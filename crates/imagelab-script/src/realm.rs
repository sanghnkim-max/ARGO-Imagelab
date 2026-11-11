//! The QuickJS side of a run: one fresh runtime + context per program, on its own thread.
//!
//! The realm talks to the host thread (which owns the registry and the call context) only
//! through a channel: `lab.<area>.<fn>(args)` becomes [`ToHost::Call`] and blocks the realm
//! until the host answers. Registry calls are synchronous, so the program's promises settle
//! through QuickJS's own job queue; no async executor is involved.
//!
//! The three Rust hooks the bootstrap needs are installed on `globalThis` and deleted by the
//! bootstrap's first statements, before any program code runs: the program reaches the host
//! only through the `lab`, `console` and `LabCallError` parameters of its wrapper function.

use std::cell::{Cell, RefCell};
use std::panic::AssertUnwindSafe;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Instant;

use rquickjs::{Context, Ctx, Function, Object, Runtime, Value};
use serde_json::Value as Json;

use crate::{Budget, ScriptError};

/// QuickJS's own interpreter-stack ceiling: well inside [`crate::REALM_THREAD_STACK`], so
/// runaway recursion is a catchable `RangeError` rather than a crash.
const JS_STACK_BYTES: usize = 4 << 20;
/// Largest argument object one call may send (JSON text).
const MAX_ARGS_BYTES: usize = 16 << 20;
/// Most console lines kept.
const MAX_LOG_LINES: usize = 1000;

/// What the realm asks of the host thread.
pub(crate) enum ToHost {
    /// Call `id` (`area.name`, or `$help`) with JSON `args`; the reply is `{"ok":…}` or
    /// `{"err":{"code","message"}}` as JSON text.
    Call { id: String, args: String, reply: mpsc::Sender<String> },
    /// `lab.keep(handle)`.
    Keep(String),
}

pub(crate) struct RealmSetup {
    pub code: String,
    /// `[(area, [function names])]`.
    pub spec: Vec<(String, Vec<String>)>,
    pub budget: Budget,
    /// The caller's cancel flag (`Ctx::cancel`).
    pub cancel: Arc<AtomicBool>,
    /// Set by the host when it gives up waiting.
    pub stop: Arc<AtomicBool>,
    pub host: mpsc::Sender<ToHost>,
}

pub(crate) struct RealmDone {
    pub result: Result<Json, ScriptError>,
    pub logs: Vec<String>,
    pub calls: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stop {
    Timeout,
    Cancelled,
}

/// Bounded console capture.
struct Logs {
    lines: Vec<String>,
    bytes: usize,
    max_bytes: usize,
    truncated: bool,
}

impl Logs {
    fn push(&mut self, level: &str, text: &str) {
        if self.truncated {
            return;
        }
        let line = match level {
            "warn" | "error" => format!("[{level}] {text}"),
            _ => text.to_string(),
        };
        let room = self.max_bytes.saturating_sub(self.bytes);
        if self.lines.len() >= MAX_LOG_LINES || line.len() > room {
            let mut cut = room.min(line.len());
            while cut > 0 && !line.is_char_boundary(cut) {
                cut -= 1;
            }
            if cut > 0 && self.lines.len() < MAX_LOG_LINES {
                self.lines.push(line[..cut].to_string());
            }
            self.lines.push("[console output truncated]".to_string());
            self.truncated = true;
            return;
        }
        self.bytes += line.len();
        self.lines.push(line);
    }
}

/// The bootstrap. `__SPEC__` is replaced by the `[[area, [names]], …]` JSON literal.
const BOOTSTRAP: &str = r#"(() => {
  'use strict';
  const call = globalThis.__lab_call;
  const keep = globalThis.__lab_keep;
  const log = globalThis.__lab_log;
  delete globalThis.__lab_call;
  delete globalThis.__lab_keep;
  delete globalThis.__lab_log;
  const parse = JSON.parse;
  const stringify = JSON.stringify;
  const freeze = Object.freeze;
  const create = Object.create;
  const CapturedError = Error;
  const CapturedSyntaxError = SyntaxError;
  const CapturedPromise = Promise;
  const CapturedProxy = Proxy;
  const SPEC = __SPEC__;

  class LabCallError extends CapturedError {
    constructor(code, message, fn) {
      super(message);
      this.name = 'LabCallError';
      this.code = code;
      if (fn !== undefined) this.fn = fn;
    }
  }

  const invoke = (id, args) => {
    let text;
    try {
      text = args === undefined ? '{}' : stringify(args);
    } catch (e) {
      throw new LabCallError('bad_params', 'the arguments of ' + id + ' are not JSON: ' + (e && e.message), id);
    }
    if (text === undefined) text = '{}';
    const reply = parse(call(id, text));
    if (reply.err) throw new LabCallError(reply.err.code, reply.err.message, id);
    return reply.ok;
  };

  const areaNames = [];
  const lab = create(null);
  for (let i = 0; i < SPEC.length; i++) {
    const area = SPEC[i][0];
    const names = SPEC[i][1];
    areaNames.push(area);
    const ns = create(null);
    for (let j = 0; j < names.length; j++) {
      const id = area + '.' + names[j];
      ns[names[j]] = async (args) => invoke(id, args);
    }
    lab[area] = new CapturedProxy(freeze(ns), {
      get: (t, k) => (typeof k !== 'string' || k in t || k === 'then') ? t[k]
        : () => CapturedPromise.reject(new LabCallError('not_found', 'no function ' + area + '.' + k + "; see lab.help('" + area + "')", area + '.' + k)),
    });
  }
  lab.help = (area) => invoke('$help', area === undefined || area === null ? null : String(area));
  lab.keep = (handle) => {
    if (typeof handle !== 'string') throw new TypeError('lab.keep takes a handle string such as "img:3"');
    keep(handle);
    return handle;
  };
  const missingArea = (area) => new CapturedProxy(create(null), {
    get: (t, k) => (typeof k !== 'string' || k === 'then') ? undefined
      : () => CapturedPromise.reject(new LabCallError('not_found', 'no area `' + area + '`; areas: ' + areaNames.join(', ') + ' (lab.help() lists them)', area + '.' + k)),
  });
  const labProxy = new CapturedProxy(freeze(lab), {
    get: (t, k) => (typeof k !== 'string' || k in t || k === 'then') ? t[k] : missingArea(k),
  });

  const show = (a) => {
    if (typeof a === 'string') return a;
    try {
      if (a instanceof CapturedError) return String(a.name) + ': ' + String(a.message);
      const s = stringify(a);
      return s === undefined ? String(a) : s;
    } catch (e) {
      try { return String(a); } catch (e2) { return '[unprintable]'; }
    }
  };
  const render = (args) => {
    const parts = [];
    for (let i = 0; i < args.length; i++) parts.push(show(args[i]));
    return parts.join(' ');
  };
  const consoleShim = freeze({
    log: (...a) => { log('log', render(a)); },
    info: (...a) => { log('log', render(a)); },
    debug: (...a) => { log('log', render(a)); },
    warn: (...a) => { log('warn', render(a)); },
    error: (...a) => { log('error', render(a)); },
  });

  const toJson = (v) => {
    if (v === undefined) return 'null';
    const s = stringify(v);
    return s === undefined ? 'null' : s;
  };
  const describe = (e) => {
    try {
      if (e instanceof LabCallError) return stringify({ code: String(e.code), message: String(e.message) });
      if (e instanceof CapturedError) {
        const stack = typeof e.stack === 'string' && e.stack.length > 0 ? '\n' + e.stack : '';
        const code = e instanceof CapturedSyntaxError ? 'syntax' : 'exception';
        return stringify({ code, message: String(e.name) + ': ' + String(e.message) + stack });
      }
      return stringify({ code: 'exception', message: 'thrown: ' + String(e) });
    } catch (x) {
      return stringify({ code: 'exception', message: 'the program threw an unrenderable value' });
    }
  };
  return { lab: labProxy, console: consoleShim, LabCallError, toJson, describe };
})()"#;

/// Run one program to its outcome. Never panics out.
pub(crate) fn run_realm(setup: RealmSetup) -> RealmDone {
    match std::panic::catch_unwind(AssertUnwindSafe(|| drive(setup))) {
        Ok(done) => done,
        Err(_) => RealmDone { result: Err(ScriptError::new("internal", "the script realm panicked")), logs: Vec::new(), calls: 0 },
    }
}

fn fail(code: &str, message: impl Into<String>) -> Result<Json, ScriptError> {
    Err(ScriptError::new(code, message))
}

fn drive(setup: RealmSetup) -> RealmDone {
    let RealmSetup { code, spec, budget, cancel, stop, host } = setup;
    let logs = Rc::new(RefCell::new(Logs { lines: Vec::new(), bytes: 0, max_bytes: budget.max_log_bytes, truncated: false }));
    let calls = Rc::new(Cell::new(0u32));
    let stopped: Rc<Cell<Option<Stop>>> = Rc::new(Cell::new(None));

    let result = (|| {
        let Ok(runtime) = Runtime::new() else { return fail("internal", "the script runtime could not start") };
        runtime.set_memory_limit(budget.memory_bytes);
        runtime.set_max_stack_size(JS_STACK_BYTES);
        let deadline = Instant::now() + budget.wall;
        {
            let stopped = stopped.clone();
            let cancel = cancel.clone();
            let stop = stop.clone();
            runtime.set_interrupt_handler(Some(Box::new(move || {
                if stopped.get().is_some() {
                    return true;
                }
                if cancel.load(Ordering::Relaxed) || stop.load(Ordering::Relaxed) {
                    stopped.set(Some(Stop::Cancelled));
                    return true;
                }
                if Instant::now() >= deadline {
                    stopped.set(Some(Stop::Timeout));
                    return true;
                }
                false
            })));
        }
        let Ok(context) = Context::full(&runtime) else { return fail("internal", "the script context could not start") };
        let result = context.with(|ctx| run_in(&ctx, &code, &spec, &budget, &host, &logs, &calls, &stopped));
        drop(context);
        runtime.run_gc();
        result
    })();

    // A stop the interrupt handler recorded wins over whatever error text the engine produced.
    let result = match (result, stopped.get()) {
        (Err(_), Some(Stop::Timeout)) => {
            fail("timeout", format!("the program ran past its {} ms budget", budget.wall.as_millis()))
        }
        (Err(_), Some(Stop::Cancelled)) => fail("cancelled", "the program was cancelled"),
        (r, _) => r,
    };
    let logs = std::mem::take(&mut logs.borrow_mut().lines);
    RealmDone { result, logs, calls: calls.get() }
}

#[allow(clippy::too_many_arguments)]
fn run_in<'js>(
    ctx: &Ctx<'js>,
    code: &str,
    spec: &[(String, Vec<String>)],
    budget: &Budget,
    host: &mpsc::Sender<ToHost>,
    logs: &Rc<RefCell<Logs>>,
    calls: &Rc<Cell<u32>>,
    stopped: &Rc<Cell<Option<Stop>>>,
) -> Result<Json, ScriptError> {
    let internal = |what: &str| ScriptError::new("internal", format!("the script realm could not start: {what}"));
    let globals = ctx.globals();

    // __lab_call(id, argsJson) -> replyJson
    let call_host = host.clone();
    let call_count = calls.clone();
    let max_calls = budget.max_calls;
    let call_stopped = stopped.clone();
    let call = Function::new(ctx.clone(), move |id: String, args: String| -> String {
        if call_stopped.get().is_some() {
            return err_reply("cancelled", "the program is stopping");
        }
        if call_count.get() >= max_calls {
            return err_reply("too_many_calls", &format!("the program made more than {max_calls} calls"));
        }
        if args.len() > MAX_ARGS_BYTES {
            return err_reply("too_large", &format!("arguments of {id} exceed {MAX_ARGS_BYTES} bytes"));
        }
        call_count.set(call_count.get() + 1);
        let (tx, rx) = mpsc::channel();
        if call_host.send(ToHost::Call { id, args, reply: tx }).is_err() {
            return err_reply("internal", "the host went away");
        }
        rx.recv().unwrap_or_else(|_| err_reply("internal", "the host went away"))
    })
    .map_err(|_| internal("call hook"))?;
    let keep_host = host.clone();
    let keep = Function::new(ctx.clone(), move |handle: String| {
        let _ = keep_host.send(ToHost::Keep(handle));
    })
    .map_err(|_| internal("keep hook"))?;
    let log_sink = logs.clone();
    let log = Function::new(ctx.clone(), move |level: String, text: String| {
        log_sink.borrow_mut().push(&level, &text);
    })
    .map_err(|_| internal("log hook"))?;
    globals.set("__lab_call", call).map_err(|_| internal("globals"))?;
    globals.set("__lab_keep", keep).map_err(|_| internal("globals"))?;
    globals.set("__lab_log", log).map_err(|_| internal("globals"))?;

    let spec_json = serde_json::to_string(spec).map_err(|_| internal("spec"))?;
    let bootstrap = BOOTSTRAP.replace("__SPEC__", &spec_json);
    let exports: Object<'js> = ctx.eval(bootstrap).map_err(|_| internal("bootstrap"))?;
    let get = |name: &str| -> Result<Value<'js>, ScriptError> { exports.get::<_, Value<'js>>(name).map_err(|_| internal(name)) };
    let lab = get("lab")?;
    let console = get("console")?;
    let error_class = get("LabCallError")?;
    let to_json: Function<'js> = exports.get("toJson").map_err(|_| internal("toJson"))?;
    let describe: Function<'js> = exports.get("describe").map_err(|_| internal("describe"))?;

    let thrown = |ctx: &Ctx<'js>| -> ScriptError {
        let value = ctx.catch();
        describe_thrown(&describe, &value)
    };

    // The program body, as an async function: top-level await and `return`. The header shares
    // the body's first line so error line numbers are the program's own (+1).
    let source = format!("(async function (lab, console, LabCallError) {{ 'use strict';\n{code}\n}})");
    let wrapper: Function<'js> = match ctx.eval::<Value<'js>, _>(source) {
        Ok(v) => match v.into_function() {
            Some(f) => f,
            None => return Err(internal("wrapper")),
        },
        Err(rquickjs::Error::Exception) => {
            let mut e = thrown(ctx);
            if e.code == "exception" {
                e.code = "syntax".into();
            }
            return Err(e);
        }
        Err(e) => return Err(ScriptError::new("internal", e.to_string())),
    };
    let returned: Value<'js> = match wrapper.call((lab, console, error_class)) {
        Ok(v) => v,
        Err(rquickjs::Error::Exception) => return Err(thrown(ctx)),
        Err(e) => return Err(ScriptError::new("internal", e.to_string())),
    };
    let value: Value<'js> = match returned.as_promise() {
        Some(promise) => match promise.finish::<Value<'js>>() {
            Ok(v) => v,
            Err(rquickjs::Error::Exception) => return Err(thrown(ctx)),
            Err(rquickjs::Error::WouldBlock) => {
                return Err(ScriptError::new("never_settled", "the program awaited a promise that never settles (there are no timers or I/O here)"));
            }
            Err(e) => return Err(ScriptError::new("internal", e.to_string())),
        },
        None => returned,
    };
    let text: String = match to_json.call((value,)) {
        Ok(t) => t,
        Err(rquickjs::Error::Exception) => {
            let e = thrown(ctx);
            return Err(ScriptError::new("bad_result", format!("the returned value is not JSON: {}", e.message)));
        }
        Err(e) => return Err(ScriptError::new("internal", e.to_string())),
    };
    if text.len() > budget.max_result_bytes {
        return Err(ScriptError::new("too_large", format!("the returned value is {} bytes of JSON; the limit is {}", text.len(), budget.max_result_bytes)));
    }
    serde_json::from_str(&text).map_err(|e| ScriptError::new("bad_result", e.to_string()))
}

fn describe_thrown<'js>(describe: &Function<'js>, value: &Value<'js>) -> ScriptError {
    // QuickJS reports an exhausted heap as an exception that is not always an object.
    let text: Option<String> = describe.call((value.clone(),)).ok();
    let parsed = text.and_then(|t| serde_json::from_str::<Json>(&t).ok());
    let (code, message) = match parsed {
        Some(Json::Object(map)) => (
            map.get("code").and_then(Json::as_str).unwrap_or("exception").to_string(),
            map.get("message").and_then(Json::as_str).unwrap_or("").to_string(),
        ),
        _ => ("exception".to_string(), "the program threw (no description available; out of memory?)".to_string()),
    };
    if message.contains("out of memory") {
        return ScriptError::new("out_of_memory", message);
    }
    ScriptError::new(&code, message)
}

fn err_reply(code: &str, message: &str) -> String {
    serde_json::json!({"err": {"code": code, "message": message}}).to_string()
}
