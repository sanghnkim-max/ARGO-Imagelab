use std::sync::{Arc, Mutex};
use std::time::Duration;

use imagelab_api::{Effects, Handles, Headless, ImageBuf, MaskBuf, NoEvents, Shared, Source};
use serde_json::json;

use super::*;

fn shared() -> Shared {
    Shared { root: std::env::temp_dir(), session: Arc::new(Mutex::new(Headless::new())), handles: Arc::new(Handles::default()), events: Arc::new(NoEvents) }
}

/// A test area: arithmetic, a pretend write, and image/mask makers.
fn registry() -> Registry {
    let mut r = Registry::new();
    r.area("math", "arithmetic for tests");
    r.function("math", "add", "add two numbers").handler(|_, p| Ok(json!(p["a"].as_i64().unwrap_or(0) + p["b"].as_i64().unwrap_or(0)))).register();
    r.function("math", "store", "pretend write").effects(Effects::Revertible).handler(|_, _| Ok(json!("stored"))).register();
    r.function("math", "fail", "always fails").handler(|_, _| Err(ApiError::bad_params("nope"))).register();
    r.area("img", "image makers for tests");
    r.function("img", "make", "a 4x4 grey image")
        .handler(|ctx, _| {
            let h = ctx.shared.handles.put_image(ImageBuf::new(4, 4, vec![128; 64])?)?;
            Ok(json!({"img": h.to_string()}))
        })
        .register();
    r.function("img", "mask", "a 4x4 mask")
        .handler(|ctx, _| {
            let h = ctx.shared.handles.put_mask(MaskBuf::new(4, 4, vec![255; 16])?)?;
            Ok(json!({"mask": h.to_string()}))
        })
        .register();
    r
}

fn run(code: &str) -> ScriptOutcome {
    let ctx = Ctx::new(shared(), Source::Agent);
    ScriptRuntime::run(&registry(), &ctx, code, Budget::default())
}

#[test]
fn arithmetic_and_return_value() {
    let o = run("const x = 6 * 7; return { x, list: [1, 'a', null] };");
    assert_eq!(o.error, None);
    assert_eq!(o.value, json!({"x": 42, "list": [1, "a", null]}));
    assert_eq!(run("1 + 1;").value, Value::Null);
}

#[test]
fn area_calls_and_help() {
    let o = run("const a = await lab.math.add({a: 2, b: 3}); const h = lab.help('math'); const all = await lab.help(); return {a, n: h.length, areas: all.map(x => x.area)};");
    assert_eq!(o.error, None, "{o:?}");
    assert_eq!(o.value, json!({"a": 5, "n": 3, "areas": ["img", "math"]}));
    assert_eq!(o.calls, 3);
}

#[test]
fn console_is_captured() {
    let o = run("console.log('hi', {a: 1}); console.warn('careful'); console.error(new Error('bad')); return 1;");
    assert_eq!(o.logs, vec!["hi {\"a\":1}", "[warn] careful", "[error] Error: bad"]);
}

#[test]
fn console_is_bounded() {
    let o = run("for (let i = 0; i < 100000; i++) console.log('line ' + i); return 1;");
    assert!(o.logs.len() <= 1001);
    assert_eq!(o.logs.last().map(String::as_str), Some("[console output truncated]"));
}

#[test]
fn read_only_refuses_writes() {
    let ctx = Ctx::new(shared(), Source::Agent).read_only();
    let code = "try { await lab.math.store({}); return 'wrote'; } catch (e) { return {name: e.name, code: e.code, isLab: e instanceof LabCallError, fn: e.fn}; }";
    let o = ScriptRuntime::run(&registry(), &ctx, code, Budget::default());
    assert_eq!(o.value, json!({"name": "LabCallError", "code": "read_only", "isLab": true, "fn": "math.store"}));
    // Uncaught, the call's code becomes the run's.
    let o = ScriptRuntime::run(&registry(), &ctx, "await lab.math.store({});", Budget::default());
    assert_eq!(o.error.map(|e| e.code), Some("read_only".to_string()));
    // Reads still work.
    let o = ScriptRuntime::run(&registry(), &ctx, "return await lab.math.add({a: 1, b: 1});", Budget::default());
    assert_eq!(o.value, json!(2));
}

#[test]
fn infinite_loop_times_out() {
    let ctx = Ctx::new(shared(), Source::Agent);
    let budget = Budget { wall: Duration::from_millis(300), ..Budget::default() };
    let started = std::time::Instant::now();
    let o = ScriptRuntime::run(&registry(), &ctx, "while (true) {}", budget.clone());
    assert_eq!(o.error.map(|e| e.code), Some("timeout".to_string()));
    assert!(started.elapsed() < Duration::from_secs(5));
    // A try/catch cannot swallow the interrupt.
    let o = ScriptRuntime::run(&registry(), &ctx, "try { while (true) {} } catch (e) { return 'caught'; }", budget);
    assert_eq!(o.error.map(|e| e.code), Some("timeout".to_string()));
}

#[test]
fn cancel_stops_a_run() {
    let ctx = Ctx::new(shared(), Source::Agent);
    let flag = ctx.cancel.clone();
    let t = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        flag.store(true, Ordering::Relaxed);
    });
    let o = ScriptRuntime::run(&registry(), &ctx, "while (true) {}", Budget::default());
    t.join().unwrap();
    assert_eq!(o.error.map(|e| e.code), Some("cancelled".to_string()));
}

#[test]
fn thrown_errors_are_values() {
    let o = run("throw new Error('boom');");
    let e = o.error.unwrap();
    assert_eq!(e.code, "exception");
    assert!(e.message.contains("boom"), "{e:?}");
    let o = run("await lab.math.fail({});");
    assert_eq!(o.error.unwrap().code, "bad_params");
    let o = run("throw 'plain';");
    assert!(o.error.unwrap().message.contains("plain"));
    let o = run("return (");
    assert_eq!(o.error.unwrap().code, "syntax");
    let o = run("await lab.math.nope({});");
    let e = o.error.unwrap();
    assert_eq!(e.code, "not_found");
    assert!(e.message.contains("lab.help('math')"), "{e:?}");
    let o = run("await lab.nowhere.fn();");
    assert!(o.error.unwrap().message.contains("areas: img, math"));
    let o = run("await new Promise(() => {});");
    assert_eq!(o.error.unwrap().code, "never_settled");
    let o = run("return 10n;");
    assert_eq!(o.error.unwrap().code, "bad_result");
}

#[test]
fn budgets_calls_and_memory() {
    let ctx = Ctx::new(shared(), Source::Agent);
    let budget = Budget { max_calls: 3, ..Budget::default() };
    let o = ScriptRuntime::run(&registry(), &ctx, "for (let i = 0; i < 10; i++) await lab.math.add({a: i, b: 1});", budget);
    assert_eq!(o.error.unwrap().code, "too_many_calls");
    let budget = Budget { memory_bytes: 16 << 20, ..Budget::default() };
    let o = ScriptRuntime::run(&registry(), &ctx, "const a = []; while (true) a.push('x'.repeat(1 << 16) + a.length);", budget);
    let e = o.error.unwrap();
    assert!(e.code == "out_of_memory" || e.code == "exception", "{e:?}");
}

#[test]
fn no_host_globals_leak() {
    let o = run("return [typeof __lab_call, typeof fetch, typeof setTimeout, typeof require, typeof lab, typeof globalThis.lab];");
    assert_eq!(o.value, json!(["undefined", "undefined", "undefined", "undefined", "object", "undefined"]));
}

#[test]
fn images_are_extracted_and_handles_freed() {
    let ctx = Ctx::new(shared(), Source::Agent);
    let r = registry();
    let code = "const a = await lab.img.make(); const b = await lab.img.make(); const m = await lab.img.mask(); \
                const k = await lab.img.make(); lab.keep(k.img); \
                return {preview: {image: a.img, note: 'x'}, again: {image: a.img}, mask: {image: m.mask}};";
    let o = ScriptRuntime::run(&r, &ctx, code, Budget::default());
    assert_eq!(o.error, None, "{o:?}");
    assert_eq!(o.images.len(), 2);
    assert!(o.images.iter().all(|i| i.mime == "image/png" && i.bytes.starts_with(b"\x89PNG")));
    assert_eq!(o.value["preview"], json!({"image": "img:1", "note": "x", "attached": 0}));
    assert_eq!(o.value["again"]["attached"], json!(0));
    assert_eq!(o.value["mask"]["attached"], json!(1));
    // img:2 (unreferenced) is freed; img:1 and mask:3 were only shown, so they are freed once
    // encoded; img:4 (kept) lives.
    let live: HashSet<String> = ctx.shared.handles.ids().iter().map(ToString::to_string).collect();
    assert_eq!(live, ["img:4"].iter().map(|s| s.to_string()).collect());
}

#[test]
fn a_shown_image_also_named_elsewhere_survives() {
    let ctx = Ctx::new(shared(), Source::Agent);
    let o = ScriptRuntime::run(&registry(), &ctx, "const a = await lab.img.make(); return {view: {image: a.img}, handle: a.img};", Budget::default());
    assert_eq!(o.error, None, "{o:?}");
    assert_eq!(o.images.len(), 1);
    let live: HashSet<String> = ctx.shared.handles.ids().iter().map(ToString::to_string).collect();
    assert_eq!(live, ["img:1"].iter().map(|s| s.to_string()).collect());
}

#[test]
fn handles_from_before_the_run_survive() {
    let ctx = Ctx::new(shared(), Source::Agent);
    let old = ctx.shared.handles.put_image(ImageBuf::new(2, 2, vec![0; 16]).unwrap()).unwrap();
    let mut r = registry();
    r.function("img", "echo", "returns the old handle").handler(move |_, _| Ok(json!({"img": old.to_string()}))).register();
    let o = ScriptRuntime::run(&r, &ctx, "await lab.img.echo(); await lab.img.make(); throw new Error('x');", Budget::default());
    assert!(o.error.is_some());
    assert_eq!(ctx.shared.handles.ids(), vec![old]);
}

#[test]
fn oversized_code_is_refused() {
    let o = run(&"x".repeat(MAX_CODE_BYTES + 1));
    assert_eq!(o.error.unwrap().code, "too_large");
}

#[test]
fn cv_free_in_a_program_only_frees_its_own_handles() {
    let ctx = Ctx::new(shared(), Source::Agent);
    let old = ctx.shared.handles.put_image(ImageBuf::new(2, 2, vec![0; 16]).unwrap()).unwrap();
    let mut r = registry();
    r.area("cv", "test cv");
    r.function("cv", "free", "free handles")
        .handler(|ctx, p| {
            let mut freed = Vec::new();
            for h in p["handles"].as_array().cloned().unwrap_or_default() {
                if let Some(id) = h.as_str().and_then(HandleId::parse) {
                    ctx.shared.handles.free(id);
                    freed.push(h);
                }
            }
            Ok(json!({"freed": freed}))
        })
        .register();
    let code = format!("const a = await lab.img.make(); return await lab.cv.free({{handles: [a.img, '{old}']}});");
    let o = ScriptRuntime::run(&r, &ctx, &code, Budget::default());
    assert_eq!(o.error, None, "{o:?}");
    assert_eq!(o.value["freed"].as_array().map(Vec::len), Some(1), "{o:?}");
    assert!(ctx.shared.handles.ids().contains(&old), "a handle from before the run is not the program's to free");
}

#[test]
fn calls_past_the_budget_are_refused_at_once() {
    let ctx = Ctx::new(shared(), Source::Agent);
    let mut r = registry();
    r.function("math", "slow", "sleeps 100 ms").handler(|_, _| {
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(json!(1))
    }).register();
    let started = std::time::Instant::now();
    let budget = Budget { wall: std::time::Duration::from_millis(250), ..Budget::default() };
    let o = ScriptRuntime::run(&r, &ctx, "for (let i = 0; i < 50; i++) await lab.math.slow(); return 'done';", budget);
    assert!(started.elapsed() < std::time::Duration::from_millis(1500), "took {:?}", started.elapsed());
    let e = o.error.expect("timed out");
    assert!(e.code == "timeout" || e.code == "cancelled", "{e:?}");
}
