//! PhotoCraft's MCP server, two ways in.
//!
//! The tools are PhotoCraft's own (`photocraft_automation::PhotocraftMcp`: `doc_open`,
//! `doc_inspect`, `command_list`, `command_run`, `command_batch`, `doc_render_preview`, …).
//! ImageLab adds no tool of its own here: what an agent can do in the app is exactly what it
//! can do with `photocraft-cli mcp` on a desktop, over the document the person sees.
//!
//! * [`InProcess`]: a JSON-lines connection over an in-memory pipe, for the in-app agent.
//! * [`HttpServer`]: streamable HTTP on loopback with a bearer token, for agents off the phone
//!   (`adb forward tcp:7700 tcp:7700`, then `claude mcp add --transport http …`).

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::Request;
use axum::http::{StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use photocraft_automation::PhotocraftMcp;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, ReadHalf, WriteHalf};
use tokio::io::DuplexStream;
use tokio_util::sync::CancellationToken;

use crate::{LabError, Result};

/// The protocol version the in-process client asks for.
const PROTOCOL_VERSION: &str = "2025-06-18";
/// A request (a long filter, a large render) may take this long before the agent gets an error.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
/// Largest response line the in-process client accepts (PhotoCraft caps its own replies lower).
const MAX_LINE_BYTES: usize = 64 << 20;

struct Conn {
    writer: WriteHalf<DuplexStream>,
    reader: BufReader<ReadHalf<DuplexStream>>,
}

/// An MCP client connected to PhotoCraft's server over an in-memory pipe. Requests are
/// serialized: PhotoCraft's session is one mutex anyway.
pub struct InProcess {
    conn: tokio::sync::Mutex<Conn>,
    next_id: AtomicU64,
}

impl InProcess {
    /// Start the server on the current runtime and complete the MCP handshake.
    pub async fn start(mcp: PhotocraftMcp) -> Result<Self> {
        let (client, server) = tokio::io::duplex(1 << 20);
        let (server_read, server_write) = tokio::io::split(server);
        tokio::spawn(async move {
            // Ends when the client half is dropped (the Lab goes away).
            let _ = mcp.serve_io(server_read, server_write).await;
        });
        let (client_read, client_write) = tokio::io::split(client);
        let this = Self { conn: tokio::sync::Mutex::new(Conn { writer: client_write, reader: BufReader::new(client_read) }), next_id: AtomicU64::new(1) };
        this.request(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "imagelab-app", "version": env!("CARGO_PKG_VERSION")}
            }),
        )
        .await?;
        this.notify("notifications/initialized", json!({})).await?;
        Ok(this)
    }

    /// Send a request and wait for its result (`result` on success; an MCP `error` becomes
    /// [`LabError::BadRequest`]). Tool failures are results with `isError: true`, as in MCP.
    pub async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let message = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let mut conn = self.conn.lock().await;
        write_line(&mut conn.writer, &message).await?;
        let wait = async {
            loop {
                let line = read_line(&mut conn.reader).await?;
                let Ok(reply) = serde_json::from_str::<Value>(&line) else { continue };
                // Notifications (progress, list_changed) carry no id; skip them.
                if reply.get("id").and_then(Value::as_u64) != Some(id) {
                    continue;
                }
                if let Some(error) = reply.get("error") {
                    let text = error.get("message").and_then(Value::as_str).unwrap_or("MCP error");
                    return Err(LabError::BadRequest(text.to_string()));
                }
                return Ok(reply.get("result").cloned().unwrap_or(Value::Null));
            }
        };
        tokio::time::timeout(REQUEST_TIMEOUT, wait).await.map_err(|_| LabError::Other(format!("MCP {method} timed out")))?
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let mut conn = self.conn.lock().await;
        write_line(&mut conn.writer, &json!({"jsonrpc": "2.0", "method": method, "params": params})).await
    }
}

async fn write_line(writer: &mut WriteHalf<DuplexStream>, message: &Value) -> Result<()> {
    let mut bytes = serde_json::to_vec(message).map_err(|e| LabError::Other(e.to_string()))?;
    bytes.push(b'\n');
    writer.write_all(&bytes).await?;
    writer.flush().await?;
    Ok(())
}

async fn read_line(reader: &mut BufReader<ReadHalf<DuplexStream>>) -> Result<String> {
    let mut buf = Vec::new();
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Err(LabError::Other("MCP server closed the connection".into()));
        }
        let newline = available.iter().position(|&b| b == b'\n');
        let take = newline.map_or(available.len(), |at| at + 1);
        if buf.len() + take > MAX_LINE_BYTES {
            return Err(LabError::Other(format!("MCP response exceeds {MAX_LINE_BYTES} bytes")));
        }
        buf.extend_from_slice(&available[..take]);
        reader.consume(take);
        if newline.is_some() {
            return String::from_utf8(buf).map_err(|e| LabError::Other(e.to_string()));
        }
    }
}

/// PhotoCraft's MCP server over streamable HTTP at `http://127.0.0.1:<port>/mcp`.
///
/// Loopback only, and every request needs `Authorization: Bearer <token>`: on Android any app
/// on the phone can reach 127.0.0.1, so the token, not the address, is the access control. The
/// Host check (rmcp's default loopback allow-list) also blocks DNS rebinding from a browser.
pub struct HttpServer {
    port: u16,
    cancel: CancellationToken,
}

impl HttpServer {
    pub async fn start(mcp: PhotocraftMcp, port: u16, token: String) -> Result<Self> {
        if token.len() < 16 {
            return Err(LabError::BadRequest("the MCP token must be at least 16 characters".into()));
        }
        let cancel = CancellationToken::new();
        let config = StreamableHttpServerConfig::default().with_cancellation_token(cancel.child_token());
        let service = StreamableHttpService::new(move || Ok(mcp.clone()), Arc::new(LocalSessionManager::default()), config);
        let token = Arc::new(token);
        let router = axum::Router::new().nest_service("/mcp", service).layer(axum::middleware::from_fn(move |req: Request, next: Next| {
            let token = token.clone();
            async move { authorize(&token, req, next).await }
        }));
        let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], port))).await?;
        let port = listener.local_addr()?.port();
        let shutdown = cancel.clone();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).with_graceful_shutdown(shutdown.cancelled_owned()).await;
        });
        Ok(Self { port, cancel })
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn stop(self) {
        self.cancel.cancel();
    }
}

async fn authorize(token: &str, req: Request, next: Next) -> Response {
    let presented = req.headers().get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer ")).unwrap_or("");
    if constant_time_eq(presented.as_bytes(), token.as_bytes()) {
        next.run(req).await
    } else {
        (StatusCode::UNAUTHORIZED, [(header::WWW_AUTHENTICATE, "Bearer")], "missing or wrong bearer token").into_response()
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_compare() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn http_requires_the_token() {
        let server = HttpServer::start(PhotocraftMcp::headless(), 0, "0123456789abcdef".into()).await.unwrap();
        let addr = format!("127.0.0.1:{}", server.port());
        let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}"#;
        let status = |auth: Option<&str>| {
            let addr = addr.clone();
            let auth = auth.map(str::to_string);
            async move {
                let mut stream = tokio::net::TcpStream::connect(&addr).await.unwrap();
                let auth = auth.map(|t| format!("Authorization: Bearer {t}\r\n")).unwrap_or_default();
                let req = format!(
                    "POST /mcp HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\n{auth}Content-Length: {}\r\nConnection: close\r\n\r\n{init}",
                    init.len()
                );
                stream.write_all(req.as_bytes()).await.unwrap();
                let mut head = [0u8; 12];
                tokio::io::AsyncReadExt::read_exact(&mut stream, &mut head).await.unwrap();
                String::from_utf8_lossy(&head[9..12]).to_string()
            }
        };
        assert_eq!(status(None).await, "401");
        assert_eq!(status(Some("wrong-token-wrong!")).await, "401");
        assert_eq!(status(Some("0123456789abcdef")).await, "200");
        server.stop();
    }
}
