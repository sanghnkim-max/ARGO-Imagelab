package dev.argo.imagelab;

import android.os.ParcelFileDescriptor;

/**
 * ImageLab's side of the companion wire: how the ARGO app's agent drives ImageLab
 * (docs/INTERFACES.md §12). Served by `dev.argo.imagelab.agent.LabService`, which answers only
 * a caller signed by a pinned ARGO certificate.
 *
 * Every request and every result is JSON, so the contract can grow without a new AIDL. A copy
 * of this file lives in the ARGO app (`apps/argo-android`, `com.argo.android.tools.lab`); keep
 * the two identical, and add methods only at the end so transaction codes stay put.
 */
interface ILabService {
    /** The contract version this service speaks (1). */
    int contractVersion();

    /**
     * Run one operation and return the read end of a pipe carrying its result: one JSON object,
     * then end of stream. Results can be megabytes (images inline as base64), more than a
     * Binder transaction holds, so they never travel in the reply itself.
     *
     * `op` is one of:
     * - `mcp`: `argsJson` is one MCP JSON-RPC message (a request or a notification); the pipe
     *   carries the JSON-RPC response, or nothing for a notification. ImageLab's MCP server
     *   (`initialize`, `ping`, `tools/list`, `tools/call`, `notifications/cancelled`) with the
     *   tools `open`, `read`, `edit`, `help` and `save`; ARGO mounts it as the server `imagelab`.
     * - `status`: `{"ok": {contract, version, document}}` (the open document, or null).
     *
     * Blocks a Binder thread of ImageLab only until the pipe is made; the work runs on
     * ImageLab's own threads.
     */
    ParcelFileDescriptor call(String op, String argsJson);

    /**
     * Copy a photo into ImageLab's workspace and open it as the document the tools work on:
     * `photo` is a readable file descriptor (ARGO opens the content URI or attachment), `name` a
     * file name with its extension. Returns `{"ok": {path, index, width, height}}` or
     * `{"error": message}` as JSON text.
     */
    String importPhoto(String name, in ParcelFileDescriptor photo);

    /** Stop every program ImageLab is running for the agent (the person pressed Stop). */
    oneway void cancel();
}
