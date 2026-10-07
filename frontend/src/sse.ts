// Minimal, dependency-free Server-Sent Events parsing for the POST /api/ask stream.

import type { CouncilEvent, EventStatus, Stage, StreamEvent } from "./api";

const LINE_BREAK = /\r\n|\r|\n/;

/**
 * Incremental text/event-stream parser. Feed it decoded text chunks of any size; it returns the
 * `data` payload of every event completed by that chunk. Handles LF, CRLF and CR line endings
 * (also when a CRLF is split across chunks), multi-line `data:` fields and `:` comment lines.
 * `event:`, `id:` and `retry:` fields are ignored: each payload is identified by its JSON `stage`.
 */
export class SSEParser {
  private buffer = "";
  private data: string[] = [];

  push(chunk: string): string[] {
    let text = this.buffer + chunk;
    // A trailing CR may be the first half of a CRLF that continues in the next chunk.
    let carry = "";
    if (text.endsWith("\r")) {
      carry = "\r";
      text = text.slice(0, -1);
    }
    const lines = text.split(LINE_BREAK);
    this.buffer = (lines.pop() ?? "") + carry;
    const out: string[] = [];
    for (const line of lines) this.processLine(line, out);
    return out;
  }

  /** Call once the stream has ended. Dispatches a final event even without a trailing blank line. */
  flush(): string[] {
    const out: string[] = [];
    const rest = this.buffer;
    this.buffer = "";
    if (rest) for (const line of rest.split(LINE_BREAK)) this.processLine(line, out);
    this.dispatch(out);
    return out;
  }

  private processLine(line: string, out: string[]): void {
    if (line === "") {
      this.dispatch(out);
      return;
    }
    if (line.startsWith(":")) return; // comment / keep-alive ping
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "data") this.data.push(value);
  }

  private dispatch(out: string[]): void {
    if (this.data.length > 0) {
      out.push(this.data.join("\n"));
      this.data = [];
    }
  }
}

/** Turn one `data:` payload into a typed event. Returns null for anything that is not a recognisable event. */
export function parseStreamEvent(payload: string): StreamEvent | null {
  let value: unknown;
  try {
    value = JSON.parse(payload);
  } catch {
    return null;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const o = value as Record<string, unknown>;
  if (typeof o.stage !== "string") return null;

  if (o.stage === "done") {
    return typeof o.run_id === "string" && o.run_id ? { stage: "done", run_id: o.run_id } : null;
  }
  if (o.stage === "error") {
    const message =
      typeof o.message === "string" ? o.message : typeof o.detail === "string" ? o.detail : "The council run failed.";
    return { stage: "error", message };
  }
  return {
    stage: o.stage as Stage,
    model: typeof o.model === "string" ? o.model : "",
    status: (typeof o.status === "string" ? o.status : "start") as EventStatus,
    round: typeof o.round === "number" ? o.round : 0,
    message: typeof o.message === "string" ? o.message : null,
  };
}

/**
 * Read a council SSE stream until its `done` event, calling `onEvent` for every progress event.
 * Resolves with the run id; rejects if the stream ends without `done` or carries an error event.
 */
export async function consumeCouncilStream(
  body: ReadableStream<Uint8Array>,
  onEvent: (event: CouncilEvent) => void,
): Promise<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser();

  const handle = (payloads: string[]): string | null => {
    for (const payload of payloads) {
      const event = parseStreamEvent(payload);
      if (!event) continue;
      if (event.stage === "done") return event.run_id;
      if (event.stage === "error") throw new Error(event.message);
      onEvent(event);
    }
    return null;
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      const payloads = done
        ? [...parser.push(decoder.decode()), ...parser.flush()]
        : parser.push(decoder.decode(value, { stream: true }));
      const runId = handle(payloads);
      if (runId) return runId;
      if (done) break;
    }
  } finally {
    // Stop reading; closes the connection when we return early on `done`.
    reader.cancel().catch(() => {});
  }
  throw new Error("The event stream ended before the council finished.");
}
