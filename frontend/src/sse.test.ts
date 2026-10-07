import { describe, expect, it } from "vitest";
import type { CouncilEvent } from "./api";
import { consumeCouncilStream, parseStreamEvent, SSEParser } from "./sse";

function parseAll(chunks: string[]): string[] {
  const parser = new SSEParser();
  const out: string[] = [];
  for (const chunk of chunks) out.push(...parser.push(chunk));
  out.push(...parser.flush());
  return out;
}

function streamOf(chunks: (string | Uint8Array)[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(typeof chunk === "string" ? encoder.encode(chunk) : chunk);
      controller.close();
    },
  });
}

describe("SSEParser", () => {
  it("splits LF-delimited events", () => {
    expect(parseAll(['data: {"a":1}\n\ndata: {"b":2}\n\n'])).toEqual(['{"a":1}', '{"b":2}']);
  });

  it("handles CRLF line endings (sse-starlette default)", () => {
    expect(parseAll(['data: {"a":1}\r\n\r\ndata: {"b":2}\r\n\r\n'])).toEqual(['{"a":1}', '{"b":2}']);
  });

  it("handles bare CR line endings", () => {
    expect(parseAll(["data: x\r\rdata: y\r\r"])).toEqual(["x", "y"]);
  });

  it("does not emit a spurious event when CRLF is split across chunks", () => {
    expect(parseAll(["data: one\r", "\n", "data: more\r", "\n\r", "\n"])).toEqual(["one\nmore"]);
  });

  it("reassembles events split at arbitrary points", () => {
    const text = 'event: progress\ndata: {"stage":"answer","model":"fake:alpha"}\n\ndata: {"stage":"done","run_id":"abc"}\n\n';
    for (let size = 1; size <= 7; size++) {
      const chunks: string[] = [];
      for (let i = 0; i < text.length; i += size) chunks.push(text.slice(i, i + size));
      expect(parseAll(chunks)).toEqual(['{"stage":"answer","model":"fake:alpha"}', '{"stage":"done","run_id":"abc"}']);
    }
  });

  it("joins multi-line data fields with newlines", () => {
    expect(parseAll(["data: line one\ndata: line two\n\n"])).toEqual(["line one\nline two"]);
  });

  it("ignores comments, event/id/retry fields and blank keep-alives", () => {
    expect(parseAll([": ping\n\nevent: council\nid: 7\nretry: 1000\ndata: payload\n\n\n\n"])).toEqual(["payload"]);
  });

  it("strips exactly one space after the colon and accepts no space", () => {
    expect(parseAll(["data:tight\n\ndata:  two spaces\n\n"])).toEqual(["tight", " two spaces"]);
  });

  it("dispatches a final event without a trailing blank line on flush", () => {
    expect(parseAll(['data: {"stage":"done","run_id":"r1"}'])).toEqual(['{"stage":"done","run_id":"r1"}']);
  });
});

describe("parseStreamEvent", () => {
  it("parses council progress events", () => {
    expect(parseStreamEvent('{"stage":"review","model":"fake:beta","status":"done","round":0,"message":null}')).toEqual({
      stage: "review",
      model: "fake:beta",
      status: "done",
      round: 0,
      message: null,
    });
  });

  it("fills defaults for omitted optional fields", () => {
    expect(parseStreamEvent('{"stage":"answer","model":"fake:a","status":"start"}')).toEqual({
      stage: "answer",
      model: "fake:a",
      status: "start",
      round: 0,
      message: null,
    });
  });

  it("recognises the done event", () => {
    expect(parseStreamEvent('{"stage":"done","run_id":"0123456789ab"}')).toEqual({ stage: "done", run_id: "0123456789ab" });
  });

  it("returns null for garbage", () => {
    expect(parseStreamEvent("not json")).toBeNull();
    expect(parseStreamEvent("[1,2]")).toBeNull();
    expect(parseStreamEvent('{"no":"stage"}')).toBeNull();
    expect(parseStreamEvent('{"stage":"done"}')).toBeNull();
  });
});

describe("consumeCouncilStream", () => {
  it("reports progress events and resolves with the run id", async () => {
    const events: CouncilEvent[] = [];
    const runId = await consumeCouncilStream(
      streamOf([
        'data: {"stage":"answer","model":"fake:alpha","status":"start","round":0,"message":null}\r\n\r\n',
        'data: {"stage":"answer","model":"fake:alpha","status":"done","round":0,"message":null}\r\n',
        "\r\n: keep-alive\r\n\r\n",
        'data: {"stage":"done","run_id":"run42"}\r\n\r\n',
      ]),
      (e) => events.push(e),
    );
    expect(runId).toBe("run42");
    expect(events.map((e) => `${e.stage}:${e.status}`)).toEqual(["answer:start", "answer:done"]);
  });

  it("decodes multi-byte UTF-8 split across chunks", async () => {
    const bytes = new TextEncoder().encode(
      'data: {"stage":"answer","model":"fake:ğüşİ","status":"error","round":0,"message":"çözülemedi"}\n\ndata: {"stage":"done","run_id":"x"}\n\n',
    );
    const chunks: Uint8Array[] = [];
    for (let i = 0; i < bytes.length; i += 3) chunks.push(bytes.slice(i, i + 3));
    const events: CouncilEvent[] = [];
    await expect(consumeCouncilStream(streamOf(chunks), (e) => events.push(e))).resolves.toBe("x");
    expect(events[0].model).toBe("fake:ğüşİ");
    expect(events[0].message).toBe("çözülemedi");
  });

  it("rejects when the stream ends without a done event", async () => {
    const stream = streamOf(['data: {"stage":"answer","model":"fake:a","status":"start"}\n\n']);
    await expect(consumeCouncilStream(stream, () => {})).rejects.toThrow(/ended before/);
  });

  it("rejects with the server message on an error event", async () => {
    const stream = streamOf(['data: {"stage":"error","message":"all members failed"}\n\n']);
    await expect(consumeCouncilStream(stream, () => {})).rejects.toThrow("all members failed");
  });
});
