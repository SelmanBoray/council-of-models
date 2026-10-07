import { describe, expect, it } from "vitest";
import type { Answer, CouncilEvent } from "./api";
import { buildLabelMap, compareLabels } from "./blind";
import { applyEvent, boardColumns, cellStatus, emptyBoard } from "./progress";

function answer(model: string, error: string | null = null): Answer {
  return { model, text: "…", latency_s: 1, error };
}

describe("buildLabelMap", () => {
  it("inverts the server labels", () => {
    const map = buildLabelMap({
      id: "r1",
      labels: { "Response A": "fake:beta", "Response B": "fake:alpha" },
      answers: [answer("fake:alpha"), answer("fake:beta")],
      debate_answers: [],
    });
    expect(map.get("fake:beta")).toBe("Response A");
    expect(map.get("fake:alpha")).toBe("Response B");
  });

  it("labels unlabeled successful answers with free letters and skips failed ones", () => {
    const map = buildLabelMap({
      id: "r2",
      labels: { "Response A": "fake:a" },
      answers: [answer("fake:a"), answer("fake:b"), answer("fake:c"), answer("fake:dead", "timeout")],
      debate_answers: [],
    });
    expect(map.get("fake:a")).toBe("Response A");
    expect(new Set([map.get("fake:b"), map.get("fake:c")])).toEqual(new Set(["Response B", "Response C"]));
    expect(map.has("fake:dead")).toBe(false);
  });

  it("sorts labels naturally", () => {
    expect(["Response C", "Response AA", "Response A"].sort(compareLabels)).toEqual([
      "Response A",
      "Response C",
      "Response AA",
    ]);
  });
});

describe("progress board", () => {
  const ev = (partial: Partial<CouncilEvent>): CouncilEvent => ({
    stage: "answer",
    model: "fake:a",
    status: "start",
    round: 0,
    message: null,
    ...partial,
  });

  it("tracks status per stage and model", () => {
    let state = emptyBoard(["fake:a", "fake:b"]);
    state = applyEvent(state, ev({}));
    state = applyEvent(state, ev({ model: "fake:b", status: "error", message: "boom" }));
    const [answerCol] = boardColumns(state, 0);
    expect(cellStatus(state, answerCol, "fake:a")).toBe("running");
    expect(cellStatus(state, answerCol, "fake:b")).toBe("error");
    expect(state.errors).toEqual([{ model: "fake:b", column: "Answer", text: "boom" }]);
    state = applyEvent(state, ev({ status: "done" }));
    expect(cellStatus(state, answerCol, "fake:a")).toBe("done");
  });

  it("maps 1-based and 0-based debate rounds onto the same columns", () => {
    const oneBased = applyEvent(emptyBoard(), ev({ stage: "debate", round: 1 }));
    const zeroBased = applyEvent(emptyBoard(), ev({ stage: "debate", round: 0 }));
    for (const state of [oneBased, zeroBased]) {
      const cols = boardColumns(state, 2).map((c) => c.label);
      expect(cols).toEqual(["Answer", "Debate 1", "Debate 2", "Review", "Synthesis"]);
      expect(cellStatus(state, { stage: "debate", round: 1, label: "Debate 1" }, "fake:a")).toBe("running");
    }
  });
});
