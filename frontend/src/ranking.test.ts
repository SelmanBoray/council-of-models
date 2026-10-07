import { describe, expect, it } from "vitest";
import type { Review } from "./api";
import { councilRanking, councilRankingRows, councilTopPick } from "./ranking";

function review(reviewer: string, ranking: string[], error: string | null = null): Review {
  return { reviewer, ranking, rationale: "", raw: "", error };
}

describe("councilRanking", () => {
  it("orders by mean 1-based position across reviews", () => {
    const run = {
      members: ["fake:alpha", "fake:beta", "fake:gamma"],
      reviews: [
        review("fake:alpha", ["fake:alpha", "fake:beta", "fake:gamma"]), // alpha 1, beta 2, gamma 3
        review("fake:beta", ["fake:beta", "fake:alpha", "fake:gamma"]), // beta 1, alpha 2, gamma 3
        review("fake:gamma", ["fake:beta", "fake:gamma", "fake:alpha"]), // beta 1, gamma 2, alpha 3
      ],
    };
    expect(councilRanking(run)).toEqual(["fake:beta", "fake:alpha", "fake:gamma"]);
    const rows = councilRankingRows(run);
    expect(rows.map((r) => r.meanPosition)).toEqual([4 / 3, 2, 8 / 3]);
    expect(rows.every((r) => r.votes === 3)).toBe(true);
  });

  it("breaks ties alphabetically by model id", () => {
    const run = {
      members: ["openrouter:z-ai/glm", "anthropic:claude", "openai:gpt"],
      reviews: [
        review("openai:gpt", ["openrouter:z-ai/glm", "anthropic:claude", "openai:gpt"]),
        review("anthropic:claude", ["anthropic:claude", "openrouter:z-ai/glm", "openai:gpt"]),
      ],
    };
    // glm and claude both average 1.5 -> alphabetical: "anthropic:..." < "openrouter:..."
    expect(councilRanking(run)).toEqual(["anthropic:claude", "openrouter:z-ai/glm", "openai:gpt"]);
  });

  it("ignores empty (failed) rankings instead of counting them", () => {
    const run = {
      members: ["fake:a", "fake:b"],
      reviews: [review("fake:a", ["fake:b", "fake:a"]), review("fake:b", [], "unparseable ranking")],
    };
    const rows = councilRankingRows(run);
    expect(rows.map((r) => r.model)).toEqual(["fake:b", "fake:a"]);
    expect(rows.map((r) => r.votes)).toEqual([1, 1]);
  });

  it("puts never-ranked members last, alphabetically", () => {
    const run = {
      members: ["fake:zeta", "fake:mid", "fake:alpha", "fake:top"],
      reviews: [review("fake:top", ["fake:mid", "fake:top"]), review("fake:mid", ["fake:top", "fake:mid"])],
    };
    // mid and top tie at 1.5 -> alphabetical; alpha and zeta were never ranked (e.g. their answers failed).
    expect(councilRanking(run)).toEqual(["fake:mid", "fake:top", "fake:alpha", "fake:zeta"]);
    const rows = councilRankingRows(run);
    expect(rows[2]).toEqual({ model: "fake:alpha", meanPosition: null, votes: 0 });
  });

  it("includes ranked models that are missing from members", () => {
    const run = { members: ["fake:a"], reviews: [review("fake:a", ["fake:x", "fake:a"])] };
    expect(councilRanking(run)).toEqual(["fake:x", "fake:a"]);
  });

  it("averages only over the rankings that include a model", () => {
    const run = {
      members: ["fake:a", "fake:b", "fake:c"],
      reviews: [
        review("fake:a", ["fake:a", "fake:b", "fake:c"]), // c: 3
        review("fake:b", ["fake:b", "fake:a"]), // c absent
      ],
    };
    const rows = councilRankingRows(run);
    expect(rows.find((r) => r.model === "fake:c")).toEqual({ model: "fake:c", meanPosition: 3, votes: 1 });
    expect(rows.map((r) => r.model)).toEqual(["fake:a", "fake:b", "fake:c"]); // a 1.5, b 1.5 (tie -> a), c 3
  });

  it("has no top pick when nobody produced a ranking", () => {
    const run = { members: ["fake:b", "fake:a"], reviews: [review("fake:a", [], "provider error")] };
    expect(councilRanking(run)).toEqual(["fake:a", "fake:b"]);
    expect(councilTopPick(run)).toBeNull();
  });
});
