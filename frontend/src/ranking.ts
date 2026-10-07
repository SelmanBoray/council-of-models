// Client-side mirror of CouncilRun.council_ranking() from the contract:
// mean 1-based position across non-empty rankings, best first; models never ranked go last;
// ties (and the unranked tail) are broken alphabetically by model id.

import type { CouncilRun, ModelId } from "./api";

export interface RankingRow {
  model: ModelId;
  /** Mean 1-based position across the reviews that ranked this model; null if nobody ranked it. */
  meanPosition: number | null;
  /** Number of non-empty rankings that included the model. */
  votes: number;
}

/** Plain code-unit comparison, matching Python's default string ordering for ASCII ids. */
export function compareIds(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

export function councilRankingRows(run: Pick<CouncilRun, "members" | "reviews">): RankingRow[] {
  const positions = new Map<ModelId, number[]>();
  for (const review of run.reviews) {
    if (!review.ranking || review.ranking.length === 0) continue;
    review.ranking.forEach((model, index) => {
      const list = positions.get(model);
      if (list) list.push(index + 1);
      else positions.set(model, [index + 1]);
    });
  }

  const models = new Set<ModelId>([...run.members, ...positions.keys()]);
  const rows: RankingRow[] = [...models].map((model) => {
    const list = positions.get(model);
    return {
      model,
      meanPosition: list ? list.reduce((sum, p) => sum + p, 0) / list.length : null,
      votes: list?.length ?? 0,
    };
  });

  rows.sort((a, b) => {
    if (a.meanPosition === null || b.meanPosition === null) {
      if (a.meanPosition !== b.meanPosition) return a.meanPosition === null ? 1 : -1;
    } else if (a.meanPosition !== b.meanPosition) {
      return a.meanPosition - b.meanPosition;
    }
    return compareIds(a.model, b.model);
  });
  return rows;
}

export function councilRanking(run: Pick<CouncilRun, "members" | "reviews">): ModelId[] {
  return councilRankingRows(run).map((row) => row.model);
}

/** The council's #1, or null when no review produced a ranking. */
export function councilTopPick(run: Pick<CouncilRun, "members" | "reviews">): ModelId | null {
  const first = councilRankingRows(run)[0];
  return first && first.meanPosition !== null ? first.model : null;
}
