// Anonymous labels for the blind vote. The server assigns them per run ("Response A" -> model id);
// this module inverts that mapping and covers models the server did not label.

import type { CouncilRun, ModelId } from "./api";

function labelName(index: number): string {
  return index < 26 ? `Response ${String.fromCharCode(65 + index)}` : `Response ${index + 1}`;
}

/** FNV-1a: a stable pseudo-random order that does not follow member order. */
function hash(text: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Compare labels so "Response B" < "Response AA" and "Response 9" < "Response 10". */
export function compareLabels(a: string, b: string): number {
  return a.length - b.length || (a < b ? -1 : a > b ? 1 : 0);
}

/**
 * model id -> anonymous label. Uses `run.labels`; any model that produced an answer (initial or
 * debate) but has no server label gets the next free letter, in an order derived from the run id
 * so that it does not leak the configured member order.
 */
export function buildLabelMap(run: Pick<CouncilRun, "id" | "labels" | "answers" | "debate_answers">): Map<ModelId, string> {
  const byModel = new Map<ModelId, string>();
  const used = new Set<string>();
  for (const [label, model] of Object.entries(run.labels ?? {})) {
    if (!byModel.has(model)) byModel.set(model, label);
    used.add(label);
  }

  const answered = [run.answers, ...(run.debate_answers ?? [])]
    .flat()
    .filter((a) => !a.error)
    .map((a) => a.model);
  const missing = [...new Set(answered)].filter((m) => !byModel.has(m));
  missing.sort((a, b) => hash(`${run.id}:${a}`) - hash(`${run.id}:${b}`));

  let next = 0;
  for (const model of missing) {
    while (used.has(labelName(next))) next++;
    const label = labelName(next);
    used.add(label);
    byModel.set(model, label);
  }
  return byModel;
}
