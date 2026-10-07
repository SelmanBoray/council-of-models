import type { ModelId } from "./api";

/** "openrouter:openai/gpt-5.1" -> "gpt-5.1", "fake:alpha" -> "alpha". */
export function shortModel(id: ModelId): string {
  const colon = id.indexOf(":");
  const rest = colon === -1 ? id : id.slice(colon + 1);
  const slash = rest.lastIndexOf("/");
  return slash === -1 ? rest : rest.slice(slash + 1);
}

/** Short display names for a set of ids; falls back to the full id where short names collide. */
export function shortNames(ids: ModelId[]): Map<ModelId, string> {
  const counts = new Map<string, number>();
  for (const id of ids) counts.set(shortModel(id), (counts.get(shortModel(id)) ?? 0) + 1);
  return new Map(ids.map((id) => [id, (counts.get(shortModel(id)) ?? 0) > 1 ? id : shortModel(id)]));
}

export function formatCost(usd: number | null | undefined): string {
  if (usd === null || usd === undefined || Number.isNaN(usd)) return "—";
  if (usd === 0) return "$0.00";
  return usd < 1 ? `$${usd.toFixed(4)}` : `$${usd.toFixed(2)}`;
}

export function formatSeconds(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "—";
  return `${seconds.toFixed(1)} s`;
}

export function formatDate(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

export function truncate(text: string, max: number): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > max ? `${clean.slice(0, max - 1)}…` : clean;
}

export function signed(value: number, digits = 2): string {
  const fixed = value.toFixed(digits);
  return value > 0 ? `+${fixed}` : fixed;
}
