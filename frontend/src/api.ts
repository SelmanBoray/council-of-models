// Typed client for the council-of-models HTTP API.
// Shapes mirror src/council/models.py and the routes in src/council/web/server.py (see CONTRACT).

import { consumeCouncilStream } from "./sse";

export type ModelId = string;

export interface Answer {
  model: ModelId;
  text: string;
  latency_s: number;
  input_tokens?: number | null;
  output_tokens?: number | null;
  cost_usd?: number | null;
  error?: string | null;
}

export interface Review {
  reviewer: ModelId;
  /** De-anonymized model ids, best -> worst, including the reviewer's own answer. Empty when unparseable. */
  ranking: ModelId[];
  rationale: string;
  raw: string;
  error?: string | null;
}

export interface Synthesis {
  chairman: ModelId;
  text: string;
  latency_s: number;
  cost_usd?: number | null;
}

export interface CouncilRun {
  id: string;
  question: string;
  /** ISO 8601, UTC. */
  created_at: string;
  members: ModelId[];
  chairman: ModelId;
  answers: Answer[];
  reviews: Review[];
  synthesis: Synthesis | null;
  debate_rounds: number;
  /** One list of revised answers per debate round. */
  debate_answers: Answer[][];
  /** Anonymous label -> model id, e.g. {"Response A": "openrouter:openai/gpt-5.1"}. */
  labels: Record<string, ModelId>;
  /** Model id the human picked as best (blind vote), or null. */
  human_vote: ModelId | null;
  total_cost_usd: number | null;
}

export type Stage = "answer" | "debate" | "review" | "synthesis";
export type EventStatus = "start" | "done" | "error";

export interface CouncilEvent {
  stage: Stage;
  model: ModelId;
  status: EventStatus;
  round: number;
  message: string | null;
}

/** Final SSE event of POST /api/ask. */
export interface DoneEvent {
  stage: "done";
  run_id: string;
}

/** Not part of the contract; handled defensively in case the server aborts the stream with a reason. */
export interface StreamErrorEvent {
  stage: "error";
  message: string;
}

export type StreamEvent = CouncilEvent | DoneEvent | StreamErrorEvent;

/** Row of GET /api/runs. */
export interface RunSummary {
  id: string;
  question: string;
  created_at: string;
  council_ranking: ModelId[];
  human_vote: ModelId | null;
}

export interface LeaderboardRow {
  model: ModelId;
  mean_rank: number;
  wins: number;
  appearances: number;
}

export interface HumanVsCouncil {
  n: number;
  agreement_rate: number | null;
  per_model_human_wins: Record<ModelId, number>;
}

/** GET /api/stats (analytics.summary). */
export interface StatsSummary {
  n_runs: number;
  total_cost_usd: number | null;
  leaderboard: LeaderboardRow[];
  self_preference: Record<ModelId, number>;
  agreement_matrix: Record<ModelId, Record<ModelId, number>>;
  human_vs_council: HumanVsCouncil;
}

/** GET /api/config. */
export interface CouncilConfigInfo {
  members: ModelId[];
  chairman: ModelId;
  debate_rounds: number;
}

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function toApiError(res: Response): Promise<ApiError> {
  let detail = res.statusText || "Request failed";
  try {
    const text = await res.text();
    if (text) {
      try {
        const body = JSON.parse(text) as { detail?: unknown };
        if (typeof body.detail === "string") detail = body.detail;
        else if (body.detail !== undefined) detail = JSON.stringify(body.detail);
        else detail = text.slice(0, 300);
      } catch {
        detail = text.slice(0, 300);
      }
    }
  } catch {
    // keep the status text
  }
  return new ApiError(res.status, `${detail} (HTTP ${res.status})`);
}

async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (init.body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(path, { ...init, headers });
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as T;
}

/** Fill defaults so the UI never trips over a missing optional field. */
export function normalizeRun(run: CouncilRun): CouncilRun {
  return {
    ...run,
    members: run.members ?? [],
    answers: run.answers ?? [],
    reviews: run.reviews ?? [],
    synthesis: run.synthesis ?? null,
    debate_rounds: run.debate_rounds ?? 0,
    debate_answers: run.debate_answers ?? [],
    labels: run.labels ?? {},
    human_vote: run.human_vote ?? null,
    total_cost_usd: run.total_cost_usd ?? null,
  };
}

export function getConfig(): Promise<CouncilConfigInfo> {
  return requestJson<CouncilConfigInfo>("/api/config");
}

export async function getRun(id: string): Promise<CouncilRun> {
  return normalizeRun(await requestJson<CouncilRun>(`/api/runs/${encodeURIComponent(id)}`));
}

export function listRuns(limit = 50): Promise<RunSummary[]> {
  return requestJson<RunSummary[]>(`/api/runs?limit=${limit}`);
}

export async function voteRun(id: string, model: ModelId): Promise<CouncilRun> {
  const run = await requestJson<CouncilRun>(`/api/runs/${encodeURIComponent(id)}/vote`, {
    method: "POST",
    body: JSON.stringify({ model }),
  });
  return normalizeRun(run);
}

export function getStats(): Promise<StatsSummary> {
  return requestJson<StatsSummary>("/api/stats");
}

export interface AskOptions {
  signal?: AbortSignal;
  onEvent?: (event: CouncilEvent) => void;
}

/**
 * POST /api/ask and follow the Server-Sent Events stream (EventSource cannot POST, so this
 * uses fetch + ReadableStream). Resolves with the run id from the final `done` event.
 */
export async function askCouncil(question: string, debateRounds: number, options: AskOptions = {}): Promise<string> {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question, debate_rounds: debateRounds }),
    signal: options.signal,
  });
  if (!res.ok) throw await toApiError(res);
  if (!res.body) throw new Error("The server returned an empty response instead of an event stream.");
  return consumeCouncilStream(res.body, options.onEvent ?? (() => {}));
}

/** Human-readable message for any thrown value. */
export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}
