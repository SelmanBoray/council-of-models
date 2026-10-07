// State of the live progress board on the Ask page: one cell per (stage, round, model).

import type { CouncilEvent, ModelId, Stage } from "./api";

export type CellStatus = "pending" | "running" | "done" | "error";

export interface Cell {
  status: CellStatus;
  message: string | null;
}

export interface LogLine {
  model: ModelId;
  column: string;
  text: string;
}

export interface BoardState {
  cells: Record<string, Cell>;
  /** Models in order of first appearance. */
  models: ModelId[];
  /** Set when a debate event arrives with round 0, i.e. the server numbers debate rounds from 0. */
  debateZeroBased: boolean;
  /** Highest (1-based) debate round seen so far. */
  maxDebateRound: number;
  /** Error messages, shown under the board. */
  errors: LogLine[];
}

export interface Column {
  stage: Stage;
  round: number;
  label: string;
}

const STAGES: readonly Stage[] = ["answer", "debate", "review", "synthesis"];

export function cellKey(stage: Stage, round: number, model: ModelId): string {
  return `${stage}:${round}:${model}`;
}

export function emptyBoard(models: ModelId[] = []): BoardState {
  return { cells: {}, models: [...new Set(models)], debateZeroBased: false, maxDebateRound: 0, errors: [] };
}

function columnLabel(stage: Stage, round: number): string {
  switch (stage) {
    case "answer":
      return "Answer";
    case "debate":
      return `Debate ${round}`;
    case "review":
      return "Review";
    case "synthesis":
      return "Synthesis";
  }
}

export function applyEvent(state: BoardState, event: CouncilEvent): BoardState {
  if (!STAGES.includes(event.stage) || !event.model) return state;

  let debateZeroBased = state.debateZeroBased;
  let round = 0;
  if (event.stage === "debate") {
    if (event.round === 0) debateZeroBased = true;
    round = event.round + (debateZeroBased ? 1 : 0);
  }

  const status: CellStatus = event.status === "done" ? "done" : event.status === "error" ? "error" : "running";
  const key = cellKey(event.stage, round, event.model);
  const previous = state.cells[key];
  const cell: Cell = { status, message: event.message ?? previous?.message ?? null };

  return {
    cells: { ...state.cells, [key]: cell },
    models: state.models.includes(event.model) ? state.models : [...state.models, event.model],
    debateZeroBased,
    maxDebateRound: Math.max(state.maxDebateRound, round),
    errors:
      status === "error"
        ? [
            ...state.errors,
            { model: event.model, column: columnLabel(event.stage, round), text: event.message ?? "failed" },
          ]
        : state.errors,
  };
}

/** Columns: Answer, Debate 1..N, Review, Synthesis. */
export function boardColumns(state: BoardState, expectedDebateRounds: number): Column[] {
  const debates = Math.max(expectedDebateRounds, state.maxDebateRound);
  const columns: Column[] = [{ stage: "answer", round: 0, label: columnLabel("answer", 0) }];
  for (let r = 1; r <= debates; r++) columns.push({ stage: "debate", round: r, label: columnLabel("debate", r) });
  columns.push({ stage: "review", round: 0, label: columnLabel("review", 0) });
  columns.push({ stage: "synthesis", round: 0, label: columnLabel("synthesis", 0) });
  return columns;
}

/**
 * Whether a model takes part in a column. Synthesis belongs to the chairman only; a chairman that is
 * not a member only synthesizes. Without config, every cell that has seen an event applies.
 */
export function cellApplies(
  state: BoardState,
  column: Column,
  model: ModelId,
  members: ModelId[],
  chairman: ModelId | null,
): boolean {
  if (state.cells[cellKey(column.stage, column.round, model)]) return true;
  if (column.stage === "synthesis") return chairman !== null && model === chairman;
  return members.length === 0 || members.includes(model);
}

export function cellStatus(state: BoardState, column: Column, model: ModelId): CellStatus {
  return state.cells[cellKey(column.stage, column.round, model)]?.status ?? "pending";
}
