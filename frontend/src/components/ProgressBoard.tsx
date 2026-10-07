import type { ModelId } from "../api";
import { shortNames } from "../format";
import { boardColumns, cellApplies, cellKey, cellStatus, type BoardState, type CellStatus } from "../progress";

const STATUS_TEXT: Record<CellStatus, string> = {
  pending: "pending",
  running: "running",
  done: "done",
  error: "error",
};

export function StatusDot({ status, title }: { status: CellStatus; title?: string }) {
  return <span className={`dot dot-${status}`} role="img" aria-label={STATUS_TEXT[status]} title={title ?? STATUS_TEXT[status]} />;
}

interface Props {
  state: BoardState;
  members: ModelId[];
  chairman: ModelId | null;
  debateRounds: number;
}

export function ProgressBoard({ state, members, chairman, debateRounds }: Props) {
  const models = [...new Set([...members, ...(chairman ? [chairman] : []), ...state.models])];
  const columns = boardColumns(state, debateRounds);
  const names = shortNames(models);

  if (models.length === 0) {
    return <p className="muted">Waiting for the first council member…</p>;
  }

  return (
    <div className="board">
      <div className="table-wrap">
        <table className="board-table">
          <thead>
            <tr>
              <th scope="col">Model</th>
              {columns.map((c) => (
                <th scope="col" key={`${c.stage}:${c.round}`}>
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={model}>
                <th scope="row" className="model-cell" title={model}>
                  {names.get(model)}
                  {model === chairman && <span className="tag">chair</span>}
                </th>
                {columns.map((c) => {
                  const key = cellKey(c.stage, c.round, model);
                  if (!cellApplies(state, c, model, members, chairman)) {
                    return (
                      <td key={key} className="na" aria-label="not applicable">
                        ·
                      </td>
                    );
                  }
                  const status = cellStatus(state, c, model);
                  const message = state.cells[key]?.message;
                  return (
                    <td key={key}>
                      <StatusDot status={status} title={message ? `${STATUS_TEXT[status]}: ${message}` : undefined} />
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="legend" aria-hidden="true">
        {(Object.keys(STATUS_TEXT) as CellStatus[]).map((s) => (
          <span key={s}>
            <span className={`dot dot-${s}`} /> {STATUS_TEXT[s]}
          </span>
        ))}
      </div>
      {state.errors.length > 0 && (
        <ul className="error-list">
          {state.errors.map((e, i) => (
            <li key={i}>
              <strong title={e.model}>{names.get(e.model) ?? e.model}</strong> · {e.column}: {e.text}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
