import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { errorMessage, listRuns, type RunSummary } from "../api";
import { formatDate, shortModel, truncate } from "../format";

export function HistoryPage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listRuns(50)
      .then((r) => !cancelled && setRuns(r))
      .catch((err) => !cancelled && setError(errorMessage(err)));
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="page">
      <h1>History</h1>
      {error && (
        <p className="alert alert-error" role="alert">
          Could not load runs: {error}
        </p>
      )}
      {!error && runs === null && <p className="muted">Loading…</p>}
      {runs && runs.length === 0 && (
        <div className="card empty">
          <p>No runs yet.</p>
          <Link className="btn btn-primary" to="/">
            Ask the council
          </Link>
        </div>
      )}
      {runs && runs.length > 0 && (
        <>
          <p className="muted small">
            For runs you have not voted on, the council&apos;s pick stays hidden so it cannot spoil the blind vote.
          </p>
          <div className="table-wrap">
            <table className="data-table history">
              <thead>
                <tr>
                  <th scope="col">Run</th>
                  <th scope="col">Question</th>
                  <th scope="col" className="hide-sm">
                    Created
                  </th>
                  <th scope="col">Council #1</th>
                  <th scope="col">Your vote</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => {
                  const top = run.council_ranking?.[0] ?? null;
                  const voted = run.human_vote != null;
                  const agree = voted && top !== null && top === run.human_vote;
                  return (
                    <tr key={run.id}>
                      <td className="nowrap">
                        <Link to={`/runs/${encodeURIComponent(run.id)}`}>
                          <code>{run.id}</code>
                        </Link>
                      </td>
                      <td className="question-cell" title={run.question}>
                        {truncate(run.question, 80)}
                        <span className="show-sm muted small">{formatDate(run.created_at)}</span>
                      </td>
                      <td className="nowrap hide-sm">{formatDate(run.created_at)}</td>
                      <td title={voted && top ? top : undefined}>
                        {!voted ? <span className="muted">hidden</span> : top ? shortModel(top) : "—"}
                      </td>
                      <td title={run.human_vote ?? undefined}>
                        {voted ? (
                          <>
                            {shortModel(run.human_vote as string)}{" "}
                            {top !== null && (
                              <span className={agree ? "verdict agree" : "verdict disagree"}>
                                {agree ? "agrees" : "differs"}
                              </span>
                            )}
                          </>
                        ) : (
                          <Link to={`/runs/${encodeURIComponent(run.id)}`}>vote</Link>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}
