import { useEffect, useState, type CSSProperties } from "react";
import { Link } from "react-router-dom";
import { errorMessage, getStats, type ModelId, type StatsSummary } from "../api";
import { formatCost, shortNames, signed } from "../format";
import { compareIds } from "../ranking";

export function StatsPage() {
  const [stats, setStats] = useState<StatsSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getStats()
      .then((s) => !cancelled && setStats(s))
      .catch((err) => !cancelled && setError(errorMessage(err)));
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <section className="page">
        <h1>Stats</h1>
        <p className="alert alert-error" role="alert">
          Could not load stats: {error}
        </p>
      </section>
    );
  }
  if (!stats) return <p className="page muted">Loading stats…</p>;

  if (!stats.n_runs) {
    return (
      <section className="page">
        <h1>Stats</h1>
        <div className="card empty">
          <p>No runs yet. Judge-bias analytics appear once the council has deliberated a few times.</p>
          <Link className="btn btn-primary" to="/">
            Ask the council
          </Link>
        </div>
      </section>
    );
  }

  const hvc = stats.human_vs_council ?? { n: 0, agreement_rate: null, per_model_human_wins: {} };
  const humanWins = hvc.per_model_human_wins ?? {};

  return (
    <section className="page stats">
      <h1>Stats</h1>

      <div className="tiles">
        <div className="tile">
          <span className="tile-label">Runs</span>
          <span className="tile-value">{stats.n_runs}</span>
        </div>
        <div className="tile">
          <span className="tile-label">Blind votes</span>
          <span className="tile-value">{hvc.n}</span>
        </div>
        <div className="tile">
          <span className="tile-label">Total cost</span>
          <span className="tile-value">{formatCost(stats.total_cost_usd)}</span>
        </div>
      </div>

      <section className="card">
        <h2 className="card-title">Human vs council</h2>
        <HumanVsCouncilPanel n={hvc.n} rate={hvc.agreement_rate} />
      </section>

      <section className="card">
        <h2 className="card-title">Leaderboard (peer review)</h2>
        <p className="muted small">Mean position given by council reviewers across all runs; 1 is best.</p>
        <Leaderboard stats={stats} humanWins={humanWins} />
      </section>

      <section className="card">
        <h2 className="card-title">Self-preference</h2>
        <p className="muted small">
          Rank others give a model minus the rank it gives itself. Positive means it rates its own answer better than its
          peers do; 0 means no self-bias.
        </p>
        <SelfPreferenceBars data={stats.self_preference ?? {}} />
      </section>

      <section className="card">
        <h2 className="card-title">Reviewer agreement</h2>
        <p className="muted small">
          Kendall&apos;s tau between two reviewers&apos; rankings, averaged over runs both reviewed. 1 = identical order,
          −1 = reversed.
        </p>
        <AgreementHeatmap matrix={stats.agreement_matrix ?? {}} />
      </section>
    </section>
  );
}

function HumanVsCouncilPanel({ n, rate }: { n: number; rate: number | null }) {
  if (!n || rate === null || rate === undefined) {
    return (
      <p className="muted">
        No blind votes yet. Open a run, pick the best anonymous answer, and your judgment is compared with the
        council&apos;s.
      </p>
    );
  }
  return (
    <div className="hero-stat">
      <span className="hero-number">{Math.round(rate * 100)}%</span>
      <span className="hero-caption">
        of your blind picks matched the council&apos;s #1
        <br />
        <span className="muted">
          n = {n} voted run{n === 1 ? "" : "s"}
        </span>
      </span>
    </div>
  );
}

function Leaderboard({ stats, humanWins }: { stats: StatsSummary; humanWins: Record<ModelId, number> }) {
  const rows = stats.leaderboard ?? [];
  if (rows.length === 0) return <p className="muted">No rankings yet.</p>;
  const names = shortNames(rows.map((r) => r.model));
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th scope="col">#</th>
            <th scope="col">Model</th>
            <th scope="col" className="num">
              Mean rank
            </th>
            <th scope="col" className="num">
              Council wins
            </th>
            <th scope="col" className="num">
              Your picks
            </th>
            <th scope="col" className="num hide-sm" title="Number of reviews that ranked this model">
              Reviews
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={row.model}>
              <td>{i + 1}</td>
              <td className="model" title={row.model}>
                {names.get(row.model)}
              </td>
              <td className="num">{row.mean_rank.toFixed(2)}</td>
              <td className="num">{row.wins}</td>
              <td className="num">{humanWins[row.model] ?? 0}</td>
              <td className="num hide-sm">{row.appearances}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SelfPreferenceBars({ data }: { data: Record<ModelId, number> }) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1] || compareIds(a[0], b[0]));
  if (entries.length === 0) return <p className="muted">Not enough reviews yet.</p>;
  const names = shortNames(entries.map(([m]) => m));
  // At least one rank position each side, so small values are not blown up to full width.
  const scale = Math.max(1, ...entries.map(([, v]) => Math.abs(v)));

  return (
    <div className="sp">
      <div className="sp-row sp-head" aria-hidden="true">
        <span />
        <span className="sp-head-labels">
          <span>← self-critical</span>
          <span>self-favoring →</span>
        </span>
        <span />
      </div>
      {entries.map(([model, value]) => {
        const width = (Math.abs(value) / scale) * 50;
        const style: CSSProperties = value >= 0 ? { left: "50%", width: `${width}%` } : { left: `${50 - width}%`, width: `${width}%` };
        return (
          <div className="sp-row" key={model} title={`${model}: ${signed(value)} positions`}>
            <span className="sp-name">{names.get(model)}</span>
            <div className="sp-track">
              <span className="sp-zero" />
              <span className={value >= 0 ? "sp-bar pos" : "sp-bar neg"} style={style} />
            </div>
            <span className="sp-value">{signed(value)}</span>
          </div>
        );
      })}
    </div>
  );
}

/** Diverging fill: blue for agreement, red for disagreement, neutral gray at 0. */
function heatStyle(value: number): CSSProperties {
  const v = Math.max(-1, Math.min(1, value));
  const pct = Math.round(Math.abs(v) * 100);
  const pole = v >= 0 ? "var(--heat-pos)" : "var(--heat-neg)";
  return {
    background: `color-mix(in srgb, ${pole} ${pct}%, var(--heat-zero))`,
    color: Math.abs(v) >= 0.55 ? "#ffffff" : undefined,
  };
}

function AgreementHeatmap({ matrix }: { matrix: Record<ModelId, Record<ModelId, number>> }) {
  const ids = [...new Set([...Object.keys(matrix), ...Object.values(matrix).flatMap((row) => Object.keys(row ?? {}))])].sort(
    compareIds,
  );
  if (ids.length === 0) return <p className="muted">Not enough reviews yet.</p>;
  const names = shortNames(ids);
  const gridStyle = { "--hm-n": ids.length } as CSSProperties;

  return (
    <div>
      <div className="table-wrap">
        <div className="heatmap" style={gridStyle} role="table" aria-label="Reviewer agreement matrix">
          <div role="row" className="hm-contents">
            <span role="columnheader" className="hm-corner" />
            {ids.map((col) => (
              <span role="columnheader" key={col} className="hm-col" title={col}>
                <span>{names.get(col)}</span>
              </span>
            ))}
          </div>
          {ids.map((row) => (
            <div role="row" className="hm-contents" key={row}>
              <span role="rowheader" className="hm-row" title={row}>
                {names.get(row)}
              </span>
              {ids.map((col) => {
                const value = matrix[row]?.[col];
                const has = typeof value === "number" && !Number.isNaN(value);
                return (
                  <span
                    key={col}
                    role="cell"
                    className={has ? "hm-cell" : "hm-cell hm-empty"}
                    style={has ? heatStyle(value) : undefined}
                    title={`${row} vs ${col}: ${has ? value.toFixed(2) : "no shared runs"}`}
                  >
                    {has ? value.toFixed(2) : "·"}
                  </span>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      <div className="heat-legend small muted" aria-hidden="true">
        <span>−1 disagree</span>
        <span className="heat-ramp" />
        <span>+1 agree</span>
      </div>
    </div>
  );
}
