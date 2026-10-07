import { useEffect, useMemo, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import { errorMessage, getRun, voteRun, type Answer, type CouncilRun, type ModelId, type Review } from "../api";
import { buildLabelMap, compareLabels } from "../blind";
import { Markdown } from "../components/Markdown";
import { formatCost, formatDate, formatSeconds } from "../format";
import { councilRankingRows, type RankingRow } from "../ranking";

type Tab = "answers" | "reviews" | "final";

export function RunPage() {
  const { id = "" } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const passed = (location.state as { run?: CouncilRun } | null)?.run;
  const [run, setRun] = useState<CouncilRun | null>(passed && passed.id === id ? passed : null);
  const [loadError, setLoadError] = useState<string | null>(null);

  // The Ask page hands over the freshly fetched run. Drop it from history state right away so a
  // page reload fetches the current version (e.g. with the vote) instead of reusing a stale copy.
  useEffect(() => {
    if (passed) navigate(location.pathname, { replace: true, state: null });
  }, [passed, navigate, location.pathname]);

  useEffect(() => {
    if (run && run.id === id) return;
    let cancelled = false;
    setRun(null);
    setLoadError(null);
    getRun(id)
      .then((r) => !cancelled && setRun(r))
      .catch((err) => !cancelled && setLoadError(errorMessage(err)));
    return () => {
      cancelled = true;
    };
    // Only refetch when the URL changes; a run handed over by the Ask page is used as-is.
  }, [id]);

  if (loadError) {
    return (
      <section className="page">
        <p className="alert alert-error" role="alert">
          Could not load run <code>{id}</code>: {loadError}
        </p>
        <Link to="/history">Back to history</Link>
      </section>
    );
  }
  if (!run) return <p className="page muted">Loading run…</p>;
  return <RunView key={run.id} run={run} onRunChange={setRun} />;
}

interface Naming {
  revealed: boolean;
  labelOf: Map<ModelId, string>;
  /** Display name: anonymous label while blind, model id once revealed. */
  name: (model: ModelId) => string;
}

function RunView({ run, onRunChange }: { run: CouncilRun; onRunChange: (run: CouncilRun) => void }) {
  const [tab, setTab] = useState<Tab>("answers");
  const [peeked, setPeeked] = useState(false);
  const [voting, setVoting] = useState<ModelId | null>(null);
  const [voteError, setVoteError] = useState<string | null>(null);

  const voted = run.human_vote !== null;
  const revealed = voted || peeked;
  const labelOf = useMemo(() => buildLabelMap(run), [run]);
  const rows = useMemo(() => councilRankingRows(run), [run]);
  const councilTop = rows[0] && rows[0].meanPosition !== null ? rows[0].model : null;

  const naming: Naming = {
    revealed,
    labelOf,
    name: (model) => (revealed ? model : (labelOf.get(model) ?? "Hidden member")),
  };

  async function vote(model: ModelId) {
    setVoting(model);
    setVoteError(null);
    try {
      onRunChange(await voteRun(run.id, model));
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      setVoteError(errorMessage(err));
    } finally {
      setVoting(null);
    }
  }

  return (
    <section className="page run">
      <header className="run-header">
        <p className="eyebrow">
          Run <code>{run.id}</code> · {formatDate(run.created_at)}
          {run.debate_rounds > 0 && ` · ${run.debate_rounds} debate round${run.debate_rounds > 1 ? "s" : ""}`}
        </p>
        <h1 className="question">{run.question}</h1>
        <p className="muted small">
          {revealed ? (
            <>
              Council: {run.members.join(", ")} · Chairman: {run.chairman}
            </>
          ) : (
            <>{run.members.length} council members · names hidden</>
          )}
        </p>
      </header>

      <VoteBanner
        run={run}
        naming={naming}
        councilTop={councilTop}
        peeked={peeked && !voted}
        onPeek={() => setPeeked(true)}
      />
      {voteError && (
        <p className="alert alert-error" role="alert">
          Vote failed: {voteError}
        </p>
      )}

      <div className="tabs" role="tablist" aria-label="Run stages">
        {(
          [
            ["answers", "Answers"],
            ["reviews", "Reviews"],
            ["final", "Final"],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            role="tab"
            type="button"
            aria-selected={tab === key}
            className={tab === key ? "tab active" : "tab"}
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </div>

      <div role="tabpanel">
        {tab === "answers" && (
          <AnswersTab
            run={run}
            naming={naming}
            councilTop={councilTop}
            canVote={!revealed}
            voting={voting}
            onVote={vote}
          />
        )}
        {tab === "reviews" && <ReviewsTab run={run} naming={naming} rows={rows} />}
        {tab === "final" && (
          <FinalTab run={run} revealed={revealed} onGoVote={() => setTab("answers")} onPeek={() => setPeeked(true)} />
        )}
      </div>
    </section>
  );
}

function VoteBanner({
  run,
  naming,
  councilTop,
  peeked,
  onPeek,
}: {
  run: CouncilRun;
  naming: Naming;
  councilTop: ModelId | null;
  peeked: boolean;
  onPeek: () => void;
}) {
  if (run.human_vote) {
    const pick = run.human_vote;
    const agrees = councilTop !== null && councilTop === pick;
    return (
      <div className="banner banner-revealed" role="status">
        <span className="pill pill-revealed">Revealed</span>
        <div>
          <p>
            Your pick: <strong>{naming.labelOf.get(pick) ?? "—"}</strong> = <strong className="model">{pick}</strong>
          </p>
          {councilTop ? (
            <p>
              Council&apos;s #1: <strong>{naming.labelOf.get(councilTop) ?? "—"}</strong> ={" "}
              <strong className="model">{councilTop}</strong>{" "}
              <span className={agrees ? "verdict agree" : "verdict disagree"}>
                {agrees ? "You agree with the council" : "You disagree with the council"}
              </span>
            </p>
          ) : (
            <p className="muted">The council produced no usable ranking for this run.</p>
          )}
        </div>
      </div>
    );
  }
  if (peeked) {
    return (
      <div className="banner banner-peeked" role="status">
        <span className="pill">Names shown</span>
        <p>You revealed the names without voting, so this run stays out of the human-vs-council statistics.</p>
      </div>
    );
  }
  return (
    <div className="banner banner-blind">
      <span className="pill pill-blind">Blind vote</span>
      <p>
        Model names are hidden. Read the answers and pick the one you think is best; names are revealed after you vote.
      </p>
      <button type="button" className="btn btn-ghost small" onClick={onPeek}>
        Skip vote &amp; reveal
      </button>
    </div>
  );
}

function AnswersTab({
  run,
  naming,
  councilTop,
  canVote,
  voting,
  onVote,
}: {
  run: CouncilRun;
  naming: Naming;
  councilTop: ModelId | null;
  canVote: boolean;
  voting: ModelId | null;
  onVote: (model: ModelId) => void;
}) {
  const rounds = [
    { label: "Initial answers", answers: run.answers },
    ...run.debate_answers.map((answers, i) => ({ label: `Debate round ${i + 1}`, answers })),
  ];
  const [roundIndex, setRoundIndex] = useState(rounds.length - 1);
  const current = rounds[Math.min(roundIndex, rounds.length - 1)];

  const ok = current.answers.filter((a) => !a.error);
  const failed = current.answers.filter((a) => a.error);
  // Always in label order, so revealing names does not reshuffle the cards.
  const sorted = [...ok].sort((a, b) =>
    compareLabels(naming.labelOf.get(a.model) ?? "~", naming.labelOf.get(b.model) ?? "~"),
  );

  return (
    <div>
      {rounds.length > 1 && (
        <>
          <div className="subtabs" role="tablist" aria-label="Answer rounds">
            {rounds.map((r, i) => (
              <button
                key={r.label}
                type="button"
                role="tab"
                aria-selected={i === roundIndex}
                className={i === roundIndex ? "subtab active" : "subtab"}
                onClick={() => setRoundIndex(i)}
              >
                {r.label}
              </button>
            ))}
          </div>
          <p className="muted small">The reviewers ranked the answers from the last debate round.</p>
        </>
      )}

      {sorted.length === 0 && <p className="muted">No successful answers in this round.</p>}

      <div className="answer-list">
        {sorted.map((answer) => (
          <AnswerCard
            key={answer.model}
            answer={answer}
            naming={naming}
            isHumanPick={run.human_vote === answer.model}
            isCouncilTop={naming.revealed && councilTop === answer.model}
            canVote={canVote}
            voting={voting}
            onVote={onVote}
          />
        ))}
      </div>

      {failed.length > 0 &&
        (naming.revealed ? (
          <div className="failed-list">
            {failed.map((a) => (
              <div key={a.model} className="card card-error">
                <strong className="model">{a.model}</strong> failed: {a.error}
              </div>
            ))}
          </div>
        ) : (
          <p className="muted small">
            {failed.length} member{failed.length > 1 ? "s" : ""} failed to answer in this round (hidden until reveal).
          </p>
        ))}
    </div>
  );
}

function AnswerCard({
  answer,
  naming,
  isHumanPick,
  isCouncilTop,
  canVote,
  voting,
  onVote,
}: {
  answer: Answer;
  naming: Naming;
  isHumanPick: boolean;
  isCouncilTop: boolean;
  canVote: boolean;
  voting: ModelId | null;
  onVote: (model: ModelId) => void;
}) {
  const label = naming.labelOf.get(answer.model) ?? "Unlabeled answer";
  const tokens =
    answer.input_tokens != null || answer.output_tokens != null
      ? `${answer.input_tokens ?? "?"} in / ${answer.output_tokens ?? "?"} out tokens`
      : null;

  return (
    <article className={isHumanPick ? "card answer picked" : "card answer"}>
      <header className="answer-head">
        <div>
          <h3 className="answer-label">{label}</h3>
          {naming.revealed && <p className="model">{answer.model}</p>}
        </div>
        <div className="badges">
          {isHumanPick && <span className="badge badge-pick">Your pick</span>}
          {isCouncilTop && <span className="badge badge-council">Council #1</span>}
        </div>
      </header>

      {naming.revealed && (
        // Latency, tokens and cost can hint at the model, so they stay hidden while blind.
        <p className="meta">
          {formatSeconds(answer.latency_s)}
          {tokens && ` · ${tokens}`}
          {answer.cost_usd != null && ` · ${formatCost(answer.cost_usd)}`}
        </p>
      )}

      <Markdown text={answer.text} />

      {canVote && (
        <footer className="answer-foot">
          <button
            type="button"
            className="btn btn-primary"
            disabled={voting !== null}
            onClick={() => onVote(answer.model)}
          >
            {voting === answer.model ? "Recording vote…" : "This one is best"}
          </button>
        </footer>
      )}
    </article>
  );
}

function ReviewsTab({ run, naming, rows }: { run: CouncilRun; naming: Naming; rows: RankingRow[] }) {
  return (
    <div>
      {!naming.revealed && (
        <p className="alert alert-info">
          Heads-up: reading the council&apos;s verdict before you vote can anchor your judgment. Reviewers are shown by
          the label of their own answer.
        </p>
      )}

      <section className="card">
        <h2 className="card-title">Aggregate council ranking</h2>
        <p className="muted small">Mean position across all usable rankings (1 = best). Unranked answers go last.</p>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Answer</th>
                <th scope="col" className="num">
                  Mean position
                </th>
                <th scope="col" className="num">
                  Rankings
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={row.model}>
                  <td>{row.meanPosition === null ? "—" : i + 1}</td>
                  <td>
                    <span className={naming.revealed ? "model" : "nowrap"}>{naming.name(row.model)}</span>
                    {naming.revealed && naming.labelOf.has(row.model) && (
                      <span className="muted small"> ({naming.labelOf.get(row.model)})</span>
                    )}
                    {run.human_vote === row.model && <span className="badge badge-pick">Your pick</span>}
                  </td>
                  <td className="num">{row.meanPosition === null ? "unranked" : row.meanPosition.toFixed(2)}</td>
                  <td className="num">{row.votes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {run.reviews.length === 0 && <p className="muted">No reviews in this run.</p>}
      <div className="review-list">
        {run.reviews.map((review, i) => (
          <ReviewCard key={`${review.reviewer}:${i}`} review={review} index={i} naming={naming} humanVote={run.human_vote} />
        ))}
      </div>
    </div>
  );
}

function ReviewCard({
  review,
  index,
  naming,
  humanVote,
}: {
  review: Review;
  index: number;
  naming: Naming;
  humanVote: ModelId | null;
}) {
  const ownLabel = naming.labelOf.get(review.reviewer);
  const reviewerName = naming.revealed
    ? review.reviewer
    : ownLabel
      ? `Author of ${ownLabel}`
      : `Reviewer ${index + 1}`;

  return (
    <article className="card review">
      <header className="answer-head">
        <div>
          <p className="eyebrow">Reviewer</p>
          <h3 className={naming.revealed ? "model" : "answer-label"}>{reviewerName}</h3>
        </div>
        {review.error && <span className="badge badge-error">Unusable ranking</span>}
      </header>

      {review.error && <p className="alert alert-error">{review.error}</p>}

      {review.ranking.length > 0 && (
        <ol className="ranking">
          {review.ranking.map((model) => (
            <li key={model}>
              <span className={naming.revealed ? "model" : undefined}>{naming.name(model)}</span>
              {model === review.reviewer && <span className="tag">own answer</span>}
              {humanVote === model && <span className="badge badge-pick">Your pick</span>}
            </li>
          ))}
        </ol>
      )}

      {review.rationale && (
        <details>
          <summary>Rationale</summary>
          <Markdown text={review.rationale} />
        </details>
      )}
      {review.error && review.raw && (
        <details>
          <summary>Raw output</summary>
          <pre className="raw">{review.raw}</pre>
        </details>
      )}
    </article>
  );
}

function FinalTab({
  run,
  revealed,
  onGoVote,
  onPeek,
}: {
  run: CouncilRun;
  revealed: boolean;
  onGoVote: () => void;
  onPeek: () => void;
}) {
  if (!revealed) {
    return (
      <div className="card locked">
        <h2 className="card-title">Final answer is locked</h2>
        <p>The chairman&apos;s synthesis refers to the models by name, so it stays hidden until you cast your blind vote.</p>
        <div className="row-buttons">
          <button type="button" className="btn btn-primary" onClick={onGoVote}>
            Go vote
          </button>
          <button type="button" className="btn btn-ghost" onClick={onPeek}>
            Reveal without voting
          </button>
        </div>
      </div>
    );
  }

  const synthesis = run.synthesis;
  return (
    <article className="card final">
      <header className="answer-head">
        <div>
          <p className="eyebrow">Chairman</p>
          <h3 className="model">{synthesis?.chairman ?? run.chairman}</h3>
        </div>
      </header>
      <p className="meta">
        {synthesis && formatSeconds(synthesis.latency_s)}
        {synthesis?.cost_usd != null && ` · synthesis ${formatCost(synthesis.cost_usd)}`}
        {run.total_cost_usd != null && ` · total run cost ${formatCost(run.total_cost_usd)}`}
      </p>
      {synthesis ? (
        <Markdown text={synthesis.text} />
      ) : (
        <p className="muted">No synthesis was produced for this run (the chairman failed or the run is incomplete).</p>
      )}
    </article>
  );
}
