import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { askCouncil, errorMessage, getConfig, getRun, type CouncilConfigInfo } from "../api";
import { ProgressBoard } from "../components/ProgressBoard";
import { shortModel } from "../format";
import { applyEvent, emptyBoard, type BoardState } from "../progress";

type Phase = "idle" | "streaming" | "loading-run" | "failed" | "stopped";

const ROUND_CHOICES = [0, 1, 2] as const;

export function AskPage() {
  const navigate = useNavigate();
  const [config, setConfig] = useState<CouncilConfigInfo | null>(null);
  const [question, setQuestion] = useState("");
  const [rounds, setRounds] = useState(0);
  const [phase, setPhase] = useState<Phase>("idle");
  const [board, setBoard] = useState<BoardState>(() => emptyBoard());
  const [error, setError] = useState<string | null>(null);
  const [finishedRunId, setFinishedRunId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    getConfig()
      .then((c) => {
        if (cancelled) return;
        setConfig(c);
        setRounds(Math.min(2, Math.max(0, c.debate_rounds ?? 0)));
      })
      .catch(() => {
        // The board still fills in from the event stream without config.
      });
    return () => {
      cancelled = true;
      abortRef.current?.abort();
    };
  }, []);

  const busy = phase === "streaming" || phase === "loading-run";

  async function submit(e?: FormEvent) {
    e?.preventDefault();
    const q = question.trim();
    if (!q || busy) return;

    const controller = new AbortController();
    abortRef.current = controller;
    setBoard(emptyBoard(config?.members ?? []));
    setError(null);
    setFinishedRunId(null);
    setPhase("streaming");

    let runId: string | null = null;
    try {
      runId = await askCouncil(q, rounds, {
        signal: controller.signal,
        onEvent: (event) => setBoard((b) => applyEvent(b, event)),
      });
      setFinishedRunId(runId);
      setPhase("loading-run");
      const run = await getRun(runId);
      navigate(`/runs/${encodeURIComponent(run.id)}`, { state: { run } });
    } catch (err) {
      if (controller.signal.aborted) {
        setPhase("stopped");
        return;
      }
      setError(errorMessage(err));
      setPhase("failed");
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      void submit();
    }
  }

  const showBoard = phase !== "idle";

  return (
    <section className="page">
      <h1>Ask the council</h1>
      <p className="lede">
        Several models answer, review each other anonymously, and a chairman writes the final answer. You get to vote
        blind before the names are revealed.
      </p>

      <form className="ask-form card" onSubmit={submit}>
        <label htmlFor="question" className="sr-only">
          Question
        </label>
        <textarea
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="What should the council deliberate on?"
          rows={5}
          disabled={busy}
          required
        />

        <div className="ask-controls">
          <fieldset className="segmented" disabled={busy}>
            <legend>Debate rounds</legend>
            {ROUND_CHOICES.map((n) => (
              <label key={n} className={rounds === n ? "active" : undefined}>
                <input type="radio" name="rounds" value={n} checked={rounds === n} onChange={() => setRounds(n)} />
                {n}
              </label>
            ))}
          </fieldset>

          <div className="ask-buttons">
            {busy ? (
              <button type="button" className="btn" onClick={stop} disabled={phase !== "streaming"}>
                Stop
              </button>
            ) : null}
            <button type="submit" className="btn btn-primary" disabled={busy || !question.trim()}>
              {busy ? "Deliberating…" : "Ask the council"}
            </button>
          </div>
        </div>

        {config && (
          <p className="council-line muted">
            Council: {config.members.map(shortModel).join(", ")} · Chairman: {shortModel(config.chairman)}
          </p>
        )}
        <p className="hint muted">Ctrl+Enter to send.</p>
      </form>

      {showBoard && (
        <div className="card">
          <h2 className="card-title">
            {phase === "streaming" && "Council in session"}
            {phase === "loading-run" && "Done. Loading the run…"}
            {phase === "failed" && "The run did not finish"}
            {phase === "stopped" && "Stopped"}
          </h2>
          <ProgressBoard
            state={board}
            members={config?.members ?? []}
            chairman={config?.chairman ?? null}
            debateRounds={rounds}
          />
          {error && (
            <p className="alert alert-error" role="alert">
              {error}
              {finishedRunId && (
                <>
                  {" "}
                  <Link to={`/runs/${encodeURIComponent(finishedRunId)}`}>Open the run</Link>
                </>
              )}
            </p>
          )}
          {phase === "stopped" && (
            <p className="muted">
              You stopped following the stream. The server may still finish the run; check <Link to="/history">History</Link>.
            </p>
          )}
        </div>
      )}
    </section>
  );
}
