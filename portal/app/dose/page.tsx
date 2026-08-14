import { InstrumentShell } from "../../components/InstrumentShell";

export default function DosePage() {
  return (
    <InstrumentShell active="dose">
      <main className="stage">
        <section className="module-page">
          <article className="panel">
            <h1>Dose</h1>
            <p>
              Ceiling-dose channel. How the hidden-norm cap changes with k. This
              module names the dose series; it does not reprint numeric claims.
            </p>
            <dl className="kv">
              <dt>channel</dt>
              <dd>ceiling-dose</dd>
              <dt>summary</dt>
              <dd className="mono">figs/summaries/A_gap_normcap.json</dd>
            </dl>
          </article>
        </section>
      </main>
    </InstrumentShell>
  );
}
