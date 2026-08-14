import { InstrumentShell } from "../../components/InstrumentShell";

export default function DosePage() {
  return (
    <InstrumentShell active="dose">
      <main className="stage">
        <section className="module-page">
          <article className="panel">
            <h1>Dose</h1>
            <p>
              Architecture slot for the ceiling-dose channel. Numeric series live in
              the warehouse summaries; this door only names the files.
            </p>
            <dl className="kv">
              <dt>index</dt>
              <dd>papers/FIGURE-INDEX.json</dd>
              <dt>summary</dt>
              <dd className="mono">figs/summaries/A_gap_normcap.json</dd>
              <dt>pipeline</dt>
              <dd>papers/figs/PIPELINE.md</dd>
            </dl>
          </article>
        </section>
      </main>
    </InstrumentShell>
  );
}
