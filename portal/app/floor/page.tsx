import { InstrumentShell } from "../../components/InstrumentShell";

export default function FloorPage() {
  return (
    <InstrumentShell active="floor">
      <main className="stage">
        <section className="module-page">
          <article className="panel">
            <h1>Floor</h1>
            <p>
              Architecture slot for the optimizer-family floor channel. Rebuild locally
              from the generator named in FIGURE-INDEX; this page does not reprint
              results.
            </p>
            <dl className="kv">
              <dt>summary</dt>
              <dd className="mono">figs/summaries/A_floor.json</dd>
              <dt>generator</dt>
              <dd className="mono">figs/make_A_figs_r.R</dd>
            </dl>
          </article>
        </section>
      </main>
    </InstrumentShell>
  );
}
