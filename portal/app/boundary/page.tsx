import { InstrumentShell } from "../../components/InstrumentShell";

export default function BoundaryPage() {
  return (
    <InstrumentShell active="boundary">
      <main className="stage">
        <section className="module-page">
          <article className="panel">
            <h1>Boundary</h1>
            <p>
              Architecture slot for scope probes (activations, plasticity, real-text).
              File names only; no caption prose.
            </p>
            <dl className="kv">
              <dt>summaries</dt>
              <dd className="mono">A_sink.json · A_plasticity.json · A_synth.json</dd>
              <dt>index</dt>
              <dd>papers/FIGURE-INDEX.json</dd>
            </dl>
          </article>
        </section>
      </main>
    </InstrumentShell>
  );
}
