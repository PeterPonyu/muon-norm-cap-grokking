import { InstrumentShell } from "../../components/InstrumentShell";

export default function LmcPage() {
  return (
    <InstrumentShell active="lmc">
      <main className="stage">
        <section className="module-page">
          <article className="panel">
            <h1>LMC</h1>
            <p>
              Linear-mode connectivity slot. Open the LMC summary to inspect the
              connectivity face.
            </p>
            <dl className="kv">
              <dt>channel</dt>
              <dd>linear-mode connectivity</dd>
              <dt>summary</dt>
              <dd className="mono">figs/summaries/A_lmc.json</dd>
            </dl>
          </article>
        </section>
      </main>
    </InstrumentShell>
  );
}
