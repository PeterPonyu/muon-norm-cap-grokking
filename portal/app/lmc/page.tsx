import { InstrumentShell } from "../../components/InstrumentShell";

export default function LmcPage() {
  return (
    <InstrumentShell active="lmc">
      <main className="stage">
        <section className="module-page">
          <article className="panel">
            <h1>LMC</h1>
            <p>
              Architecture slot for linear-mode connectivity. Pointer only — open the
              warehouse JSON or the pointer manuscript.
            </p>
            <dl className="kv">
              <dt>summary</dt>
              <dd className="mono">figs/summaries/A_lmc.json</dd>
              <dt>manuscript</dt>
              <dd>papers/A/main.tex</dd>
            </dl>
          </article>
        </section>
      </main>
    </InstrumentShell>
  );
}
