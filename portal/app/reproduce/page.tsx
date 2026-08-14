import figureMap from "../../content/figure-map.json";
import { InstrumentShell } from "../../components/InstrumentShell";

export default function ReproducePage() {
  return (
    <InstrumentShell active="reproduce">
      <main className="stage">
        <section className="module-page">
          <article className="panel">
            <h1>Reproduce-as-rebuild</h1>
            <p>
              Clone the repo and rebuild the instrument channels from the generators
              below. This site is the live instrument, not an archive dump.
            </p>
            <dl className="kv">
              <dt>clone</dt>
              <dd>https://github.com/PeterPonyu/muon-norm-cap-grokking</dd>
              <dt>Zenodo concept</dt>
              <dd>10.5281/zenodo.21020291</dd>
              <dt>license</dt>
              <dd>MIT (code) · CC BY 4.0 (data + figures)</dd>
            </dl>
            <p>Channel IDs (generators only):</p>
            <ul className="fig-list">
              {figureMap.figures.map((fig) => (
                <li key={fig.id}>
                  <span className="id">{fig.id}</span>
                  <span>{fig.generator}</span>
                </li>
              ))}
            </ul>
          </article>
        </section>
      </main>
    </InstrumentShell>
  );
}
