import figureMap from "../../content/figure-map.json";
import { InstrumentShell } from "../../components/InstrumentShell";

export default function ReproducePage() {
  return (
    <InstrumentShell active="reproduce">
      <main className="stage">
        <section className="module-page">
          <article className="panel">
            <h1>Reproduce</h1>
            <p>
              Warehouse door: clone the repo, read the pointer TeX, rebuild figures
              via the R → TikZ pipeline. This site is not a preprint.
            </p>
            <dl className="kv">
              <dt>clone</dt>
              <dd>https://github.com/PeterPonyu/muon-norm-cap-grokking</dd>
              <dt>manuscript</dt>
              <dd>{figureMap.manuscript}</dd>
              <dt>index</dt>
              <dd>{figureMap.index}</dd>
              <dt>schema</dt>
              <dd>{figureMap.schema}</dd>
              <dt>pipeline</dt>
              <dd>{figureMap.pipeline}</dd>
              <dt>Zenodo concept</dt>
              <dd>10.5281/zenodo.21020291</dd>
              <dt>license</dt>
              <dd>MIT (code) · CC BY 4.0 (data + figures)</dd>
            </dl>
            <p>Figure IDs (paths only):</p>
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
