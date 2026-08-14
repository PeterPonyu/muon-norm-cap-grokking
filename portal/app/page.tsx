import { InstrumentShell } from "../components/InstrumentShell";

export default function CapPage() {
  return (
    <InstrumentShell active="cap">
      <main className="stage">
        <section className="readout-grid" aria-label="Instrument readouts">
          <article className="readout" aria-labelledby="dose-title">
            <h2 className="readout-title" id="dose-title">
              DOSE
            </h2>
            <div className="readout-body">
              <svg
                className="chart"
                viewBox="0 0 320 180"
                role="img"
                aria-label="Schematic dose channel"
              >
                <text className="chart-label" x="36" y="22">
                  high
                </text>
                <text className="chart-label" x="36" y="118">
                  low
                </text>
                <line x1="58" y1="16" x2="58" y2="128" stroke="#2A3140" strokeWidth="1" />
                <line x1="58" y1="128" x2="310" y2="128" stroke="#2A3140" strokeWidth="1" />
                <rect x="74" y="46" width="28" height="82" fill="#3EC8D8" />
                <rect x="118" y="76" width="28" height="52" fill="#3EC8D8" />
                <rect x="162" y="82" width="28" height="46" fill="#3EC8D8" />
                <rect x="206" y="86" width="28" height="42" fill="#3EC8D8" />
                <rect x="250" y="92" width="28" height="36" fill="#3EC8D8" />
                <text className="chart-label" x="180" y="168">
                  k
                </text>
              </svg>
              <p className="readout-caption">
                pointer <span className="mono">figs/summaries/A_gap_normcap.json</span>
              </p>
            </div>
          </article>

          <article className="readout" aria-labelledby="tc-title">
            <h2 className="readout-title" id="tc-title">
              TIMECOURSE
            </h2>
            <div className="readout-body">
              <svg
                className="chart"
                viewBox="0 0 320 180"
                role="img"
                aria-label="Schematic amber trace into a cyan cap"
              >
                <line x1="58" y1="16" x2="58" y2="128" stroke="#2A3140" strokeWidth="1" />
                <line x1="58" y1="128" x2="310" y2="128" stroke="#2A3140" strokeWidth="1" />
                <line x1="70" y1="96" x2="300" y2="96" stroke="#3EC8D8" strokeWidth="1.5" />
                <polyline
                  fill="none"
                  stroke="#E8A838"
                  strokeWidth="2"
                  points="70,118 110,108 150,70 190,28 214,22 228,96 300,98"
                />
                <text className="chart-label" x="180" y="168">
                  step
                </text>
              </svg>
              <p className="readout-caption">
                pointer <span className="mono">figs/summaries/A_normctl.json</span>
              </p>
            </div>
          </article>

          <article className="readout" aria-labelledby="disc-title">
            <h2 className="readout-title" id="disc-title">
              DISCRIMINATOR
            </h2>
            <div className="readout-body">
              <div className="chip-stack" role="list">
                <div className="chip is-active" role="listitem">
                  Muon
                </div>
                <div className="chip" role="listitem">
                  AdamW
                </div>
                <div className="chip" role="listitem">
                  SGDM
                </div>
              </div>
              <p className="readout-caption">
                pointer <span className="mono">figs/summaries/A_norm_discriminator.json</span>
              </p>
            </div>
          </article>

          <article className="readout" aria-labelledby="lmc-title">
            <h2 className="readout-title" id="lmc-title">
              LMC
            </h2>
            <div className="readout-body">
              <div className="lmc-face">linear-mode slot</div>
              <p className="readout-caption">
                pointer <span className="mono">figs/summaries/A_lmc.json</span>
              </p>
            </div>
          </article>
        </section>

        <section className="claim-ledger" aria-label="Claim ledger">
          <div className="ledger-row">
            <span className="verb">PRESERVE</span>
            <span className="detail">Cap module</span>
            <span className="tick">.</span>
          </div>
          <div className="ledger-row accelerate">
            <span className="verb">ACCELERATE</span>
            <span className="detail">Dose module</span>
            <span className="tick">.</span>
          </div>
          <div className="ledger-row boundary">
            <span className="verb">BOUNDARY</span>
            <span className="detail">scope module</span>
            <span className="tick">.</span>
          </div>
        </section>
      </main>
    </InstrumentShell>
  );
}
