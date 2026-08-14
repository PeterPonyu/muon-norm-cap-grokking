import { InstrumentShell } from "../components/InstrumentShell";
import { LiveFace } from "../components/LiveFace";

export default function CapPage() {
  return (
    <InstrumentShell active="cap">
      <main className="stage">
        <section className="readout-grid" aria-label="Instrument readouts">
          <LiveFace id="dose" showChart="dose" />
          <LiveFace id="timecourse" showChart="timecourse" />
          <LiveFace id="discriminator" showChart="chips" />
          <LiveFace id="lmc" showChart="lmc" />
        </section>

        <section className="claim-ledger" aria-label="Claim ledger">
          <div className="ledger-row">
            <span className="verb">PRESERVE</span>
            <span className="detail">Cap holds the ceiling intervention</span>
            <span className="tick">.</span>
          </div>
          <div className="ledger-row accelerate">
            <span className="verb">ACCELERATE</span>
            <span className="detail">Dose tightens the ceiling series</span>
            <span className="tick">.</span>
          </div>
          <div className="ledger-row boundary">
            <span className="verb">BOUNDARY</span>
            <span className="detail">Floor, sink, and plasticity mark scope</span>
            <span className="tick">.</span>
          </div>
        </section>
      </main>
    </InstrumentShell>
  );
}
