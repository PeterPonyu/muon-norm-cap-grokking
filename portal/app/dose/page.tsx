import { InstrumentShell } from "../../components/InstrumentShell";
import { ModulePage } from "../../components/ModulePage";

export default function DosePage() {
  return (
    <InstrumentShell active="dose">
      <ModulePage
        title="Dose"
        intro="Ceiling-dose channel. How the hidden-norm cap changes with k. The live summary names the dose series, panels, and relations."
        faceIds={["dose"]}
      />
    </InstrumentShell>
  );
}
