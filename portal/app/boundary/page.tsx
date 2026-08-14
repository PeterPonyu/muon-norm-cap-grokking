import { InstrumentShell } from "../../components/InstrumentShell";
import { ModulePage } from "../../components/ModulePage";

export default function BoundaryPage() {
  return (
    <InstrumentShell active="boundary">
      <ModulePage
        title="Boundary"
        intro="Scope probes: activation sinks, plasticity, and the synthesis bound. Each face loads its public summary and shows questions plus panel structure."
        faceIds={["sink", "plasticity", "synth"]}
      />
    </InstrumentShell>
  );
}
