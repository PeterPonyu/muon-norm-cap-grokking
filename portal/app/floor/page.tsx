import { InstrumentShell } from "../../components/InstrumentShell";
import { ModulePage } from "../../components/ModulePage";

export default function FloorPage() {
  return (
    <InstrumentShell active="floor">
      <ModulePage
        title="Floor"
        intro="Optimizer-family floor. The live summary lists the group ladder, seed spread, and family split that bound the instrument."
        faceIds={["floor"]}
      />
    </InstrumentShell>
  );
}
