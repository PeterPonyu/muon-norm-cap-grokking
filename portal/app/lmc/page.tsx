import { InstrumentShell } from "../../components/InstrumentShell";
import { ModulePage } from "../../components/ModulePage";

export default function LmcPage() {
  return (
    <InstrumentShell active="lmc">
      <ModulePage
        title="LMC"
        intro="Linear-mode connectivity. The live summary names the barrier, family, and spawn faces that test Cap basins."
        faceIds={["lmc"]}
      />
    </InstrumentShell>
  );
}
