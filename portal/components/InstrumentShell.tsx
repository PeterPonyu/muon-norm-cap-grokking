import type { ReactNode } from "react";

import { ModuleNav, type ModuleId } from "./ModuleNav";
import { ScienceProvider } from "./ScienceProvider";
import { StatusStrip } from "./StatusStrip";

const GITHUB = "https://github.com/PeterPonyu/muon-norm-cap-grokking";
const DOI = "10.5281/zenodo.21020291";

export function InstrumentShell({
  active,
  children,
}: {
  active: ModuleId;
  children: ReactNode;
}) {
  return (
    <ScienceProvider>
      <div className="instrument-shell">
        <header className="instrument" role="banner" aria-label="Norm-cap instrument">
          <StatusStrip />
          <ModuleNav active={active} />
        </header>
        {children}
        <footer className="instrument-foot">
          MIT<span className="sep">•</span>CC BY 4.0<span className="sep">•</span>
          <a href={GITHUB}>github.com/PeterPonyu/muon-norm-cap-grokking</a>
          <span className="sep">•</span>
          <a href={`https://doi.org/${DOI}`}>{DOI}</a>
        </footer>
      </div>
    </ScienceProvider>
  );
}
