import type { ReactNode } from "react";

import { ModuleNav, type ModuleId } from "./ModuleNav";

const GITHUB = "https://github.com/PeterPonyu/muon-norm-cap-grokking";
const DOI = "10.5281/zenodo.21020291";

const CHANNELS = [
  { key: "CEILING", value: "—", tone: "cyan" as const },
  { key: "STEP", value: "—", tone: "muted" as const },
  { key: "RATIO", value: "—", tone: "muted" as const },
  { key: "INTERVAL", value: "—", tone: "muted" as const },
];

export function InstrumentShell({
  active,
  children,
}: {
  active: ModuleId;
  children: ReactNode;
}) {
  return (
    <div className="instrument-shell">
      <header className="instrument" role="banner" aria-label="Norm-cap instrument">
        <div className="status-strip" aria-label="Instrument channels">
          {CHANNELS.map((ch) => (
            <div className="status-item" key={ch.key}>
              <span className="key">{ch.key}</span>
              <span className={`val ${ch.tone}`}>{ch.value}</span>
            </div>
          ))}
        </div>
        <ModuleNav active={active} />
      </header>
      {children}
      <footer className="warehouse">
        MIT<span className="sep">•</span>CC BY 4.0<span className="sep">•</span>
        <a href={GITHUB}>github.com/PeterPonyu/muon-norm-cap-grokking</a>
        <span className="sep">•</span>
        <a href={`https://doi.org/${DOI}`}>{DOI}</a>
        <span className="sep">•</span>papers/FIGURE-INDEX.json
        <span className="sep">•</span>papers/figs/PIPELINE.md
        <span className="sep">•</span>papers/A/main.tex
      </footer>
    </div>
  );
}
