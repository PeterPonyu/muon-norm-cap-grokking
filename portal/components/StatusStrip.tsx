"use client";

import { stripCounts, useScience } from "./ScienceProvider";

export function StatusStrip() {
  const { status, live } = useScience();
  const counts = stripCounts(live);
  const channels = [
    { key: "MODULES", value: String(counts.modules), tone: "cyan" as const },
    { key: "FACES", value: String(counts.faces), tone: "muted" as const },
    { key: "FAMILIES", value: String(counts.families), tone: "muted" as const },
    { key: "PANELS", value: String(counts.panels), tone: "muted" as const },
  ];

  return (
    <div className="status-strip" aria-label="Instrument channels" data-live={status}>
      {channels.map((ch) => (
        <div className="status-item" key={ch.key}>
          <span className="key">{ch.key}</span>
          <span className={`val ${ch.tone}`}>{ch.value}</span>
        </div>
      ))}
    </div>
  );
}
