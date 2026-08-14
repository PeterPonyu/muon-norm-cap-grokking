"use client";

import { FACES } from "../lib/modules";
import { useLiveFace, useScience } from "./ScienceProvider";

export function LiveFace({
  id,
  showChart,
}: {
  id: string;
  showChart?: "dose" | "timecourse" | "chips" | "lmc";
}) {
  const spec = FACES[id];
  const live = useLiveFace(id);
  const { status } = useScience();
  const questions = live?.questions ?? [];
  const ceilings = live?.ceilings.length ? live.ceilings : null;
  const families = live?.families.length ? live.families : null;
  const seeds = live?.seeds ?? null;
  const panels = spec?.panels ?? [];

  return (
    <article className="readout" aria-labelledby={`${id}-title`} data-face={id}>
      <h2 className="readout-title" id={`${id}-title`}>
        {spec?.title ?? id}
      </h2>
      <div className="readout-body">
        {showChart === "dose" ? <DoseChart /> : null}
        {showChart === "timecourse" ? <TimecourseChart /> : null}
        {showChart === "chips" ? (
          <div className="chip-stack" role="list">
            {(families ?? ["Muon", "AdamW", "SGDM"]).map((name) => (
              <div className={name === "Muon" ? "chip is-active" : "chip"} role="listitem" key={name}>
                {name}
              </div>
            ))}
          </div>
        ) : null}
        {showChart === "lmc" ? <div className="lmc-face">linear-mode slot</div> : null}

        <p className="readout-role">{spec?.role}</p>
        {panels.length ? (
          <ul className="panel-list">
            {panels.map((panel) => (
              <li key={panel}>{panel}</li>
            ))}
          </ul>
        ) : null}
        {questions.length ? (
          <ol className="question-list" data-live="questions">
            {questions.map((q) => (
              <li key={q}>{q}</li>
            ))}
          </ol>
        ) : (
          <p className="readout-caption" data-live={status}>
            {status === "loading" ? "reading public summaries" : spec?.relates.join(" · ")}
          </p>
        )}
        <p className="qty-row">
          {ceilings ? <span>ceilings {ceilings.join(" · ")}</span> : null}
          {seeds != null ? <span>seeds {seeds}</span> : null}
          {live?.panelCount != null ? <span>panels {live.panelCount}</span> : null}
        </p>
      </div>
    </article>
  );
}

function DoseChart() {
  return (
    <svg className="chart" viewBox="0 0 320 180" role="img" aria-label="Schematic dose channel">
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
  );
}

function TimecourseChart() {
  return (
    <svg className="chart" viewBox="0 0 320 180" role="img" aria-label="Schematic amber trace into a cyan cap">
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
  );
}
