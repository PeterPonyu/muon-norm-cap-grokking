"use client";

import { FACES, INSTRUMENT, MODULE_FACES, RELATIONS } from "../lib/modules";
import { useScience } from "./ScienceProvider";

const TARGETS = [
  { id: "cap", label: "Cap", faces: MODULE_FACES.cap },
  { id: "dose", label: "Dose", faces: MODULE_FACES.dose },
  { id: "floor", label: "Floor", faces: MODULE_FACES.floor },
  { id: "lmc", label: "LMC", faces: MODULE_FACES.lmc },
  { id: "boundary", label: "Boundary", faces: MODULE_FACES.boundary },
] as const;

export function RebuildPage() {
  const { status, live } = useScience();

  return (
    <main className="stage">
      <section className="module-page">
        <article className="panel">
          <h1>Rebuild</h1>
          <p>
            Clone the repo and rebuild the instrument channels. This site is the live
            door: Cap, Dose, Floor, LMC, Boundary, and rebuild.
          </p>
          <dl className="kv">
            <dt>clone</dt>
            <dd>https://github.com/PeterPonyu/muon-norm-cap-grokking</dd>
            <dt>archive</dt>
            <dd>10.5281/zenodo.21020291</dd>
            <dt>license</dt>
            <dd>MIT (code) · CC BY 4.0 (data + figures)</dd>
            <dt>live faces</dt>
            <dd data-live={status}>
              {status === "ready"
                ? `${live?.faceCount ?? INSTRUMENT.faces} summaries linked`
                : `${INSTRUMENT.faces} faces on the map`}
            </dd>
          </dl>
          <p>Rebuild targets:</p>
          <ul className="fig-list">
            {TARGETS.map((mod) => (
              <li key={mod.id}>
                <span className="id">{mod.label}</span>
                <span>
                  {mod.faces
                    .map((id) => FACES[id]?.title)
                    .filter(Boolean)
                    .join(" · ")}
                </span>
              </li>
            ))}
          </ul>
          <ul className="relation-list">
            {RELATIONS.map((rel) => (
              <li key={`${rel.from}-${rel.to}`}>
                {rel.from} → {rel.to}
                <span className="via">{rel.via}</span>
              </li>
            ))}
          </ul>
        </article>
      </section>
    </main>
  );
}
