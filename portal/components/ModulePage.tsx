"use client";

import { FACES, RELATIONS } from "../lib/modules";
import { LiveFace } from "./LiveFace";
import { useScience } from "./ScienceProvider";

export function ModulePage({
  title,
  intro,
  faceIds,
}: {
  title: string;
  intro: string;
  faceIds: string[];
}) {
  const { status, live } = useScience();
  const relations = RELATIONS.filter((rel) =>
    faceIds.some((id) => {
      const spec = FACES[id];
      return spec && (rel.from === spec.title || rel.to === spec.title);
    }),
  );

  return (
    <main className="stage">
      <section className="module-page">
        <article className="panel">
          <h1>{title}</h1>
          <p>{intro}</p>
          <p className="readout-caption" data-live={status}>
            {status === "ready"
              ? `${live?.faceCount ?? 0} public faces linked`
              : status === "loading"
                ? "reading public summaries"
                : "structure from the instrument map"}
          </p>
          {relations.length ? (
            <ul className="relation-list">
              {relations.map((rel) => (
                <li key={`${rel.from}-${rel.to}`}>
                  {rel.from} → {rel.to}
                  <span className="via">{rel.via}</span>
                </li>
              ))}
            </ul>
          ) : null}
        </article>
        {faceIds.map((id) => (
          <div className="panel face-panel" key={id}>
            <LiveFace id={id} />
          </div>
        ))}
      </section>
    </main>
  );
}
