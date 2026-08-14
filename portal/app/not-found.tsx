import Link from "next/link";

import { InstrumentShell } from "../components/InstrumentShell";

export default function NotFound() {
  return (
    <InstrumentShell active="cap">
      <main className="stage">
        <section className="module-page">
          <article className="panel">
            <h1>Not found</h1>
            <p>
              Unknown instrument module. Return to{" "}
              <Link className="mono" href="/">
                Cap
              </Link>
              .
            </p>
          </article>
        </section>
      </main>
    </InstrumentShell>
  );
}
