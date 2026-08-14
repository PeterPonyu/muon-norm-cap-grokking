import Link from "next/link";

export const MODULES = [
  { id: "cap", label: "Cap", href: "/" },
  { id: "dose", label: "Dose", href: "/dose/" },
  { id: "floor", label: "Floor", href: "/floor/" },
  { id: "lmc", label: "LMC", href: "/lmc/" },
  { id: "boundary", label: "Boundary", href: "/boundary/" },
  { id: "reproduce", label: "Reproduce", href: "/reproduce/" },
] as const;

export type ModuleId = (typeof MODULES)[number]["id"];

export function ModuleNav({ active }: { active: ModuleId }) {
  return (
    <nav className="module-tabs" aria-label="Instrument modules">
      {MODULES.map((mod) => (
        <Link
          key={mod.id}
          href={mod.href}
          className={mod.id === active ? "is-active" : undefined}
          aria-current={mod.id === active ? "page" : undefined}
        >
          {mod.label}
        </Link>
      ))}
    </nav>
  );
}
