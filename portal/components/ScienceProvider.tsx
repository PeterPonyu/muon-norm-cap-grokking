"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

import { INSTRUMENT } from "../lib/modules";
import { loadLiveScience, type LiveFace, type LiveScience } from "../lib/publicData";

type ScienceState = {
  status: "loading" | "ready" | "offline";
  live: LiveScience | null;
};

const ScienceContext = createContext<ScienceState>({ status: "loading", live: null });

export function ScienceProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<ScienceState>({ status: "loading", live: null });

  useEffect(() => {
    let cancelled = false;
    loadLiveScience().then((live) => {
      if (cancelled) return;
      setState(live ? { status: "ready", live } : { status: "offline", live: null });
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return <ScienceContext.Provider value={state}>{children}</ScienceContext.Provider>;
}

export function useScience(): ScienceState {
  return useContext(ScienceContext);
}

export function useLiveFace(id: string): LiveFace | null {
  return useScience().live?.faces[id] ?? null;
}

export function stripCounts(live: LiveScience | null) {
  const sample = live?.faces.dose ?? live?.faces.cap ?? live?.faces.floor;
  return {
    modules: INSTRUMENT.modules,
    faces: live?.faceCount || INSTRUMENT.faces,
    families: sample?.families.length || INSTRUMENT.families.length,
    panels: sample?.panelCount || INSTRUMENT.panelsPerFace,
  };
}
