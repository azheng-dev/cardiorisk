"use client";

import { create } from "zustand";
import { decideCase, getCase, isMockMode, startCase } from "./client";
import type { CaseSnapshot, DecideRequest, PatientInput } from "./schema";

/**
 * Single source of truth for the in-progress case across the 5
 * Phase-5.3 screens. Backed by zustand (per AGENTS §4) so the
 * patient-input form, dashboard, guideline panel, letter editor and
 * audit log all read the same snapshot without prop-drilling and
 * without us reaching for React Context (which would force every
 * screen to be a client component).
 *
 * The store deliberately holds *only* the active case. Persistence
 * across navigation is provided by the URL (`?case=...`) so a refresh
 * keeps the user in the same state — important for the demo GIF.
 */

interface CaseStore {
  active: CaseSnapshot | null;
  loading: boolean;
  error: string | null;
  start: (patient: PatientInput) => Promise<CaseSnapshot>;
  load: (caseId: string) => Promise<CaseSnapshot | null>;
  decide: (payload: DecideRequest) => Promise<CaseSnapshot>;
  reset: () => void;
}

export const useCaseStore = create<CaseStore>((set, get) => ({
  active: null,
  loading: false,
  error: null,
  start: async (patient) => {
    set({ loading: true, error: null });
    try {
      const snap = await startCase(patient);
      set({ active: snap, loading: false });
      return snap;
    } catch (err) {
      const message = err instanceof Error ? err.message : "unknown error";
      set({ loading: false, error: message });
      throw err;
    }
  },
  load: async (caseId) => {
    if (get().active?.case_id === caseId) return get().active;
    set({ loading: true, error: null });
    try {
      const snap = await getCase(caseId);
      set({ active: snap, loading: false });
      return snap;
    } catch (err) {
      const message = err instanceof Error ? err.message : "unknown error";
      set({ loading: false, error: message });
      return null;
    }
  },
  decide: async (payload) => {
    const current = get().active;
    if (!current) throw new Error("no active case");
    set({ loading: true, error: null });
    try {
      const snap = await decideCase(current.case_id, payload);
      set({ active: snap, loading: false });
      return snap;
    } catch (err) {
      const message = err instanceof Error ? err.message : "unknown error";
      set({ loading: false, error: message });
      throw err;
    }
  },
  reset: () => set({ active: null, error: null, loading: false }),
}));

export const isMock = isMockMode;
