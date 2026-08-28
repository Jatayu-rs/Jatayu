// src/lib/api.ts
import type { QueryResponse } from "./types";
import type { Sample } from "./local-types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function getSamples(): Promise<Sample[]> {
  const res = await fetch(`${BASE}/api/samples`);
  if (!res.ok) throw new Error("Failed to load samples");
  return res.json();
}

export async function analyze(params: {
  query: string;
  files?: File[];
  sampleId?: string;
}): Promise<QueryResponse> {
  const form = new FormData();
  form.append("query", params.query);
  if (params.sampleId) form.append("sample_id", params.sampleId);
  params.files?.forEach((f) => form.append("files", f));

  const res = await fetch(`${BASE}/api/analyze`, { method: "POST", body: form });
  // Backend guarantees a valid QueryResponse refusal even on error — never a raw 500.
  return res.json();
}

export function artifactUrl(path: string): string {
  return `${BASE}${path}`;
}