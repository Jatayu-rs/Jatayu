import type { AnalyzeResponse, HealthResponse, SampleScenario } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || " ";

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function fetchSamples(): Promise<SampleScenario[]> {
  try {
    const res = await fetch(`${API_BASE}/api/samples`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export async function analyzeWithSample(
  query: string,
  sampleId: string
): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("query", query);
  form.append("sample_id", sampleId);

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Analysis failed: ${res.status} — ${text}`);
  }
  return res.json();
}

export async function analyzeWithFiles(
  query: string,
  files: File[]
): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("query", query);
  for (const file of files) {
    form.append("files", file);
  }

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Analysis failed: ${res.status} — ${text}`);
  }
  return res.json();
}

export function overlayUrl(path: string): string {
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}
