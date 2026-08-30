"use client";

import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

/* ================================================================
   BACKEND
================================================================ */

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type View = "home" | "analysis";

type TraceStep = {
  step: number;
  stage: "validate" | "classify" | "route" | "execute" | "combine";
  detail: string;
  tool_name?: string | null;
  model_id?: string | null;
  params?: Record<string, unknown>;
  duration_ms?: number;
};

type ValidationIssue = {
  severity: "error" | "warning" | "info";
  code: string;
  message: string;
  image_index?: number | null;
};

type Evidence = {
  kind: "bbox" | "mask" | "overlay" | "table" | "none";
  boxes: Array<{
    x_min: number;
    y_min: number;
    x_max: number;
    y_max: number;
    label?: string | null;
    score?: number | null;
    image_index?: number;
  }>;
  mask_path?: string | null;
  overlay_png?: string | null;
  geojson?: Record<string, unknown> | null;
  legend: Record<string, string>;
  caption?: string | null;
};

type QueryResult = {
  answer: string;
  evidence: Evidence;
  confidence: number;
  confidence_method: string;
  task_family: string;
  tools_used: string[];
  trace: TraceStep[];
  validation: {
    ok: boolean;
    issues: ValidationIssue[];
  };
  total_latency_ms: number;
  request_id?: string | null;
};

type AnalyzeResponse = {
  result: QueryResult;
  language?: {
    detected: string;
    display_name: string;
    was_translated: boolean;
    original_query?: string | null;
  } | null;
  request_id?: string | null;
};

type SampleScenario = {
  id: string;
  title: string;
  description: string;
  family: string;
  suggested_query: string;
  files: string[];
  modalities: string[];
};

/* ================================================================
   HELPERS
================================================================ */

function backendUrl(path: string | null | undefined) {
  if (!path) return null;

  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }

  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

function formatToolName(name: string) {
  return name
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatTaskFamily(value: string) {
  return value.replaceAll("_", " ");
}

function formatLatency(ms: number) {
  if (!ms) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

function formatTimestamp() {
  return new Date().toLocaleTimeString([], {
    hour12: false,
  });
}

/* ================================================================
   SIDEBAR
================================================================ */

function Sidebar({
  active,
  setActive,
  setQuery,
  setView,
}: {
  active: string;
  setActive: (value: string) => void;
  setQuery: (value: string) => void;
  setView: (value: View) => void;
}) {
  return (
    <aside className="hidden w-[260px] shrink-0 border-r border-[#e4ddd5] bg-[#faf7f3] lg:flex lg:flex-col">
      <div className="px-7 pt-8">
        <div className="mb-10">
          <button
            onClick={() => setView("home")}
            className="font-serif text-[32px] font-semibold tracking-[-0.04em] text-[#a84300]"
          >
            Jatayu
          </button>

          <div className="mt-0.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-[#665e57]">
            Physics-Grounded Earth Intelligence
          </div>
        </div>

        <motion.button
          whileHover={{ scale: 1.015 }}
          whileTap={{ scale: 0.985 }}
          onClick={() => {
            setQuery("");
            setView("home");
          }}
          className="mb-8 flex w-full items-center justify-center gap-2 rounded-full bg-[#a84300] px-5 py-3.5 text-sm font-semibold text-white shadow-[0_8px_24px_rgba(168,67,0,0.18)]"
        >
          <span className="text-lg">+</span>
          New Analysis
        </motion.button>

        <nav className="space-y-1">
          {[
            ["◈", "Imagery Workspace"],
            ["◷", "Historical Archives"],
            ["⌘", "Intelligence Logs"],
            ["▥", "Strategic Reports"],
            ["⊞", "System Status"],
          ].map(([icon, label]) => (
            <button
              key={label}
              onClick={() => setActive(label)}
              className={`flex w-full items-center gap-3 rounded-r-full px-4 py-3 text-left text-[14px] transition ${
                active === label
                  ? "border-l-[3px] border-[#a84300] bg-[#f2e3d9] font-semibold text-[#a84300]"
                  : "border-l-[3px] border-transparent text-[#625b55] hover:bg-[#f2ede8]"
              }`}
            >
              <span className="w-5 text-[17px]">{icon}</span>
              {label}
            </button>
          ))}
        </nav>
      </div>

      <div className="mt-auto border-t border-[#e4ddd5] p-6">
        <div className="mb-5 flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[#66734c] text-xs font-bold text-white">
            JA
          </div>

          <div>
            <div className="text-xs font-semibold">Analyst 04</div>
            <div className="mt-0.5 flex items-center gap-1.5 text-[10px] text-[#6c7460]">
              <span className="h-1.5 w-1.5 rounded-full bg-[#6b7b47]" />
              System Ready
            </div>
          </div>
        </div>

        <button className="flex w-full items-center gap-3 text-sm text-[#655d56]">
          ⚙ <span>Settings</span>
        </button>

        <button className="mt-4 flex w-full items-center gap-3 text-sm text-[#655d56]">
          ? <span>Support</span>
        </button>
      </div>
    </aside>
  );
}

/* ================================================================
   HOME
================================================================ */

function HomeScreen({
  query,
  setQuery,
  execute,
  files,
  setFiles,
  samples,
  loading,
}: {
  query: string;
  setQuery: (value: string) => void;
  execute: () => void;
  files: File[];
  setFiles: (files: File[]) => void;
  samples: SampleScenario[];
  loading: boolean;
}) {
  const [showSamples, setShowSamples] = useState(false);

  const scenarios = [
    {
      eyebrow: "AGRICULTURE",
      title: "Crop Health Analysis",
      description:
        "Assess crop health using NDVI, NDRE, NDMI and complementary satellite evidence.",
      meta: "Spectral analysis",
      icon: "🌾",
      query: "Is the crop in this area stressed?",
    },
    {
      eyebrow: "CHANGE DETECTION",
      title: "Detect What Changed",
      description:
        "Compare satellite imagery across time to identify vegetation loss, construction and land-use change.",
      meta: "Multi-temporal",
      featured: true,
      icon: "◈",
      query: "What changed between these two images?",
    },
    {
      eyebrow: "MULTIMODAL",
      title: "Optical + SAR Fusion",
      description:
        "Combine Sentinel-1 SAR and Sentinel-2 optical imagery for analysis even under cloud cover.",
      meta: "Cross-modal",
      icon: "◉",
      query:
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
    },
  ];

  return (
    <section className="relative flex min-h-screen flex-1 flex-col overflow-hidden">
      <div className="pointer-events-none absolute right-[-180px] top-[-180px] h-[500px] w-[500px] rounded-full bg-[#ead9cc] opacity-30 blur-3xl" />

      <div className="mx-auto flex w-full max-w-[1120px] flex-1 flex-col px-6 pb-16 pt-20 sm:px-10 lg:px-16 lg:pt-28">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col items-center text-center"
        >
          <div className="mb-7 flex h-11 w-11 items-center justify-center rounded-full bg-[#e7ddd5] text-xl text-[#8b8179]">
            ◉
          </div>

          <h1 className="max-w-[900px] font-serif text-[48px] font-semibold leading-[0.98] tracking-[-0.045em] text-[#292522] sm:text-[60px] lg:text-[72px]">
            Ask the Earth a question.
          </h1>

          <p className="mt-6 max-w-[650px] text-[15px] leading-7 text-[#756d66] sm:text-[16px]">
            Analyze satellite imagery using natural language, physics-grounded
            models and multimodal geospatial intelligence.
          </p>
        </motion.div>

        {/* INPUT */}

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.12 }}
          className="mx-auto mt-12 w-full max-w-[900px]"
        >
          <div className="rounded-[28px] border border-[#e2dcd5] bg-white p-2 shadow-[0_12px_45px_rgba(72,52,37,0.06)] focus-within:border-[#caa991]">
            <div className="flex items-center pl-5">
              <span className="mr-3 text-xl text-[#8b8179]">⌕</span>

              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !loading) execute();
                }}
                placeholder="e.g. Is my crop stressed?"
                className="min-w-0 flex-1 bg-transparent py-3 text-[14px] text-[#292522] outline-none placeholder:text-[#aaa19a]"
              />

              <motion.button
                whileHover={{ scale: loading ? 1 : 1.02 }}
                whileTap={{ scale: loading ? 1 : 0.97 }}
                onClick={execute}
                disabled={loading}
                className="rounded-full bg-[#a84300] px-7 py-3.5 text-[13px] font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? "Analysing..." : "Execute →"}
              </motion.button>
            </div>

            {/* FILE UPLOAD */}

            <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-[#f0ebe6] px-5 py-3">
              <label className="cursor-pointer rounded-full border border-[#ddd5ce] bg-[#faf7f3] px-4 py-2 text-[10px] font-semibold text-[#675e56] transition hover:border-[#c9ad98] hover:text-[#a84300]">
                ＋ Add GeoTIFF
                <input
                  type="file"
                  accept=".tif,.tiff"
                  multiple
                  className="hidden"
                  onChange={(e) => {
                    const selected = Array.from(e.target.files || []);

                    if (selected.length > 2) {
                      setFiles(selected.slice(0, 2));
                    } else {
                      setFiles(selected);
                    }
                  }}
                />
              </label>

              {files.length === 0 ? (
                <span className="text-[10px] text-[#aaa19a]">
                  Upload 1 image, or 2 for temporal / multimodal analysis
                </span>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {files.map((file, index) => (
                    <div
                      key={`${file.name}-${index}`}
                      className="flex items-center gap-2 rounded-full bg-[#f3eee9] px-3 py-1.5 text-[9px] text-[#655d56]"
                    >
                      <span>▧</span>
                      <span className="max-w-[180px] truncate">
                        {file.name}
                      </span>
                      <button
                        onClick={() =>
                          setFiles(files.filter((_, i) => i !== index))
                        }
                        className="font-bold text-[#a84300]"
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="mt-4 flex flex-wrap justify-center gap-2">
            {[
              "Detect water bodies",
              "Check crop stress",
              "Find recent changes",
              "Analyze with SAR",
            ].map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => setQuery(suggestion)}
                className="rounded-full border border-[#e3dcd5] bg-[#faf7f3] px-4 py-2 text-[11px] text-[#716960] transition hover:border-[#c9ad98] hover:bg-white hover:text-[#a84300]"
              >
                {suggestion}
              </button>
            ))}
          </div>

          {/* BACKEND SAMPLES */}

          {samples.length > 0 && (
            <div className="mt-5 text-center">
              <button
                onClick={() => setShowSamples(!showSamples)}
                className="text-[10px] font-semibold uppercase tracking-[0.15em] text-[#a84300]"
              >
                {showSamples ? "Hide backend samples ↑" : "Use backend sample data ↓"}
              </button>

              <AnimatePresence>
                {showSamples && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-3 grid gap-2 text-left sm:grid-cols-2"
                  >
                    {samples.map((sample) => (
                      <button
                        key={sample.id}
                        onClick={() => {
                          setQuery(sample.suggested_query);
                          setFiles([]);
                        }}
                        className="rounded-[16px] border border-[#e4ddd5] bg-white p-4 transition hover:border-[#c9ad98]"
                      >
                        <div className="text-[11px] font-semibold text-[#3f3934]">
                          {sample.title}
                        </div>

                        <div className="mt-1 text-[9px] leading-4 text-[#8a8179]">
                          {sample.description}
                        </div>

                        <div className="mt-2 font-mono text-[8px] text-[#a84300]">
                          {sample.id}
                        </div>
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}
        </motion.div>

        {/* SCENARIOS */}

        <div className="mx-auto mt-16 grid w-full max-w-[1050px] gap-5 md:grid-cols-3">
          {scenarios.map((scenario, index) => (
            <motion.button
              key={scenario.title}
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 + index * 0.08 }}
              whileHover={{ y: -5 }}
              onClick={() => setQuery(scenario.query)}
              className={`group relative min-h-[270px] overflow-hidden rounded-[30px] border text-left transition ${
                scenario.featured
                  ? "border-[#ddd1c6] bg-[#292522] text-white shadow-[0_18px_50px_rgba(41,37,34,0.16)]"
                  : "border-[#e6dfd8] bg-white text-[#292522] shadow-[0_12px_40px_rgba(72,52,37,0.035)]"
              }`}
            >
              {scenario.featured && (
                <>
                  <div className="absolute inset-x-0 top-0 h-[150px] bg-[radial-gradient(circle_at_35%_40%,#7e756c,transparent_28%),linear-gradient(135deg,#403a35,#171614)]" />
                  <div className="absolute inset-x-0 top-0 h-[150px] bg-gradient-to-b from-transparent to-[#292522]" />
                </>
              )}

              <div className="relative flex h-full flex-col p-7">
                <div className="flex items-start justify-between">
                  <div
                    className={`flex h-11 w-11 items-center justify-center rounded-full text-lg ${
                      scenario.featured
                        ? "border border-white/20 bg-white/10"
                        : "bg-[#ece8df]"
                    }`}
                  >
                    {scenario.icon}
                  </div>

                  <span
                    className={`rounded-full px-3 py-1.5 text-[8px] font-semibold tracking-[0.16em] ${
                      scenario.featured
                        ? "bg-white/10 text-white/70"
                        : "bg-[#f4eee8] text-[#8a7c70]"
                    }`}
                  >
                    {scenario.eyebrow}
                  </span>
                </div>

                <div className="mt-auto">
                  <h2
                    className={`font-serif text-[25px] font-semibold ${
                      scenario.featured ? "text-white" : "text-[#292522]"
                    }`}
                  >
                    {scenario.title}
                  </h2>

                  <p
                    className={`mt-3 text-[12px] leading-5 ${
                      scenario.featured
                        ? "text-white/65"
                        : "text-[#756d66]"
                    }`}
                  >
                    {scenario.description}
                  </p>

                  <div
                    className={`mt-6 border-t pt-4 text-[10px] ${
                      scenario.featured
                        ? "border-white/15 text-white/55"
                        : "border-[#eee7e0] text-[#777069]"
                    }`}
                  >
                    ◈ &nbsp; {scenario.meta}
                  </div>
                </div>
              </div>
            </motion.button>
          ))}
        </div>

        <div className="mt-auto pt-16 text-center">
          <p className="text-[10px] uppercase tracking-[0.2em] text-[#aaa099]">
            Natural language → satellite evidence → explainable intelligence
          </p>
        </div>
      </div>
    </section>
  );
}

/* ================================================================
   LOADING SCREEN
================================================================ */

function LoadingScreen({ query }: { query: string }) {
  return (
    <section className="flex min-h-screen flex-1 items-center justify-center bg-[#f7f3ee]">
      <div className="w-full max-w-[620px] px-6">
        <div className="rounded-[28px] border border-[#e4ddd5] bg-white p-8 shadow-[0_15px_50px_rgba(70,50,35,0.06)]">
          <div className="flex items-center gap-4">
            <div className="flex h-11 w-11 animate-pulse items-center justify-center rounded-full bg-[#e7ddd5] text-[#a84300]">
              ◉
            </div>

            <div>
              <div className="font-serif text-[24px] font-semibold">
                Jatayu is analysing
              </div>

              <div className="mt-1 text-[10px] uppercase tracking-[0.15em] text-[#958980]">
                Physics-grounded inference
              </div>
            </div>
          </div>

          <div className="mt-7 rounded-[18px] bg-[#f8f3ee] p-5 text-[13px] leading-6 text-[#625850]">
            {query}
          </div>

          <div className="mt-6 space-y-3">
            {[
              "Validating imagery",
              "Classifying task",
              "Routing specialist tool",
              "Executing analysis",
            ].map((item, i) => (
              <motion.div
                key={item}
                initial={{ opacity: 0.35 }}
                animate={{ opacity: [0.35, 1, 0.35] }}
                transition={{
                  duration: 1.5,
                  repeat: Infinity,
                  delay: i * 0.25,
                }}
                className="flex items-center gap-3 text-[10px] uppercase tracking-[0.12em] text-[#81766e]"
              >
                <span className="h-2 w-2 rounded-full bg-[#a84300]" />
                {item}
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

/* ================================================================
   EVIDENCE IMAGE
================================================================ */
function EvidenceImage({
  src,
  label,
}: {
  src: string | null;
  label?: string;
}) {
  if (!src) {
    return (
      <div className="flex aspect-[4/5] w-full items-center justify-center rounded-[22px] bg-[#ece6df]">
        <div className="text-center">
          <div className="text-2xl text-[#a49a91]">◈</div>
          <div className="mt-2 text-[10px] uppercase tracking-[0.12em] text-[#8d8279]">
            No rendered evidence
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full overflow-hidden rounded-[22px] bg-[#ddd7cf]" style={{ aspectRatio: "4 / 5" }}>
      <img
        src={src}
        alt={label || "Jatayu analysis evidence"}
        className="absolute inset-0 h-full w-full object-contain"
      />

      {label && (
        <div className="absolute left-4 top-4 rounded-full bg-black/60 px-3 py-1.5 text-[9px] font-medium text-white backdrop-blur">
          {label}
        </div>
      )}
    </div>
  );
}
/* ================================================================
   ANALYSIS
================================================================ */

function AnalysisScreen({
  query,
  response,
  goHome,
}: {
  query: string;
  response: AnalyzeResponse;
  goHome: () => void;
}) {
  const result = response.result;

  const confidence = Math.round(result.confidence * 100);

  const evidenceUrl = backendUrl(result.evidence.overlay_png);
  const maskUrl = backendUrl(result.evidence.mask_path);

  const modelIds = Array.from(
    new Set(
      result.trace
        .map((step) => step.model_id)
        .filter(Boolean) as string[]
    )
  );

  const issues = result.validation?.issues || [];

  const stages = result.trace.length
    ? result.trace
    : [
        {
          step: 1,
          stage: "validate" as const,
          detail: "Request processed",
          duration_ms: 0,
        },
      ];

  const copyJson = async () => {
    try {
      await navigator.clipboard.writeText(
        JSON.stringify(response, null, 2)
      );
    } catch {
      // Clipboard permissions may be unavailable.
    }
  };

  const downloadEvidence = () => {
    if (!evidenceUrl) return;

    const a = document.createElement("a");
    a.href = evidenceUrl;
    a.download = "jatayu-evidence.png";
    a.target = "_blank";
    a.click();
  };

  return (
    <section className="min-h-screen flex-1 bg-[#f7f3ee] text-[#292522]">
      {/* TOP BAR */}

      <header className="sticky top-0 z-20 flex h-[82px] items-center justify-between border-b border-[#e4ddd5] bg-[#faf7f3]/95 px-6 backdrop-blur lg:px-10">
        <div className="flex items-center gap-5">
          <button
            onClick={goHome}
            className="flex h-9 w-9 items-center justify-center rounded-full border border-[#ddd4cb] bg-white text-lg transition hover:bg-[#f3ebe4]"
          >
            ←
          </button>

          <div>
            <div className="flex items-center gap-3">
              <h1 className="font-serif text-[25px] font-semibold">
                Analysis Complete
              </h1>

              <span
                className={`hidden rounded-full px-3 py-1 text-[9px] font-bold uppercase tracking-[0.12em] sm:block ${
                  result.validation.ok
                    ? "bg-[#e9eee0] text-[#59663f]"
                    : "bg-[#f6e4dc] text-[#a84300]"
                }`}
              >
                {result.validation.ok ? "Ready" : "Review"}
              </span>
            </div>

            <p className="mt-1 font-mono text-[9px] uppercase tracking-[0.12em] text-[#8c8178]">
              Query ID: {response.request_id || result.request_id || "—"}
            </p>
          </div>
        </div>

        <div className="hidden items-center gap-3 sm:flex">
          <button
            onClick={copyJson}
            className="rounded-full border border-[#d9d0c8] bg-white px-5 py-2.5 text-[11px] font-semibold text-[#5f564e]"
          >
            {"{ }"} Copy as JSON
          </button>

          {evidenceUrl && (
            <button
              onClick={downloadEvidence}
              className="rounded-full bg-[#a84300] px-5 py-2.5 text-[11px] font-semibold text-white shadow-[0_5px_20px_rgba(168,67,0,0.18)]"
            >
              ↓ &nbsp; Export Evidence
            </button>
          )}
        </div>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_340px]">
        <main className="min-w-0 p-5 sm:p-8 lg:p-10">
          <div className="mx-auto max-w-[930px] space-y-6">
            {/* USER QUERY */}

            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-[25px] border border-[#e5ddd5] bg-white p-7 shadow-[0_8px_30px_rgba(70,50,35,0.035)]"
            >
              <div className="mb-4 flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[#f1dfd3] text-[#a84300]">
                  ♙
                </div>

                <div>
                  <div className="text-[10px] font-bold uppercase tracking-[0.15em] text-[#8b7d73]">
                    User Query
                  </div>

                  <div className="mt-1 text-[9px] text-[#aaa099]">
                    Natural language request
                  </div>
                </div>
              </div>

              <p className="pl-0 text-[15px] leading-7 text-[#403a35] sm:pl-12">
                {query}
              </p>
            </motion.div>

            {/* REAL ANALYSIS */}

            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.08 }}
              className="rounded-[25px] border border-[#e5ddd5] bg-white p-7 shadow-[0_8px_30px_rgba(70,50,35,0.035)]"
            >
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[#e4e8da] text-[#657044]">
                  ◉
                </div>

                <div>
                  <div className="text-[10px] font-bold uppercase tracking-[0.15em] text-[#657044]">
                    Jatayu Analysis
                  </div>

                  <div className="mt-1 flex items-center gap-2">
                    <span className="h-1.5 w-1.5 rounded-full bg-[#697747]" />
                    <span className="text-[9px] text-[#8c8178]">
                      {formatTaskFamily(result.task_family)}
                    </span>
                  </div>
                </div>
              </div>

              <div className="mt-6 pl-0 sm:pl-12">
                <p className="whitespace-pre-wrap text-[15px] leading-7 text-[#403a35]">
                  {result.answer}
                </p>

                {result.evidence.caption && (
                  <p className="mt-4 text-[11px] italic leading-5 text-[#81766e]">
                    {result.evidence.caption}
                  </p>
                )}

                {/* METRICS */}

                <div className="mt-7 grid gap-3 sm:grid-cols-3">
                  <div className="rounded-[18px] bg-[#f8f3ee] p-5">
                    <div className="flex items-center justify-between">
                      <span className="text-[9px] font-bold uppercase tracking-[0.14em] text-[#85786e]">
                        Confidence
                      </span>

                      <span className="font-mono text-[18px] font-semibold text-[#59663f]">
                        {confidence}%
                      </span>
                    </div>

                    <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-[#ddd8d1]">
                      <div
                        className="h-full rounded-full bg-[#697747]"
                        style={{ width: `${confidence}%` }}
                      />
                    </div>

                    <div className="mt-2 text-[8px] text-[#958980]">
                      {result.confidence_method}
                    </div>
                  </div>

                  <div className="rounded-[18px] bg-[#f8f3ee] p-5">
                    <div className="text-[9px] font-bold uppercase tracking-[0.14em] text-[#85786e]">
                      Task Family
                    </div>

                    <div className="mt-3 font-mono text-[12px] text-[#4d4640]">
                      {formatTaskFamily(result.task_family)}
                    </div>
                  </div>

                  <div className="rounded-[18px] bg-[#f8f3ee] p-5">
                    <div className="text-[9px] font-bold uppercase tracking-[0.14em] text-[#85786e]">
                      Execution
                    </div>

                    <div className="mt-3 font-mono text-[12px] text-[#4d4640]">
                      {formatLatency(result.total_latency_ms)}
                    </div>
                  </div>
                </div>

                {/* TOOLS */}

                {result.tools_used.length > 0 && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {result.tools_used.map((tool) => (
                      <span
                        key={tool}
                        className="rounded-full bg-[#eef0e7] px-3 py-1.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-[#59663f]"
                      >
                        {formatToolName(tool)}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>

            {/* VALIDATION / ISSUES */}

            {issues.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                className="overflow-hidden rounded-[25px] border border-[#eadbd1] bg-[#f8eee8]"
              >
                <div className="flex">
                  <div className="w-2 bg-[#a84300]" />

                  <div className="w-full p-7">
                    <div className="text-[10px] font-bold uppercase tracking-[0.16em] text-[#a84300]">
                      Validation
                    </div>

                    <div className="mt-4 space-y-3">
                      {issues.map((issue, index) => (
                        <div
                          key={`${issue.code}-${index}`}
                          className="rounded-[14px] bg-white/70 p-4"
                        >
                          <div className="flex items-center gap-2">
                            <span
                              className={`rounded-full px-2 py-1 text-[8px] font-bold uppercase ${
                                issue.severity === "error"
                                  ? "bg-[#f3d8d0] text-[#9d3c1e]"
                                  : issue.severity === "warning"
                                    ? "bg-[#f3e6cc] text-[#86631d]"
                                    : "bg-[#e5ebdd] text-[#59663f]"
                              }`}
                            >
                              {issue.severity}
                            </span>

                            <span className="font-mono text-[9px] text-[#8c8178]">
                              {issue.code}
                            </span>
                          </div>

                          <p className="mt-2 text-[11px] leading-5 text-[#625850]">
                            {issue.message}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* EXECUTION TRACE */}

            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.16 }}
              className="rounded-[25px] border border-[#e5ddd5] bg-white p-7"
            >
              <div className="flex items-center justify-between">
                <h2 className="flex items-center gap-3 text-[16px] font-semibold">
                  <span className="text-[#a84300]">⌘</span>
                  Execution Trace
                </h2>

                <span className="text-[11px] text-[#968b82]">
                  {formatLatency(result.total_latency_ms)}
                </span>
              </div>

              <div className="mt-7 overflow-x-auto pb-3">
                <div className="flex min-w-[600px] items-start">
                  {stages.map((step, index) => (
                    <div
                      key={`${step.step}-${index}`}
                      className="flex flex-1 items-start"
                    >
                      <div className="flex min-w-[100px] flex-col items-center text-center">
                        <div className="flex h-7 w-7 items-center justify-center rounded-full bg-[#a84300] text-[9px] font-bold text-white">
                          ✓
                        </div>

                        <span className="mt-2 text-[8px] font-semibold uppercase tracking-[0.08em] text-[#766b63]">
                          {step.stage}
                        </span>

                        <span className="mt-1 max-w-[110px] text-[8px] leading-4 text-[#9a8f87]">
                          {step.tool_name
                            ? formatToolName(step.tool_name)
                            : step.detail}
                        </span>
                      </div>

                      {index < stages.length - 1 && (
                        <div className="mx-2 mt-3 h-[1px] flex-1 bg-[#d9c9be]" />
                      )}
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-4 rounded-[18px] bg-[#292522] p-5 font-mono text-[10px] leading-6 text-[#d6cec6]">
                {stages.map((step, index) => (
                  <div key={`${step.step}-log-${index}`}>
                    <span className="text-[#c7a890]">
                      {formatTimestamp()}
                    </span>{" "}
                    <span className="text-[#a7a098]">
                      [{step.stage.toUpperCase()}]
                    </span>{" "}
                    {step.detail}{" "}
                    <span className="text-[#a5b379]">OK</span>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* EVIDENCE */}

            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.24 }}
              className="rounded-[25px] border border-[#e5ddd5] bg-white p-7"
            >
              <div className="mb-5 flex items-center justify-between">
                <h2 className="flex items-center gap-3 text-[16px] font-semibold">
                  <span className="text-[#a84300]">◈</span>
                  Evidentiary Imagery
                </h2>

                <span className="rounded-full bg-[#f3eee9] px-3 py-1 text-[8px] font-semibold uppercase tracking-[0.12em] text-[#7e7167]">
                  {result.evidence.kind}
                </span>
              </div>

              <EvidenceImage
                src={evidenceUrl}
                label="BACKEND GENERATED EVIDENCE"
              />

              {maskUrl && (
                <div className="mt-3">
                  <a
                    href={maskUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="text-[10px] font-semibold text-[#a84300]"
                  >
                    Open analysis mask →
                  </a>
                </div>
              )}

              {Object.keys(result.evidence.legend || {}).length > 0 && (
                <div className="mt-5 rounded-[18px] bg-[#f8f3ee] p-5">
                  <div className="mb-3 text-[9px] font-bold uppercase tracking-[0.14em] text-[#81766e]">
                    Legend
                  </div>

                  <div className="grid gap-2 sm:grid-cols-2">
                    {Object.entries(result.evidence.legend).map(
                      ([value, label]) => (
                        <div
                          key={value}
                          className="flex items-center justify-between rounded-[12px] bg-white px-3 py-2"
                        >
                          <span className="font-mono text-[9px] text-[#81766e]">
                            {value}
                          </span>

                          <span className="text-[10px] font-medium text-[#4f4842]">
                            {label}
                          </span>
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}

              {result.evidence.boxes?.length > 0 && (
                <div className="mt-5 rounded-[18px] bg-[#f8f3ee] p-5">
                  <div className="mb-3 text-[9px] font-bold uppercase tracking-[0.14em] text-[#81766e]">
                    Detected Regions
                  </div>

                  <div className="space-y-2">
                    {result.evidence.boxes.map((box, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between rounded-[12px] bg-white px-4 py-3"
                      >
                        <div>
                          <div className="text-[10px] font-semibold">
                            {box.label || `Region ${index + 1}`}
                          </div>

                          <div className="mt-1 font-mono text-[8px] text-[#958980]">
                            [{box.x_min.toFixed(0)}, {box.y_min.toFixed(0)}] → [
                            {box.x_max.toFixed(0)}, {box.y_max.toFixed(0)}]
                          </div>
                        </div>

                        {box.score != null && (
                          <span className="font-mono text-[10px] text-[#59663f]">
                            {Math.round(box.score * 100)}%
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>

            {/* FOLLOW UP */}

            <FollowUp
              initialQuery={query}
              onSubmit={() => {}}
            />
          </div>
        </main>

        {/* CONTEXT SIDEBAR */}

        <aside className="border-l border-[#e4ddd5] bg-[#faf7f3] p-6 xl:min-h-[calc(100vh-82px)]">
          <div className="sticky top-[105px] space-y-6">
            <div>
              <h2 className="font-serif text-[24px] font-semibold">
                Context
              </h2>

              <p className="mt-1 text-[10px] uppercase tracking-[0.14em] text-[#958980]">
                Backend execution metadata
              </p>
            </div>

            {/* TASK */}

            <div>
              <div className="mb-3 text-[9px] font-bold uppercase tracking-[0.14em] text-[#81766e]">
                Task Classification
              </div>

              <div className="rounded-[20px] border border-[#e1d8d0] bg-white p-5">
                <div className="text-[9px] text-[#958980]">
                  Task family
                </div>

                <div className="mt-2 font-mono text-[13px] font-semibold text-[#4f4842]">
                  {formatTaskFamily(result.task_family)}
                </div>

                <div className="mt-4 text-[9px] text-[#958980]">
                  Evidence type
                </div>

                <div className="mt-2 font-mono text-[11px] text-[#4f4842]">
                  {result.evidence.kind}
                </div>
              </div>
            </div>

            {/* TOOLS */}

            <div>
              <div className="mb-3 text-[9px] font-bold uppercase tracking-[0.14em] text-[#81766e]">
                Tools Used
              </div>

              <div className="space-y-2">
                {result.tools_used.length ? (
                  result.tools_used.map((tool) => (
                    <div
                      key={tool}
                      className="flex items-center gap-3 rounded-[16px] border border-[#e1d8d0] bg-white p-3"
                    >
                      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#eee8df] text-[11px]">
                        ◈
                      </div>

                      <div>
                        <div className="text-[10px] font-semibold">
                          {formatToolName(tool)}
                        </div>

                        <div className="mt-0.5 text-[8px] text-[#978c84]">
                          Specialist analysis tool
                        </div>
                      </div>

                      <span className="ml-auto h-2 w-2 rounded-full bg-[#697747]" />
                    </div>
                  ))
                ) : (
                  <div className="rounded-[16px] border border-[#e1d8d0] bg-white p-4 text-[9px] text-[#978c84]">
                    No specialist tool reported.
                  </div>
                )}
              </div>
            </div>

            {/* MODELS */}

            <div>
              <div className="mb-3 text-[9px] font-bold uppercase tracking-[0.14em] text-[#81766e]">
                Models
              </div>

              <div className="rounded-[20px] border border-[#e1d8d0] bg-white p-5">
                {modelIds.length ? (
                  modelIds.map((model) => (
                    <div
                      key={model}
                      className="border-b border-[#eee8e2] py-3 last:border-0 last:pb-0 first:pt-0"
                    >
                      <div className="text-[8px] uppercase tracking-[0.1em] text-[#958980]">
                        Model ID
                      </div>

                      <div className="mt-1 break-all font-mono text-[9px] text-[#4f4842]">
                        {model}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="font-mono text-[9px] text-[#8b8077]">
                    No model ID reported.
                  </div>
                )}
              </div>
            </div>

            {/* PERFORMANCE */}

            <div>
              <div className="mb-3 text-[9px] font-bold uppercase tracking-[0.14em] text-[#81766e]">
                Performance
              </div>

              <div className="rounded-[20px] border border-[#e1d8d0] bg-white p-5">
                <div className="flex items-center justify-between border-b border-[#eee8e2] py-3 first:pt-0">
                  <span className="text-[9px] text-[#958980]">
                    Total latency
                  </span>

                  <span className="font-mono text-[9px] font-medium">
                    {formatLatency(result.total_latency_ms)}
                  </span>
                </div>

                <div className="flex items-center justify-between py-3">
                  <span className="text-[9px] text-[#958980]">
                    Trace steps
                  </span>

                  <span className="font-mono text-[9px] font-medium">
                    {result.trace.length}
                  </span>
                </div>

                <div className="flex items-center justify-between border-t border-[#eee8e2] py-3">
                  <span className="text-[9px] text-[#958980]">
                    Validation
                  </span>

                  <span
                    className={`font-mono text-[9px] font-medium ${
                      result.validation.ok
                        ? "text-[#59663f]"
                        : "text-[#a84300]"
                    }`}
                  >
                    {result.validation.ok ? "PASS" : "REVIEW"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </section>
  );
}

/* ================================================================
   FOLLOW UP
================================================================ */

function FollowUp({
  initialQuery,
  onSubmit,
}: {
  initialQuery: string;
  onSubmit: (query: string) => void;
}) {
  const [value, setValue] = useState("");

  return (
    <div className="sticky bottom-4 z-10 rounded-[24px] border border-[#ded4cb] bg-[#faf7f3]/95 p-2 shadow-[0_10px_35px_rgba(50,35,25,0.12)] backdrop-blur">
      <div className="flex items-center rounded-[18px] bg-white px-5 py-3">
        <span className="mr-3 text-[#8d8178]">⌕</span>

        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && value.trim()) {
              onSubmit(value.trim());
              setValue("");
            }
          }}
          placeholder={`Refine "${initialQuery.slice(0, 35)}${initialQuery.length > 35 ? "..." : ""}"`}
          className="flex-1 bg-transparent text-[12px] outline-none placeholder:text-[#aaa099]"
        />

        <button
          onClick={() => {
            if (!value.trim()) return;
            onSubmit(value.trim());
            setValue("");
          }}
          className="flex h-9 w-9 items-center justify-center rounded-full bg-[#a84300] text-white"
        >
          →
        </button>
      </div>
    </div>
  );
}

/* ================================================================
   APP
================================================================ */

export default function Home() {
  const [query, setQuery] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [active, setActive] = useState("Imagery Workspace");
  const [view, setView] = useState<View>("home");

  const [response, setResponse] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleScenario[]>([]);

  /* --------------------------------------------------------------
     Load backend samples
  -------------------------------------------------------------- */

  useEffect(() => {
    let cancelled = false;

    fetch(`${API_BASE}/api/samples`)
      .then(async (res) => {
        if (!res.ok) throw new Error("Could not load backend samples.");
        return res.json();
      })
      .then((data) => {
        if (!cancelled && Array.isArray(data)) {
          setSamples(data);
        }
      })
      .catch(() => {
        // Samples are optional. Do not block the application.
      });

    return () => {
      cancelled = true;
    };
  }, []);

  /* --------------------------------------------------------------
     Execute real backend request
  -------------------------------------------------------------- */

  const execute = async () => {
    const cleanQuery = query.trim();

    if (!cleanQuery) {
      setError("Please enter a question.");
      return;
    }

    if (files.length > 2) {
      setError("Jatayu accepts a maximum of two images.");
      return;
    }

    setLoading(true);
    setError(null);
    setView("analysis");

    try {
      const form = new FormData();

      form.append("query", cleanQuery);
      form.append("language", "eng_Latn");

      for (const file of files) {
        form.append("files", file);
      }

      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: form,
      });

      let data: AnalyzeResponse;

      try {
        data = await res.json();
      } catch {
        throw new Error(
          `Backend returned an invalid response (${res.status}).`
        );
      }

      if (!res.ok) {
        throw new Error(
          data?.result?.answer ||
            `Backend request failed with HTTP ${res.status}.`
        );
      }

      if (!data.result) {
        throw new Error("Backend response did not contain a result.");
      }

      setResponse(data);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Could not connect to the Jatayu backend.";

      setError(message);
      setResponse(null);
    } finally {
      setLoading(false);
    }
  };

  /* --------------------------------------------------------------
     Follow-up queries
  -------------------------------------------------------------- */

  const executeFollowUp = async (followUpQuery: string) => {
    setQuery(followUpQuery);

    // Follow-up uses the same uploaded imagery currently selected.
    setLoading(true);
    setError(null);

    try {
      const form = new FormData();

      form.append("query", followUpQuery);
      form.append("language", "eng_Latn");

      for (const file of files) {
        form.append("files", file);
      }

      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: form,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(
          data?.result?.answer || `HTTP ${res.status}`
        );
      }

      setResponse(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Follow-up analysis failed."
      );
    } finally {
      setLoading(false);
    }
  };

  const analysisContent = useMemo(() => {
    if (loading) {
      return <LoadingScreen query={query} />;
    }

    if (response) {
      return (
        <AnalysisScreen
          query={query}
          response={response}
          goHome={() => setView("home")}
        />
      );
    }

    return (
      <section className="flex min-h-screen flex-1 items-center justify-center bg-[#f7f3ee]">
        <div className="mx-6 w-full max-w-[600px] rounded-[28px] border border-[#eadbd1] bg-[#f8eee8] p-8">
          <div className="text-[10px] font-bold uppercase tracking-[0.16em] text-[#a84300]">
            Backend Connection Error
          </div>

          <h2 className="mt-3 font-serif text-[28px] font-semibold">
            Jatayu could not complete the request.
          </h2>

          <p className="mt-3 text-[13px] leading-6 text-[#625850]">
            {error || "No response was returned by the backend."}
          </p>

          <div className="mt-5 rounded-[15px] bg-white p-4 font-mono text-[10px] text-[#756d66]">
            Backend: {API_BASE}
          </div>

          <button
            onClick={() => {
              setError(null);
              setView("home");
            }}
            className="mt-5 rounded-full bg-[#a84300] px-5 py-3 text-[11px] font-semibold text-white"
          >
            ← Back to workspace
          </button>
        </div>
      </section>
    );
  }, [loading, response, query, error]);

  return (
    <main className="min-h-screen bg-[#f7f3ee] text-[#292522]">
      <div className="flex min-h-screen">
        <Sidebar
          active={active}
          setActive={setActive}
          setQuery={setQuery}
          setView={setView}
        />

        <AnimatePresence mode="wait">
          {view === "home" ? (
            <motion.div
              key="home"
              className="flex min-w-0 flex-1"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <HomeScreen
                query={query}
                setQuery={setQuery}
                execute={execute}
                files={files}
                setFiles={setFiles}
                samples={samples}
                loading={loading}
              />
            </motion.div>
          ) : (
            <motion.div
              key="analysis"
              className="flex min-w-0 flex-1"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              {analysisContent}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </main>
  );
}
