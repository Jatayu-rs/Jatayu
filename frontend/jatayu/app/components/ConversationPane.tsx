// app/components/ConversationPane.tsx
"use client";
import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Send } from "lucide-react";
import { analyze } from "../../src/lib/api";
import { isRefusal, validationErrors, type Sample, type ImageRef } from "../../src/lib/local-types";
import type { QueryResponse } from "../../src/lib/types";
import { SampleCards } from "./SampleCards";
import { PipelineStages } from "./PipelineStages";
import { EvidenceDisplay } from "./EvidenceDisplay";
import { ConfidenceBar } from "./ConfidenceBar";
import { RefusalCard } from "./RefusalCard";
import { TracePanel } from "./TracePanel";

export function ConversationPane({ onImagesChange }: { onImagesChange: (images: ImageRef[]) => void }) {
  const [query, setQuery] = useState("");
  const [sentQuery, setSentQuery] = useState<string | null>(null);
  const [baseImageUrl, setBaseImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);

  async function run(opts: { query: string; sampleId?: string; files?: File[] }) {
    setLoading(true);
    setError(null);
    setResult(null);
    setSentQuery(opts.query);
    try {
      const res = await analyze(opts);
      setResult(res);
    } catch {
      // Network/parse failure only — the backend itself always returns a valid
      // refusal QueryResponse, per the contract, so this path is transport-level.
      setError("Couldn't reach the analysis server. Check your connection and try again.");
    } finally {
      setLoading(false);
    }
  }

  function handleSample(s: Sample) {
    onImagesChange(s.images);
    setBaseImageUrl(s.thumbnail_url);
    setQuery(s.suggested_query);
    run({ query: s.suggested_query, sampleId: s.id });
  }

  function handleSubmit() {
    if (!query.trim()) return;
    run({ query });
  }

  return (
    <main className="flex h-full flex-col overflow-y-auto p-8">
      <h2 className="mb-6 font-serif text-xl">Analysis Session</h2>

      {!sentQuery && (
        <div className="my-auto">
          <p className="mb-4 text-sm text-ink/50">A judge won&apos;t have a GeoTIFF on hand — start with a sample:</p>
          <SampleCards onPick={handleSample} />
        </div>
      )}

      {sentQuery && (
        <div className="flex-1 space-y-6">
          <div className="ml-auto max-w-lg rounded-2xl bg-white px-4 py-3 shadow-sm">{sentQuery}</div>

          <AnimatePresence mode="wait">
            {loading && (
              <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <PipelineStages inFlight />
              </motion.div>
            )}

            {!loading && error && (
              <motion.p key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-sm text-terracotta">
                {error}
              </motion.p>
            )}

            {!loading && result && (
              <motion.div key="result" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
                <PipelineStages trace={result.trace} inFlight={false} />

                {isRefusal(result) ? (
                  <RefusalCard answer={result.answer} issues={validationErrors(result)} />
                ) : (
                  <div className="rounded-xl border border-ink/10 bg-white p-5">
                    <p className="mb-4">{result.answer}</p>
                    {baseImageUrl && <EvidenceDisplay evidence={result.evidence} baseImageUrl={baseImageUrl} />}
                    <div className="mt-4">
                      <ConfidenceBar value={result.confidence} method={result.confidence_method} />
                    </div>
                  </div>
                )}

                {(result.trace ?? []).length > 0 && <TracePanel trace={result.trace ?? []} />}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      <div className="mt-6 flex items-center gap-2 rounded-full border border-ink/10 bg-white px-4 py-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          placeholder="Ask about this imagery…"
          className="flex-1 bg-transparent text-sm outline-none"
        />
        <button onClick={handleSubmit} className="rounded-full bg-ink p-2 text-cream">
          <Send size={14} />
        </button>
      </div>
    </main>
  );
}