// src/components/TracePanel.tsx
"use client";
import { useState } from "react";
import type { TraceStep } from "../../src/lib/types";

export function TracePanel({ trace }: { trace: TraceStep[] }) {
  const [copied, setCopied] = useState(false);

  return (
    <div className="rounded-xl border border-ink/10 bg-white p-4">
      <div className="mb-3 flex items-center justify-between">
        <h4 className="font-serif">How this was answered</h4>
        <button
          onClick={() => {
            navigator.clipboard.writeText(JSON.stringify(trace, null, 2));
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }}
          className="text-xs text-terracotta hover:underline"
        >
          {copied ? "Copied" : "Copy as JSON"}
        </button>
      </div>
      <ol className="space-y-3">
        {trace.map((step, i) => (
          <li key={i} className="flex gap-3 text-sm">
            <span className="font-mono text-ink/40">{i + 1}</span>
            <div>
              <p className="font-medium capitalize">{step.stage}</p>
              <p className="text-ink/60">{step.detail}</p>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}