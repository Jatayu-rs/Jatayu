// app/components/PipelineStages.tsx
"use client";
import { motion } from "framer-motion";
import type { TraceStep } from "../../src/lib/types";

type Stage = TraceStep["stage"];

const STAGES: { key: Stage; label: string }[] = [
  { key: "validate", label: "Validated" },
  { key: "classify", label: "Interpreted" },
  { key: "route", label: "Routed" },
  { key: "execute", label: "Executed" },
  { key: "combine", label: "Reported" },
];

export function PipelineStages({ trace, inFlight }: { trace?: TraceStep[]; inFlight: boolean }) {
  const done = new Set((trace ?? []).map((t) => t.stage));
  const optimistic = inFlight && !trace;

  return (
    <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-ink/50">
      {STAGES.map((s, i) => (
        <div key={s.key} className="flex items-center gap-2">
          <motion.span
            animate={
              done.has(s.key) ? { opacity: 1 } : optimistic ? { opacity: [0.3, 1, 0.3] } : { opacity: 0.3 }
            }
            transition={optimistic ? { repeat: Infinity, duration: 1.2, delay: i * 0.2 } : { duration: 0.2 }}
            className={done.has(s.key) ? "text-terracotta" : ""}
          >
            {s.label}
          </motion.span>
          {i < STAGES.length - 1 && <span className="text-ink/20">→</span>}
        </div>
      ))}
    </div>
  );
}
