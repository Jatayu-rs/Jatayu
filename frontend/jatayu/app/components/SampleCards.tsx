// src/components/SampleCards.tsx
"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { getSamples } from "../../src/lib/api";
import type { Sample } from "../../src/lib/local-types";

export function SampleCards({ onPick }: { onPick: (s: Sample) => void }) {
  const [samples, setSamples] = useState<Sample[]>([]);

  useEffect(() => {
    getSamples().then(setSamples).catch(() => setSamples([]));
  }, []);

  if (!samples.length) return null;

  return (
    <div className="grid grid-cols-3 gap-4">
      {samples.map((s, i) => (
        <motion.button
          key={s.id}
          onClick={() => onPick(s)}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.05 }}
          whileHover={{ y: -2 }}
          className="rounded-xl border border-ink/10 bg-white p-4 text-left shadow-sm transition-shadow hover:shadow-md"
        >
          <img src={s.thumbnail_url} alt="" className="mb-3 h-28 w-full rounded-md object-cover" />
          <p className="font-medium">{s.label}</p>
          <p className="mt-1 text-sm text-ink/60">{s.suggested_query}</p>
        </motion.button>
      ))}
    </div>
  );
}