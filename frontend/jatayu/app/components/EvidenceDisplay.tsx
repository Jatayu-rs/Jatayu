// app/components/EvidenceDisplay.tsx
"use client";
import { useRef, useState } from "react";
import type { Evidence } from "../../src/lib/types";
import { artifactUrl } from "../../src/lib/api";

export function EvidenceDisplay({ evidence, baseImageUrl }: { evidence: Evidence; baseImageUrl: string }) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [natural, setNatural] = useState({ w: 1, h: 1 });

  if (evidence.kind === "none") {
    return <p className="text-sm italic text-ink/50">No visual evidence for this answer.</p>;
  }

  return (
    <div>
      <div className="relative inline-block">
        <img
          ref={imgRef}
          src={baseImageUrl}
          alt=""
          className="max-w-full rounded-lg"
          onLoad={(e) => setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
        />
        {evidence.overlay_png && (
          <img src={artifactUrl(evidence.overlay_png)} alt="" className="absolute inset-0 h-full w-full rounded-lg opacity-80" />
        )}
        {(evidence.boxes ?? []).map((b, i) => {
          const el = imgRef.current;
          const sx = el ? el.clientWidth / natural.w : 1;
          const sy = el ? el.clientHeight / natural.h : 1;
          return (
            <div
              key={i}
              className="absolute border-2 border-terracotta"
              style={{
                left: b.x_min * sx,
                top: b.y_min * sy,
                width: (b.x_max - b.x_min) * sx,
                height: (b.y_max - b.y_min) * sy,
              }}
            >
              {b.label && (
                <span className="absolute -top-6 left-0 whitespace-nowrap bg-terracotta px-1.5 py-0.5 text-xs text-white">
                  {b.label}{b.score != null ? ` · ${Math.round(b.score * 100)}%` : ""}
                </span>
              )}
            </div>
          );
        })}
      </div>
      {evidence.legend && Object.keys(evidence.legend).length > 0 && (
        <div className="mt-3 flex flex-wrap gap-3">
          {Object.entries(evidence.legend).map(([label, color]) => (
            <span key={label} className="flex items-center gap-1.5 text-xs">
              <span className="h-3 w-3 rounded-sm" style={{ background: color }} />
              {label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}