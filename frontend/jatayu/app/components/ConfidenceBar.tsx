// src/components/ConfidenceBar.tsx
export function ConfidenceBar({ value, method }: { value: number; method: string }) {
  const pct = Math.round(value * 100);
  const low = pct < 35;

  return (
    <div>
      <div className="mb-1 flex justify-between text-sm">
        <span className="font-medium">Confidence</span>
        <span>{pct}%</span>
      </div>
      <div className="h-2 rounded-full bg-ink/10">
        <div
          className={`h-2 rounded-full ${low ? "bg-terracotta" : "bg-sage"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="mt-1 font-mono text-xs text-ink/50">{method}</p>
      {low && (
        <p className="mt-2 rounded-md bg-terracotta-soft px-3 py-2 text-sm text-terracotta">
          Low confidence — treat this as a lead, not a conclusion.
        </p>
      )}
    </div>
  );
}
