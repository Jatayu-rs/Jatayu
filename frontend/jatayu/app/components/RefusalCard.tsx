// app/components/RefusalCard.tsx
import { AlertTriangle } from "lucide-react";
import type { ValidationIssue } from "../../src/lib/types";

export function RefusalCard({ answer, issues }: { answer: string; issues?: ValidationIssue[] }) {
  return (
    <div className="rounded-xl border border-ink/10 bg-white p-6">
      <div className="mb-3 flex items-center gap-2 text-terracotta">
        <AlertTriangle size={18} />
        <h3 className="font-serif text-lg">Couldn't process this request</h3>
      </div>
      <p className="text-sm text-ink/80">{answer}</p>
      {(issues ?? []).length > 0 && (
        <ul className="mt-4 space-y-1 rounded-lg bg-cream p-4 text-sm text-ink/70">
          {(issues ?? []).map((issue) => (
            <li key={issue.code}>
              <span className="font-mono text-xs text-ink/40">{issue.code}</span> — {issue.message}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}