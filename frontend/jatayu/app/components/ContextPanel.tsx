// app/components/ContextPanel.tsx
import type { ImageRef } from "../../src/lib/local-types";

export function ContextPanel({ images }: { images: ImageRef[] }) {
  if (images.length === 0) {
    return (
      <aside className="border-l border-ink/10 bg-cream p-6">
        <h2 className="text-sm font-medium text-ink/50">Spatial Context</h2>
        <p className="mt-4 text-sm text-ink/40">Load a sample or upload imagery to see coordinates and CRS here.</p>
      </aside>
    );
  }

  return (
    <aside className="space-y-4 overflow-y-auto border-l border-ink/10 bg-cream p-6">
      <h2 className="text-sm font-medium text-ink/50">Spatial Context</h2>

      {/* Placeholder footprint — swap for a real map (maplibre/leaflet) once bounds are reliable */}
      <div className="flex h-40 items-center justify-center rounded-lg border border-dashed border-ink/20 bg-white text-xs text-ink/30">
        Map preview
      </div>

      {images.map((img, i) => {
        const mismatched =
          i > 0 && images[0].crs && img.crs && images[0].crs !== img.crs;
        return (
          <div
            key={i}
            className={`rounded-lg border p-3 text-sm ${mismatched ? "border-terracotta bg-terracotta-soft" : "border-ink/10 bg-white"}`}
          >
            <p className="mb-1 flex items-center gap-2 font-medium">
              <span className={`h-2 w-2 rounded-full ${mismatched ? "bg-terracotta" : "bg-sage"}`} />
              Source {i + 1}
            </p>
            <dl className="space-y-1 text-xs text-ink/60">
              <div className="flex justify-between"><dt>CRS</dt><dd className="font-mono">{img.crs ?? "unknown"}</dd></div>
              {img.bounds && (
                <div className="flex justify-between">
                  <dt>Bounds</dt>
                  <dd className="font-mono">{img.bounds.map((b) => b.toFixed(2)).join(", ")}</dd>
                </div>
              )}
              {img.acquired && (
                <div className="flex justify-between"><dt>Acquired</dt><dd>{new Date(img.acquired).toLocaleDateString()}</dd></div>
              )}
              <div className="flex justify-between"><dt>Modality</dt><dd className="capitalize">{img.modality}</dd></div>
            </dl>
          </div>
        );
      })}
    </aside>
  );
}