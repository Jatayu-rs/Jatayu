// src/lib/local-types.ts — hand-maintained, never touched by codegen
import type { QueryResponse, ValidationIssue } from "../lib/types";

export type Modality = "optical" | "multispectral" | "sar" | "unknown";

export interface ImageRef {
  path: string;
  modality: Modality;
  crs: string | null;
  bounds: [number, number, number, number] | null;
  transform: [number, number, number, number, number, number] | null;
  acquired: string | null;
  width: number;
  height: number;
  band_count: number;
  band_names: string[];
  dtype: string;
  nodata: number | null;
}

// Your own /api/samples shape — not part of the tool contract. Confirm with Dev C.
export interface Sample {
  id: string;
  label: string;
  thumbnail_url: string;
  suggested_query: string;
  images: ImageRef[];
}

export const validationErrors = (r: QueryResponse): ValidationIssue[] =>
  (r.validation.issues ?? []).filter((i) => i.severity === "error");

export const isRefusal = (r: QueryResponse): boolean =>
  r.confidence_method === "not_attempted" || validationErrors(r).length > 0;
