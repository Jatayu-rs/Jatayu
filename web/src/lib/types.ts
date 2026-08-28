// Types matching src/jatayu/schemas.py — DO NOT hand-edit, regenerate if schemas change

export interface BoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  label?: string;
  score?: number;
  image_index: number;
}

export interface Evidence {
  kind: "bbox" | "mask" | "overlay" | "table" | "none";
  boxes: BoundingBox[];
  mask_path?: string;
  overlay_png?: string;
  geojson?: Record<string, unknown>;
  legend: Record<string, string>;
  caption?: string;
}

export interface TraceStep {
  step: number;
  stage: "validate" | "classify" | "route" | "execute" | "combine";
  detail: string;
  tool_name?: string;
  model_id?: string;
  params: Record<string, unknown>;
  duration_ms: number;
}

export interface ValidationIssue {
  severity: "error" | "warning" | "info";
  code: string;
  message: string;
  image_index?: number;
}

export interface ValidationReport {
  ok: boolean;
  issues: ValidationIssue[];
}

export interface QueryResponse {
  answer: string;
  evidence: Evidence;
  confidence: number;
  confidence_method: string;
  task_family: string;
  tools_used: string[];
  trace: TraceStep[];
  validation: ValidationReport;
  total_latency_ms: number;
  request_id?: string;
}

export interface LanguageInfo {
  detected: string;
  display_name: string;
  was_translated: boolean;
  original_query?: string;
}

export interface AnalyzeResponse {
  result: QueryResponse;
  language?: LanguageInfo;
  request_id?: string;
}

export interface SampleScenario {
  id: string;
  title: string;
  description: string;
  family: string;
  suggested_query: string;
  files: string[];
  modalities: string[];
}

export interface HealthResponse {
  status: string;
  tools_registered: string[];
  samples_available: number;
}
