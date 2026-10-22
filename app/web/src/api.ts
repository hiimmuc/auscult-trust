// Client for the inference API. Types mirror the JSON written by api/engine.py and api/main.py.

export type Class4 = "normal" | "crackle" | "wheeze" | "both";
export type Class2 = "normal" | "abnormal";

export interface Segment {
  index: number;
  start: number;
  end: number;
  probs4: Record<Class4, number>;
  pred4: Class4;
  set4: Class4[];
  probs2: Record<Class2, number>;
  pred2: Class2;
  set2: Class2[];
  truth4?: Class4;
  truth2?: Class2;
}

export interface PipelineResult {
  label: string;
  sc: { applied: false } | { applied: true; mode: string; coefficient_source: string; gain_db: number[] };
  segments: Segment[];
  recording: {
    p_abnormal: number;
    set: Class2[];
    triage: "refer" | "no_referral" | "uncertain" | "rerecord";
    aggregation: string;
    windows: { start: number; end: number; p_abnormal: number }[];
  };
}

export interface InferResult {
  request_id: string;
  source: { kind: "sample" | "upload"; id?: string };
  audio: { filename: string; sample_rate: number; channels: number; duration_sec: number; subtype: string; peak_dbfs: number; clipped_share: number; sha256: string };
  device: { declared: string | null };
  segmentation: { mode: "cycles" | "windows"; window_sec?: number; hop_sec?: number };
  alignment?: { reference_sample: string; lag_sec: number; peak_score: number; reliable: boolean; trimmed_to_sec: number };
  pipelines: Record<string, PipelineResult>;
  cached: boolean; // served from the API's result cache (same file, same settings)
  timing_ms: number;
}

export interface ModelInfo {
  classes: Class4[];
  classes2: Class2[];
  devices: string[];
  alpha: number;
  preprocessing: Record<string, string | number>;
  pipelines: Record<string, {
    label: string; checkpoint_sha256: string; epoch: number; encoder: string; method: string; spectrum_correction: boolean;
    thresholds: { cycle4: number; cycle2: number; screen: number };
    calibration: { cycles: number; recordings: number; patients: number };
    sc?: { mode: string; reference: string; s_ref_sha256: string };
  }>;
  limits: { max_bytes: number; min_sec: number; max_sec: number; sample_rate: [number, number]; channels: [number, number] };
  sc_freqs_hz: number[];
}

export interface Sample {
  id: string;
  class: Class4;
  patient: string;
  device: string;
  chest_location: string;
  acquisition: string;
  class_share: number;
  class_cycles: number;
  cycles_total: number;
  cycles: { start: number; end: number; label: Class4 }[];
  sha256: string;
}

export class ApiError extends Error {
  constructor(public code: string, message: string, public requestId?: string, public status?: number) {
    super(message);
  }
}

const KEY = "auscult-trust.connection";

export function connection(): { base: string; token: string } {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) || "null");
    if (saved?.base) return saved;
  } catch { /* storage blocked: fall back to the build default */ }
  return { base: import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000", token: "" };
}

export function saveConnection(c: { base: string; token: string }) {
  try { localStorage.setItem(KEY, JSON.stringify(c)); } catch { /* not persisted */ }
  active = "";
}

/** The base in use: the chosen API, or the fallback after the chosen one failed to answer. */
let active = "";
export const activeBase = () => active || connection().base;
export const useBase = (base: string) => (active = base);
/** The chosen API first, then the build's fallback (VITE_API_FALLBACK), when there is one. */
export const candidateBases = () => [...new Set([connection().base, import.meta.env.VITE_API_FALLBACK].filter(Boolean) as string[])];
export const onFallback = () => active !== "" && active !== connection().base;

async function call(path: string, init: RequestInit = {}, timeoutMs = 30000, signal?: AbortSignal, baseOverride?: string): Promise<Response> {
  const base = baseOverride ?? activeBase(), { token } = connection();
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort("timeout"), timeoutMs);
  signal?.addEventListener("abort", () => ctl.abort("cancelled"));
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  let res: Response;
  try {
    res = await fetch(base.replace(/\/$/, "") + path, { ...init, headers, signal: ctl.signal });
  } catch (e) {
    if (ctl.signal.aborted) {
      throw ctl.signal.reason === "cancelled" ? new ApiError("cancelled", "Cancelled.") : new ApiError("timeout", `No answer from the API after ${timeoutMs / 1000} s.`);
    }
    throw new ApiError("network", `Cannot reach the API at ${base}. Check the address, your connection, and CORS.`);
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    let body: any = null;
    try { body = await res.json(); } catch { /* not JSON */ }
    const rid = body?.request_id || res.headers.get("X-Request-ID") || undefined;
    throw new ApiError(body?.error?.code || `http_${res.status}`, body?.error?.message || `API answered ${res.status}.`, rid, res.status);
  }
  return res;
}

export const api = {
  // 60 s: a free host sleeping after idle needs about a minute to wake
  health: async (base?: string) => (await call("/v1/health", {}, 60000, undefined, base)).json() as Promise<{ status: "ok" | "unavailable"; pipelines: string[]; gpu: boolean; reason?: string }>,
  model: async () => (await call("/v1/model")).json() as Promise<ModelInfo>,
  samples: async () => ((await (await call("/v1/samples")).json()).samples as Sample[]),
  sampleAudio: async (id: string, signal?: AbortSignal) => (await call(`/v1/samples/${encodeURIComponent(id)}/audio`, {}, 30000, signal)).arrayBuffer(),
  sampleResult: async (id: string, signal?: AbortSignal) =>
    (await call(`/v1/samples/${encodeURIComponent(id)}/result`, {}, 130000, signal)).json() as Promise<InferResult>,
  infer: async (fields: Record<string, Blob | string>, signal?: AbortSignal) => {
    const fd = new FormData();
    for (const [k, v] of Object.entries(fields)) v instanceof File ? fd.append(k, v, v.name) : v instanceof Blob ? fd.append(k, v, `${k}.wav`) : fd.append(k, v);
    return (await call("/v1/infer", { method: "POST", body: fd }, 130000, signal)).json() as Promise<InferResult>;
  },
};
