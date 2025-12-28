export type HealthResponse = { status: string }

export type DomainsResponse = { domains: string[] }

export type RetrievedChunk = {
  chunk_id: number
  source: string
  text_preview: string
  score?: number | null
}

export type PipelineResult = {
  pipeline: string
  model: string
  generation_config: Record<string, unknown>
  answer: string
  latency_ms: number
  tokens_in?: number | null
  tokens_out?: number | null
  cost_estimate_usd?: number | null
  retrieved_chunks?: RetrievedChunk[] | null
  flags: Record<string, unknown>
  error?: string | null
}

export type EvaluationResult = {
  quality_score?: number | null
  hallucination_flags: string[]
  grounding_flags: string[]
  notes?: string | null
}

export type RunResponse = {
  run_id: string
  timestamp: string
  domain: string
  query: string
  results: Record<string, PipelineResult>
  evaluations: Record<string, EvaluationResult>
  summary_metrics: Record<string, unknown>
}

export type RunRequest = {
  query: string
  domain: string
  pipelines: string[]
  run_id?: string
  client_metadata?: Record<string, unknown> | null
}

