export type HealthResponse = { status: string }

export type DomainsResponse = { domains: string[] }

export type SuiteInfo = {
  id: string
  suite: string
  domain?: string | null
  description?: string | null
  cases: number
}

export type ExpectationSpec = {
  must_include?: string[] | string
  must_not_include?: string[] | string
  must_include_any?: (string | string[])[] | string | string[]
  expect_idk?: boolean
}

export type SuiteCase = {
  id: string
  query: string
  tags: string[]
  expect: ExpectationSpec
}

export type SuitesResponse = { suites: SuiteInfo[] }

export type SuiteResponse = {
  id: string
  suite: string
  domain?: string | null
  description?: string | null
  queries: SuiteCase[]
}

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
  penalty_total?: number | null
  rule_breakdown?: {
    rule: string
    penalty: number
    hallucination_flags: string[]
    grounding_flags: string[]
    note?: string | null
  }[]
  expect_score?: number | null
  expect_details?: {
    missing?: string[]
    forbidden?: string[]
  } | null
  hallucination_flags: string[]
  grounding_flags: string[]
  notes?: string | null
}

export type RunResponse = {
  run_id: string
  timestamp: string
  domain: string
  mode: 'docs' | 'general'
  query: string
  results: Record<string, PipelineResult>
  evaluations: Record<string, EvaluationResult>
  summary_metrics: Record<string, unknown>
  judge?: {
    winner?: string | null
    scores: Record<string, number>
    criteria: Record<string, Record<string, number>>
    rationale?: string | null
    model?: string | null
    latency_ms?: number | null
    tokens_in?: number | null
    tokens_out?: number | null
    cost_estimate_usd?: number | null
    error?: string | null
  } | null
}

export type RunRequest = {
  query: string
  domain: string
  mode: 'docs' | 'general'
  pipelines: string[]
  expect?: ExpectationSpec | null
  judge?: boolean
  judge_model?: string | null
  run_id?: string
  client_metadata?: Record<string, unknown> | null
}
