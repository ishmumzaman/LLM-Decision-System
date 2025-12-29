import type { EvaluationResult, RunResponse } from '../types'

export type MetricKey =
  | 'expect'
  | 'heuristic'
  | 'abstention'
  | 'latency'
  | 'cost'
  | 'judge'
  | 'evidence'

export type ScoringPresetId = 'docs_default' | 'general_default' | 'cost_sensitive' | 'latency_sensitive'

export type MetricSetting = { enabled: boolean; weight: number }

export type GateSetting = {
  require_expect_full: boolean
  require_abstention_correct: boolean
  require_no_grounding_flags: boolean
  require_no_hallucination_flags: boolean
}

export type ScoringConfig = {
  preset: ScoringPresetId
  metrics: Record<MetricKey, MetricSetting>
  gates: GateSetting
}

export type ScoringConfigWithHash = ScoringConfig & {
  hash: string
  is_custom: boolean
}

export const PRESETS: Record<
  ScoringPresetId,
  {
    label: string
    config: ScoringConfig
  }
> = {
  docs_default: {
    label: 'Docs-grounded (preset)',
    config: {
      preset: 'docs_default',
      metrics: {
        expect: { enabled: true, weight: 5 },
        heuristic: { enabled: true, weight: 3 },
        abstention: { enabled: true, weight: 3 },
        latency: { enabled: true, weight: 1 },
        cost: { enabled: true, weight: 1 },
        judge: { enabled: false, weight: 2 },
        evidence: { enabled: false, weight: 2 },
      },
      gates: {
        require_expect_full: false,
        require_abstention_correct: false,
        require_no_grounding_flags: false,
        require_no_hallucination_flags: false,
      },
    },
  },
  general_default: {
    label: 'General (preset)',
    config: {
      preset: 'general_default',
      metrics: {
        expect: { enabled: true, weight: 5 },
        heuristic: { enabled: true, weight: 3 },
        abstention: { enabled: true, weight: 2 },
        latency: { enabled: true, weight: 1 },
        cost: { enabled: true, weight: 1 },
        judge: { enabled: false, weight: 2 },
        evidence: { enabled: false, weight: 0 },
      },
      gates: {
        require_expect_full: false,
        require_abstention_correct: false,
        require_no_grounding_flags: false,
        require_no_hallucination_flags: false,
      },
    },
  },
  cost_sensitive: {
    label: 'Cost-sensitive (preset)',
    config: {
      preset: 'cost_sensitive',
      metrics: {
        expect: { enabled: true, weight: 4 },
        heuristic: { enabled: true, weight: 2 },
        abstention: { enabled: true, weight: 2 },
        latency: { enabled: true, weight: 1 },
        cost: { enabled: true, weight: 4 },
        judge: { enabled: false, weight: 2 },
        evidence: { enabled: false, weight: 2 },
      },
      gates: {
        require_expect_full: false,
        require_abstention_correct: false,
        require_no_grounding_flags: false,
        require_no_hallucination_flags: false,
      },
    },
  },
  latency_sensitive: {
    label: 'Latency-sensitive (preset)',
    config: {
      preset: 'latency_sensitive',
      metrics: {
        expect: { enabled: true, weight: 4 },
        heuristic: { enabled: true, weight: 2 },
        abstention: { enabled: true, weight: 2 },
        latency: { enabled: true, weight: 4 },
        cost: { enabled: true, weight: 1 },
        judge: { enabled: false, weight: 2 },
        evidence: { enabled: false, weight: 2 },
      },
      gates: {
        require_expect_full: false,
        require_abstention_correct: false,
        require_no_grounding_flags: false,
        require_no_hallucination_flags: false,
      },
    },
  },
}

function stableStringify(value: unknown): string {
  if (value == null) return 'null'
  if (typeof value !== 'object') return JSON.stringify(value)
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`
  const obj = value as Record<string, unknown>
  const keys = Object.keys(obj).sort()
  return `{${keys.map((k) => `${JSON.stringify(k)}:${stableStringify(obj[k])}`).join(',')}}`
}

function fnv1aHex(input: string): string {
  let hash = 0x811c9dc5
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(16).padStart(8, '0')
}

export function withHash(config: ScoringConfig): ScoringConfigWithHash {
  const payload = { preset: config.preset, metrics: config.metrics, gates: config.gates }
  const hash = fnv1aHex(stableStringify(payload))
  const presetHash = fnv1aHex(stableStringify(PRESETS[config.preset].config))
  return { ...config, hash, is_custom: hash !== presetHash }
}

type MetricValue = { value: number; note?: string }

export type PipelineScore = {
  score_0_10: number | null
  failed_gates: string[]
  used: Record<string, { value: number; weight: number; contrib: number }>
}

export function computeScores(run: RunResponse, config: ScoringConfigWithHash): Record<string, PipelineScore> {
  const pipelines = Object.keys(run.results || {})

  const latencies = pipelines
    .map((p) => ({ p, v: run.results[p]?.latency_ms }))
    .filter((x) => typeof x.v === 'number') as { p: string; v: number }[]
  const costs = pipelines
    .map((p) => ({ p, v: run.results[p]?.cost_estimate_usd }))
    .filter((x) => typeof x.v === 'number') as { p: string; v: number }[]

  const minLatency = latencies.length ? Math.min(...latencies.map((x) => x.v)) : null
  const maxLatency = latencies.length ? Math.max(...latencies.map((x) => x.v)) : null
  const minCost = costs.length ? Math.min(...costs.map((x) => x.v)) : null
  const maxCost = costs.length ? Math.max(...costs.map((x) => x.v)) : null

  function normalizeLowerIsBetter(v: number, min: number | null, max: number | null): number | null {
    if (min == null || max == null) return null
    if (max === min) return 1
    return 1 - (v - min) / (max - min)
  }

  function metricValues(p: string, evaluation: EvaluationResult | undefined): Record<MetricKey, MetricValue | null> {
    const v: Record<MetricKey, MetricValue | null> = {
      expect:
        typeof evaluation?.expect_score === 'number'
          ? { value: clamp01(evaluation.expect_score) }
          : null,
      heuristic:
        typeof evaluation?.quality_score === 'number' ? { value: clamp01(evaluation.quality_score) } : null,
      abstention:
        typeof evaluation?.abstention_score === 'number'
          ? { value: clamp01(evaluation.abstention_score) }
          : null,
      latency:
        typeof run.results[p]?.latency_ms === 'number'
          ? (() => {
              const norm = normalizeLowerIsBetter(run.results[p].latency_ms, minLatency, maxLatency)
              return norm == null ? null : { value: clamp01(norm) }
            })()
          : null,
      cost:
        typeof run.results[p]?.cost_estimate_usd === 'number'
          ? (() => {
              const norm = normalizeLowerIsBetter(run.results[p].cost_estimate_usd!, minCost, maxCost)
              return norm == null ? null : { value: clamp01(norm) }
            })()
          : null,
      judge:
        typeof run.judge?.scores?.[p] === 'number'
          ? { value: clamp01(run.judge.scores[p] / 10) }
          : null,
      evidence:
        typeof run.proxies?.evidence_support?.[p]?.support_score === 'number'
          ? { value: clamp01(run.proxies.evidence_support[p].support_score! / 2) }
          : null,
    }
    return v
  }

  const out: Record<string, PipelineScore> = {}
  for (const p of pipelines) {
    const result = run.results[p]
    const evaluation = run.evaluations[p]
    const failed: string[] = []
    if (result?.error) failed.push('pipeline_error')
    if (config.gates.require_expect_full && typeof evaluation?.expect_score === 'number') {
      if (evaluation.expect_score < 1) failed.push('expect_full')
    }
    if (config.gates.require_abstention_correct && typeof evaluation?.abstention_score === 'number') {
      if (evaluation.abstention_score < 1) failed.push('abstention_correct')
    }
    if (config.gates.require_no_grounding_flags && (evaluation?.grounding_flags?.length || 0) > 0) {
      failed.push('no_grounding_flags')
    }
    if (config.gates.require_no_hallucination_flags && (evaluation?.hallucination_flags?.length || 0) > 0) {
      failed.push('no_hallucination_flags')
    }

    const values = metricValues(p, evaluation)
    const used: Record<string, { value: number; weight: number; contrib: number }> = {}
    if (failed.length) {
      out[p] = { score_0_10: null, failed_gates: failed, used }
      continue
    }

    let sum = 0
    let sumW = 0
    for (const [k, setting] of Object.entries(config.metrics) as [MetricKey, MetricSetting][]) {
      if (!setting.enabled) continue
      const mv = values[k]
      if (!mv) continue
      const w = Math.max(0, Number(setting.weight) || 0)
      if (w <= 0) continue
      sum += mv.value * w
      sumW += w
      used[k] = { value: mv.value, weight: w, contrib: mv.value * w }
    }
    const score = sumW > 0 ? (sum / sumW) * 10 : null
    out[p] = { score_0_10: score != null ? round2(score) : null, failed_gates: [], used }
  }
  return out
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v))
}

function round2(v: number): number {
  return Math.round(v * 100) / 100
}
