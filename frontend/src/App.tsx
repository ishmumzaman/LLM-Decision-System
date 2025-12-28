import { useEffect, useMemo, useState } from 'react'
import { getDomains, getHealth, runOnce } from './lib/api'
import { formatInt, formatIsoTimestamp, formatUsd } from './lib/format'
import type { EvaluationResult, PipelineResult, RetrievedChunk, RunResponse } from './types'

function App() {
  const [backendOk, setBackendOk] = useState<boolean | null>(null)
  const [domains, setDomains] = useState<string[]>([])
  const [domain, setDomain] = useState('fastapi_docs')
  const [query, setQuery] = useState('')
  const [pipelines, setPipelines] = useState<Record<string, boolean>>({ prompt: true, rag: true })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [run, setRun] = useState<RunResponse | null>(null)
  const [history, setHistory] = useState<RunResponse[]>([])

  const selectedPipelines = useMemo(() => {
    return Object.entries(pipelines)
      .filter(([, on]) => on)
      .map(([p]) => p)
  }, [pipelines])

  useEffect(() => {
    let canceled = false

    async function boot() {
      try {
        await getHealth()
        if (!canceled) setBackendOk(true)
      } catch {
        if (!canceled) setBackendOk(false)
      }

      try {
        const resp = await getDomains()
        if (!canceled) {
          setDomains(resp.domains || [])
          if (resp.domains?.includes(domain) === false && resp.domains.length > 0) {
            setDomain(resp.domains[0])
          }
        }
      } catch (e) {
        if (!canceled) setError(String(e))
      }
    }

    void boot()
    return () => {
      canceled = true
    }
  }, [])

  async function onRun() {
    setError(null)
    setLoading(true)
    try {
      const resp = await runOnce({
        domain,
        query,
        pipelines: selectedPipelines,
        run_id: crypto.randomUUID(),
      })
      setRun(resp)
      setHistory((prev) => [resp, ...prev].slice(0, 10))
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const canRun = query.trim().length > 0 && selectedPipelines.length > 0 && !loading

  return (
    <div className="min-h-full bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-6xl px-4 py-8">
        <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">LLM Decision System</h1>
            <p className="text-sm text-slate-300">
              Compare <span className="font-medium text-slate-100">prompt</span> vs{' '}
              <span className="font-medium text-slate-100">RAG</span> on identical inputs.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <StatusPill ok={backendOk} />
          </div>
        </header>

        <div className="mt-6 rounded-xl border border-slate-800 bg-slate-900/40 p-4">
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="sm:col-span-1">
              <label className="text-xs font-medium text-slate-300">Domain</label>
              <select
                value={domain}
                onChange={(e) => setDomain(e.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-slate-500"
              >
                {(domains.length ? domains : [domain]).map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>

            <div className="sm:col-span-2">
              <label className="text-xs font-medium text-slate-300">Pipelines</label>
              <div className="mt-1 flex flex-wrap gap-3">
                <PipelineToggle
                  label="Prompt"
                  checked={pipelines.prompt}
                  onChange={(v) => setPipelines((p) => ({ ...p, prompt: v }))}
                />
                <PipelineToggle
                  label="RAG"
                  checked={pipelines.rag}
                  onChange={(v) => setPipelines((p) => ({ ...p, rag: v }))}
                />
              </div>
            </div>
          </div>

          <div className="mt-4">
            <label className="text-xs font-medium text-slate-300">Query</label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question about FastAPI docs…"
              rows={4}
              className="mt-1 w-full resize-y rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-slate-500"
            />
            <div className="mt-2 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div className="text-xs text-slate-400">{query.trim().length}/4000</div>
              <button
                onClick={onRun}
                disabled={!canRun}
                className="inline-flex items-center justify-center rounded-lg bg-indigo-500 px-4 py-2 text-sm font-medium text-white transition disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? 'Running…' : 'Run'}
              </button>
            </div>
          </div>
        </div>

        {error ? (
          <div className="mt-4 rounded-lg border border-rose-900/60 bg-rose-950/30 p-3 text-sm text-rose-200">
            {error}
          </div>
        ) : null}

        {run ? (
          <div className="mt-6">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div className="text-sm text-slate-300">
                <span className="font-medium text-slate-100">Run</span> {run.run_id}{' '}
                <span className="text-slate-500">·</span> {formatIsoTimestamp(run.timestamp)}
              </div>
              <SummaryBar summary={run.summary_metrics} />
            </div>

            <div className="mt-4 grid gap-4 md:grid-cols-2">
              {selectedPipelines.map((p) => (
                <PipelineCard
                  key={p}
                  pipeline={p}
                  result={run.results[p]}
                  evaluation={run.evaluations[p]}
                />
              ))}
            </div>
          </div>
        ) : null}

        <div className="mt-10 border-t border-slate-800 pt-6">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-100">Recent runs</h2>
            <button
              onClick={() => setHistory([])}
              className="text-xs text-slate-400 hover:text-slate-200"
              type="button"
            >
              Clear
            </button>
          </div>
          <div className="mt-3 grid gap-2">
            {history.length === 0 ? (
              <div className="text-sm text-slate-400">No runs yet.</div>
            ) : (
              history.map((h) => (
                <button
                  key={h.run_id}
                  onClick={() => setRun(h)}
                  className="flex w-full items-start justify-between gap-3 rounded-lg border border-slate-800 bg-slate-900/30 px-3 py-2 text-left hover:border-slate-700"
                  type="button"
                >
                  <div className="min-w-0">
                    <div className="truncate text-sm text-slate-100">{h.query}</div>
                    <div className="text-xs text-slate-400">{formatIsoTimestamp(h.timestamp)}</div>
                  </div>
                  <div className="shrink-0 text-xs text-slate-400">{h.domain}</div>
                </button>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

function StatusPill({ ok }: { ok: boolean | null }) {
  const label = ok == null ? 'Backend: unknown' : ok ? 'Backend: ok' : 'Backend: down'
  const classes =
    ok == null
      ? 'border-slate-700 bg-slate-900/40 text-slate-200'
      : ok
        ? 'border-emerald-700/60 bg-emerald-950/30 text-emerald-200'
        : 'border-rose-800/60 bg-rose-950/30 text-rose-200'
  return (
    <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs ${classes}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-current opacity-80" />
      {label}
    </span>
  )
}

function PipelineToggle({
  label,
  checked,
  onChange,
}: {
  label: string
  checked: boolean
  onChange: (value: boolean) => void
}) {
  return (
    <label className="inline-flex cursor-pointer select-none items-center gap-2 text-sm text-slate-100">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="h-4 w-4 rounded border-slate-600 bg-slate-950 text-indigo-500"
      />
      {label}
    </label>
  )
}

function SummaryBar({ summary }: { summary: Record<string, unknown> }) {
  const winnerByQuality = summary.winner_by_quality as string | null | undefined
  const winnerByLatency = summary.winner_by_latency as string | null | undefined
  const winnerByCost = summary.winner_by_cost as string | null | undefined
  const qualityTies = summary.winner_by_quality_ties as string[] | undefined

  return (
    <div className="flex flex-wrap gap-2">
      <Chip label={`quality: ${winnerByQuality ?? (qualityTies?.length ? 'tie' : '—')}`} tone="indigo" />
      <Chip label={`latency: ${winnerByLatency ?? '—'}`} tone="slate" />
      <Chip label={`cost: ${winnerByCost ?? '—'}`} tone="slate" />
    </div>
  )
}

function PipelineCard({
  pipeline,
  result,
  evaluation,
}: {
  pipeline: string
  result: PipelineResult | undefined
  evaluation: EvaluationResult | undefined
}) {
  if (!result) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4">
        <div className="text-sm text-slate-300">No result for pipeline: {pipeline}</div>
      </div>
    )
  }

  const isRag = pipeline === 'rag'
  const title = pipeline === 'prompt' ? 'Prompt' : pipeline.toUpperCase()
  const latency = result.latency_ms
  const tokensIn = result.tokens_in ?? null
  const tokensOut = result.tokens_out ?? null
  const costUsd = result.cost_estimate_usd ?? null
  const quality = evaluation?.quality_score ?? null

  const pipelineFlags = Object.keys(result.flags || {})
  const hallucFlags = evaluation?.hallucination_flags || []
  const groundingFlags = evaluation?.grounding_flags || []

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-slate-100">{title}</div>
          <div className="mt-1 text-xs text-slate-400">{result.model}</div>
        </div>
        <div className="flex flex-wrap justify-end gap-2">
          <Chip label={`quality: ${quality == null ? '—' : quality.toFixed(2)}`} tone="indigo" />
          <Chip label={`${latency}ms`} tone="slate" />
          <Chip label={`in ${formatInt(tokensIn)} · out ${formatInt(tokensOut)}`} tone="slate" />
          <Chip label={formatUsd(costUsd)} tone="slate" />
        </div>
      </div>

      {result.error ? (
        <div className="mt-3 rounded-lg border border-rose-900/60 bg-rose-950/30 p-3 text-sm text-rose-200">
          {result.error}
        </div>
      ) : (
        <>
          <div className="mt-4 whitespace-pre-wrap text-sm leading-relaxed text-slate-100">{result.answer}</div>

          <div className="mt-4 flex flex-wrap gap-2">
            {pipelineFlags.map((f) => (
              <Chip key={`pf:${f}`} label={f} tone="blue" />
            ))}
            {hallucFlags.map((f) => (
              <Chip key={`hf:${f}`} label={f} tone="rose" />
            ))}
            {groundingFlags.map((f) => (
              <Chip key={`gf:${f}`} label={f} tone="amber" />
            ))}
            {pipelineFlags.length + hallucFlags.length + groundingFlags.length === 0 ? (
              <div className="text-xs text-slate-400">No flags</div>
            ) : null}
          </div>

          {evaluation?.notes ? <div className="mt-2 text-xs text-slate-400">{evaluation.notes}</div> : null}

          {isRag ? <EvidencePanel chunks={result.retrieved_chunks || []} /> : null}
        </>
      )}
    </div>
  )
}

function EvidencePanel({ chunks }: { chunks: RetrievedChunk[] }) {
  if (!chunks.length) {
    return <div className="mt-4 text-xs text-slate-400">No retrieved chunks.</div>
  }
  return (
    <div className="mt-4">
      <div className="text-xs font-medium text-slate-300">Evidence</div>
      <div className="mt-2 grid gap-2">
        {chunks.map((c) => (
          <details
            key={c.chunk_id}
            className="rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2"
          >
            <summary className="cursor-pointer text-xs text-slate-200">
              [{c.chunk_id}] <span className="text-slate-400">{c.source}</span>{' '}
              {typeof c.score === 'number' ? (
                <span className="text-slate-500">· score {c.score.toFixed(3)}</span>
              ) : null}
            </summary>
            <pre className="mt-2 whitespace-pre-wrap break-words text-xs text-slate-200">{c.text_preview}</pre>
          </details>
        ))}
      </div>
    </div>
  )
}

function Chip({
  label,
  tone,
}: {
  label: string
  tone: 'slate' | 'indigo' | 'rose' | 'amber' | 'blue'
}) {
  const cls =
    tone === 'indigo'
      ? 'border-indigo-700/60 bg-indigo-950/40 text-indigo-200'
      : tone === 'rose'
        ? 'border-rose-800/60 bg-rose-950/30 text-rose-200'
        : tone === 'amber'
          ? 'border-amber-700/60 bg-amber-950/20 text-amber-200'
          : tone === 'blue'
            ? 'border-sky-700/60 bg-sky-950/30 text-sky-200'
            : 'border-slate-700 bg-slate-900/40 text-slate-200'

  return <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs ${cls}`}>{label}</span>
}
