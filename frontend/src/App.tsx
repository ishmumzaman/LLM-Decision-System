import { useEffect, useMemo, useState } from 'react'
import { getDomains, getHealth, getSuite, getSuites, runOnce } from './lib/api'
import { formatInt, formatIsoTimestamp, formatUsd } from './lib/format'
import { PRESETS, computeScores, withHash } from './lib/scoring'
import type {
  EvaluationResult,
  PipelineResult,
  RetrievedChunk,
  RunResponse,
  SuiteCase,
  SuiteInfo,
} from './types'

function App() {
  const [backendOk, setBackendOk] = useState<boolean | null>(null)
  const [domains, setDomains] = useState<string[]>([])
  const [domain, setDomain] = useState('fastapi_docs')
  const [mode, setMode] = useState<'docs' | 'general'>('docs')
  const [querySource, setQuerySource] = useState<'custom' | 'suite'>('custom')
  const [suites, setSuites] = useState<SuiteInfo[]>([])
  const [suiteId, setSuiteId] = useState<string | null>(null)
  const [suiteCases, setSuiteCases] = useState<SuiteCase[]>([])
  const [caseId, setCaseId] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [pipelines, setPipelines] = useState<Record<string, boolean>>({
    prompt: true,
    rag: true,
    finetune: false,
  })
  const [useJudge, setUseJudge] = useState(false)
  const [judgeModel, setJudgeModel] = useState('gpt-4o')
  const [proxyEvidence, setProxyEvidence] = useState(false)
  const [proxyAnswerability, setProxyAnswerability] = useState(false)
  const [scoringConfig, setScoringConfig] = useState(() => PRESETS.docs_default.config)
  const [customScoring, setCustomScoring] = useState(false)
  const [customScoringAck, setCustomScoringAck] = useState(() => {
    try {
      return localStorage.getItem('scoring_custom_ack_v1') === 'true'
    } catch {
      return false
    }
  })
  const [showCustomScoringWarning, setShowCustomScoringWarning] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [run, setRun] = useState<RunResponse | null>(null)
  const [history, setHistory] = useState<RunResponse[]>([])

  const scoring = useMemo(() => withHash(scoringConfig), [scoringConfig])
  const scoreByPipeline = useMemo(() => (run ? computeScores(run, scoring) : null), [run, scoring])

  const selectedCase = useMemo(() => {
    if (querySource !== 'suite') return null
    return suiteCases.find((c) => c.id === caseId) || null
  }, [querySource, suiteCases, caseId])

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

      try {
        const resp = await getSuites()
        if (!canceled) {
          setSuites(resp.suites || [])
        }
      } catch {
        // optional (dev-only endpoint)
      }
    }

    void boot()
    return () => {
      canceled = true
    }
  }, [])

  useEffect(() => {
    if (querySource !== 'suite') return
    if (suiteId) return
    if (suites.length) {
      setSuiteId(suites[0].id)
    }
  }, [querySource, suiteId, suites])

  useEffect(() => {
    let canceled = false

    async function load() {
      if (!suiteId) return
      try {
        const resp = await getSuite(suiteId)
        if (canceled) return
        const cases = resp.queries || []
        setSuiteCases(cases)
        const first = cases[0] || null
        setCaseId(first?.id ?? null)
        if (first) setQuery(first.query)
        if (typeof resp.domain === 'string' && resp.domain.trim()) {
          setDomain(resp.domain)
        }
      } catch (e) {
        if (!canceled) setError(String(e))
      }
    }

    if (querySource === 'suite') {
      void load()
    }
    return () => {
      canceled = true
    }
  }, [querySource, suiteId])

  useEffect(() => {
    if (querySource !== 'suite') return
    if (!caseId) return
    const c = suiteCases.find((c) => c.id === caseId) || null
    if (c) setQuery(c.query)
  }, [querySource, suiteCases, caseId])

  async function onRun() {
    setError(null)
    setLoading(true)
    try {
      if (querySource === 'suite' && !selectedCase) {
        throw new Error('Select a suite and case first.')
      }
      const resp = await runOnce({
        domain,
        mode,
        query,
        pipelines: selectedPipelines,
        expect: querySource === 'suite' ? selectedCase?.expect ?? null : null,
        case:
          querySource === 'suite' && selectedCase
            ? {
                id: selectedCase.id,
                tags: selectedCase.tags,
                answerable_from_general_knowledge:
                  selectedCase.answerable_from_general_knowledge ?? null,
                requires_docs: selectedCase.requires_docs ?? null,
                expected_abstain_in_docs: selectedCase.expected_abstain_in_docs ?? null,
              }
            : null,
        judge: useJudge,
        judge_model: useJudge ? judgeModel.trim() || null : null,
        proxy_evidence: proxyEvidence,
        proxy_answerability: proxyAnswerability,
        scoring,
        run_id: crypto.randomUUID(),
        client_metadata:
          querySource === 'suite' && suiteId && selectedCase
            ? { suite: suiteId, case_id: selectedCase.id, tags: selectedCase.tags }
            : null,
      })
      setRun(resp)
      setHistory((prev) => [resp, ...prev].slice(0, 10))
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const canRun =
    query.trim().length > 0 &&
    selectedPipelines.length > 0 &&
    !loading &&
    (querySource !== 'suite' || Boolean(selectedCase))

  return (
    <div className="min-h-full bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-6xl px-4 py-8">
        <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">LLM Decision System</h1>
            <p className="text-sm text-slate-300">
              Compare <span className="font-medium text-slate-100">prompt</span>,{' '}
              <span className="font-medium text-slate-100">RAG</span>, and{' '}
              <span className="font-medium text-slate-100">fine-tuned</span> on identical inputs.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <StatusPill ok={backendOk} />
          </div>
        </header>

        <div className="mt-6 rounded-xl border border-slate-800 bg-slate-900/40 p-4">
          <div className="grid gap-3 sm:grid-cols-4">
            <div className="sm:col-span-1">
              <label className="text-xs font-medium text-slate-300">Domain</label>
              <select
                value={domain}
                onChange={(e) => setDomain(e.target.value)}
                disabled={querySource === 'suite'}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-slate-500"
              >
                {(domains.length ? domains : [domain]).map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>

            <div className="sm:col-span-1">
              <label className="text-xs font-medium text-slate-300">Mode</label>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as 'docs' | 'general')}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-slate-500"
              >
                <option value="docs">Docs-grounded</option>
                <option value="general">General</option>
              </select>
              <div className="mt-1 text-xs text-slate-400">
                {mode === 'docs'
                  ? 'Answers should be grounded in the selected domain corpus (RAG uses only retrieved context).'
                  : 'Answers may use general knowledge (RAG adds optional domain context).'}{' '}
              </div>
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
                <PipelineToggle
                  label="Fine-tuned"
                  checked={pipelines.finetune}
                  onChange={(v) => setPipelines((p) => ({ ...p, finetune: v }))}
                />
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <PipelineToggle label="Judge" checked={useJudge} onChange={setUseJudge} />
                <div className="text-xs text-slate-400">
                  Runs an extra model call to score/rank outputs (costs $).
                </div>
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <PipelineToggle
                  label="Evidence (LLM)"
                  checked={proxyEvidence}
                  onChange={setProxyEvidence}
                />
                <PipelineToggle
                  label="Answerability (LLM)"
                  checked={proxyAnswerability}
                  onChange={setProxyAnswerability}
                />
                <div className="text-xs text-slate-400">Extra eval calls (costs $).</div>
              </div>
              {useJudge ? (
                <div className="mt-2 grid gap-2 sm:max-w-sm">
                  <label className="text-xs font-medium text-slate-300">Judge model</label>
                  <input
                    value={judgeModel}
                    onChange={(e) => setJudgeModel(e.target.value)}
                    placeholder="gpt-4o"
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-slate-500"
                  />
                  <div className="text-xs text-slate-400">
                    Recommended: use a stronger/different model than the participants.
                  </div>
                </div>
              ) : null}

              <div className="mt-4 grid gap-2 sm:max-w-sm">
                <label className="text-xs font-medium text-slate-300">Overall scoring</label>
                <select
                  value={scoringConfig.preset}
                  onChange={(e) => {
                    const preset = e.target.value as keyof typeof PRESETS
                    setScoringConfig(PRESETS[preset].config)
                    setCustomScoring(false)
                  }}
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-slate-500"
                >
                  {Object.entries(PRESETS).map(([id, p]) => (
                    <option key={id} value={id}>
                      {p.label}
                    </option>
                  ))}
                </select>

                <div className="text-xs text-slate-400">
                  config hash: <span className="font-mono">{scoring.hash}</span>{' '}
                  {scoring.is_custom ? (
                    <span className="text-rose-300">custom (non-comparable)</span>
                  ) : (
                    <span className="text-slate-400">preset</span>
                  )}
                </div>

                <label className="mt-1 inline-flex cursor-pointer select-none items-center gap-2 text-xs text-slate-200">
                  <input
                    type="checkbox"
                    checked={customScoring}
                    onChange={(e) => {
                      const next = e.target.checked
                      if (next) {
                        if (!customScoringAck) {
                          setShowCustomScoringWarning(true)
                          return
                        }
                        setCustomScoring(true)
                        return
                      }
                      setCustomScoring(false)
                      setScoringConfig(PRESETS[scoringConfig.preset].config)
                    }}
                    className="h-4 w-4 rounded border-slate-600 bg-slate-950 text-indigo-500"
                  />
                  Advanced: custom scoring (non-comparable)
                </label>

                <details className="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
                  <summary className="cursor-pointer text-xs font-medium text-slate-200">
                    Metrics board
                  </summary>
                  <div className="mt-2 grid gap-2">
                    {(
                      [
                        ['expect', 'Expect score (deterministic)'],
                        ['heuristic', 'Heuristic score (rules/penalties)'],
                        ['abstention', 'Abstention correctness (deterministic)'],
                        ['latency', 'Latency (normalized, lower is better)'],
                        ['cost', 'Cost (normalized, lower is better)'],
                        ['judge', 'Pairwise judge score (LLM)'],
                        ['evidence', 'Evidence support score (LLM)'],
                      ] as const
                    ).map(([key, label]) => {
                      const setting = scoringConfig.metrics[key]
                      return (
                        <div key={key} className="flex items-center justify-between gap-3">
                          <label className="flex items-center gap-2 text-xs text-slate-200">
                            <input
                              type="checkbox"
                              checked={setting.enabled}
                              disabled={!customScoring}
                              onChange={(e) =>
                                setScoringConfig((cfg) => ({
                                  ...cfg,
                                  metrics: {
                                    ...cfg.metrics,
                                    [key]: { ...cfg.metrics[key], enabled: e.target.checked },
                                  },
                                }))
                              }
                              className="h-4 w-4 rounded border-slate-600 bg-slate-950 text-indigo-500 disabled:opacity-50"
                            />
                            {label}
                          </label>
                          <input
                            type="number"
                            min={0}
                            step={0.5}
                            value={setting.weight}
                            disabled={!customScoring || !setting.enabled}
                            onChange={(e) =>
                              setScoringConfig((cfg) => ({
                                ...cfg,
                                metrics: {
                                  ...cfg.metrics,
                                  [key]: { ...cfg.metrics[key], weight: Number(e.target.value) || 0 },
                                },
                              }))
                            }
                            className="w-20 rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-100 outline-none focus:border-slate-500 disabled:opacity-50"
                          />
                        </div>
                      )
                    })}
                  </div>

                  <div className="mt-3 border-t border-slate-800 pt-3">
                    <div className="text-xs font-medium text-slate-200">Hard gates</div>
                    <div className="mt-2 grid gap-2">
                      {(
                        [
                          ['require_expect_full', 'Require full expect match (expect=1.0)'],
                          ['require_abstention_correct', 'Require abstention correctness (1.0)'],
                          ['require_no_grounding_flags', 'Require no grounding flags'],
                          ['require_no_hallucination_flags', 'Require no hallucination flags'],
                        ] as const
                      ).map(([gateKey, label]) => (
                        <label
                          key={gateKey}
                          className="flex cursor-pointer select-none items-center gap-2 text-xs text-slate-200"
                        >
                          <input
                            type="checkbox"
                            checked={scoringConfig.gates[gateKey]}
                            disabled={!customScoring}
                            onChange={(e) =>
                              setScoringConfig((cfg) => ({
                                ...cfg,
                                gates: { ...cfg.gates, [gateKey]: e.target.checked },
                              }))
                            }
                            className="h-4 w-4 rounded border-slate-600 bg-slate-950 text-indigo-500 disabled:opacity-50"
                          />
                          {label}
                        </label>
                      ))}
                    </div>
                  </div>
                </details>
              </div>
            </div>
          </div>

          <div className="mt-4">
            <label className="text-xs font-medium text-slate-300">Query</label>
            <div className="mt-2 grid gap-3 sm:grid-cols-3">
              <div>
                <label className="text-xs font-medium text-slate-300">Source</label>
                <select
                  value={querySource}
                  onChange={(e) => setQuerySource(e.target.value as 'custom' | 'suite')}
                  className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-slate-500"
                >
                  <option value="custom">Custom</option>
                  <option value="suite">Regression case</option>
                </select>
              </div>
              {querySource === 'suite' ? (
                <>
                  <div>
                    <label className="text-xs font-medium text-slate-300">Suite</label>
                    <select
                      value={suiteId ?? ''}
                      onChange={(e) => setSuiteId(e.target.value)}
                      className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-slate-500"
                    >
                      {(suites.length ? suites : suiteId ? [{ id: suiteId, suite: suiteId, cases: 0 }] : []).map(
                        (s) => (
                          <option key={s.id} value={s.id}>
                            {s.id}
                          </option>
                        ),
                      )}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-slate-300">Case</label>
                    <select
                      value={caseId ?? ''}
                      onChange={(e) => setCaseId(e.target.value)}
                      className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-slate-500"
                    >
                      {(suiteCases.length ? suiteCases : caseId ? [{ id: caseId, query: '', tags: [], expect: {} }] : []).map(
                        (c) => (
                          <option key={c.id} value={c.id}>
                            {c.id}
                          </option>
                        ),
                      )}
                    </select>
                  </div>
                </>
              ) : null}
            </div>
            {querySource === 'suite' ? (
              <div className="mt-2 text-xs text-slate-400">
                {selectedCase
                  ? `tags: ${selectedCase.tags?.length ? selectedCase.tags.join(', ') : 'none'}`
                  : 'Select a regression case to load the query + expectations.'}
              </div>
            ) : null}
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              readOnly={querySource === 'suite'}
              placeholder={
                mode === 'docs'
                  ? `Ask a question about ${domain} docs…`
                  : 'Ask a question (general mode)…'
              }
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
                <span className="text-slate-500">·</span> mode: {run.mode}
              </div>
              <SummaryBar summary={run.summary_metrics} />
            </div>

            {scoreByPipeline ? (
              <div className="mt-3 rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2">
                <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <div className="text-xs text-slate-300">
                    {scoring.is_custom
                      ? 'User score (custom)'
                      : `Overall score (preset: ${scoringConfig.preset})`}{' '}
                    <span className="text-slate-500">·</span>{' '}
                    <span className="font-mono text-slate-400">{scoring.hash}</span>
                    {(() => {
                      const runHash = (run.scoring as any)?.hash as string | undefined
                      if (!runHash) return null
                      if (runHash === scoring.hash) return null
                      return <span className="ml-2 text-rose-200">not comparable (run cfg {runHash})</span>
                    })()}
                  </div>
                  {scoring.is_custom ? (
                    <div className="text-xs text-rose-200">
                      Custom scoring is non-comparable across runs unless hashes match.
                    </div>
                  ) : null}
                </div>

                {run.proxies?.answerability ? (
                  <div className="mt-2 text-xs text-slate-400">
                    answerability (LLM):{' '}
                    <span className="text-slate-200">
                      {run.proxies.answerability.label ?? 'unknown'}
                    </span>
                    {typeof run.proxies.answerability.confidence === 'number'
                      ? ` (${run.proxies.answerability.confidence.toFixed(2)})`
                      : null}
                    {run.proxies.answerability.error ? ` · error: ${run.proxies.answerability.error}` : null}
                  </div>
                ) : null}

                <div className="mt-3 overflow-x-auto">
                  <table className="w-full min-w-[720px] text-left text-xs">
                    <thead className="text-slate-400">
                      <tr>
                        <th className="py-1 pr-3">Pipeline</th>
                        <th className="py-1 pr-3">Score</th>
                        <th className="py-1 pr-3">Expect</th>
                        <th className="py-1 pr-3">Heuristic</th>
                        <th className="py-1 pr-3">Abstain</th>
                        <th className="py-1 pr-3">Latency</th>
                        <th className="py-1 pr-3">Cost</th>
                        <th className="py-1 pr-3">Judge</th>
                        <th className="py-1 pr-3">Evidence</th>
                      </tr>
                    </thead>
                    <tbody className="text-slate-200">
                      {selectedPipelines.map((p) => {
                        const r = run.results[p]
                        const e = run.evaluations[p]
                        const score = scoreByPipeline[p]
                        const judgeScore = run.judge?.scores?.[p]
                        const evidenceScore = run.proxies?.evidence_support?.[p]?.support_score
                        return (
                          <tr key={p} className="border-t border-slate-900/60">
                            <td className="py-1 pr-3 font-medium text-slate-100">{p}</td>
                            <td className="py-1 pr-3">
                              {score?.score_0_10 != null
                                ? score.score_0_10.toFixed(2)
                                : score?.failed_gates?.length
                                  ? `failed (${score.failed_gates.join(', ')})`
                                  : 'n/a'}
                            </td>
                            <td className="py-1 pr-3">
                              {typeof e?.expect_score === 'number' ? e.expect_score.toFixed(2) : '—'}
                            </td>
                            <td className="py-1 pr-3">
                              {typeof e?.quality_score === 'number' ? e.quality_score.toFixed(2) : '—'}
                            </td>
                            <td className="py-1 pr-3">
                              {typeof e?.abstention_score === 'number' ? e.abstention_score.toFixed(2) : '—'}
                            </td>
                            <td className="py-1 pr-3">{r?.latency_ms != null ? `${r.latency_ms}ms` : '—'}</td>
                            <td className="py-1 pr-3">
                              {typeof r?.cost_estimate_usd === 'number' ? formatUsd(r.cost_estimate_usd) : '—'}
                            </td>
                            <td className="py-1 pr-3">
                              {typeof judgeScore === 'number' ? judgeScore.toFixed(2) : '—'}
                            </td>
                            <td className="py-1 pr-3">
                              {typeof evidenceScore === 'number' ? evidenceScore.toFixed(2) : '—'}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}

            {run.judge ? (
              <details className="mt-3 rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2">
                <summary className="cursor-pointer text-xs text-slate-300">Judge details</summary>
                {run.judge.error ? (
                  <div className="mt-2 text-xs text-rose-200">Judge error: {run.judge.error}</div>
                ) : (
                  <div className="mt-2 grid gap-1 text-xs text-slate-200">
                    <div>
                      <span className="font-medium text-slate-100">winner</span>{' '}
                      <span className="text-slate-400">{run.judge.winner ?? '—'}</span>
                    </div>
                    {run.judge.rationale ? (
                      <div className="text-slate-400">{run.judge.rationale}</div>
                    ) : null}
                    <div className="text-slate-500">
                      {run.judge.model ? `model: ${run.judge.model}` : null}
                      {run.judge.latency_ms != null ? ` · ${run.judge.latency_ms}ms` : null}
                      {run.judge.cost_estimate_usd != null ? ` · ${formatUsd(run.judge.cost_estimate_usd)}` : null}
                    </div>
                  </div>
                )}
              </details>
            ) : null}

            <div className="mt-4 grid gap-4 md:grid-cols-2">
              {selectedPipelines.map((p) => (
                <PipelineCard
                  key={p}
                  pipeline={p}
                  result={run.results[p]}
                  evaluation={run.evaluations[p]}
                  mode={run.mode}
                  judge={run.judge ?? null}
                  proxies={run.proxies ?? null}
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
                  <div className="shrink-0 text-xs text-slate-400">
                    <div>
                      {h.domain} <span className="text-slate-600">·</span> {h.mode}
                    </div>
                    {(() => {
                      const hHash = (h.scoring as any)?.hash as string | undefined
                      if (!hHash) return null
                      const mismatch = hHash !== scoring.hash
                      return (
                        <div className={mismatch ? 'text-rose-200' : 'text-slate-500'}>
                          cfg {hHash}
                          {mismatch ? ' (not comparable)' : ''}
                        </div>
                      )
                    })()}
                  </div>
                </button>
              ))
            )}
          </div>
        </div>
      </div>

      {showCustomScoringWarning ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div className="w-full max-w-lg rounded-xl border border-slate-800 bg-slate-950 p-5">
            <div className="text-sm font-semibold text-slate-100">Enable custom scoring?</div>
            <div className="mt-2 text-sm text-rose-200">
              Warning: custom “overall score” can create false precision and destroy comparability across runs. Only
              compare results when the scoring config hash matches.
            </div>
            <div className="mt-4 flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowCustomScoringWarning(false)}
                className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200 hover:border-slate-600"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  setCustomScoringAck(true)
                  try {
                    localStorage.setItem('scoring_custom_ack_v1', 'true')
                  } catch {
                    // ignore
                  }
                  setCustomScoring(true)
                  setShowCustomScoringWarning(false)
                }}
                className="rounded-lg bg-rose-600 px-3 py-2 text-sm font-medium text-white hover:bg-rose-500"
              >
                I understand — enable
              </button>
            </div>
          </div>
        </div>
      ) : null}
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
  const winnerByExpect = summary.winner_by_expect as string | null | undefined
  const expectTies = summary.winner_by_expect_ties as string[] | undefined
  const winnerByJudge = summary.winner_by_judge as string | null | undefined
  const judgeTies = summary.winner_by_judge_ties as string[] | undefined

  return (
    <div className="flex flex-wrap gap-2">
      <Chip label={`heuristic: ${winnerByQuality ?? (qualityTies?.length ? 'tie' : '—')}`} tone="indigo" />
      {winnerByExpect != null || expectTies?.length ? (
        <Chip label={`expect: ${winnerByExpect ?? 'tie'}`} tone="blue" />
      ) : null}
      {winnerByJudge != null || judgeTies?.length ? (
        <Chip label={`judge: ${winnerByJudge ?? 'tie'}`} tone="blue" />
      ) : null}
      <Chip label={`latency: ${winnerByLatency ?? '—'}`} tone="slate" />
      <Chip label={`cost: ${winnerByCost ?? '—'}`} tone="slate" />
    </div>
  )
}

function PipelineCard({
  pipeline,
  result,
  evaluation,
  mode,
  judge,
  proxies,
}: {
  pipeline: string
  result: PipelineResult | undefined
  evaluation: EvaluationResult | undefined
  mode: 'docs' | 'general'
  judge: RunResponse['judge'] | null
  proxies: RunResponse['proxies'] | null
}) {
  if (!result) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4">
        <div className="text-sm text-slate-300">No result for pipeline: {pipeline}</div>
      </div>
    )
  }

  const isRag = pipeline === 'rag'
  const title =
    pipeline === 'prompt'
      ? 'Prompt'
      : pipeline === 'finetune'
        ? 'Fine-tuned'
        : pipeline.toUpperCase()
  const latency = result.latency_ms
  const tokensIn = result.tokens_in ?? null
  const tokensOut = result.tokens_out ?? null
  const costUsd = result.cost_estimate_usd ?? null
  const heuristic = evaluation?.quality_score ?? null
  const ruleBreakdown = evaluation?.rule_breakdown || []
  const expectScore = evaluation?.expect_score ?? null
  const expectDetails = evaluation?.expect_details ?? null
  const abstentionScore = evaluation?.abstention_score ?? null
  const abstentionExpected = evaluation?.abstention_expected ?? null
  const abstained = evaluation?.abstained ?? null
  const judgeScore = typeof judge?.scores?.[pipeline] === 'number' ? judge.scores[pipeline] : null
  const judgeCriteria = judge?.criteria?.[pipeline] ?? null
  const evidence = proxies?.evidence_support?.[pipeline] ?? null

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
          <Chip label={`heuristic: ${heuristic == null ? '—' : heuristic.toFixed(2)}`} tone="indigo" />
          {expectScore == null ? null : (
            <Chip label={`expect: ${expectScore.toFixed(2)}`} tone="blue" />
          )}
          {abstentionScore == null ? null : (
            <Chip label={`abstain: ${abstentionScore.toFixed(2)}`} tone="amber" />
          )}
          {judgeScore == null ? null : <Chip label={`judge: ${judgeScore.toFixed(1)}/10`} tone="blue" />}
          {typeof evidence?.support_score === 'number' ? (
            <Chip label={`evidence: ${evidence.support_score.toFixed(1)}/2`} tone="amber" />
          ) : null}
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

          {ruleBreakdown.length ? (
            <details className="mt-3 rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2">
              <summary className="cursor-pointer text-xs text-slate-300">Heuristic breakdown</summary>
              <div className="mt-2 grid gap-1 text-xs text-slate-200">
                {ruleBreakdown.map((r) => {
                  const flags = [...(r.hallucination_flags || []), ...(r.grounding_flags || [])]
                  const penalty = Number.isFinite(r.penalty) ? r.penalty : 0
                  return (
                    <div key={r.rule} className="leading-relaxed">
                      <span className="font-medium text-slate-100">
                        -{penalty.toFixed(2)} {r.rule}
                      </span>
                      {r.note ? <span className="text-slate-400">: {r.note}</span> : null}
                      {flags.length ? <span className="text-slate-500"> ({flags.join(', ')})</span> : null}
                    </div>
                  )
                })}
              </div>
            </details>
          ) : null}

          {expectScore == null ? null : (
            <details className="mt-3 rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2">
              <summary className="cursor-pointer text-xs text-slate-300">
                Expectation score: {expectScore.toFixed(2)}
              </summary>
              <div className="mt-2 grid gap-1 text-xs text-slate-200">
                {expectDetails?.missing?.length ? (
                  <div>
                    <span className="font-medium text-slate-100">Missing</span>{' '}
                    <span className="text-slate-400">{expectDetails.missing.join(', ')}</span>
                  </div>
                ) : null}
                {expectDetails?.forbidden?.length ? (
                  <div>
                    <span className="font-medium text-slate-100">Forbidden</span>{' '}
                    <span className="text-slate-400">{expectDetails.forbidden.join(', ')}</span>
                  </div>
                ) : null}
                {!(expectDetails?.missing?.length || expectDetails?.forbidden?.length) ? (
                  <div className="text-slate-400">All expectations satisfied.</div>
                ) : null}
              </div>
            </details>
          )}

          {abstentionScore == null ? null : (
            <details className="mt-3 rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2">
              <summary className="cursor-pointer text-xs text-slate-300">
                Abstention correctness: {abstentionScore.toFixed(2)}
              </summary>
              <div className="mt-2 grid gap-1 text-xs text-slate-200">
                <div>
                  <span className="font-medium text-slate-100">expected abstain</span>{' '}
                  <span className="text-slate-400">
                    {abstentionExpected == null ? '—' : abstentionExpected ? 'true' : 'false'}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-slate-100">abstained</span>{' '}
                  <span className="text-slate-400">
                    {abstained == null ? '—' : abstained ? 'true' : 'false'}
                  </span>
                </div>
              </div>
            </details>
          )}

          {evidence ? (
            <details className="mt-3 rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2">
              <summary className="cursor-pointer text-xs text-slate-300">Evidence support (LLM)</summary>
              {evidence.error ? (
                <div className="mt-2 text-xs text-rose-200">Evidence error: {evidence.error}</div>
              ) : (
                <div className="mt-2 grid gap-1 text-xs text-slate-200">
                  {typeof evidence.support_score === 'number' ? (
                    <div>
                      <span className="font-medium text-slate-100">support score</span>{' '}
                      <span className="text-slate-400">{evidence.support_score.toFixed(1)}/2</span>
                    </div>
                  ) : null}
                  {evidence.rationale ? <div className="text-slate-400">{evidence.rationale}</div> : null}
                  {evidence.unsupported_claims?.length ? (
                    <div className="text-slate-400">
                      <span className="font-medium text-slate-100">unsupported</span>{' '}
                      {evidence.unsupported_claims.join(' · ')}
                    </div>
                  ) : null}
                  {!(evidence.unsupported_claims?.length || evidence.rationale) ? (
                    <div className="text-slate-400">No evidence notes.</div>
                  ) : null}
                </div>
              )}
            </details>
          ) : null}

          {judgeScore == null ? null : (
            <details className="mt-3 rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-2">
              <summary className="cursor-pointer text-xs text-slate-300">
                Judge score: {judgeScore.toFixed(1)}/10
              </summary>
              {judgeCriteria && Object.keys(judgeCriteria).length ? (
                <div className="mt-2 grid gap-1 text-xs text-slate-200">
                  {Object.entries(judgeCriteria).map(([k, v]) => (
                    <div key={k}>
                      <span className="font-medium text-slate-100">{k}</span>{' '}
                      <span className="text-slate-400">{Number(v).toFixed(1)}/2</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="mt-2 text-xs text-slate-400">No per-criterion breakdown.</div>
              )}
            </details>
          )}

          {isRag ? (
            <EvidencePanel
              title={mode === 'docs' ? 'Evidence' : 'Retrieved context'}
              chunks={result.retrieved_chunks || []}
            />
          ) : null}
        </>
      )}
    </div>
  )
}

function EvidencePanel({ title, chunks }: { title: string; chunks: RetrievedChunk[] }) {
  if (!chunks.length) {
    return <div className="mt-4 text-xs text-slate-400">No retrieved chunks.</div>
  }
  return (
    <div className="mt-4">
      <div className="text-xs font-medium text-slate-300">{title}</div>
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
