import type { DomainsResponse, HealthResponse, RunRequest, RunResponse } from '../types'

const DEFAULT_BASE_URL = 'http://127.0.0.1:8000'

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() || DEFAULT_BASE_URL

export class ApiError extends Error {
  status: number
  body: unknown

  constructor(message: string, status: number, body: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.headers || {}),
    },
  })

  let body: unknown = null
  const text = await res.text()
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = text
    }
  }

  if (!res.ok) {
    throw new ApiError(`Request failed: ${res.status} ${res.statusText}`, res.status, body)
  }

  return body as T
}

export function getHealth(): Promise<HealthResponse> {
  return fetchJson('/health')
}

export function getDomains(): Promise<DomainsResponse> {
  return fetchJson('/domains')
}

export function runOnce(payload: RunRequest): Promise<RunResponse> {
  return fetchJson('/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

