export function formatIsoTimestamp(value: string): string {
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return value
  return d.toLocaleString()
}

export function formatUsd(value: number | null | undefined): string {
  if (value == null) return '—'
  return `$${value.toFixed(6)}`
}

export function formatInt(value: number | null | undefined): string {
  if (value == null) return '—'
  return String(value)
}

