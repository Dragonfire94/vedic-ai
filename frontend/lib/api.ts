function resolveApiBaseUrl(): string {
  // Server-side (SSR / Route handlers): prefer internal service DNS.
  if (typeof window === 'undefined') {
    const internalApiUrl = (
      process.env.INTERNAL_API_URL ||
      process.env.API_URL ||
      ''
    ).trim()
    if (internalApiUrl) {
      return internalApiUrl
    }

    const publicApiUrl = (process.env.NEXT_PUBLIC_API_URL || '').trim()
    if (publicApiUrl) {
      return publicApiUrl
    }

    if (process.env.NODE_ENV === 'development') {
      return 'http://127.0.0.1:8000'
    }

    throw new Error(
      'INTERNAL_API_URL (or API_URL) is required for SSR in production.'
    )
  }

  // Browser-side:
  // If frontend runs on localhost, prefer local backend by default to avoid
  // accidental dependency on stale remote NEXT_PUBLIC_API_URL values.
  const isLocalFrontendHost =
    typeof window !== 'undefined' &&
    (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  if (isLocalFrontendHost && process.env.NEXT_PUBLIC_ALLOW_REMOTE_API !== '1') {
    return 'http://127.0.0.1:8000'
  }

  // Otherwise require explicit public API URL in non-dev deployments.
  const publicApiUrl = (process.env.NEXT_PUBLIC_API_URL || '').trim()
  if (publicApiUrl) {
    return publicApiUrl
  }
  if (process.env.NODE_ENV === 'development') {
    return 'http://127.0.0.1:8000'
  }
  throw new Error(
    'NEXT_PUBLIC_API_URL is required in production. Set it to your public backend URL.'
  )
}

const API_BASE_URL = resolveApiBaseUrl()

async function safeFetch(path: string, init?: RequestInit): Promise<Response> {
  const url = `${API_BASE_URL}${path}`
  try {
    return await fetch(url, init)
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error)
    throw new Error(
      `Network request failed. API server unreachable.\nURL: ${url}\nReason: ${reason}`
    )
  }
}

export type EventType =
  | 'career_change'
  | 'relationship'
  | 'relocation'
  | 'health'
  | 'finance'
  | 'other'

// Type definitions
export type BTREvent = {
  event_type: EventType
  precision_level: 'exact' | 'range' | 'unknown'
  year?: number
  age_range?: [number, number]
  other_label?: string
  weight?: number
  dasha_lords?: string[]
  house_triggers?: number[]
}

export interface BTRAnalyzeRequest {
  year: number
  month: number
  day: number
  lat: number
  lon: number
  timezone?: number
  events: BTREvent[]
  tune_mode?: boolean
}

export interface ChartRequest {
  year: number
  month: number
  day: number
  hour: number
  lat: number
  lon: number
  house_system?: string
  include_nodes?: boolean
  include_d9?: boolean
  include_vargas?: string[]
  gender?: string
  timezone?: number
}

function applyVargaParams(params: URLSearchParams, data: ChartRequest): void {
  if (Array.isArray(data.include_vargas) && data.include_vargas.length > 0) {
    const normalized = data.include_vargas
      .map((v) => v.trim().toLowerCase())
      .filter((v) => v.length > 0)
    if (normalized.length > 0) {
      params.set('include_vargas', normalized.join(','))
    }
  }
}

// API Functions
async function buildApiError(response: Response, fallbackMessage: string): Promise<Error> {
  try {
    const contentType = response.headers.get('content-type') || ''
    if (contentType.includes('application/json')) {
      const data = await response.json()
      if (typeof data?.detail === 'string' && data.detail.trim()) {
        return new Error(data.detail)
      }
      if (Array.isArray(data?.detail) && data.detail.length > 0) {
        const first = data.detail[0]
        if (typeof first?.msg === 'string' && first.msg.trim()) {
          return new Error(first.msg)
        }
      }
      if (typeof data?.message === 'string' && data.message.trim()) {
        return new Error(data.message)
      }
    } else {
      const text = await response.text()
      if (text.trim()) {
        return new Error(text)
      }
    }
  } catch {
    // Fall through to fallback below.
  }
  return new Error(fallbackMessage)
}

export async function getBTRQuestions(age: number, language: string = 'ko') {
  const response = await safeFetch(`/btr/questions?age=${age}&language=${language}`)
  if (!response.ok) {
    throw await buildApiError(response, 'Failed to fetch BTR questions')
  }
  return response.json()
}

export async function analyzeBTR(data: BTRAnalyzeRequest) {
  const response = await safeFetch('/btr/analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })
  if (!response.ok) {
    throw await buildApiError(response, 'Failed to analyze BTR')
  }
  return response.json()
}

export async function getChart(data: ChartRequest) {
  const params = new URLSearchParams({
    year: data.year.toString(),
    month: data.month.toString(),
    day: data.day.toString(),
    hour: data.hour.toString(),
    lat: data.lat.toString(),
    lon: data.lon.toString(),
    house_system: data.house_system || 'W',
    include_nodes: data.include_nodes !== false ? '1' : '0',
    include_d9: data.include_d9 ? '1' : '0',
    gender: data.gender || 'male',
  })
  applyVargaParams(params, data)
  if (Number.isFinite(data.timezone)) {
    params.set('timezone', String(data.timezone))
  }

  const response = await safeFetch(`/chart?${params}`)
  if (!response.ok) {
    throw await buildApiError(response, 'Failed to fetch chart')
  }
  return response.json()
}

export async function getAIReading(data: ChartRequest & { language?: string }) {
  const params = new URLSearchParams({
    year: data.year.toString(),
    month: data.month.toString(),
    day: data.day.toString(),
    hour: data.hour.toString(),
    lat: data.lat.toString(),
    lon: data.lon.toString(),
    house_system: data.house_system || 'W',
    include_nodes: data.include_nodes !== false ? '1' : '0',
    include_d9: data.include_d9 ? '1' : '0',
    language: data.language || 'ko',
    gender: data.gender || 'male',
  })
  applyVargaParams(params, data)
  if (Number.isFinite(data.timezone)) {
    params.set('timezone', String(data.timezone))
  }

  const response = await safeFetch(`/ai_reading?${params}`)
  if (!response.ok) {
    throw await buildApiError(response, 'Failed to fetch AI reading')
  }
  return response.json()
}

export async function getPDF(data: ChartRequest & { language?: string }) {
  const params = new URLSearchParams({
    year: data.year.toString(),
    month: data.month.toString(),
    day: data.day.toString(),
    hour: data.hour.toString(),
    lat: data.lat.toString(),
    lon: data.lon.toString(),
    house_system: data.house_system || 'W',
    include_nodes: data.include_nodes !== false ? '1' : '0',
    include_d9: data.include_d9 ? '1' : '0',
    language: data.language || 'ko',
    gender: data.gender || 'male',
  })
  applyVargaParams(params, data)
  if (Number.isFinite(data.timezone)) {
    params.set('timezone', String(data.timezone))
  }

  const response = await safeFetch(`/pdf?${params}`)
  if (!response.ok) {
    throw await buildApiError(response, 'Failed to fetch PDF')
  }
  return response.blob()
}
