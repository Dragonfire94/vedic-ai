function resolveApiBaseUrl(): string {
  // Server-side (SSR / Route handlers): prefer internal service DNS.
  if (typeof window === 'undefined') {
    return (
      process.env.API_URL ||
      process.env.NEXT_PUBLIC_API_URL ||
      'http://127.0.0.1:8000'
    )
  }

  // Browser-side: only public URL should be used.
  return process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'
}

const API_BASE_URL = resolveApiBaseUrl()

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
  gender?: string
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
  const response = await fetch(`${API_BASE_URL}/btr/questions?age=${age}&language=${language}`)
  if (!response.ok) {
    throw await buildApiError(response, 'Failed to fetch BTR questions')
  }
  return response.json()
}

export async function analyzeBTR(data: BTRAnalyzeRequest) {
  const response = await fetch(`${API_BASE_URL}/btr/analyze`, {
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

  const response = await fetch(`${API_BASE_URL}/chart?${params}`)
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

  const response = await fetch(`${API_BASE_URL}/ai_reading?${params}`)
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

  const response = await fetch(`${API_BASE_URL}/pdf?${params}`)
  if (!response.ok) {
    throw await buildApiError(response, 'Failed to fetch PDF')
  }
  return response.blob()
}
