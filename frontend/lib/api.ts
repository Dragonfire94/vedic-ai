const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Type definitions
export interface BTREvent {
  type: string
  year: number
  month?: number
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

export async function getBTRQuestions(age: number, language: string = 'ko') {
  const response = await fetch(`${API_BASE_URL}/btr/questions?age=${age}&language=${language}`)
  if (!response.ok) {
    throw new Error('Failed to fetch BTR questions')
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
    throw new Error('Failed to analyze BTR')
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
    house_system: data.house_system || 'P',
    include_nodes: data.include_nodes !== false ? '1' : '0',
    include_d9: data.include_d9 ? '1' : '0',
    gender: data.gender || 'male',
  })

  const response = await fetch(`${API_BASE_URL}/chart?${params}`)
  if (!response.ok) {
    throw new Error('Failed to fetch chart')
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
    house_system: data.house_system || 'P',
    include_nodes: data.include_nodes !== false ? '1' : '0',
    include_d9: data.include_d9 ? '1' : '0',
    language: data.language || 'ko',
    gender: data.gender || 'male',
  })

  const response = await fetch(`${API_BASE_URL}/ai_reading?${params}`)
  if (!response.ok) {
    throw new Error('Failed to fetch AI reading')
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
    house_system: data.house_system || 'P',
    include_nodes: data.include_nodes !== false ? '1' : '0',
    include_d9: data.include_d9 ? '1' : '0',
    language: data.language || 'ko',
    gender: data.gender || 'male',
  })

  const response = await fetch(`${API_BASE_URL}/pdf?${params}`)
  if (!response.ok) {
    throw new Error('Failed to fetch PDF')
  }
  return response.json()
}
