import type { BTREvent, EventType } from '@/lib/api'

export const AGE_RANGE_OPTIONS: Array<{ label: string; value: [number, number] }> = [
  { label: '10–20', value: [10, 20] },
  { label: '20–30', value: [20, 30] },
  { label: '30–40', value: [30, 40] },
  { label: '40–50', value: [40, 50] },
  { label: '50–60', value: [50, 60] },
  { label: '60+', value: [60, 120] },
]

export function mapQuestionEventType(rawType: string): EventType {
  const normalized = (rawType || '').toLowerCase()

  if (normalized.includes('career')) return 'career_change'
  if (normalized.includes('relation') || normalized.includes('marriage')) return 'relationship'
  if (normalized.includes('relocat') || normalized.includes('move')) return 'relocation'
  if (normalized.includes('health')) return 'health'
  if (normalized.includes('financ') || normalized.includes('money')) return 'finance'
  return 'other'
}

export function hasAnyTimedEvent(events: BTREvent[]): boolean {
  return events.some((event) => event.precision_level !== 'unknown')
}

export function validateOtherLabel(event: BTREvent): boolean {
  if (event.event_type !== 'other') return true
  return Boolean(event.other_label?.trim())
}

export function buildEventPayload(input: {
  eventType: EventType
  precisionLevel: 'exact' | 'range' | 'unknown'
  year?: string
  ageRangeKey?: string
  otherLabel?: string
}): BTREvent {
  const payload: BTREvent = {
    event_type: input.eventType,
    precision_level: input.precisionLevel,
  }

  if (input.precisionLevel === 'exact' && input.year) {
    payload.year = parseInt(input.year, 10)
  }

  if (input.precisionLevel === 'range' && input.ageRangeKey) {
    const selected = AGE_RANGE_OPTIONS.find((option) => option.label === input.ageRangeKey)
    if (selected) {
      payload.age_range = selected.value
    }
  }

  if (input.eventType === 'other' && input.otherLabel?.trim()) {
    payload.other_label = input.otherLabel.trim()
  }

  return payload
}
