import {
  AGE_RANGE_OPTIONS,
  buildEventPayload,
  hasAnyTimedEvent,
  validateOtherLabel,
} from '@/lib/btr-event-flow'

function test_precision_exact() {
  const payload = buildEventPayload({
    eventType: 'career_change',
    precisionLevel: 'exact',
    year: '2019',
  })

  if (payload.event_type !== 'career_change' || payload.precision_level !== 'exact' || payload.year !== 2019) {
    throw new Error('Exact precision payload mismatch with backend BTREvent schema')
  }
}

function test_precision_range() {
  const option = AGE_RANGE_OPTIONS[1]
  const payload = buildEventPayload({
    eventType: 'relationship',
    precisionLevel: 'range',
    ageRangeKey: option.label,
  })

  if (!payload.age_range || payload.age_range[0] !== option.value[0] || payload.age_range[1] !== option.value[1]) {
    throw new Error('Range precision payload mismatch with backend BTREvent schema')
  }
}

function test_unknown_followup() {
  const onlyUnknown = [
    buildEventPayload({ eventType: 'health', precisionLevel: 'unknown' }),
    buildEventPayload({ eventType: 'finance', precisionLevel: 'unknown' }),
  ]

  if (hasAnyTimedEvent(onlyUnknown)) {
    throw new Error('Unknown follow-up detection should reject all-unknown event sets')
  }
}

function test_other_requires_label() {
  const missing = buildEventPayload({ eventType: 'other', precisionLevel: 'unknown' })
  const valid = buildEventPayload({
    eventType: 'other',
    precisionLevel: 'unknown',
    otherLabel: 'Moved abroad',
  })

  if (validateOtherLabel(missing) || !validateOtherLabel(valid)) {
    throw new Error('Other label validation does not match backend BTREvent requirements')
  }
}

export {
  test_precision_exact,
  test_precision_range,
  test_unknown_followup,
  test_other_requires_label,
}
