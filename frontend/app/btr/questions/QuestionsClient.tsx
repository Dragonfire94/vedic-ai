'use client'

import { useEffect, useMemo, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Progress } from '@/components/ui/progress'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ChevronLeft, ChevronRight, Sparkles } from 'lucide-react'
import { analyzeBTR, getBTRQuestions, type BTRQuestion, type BTREvent } from '@/lib/api'
import { useBTRStore } from '@/store/btrStore'
import { toNum } from '@/lib/utils'
import {
  AGE_RANGE_OPTIONS,
  buildEventPayload,
  hasAnyTimedEvent,
  mapQuestionEventType,
  validateOtherLabel,
} from '@/lib/btr-event-flow'

type PrecisionLevel = 'exact' | 'range' | 'unknown'

type EventAnswer = {
  hasEvent?: boolean
  precision_level?: PrecisionLevel
  year?: string
  age_range_label?: string
  other_label?: string
}

type PersonalityAnswer = { choice: string }

const HELPER_TEXT: Record<string, string> = {
  career_change: '예: 취업, 승진, 이직',
  relationship: '예: 결혼, 장기 연애 시작/종료',
  relocation: '예: 이사, 해외 이동',
  health: '예: 수술, 회복이 필요했던 시기',
}

export default function BTRQuestionsPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const setResult = useBTRStore((s) => s.setResult)
  const birthData = {
    year: toNum(searchParams.get('year'), 1994),
    month: toNum(searchParams.get('month'), 12),
    day: toNum(searchParams.get('day'), 18),
    hour: toNum(searchParams.get('hour'), 12),
    lat: toNum(searchParams.get('lat'), 37.5665),
    lon: toNum(searchParams.get('lon'), 126.978),
    timezone: toNum(searchParams.get('timezone'), 9),
  }

  const [questions, setQuestions] = useState<BTRQuestion[]>([])
  const [currentStep, setCurrentStep] = useState(0)
  const [answers, setAnswers] = useState<Record<string, EventAnswer | PersonalityAnswer>>({})
  const [eventsByQuestion, setEventsByQuestion] = useState<Record<string, BTREvent>>({})
  const [loading, setLoading] = useState(true)
  const [analyzing, setAnalyzing] = useState(false)
  const [inlineError, setInlineError] = useState('')

  const [followUpAttempts, setFollowUpAttempts] = useState(0)
  const [showFollowUpPrompt, setShowFollowUpPrompt] = useState(false)
  const [allowLowPrecisionContinue, setAllowLowPrecisionContinue] = useState(false)
  const [followUpAnswer, setFollowUpAnswer] = useState<EventAnswer>({ hasEvent: true, precision_level: 'unknown' })

  useEffect(() => {
    const loadQuestions = async () => {
      try {
        const age = new Date().getFullYear() - birthData.year
        const data = await getBTRQuestions(age, 'ko')
        setQuestions(data.questions || [])
      } catch (error) {
        console.error('Failed to load questions:', error)
        setQuestions([])
      } finally {
        setLoading(false)
      }
    }
    loadQuestions()
  }, [birthData.year])

  const currentQuestion = questions[currentStep]
  const progress = questions.length > 0 ? ((currentStep + 1) / questions.length) * 100 : 0
  const isPersonalityQuestion = currentQuestion?.event_type === 'personality'
  const isEventQuestion = !isPersonalityQuestion && currentQuestion?.type === 'yesno_date'

  const eventList = useMemo(() => Object.values(eventsByQuestion), [eventsByQuestion])
  const timedEventExists = hasAnyTimedEvent(eventList)

  const handleAnswer = (questionId: string, value: EventAnswer | PersonalityAnswer) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }))
  }

  const upsertQuestionEvent = (questionId: string, questionEventType: string, answer: EventAnswer) => {
    if (!answer.hasEvent) {
      setEventsByQuestion((prev) => {
        const next = { ...prev }
        delete next[questionId]
        return next
      })
      return
    }
    if (!answer.precision_level) return

    const payload = buildEventPayload({
      eventType: mapQuestionEventType(questionEventType),
      precisionLevel: answer.precision_level,
      year: answer.year,
      ageRangeKey: answer.age_range_label,
      otherLabel: answer.other_label,
    })
    if (!validateOtherLabel(payload)) return
    setEventsByQuestion((prev) => ({ ...prev, [questionId]: payload }))
  }

  const buildMergedEvents = (existing: Record<string, BTREvent>, questionId: string, currentAnswer: EventAnswer) => {
    const updated = { ...existing }
    if (!currentAnswer.hasEvent) {
      delete updated[questionId]
      return Object.values(updated)
    }
    if (!currentAnswer.precision_level) return Object.values(updated)

    const questionEventType = questionId.startsWith('followup-')
      ? 'other'
      : mapQuestionEventType(questions.find((q) => q.id === questionId)?.event_type ?? '')

    const payload = buildEventPayload({
      eventType: questionEventType,
      precisionLevel: currentAnswer.precision_level,
      year: currentAnswer.year,
      ageRangeKey: currentAnswer.age_range_label,
      otherLabel: currentAnswer.other_label,
    })
    if (!validateOtherLabel(payload)) return Object.values(updated)
    updated[questionId] = payload
    return Object.values(updated)
  }

  const canProceedEventQuestion = () => {
    const answer = answers[currentQuestion?.id] as EventAnswer | undefined
    if (!answer || answer.hasEvent === undefined) return false
    if (!answer.hasEvent) return true
    if (!answer.precision_level) return false
    if (answer.precision_level === 'exact') return Boolean(answer.year)
    if (answer.precision_level === 'range') return Boolean(answer.age_range_label)
    if (mapQuestionEventType(currentQuestion?.event_type ?? '') === 'other') return Boolean(answer.other_label?.trim())
    return true
  }

  const handleAnalyze = async (overrideEvents?: BTREvent[]) => {
    const payloadEvents = overrideEvents ?? Object.values(eventsByQuestion)
    if (payloadEvents.length === 0) {
      setInlineError('최소 1개 이상의 이벤트를 입력해 주세요')
      return
    }

    if (!hasAnyTimedEvent(payloadEvents) && !allowLowPrecisionContinue) {
      if (followUpAttempts < 3) {
        setShowFollowUpPrompt(true)
        setInlineError('연도/나이 정보가 있는 이벤트가 없으면 정확도가 떨어질 수 있어요')
      } else {
        setInlineError('시간 정보가 없어 정확도가 낮아질 수 있어요')
      }
      return
    }

    setAnalyzing(true)
    setInlineError('')
    try {
      const result = await analyzeBTR({
        year: birthData.year,
        month: birthData.month,
        day: birthData.day,
        hour: birthData.hour,
        lat: birthData.lat,
        lon: birthData.lon,
        timezone: birthData.timezone,
        events: payloadEvents,
        personality_answers: Object.fromEntries(
          Object.entries(answers)
            .map(([key, value]) => [key, (value as PersonalityAnswer).choice])
            .filter(([, value]) => Boolean(value))
        ),
      })
      const params = new URLSearchParams({
        year: String(birthData.year),
        month: String(birthData.month),
        day: String(birthData.day),
        lat: String(birthData.lat),
        lon: String(birthData.lon),
        timezone: String(birthData.timezone),
        gender: searchParams.get('gender') || 'female',
      })
      setResult(result)
      router.push(`/btr/results?${params}`)
    } catch (error) {
      console.error('BTR analysis failed:', error)
      const msg = error instanceof Error ? error.message : 'Unknown error'
      setInlineError(`분석 중 오류가 발생했습니다: ${msg}`)
    } finally {
      setAnalyzing(false)
    }
  }

  const handleNext = () => {
    if (isEventQuestion && currentQuestion) {
      const answer = answers[currentQuestion.id] as EventAnswer | undefined
      if (answer) {
        if (currentStep === questions.length - 1) {
          const merged = buildMergedEvents(eventsByQuestion, currentQuestion.id, answer)
          handleAnalyze(merged)
          return
        }
        upsertQuestionEvent(currentQuestion.id, currentQuestion.event_type ?? '', answer)
      }
    }
    if (currentStep < questions.length - 1) {
      setCurrentStep((prev) => prev + 1)
      return
    }
    handleAnalyze()
  }

  const submitFollowUpEvent = async () => {
    const payload = buildEventPayload({
      eventType: 'other',
      precisionLevel: followUpAnswer.precision_level || 'unknown',
      year: followUpAnswer.year,
      ageRangeKey: followUpAnswer.age_range_label,
      otherLabel: followUpAnswer.other_label,
    })
    if (!validateOtherLabel(payload)) {
      setInlineError('추가 이벤트 설명을 입력해 주세요')
      return
    }
    if (payload.precision_level === 'exact' && !payload.year) {
      setInlineError('연도를 입력해 주세요')
      return
    }
    if (payload.precision_level === 'range' && !payload.age_range) {
      setInlineError('나이 구간을 선택해 주세요')
      return
    }
    const key = `followup-${followUpAttempts + 1}`
    const merged = buildMergedEvents(eventsByQuestion, key, { ...followUpAnswer, hasEvent: true })
    setEventsByQuestion((prev) => ({ ...prev, [key]: payload }))
    setFollowUpAttempts((prev) => prev + 1)
    setInlineError('')
    setFollowUpAnswer({ hasEvent: true, precision_level: 'unknown' })
    await handleAnalyze(merged)
  }

  if (loading || analyzing) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#f7f6f3]">
        <div className="text-center">
          <Sparkles className="w-10 h-10 mx-auto mb-3 text-[#8d3d56] animate-pulse" />
          <p className="text-[#534e57]">{analyzing ? '분석 중입니다...' : '질문을 불러오는 중입니다...'}</p>
        </div>
      </div>
    )
  }

  if (!currentQuestion) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#f7f6f3]">
        <Card className="max-w-md w-full">
          <CardHeader>
            <CardTitle>질문이 없습니다</CardTitle>
            <CardDescription>잠시 후 다시 시도해 주세요</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => router.push('/')} className="w-full">처음으로</Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  const currentEventType = mapQuestionEventType(currentQuestion?.event_type || '')
  const helperText = HELPER_TEXT[currentEventType]
  const currentEventAnswer = (answers[currentQuestion.id] as EventAnswer | undefined) || {}

  return (
    <div className="min-h-screen bg-[linear-gradient(180deg,#f7f6f3_0%,#fff_36%)]">
      <div className="container mx-auto px-4 py-12 max-w-3xl">
        <div className="text-center mb-8">
          <p className="text-sm tracking-[0.18em] uppercase text-[#8a808a] mb-3">Birth Time Check</p>
          <h1 className="text-3xl font-semibold text-[#2b2731]">질문으로 시간을 찾기</h1>
          <p className="text-[#5f5a64] mt-3">기억나는 경험만 답해도 됩니다. 모르면 중립적인 선택지를 선택해 주세요.</p>
        </div>

        <Card className="mb-6 border-[#e5d9de]">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between text-sm text-[#756d77] mb-2">
              <span>{currentStep + 1} / {questions.length}</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <Progress value={progress} className="h-2" />
          </CardContent>
        </Card>

        <Card className="border-[#e5d9de]">
          <CardHeader>
            <CardTitle className="text-xl text-[#2e2831]">{currentQuestion.text_ko || currentQuestion.text}</CardTitle>
            <CardDescription>
              {isPersonalityQuestion ? '가장 가까운 답을 골라 주세요' : '있음/없음과 시기 정보를 알려 주세요'}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {isPersonalityQuestion && currentQuestion.options && (
              <RadioGroup
                value={(answers[currentQuestion.id] as PersonalityAnswer | undefined)?.choice || ''}
                onValueChange={(v) => handleAnswer(currentQuestion.id, { choice: v })}
              >
                {Object.entries(currentQuestion.options as Record<string, string>).map(([key, value]) => (
                  <div key={key} className="flex items-center space-x-2 rounded-lg border p-3 hover:bg-[#faf7f8]">
                    <RadioGroupItem value={key} id={`${currentQuestion.id}-${key}`} />
                    <Label htmlFor={`${currentQuestion.id}-${key}`} className="flex-1 cursor-pointer">{value}</Label>
                  </div>
                ))}
              </RadioGroup>
            )}

            {isEventQuestion && (
              <>
                <div>
                  <Label className="mb-3 block">해당 일이 있었나요?</Label>
                  <RadioGroup
                    value={currentEventAnswer.hasEvent ? 'yes' : 'no'}
                    onValueChange={(v) => handleAnswer(currentQuestion.id, { ...currentEventAnswer, hasEvent: v === 'yes' })}
                  >
                    <div className="flex items-center space-x-2 rounded-lg border p-3 hover:bg-[#faf7f8]">
                      <RadioGroupItem value="yes" id="event-yes" />
                      <Label htmlFor="event-yes" className="flex-1 cursor-pointer">예</Label>
                    </div>
                    <div className="flex items-center space-x-2 rounded-lg border p-3 hover:bg-[#faf7f8]">
                      <RadioGroupItem value="no" id="event-no" />
                      <Label htmlFor="event-no" className="flex-1 cursor-pointer">아니요</Label>
                    </div>
                  </RadioGroup>
                </div>

                {currentEventAnswer.hasEvent && (
                  <div className="space-y-4 rounded-lg border border-[#eadfe4] bg-[#faf5f7] p-4">
                    <Label>언제가 있었나요?</Label>
                    {helperText && <p className="text-xs text-[#736a74]">{helperText}</p>}

                    <RadioGroup
                      value={currentEventAnswer.precision_level || ''}
                      onValueChange={(v) =>
                        handleAnswer(currentQuestion.id, {
                          ...currentEventAnswer,
                          precision_level: v as PrecisionLevel,
                          year: undefined,
                          age_range_label: undefined,
                        })
                      }
                    >
                      <div className="flex items-center space-x-2 rounded-lg border bg-white p-3">
                        <RadioGroupItem value="exact" id="precision-exact" />
                        <Label htmlFor="precision-exact" className="flex-1">정확한 연도 기억</Label>
                      </div>
                      <div className="flex items-center space-x-2 rounded-lg border bg-white p-3">
                        <RadioGroupItem value="range" id="precision-range" />
                        <Label htmlFor="precision-range" className="flex-1">나이대 구간 기억</Label>
                      </div>
                      <div className="flex items-center space-x-2 rounded-lg border bg-white p-3">
                        <RadioGroupItem value="unknown" id="precision-unknown" />
                        <Label htmlFor="precision-unknown" className="flex-1">기억 없음</Label>
                      </div>
                    </RadioGroup>

                    {currentEventAnswer.precision_level === 'exact' && (
                      <div>
                        <Label htmlFor="year">연도</Label>
                        <Input
                          id="year"
                          type="number"
                          min={birthData.year}
                          max={new Date().getFullYear()}
                          value={currentEventAnswer.year || ''}
                          onChange={(e) => handleAnswer(currentQuestion.id, { ...currentEventAnswer, year: e.target.value })}
                          placeholder="예: 2018"
                        />
                      </div>
                    )}

                    {currentEventAnswer.precision_level === 'range' && (
                      <div>
                        <Label>나이 구간</Label>
                        <Select
                          value={currentEventAnswer.age_range_label || ''}
                          onValueChange={(v) => handleAnswer(currentQuestion.id, { ...currentEventAnswer, age_range_label: v })}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="구간 선택" />
                          </SelectTrigger>
                          <SelectContent>
                            {AGE_RANGE_OPTIONS.map((option) => (
                              <SelectItem key={option.label} value={option.label}>{option.label}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    )}

                    {currentEventType === 'other' && (
                      <div>
                        <Label>이벤트 내용</Label>
                        <Input
                          value={currentEventAnswer.other_label || ''}
                          onChange={(e) => handleAnswer(currentQuestion.id, { ...currentEventAnswer, other_label: e.target.value })}
                          placeholder="예: 유학 시작, 수술"
                        />
                      </div>
                    )}
                  </div>
                )}
              </>
            )}

            {inlineError && <p className="text-sm text-red-600">{inlineError}</p>}

            {showFollowUpPrompt && !timedEventExists && (
              <div className="rounded-lg border border-amber-300 bg-amber-50 p-4 space-y-3">
                <p className="text-sm font-medium">정확도를 위해 추가 이벤트 1개만 더 입력해 주세요</p>
                <p className="text-xs text-[#6b6470]">추가 시도: {followUpAttempts}/3</p>

                <RadioGroup
                  value={followUpAnswer.precision_level || 'unknown'}
                  onValueChange={(v) => setFollowUpAnswer((prev) => ({ ...prev, precision_level: v as PrecisionLevel, year: undefined, age_range_label: undefined }))}
                >
                  <div className="flex items-center space-x-2"><RadioGroupItem value="exact" id="f-exact" /><Label htmlFor="f-exact">정확한 연도</Label></div>
                  <div className="flex items-center space-x-2"><RadioGroupItem value="range" id="f-range" /><Label htmlFor="f-range">나이 구간</Label></div>
                  <div className="flex items-center space-x-2"><RadioGroupItem value="unknown" id="f-unknown" /><Label htmlFor="f-unknown">기억 없음</Label></div>
                </RadioGroup>

                {followUpAnswer.precision_level === 'exact' && (
                  <Input type="number" placeholder="연도" value={followUpAnswer.year || ''} onChange={(e) => setFollowUpAnswer((prev) => ({ ...prev, year: e.target.value }))} />
                )}
                {followUpAnswer.precision_level === 'range' && (
                  <Select value={followUpAnswer.age_range_label || ''} onValueChange={(v) => setFollowUpAnswer((prev) => ({ ...prev, age_range_label: v }))}>
                    <SelectTrigger><SelectValue placeholder="나이 구간" /></SelectTrigger>
                    <SelectContent>
                      {AGE_RANGE_OPTIONS.map((option) => (
                        <SelectItem key={option.label} value={option.label}>{option.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}

                <Input
                  placeholder="이벤트 내용 (예: 이직, 결혼, 이사)"
                  value={followUpAnswer.other_label || ''}
                  onChange={(e) => setFollowUpAnswer((prev) => ({ ...prev, other_label: e.target.value }))}
                />

                <div className="flex gap-2">
                  <Button variant="outline" onClick={submitFollowUpEvent} disabled={followUpAttempts >= 3}>추가 이벤트 제출</Button>
                  {followUpAttempts >= 3 && (
                    <Button
                      onClick={() => {
                        setAllowLowPrecisionContinue(true)
                        setInlineError('정확도가 낮을 수 있음을 확인하고 진행합니다')
                      }}
                    >
                      경고 확인 후 진행
                    </Button>
                  )}
                </div>
              </div>
            )}

            <div className="flex gap-3 pt-2">
              <Button variant="outline" className="flex-1" onClick={() => setCurrentStep((s) => Math.max(0, s - 1))} disabled={currentStep === 0}>
                <ChevronLeft className="w-4 h-4 mr-1" />
                이전
              </Button>
              <Button
                className="flex-1 bg-[#8d3d56] hover:bg-[#7a344a]"
                onClick={handleNext}
                disabled={
                  (isPersonalityQuestion && !(answers[currentQuestion.id] as PersonalityAnswer | undefined)?.choice) ||
                  (isEventQuestion && !canProceedEventQuestion())
                }
              >
                {currentStep === questions.length - 1 ? '분석 시작' : '다음'}
                <ChevronRight className="w-4 h-4 ml-1" />
              </Button>
            </div>
          </CardContent>
        </Card>

        <div className="mt-6 text-center text-sm text-[#726a75]">
          성향 답변 {Object.values(answers).filter((answer): answer is PersonalityAnswer => {
            return typeof (answer as PersonalityAnswer).choice === 'string'
          }).length}개 / 이벤트 {eventList.length}개 / 시기 정보 포함 {eventList.filter((e) => e.precision_level !== 'unknown').length}개
        </div>
      </div>
    </div>
  )
}

