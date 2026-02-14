'use client'

import { useEffect, useMemo, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Progress } from '@/components/ui/progress'
import { Sparkles, ChevronRight, ChevronLeft } from 'lucide-react'
import { getBTRQuestions, analyzeBTR, type BTREvent } from '@/lib/api'
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

const HELPER_TEXT: Record<string, string> = {
  career_change: 'For example: first job, major promotion, career switch.',
  relationship: 'Marriage, serious breakup, engagement.',
}

export default function BTRQuestionsPage() {
  const router = useRouter()
  const searchParams = useSearchParams()

  const birthData = {
    year: parseInt(searchParams.get('year') || '1994', 10),
    month: parseInt(searchParams.get('month') || '12', 10),
    day: parseInt(searchParams.get('day') || '18', 10),
    lat: parseFloat(searchParams.get('lat') || '37.5665'),
    lon: parseFloat(searchParams.get('lon') || '126.978'),
  }

  const [questions, setQuestions] = useState<any[]>([])
  const [currentStep, setCurrentStep] = useState(0)
  const [answers, setAnswers] = useState<Record<string, EventAnswer | { choice: string }>>({})
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

  const yearOptions = Array.from(
    { length: new Date().getFullYear() - birthData.year + 1 },
    (_, i) => (new Date().getFullYear() - i).toString(),
  )

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

    const eventType = mapQuestionEventType(questionEventType)
    const payload = buildEventPayload({
      eventType,
      precisionLevel: answer.precision_level,
      year: answer.year,
      ageRangeKey: answer.age_range_label,
      otherLabel: answer.other_label,
    })

    if (!validateOtherLabel(payload)) return

    setEventsByQuestion((prev) => ({ ...prev, [questionId]: payload }))
  }

  const handleAnswer = (questionId: string, value: EventAnswer | { choice: string }) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }))
  }

  const buildMergedEvents = (existing: Record<string, BTREvent>, questionId: string, currentAnswer: EventAnswer) => {
    const updated = { ...existing }

    if (!currentAnswer.hasEvent) {
      delete updated[questionId]
      return Object.values(updated)
    }

    if (!currentAnswer.precision_level) {
      return Object.values(updated)
    }

    const questionEventType = questionId.startsWith('followup-')
      ? 'other'
      : mapQuestionEventType(questions.find((question) => question.id === questionId)?.event_type)

    const payload = buildEventPayload({
      eventType: questionEventType,
      precisionLevel: currentAnswer.precision_level,
      year: currentAnswer.year,
      ageRangeKey: currentAnswer.age_range_label,
      otherLabel: currentAnswer.other_label,
    })

    if (!validateOtherLabel(payload)) {
      return Object.values(updated)
    }

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

    const eventType = mapQuestionEventType(currentQuestion?.event_type)
    if (eventType === 'other') return Boolean(answer.other_label?.trim())
    return true
  }

  const handleNext = () => {
    if (isEventQuestion && currentQuestion) {
      const answer = answers[currentQuestion.id] as EventAnswer | undefined
      if (answer) {
        if (currentStep === questions.length - 1) {
          const mergedEvents = buildMergedEvents(eventsByQuestion, currentQuestion.id, answer)
          handleAnalyze(mergedEvents)
          return
        }

        upsertQuestionEvent(currentQuestion.id, currentQuestion.event_type, answer)
      }
    }

    if (currentStep < questions.length - 1) {
      setCurrentStep((prev) => prev + 1)
      return
    }

    handleAnalyze()
  }

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1)
    }
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
      setInlineError('기타 이벤트 설명을 입력해주세요.')
      return
    }

    if (payload.precision_level === 'exact' && !payload.year) {
      setInlineError('정확한 연도를 입력해주세요.')
      return
    }

    if (payload.precision_level === 'range' && !payload.age_range) {
      setInlineError('나이 구간을 선택해주세요.')
      return
    }

    const key = `followup-${followUpAttempts + 1}`
    const mergedEvents = buildMergedEvents(eventsByQuestion, key, {
      ...followUpAnswer,
      hasEvent: true,
    })

    setEventsByQuestion((prev) => ({ ...prev, [key]: payload }))
    setFollowUpAttempts((prev) => prev + 1)
    setInlineError('')
    setFollowUpAnswer({ hasEvent: true, precision_level: 'unknown' })

    await handleAnalyze(mergedEvents)
  }

  const handleAnalyze = async (overrideEvents?: BTREvent[]) => {
    const payloadEvents = overrideEvents ?? Object.values(eventsByQuestion)

    if (payloadEvents.length === 0) {
      setInlineError('최소 1개 이상의 이벤트를 입력해주세요.')
      return
    }

    const hasTimed = hasAnyTimedEvent(payloadEvents)
    if (!hasTimed && !allowLowPrecisionContinue) {
      if (followUpAttempts < 3) {
        setShowFollowUpPrompt(true)
        setInlineError('We don’t yet have enough timing data. Can you recall another event?')
      } else {
        setInlineError('Low timing precision may reduce rectification confidence.')
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
        lat: birthData.lat,
        lon: birthData.lon,
        events: payloadEvents,
      })

      const params = new URLSearchParams(searchParams.toString())
      params.set('result', JSON.stringify(result))
      router.push(`/btr/results?${params}`)
    } catch (error) {
      console.error('BTR analysis failed:', error)
      setInlineError('분석 중 오류가 발생했습니다. 다시 시도해주세요.')
    } finally {
      setAnalyzing(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Sparkles className="w-12 h-12 text-purple-600 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">질문을 불러오는 중...</p>
        </div>
      </div>
    )
  }

  if (analyzing) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Sparkles className="w-12 h-12 text-purple-600 animate-spin mx-auto mb-4" />
          <p className="text-xl font-medium mb-2">생시를 분석하고 있습니다...</p>
          <p className="text-gray-600">잠시만 기다려주세요</p>
        </div>
      </div>
    )
  }

  const currentEventType = mapQuestionEventType(currentQuestion?.event_type || '')
  const helperText = HELPER_TEXT[currentEventType]
  const currentEventAnswer = (answers[currentQuestion?.id] as EventAnswer | undefined) || {}

  return (
    <div className="min-h-screen bg-gradient-to-b from-purple-50 to-white">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">생시 보정 질문</h1>
          <p className="text-gray-600">
            {isPersonalityQuestion ? '당신의 성향을 알려주세요' : '인생의 중요한 이벤트에 대해 알려주세요'}
          </p>
        </div>

        <div className="max-w-2xl mx-auto mb-8">
          <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
            <span>
              질문 {currentStep + 1} / {questions.length}
            </span>
            <span>{Math.round(progress)}% 완료</span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        {currentQuestion && (
          <Card className="max-w-2xl mx-auto">
            <CardHeader>
              <CardTitle className="text-xl">{currentQuestion.text_ko || currentQuestion.text}</CardTitle>
              <CardDescription>
                {isPersonalityQuestion
                  ? '가장 가까운 답변을 선택해주세요'
                  : '해당 이벤트가 있었다면 시기를 선택해주세요'}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {isPersonalityQuestion && currentQuestion.options && (
                <RadioGroup
                  value={(answers[currentQuestion.id] as { choice?: string })?.choice || ''}
                  onValueChange={(v) => handleAnswer(currentQuestion.id, { choice: v })}
                >
                  {Object.entries(currentQuestion.options).map(([key, value]: [string, any]) => (
                    <div key={key} className="flex items-center space-x-2 p-3 border rounded-lg hover:bg-gray-50 cursor-pointer">
                      <RadioGroupItem value={key} id={`${currentQuestion.id}-${key}`} />
                      <Label htmlFor={`${currentQuestion.id}-${key}`} className="flex-1 cursor-pointer">
                        {value}
                      </Label>
                    </div>
                  ))}
                </RadioGroup>
              )}

              {isEventQuestion && (
                <>
                  <div>
                    <Label className="mb-3 block">해당 사항이 있나요?</Label>
                    <RadioGroup
                      value={currentEventAnswer.hasEvent ? 'yes' : 'no'}
                      onValueChange={(v) =>
                        handleAnswer(currentQuestion.id, {
                          ...currentEventAnswer,
                          hasEvent: v === 'yes',
                        })
                      }
                    >
                      <div className="flex items-center space-x-2 p-3 border rounded-lg hover:bg-gray-50 cursor-pointer">
                        <RadioGroupItem value="yes" id="yes" />
                        <Label htmlFor="yes" className="flex-1 cursor-pointer">
                          예
                        </Label>
                      </div>
                      <div className="flex items-center space-x-2 p-3 border rounded-lg hover:bg-gray-50 cursor-pointer">
                        <RadioGroupItem value="no" id="no" />
                        <Label htmlFor="no" className="flex-1 cursor-pointer">
                          아니오
                        </Label>
                      </div>
                    </RadioGroup>
                  </div>

                  {currentEventAnswer.hasEvent && (
                    <div className="space-y-4 p-4 bg-purple-50 rounded-lg">
                      <Label className="font-medium">How precisely do you remember when this happened?</Label>
                      {helperText && <p className="text-xs text-gray-600">{helperText}</p>}

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
                        <div className="flex items-center space-x-2 p-3 border rounded-lg bg-white">
                          <RadioGroupItem value="exact" id="precision-exact" />
                          <Label htmlFor="precision-exact" className="flex-1">
                            Exact Year
                          </Label>
                        </div>
                        <div className="flex items-center space-x-2 p-3 border rounded-lg bg-white">
                          <RadioGroupItem value="range" id="precision-range" />
                          <Label htmlFor="precision-range" className="flex-1">
                            Age Range
                          </Label>
                        </div>
                        <div className="flex items-center space-x-2 p-3 border rounded-lg bg-white">
                          <RadioGroupItem value="unknown" id="precision-unknown" />
                          <Label htmlFor="precision-unknown" className="flex-1">
                            I don&apos;t remember
                          </Label>
                        </div>
                      </RadioGroup>

                      {currentEventAnswer.precision_level === 'exact' && (
                        <div>
                          <Label htmlFor="year">년도 *</Label>
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
                          <Label>나이 구간 *</Label>
                          <Select
                            value={currentEventAnswer.age_range_label || ''}
                            onValueChange={(v) =>
                              handleAnswer(currentQuestion.id, {
                                ...currentEventAnswer,
                                age_range_label: v,
                              })
                            }
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="선택하세요" />
                            </SelectTrigger>
                            <SelectContent>
                              {AGE_RANGE_OPTIONS.map((option) => (
                                <SelectItem key={option.label} value={option.label}>
                                  {option.label}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      )}

                      {currentEventType === 'other' && (
                        <div>
                          <Label>What kind of event was this?</Label>
                          <Input
                            value={currentEventAnswer.other_label || ''}
                            onChange={(e) =>
                              handleAnswer(currentQuestion.id, {
                                ...currentEventAnswer,
                                other_label: e.target.value,
                              })
                            }
                            placeholder="예: 유학 시작, 큰 수술"
                          />
                        </div>
                      )}
                    </div>
                  )}
                </>
              )}

              {inlineError && <p className="text-sm text-red-600">{inlineError}</p>}

              {showFollowUpPrompt && !timedEventExists && (
                <div className="p-4 rounded-lg border border-amber-300 bg-amber-50 space-y-3">
                  <p className="text-sm font-medium">We don’t yet have enough timing data. Can you recall another event?</p>
                  <p className="text-xs text-gray-600">Follow-up attempts: {followUpAttempts}/3</p>

                  <div>
                    <Label className="mb-2 block">How precisely do you remember when this happened?</Label>
                    <RadioGroup
                      value={followUpAnswer.precision_level || 'unknown'}
                      onValueChange={(v) =>
                        setFollowUpAnswer((prev) => ({
                          ...prev,
                          precision_level: v as PrecisionLevel,
                          year: undefined,
                          age_range_label: undefined,
                        }))
                      }
                    >
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="exact" id="followup-exact" />
                        <Label htmlFor="followup-exact">Exact Year</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="range" id="followup-range" />
                        <Label htmlFor="followup-range">Age Range</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="unknown" id="followup-unknown" />
                        <Label htmlFor="followup-unknown">I don&apos;t remember</Label>
                      </div>
                    </RadioGroup>
                  </div>

                  {followUpAnswer.precision_level === 'exact' && (
                    <Input
                      type="number"
                      placeholder="연도"
                      value={followUpAnswer.year || ''}
                      onChange={(e) => setFollowUpAnswer((prev) => ({ ...prev, year: e.target.value }))}
                    />
                  )}

                  {followUpAnswer.precision_level === 'range' && (
                    <Select
                      value={followUpAnswer.age_range_label || ''}
                      onValueChange={(v) => setFollowUpAnswer((prev) => ({ ...prev, age_range_label: v }))}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="나이 구간" />
                      </SelectTrigger>
                      <SelectContent>
                        {AGE_RANGE_OPTIONS.map((option) => (
                          <SelectItem key={option.label} value={option.label}>
                            {option.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}

                  <Input
                    placeholder="What kind of event was this?"
                    value={followUpAnswer.other_label || ''}
                    onChange={(e) => setFollowUpAnswer((prev) => ({ ...prev, other_label: e.target.value }))}
                  />

                  <div className="flex gap-2">
                    <Button variant="outline" onClick={submitFollowUpEvent} disabled={followUpAttempts >= 3}>
                      Add Follow-Up Event
                    </Button>
                    {followUpAttempts >= 3 && (
                      <Button
                        onClick={() => {
                          setAllowLowPrecisionContinue(true)
                          setInlineError('Low timing precision may reduce rectification confidence.')
                        }}
                      >
                        Continue with warning
                      </Button>
                    )}
                  </div>
                </div>
              )}

              <div className="flex gap-4 pt-4">
                <Button variant="outline" onClick={handlePrevious} disabled={currentStep === 0} className="flex-1">
                  <ChevronLeft className="w-4 h-4 mr-2" />
                  이전
                </Button>
                <Button
                  onClick={handleNext}
                  className="flex-1"
                  disabled={
                    (isPersonalityQuestion && !(answers[currentQuestion.id] as { choice?: string })?.choice) ||
                    (isEventQuestion && !canProceedEventQuestion())
                  }
                >
                  {currentStep === questions.length - 1 ? '분석 시작' : '다음'}
                  <ChevronRight className="w-4 h-4 ml-2" />
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        <div className="max-w-2xl mx-auto mt-6 text-center">
          <div className="inline-flex gap-6 text-sm text-gray-600">
            <span>성격 답변: {Object.values(answers).filter((a: any) => a.choice).length}개</span>
            <span>입력된 이벤트: {eventList.length}개</span>
            <span>정밀 이벤트: {eventList.filter((event) => event.precision_level !== 'unknown').length}개</span>
          </div>
        </div>
      </div>
    </div>
  )
}
