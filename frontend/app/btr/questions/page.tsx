'use client'

import { useState, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Progress } from '@/components/ui/progress'
import { Sparkles, ChevronRight, ChevronLeft, Calendar } from 'lucide-react'
import { getBTRQuestions, analyzeBTR, type BTREvent } from '@/lib/api'

export default function BTRQuestionsPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  
  const birthData = {
    year: parseInt(searchParams.get('year') || '1994'),
    month: parseInt(searchParams.get('month') || '12'),
    day: parseInt(searchParams.get('day') || '18'),
    lat: parseFloat(searchParams.get('lat') || '37.5665'),
    lon: parseFloat(searchParams.get('lon') || '126.978'),
    gender: searchParams.get('gender') || 'male',
    timeBracket: searchParams.get('timeBracket') || 'all',
  }

  const [questions, setQuestions] = useState<any[]>([])
  const [currentStep, setCurrentStep] = useState(0)
  const [answers, setAnswers] = useState<Record<string, any>>({})
  const [events, setEvents] = useState<BTREvent[]>([])
  const [loading, setLoading] = useState(true)
  const [analyzing, setAnalyzing] = useState(false)

  // 질문 불러오기
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
  
  // 성격 질문인지 이벤트 질문인지 판단
  const isPersonalityQuestion = currentQuestion?.event_type === 'personality'
  const isEventQuestion = !isPersonalityQuestion && currentQuestion?.type === 'yesno_date'

  const handleAnswer = (questionId: string, value: any) => {
    setAnswers({ ...answers, [questionId]: value })
  }

  const handleNext = () => {
    // 이벤트 질문이고 답변이 있으면 이벤트 추가
    if (isEventQuestion && currentQuestion && answers[currentQuestion.id]?.hasEvent) {
      const year = answers[currentQuestion.id].year
      const month = answers[currentQuestion.id].month
      
      if (year) {
        const event: BTREvent = {
          type: currentQuestion.event_type,
          year: parseInt(year),
          month: month ? parseInt(month) : undefined,
          weight: currentQuestion.weight || 1.0,
          dasha_lords: currentQuestion.dasha_lords || [],
          house_triggers: currentQuestion.house_triggers || [],
        }
        
        setEvents([...events, event])
      }
    }

    if (currentStep < questions.length - 1) {
      setCurrentStep(currentStep + 1)
    } else {
      // 마지막 질문 → 분석 시작
      handleAnalyze()
    }
  }

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const handleAnalyze = async () => {
    if (events.length === 0) {
      alert('최소 1개 이상의 이벤트를 입력해주세요.')
      return
    }

    setAnalyzing(true)
    
    try {
      const result = await analyzeBTR({
        year: birthData.year,
        month: birthData.month,
        day: birthData.day,
        lat: birthData.lat,
        lon: birthData.lon,
        events: events,
      })

      // 결과를 URL 파라미터로 전달
      const params = new URLSearchParams(searchParams.toString())
      params.set('result', JSON.stringify(result))
      router.push(`/btr/results?${params}`)
    } catch (error) {
      console.error('BTR analysis failed:', error)
      alert('분석 중 오류가 발생했습니다. 다시 시도해주세요.')
    } finally {
      setAnalyzing(false)
    }
  }

  // 년도 옵션 생성 (출생년도 ~ 현재년도)
  const yearOptions = Array.from(
    { length: new Date().getFullYear() - birthData.year + 1 },
    (_, i) => new Date().getFullYear() - i
  )

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

  return (
    <div className="min-h-screen bg-gradient-to-b from-purple-50 to-white">
      <div className="container mx-auto px-4 py-16">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">생시 보정 질문</h1>
          <p className="text-gray-600">
            {isPersonalityQuestion 
              ? '당신의 성향을 알려주세요' 
              : '인생의 중요한 이벤트에 대해 알려주세요'}
          </p>
        </div>

        {/* Progress */}
        <div className="max-w-2xl mx-auto mb-8">
          <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
            <span>질문 {currentStep + 1} / {questions.length}</span>
            <span>{Math.round(progress)}% 완료</span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        {/* Question Card */}
        {currentQuestion && (
          <Card className="max-w-2xl mx-auto">
            <CardHeader>
              <CardTitle className="text-xl">
                {currentQuestion.text_ko || currentQuestion.text}
              </CardTitle>
              <CardDescription>
                {isPersonalityQuestion 
                  ? '가장 가까운 답변을 선택해주세요'
                  : '해당 이벤트가 있었다면 시기를 선택해주세요'}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* 성격 질문 - 객관식 4지선다 */}
              {isPersonalityQuestion && currentQuestion.options && (
                <RadioGroup
                  value={answers[currentQuestion.id]?.choice || ''}
                  onValueChange={(v) => handleAnswer(currentQuestion.id, { choice: v })}
                >
                  {Object.entries(currentQuestion.options).map(([key, value]: [string, any]) => (
                    <div
                      key={key}
                      className="flex items-center space-x-2 p-3 border rounded-lg hover:bg-gray-50 cursor-pointer"
                    >
                      <RadioGroupItem value={key} id={`${currentQuestion.id}-${key}`} />
                      <Label 
                        htmlFor={`${currentQuestion.id}-${key}`}
                        className="flex-1 cursor-pointer"
                      >
                        {value}
                      </Label>
                    </div>
                  ))}
                </RadioGroup>
              )}

              {/* 이벤트 질문 - Yes/No + 날짜 */}
              {isEventQuestion && (
                <>
                  {/* Yes/No */}
                  <div>
                    <Label className="mb-3 block">해당 사항이 있나요?</Label>
                    <RadioGroup
                      value={answers[currentQuestion.id]?.hasEvent ? 'yes' : 'no'}
                      onValueChange={(v) => handleAnswer(currentQuestion.id, { 
                        ...answers[currentQuestion.id],
                        hasEvent: v === 'yes' 
                      })}
                    >
                      <div className="flex items-center space-x-2 p-3 border rounded-lg hover:bg-gray-50 cursor-pointer">
                        <RadioGroupItem value="yes" id="yes" />
                        <Label htmlFor="yes" className="flex-1 cursor-pointer">예</Label>
                      </div>
                      <div className="flex items-center space-x-2 p-3 border rounded-lg hover:bg-gray-50 cursor-pointer">
                        <RadioGroupItem value="no" id="no" />
                        <Label htmlFor="no" className="flex-1 cursor-pointer">아니오</Label>
                      </div>
                    </RadioGroup>
                  </div>

                  {/* Date selects (if yes) */}
                  {answers[currentQuestion.id]?.hasEvent && (
                    <div className="space-y-4 p-4 bg-purple-50 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <Calendar className="w-4 h-4 text-purple-600" />
                        <Label className="font-medium">언제였나요?</Label>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <Label htmlFor="year">년도 *</Label>
                          <Select
                            value={answers[currentQuestion.id]?.year || ''}
                            onValueChange={(v) => handleAnswer(currentQuestion.id, {
                              ...answers[currentQuestion.id],
                              year: v
                            })}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="선택하세요" />
                            </SelectTrigger>
                            <SelectContent>
                              {yearOptions.map((year) => (
                                <SelectItem key={year} value={year.toString()}>
                                  {year}년
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        <div>
                          <Label htmlFor="month">월 (선택)</Label>
                          <Select
                            value={answers[currentQuestion.id]?.month || ''}
                            onValueChange={(v) => handleAnswer(currentQuestion.id, {
                              ...answers[currentQuestion.id],
                              month: v
                            })}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="선택" />
                            </SelectTrigger>
                            <SelectContent>
                              {Array.from({ length: 12 }, (_, i) => i + 1).map((m) => (
                                <SelectItem key={m} value={m.toString()}>
                                  {m}월
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* Navigation */}
              <div className="flex gap-4 pt-4">
                <Button
                  variant="outline"
                  onClick={handlePrevious}
                  disabled={currentStep === 0}
                  className="flex-1"
                >
                  <ChevronLeft className="w-4 h-4 mr-2" />
                  이전
                </Button>
                <Button
                  onClick={handleNext}
                  className="flex-1"
                  disabled={
                    // 성격 질문: 선택 필수
                    (isPersonalityQuestion && !answers[currentQuestion.id]?.choice) ||
                    // 이벤트 질문: 예 선택 시 년도 필수
                    (isEventQuestion && answers[currentQuestion.id]?.hasEvent && !answers[currentQuestion.id]?.year)
                  }
                >
                  {currentStep === questions.length - 1 ? '분석 시작' : '다음'}
                  <ChevronRight className="w-4 h-4 ml-2" />
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Stats */}
        <div className="max-w-2xl mx-auto mt-6 text-center">
          <div className="inline-flex gap-6 text-sm text-gray-600">
            <span>성격 답변: {Object.values(answers).filter((a: any) => a.choice).length}개</span>
            <span>입력된 이벤트: {events.length}개</span>
          </div>
        </div>
      </div>
    </div>
  )
}
