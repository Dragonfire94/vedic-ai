'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Sparkles, Calendar, MapPin, Clock } from 'lucide-react'
import { CitySearch } from '@/components/CitySearch'

type TimeKnown = 'exact' | 'approximate' | 'unknown'

export default function HomePage() {
  const router = useRouter()
  const [step, setStep] = useState(1)
  const [formData, setFormData] = useState({
    year: 1994,
    month: 12,
    day: 18,
    city: '',
    lat: 0,
    lon: 0,
    timeKnown: 'unknown' as TimeKnown,
    hour: 23,
    minute: 45,
    timeBracket: '',
    gender: 'male',
  })

  const handleNext = () => {
    if (step === 1) {
      if (!formData.city || formData.lat === 0) {
        alert('출생 도시를 선택해 주세요.')
        return
      }
      setStep(2)
      return
    }

    const params = new URLSearchParams({
      year: formData.year.toString(),
      month: formData.month.toString(),
      day: formData.day.toString(),
      lat: formData.lat.toString(),
      lon: formData.lon.toString(),
      gender: formData.gender,
    })

    if (formData.timeKnown === 'exact') {
      params.append('hour', (formData.hour + formData.minute / 60).toString())
      params.append('house_system', 'W')
      router.push(`/chart?${params}`)
      return
    }

    if (formData.timeKnown === 'approximate') {
      if (!formData.timeBracket) {
        alert('시간대를 선택해 주세요.')
        return
      }
      params.append('timeBracket', formData.timeBracket)
      router.push(`/btr/questions?${params}`)
      return
    }

    params.append('timeBracket', 'all')
    router.push(`/btr/questions?${params}`)
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-purple-50 to-white">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Sparkles className="w-8 h-8 text-purple-600" />
            <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
              Vedic AI
            </h1>
          </div>
          <p className="text-2xl text-gray-700 mb-4">당신의 진짜 별자리를 찾아보세요</p>
          <p className="text-gray-600">출생 정보와 시간 정보로 차트 또는 BTR 분석을 시작할 수 있습니다.</p>
        </div>

        <div className="max-w-2xl mx-auto mb-8">
          <div className="flex items-center justify-center gap-4">
            <div className={`flex items-center gap-2 ${step >= 1 ? 'text-purple-600' : 'text-gray-400'}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${step >= 1 ? 'bg-purple-600 text-white' : 'bg-gray-200'}`}>
                1
              </div>
              <span className="font-medium">출생일</span>
            </div>
            <div className="h-px w-12 bg-gray-300" />
            <div className={`flex items-center gap-2 ${step >= 2 ? 'text-purple-600' : 'text-gray-400'}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${step >= 2 ? 'bg-purple-600 text-white' : 'bg-gray-200'}`}>
                2
              </div>
              <span className="font-medium">출생 시간</span>
            </div>
            <div className="h-px w-12 bg-gray-300" />
            <div className="flex items-center gap-2 text-gray-400">
              <div className="w-8 h-8 rounded-full flex items-center justify-center bg-gray-200">3</div>
              <span className="font-medium">결과</span>
            </div>
          </div>
        </div>

        <div className="max-w-2xl mx-auto">
          {step === 1 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Calendar className="w-5 h-5" />
                  출생 정보 입력
                </CardTitle>
                <CardDescription>출생 연월일, 도시, 성별을 입력해 주세요.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <Label htmlFor="year">연도</Label>
                    <Input
                      id="year"
                      type="number"
                      value={formData.year}
                      onChange={(e) => setFormData({ ...formData, year: parseInt(e.target.value, 10) })}
                      min={1900}
                      max={2025}
                    />
                  </div>
                  <div>
                    <Label htmlFor="month">월</Label>
                    <Select
                      value={formData.month.toString()}
                      onValueChange={(v) => setFormData({ ...formData, month: parseInt(v, 10) })}
                    >
                      <SelectTrigger id="month">
                        <SelectValue />
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
                  <div>
                    <Label htmlFor="day">일</Label>
                    <Select
                      value={formData.day.toString()}
                      onValueChange={(v) => setFormData({ ...formData, day: parseInt(v, 10) })}
                    >
                      <SelectTrigger id="day">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {Array.from({ length: 31 }, (_, i) => i + 1).map((d) => (
                          <SelectItem key={d} value={d.toString()}>
                            {d}일
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div>
                  <CitySearch
                    defaultValue={formData.city}
                    onCitySelect={(data) => {
                      setFormData({
                        ...formData,
                        city: data.city,
                        lat: data.lat,
                        lon: data.lon,
                      })
                    }}
                  />

                  {formData.city && (
                    <p className="text-sm text-gray-500 mt-2 flex items-center gap-1">
                      <MapPin className="w-3 h-3" />
                      {formData.city} ({formData.lat.toFixed(4)}, {formData.lon.toFixed(4)})
                    </p>
                  )}
                </div>

                <div>
                  <Label>성별</Label>
                  <RadioGroup
                    value={formData.gender}
                    onValueChange={(v) => setFormData({ ...formData, gender: v })}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="male" id="male" />
                      <Label htmlFor="male">남성</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="female" id="female" />
                      <Label htmlFor="female">여성</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="other" id="other" />
                      <Label htmlFor="other">기타</Label>
                    </div>
                  </RadioGroup>
                </div>

                <Button onClick={handleNext} className="w-full" size="lg">
                  다음 단계
                </Button>
              </CardContent>
            </Card>
          )}

          {step === 2 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="w-5 h-5" />
                  출생 시간 정보
                </CardTitle>
                <CardDescription>출생 시간을 어느 정도 정확하게 알고 있는지 선택해 주세요.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <RadioGroup
                  value={formData.timeKnown}
                  onValueChange={(v) => setFormData({ ...formData, timeKnown: v as TimeKnown })}
                >
                  <Card className="cursor-pointer hover:border-purple-600 transition-colors">
                    <CardContent className="p-4">
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="exact" id="exact" />
                        <Label htmlFor="exact" className="cursor-pointer flex-1">
                          <div className="font-medium">정확히 알고 있어요</div>
                          <div className="text-sm text-gray-500">시간과 분까지 입력할 수 있어요.</div>
                        </Label>
                      </div>
                      {formData.timeKnown === 'exact' && (
                        <div className="mt-4 grid grid-cols-2 gap-4">
                          <div>
                            <Label>시</Label>
                            <Select
                              value={formData.hour.toString()}
                              onValueChange={(v) => setFormData({ ...formData, hour: parseInt(v, 10) })}
                            >
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                {Array.from({ length: 24 }, (_, i) => i).map((h) => (
                                  <SelectItem key={h} value={h.toString()}>
                                    {h}시
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>
                          <div>
                            <Label>분</Label>
                            <Select
                              value={formData.minute.toString()}
                              onValueChange={(v) => setFormData({ ...formData, minute: parseInt(v, 10) })}
                            >
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                {Array.from({ length: 60 }, (_, i) => i).map((m) => (
                                  <SelectItem key={m} value={m.toString()}>
                                    {m}분
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  <Card className="cursor-pointer hover:border-purple-600 transition-colors">
                    <CardContent className="p-4">
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="approximate" id="approximate" />
                        <Label htmlFor="approximate" className="cursor-pointer flex-1">
                          <div className="font-medium">대략 알고 있어요</div>
                          <div className="text-sm text-gray-500">예: 오전, 저녁 같은 시간대 정보만 있어요.</div>
                        </Label>
                      </div>
                      {formData.timeKnown === 'approximate' && (
                        <div className="mt-4">
                          <Label>시간대 선택</Label>
                          <Select
                            value={formData.timeBracket}
                            onValueChange={(v) => setFormData({ ...formData, timeBracket: v })}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="시간대를 선택해 주세요." />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="00:00-03:00">00:00 - 03:00 (심야)</SelectItem>
                              <SelectItem value="03:00-06:00">03:00 - 06:00 (이른 새벽)</SelectItem>
                              <SelectItem value="06:00-09:00">06:00 - 09:00 (아침)</SelectItem>
                              <SelectItem value="09:00-12:00">09:00 - 12:00 (오전)</SelectItem>
                              <SelectItem value="12:00-15:00">12:00 - 15:00 (오후)</SelectItem>
                              <SelectItem value="15:00-18:00">15:00 - 18:00 (늦은 오후)</SelectItem>
                              <SelectItem value="18:00-21:00">18:00 - 21:00 (저녁)</SelectItem>
                              <SelectItem value="21:00-00:00">21:00 - 00:00 (밤)</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  <Card className="cursor-pointer hover:border-purple-600 transition-colors">
                    <CardContent className="p-4">
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="unknown" id="unknown" />
                        <Label htmlFor="unknown" className="cursor-pointer flex-1">
                          <div className="font-medium">전혀 모르겠어요</div>
                          <div className="text-sm text-gray-500">BTR 질문으로 시간대를 추정합니다.</div>
                        </Label>
                      </div>
                    </CardContent>
                  </Card>
                </RadioGroup>

                <div className="flex gap-4">
                  <Button onClick={() => setStep(1)} variant="outline" className="flex-1">
                    이전
                  </Button>
                  <Button onClick={handleNext} className="flex-1" size="lg">
                    {formData.timeKnown === 'exact' ? '차트 보기' : 'BTR 시작하기'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
