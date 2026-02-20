'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Sparkles, Calendar, Clock, MapPin } from 'lucide-react'
import { CitySearch } from '@/components/CitySearch'
import tzlookup from 'tz-lookup'

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
    gender: 'female',
    timeKnown: 'unknown' as TimeKnown,
    hour: 12,
    minute: 0,
    ampm: 'AM' as 'AM' | 'PM',
    timeBracket: '',
    timezone: 0,
  })

  const handleNext = () => {
    if (step === 1) {
      if (!formData.city || !Number.isFinite(formData.lat) || !Number.isFinite(formData.lon)) {
        alert('출생 도시를 먼저 선택해 주세요.')
        return
      }
      setStep(2)
      return
    }

    const safeTimezone = Number.isFinite(formData.timezone) ? formData.timezone : 0
    const params = new URLSearchParams({
      year: String(formData.year),
      month: String(formData.month),
      day: String(formData.day),
      lat: String(formData.lat),
      lon: String(formData.lon),
      gender: formData.gender,
      timezone: String(safeTimezone),
    })

    if (formData.timeKnown === 'exact') {
      const hour24 =
        formData.ampm === 'AM'
          ? formData.hour === 12 ? 0 : formData.hour
          : formData.hour === 12 ? 12 : formData.hour + 12
      const decimalHour = hour24 + formData.minute / 60
      params.set('hour', String(decimalHour))
      params.set('house_system', 'W')
      router.push(`/chart?${params.toString()}`)
      return
    }

    if (formData.timeKnown === 'approximate') {
      if (!formData.timeBracket) {
        alert('기억나는 시간대를 선택해 주세요.')
        return
      }
      params.set('timeBracket', formData.timeBracket)
      router.push(`/btr/questions?${params.toString()}`)
      return
    }

    params.set('timeBracket', 'all')
    router.push(`/btr/questions?${params.toString()}`)
  }

  return (
    <div className="min-h-screen bg-[linear-gradient(180deg,#f7f6f3_0%,#fff_36%)]">
      <div className="container mx-auto px-4 py-12 max-w-3xl">
        <div className="text-center mb-10">
          <div className="flex items-center justify-center gap-2 mb-3">
            <Sparkles className="w-6 h-6 text-[#8d3d56]" />
            <h1 className="text-4xl font-semibold text-[#2b2731]">Vedic AI</h1>
          </div>
          <p className="text-[#5f5a64]">출생 정보와 시간 기억 수준으로 리포트를 시작합니다.</p>
        </div>

        <div className="flex items-center justify-center gap-3 text-sm mb-8">
          <span className={`px-3 py-1 rounded-full ${step >= 1 ? 'bg-[#8d3d56] text-white' : 'bg-gray-200 text-gray-600'}`}>1. 기본 정보</span>
          <span className="text-gray-400">→</span>
          <span className={`px-3 py-1 rounded-full ${step >= 2 ? 'bg-[#8d3d56] text-white' : 'bg-gray-200 text-gray-600'}`}>2. 출생 시간</span>
        </div>

        {step === 1 && (
          <Card className="border-[#e5d9de]">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Calendar className="w-5 h-5 text-[#8d3d56]" />
                출생 기본 정보
              </CardTitle>
              <CardDescription>정확한 출생 도시를 선택하면 타임존이 자동 설정됩니다.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <Label>연도</Label>
                  <Input type="number" value={formData.year} onChange={(e) => setFormData({ ...formData, year: Number(e.target.value) })} />
                </div>
                <div>
                  <Label>월</Label>
                  <Select value={String(formData.month)} onValueChange={(v) => setFormData({ ...formData, month: Number(v) })}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {Array.from({ length: 12 }, (_, i) => i + 1).map((m) => (
                        <SelectItem key={m} value={String(m)}>{m}월</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>일</Label>
                  <Select value={String(formData.day)} onValueChange={(v) => setFormData({ ...formData, day: Number(v) })}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {Array.from({ length: 31 }, (_, i) => i + 1).map((d) => (
                        <SelectItem key={d} value={String(d)}>{d}일</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div>
                <CitySearch
                  defaultValue={formData.city}
                  onCitySelect={(data) => {
                    let timezoneOffset = 0
                    try {
                      const ianaZone = tzlookup(data.lat, data.lon)
                      const now = new Date()
                      const utcOffset = new Intl.DateTimeFormat('en-US', {
                        timeZone: ianaZone,
                        timeZoneName: 'shortOffset',
                      }).formatToParts(now).find((p) => p.type === 'timeZoneName')?.value ?? 'UTC+0'
                      const match = utcOffset.match(/UTC([+-]\d+(?:\.\d+)?)/)
                      timezoneOffset = match ? Number(match[1]) : 0
                    } catch {
                      timezoneOffset = 0
                    }
                    setFormData({
                      ...formData,
                      city: data.city,
                      lat: data.lat,
                      lon: data.lon,
                      timezone: timezoneOffset,
                    })
                  }}
                />
                {formData.city && (
                  <p className="text-sm text-[#726a75] mt-2 flex items-center gap-1">
                    <MapPin className="w-3 h-3" />
                    {formData.city} ({formData.lat.toFixed(4)}, {formData.lon.toFixed(4)})
                  </p>
                )}
              </div>

              <div>
                <Label>성별</Label>
                <RadioGroup value={formData.gender} onValueChange={(v) => setFormData({ ...formData, gender: v })}>
                  <div className="flex items-center space-x-2"><RadioGroupItem value="female" id="female" /><Label htmlFor="female">여성</Label></div>
                  <div className="flex items-center space-x-2"><RadioGroupItem value="male" id="male" /><Label htmlFor="male">남성</Label></div>
                </RadioGroup>
              </div>

              <Button onClick={handleNext} className="w-full bg-[#8d3d56] hover:bg-[#7a344a]">다음</Button>
            </CardContent>
          </Card>
        )}

        {step === 2 && (
          <Card className="border-[#e5d9de]">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="w-5 h-5 text-[#8d3d56]" />
                출생 시간 정보
              </CardTitle>
              <CardDescription>출생 시간을 얼마나 기억하시나요?</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <RadioGroup value={formData.timeKnown} onValueChange={(v) => setFormData({ ...formData, timeKnown: v as TimeKnown })}>
                <Card className="border"><CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <RadioGroupItem value="exact" id="exact" />
                    <Label htmlFor="exact">정확히 기억함</Label>
                  </div>
                  {formData.timeKnown === 'exact' && (
                    <div className="grid grid-cols-3 gap-3 mt-3">
                      <div>
                        <Label>오전/오후</Label>
                        <Select
                          value={formData.ampm}
                          onValueChange={(v) => setFormData({ ...formData, ampm: v as 'AM' | 'PM' })}
                        >
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="AM">오전 (AM)</SelectItem>
                            <SelectItem value="PM">오후 (PM)</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label>시</Label>
                        <Select value={String(formData.hour)} onValueChange={(v) => setFormData({ ...formData, hour: Number(v) })}>
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            {Array.from({ length: 12 }, (_, i) => i + 1).map((h) => (
                              <SelectItem key={h} value={String(h)}>{h}시</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label>분</Label>
                        <Select value={String(formData.minute)} onValueChange={(v) => setFormData({ ...formData, minute: Number(v) })}>
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            {Array.from({ length: 12 }, (_, i) => i * 5).map((m) => (
                              <SelectItem key={m} value={String(m)}>{m}분</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  )}
                </CardContent></Card>

                <Card className="border"><CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <RadioGroupItem value="approximate" id="approximate" />
                    <Label htmlFor="approximate">대략 기억함</Label>
                  </div>
                  {formData.timeKnown === 'approximate' && (
                    <div className="mt-3">
                      <Label>시간대</Label>
                      <Select value={formData.timeBracket} onValueChange={(v) => setFormData({ ...formData, timeBracket: v })}>
                        <SelectTrigger><SelectValue placeholder="시간대 선택" /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="00:00-03:00">00:00 - 03:00</SelectItem>
                          <SelectItem value="03:00-06:00">03:00 - 06:00</SelectItem>
                          <SelectItem value="06:00-09:00">06:00 - 09:00</SelectItem>
                          <SelectItem value="09:00-12:00">09:00 - 12:00</SelectItem>
                          <SelectItem value="12:00-15:00">12:00 - 15:00</SelectItem>
                          <SelectItem value="15:00-18:00">15:00 - 18:00</SelectItem>
                          <SelectItem value="18:00-21:00">18:00 - 21:00</SelectItem>
                          <SelectItem value="21:00-00:00">21:00 - 00:00</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  )}
                </CardContent></Card>

                <Card className="border"><CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <RadioGroupItem value="unknown" id="unknown" />
                    <Label htmlFor="unknown">잘 모르겠음 (생시보정 필요)</Label>
                  </div>
                </CardContent></Card>
              </RadioGroup>

              <div className="flex gap-3">
                <Button variant="outline" className="flex-1" onClick={() => setStep(1)}>이전</Button>
                <Button className="flex-1 bg-[#8d3d56] hover:bg-[#7a344a]" onClick={handleNext}>
                  {formData.timeKnown === 'exact' ? '차트 보기' : '생시보정 시작하기'}
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
