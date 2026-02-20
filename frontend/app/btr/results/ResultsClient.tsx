'use client'

import { useEffect, useMemo, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { ChevronDown, ChevronUp, Compass, Home, Sparkles } from 'lucide-react'
import { ASCENDANT_TRAITS } from '@/lib/utils'
import type { BTRCandidate } from '@/lib/api'
import { useBTRStore } from '@/store/btrStore'

function formatConfidencePercent(value: unknown): number {
  const numeric = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(numeric)) return 0
  const normalized = numeric <= 1 ? numeric * 100 : numeric
  return Math.max(0, Math.min(100, Math.round(normalized)))
}

function parseMidHour(candidate: BTRCandidate): string {
  return Number.isFinite(candidate.mid_hour)
    ? candidate.mid_hour.toFixed(2)
    : '-'
}

function confidenceLabel(pct: number): string {
  if (pct >= 80) return '높음'
  if (pct >= 60) return '보통 이상'
  if (pct >= 40) return '보통'
  return '참고'
}

export default function BTRResultsPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const result = useBTRStore((s) => s.result)

  const [expandedCard, setExpandedCard] = useState<number | null>(0)

  useEffect(() => {
    if (!result) {
      router.replace('/btr/questions')
    }
  }, [result, router])

  const candidates = useMemo(() => {
    const rows = Array.isArray(result?.candidates) ? result.candidates : []
    return rows.slice(0, 3)
  }, [result])

  const top = candidates[0]
  const topPct = formatConfidencePercent(top?.confidence)

  const handleSelectCandidate = (candidate: any) => {
  const hour = parseMidHour(candidate)
  const fallbackHour = searchParams.get('hour') || '12'
  const params = new URLSearchParams({
      year: searchParams.get('year') || '',
      month: searchParams.get('month') || '',
      day: searchParams.get('day') || '',
      lat: searchParams.get('lat') || '',
      lon: searchParams.get('lon') || '',
      hour: hour || fallbackHour,
      gender: searchParams.get('gender') || 'female',
      house_system: 'W',
      timezone: searchParams.get('timezone') || '9',
    })
    router.push(`/chart?${params}`)
  }

  if (!result || candidates.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#f7f6f3]">
        <Card className="max-w-md w-full">
          <CardHeader>
            <CardTitle>분석 결과를 불러오지 못했어요</CardTitle>
            <CardDescription>다시 분석을 실행해 주세요.</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => router.push('/btr/questions')} className="w-full">
              다시 분석하기
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-[linear-gradient(180deg,#f7f6f3_0%,#fff_36%)]">
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        <div className="text-center mb-8">
          <p className="text-sm tracking-[0.18em] uppercase text-[#8a808a] mb-3">Birth Time Check</p>
          <h1 className="text-3xl font-semibold text-[#2b2731]">가장 가능성 높은 시간대</h1>
          <p className="text-[#5f5a64] mt-3">입력한 이벤트를 기준으로 후보를 정리했어요.</p>
        </div>

        <Card className="mb-7 border-[#e5d9de] bg-white">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-[#3a3240]">
              <Sparkles className="w-5 h-5 text-[#8d3d56]" />
              추천 후보
            </CardTitle>
            <CardDescription>우선 이 후보부터 확인하는 것을 권장해요.</CardDescription>
          </CardHeader>
          <CardContent className="grid md:grid-cols-3 gap-3">
            <div className="rounded-lg border border-[#ece5ea] p-4 bg-[#fdfcfc]">
              <p className="text-xs text-[#877b86] mb-1">시간대</p>
              <p className="font-semibold text-[#302a33]">{top?.time_range || '-'}</p>
            </div>
            <div className="rounded-lg border border-[#ece5ea] p-4 bg-[#fdfcfc]">
              <p className="text-xs text-[#877b86] mb-1">신뢰도</p>
              <p className="font-semibold text-[#302a33]">{topPct}% ({confidenceLabel(topPct)})</p>
            </div>
            <div className="rounded-lg border border-[#ece5ea] p-4 bg-[#fdfcfc]">
              <p className="text-xs text-[#877b86] mb-1">상승궁</p>
              <p className="font-semibold text-[#302a33]">{top?.ascendant || '-'}</p>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4">
          {candidates.map((candidate: any, index: number) => {
            const open = expandedCard === index
            const pct = formatConfidencePercent(candidate?.confidence)
            const ascInfo = ASCENDANT_TRAITS[candidate?.ascendant] || {
              name_kr: candidate?.ascendant || '알 수 없음',
              emoji: '⭐',
              keywords: [],
              preview: '',
            }

            return (
              <Card key={`${candidate?.time_range}-${index}`} className="border-[#e9e1e6]">
                <CardHeader
                  className="cursor-pointer"
                  onClick={() => setExpandedCard(open ? null : index)}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <CardTitle className="text-base text-[#352f38] flex items-center gap-2">
                        <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-[#f5edf1] text-[#6e4557] text-sm">
                          {index + 1}
                        </span>
                        {candidate?.time_range || '시간대 정보 없음'}
                        {index === 0 && <Badge className="bg-[#8d3d56]">추천</Badge>}
                      </CardTitle>
                      <CardDescription className="mt-1">
                        {ascInfo.emoji} {ascInfo.name_kr} 상승궁 후보
                      </CardDescription>
                    </div>
                    {open ? <ChevronUp className="w-4 h-4 text-[#8e8390]" /> : <ChevronDown className="w-4 h-4 text-[#8e8390]" />}
                  </div>
                  <div className="mt-2">
                    <Progress value={pct} className="h-2" />
                    <p className="text-xs text-[#7a707c] mt-1">신뢰도 {pct}%</p>
                  </div>
                </CardHeader>

                {open && (
                  <CardContent className="space-y-4 text-sm text-[#5b5560]">
                    <div className="rounded-md bg-[#f6f2f4] border border-[#e9dde2] p-3">
                      {ascInfo.preview || '이 후보는 성향 일치율이 높게 계산되었습니다.'}
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {(ascInfo.keywords || []).slice(0, 4).map((k: string, i: number) => (
                        <Badge key={`${k}-${i}`} variant="secondary" className="bg-[#f3ecf0] text-[#684857]">
                          {k}
                        </Badge>
                      ))}
                    </div>
                    <Button
                      className="w-full bg-[#8d3d56] hover:bg-[#7a344a]"
                      onClick={() => handleSelectCandidate(candidate)}
                    >
                      <Compass className="w-4 h-4 mr-2" />
                      이 시간대로 차트 보기
                    </Button>
                  </CardContent>
                )}
              </Card>
            )
          })}
        </div>

        <div className="mt-8 text-center">
          <Button variant="outline" onClick={() => router.push('/')} className="border-[#cdb9c2]">
            <Home className="w-4 h-4 mr-2" />
            처음으로
          </Button>
        </div>
      </div>
    </div>
  )
}
