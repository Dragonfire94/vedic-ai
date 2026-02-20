'use client'

import { useEffect, useMemo, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { ChevronDown, ChevronUp, Compass, Home, Sparkles } from 'lucide-react'
import { ASCENDANT_TRAITS } from '@/lib/utils'
import type { BTRAnalyzeResponse, BTRCandidate } from '@/lib/api'
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
  if (pct >= 80) return '??'
  if (pct >= 60) return '?? ??'
  if (pct >= 40) return '??'
  return '??'
}

export default function BTRResultsPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const result = useBTRStore((s) => s.result) as BTRAnalyzeResponse | null

  const [expandedCard, setExpandedCard] = useState<number | null>(0)

  useEffect(() => {
    if (!result) {
      router.replace('/btr/questions')
    }
  }, [result, router])

  const candidates = useMemo<BTRCandidate[]>(() => {
    const rows = result?.candidates ?? []
    return rows.slice(0, 3)
  }, [result])

  const top = candidates[0]
  const topPct = formatConfidencePercent(top?.confidence)

  const handleSelectCandidate = (candidate: BTRCandidate) => {
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
    })
    router.push(`/chart?${params}`)
  }

  if (!result || candidates.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#f7f6f3]">
        <Card className="max-w-md w-full">
          <CardHeader>
            <CardTitle>遺꾩꽍 寃곌낵瑜?遺덈윭?ㅼ? 紐삵뻽?댁슂</CardTitle>
            <CardDescription>?ㅼ떆 遺꾩꽍??ㅽ뻾??二쇱꽭??</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => router.push('/btr/questions')} className="w-full">
              ?ㅼ떆 遺꾩꽍?섍린
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
          <h1 className="text-3xl font-semibold text-[#2b2731]">媛??媛?μ꽦 ?믪? ?쒓컙?</h1>
          <p className="text-[#5f5a64] mt-3">?낅젰??대깽?몃? 湲곗??쇰줈 ?꾨낫瑜??뺣━?덉뼱??</p>
        </div>

        <Card className="mb-7 border-[#e5d9de] bg-white">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-[#3a3240]">
              <Sparkles className="w-5 h-5 text-[#8d3d56]" />
              異붿쿇 ?꾨낫
            </CardTitle>
            <CardDescription>?곗꽑 ??꾨낫遺??뺤씤?섎뒗 寃껋쓣 沅뚯옣?댁슂.</CardDescription>
          </CardHeader>
          <CardContent className="grid md:grid-cols-3 gap-3">
            <div className="rounded-lg border border-[#ece5ea] p-4 bg-[#fdfcfc]">
              <p className="text-xs text-[#877b86] mb-1">?쒓컙?</p>
              <p className="font-semibold text-[#302a33]">{top?.time_range || '-'}</p>
            </div>
            <div className="rounded-lg border border-[#ece5ea] p-4 bg-[#fdfcfc]">
              <p className="text-xs text-[#877b86] mb-1">?좊ː??</p>
              <p className="font-semibold text-[#302a33]">{topPct}% ({confidenceLabel(topPct)})</p>
            </div>
            <div className="rounded-lg border border-[#ece5ea] p-4 bg-[#fdfcfc]">
              <p className="text-xs text-[#877b86] mb-1">?곸듅沅?</p>
              <p className="font-semibold text-[#302a33]">{top?.ascendant || '-'}</p>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4">
          {candidates.map((candidate: BTRCandidate, index: number) => {
            const open = expandedCard === index
            const pct = formatConfidencePercent(candidate?.confidence)
            const ascInfo = candidate?.ascendant ? ASCENDANT_TRAITS[candidate.ascendant] : undefined
            const displayAsc = ascInfo || {
              name_kr: candidate?.ascendant || '? ? ??',
              emoji: '?',
              keywords: [],
              preview: '? ?? ?? ?? ?? ???.',
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
                        {candidate?.time_range || '?쒓컙? ?뺣낫 ?놁쓬'}
                        {index === 0 && <Badge className="bg-[#8d3d56]">異붿쿇</Badge>}
                      </CardTitle>
                      <CardDescription className="mt-1">
                        {displayAsc.emoji} {displayAsc.name_kr} ?곸듅沅??꾨낫
                      </CardDescription>
                    </div>
                    {open ? <ChevronUp className="w-4 h-4 text-[#8e8390]" /> : <ChevronDown className="w-4 h-4 text-[#8e8390]" />}
                  </div>
                  <div className="mt-2">
                    <Progress value={pct} className="h-2" />
                    <p className="text-xs text-[#7a707c] mt-1">?좊ː??{pct}%</p>
                  </div>
                </CardHeader>

                {open && (
                  <CardContent className="space-y-4 text-sm text-[#5b5560]">
                    <div className="rounded-md bg-[#f6f2f4] border border-[#e9dde2] p-3">
                      {displayAsc.preview || '??꾨낫??깊뼢 ?쇱튂?⑥씠 ?믨쾶 怨꾩궛?섏뿀?듬땲??'}
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {(displayAsc.keywords || []).slice(0, 4).map((k: string, i: number) => (
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
                      ??쒓컙?濡?李⑦듃 蹂닿린
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
            泥섏쓬?쇰줈
          </Button>
        </div>
      </div>
    </div>
  )
}

