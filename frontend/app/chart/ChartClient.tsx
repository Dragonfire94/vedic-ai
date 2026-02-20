'use client'

import { useEffect, useMemo, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Compass, Download, Home, Sparkles, ChevronDown, ChevronUp, Star } from 'lucide-react'
import { getAIReading, getChart, getPDF, type AIReadingResponse, type ChartResponse, type PlanetData } from '@/lib/api'
import { ASCENDANT_TRAITS, PLANET_NAMES_KR, toNum } from '@/lib/utils'

type PlanetRow = {
  name: string
  sign: string
  house: number | string
  easyMeaning: string
}

const HOUSE_MEANING: Record<number, string> = {
  1: '자신, 첫인상',
  2: '재물과 말',
  3: '소통과 이동',
  4: '가정, 마음 안정',
  5: '연애, 창의성',
  6: '건강 관리, 노력',
  7: '관계, 파트너십',
  8: '변화, 위기',
  9: '가치관, 성장',
  10: '직업, 커리어',
  11: '목표, 기회',
  12: '휴식, 정리',
}

function getPlanetEmoji(name: string): string {
  const m: Record<string, string> = {
    Sun: '☉',
    Moon: '☾',
    Mars: '♂',
    Mercury: '☿',
    Jupiter: '♃',
    Venus: '♀',
    Saturn: '♄',
    Rahu: '☊',
    Ketu: '☋',
  }
  return m[name] || '✦'
}

function getEasyPlanetMeaning(name: string, house: number | undefined): string {
  const base: Record<string, string> = {
    Sun: '자신감과 존재감',
    Moon: '감정과 안정감',
    Mars: '추진력과 결단',
    Mercury: '감각과 소통',
    Jupiter: '성장과 기회',
    Venus: '관계와 매력',
    Saturn: '책임감과 규칙',
    Rahu: '새로운 욕망',
    Ketu: '놓아줌과 지혜',
  }
  const houseText = house ? HOUSE_MEANING[house] || '삶의 영역' : '삶의 영역'
  return `${base[name] || '주제'}에 해당하는 "${houseText}"을(를) 중점적으로 보여줘요.`
}

export default function ChartPage() {
  const router = useRouter()
  const searchParams = useSearchParams()

  const [chart, setChart] = useState<ChartResponse | null>(null)
  const [aiReading, setAIReading] = useState<AIReadingResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadingAI, setLoadingAI] = useState(false)
  const [loadingPDF, setLoadingPDF] = useState(false)
  const [activeTab, setActiveTab] = useState('summary')
  const [expandedPlanet, setExpandedPlanet] = useState<string | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const birthData = {
    year: toNum(searchParams.get('year'), 1994),
    month: toNum(searchParams.get('month'), 12),
    day: toNum(searchParams.get('day'), 18),
    hour: toNum(searchParams.get('hour'), 23.75),
    lat: toNum(searchParams.get('lat'), 37.5665),
    lon: toNum(searchParams.get('lon'), 126.978),
    gender: searchParams.get('gender') || 'female',
    house_system: searchParams.get('house_system') || 'W',
  }

  useEffect(() => {
    const loadChart = async () => {
      try {
        const data = await getChart({
          ...birthData,
          include_nodes: true,
          include_d9: false,
          include_vargas: [],
        })
        setChart(data)
      } catch (error) {
        console.error('Failed to load chart:', error)
        const msg = error instanceof Error ? error.message : 'Unknown error'
        alert(`차트 불러오기에 실패했습니다.\n원인: ${msg}`)
      } finally {
        setLoading(false)
      }
    }
    loadChart()
  }, [])

  const handleLoadAIReading = async () => {
    if (aiReading) {
      setActiveTab('reading')
      return
    }
    setLoadingAI(true)
    try {
      const data = await getAIReading({
        ...birthData,
        language: 'ko',
        include_nodes: true,
        include_d9: true,
        include_vargas: ['d7', 'd9', 'd10', 'd12'],
        analysis_mode: 'pro',
      })
      setAIReading(data)
      setActiveTab('reading')
    } catch (error) {
      console.error('Failed to load AI reading:', error)
      const msg = error instanceof Error ? error.message : 'Unknown error'
      alert(`AI 해석 불러오기에 실패했습니다.\n원인: ${msg}`)
    } finally {
      setLoadingAI(false)
    }
  }

  const handleDownloadPDF = async () => {
    setLoadingPDF(true)
    try {
      const blob = await getPDF({
        ...birthData,
        language: 'ko',
        include_nodes: true,
        include_d9: true,
        include_vargas: ['d9', 'd10', 'd12'],
        analysis_mode: 'pro',
      })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `vedic-report-${birthData.year}${birthData.month}${birthData.day}.pdf`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Failed to download PDF:', error)
      const msg = error instanceof Error ? error.message : 'Unknown error'
      alert(`PDF 다운로드에 실패했습니다.\n원인: ${msg}`)
    } finally {
      setLoadingPDF(false)
    }
  }

  const ascendant = chart?.houses?.ascendant
  const ascendantName = ascendant?.rasi?.name
  const ascendantInfo = ascendantName ? ASCENDANT_TRAITS[ascendantName] : undefined
  const displayAscendant = ascendantInfo || {
    name_kr: ascendant?.rasi?.name_kr || '알 수 없음',
    emoji: '❓',
    keywords: [],
    preview: '상승궁 정보를 확인할 수 없습니다.',
  }

  const planetRows: PlanetRow[] = useMemo(() => {
    if (!chart?.planets || typeof chart.planets !== 'object') return []
    return Object.entries(chart.planets).map(([name, data]: [string, PlanetData]) => {
      const houseNum = Number(data?.house || 0)
      return {
        name,
        sign: data?.rasi?.name_kr || data?.rasi?.name || '-',
        house: data?.house || '-',
        easyMeaning: getEasyPlanetMeaning(name, houseNum || undefined),
      }
    })
  }, [chart])

  const top3 = planetRows.slice(0, 3)
  const hourInt = Math.floor(birthData.hour)
  const minInt = Math.round((birthData.hour - hourInt) * 60)
  const readingText = aiReading?.polished_reading ?? ''

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#f7f6f3]">
        <div className="text-center">
          <Sparkles className="w-10 h-10 mx-auto mb-3 text-[#8d3d56] animate-pulse" />
          <p className="text-[#534e57]">당신의 차트를 준비하고 있어요...</p>
        </div>
      </div>
    )
  }

  if (!chart) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#f7f6f3]">
        <Card className="max-w-md w-full">
          <CardHeader>
            <CardTitle>차트를 불러오지 못했습니다</CardTitle>
            <CardDescription>잠시 후 다시 시도해 주세요.</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => router.push('/')} className="w-full">
              처음으로 돌아가기
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-[linear-gradient(180deg,#f7f6f3_0%,#fff_36%)]">
      <div className="container mx-auto px-4 py-12 max-w-6xl">
        <div className="text-center mb-10">
          <p className="text-sm tracking-[0.18em] uppercase text-[#8a808a] mb-3">Vedic Signature</p>
          <h1 className="text-3xl md:text-4xl font-semibold text-[#2b2731]">쉽게 보는 내 성향 리포트</h1>
          <p className="text-[#5f5a64] mt-3">
            {birthData.year}.{birthData.month}.{birthData.day} {hourInt}:{String(minInt).padStart(2, '0')}
          </p>
        </div>

        <Card className="border-[#e5d9de] bg-white shadow-sm mb-7">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-[#3a3240]">
              <Compass className="w-5 h-5 text-[#8d3d56]" />
              요약
            </CardTitle>
            <CardDescription>전문 용어 대신 쉬운 표현으로 정리했어요.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="flex items-start gap-4 rounded-xl border border-[#f0e6ea] bg-[#fff9fb] p-4">
              <div className="text-4xl">{displayAscendant.emoji}</div>
              <div>
                <p className="text-sm text-[#866878] mb-1">상승궁</p>
                <h2 className="text-xl font-semibold text-[#2f2a33]">{displayAscendant.name_kr}</h2>
                <p className="text-[#5f5a64] mt-2">{displayAscendant.preview}</p>
                <div className="flex flex-wrap gap-2 mt-3">
                  {displayAscendant.keywords.slice(0, 4).map((k: string, i: number) => (
                    <Badge key={`${k}-${i}`} variant="secondary" className="bg-[#f5edf1] text-[#694958]">
                      {k}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-3">
              {top3.map((p) => (
                <div key={p.name} className="rounded-lg border border-[#ece5ea] p-3 bg-[#fdfcfc]">
                  <p className="text-xs text-[#877b86] mb-1">대표 행성</p>
                  <p className="font-medium text-[#302a33]">
                    {getPlanetEmoji(p.name)} {PLANET_NAMES_KR[p.name] || p.name}
                  </p>
                  <p className="text-sm text-[#5e5761] mt-1">{p.easyMeaning}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <div className="flex flex-wrap gap-3 justify-center mb-8">
          <Button onClick={handleLoadAIReading} disabled={loadingAI} className="bg-[#8d3d56] hover:bg-[#7a344a]">
            <Sparkles className="w-4 h-4 mr-2" />
            {loadingAI ? 'AI 해석 생성 중...' : aiReading ? 'AI 해석 보기' : 'AI 해석 생성'}
          </Button>
          <Button onClick={handleDownloadPDF} disabled={loadingPDF} variant="outline" className="border-[#ccb8c2]">
            <Download className="w-4 h-4 mr-2" />
            {loadingPDF ? 'PDF 준비 중...' : 'PDF 다운로드'}
          </Button>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3 mb-6 bg-[#f3eff2]">
            <TabsTrigger value="summary">요약</TabsTrigger>
            <TabsTrigger value="planets">행성 해석</TabsTrigger>
            <TabsTrigger value="reading" disabled={!aiReading}>AI 해석</TabsTrigger>
          </TabsList>

          <TabsContent value="summary">
            <Card>
              <CardHeader>
                <CardTitle className="text-[#362f39]">삶의 가이드</CardTitle>
                <CardDescription>오늘 바로 적용할 수 있는 조언을 모았어요.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 text-[#504a54]">
                <p>1) 중요한 결정은 감정이 가라앉은 뒤에 해 주세요.</p>
                <p>2) 관계 속에서는 솔직함과 경계를 함께 지켜주세요.</p>
                <p>3) 루틴(수면, 식사, 이동)을 지키면 안정감이 커져요.</p>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="planets">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {planetRows.map((p) => {
                const open = expandedPlanet === p.name
                return (
                  <Card key={p.name} className="border-[#ece6ea]">
                    <CardHeader
                      className="cursor-pointer"
                      onClick={() => setExpandedPlanet(open ? null : p.name)}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <CardTitle className="text-base text-[#352f38] flex items-center gap-2">
                            <span>{getPlanetEmoji(p.name)}</span>
                            {PLANET_NAMES_KR[p.name] || p.name}
                          </CardTitle>
                          <CardDescription>{p.sign} / {p.house} 하우스</CardDescription>
                        </div>
                        {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                      </div>
                    </CardHeader>
                    {open && (
                      <CardContent className="text-sm text-[#5b5560] space-y-2">
                        <p>{p.easyMeaning}</p>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="px-0 text-[#7b5366] hover:text-[#6b4557]"
                          onClick={() => setShowAdvanced((v) => !v)}
                        >
                          <Star className="w-4 h-4 mr-1" />
                          {showAdvanced ? '상세 정보 숨기기' : '상세 정보 보기'}
                        </Button>
                        {showAdvanced && (
                          <div className="rounded-md bg-[#f6f2f4] border border-[#e9dde2] p-3 text-xs text-[#6f6470]">
                            <p>별자리(라시): {chart.planets?.[p.name]?.rasi?.name || '-'}</p>
                            <p>낙샤트라: {chart.planets?.[p.name]?.nakshatra?.name || '-'}</p>
                            <p>파다: {chart.planets?.[p.name]?.nakshatra?.pada || '-'}</p>
                          </div>
                        )}
                      </CardContent>
                    )}
                  </Card>
                )
              })}
            </div>
          </TabsContent>

          <TabsContent value="reading">
            <Card>
              <CardHeader>
                <CardTitle className="text-[#362f39]">AI 해석</CardTitle>
                <CardDescription>필요하면 다시 생성해서 최신 버전으로 볼 수 있어요.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="whitespace-pre-wrap break-words">
                  {readingText}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

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
