'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Compass, Download, Home, Sparkles, ChevronDown, ChevronUp, Star } from 'lucide-react'
import {
  getAIReading,
  getChart,
  getPDF,
  type AIReadingRequest,
  type AIReadingResponse,
  type ChartResponse,
  type LifeCycleOnboardingGoal,
  type PlanetData,
  type ProductType,
} from '@/lib/api'
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

const ONBOARDING_GOAL_LABELS: Record<LifeCycleOnboardingGoal, string> = {
  career_money: '일과 돈의 방향',
  relationship: '관계와 친밀감',
  condition: '컨디션과 회복 리듬',
  life_direction: '삶의 큰 방향',
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

function parseOptionalNumber(value: string | null): number | undefined {
  if (value === null || value.trim().length === 0) {
    return undefined
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function parseCsvParam(value: string | null, maxItems: number): string[] {
  if (!value) {
    return []
  }
  return value
    .split(',')
    .map((item) => item.trim())
    .filter((item, index, items) => item.length > 0 && items.indexOf(item) === index)
    .slice(0, maxItems)
}

function normalizeProductType(value: string | null): ProductType | undefined {
  switch ((value || '').trim().toLowerCase()) {
    case 'life_cycle':
      return 'life_cycle'
    case 'yearly_forecast':
      return 'yearly_forecast'
    case 'compatibility':
      return 'compatibility'
    default:
      return undefined
  }
}

function normalizeOnboardingGoal(value: string | null): LifeCycleOnboardingGoal | undefined {
  switch ((value || '').trim().toLowerCase()) {
    case 'career_money':
      return 'career_money'
    case 'relationship':
      return 'relationship'
    case 'condition':
      return 'condition'
    case 'life_direction':
      return 'life_direction'
    default:
      return undefined
  }
}

function formatOnboardingGoal(value: string | undefined): string | undefined {
  if (!value) {
    return undefined
  }
  return ONBOARDING_GOAL_LABELS[value as LifeCycleOnboardingGoal] || value
}

function loadAIReadingFromSessionCache(cacheKey: string): AIReadingResponse | null {
  if (typeof window === 'undefined') {
    return null
  }
  try {
    const raw = window.sessionStorage.getItem(cacheKey)
    if (!raw) {
      return null
    }
    return JSON.parse(raw) as AIReadingResponse
  } catch {
    return null
  }
}

function saveAIReadingToSessionCache(cacheKey: string, data: AIReadingResponse): void {
  if (typeof window === 'undefined') {
    return
  }
  try {
    window.sessionStorage.setItem(cacheKey, JSON.stringify(data))
  } catch {
    // Ignore client cache failures and continue with in-memory state.
  }
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
  const autoLoadAttemptedRef = useRef(false)

  const routeQueryString = searchParams.toString()
  const productType = normalizeProductType(searchParams.get('product_type'))
  const subjectName = searchParams.get('subject_name')?.trim() || undefined
  const onboardingGoal = normalizeOnboardingGoal(searchParams.get('onboarding_goal'))
  const focusTokens = parseCsvParam(searchParams.get('focus_tokens'), 2)
  const concernTokens = parseCsvParam(searchParams.get('concern_tokens'), 3)
  const occupationContext = searchParams.get('occupation_context')?.trim() || undefined
  const relationshipStatus = searchParams.get('relationship_status')?.trim() || undefined

  const chartRequest = useMemo(
    () => ({
      year: toNum(searchParams.get('year'), 1994),
      month: toNum(searchParams.get('month'), 12),
      day: toNum(searchParams.get('day'), 18),
      hour: toNum(searchParams.get('hour'), 23.75),
      lat: toNum(searchParams.get('lat'), 37.5665),
      lon: toNum(searchParams.get('lon'), 126.978),
      gender: searchParams.get('gender') || 'female',
      house_system: searchParams.get('house_system') || 'W',
      timezone: parseOptionalNumber(searchParams.get('timezone')),
      include_nodes: true,
      include_d9: false,
      include_vargas: [] as string[],
    }),
    [searchParams]
  )

  const aiReadingRequest = useMemo<AIReadingRequest>(
    () => ({
      ...chartRequest,
      language: 'ko',
      include_nodes: true,
      include_d9: true,
      include_vargas: productType === 'life_cycle' ? ['d9', 'd10', 'd12'] : ['d7', 'd9', 'd10', 'd12'],
      analysis_mode: 'pro',
      detail_level: 'full',
      product_type: productType,
      subject_name: subjectName,
      onboarding_goal: onboardingGoal,
      focus_tokens: focusTokens,
      concern_tokens: concernTokens,
      occupation_context: occupationContext,
      relationship_status: relationshipStatus,
    }),
    [chartRequest, concernTokens, focusTokens, occupationContext, onboardingGoal, productType, relationshipStatus, subjectName]
  )

  const pdfRequest = useMemo<AIReadingRequest>(
    () => ({
      ...aiReadingRequest,
      include_vargas: ['d9', 'd10', 'd12'],
    }),
    [aiReadingRequest]
  )
  const aiReadingSessionCacheKey = useMemo(
    () => `chart-ai-reading:${routeQueryString || 'default'}`,
    [routeQueryString]
  )

  useEffect(() => {
    autoLoadAttemptedRef.current = false
    setChart(null)
    setAIReading(null)
    setLoading(true)
    setLoadingAI(false)
    setActiveTab('summary')
  }, [aiReadingSessionCacheKey])

  useEffect(() => {
    const cached = loadAIReadingFromSessionCache(aiReadingSessionCacheKey)
    if (!cached) {
      return
    }
    setAIReading(cached)
    if ((cached.product_type || productType) === 'life_cycle') {
      setActiveTab('reading')
    }
  }, [aiReadingSessionCacheKey, productType])

  useEffect(() => {
    const loadChart = async () => {
      try {
        const data = await getChart(chartRequest)
        setChart(data)
      } catch (error) {
        console.error('Failed to load chart:', error)
        const msg = error instanceof Error ? error.message : 'Unknown error'
        alert(`차트 불러오기에 실패했습니다.
원인: ${msg}`)
      } finally {
        setLoading(false)
      }
    }
    loadChart()
  }, [chartRequest])

  const loadAIReading = async ({ activateTab }: { activateTab: boolean }) => {
    const cached = loadAIReadingFromSessionCache(aiReadingSessionCacheKey)
    if (cached) {
      setAIReading(cached)
      if (activateTab) {
        setActiveTab('reading')
      }
      return
    }

    setLoadingAI(true)
    try {
      const data = await getAIReading(aiReadingRequest)
      setAIReading(data)
      saveAIReadingToSessionCache(aiReadingSessionCacheKey, data)
      if (activateTab) {
        setActiveTab('reading')
      }
    } catch (error) {
      console.error('Failed to load AI reading:', error)
      const msg = error instanceof Error ? error.message : 'Unknown error'
      alert(`AI 해석 불러오기에 실패했습니다.
원인: ${msg}`)
    } finally {
      setLoadingAI(false)
    }
  }

  const handleLoadAIReading = async () => {
    if (aiReading) {
      setActiveTab('reading')
      return
    }
    await loadAIReading({ activateTab: true })
  }

  const handleDownloadPDF = async () => {
    setLoadingPDF(true)
    try {
      const blob = await getPDF(pdfRequest)
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${productType === 'life_cycle' ? 'vedic-life-cycle-report' : 'vedic-report'}-${chartRequest.year}${chartRequest.month}${chartRequest.day}.pdf`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Failed to download PDF:', error)
      const msg = error instanceof Error ? error.message : 'Unknown error'
      alert(`PDF 다운로드에 실패했습니다.
원인: ${msg}`)
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

  const isLifeCycleReport = aiReading?.product_type === 'life_cycle' || productType === 'life_cycle'
  const top3 = planetRows.slice(0, 3)
  const hourInt = Math.floor(chartRequest.hour)
  const minInt = Math.round((chartRequest.hour - hourInt) * 60)
  const readingText = aiReading?.polished_reading ?? ''
  const readingMeta = aiReading?.meta
  const displaySubjectName = aiReading?.summary?.structured_summary?.subject_name || subjectName
  const goalLabel = formatOnboardingGoal(readingMeta?.onboarding_goal || onboardingGoal)
  const heroEyebrow = isLifeCycleReport ? 'Life Cycle Report' : 'Vedic Signature'
  const heroTitle = isLifeCycleReport ? 'Vedic Life Cycle Report' : '쉽게 보는 내 성향 리포트'
  const heroDescription = isLifeCycleReport
    ? '현재 시즌, 다음 전환 시점, 그리고 지금 붙잡아야 할 기준을 한 번에 읽는 리포트입니다.'
    : '출생 차트의 핵심 성향을 쉬운 말로 먼저 훑어보는 화면입니다.'
  const readingTabLabel = isLifeCycleReport ? '리포트' : 'AI 해석'
  const readingCardTitle = isLifeCycleReport ? '인생 주기 리포트' : 'AI 해석'
  const readingButtonLabel = loadingAI
    ? isLifeCycleReport
      ? '리포트 생성 중...'
      : 'AI 해석 생성 중...'
    : aiReading
      ? isLifeCycleReport
        ? '리포트 보기'
        : 'AI 해석 보기'
      : isLifeCycleReport
        ? '리포트 생성'
        : 'AI 해석 생성'
  const pdfButtonLabel = loadingPDF
    ? 'PDF 준비 중...'
    : isLifeCycleReport
      ? '리포트 PDF 다운로드'
      : 'PDF 다운로드'

  const lifeCycleHighlights = [
    displaySubjectName ? { label: '대상', value: displaySubjectName } : null,
    goalLabel ? { label: '핵심 질문', value: goalLabel } : null,
    readingMeta?.valid_until ? { label: '리포트 유효기간', value: readingMeta.valid_until } : null,
    readingMeta?.next_mahadasha_date ? { label: '다음 큰 전환일', value: readingMeta.next_mahadasha_date } : null,
    readingMeta?.current_mahadasha_planet ? { label: '현재 마하다샤', value: readingMeta.current_mahadasha_planet } : null,
    readingMeta?.render_profile ? { label: '렌더 프로필', value: readingMeta.render_profile } : null,
  ].filter((item): item is { label: string; value: string } => Boolean(item))

  const lifeCycleBadges = [
    ...focusTokens.map((token) => `집중: ${token}`),
    ...concernTokens.map((token) => `걱정: ${token}`),
    occupationContext ? `현재 맥락: ${occupationContext}` : null,
    relationshipStatus ? `관계 상태: ${relationshipStatus}` : null,
  ].filter((item): item is string => Boolean(item))

  const summaryGuides = isLifeCycleReport
    ? [
        '1) 먼저 현재 위치와 다음 큰 전환일을 확인해 지금 시즌을 잡아두세요.',
        '2) valid_until 전까지 유지할 기준 1개만 남기고 나머지는 줄여 보세요.',
        '3) 마하다샤 단계 목록에서 반복되는 행성 톤을 비교하며 행동 기준을 점검하세요.',
      ]
    : [
        '1) 중요한 결정은 감정이 가라앉은 뒤에 해 주세요.',
        '2) 관계 속에서는 솔직함과 경계를 함께 지켜주세요.',
        '3) 루틴(수면, 식사, 이동)을 지키면 안정감이 커져요.',
      ]

  useEffect(() => {
    if (!isLifeCycleReport || !chart || loading || aiReading || autoLoadAttemptedRef.current) {
      return
    }
    autoLoadAttemptedRef.current = true
    void loadAIReading({ activateTab: true })
  }, [aiReading, chart, isLifeCycleReport, loading, loadAIReading])

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
          <p className="text-sm tracking-[0.18em] uppercase text-[#8a808a] mb-3">{heroEyebrow}</p>
          <h1 className="text-3xl md:text-4xl font-semibold text-[#2b2731]">{heroTitle}</h1>
          <p className="text-[#5f5a64] mt-3">{heroDescription}</p>
          <p className="text-[#5f5a64] mt-3">
            {chartRequest.year}.{chartRequest.month}.{chartRequest.day} {hourInt}:{String(minInt).padStart(2, '0')}
          </p>
        </div>

        {isLifeCycleReport && (
          <Card className="border-[#e5d9de] bg-white shadow-sm mb-7">
            <CardHeader>
              <CardTitle className="text-[#3a3240]">리포트 설정</CardTitle>
              <CardDescription>프론트에서 현재 소비 중인 life_cycle 계약 입력과 응답 메타입니다.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="grid gap-3 md:grid-cols-3">
                {lifeCycleHighlights.map((item) => (
                  <div key={`${item.label}-${item.value}`} className="rounded-lg border border-[#ece5ea] p-3 bg-[#fdfcfc]">
                    <p className="text-xs text-[#877b86] mb-1">{item.label}</p>
                    <p className="font-medium text-[#302a33] break-words">{item.value}</p>
                  </div>
                ))}
              </div>
              {lifeCycleBadges.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {lifeCycleBadges.map((label) => (
                    <Badge key={label} variant="secondary" className="bg-[#f5edf1] text-[#694958]">
                      {label}
                    </Badge>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}

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
            {readingButtonLabel}
          </Button>
          <Button onClick={handleDownloadPDF} disabled={loadingPDF} variant="outline" className="border-[#ccb8c2]">
            <Download className="w-4 h-4 mr-2" />
            {pdfButtonLabel}
          </Button>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3 mb-6 bg-[#f3eff2]">
            <TabsTrigger value="summary">요약</TabsTrigger>
            <TabsTrigger value="planets">행성 해석</TabsTrigger>
            <TabsTrigger value="reading" disabled={!aiReading}>{readingTabLabel}</TabsTrigger>
          </TabsList>

          <TabsContent value="summary">
            <Card>
              <CardHeader>
                <CardTitle className="text-[#362f39]">{isLifeCycleReport ? '리포트 가이드' : '삶의 가이드'}</CardTitle>
                <CardDescription>
                  {isLifeCycleReport ? '리포트를 읽는 순서와 지금 붙잡을 기준부터 짚어보세요.' : '오늘 바로 적용할 수 있는 조언을 모았어요.'}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 text-[#504a54]">
                {summaryGuides.map((item) => (
                  <p key={item}>{item}</p>
                ))}
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
                <CardTitle className="text-[#362f39]">{readingCardTitle}</CardTitle>
                <CardDescription>
                  {isLifeCycleReport
                    ? 'product_type=life_cycle 경로의 메타와 polished markdown를 그대로 보여줍니다.'
                    : '필요하면 다시 생성해서 최신 버전으로 볼 수 있어요.'}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {isLifeCycleReport && readingMeta && (
                  <div className="rounded-xl border border-[#efe4e8] bg-[#fffafc] p-4 text-sm text-[#5b5560]">
                    <div className="grid gap-3 md:grid-cols-2">
                      <p><span className="font-medium text-[#3b3340]">contract_version</span>: {readingMeta.contract_version || '-'}</p>
                      <p><span className="font-medium text-[#3b3340]">render_profile</span>: {readingMeta.render_profile || '-'}</p>
                      <p><span className="font-medium text-[#3b3340]">as_of_local</span>: {readingMeta.as_of_local || '-'}</p>
                      <p><span className="font-medium text-[#3b3340]">valid_until</span>: {readingMeta.valid_until || '-'}</p>
                    </div>
                  </div>
                )}
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
