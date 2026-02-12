'use client'

import { useState, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Sparkles,
  Download,
  Home,
  Star,
  Moon,
  Compass,
  Info,
  TrendingUp,
  ChevronDown,
  ChevronUp
} from 'lucide-react'
import { getChart, getAIReading, getPDF } from '@/lib/api'
import { ASCENDANT_TRAITS, PLANET_NAMES_KR, DIGNITY_LABELS_KR } from '@/lib/utils'

export default function ChartPage() {
  const router = useRouter()
  const searchParams = useSearchParams()

  const [chart, setChart] = useState<any>(null)
  const [aiReading, setAIReading] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [loadingAI, setLoadingAI] = useState(false)
  const [loadingPDF, setLoadingPDF] = useState(false)
  const [expandedPlanet, setExpandedPlanet] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('chart')

  // Parse query parameters
  const birthData = {
    year: parseInt(searchParams.get('year') || '1994'),
    month: parseInt(searchParams.get('month') || '12'),
    day: parseInt(searchParams.get('day') || '18'),
    hour: parseFloat(searchParams.get('hour') || '23.75'),
    lat: parseFloat(searchParams.get('lat') || '37.5665'),
    lon: parseFloat(searchParams.get('lon') || '126.978'),
    gender: searchParams.get('gender') || 'male',
    house_system: searchParams.get('house_system') || 'W',  // Vedic uses Whole Sign
  }

  // Load chart data
  useEffect(() => {
    const loadChart = async () => {
      try {
        const data = await getChart({
          ...birthData,
          include_nodes: true,
          include_d9: true,
        })
        setChart(data)
      } catch (error) {
        console.error('Failed to load chart:', error)
        alert('차트를 불러오는데 실패했습니다.')
      } finally {
        setLoading(false)
      }
    }
    loadChart()
  }, [])

  // Load AI Reading
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
      })
      setAIReading(data)
      setActiveTab('reading')
    } catch (error) {
      console.error('Failed to load AI reading:', error)
      alert('AI 리딩을 불러오는데 실패했습니다.')
    } finally {
      setLoadingAI(false)
    }
  }

  // Download PDF
  const handleDownloadPDF = async () => {
    setLoadingPDF(true)
    try {
      const data = await getPDF({
        ...birthData,
        language: 'ko',
        include_nodes: true,
        include_d9: true,
      })

      if (data.pdf_base64) {
        // Convert base64 to blob and download
        const byteCharacters = atob(data.pdf_base64)
        const byteNumbers = new Array(byteCharacters.length)
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i)
        }
        const byteArray = new Uint8Array(byteNumbers)
        const blob = new Blob([byteArray], { type: 'application/pdf' })
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `vedic-chart-${birthData.year}${birthData.month}${birthData.day}.pdf`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        window.URL.revokeObjectURL(url)
      }
    } catch (error) {
      console.error('Failed to download PDF:', error)
      alert('PDF 다운로드에 실패했습니다.')
    } finally {
      setLoadingPDF(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Sparkles className="w-12 h-12 text-purple-600 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">차트를 계산하고 있습니다...</p>
        </div>
      </div>
    )
  }

  if (!chart) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600">차트를 불러올 수 없습니다.</p>
          <Button onClick={() => router.push('/')} className="mt-4">
            처음으로
          </Button>
        </div>
      </div>
    )
  }

  const ascendant = chart.houses?.ascendant
  const ascendantInfo = ASCENDANT_TRAITS[ascendant?.rasi?.name] || {
    name_kr: ascendant?.rasi?.name_kr || '알 수 없음',
    emoji: '✨',
    keywords: [],
    preview: '',
  }

  // Format hour for display
  const hours = Math.floor(birthData.hour)
  const minutes = Math.round((birthData.hour - hours) * 60)

  return (
    <div className="min-h-screen bg-gradient-to-b from-purple-50 to-white">
      <div className="container mx-auto px-4 py-16">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Sparkles className="w-8 h-8 text-purple-600" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
              베딕 출생 차트
            </h1>
          </div>
          <p className="text-xl text-gray-700 mb-2">
            {birthData.year}년 {birthData.month}월 {birthData.day}일 {hours}시 {minutes}분
          </p>
          <p className="text-gray-600">
            {ascendantInfo.emoji} {ascendantInfo.name_kr} 상승궁
          </p>
        </div>

        {/* Action Buttons */}
        <div className="max-w-4xl mx-auto mb-8 flex gap-4 justify-center">
          <Button
            onClick={handleLoadAIReading}
            disabled={loadingAI}
            size="lg"
            className="gap-2"
          >
            <Sparkles className="w-5 h-5" />
            {loadingAI ? 'AI 분석 중...' : aiReading ? 'AI 리딩 보기' : 'AI 리딩 생성'}
          </Button>
          <Button
            onClick={handleDownloadPDF}
            disabled={loadingPDF}
            variant="outline"
            size="lg"
            className="gap-2"
          >
            <Download className="w-5 h-5" />
            {loadingPDF ? '다운로드 중...' : 'PDF 다운로드'}
          </Button>
        </div>

        {/* Tabs */}
        <div className="max-w-6xl mx-auto">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3 mb-8">
              <TabsTrigger value="chart" className="gap-2">
                <Star className="w-4 h-4" />
                차트 상세
              </TabsTrigger>
              <TabsTrigger value="houses" className="gap-2">
                <Compass className="w-4 h-4" />
                하우스
              </TabsTrigger>
              <TabsTrigger value="reading" className="gap-2" disabled={!aiReading}>
                <Sparkles className="w-4 h-4" />
                AI 리딩
              </TabsTrigger>
            </TabsList>

            {/* Chart Tab */}
            <TabsContent value="chart" className="space-y-6">
              {/* Ascendant Card */}
              <Card className="border-2 border-purple-200 bg-gradient-to-br from-purple-50 to-white">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Compass className="w-6 h-6 text-purple-600" />
                    상승궁 (Ascendant)
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center gap-4">
                      <div className="text-5xl">{ascendantInfo.emoji}</div>
                      <div>
                        <div className="text-2xl font-bold">
                          {ascendantInfo.name_kr}
                        </div>
                        <div className="text-gray-600">
                          {ascendant?.rasi?.name} • {ascendant?.longitude?.toFixed(2)}°
                        </div>
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {ascendantInfo.keywords.map((keyword: string, i: number) => (
                        <Badge key={i} variant="secondary">
                          {keyword}
                        </Badge>
                      ))}
                    </div>
                    <p className="text-gray-700 p-4 bg-purple-50 rounded-lg">
                      {ascendantInfo.preview}
                    </p>
                  </div>
                </CardContent>
              </Card>

              {/* Planets */}
              <div>
                <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                  <Star className="w-6 h-6 text-purple-600" />
                  행성 배치
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(chart.planets || {}).map(([planetName, planetData]: [string, any]) => {
                    const isExpanded = expandedPlanet === planetName
                    const planetKr = PLANET_NAMES_KR[planetName] || planetName
                    const dignityKr = DIGNITY_LABELS_KR[planetData.features?.dignity] || planetData.features?.dignity

                    return (
                      <Card
                        key={planetName}
                        className="cursor-pointer hover:border-purple-300 transition-colors"
                        onClick={() => setExpandedPlanet(isExpanded ? null : planetName)}
                      >
                        <CardHeader className="pb-3">
                          <div className="flex items-center justify-between">
                            <div>
                              <CardTitle className="text-lg flex items-center gap-2">
                                {planetName === 'Sun' && '☀️'}
                                {planetName === 'Moon' && '🌙'}
                                {planetName === 'Mars' && '♂️'}
                                {planetName === 'Mercury' && '☿️'}
                                {planetName === 'Jupiter' && '♃'}
                                {planetName === 'Venus' && '♀️'}
                                {planetName === 'Saturn' && '♄'}
                                {planetName === 'Rahu' && '🐉'}
                                {planetName === 'Ketu' && '🐲'}
                                {planetKr}
                                {planetData.features?.retrograde && (
                                  <Badge variant="destructive" className="text-xs">R</Badge>
                                )}
                                {planetData.features?.combust && (
                                  <Badge variant="outline" className="text-xs">연소</Badge>
                                )}
                              </CardTitle>
                              <CardDescription>
                                {planetData.rasi?.name_kr} • {planetData.house}하우스
                              </CardDescription>
                            </div>
                            <div>
                              {isExpanded ? (
                                <ChevronUp className="w-5 h-5 text-gray-400" />
                              ) : (
                                <ChevronDown className="w-5 h-5 text-gray-400" />
                              )}
                            </div>
                          </div>
                        </CardHeader>

                        {isExpanded && (
                          <CardContent>
                            <div className="space-y-3">
                              <div className="grid grid-cols-2 gap-3">
                                <div className="p-2 bg-gray-50 rounded">
                                  <div className="text-xs text-gray-600">라시</div>
                                  <div className="font-medium">
                                    {planetData.rasi?.name_kr} ({planetData.rasi?.deg_in_sign?.toFixed(2)}°)
                                  </div>
                                </div>
                                <div className="p-2 bg-gray-50 rounded">
                                  <div className="text-xs text-gray-600">하우스</div>
                                  <div className="font-medium">{planetData.house}</div>
                                </div>
                              </div>

                              <div className="p-2 bg-gray-50 rounded">
                                <div className="text-xs text-gray-600">나크샤트라</div>
                                <div className="font-medium">
                                  {planetData.nakshatra?.name} (파다 {planetData.nakshatra?.pada})
                                </div>
                              </div>

                              <div className="p-2 bg-gray-50 rounded">
                                <div className="text-xs text-gray-600">존엄 (Dignity)</div>
                                <div className="font-medium">
                                  <Badge variant={
                                    planetData.features?.dignity === 'Exalted' ? 'default' :
                                    planetData.features?.dignity === 'Own' ? 'secondary' :
                                    planetData.features?.dignity === 'Debilitated' ? 'destructive' :
                                    'outline'
                                  }>
                                    {dignityKr}
                                  </Badge>
                                </div>
                              </div>

                              {chart.d9?.planets?.[planetName] && (
                                <div className="p-2 bg-purple-50 rounded border border-purple-200">
                                  <div className="text-xs text-purple-700 font-medium">나밤샤 (D9)</div>
                                  <div className="text-sm">
                                    {chart.d9.planets[planetName].rasi_kr}
                                  </div>
                                </div>
                              )}
                            </div>
                          </CardContent>
                        )}
                      </Card>
                    )
                  })}
                </div>
              </div>

              {/* Yogas */}
              {chart.features?.yogas?.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="w-6 h-6 text-purple-600" />
                      요가 (Yogas)
                    </CardTitle>
                    <CardDescription>
                      발견된 행성 조합
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {chart.features.yogas.map((yoga: any, i: number) => (
                        <div key={i} className="p-4 bg-green-50 border border-green-200 rounded-lg">
                          <div className="font-medium text-green-900">{yoga.name}</div>
                          <div className="text-sm text-green-700 mt-1">{yoga.note}</div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            {/* Houses Tab */}
            <TabsContent value="houses" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Compass className="w-6 h-6 text-purple-600" />
                    하우스 시스템: {chart.input?.house_system === 'P' ? 'Placidus' : 'Whole Sign'}
                  </CardTitle>
                  <CardDescription>
                    12개 하우스의 커스프와 라시
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Array.from({ length: 12 }, (_, i) => i + 1).map((houseNum) => {
                      const houseKey = `house_${houseNum}`
                      const house = chart.houses?.[houseKey]

                      return (
                        <div key={houseNum} className="p-4 border rounded-lg hover:border-purple-300 transition-colors">
                          <div className="font-bold text-lg mb-2">
                            {houseNum}하우스
                          </div>
                          <div className="text-sm text-gray-600">
                            커스프: {house?.cusp_longitude?.toFixed(2)}°
                          </div>
                          <div className="text-sm font-medium mt-1">
                            {house?.rasi}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* AI Reading Tab */}
            <TabsContent value="reading" className="space-y-6">
              {aiReading && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Sparkles className="w-6 h-6 text-purple-600" />
                      AI 베딕 리딩
                    </CardTitle>
                    <CardDescription>
                      {aiReading.cached && '(캐시됨) '}
                      모델: {aiReading.model || 'gpt-4o-mini'}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="prose max-w-none">
                      <div className="whitespace-pre-wrap text-gray-700">
                        {aiReading.reading || '리딩을 생성할 수 없습니다.'}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>
          </Tabs>
        </div>

        {/* Info Box */}
        <div className="max-w-6xl mx-auto mt-12">
          <div className="p-6 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex gap-3">
              <Info className="w-5 h-5 text-blue-600 mt-0.5" />
              <div>
                <h3 className="font-medium text-blue-900 mb-2">베딕 점성술이란?</h3>
                <p className="text-sm text-blue-800">
                  베딕 점성술은 인도의 고대 점성술 체계로, 항성계(Sidereal) 황도를 사용합니다.
                  서양 점성술과 약 23도 차이가 나며, 나크샤트라, 다샤 시스템 등 독특한 기법을 사용합니다.
                  이 차트는 라히리 아야남샤(Lahiri Ayanamsa)를 기준으로 계산되었습니다.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <div className="max-w-6xl mx-auto mt-8 text-center">
          <Button
            variant="outline"
            onClick={() => router.push('/')}
            className="gap-2"
          >
            <Home className="w-4 h-4" />
            처음으로 돌아가기
          </Button>
        </div>
      </div>
    </div>
  )
}
