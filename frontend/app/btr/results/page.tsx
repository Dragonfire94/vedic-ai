'use client'

import { useState, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Sparkles, ChevronDown, ChevronUp, Target, TrendingUp } from 'lucide-react'
import { ASCENDANT_TRAITS } from '@/lib/utils'

export default function BTRResultsPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  
  const [result, setResult] = useState<any>(null)
  const [expandedCard, setExpandedCard] = useState<number | null>(0)
  const [selectedAscendant, setSelectedAscendant] = useState<string | null>(null)

  useEffect(() => {
    const resultParam = searchParams.get('result')
    if (resultParam) {
      try {
        const parsed = JSON.parse(resultParam)
        setResult(parsed)
      } catch (error) {
        console.error('Failed to parse result:', error)
      }
    }
  }, [searchParams])

  const handleSelectAscendant = (ascendant: string) => {
    setSelectedAscendant(ascendant)
    
    // 차트 페이지로 이동 (선택된 상승궁으로)
    const params = new URLSearchParams({
      year: searchParams.get('year') || '',
      month: searchParams.get('month') || '',
      day: searchParams.get('day') || '',
      lat: searchParams.get('lat') || '',
      lon: searchParams.get('lon') || '',
      ascendant: ascendant,
      btr_result: 'true',
    })
    router.push(`/chart?${params}`)
  }

  if (!result || !result.candidates) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600">결과를 불러오는 중...</p>
        </div>
      </div>
    )
  }

  const candidates = result.candidates || []
  const topCandidate = candidates[0]

  return (
    <div className="min-h-screen bg-gradient-to-b from-purple-50 to-white">
      <div className="container mx-auto px-4 py-16">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Sparkles className="w-8 h-8 text-purple-600" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
              생시 분석 결과
            </h1>
          </div>
          <p className="text-xl text-gray-700 mb-2">
            가능성 높은 출생 시간대
          </p>
          <p className="text-gray-600">
            총 {result.total_events || 0}개의 이벤트를 분석했습니다
          </p>
        </div>

        {/* Top Result Summary */}
        <Card className="max-w-3xl mx-auto mb-8 border-2 border-purple-200 bg-gradient-to-br from-purple-50 to-white">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-2xl mb-2">
                  가장 가능성 높은 시간대
                </CardTitle>
                <CardDescription>
                  {topCandidate?.time_range || '알 수 없음'}
                </CardDescription>
              </div>
              <div className="text-right">
                <div className="text-3xl font-bold text-purple-600">
                  {topCandidate?.confidence || 0}%
                </div>
                <div className="text-sm text-gray-600">신뢰도</div>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div className="p-4 bg-white rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {topCandidate?.ascendant || '-'}
                </div>
                <div className="text-sm text-gray-600 mt-1">상승궁</div>
              </div>
              <div className="p-4 bg-white rounded-lg">
                <div className="text-2xl font-bold text-green-600">
                  {topCandidate?.matched_events || 0}
                </div>
                <div className="text-sm text-gray-600 mt-1">매칭 이벤트</div>
              </div>
              <div className="p-4 bg-white rounded-lg">
                <div className="text-2xl font-bold text-blue-600">
                  {topCandidate?.score?.toFixed(1) || '0.0'}
                </div>
                <div className="text-sm text-gray-600 mt-1">점수</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* All Candidates */}
        <div className="max-w-3xl mx-auto space-y-4">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
            <Target className="w-6 h-6 text-purple-600" />
            후보 시간대 (Top 3)
          </h2>

          {candidates.slice(0, 3).map((candidate: any, index: number) => {
            const isExpanded = expandedCard === index
            const ascendantInfo = ASCENDANT_TRAITS[candidate.ascendant] || {
              name_kr: candidate.ascendant,
              emoji: '✨',
              keywords: [],
              preview: '',
            }

            return (
              <Card 
                key={index}
                className={`cursor-pointer transition-all ${
                  index === 0 ? 'border-2 border-purple-300' : ''
                }`}
              >
                <CardHeader 
                  onClick={() => setExpandedCard(isExpanded ? null : index)}
                  className="cursor-pointer hover:bg-gray-50"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="text-3xl">{index + 1}</div>
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          <span className="text-2xl">{ascendantInfo.emoji}</span>
                          {candidate.time_range}
                          {index === 0 && (
                            <Badge className="bg-purple-600">추천</Badge>
                          )}
                        </CardTitle>
                        <CardDescription className="mt-1">
                          {ascendantInfo.name_kr} 상승궁 • 신뢰도 {candidate.confidence}%
                        </CardDescription>
                      </div>
                    </div>
                    <div className="text-right">
                      {isExpanded ? (
                        <ChevronUp className="w-5 h-5 text-gray-400" />
                      ) : (
                        <ChevronDown className="w-5 h-5 text-gray-400" />
                      )}
                    </div>
                  </div>
                  
                  {/* Progress bar */}
                  <div className="mt-4">
                    <Progress value={candidate.confidence || 0} className="h-2" />
                  </div>
                </CardHeader>

                {isExpanded && (
                  <CardContent>
                    {/* Stats */}
                    <div className="grid grid-cols-2 gap-4 mb-6">
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <div className="text-sm text-gray-600 mb-1">매칭 이벤트</div>
                        <div className="text-xl font-bold">
                          {candidate.matched_events} / {result.total_events}
                        </div>
                      </div>
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <div className="text-sm text-gray-600 mb-1">분석 점수</div>
                        <div className="text-xl font-bold">
                          {candidate.score?.toFixed(1) || '0.0'}
                        </div>
                      </div>
                    </div>

                    {/* Keywords */}
                    <div className="mb-6">
                      <div className="text-sm font-medium text-gray-700 mb-2">
                        핵심 성향
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {ascendantInfo.keywords.map((keyword: string, i: number) => (
                          <Badge key={i} variant="outline" className="text-sm">
                            {keyword}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    {/* Preview */}
                    <div className="mb-6 p-4 bg-purple-50 rounded-lg">
                      <div className="text-sm text-gray-700">
                        {ascendantInfo.preview}
                      </div>
                    </div>

                    {/* Action Button */}
                    <Button 
                      onClick={() => handleSelectAscendant(candidate.ascendant)}
                      className="w-full"
                      variant={index === 0 ? 'default' : 'outline'}
                    >
                      <TrendingUp className="w-4 h-4 mr-2" />
                      이 상승궁으로 상세 보기
                    </Button>
                  </CardContent>
                )}
              </Card>
            )
          })}
        </div>

        {/* Info */}
        <div className="max-w-3xl mx-auto mt-12 text-center">
          <div className="p-6 bg-blue-50 border border-blue-200 rounded-lg">
            <h3 className="font-medium text-blue-900 mb-2">💡 결과 해석 방법</h3>
            <p className="text-sm text-blue-800">
              신뢰도는 입력하신 이벤트들이 해당 시간대의 다샤(Dasha)와 얼마나 잘 맞는지를 나타냅니다.
              80% 이상이면 매우 높은 확률, 60-79%는 높은 확률, 40-59%는 중간 확률로 해석할 수 있습니다.
            </p>
          </div>
        </div>

        {/* Navigation */}
        <div className="max-w-3xl mx-auto mt-8 text-center">
          <Button
            variant="outline"
            onClick={() => router.push('/')}
          >
            처음으로 돌아가기
          </Button>
        </div>
      </div>
    </div>
  )
}
