'use client'

import { useState, useEffect, useRef } from 'react'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { MapPin, Loader2 } from 'lucide-react'

interface CitySearchProps {
  onCitySelect: (data: { city: string; lat: number; lon: number }) => void
  defaultValue?: string
}

export function CitySearch({ onCitySelect, defaultValue = '' }: CitySearchProps) {
  const [inputValue, setInputValue] = useState(defaultValue)
  const [predictions, setPredictions] = useState<any[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [showDropdown, setShowDropdown] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)
  const debounceTimer = useRef<NodeJS.Timeout>()

  // Click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(event.target as Node)
      ) {
        setShowDropdown(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const searchCities = async (query: string) => {
    if (query.length < 2) {
      setPredictions([])
      setShowDropdown(false)
      return
    }

    setIsLoading(true)

    try {
      // Nominatim API (OpenStreetMap - 무료!)
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?` +
        new URLSearchParams({
          q: query,
          format: 'json',
          addressdetails: '1',
          limit: '5',
          'accept-language': 'ko',
        }),
        {
          headers: {
            'User-Agent': 'VedicAI/1.0', // Nominatim requires User-Agent
          },
        }
      )

      const data = await response.json()

      // 도시만 필터링
      const cities = data.filter((item: any) => {
        const type = item.type || item.addresstype
        return (
          type === 'city' ||
          type === 'town' ||
          type === 'village' ||
          type === 'administrative' ||
          item.class === 'place'
        )
      })

      setPredictions(cities)
      setShowDropdown(cities.length > 0)
    } catch (error) {
      console.error('City search error:', error)
      setPredictions([])
      setShowDropdown(false)
    } finally {
      setIsLoading(false)
    }
  }

  const handleInputChange = (value: string) => {
    setInputValue(value)

    // Debounce search
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current)
    }

    debounceTimer.current = setTimeout(() => {
      searchCities(value)
    }, 300) // 300ms 대기
  }

  const handleSelectCity = (city: any) => {
    const displayName = city.display_name.split(',').slice(0, 2).join(',')
    
    setInputValue(displayName)
    setShowDropdown(false)

    onCitySelect({
      city: displayName,
      lat: parseFloat(city.lat),
      lon: parseFloat(city.lon),
    })
  }

  return (
    <div className="relative">
      <Label htmlFor="city" className="flex items-center gap-2 mb-2">
        <MapPin className="w-4 h-4" />
        출생 도시
      </Label>
      
      <div className="relative">
        <Input
          ref={inputRef}
          id="city"
          type="text"
          placeholder="서울, Tokyo, New York..."
          value={inputValue}
          onChange={(e) => handleInputChange(e.target.value)}
          onFocus={() => {
            if (predictions.length > 0) {
              setShowDropdown(true)
            }
          }}
          className="pr-10"
        />
        
        {isLoading && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2">
            <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
          </div>
        )}
      </div>

      {/* Dropdown */}
      {showDropdown && predictions.length > 0 && (
        <div
          ref={dropdownRef}
          className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto"
        >
          {predictions.map((city, index) => {
            const parts = city.display_name.split(',')
            const mainText = parts[0]
            const secondaryText = parts.slice(1, 3).join(',')

            return (
              <button
                key={`${city.place_id}-${index}`}
                onClick={() => handleSelectCity(city)}
                className="w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors flex items-start gap-3 border-b border-gray-100 last:border-b-0"
              >
                <MapPin className="w-4 h-4 text-purple-600 flex-shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-gray-900 truncate">
                    {mainText}
                  </div>
                  <div className="text-sm text-gray-500 truncate">
                    {secondaryText}
                  </div>
                </div>
              </button>
            )
          })}
        </div>
      )}

      {/* No results */}
      {showDropdown && predictions.length === 0 && !isLoading && inputValue.length >= 2 && (
        <div
          ref={dropdownRef}
          className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg p-4 text-center text-sm text-gray-500"
        >
          검색 결과가 없습니다
        </div>
      )}
    </div>
  )
}
