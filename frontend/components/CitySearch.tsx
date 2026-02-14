'use client'

import { useEffect, useRef, useState } from 'react'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Loader2, MapPin } from 'lucide-react'

type Provider = 'google' | 'nominatim'

type CitySelection = {
  city: string
  lat: number
  lon: number
}

interface CitySearchProps {
  onCitySelect: (data: CitySelection) => void
  defaultValue?: string
}

export function CitySearch({ onCitySelect, defaultValue = '' }: CitySearchProps) {
  const hasGoogleApiKey = Boolean(process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY?.trim())

  const [provider, setProvider] = useState<Provider>(hasGoogleApiKey ? 'google' : 'nominatim')
  const [inputValue, setInputValue] = useState(defaultValue)
  const [predictions, setPredictions] = useState<any[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [showDropdown, setShowDropdown] = useState(false)

  const inputRef = useRef<HTMLInputElement>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)
  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const autocompleteService = useRef<any>(null)
  const geocoder = useRef<any>(null)

  const initializeGoogleServices = () => {
    if (!window.google?.maps?.places?.AutocompleteService || !window.google?.maps?.Geocoder) {
      return false
    }

    autocompleteService.current = new window.google.maps.places.AutocompleteService()
    geocoder.current = new window.google.maps.Geocoder()
    return true
  }

  useEffect(() => {
    if (!hasGoogleApiKey) {
      setProvider('nominatim')
      return
    }

    if (window.google) {
      if (!initializeGoogleServices()) {
        setProvider('nominatim')
      }
      return
    }

    const scriptId = 'google-maps-places-script'
    let script = document.getElementById(scriptId) as HTMLScriptElement | null

    if (!script) {
      script = document.createElement('script')
      script.id = scriptId
      script.src = `https://maps.googleapis.com/maps/api/js?key=${process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY}&libraries=places&language=ko`
      script.async = true
      script.defer = true
      document.head.appendChild(script)
    }

    script.onload = () => {
      if (!initializeGoogleServices()) {
        setProvider('nominatim')
      }
    }

    script.onerror = () => {
      setProvider('nominatim')
    }
  }, [hasGoogleApiKey])

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

  const searchGoogleCities = (query: string) => {
    if (!autocompleteService.current) {
      setProvider('nominatim')
      searchNominatimCities(query)
      return
    }

    setIsLoading(true)

    autocompleteService.current.getPlacePredictions(
      {
        input: query,
        types: ['(cities)'],
        language: 'ko',
      },
      (googlePredictions: any, status: any) => {
        setIsLoading(false)

        if (status === window.google.maps.places.PlacesServiceStatus.OK && googlePredictions) {
          setPredictions(googlePredictions)
          setShowDropdown(true)
          return
        }

        setPredictions([])
        setShowDropdown(false)
      }
    )
  }

  const searchNominatimCities = async (query: string) => {
    setIsLoading(true)

    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?${
          new URLSearchParams({
            q: query,
            format: 'json',
            addressdetails: '1',
            limit: '5',
            'accept-language': 'ko',
          }).toString()
        }`
      )

      const data = await response.json()
      const cities = Array.isArray(data)
        ? data.filter((item: any) => {
            const type = item.type || item.addresstype
            return (
              type === 'city' ||
              type === 'town' ||
              type === 'village' ||
              type === 'administrative' ||
              item.class === 'place'
            )
          })
        : []

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

    if (value.length < 2) {
      setPredictions([])
      setShowDropdown(false)
      return
    }

    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current)
    }

    debounceTimer.current = setTimeout(() => {
      if (provider === 'google') {
        searchGoogleCities(value)
      } else {
        searchNominatimCities(value)
      }
    }, 250)
  }

  const handleGoogleSelectCity = (prediction: any) => {
    setInputValue(prediction.description)
    setShowDropdown(false)
    setIsLoading(true)

    if (!geocoder.current) {
      setIsLoading(false)
      setProvider('nominatim')
      return
    }

    geocoder.current.geocode({ placeId: prediction.place_id }, (results: any, status: any) => {
      setIsLoading(false)

      if (status === 'OK' && results[0]) {
        const location = results[0].geometry.location
        onCitySelect({
          city: prediction.description,
          lat: location.lat(),
          lon: location.lng(),
        })
      }
    })
  }

  const handleNominatimSelectCity = (city: any) => {
    const displayName = String(city.display_name || '').split(',').slice(0, 2).join(',').trim()

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

      {showDropdown && predictions.length > 0 && (
        <div
          ref={dropdownRef}
          className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto"
        >
          {predictions.map((item: any, index: number) => {
            const key = provider === 'google' ? item.place_id : `${item.place_id}-${index}`
            const mainText =
              provider === 'google'
                ? item.structured_formatting?.main_text || item.description
                : String(item.display_name || '').split(',')[0]
            const secondaryText =
              provider === 'google'
                ? item.structured_formatting?.secondary_text || ''
                : String(item.display_name || '').split(',').slice(1, 3).join(',').trim()

            return (
              <button
                key={key}
                onClick={() =>
                  provider === 'google' ? handleGoogleSelectCity(item) : handleNominatimSelectCity(item)
                }
                className="w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors flex items-start gap-3 border-b border-gray-100 last:border-b-0"
              >
                <MapPin className="w-4 h-4 text-purple-600 flex-shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-gray-900 truncate">{mainText}</div>
                  <div className="text-sm text-gray-500 truncate">{secondaryText}</div>
                </div>
              </button>
            )
          })}
        </div>
      )}

      {showDropdown && predictions.length === 0 && !isLoading && inputValue.length >= 2 && (
        <div
          ref={dropdownRef}
          className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg p-4 text-center text-sm text-gray-500"
        >
          검색 결과가 없습니다.
        </div>
      )}
    </div>
  )
}

declare global {
  interface Window {
    google: any
  }
}
