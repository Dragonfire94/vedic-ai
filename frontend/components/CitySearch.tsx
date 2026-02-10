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

  // Google Places Autocomplete Service
  const autocompleteService = useRef<any>(null)
  const geocoder = useRef<any>(null)

  useEffect(() => {
    // Load Google Maps API
    if (typeof window !== 'undefined' && !window.google) {
      const script = document.createElement('script')
      script.src = `https://maps.googleapis.com/maps/api/js?key=${process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY}&libraries=places&language=ko`
      script.async = true
      script.defer = true
      document.head.appendChild(script)

      script.onload = () => {
        initializeServices()
      }
    } else if (window.google) {
      initializeServices()
    }
  }, [])

  const initializeServices = () => {
    if (window.google) {
      autocompleteService.current = new window.google.maps.places.AutocompleteService()
      geocoder.current = new window.google.maps.Geocoder()
    }
  }

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

  const handleInputChange = (value: string) => {
    setInputValue(value)

    if (value.length < 2) {
      setPredictions([])
      setShowDropdown(false)
      return
    }

    if (!autocompleteService.current) {
      return
    }

    setIsLoading(true)

    // Google Places Autocomplete
    autocompleteService.current.getPlacePredictions(
      {
        input: value,
        types: ['(cities)'], // 도시만
        language: 'ko',
      },
      (predictions: any, status: any) => {
        setIsLoading(false)

        if (status === window.google.maps.places.PlacesServiceStatus.OK && predictions) {
          setPredictions(predictions)
          setShowDropdown(true)
        } else {
          setPredictions([])
          setShowDropdown(false)
        }
      }
    )
  }

  const handleSelectCity = async (prediction: any) => {
    setInputValue(prediction.description)
    setShowDropdown(false)
    setIsLoading(true)

    // Geocode to get lat/lon
    if (!geocoder.current) {
      setIsLoading(false)
      return
    }

    geocoder.current.geocode(
      { placeId: prediction.place_id },
      (results: any, status: any) => {
        setIsLoading(false)

        if (status === 'OK' && results[0]) {
          const location = results[0].geometry.location
          
          onCitySelect({
            city: prediction.description,
            lat: location.lat(),
            lon: location.lng(),
          })
        }
      }
    )
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
          {predictions.map((prediction) => (
            <button
              key={prediction.place_id}
              onClick={() => handleSelectCity(prediction)}
              className="w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors flex items-start gap-3 border-b border-gray-100 last:border-b-0"
            >
              <MapPin className="w-4 h-4 text-purple-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1 min-w-0">
                <div className="font-medium text-gray-900 truncate">
                  {prediction.structured_formatting.main_text}
                </div>
                <div className="text-sm text-gray-500 truncate">
                  {prediction.structured_formatting.secondary_text}
                </div>
              </div>
            </button>
          ))}
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

// TypeScript types for window.google
declare global {
  interface Window {
    google: any
  }
}
