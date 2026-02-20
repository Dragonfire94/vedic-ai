'use client'

import React from 'react'

interface Props {
  children: React.ReactNode
  fallback?: React.ReactNode
}

interface State {
  hasError: boolean
  message: string
}

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, message: '' }
  }

  static getDerivedStateFromError(error: unknown): State {
    const message =
      error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.'
    return { hasError: true, message }
  }

  override componentDidCatch(error: unknown, info: React.ErrorInfo) {
    console.error('[ErrorBoundary]', error, info)
  }

  override render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback
      return (
        <div className="min-h-screen flex items-center justify-center bg-[#f7f6f3]">
          <div className="max-w-md w-full rounded-xl border border-[#e5d9de] bg-white p-8 text-center shadow-sm">
            <p className="text-lg font-semibold text-[#2b2731] mb-2">오류가 발생했습니다</p>
            <p className="text-sm text-[#5f5a64] mb-6">{this.state.message}</p>
            <button
              className="bg-[#8d3d56] hover:bg-[#7a344a] text-white text-sm px-5 py-2 rounded-lg"
              onClick={() => window.location.reload()}
            >
              새로고침
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
