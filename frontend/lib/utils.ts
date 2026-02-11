import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Ascendant traits for each zodiac sign
export const ASCENDANT_TRAITS: Record<string, {
  name_kr: string
  emoji: string
  keywords: string[]
  preview: string
}> = {
  "Aries": {
    name_kr: "양자리",
    emoji: "🐏",
    keywords: ["열정적", "리더십", "독립적", "직설적"],
    preview: "양자리 상승궁은 강한 추진력과 리더십을 가진 사람입니다. 목표를 향해 직진하며, 새로운 시작을 두려워하지 않습니다."
  },
  "Taurus": {
    name_kr: "황소자리",
    emoji: "🐂",
    keywords: ["안정적", "인내심", "실용적", "감각적"],
    preview: "황소자리 상승궁은 안정과 물질적 풍요를 중요하게 여깁니다. 인내심이 강하고 실용적인 접근을 선호합니다."
  },
  "Gemini": {
    name_kr: "쌍둥이자리",
    emoji: "👯",
    keywords: ["호기심", "의사소통", "다재다능", "민첩함"],
    preview: "쌍둥이자리 상승궁은 뛰어난 커뮤니케이션 능력과 지적 호기심을 가졌습니다. 다양한 분야에 관심이 많습니다."
  },
  "Cancer": {
    name_kr: "게자리",
    emoji: "🦀",
    keywords: ["감성적", "보호본능", "직관적", "가정적"],
    preview: "게자리 상승궁은 깊은 감수성과 보호 본능을 지녔습니다. 가족과 친밀한 관계를 매우 중요하게 생각합니다."
  },
  "Leo": {
    name_kr: "사자자리",
    emoji: "🦁",
    keywords: ["자신감", "창의적", "관대함", "카리스마"],
    preview: "사자자리 상승궁은 타고난 카리스마와 자신감을 가졌습니다. 창의적이며 주목받는 것을 즐깁니다."
  },
  "Virgo": {
    name_kr: "처녀자리",
    emoji: "👸",
    keywords: ["분석적", "완벽주의", "실용적", "섬세함"],
    preview: "처녀자리 상승궁은 뛰어난 분석력과 세심함을 가졌습니다. 완벽을 추구하며 실용적인 해결책을 찾습니다."
  },
  "Libra": {
    name_kr: "천칭자리",
    emoji: "⚖️",
    keywords: ["조화", "외교적", "예술적", "공정함"],
    preview: "천칭자리 상승궁은 조화와 균형을 중시합니다. 외교적 능력이 뛰어나고 예술적 감각이 있습니다."
  },
  "Scorpio": {
    name_kr: "전갈자리",
    emoji: "🦂",
    keywords: ["강렬함", "통찰력", "변화", "집중력"],
    preview: "전갈자리 상승궁은 강렬한 에너지와 깊은 통찰력을 가졌습니다. 변화와 재생의 힘을 가지고 있습니다."
  },
  "Sagittarius": {
    name_kr: "궁수자리",
    emoji: "🏹",
    keywords: ["자유로움", "철학적", "낙천적", "모험적"],
    preview: "궁수자리 상승궁은 자유를 사랑하고 진리를 추구합니다. 낙천적이며 모험을 즐깁니다."
  },
  "Capricorn": {
    name_kr: "염소자리",
    emoji: "🐐",
    keywords: ["목표지향", "책임감", "인내", "현실적"],
    preview: "염소자리 상승궁은 강한 목표의식과 책임감을 가졌습니다. 인내심 있게 장기적 목표를 향해 나아갑니다."
  },
  "Aquarius": {
    name_kr: "물병자리",
    emoji: "🏺",
    keywords: ["혁신적", "독창적", "인도주의", "독립적"],
    preview: "물병자리 상승궁은 독창적이고 혁신적인 사고를 합니다. 인도주의적 가치를 중시하고 독립적입니다."
  },
  "Pisces": {
    name_kr: "물고기자리",
    emoji: "🐟",
    keywords: ["직관적", "공감능력", "영적", "창의적"],
    preview: "물고기자리 상승궁은 뛰어난 직관력과 공감 능력을 가졌습니다. 영적이고 창의적인 재능이 있습니다."
  }
}

// Planet names in Korean
export const PLANET_NAMES_KR: Record<string, string> = {
  "Sun": "태양",
  "Moon": "달",
  "Mars": "화성",
  "Mercury": "수성",
  "Jupiter": "목성",
  "Venus": "금성",
  "Saturn": "토성",
  "Rahu": "라후",
  "Ketu": "케투"
}

// Dignity labels in Korean
export const DIGNITY_LABELS_KR: Record<string, string> = {
  "Own": "본거지",
  "Exalted": "고양",
  "Debilitated": "약화",
  "Neutral": "중립",
  "Shadow": "그림자"
}
