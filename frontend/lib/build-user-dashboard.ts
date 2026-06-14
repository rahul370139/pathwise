import type { Achievement, DashboardData, RecentActivityItem } from "@/types/dashboard"

/** Zeroed dashboard for a signed-in user with no activity yet. */
export function generateEmptyUserData(userRole: "student" | "professional" = "student"): DashboardData {
  return {
    progress: {
      completed_lessons: 0,
      total_lessons: 0,
      progress_percentage: 0,
      hours_spent: 0,
      current_streak: 0,
      skills_learned: 0,
      career_advancement_percentage: 0,
      current_level: 1,
      next_milestone: "Complete your first lesson",
    },
    recent_activity: [],
    achievements: [],
    saved_content: [],
    chat_history: [],
    recommendations: {
      next_lessons: [],
      career_opportunities: [],
      skill_gaps: [],
      market_trends: [],
      general_recommendations: [
        {
          id: "start-learn",
          title: "Start on the Learn page",
          description: "Pick a topic, apply your settings, and generate lessons, quizzes, and flashcards.",
          link: "/learn",
        },
        {
          id: "career-sim",
          title: "Run the Career Simulator",
          description: "Upload your resume and get a personalised 30/60/90-day upgrade plan.",
          link: "/simulator",
        },
      ],
    },
    analytics: {
      learning_timeline: [],
      skill_levels: [],
      time_distribution: [],
      career_progress: [],
    },
    user_role: userRole,
  }
}

function normalizeMastery(raw: unknown): {
  overall_score: number
  topic_scores: Record<string, number>
} {
  const payload = (raw as { mastery?: unknown })?.mastery ?? raw
  if (payload && typeof payload === "object" && "topic_scores" in (payload as object)) {
    const m = payload as { overall_score?: number; topic_scores?: Record<string, number> }
    return {
      overall_score: m.overall_score ?? 0,
      topic_scores: m.topic_scores ?? {},
    }
  }
  if (payload && typeof payload === "object" && !Array.isArray(payload)) {
    const topic_scores: Record<string, number> = {}
    for (const [key, value] of Object.entries(payload as Record<string, unknown>)) {
      if (typeof value === "number") topic_scores[key] = value
    }
    const scores = Object.values(topic_scores)
    const overall_score = scores.length
      ? Math.round((scores.reduce((a, b) => a + b, 0) / scores.length) * 100)
      : 0
    return { overall_score, topic_scores }
  }
  return { overall_score: 0, topic_scores: {} }
}

function deriveAchievements(completedLessons: number, streak: number, topicCount: number): Achievement[] {
  const earned: Achievement[] = []
  if (completedLessons >= 1) {
    earned.push({
      title: "First Steps",
      description: "Completed your first lesson",
      earned_date: new Date().toISOString().slice(0, 10),
      badge_type: "bronze",
      icon: "🎯",
    })
  }
  if (streak >= 7) {
    earned.push({
      title: "Week Warrior",
      description: "Seven days of continuous learning",
      earned_date: new Date().toISOString().slice(0, 10),
      badge_type: "gold",
      icon: "🔥",
    })
  }
  if (topicCount >= 3) {
    earned.push({
      title: "Skill Explorer",
      description: "Practised three or more topics",
      earned_date: new Date().toISOString().slice(0, 10),
      badge_type: "silver",
      icon: "⭐",
    })
  }
  return earned
}

function mapCompletedLessonsToActivity(completions: unknown[]): RecentActivityItem[] {
  return completions.slice(0, 8).map((row) => {
    const item = row as {
      lesson_id?: string | number
      completed_at?: string
      progress_percentage?: number
      title?: string
    }
    const date = item.completed_at
      ? new Date(item.completed_at).toISOString().slice(0, 10)
      : new Date().toISOString().slice(0, 10)
    return {
      type: "lesson",
      title: item.title ? String(item.title) : `Lesson ${item.lesson_id ?? ""}`.trim(),
      date,
      progress: item.progress_percentage ?? 100,
      status: "completed",
    }
  })
}

export type UserDashboardSources = {
  progressStats?: {
    total_lessons?: number
    completed_lessons?: number
    completion_rate?: number
  } | null
  completedLessons?: unknown[] | null
  masteryRaw?: unknown | null
  userRole?: { role?: string } | null
  recommendationsRaw?: { recommendations?: unknown[] } | null
  careerPlan?: {
    has_plan?: boolean
    plan?: { target_role?: string; phases?: Array<{ name?: string; goals?: string[] }> }
    saved_at?: string
  } | null
}

/** Merge live API payloads into a per-user dashboard (starts empty, never demo numbers). */
export function buildUserDashboardData(sources: UserDashboardSources): DashboardData {
  const roleRaw = (sources.userRole?.role || "student").toLowerCase()
  const userRole: "student" | "professional" =
    roleRaw.includes("prof") || roleRaw.includes("senior") ? "professional" : "student"

  const data = generateEmptyUserData(userRole)

  const totalLessons = sources.progressStats?.total_lessons ?? 0
  const completedLessons = sources.progressStats?.completed_lessons ?? 0
  const completionRate =
    sources.progressStats?.completion_rate ??
    (totalLessons > 0 ? Math.round((completedLessons / totalLessons) * 100) : 0)

  const { overall_score, topic_scores } = normalizeMastery(sources.masteryRaw)
  const topicEntries = Object.entries(topic_scores)
  const skillsLearned = topicEntries.length

  data.progress = {
    ...data.progress,
    completed_lessons: completedLessons,
    total_lessons: totalLessons,
    progress_percentage: Math.round(completionRate),
    hours_spent: Math.round(completedLessons * 0.75 * 10) / 10,
    current_streak: 0,
    skills_learned: skillsLearned,
    career_advancement_percentage: overall_score,
    current_level: completedLessons >= 10 ? 3 : completedLessons >= 3 ? 2 : 1,
    next_milestone:
      sources.careerPlan?.plan?.target_role ??
      (completedLessons === 0 ? "Complete your first lesson" : "Keep your learning streak"),
  }

  const completions = sources.completedLessons ?? []
  data.recent_activity = mapCompletedLessonsToActivity(completions)

  if (sources.careerPlan?.has_plan && sources.careerPlan.plan?.target_role) {
    data.recent_activity.unshift({
      type: "career",
      title: `Career plan: ${sources.careerPlan.plan.target_role}`,
      date: sources.careerPlan.saved_at
        ? new Date(sources.careerPlan.saved_at).toISOString().slice(0, 10)
        : new Date().toISOString().slice(0, 10),
      status: "completed",
    })

    const phases = sources.careerPlan.plan.phases ?? []
    data.analytics.career_progress = phases.slice(0, 6).map((phase, index) => ({
      milestone: phase.name ?? `Phase ${index + 1}`,
      completed: index === 0 && completedLessons > 0,
      date: sources.careerPlan?.saved_at,
      progress_percentage: index === 0 && completedLessons > 0 ? 100 : 0,
    }))
  }

  data.achievements = deriveAchievements(completedLessons, 0, skillsLearned)

  if (topicEntries.length > 0) {
    data.analytics.skill_levels = topicEntries.slice(0, 8).map(([skill, score]) => ({
      skill,
      level: Math.max(1, Math.round(score * 5)),
      max_level: 5,
      progress_percentage: Math.round(score * 100),
    }))

    data.recommendations.skill_gaps = topicEntries
      .filter(([, score]) => score < 0.7)
      .slice(0, 4)
      .map(([skill, score]) => ({
        skill,
        current_level: Math.max(1, Math.round(score * 5)),
        target_level: 4,
        priority: score < 0.4 ? "high" : "medium",
        market_demand: "High",
      }))
  }

  const recs = sources.recommendationsRaw?.recommendations
  if (Array.isArray(recs) && recs.length > 0) {
    data.recommendations.next_lessons = recs.slice(0, 5).map((rec, index) => {
      const r = rec as {
        id?: string
        title?: string
        description?: string
        duration?: string
        difficulty?: string
      }
      return {
        id: String(r.id ?? index + 1),
        title: r.title ?? "Recommended lesson",
        description: r.description ?? "",
        estimated_time: r.duration ?? "1 hour",
        difficulty: r.difficulty ?? "intermediate",
        priority: index + 1,
      }
    })
  }

  if (completedLessons > 0) {
    const today = new Date()
    data.analytics.learning_timeline = [
      {
        date: new Date(today.getTime() - 2 * 86400000).toISOString().slice(0, 10),
        lessons: Math.max(1, Math.floor(completedLessons / 2)),
        hours: Math.round(completedLessons * 0.3 * 10) / 10,
        skills_gained: Math.min(skillsLearned, 2),
      },
      {
        date: today.toISOString().slice(0, 10),
        lessons: completedLessons,
        hours: Math.round(completedLessons * 0.75 * 10) / 10,
        skills_gained: skillsLearned,
      },
    ]

    const lessonHours = Math.round(completedLessons * 0.75 * 10) / 10
    data.analytics.time_distribution = [
      { activity: "Lessons", hours: lessonHours, percentage: 100, color: "#3B82F6" },
    ]
  }

  return data
}
