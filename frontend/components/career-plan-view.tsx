"use client"

import Link from "next/link"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import {
  Brain,
  Target,
  TrendingUp,
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  Code,
  Briefcase,
  Calendar,
  BookOpen,
  Lightbulb,
  GraduationCap,
  ArrowRight,
  Rocket,
  MessageSquare,
  Layers,
} from "lucide-react"

export interface CareerPlanResponse {
  target_role: string
  interests?: string[]
  onet_match?: {
    title?: string | null
    soc_code?: string | null
    salary_low?: number | null
    salary_high?: number | null
    growth_pct?: number | null
  }
  plan: {
    skill_gap?: {
      strengths?: string[]
      transferable?: string[]
      gaps?: string[]
      blockers?: string[]
      readiness_score?: number
    }
    roadmap?: Record<
      string,
      {
        title?: string
        duration?: string
        salary_range?: string
        skills_to_acquire?: string[]
        responsibilities?: string[]
        milestones?: string[]
      }
    >
    ninety_day_plan?: Array<{ week: string; focus: string; actions?: string[]; deliverable?: string }>
    projects_to_build?: Array<{ title: string; why?: string; stack?: string[]; scope?: string; stretch_goal?: string }>
    resources?: Array<{ title: string; type: string; why?: string }>
    interview_prep?: {
      core_topics?: string[]
      behavioral_themes?: string[]
      system_design_targets?: string[]
    }
    market_insights?: {
      salary_range?: string
      growth_outlook?: string
      hot_skills?: string[]
    }
  }
}

/** Build a deep link into the Learn chatbox, pre-focused on a topic. */
export function learnHref(topic: string) {
  return `/learn?topic=${encodeURIComponent(topic)}`
}

type WeekItem = { week: string; focus: string; actions?: string[]; deliverable?: string }

const PHASE_META = [
  { label: "Days 1–30", tagline: "Foundations & quick wins" },
  { label: "Days 31–60", tagline: "Build depth & ship" },
  { label: "Days 61–90", tagline: "Prove impact & interview-ready" },
]

const PHASE_STYLES = [
  { card: "border-blue-200 dark:border-blue-900 bg-blue-50/50 dark:bg-blue-950/20", badge: "bg-blue-600", rule: "border-blue-400" },
  { card: "border-purple-200 dark:border-purple-900 bg-purple-50/50 dark:bg-purple-950/20", badge: "bg-purple-600", rule: "border-purple-400" },
  { card: "border-green-200 dark:border-green-900 bg-green-50/50 dark:bg-green-950/20", badge: "bg-green-600", rule: "border-green-400" },
]

/**
 * Group a weekly plan into 30/60/90 phases. Buckets by the week number when
 * present (1–4 → phase 1, 5–8 → phase 2, 9+ → phase 3); if the labels don't
 * carry usable numbers, falls back to splitting the list into even thirds.
 */
function groupIntoPhases(weeks: WeekItem[]): { label: string; tagline: string; weeks: WeekItem[] }[] {
  const phases: WeekItem[][] = [[], [], []]
  let usedNumbers = false
  weeks.forEach((w) => {
    const n = Number.parseInt(String(w.week).match(/\d+/)?.[0] ?? "", 10)
    if (!Number.isNaN(n)) {
      usedNumbers = true
      phases[n <= 4 ? 0 : n <= 8 ? 1 : 2].push(w)
    }
  })
  if (!usedNumbers || phases.every((p) => p.length === 0)) {
    phases[0] = []
    phases[1] = []
    phases[2] = []
    const size = Math.ceil(weeks.length / 3)
    weeks.forEach((w, i) => phases[Math.min(2, Math.floor(i / size))].push(w))
  }
  return PHASE_META.map((m, i) => ({ ...m, weeks: phases[i] }))
}

/** A topic that, when clicked, opens the Learn page focused on it. */
function LearnTopic({ topic, className = "" }: { topic: string; className?: string }) {
  return (
    <Link
      href={learnHref(topic)}
      className={`inline-flex items-center gap-1 rounded-md border border-blue-200 dark:border-blue-900 bg-blue-50 dark:bg-blue-950/30 px-2 py-0.5 text-xs text-blue-700 dark:text-blue-300 hover:bg-blue-100 dark:hover:bg-blue-900/40 transition-colors ${className}`}
      title={`Learn "${topic}" in the AI tutor`}
    >
      <BookOpen className="h-3 w-3" />
      {topic}
    </Link>
  )
}

export function CareerPlanView({ plan, userName }: { plan: CareerPlanResponse; userName?: string | null }) {
  const gap = plan.plan.skill_gap ?? {}
  const market = plan.plan.market_insights ?? {}
  const onet = plan.onet_match
  const readiness = typeof gap.readiness_score === "number" ? Math.max(0, Math.min(100, gap.readiness_score)) : null

  return (
    <div className="space-y-6">
      <Card className="border-blue-200 dark:border-blue-900">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-blue-600" />
            Your upgrade plan: {plan.target_role}
          </CardTitle>
          <CardDescription>
            {userName ? `${userName}, here's where you stand and what to do next.` : "Here's where you stand and what to do next."}
            {onet?.title ? ` Anchored to O*NET row "${onet.title}".` : ""}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Stat label="Readiness" value={readiness !== null ? `${readiness}%` : "—"} icon={<Target className="h-4 w-4" />} />
            <Stat
              label="Salary band"
              value={market.salary_range || (onet?.salary_low && onet?.salary_high ? `$${onet.salary_low.toLocaleString()} – $${onet.salary_high.toLocaleString()}` : "—")}
              icon={<Briefcase className="h-4 w-4" />}
            />
            <Stat
              label="Growth"
              value={market.growth_outlook || (onet?.growth_pct != null ? `${onet.growth_pct}%` : "—")}
              icon={<TrendingUp className="h-4 w-4" />}
            />
          </div>
          {readiness !== null && (
            <div className="mt-4">
              <Progress value={readiness} className="h-2" />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Skill gap */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Skill gap analysis
          </CardTitle>
          <CardDescription>Click a gap to learn it in the AI tutor.</CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <GapList label="Strengths" items={gap.strengths || []} tone="green" icon={<CheckCircle2 className="h-4 w-4 text-green-600" />} />
          <GapList label="Transferable" items={gap.transferable || []} tone="blue" icon={<Sparkles className="h-4 w-4 text-blue-600" />} />
          <GapList label="Gaps to close" items={gap.gaps || []} tone="orange" icon={<AlertTriangle className="h-4 w-4 text-orange-600" />} learnable />
          <GapList label="Hard blockers" items={gap.blockers || []} tone="red" icon={<AlertTriangle className="h-4 w-4 text-red-600" />} />
        </CardContent>
      </Card>

      {/* Roadmap */}
      {plan.plan.roadmap && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              3-stage roadmap
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {Object.entries(plan.plan.roadmap).map(([level, details], ri) => (
                <Card key={level} className={`border-2 ${PHASE_STYLES[ri % 3].card}`}>
                  <CardHeader className="pb-2">
                    <div className="flex items-center gap-2">
                      <span className={`flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold text-white ${PHASE_STYLES[ri % 3].badge}`}>
                        {ri + 1}
                      </span>
                      <CardTitle className="text-base">{details.title || level}</CardTitle>
                    </div>
                    <CardDescription className="text-xs">
                      {details.duration} · {details.salary_range}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {details.skills_to_acquire && details.skills_to_acquire.length > 0 && (
                      <div>
                        <p className="text-xs font-medium mb-1">Skills to acquire</p>
                        <div className="flex flex-wrap gap-1">
                          {details.skills_to_acquire.map((s, i) => (
                            <Badge key={i} variant="outline" className="text-xs">
                              {s}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                    {details.responsibilities && details.responsibilities.length > 0 && (
                      <div>
                        <p className="text-xs font-medium mb-1">Responsibilities</p>
                        <ul className="text-xs space-y-1 text-muted-foreground">
                          {details.responsibilities.slice(0, 3).map((r, i) => (
                            <li key={i}>• {r}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {details.milestones && details.milestones.length > 0 && (
                      <div>
                        <p className="text-xs font-medium mb-1">Milestones</p>
                        <ul className="text-xs space-y-1 text-muted-foreground">
                          {details.milestones.slice(0, 3).map((m, i) => (
                            <li key={i}>→ {m}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* 30 / 60 / 90-day plan, grouped into three phases */}
      {plan.plan.ninety_day_plan && plan.plan.ninety_day_plan.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Calendar className="h-5 w-5 text-blue-600" />
              Your 30 / 60 / 90-day plan
            </CardTitle>
            <CardDescription>Three phases, paced to your gaps — stick this on your calendar.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {groupIntoPhases(plan.plan.ninety_day_plan).map((phase, pi) =>
                phase.weeks.length === 0 ? null : (
                  <div
                    key={pi}
                    className={`rounded-xl border p-4 ${PHASE_STYLES[pi].card}`}
                  >
                    <div className="flex items-center gap-2 mb-3">
                      <span className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-bold text-white ${PHASE_STYLES[pi].badge}`}>
                        {pi + 1}
                      </span>
                      <div>
                        <p className="text-sm font-semibold leading-none">{phase.label}</p>
                        <p className="text-[11px] text-muted-foreground mt-0.5">{phase.tagline}</p>
                      </div>
                    </div>
                    <div className="space-y-3">
                      {phase.weeks.map((w, i) => (
                        <div key={i} className={`border-l-2 pl-3 ${PHASE_STYLES[pi].rule}`}>
                          <p className="text-xs font-semibold">
                            Week {w.week} — {w.focus}
                          </p>
                          {w.actions && w.actions.length > 0 && (
                            <ul className="mt-1 text-xs text-muted-foreground space-y-0.5">
                              {w.actions.map((a, ai) => (
                                <li key={ai}>• {a}</li>
                              ))}
                            </ul>
                          )}
                          {w.deliverable && (
                            <p className="mt-1 text-[11px]">
                              <span className="font-medium">Ship:</span> {w.deliverable}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ),
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Projects — numbered portfolio cards (matches roadmap / 90-day phase styling) */}
      {plan.plan.projects_to_build && plan.plan.projects_to_build.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Code className="h-5 w-5 text-indigo-600" />
              Projects to build
            </CardTitle>
            <CardDescription>
              Portfolio-grade work tailored to your stack — ship these to prove readiness.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {plan.plan.projects_to_build.map((p, i) => {
                const style = PHASE_STYLES[i % PHASE_STYLES.length]
                return (
                  <div
                    key={i}
                    className={`rounded-xl border-2 p-4 ${style.card} flex flex-col gap-3`}
                  >
                    <div className="flex items-start gap-3">
                      <span
                        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-bold text-white ${style.badge}`}
                      >
                        {i + 1}
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-semibold leading-snug">{p.title}</p>
                        {p.why && (
                          <p className="mt-1 text-xs text-muted-foreground leading-relaxed">{p.why}</p>
                        )}
                      </div>
                    </div>
                    {(p.scope || p.stretch_goal) && (
                      <div className={`space-y-2 border-l-2 pl-3 ${style.rule}`}>
                        {p.scope && (
                          <div>
                            <p className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                              Scope
                            </p>
                            <p className="text-xs leading-relaxed">{p.scope}</p>
                          </div>
                        )}
                        {p.stretch_goal && (
                          <div>
                            <p className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                              Stretch goal
                            </p>
                            <p className="text-xs leading-relaxed">{p.stretch_goal}</p>
                          </div>
                        )}
                      </div>
                    )}
                    {p.stack && p.stack.length > 0 && (
                      <div>
                        <p className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground mb-1.5">
                          Tech stack
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {p.stack.map((s, si) => (
                            <Badge key={si} variant="outline" className="text-[10px] bg-background/60">
                              {s}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Resources — clickable study cards */}
      {plan.plan.resources && plan.plan.resources.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-blue-600" />
              Resources
            </CardTitle>
            <CardDescription>Click any resource to study it in the Learn tutor.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {plan.plan.resources.map((r, i) => (
                <Link
                  key={i}
                  href={learnHref(r.title)}
                  title={`Learn "${r.title}" in the AI tutor`}
                  className="group rounded-xl border border-blue-200 dark:border-blue-900 bg-blue-50/40 dark:bg-blue-950/20 p-4 transition-all hover:border-blue-400 hover:shadow-md hover:bg-blue-50/70 dark:hover:bg-blue-950/40"
                >
                  <div className="flex items-start justify-between gap-2">
                    <Badge variant="outline" className="text-[10px] shrink-0 bg-background/70">
                      {r.type}
                    </Badge>
                    <ArrowRight className="h-3.5 w-3.5 text-blue-500 opacity-0 -translate-x-1 transition-all group-hover:opacity-100 group-hover:translate-x-0" />
                  </div>
                  <p className="mt-2 text-sm font-semibold text-blue-900 dark:text-blue-100 group-hover:text-blue-700 dark:group-hover:text-blue-300">
                    {r.title}
                  </p>
                  {r.why && (
                    <p className="mt-1.5 text-xs text-muted-foreground leading-relaxed line-clamp-3">{r.why}</p>
                  )}
                  <p className="mt-2 text-[10px] font-medium text-blue-600 dark:text-blue-400 flex items-center gap-1">
                    <BookOpen className="h-3 w-3" />
                    Open in Learn
                  </p>
                </Link>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Interview prep — three tone panels (matches skill-gap quadrants) */}
      {plan.plan.interview_prep &&
        ((plan.plan.interview_prep.core_topics?.length ?? 0) > 0 ||
          (plan.plan.interview_prep.behavioral_themes?.length ?? 0) > 0 ||
          (plan.plan.interview_prep.system_design_targets?.length ?? 0) > 0) && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Lightbulb className="h-5 w-5 text-amber-600" />
                Interview prep
              </CardTitle>
              <CardDescription>
                Grounded topics from your plan — click technical areas to drill in the tutor.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {plan.plan.interview_prep.core_topics && plan.plan.interview_prep.core_topics.length > 0 && (
                  <div className="rounded-xl border border-blue-200 dark:border-blue-900 bg-blue-50/50 dark:bg-blue-950/20 p-4">
                    <div className="flex items-center gap-2 text-sm font-semibold mb-3">
                      <BookOpen className="h-4 w-4 text-blue-600" />
                      Core topics
                      <span className="text-xs font-normal text-muted-foreground">
                        ({plan.plan.interview_prep.core_topics.length})
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {plan.plan.interview_prep.core_topics.map((t, i) => (
                        <LearnTopic key={i} topic={t} />
                      ))}
                    </div>
                  </div>
                )}
                {plan.plan.interview_prep.behavioral_themes &&
                  plan.plan.interview_prep.behavioral_themes.length > 0 && (
                    <div className="rounded-xl border border-purple-200 dark:border-purple-900 bg-purple-50/50 dark:bg-purple-950/20 p-4">
                      <div className="flex items-center gap-2 text-sm font-semibold mb-3">
                        <MessageSquare className="h-4 w-4 text-purple-600" />
                        Behavioral STAR
                        <span className="text-xs font-normal text-muted-foreground">
                          ({plan.plan.interview_prep.behavioral_themes.length})
                        </span>
                      </div>
                      <ul className="space-y-2">
                        {plan.plan.interview_prep.behavioral_themes.map((t, i) => (
                          <li
                            key={i}
                            className="flex items-start gap-2 text-xs text-muted-foreground leading-relaxed"
                          >
                            <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-purple-200/80 dark:bg-purple-900/50 text-[10px] font-bold text-purple-700 dark:text-purple-300">
                              {i + 1}
                            </span>
                            {t}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                {plan.plan.interview_prep.system_design_targets &&
                  plan.plan.interview_prep.system_design_targets.length > 0 && (
                    <div className="rounded-xl border border-green-200 dark:border-green-900 bg-green-50/50 dark:bg-green-950/20 p-4">
                      <div className="flex items-center gap-2 text-sm font-semibold mb-3">
                        <Layers className="h-4 w-4 text-green-600" />
                        System design
                        <span className="text-xs font-normal text-muted-foreground">
                          ({plan.plan.interview_prep.system_design_targets.length})
                        </span>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {plan.plan.interview_prep.system_design_targets.map((t, i) => (
                          <LearnTopic key={i} topic={t} />
                        ))}
                      </div>
                    </div>
                  )}
              </div>
            </CardContent>
          </Card>
        )}

      {/* Hot skills — market pulse strip */}
      {market.hot_skills && market.hot_skills.length > 0 && (
        <Card className="border-amber-200 dark:border-amber-900/60">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <GraduationCap className="h-5 w-5 text-amber-600" />
              Hot skills in this market
            </CardTitle>
            <CardDescription>
              In-demand skills for {plan.target_role} — tap any to start learning.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-xl border border-amber-200 dark:border-amber-900/60 bg-gradient-to-br from-amber-50/80 via-orange-50/40 to-yellow-50/60 dark:from-amber-950/25 dark:via-orange-950/15 dark:to-yellow-950/10 p-5">
              <div className="flex items-center gap-2 mb-3">
                <Rocket className="h-4 w-4 text-amber-600" />
                <p className="text-xs font-semibold text-amber-800 dark:text-amber-300">
                  Trending for your target role
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
                {market.hot_skills.map((s, i) => (
                  <LearnTopic
                    key={i}
                    topic={s}
                    className="text-xs px-2.5 py-1 border-amber-200 dark:border-amber-800 bg-white/70 dark:bg-amber-950/40 text-amber-900 dark:text-amber-200 hover:bg-amber-100 dark:hover:bg-amber-900/50"
                  />
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function Stat({ label, value, icon }: { label: string; value: string; icon: React.ReactNode }) {
  return (
    <div className="rounded-lg border p-3">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        {icon}
        {label}
      </div>
      <p className="mt-1 text-lg font-semibold">{value}</p>
    </div>
  )
}

function GapList({
  label,
  items,
  tone,
  icon,
  learnable = false,
}: {
  label: string
  items: string[]
  tone: "green" | "blue" | "orange" | "red"
  icon: React.ReactNode
  learnable?: boolean
}) {
  const toneClass: Record<typeof tone, string> = {
    green: "border-green-200 bg-green-50 dark:bg-green-950/20",
    blue: "border-blue-200 bg-blue-50 dark:bg-blue-950/20",
    orange: "border-orange-200 bg-orange-50 dark:bg-orange-950/20",
    red: "border-red-200 bg-red-50 dark:bg-red-950/20",
  }
  return (
    <div className={`rounded-lg border p-3 ${toneClass[tone]}`}>
      <div className="flex items-center gap-2 text-sm font-medium">
        {icon}
        {label}
        <span className="text-xs text-muted-foreground">({items.length})</span>
      </div>
      {items.length === 0 ? (
        <p className="mt-2 text-xs text-muted-foreground">— none —</p>
      ) : learnable ? (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {items.map((it, i) => (
            <LearnTopic key={i} topic={it} />
          ))}
        </div>
      ) : (
        <ul className="mt-2 space-y-1 text-sm">
          {items.map((it, i) => (
            <li key={i}>• {it}</li>
          ))}
        </ul>
      )}
    </div>
  )
}
