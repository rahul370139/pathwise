"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { simulatorAPI, careerAPI } from "@/lib/api"
import { useAuth } from "@/components/auth-provider"
import { AgentThinkingPanel } from "@/components/agent-thinking-panel"
import { ReadinessReport } from "@/components/readiness-report"
import { EvalCard } from "@/components/eval-card"
import { learnHref } from "@/components/career-plan-view"
import {
  Upload,
  Send,
  Sparkles,
  Loader2,
  AlertTriangle,
  SkipForward,
  MessageSquarePlus,
  Brain,
  ArrowRight,
  Briefcase,
  FileText,
  MessagesSquare,
  CheckCircle2,
  RotateCcw,
} from "lucide-react"

type Step = "upload" | "interview" | "report"
type EntryMode = "jd" | "role" | "quiz"

interface Question {
  competency: string
  question: string
  citations?: { doc?: string; section?: string }[]
  grounded?: boolean
  index: number
  total: number
}

const POPULAR_ROLES = [
  "Data Scientist",
  "Software Engineer",
  "Machine Learning Engineer",
  "Product Manager",
  "DevOps Engineer",
  "Backend Developer",
]

const INTEREST_SUGGESTIONS = [
  "Machine Learning",
  "Cloud / DevOps",
  "Generative AI",
  "Data Engineering",
  "Backend Systems",
  "Frontend / UX",
]

// "Not sure?" assessment — surfaces matching careers (moved from the old Career page).
const ASSESSMENT_QUESTIONS = [
  "I enjoy working with technology and solving technical problems",
  "I prefer working independently rather than in large teams",
  "I like analyzing data to find patterns and insights",
  "I enjoy creating visual designs and user interfaces",
  "I'm comfortable presenting ideas to groups of people",
  "I prefer structured, predictable work environments",
  "I enjoy mentoring and helping others learn",
  "I like working on long-term strategic projects",
  "I'm energized by fast-paced, changing environments",
  "I enjoy learning new technologies and staying updated with industry trends",
]
const LIKERT = [
  { value: "1", label: "Strongly Disagree" },
  { value: "2", label: "Disagree" },
  { value: "3", label: "Neutral" },
  { value: "4", label: "Agree" },
  { value: "5", label: "Strongly Agree" },
]

const SAMPLE_RESUME = `Priya Sharma — Software Engineer
Summary: 3 years building web apps. Comfortable with Python, JavaScript, React, and REST APIs.
Experience:
- Software Engineer, Acme Corp (2022–present): Built internal dashboards in React, wrote FastAPI services, set up CI on GitHub Actions.
- Junior Developer, StartHub (2021–2022): Maintained a Django app, wrote unit tests, fixed bugs.
Skills: Python, JavaScript, React, FastAPI, SQL, Git, REST APIs, unit testing
Education: B.Tech Computer Science, 2021
Projects: Personal finance tracker (React + FastAPI), Markdown note app.`

const SAMPLE_JD = `Backend Engineer (Distributed Systems)
We are hiring a backend engineer to design and operate high-throughput services.
Responsibilities: design scalable microservices, own data modeling for Postgres at scale,
build event-driven pipelines (Kafka), and ensure reliability via observability and on-call.
Requirements: strong Python or Go, deep understanding of distributed systems, message queues,
database indexing and query optimization, caching strategies, system design, and production
debugging. Nice to have: Kubernetes, AWS, and experience with SLOs/SLIs.`

export default function SimulatorPage() {
  const { user } = useAuth()
  const [step, setStep] = useState<Step>("upload")
  const [entryMode, setEntryMode] = useState<EntryMode>("jd")

  const [file, setFile] = useState<File | null>(null)
  const [jdText, setJdText] = useState("")
  const [targetRole, setTargetRole] = useState("")
  const [interests, setInterests] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Quiz state
  const [quizAnswers, setQuizAnswers] = useState<Record<number, string>>({})
  const [matches, setMatches] = useState<any[] | null>(null)
  const [matching, setMatching] = useState(false)

  const [sessionId, setSessionId] = useState<string>("")
  const [plan, setPlan] = useState<any>(null)
  const [question, setQuestion] = useState<Question | null>(null)
  const [answer, setAnswer] = useState("")
  const [comment, setComment] = useState("")
  const [showComment, setShowComment] = useState(false)
  const [lastFeedback, setLastFeedback] = useState<any>(null)
  const [report, setReport] = useState<any>(null)
  const [confirmRestart, setConfirmRestart] = useState(false)

  const toggleInterest = (i: string) =>
    setInterests((prev) => (prev.includes(i) ? prev.filter((x) => x !== i) : [...prev, i]))

  // Persist simulator progress so switching to Learn and back doesn't wipe the
  // plan/report. (The resume file isn't restored — the server session holds it.)
  useEffect(() => {
    try {
      const raw = localStorage.getItem("pathwise_simulator_state")
      if (raw) {
        const s = JSON.parse(raw)
        if (s && (s.report || s.plan)) {
          if (s.step) setStep(s.step)
          if (s.sessionId) setSessionId(s.sessionId)
          if (s.plan) setPlan(s.plan)
          if (s.question) setQuestion(s.question)
          if (s.report) setReport(s.report)
          if (s.entryMode) setEntryMode(s.entryMode)
        }
      }
    } catch {}
  }, [])

  useEffect(() => {
    try {
      if (!plan && !report) return
      localStorage.setItem(
        "pathwise_simulator_state",
        JSON.stringify({ step, sessionId, plan, question, report, entryMode }),
      )
    } catch {}
  }, [step, sessionId, plan, question, report, entryMode])

  function resetSimulator() {
    setStep("upload")
    setSessionId("")
    setPlan(null)
    setQuestion(null)
    setReport(null)
    setAnswer("")
    setComment("")
    setShowComment(false)
    setLastFeedback(null)
    setConfirmRestart(false)
    setFile(null)
    setJdText("")
    setTargetRole("")
    setInterests([])
    setMatches(null)
    setError(null)
    try {
      localStorage.removeItem("pathwise_simulator_state")
    } catch {}
  }

  function applyStartResponse(res: any) {
    if (res.refused) {
      setError(res.reason || "This request is out of scope.")
      return false
    }
    setSessionId(res.session_id)
    setPlan(res.plan)
    setQuestion(res.first_question)
    setStep("interview")
    return true
  }

  async function handleStart() {
    setError(null)
    // Validate per entry mode
    if (entryMode === "jd" && (!file || jdText.trim().length < 40)) {
      setError("Upload a resume PDF and paste the full job posting (40+ chars).")
      return
    }
    if (entryMode === "role" && (!file || !targetRole.trim())) {
      setError("Upload a resume PDF and choose a target role.")
      return
    }
    setLoading(true)
    try {
      const res =
        entryMode === "jd"
          ? await simulatorAPI.start(file as File, jdText)
          : await simulatorAPI.startWithRole(file, targetRole, interests)
      applyStartResponse(res)
    } catch (e: any) {
      setError(e?.message || "Failed to start the simulator.")
    } finally {
      setLoading(false)
    }
  }

  async function handleSample() {
    setError(null)
    setLoading(true)
    try {
      const res = await simulatorAPI.startWithText(SAMPLE_RESUME, SAMPLE_JD)
      applyStartResponse(res)
    } catch (e: any) {
      setError(e?.message || "Failed to start the sample simulation.")
    } finally {
      setLoading(false)
    }
  }

  async function runQuiz() {
    setMatching(true)
    setError(null)
    try {
      const answersArray = ASSESSMENT_QUESTIONS.map((_, idx) => ({
        question_id: idx + 1,
        rating: Number.parseInt(quizAnswers[idx + 1] || "3"),
      }))
      const result: any = await careerAPI.matchCareer({
        user_id: user?.id || "anonymous-user",
        answers: answersArray,
      })
      setMatches(result.career_matches || result.results || [])
    } catch {
      setError("Couldn't reach the career matcher. Pick a target role manually instead.")
    } finally {
      setMatching(false)
    }
  }

  function chooseMatch(title: string) {
    setTargetRole(title)
    setEntryMode("role")
    setMatches(null)
  }

  async function advance(skipped: boolean) {
    if (!sessionId) return
    if (!skipped && !answer.trim()) return
    setLoading(true)
    try {
      const res = await simulatorAPI.answer(sessionId, answer, {
        skipped,
        comment: comment.trim() || undefined,
      })
      setLastFeedback(res.skipped ? { skipped: true } : res.score)
      setAnswer("")
      setComment("")
      setShowComment(false)
      if (res.finished || !res.next) {
        const rep = await simulatorAPI.report(sessionId)
        setReport(rep.report)
        setStep("report")
      } else {
        setQuestion(res.next)
      }
    } catch (e: any) {
      setError(e?.message || "Failed to submit.")
    } finally {
      setLoading(false)
    }
  }

  const quizProgress = (Object.keys(quizAnswers).length / ASSESSMENT_QUESTIONS.length) * 100

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="flex items-center gap-2 mb-2">
        <Sparkles className="h-6 w-6 text-blue-500" />
        <h1 className="text-2xl font-bold">Career Simulator</h1>
      </div>
      <p className="text-muted-foreground mb-4">
        Upload your resume and pick a target. A team of grounded agents runs an adaptive interview,
        closes your gaps with micro-lessons, and produces a full role-readiness plan — skill gap,
        roadmap, 30/60/90-day plan, projects, resources, and interview prep.
      </p>

      <div className="flex flex-wrap items-center justify-between gap-3 mb-6">
        <Stepper step={step} />
        {step !== "upload" &&
          (confirmRestart ? (
            <div className="flex items-center gap-2 rounded-lg border bg-muted/40 px-3 py-1.5">
              <span className="text-xs text-muted-foreground">Discard this session?</span>
              <Button variant="ghost" size="sm" className="h-7 px-2" onClick={() => setConfirmRestart(false)}>
                Cancel
              </Button>
              <Button variant="outline" size="sm" className="h-7 px-2" onClick={resetSimulator}>
                Yes, start over
              </Button>
            </div>
          ) : (
            <Button
              variant="ghost"
              size="sm"
              className="h-8 text-muted-foreground hover:text-foreground"
              disabled={loading}
              onClick={() => setConfirmRestart(true)}
            >
              <RotateCcw className="h-3.5 w-3.5 mr-1.5" />
              Start over
            </Button>
          ))}
      </div>

      {error && (
        <Card className="p-3 mb-4 border-red-300 bg-red-50 dark:bg-red-950/30">
          <p className="text-sm text-red-700 dark:text-red-300 flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" /> {error}
          </p>
        </Card>
      )}

      {step === "upload" && (
        <div className="space-y-4 max-w-2xl">
          {/* Resume */}
          <Card className="p-6 space-y-4">
            <div>
              <label className="text-sm font-medium mb-1 block">Resume (PDF)</label>
              <label className="flex items-center gap-2 border-2 border-dashed rounded-lg p-4 cursor-pointer hover:bg-accent">
                <Upload className="h-4 w-4" />
                <span className="text-sm">{file ? file.name : "Choose a PDF resume…"}</span>
                <input
                  type="file"
                  accept="application/pdf"
                  className="hidden"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                />
              </label>
            </div>

            {/* Entry mode tabs */}
            <div className="flex flex-wrap gap-2">
              <ModeTab active={entryMode === "jd"} onClick={() => setEntryMode("jd")} icon={<Briefcase className="h-3.5 w-3.5" />}>
                Paste job posting
              </ModeTab>
              <ModeTab active={entryMode === "role"} onClick={() => setEntryMode("role")} icon={<Sparkles className="h-3.5 w-3.5" />}>
                Pick a target role
              </ModeTab>
              <ModeTab active={entryMode === "quiz"} onClick={() => setEntryMode("quiz")} icon={<Brain className="h-3.5 w-3.5" />}>
                Not sure? Take a quiz
              </ModeTab>
            </div>

            {entryMode === "jd" && (
              <div>
                <Label className="text-sm">Target job posting</Label>
                <textarea
                  className="w-full min-h-[160px] rounded-md border bg-background p-3 text-sm"
                  placeholder="Paste the full job description here…"
                  value={jdText}
                  onChange={(e) => setJdText(e.target.value)}
                />
              </div>
            )}

            {entryMode === "role" && (
              <div className="space-y-3">
                <div>
                  <Label htmlFor="target-role" className="text-sm">Target role</Label>
                  <Input
                    id="target-role"
                    placeholder="e.g. Machine Learning Engineer"
                    value={targetRole}
                    onChange={(e) => setTargetRole(e.target.value)}
                  />
                  <div className="mt-2 flex flex-wrap gap-2">
                    {POPULAR_ROLES.map((r) => (
                      <Button key={r} variant="outline" size="sm" className="h-7 text-xs" onClick={() => setTargetRole(r)}>
                        {r}
                      </Button>
                    ))}
                  </div>
                </div>
                <div>
                  <Label className="text-sm">Interests (optional)</Label>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {INTEREST_SUGGESTIONS.map((i) => (
                      <Badge
                        key={i}
                        variant={interests.includes(i) ? "default" : "outline"}
                        className="cursor-pointer"
                        onClick={() => toggleInterest(i)}
                      >
                        {i}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {entryMode === "quiz" && (
              <div className="space-y-4">
                {!matches ? (
                  <>
                    <p className="text-sm text-muted-foreground">
                      Answer 10 quick questions and we'll match you against O*NET careers. Pick one to
                      simulate.
                    </p>
                    <Progress value={quizProgress} className="h-1.5" />
                    <div className="space-y-3 max-h-[320px] overflow-y-auto pr-1">
                      {ASSESSMENT_QUESTIONS.map((q, idx) => (
                        <div key={idx} className="rounded-md border p-3">
                          <p className="text-sm mb-2">{idx + 1}. {q}</p>
                          <RadioGroup
                            value={quizAnswers[idx + 1] || ""}
                            onValueChange={(v) => setQuizAnswers((p) => ({ ...p, [idx + 1]: v }))}
                            className="flex flex-wrap gap-3"
                          >
                            {LIKERT.map((o) => (
                              <div key={o.value} className="flex items-center gap-1.5">
                                <RadioGroupItem value={o.value} id={`q${idx}-${o.value}`} />
                                <Label htmlFor={`q${idx}-${o.value}`} className="text-xs cursor-pointer">
                                  {o.label}
                                </Label>
                              </div>
                            ))}
                          </RadioGroup>
                        </div>
                      ))}
                    </div>
                    <Button onClick={runQuiz} disabled={matching || Object.keys(quizAnswers).length < ASSESSMENT_QUESTIONS.length}>
                      {matching ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <Brain className="h-4 w-4 mr-1" />}
                      See my matches
                    </Button>
                  </>
                ) : (
                  <div className="space-y-2">
                    <p className="text-sm font-medium">Pick a role to simulate:</p>
                    {matches.length === 0 && (
                      <p className="text-sm text-muted-foreground">No matches — switch to "Pick a target role".</p>
                    )}
                    {matches.map((m, i) => (
                      <button
                        key={i}
                        onClick={() => chooseMatch(m.career || m.title)}
                        className="w-full text-left rounded-md border p-3 hover:bg-accent transition-colors flex items-center justify-between"
                      >
                        <div>
                          <p className="text-sm font-medium">{m.career || m.title}</p>
                          {m.description && <p className="text-xs text-muted-foreground line-clamp-1">{m.description}</p>}
                        </div>
                        <div className="flex items-center gap-2">
                          {(m.match_score || m.similarity) != null && (
                            <Badge variant="secondary">{Math.round((m.match_score || m.similarity) * 100)}%</Badge>
                          )}
                          <ArrowRight className="h-4 w-4 text-muted-foreground" />
                        </div>
                      </button>
                    ))}
                    <Button variant="ghost" size="sm" onClick={() => setMatches(null)}>
                      Retake quiz
                    </Button>
                  </div>
                )}
              </div>
            )}

            {entryMode !== "quiz" && (
              <div className="flex flex-wrap items-center gap-2">
                <Button onClick={handleStart} disabled={loading}>
                  {loading ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <Sparkles className="h-4 w-4 mr-1" />}
                  Start simulation
                </Button>
                <Button variant="outline" onClick={handleSample} disabled={loading}>
                  Try a sample
                </Button>
                <span className="text-xs text-muted-foreground">No PDF handy? Run a built-in resume + job posting.</span>
              </div>
            )}
          </Card>
        </div>
      )}

      {step === "interview" && (
        <div className="grid md:grid-cols-3 gap-4">
          <div className="md:col-span-2 space-y-4">
            {plan && (
              <Card className="p-4">
                <div className="flex items-center justify-between">
                  <h2 className="font-semibold">{plan.target_role}</h2>
                  {typeof plan.readiness_estimate === "number" && (
                    <Badge variant="secondary">Est. readiness {plan.readiness_estimate}%</Badge>
                  )}
                </div>
                {plan.summary && <p className="text-sm text-muted-foreground mt-1">{plan.summary}</p>}
              </Card>
            )}

            {question && (
              <Card className="p-5">
                <div className="flex items-center gap-2 mb-2">
                  <Badge>{question.competency}</Badge>
                  <span className="text-xs text-muted-foreground">
                    Question {question.index + 1} of {question.total}
                  </span>
                  {question.grounded === false && (
                    <Badge variant="outline" className="text-amber-600">low grounding</Badge>
                  )}
                </div>
                <Progress value={(question.index / Math.max(1, question.total)) * 100} className="h-1 mb-3" />
                <p className="font-medium mb-3">{question.question}</p>
                {question.citations && question.citations.length > 0 && (
                  <div className="mb-3">
                    <p className="text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
                      Grounded in (click to learn)
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {question.citations.slice(0, 4).map((c, i) => (
                        <Link
                          key={i}
                          href={learnHref(c.section || c.doc || "interview prep")}
                          className="inline-flex items-center rounded-md border border-blue-200 dark:border-blue-900 bg-blue-50 dark:bg-blue-950/30 px-2 py-0.5 text-[10px] text-blue-700 dark:text-blue-300 hover:bg-blue-100 dark:hover:bg-blue-900/40 transition-colors"
                          title="Open this topic in the Learn tutor"
                        >
                          {(c.doc || "kb").replace(/\.md$/, "")}
                          {c.section ? ` · ${c.section}` : ""}
                        </Link>
                      ))}
                    </div>
                  </div>
                )}
                <textarea
                  className="w-full min-h-[120px] rounded-md border bg-background p-3 text-sm"
                  placeholder="Type your answer… (or skip if you're not sure)"
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                />

                {showComment ? (
                  <textarea
                    className="w-full mt-2 min-h-[60px] rounded-md border bg-background p-3 text-sm"
                    placeholder="Optional note to yourself or the panel…"
                    value={comment}
                    onChange={(e) => setComment(e.target.value)}
                  />
                ) : (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="mt-2 text-muted-foreground"
                    onClick={() => setShowComment(true)}
                  >
                    <MessageSquarePlus className="h-4 w-4 mr-1" /> Add a comment (optional)
                  </Button>
                )}

                <div className="flex items-center gap-2 mt-3">
                  {lastFeedback && !lastFeedback.skipped && (
                    <div className="text-xs text-muted-foreground">
                      Last: <span className="font-medium">{lastFeedback.score}/100</span> — {lastFeedback.feedback}
                    </div>
                  )}
                  {lastFeedback?.skipped && <div className="text-xs text-muted-foreground">Last question skipped.</div>}
                  <div className="ml-auto flex items-center gap-2">
                    <Button variant="outline" onClick={() => advance(true)} disabled={loading}>
                      <SkipForward className="h-4 w-4 mr-1" /> Skip
                    </Button>
                    <Button onClick={() => advance(false)} disabled={loading || !answer.trim()}>
                      {loading ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <Send className="h-4 w-4 mr-1" />}
                      Submit
                    </Button>
                  </div>
                </div>
              </Card>
            )}
          </div>

          <div className="md:col-span-1">
            {sessionId && <AgentThinkingPanel sessionId={sessionId} />}
          </div>
        </div>
      )}

      {step === "report" && report && (
        <div className="grid md:grid-cols-3 gap-4">
          <div className="md:col-span-2 space-y-3">
            <ReadinessReport report={report} userName={user?.email?.split("@")[0]} />
          </div>
          <div className="md:col-span-1 space-y-4">
            {sessionId && <AgentThinkingPanel sessionId={sessionId} />}
            <EvalCard />
          </div>
        </div>
      )}
    </div>
  )
}

function Stepper({ step, className = "" }: { step: Step; className?: string }) {
  const steps: { key: Step; label: string; icon: React.ReactNode }[] = [
    { key: "upload", label: "Setup", icon: <FileText className="h-4 w-4" /> },
    { key: "interview", label: "Interview", icon: <MessagesSquare className="h-4 w-4" /> },
    { key: "report", label: "Readiness plan", icon: <CheckCircle2 className="h-4 w-4" /> },
  ]
  const activeIdx = steps.findIndex((s) => s.key === step)
  return (
    <div className={`flex items-center ${className}`}>
      {steps.map((s, i) => {
        const done = i < activeIdx
        const active = i === activeIdx
        return (
          <div key={s.key} className="flex items-center">
            <div
              className={`flex items-center gap-2 rounded-full px-3 py-1.5 text-sm font-medium transition-colors ${
                active
                  ? "bg-blue-600 text-white"
                  : done
                    ? "bg-blue-100 text-blue-700 dark:bg-blue-950/50 dark:text-blue-300"
                    : "bg-muted text-muted-foreground"
              }`}
            >
              {done ? <CheckCircle2 className="h-4 w-4" /> : s.icon}
              <span className="hidden sm:inline">{s.label}</span>
            </div>
            {i < steps.length - 1 && (
              <div className={`h-0.5 w-6 sm:w-12 mx-1 ${i < activeIdx ? "bg-blue-500" : "bg-border"}`} />
            )}
          </div>
        )
      })}
    </div>
  )
}

function ModeTab({
  active,
  onClick,
  icon,
  children,
}: {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
        active
          ? "border-blue-500 bg-blue-50 text-blue-700 dark:bg-blue-950/40 dark:text-blue-300"
          : "border-border text-muted-foreground hover:bg-accent"
      }`}
    >
      {icon}
      {children}
    </button>
  )
}
