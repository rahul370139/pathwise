"use client"

import { useEffect, useRef, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { simulatorAPI } from "@/lib/api"
import { Brain, Search, MessageSquare, CheckCircle2, BookOpen, FileText, ShieldAlert, Loader2 } from "lucide-react"

export interface AgentEvent {
  ts: number
  agent: string
  status: string
  message: string
  data?: {
    citations?: { doc?: string; section?: string; source_uri?: string; score?: number }[]
    grounded?: boolean
    score?: number
    supported?: boolean
    [k: string]: any
  }
}

const AGENT_META: Record<string, { label: string; icon: any; color: string }> = {
  planner: { label: "Planner", icon: Brain, color: "text-purple-500" },
  retrieval: { label: "Retrieval", icon: Search, color: "text-blue-500" },
  interviewer: { label: "Interviewer", icon: MessageSquare, color: "text-cyan-500" },
  scorer: { label: "Scorer", icon: CheckCircle2, color: "text-green-500" },
  remediation: { label: "Remediation", icon: BookOpen, color: "text-amber-500" },
  report: { label: "Report", icon: FileText, color: "text-indigo-500" },
  safety: { label: "Safety", icon: ShieldAlert, color: "text-red-500" },
  system: { label: "System", icon: Loader2, color: "text-muted-foreground" },
}

export function AgentThinkingPanel({ sessionId }: { sessionId: string }) {
  const [events, setEvents] = useState<AgentEvent[]>([])
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!sessionId) return
    const es = new EventSource(simulatorAPI.streamUrl(sessionId))
    es.onmessage = (e) => {
      try {
        const ev = JSON.parse(e.data) as AgentEvent
        if (ev.status === "close") {
          es.close()
          return
        }
        setEvents((prev) => [...prev, ev])
      } catch {
        /* ignore keep-alive frames */
      }
    }
    es.onerror = () => es.close()
    return () => es.close()
  }, [sessionId])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
  }, [events])

  return (
    <Card className="p-4 h-full">
      <div className="flex items-center gap-2 mb-3">
        <Brain className="h-4 w-4 text-purple-500" />
        <h3 className="font-semibold text-sm">Agent Thinking</h3>
        <span className="text-xs text-muted-foreground ml-auto">{events.length} steps</span>
      </div>
      <div ref={scrollRef} className="space-y-2 max-h-[420px] overflow-y-auto pr-1">
        {events.length === 0 && (
          <p className="text-xs text-muted-foreground">Waiting for the agents to start…</p>
        )}
        {events.map((ev, i) => {
          const meta = AGENT_META[ev.agent] || AGENT_META.system
          const Icon = meta.icon
          const running = ev.status === "running"
          return (
            <div key={i} className="flex gap-2 text-xs">
              <Icon className={`h-3.5 w-3.5 mt-0.5 shrink-0 ${meta.color} ${running ? "animate-pulse" : ""}`} />
              <div className="flex-1">
                <span className="font-medium">{meta.label}</span>
                <span className="text-muted-foreground"> — {ev.message || ev.status}</span>
                {typeof ev.data?.score === "number" && (
                  <Badge variant="secondary" className="ml-2">{ev.data.score}/100</Badge>
                )}
                {ev.data?.grounded === false && (
                  <Badge variant="outline" className="ml-2 text-amber-600">low grounding</Badge>
                )}
                {ev.data?.citations && ev.data.citations.length > 0 && (
                  <div className="mt-1 flex flex-wrap gap-1">
                    {ev.data.citations.slice(0, 4).map((c, j) => (
                      <Badge key={j} variant="outline" className="text-[10px] font-normal">
                        {(c.doc || "kb").replace(/\.md$/, "")}
                        {c.section ? ` · ${c.section}` : ""}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </Card>
  )
}
