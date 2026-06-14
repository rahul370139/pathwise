"use client"

import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { simulatorAPI } from "@/lib/api"
import { ShieldCheck } from "lucide-react"

interface EvalReport {
  available?: boolean
  message?: string
  scenario_count?: number
  metrics?: {
    groundedness?: number | null
    answer_relevance?: number | null
    refusal_correctness?: number | null
    latency_p95_sec?: number | null
  }
}

function pct(v?: number | null): string {
  return v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`
}

export function EvalCard() {
  const [data, setData] = useState<EvalReport | null>(null)

  useEffect(() => {
    simulatorAPI.evalReport().then(setData).catch(() => setData(null))
  }, [])

  if (!data) return null

  const m = data.metrics || {}
  return (
    <Card className="p-4">
      <div className="flex items-center gap-2 mb-3">
        <ShieldCheck className="h-4 w-4 text-green-500" />
        <h3 className="font-semibold text-sm">Reliability & Eval</h3>
        {data.scenario_count ? (
          <span className="text-xs text-muted-foreground ml-auto">{data.scenario_count} scenarios</span>
        ) : null}
      </div>
      {data.available === false ? (
        <p className="text-xs text-muted-foreground">{data.message}</p>
      ) : (
        <div className="grid grid-cols-2 gap-2 text-sm">
          <Metric label="Groundedness" value={pct(m.groundedness)} />
          <Metric label="Answer relevance" value={pct(m.answer_relevance)} />
          <Metric label="Refusal correctness" value={pct(m.refusal_correctness)} />
          <Metric label="Latency p95" value={m.latency_p95_sec != null ? `${m.latency_p95_sec}s` : "—"} />
        </div>
      )}
    </Card>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-xs text-muted-foreground">{label}</span>
      <Badge variant="secondary" className="w-fit mt-0.5">{value}</Badge>
    </div>
  )
}
