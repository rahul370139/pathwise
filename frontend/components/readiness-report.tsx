"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Download, Target, MessageSquare } from "lucide-react"
import { CareerPlanView, type CareerPlanResponse } from "@/components/career-plan-view"

interface Gap {
  competency: string
  score: number
  skipped?: boolean
}

interface ReadinessReportProps {
  report: {
    overall_readiness?: number
    answered?: number
    skipped?: number
    top_gaps?: Gap[]
    comments?: { competency: string; comment: string }[]
    career_plan?: CareerPlanResponse
  }
  userName?: string | null
}

export function ReadinessReport({ report, userName }: ReadinessReportProps) {
  const readiness = report.overall_readiness ?? 0

  return (
    <div className="space-y-4 print:space-y-2" id="readiness-report">
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Target className="h-5 w-5 text-blue-600" />
            <h2 className="text-xl font-bold">Readiness Report</h2>
          </div>
          <Button size="sm" variant="outline" onClick={() => window.print()} className="print:hidden">
            <Download className="h-4 w-4 mr-1" /> Save as PDF
          </Button>
        </div>

        <div className="flex items-center gap-4 mb-2">
          <div className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            {readiness}%
          </div>
          <div className="flex-1">
            <p className="text-sm text-muted-foreground mb-1">
              Role readiness across {report.answered ?? 0} answered competencies
              {report.skipped ? ` · ${report.skipped} skipped` : ""}
            </p>
            <Progress value={readiness} className="h-2" />
          </div>
        </div>
      </Card>

      {report.top_gaps && report.top_gaps.length > 0 && (
        <Card className="p-6">
          <h3 className="font-semibold mb-3">Top gaps to close</h3>
          <div className="space-y-2">
            {report.top_gaps.map((g, i) => (
              <div key={i} className="flex items-center gap-3">
                <span className="text-sm flex-1">
                  {g.competency}
                  {g.skipped && (
                    <Badge variant="outline" className="ml-2 text-[10px] text-amber-600">
                      skipped
                    </Badge>
                  )}
                </span>
                <Progress value={g.score} className="h-2 w-32" />
                <span className="text-xs text-muted-foreground w-10 text-right">{g.score}%</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {report.comments && report.comments.length > 0 && (
        <Card className="p-6">
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <MessageSquare className="h-4 w-4" /> Your notes
          </h3>
          <ul className="space-y-2 text-sm">
            {report.comments.map((c, i) => (
              <li key={i}>
                <span className="font-medium">{c.competency}:</span>{" "}
                <span className="text-muted-foreground">{c.comment}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {/* Full grounded career plan: skill gap, roadmap, 90-day, projects,
          resources (deep-link to Learn), interview prep, hot skills. */}
      {report.career_plan && report.career_plan.plan && (
        <CareerPlanView plan={report.career_plan} userName={userName} />
      )}
    </div>
  )
}
