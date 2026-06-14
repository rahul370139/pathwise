"use client"

import Link from "next/link"
import { Suspense, useEffect, useState } from "react"
import { useSearchParams } from "next/navigation"
import { supabase, getAuthCallbackUrl } from "@/lib/supabase"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Loader2, Mail, AlertCircle } from "lucide-react"

function friendlyAuthError(message: string, code?: string): string {
  const m = message.toLowerCase()
  if (code === "over_email_send_rate_limit" || m.includes("rate limit")) {
    return "Supabase’s built-in email limit was hit for this project. Wait about an hour and try again, or configure custom SMTP in the Supabase dashboard (Authentication → Email)."
  }
  if (m.includes("redirect") || m.includes("redirect_to")) {
    const callback = typeof window !== "undefined" ? getAuthCallbackUrl() : "/auth/callback"
    return `This app URL is not allowed in Supabase. Add ${callback} under Authentication → URL configuration → Redirect URLs (and set Site URL to your frontend domain).`
  }
  if (m.includes("invalid") && m.includes("email")) {
    return "Enter a valid email address."
  }
  return message || "Could not send the magic link. Please try again."
}

export default function LoginPage() {
  return (
    <Suspense fallback={<div className="container mx-auto px-4 py-12 text-center text-sm text-muted-foreground">Loading…</div>}>
      <LoginForm />
    </Suspense>
  )
}

function LoginForm() {
  const searchParams = useSearchParams()
  const [email, setEmail] = useState("")
  const [sent, setSent] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const q = searchParams.get("error")
    if (q) setError(decodeURIComponent(q))
  }, [searchParams])

  const handleLogin = async (e?: React.FormEvent) => {
    e?.preventDefault()
    const trimmed = email.trim()
    if (!trimmed) {
      setError("Enter your email address.")
      return
    }

    setLoading(true)
    setError(null)
    setSent(false)

    try {
      const { error: authError } = await supabase.auth.signInWithOtp({
        email: trimmed,
        options: {
          emailRedirectTo: getAuthCallbackUrl(),
          shouldCreateUser: true,
        },
      })

      if (authError) {
        const err = authError as { message?: string; code?: string; error_code?: string }
        setError(
          friendlyAuthError(
            err.message ?? "Failed to send magic link",
            err.code ?? err.error_code,
          ),
        )
        return
      }

      setSent(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error sending magic link.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container mx-auto flex min-h-[60vh] max-w-md items-center px-4 py-12">
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Mail className="h-5 w-5" />
            Log in or sign up
          </CardTitle>
          <CardDescription>
            We&apos;ll email you a magic link — no password needed.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {sent ? (
            <Alert>
              <Mail className="h-4 w-4" />
              <AlertTitle>Check your inbox</AlertTitle>
              <AlertDescription>
                Magic link sent to <span className="font-medium">{email.trim()}</span>. Open the
                email from Supabase and click the link. Check spam if you don&apos;t see it within
                a minute.
              </AlertDescription>
            </Alert>
          ) : (
            <form onSubmit={handleLogin} className="space-y-4">
              <Input
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
                disabled={loading}
              />
              {error && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Could not send link</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Sending…
                  </>
                ) : (
                  "Send magic link"
                )}
              </Button>
            </form>
          )}

          {sent && (
            <Button variant="outline" className="w-full" onClick={() => setSent(false)}>
              Use a different email
            </Button>
          )}

          <p className="text-center text-xs text-muted-foreground">
            <Link href="/" className="underline hover:text-foreground">
              Back to home
            </Link>
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
