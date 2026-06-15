import { createClient } from "@supabase/supabase-js"
import { NextRequest, NextResponse } from "next/server"
import { getAuthCallbackUrl } from "@/lib/site-url"

export async function POST(request: NextRequest) {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

  if (!supabaseUrl || !supabaseAnonKey) {
    return NextResponse.json(
      { error: "Supabase is not configured on the server." },
      { status: 500 },
    )
  }

  let email: string
  try {
    const body = await request.json()
    email = typeof body.email === "string" ? body.email.trim() : ""
  } catch {
    return NextResponse.json({ error: "Invalid request body." }, { status: 400 })
  }

  if (!email) {
    return NextResponse.json({ error: "Enter a valid email address." }, { status: 400 })
  }

  const emailRedirectTo = getAuthCallbackUrl(request)
  const supabase = createClient(supabaseUrl, supabaseAnonKey)

  const { error } = await supabase.auth.signInWithOtp({
    email,
    options: {
      emailRedirectTo,
      shouldCreateUser: true,
    },
  })

  if (error) {
    const code = (error as { code?: string }).code
    return NextResponse.json(
      {
        error: error.message,
        code,
        emailRedirectTo,
      },
      { status: 400 },
    )
  }

  return NextResponse.json({ ok: true, emailRedirectTo })
}
