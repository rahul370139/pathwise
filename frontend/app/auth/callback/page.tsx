'use client'

import { Suspense, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import type { Session } from '@supabase/supabase-js'
import { supabase } from '@/lib/supabase'

/** After initialize(), session may land a tick later via onAuthStateChange. */
async function waitForSession(timeoutMs = 4000): Promise<Session | null> {
  const existing = await supabase.auth.getSession()
  if (existing.data.session) return existing.data.session

  return new Promise<Session | null>((resolve) => {
    let subscription: { unsubscribe: () => void } | null = null
    const timer = setTimeout(() => {
      subscription?.unsubscribe()
      resolve(null)
    }, timeoutMs)

    const sub = supabase.auth.onAuthStateChange((event, session) => {
      if ((event === 'SIGNED_IN' || event === 'TOKEN_REFRESHED') && session) {
        clearTimeout(timer)
        sub.data.subscription.unsubscribe()
        resolve(session)
      }
    })
    subscription = sub.data.subscription
  })
}

function CallbackInner() {
  const router = useRouter()

  useEffect(() => {
    let isMounted = true

    const handleCallback = async () => {
      try {
        const { error: initError } = await supabase.auth.initialize()

        if (initError) {
          const msg =
            initError.message ||
            'Sign-in link is invalid or has expired. Request a new magic link.'
          if (isMounted) router.replace(`/login?error=${encodeURIComponent(msg)}`)
          return
        }

        const session = await waitForSession()

        if (session) {
          if (isMounted) router.replace('/')
          return
        }

        if (isMounted) {
          router.replace(
            `/login?error=${encodeURIComponent(
              'Could not complete sign-in. Open the latest magic link in the same browser, or request a new one.',
            )}`,
          )
        }
      } catch (err) {
        console.error('Auth callback error:', err)
        const msg = err instanceof Error ? err.message : 'Sign-in failed'
        if (isMounted) router.replace(`/login?error=${encodeURIComponent(msg)}`)
      }
    }

    void handleCallback()
    return () => {
      isMounted = false
    }
  }, [router])

  return (
    <div className="min-h-[60vh] flex items-center justify-center">
      <div className="space-y-3 text-center">
        <div className="mx-auto h-10 w-10 rounded-full border-2 border-primary border-t-transparent animate-spin" />
        <p className="text-sm text-muted-foreground">Finishing sign-in…</p>
      </div>
    </div>
  )
}

export default function AuthCallback() {
  return (
    <Suspense
      fallback={
        <div className="min-h-[60vh] flex items-center justify-center">
          <div className="space-y-3 text-center">
            <div className="mx-auto h-10 w-10 rounded-full border-2 border-primary border-t-transparent animate-spin" />
            <p className="text-sm text-muted-foreground">Loading…</p>
          </div>
        </div>
      }
    >
      <CallbackInner />
    </Suspense>
  )
}
