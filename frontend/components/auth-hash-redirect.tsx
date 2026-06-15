"use client"

import { useEffect } from "react"

/**
 * Supabase magic links sometimes land on the site root with tokens in the URL hash
 * (e.g. /#access_token=...) instead of /auth/callback. Forward those to the callback
 * handler on the same origin so sign-in can complete.
 */
export function AuthHashRedirect() {
  useEffect(() => {
    const { pathname, hash } = window.location
    if (!hash) return
    if (pathname === "/auth/callback") return

    const isAuthHash =
      hash.includes("access_token=") ||
      hash.includes("refresh_token=") ||
      hash.includes("type=magiclink") ||
      hash.includes("type=recovery")

    if (isAuthHash) {
      window.location.replace(`/auth/callback${hash}`)
    }
  }, [])

  return null
}
