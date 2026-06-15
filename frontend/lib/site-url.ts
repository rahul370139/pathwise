/**
 * Resolve the public site origin for auth redirects.
 * Server routes use request headers; the browser uses window.location.
 */
export function getSiteOrigin(request?: Request): string {
  const configured = process.env.NEXT_PUBLIC_SITE_URL?.trim().replace(/\/+$/, "")
  if (configured) return configured

  if (request) {
    const host = request.headers.get("x-forwarded-host") || request.headers.get("host")
    if (host) {
      const proto =
        request.headers.get("x-forwarded-proto") ||
        (host.startsWith("localhost") || host.startsWith("127.0.0.1") ? "http" : "https")
      return `${proto}://${host}`.replace(/\/+$/, "")
    }
  }

  const vercelUrl = process.env.VERCEL_URL?.trim().replace(/\/+$/, "")
  if (vercelUrl) return `https://${vercelUrl}`

  return "http://localhost:3000"
}

export function getAuthCallbackUrl(request?: Request): string {
  if (typeof window !== "undefined") {
    return `${window.location.origin.replace(/\/+$/, "")}/auth/callback`
  }
  return `${getSiteOrigin(request)}/auth/callback`
}
