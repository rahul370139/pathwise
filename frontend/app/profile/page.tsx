"use client"

import Link from "next/link"
import { Suspense, useMemo } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { useAuth } from "@/components/auth-provider"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Separator } from "@/components/ui/separator"
import {
  User,
  Shield,
  CreditCard,
  Settings,
  LayoutDashboard,
  Download,
  LogOut,
  LogIn,
  BookOpen,
  Sparkles,
} from "lucide-react"
import { toast } from "@/components/ui/use-toast"
import { ModeToggle } from "@/components/mode-toggle"

const VALID_TABS = ["account", "privacy", "payments", "settings"] as const
type ProfileTab = (typeof VALID_TABS)[number]

function ProfilePageContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { user, loading, signOut } = useAuth()

  const activeTab = useMemo(() => {
    const tab = (searchParams.get("tab") || "account").toLowerCase()
    return (VALID_TABS.includes(tab as ProfileTab) ? tab : "account") as ProfileTab
  }, [searchParams])

  const setTab = (tab: ProfileTab) => {
    router.replace(`/profile?tab=${tab}`, { scroll: false })
  }

  const displayName =
    (user?.user_metadata?.full_name as string | undefined) ||
    (user?.user_metadata?.name as string | undefined) ||
    (user?.email ? user.email.split("@")[0] : "")
  const avatarUrl =
    (user?.user_metadata?.avatar_url as string | undefined) ||
    (user?.user_metadata?.picture as string | undefined) ||
    "/placeholder-user.jpg"
  const initials = (displayName || user?.email || "U").slice(0, 2).toUpperCase()
  const memberSince = user?.created_at
    ? new Date(user.created_at).toLocaleDateString(undefined, { dateStyle: "long" })
    : null

  const handleExportLocalData = () => {
    try {
      const payload = {
        exported_at: new Date().toISOString(),
        user_id: user?.id,
        email: user?.email,
        local: {
          chat_messages: localStorage.getItem("pathwise_chat_messages"),
          conversation_id: localStorage.getItem("pathwise_conversation_id"),
          simulator_state: localStorage.getItem("pathwise_simulator_state"),
        },
      }
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `pathwise-data-export-${new Date().toISOString().split("T")[0]}.json`
      a.click()
      URL.revokeObjectURL(url)
      toast({ title: "Export ready", description: "Your local PathWise data was downloaded." })
    } catch {
      toast({
        title: "Export failed",
        description: "Could not read local data from this browser.",
        variant: "destructive",
      })
    }
  }

  if (loading) {
    return (
      <div className="container mx-auto max-w-4xl px-4 py-10">
        <div className="h-8 w-48 animate-pulse rounded bg-muted mb-6" />
        <div className="h-64 animate-pulse rounded-lg bg-muted" />
      </div>
    )
  }

  if (!user) {
    return (
      <div className="container mx-auto max-w-lg px-4 py-16 text-center">
        <Card>
          <CardHeader>
            <CardTitle>Sign in to manage your profile</CardTitle>
            <CardDescription>
              Account, privacy, billing, and settings are available after you sign in.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <Button asChild>
              <Link href="/login">
                <LogIn className="h-4 w-4 mr-2" />
                Sign in
              </Link>
            </Button>
            <Button asChild variant="outline">
              <Link href="/dashboard">View demo dashboard</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="container mx-auto max-w-4xl px-4 py-8 md:py-10">
      <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-4">
          <Avatar className="h-14 w-14">
            <AvatarImage src={avatarUrl} alt={displayName || "Profile"} />
            <AvatarFallback className="bg-gradient-to-r from-blue-500 to-purple-500 text-white text-lg">
              {initials}
            </AvatarFallback>
          </Avatar>
          <div>
            <h1 className="text-2xl font-bold">{displayName || "Your profile"}</h1>
            <p className="text-sm text-muted-foreground">{user.email}</p>
            {memberSince && (
              <p className="text-xs text-muted-foreground mt-1">Member since {memberSince}</p>
            )}
          </div>
        </div>
        <Button variant="outline" size="sm" onClick={() => void signOut()}>
          <LogOut className="h-4 w-4 mr-2" />
          Sign out
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={(v) => setTab(v as ProfileTab)} className="w-full">
        <TabsList className="grid w-full grid-cols-2 md:grid-cols-4 h-auto">
          <TabsTrigger value="account" className="gap-2 py-2">
            <User className="h-4 w-4" />
            Account
          </TabsTrigger>
          <TabsTrigger value="privacy" className="gap-2 py-2">
            <Shield className="h-4 w-4" />
            Privacy
          </TabsTrigger>
          <TabsTrigger value="payments" className="gap-2 py-2">
            <CreditCard className="h-4 w-4" />
            Payments
          </TabsTrigger>
          <TabsTrigger value="settings" className="gap-2 py-2">
            <Settings className="h-4 w-4" />
            Settings
          </TabsTrigger>
        </TabsList>

        <TabsContent value="account" className="mt-6 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Account overview</CardTitle>
              <CardDescription>Your PathWise identity and quick links.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-lg border p-3">
                  <p className="text-xs text-muted-foreground">Email</p>
                  <p className="font-medium break-all">{user.email}</p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-xs text-muted-foreground">User ID</p>
                  <p className="font-mono text-xs break-all">{user.id}</p>
                </div>
              </div>
              <Separator />
              <div className="flex flex-wrap gap-2">
                <Button asChild>
                  <Link href="/dashboard">
                    <LayoutDashboard className="h-4 w-4 mr-2" />
                    Open dashboard
                  </Link>
                </Button>
                <Button asChild variant="secondary">
                  <Link href="/learn">
                    <BookOpen className="h-4 w-4 mr-2" />
                    Continue learning
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="privacy" className="mt-6 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Privacy &amp; security</CardTitle>
              <CardDescription>Control how your data is stored and exported on this device.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="rounded-lg border p-4 space-y-2">
                <div className="flex items-center justify-between gap-2">
                  <p className="font-medium">Local session data</p>
                  <Badge variant="secondary">This browser</Badge>
                </div>
                <p className="text-sm text-muted-foreground">
                  Chat history, conversation IDs, and simulator progress are stored in your browser until you clear
                  them or sign out on a shared device.
                </p>
              </div>
              <div className="rounded-lg border p-4 space-y-3">
                <p className="font-medium">Export your data</p>
                <p className="text-sm text-muted-foreground">
                  Download a JSON copy of locally stored PathWise data (chat messages, conversation id, simulator
                  state).
                </p>
                <Button variant="outline" onClick={handleExportLocalData}>
                  <Download className="h-4 w-4 mr-2" />
                  Export local data
                </Button>
              </div>
              <div className="rounded-lg border border-dashed p-4 space-y-2">
                <p className="font-medium">Account deletion</p>
                <p className="text-sm text-muted-foreground">
                  To permanently delete your account and server-side records, contact support. Self-serve deletion
                  will be added in a future release.
                </p>
              </div>
              <ul className="text-sm text-muted-foreground list-disc pl-5 space-y-1">
                <li>We do not sell your personal learning data.</li>
                <li>Career plans and mastery scores are tied to your user ID when signed in.</li>
                <li>Sign out on shared computers after each session.</li>
              </ul>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="payments" className="mt-6 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Billing &amp; plans</CardTitle>
              <CardDescription>Manage your PathWise subscription.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="rounded-lg border bg-muted/30 p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2">
                    <p className="font-semibold">Free plan</p>
                    <Badge>Current</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mt-1">
                    Learn, Career Simulator, and dashboard tracking at no cost.
                  </p>
                </div>
                <Button disabled className="shrink-0">
                  <Sparkles className="h-4 w-4 mr-2" />
                  Active
                </Button>
              </div>
              <div className="rounded-lg border border-amber-200/80 bg-amber-50/50 dark:bg-amber-950/20 p-4 space-y-3">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-amber-600" />
                  <p className="font-semibold">PathWise Pro</p>
                  <Badge variant="outline">Coming soon</Badge>
                </div>
                <p className="text-sm text-muted-foreground">
                  Priority AI models, unlimited career simulations, advanced analytics, and team workspace features.
                </p>
                <Button variant="secondary" disabled>
                  Upgrade to Pro — coming soon
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                No payment method on file. Billing integrations will appear here when Pro launches.
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="mt-6 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Preferences</CardTitle>
              <CardDescription>Appearance and learning defaults.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="flex items-center justify-between rounded-lg border p-4">
                <div>
                  <p className="font-medium">Theme</p>
                  <p className="text-sm text-muted-foreground">Switch between light and dark mode.</p>
                </div>
                <ModeToggle />
              </div>
              <div className="rounded-lg border p-4 space-y-3">
                <p className="font-medium">Learning settings</p>
                <p className="text-sm text-muted-foreground">
                  Explanation level and framework focus are configured on the Learn page sidebar (Apply button).
                </p>
                <Button asChild variant="outline">
                  <Link href="/learn">Open Learn settings</Link>
                </Button>
              </div>
              <div className="rounded-lg border p-4 space-y-3">
                <p className="font-medium">Career simulator</p>
                <p className="text-sm text-muted-foreground">
                  Reset or continue your saved simulator session from the Career Simulator page.
                </p>
                <Button asChild variant="outline">
                  <Link href="/simulator">Open Career Simulator</Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default function ProfilePage() {
  return (
    <Suspense
      fallback={
        <div className="container mx-auto max-w-4xl px-4 py-10">
          <div className="h-8 w-48 animate-pulse rounded bg-muted" />
        </div>
      }
    >
      <ProfilePageContent />
    </Suspense>
  )
}
