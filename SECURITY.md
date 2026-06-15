# Security

## Reporting a vulnerability

Do **not** open a public issue for security problems.

Use [GitHub Security Advisories](https://github.com/rahul370139/PathWise/security/advisories/new) on this repository, or contact the maintainer privately.

## Secrets

- Never commit `backend.env`, `.env`, or `.env.local` — they are gitignored.
- Store production secrets only in:
  - **VPS:** `backend/backend.env` (chmod 600)
  - **Vercel:** project environment variables
  - **Supabase / Azure / Groq / Cohere dashboards:** provider consoles

If a secret is accidentally pushed, **rotate it immediately** in the provider console and remove it from git history.

## Public demo

The production API (`http://2.24.74.235:8000`) is intentionally public for the Agents League demo. Do not expose admin credentials or internal-only endpoints.
