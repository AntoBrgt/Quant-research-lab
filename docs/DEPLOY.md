# Deploying the site

The site in `site/` is a static export: plain HTML, JS and CSS built from `site/public/data/*.json`. It has no
backend, no API keys and no runtime data calls. Everything below except the browser steps is already in the repo.

## Option A: Vercel (primary, free Hobby plan)

In the browser, once:

1. Go to <https://vercel.com/signup> and choose **Continue with GitHub** (Hobby plan, free).
2. **Add New... -> Project -> Import Git Repository**, and pick `AntoBrgt/Quant-research-lab`. If it isn't listed,
   click **Adjust GitHub App Permissions** and grant access to that repository.
3. On the **Configure Project** screen:
   - **Root Directory**: click **Edit** and choose `site`.
   - **Framework Preset**: leave it as detected. `site/vercel.json` overrides it with a pure static build
     (`npm ci`, `npm run build`, output `out`).
   - Leave the build and output settings untouched, and add no environment variables.
4. Click **Deploy**. The URL looks like `https://quant-research-lab-<suffix>.vercel.app`.
5. Optional: **Settings -> Domains** to choose a nicer `*.vercel.app` name, or to add your own domain.
6. Send me the URL. It goes into the README's "Live site" link.

After that, every push to `main` redeploys automatically through Vercel's Git integration. Pushes to other branches
get preview URLs.
- Optional backup trigger: **Settings -> Git -> Deploy Hooks**, create a hook for `main`, and save its URL as the
  GitHub secret `VERCEL_DEPLOY_HOOK_URL` (repository **Settings -> Secrets and variables -> Actions**). The `site`
  workflow then calls it after each green build.

## Option B: GitHub Pages (fallback)

In the browser, once:

1. Repository **Settings -> Pages -> Build and deployment -> Source: GitHub Actions**.
2. Repository **Settings -> Secrets and variables -> Actions -> Variables -> New repository variable**:
   `DEPLOY_TARGET` = `pages`.
3. **Actions -> site -> Run workflow** (or push to `main`).

The site is then served at `https://antobrgt.github.io/Quant-research-lab/`. The workflow builds it with
`BASE_PATH=/Quant-research-lab`, so links and assets work under that subpath (tested locally).

## What the `site` workflow does (`.github/workflows/site.yml`)

On every push or pull request that touches `site/` (including new data in `site/public/data/`):
1. `npm ci && npm run build`. The build includes `scripts/flatten-segments.mjs`, a post-build fix for Next 16's
   static export.
2. `node scripts/check-build.mjs` fails the run if any page lacks the disclaimer footer, if any verdict row from
   `verdicts.json` is missing from `/results/`, or if BUY/SELL wording appears outside the quoted pre-registrations.
3. On `main` only: deploys to GitHub Pages if `DEPLOY_TARGET=pages`, and calls the Vercel deploy hook if that secret
   exists.

The Python `tests` workflow separately checks that `site/public/data/` matches the README verdict table.

## Weekly update path (Phase 3, once the forward test runs)

The research run is not automated: it needs local data caches and, for the agent arm, LLM keys, and both stay at home.

```bash
python src/run_phase3_weekly.py      # refresh data -> funnel -> ranker -> agent -> ledger -> fill prices -> close
python src/export_site_data.py       # rewrites site/public/data/*.json (including ledger.json)
pytest tests/test_export_site_data.py
git add site/public/data && git commit -m "Weekly ledger update YYYY-Www"
git push                             # -> the site workflow checks the build, then the site redeploys
```

`src/run_phase3_weekly.py` belongs to Phase 3 (STEP 21) and does not exist yet. Until then,
`ledger.json` stays `status: "not_started"` and the ledger page shows its empty state.

## Building locally

Needs Node 20.9+ (`node --version`).

```bash
cd site
npm ci
npm run build                        # -> site/out/
node scripts/check-build.mjs
npx serve out                        # http://localhost:3000
```
