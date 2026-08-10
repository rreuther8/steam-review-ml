---
name: pr-review
description: >-
  Staff-architect PR review for steam_recommendations. Use when the user asks to
  review a pull request, PR diff, or branch changes vs main.
---

# PR review (manual)

## When to use

- User asks to review a PR or branch vs `main`.
- User wants architecture / maintainability feedback before merge.

## Setup

- Diff base is always **`main`** (not the PR's GitHub base branch).
- Rubric: [prompts/pr_review.md](../../../prompts/pr_review.md)
- CI uses the same rubric via [.github/scripts/review_pr.py](../../../.github/scripts/review_pr.py) when the PR has label **`review`**.

## Manual workflow

1. Resolve PR number (from user or `gh pr list`).
2. Fetch context:
   - `git fetch origin main`
   - `gh pr view <N> --json title,body,files,additions,deletions,url`
   - `git diff origin/main...$(gh pr view <N> -q .headRefOid)`
3. Read `prompts/pr_review.md` and follow it exactly for structure and tone.
4. Ground findings in the diff and repo layout under `src/`, `scripts/`, `tests/`, `configs/`, `docs/`.
5. Do not propose repo-wide rewrites; prefer incremental, file-scoped changes.

## Automated workflow (GitHub Actions)

- Workflow: [.github/workflows/pr-review.yml](../../../.github/workflows/pr-review.yml)
- Runs when someone adds the **`review`** label (or opens a PR that already has it).
- Does **not** run on every push; to re-run after new commits, remove and re-add `review`, or re-add the label.
- Posts one PR comment via `gh pr comment`.
- Calls the Claude API directly (`claude-opus-5`, via [.github/scripts/review_pr.py](../../../.github/scripts/review_pr.py)) — same rubric as the manual workflow above.

### CI secret

- Store an **`ANTHROPIC_API_KEY`** repository secret ([console.anthropic.com](https://console.anthropic.com)) — the Action needs it to call the Messages API.

## Optional local dry-run (no API call)

```bash
export GH_TOKEN=...   # only if gh is not already authenticated
python .github/scripts/review_pr.py --pr 123 --dry-run
```
