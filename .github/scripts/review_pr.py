#!/usr/bin/env python3
"""Run architect-style PR review via the Claude API and post a GitHub comment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import anthropic

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = REPO_ROOT / "prompts" / "pr_review.md"
BASE_BRANCH = "main"
MAX_DIFF_CHARS = 200_000
MAX_COMMENT_CHARS = 60_000
MODEL = "claude-opus-5"
MAX_TOKENS = 16_000


def _run(cmd: list[str], *, check: bool = True) -> str:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr.strip()}"
        )
    return result.stdout


def _pr_number_from_event() -> int | None:
    path = os.environ.get("GITHUB_EVENT_PATH")
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        event = json.load(f)
    pr = event.get("pull_request")
    if not pr:
        return None
    return int(pr["number"])


def _fetch_main() -> None:
    _run(["git", "fetch", "origin", BASE_BRANCH], check=False)


def _diff_vs_main(pr_number: int) -> str:
    _fetch_main()
    # Always compare PR head to origin/main, not the PR's GitHub base branch.
    head_ref = _run(
        ["gh", "pr", "view", str(pr_number), "--json", "headRefOid", "-q", ".headRefOid"]
    ).strip()
    if not head_ref:
        raise RuntimeError(f"Could not resolve head commit for PR #{pr_number}")
    return _run(["git", "diff", f"origin/{BASE_BRANCH}...{head_ref}"])


def _pr_metadata(pr_number: int) -> dict:
    out = _run(
        [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--json",
            "title,body,files,additions,deletions,url",
        ]
    )
    return json.loads(out)


def _build_prompt(diff: str, metadata: dict) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    parts = [
        template,
        "\n\n---\n\n## PR metadata\n\n",
        f"- **Title:** {metadata.get('title', '')}\n",
        f"- **URL:** {metadata.get('url', '')}\n",
        f"- **Additions / deletions:** +{metadata.get('additions', '?')} / -{metadata.get('deletions', '?')}\n",
    ]
    files = metadata.get("files") or []
    if files:
        parts.append("\n**Changed files:**\n")
        for entry in files[:150]:
            path = entry.get("path", entry) if isinstance(entry, dict) else str(entry)
            parts.append(f"- {path}\n")
    parts.append(f"\n\n## Diff vs `{BASE_BRANCH}`\n\n```diff\n")
    if len(diff) > MAX_DIFF_CHARS:
        parts.append(diff[:MAX_DIFF_CHARS])
        parts.append(f"\n\n... [diff truncated at {MAX_DIFF_CHARS} characters] ...\n")
    else:
        parts.append(diff)
    parts.append("\n```\n")
    return "".join(parts)


def _post_comment(pr_number: int, body: str) -> None:
    if len(body) > MAX_COMMENT_CHARS:
        body = (
            body[:MAX_COMMENT_CHARS]
            + "\n\n_(Review truncated for GitHub comment size limit.)_"
        )
    _run(["gh", "pr", "comment", str(pr_number), "--body", body])


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a PR and post a comment.")
    parser.add_argument("--pr", type=int, help="Pull request number")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompt prefix only; do not call the Claude API or post a comment",
    )
    args = parser.parse_args()

    pr_number = args.pr or _pr_number_from_event()
    if pr_number is None:
        print("PR number required: pass --pr or run inside a pull_request workflow.", file=sys.stderr)
        sys.exit(1)

    if not PROMPT_PATH.is_file():
        print(f"Missing prompt file: {PROMPT_PATH}", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY") and not args.dry_run:
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    metadata = _pr_metadata(pr_number)
    diff = _diff_vs_main(pr_number)
    prompt = _build_prompt(diff, metadata)

    if args.dry_run:
        print(prompt[:8000])
        if len(prompt) > 8000:
            print(f"\n... [{len(prompt) - 8000} more characters] ...", file=sys.stderr)
        return

    client = anthropic.Anthropic()

    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            response = stream.get_final_message()
    except anthropic.AuthenticationError:
        print("ANTHROPIC_API_KEY is invalid.", file=sys.stderr)
        sys.exit(1)
    except anthropic.RateLimitError as err:
        print(f"Rate limited by the Claude API: {err}", file=sys.stderr)
        sys.exit(1)
    except anthropic.APIStatusError as err:
        print(f"Claude API error ({err.status_code}): {err.message}", file=sys.stderr)
        sys.exit(1)
    except anthropic.APIConnectionError as err:
        print(f"Could not reach the Claude API: {err}", file=sys.stderr)
        sys.exit(1)

    if response.stop_reason == "refusal":
        print(f"Claude declined to review this PR: {response.stop_details}", file=sys.stderr)
        sys.exit(2)

    review_text = "\n".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
    if not review_text:
        print("Claude returned an empty review.", file=sys.stderr)
        sys.exit(2)

    header = (
        "## Automated PR review\n\n"
        "_Triggered by the `review` label. Diff base: `main`._\n\n"
    )
    _post_comment(pr_number, header + review_text)
    print(f"Posted review on PR #{pr_number}")


if __name__ == "__main__":
    main()
