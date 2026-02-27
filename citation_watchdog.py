"""
citation_watchdog.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gimel (⌐■_■)✓ — uncited-derivative detection for MBD-Framework

Two-stage strategy
  Stage 1 – KNOWN CITERS: query OpenAlex for every paper/work that
            explicitly cites one of our registered DOIs.
  Stage 2 – CONCEPT MENTIONS: search Semantic Scholar and OpenAlex
            full-text for distinctive phrases that appear in the
            papers but are NOT in the known-citer set.
  Stage 3 – CODE FINGERPRINTS: query the GitHub Search API for
            distinctive variable names and equation fragments.

The gap between Stage 2/3 results and Stage 1 is the suspect list.

Usage
─────
  python citation_watchdog.py             # full report to stdout
  python citation_watchdog.py --json      # machine-readable JSON
  python citation_watchdog.py --github    # GitHub code scan only

Requires: requests  (pip install requests)
Optional: Set GITHUB_TOKEN env var for higher GitHub rate limits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

try:
    import requests
except ImportError:
    sys.exit("Install requests first:  pip install requests")

# ── DOI registry ──────────────────────────────────────────────────────────────

PAPERS = [
    {
        "short": "MBD Core",
        "title": "Memory as Baseline Deviation: A Formal Framework for Personality as State-Space Dynamics",
        "doi": "10.5281/zenodo.17381536",
    },
    {
        "short": "Markov Tensor",
        "title": "In Pursuit of the Markov Tensor: A Geometric Framework for Social Cognition",
        "doi": "10.5281/zenodo.17537185",
    },
    {
        "short": "Episodic Recall",
        "title": "Episodic Recall as Resonant Re-instantiation: A Fokker–Planck Account of Memory Retrieval",
        "doi": "10.5281/zenodo.17374270",
    },
    {
        "short": "Coupling Asymmetry",
        "title": "The Coupling Asymmetry: Executive Dysfunction as an Eigenstate of the Memory–Baseline System",
        "doi": "10.5281/zenodo.18519187",
    },
    {
        "short": "Emergent Gate",
        "title": "The Emergent Gate: Memory Encoding as Threshold-Dependent Consolidation",
        "doi": "10.5281/zenodo.17344091",
    },
    {
        "short": "Resonant Gate",
        "title": "The Resonant Gate: Conversational Insight as Phase-Locked Coupling in the MBD Framework",
        "doi": "10.5281/zenodo.17352481",
    },
    {
        "short": "Software package",
        "title": "MBD-Framework: Memory as Baseline Deviation — Reference Implementations",
        "doi": "10.5281/zenodo.18652919",
    },
]

# ── Distinctive search phrases ─────────────────────────────────────────────────
# Chosen to be rare enough to be fingerprints, not so rare they miss paraphrases.

SEMANTIC_PHRASES = [
    "Memory as Baseline Deviation",
    "Markov Tensor",
    "Resonant Re-instantiation",
    "Coupling Asymmetry eigenstate",
    "Emergent Gate memory encoding",
    "Resonant Gate phase-locked coupling",
    "baseline deviation vector",
    "kappa coupling coefficient personality",
]

# ── GitHub code fingerprints ───────────────────────────────────────────────────

GITHUB_FINGERPRINTS = [
    "baseline_deviation",
    "kappa_coupling",
    "delta_baseline",
    "MBD framework",
    "memory_as_baseline",
    '"Memory as Baseline Deviation"',
    '"Markov Tensor"',
    "resonant_reinstantiation",
    "coupling_asymmetry",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MBD-citation-watchdog/1.0 (YellowHapax)"})

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


def _get(url: str, params: dict | None = None, headers: dict | None = None,
         retries: int = 3) -> dict | list | None:
    h = headers or {}
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, headers=h, timeout=20)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                print(f"  [rate-limit] sleeping {wait}s …", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                print(f"  [warn] {url} → {exc}", flush=True)
                return None
            time.sleep(2 ** attempt)
    return None


# ── Stage 1: OpenAlex — known citers ──────────────────────────────────────────

def fetch_openalex_citing(doi: str) -> list[dict]:
    """Return list of OpenAlex Work objects that cite *doi*."""
    # Resolve openalex ID from DOI first
    work = _get("https://api.openalex.org/works", params={"filter": f"doi:{doi}"})
    if not work or not work.get("results"):
        return []
    oa_id = work["results"][0].get("id", "")  # e.g. https://openalex.org/W1234
    short_id = oa_id.rsplit("/", 1)[-1]

    # Paginate cited_by
    citing = []
    cursor = "*"
    while cursor:
        resp = _get(
            "https://api.openalex.org/works",
            params={
                "filter": f"cites:{short_id}",
                "per-page": 200,
                "cursor": cursor,
            },
        )
        if not resp:
            break
        results = resp.get("results", [])
        citing.extend(results)
        cursor = resp.get("meta", {}).get("next_cursor")
        if not results:
            break
        time.sleep(0.3)

    return citing


# ── Stage 2: Semantic Scholar — concept mention search ────────────────────────

def search_semantic_scholar(phrase: str) -> list[dict]:
    """Full-text search on Semantic Scholar for *phrase*."""
    resp = _get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": phrase,
            "fields": "title,externalIds,year,authors,openAccessPdf",
            "limit": 20,
        },
    )
    if not resp:
        return []
    return resp.get("data", [])


def search_openalex_fulltext(phrase: str) -> list[dict]:
    """Free-text search on OpenAlex for *phrase*."""
    resp = _get(
        "https://api.openalex.org/works",
        params={
            "search": phrase,
            "per-page": 20,
            "sort": "relevance_score:desc",
        },
    )
    if not resp:
        return []
    return resp.get("results", [])


# ── Stage 3: GitHub code search ───────────────────────────────────────────────

def search_github_code(fingerprint: str) -> list[dict]:
    """Search GitHub for code containing *fingerprint*."""
    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    resp = _get(
        "https://api.github.com/search/code",
        params={"q": fingerprint, "per_page": 10},
        headers=headers,
    )
    time.sleep(1.5)  # GitHub secondary rate-limit courtesy delay
    if not resp:
        return []
    return resp.get("items", [])


# ── Key extractor helpers ──────────────────────────────────────────────────────

def _oa_doi(work: dict) -> str:
    raw = work.get("doi", "") or ""
    # OpenAlex stores as full URL, normalise
    return raw.replace("https://doi.org/", "").lower().strip()


def _oa_title(work: dict) -> str:
    return work.get("display_name") or work.get("title") or "(no title)"


def _ss_doi(work: dict) -> str:
    ext = work.get("externalIds", {}) or {}
    doi = ext.get("DOI", "") or ""
    return doi.lower().strip()


def _ss_title(work: dict) -> str:
    return work.get("title") or "(no title)"


# ── Report assembly ────────────────────────────────────────────────────────────

def run_watchdog(github_only: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {
        "known_citers": [],       # DOIs that explicitly cite our work
        "concept_mentions": [],   # Works mentioning our phrases (suspect pool)
        "uncited_suspects": [],   # concept_mentions minus known_citers
        "github_hits": [],        # Code repos with our fingerprints
    }

    our_dois = {p["doi"].lower() for p in PAPERS}

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if not github_only:
        print("\n[Stage 1] Fetching known citers from OpenAlex …", flush=True)
        known_citer_dois: set[str] = set()
        for paper in PAPERS:
            print(f"  . {paper['short']} ({paper['doi']})", flush=True)
            citers = fetch_openalex_citing(paper["doi"])
            for c in citers:
                d = _oa_doi(c)
                t = _oa_title(c)
                if d and d not in our_dois:
                    known_citer_dois.add(d)
                    record = {"doi": d, "title": t, "cited_paper": paper["short"]}
                    if record not in report["known_citers"]:
                        report["known_citers"].append(record)
            time.sleep(0.5)

        # ── Stage 2 ──────────────────────────────────────────────────────────
        print("\n[Stage 2] Scanning for concept mentions...", flush=True)
        seen_titles: set[str] = set()

        for phrase in SEMANTIC_PHRASES:
            print(f"  . SS  '{phrase}'", flush=True)
            for hit in search_semantic_scholar(phrase):
                d = _ss_doi(hit)
                t = _ss_title(hit)
                key = t.lower()[:80]
                if key in seen_titles or d in our_dois:
                    continue
                seen_titles.add(key)
                report["concept_mentions"].append({
                    "source": "SemanticScholar",
                    "doi": d,
                    "title": t,
                    "matched_phrase": phrase,
                    "year": hit.get("year"),
                })
            time.sleep(0.5)

            print(f"  . OA  '{phrase}'", flush=True)
            for hit in search_openalex_fulltext(phrase):
                d = _oa_doi(hit)
                t = _oa_title(hit)
                key = t.lower()[:80]
                if key in seen_titles or d in our_dois:
                    continue
                seen_titles.add(key)
                report["concept_mentions"].append({
                    "source": "OpenAlex",
                    "doi": d,
                    "title": t,
                    "matched_phrase": phrase,
                    "year": (hit.get("publication_year") or ""),
                })
            time.sleep(0.5)

        # ── Diff: suspects = mentions that are NOT known citers ───────────────
        known_citer_dois_norm = {d.lower() for d in known_citer_dois}
        for mention in report["concept_mentions"]:
            md = mention["doi"].lower()
            if md and md not in known_citer_dois_norm and md not in our_dois:
                report["uncited_suspects"].append(mention)

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    print("\n[Stage 3] GitHub code fingerprint scan …", flush=True)
    if not GITHUB_TOKEN:
        print("  [hint] Set GITHUB_TOKEN for higher rate limits and better results.")
    seen_repos: set[str] = set()
    for fp in GITHUB_FINGERPRINTS:
        print(f"  . '{fp}'", flush=True)
        hits = search_github_code(fp)
        for item in hits:
            repo = item.get("repository", {}).get("full_name", "?")
            if repo in seen_repos:
                continue
            seen_repos.add(repo)
            report["github_hits"].append({
                "repo": repo,
                "repo_url": item.get("repository", {}).get("html_url", ""),
                "file": item.get("path", ""),
                "file_url": item.get("html_url", ""),
                "matched_fingerprint": fp,
            })

    return report


# ── Pretty printer ─────────────────────────────────────────────────────────────

def print_report(report: dict[str, Any]) -> None:
    sep = "─" * 72

    print(f"\n{sep}")
    print("MBD-Framework Citation Watchdog  —  (⌐■_■)✓")
    print(sep)

    kc = report["known_citers"]
    print(f"\n✅ KNOWN CITERS ({len(kc)} works explicitly cite our DOIs)")
    if kc:
        for r in kc:
            print(f"   [{r['cited_paper']}]  {r['title']}")
            if r["doi"]:
                print(f"      doi: {r['doi']}")
    else:
        print("   (none found — you may be first!)")

    cm = report["concept_mentions"]
    us = report["uncited_suspects"]
    print(f"\n⚠️  CONCEPT MENTIONS ({len(cm)} total, {len(us)} NOT in citer list)")
    if us:
        for r in us:
            print(f"   [{r['source']}] {r['title']} ({r.get('year','?')})")
            print(f"      matched: '{r['matched_phrase']}'")
            if r["doi"]:
                print(f"      doi: {r['doi']}")
    else:
        print("   (no uncited suspects found)")

    gh = report["github_hits"]
    print(f"\n🔍 GITHUB CODE HITS ({len(gh)} repos)")
    if gh:
        for r in gh:
            print(f"   {r['repo']}  →  {r['file']}")
            print(f"      matched: '{r['matched_fingerprint']}'")
            print(f"      {r['file_url']}")
    else:
        print("   (no code matches found)")

    print(f"\n{sep}")
    print("Next steps for suspects:")
    print("  1. Read the abstract — is the concept actually from MBD?")
    print("  2. Check references section for our DOIs.")
    print("  3. If absent: screenshot + note date, then reach out to authors.")
    print(sep)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MBD uncited-derivative detector")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--github", action="store_true", help="GitHub scan only")
    args = parser.parse_args()

    report = run_watchdog(github_only=args.github)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
