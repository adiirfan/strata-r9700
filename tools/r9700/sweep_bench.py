#!/usr/bin/env python3
"""tools/r9700/sweep_bench.py URL OUT.json [--max-context N]: prompt reading and writing from short to long context.

Against a Strata server already running (any engine behind it), with the same request as tools/strata_bench.py
(streamed, greedy, thinking off): prompts of about 1K, 4K, 16K, 32K, 64K, 128K and 256K tokens of real text (Strata's
README and docs, in name order, never repeated), each answered once with up to 128 and once with up to 1,024 tokens.
Every prompt starts with a fresh number, so no conversation cache helps.  Speeds are the engine's own timings; the
first-token time is measured here.  Sizes that do not fit the server's context are left out.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from strata_bench import chat  # noqa: E402  the same streamed, greedy, thinking-off request

SIZES = [1024, 4096, 16384, 32768, 65536, 131072, 250000]   # (the last: 256K less room for the answer)
GENS = [128, 1024]
ASK = "\n\nExplain the documents above in detail, section by section."


def corpus() -> str:
    files = [ROOT / "README-STRATA.md"] + sorted((ROOT / "docs").glob("*.md"))
    return "\n\n".join(f"=== {f.name} ===\n\n" + f.read_text(encoding="utf-8", errors="replace") for f in files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--max-context", type=int, default=262144)
    a = ap.parse_args()
    base = a.url.rstrip("/")
    text = corpus()
    # characters per token for this model: one short request, then each size cut to its share of the text
    probe = chat(base, "", [{"role": "user", "content": text[:20000] + ASK}], 1)
    n = ((probe.get("timings") or {}).get("prompt_n")) or ((probe.get("usage") or {}).get("prompt_tokens"))
    if not n:
        print("no token count from the probe:", probe.get("err"), file=sys.stderr)
        return 1
    cpt = 20000 / n
    rows = []
    for size in SIZES:
        if size + max(GENS) + 512 > a.max_context:
            continue
        if int(size * cpt) > len(text):
            print(f"{size}: the text is too short", file=sys.stderr)
            continue
        for gen in GENS:
            body = f"Request {random.randint(0, 10 ** 9)}.\n\n" + text[: int(size * cpt)] + ASK
            r = chat(base, "", [{"role": "user", "content": body}], gen)
            t = r.get("timings") or {}
            row = {"size": size, "gen": gen, "prompt_n": t.get("prompt_n"), "prompt_tps": t.get("prompt_per_second"),
                   "first_s": r.get("first"), "gen_n": t.get("predicted_n"), "gen_tps": t.get("predicted_per_second"),
                   "err": r.get("err")}
            rows.append(row)
            print(f"{size:>7} +{gen:<5} prompt {row['prompt_n']} at {row['prompt_tps']} tok/s, first token "
                  f"{row['first_s'] and round(row['first_s'], 2)} s | wrote {row['gen_n']} at {row['gen_tps']} tok/s"
                  f"{' | ' + str(row['err']) if row['err'] else ''}", flush=True)
            pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
