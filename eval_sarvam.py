#!/usr/bin/env python3
"""Evaluate Sarvam speech-to-text on Vistaar's Kathbath (clean) and Kathbath-Hard (noisy) test sets.

The full benchmark is ~37 h per set (~74 h for both), far more than Sarvam's free credits
(Rs 100 at Rs 30/hour, billed per second). So this script evaluates a *paired, stratified sample*:
the same utterances in both clean and noisy form, with equal audio time per language, sized to a
hard budget. Only the sampled WAVs are fetched (HTTP range reads into the remote zips), not 7 GB.

Usage:
    export SARVAM_API_KEY=...            # never hard-code the key
    python eval_sarvam.py prepare --budget-inr 90       # sample + download audio (free)
    python eval_sarvam.py run                           # transcribe (spends credits, capped)
    python eval_sarvam.py score                         # WER/CER per language and set

`run` is resumable: results are appended to results.jsonl and never re-requested.
"""
import argparse
import json
import math
import os
import random
import sys
import threading
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE_URL = "https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/"
SETS = {"clean": "kathbath", "noisy": "kathbath_noisy"}
API_URL = "https://api.sarvam.ai/speech-to-text"
RS_PER_HOUR = 30.0

LANG_CODES = {
    "bengali": "bn-IN", "gujarati": "gu-IN", "hindi": "hi-IN", "kannada": "kn-IN",
    "malayalam": "ml-IN", "marathi": "mr-IN", "odia": "od-IN", "punjabi": "pa-IN",
    "sanskrit": "sa-IN", "tamil": "ta-IN", "telugu": "te-IN", "urdu": "ur-IN",
}

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SAMPLE = ROOT / "sample.jsonl"
RESULTS = ROOT / "results.jsonl"


def billed_seconds(duration):
    # Sarvam bills per second, rounded up per request.
    return math.ceil(duration)


def cost_inr(seconds):
    return seconds / 3600 * RS_PER_HOUR


# --------------------------------------------------------------------------- prepare

def fetch_manifests():
    from remotezip import RemoteZip
    for zipname in SETS.values():
        if all((DATA / zipname / lang / "manifest.json").exists() for lang in LANG_CODES):
            continue
        with RemoteZip(BASE_URL + zipname + ".zip") as z:
            for info in z.infolist():
                if info.filename.endswith("manifest.json"):
                    z.extract(info, DATA)


def load_manifest(zipname, lang):
    with open(DATA / zipname / lang / "manifest.json", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_prepare(args):
    fetch_manifests()
    langs = args.langs or sorted(LANG_CODES)
    # Each sampled utterance is transcribed twice (clean + noisy).
    budget_sec = args.budget_inr / RS_PER_HOUR * 3600
    per_lang_sec = budget_sec / 2 / len(langs)
    rng = random.Random(args.seed)

    sample = []
    for lang in langs:
        clean = load_manifest(SETS["clean"], lang)
        noisy = {Path(r["audio_filepath"]).name: r for r in load_manifest(SETS["noisy"], lang)}
        rows = [r for r in clean if Path(r["audio_filepath"]).name in noisy]
        rng.shuffle(rows)
        used = 0
        for r in rows:
            b = billed_seconds(r["duration"])
            if used + b > per_lang_sec:
                continue
            used += b
            name = Path(r["audio_filepath"]).name
            sample.append({"lang": lang, "id": name, "duration": r["duration"], "text": r["text"]})
        print(f"{lang:10s} {sum(1 for s in sample if s['lang'] == lang):4d} utts  {used / 60:5.1f} min/set")

    total = sum(billed_seconds(s["duration"]) for s in sample) * 2
    print(f"\n{len(sample)} utterances x 2 sets = {total / 3600:.2f} h billed -> Rs {cost_inr(total):.2f}"
          f" (budget Rs {args.budget_inr})")
    with open(SAMPLE, "w", encoding="utf-8") as f:
        for s in sample:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    download_audio(sample)


def download_audio(sample):
    from remotezip import RemoteZip
    for key, zipname in SETS.items():
        todo = [s for s in sample if not audio_path(key, s).exists()]
        if not todo:
            continue
        print(f"downloading {len(todo)} {key} clips ...")
        with RemoteZip(BASE_URL + zipname + ".zip") as z:
            for i, s in enumerate(todo, 1):
                z.extract(f"{zipname}/{s['lang']}/wavs/{s['id']}", DATA)
                if i % 100 == 0:
                    print(f"  {i}/{len(todo)}")


def audio_path(key, s):
    return DATA / SETS[key] / s["lang"] / "wavs" / s["id"]


# --------------------------------------------------------------------------- run

def load_sample():
    with open(SAMPLE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_results():
    if not RESULTS.exists():
        return {}
    out = {}
    with open(RESULTS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[(r["set"], r["lang"], r["id"], r["model"])] = r
    return out


def transcribe(api_key, path, lang, model, mode, retries=5):
    data = {"model": model, "language_code": LANG_CODES[lang]}
    if mode:
        data["mode"] = mode
    for attempt in range(retries):
        with open(path, "rb") as f:
            resp = requests.post(
                API_URL,
                headers={"api-subscription-key": api_key},
                files={"file": (path.name, f, "audio/wav")},
                data=data,
                timeout=120,
            )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt + random.random())
            continue
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError(f"gave up after {retries} retries (last HTTP {resp.status_code})")


def cmd_run(args):
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        sys.exit("Set SARVAM_API_KEY in the environment.")
    sample = load_sample()
    # Keep the comparison paired: skip utterances whose clean or noisy audio isn't downloaded.
    sample = [s for s in sample if all(audio_path(k, s).exists() for k in SETS)]
    if args.limit:
        sample = sample[: args.limit]
    done = load_results()
    sets = args.sets or list(SETS)

    jobs = [(key, s) for s in sample for key in sets
            if (key, s["lang"], s["id"], args.model) not in done]
    already = sum(billed_seconds(r["duration"]) for r in done.values() if r["model"] == args.model)
    cap = args.budget_inr / RS_PER_HOUR * 3600
    print(f"{len(jobs)} requests pending; {already / 3600:.2f} h already billed this model "
          f"(Rs {cost_inr(already):.2f}); cap Rs {args.budget_inr}")

    lock = threading.Lock()
    state = {"spent": already, "n": 0, "errors": 0}
    out = open(RESULTS, "a", encoding="utf-8")

    def work(key, s):
        b = billed_seconds(s["duration"])
        with lock:
            if state["spent"] + b > cap:
                return "skipped"
            state["spent"] += b  # reserve before sending; failed calls may still be billed
        try:
            resp = transcribe(api_key, audio_path(key, s), s["lang"], args.model, args.mode)
        except Exception as e:
            with lock:
                state["errors"] += 1
            print(f"  ERROR {key}/{s['lang']}/{s['id']}: {e}", file=sys.stderr)
            return "error"
        rec = {"set": key, "lang": s["lang"], "id": s["id"], "model": args.model,
               "duration": s["duration"], "ref": s["text"], "hyp": resp.get("transcript", ""),
               "request_id": resp.get("request_id")}
        with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            state["n"] += 1
            if state["n"] % 50 == 0:
                print(f"  {state['n']}/{len(jobs)} done, ~Rs {cost_inr(state['spent']):.2f} spent")
        return "ok"

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        statuses = [f.result() for f in as_completed([ex.submit(work, k, s) for k, s in jobs])]
    out.close()
    print(f"\nfinished: {statuses.count('ok')} ok, {statuses.count('error')} errors, "
          f"{statuses.count('skipped')} skipped (budget). Est. total spend Rs {cost_inr(state['spent']):.2f}")
    if statuses.count("error"):
        print("Errors may still have been billed; re-run `run` to retry them (budget cap still applies).")


# --------------------------------------------------------------------------- score

def normalize(text):
    """Vistaar-style normalisation: drop punctuation (incl. danda), unify whitespace, lowercase."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return " ".join(text.lower().split())


def cmd_score(args):
    import jiwer
    results = [r for r in load_results().values() if r["model"] == args.model]
    if not results:
        sys.exit("No results for this model yet; run `run` first.")
    # Only score utterances present in both sets, so clean vs noisy is a fair comparison.
    keys = defaultdict(set)
    for r in results:
        keys[r["set"]].add((r["lang"], r["id"]))
    paired = keys["clean"] & keys["noisy"]

    groups = defaultdict(lambda: ([], []))
    for r in results:
        if (r["lang"], r["id"]) not in paired:
            continue
        ref, hyp = normalize(r["ref"]), normalize(r["hyp"])
        for g in ((r["set"], r["lang"]), (r["set"], "ALL")):
            groups[g][0].append(ref)
            groups[g][1].append(hyp)

    langs = sorted({l for _, l in groups if l != "ALL"}) + ["ALL"]
    rows = []
    print(f"model={args.model}  paired utterances={len(paired)}\n")
    print(f"{'language':10s} {'n':>5s} {'WER clean':>10s} {'WER noisy':>10s} {'ΔWER':>7s} "
          f"{'CER clean':>10s} {'CER noisy':>10s}")
    for lang in langs:
        m = {}
        for key in ("clean", "noisy"):
            refs, hyps = groups[(key, lang)]
            m[key] = (jiwer.wer(refs, hyps) * 100, jiwer.cer(refs, hyps) * 100, len(refs))
        print(f"{lang:10s} {m['clean'][2]:5d} {m['clean'][0]:10.2f} {m['noisy'][0]:10.2f} "
              f"{m['noisy'][0] - m['clean'][0]:+7.2f} {m['clean'][1]:10.2f} {m['noisy'][1]:10.2f}")
        rows.append({"language": lang, "n": m["clean"][2],
                     "wer_clean": m["clean"][0], "wer_noisy": m["noisy"][0],
                     "cer_clean": m["clean"][1], "cer_noisy": m["noisy"][1]})
    print("\nALL = pooled (corpus-level) over all languages, not the mean of per-language WERs.")
    with open(ROOT / f"scores_{args.model.replace(':', '_')}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


# --------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="sample utterances within budget and download their audio")
    pp.add_argument("--budget-inr", type=float, default=90.0)
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--langs", nargs="+", choices=sorted(LANG_CODES))

    pr = sub.add_parser("run", help="transcribe the sample with Sarvam (spends credits)")
    pr.add_argument("--budget-inr", type=float, default=90.0, help="hard cap on cumulative spend")
    pr.add_argument("--model", default="saaras:v4")
    pr.add_argument("--mode", help="saaras:v3 only (transcribe/verbatim/...); v4 rejects it")
    pr.add_argument("--workers", type=int, default=4)
    pr.add_argument("--limit", type=int, help="only the first N sampled utterances (smoke test)")
    pr.add_argument("--sets", nargs="+", choices=list(SETS))

    ps = sub.add_parser("score", help="compute WER/CER")
    ps.add_argument("--model", default="saaras:v4")

    args = p.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
