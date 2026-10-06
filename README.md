# Sarvam STT on Kathbath / Kathbath-Hard

Evaluates Sarvam AI's speech-to-text API (`saaras:v4`) against
[AI4Bharat/Vistaar](https://github.com/AI4Bharat/Vistaar)'s **Kathbath**
(clean) and **Kathbath-Hard** (ESC-50 noise mixed in at a random 3–30 dB
SNR) benchmarks, under a hard rupee budget — built to fit inside Sarvam's
free trial credit rather than transcribing the full ~37-hour-per-set
benchmark.

Full write-up with results: https://claude.ai/artifact/AJcPL5ftqcJidGXiZqrh2b

## How it works

Both benchmark zips (`kathbath.zip`, `kathbath_noisy.zip`, ~3.5 GB each) are
read as **remote zips** over HTTP range requests, so only the manifest and
the specific sampled audio clips are ever downloaded — not the full 7 GB.

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export SARVAM_API_KEY=...          # never commit this — see below

python eval_sarvam.py prepare --budget-inr 90   # sample utterances + download audio (free)
python eval_sarvam.py run     --budget-inr 90   # transcribe (spends credits, hard-capped)
python eval_sarvam.py score                      # WER / CER per language + pooled
```

- **`prepare`** splits the budget evenly across all 12 languages and, within
  each language, evenly between the clean and noisy pass (every sampled
  utterance is transcribed twice). It samples utterances present in *both*
  sets so the clean/noisy comparison stays paired, and only fetches their
  audio.
- **`run`** transcribes concurrently, tracks cumulative billed seconds
  (Sarvam bills per second, rounded up per request) against the cap, and
  stops issuing new requests once it's reached. Results append to
  `results.jsonl` as they land — a second invocation only retries what's
  missing or failed, never double-pays for a clip already done.
- **`score`** normalizes text the way Vistaar's own evaluation does
  (Unicode NFC, strip punctuation, lowercase) and reports WER/CER with
  `jiwer`, per language and pooled.

## API key handling

The script reads `SARVAM_API_KEY` from the environment at run time — it is
**never** written to disk, logged, or embedded in any output file.
`results.jsonl` and `sample.jsonl` contain only reference/hypothesis text
and durations, no credentials.

If you fork or run this, set your own key as an environment variable (or a
local `.env` you keep out of git — see `.gitignore`); do not paste it into
code or commit history.

## What's committed vs. ignored

Committed: the script, `README.md`, `requirements.txt`, and the run's
output (`sample.jsonl`, `results.jsonl`, `scores_saaras_v4.json`) as
evidence of what was actually measured.

Ignored (`.gitignore`): the Python venv, `__pycache__`, downloaded audio
under `data/` (hundreds of MB, trivially re-fetched by `prepare`), and
`run.log`.

## Results summary

Pooled over 736 paired utterances across 12 languages: **15.4% WER** on
clean Kathbath vs. **16.2% WER** on Kathbath-Hard (+0.8 points for added
noise). Full per-language breakdown and caveats in the linked report above
and in `scores_saaras_v4.json`.
