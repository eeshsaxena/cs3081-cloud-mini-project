# Handwritten Reflection — DRAFT (write this out by hand)

> **This must be HANDWRITTEN by you, then scanned and uploaded as `230101032.pdf`** (exact
> filename, no extra spaces/text, or it scores zero). Max 3 pages. This draft is only a skeleton of
> points — put them in your own words and your own handwriting, and only claim numbers you actually
> measured. Do not print this file and submit it; the reflection is assessed as handwritten.

**Top of page 1:** Name (Eesh Saxena), Roll number (230101032), and the GitHub repository link.

## What I learnt (pick the ones that are true for you)
- How knowledge distillation works in practice: a small student is trained on a larger teacher's
  *outputs* (sequence-level), not its weights — and why we couldn't use the teacher's logits here
  (Ollama returns text only, different vocabularies).
- Running Kubernetes locally with k3d, and the pieces that make a service production-ish: readiness vs
  liveness probes, requests/limits, an HPA, rolling updates and rollback.
- Object storage with an S3 API (SeaweedFS), and keeping model weights out of git and the image
  (Release assets + checksums instead).
- Stateless networking: why a NACL needs explicit rules for reply traffic and why rule order decides
  whether a deny actually wins.
- CI/CD with a real quality gate, and secret scanning.

## Challenges I faced, and how I solved them (use the ones you hit)
- **Teacher labelling was slow on CPU** → sharded the work across two parallel workers (indexed Job)
  and added resume/checkpointing so a restart didn't redo everything.
- **Noisy teacher output** → wrote `clean.py` with explicit, countable rejection reasons and verified
  `input = a+b+c+d+duplicates+kept`.
- **Keeping the image ≤ 2 GB** → installed CPU-only PyTorch from the CPU wheel index instead of the
  default CUDA build.
- **Service must never return broken JSON with 200** → validate every output against `schema.json`,
  return 500 `invalid_model_output` otherwise; tested with `FAULT_INJECT=1`.
- **NACL correctness** → used the provided simulator (`nacl_sim.py`) and got 6/6; fixed rule ordering
  so the abusive-range deny beats the port-80/443 allows.
- **[Add a real one of your own — e.g. a k3d/Docker/Ollama error you debugged.]**

## Findings (fill with your measured numbers)
- Student full-frame accuracy was **[__]%** vs the teacher's **[__]%** — distillation kept some skill
  but a 70M model clearly underfits.
- p95 latency **[__] ms**, throughput **[__] rps**, cold start **[__] s**, cost/1000 req **[__] INR**.
- Cleaning kept **[__]** of **[__]** records.

## What I would do better / improvements
- Request batching in the service for higher throughput.
- A shared work queue instead of static modulo sharding, to handle stragglers.
- More epochs or a slightly larger student; a better teacher prompt to reduce label noise.
- A real multi-node cluster for genuine fault tolerance (the single laptop node is the biggest
  limitation).

## How I used AI (be honest — it's required and marked)
- Used an AI assistant to scaffold code/manifests and draft docs; I ran everything myself, verified
  with the provided checkers (`nacl_sim.py`, `score.py`, Locust), and every number here is my own
  measurement. See `AI_USE.md`.
