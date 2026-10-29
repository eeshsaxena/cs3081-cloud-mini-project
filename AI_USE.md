# AI Use

This project was built with AI assistance, as the handout permits ("You'll be using AI and no
one can stop you. But do use it with your eyes wide open."). This file records what was used,
for what, and how the output was checked. Edit it so it matches what *you* actually did.

## Tools used
- **An AI coding assistant (Claude / Claude Code)** — to scaffold the repository, write the
  Python code, Dockerfiles, Kubernetes manifests, the NACL and subnet plan, the CI workflow,
  and the first drafts of the report and README.
- **Ollama + phi3.5** — the *fixed teacher model* required by the handout (not a coding aid):
  used to label the command pool. Temperature 0, seed 42.

## What AI was used for
| Area | AI's role | My role |
|------|-----------|---------|
| Teacher prompt (`prompts/teacher_prompt.txt`) | drafted the instructions + few-shot examples | reviewed against the Section 4 slot rules; ran it and spot-checked replies |
| Worker + `clean.py` (Task 1) | wrote the sharding and cleaning logic | ran it, checked the counts add up, confirmed the checksum is stable |
| Training / eval scripts (Task 2) | wrote `train.py`, `predict.py` | ran training on CPU, read `loss.csv`, scored all three models |
| Service + Dockerfile (Task 3) | wrote the FastAPI app and image | tested the three endpoints, the headers, `FAULT_INJECT`, and the image size |
| K8s manifests (Task 4) | wrote Deployment/Service/Ingress/HPA | applied them, tested rollout/rollback/recovery/cold-start myself |
| NACL + subnets (Task 5) | wrote `nacl.json` and the subnet table | **verified with `nacl_sim.py` (6/6 on the sample packets)** and re-did the subnet maths by hand |
| CI + Makefile (Task 6) | wrote `ci.yml`, `Makefile`, helper scripts | ran the pipeline, including the deliberately-failing gate run |
| Report / README | drafted structure and prose | replaced every number with my own measured result |

## How I checked what the AI gave me
- **Ran everything.** No result in the report is copied from the AI; every measured number
  (scores, latency, memory, cold-start, cost) comes from my own run and the files in `results/`.
- **Used the provided checkers.** `nacl_sim.py` for the NACL, `score.py` for model quality,
  the Locust file for load. These are ground truth, not the AI's claims.
- **Unit tests.** `tests/` covers `clean.py`, the output validation, and the subnet maths; they
  run in CI.
- **Read the code.** I can explain every file; where I changed the AI's output I noted why.

## Limitations I am aware of
- A 70M student trained on ~1,500 noisy teacher labels makes mistakes; the scores in the report
  reflect that honestly.
- The AI occasionally produced plausible-but-wrong details (e.g. version pins, k8s field names);
  I caught these by actually running the build, which is why the provided checkers matter.
