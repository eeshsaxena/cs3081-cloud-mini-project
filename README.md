# Cloud Computing Mini Project — Command Parser (CS3081)

Knowledge distillation of a command-parsing model (`phi3.5` teacher → `pythia-70m` student),
served on a local k3d Kubernetes cluster with SeaweedFS object storage.

> **Student:** Eesh Saxena · **Roll number:** 230101032 · **Repo:** _<github link>_

## What it does
Takes one English command and returns a four-key JSON object:

```
POST /parse  {"text": "set an alarm for seven thirty am tomorrow"}
->  {"intent": "alarm_set", "date": "tomorrow", "time": "seven thirty am", "place": null}
```

## Prerequisites
| Tool | Used for | Notes |
|------|----------|-------|
| Docker | build images, run k3d | Docker Desktop on Windows/Mac |
| [k3d](https://k3d.io) + kubectl | local Kubernetes | `k3d` runs k3s in Docker |
| [Ollama](https://ollama.com) + `phi3.5` | the teacher | `ollama pull phi3.5` (~2.2 GB) |
| Python 3.11 | training, scripts, tests | CPU-only |
| make, [locust](https://locust.io), [gh](https://cli.github.com) | orchestration, load test, releases | |

> On Windows, run the `.sh` helpers from Git Bash, or install `make` (e.g. via Chocolatey:
> `choco install make`). Every `make` target just calls a script in `scripts/`.

## Repository layout
```
cluster/      k3d.yaml                     Task 1  cluster config
prompts/      teacher_prompt.txt           Task 1  teacher prompt
worker/       label_worker.py, clean.py    Task 1  sharded labelling + cleaning
training/     train.py, predict.py, config Task 2  student training + inference
service/      app.py, model_runner.py, Dockerfile   Task 3  FastAPI /parse service
k8s/          *.yaml                       Tasks 1 & 4  namespaces, SeaweedFS, deploy, svc, ingress, hpa
network/      subnets.md, nacl.json        Task 5  subnetting + NACL
scripts/      up/deploy/evaluate/cost/...  Tasks 1-7  helpers
tools/        score.py, nacl_sim.py, locustfile.py   provided checkers
tests/        unit tests                   Task 6
data/         commands_pool, validation, schema, rate_card   provided kit
results/       logs, score reports, locust output           Task 7 evidence
report/       report.pdf                   final report
```

## Quick start (three commands)
```bash
cp .env.example .env            # 1. local dev secret (edit the value)
make up                         # 2. cluster + storage + build + deploy the service
make train                      # 3. produce the model  (needs the dataset; see below)
```
`make up` creates the k3d cluster, deploys SeaweedFS, creates the `datasets`/`models` buckets,
builds and imports the images, seeds the model from the GitHub Release (if present), and deploys
the service behind an Ingress at `http://localhost:8080/parse`.

## Full workflow

### Task 1 — object storage + parallel teacher labelling
```bash
ollama pull phi3.5 && ollama serve          # teacher on the laptop
make up                                      # cluster + SeaweedFS + buckets (uploads commands_pool.jsonl)
make label                                   # indexed Job: 2 workers, SHARD_INDEX 0 and 1
# (or locally without k8s:)  bash scripts/run_workers_local.sh
make clean-data                              # -> datasets/v1/train.jsonl + train.sha256, prints counts
```

### Task 2 — train the student
```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r training/requirements.txt
make train SUBSET=50          # smoke test (fast)
make train                    # full v1 -> models/v1/ (+ models/v1.tar.gz + model.sha256)
make train-v2                 # v2 (different seed) -> models/v2/  (needed for Task 4)
make eval                     # scores teacher, untrained base, trained student -> results/
```
Publish `models/v1.tar.gz` (and `v2.tar.gz`) as **GitHub Release assets** under tag `model-v1`;
the checksum in `models/v1/model.sha256` must match the uploaded asset. **Do not commit weights.**

### Task 3 — the container
```bash
docker build -f service/Dockerfile -t cloud-mini/service:v1 .
docker images cloud-mini/service:v1           # confirm <= 2 GB
# local smoke test:
docker run --rm -p 8000:8000 -e MODEL_URI=s3://models/v1 -e MODEL_VERSION=v1 \
  -e S3_ENDPOINT=http://host.docker.internal:8333 -e MAX_NEW_TOKENS=64 \
  -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... cloud-mini/service:v1
curl -s localhost:8000/parse -H 'content-type: application/json' -d '{"text":"weather in london"}'
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/parse \
  -H 'content-type: application/json' -d '{"text":"x"}'   # with FAULT_INJECT=1 -> 500
```

### Task 4 — deploy, scale, reliability
```bash
make deploy                                   # Deployment(2) + Service + Ingress + HPA
kubectl -n app get deploy,svc,ingress,hpa,pods
curl -s localhost:8080/parse -H 'content-type: application/json' -d '{"text":"weather in london"}' -i | head
# rolling update / rollback, recovery, cold start, load balancing: see docs/RUNBOOK.md
```

### Task 5 — networking (self-contained, no cluster)
```bash
python tools/nacl_sim.py network/nacl.json network/packets_sample.jsonl --check   # 6/6
python scripts/subnet_calc.py                                                      # subnet table
```

### Task 6 — tests & CI
```bash
make lint && make test        # pyflakes + pytest (runs in CI too)
```
CI (`.github/workflows/ci.yml`): lint → tests → model evaluation from the Release → quality gate
(`score.py --min-validity 95`) → gitleaks. Run it against the untrained base to see the gate fail:
Actions → **ci** → *Run workflow* → `model_source = base`.

### Task 7 — performance & cost
```bash
make load                                                     # Locust, standard marking command
python scripts/cost.py --replicas 2 --cpu-limit 1 --mem-gib 1 --rps <measured>   # from loadtest
```

## Teardown
```bash
make down            # k3d cluster delete cloud-mini
```

See [`docs/RUNBOOK.md`](docs/RUNBOOK.md) for the exact step-by-step (rolling update, recovery,
cold-start timing, load-balancing evidence) and the 4-day commit plan.
