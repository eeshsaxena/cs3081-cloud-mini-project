#!/usr/bin/env bash
# Apply namespaces, secrets, SeaweedFS, seed buckets + model, then deploy the service.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

kubectl apply -f k8s/00-namespaces.yaml
bash scripts/create_secrets.sh
kubectl apply -f k8s/seaweedfs.yaml
kubectl -n data rollout status deploy/seaweedfs --timeout=180s

# Seed buckets + model through a temporary port-forward (python + boto3 run on the host).
kubectl -n data port-forward svc/seaweedfs 8333:8333 >/tmp/seaweed-pf.log 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null || true' EXIT
sleep 4
export S3_ENDPOINT=http://localhost:8333
python scripts/init_storage.py
bash scripts/fetch_model.sh          # get models/v1 (+v2) from the GitHub Release if not present
python scripts/seed_model.py         # upload local models/* into the models bucket
kill $PF 2>/dev/null || true; trap - EXIT

kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/ingress.yaml -f k8s/hpa.yaml
kubectl -n app rollout status deploy/command-parser --timeout=300s
