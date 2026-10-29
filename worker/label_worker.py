"""Parallel teacher-labelling worker (Task 1).

Each worker handles one shard of the command pool. A command is in this worker's shard when
its line number (counting from 0) modulo SHARD_COUNT equals SHARD_INDEX. For each command it
sends the teacher prompt to Ollama (model phi3.5, temperature 0, seed 42) and writes one line

    {"id": "...", "text": "...", "teacher_raw": "<the teacher's exact reply>"}

to its own object, datasets/teacher/shard-<SHARD_INDEX>.jsonl. The raw reply is stored exactly
as generated; it is never edited here.

Environment variables
  SHARD_INDEX / SHARD_COUNT   which shard and how many (SHARD_INDEX may also come from an
                              indexed Job via JOB_COMPLETION_INDEX)
  OLLAMA_HOST                 default http://localhost:11434 (use http://host.k3d.internal:11434
                              from inside the cluster when Ollama runs on the laptop)
  TEACHER_MODEL               default phi3.5
  PROMPT_FILE                 default prompts/teacher_prompt.txt
  DATASETS_BUCKET             default datasets
  S3_ENDPOINT                 if set, read/write via SeaweedFS; otherwise use local files
"""
import json
import os
import sys
import time
import urllib.request

import storage

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "phi3.5")
PROMPT_FILE = os.environ.get("PROMPT_FILE", "prompts/teacher_prompt.txt")
DATASETS_BUCKET = os.environ.get("DATASETS_BUCKET", "datasets")
SEED = int(os.environ.get("TEACHER_SEED", "42"))
MAX_TOKENS = int(os.environ.get("TEACHER_MAX_TOKENS", "128"))


def shard_index():
    for var in ("SHARD_INDEX", "JOB_COMPLETION_INDEX"):
        if os.environ.get(var) not in (None, ""):
            return int(os.environ[var])
    sys.exit("SHARD_INDEX (or JOB_COMPLETION_INDEX) must be set")


def load_prompt():
    with open(PROMPT_FILE, encoding="utf-8") as f:
        return f.read()


def call_teacher(prompt_template, command):
    prompt = prompt_template.replace("{{COMMAND}}", command)
    payload = json.dumps({
        "model": TEACHER_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",  # constrain decoding to one valid JSON object (no trailing prose)
        "options": {"temperature": 0, "seed": SEED, "num_predict": MAX_TOKENS},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "").strip()


def read_command_pool():
    """Read commands_pool.jsonl from the datasets bucket, or from the local kit as a fallback."""
    try:
        text = storage.read_text(DATASETS_BUCKET, "commands_pool.jsonl")
    except Exception:
        with open(os.path.join("data", "commands_pool.jsonl"), encoding="utf-8") as f:
            text = f.read()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def already_done(local_path):
    done = set()
    if os.path.exists(local_path):
        with open(local_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["id"])
                    except Exception:
                        pass
    return done


def main():
    idx = shard_index()
    count = int(os.environ.get("SHARD_COUNT", "2"))
    prompt_template = load_prompt()
    commands = read_command_pool()
    mine = [(i, c) for i, c in enumerate(commands) if i % count == idx]

    storage.ensure_bucket(DATASETS_BUCKET)
    local_path = os.path.join("datasets", "teacher", f"shard-{idx}.jsonl")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    done = already_done(local_path)

    key = f"teacher/shard-{idx}.jsonl"
    print(f"[shard {idx}/{count}] {len(mine)} commands, {len(done)} already done", flush=True)

    with open(local_path, "a", encoding="utf-8") as out:
        for n, (i, cmd) in enumerate(mine, 1):
            if cmd["id"] in done:
                continue
            for attempt in range(3):
                try:
                    raw = call_teacher(prompt_template, cmd["text"])
                    break
                except Exception as e:  # transient Ollama / network error -> retry
                    print(f"[shard {idx}] retry {cmd['id']} ({e})", flush=True)
                    time.sleep(2 * (attempt + 1))
            else:
                raw = ""  # give up after retries; clean.py will reject an empty reply
            out.write(json.dumps({"id": cmd["id"], "text": cmd["text"], "teacher_raw": raw},
                                 ensure_ascii=False) + "\n")
            out.flush()
            if n % 25 == 0:
                print(f"[shard {idx}] {n}/{len(mine)}", flush=True)
                storage.upload_file(DATASETS_BUCKET, key, local_path)

    storage.upload_file(DATASETS_BUCKET, key, local_path)
    print(f"[shard {idx}] done -> {DATASETS_BUCKET}/{key}", flush=True)


if __name__ == "__main__":
    main()
