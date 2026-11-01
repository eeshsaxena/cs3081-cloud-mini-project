"""Fine-tune the student (EleutherAI/pythia-70m) on the teacher's replies (Task 2).

Sequence-level distillation: each example is the teacher's JSON reply conditioned on the command.
Only training/config.yaml is read for hyperparameters. Everything runs on CPU.

  make train                 # v1, full config
  make train SUBSET=50       # smoke test on the first 50 records
  make train-v2              # v2 (another seed) -> models/v2/, required for Task 4

Outputs:
  models/<version>/                 model + tokenizer files (also uploaded to the models bucket)
  models/<version>.tar.gz           the archive published as a GitHub Release asset
  models/<version>/model.sha256     SHA-256 of that archive (checksum the Release asset must match)
  training/loss.csv                 one average training loss per epoch
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import tarfile

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "worker"))
import storage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_config():
    with open(os.path.join(HERE, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_train_rows(cfg, limit):
    """Read train.jsonl from the datasets bucket, falling back to the local file."""
    try:
        text = storage.read_text(cfg["datasets_bucket"], cfg["train_file"])
    except Exception:
        with open(os.path.join(ROOT, "datasets", "v1", "train.jsonl"), encoding="utf-8") as f:
            text = f.read()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return rows[:limit]


def build_example(tokenizer, row, max_len):
    target = json.dumps({k: row[k] for k in ("intent", "date", "time", "place")}, ensure_ascii=False)
    # The prompt must NOT end in a space: a trailing space becomes its own token when the prompt is
    # tokenised alone, but merges into the next token (" {") inside the full string. That mismatch
    # left the opening brace unlearned and made the model emit EOS immediately at inference. Ending
    # the prompt at "JSON:" and giving the completion the leading space keeps the prompt a clean
    # token prefix of the full sequence, so the mask boundary is exact and the model learns " {...}".
    prompt = f"Command: {row['text']}\nJSON:"
    full = prompt + " " + target + tokenizer.eos_token
    ids = tokenizer(full, truncation=True, max_length=max_len)["input_ids"]
    prompt_len = min(len(tokenizer(prompt)["input_ids"]), len(ids))
    labels = [-100] * prompt_len + ids[prompt_len:]
    return ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--subset", type=int, default=0, help="train on the first N records (smoke test)")
    ap.add_argument("--seed", type=int, default=None, help="override config seed (used for v2)")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--no-upload", action="store_true", help="skip object-storage upload (local only)")
    a = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config()
    seed = a.seed if a.seed is not None else cfg["seed"]
    epochs = a.epochs if a.epochs is not None else cfg["epochs"]
    limit = a.subset if a.subset else cfg["train_limit"]

    random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
    model.train()

    rows = load_train_rows(cfg, limit)
    print(f"[train] {len(rows)} examples, version={a.version}, seed={seed}, epochs={epochs}")

    examples = [build_example(tokenizer, r, cfg["max_seq_len"]) for r in rows]

    class DS(Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, i):
            return examples[i]

    pad_id = tokenizer.pad_token_id

    def collate(batch):
        m = max(len(ids) for ids, _ in batch)
        input_ids, attn, labels = [], [], []
        for ids, lab in batch:
            pad = m - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            labels.append(lab + [-100] * pad)
        return (torch.tensor(input_ids), torch.tensor(attn), torch.tensor(labels))

    loader = DataLoader(DS(), batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    loss_rows = []
    for epoch in range(1, epochs + 1):
        total, nb = 0.0, 0
        for input_ids, attn, labels in loader:
            optim.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            optim.step()
            total += out.loss.item()
            nb += 1
        avg = total / max(nb, 1)
        loss_rows.append((epoch, round(avg, 6)))
        print(f"[train] epoch {epoch}/{epochs} loss {avg:.6f}")

    with open(os.path.join(HERE, "loss.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss"])
        w.writerows(loss_rows)

    out_dir = os.path.join(ROOT, "models", a.version)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[train] saved model + tokenizer to {out_dir}")

    # Archive for the GitHub Release asset + checksum.
    archive = os.path.join(ROOT, "models", f"{a.version}.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for name in sorted(os.listdir(out_dir)):
            if name == "model.sha256":
                continue
            tar.add(os.path.join(out_dir, name), arcname=name)
    digest = hashlib.sha256(open(archive, "rb").read()).hexdigest()
    with open(os.path.join(out_dir, "model.sha256"), "w", encoding="utf-8") as f:
        f.write(f"{digest}  {a.version}.tar.gz\n")
    print(f"[train] {a.version}.tar.gz sha256 {digest}")

    if not a.no_upload:
        try:
            storage.ensure_bucket(cfg["models_bucket"])
            for name in sorted(os.listdir(out_dir)):
                storage.upload_file(cfg["models_bucket"], f"{a.version}/{name}", os.path.join(out_dir, name))
            print(f"[train] uploaded files to {cfg['models_bucket']}/{a.version}/")
        except Exception as e:
            print(f"[train] WARNING: upload skipped ({e}). Files are in {out_dir}.")


if __name__ == "__main__":
    main()
