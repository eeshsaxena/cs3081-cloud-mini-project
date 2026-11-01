"""Loads the student model from object storage and runs inference (Task 3).

The model weights are NEVER baked into the image. At start-up the service downloads the files
under MODEL_URI (e.g. s3://models/v1) from SeaweedFS, using credentials taken from environment
variables that come from a Kubernetes Secret.
"""
import json
import os
import tempfile

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")
with open(SCHEMA_PATH, encoding="utf-8") as _f:
    _SCHEMA = json.load(_f)
KEYS = set(_SCHEMA["keys"])
INTENTS = set(_SCHEMA["intents"])


def extract_json(generated):
    """From the first '{' to its matching '}'. Identical to training/predict.py."""
    s = generated.strip()
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return s[start:]


def validate(raw):
    """Return the parsed object if it matches schema.json exactly, else None."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj.keys()) != KEYS:
        return None
    if obj.get("intent") not in INTENTS:
        return None
    for k in ("date", "time", "place"):
        if obj[k] is not None and not isinstance(obj[k], str):
            return None
    if not isinstance(obj["intent"], str):
        return None
    return obj


def _download_model(model_uri):
    assert model_uri.startswith("s3://"), "MODEL_URI must be s3://<bucket>/<prefix>"
    bucket, _, prefix = model_uri[len("s3://"):].partition("/")
    prefix = prefix.rstrip("/") + "/"
    import boto3
    from botocore.client import Config
    s3 = boto3.client(
        "s3", endpoint_url=os.environ["S3_ENDPOINT"],
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    dest = tempfile.mkdtemp(prefix="model-")
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    objs = resp.get("Contents", [])
    if not objs:
        raise RuntimeError(f"no objects under {model_uri}")
    for o in objs:
        name = o["Key"][len(prefix):]
        if not name:
            continue
        s3.download_file(bucket, o["Key"], os.path.join(dest, name))
    return dest


class Runner:
    def __init__(self):
        self.ready = False
        self.model = None
        self.tokenizer = None
        self.max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "64"))

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        local_dir = _download_model(os.environ["MODEL_URI"])
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(local_dir)
        self.model.eval()
        self.ready = True

    def raw_output(self, text):
        """Return the model's raw candidate JSON text for a command."""
        import torch
        prompt = f"Command: {text}\nJSON:"  # no trailing space (must match training)
        enc = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            gen = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new = gen[0][enc["input_ids"].shape[1]:]
        decoded = self.tokenizer.decode(new, skip_special_tokens=True)
        return extract_json(decoded)
