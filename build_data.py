#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build data.json for the DaruSim AI web app.

Takes the 241 drugs + 3072-dim cached embeddings and writes a compact
single file: first 768 dims (Matryoshka truncation, valid for
gemini-embedding-001) rounded to 4 decimals. Also validates that retrieval
quality survives the truncation, using the same naproxen test question.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.expanduser("~/workspace/custom-connectors/custom.gemini/bin"))
from gemini import api_post

RAG_DIR = os.path.expanduser("~/workspace/darusim-rag")
APP_DIR = os.path.expanduser("~/workspace/darusim-ai-app")
DIMS = 768

docs = json.load(open(os.path.join(RAG_DIR, "drugs.json"), encoding="utf-8"))
emb = json.load(open(os.path.join(RAG_DIR, "embeddings.json"), encoding="utf-8"))

out = []
for d in docs:
    v = emb[d["id"]][:DIMS]
    out.append({
        "id": d["id"],
        "fa": d["fa"],
        "en": d["en"],
        "cat": d["cat"],
        "text": d["text"],
        "vec": [round(x, 4) for x in v],
    })

os.makedirs(APP_DIR, exist_ok=True)
path = os.path.join(APP_DIR, "data.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
print("wrote", path, round(os.path.getsize(path) / 1e6, 2), "MB")

# Validate retrieval with truncated vectors.
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

q = api_post("models/gemini-embedding-001:embedContent",
             {"model": "models/gemini-embedding-001",
              "content": {"parts": [{"text": "ناپروکسن برای درد قاعدگی خوبه؟"}]}})
qvec = q["embedding"]["values"][:DIMS]
ranked = sorted(out, key=lambda d: cosine(qvec, d["vec"]), reverse=True)
print("top-4 with 768-dim truncated vectors:",
      "، ".join(d["fa"] for d in ranked[:4]))
