# Speculative Decoding — Lesson 22

Workshop and assignment for the speculative-decoding lesson of the Nebius ML course.

## Contents

| Path | What it is |
|---|---|
| `workshop.ipynb` | In-class workshop notebook |
| `spec_decoding.ipynb` | Speculative decoding implementation |
| `solution_1.1.ipynb` | Solution notebook for task 1.1 |
| `*.md` | Instructions, plan, insights and output notes |

## Setup

```bash
cp .env.example .env   # then fill in HF_TOKEN
```

The notebooks read the HuggingFace token from the environment:

```python
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])
```

`.env` is git-ignored; never commit real credentials.
