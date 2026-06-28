# GAMA

GAMA is a guideline-aware multi-agent framework for biomedical named entity recognition (BioNER). It combines guideline summarization, entity planning, deterministic coding, verification, and F1 evaluation in a reusable Python package.

## Repository Layout

```text
GAMA/
  run_pipeline.py           Runnable pipeline entry point
  src/gama/                 Core package
    pipeline.py             Main dual-loop BioNER pipeline
    planning_agent.py       LLM-based candidate planning
    coding_agent.py         Pydantic-based entity object construction
    verification_agent.py   Semantic/type/structure verification
    guideline_summarizer.py Dataset-driven guideline summarization
    metrics.py              Entity-level precision/recall/F1 utilities
  examples/data/            Small example dataset
  scripts/                  Data conversion utilities
```

## Installation

```bash
python -m venv .venv
pip install -r requirements.txt
```

Set model service credentials with environment variables:

```bash
export BIO_NER_API_KEY="your-api-key"
export BIO_NER_BASE_URL="your-base-url"
```

PowerShell:

```powershell
$env:BIO_NER_API_KEY="your-api-key"
$env:BIO_NER_BASE_URL="your-base-url"
```

## Quick Start

```bash
python run_pipeline.py \
  --dataset examples/data/sample_bioner.json \
  --train-data examples/data/sample_bioner.json \
  --schema GENE \
  --dataset-name sample_bioner \
  --output-dir results \
  --guideline-sample-size 5
```

## Dataset Format

Input datasets are JSON lists. Each item should contain a `text` field and, when evaluation is needed, an `entities` list:

```json
[
  {
    "text": "Mutations in BRCA1 are associated with breast cancer.",
    "entities": [
      {"text": "BRCA1", "type": "GENE"}
    ]
  }
]
```

## Convert IOB/TSV Data

For a single TSV file:

```bash
python scripts/convert_iob_to_json.py --input path/to/test.tsv --output test.json --entity-type GENE
```

For a folder containing `train.tsv`, `devel.tsv`, and `test.tsv`:

```bash
python scripts/convert_iob_to_json.py --folder --input path/to/dataset --output merged.json --entity-type GENE
```

