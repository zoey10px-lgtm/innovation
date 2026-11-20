# Innovation

Independent APCTT TRL evidence extraction and cross-verification experiment.

The engine retrieves sentence windows using an encoder and a cross-encoder. Each window has a center sentence and one neighbor in each direction. The shortlist covers up to five pages. Qwen proposes one existing sentence ID per supported criterion. DeBERTa verifies the exact center sentence against the target and criterion and checks that work was performed for C03–C09. A second Qwen call reviews the surviving citations. The output reports the highest criterion both checks accepted, with the original excerpt and page, or abstains.

The diagnostic CLI covers C01–C09. The outer worker enforces a default 180-second wall-time budget including model load. The target profile runs retrieval on CUDA, releases retrieval models before loading Qwen, and uses CPU for NLI.

## Target profile

Use the local checkpoint directories for `scincl`, `bge-reranker-v2-m3`, `qwen3-4b-instruct-2507-nf4`, and `deberta-v3-large-zeroshot-v2.0-c`. Install dependencies before going offline. Point Qwen at the catalog-pinned publisher checkpoint. The CLI quantizes it at load time using bitsandbytes NF4 with double quantization and BF16 compute. See `model-profile.json` for pinned revisions.

```sh
python -m innovation --pdf report.pdf --target "Assessed technology" \
  --encoder /models/scincl --reranker /models/bge-reranker-v2-m3 \
  --qwen /models/qwen3-4b-instruct-2507-nf4 \
  --nli /models/deberta-v3-large-zeroshot-v2.0-c
```

## Validation

```sh
python -m unittest discover -s tests -v
python benchmarks/run.py --output results -- --encoder /models/scincl \
  --reranker /models/bge-reranker-v2-m3 --qwen /models/qwen3-4b-instruct-2507-nf4 \
  --nli /models/deberta-v3-large-zeroshot-v2.0-c
```

Place the 11 PDFs at the paths in `benchmarks/cases/`. Original filenames and checksums are listed in `benchmarks/pdf_sources.csv`. Reference TRLs are provisional. The target GPU profile requires separate validation. Local-model CPU diagnostics use MiniLM, the MS MARCO MiniLM reranker, Qwen3-1.7B, and DeBERTa Large with `--qwen-device cpu`.

## Optional LoRA experiment

The default pipeline uses unmodified Qwen weights. To fine-tune locally, install `peft` on a CUDA machine. Run `python -m training.prepare labeled_pairs.jsonl train.jsonl` on labeled evidence pairs with `target`, `criterion_id`, `criterion`, `passage`, binary `label`, and source `group`. Retain source grouping for synthetic derivatives, and reserve independent groups for development and final evaluation.

Run `python -m training.train_lora --qwen /models/qwen --train train.jsonl --dev dev.jsonl --output adapter`. To evaluate an adapter, pass `--qwen-adapter adapter` to the assessment CLI. The offline data conversion was checked with 468 training and 18 validation examples from earlier evidence-pair data, with no shared group identifiers. No LoRA weights have been trained for this repository.

CUDA runs record PyTorch peak allocated and reserved memory. Validate the full target GPU profile with the exact checkpoint exports before interpreting those measurements as an 8 GB deployment result. The kit adapter entry point is `innovation.adapter:AssessmentModule`. Set `APCTT_MODEL_STORE` to the catalog-layout model directory. C10–C12 are surfaced as insufficient evidence by this TRL-only experiment. The kit adapter and official packaging still require a CUDA kit smoke run.
