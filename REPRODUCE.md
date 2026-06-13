# Reproducing The Code Artifact

This file covers the code-facing repository: installation, tests, smoke checks,
and runnable evaluation entry points. Paper sources and generated experiment
artifacts are not part of this tree.

## CPU Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements-cpu.txt
pip install -e ".[dev]"
```

Run CPU tests:

```bash
pytest tests/ -q --ignore=tests/test_integration_identity.py
```

Or:

```bash
make setup-cpu
make test
```

## CUDA Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements-cuda.txt
pip install -e ".[dev,eval,ruler]"
```

Run GPU-dependent checks when CUDA and model checkpoints are available:

```bash
pytest tests/test_integration_identity.py -q
pytest tests/test_triton_chunked.py -q
```

## Smoke Checks

```bash
python scripts/smoke_path_d_wiring.py
python scripts/smoke_chunked_prefill.py
python scripts/smoke_quest_wiring.py
```

Some smoke checks require CUDA, Hugging Face model access, or prepared local
benchmark data.

## Evaluation Runners

Most runners write outputs under `experiments/` by default. That directory is
ignored by Git.

```bash
python scripts/run_longbench.py --help
python scripts/run_ruler.py --help
python scripts/run_infinitebench.py --help
python scripts/run_pathd_ruler.py --help
python scripts/run_vllm_baseline.py --help
```

Prepare RULER data when needed:

```bash
bash scripts/repro/prepare_ruler_data.sh
```

## Benchmarks

```bash
python scripts/benchmark_memory.py --help
python scripts/benchmark_triton_chunked.py --help
python scripts/benchmark_triton_single_launch.py --help
python scripts/run_chunked_benchmark.py --help
```

Benchmark outputs are generated locally and should not be committed.
