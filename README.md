# HALO

HALO is a research-preview Python package for long-context LLM KV-cache
tiering and offloading. The repository now contains the code-facing artifact:
runtime modules, baseline wrappers, configs, runnable scripts, examples, and
tests.

## What Is Included

- `halo/`: core runtime code, including policy wrapping, KV-cache variants,
  chunked attention, tiered storage, scoring, demotion, and refetch helpers.
- `baselines/`: baseline cache and evaluation wrappers.
- `configs/`: model, task, and HALO configuration files.
- `scripts/`: runnable evaluation, benchmark, smoke-test, diagnostic, and data
  preparation utilities.
- `tests/`: unit and integration tests for the code paths.
- `examples/`: quickstart notebook.

Paper sources, generated paper tables/figures, experiment outputs, logs, and
supplement archives are intentionally excluded from this code repository.

## Install

CPU test environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements-cpu.txt
pip install -e ".[dev]"
```

CUDA environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements-cuda.txt
pip install -e ".[dev,eval,ruler]"
```

## Quick API

```python
from halo import HALOConfig, wrap_with_halo

cfg = HALOConfig(chunked=True, hot_ratio=0.25)
model = wrap_with_halo(hf_causal_lm, cfg)
```

## Common Commands

```bash
make setup-cpu
make test

# CUDA and model-cache dependent
make setup-cuda
make test-gpu
```

Useful script entry points:

```bash
python scripts/run_longbench.py --help
python scripts/run_ruler.py --help
python scripts/benchmark_triton_chunked.py --help
python scripts/smoke_path_d_wiring.py
```

## CLI Entry Points

After installation:

```bash
halo-trace --help
halo-eval --help
```
