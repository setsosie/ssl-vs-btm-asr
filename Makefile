.PHONY: install data exp aggregate test lint check precommit

ARM   ?= A_ssl
SCALE ?= 3
DEVICE ?= cuda
SEEDS ?= 0 1 2 3 4

# --frozen: install exactly what uv.lock pins and fail if it has drifted from
# pyproject.toml, rather than silently re-resolving. Reproducibility is the
# point of this repo.
install:
	uv sync --frozen --extra dev

data:
	bash scripts/download_data.sh

# Run all seeds for one (arm, scale). Sequential here; use scripts/run_matrix.py
# + your scheduler to parallelize across GPUs.
exp:
	@for s in $(SEEDS); do \
		echo "=== $(ARM) scale=$(SCALE) seed=$$s ==="; \
		uv run svb run --arm $(ARM) --scale $(SCALE) --seed $$s --config configs/base.yaml --device $(DEVICE); \
	done

aggregate:
	uv run svb aggregate --arm $(ARM) --scale $(SCALE)

test:
	uv run pytest

# Same three commands, over the same paths, as the `checks` job in
# .github/workflows/ci.yml. If this passes locally, CI passes.
lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src tests

# Everything CI runs.
check: lint test

# Run the git hooks against the whole tree without needing them installed.
precommit:
	uv run pre-commit run --all-files
