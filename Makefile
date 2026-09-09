.PHONY: install fetch data exp aggregate tables test lint check precommit

ARM   ?= A_ssl
SCALE ?= 3
DEVICE ?= cuda
SEEDS ?= 0 1 2 3 4

# --frozen: install exactly what uv.lock pins and fail if it has drifted from
# pyproject.toml, rather than silently re-resolving. Reproducibility is the
# point of this repo.
install:
	uv sync --frozen --extra dev

# Download and extract the held-out OpenSLR corpora into $OPENSLR_ROOT. Common
# Voice is not fetched here: it is only distributed through Mozilla Data
# Collective, behind an account and a terms acceptance no script can give.
fetch:
	uv run python scripts/fetch_openslr.py --root $${OPENSLR_ROOT:?set OPENSLR_ROOT to the extraction root}

# Report split sizes for every configured language, from metadata only.
data:
	python scripts/check_data.py

# Run all seeds for one (arm, scale). Sequential here; use scripts/run_matrix.py
# + your scheduler to parallelize across GPUs.
exp:
	@for s in $(SEEDS); do \
		echo "=== $(ARM) scale=$(SCALE) seed=$$s ==="; \
		uv run svb run --arm $(ARM) --scale $(SCALE) --seed $$s --config configs/base.yaml --device $(DEVICE); \
	done

aggregate:
	uv run svb aggregate --arm $(ARM) --scale $(SCALE)

# Regenerate tables/ from the run artifacts. Nothing in tables/ is hand-typed;
# every number there traces back to a predictions sidecar on disk. Set
# COMPARE_TO to add a seed-for-seed paired permutation test against another arm:
#   make tables SCALE=3 ARM=A_ssl COMPARE_TO=B_btm_ssl
tables:
	uv run svb analyze --arm $(ARM) --scale $(SCALE) $(if $(COMPARE_TO),--compare-to $(COMPARE_TO),)

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
