.PHONY: install fetch data exp aggregate tables test lint check precommit

ARM   ?= A_ssl
SCALE ?= 3
DEVICE ?= cuda
CONFIG ?= configs/base.yaml

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

# Run every committed seed for one (arm, scale). Sequential here; use
# scripts/run_matrix.py + your scheduler to parallelize across GPUs. The seeds
# come from configs/seeds.yaml through run_matrix.py, never from a list here.
# N=3 takes the first three. A failed seed stops the loop rather than letting
# the next one scroll the error away, and the cell list is captured before the
# loop rather than piped into it: a pipeline's status is its last command's, so
# a failing run_matrix.py would otherwise leave make reporting success after
# running nothing.
exp:
	@cells=$$(uv run python scripts/run_matrix.py --arm $(ARM) --scale $(SCALE) $(if $(N),--n-seeds $(N),)) || exit 1; \
	printf '%s\n' "$$cells" | while read -r cell; do \
		echo "=== $$cell ==="; \
		uv run svb run $$cell --config $(CONFIG) --device $(DEVICE) || exit 1; \
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
