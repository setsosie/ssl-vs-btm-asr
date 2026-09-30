.PHONY: install fetch data exp study aggregate tables compare test lint check precommit

ARM   ?= A_ssl
SCALE ?= 3
DEVICE ?= cuda
# STUDY names an overlay in configs/studies/, laid over configs/base.yaml:
# `make study STUDY=small` is the small study. RESUME=1 continues runs from the
# stages they already finished instead of starting them again.
STUDY ?=
# A misspelt study fails here, naming the file, rather than fifteen cells later
# inside svb run with a bare FileNotFoundError.
ifneq ($(STUDY),)
ifeq ($(wildcard configs/studies/$(STUDY).yaml),)
$(error STUDY=$(STUDY): no such overlay configs/studies/$(STUDY).yaml)
endif
endif
CONFIGS = --config configs/base.yaml $(if $(STUDY),--config configs/studies/$(STUDY).yaml,)
RUN_FLAGS = $(CONFIGS) --device $(DEVICE) $(if $(filter 1,$(RESUME)),--resume,)

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
	uv run python scripts/check_data.py

# Run every committed seed for one (arm, scale). Sequential here; use
# scripts/run_matrix.py + your scheduler to parallelize across GPUs. The seeds
# come from configs/seeds.yaml through run_matrix.py, never from a list here.
# N=3 takes the first three. A failed seed stops the loop rather than letting
# the next one scroll the error away.
# The cell list is captured before the loop, not piped into it: a pipeline's
# status is its last command's, so a failing run_matrix.py would otherwise
# leave make reporting success after running nothing.
exp:
	@cells=$$(uv run python scripts/run_matrix.py --arm $(ARM) --scale $(SCALE) $(if $(N),--n-seeds $(N),)) || exit 1; \
	printf '%s\n' "$$cells" | while read -r cell; do \
		echo "=== $$cell ==="; \
		uv run svb run $$cell $(RUN_FLAGS) || exit 1; \
	done

# Every arm and every seed of one scale, in the matrix's order. The same loop as
# `exp` without the arm filter; on one GPU this is days, so expect to use
# RESUME=1, or the scheduler route in docs/runbooks/small-study.md. Without
# RESUME=1 a cell that already has finished stages refuses to run: `svb run`
# needs --resume or --restart to touch one, and only the first is offered here.
study:
	@cells=$$(uv run python scripts/run_matrix.py --scale $(SCALE) $(if $(N),--n-seeds $(N),)) || exit 1; \
	printf '%s\n' "$$cells" | while read -r cell; do \
		echo "=== $$cell ==="; \
		uv run svb run $$cell $(RUN_FLAGS) || exit 1; \
	done

aggregate:
	uv run svb aggregate --arm $(ARM) --scale $(SCALE)

# Regenerate tables/ from the run artifacts. Nothing in tables/ is hand-typed;
# every number there traces back to a predictions sidecar on disk. Set
# COMPARE_TO to add a seed-for-seed paired permutation test against another arm:
#   make tables SCALE=3 ARM=A_ssl COMPARE_TO=B_btm_ssl
tables:
	uv run svb analyze --arm $(ARM) --scale $(SCALE) $(if $(COMPARE_TO),--compare-to $(COMPARE_TO),)

# The arms side by side, with intervals on the differences that resample seeds
# and test utterances together. Reads every arm that has finished runs.
compare:
	uv run svb compare --scale $(SCALE)

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
