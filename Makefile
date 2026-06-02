.PHONY: install data exp aggregate test lint

ARM   ?= A_ssl
SCALE ?= 3
DEVICE ?= cuda
SEEDS ?= 0 1 2 3 4

install:
	uv sync --extra dev

data:
	bash scripts/download_data.sh

# Run all seeds for one (arm, scale). Sequential here; use scripts/run_matrix.py
# + your scheduler to parallelize across GPUs.
exp:
	@for s in $(SEEDS); do \
		echo "=== $(ARM) scale=$(SCALE) seed=$$s ==="; \
		svb run --arm $(ARM) --scale $(SCALE) --seed $$s --config configs/base.yaml --device $(DEVICE); \
	done

aggregate:
	svb aggregate --arm $(ARM) --scale $(SCALE)

test:
	pytest -q

lint:
	ruff check src tests && ruff format --check src tests
