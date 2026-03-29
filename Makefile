.PHONY: install install-dev lint format typecheck test test-cov clean docker docker-gpu api

# ── Setup ──────────────────────────────────────────────────────────────────────

install:
	pip install -e .

install-dev:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install torch-geometric
	pip install -e ".[dev,api]"

# ── Code quality ───────────────────────────────────────────────────────────────

lint:
	ruff check spatialppiv2/

format:
	ruff format spatialppiv2/

typecheck:
	mypy spatialppiv2/ --ignore-missing-imports

# ── Tests ──────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=spatialppiv2 --cov-report=term-missing --cov-report=html

# ── Docker ─────────────────────────────────────────────────────────────────────

docker:
	docker build -t spatialppiv2:latest .

docker-gpu:
	docker build -f Dockerfile.gpu -t spatialppiv2-gpu:latest .

up:
	docker compose up api

up-gpu:
	docker compose --profile gpu up api-gpu

# ── Dev server ─────────────────────────────────────────────────────────────────

api:
	sppi-api --host 0.0.0.0 --port 8000 --device cpu

# ── Data pipeline ──────────────────────────────────────────────────────────────

download-pdbs:
	sppi-pdbs

score:
	sppi-score

eval:
	sppi-eval --save-figs

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build *.egg-info
