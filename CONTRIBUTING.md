# Contributing to SpatialPPIv2

Thank you for your interest in contributing! Below are the guidelines to keep the codebase consistent and the CI green.

---

## Development setup

```bash
git clone https://github.com/NaMEleSs0616/SpatialPPIv2.git
cd SpatialPPIv2

python -m venv .venv && source .venv/bin/activate

# CPU-only (fastest for development)
make install-dev
```

---

## Code style

We use **Ruff** for linting and formatting (replaces flake8 + black + isort).

```bash
make lint      # check for issues
make format    # auto-fix formatting
make typecheck # mypy type checking
```

All three must pass before opening a PR. The CI will enforce them.

---

## Tests

```bash
make test       # run the full test suite
make test-cov   # run with HTML coverage report (opens htmlcov/index.html)
```

New features must include tests. Keep test files in `tests/` with the prefix `test_`.  
Fixtures shared across test files go in `tests/conftest.py`.

---

## Pull request checklist

- [ ] `make lint` passes (no ruff errors)
- [ ] `make format` produces no diff
- [ ] `make test` passes (all tests green)
- [ ] New public functions/classes have docstrings
- [ ] Config-driven behaviour uses `config/default.yaml` — no hardcoded paths or hyperparameters
- [ ] Large binary files (checkpoints, PDBs) are **never** committed — add to `.gitignore` if needed

---

## Branching

| Branch | Purpose |
|--------|---------|
| `main` | Stable, always passing CI |
| `develop` | Integration branch for features |
| `feature/<name>` | Individual feature branches — PR into `develop` |
| `fix/<name>` | Bug-fix branches |

---

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add pLDDT confidence weighting to node features
fix: handle missing chain in extractPDB
docs: update README Quick Start section
test: add edge cases for NT-Xent loss
refactor: split run_scoring into loader and scorer modules
```

---

## Reporting issues

Please include:
1. Python version (`python --version`)
2. PyTorch + PyG versions (`pip show torch torch-geometric`)
3. Minimal reproduction steps
4. Full traceback

---

## License

By contributing you agree that your contributions will be licensed under the MIT License.
