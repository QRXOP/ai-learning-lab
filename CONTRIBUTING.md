# Contributing to AI Learning Lab

Thanks for your interest. This project values small, readable, well-tested examples over breadth.

## Getting set up

```bash
git clone https://github.com/QRXOP/ai-learning-lab.git
cd ai-learning-lab
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check .
python -m pytest
python examples/sentiment_demo.py
python examples/classification_demo.py
python examples/benchmark_demo.py
```

All of these run in CI on Python 3.11–3.13.

## Guidelines

- **Keep it educational.** Prefer clear code and docstrings that explain *why* over clever code.
- **Be reproducible.** Seed randomness, and record any data source and its version.
- **Test what you add.** New public functions need tests, including invalid-input cases.
- **No unbacked claims.** Don't state accuracy, speed, or adoption figures without a reproducible way to check them.
- **Licenses matter.** Only add models, datasets, or assets whose licenses permit it, and link the license in the PR.
- **No secrets or private data.** Never commit credentials, API keys, or personal data.
- **Local-first.** Core examples must run offline without paid services. Optional integrations go behind optional dependencies.

## Issues and labels

Use the issue forms for bugs and feature requests. Maintainers may apply `good first issue`, `help wanted`, `documentation`, `tests`, or `enhancement`.

## Conduct

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
