# AI Learning Lab

**A local-first, open-source Python laboratory for learning and experimenting with AI.**

AI Learning Lab is designed for students, researchers, and developers who want to understand practical machine-learning workflows through small, readable, reproducible examples—not opaque demos or paid API calls. The project is organized as an extensible Python package, with tests, examples, documentation, and continuous integration.

> **Status:** MVP development. This is an educational project, not a production sentiment service. Its small built-in sentiment dataset is for demonstrating the workflow; it does not establish real-world model quality.

## What this project is about

AI Learning Lab aims to make it easier to learn how AI software is built as well as how an individual model works. Users can install the package locally, run an example, inspect the implementation, change it, and verify behavior with tests.

The project emphasizes:

- **Learning by doing:** understandable code and runnable examples.
- **Local-first use:** no commercial APIs, credentials, or network service are required by the MVP.
- **Reproducibility:** fixed random seeds where applicable, documented inputs, and captured environment metadata.
- **Modularity:** separate areas for classification, NLP, computer vision, benchmarking, and utilities.
- **Responsible claims:** no claimed accuracy, performance, popularity, or adoption figures without evidence.
- **Open collaboration:** contribution guidance, issue templates, a code of conduct, and automated checks.

It is intended for AI/ML learners, developers building a portfolio, researchers prototyping small experiments, and open-source contributors. It is not a substitute for model governance, domain review, or production MLOps.

## MVP features

- **Sentiment classification:** scikit-learn TF-IDF and logistic regression, with fit, predict, and probability methods. The ten hand-authored examples are for demonstration and tests, not representative training data.
- **Classical classification:** Iris-dataset logistic regression or random forest examples, with a deterministic held-out evaluation helper.
- **Benchmarking:** local inference latency measurements, Python allocation estimates, runtime/dependency metadata, and JSON/CSV export. Results vary by machine and should only be compared under controlled conditions.
- **Quality scaffolding:** pytest tests, Ruff linting, GitHub Actions workflows, packaging metadata, documentation, and contribution templates.

The MVP does not download pretrained models or datasets and does not require paid services. PyTorch, Transformers, computer-vision inference, chatbot functionality, regression/K-Means examples, and a user interface are future possibilities—not implemented features.

## Installation

Requirements: Python 3.11 or later and pip.

```bash
git clone https://github.com/QRXOP/ai-learning-lab.git
cd ai-learning-lab
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the package and runtime dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

For tests and linting:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

Run the examples from the repository root:

```bash
python examples/sentiment_demo.py
python examples/classification_demo.py
python examples/benchmark_demo.py
```

The benchmark example writes `benchmark_results.json` and `benchmark_results.csv` to the current directory (both are git-ignored).

### Sentiment analysis

```python
from ai_learning_lab.nlp.sentiment import SentimentClassifier

# Tiny built-in data is for demonstrating the API only.
model = SentimentClassifier.from_demo_dataset()
labels = model.predict([
    "I love how useful and clear this is.",
    "This is frustrating and difficult to use.",
])
print(labels)
print(model.classes)  # column order for predict_proba
```

For meaningful results, use a sufficiently large, representative, appropriately licensed labeled dataset and evaluate on data held out from training. Predictions and `predict_proba` values are model outputs, not calibrated truth or professional advice. With only ten training sentences, the demo model often gets simple inputs wrong. For example, it may label a neutral sentence as negative.

### Scikit-learn classification

```python
from ai_learning_lab.classification.sklearn_models import (
    evaluate_iris_classifier,
    train_iris_classifier,
)

model = train_iris_classifier(model_name="logistic_regression")
metrics = evaluate_iris_classifier(model_name="random_forest")
print(metrics)
```

The Iris dataset is bundled with scikit-learn and is loaded locally. The evaluation helper uses a fixed random seed for its split and model configuration.

### Benchmarking

```python
from ai_learning_lab.benchmarking import export_benchmark, run_benchmark

result = run_benchmark(repeats=20, warmup=3)
print(result["mean_latency_ms"])
json_path, csv_path = export_benchmark(
    result, output_dir="results", file_name="iris_inference"
)
```

See [docs/benchmarking.md](docs/benchmarking.md) for methodology and limitations.

## Project layout

```text
.
├── .github/                 # CI, issue forms, pull-request template
├── assets/                  # Reviewed project assets
├── docs/                    # Roadmap, methodology, publishing guide
├── examples/                # Runnable demonstrations
├── notebooks/               # Future educational notebooks
├── src/ai_learning_lab/
│   ├── benchmarking/        # Timing and result export
│   ├── classification/      # Scikit-learn classification examples
│   ├── computer_vision/     # Reserved for future CV work
│   ├── nlp/                 # Sentiment pipeline
│   └── utils/               # Runtime metadata helpers
└── tests/                   # Automated tests
```

## Development and checks

Install development dependencies, then run:

```bash
python -m pytest
ruff check .
```

GitHub Actions runs linting, tests, and all three examples on Python 3.11, 3.12, and 3.13 for pushes to `main` and for pull requests. Check the workflow results in GitHub before merging or releasing.

## License and dependencies

Original project code is licensed under the MIT License; see [LICENSE](LICENSE). Dependencies are separately licensed by their maintainers. The MVP uses scikit-learn and its built-in Iris dataset; consult their upstream license and dataset documentation before redistribution or use in another context. No externally hosted model weights or third-party datasets are included.

Review software, model, and dataset licenses separately before adding optional integrations. See [SECURITY.md](SECURITY.md) for security guidance.

## Roadmap

- **v0.1.0 — MVP:** sentiment demo, scikit-learn classification, benchmark export, tests, documentation, and CI.
- **v0.2.0 — Learning modules:** expand classical ML examples and add educational notebooks; investigate computer vision only after reviewing model and dataset licenses.
- **v1.0.0 — Stable foundation:** stabilize public APIs, improve evaluation and documentation, and consider optional open-source model integrations.

These are goals, not current features or release commitments. See [docs/roadmap.md](docs/roadmap.md).

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and use the issue and pull-request templates. Suggested starter labels include `good first issue`, `help wanted`, `documentation`, `tests`, and `enhancement`; maintainers can create these in GitHub.

## Publishing and visibility

The repository is currently private. An administrator can change its visibility in **Settings → General → Danger Zone → Change repository visibility** after reviewing all files and commit history. Follow [docs/publishing.md](docs/publishing.md) and do not make the repository public until secrets, licenses, and content have been reviewed.

## Limitations

- The sentiment corpus contains only ten hand-authored examples. It is not suitable for measuring real-world model quality or training a useful application.
- The chatbot, image classifier, expanded ML playground, and interactive interface are not implemented.
- Benchmarking is a small local demonstration. `tracemalloc` reports Python-tracked allocations, not total process or accelerator memory, and adds overhead to the timed loop. Timing varies with hardware, load, software versions, and sample size.
- No accuracy or latency guarantee is made. Evaluate models against suitable held-out data and use case-specific criteria.

## Acknowledgments

AI Learning Lab is independently developed. It uses open-source Python libraries and scikit-learn's built-in Iris dataset for educational examples. No private project code, credentials, or confidential datasets are intended to be included.
