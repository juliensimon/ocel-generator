# PyPI Publishing & Discoverability — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish `ocelgen` to PyPI as `open-agent-traces`, set up automated CI publishing, polish the repo, and draft announcements.

**Architecture:** Update `pyproject.toml` metadata, add a GitHub Actions publish workflow with OIDC trusted publishers, update the README for pip-install-first onboarding, and write announcement copy.

**Tech Stack:** uv (build), pypa/gh-action-pypi-publish (CI), GitHub OIDC trusted publishers, gh CLI (repo metadata)

---

### Task 1: Update `pyproject.toml` metadata

**Files:**
- Modify: `pyproject.toml:1-9` (project name, license, keywords, classifiers, urls)

- [ ] **Step 1: Update the project name and add metadata fields**

In `pyproject.toml`, replace the current `[project]` section's name and add new fields. The final `[project]` section should read:

```toml
[project]
name = "open-agent-traces"
version = "0.1.0"
description = "Mock OCEL 2.0 event log generator for LangChain multi-agent runs"
readme = "README.md"
authors = [
    { name = "Julien Simon", email = "julien@arcee.ai" }
]
requires-python = ">=3.11"
license = "MIT"
keywords = [
    "ocel", "process-mining", "agent-traces", "multi-agent",
    "langchain", "synthetic-data", "observability", "anomaly-detection",
    "llm", "ai-agents",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Testing :: Traffic Generation",
]
dependencies = [
    "pydantic>=2.6",
    "typer>=0.12",
    "rich>=13.0",
    "jsonschema>=4.21",
    "faker>=24.0",
    "openai>=1.0",
    "huggingface_hub>=0.20",
    "pyarrow>=15.0",
    "pyyaml>=6.0",
]
```

- [ ] **Step 2: Add `[project.urls]` section**

Add this after the `[project.scripts]` section, before `[build-system]`:

```toml
[project.urls]
Homepage = "https://github.com/juliensimon/ocel-generator"
Documentation = "https://github.com/juliensimon/ocel-generator/tree/main/docs"
Repository = "https://github.com/juliensimon/ocel-generator"
Issues = "https://github.com/juliensimon/ocel-generator/issues"
Dataset = "https://huggingface.co/datasets/juliensimon/open-agent-traces"
```

- [ ] **Step 3: Verify the build works**

Run:
```bash
uv build
```
Expected: Creates `dist/open_agent_traces-0.1.0.tar.gz` and `dist/open_agent_traces-0.1.0-py3-none-any.whl` without errors.

- [ ] **Step 4: Verify the metadata in the built wheel**

Run:
```bash
unzip -p dist/open_agent_traces-0.1.0-py3-none-any.whl '*/METADATA' | head -30
```
Expected: Shows `Name: open-agent-traces`, `License: MIT`, keywords and classifiers present.

- [ ] **Step 5: Clean up dist and commit**

Run:
```bash
rm -rf dist/
git add pyproject.toml
git commit -m "feat: rename package to open-agent-traces, add PyPI metadata"
```

---

### Task 2: Create GitHub Actions publish workflow

**Files:**
- Create: `.github/workflows/publish.yml`

- [ ] **Step 1: Create the publish workflow**

Create `.github/workflows/publish.yml` with this content:

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - "v*"

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync --extra dev

      - name: Lint
        run: uv run ruff check src tests

      - name: Type check
        run: uv run mypy src

      - name: Test
        run: uv run pytest tests/ -v --tb=short

  publish-testpypi:
    needs: test
    if: contains(github.ref_name, 'rc') || contains(github.ref_name, 'dev')
    runs-on: ubuntu-latest
    environment: release
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Build package
        run: uv build

      - name: Publish to TestPyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/

  publish-pypi:
    needs: test
    if: "!contains(github.ref_name, 'rc') && !contains(github.ref_name, 'dev')"
    runs-on: ubuntu-latest
    environment: release
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Build package
        run: uv build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

- [ ] **Step 2: Validate the workflow syntax**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/publish.yml'))" && echo "YAML valid"
```
Expected: `YAML valid`

- [ ] **Step 3: Commit**

Run:
```bash
git add .github/workflows/publish.yml
git commit -m "ci: add publish workflow with trusted publishers"
```

---

### Task 3: PyPI & TestPyPI account setup (manual)

This task is performed by the user in a browser — no code changes.

- [ ] **Step 1: Create PyPI account**

Go to https://pypi.org/account/register/ — create account, enable 2FA.

- [ ] **Step 2: Create TestPyPI account**

Go to https://test.pypi.org/account/register/ — create account, enable 2FA. (This is a separate account from production PyPI.)

- [ ] **Step 3: Register pending trusted publisher on TestPyPI**

Go to https://test.pypi.org/manage/account/publishing/ and add:
- PyPI project name: `open-agent-traces`
- Owner: `juliensimon`
- Repository name: `ocel-generator`
- Workflow name: `publish.yml`
- Environment name: `release`

- [ ] **Step 4: Register pending trusted publisher on PyPI**

Go to https://pypi.org/manage/account/publishing/ and add the same values as Step 3.

- [ ] **Step 5: Create GitHub `release` environment**

Go to https://github.com/juliensimon/ocel-generator/settings/environments — click "New environment", name it `release`, save. No additional protection rules needed.

---

### Task 4: TestPyPI dry run

- [ ] **Step 1: Tag a release candidate**

Run:
```bash
git tag v0.1.0rc1
git push origin v0.1.0rc1
```

- [ ] **Step 2: Watch the workflow**

Run:
```bash
gh run watch --exit-status
```
Expected: `test` job passes, `publish-testpypi` job passes, `publish-pypi` job is skipped.

- [ ] **Step 3: Verify on TestPyPI**

Run:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ open-agent-traces==0.1.0rc1
ocelgen --help
```
Expected: Package installs, `ocelgen --help` shows the CLI help text.

- [ ] **Step 4: Clean up test install**

Run:
```bash
pip uninstall -y open-agent-traces
```

---

### Task 5: Production PyPI publish

- [ ] **Step 1: Tag the production release**

Run:
```bash
git tag v0.1.0
git push origin v0.1.0
```

- [ ] **Step 2: Watch the workflow**

Run:
```bash
gh run watch --exit-status
```
Expected: `test` job passes, `publish-pypi` job passes, `publish-testpypi` job is skipped.

- [ ] **Step 3: Verify on PyPI**

Run:
```bash
pip install open-agent-traces
ocelgen --help
```
Expected: Package installs from production PyPI, CLI works.

- [ ] **Step 4: Verify PyPI page**

Go to https://pypi.org/project/open-agent-traces/ — confirm:
- Description renders from README
- Sidebar shows project URLs (Homepage, Docs, Repository, Issues, Dataset)
- Classifiers and keywords are visible
- License shows MIT

- [ ] **Step 5: Clean up test install**

Run:
```bash
pip uninstall -y open-agent-traces
```

---

### Task 6: Update GitHub repo metadata

- [ ] **Step 1: Update repo description**

Run:
```bash
gh repo edit juliensimon/ocel-generator \
  --description "Generate realistic multi-agent workflow traces with LLM-enriched content. pip install open-agent-traces"
```

- [ ] **Step 2: Set repo topics**

Run:
```bash
gh repo edit juliensimon/ocel-generator \
  --add-topic ocel \
  --add-topic process-mining \
  --add-topic multi-agent \
  --add-topic langchain \
  --add-topic synthetic-data \
  --add-topic agent-observability \
  --add-topic llm \
  --add-topic ai-agents \
  --add-topic anomaly-detection \
  --add-topic dataset-generation
```

---

### Task 7: Update README for pip-install onboarding

**Files:**
- Modify: `README.md:5-6` (add badge)
- Modify: `README.md:29-34` (rewrite quick start)

- [ ] **Step 1: Add PyPI badge**

In `README.md`, on line 5 (the badge row), add this badge after the Dataset badge:

```markdown
[![PyPI](https://img.shields.io/pypi/v/open-agent-traces)](https://pypi.org/project/open-agent-traces/)
```

The full badge block becomes:
```markdown
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-open--agent--traces-yellow)](https://huggingface.co/datasets/juliensimon/open-agent-traces)
[![PyPI](https://img.shields.io/pypi/v/open-agent-traces)](https://pypi.org/project/open-agent-traces/)
[![CI](https://github.com/juliensimon/ocel-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/juliensimon/ocel-generator/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![OCEL 2.0](https://img.shields.io/badge/OCEL-2.0-orange.svg)](https://www.ocel-standard.org/)
[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20Compatible-lightgrey.svg)](docs/user-guide.md#model-and-endpoint-configuration)
```

- [ ] **Step 2: Rewrite Quick Start to lead with pip install**

Replace the current Quick Start section (lines 29-33):

```markdown
## Quick start

```bash
pip install open-agent-traces
```

### Development setup

```bash
git clone https://github.com/juliensimon/ocel-generator.git && cd ocel-generator
uv sync
```
```

- [ ] **Step 3: Commit**

Run:
```bash
git add README.md
git commit -m "docs: add PyPI badge and pip install quickstart"
```

---

### Task 8: Draft announcements

**Files:**
- Create: `docs/announcements/linkedin.md`
- Create: `docs/announcements/x.md`

- [ ] **Step 1: Create announcements directory**

Run:
```bash
mkdir -p docs/announcements
```

- [ ] **Step 2: Write LinkedIn announcement**

Create `docs/announcements/linkedin.md`:

```markdown
Real agent traces are scarce. If you're building observability, evaluation, or debugging tools for multi-agent systems, you know the pain — production traces are proprietary, and toy examples don't cut it.

I built open-agent-traces to fix this. It generates structurally valid, semantically rich execution traces that look and feel like real multi-agent workflows:

- 10 enterprise domains (customer support, code review, incident response, financial analysis...)
- 3 workflow patterns (sequential, supervisor/worker, parallel fan-out)
- LLM-enriched content — real prompts, completions, tool calls, agent reasoning
- Labeled anomalies for training detectors (wrong tools, skipped steps, timeouts)
- OCEL 2.0 standard — works with PM4Py, Celonis, and other process mining tools

pip install open-agent-traces

Pre-built dataset on Hugging Face: https://huggingface.co/datasets/juliensimon/open-agent-traces
Code: https://github.com/juliensimon/ocel-generator

MIT licensed. Contributions welcome.

#AI #agents #opensource #processmining
```

- [ ] **Step 3: Write X/Twitter announcement**

Create `docs/announcements/x.md`:

```markdown
No open trace data for multi-agent systems? I built a fix.

open-agent-traces generates realistic LLM-enriched execution traces — 10 domains, 3 workflow patterns, labeled anomalies, OCEL 2.0 compliant.

pip install open-agent-traces

https://github.com/juliensimon/ocel-generator
```

- [ ] **Step 4: Commit**

Run:
```bash
git add docs/announcements/
git commit -m "docs: draft LinkedIn and X announcements"
```
