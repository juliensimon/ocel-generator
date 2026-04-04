# PyPI Publishing & Discoverability — Design Spec

**Date**: 2026-04-04
**Status**: Draft
**Package name**: `open-agent-traces` (PyPI) / `ocelgen` (import & CLI)

## Goal

Publish `ocelgen` to PyPI as `open-agent-traces`, make it discoverable across process mining, AI agent, and ML research audiences, and set up automated publishing for future releases. Announce on LinkedIn and X.

## 1. Package Metadata & Renaming

### Name change

- PyPI distribution name: `open-agent-traces` (`pip install open-agent-traces`)
- Python import name: `ocelgen` (unchanged — `src/ocelgen/` stays as-is)
- CLI command: `ocelgen` (unchanged — `[project.scripts]` is independent)

### Metadata additions to `pyproject.toml`

```toml
[project]
name = "open-agent-traces"
license = "MIT"
keywords = [
    "ocel", "process-mining", "agent-traces", "multi-agent",
    "langchain", "synthetic-data", "observability", "anomaly-detection",
    "llm", "ai-agents"
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

[project.urls]
Homepage = "https://github.com/juliensimon/ocel-generator"
Documentation = "https://github.com/juliensimon/ocel-generator/tree/main/docs"
Repository = "https://github.com/juliensimon/ocel-generator"
Issues = "https://github.com/juliensimon/ocel-generator/issues"
Dataset = "https://huggingface.co/datasets/juliensimon/open-agent-traces"
```

## 2. PyPI Account & Trusted Publishers

### Account setup (manual, one-time)

1. Create accounts at pypi.org and test.pypi.org (separate registrations)
2. Enable 2FA on both (required by PyPI since 2024)
3. On each, register a **pending trusted publisher** before first upload:
   - PyPI project name: `open-agent-traces`
   - Owner: `juliensimon`
   - Repository: `ocel-generator`
   - Workflow: `publish.yml`
   - Environment: `release`

### How it works

PyPI's trusted publisher mechanism uses GitHub's OIDC identity provider. When the Actions workflow runs, GitHub generates a short-lived OIDC token proving the request comes from the configured repo/workflow/environment. PyPI verifies this token — no API keys or secrets needed.

## 3. GitHub Actions Workflow

### File: `.github/workflows/publish.yml`

**Trigger**: git tags matching `v*`

**Routing** (single workflow, conditional jobs):
- The workflow uses an `if` condition to check the tag string
- If the tag contains `rc` or `dev` → publish to TestPyPI
- Otherwise → publish to production PyPI
- This is implemented as two separate publish jobs with mutually exclusive `if` guards, not pattern matching on the trigger

### Jobs

**`test`**: Runs existing test suite to gate the release.

**`publish`**: Depends on `test` passing.
- Runs in the `release` GitHub environment
- Uses `uv build` to produce sdist + wheel
- Uses `pypa/gh-action-pypi-publish` action for upload (handles OIDC automatically)

### Tag conventions

```
v0.1.0rc1  →  TestPyPI (dry run)
v0.1.0     →  PyPI (production)
```

### GitHub environment

Create a `release` environment in repo Settings > Environments. No additional protection rules needed — the trusted publisher config on PyPI is the gate.

## 4. GitHub Repo Polish

### Repo description

> Generate realistic multi-agent workflow traces with LLM-enriched content. pip install open-agent-traces

### Repo topics

`ocel` `process-mining` `multi-agent` `langchain` `synthetic-data` `agent-observability` `llm` `ai-agents` `anomaly-detection` `dataset-generation`

### README updates

1. Add PyPI badge: `[![PyPI](https://img.shields.io/pypi/v/open-agent-traces)](https://pypi.org/project/open-agent-traces/)`
2. Change Quick Start to lead with `pip install open-agent-traces`
3. Move `git clone && uv sync` to a "Development setup" section

## 5. Announcements

### LinkedIn (~150 words)

- Hook: scarcity of open agent trace data for building observability/eval tools
- What it does: generates structurally valid, semantically rich multi-agent traces
- Key differentiators: 10 enterprise domains, 3 workflow patterns, OCEL 2.0 compliant, labeled anomalies
- CTA: `pip install open-agent-traces` + links to HF dataset and GitHub repo
- 3-4 hashtags: #AI #agents #opensource #processmining

### X/Twitter (~200 chars + link)

- One-liner on the problem (no open agent trace data)
- What it does
- `pip install open-agent-traces`
- Link to repo

Both announcements will be saved as text files for copy-paste.

## Release checklist (first publish)

1. Create PyPI + TestPyPI accounts, enable 2FA
2. Register pending trusted publisher on both
3. Update `pyproject.toml` metadata
4. Create `release` GitHub environment
5. Add `.github/workflows/publish.yml`
6. Tag `v0.1.0rc1`, verify TestPyPI upload
7. Tag `v0.1.0`, verify PyPI upload
8. Verify `pip install open-agent-traces` works in a fresh venv
9. Update GitHub repo description + topics
10. Update README (badge, pip install quickstart)
11. Draft and post LinkedIn announcement
12. Draft and post X announcement
