# Contributing to Jatayu

Six people, eight weeks, one repository. These rules exist so we spend that time
building rather than resolving conflicts.

---

## Setup (once)

```bash
git clone https://github.com/pandatech717-code/jatayu.git
cd jatayu
uv sync --dev
uv run pre-commit install     # do not skip — this is what stops bad commits
cp .env.example .env
```

Check it works:

```bash
uv run pytest -m "not gpu"
uv run uvicorn jatayu.api.main:app --reload    # then http://localhost:8000/docs
```

If you are working on models, add the heavy stack: `uv sync --extra models`.
Nobody else needs it.

---

## Who owns what

Edit files in your own area freely. Touching someone else's area means tagging
them on the PR.

| Area | Owner | Directories |
|---|---|---|
| Vision & adaptation | Srijoy | `training/clip_bigearthnet/`, `src/jatayu/tools/vqa.py`, `grounding.py` |
| Temporal & change | Srijoy | `src/jatayu/tools/change_vqa.py`, `fusion.py`, `training/lora_vqa/` |
| Orchestration & geospatial | Sumit | `src/jatayu/io/`, `controller/`, `api/`, **`schemas.py`** |
| Data & evaluation | Sumit | `eval/`, `scripts/`, `docs/datasets.md`, `docs/evaluation.md` |
| Frontend & docs | Apoorv | `app/`, `README.md`, `CONTRIBUTING.md` |
| Reporting & demo | Apoorv | `src/jatayu/report/`, `assets/`, demo materials |

### `src/jatayu/schemas.py` is special

Every module depends on it. Changing it is a team decision:

1. Raise it in the team chat first, with what you need and why.
2. Open the PR, tag **C**, and use the `contract` label.
3. Wait for approval. Do not merge on a self-review.

Adding an optional field with a default is usually fine. Renaming or removing a
field breaks five people's code simultaneously.

---

## Branches and commits

Branch off `main`, one branch per piece of work:

```
feat/io-geotiff-loader
feat/grounding-box-parsing
fix/router-two-image-gate
docs/eval-table-week3
chore/ci-cache
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) —
`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`. Not enforced by CI, but
it makes the history readable when you are writing the final report at 2 a.m.

**Never commit:** datasets, model weights, `.tif` files, `.env`, API keys, or
notebook outputs. `.gitignore` and the pre-commit hooks cover most of it, but
they are a safety net, not a substitute for looking at `git status`.

---

## Pull requests

`main` is protected: no direct pushes, one approval required.

Keep PRs small. A 200-line PR gets a real review; a 2,000-line PR gets an
approval nobody read. If a piece of work is going to be large, split it — the
contract exists precisely so pieces can land independently.

Before you open one:

```bash
uv run ruff check . --fix
uv run ruff format .
uv run pytest -m "not gpu"
```

CI runs the same three things. If they pass locally they pass in CI.

### Reviewing

You are not being asked to prove the code is perfect. Look for: does it honour
the contract, will it break someone else, is there a test, would you understand
it in three weeks. Approve if yes. Speed matters more than rigour at this scale —
a PR sitting unreviewed for two days blocks a person.

---

## Adding a new tool

The whole point of the contract is that this takes one file and no changes
anywhere else.

1. Create `src/jatayu/tools/<name>.py`.
2. Write `run(req: ToolRequest) -> ToolResult`, decorated with `@register(...)`
   and `@timed`.
3. Import it in `src/jatayu/tools/__init__.py`.
4. Done. The registry test suite picks it up automatically, the router can select
   it, the UI renders it, and the report generator handles its output.

Write the stub version first, merge it, then replace the body with the real
implementation in a second PR. That way the rest of the team unblocks immediately.

```python
@register(
    TaskName.MY_TASK,
    families={TaskFamily.SINGLE_IMAGE},
    description="One or two sentences, written for the router LLM. Say what "
                "question this answers, not how the model works.",
)
@timed
def run(req: ToolRequest) -> ToolResult:
    ...
```

---

## Testing

```bash
uv run pytest -m "not gpu"      # fast suite: no models, no data, seconds
uv run pytest                    # everything, needs GPU and weights
uv run pytest --cov=jatayu       # with coverage
```

Mark anything needing a GPU or downloaded weights with `@pytest.mark.gpu` so CI
skips it. CI runners have no GPU.

Prefer testing pure functions over mocking model inference. `fusion.classify`
takes arrays and returns arrays specifically so its physics can be tested without
opening a single raster — do the same where you can.

---

## Benchmarks

If your PR changes a model or a tool, put the before/after numbers in the PR
description and update `docs/evaluation.md` **in the same PR**. A score table
that drifts out of date is worse than not having one, because we will quote it in
the pitch.

Every number needs: benchmark, split, sample count, date, and commit hash. A
number without those is not a result.

---

## Using Claude Code

`CLAUDE.md` at the repo root holds the project context. Read it before your first
session; update it when a convention changes.

Good uses: boilerplate, Streamlit layouts, rasterio utilities, test scaffolding,
CI configs, docstrings, GeoJSON handling.

Bad uses: choosing a model architecture, picking hyperparameters, debugging a
training run that will not converge, setting fusion thresholds. Those need your
judgement, and a confident wrong answer there costs a week.

Always read generated code before committing it. You are the author on the PR.
