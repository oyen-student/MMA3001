# MMA3001 Project Context for Codex

## Purpose of this file

Use this as persistent project context when working in this repository. It captures the project's goals, data assumptions, local development setup, and known risks. It is a working summary, not a substitute for inspecting the current code. **Always inspect the current branch and relevant source files before making changes.** Treat source code and executed tests as more authoritative than historical notes here.

## Project objective

This is an MMA3001 university machine-learning / numerical-methods project. The intended task is to build and evaluate a regression pipeline that predicts a selected hydrodynamic coefficient from the DTMB 5415 hull-shape dataset.

The user wants to understand the engineering meaning of the work, so:
- Prefer physically interpretable explanations and variables where possible.
- Explain important modelling decisions, preprocessing, data leakage, metrics, and plots.
- Do not claim that the 27 inputs are direct dimensions such as length, beam, or draft. They are normalized shape-design variables, and their individual physical interpretations have not been verified.
- Do not change datasets or expand the project scope without discussing the trade-off first. The project is intended to be completed using the existing dataset.

## Dataset

Source: https://zenodo.org/records/19222008

Expected local folder layout:
```
data/
  dr-benchmark-5415hull/
    DTMB5415-database.mat
```

The dataset is large (approximately 1 GB) and should not be committed to Git. README instructions should tell the user to download the Zenodo archive, extract it, and put the full extracted folder named `dr-benchmark-5415hull` inside the repository's `data` folder.

Dataset facts from prior inspection:
- 9,001 hull configurations/samples.
- `variablesMatrix`: 27 × 9001 normalized hull shape-design variables.
- `xMatrix`, `yMatrix`, `zMatrix`: each 2250 × 9001 surface-coordinate arrays.
- `coefStatMatrix`: 3 × 9001 integrated coefficient array.
- `preMatrix0250`, `preMatrix0330`, `preMatrix0410`: each 2250 × 9001 pressure-related arrays.
- `intfrMatrix0250`, `intfrMatrix0330`, `intfrMatrix0410`: each 1500 × 9001 interface/wave-related arrays.
- Other arrays include motion and acceleration RMS metrics.

MATLAB arrays are stored with features/fields in rows and samples in columns. Check the loader and actual file keys before changing orientation assumptions.

**Target-row mapping:** The local archive's `metadata.txt` maps one-based `coefStatMatrix` rows 1, 2, and 3 to model-scale wave-resistance coefficient (`Cw`) at Fr=0.25, 0.33, and 0.41. Python `--target-row 0` therefore selects row 1 / Fr=0.25. `data_prep.py` records this mapping and source in results. Cite that archive metadata and seek publication-level corroboration when a report requires independent confirmation. Do not claim the individual normalized design variables have known physical meanings unless separate documentation establishes them.

## Modelling and evaluation expectations

A sensible comparison is:
1. Mean-prediction baseline.
2. Linear regression baseline.
3. Multilayer perceptron (MLP) regression.
4. Optional hyperparameter tuning (for example Optuna), compared against a fixed/default MLP.

Keep the held-out test set separate from model selection and tuning. Fit feature and target scalers using training data only; transform validation/test data using those fitted scalers. Avoid leakage. Report metrics on a clearly identified data split and state whether target values were inverse-transformed before metrics are calculated.

Useful evaluation artefacts include:
- training/validation loss curves,
- measured-versus-predicted test plot,
- residual/error diagnostics where appropriate,
- machine-readable metrics such as JSON.

Do not hardcode historical model scores as if freshly reproduced. Prior outputs reported approximate values (mean baseline RMSE 0.000467, R² -0.0016; linear regression RMSE 0.000420, R² 0.192; tuned MLP RMSE 0.000283, R² 0.632; another fixed/tuned MLP run RMSE about 0.000231–0.000232 and R² about 0.752–0.756). These are historical artefacts and have not been independently reproduced in the current session. Re-run the pipeline before relying on them in a report.

## Repository and branch context

Repository: https://github.com/oyen-student/MMA3001

Relevant branch names previously used:
- `main`
- `review/fix-ml-pipeline`
- `experiment/stable-mlp-training`
- `test/pytest-pipeline` (testing branch created for test/CI improvements)

Branch names and contents can change. Run `git status`, `git branch -vv`, and `git log --oneline --decorate -n 10` before deciding what is current. Do not assume the current branch is the one listed above.

The testing branch was created from `experiment/stable-mlp-training` and received:
- `tests/test_pipeline.py` with synthetic-data tests,
- `requirements-dev.txt` including `requirements.txt` and pytest,
- `.github/workflows/tests.yml` for Python 3.12 CI,
- a correction to a malformed docstring in `neural_network.py`,
- README test/CI instructions.

The synthetic suite passed locally on the user's Anaconda Python 3.12.14 with 21 tests on 2026-10-22. The no-write syntax scan also passed. CI workflow success still needs to be checked independently. Re-run pytest after any code/test change. In particular, make sure synthetic tests do not get presented as validation against the full real dataset.

## Local Windows / Python environment

The user's repository clone has been located at:
`C:\Users\Orlando\OneDrive\Documents\MMA3001\project\MMA3001`

Python is available through Anaconda:
`C:\Users\Orlando\anaconda3\python.exe`

This interpreter previously reported Python 3.12.14. The commands `python` and `py` were not recognized in the user's PowerShell session, so use the explicit executable path unless the environment has since been fixed.

Example commands from the repository root:
```powershell
& "C:\Users\Orlando\anaconda3\python.exe" --version
& "C:\Users\Orlando\anaconda3\python.exe" -m pip install -r requirements-dev.txt
& "C:\Users\Orlando\anaconda3\python.exe" -B check_syntax.py
& "C:\Users\Orlando\anaconda3\python.exe" -m pytest -q
```

To sync a remote branch:
```powershell
git status
git fetch origin
git branch -a
git switch --track -c test/pytest-pipeline origin/test/pytest-pipeline
```
If the local branch already exists, switch to it and run `git pull`. Do not overwrite uncommitted user work; inspect `git status` first and ask before discarding changes.

Example pipeline run used previously (verify current CLI options with `main.py --help` first):
```powershell
& "C:\Users\Orlando\anaconda3\python.exe" main.py --target-row 0 --skip-tuning --output-dir outputs_test
```
This requires the dataset to be downloaded into the expected data directory. Check the actual CLI help and README rather than assuming these options remain valid.

## How Codex should work in this repository

1. Start by checking branch, working tree, project tree, README, requirements, and relevant Python files.
2. State what is known versus what needs verification. Never claim tests, model training, or plots were run unless you actually ran them and inspected the result.
3. Make focused changes that preserve the existing project structure and user's approach unless there is a clear reason to refactor.
4. For bugs, identify the failing assumption and the smallest robust fix. Avoid speculative rewrites.
5. Add or update tests for behaviour changes. Use synthetic MATLAB-shaped data for unit/integration tests so the large private/local dataset is not required for routine CI.
6. Keep the real dataset and generated large outputs out of Git unless the project explicitly requires otherwise.
7. After changes, run syntax checks and relevant tests using the available interpreter. If unable to run them, say exactly why and give the user the command to run.
8. For ML results, record the command, target row, split/random seed, tuning settings, metrics, and output locations so results can be reproduced.
9. Explain results in engineering terms, but do not overstate what a model's performance proves. R² or RMSE alone does not establish physical validity or generalization to arbitrary hulls.
10. Before committing or pushing, summarize the changed files and checks. Never force-push, discard user changes, or merge branches without explicit approval.

## User's working preferences

- Wants step-by-step explanations from first principles when the maths matters; do not skip key algebra or silently jump to conclusions.
- Prefers direct fixes that preserve the current approach over broad rewrites.
- Values physical interpretability and wants to understand what the model is learning.
- Is still getting familiar with VS Code's interpreter, Git branch, and sync behaviour. Give explicit PowerShell commands and state which folder they should be run from.
- Be careful with spelling of branch name: `fix-ml-pipline` was how the user previously referred to one branch, but verify actual remote branch names instead of assuming.

## Current project caveats

- `data/train.csv`, `data/val.csv`, and `data/test.csv` are legacy artifacts with undocumented provenance. The active CLI does not read them; do not treat them as experiment evidence.
- `src/data_prep.py` is a deprecated compatibility wrapper around the project-root loader. It must never synthesize fallback/random data.
- The current split is random and sample-level. Metrics include an exact duplicate design-vector audit across splits. Near-duplicates are not measured because no tolerance is justified, and no hull-family labels are available for a group-aware split.
- Saved output folders may represent different runs. Treat them as historical artifacts unless a run record documents command, seed, target, split, tuning settings, environment, and output location.
- Before report conclusions, independently verify the archive metadata if needed, audit split relationships, run on the real MAT file, and check the GitHub Actions result.
