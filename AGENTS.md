# AGENTS.md

This document tells the coding agent **how to work in this repository**: environment setup, build & test and PR workflow. It is aligned with `.github/copilot-instructions.md`. Remember to open a new branch with the name of the feature or the bug.


## 1) Repository layout

- Top repo: **colon_matching** (this repo: Superpoint: SP).
- Sub repo: **utils/ColonSuperGlue_demo** (the superglue (SG) demo cloned without being a git submodule).
  - SG contains a more advanced demo by the same researchers. It serves as inspiration

- All modifications should be done in the Top repo, given that the SuperGLue_demo is listed in .gitignore
---

## 2) Environment setup

1. If conda environment already created just activate it

```bash
conda activate py38-sp
```

2. Create conda environment:
    
```bash
conda create -n py38-sp python=3.8
conda activate py38-sp
```

3. Install dependencies:

```bash
pip install opencv-python torch pyyaml
```

---

## 3) Test run

Always run a test run to search for any error.

```bash
./demo_superpoint.py --config configs/config.yaml
```

---

## 4) Code style, typing, docs

* **Style**: Black (line length 88) + isort. Prefer early returns and expressive `snake_case` names.
* **Typing**: Add Python type hints on public functions/classes.
* **Docstrings**: Use **Google-style** docstrings with `Args:` and `Returns:`, focused on **why** as well as what.

Example:

```python
def example(x: int) -> str:
    """Summarize purpose and result.

    Args:
        x: Meaningful description.

    Returns:
        Description of the returned value.
    """
```

---

## 5) Branching, rebasing, PRs

* Work on a **feature branch**; **do not** push directly to `main`.
* **Rebase** your branch onto the latest `main` before merging:

  ```bash
  git fetch origin
  git rebase origin/main
  ```
* **Squash** trivial/iterative commits before merging.
* **PR checklist**:

  * Include a short **What / Why / Test** section.
  * Ensure CI checks (tests) are **green**.
  * For breaking changes, update README/docs.
  * For dependency changes, update packaging/requirements.

After merging, **delete the feature branch**.

---

## 6) Agent task flow (step-by-step)

1. **Plan**

   * Identify the components to change and make a plan in no less than 6 steps.
2. **Prepare**

   * Confirm conda environment is activated.

3. **Implement**

4. **Validate**

   * Make a test run looking for possible errors.
   ```bash
   ./demo_superpoint.py --config configs/config.yaml
   ```


5. **Prepare PR**

   * Provide **What / Why / Test**, note decisions/assumptions, and keep scope tight (split large changes into smaller PRs).
6. **Safety rails**

   * Do not fetch LFS assets unless the task requires them.
 
