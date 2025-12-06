# AGENTS.md

This document tells the coding agent **how to work in this repository**: environment setup, build & test and PR workflow. Remember to open a new branch with the name of the feature or the bug.


## Repository layout

- Top repo: **ColonSuperpoint_demo** (this repo: Superpoint: SP).
- Sub repo: **utils/ColonSuperGlue_demo** (the superglue (SG) demo cloned without being a git submodule).
- Sub repo: **utils/LightGlue** (the lightglue (LG) demo cloned without being a git submodule).

- All modifications should be done in the Top repo, given that the SuperGLue_demo is listed in .gitignore
---

## Environment setup

1. If conda environment already created just activate it

```bash
conda activate colon-demo
```
---


## Code style, typing, docs

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

## Branching, rebasing, PRs

* Work on a **feature branch** for every feature request or bug fix; **do not** push directly to `main`.
* Before creating the branch, ensure the worktree is clean. If there are local modifications on `master`, run `git stash push --include-untracked -m "<reason>"` to shelve them, then create the branch from the clean state. Leave the stash untouched while you work so you can decide later—when you return to `master`—whether to reapply it with `git stash pop` or discard it with `git stash drop`.
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

   * Confirm `conda activate colon-demo` has been run so commands use the correct environment.
   * Run `git status --short` on `master`; if it is not clean, stash the local edits with `git stash push --include-untracked -m "<reason>"` and keep the stash for later review when you return to `master`.

3. **Implement**

4. **Validate**

   * Make a test run looking for possible errors (reusing the lightweight frame directory and disabling step mode).

   <!-- * If execution is blocked by sandbox restrictions (e.g., macOS Seatbelt), report the limitation and request the user to run locally. -->


5. **Hand off for review**

   * Present the changes and wait for user confirmation. Do not commit or merge until the user explicitly requests it.

6. **Prepare PR**

   * Provide **What / Why / Test**, note decisions/assumptions, and keep scope tight (split large changes into smaller PRs).

7. **Safety rails**

   * Do not fetch LFS assets unless the task requires them.

8. **Finalization on request**

   * When the user approves, squash commits into a single commit, rebase onto `master`, and merge as instructed. Delete the feature branch after merging.
