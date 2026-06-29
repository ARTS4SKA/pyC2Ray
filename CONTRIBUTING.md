# Contributing to pyc2ray

Thank you for your interest in contributing to **pyc2ray**!

> ⚠️ This is an initial version of the contributing guide and does not cover every workflow or edge case yet.

If you have questions, please reach out to the main maintainers.

## Issues, Questions, and Suggestions

Users of pyc2ray are welcome to open GitHub issues:
- to report bugs or unexpected behavior,
- to ask for help,
- and to share suggestions for improvements.

## Pull Request Guidelines

To keep the repository history clean and maintainable, please follow these rules when opening a pull request:

- Your branch should be updated against the branch it is being merged into before review/merge.
- **Rebase and merge** is the preferred merge strategy for typical contributions, to keep history linear.
- It is strongly suggested to squash commits so a regular PR is merged as **one commit**.
- Large development branches may be merged without squashing when preserving commit history is valuable.
- If your PR depends on an existing PRs, please specify that in the PR description with `Depends on #<PR number>`.

## Commit Message Style

Commit messages should follow the [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/) style guide.

At minimum:
- Start the subject with `type:`
- Typical types are:
  - `fix:` for bug fixes
  - `feat:` for new features

Examples:
- `fix: handle missing proxy configuration`
- `feat: add support for multiple outbound profiles`

## Suggested Contributor Workflow

0. **Only the first time**, install `pre-commit` in your environment and install the hooks:
   ```bash
   pre-commit install
   ```
1. Create a new branch off the main trunk and make your modifications there.
2. Commit your changes and fix any issue highlighted by the pre-commit hooks; code format is automatically fixed.
3. Push your branch to the remote repository.
4. Open a Pull Request on GitHub to the `main` branch.
5. It is strongly suggested to squash all commits into one.
6. In the PR description on GitHub, specify blocking dependencies with `Depends on #...` and close issues with `Closes #...`.
7. Ask for code review before merging.

---

Thanks again for helping improve pyc2ray 🚀
