# Contributing

## Purpose

This code release must remain reproducible. The hashed files in `docker/` (listed in `verification/CHECKSUMS.txt` and `docker/vendor/INVENTORY.md`) are byte-identical to their copies in the image of Codabench submission 887183.

Do not edit `docker/predict.py`, `docker/model.py`, `docker/requirements.txt` or `docker/Dockerfile` unless the change is part of a new release with updated `verification/CHECKSUMS.txt`. Such a release is no longer the submitted system and must say so clearly.

## Branching model

This repository uses git-flow:

- `main` is release-only. Every release merge is tagged `vX.Y.Z`. Branch protection (pull request required, `ci / test` check required) should be enabled on `main` in the repository settings. Merge into it only from `release/*` or `hotfix/*`, with `--no-ff`.
- `develop` is the integration branch and the default pull-request target.
- `feature/<short-slug>` starts from `develop` and returns to `develop` through a pull request.
- `release/vX.Y.Z` starts from `develop`. Only version bumps, the changelog and documentation belong there. Merge it into `main`, tag it, and merge it back into `develop`.
- `hotfix/<slug>` starts from `main`. Merge it into `main` with a patch tag and into `develop`.

```text
feature/* ----> develop ----> release/vX.Y.Z ----> main
                  ^                    |             |
                  |                    +-------------+
                  +-------------- hotfix/* <---------+
```

Start and finish a feature:

```bash
git switch develop
git pull --ff-only
git switch -c feature/<short-slug>
git push -u origin feature/<short-slug>
# Open a pull request into develop, then merge with:
git switch develop
git merge --no-ff feature/<short-slug>
git push origin develop
```

Cut and finish a release:

```bash
git switch develop
git pull --ff-only
git switch -c release/vX.Y.Z
# Update version, CHANGELOG and documentation, then open a pull request.
git switch main
git merge --no-ff release/vX.Y.Z
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin main vX.Y.Z
git switch develop
git merge --no-ff release/vX.Y.Z
git push origin develop
```

Start and finish a hotfix:

```bash
git switch main
git pull --ff-only
git switch -c hotfix/<slug>
# Open a pull request, then merge and tag the patch release.
git switch main
git merge --no-ff hotfix/<slug>
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin main vX.Y.Z
git switch develop
git merge --no-ff hotfix/<slug>
git push origin develop
```

## Versioning

Semantic Versioning is used:

- MAJOR changes alter the submitted inference system or weights.
- MINOR changes add tooling, documentation or scripts without altering predictions.
- PATCH changes cover typos and CI corrections.

## Commit messages

Use Conventional Commits with `feat`, `fix`, `docs`, `chore`, `ci`, `refactor` or `test`. Useful scopes include `docker`, `training`, `verification` and `docs`. Write an imperative subject no longer than 72 characters, and use the body to explain why the change is needed.

Examples:

```text
docs(method): clarify task-specific fusion
test(verification): cover duplicate paths
```

## Pull requests

Keep one topic per pull request, complete the pull-request template, require green CI, and obtain at least one review.

## Definition of done for inference changes

Anything touching inference must run `verification/compare_to_oracle.py` on all 619 validation images and exit 0, with maximum drift below 0.5 px. Update `verification/CHECKSUMS.txt` whenever a hashed file changes.

## Reporting issues

Use GitHub issues. Include the image, command and output needed to reproduce the problem.
