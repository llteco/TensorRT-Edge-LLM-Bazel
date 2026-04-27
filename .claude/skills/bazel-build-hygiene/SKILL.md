---
name: bazel-build-hygiene
description: 'Write and refactor Bazel BUILD.bazel and .bzl files with reliable linting and automated edits. Use when adding targets, fixing Bazel style, running buildifier, or applying bulk changes with buildozer.'
argument-hint: 'What Bazel package/target/files should be updated?'
---

# Bazel Build Hygiene

Create or update Bazel BUILD/BZL code with consistent style, safe refactors, and verifiable results.

## When to Use

- Add or modify rules in BUILD.bazel or .bzl files
- Normalize formatting and lint using buildifier
- Perform repeatable structured edits using buildozer
- Update deps, visibility, srcs, hdrs, copts, tags, or rule kinds in bulk

## Procedure

1. Confirm Bazel module mode first.
   - Ask the user whether this repo should use legacy WORKSPACE mode or bzlmod.
   - If legacy WORKSPACE is requested, ensure `.bazelrc` contains:
     - `common --noenable_bzlmod`
   - If bzlmod is requested, do not add `--noenable_bzlmod`; align with MODULE.bazel-based workflow.

2. Define scope and success criteria.
   - Identify changed packages and affected targets.
   - Decide whether this is a single-target edit or a cross-package refactor.

3. Author Bazel changes first.
   - Keep rule names stable unless a rename is required.
   - Prefer small, reviewable edits per package.
   - Preserve existing repository conventions for load statements, macro usage, and visibility.

4. Run buildifier in check mode, then fix mode.
   - Check style for selected paths:
     - `buildifier -mode=check -r <path_or_paths>`
   - Apply formatting:
     - `buildifier -r <path_or_paths>`
   - Re-run check mode and ensure zero style errors.

5. Use buildozer for structured edits when repeating the same operation.
   - Add a dependency:
     - `buildozer 'add deps //pkg:dep' //pkg:target`
   - Remove a dependency:
     - `buildozer 'remove deps //pkg:dep' //pkg:target`
   - Set visibility:
     - `buildozer 'set visibility //visibility:public' //pkg:target`
   - Preview edits before writing when needed:
     - `buildozer -stdout '<command>' //<pkg>:<target>`

6. Validate with Bazel build.
   - Build changed targets first:
     - `bazelisk build //<pkg>:<target>`
   - If scope is broad, run a wider build sweep appropriate for the change.

7. Report outcomes.
   - List touched BUILD/BZL files.
   - Summarize buildifier status (check pass/fail).
   - Summarize buildozer commands used.
   - Confirm Bazel build targets that passed.

## Decision Points

- If module mode is unclear, stop and ask: legacy WORKSPACE or bzlmod.
- If legacy WORKSPACE is chosen, set `common --noenable_bzlmod` in `.bazelrc`.
- If the change is pure formatting, use buildifier only.
- If the change repeats across many targets, prefer buildozer over manual edits.
- If buildozer command impact is uncertain, run with `-stdout` first.
- If package-wide risk is high, split into two commits: mechanical edit first, semantic edit second.

## Completion Criteria

- BUILD/BZL edits satisfy requested behavior.
- Bazel module mode decision is explicit and reflected in `.bazelrc` when needed.
- buildifier check is clean on all touched Bazel files.
- Bazel build succeeds for all required targets in scope.
- Final summary includes commands run and verification status.