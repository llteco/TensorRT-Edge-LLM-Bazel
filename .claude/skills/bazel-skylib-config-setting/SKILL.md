---
name: bazel-skylib-config-setting
description: 'Create robust Bazel config_setting selectors with skylib patterns. Use when designing feature flags, platform conditions, or reusable build constraints in BUILD.bazel/.bzl.'
argument-hint: 'What condition dimensions (cpu/os/cuda/define/flag) and target rules should use select()?'
---

# Bazel Skylib Config Setting Guide

Use this skill to design and implement maintainable Bazel condition logic for `select()` using `config_setting` and common skylib-oriented patterns.

## When to Use

- Add conditional deps/srcs/copts via `select()`
- Replace ad-hoc `--define` checks with clearer config structure
- Create reusable config conditions across multiple packages
- Migrate from duplicated conditions to centralized settings

## Outcome

Produce a clear condition matrix and the corresponding Bazel code:
- `config_setting` declarations
- optional grouping aliases or helper macros
- `select()` usage in concrete rules
- validation commands and pass/fail checks

## Procedure

1. Define condition matrix first.
- List each dimension to branch on: `cpu`, `os`, `compilation_mode`, `--define`, Starlark build setting.
- Define exact expected values and where they come from.
- Identify default/fallback behavior for unmatched configs.

2. Choose the condition source type.
- Use `values = {...}` for native flags (for example `cpu`, `compilation_mode`).
- Use `define_values = {...}` for legacy `--define key=value`.
- Use `flag_values = {...}` when matching Starlark build settings.
- Prefer typed Starlark build settings for new feature switches; keep `--define` for compatibility-only cases.

3. Create `config_setting` targets.
- Name conditions by intent, not by implementation detail.
- Keep them near usage package, or in a shared `//build_config/...` package for cross-package reuse.
- Always add a default branch in all `select()` calls.

4. Apply conditions via `select()`.
- Attach conditions to `deps`, `srcs`, `copts`, `linkopts`, or `defines`.
- Keep each `select()` focused on one concern when possible.
- If multiple dimensions are needed, prefer composing explicit conditions instead of deeply nested `select()` chains.

5. Add reusable skylib-style organization.
- If many targets share the same branch map, factor it into a macro in `.bzl`.
- Keep one canonical condition target per semantic condition to avoid drift.
- For broad reuse, expose helper constants/macros from a dedicated config module.

6. Validate behavior.
- Run `bazelisk query` to confirm condition targets exist and are referenced.
- Run representative builds for each branch with explicit flags.
- Verify selected outputs/deps change as intended.

7. Document usage.
- Record supported flags and examples in comments or package README.
- Include at least one canonical build command per branch.

## Decision Points

- If you need backwards compatibility with existing CI flags: keep `define_values` path.
- If you need long-term maintainability: introduce typed Starlark build settings and `flag_values`.
- If one condition is used by 3+ targets: extract to shared location/macro.
- If a condition is package-local and unlikely to spread: keep local for readability.

## Quality Checks

- Every `select()` has a default branch (`//conditions:default`).
- No duplicated condition targets with equivalent semantics.
- Condition names reflect product/build intent.
- At least one build command validates each non-default branch.
- BUILD/BZL files pass `buildifier`.

## Minimal Example

```bzl
config_setting(
    name = "cuda_enabled",
    define_values = {"with_cuda": "true"},
)

cc_library(
    name = "core",
    srcs = ["core.cc"] + select({
        ":cuda_enabled": ["core_cuda.cc"],
        "//conditions:default": [],
    }),
)
```

## Declare in WORKSPACE

```bzl
# https://github.com/bazelbuild/bazel-skylib/releases
skylib_version = "1.9.0"
http_archive(
    name = "bazel_skylib",
    sha256 = "3b5b49006181f5f8ff626ef8ddceaa95e9bb8ad294f7b5d7b11ea9f7ddaf8c59",
    urls = [
        "https://github.com/bazelbuild/bazel-skylib/releases/download/{0}/bazel-skylib-{0}.tar.gz".format(skylib_version),
    ],
)
```

## Declare in BZLMOD

```bzl
bazel_dep(name = "bazel_skylib", version = "1.9.0")
```

## Validation Commands

```bash
bazelisk build //path:target
bazelisk build --define with_cuda=true //path:target
buildifier -mode=check -r .
```
