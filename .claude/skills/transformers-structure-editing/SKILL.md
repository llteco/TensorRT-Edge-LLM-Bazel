---
name: transformers-structure-editing
description: 'Understand Hugging Face transformers model structure and apply safe modifications for loading, patching, export, and compatibility. Use when adding model support, changing modeling behavior, or debugging custom model integration.'
argument-hint: 'Which model family and what behavior should be changed?'
---

# Transformers Structure Editing

Systematically understand and modify transformers-based model code used by this repository, with compatibility checks for quantization and ONNX export workflows.

## When to Use

- Add support for a new model family from transformers
- Patch attention, rotary embedding, or forward behavior for export/runtime constraints
- Register custom config/model classes into AutoConfig or AutoModel
- Debug failures caused by transformers version upgrades or API changes
- Decide between subclass wrappers, runtime monkey-patches, or registry-based integration

## Procedure

1. Confirm target and constraints.
   - Ask which model family is being changed (for example qwen2_vl, qwen3_vl, qwen3_5, llama).
   - Confirm the goal: feature addition, bug fix, export compatibility, or performance.
   - Confirm constraints: preserve upstream behavior, preserve quantized weights, keep ONNX I/O names stable, or avoid API breakage.

2. Build a structure map before editing.
   - Locate config entrypoints (`AutoConfig`, model config classes).
   - Locate model entrypoints (`AutoModel*`, direct imports from `transformers.models.<family>.modeling_*`).
   - Locate project wrappers and patches in `tensorrt_edgellm/visual_models`, `tensorrt_edgellm/llm_models`, `tensorrt_edgellm/quantization`, and `tensorrt_edgellm/onnx_export`.
   - Identify the real call chain from load -> patch/wrapper -> export/quantization.

3. Choose modification strategy explicitly.
   - Wrapper subclass strategy: best for preserving original modules and reusing existing parameters.
   - Auto registry strategy (`AutoConfig.register`, `AutoModel.register`): best for custom model types.
   - Runtime patch strategy (monkey-patch): only when no stable extension point exists.
   - Prefer the least invasive strategy that satisfies the requirement.

4. Implement with compatibility guardrails.
   - Keep constructor signatures and critical forward inputs/outputs compatible.
   - If changing attention internals, preserve tensor layout contracts expected by downstream plugins/export.
   - If adding required inputs, fail fast with clear error messages.
   - Keep path and loader behavior environment-agnostic (no personal absolute paths in code or docs).

5. Validate integration points.
   - Load-path validation: model/config/tokenizer or processor can still be resolved.
   - Export-path validation: visual/llm export code still executes for intended model type.
   - Quantization-path validation: quantization entrypoints still accept modified model objects.
   - Build/runtime validation: run the smallest representative command set that proves no regression.

6. Document change intent and risk.
   - Explain why this strategy was chosen over alternatives.
   - Record behavior compatibility guarantees and known limitations.
   - Include follow-up tasks if temporary patches were introduced.

## How To Locate Model Definitions

Use this two-layer method before any model edit.

1. Find repository adaptation entrypoints first.
    - Search in `tensorrt_edgellm/visual_models`, `tensorrt_edgellm/llm_models`, `tensorrt_edgellm/quantization`, and `tensorrt_edgellm/onnx_export`.
    - Look for imports like:
       - `from transformers.models.<family>.modeling_<family> import ...`
    - This reveals where the project wraps or patches upstream classes.

2. Jump to upstream transformers implementation.
    - Open:
       - `.venv/lib/python*/site-packages/transformers/models/<family>/modeling_<family>.py`
    - For config-driven behavior, also check:
       - `configuration_<family>.py`
    - For processor/tokenizer behavior, also check:
       - `processing_<family>.py` and tokenization/image/video processing files.

3. Validate the call chain.
    - Confirm path from `AutoConfig`/`AutoModel*` or direct class import -> repository wrapper/patch -> export/quantization entrypoints.
    - Only edit after this chain is explicit.

### Family Examples

- Qwen3:
   - Repository wrapper example: `tensorrt_edgellm/visual_models/qwen3_vl_model.py`
   - Upstream model definition: `transformers/models/qwen3/modeling_qwen3.py`

- Gemma4:
   - Upstream model definition: `transformers/models/gemma4/modeling_gemma4.py`
   - Upstream config: `transformers/models/gemma4/configuration_gemma4.py`
   - If repository-level references are absent, integration work is likely not implemented yet and should start from loader/registry design.

## Decision Points

- If behavior can be achieved by wrapping a module, prefer wrapper over monkey-patch.
- If model type is custom and must work with `from_pretrained`, prefer Auto registry integration.
- If export ABI is sensitive, prioritize I/O and shape stability over internal refactor purity.
- If transformers upstream changed APIs, isolate version-sensitive logic in one place.

## Completion Criteria

- Target model behavior change is implemented and scoped to the requested family.
- Loading, export, and quantization paths are checked for regressions in affected flows.
- Compatibility assumptions (tensor shapes, dtypes, input names, registry behavior) are explicit.
- Summary clearly states what changed, why, and what remains as known risk.