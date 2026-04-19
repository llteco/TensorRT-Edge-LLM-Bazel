---
description: "Use when building C++ plugins/examples, exporting ONNX from Python, or creating TensorRT engines from ONNX in this repository. Includes canonical bazelisk and uv command sequences."
name: "Build And Export Workflow"
---
# TensorRT Edge LLM Build And Export Workflow

Follow these canonical commands for build and export tasks.

## Build C++

- Build plugin:

```bash
bazelisk build //cpp:NvInfer_edgellm_plugin
```

- Build examples:

```bash
bazelisk build //examples/...
```

## Create Engine From ONNX

- Visual engine build:

```bash
# If visual_build is linked with dynamic plugin library, set EDGELLM_PLUGIN_PATH to that library
EDGELLM_PLUGIN_PATH=bazel-bin/cpp/libNvInfer_edgellm_plugin.so bazel-bin/examples/visual_build --onnxDir <visual_onnx_dir> --engineDir <visual_engine_dir>
```

- LLM engine build:

```bash
# If visual_build is linked with dynamic plugin library, set EDGELLM_PLUGIN_PATH to that library
EDGELLM_PLUGIN_PATH=bazel-bin/cpp/libNvInfer_edgellm_plugin.so bazel-bin/examples/llm_build --onnxDir <llm_onnx_dir> --engineDir <llm_engine_dir>
```

## Create ONNX From Original Python

1. Setup venv:

```bash
uv sync --dev -U
```

2. Export visual ONNX:

```bash
uv run tensorrt-edgellm-export-visual --model_dir <huggingface_id> --output_dir <visual_onnx_dir>
```

3. Export LLM ONNX:

```bash
uv run tensorrt-edgellm-export-llm --model_dir <huggingface_id> --output_dir <llm_onnx_dir>
```

## Notes

- Build the plugin before running visual/llm engine builders, if they are linked dynamically.
- Replace `<huggingface_id>` with the actual model identifier.
- Replace all `<..._dir>` placeholders with local paths that match your environment.
- Do not hardcode personal or host-specific absolute paths in shared instructions.