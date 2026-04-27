/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInferRuntime.h>
#include <cstddef>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

//! \brief TensorRT plugin for Gated Delta Rule (recurrent linear attention core)
//!
//! Registered as "GatedDeltaRule" under the "trt_edgellm" ONNX domain.
//!
//! Implements the recurrent gated delta rule from Qwen3.5:
//!   S = S * exp(g) + outer(k, (v - S^T @ k) * beta)
//!   y = S^T @ q
//!
//! Inputs:
//!   [0] query         [batch, seq_len, num_v_heads, head_k_dim]  FP16
//!   [1] key           [batch, seq_len, num_v_heads, head_k_dim]  FP16 (after repeat_interleave)
//!   [2] value         [batch, seq_len, num_v_heads, head_v_dim]  FP16
//!   [3] g             [batch, seq_len, num_v_heads]              FP16
//!   [4] beta          [batch, seq_len, num_v_heads]              FP16
//!   [5] initial_state [batch, num_v_heads, head_k_dim, head_v_dim] FP16 (optional)
//!
//! Outputs:
//!   [0] output      [batch, seq_len, num_v_heads, head_v_dim]    FP16
//!   [1] final_state [batch, num_v_heads, head_k_dim, head_v_dim] FP16
//!
//! Attributes:
//!   num_v_heads     - Number of value heads
//!   head_v_dim      - Value head dimension
//!   head_k_dim      - Key head dimension
//!   use_qk_l2norm   - Whether to apply L2 norm to q/k (1 = yes)
class GatedDeltaRulePlugin : public nvinfer1::IPluginV3,
                             public nvinfer1::IPluginV3OneCore,
                             public nvinfer1::IPluginV3OneBuild,
                             public nvinfer1::IPluginV3OneRuntime
{
public:
    GatedDeltaRulePlugin(
        std::string const& name, int32_t numVHeads, int32_t headVDim, int32_t headKDim, int32_t useQKL2Norm);
    GatedDeltaRulePlugin() = delete;
    GatedDeltaRulePlugin(GatedDeltaRulePlugin const&) = delete;
    ~GatedDeltaRulePlugin() override;

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    std::string mLayerName;
    std::string mNamespace;
    int32_t mNumVHeads{};
    int32_t mHeadVDim{};
    int32_t mHeadKDim{};
    int32_t mUseQKL2Norm{1};
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class GatedDeltaRulePluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    GatedDeltaRulePluginCreator();
    ~GatedDeltaRulePluginCreator() override = default;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    char const* getPluginNamespace() const noexcept override;
    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
