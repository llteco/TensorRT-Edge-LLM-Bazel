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

//! \brief TensorRT plugin for Gated Delta Net causal conv1d
//!
//! Registered as "GatedDeltaNetCausalConv1d" under the "trt_edgellm" ONNX domain.
//!
//! Inputs:
//!   [0] x          [batch, conv_dim, seq_len]     FP16
//!   [1] conv_state [batch, conv_dim, kernel_size] FP16 (optional)
//!   [2] weight     [conv_dim, kernel_size]        FP16
//!   [3] bias       [conv_dim]                     FP16
//!
//! Outputs:
//!   [0] output          [batch, conv_dim, seq_len]     FP16
//!   [1] conv_state_out  [batch, conv_dim, kernel_size] FP16
//!
//! Attributes:
//!   kernel_size  - Convolution kernel size
//!   activation   - Activation type (0 = silu)
class GatedDeltaNetCausalConv1dPlugin : public nvinfer1::IPluginV3,
                                        public nvinfer1::IPluginV3OneCore,
                                        public nvinfer1::IPluginV3OneBuild,
                                        public nvinfer1::IPluginV3OneRuntime
{
public:
    GatedDeltaNetCausalConv1dPlugin(std::string const& name, int32_t kernelSize, int32_t activation = 0);
    GatedDeltaNetCausalConv1dPlugin() = delete;
    GatedDeltaNetCausalConv1dPlugin(GatedDeltaNetCausalConv1dPlugin const&) = delete;
    ~GatedDeltaNetCausalConv1dPlugin() override;

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
    int32_t mKernelSize{4};
    int32_t mActivation{0};
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class GatedDeltaNetCausalConv1dPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    GatedDeltaNetCausalConv1dPluginCreator();
    ~GatedDeltaNetCausalConv1dPluginCreator() override = default;
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
