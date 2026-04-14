/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gatedDeltaRulePlugin.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "kernels/gdn/gatedDeltaNetKernel.cuh"
#include "plugins/utils/pluginUtils.h"

#include <NvInferPluginBase.h>
#include <cuda_runtime.h>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kGATED_DELTA_RULE_PLUGIN_NAME{"GatedDeltaRule"};
constexpr char const* kGATED_DELTA_RULE_PLUGIN_VERSION{"1"};

constexpr int32_t kIN_QUERY_IDX{0};
constexpr int32_t kIN_KEY_IDX{1};
constexpr int32_t kIN_VALUE_IDX{2};
constexpr int32_t kIN_G_IDX{3};
constexpr int32_t kIN_BETA_IDX{4};
constexpr int32_t kIN_INITIAL_STATE_IDX{5};
constexpr int32_t kNUM_INPUTS{6};

constexpr int32_t kOUT_OUTPUT_IDX{0};
constexpr int32_t kOUT_FINAL_STATE_IDX{1};
constexpr int32_t kNUM_OUTPUTS{2};

constexpr char const* FIELD_NUM_V_HEADS{"num_v_heads"};
constexpr char const* FIELD_HEAD_V_DIM{"head_v_dim"};
constexpr char const* FIELD_HEAD_K_DIM{"head_k_dim"};
constexpr char const* FIELD_USE_QK_L2NORM{"use_qk_l2norm"};
} // namespace

GatedDeltaRulePlugin::GatedDeltaRulePlugin(
    std::string const& name, int32_t numVHeads, int32_t headVDim, int32_t headKDim, int32_t useQKL2Norm)
    : mLayerName(name)
    , mNumVHeads(numVHeads)
    , mHeadVDim(headVDim)
    , mHeadKDim(headKDim)
    , mUseQKL2Norm(useQKL2Norm)
{
}

GatedDeltaRulePlugin::~GatedDeltaRulePlugin() = default;

nvinfer1::IPluginCapability* GatedDeltaRulePlugin::getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept
{
    if (type == nvinfer1::PluginCapabilityType::kBUILD)
    {
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kRUNTIME)
    {
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return static_cast<nvinfer1::IPluginV3OneCore*>(this);
}

nvinfer1::IPluginV3* GatedDeltaRulePlugin::clone() noexcept
{
    auto* plugin = new GatedDeltaRulePlugin(mLayerName, mNumVHeads, mHeadVDim, mHeadKDim, mUseQKL2Norm);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

char const* GatedDeltaRulePlugin::getPluginName() const noexcept
{
    return kGATED_DELTA_RULE_PLUGIN_NAME;
}

char const* GatedDeltaRulePlugin::getPluginVersion() const noexcept
{
    return kGATED_DELTA_RULE_PLUGIN_VERSION;
}

char const* GatedDeltaRulePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

int32_t GatedDeltaRulePlugin::getNbOutputs() const noexcept
{
    return kNUM_OUTPUTS;
}

int32_t GatedDeltaRulePlugin::getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    outputTypes[kOUT_OUTPUT_IDX] = inputTypes[kIN_QUERY_IDX];
    if (nbInputs > kIN_INITIAL_STATE_IDX)
    {
        outputTypes[kOUT_FINAL_STATE_IDX] = inputTypes[kIN_INITIAL_STATE_IDX];
    }
    return 0;
}

int32_t GatedDeltaRulePlugin::getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/, nvinfer1::DimsExprs* outputs,
    int32_t nbOutputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (nbInputs < kNUM_INPUTS - 1 || nbOutputs != kNUM_OUTPUTS)
    {
        return -1;
    }
    // output: [batch, seq_len, num_v_heads, head_v_dim]
    outputs[kOUT_OUTPUT_IDX].nbDims = 4;
    outputs[kOUT_OUTPUT_IDX].d[0] = inputs[kIN_QUERY_IDX].d[0];
    outputs[kOUT_OUTPUT_IDX].d[1] = inputs[kIN_QUERY_IDX].d[1];
    outputs[kOUT_OUTPUT_IDX].d[2] = exprBuilder.constant(mNumVHeads);
    outputs[kOUT_OUTPUT_IDX].d[3] = exprBuilder.constant(mHeadVDim);
    // final_state: [batch, num_v_heads, head_k_dim, head_v_dim]
    outputs[kOUT_FINAL_STATE_IDX].nbDims = 4;
    outputs[kOUT_FINAL_STATE_IDX].d[0] = inputs[kIN_QUERY_IDX].d[0];
    outputs[kOUT_FINAL_STATE_IDX].d[1] = exprBuilder.constant(mNumVHeads);
    outputs[kOUT_FINAL_STATE_IDX].d[2] = exprBuilder.constant(mHeadKDim);
    outputs[kOUT_FINAL_STATE_IDX].d[3] = exprBuilder.constant(mHeadVDim);
    return 0;
}

bool GatedDeltaRulePlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (nbInputs < kNUM_INPUTS - 1 || nbOutputs != kNUM_OUTPUTS)
    {
        return false;
    }
    bool const isLinear = inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;
    bool const isHalf = inOut[pos].desc.type == nvinfer1::DataType::kHALF;
    if (!isLinear || !isHalf)
    {
        return false;
    }
    if (pos > 0)
    {
        return inOut[pos].desc.type == inOut[kIN_QUERY_IDX].desc.type;
    }
    return true;
}

int32_t GatedDeltaRulePlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (nbInputs < kNUM_INPUTS - 1 || nbOutputs != kNUM_OUTPUTS)
    {
        return -1;
    }
    return 0;
}

size_t GatedDeltaRulePlugin::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t /*nbInputs*/,
    nvinfer1::DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept
{
    int32_t batch = inputs[kIN_QUERY_IDX].desc.dims.d[0];
    int32_t seqLen = inputs[kIN_QUERY_IDX].desc.dims.d[1];
    size_t size = 0;
    // qNormed, kNormed if use_qk_l2norm
    if (mUseQKL2Norm)
    {
        size = accumulateWorkspaceSize(size, {batch, seqLen, mNumVHeads, mHeadKDim, 2}, nvinfer1::DataType::kHALF);
    }
    return size;
}

int32_t GatedDeltaRulePlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int32_t batch = inputDesc[kIN_QUERY_IDX].dims.d[0];
    int32_t seqLen = inputDesc[kIN_QUERY_IDX].dims.d[1];

    if (batch <= 0 || seqLen <= 0)
    {
        return 0;
    }

    half const* query = static_cast<half const*>(inputs[kIN_QUERY_IDX]);
    half const* key = static_cast<half const*>(inputs[kIN_KEY_IDX]);
    half const* value = static_cast<half const*>(inputs[kIN_VALUE_IDX]);
    half const* g = static_cast<half const*>(inputs[kIN_G_IDX]);
    half const* beta = static_cast<half const*>(inputs[kIN_BETA_IDX]);
    half const* initialState = nullptr;
    if (inputs[kIN_INITIAL_STATE_IDX] != nullptr)
    {
        initialState = static_cast<half const*>(inputs[kIN_INITIAL_STATE_IDX]);
    }
    half* output = static_cast<half*>(outputs[kOUT_OUTPUT_IDX]);
    half* finalState = static_cast<half*>(outputs[kOUT_FINAL_STATE_IDX]);

    half* qNormed = const_cast<half*>(query);
    half* kNormed = const_cast<half*>(key);

    if (mUseQKL2Norm)
    {
        char* ws = static_cast<char*>(workspace);
        qNormed = reinterpret_cast<half*>(ws);
        kNormed = reinterpret_cast<half*>(ws + batch * seqLen * mNumVHeads * mHeadKDim * sizeof(half));

        CUDA_CHECK(cudaMemcpyAsync(
            qNormed, query, batch * seqLen * mNumVHeads * mHeadKDim * sizeof(half), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            kNormed, key, batch * seqLen * mNumVHeads * mHeadKDim * sizeof(half), cudaMemcpyDeviceToDevice, stream));

        trt_edgellm::kernels::computeGatedDeltaQKNorm(
            qNormed, kNormed, qNormed, kNormed, batch * seqLen, mNumVHeads, mHeadKDim, stream);
    }

    // Scale q/k by 1/sqrt(head_k_dim)
    trt_edgellm::kernels::scaleQK(qNormed, batch * seqLen, mNumVHeads, mHeadKDim, stream);

    if (seqLen == 1)
    {
        trt_edgellm::kernels::recurrentGatedDeltaStep(qNormed, kNormed, value, g, beta, initialState, output,
            finalState, batch, mNumVHeads, mHeadVDim, mHeadKDim, stream);
    }
    else
    {
        trt_edgellm::kernels::gatedDeltaNetSerialLoop(qNormed, kNormed, value, g, beta, initialState, output,
            finalState, batch, seqLen, mNumVHeads, mHeadVDim, mHeadKDim, stream);
    }
    return 0;
}

int32_t GatedDeltaRulePlugin::onShapeChange(nvinfer1::PluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    nvinfer1::PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept
{
    return 0;
}

nvinfer1::IPluginV3* GatedDeltaRulePlugin::attachToContext(nvinfer1::IPluginResourceContext* /*context*/) noexcept
{
    return clone();
}

nvinfer1::PluginFieldCollection const* GatedDeltaRulePlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(FIELD_NUM_V_HEADS, &mNumVHeads, nvinfer1::PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(FIELD_HEAD_V_DIM, &mHeadVDim, nvinfer1::PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(FIELD_HEAD_K_DIM, &mHeadKDim, nvinfer1::PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(FIELD_USE_QK_L2NORM, &mUseQKL2Norm, nvinfer1::PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

void GatedDeltaRulePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

// Creator
GatedDeltaRulePluginCreator::GatedDeltaRulePluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(FIELD_NUM_V_HEADS, nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back(FIELD_HEAD_V_DIM, nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back(FIELD_HEAD_K_DIM, nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back(FIELD_USE_QK_L2NORM, nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* GatedDeltaRulePluginCreator::getPluginName() const noexcept
{
    return kGATED_DELTA_RULE_PLUGIN_NAME;
}

char const* GatedDeltaRulePluginCreator::getPluginVersion() const noexcept
{
    return kGATED_DELTA_RULE_PLUGIN_VERSION;
}

nvinfer1::PluginFieldCollection const* GatedDeltaRulePluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void GatedDeltaRulePluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

char const* GatedDeltaRulePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

nvinfer1::IPluginV3* GatedDeltaRulePluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase /*phase*/) noexcept
{
    int32_t numVHeads = 0;
    int32_t headVDim = 0;
    int32_t headKDim = 0;
    int32_t useQKL2Norm = 1;
    try
    {
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            std::string fieldName(fc->fields[i].name);
            if (fc->fields[i].data == nullptr)
            {
                continue;
            }
            if (fieldName == FIELD_NUM_V_HEADS)
            {
                numVHeads = *static_cast<int32_t const*>(fc->fields[i].data);
            }
            else if (fieldName == FIELD_HEAD_V_DIM)
            {
                headVDim = *static_cast<int32_t const*>(fc->fields[i].data);
            }
            else if (fieldName == FIELD_HEAD_K_DIM)
            {
                headKDim = *static_cast<int32_t const*>(fc->fields[i].data);
            }
            else if (fieldName == FIELD_USE_QK_L2NORM)
            {
                useQKL2Norm = *static_cast<int32_t const*>(fc->fields[i].data);
            }
        }
        auto* plugin = new GatedDeltaRulePlugin(name, numVHeads, headVDim, headKDim, useQKL2Norm);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create GatedDeltaRulePlugin: %s", e.what());
        return nullptr;
    }
}

nvinfer1::PluginFieldCollection GatedDeltaRulePluginCreator::mFieldCollection{};
std::vector<nvinfer1::PluginField> GatedDeltaRulePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GatedDeltaRulePluginCreator);

} // namespace plugins
} // namespace trt_edgellm
