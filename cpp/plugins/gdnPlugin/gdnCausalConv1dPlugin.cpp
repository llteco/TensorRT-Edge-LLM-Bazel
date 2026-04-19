/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gdnCausalConv1dPlugin.h"

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
constexpr char const* kGDN_CAUSAL_CONV_PLUGIN_NAME{"GatedDeltaNetCausalConv1d"};
constexpr char const* kGDN_CAUSAL_CONV_PLUGIN_VERSION{"1"};

// ONNX input order: [x, weight, bias, conv_state]
constexpr int32_t kIN_X_IDX{0};
constexpr int32_t kIN_WEIGHT_IDX{1};
constexpr int32_t kIN_BIAS_IDX{2};
constexpr int32_t kIN_CONV_STATE_IDX{3};
constexpr int32_t kNUM_INPUTS{4};

constexpr int32_t kOUT_IDX{0};
constexpr int32_t kOUT_CONV_STATE_IDX{1};
constexpr int32_t kNUM_OUTPUTS{2};

constexpr char const* FIELD_KERNEL_SIZE{"kernel_size"};
constexpr char const* FIELD_ACTIVATION{"activation"};

// Helper: ensure a pointer is half*; if input is FLOAT, allocate a temp buffer and convert.
struct HalfBuffer
{
    half* data = nullptr;
    bool owned = false;

    void acquire(void const* ptr, nvinfer1::DataType type, int32_t numel, cudaStream_t stream)
    {
        if (type == nvinfer1::DataType::kHALF)
        {
            data = const_cast<half*>(static_cast<half const*>(ptr));
            owned = false;
        }
        else if (type == nvinfer1::DataType::kFLOAT)
        {
            cudaMallocAsync(&data, numel * sizeof(half), stream);
            trt_edgellm::kernels::castFloatToHalf(ptr, data, numel, stream);
            owned = true;
        }
    }

    void release(cudaStream_t stream)
    {
        if (owned && data != nullptr)
        {
            cudaFreeAsync(data, stream);
            data = nullptr;
            owned = false;
        }
    }
};

} // namespace

GatedDeltaNetCausalConv1dPlugin::GatedDeltaNetCausalConv1dPlugin(
    std::string const& name, int32_t kernelSize, int32_t activation)
    : mLayerName(name)
    , mKernelSize(kernelSize)
    , mActivation(activation)
{
}

GatedDeltaNetCausalConv1dPlugin::~GatedDeltaNetCausalConv1dPlugin() = default;

nvinfer1::IPluginCapability* GatedDeltaNetCausalConv1dPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept
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

nvinfer1::IPluginV3* GatedDeltaNetCausalConv1dPlugin::clone() noexcept
{
    auto* plugin = new GatedDeltaNetCausalConv1dPlugin(mLayerName, mKernelSize, mActivation);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

char const* GatedDeltaNetCausalConv1dPlugin::getPluginName() const noexcept
{
    return kGDN_CAUSAL_CONV_PLUGIN_NAME;
}

char const* GatedDeltaNetCausalConv1dPlugin::getPluginVersion() const noexcept
{
    return kGDN_CAUSAL_CONV_PLUGIN_VERSION;
}

char const* GatedDeltaNetCausalConv1dPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

int32_t GatedDeltaNetCausalConv1dPlugin::getNbOutputs() const noexcept
{
    return kNUM_OUTPUTS;
}

int32_t GatedDeltaNetCausalConv1dPlugin::getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (nbOutputs != kNUM_OUTPUTS || nbInputs != kNUM_INPUTS)
    {
        return -1;
    }
    outputTypes[kOUT_IDX] = inputTypes[kIN_X_IDX];
    outputTypes[kOUT_CONV_STATE_IDX] = inputTypes[kIN_CONV_STATE_IDX];
    return 0;
}

int32_t GatedDeltaNetCausalConv1dPlugin::getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/, nvinfer1::DimsExprs* outputs,
    int32_t nbOutputs, nvinfer1::IExprBuilder& /*exprBuilder*/) noexcept
{
    if (nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS)
    {
        return -1;
    }
    // output: same as x [batch, conv_dim, seq_len]
    outputs[kOUT_IDX].nbDims = inputs[kIN_X_IDX].nbDims;
    for (int32_t i = 0; i < outputs[kOUT_IDX].nbDims; ++i)
    {
        outputs[kOUT_IDX].d[i] = inputs[kIN_X_IDX].d[i];
    }
    // conv_state_out: same as conv_state input [batch, conv_dim, kernel_size]
    outputs[kOUT_CONV_STATE_IDX].nbDims = inputs[kIN_CONV_STATE_IDX].nbDims;
    for (int32_t i = 0; i < outputs[kOUT_CONV_STATE_IDX].nbDims; ++i)
    {
        outputs[kOUT_CONV_STATE_IDX].d[i] = inputs[kIN_CONV_STATE_IDX].d[i];
    }
    return 0;
}

bool GatedDeltaNetCausalConv1dPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS)
    {
        return false;
    }
    bool const isLinear = inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;
    if (!isLinear)
    {
        return false;
    }

    // x must be HALF
    if (pos == kIN_X_IDX)
    {
        return inOut[pos].desc.type == nvinfer1::DataType::kHALF;
    }
    // weight and bias may be HALF or FLOAT
    if (pos == kIN_WEIGHT_IDX || pos == kIN_BIAS_IDX)
    {
        return inOut[pos].desc.type == nvinfer1::DataType::kHALF
            || inOut[pos].desc.type == nvinfer1::DataType::kFLOAT;
    }
    // conv_state must be HALF and same type as x
    if (pos == kIN_CONV_STATE_IDX)
    {
        return inOut[pos].desc.type == nvinfer1::DataType::kHALF
            && inOut[pos].desc.type == inOut[kIN_X_IDX].desc.type;
    }
    // outputs must match corresponding inputs
    int32_t const outPos = pos - kNUM_INPUTS;
    if (outPos == kOUT_IDX)
    {
        return inOut[pos].desc.type == inOut[kIN_X_IDX].desc.type;
    }
    if (outPos == kOUT_CONV_STATE_IDX)
    {
        return inOut[pos].desc.type == inOut[kIN_CONV_STATE_IDX].desc.type;
    }
    return false;
}

int32_t GatedDeltaNetCausalConv1dPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    if (nbInputs != kNUM_INPUTS || nbOutputs != kNUM_OUTPUTS)
    {
        return -1;
    }
    return 0;
}

size_t GatedDeltaNetCausalConv1dPlugin::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* /*inputs*/,
    int32_t /*nbInputs*/, nvinfer1::DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept
{
    return 0;
}

int32_t GatedDeltaNetCausalConv1dPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* /*workspace*/,
    cudaStream_t stream) noexcept
{
    auto const& xDesc = inputDesc[kIN_X_IDX];
    int32_t batch = xDesc.dims.d[0];
    int32_t convDim = xDesc.dims.d[1];
    int32_t seqLen = xDesc.dims.d[2];

    if (batch <= 0 || convDim <= 0 || seqLen <= 0)
    {
        return 0;
    }

    half const* x = static_cast<half const*>(inputs[kIN_X_IDX]);
    half const* convState = nullptr;
    if (inputs[kIN_CONV_STATE_IDX] != nullptr)
    {
        convState = static_cast<half const*>(inputs[kIN_CONV_STATE_IDX]);
    }

    // weight and bias may be FLOAT; convert on-the-fly if needed.
    HalfBuffer weightBuf;
    HalfBuffer biasBuf;

    int32_t const weightNumel = convDim * mKernelSize;
    weightBuf.acquire(inputs[kIN_WEIGHT_IDX], inputDesc[kIN_WEIGHT_IDX].type, weightNumel, stream);
    if (weightBuf.data == nullptr)
    {
        return -1;
    }

    int32_t const biasNumel = convDim;
    biasBuf.acquire(inputs[kIN_BIAS_IDX], inputDesc[kIN_BIAS_IDX].type, biasNumel, stream);
    if (biasBuf.data == nullptr)
    {
        weightBuf.release(stream);
        return -1;
    }

    half* output = static_cast<half*>(outputs[kOUT_IDX]);
    half* convStateOut = static_cast<half*>(outputs[kOUT_CONV_STATE_IDX]);

    int32_t status = 0;
    if (seqLen == 1)
    {
        trt_edgellm::kernels::causalConv1dDecode(
            x, convState, weightBuf.data, biasBuf.data, output, convStateOut, batch, convDim, mKernelSize, mActivation,
            stream);
    }
    else
    {
        trt_edgellm::kernels::causalConv1dPrefill(
            x, weightBuf.data, biasBuf.data, output, convStateOut, batch, convDim, seqLen, mKernelSize, mActivation,
            stream);
    }

    weightBuf.release(stream);
    biasBuf.release(stream);
    return status;
}

int32_t GatedDeltaNetCausalConv1dPlugin::onShapeChange(nvinfer1::PluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    nvinfer1::PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept
{
    return 0;
}

nvinfer1::IPluginV3* GatedDeltaNetCausalConv1dPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* /*context*/) noexcept
{
    return clone();
}

nvinfer1::PluginFieldCollection const* GatedDeltaNetCausalConv1dPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(FIELD_KERNEL_SIZE, &mKernelSize, nvinfer1::PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back(FIELD_ACTIVATION, &mActivation, nvinfer1::PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

void GatedDeltaNetCausalConv1dPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

// Creator
GatedDeltaNetCausalConv1dPluginCreator::GatedDeltaNetCausalConv1dPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(FIELD_KERNEL_SIZE, nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back(FIELD_ACTIVATION, nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* GatedDeltaNetCausalConv1dPluginCreator::getPluginName() const noexcept
{
    return kGDN_CAUSAL_CONV_PLUGIN_NAME;
}

char const* GatedDeltaNetCausalConv1dPluginCreator::getPluginVersion() const noexcept
{
    return kGDN_CAUSAL_CONV_PLUGIN_VERSION;
}

nvinfer1::PluginFieldCollection const* GatedDeltaNetCausalConv1dPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void GatedDeltaNetCausalConv1dPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

char const* GatedDeltaNetCausalConv1dPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

nvinfer1::IPluginV3* GatedDeltaNetCausalConv1dPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase /*phase*/) noexcept
{
    int32_t kernelSize = 4;
    int32_t activation = 0;
    try
    {
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            std::string fieldName(fc->fields[i].name);
            if (fieldName == FIELD_KERNEL_SIZE && fc->fields[i].data != nullptr)
            {
                kernelSize = *static_cast<int32_t const*>(fc->fields[i].data);
            }
            else if (fieldName == FIELD_ACTIVATION && fc->fields[i].data != nullptr)
            {
                activation = *static_cast<int32_t const*>(fc->fields[i].data);
            }
        }
        auto* plugin = new GatedDeltaNetCausalConv1dPlugin(name, kernelSize, activation);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create GatedDeltaNetCausalConv1dPlugin: %s", e.what());
    }
    return nullptr;
}

nvinfer1::PluginFieldCollection GatedDeltaNetCausalConv1dPluginCreator::mFieldCollection{};
std::vector<nvinfer1::PluginField> GatedDeltaNetCausalConv1dPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GatedDeltaNetCausalConv1dPluginCreator);

} // namespace plugins
} // namespace trt_edgellm
