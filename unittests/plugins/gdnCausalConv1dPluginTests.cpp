/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "common/cudaUtils.h"
#include "common/safetensorsUtils.h"
#include "common/tensor.h"
#include "plugins/gdnPlugin/gdnCausalConv1dPlugin.h"
#include "testUtils.h"

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

rt::Tensor const* findTensorByName(std::vector<rt::Tensor> const& tensors, std::string const& name)
{
    for (auto const& t : tensors)
    {
        if (t.getName() == name)
        {
            return &t;
        }
    }
    return nullptr;
}

void copyTensorToHost(rt::Tensor const& src, std::vector<half>& dst, cudaStream_t stream)
{
    dst.resize(src.getShape().volume());
    CUDA_CHECK(
        cudaMemcpyAsync(dst.data(), src.rawPointer(), dst.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
}

PluginTensorDesc makePluginTensorDesc(Dims const& dims, DataType dtype)
{
    PluginTensorDesc desc{};
    desc.dims.nbDims = dims.nbDims;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        desc.dims.d[i] = dims.d[i];
    }
    desc.type = dtype;
    desc.format = TensorFormat::kLINEAR;
    return desc;
}

std::string getResourcePath(std::string const& filename)
{
    char const* testSrcdir = std::getenv("TEST_SRCDIR");
    char const* testWorkspace = std::getenv("TEST_WORKSPACE");
    if (testSrcdir && testWorkspace)
    {
        return std::string(testSrcdir) + "/" + testWorkspace + "/unittests/resources/" + filename;
    }
    // Fallback for local execution outside bazel test
    return "unittests/resources/" + filename;
}

} // anonymous namespace

// ============================================================================
// Plugin Creator API tests
// ============================================================================

TEST(GatedDeltaNetCausalConv1dPluginCreator, BasicApi)
{
    plugins::GatedDeltaNetCausalConv1dPluginCreator creator;
    EXPECT_STREQ(creator.getPluginName(), "GatedDeltaNetCausalConv1d");
    EXPECT_STREQ(creator.getPluginVersion(), "1");

    PluginFieldCollection const* fc = creator.getFieldNames();
    ASSERT_NE(fc, nullptr);
    EXPECT_EQ(fc->nbFields, 2);

    creator.setPluginNamespace("trt_edgellm");
    EXPECT_STREQ(creator.getPluginNamespace(), "trt_edgellm");
}

TEST(GatedDeltaNetCausalConv1dPluginCreator, CreatePlugin)
{
    plugins::GatedDeltaNetCausalConv1dPluginCreator creator;

    int32_t kernelSize = 4;
    int32_t activation = 0;
    std::vector<PluginField> fields;
    fields.emplace_back("kernel_size", &kernelSize, PluginFieldType::kINT32, 1);
    fields.emplace_back("activation", &activation, PluginFieldType::kINT32, 1);
    PluginFieldCollection fc{static_cast<int32_t>(fields.size()), fields.data()};

    IPluginV3* plugin = creator.createPlugin("test_conv", &fc, TensorRTPhase::kBUILD);
    ASSERT_NE(plugin, nullptr);

    auto* core = plugin->getCapabilityInterface(PluginCapabilityType::kCORE);
    ASSERT_NE(core, nullptr);
    EXPECT_STREQ(static_cast<IPluginV3OneCore*>(core)->getPluginName(), "GatedDeltaNetCausalConv1d");
    EXPECT_STREQ(static_cast<IPluginV3OneCore*>(core)->getPluginVersion(), "1");

    delete plugin;
}

// ============================================================================
// Plugin Core / Build / Runtime API tests (without enqueue)
// ============================================================================

TEST(GatedDeltaNetCausalConv1dPlugin, CoreApi)
{
    plugins::GatedDeltaNetCausalConv1dPlugin plugin("test_layer", 4, 0);
    plugin.setPluginNamespace("trt_edgellm");

    EXPECT_STREQ(plugin.getPluginName(), "GatedDeltaNetCausalConv1d");
    EXPECT_STREQ(plugin.getPluginVersion(), "1");
    EXPECT_STREQ(plugin.getPluginNamespace(), "trt_edgellm");
    EXPECT_EQ(plugin.getNbOutputs(), 2);

    // getCapabilityInterface
    EXPECT_EQ(plugin.getCapabilityInterface(PluginCapabilityType::kBUILD), static_cast<IPluginV3OneBuild*>(&plugin));
    EXPECT_EQ(
        plugin.getCapabilityInterface(PluginCapabilityType::kRUNTIME), static_cast<IPluginV3OneRuntime*>(&plugin));
    EXPECT_EQ(plugin.getCapabilityInterface(PluginCapabilityType::kCORE), static_cast<IPluginV3OneCore*>(&plugin));

    // clone
    IPluginV3* cloned = plugin.clone();
    ASSERT_NE(cloned, nullptr);
    auto* clonedCore = cloned->getCapabilityInterface(PluginCapabilityType::kCORE);
    ASSERT_NE(clonedCore, nullptr);
    EXPECT_STREQ(static_cast<IPluginV3OneCore*>(clonedCore)->getPluginName(), "GatedDeltaNetCausalConv1d");
    delete cloned;

    // onShapeChange
    EXPECT_EQ(plugin.onShapeChange(nullptr, 0, nullptr, 0), 0);
}

TEST(GatedDeltaNetCausalConv1dPlugin, BuildApi)
{
    plugins::GatedDeltaNetCausalConv1dPlugin plugin("test_layer", 4, 0);

    // getOutputDataTypes
    DataType inputTypes[4] = {DataType::kHALF, DataType::kHALF, DataType::kHALF, DataType::kHALF};
    DataType outputTypes[2];
    EXPECT_EQ(plugin.getOutputDataTypes(outputTypes, 2, inputTypes, 4), 0);
    EXPECT_EQ(outputTypes[0], DataType::kHALF);
    EXPECT_EQ(outputTypes[1], DataType::kHALF);

    // getOutputShapes requires IExprBuilder which cannot be easily constructed here.
    // It is tested indirectly via enqueue path.

    // supportsFormatCombination
    DynamicPluginTensorDesc inOut[6];
    for (int i = 0; i < 6; ++i)
    {
        inOut[i].desc.type = DataType::kHALF;
        inOut[i].desc.format = TensorFormat::kLINEAR;
    }
    EXPECT_TRUE(plugin.supportsFormatCombination(0, inOut, 4, 2));
    EXPECT_TRUE(plugin.supportsFormatCombination(3, inOut, 4, 2));

    // configurePlugin
    DynamicPluginTensorDesc inDesc[4];
    DynamicPluginTensorDesc outDesc[2];
    EXPECT_EQ(plugin.configurePlugin(inDesc, 4, outDesc, 2), 0);

    // getWorkspaceSize
    EXPECT_EQ(plugin.getWorkspaceSize(inDesc, 4, outDesc, 2), 0);
}

TEST(GatedDeltaNetCausalConv1dPlugin, SerializationApi)
{
    plugins::GatedDeltaNetCausalConv1dPlugin plugin("test_layer", 4, 0);

    PluginFieldCollection const* fc = plugin.getFieldsToSerialize();
    ASSERT_NE(fc, nullptr);
    EXPECT_EQ(fc->nbFields, 2);

    // attachToContext should clone
    IPluginV3* attached = plugin.attachToContext(nullptr);
    ASSERT_NE(attached, nullptr);
    delete attached;

    EXPECT_EQ(plugin.onShapeChange(nullptr, 0, nullptr, 0), 0);
}

// ============================================================================
// Numerical correctness: Prefill path
// ============================================================================

TEST(GatedDeltaNetCausalConv1dPlugin, PrefillNumerical)
{
    std::string const resourcePath = getResourcePath("gdn_causal_conv1d_1b_32d_4s_4k_prefill.safetensors");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<rt::Tensor> tensors;
    ASSERT_TRUE(rt::safetensors::loadSafetensors(resourcePath, tensors, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    rt::Tensor const* xTensor = findTensorByName(tensors, "x");
    rt::Tensor const* wTensor = findTensorByName(tensors, "weight");
    rt::Tensor const* bTensor = findTensorByName(tensors, "bias");
    rt::Tensor const* outRefTensor = findTensorByName(tensors, "output");
    rt::Tensor const* stateRefTensor = findTensorByName(tensors, "conv_state_out");

    ASSERT_NE(xTensor, nullptr);
    ASSERT_NE(wTensor, nullptr);
    ASSERT_NE(bTensor, nullptr);
    ASSERT_NE(outRefTensor, nullptr);
    ASSERT_NE(stateRefTensor, nullptr);

    rt::Coords xShape = xTensor->getShape();
    int32_t kernelSize = static_cast<int32_t>(wTensor->getShape()[1]);

    // loadSafetensors already places tensors on GPU; use them directly as inputs.
    rt::Tensor outDevice(xShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor stateDevice(stateRefTensor->getShape(), rt::DeviceType::kGPU, DataType::kHALF);
    CUDA_CHECK(cudaMemsetAsync(outDevice.rawPointer(), 0, outDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(stateDevice.rawPointer(), 0, stateDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    plugins::GatedDeltaNetCausalConv1dPlugin plugin("test_conv", kernelSize, 0);

    PluginTensorDesc inputDesc[4];
    inputDesc[0] = makePluginTensorDesc(xTensor->getTRTDims(), DataType::kHALF);
    inputDesc[1] = makePluginTensorDesc(Dims(), DataType::kHALF); // optional conv_state (not used for prefill)
    inputDesc[2] = makePluginTensorDesc(wTensor->getTRTDims(), DataType::kHALF);
    inputDesc[3] = makePluginTensorDesc(bTensor->getTRTDims(), DataType::kHALF);

    PluginTensorDesc outputDesc[2];
    outputDesc[0] = makePluginTensorDesc(outDevice.getTRTDims(), DataType::kHALF);
    outputDesc[1] = makePluginTensorDesc(stateDevice.getTRTDims(), DataType::kHALF);

    void const* inputs[4] = {xTensor->rawPointer(), nullptr, wTensor->rawPointer(), bTensor->rawPointer()};
    void* outputs[2] = {outDevice.rawPointer(), stateDevice.rawPointer()};

    EXPECT_EQ(plugin.enqueue(inputDesc, outputDesc, inputs, outputs, nullptr, stream), 0);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Compare output
    std::vector<half> outHost;
    copyTensorToHost(outDevice, outHost, stream);
    std::vector<half> outRefHost;
    copyTensorToHost(*outRefTensor, outRefHost, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto [rtol, atol] = getTolerance<half>();
    int32_t mismatches = 0;
    for (size_t i = 0; i < outHost.size(); ++i)
    {
        if (!isclose(outHost[i], outRefHost[i], rtol, atol))
        {
            if (mismatches < 5)
            {
                std::cout << "Output mismatch at " << i << ": got " << __half2float(outHost[i]) << ", expected "
                          << __half2float(outRefHost[i]) << std::endl;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << "Output mismatch count: " << mismatches;

    // Compare conv_state_out
    std::vector<half> stateHost;
    copyTensorToHost(stateDevice, stateHost, stream);
    std::vector<half> stateRefHost;
    copyTensorToHost(*stateRefTensor, stateRefHost, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    mismatches = 0;
    for (size_t i = 0; i < stateHost.size(); ++i)
    {
        if (!isclose(stateHost[i], stateRefHost[i], rtol, atol))
        {
            if (mismatches < 5)
            {
                std::cout << "State mismatch at " << i << ": got " << __half2float(stateHost[i]) << ", expected "
                          << __half2float(stateRefHost[i]) << std::endl;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << "State mismatch count: " << mismatches;

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// Numerical correctness: Decode path
// ============================================================================

TEST(GatedDeltaNetCausalConv1dPlugin, DecodeNumerical)
{
    std::string const resourcePath = getResourcePath("gdn_causal_conv1d_1b_16d_1s_4k_decode.safetensors");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<rt::Tensor> tensors;
    ASSERT_TRUE(rt::safetensors::loadSafetensors(resourcePath, tensors, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    rt::Tensor const* xTensor = findTensorByName(tensors, "x");
    rt::Tensor const* wTensor = findTensorByName(tensors, "weight");
    rt::Tensor const* bTensor = findTensorByName(tensors, "bias");
    rt::Tensor const* outRefTensor = findTensorByName(tensors, "output");
    rt::Tensor const* stateRefTensor = findTensorByName(tensors, "conv_state_out");

    ASSERT_NE(xTensor, nullptr);
    ASSERT_NE(wTensor, nullptr);
    ASSERT_NE(bTensor, nullptr);
    ASSERT_NE(outRefTensor, nullptr);
    ASSERT_NE(stateRefTensor, nullptr);

    rt::Coords xShape = xTensor->getShape();
    int32_t batch = static_cast<int32_t>(xShape[0]);
    int32_t convDim = static_cast<int32_t>(xShape[1]);
    int32_t kernelSize = static_cast<int32_t>(wTensor->getShape()[1]);

    // For decode path we need conv_state input (zero-initialized)
    rt::Tensor convStateIn(rt::Coords{batch, convDim, kernelSize}, rt::DeviceType::kGPU, DataType::kHALF);
    CUDA_CHECK(cudaMemsetAsync(convStateIn.rawPointer(), 0, convStateIn.getMemoryCapacity(), stream));

    rt::Tensor outDevice(xShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor stateDevice(stateRefTensor->getShape(), rt::DeviceType::kGPU, DataType::kHALF);
    CUDA_CHECK(cudaMemsetAsync(outDevice.rawPointer(), 0, outDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(stateDevice.rawPointer(), 0, stateDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    plugins::GatedDeltaNetCausalConv1dPlugin plugin("test_conv", kernelSize, 0);

    PluginTensorDesc inputDesc[4];
    inputDesc[0] = makePluginTensorDesc(xTensor->getTRTDims(), DataType::kHALF);
    inputDesc[1] = makePluginTensorDesc(convStateIn.getTRTDims(), DataType::kHALF);
    inputDesc[2] = makePluginTensorDesc(wTensor->getTRTDims(), DataType::kHALF);
    inputDesc[3] = makePluginTensorDesc(bTensor->getTRTDims(), DataType::kHALF);

    PluginTensorDesc outputDesc[2];
    outputDesc[0] = makePluginTensorDesc(outDevice.getTRTDims(), DataType::kHALF);
    outputDesc[1] = makePluginTensorDesc(stateDevice.getTRTDims(), DataType::kHALF);

    void const* inputs[4]
        = {xTensor->rawPointer(), convStateIn.rawPointer(), wTensor->rawPointer(), bTensor->rawPointer()};
    void* outputs[2] = {outDevice.rawPointer(), stateDevice.rawPointer()};

    EXPECT_EQ(plugin.enqueue(inputDesc, outputDesc, inputs, outputs, nullptr, stream), 0);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Compare output
    std::vector<half> outHost;
    copyTensorToHost(outDevice, outHost, stream);
    std::vector<half> outRefHost;
    copyTensorToHost(*outRefTensor, outRefHost, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto [rtol, atol] = getTolerance<half>();
    int32_t mismatches = 0;
    for (size_t i = 0; i < outHost.size(); ++i)
    {
        if (!isclose(outHost[i], outRefHost[i], rtol, atol))
        {
            if (mismatches < 5)
            {
                std::cout << "Decode output mismatch at " << i << ": got " << __half2float(outHost[i]) << ", expected "
                          << __half2float(outRefHost[i]) << std::endl;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << "Decode output mismatch count: " << mismatches;

    // Compare conv_state_out
    std::vector<half> stateHost;
    copyTensorToHost(stateDevice, stateHost, stream);
    std::vector<half> stateRefHost;
    copyTensorToHost(*stateRefTensor, stateRefHost, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    mismatches = 0;
    for (size_t i = 0; i < stateHost.size(); ++i)
    {
        if (!isclose(stateHost[i], stateRefHost[i], rtol, atol))
        {
            if (mismatches < 5)
            {
                std::cout << "Decode state mismatch at " << i << ": got " << __half2float(stateHost[i]) << ", expected "
                          << __half2float(stateRefHost[i]) << std::endl;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << "Decode state mismatch count: " << mismatches;

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// Batch2 prefill numerical test
// ============================================================================

TEST(GatedDeltaNetCausalConv1dPlugin, Batch2PrefillNumerical)
{
    std::string const resourcePath = getResourcePath("gdn_causal_conv1d_2b_64d_16s_3k_prefill.safetensors");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<rt::Tensor> tensors;
    ASSERT_TRUE(rt::safetensors::loadSafetensors(resourcePath, tensors, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    rt::Tensor const* xTensor = findTensorByName(tensors, "x");
    rt::Tensor const* wTensor = findTensorByName(tensors, "weight");
    rt::Tensor const* bTensor = findTensorByName(tensors, "bias");
    rt::Tensor const* outRefTensor = findTensorByName(tensors, "output");
    rt::Tensor const* stateRefTensor = findTensorByName(tensors, "conv_state_out");

    ASSERT_NE(xTensor, nullptr);
    ASSERT_NE(wTensor, nullptr);
    ASSERT_NE(bTensor, nullptr);
    ASSERT_NE(outRefTensor, nullptr);
    ASSERT_NE(stateRefTensor, nullptr);

    rt::Coords xShape = xTensor->getShape();
    int32_t kernelSize = static_cast<int32_t>(wTensor->getShape()[1]);

    rt::Tensor outDevice(xShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor stateDevice(stateRefTensor->getShape(), rt::DeviceType::kGPU, DataType::kHALF);
    CUDA_CHECK(cudaMemsetAsync(outDevice.rawPointer(), 0, outDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(stateDevice.rawPointer(), 0, stateDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    plugins::GatedDeltaNetCausalConv1dPlugin plugin("test_conv", kernelSize, 0);

    PluginTensorDesc inputDesc[4];
    inputDesc[0] = makePluginTensorDesc(xTensor->getTRTDims(), DataType::kHALF);
    Dims emptyDims{};
    emptyDims.nbDims = 0;
    inputDesc[1] = makePluginTensorDesc(emptyDims, DataType::kHALF); // optional conv_state
    inputDesc[2] = makePluginTensorDesc(wTensor->getTRTDims(), DataType::kHALF);
    inputDesc[3] = makePluginTensorDesc(bTensor->getTRTDims(), DataType::kHALF);

    PluginTensorDesc outputDesc[2];
    outputDesc[0] = makePluginTensorDesc(outDevice.getTRTDims(), DataType::kHALF);
    outputDesc[1] = makePluginTensorDesc(stateDevice.getTRTDims(), DataType::kHALF);

    void const* inputs[4] = {xTensor->rawPointer(), nullptr, wTensor->rawPointer(), bTensor->rawPointer()};
    void* outputs[2] = {outDevice.rawPointer(), stateDevice.rawPointer()};

    EXPECT_EQ(plugin.enqueue(inputDesc, outputDesc, inputs, outputs, nullptr, stream), 0);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Compare output
    std::vector<half> outHost;
    copyTensorToHost(outDevice, outHost, stream);
    std::vector<half> outRefHost;
    copyTensorToHost(*outRefTensor, outRefHost, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto [rtol, atol] = getTolerance<half>();
    int32_t mismatches = 0;
    for (size_t i = 0; i < outHost.size(); ++i)
    {
        if (!isclose(outHost[i], outRefHost[i], rtol, atol))
        {
            if (mismatches < 5)
            {
                std::cout << "Batch2 output mismatch at " << i << ": got " << __half2float(outHost[i]) << ", expected "
                          << __half2float(outRefHost[i]) << std::endl;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << "Batch2 output mismatch count: " << mismatches;

    // Compare conv_state_out
    std::vector<half> stateHost;
    copyTensorToHost(stateDevice, stateHost, stream);
    std::vector<half> stateRefHost;
    copyTensorToHost(*stateRefTensor, stateRefHost, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    mismatches = 0;
    for (size_t i = 0; i < stateHost.size(); ++i)
    {
        if (!isclose(stateHost[i], stateRefHost[i], rtol, atol))
        {
            if (mismatches < 5)
            {
                std::cout << "Batch2 state mismatch at " << i << ": got " << __half2float(stateHost[i]) << ", expected "
                          << __half2float(stateRefHost[i]) << std::endl;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << "Batch2 state mismatch count: " << mismatches;

    CUDA_CHECK(cudaStreamDestroy(stream));
}
