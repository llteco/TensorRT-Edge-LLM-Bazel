/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "common/cudaUtils.h"
#include "common/safetensorsUtils.h"
#include "common/tensor.h"
#include "plugins/gdnPlugin/gatedDeltaRulePlugin.h"
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
    CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.rawPointer(), dst.size() * sizeof(half), cudaMemcpyDeviceToHost, stream));
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
    return "unittests/resources/" + filename;
}

} // anonymous namespace

// ============================================================================
// Plugin Creator API tests
// ============================================================================

TEST(GatedDeltaRulePluginCreator, BasicApi)
{
    plugins::GatedDeltaRulePluginCreator creator;
    EXPECT_STREQ(creator.getPluginName(), "GatedDeltaRule");
    EXPECT_STREQ(creator.getPluginVersion(), "1");

    PluginFieldCollection const* fc = creator.getFieldNames();
    ASSERT_NE(fc, nullptr);
    EXPECT_EQ(fc->nbFields, 4);

    creator.setPluginNamespace("trt_edgellm");
    EXPECT_STREQ(creator.getPluginNamespace(), "trt_edgellm");
}

TEST(GatedDeltaRulePluginCreator, CreatePlugin)
{
    plugins::GatedDeltaRulePluginCreator creator;

    int32_t numVHeads = 8;
    int32_t headVDim = 128;
    int32_t headKDim = 64;
    int32_t useQKL2Norm = 1;
    std::vector<PluginField> fields;
    fields.emplace_back("num_v_heads", &numVHeads, PluginFieldType::kINT32, 1);
    fields.emplace_back("head_v_dim", &headVDim, PluginFieldType::kINT32, 1);
    fields.emplace_back("head_k_dim", &headKDim, PluginFieldType::kINT32, 1);
    fields.emplace_back("use_qk_l2norm", &useQKL2Norm, PluginFieldType::kINT32, 1);
    PluginFieldCollection fc{static_cast<int32_t>(fields.size()), fields.data()};

    IPluginV3* plugin = creator.createPlugin("test_delta", &fc, TensorRTPhase::kBUILD);
    ASSERT_NE(plugin, nullptr);

    auto* core = plugin->getCapabilityInterface(PluginCapabilityType::kCORE);
    ASSERT_NE(core, nullptr);
    EXPECT_STREQ(static_cast<IPluginV3OneCore*>(core)->getPluginName(), "GatedDeltaRule");
    EXPECT_STREQ(static_cast<IPluginV3OneCore*>(core)->getPluginVersion(), "1");

    delete plugin;
}

// ============================================================================
// Plugin Core / Build / Runtime API tests (without enqueue)
// ============================================================================

TEST(GatedDeltaRulePlugin, CoreApi)
{
    plugins::GatedDeltaRulePlugin plugin("test_layer", 8, 128, 64, 1);
    plugin.setPluginNamespace("trt_edgellm");

    EXPECT_STREQ(plugin.getPluginName(), "GatedDeltaRule");
    EXPECT_STREQ(plugin.getPluginVersion(), "1");
    EXPECT_STREQ(plugin.getPluginNamespace(), "trt_edgellm");
    EXPECT_EQ(plugin.getNbOutputs(), 2);

    EXPECT_EQ(plugin.getCapabilityInterface(PluginCapabilityType::kBUILD), static_cast<IPluginV3OneBuild*>(&plugin));
    EXPECT_EQ(plugin.getCapabilityInterface(PluginCapabilityType::kRUNTIME), static_cast<IPluginV3OneRuntime*>(&plugin));
    EXPECT_EQ(plugin.getCapabilityInterface(PluginCapabilityType::kCORE), static_cast<IPluginV3OneCore*>(&plugin));

    IPluginV3* cloned = plugin.clone();
    ASSERT_NE(cloned, nullptr);
    auto* clonedCore = cloned->getCapabilityInterface(PluginCapabilityType::kCORE);
    ASSERT_NE(clonedCore, nullptr);
    EXPECT_STREQ(static_cast<IPluginV3OneCore*>(clonedCore)->getPluginName(), "GatedDeltaRule");
    delete cloned;

    EXPECT_EQ(plugin.onShapeChange(nullptr, 0, nullptr, 0), 0);
}

TEST(GatedDeltaRulePlugin, BuildApi)
{
    plugins::GatedDeltaRulePlugin plugin("test_layer", 8, 128, 64, 1);

    // getOutputDataTypes
    DataType inputTypes[6] = {DataType::kHALF, DataType::kHALF, DataType::kHALF, DataType::kHALF, DataType::kHALF,
        DataType::kHALF};
    DataType outputTypes[2];
    EXPECT_EQ(plugin.getOutputDataTypes(outputTypes, 2, inputTypes, 6), 0);
    EXPECT_EQ(outputTypes[0], DataType::kHALF);
    EXPECT_EQ(outputTypes[1], DataType::kHALF);

    // supportsFormatCombination
    DynamicPluginTensorDesc inOut[8];
    for (int i = 0; i < 8; ++i)
    {
        inOut[i].desc.type = DataType::kHALF;
        inOut[i].desc.format = TensorFormat::kLINEAR;
    }
    EXPECT_TRUE(plugin.supportsFormatCombination(0, inOut, 6, 2));
    EXPECT_TRUE(plugin.supportsFormatCombination(5, inOut, 6, 2));

    // configurePlugin
    DynamicPluginTensorDesc inDesc[6];
    DynamicPluginTensorDesc outDesc[2];
    EXPECT_EQ(plugin.configurePlugin(inDesc, 6, outDesc, 2), 0);

    // getWorkspaceSize (with use_qk_l2norm=1)
    EXPECT_GT(plugin.getWorkspaceSize(inDesc, 6, outDesc, 2), 0);
}

TEST(GatedDeltaRulePlugin, SerializationApi)
{
    plugins::GatedDeltaRulePlugin plugin("test_layer", 8, 128, 64, 1);

    PluginFieldCollection const* fc = plugin.getFieldsToSerialize();
    ASSERT_NE(fc, nullptr);
    EXPECT_EQ(fc->nbFields, 4);

    IPluginV3* attached = plugin.attachToContext(nullptr);
    ASSERT_NE(attached, nullptr);
    delete attached;
}

// ============================================================================
// Numerical correctness: decode path
// ============================================================================

TEST(GatedDeltaRulePlugin, DecodeNumerical)
{
    std::string const resourcePath = getResourcePath("gated_delta_rule_1b_1s_8h_64k_64v_decode.safetensors");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<rt::Tensor> tensors;
    ASSERT_TRUE(rt::safetensors::loadSafetensors(resourcePath, tensors, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    rt::Tensor const* qTensor = findTensorByName(tensors, "query");
    rt::Tensor const* kTensor = findTensorByName(tensors, "key");
    rt::Tensor const* vTensor = findTensorByName(tensors, "value");
    rt::Tensor const* gTensor = findTensorByName(tensors, "g");
    rt::Tensor const* betaTensor = findTensorByName(tensors, "beta");
    rt::Tensor const* outRefTensor = findTensorByName(tensors, "output");
    rt::Tensor const* stateRefTensor = findTensorByName(tensors, "final_state");

    ASSERT_NE(qTensor, nullptr);
    ASSERT_NE(kTensor, nullptr);
    ASSERT_NE(vTensor, nullptr);
    ASSERT_NE(gTensor, nullptr);
    ASSERT_NE(betaTensor, nullptr);
    ASSERT_NE(outRefTensor, nullptr);
    ASSERT_NE(stateRefTensor, nullptr);

    rt::Coords qShape = qTensor->getShape();
    int32_t numVHeads = static_cast<int32_t>(qShape[2]);
    int32_t headKDim = static_cast<int32_t>(qShape[3]);
    int32_t headVDim = static_cast<int32_t>(vTensor->getShape()[3]);

    rt::Tensor outDevice(outRefTensor->getShape(), rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor finalStateDevice(stateRefTensor->getShape(), rt::DeviceType::kGPU, DataType::kHALF);
    CUDA_CHECK(cudaMemsetAsync(outDevice.rawPointer(), 0, outDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(finalStateDevice.rawPointer(), 0, finalStateDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    plugins::GatedDeltaRulePlugin plugin("test_delta", numVHeads, headVDim, headKDim, 1);

    PluginTensorDesc inputDesc[6];
    inputDesc[0] = makePluginTensorDesc(qTensor->getTRTDims(), DataType::kHALF);
    inputDesc[1] = makePluginTensorDesc(kTensor->getTRTDims(), DataType::kHALF);
    inputDesc[2] = makePluginTensorDesc(vTensor->getTRTDims(), DataType::kHALF);
    inputDesc[3] = makePluginTensorDesc(gTensor->getTRTDims(), DataType::kHALF);
    inputDesc[4] = makePluginTensorDesc(betaTensor->getTRTDims(), DataType::kHALF);
    Dims emptyDims{};
    emptyDims.nbDims = 0;
    inputDesc[5] = makePluginTensorDesc(emptyDims, DataType::kHALF); // optional initial_state

    PluginTensorDesc outputDesc[2];
    outputDesc[0] = makePluginTensorDesc(outDevice.getTRTDims(), DataType::kHALF);
    outputDesc[1] = makePluginTensorDesc(finalStateDevice.getTRTDims(), DataType::kHALF);

    // Allocate workspace for QKNorm (required when use_qk_l2norm=1)
    DynamicPluginTensorDesc dynInDesc[6];
    for (int i = 0; i < 6; ++i)
    {
        dynInDesc[i].desc = inputDesc[i];
    }
    size_t workspaceSize = plugin.getWorkspaceSize(dynInDesc, 6, nullptr, 2);
    void* workspace = nullptr;
    if (workspaceSize > 0)
    {
        CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));
    }

    void const* inputs[6]
        = {qTensor->rawPointer(), kTensor->rawPointer(), vTensor->rawPointer(), gTensor->rawPointer(),
            betaTensor->rawPointer(), nullptr};
    void* outputs[2] = {outDevice.rawPointer(), finalStateDevice.rawPointer()};

    EXPECT_EQ(plugin.enqueue(inputDesc, outputDesc, inputs, outputs, workspace, stream), 0);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto [rtol, atol] = getTolerance<half>();

    // Compare output
    std::vector<half> outHost;
    copyTensorToHost(outDevice, outHost, stream);
    std::vector<half> outRefHost;
    copyTensorToHost(*outRefTensor, outRefHost, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

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

    // Compare final_state
    std::vector<half> stateHost;
    copyTensorToHost(finalStateDevice, stateHost, stream);
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
// Numerical correctness: decode path with asymmetric K/V dims
// ============================================================================

TEST(GatedDeltaRulePlugin, DecodeAsymNumerical)
{
    std::string const resourcePath = getResourcePath("gated_delta_rule_1b_1s_8h_64k_128v_decode_asym.safetensors");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<rt::Tensor> tensors;
    ASSERT_TRUE(rt::safetensors::loadSafetensors(resourcePath, tensors, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    rt::Tensor const* qTensor = findTensorByName(tensors, "query");
    rt::Tensor const* kTensor = findTensorByName(tensors, "key");
    rt::Tensor const* vTensor = findTensorByName(tensors, "value");
    rt::Tensor const* gTensor = findTensorByName(tensors, "g");
    rt::Tensor const* betaTensor = findTensorByName(tensors, "beta");
    rt::Tensor const* outRefTensor = findTensorByName(tensors, "output");
    rt::Tensor const* stateRefTensor = findTensorByName(tensors, "final_state");

    ASSERT_NE(qTensor, nullptr);
    ASSERT_NE(kTensor, nullptr);
    ASSERT_NE(vTensor, nullptr);
    ASSERT_NE(gTensor, nullptr);
    ASSERT_NE(betaTensor, nullptr);
    ASSERT_NE(outRefTensor, nullptr);
    ASSERT_NE(stateRefTensor, nullptr);

    rt::Coords qShape = qTensor->getShape();
    int32_t numVHeads = static_cast<int32_t>(qShape[2]);
    int32_t headKDim = static_cast<int32_t>(qShape[3]);
    int32_t headVDim = static_cast<int32_t>(vTensor->getShape()[3]);

    rt::Tensor outDevice(outRefTensor->getShape(), rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor finalStateDevice(stateRefTensor->getShape(), rt::DeviceType::kGPU, DataType::kHALF);
    CUDA_CHECK(cudaMemsetAsync(outDevice.rawPointer(), 0, outDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(finalStateDevice.rawPointer(), 0, finalStateDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    plugins::GatedDeltaRulePlugin plugin("test_delta", numVHeads, headVDim, headKDim, 1);

    PluginTensorDesc inputDesc[6];
    inputDesc[0] = makePluginTensorDesc(qTensor->getTRTDims(), DataType::kHALF);
    inputDesc[1] = makePluginTensorDesc(kTensor->getTRTDims(), DataType::kHALF);
    inputDesc[2] = makePluginTensorDesc(vTensor->getTRTDims(), DataType::kHALF);
    inputDesc[3] = makePluginTensorDesc(gTensor->getTRTDims(), DataType::kHALF);
    inputDesc[4] = makePluginTensorDesc(betaTensor->getTRTDims(), DataType::kHALF);
    Dims emptyDims{};
    emptyDims.nbDims = 0;
    inputDesc[5] = makePluginTensorDesc(emptyDims, DataType::kHALF);

    PluginTensorDesc outputDesc[2];
    outputDesc[0] = makePluginTensorDesc(outDevice.getTRTDims(), DataType::kHALF);
    outputDesc[1] = makePluginTensorDesc(finalStateDevice.getTRTDims(), DataType::kHALF);

    // Allocate workspace for QKNorm (required when use_qk_l2norm=1)
    DynamicPluginTensorDesc dynInDesc[6];
    for (int i = 0; i < 6; ++i)
    {
        dynInDesc[i].desc = inputDesc[i];
    }
    size_t workspaceSize = plugin.getWorkspaceSize(dynInDesc, 6, nullptr, 2);
    void* workspace = nullptr;
    if (workspaceSize > 0)
    {
        CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));
    }

    void const* inputs[6]
        = {qTensor->rawPointer(), kTensor->rawPointer(), vTensor->rawPointer(), gTensor->rawPointer(),
            betaTensor->rawPointer(), nullptr};
    void* outputs[2] = {outDevice.rawPointer(), finalStateDevice.rawPointer()};

    EXPECT_EQ(plugin.enqueue(inputDesc, outputDesc, inputs, outputs, workspace, stream), 0);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto [rtol, atol] = getTolerance<half>();

    std::vector<half> outHost;
    copyTensorToHost(outDevice, outHost, stream);
    std::vector<half> outRefHost;
    copyTensorToHost(*outRefTensor, outRefHost, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int32_t mismatches = 0;
    for (size_t i = 0; i < outHost.size(); ++i)
    {
        if (!isclose(outHost[i], outRefHost[i], rtol, atol))
        {
            if (mismatches < 5)
            {
                std::cout << "Asym output mismatch at " << i << ": got " << __half2float(outHost[i]) << ", expected "
                          << __half2float(outRefHost[i]) << std::endl;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << "Asym output mismatch count: " << mismatches;

    std::vector<half> stateHost;
    copyTensorToHost(finalStateDevice, stateHost, stream);
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
                std::cout << "Asym state mismatch at " << i << ": got " << __half2float(stateHost[i]) << ", expected "
                          << __half2float(stateRefHost[i]) << std::endl;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << "Asym state mismatch count: " << mismatches;

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// Numerical correctness: prefill path
// ============================================================================

TEST(GatedDeltaRulePlugin, PrefillNumerical)
{
    std::string const resourcePath = getResourcePath("gated_delta_rule_1b_4s_16h_128k_128v_prefill.safetensors");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<rt::Tensor> tensors;
    ASSERT_TRUE(rt::safetensors::loadSafetensors(resourcePath, tensors, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    rt::Tensor const* qTensor = findTensorByName(tensors, "query");
    rt::Tensor const* kTensor = findTensorByName(tensors, "key");
    rt::Tensor const* vTensor = findTensorByName(tensors, "value");
    rt::Tensor const* gTensor = findTensorByName(tensors, "g");
    rt::Tensor const* betaTensor = findTensorByName(tensors, "beta");
    rt::Tensor const* outRefTensor = findTensorByName(tensors, "output");
    rt::Tensor const* stateRefTensor = findTensorByName(tensors, "final_state");

    ASSERT_NE(qTensor, nullptr);
    ASSERT_NE(kTensor, nullptr);
    ASSERT_NE(vTensor, nullptr);
    ASSERT_NE(gTensor, nullptr);
    ASSERT_NE(betaTensor, nullptr);
    ASSERT_NE(outRefTensor, nullptr);
    ASSERT_NE(stateRefTensor, nullptr);

    rt::Coords qShape = qTensor->getShape();
    int32_t numVHeads = static_cast<int32_t>(qShape[2]);
    int32_t headKDim = static_cast<int32_t>(qShape[3]);
    int32_t headVDim = static_cast<int32_t>(vTensor->getShape()[3]);

    rt::Tensor outDevice(outRefTensor->getShape(), rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor finalStateDevice(stateRefTensor->getShape(), rt::DeviceType::kGPU, DataType::kHALF);
    CUDA_CHECK(cudaMemsetAsync(outDevice.rawPointer(), 0, outDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(finalStateDevice.rawPointer(), 0, finalStateDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    plugins::GatedDeltaRulePlugin plugin("test_delta", numVHeads, headVDim, headKDim, 1);

    PluginTensorDesc inputDesc[6];
    inputDesc[0] = makePluginTensorDesc(qTensor->getTRTDims(), DataType::kHALF);
    inputDesc[1] = makePluginTensorDesc(kTensor->getTRTDims(), DataType::kHALF);
    inputDesc[2] = makePluginTensorDesc(vTensor->getTRTDims(), DataType::kHALF);
    inputDesc[3] = makePluginTensorDesc(gTensor->getTRTDims(), DataType::kHALF);
    inputDesc[4] = makePluginTensorDesc(betaTensor->getTRTDims(), DataType::kHALF);
    Dims emptyDims{};
    emptyDims.nbDims = 0;
    inputDesc[5] = makePluginTensorDesc(emptyDims, DataType::kHALF);

    PluginTensorDesc outputDesc[2];
    outputDesc[0] = makePluginTensorDesc(outDevice.getTRTDims(), DataType::kHALF);
    outputDesc[1] = makePluginTensorDesc(finalStateDevice.getTRTDims(), DataType::kHALF);

    // Allocate workspace for QKNorm (required when use_qk_l2norm=1)
    DynamicPluginTensorDesc dynInDesc[6];
    for (int i = 0; i < 6; ++i)
    {
        dynInDesc[i].desc = inputDesc[i];
    }
    size_t workspaceSize = plugin.getWorkspaceSize(dynInDesc, 6, nullptr, 2);
    void* workspace = nullptr;
    if (workspaceSize > 0)
    {
        CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));
    }

    void const* inputs[6]
        = {qTensor->rawPointer(), kTensor->rawPointer(), vTensor->rawPointer(), gTensor->rawPointer(),
            betaTensor->rawPointer(), nullptr};
    void* outputs[2] = {outDevice.rawPointer(), finalStateDevice.rawPointer()};

    EXPECT_EQ(plugin.enqueue(inputDesc, outputDesc, inputs, outputs, workspace, stream), 0);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto [rtol, atol] = getTolerance<half>();

    std::vector<half> outHost;
    copyTensorToHost(outDevice, outHost, stream);
    std::vector<half> outRefHost;
    copyTensorToHost(*outRefTensor, outRefHost, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int32_t mismatches = 0;
    for (size_t i = 0; i < outHost.size(); ++i)
    {
        if (!isclose(outHost[i], outRefHost[i], rtol, atol))
        {
            if (mismatches < 5)
            {
                std::cout << "Prefill output mismatch at " << i << ": got " << __half2float(outHost[i]) << ", expected "
                          << __half2float(outRefHost[i]) << std::endl;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << "Prefill output mismatch count: " << mismatches;

    std::vector<half> stateHost;
    copyTensorToHost(finalStateDevice, stateHost, stream);
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
                std::cout << "Prefill state mismatch at " << i << ": got " << __half2float(stateHost[i]) << ", expected "
                          << __half2float(stateRefHost[i]) << std::endl;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << "Prefill state mismatch count: " << mismatches;

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// Numerical correctness: prefill with initial state
// ============================================================================

TEST(GatedDeltaRulePlugin, PrefillWithInitialState)
{
    std::string const resourcePath
        = getResourcePath("gated_delta_rule_1b_8s_16h_128k_128v_prefill_with_state.safetensors");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<rt::Tensor> tensors;
    ASSERT_TRUE(rt::safetensors::loadSafetensors(resourcePath, tensors, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    rt::Tensor const* qTensor = findTensorByName(tensors, "query");
    rt::Tensor const* kTensor = findTensorByName(tensors, "key");
    rt::Tensor const* vTensor = findTensorByName(tensors, "value");
    rt::Tensor const* gTensor = findTensorByName(tensors, "g");
    rt::Tensor const* betaTensor = findTensorByName(tensors, "beta");
    rt::Tensor const* initStateTensor = findTensorByName(tensors, "initial_state");
    rt::Tensor const* outRefTensor = findTensorByName(tensors, "output");
    rt::Tensor const* stateRefTensor = findTensorByName(tensors, "final_state");

    ASSERT_NE(qTensor, nullptr);
    ASSERT_NE(kTensor, nullptr);
    ASSERT_NE(vTensor, nullptr);
    ASSERT_NE(gTensor, nullptr);
    ASSERT_NE(betaTensor, nullptr);
    ASSERT_NE(initStateTensor, nullptr);
    ASSERT_NE(outRefTensor, nullptr);
    ASSERT_NE(stateRefTensor, nullptr);

    rt::Coords qShape = qTensor->getShape();
    int32_t numVHeads = static_cast<int32_t>(qShape[2]);
    int32_t headKDim = static_cast<int32_t>(qShape[3]);
    int32_t headVDim = static_cast<int32_t>(vTensor->getShape()[3]);

    rt::Tensor outDevice(outRefTensor->getShape(), rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor finalStateDevice(stateRefTensor->getShape(), rt::DeviceType::kGPU, DataType::kHALF);
    CUDA_CHECK(cudaMemsetAsync(outDevice.rawPointer(), 0, outDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(finalStateDevice.rawPointer(), 0, finalStateDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    plugins::GatedDeltaRulePlugin plugin("test_delta", numVHeads, headVDim, headKDim, 1);

    PluginTensorDesc inputDesc[6];
    inputDesc[0] = makePluginTensorDesc(qTensor->getTRTDims(), DataType::kHALF);
    inputDesc[1] = makePluginTensorDesc(kTensor->getTRTDims(), DataType::kHALF);
    inputDesc[2] = makePluginTensorDesc(vTensor->getTRTDims(), DataType::kHALF);
    inputDesc[3] = makePluginTensorDesc(gTensor->getTRTDims(), DataType::kHALF);
    inputDesc[4] = makePluginTensorDesc(betaTensor->getTRTDims(), DataType::kHALF);
    inputDesc[5] = makePluginTensorDesc(initStateTensor->getTRTDims(), DataType::kHALF);

    PluginTensorDesc outputDesc[2];
    outputDesc[0] = makePluginTensorDesc(outDevice.getTRTDims(), DataType::kHALF);
    outputDesc[1] = makePluginTensorDesc(finalStateDevice.getTRTDims(), DataType::kHALF);

    // Allocate workspace for QKNorm (required when use_qk_l2norm=1)
    DynamicPluginTensorDesc dynInDesc[6];
    for (int i = 0; i < 6; ++i)
    {
        dynInDesc[i].desc = inputDesc[i];
    }
    size_t workspaceSize = plugin.getWorkspaceSize(dynInDesc, 6, nullptr, 2);
    void* workspace = nullptr;
    if (workspaceSize > 0)
    {
        CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));
    }

    void const* inputs[6]
        = {qTensor->rawPointer(), kTensor->rawPointer(), vTensor->rawPointer(), gTensor->rawPointer(),
            betaTensor->rawPointer(), initStateTensor->rawPointer()};
    void* outputs[2] = {outDevice.rawPointer(), finalStateDevice.rawPointer()};

    EXPECT_EQ(plugin.enqueue(inputDesc, outputDesc, inputs, outputs, workspace, stream), 0);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto [rtol, atol] = getTolerance<half>();

    std::vector<half> outHost;
    copyTensorToHost(outDevice, outHost, stream);
    std::vector<half> outRefHost;
    copyTensorToHost(*outRefTensor, outRefHost, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int32_t mismatches = 0;
    for (size_t i = 0; i < outHost.size(); ++i)
    {
        if (!isclose(outHost[i], outRefHost[i], rtol, atol))
        {
            if (mismatches < 5)
            {
                std::cout << "WithState output mismatch at " << i << ": got " << __half2float(outHost[i])
                          << ", expected " << __half2float(outRefHost[i]) << std::endl;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << "WithState output mismatch count: " << mismatches;

    std::vector<half> stateHost;
    copyTensorToHost(finalStateDevice, stateHost, stream);
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
                std::cout << "WithState state mismatch at " << i << ": got " << __half2float(stateHost[i])
                          << ", expected " << __half2float(stateRefHost[i]) << std::endl;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << "WithState state mismatch count: " << mismatches;

    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// Numerical correctness: batch=2 prefill
// ============================================================================

TEST(GatedDeltaRulePlugin, Batch2PrefillNumerical)
{
    std::string const resourcePath = getResourcePath("gated_delta_rule_2b_16s_8h_64k_64v_prefill_batch2.safetensors");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<rt::Tensor> tensors;
    ASSERT_TRUE(rt::safetensors::loadSafetensors(resourcePath, tensors, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    rt::Tensor const* qTensor = findTensorByName(tensors, "query");
    rt::Tensor const* kTensor = findTensorByName(tensors, "key");
    rt::Tensor const* vTensor = findTensorByName(tensors, "value");
    rt::Tensor const* gTensor = findTensorByName(tensors, "g");
    rt::Tensor const* betaTensor = findTensorByName(tensors, "beta");
    rt::Tensor const* outRefTensor = findTensorByName(tensors, "output");
    rt::Tensor const* stateRefTensor = findTensorByName(tensors, "final_state");

    ASSERT_NE(qTensor, nullptr);
    ASSERT_NE(kTensor, nullptr);
    ASSERT_NE(vTensor, nullptr);
    ASSERT_NE(gTensor, nullptr);
    ASSERT_NE(betaTensor, nullptr);
    ASSERT_NE(outRefTensor, nullptr);
    ASSERT_NE(stateRefTensor, nullptr);

    rt::Coords qShape = qTensor->getShape();
    int32_t numVHeads = static_cast<int32_t>(qShape[2]);
    int32_t headKDim = static_cast<int32_t>(qShape[3]);
    int32_t headVDim = static_cast<int32_t>(vTensor->getShape()[3]);

    rt::Tensor outDevice(outRefTensor->getShape(), rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor finalStateDevice(stateRefTensor->getShape(), rt::DeviceType::kGPU, DataType::kHALF);
    CUDA_CHECK(cudaMemsetAsync(outDevice.rawPointer(), 0, outDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaMemsetAsync(finalStateDevice.rawPointer(), 0, finalStateDevice.getMemoryCapacity(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    plugins::GatedDeltaRulePlugin plugin("test_delta", numVHeads, headVDim, headKDim, 1);

    PluginTensorDesc inputDesc[6];
    inputDesc[0] = makePluginTensorDesc(qTensor->getTRTDims(), DataType::kHALF);
    inputDesc[1] = makePluginTensorDesc(kTensor->getTRTDims(), DataType::kHALF);
    inputDesc[2] = makePluginTensorDesc(vTensor->getTRTDims(), DataType::kHALF);
    inputDesc[3] = makePluginTensorDesc(gTensor->getTRTDims(), DataType::kHALF);
    inputDesc[4] = makePluginTensorDesc(betaTensor->getTRTDims(), DataType::kHALF);
    Dims emptyDims{};
    emptyDims.nbDims = 0;
    inputDesc[5] = makePluginTensorDesc(emptyDims, DataType::kHALF);

    PluginTensorDesc outputDesc[2];
    outputDesc[0] = makePluginTensorDesc(outDevice.getTRTDims(), DataType::kHALF);
    outputDesc[1] = makePluginTensorDesc(finalStateDevice.getTRTDims(), DataType::kHALF);

    // Allocate workspace for QKNorm (required when use_qk_l2norm=1)
    DynamicPluginTensorDesc dynInDesc[6];
    for (int i = 0; i < 6; ++i)
    {
        dynInDesc[i].desc = inputDesc[i];
    }
    size_t workspaceSize = plugin.getWorkspaceSize(dynInDesc, 6, nullptr, 2);
    void* workspace = nullptr;
    if (workspaceSize > 0)
    {
        CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));
    }

    void const* inputs[6]
        = {qTensor->rawPointer(), kTensor->rawPointer(), vTensor->rawPointer(), gTensor->rawPointer(),
            betaTensor->rawPointer(), nullptr};
    void* outputs[2] = {outDevice.rawPointer(), finalStateDevice.rawPointer()};

    EXPECT_EQ(plugin.enqueue(inputDesc, outputDesc, inputs, outputs, workspace, stream), 0);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto [rtol, atol] = getTolerance<half>();

    std::vector<half> outHost;
    copyTensorToHost(outDevice, outHost, stream);
    std::vector<half> outRefHost;
    copyTensorToHost(*outRefTensor, outRefHost, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

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

    std::vector<half> stateHost;
    copyTensorToHost(finalStateDevice, stateHost, stream);
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
