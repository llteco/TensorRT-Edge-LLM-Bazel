workspace(name = "tensorrt_edge_llm")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

rules_cc_version = "0.0.14"  # this is the last working version for legacy workspace

http_archive(
    name = "rules_cc",
    integrity = "sha256-kG6JKGrMZ8IIGcPIizKD3g1YaK/aM2NdcKyuDel3e7c=",
    strip_prefix = "rules_cc-%s" % rules_cc_version,
    url = "https://github.com/bazelbuild/rules_cc/archive/refs/tags/%s.tar.gz" % rules_cc_version,
)

# https://github.com/bazel_contrib/rules_cuda
rules_cuda_commit = "6d612b8de45576f3999673b44ad2bacb713df0d1"

http_archive(
    name = "rules_cuda",
    integrity = "sha256-ZDur9PHE/0rnd3bkJaOPID2G2zRzFv0heV378p/EiAk=",
    strip_prefix = "rules_cuda-%s" % rules_cuda_commit,
    url = "https://github.com/bazel-contrib/rules_cuda/archive/%s.tar.gz" % rules_cuda_commit,
)

load("@rules_cuda//cuda:repositories.bzl", "rules_cuda_toolchains")

rules_cuda_toolchains(register_toolchains = True)

# TensorRT
# tensorrt_version = "10.15.1"
git_repository(
    name = "rules_tensorrt",
    commit = "e12725a6c1382fb6320cc49a01d20d00e31dda8e",
    remote = "https://github.com/loseall/rules_tensorrt.git",
)

load("@rules_tensorrt//:repo.bzl", "config_tensorrt")

config_tensorrt(
    name = "tensorrt",
    required = False,
)

local_repository(
    name = "nlohmann_json",
    path = "3rdParty/nlohmannJson",
)

local_repository(
    name = "com_google_googletest",
    path = "3rdParty/googletest",
)

# https://github.com/bazelbuild/bazel-skylib/releases
skylib_version = "1.9.0"

http_archive(
    name = "bazel_skylib",
    sha256 = "3b5b49006181f5f8ff626ef8ddceaa95e9bb8ad294f7b5d7b11ea9f7ddaf8c59",
    urls = [
        "https://github.com/bazelbuild/bazel-skylib/releases/download/{0}/bazel-skylib-{0}.tar.gz".format(skylib_version),
    ],
)

# Hedron's Compile Commands Extractor for Bazel (fork'd version)
# https://github.com/loseall/bazel-compile-commands-extractor
hedron_commit = "2c33c1b2a6a06ef9d4a371ba90b7325341a15f39"

http_archive(
    name = "hedron_compile_commands",
    integrity = "sha256-u0AvEvL4uf4pUBZ+LQTekwn+hflpJi/3ZKTOA1K8U6U=",
    strip_prefix = "bazel-compile-commands-extractor-%s" % hedron_commit,
    url = "https://github.com/loseall/bazel-compile-commands-extractor/archive/%s.zip" % hedron_commit,
)
