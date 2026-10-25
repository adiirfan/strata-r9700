#!/bin/bash
# tools/r9700/build-glm.sh <ROCm 10 SDK dir> [build dir] [llama.cpp checkout]: strata-glm, the GLM-5.3-Flash engine
# (src/glm, from Project Maya), for the Radeon AI PRO R9700 / RX 9070 (gfx1201) against AMD TheRock ROCm 10, copied to
# engine/strata-glm.  src/glm is its own CMake project (its libraries share names with Strata's own), built apart.
# The SDK dir is the _rocm_sdk_devel folder of a `pip install "rocm[libraries,devel,device-gfx1201]"` from
# https://nightly.repo.amd.com/rocm/whl-next/ (for example 10.2.0a20261009).  Without a llama.cpp checkout at the
# commit in src/glm/third_party/ggml/VERSION.txt the build fetches it.
set -e
SDK=${1:?the ROCm 10 SDK dir (.../_rocm_sdk_devel)}; B=${2:-build-glm}; LLAMA=${3:-}
cd "$(dirname "$0")/../.."
export ROCM_PATH=$SDK HIP_PATH=$SDK HIP_PLATFORM=amd
export PATH=$SDK/bin:$SDK/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=$SDK/lib:$SDK/lib/rocm_sysdeps/lib:$SDK/lib/llvm/lib
cmake -S src/glm -B "$B" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PREFILL_MMQ=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DGPU_TARGETS=gfx1201 -DHIP_PLATFORM=amd -DCMAKE_HIP_PLATFORM=amd \
  -DCMAKE_HIP_COMPILER=$SDK/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$SDK \
  "-DCMAKE_PREFIX_PATH=$SDK;$SDK/lib/rocm_sysdeps;$SDK/lib/llvm" \
  "-DCMAKE_HIP_FLAGS=-isystem $SDK/include --rocm-path=$SDK --rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode" \
  "-DCMAKE_CXX_FLAGS=-isystem $SDK/include" ${LLAMA:+"-DSTRATA_GGML_DIR=$LLAMA"}
cmake --build "$B" --target strata-glm -j "$(nproc)"
mkdir -p engine && cp "$B/strata-glm" engine/strata-glm
echo "built engine/strata-glm"
