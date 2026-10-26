#!/bin/bash
# tools/r9700/build-vision-glm.sh <ROCm 10 SDK dir> <llama.cpp checkout> [build dir]: strata-vision-hip with GLM-5.3-Flash's
# vision tower, for the R9700 (gfx1201).  The checkout must be at the commit in src/glm/third_party/ggml/VERSION.txt; it
# is patched in place (tools/vision/glm5next_patch.py adds the glm5next projector to its mtmd; a second run changes
# nothing), so give it a copy of its own.  Writes <build dir>/bin/strata-vision-hip (default build-vision-glm).
set -e
SDK=${1:?the ROCm 10 SDK dir (.../_rocm_sdk_devel)}; LLAMA=${2:?a llama.cpp checkout}; B=${3:-build-vision-glm}
cd "$(dirname "$0")/../.."
python3 tools/vision/glm5next_patch.py "$LLAMA"
export ROCM_PATH=$SDK HIP_PATH=$SDK HIP_PLATFORM=amd PATH=$SDK/bin:$SDK/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=$SDK/lib:$SDK/lib/rocm_sysdeps/lib:$SDK/lib/llvm/lib
cmake -S tools/vision -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="$(realpath "$LLAMA")" \
  -DSTRATA_PORTABLE=OFF -DSTRATA_VISION_HIP=ON -DSTRATA_VISION_HIP_ARCHS=gfx1201 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DGPU_TARGETS=gfx1201 -DAMDGPU_TARGETS=gfx1201 -DCMAKE_HIP_PLATFORM=amd \
  -DCMAKE_HIP_COMPILER=$SDK/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$SDK \
  "-DCMAKE_PREFIX_PATH=$SDK;$SDK/lib/rocm_sysdeps;$SDK/lib/llvm" \
  "-DCMAKE_HIP_FLAGS=-isystem $SDK/include --rocm-path=$SDK --rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode"
cmake --build "$B" -j "$(nproc)"
echo "built $B/bin/strata-vision-hip"
