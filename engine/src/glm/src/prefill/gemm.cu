// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
// The HIP compatibility shim maps CUDA shuffle spellings to Strata helpers.
// hipBLASLt's public headers declare native HIP shuffle functions, so keep
// those declarations from being macro-expanded in this translation unit.
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#endif

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
#include "hipblaslt_tuning.hpp"
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>
#include <set>
#include <tuple>
#endif

namespace strata::prefill {
namespace {

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
struct HipLtCallKey {
    strata::prefill::hipblaslt::InputType type;
    int t;
    int n;
    int k;
    int ldy;
    uint32_t beta_bits;

    bool operator<(const HipLtCallKey& other) const {
        return std::tie(type, n, k, ldy, t, beta_bits) <
               std::tie(other.type, other.n, other.k, other.ldy, other.t, other.beta_bits);
    }
};

struct HipLtCachedAlgo {
    bool supported = false;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
};

struct HipLtState {
    hipblasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    strata::prefill::hipblaslt::TuningTable table;
    std::map<HipLtCallKey, HipLtCachedAlgo> cache;
    uint64_t lt_launches = 0;
    uint64_t fallbacks = 0;
    std::set<std::tuple<strata::prefill::hipblaslt::InputType, int, int, int, int>> fallback_shapes;

    ~HipLtState() {
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: hipBLASLt summary launches=%llu fallbacks=%llu unique_fallback_shapes=%zu\n",
                         (unsigned long long) lt_launches, (unsigned long long) fallbacks, fallback_shapes.size());
            for (const auto& shape : fallback_shapes) {
                const auto type = std::get<0>(shape);
                std::fprintf(stderr, "prefill gemm: fallback shape dtype=%s T=%d N=%d K=%d ldy=%d\n",
                             type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16",
                             std::get<1>(shape), std::get<2>(shape), std::get<3>(shape), std::get<4>(shape));
            }
        }
        if (handle) hipblasLtDestroy(handle);
    }
};

struct HipLtDescriptors {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr;
    hipblasLtMatrixLayout_t b = nullptr;
    hipblasLtMatrixLayout_t c = nullptr;

    ~HipLtDescriptors() {
        if (op) hipblasLtMatmulDescDestroy(op);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
    }

    bool init(hipDataType type, int t, int n, int k, int ldy) {
        const hipblasOperation_t trans_a = HIPBLAS_OP_T;
        const hipblasOperation_t trans_b = HIPBLAS_OP_N;
        if (hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&a, type, k, n, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&b, type, k, t, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, n, t, ldy) != HIPBLAS_STATUS_SUCCESS) {
            return false;
        }
        return true;
    }
};

std::unique_ptr<HipLtState> create_hipblaslt_state(void* workspace, size_t workspace_bytes) {
    const char* path = std::getenv("STRATA_HIPBLASLT_TUNING");
    if (!path || !*path) return nullptr;

    auto state = std::make_unique<HipLtState>();
    state->workspace = workspace;
    state->workspace_bytes = workspace_bytes;
    if (hipblasLtCreate(&state->handle) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt handle creation failed; using hipBLASEx\n");
        return nullptr;
    }

    int version = 0;
    if (hipblasLtGetVersion(state->handle, &version) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt version query failed; using hipBLASEx\n");
        return nullptr;
    }
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess) {
        std::fprintf(stderr, "prefill gemm: HIP device query failed; using hipBLASEx\n");
        return nullptr;
    }
    std::string arch(properties.gcnArchName);
    const auto suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);

    // A ':'-separated list (one table per architecture, for a layer split across different cards): each device
    // takes the first table whose header names its architecture and hipBLASLt version.
    std::string error, list(path);
    bool loaded = false;
    for (size_t at = 0; at <= list.size() && !loaded;) {
        const size_t end = std::min(list.find(':', at), list.size());
        const std::string one = list.substr(at, end - at);
        at = end + 1;
        if (one.empty()) continue;
        strata::prefill::hipblaslt::TuningTable table;
        std::string e;
        if (table.load(one, arch, version, e)) {
            state->table = std::move(table);
            loaded = true;
        } else {
            error += (error.empty() ? "" : "; ") + one + ": " + e;
        }
    }
    if (!loaded) {
        std::fprintf(stderr, "prefill gemm: %s; using hipBLASEx\n", error.c_str());
        return nullptr;
    }
    std::fprintf(stderr, "prefill gemm: hipBLASLt tuning enabled (%zu rows, %s, version %d)\n",
                 state->table.rows().size(), arch.c_str(), version);
    return state;
}

HipLtCachedAlgo resolve_hipblaslt_algo(HipLtState& state, strata::prefill::hipblaslt::InputType type, int t,
                                       int n, int k, int ldy, float beta) {
    uint32_t beta_bits = 0;
    static_assert(sizeof(beta_bits) == sizeof(beta));
    std::memcpy(&beta_bits, &beta, sizeof(beta));
    const HipLtCallKey key{type, t, n, k, ldy, beta_bits};
    const auto cached = state.cache.find(key);
    if (cached != state.cache.end()) return cached->second;

    HipLtCachedAlgo resolved;
    const bool verbose = std::getenv("STRATA_HIPBLASLT_VERBOSE") != nullptr;
    const auto* row = state.table.closest(type, n, k, ldy, t);
    if (!row) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; no calibration for dtype=%s T=%d N=%d K=%d ldy=%d\n",
                         type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, t, n, k, ldy)) {
        return state.cache.emplace(key, resolved).first->second;
    }

    std::vector<int> solution_ids{row->solution_id};
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
    if (hipblaslt_ext::getAlgosFromIndex(state.handle, solution_ids, candidates) != HIPBLAS_STATUS_SUCCESS ||
        candidates.empty() || candidates.front().state != HIPBLAS_STATUS_SUCCESS ||
        hipblaslt_ext::getIndexFromAlgo(candidates.front().algo) != row->solution_id) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d unavailable for T=%d N=%d K=%d ldy=%d\n",
                         row->solution_id, t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    const float alpha = 1.0f;
    size_t required_workspace = 0;
    auto algo = candidates.front().algo;
    if (hipblaslt_ext::matmulIsAlgoSupported(state.handle, desc.op, &alpha, desc.a, desc.b, &beta, desc.c, desc.c,
                                             algo, required_workspace) != HIPBLAS_STATUS_SUCCESS) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d rejects actual T=%d N=%d K=%d ldy=%d beta=%.9g\n",
                         row->solution_id, t, n, k, ldy, beta);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    resolved.supported = true;
    resolved.algo = algo;
    resolved.workspace_bytes = required_workspace;
    if (verbose) {
        std::fprintf(stderr,
                     "prefill gemm: Lt solution=%d dtype=%s T=%d N=%d K=%d ldy=%d beta=%.9g workspace=%zu\n",
                     row->solution_id, type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n,
                     k, ldy, beta, required_workspace);
    }
    return state.cache.emplace(key, resolved).first->second;
}

bool try_hipblaslt(void* opaque_state, strata::prefill::hipblaslt::InputType type, const uint16_t* x,
                   const uint16_t* w, float* y, int64_t t, int64_t n, int64_t k, int64_t ldy, float beta,
                   void* stream) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX ||
        ldy < n) {
        return false;
    }
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        return false;
    }
    if (resolved.workspace_bytes > state->workspace_bytes) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution needs %zu workspace bytes, have %zu\n",
                         resolved.workspace_bytes, state->workspace_bytes);
        }
        return false;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, (int) t, (int) n, (int) k, (int) ldy)) return false;
    const float alpha = 1.0f;
    const hipblasStatus_t status = hipblasLtMatmul(state->handle, desc.op, &alpha, w, desc.a, x, desc.b, &beta, y,
                                                   desc.c, y, desc.c, &resolved.algo, state->workspace,
                                                   state->workspace_bytes, (hipStream_t) stream);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        ++state->lt_launches;
        return true;
    }

    std::fprintf(stderr, "prefill gemm: hipBLASLt launch failed with status %d\n", (int) status);
    if (beta != 0.0f) {
        std::fprintf(stderr, "prefill gemm: refusing a fallback after hipBLASLt failed with nonzero beta\n");
        std::exit(1);
    }
    auto* mutable_state = static_cast<HipLtState*>(opaque_state);
    uint32_t beta_bits = 0;
    std::memcpy(&beta_bits, &beta, sizeof(beta_bits));
    auto cached = mutable_state->cache.find(HipLtCallKey{type, (int) t, (int) n, (int) k, (int) ldy, beta_bits});
    if (cached != mutable_state->cache.end()) cached->second.supported = false;
    ++mutable_state->fallbacks;
    mutable_state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
    return false;
}
#endif

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
// ---- f16_fp8: an FP16 product as FP8 E4M3 on both sides (as Strata-r9700's Gemm::native_fp8 for Qwen)
}  // namespace
}  // namespace strata::prefill
#include <hip/hip_fp8.h>
namespace strata::prefill {
namespace {
constexpr float kE4M3Max = 448.0f;
__device__ __forceinline__ uint8_t to_e4m3(float v) {
    return (uint8_t) __hip_cvt_float_to_fp8(v, __HIP_SATFINITE, __HIP_E4M3);
}
__device__ __forceinline__ float block_max256(float m) {
    __shared__ float red[256];
    red[threadIdx.x] = m;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    return red[0];
}
// one block per weight row: scale = max |w| / 448, the row in E4M3 at that scale
__global__ void __launch_bounds__(256) fp8_rows_kernel(const uint16_t* __restrict__ w16, int64_t K, uint8_t* __restrict__ w8,
                                                       float* __restrict__ scale) {
    const int64_t r = blockIdx.x;
    const __half* row = reinterpret_cast<const __half*>(w16) + r * K;
    float m = 0.0f;
    for (int64_t i = threadIdx.x; i < K; i += 256) m = fmaxf(m, fabsf(__half2float(row[i])));
    m = block_max256(m);
    const float sc = m > 0.0f ? m / kE4M3Max : 1.0f, inv = 1.0f / sc;
    for (int64_t i = threadIdx.x; i < K; i += 256) w8[r * K + i] = to_e4m3(__half2float(row[i]) * inv);
    if (threadIdx.x == 0) scale[r] = sc;
}
// the largest |x| of n FP16 values into *amax (float bits: non-negative floats order as unsigned ints)
__global__ void __launch_bounds__(256) fp8_amax_kernel(const uint16_t* __restrict__ x16, int64_t n, uint32_t* __restrict__ amax) {
    const __half* x = reinterpret_cast<const __half*>(x16);
    float m = 0.0f;
    for (int64_t i = (int64_t) blockIdx.x * 256 + threadIdx.x; i < n; i += (int64_t) gridDim.x * 256)
        m = fmaxf(m, fabsf(__half2float(x[i])));
    m = block_max256(m);
    if (threadIdx.x == 0 && m > 0.0f) atomicMax(amax, __float_as_uint(m));
}
__global__ void __launch_bounds__(256) fp8_act_kernel(const uint16_t* __restrict__ x16, int64_t n, const uint32_t* __restrict__ amax,
                                                      uint8_t* __restrict__ x8) {
    const __half* x = reinterpret_cast<const __half*>(x16);
    const float m = __uint_as_float(*amax), inv = m > 0.0f ? kE4M3Max / m : 1.0f;
    for (int64_t i = (int64_t) blockIdx.x * 256 + threadIdx.x; i < n; i += (int64_t) gridDim.x * 256)
        x8[i] = to_e4m3(__half2float(x[i]) * inv);
}
__global__ void fp8_alpha_kernel(const float* __restrict__ wscale, int64_t N, const uint32_t* __restrict__ amax,
                                 float* __restrict__ alpha) {
    const int64_t i = (int64_t) blockIdx.x * 256 + threadIdx.x;
    const float m = __uint_as_float(*amax), xs = m > 0.0f ? m / kE4M3Max : 1.0f;
    if (i < N) alpha[i] = wscale[i] * xs;
}
struct Fp8Plan {
    bool ok = false, tried = false;
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    hipblasLtMatmulAlgo_t algo{};
};
// the descriptors and hipBLASLt's first heuristic answer per shape (asked at the row count rounded up to 1024, as for
// Qwen: a solution per exact row count reloaded code on most requests), Y[t, n] fp32 = X8[t, k] . W8[n, k]^T, alpha per
// row of W, beta on the host
struct Fp8Cache {
    std::map<std::tuple<int, int, int, int, uint32_t>, Fp8Plan> plans;
    std::map<std::tuple<int, int, int, int, uint32_t>, std::pair<bool, hipblasLtMatmulAlgo_t>> algos;
    ~Fp8Cache() {
        for (auto& [k, p] : plans) {
            if (p.op) hipblasLtMatmulDescDestroy(p.op);
            if (p.a) hipblasLtMatrixLayoutDestroy(p.a);
            if (p.b) hipblasLtMatrixLayoutDestroy(p.b);
            if (p.c) hipblasLtMatrixLayoutDestroy(p.c);
        }
    }
};
const Fp8Plan& fp8_plan(Fp8Cache& cache, HipLtState& st, int t, int n, int k, int ldy, float beta) {
    uint32_t bb = 0;
    std::memcpy(&bb, &beta, sizeof bb);
    Fp8Plan& p = cache.plans[{t, n, k, ldy, bb}];
    if (p.tried) return p;
    p.tried = true;
    const hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
    const hipblasLtPointerMode_t pm = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
    if (hipblasLtMatmulDescCreate(&p.op, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatmulDescSetAttribute(p.op, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatmulDescSetAttribute(p.op, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatmulDescSetAttribute(p.op, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof pm) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&p.a, HIP_R_8F_E4M3, k, n, k) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&p.b, HIP_R_8F_E4M3, k, t, k) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&p.c, HIP_R_32F, n, t, ldy) != HIPBLAS_STATUS_SUCCESS)
        return p;
    const int bucket = (t + 1023) / 1024 * 1024;
    auto& fa = cache.algos[{bucket, n, k, ldy, bb}];
    if (!fa.first) {
        hipblasLtMatrixLayout_t bb_l = nullptr, bc_l = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;
        hipblasLtMatmulHeuristicResult_t res[4];
        int got = 0;
        const uint64_t ws = st.workspace_bytes;
        if (hipblasLtMatrixLayoutCreate(&bb_l, HIP_R_8F_E4M3, k, bucket, k) == HIPBLAS_STATUS_SUCCESS &&
            hipblasLtMatrixLayoutCreate(&bc_l, HIP_R_32F, n, bucket, std::max(ldy, n)) == HIPBLAS_STATUS_SUCCESS &&
            hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS &&
            hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws) == HIPBLAS_STATUS_SUCCESS &&
            hipblasLtMatmulAlgoGetHeuristic(st.handle, p.op, p.a, bb_l, bc_l, bc_l, pref, 4, res, &got) == HIPBLAS_STATUS_SUCCESS && got > 0)
            fa = {true, res[0].algo};
        if (pref) hipblasLtMatmulPreferenceDestroy(pref);
        if (bb_l) hipblasLtMatrixLayoutDestroy(bb_l);
        if (bc_l) hipblasLtMatrixLayoutDestroy(bc_l);
    }
    if (fa.first) {
        size_t need = 0;
        const float one = 1.0f;
        hipblasLtMatmulAlgo_t algo = fa.second;
        if (hipblaslt_ext::matmulIsAlgoSupported(st.handle, p.op, &one, p.a, p.b, &beta, p.c, p.c, algo, need) ==
                HIPBLAS_STATUS_SUCCESS && need <= st.workspace_bytes) {
            p.algo = algo;
            p.ok = true;
        }
    }
    if (!p.ok)
        std::fprintf(stderr, "prefill gemm: no FP8 hipBLASLt solution for t %d n %d k %d (the FP16 path takes it)\n", t, n, k);
    return p;
}
#endif

}  // namespace

Gemm::~Gemm() {
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    delete static_cast<Fp8Cache*>(fp8_cache_);
    if (fp8_w_) cudaFree(fp8_w_);
    if (fp8_vec_) cudaFree(fp8_vec_);
    if (fp8_amax_) cudaFree(fp8_amax_);
    delete static_cast<HipLtState*>(hipblaslt_state_);
#endif
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { err = "prefill gemm: cublasCreate failed"; return false; }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    cublasSetStream(h, (cudaStream_t) stream);
    workspace_ = workspace;
    cublasSetWorkspace(h, workspace_, ws_bytes);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws_bytes).release();
#endif
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    cublasSetWorkspace((cublasHandle_t) handle_, workspace_, ws_bytes);
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (hipblaslt_state_) {
        auto* state = static_cast<HipLtState*>(hipblaslt_state_);
        state->workspace = workspace_;
        state->workspace_bytes = ws_bytes;
    }
#endif
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { err = "prefill gemm: cublasCreate failed"; return false; }
    handle_ = h;
    stream_ = stream;
    cublasSetStream(h, (cudaStream_t) stream);
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (cudaMalloc(&workspace_, ws) != cudaSuccess) { err = "prefill gemm: workspace"; return false; }
    cublasSetWorkspace(h, workspace_, ws);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws).release();
#endif
    if (scratch_elems > 0 && cudaMalloc((void**) &scratch_, (size_t) scratch_elems * 2) != cudaSuccess) {
        err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB";
        return false;
    }
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, X, W, Y, T, N, K, ldy,
                      beta, stream_)) {
        return;
    }
#endif
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx");
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::f16, X, W, Y, T, N, K, ldy,
                      beta, stream_)) {
        return;
    }
#endif
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16");
}

// W (FP16, in the dequantization scratch) . X (FP16) as FP8 E4M3 on both sides: the weight scaled per row to E4M3's range
// into a buffer of its own, then the activations per slice of rows (by their largest |x|) through the scratch the
// weight left; hipBLASLt's alpha vector (row scale x activation scale) puts the product back on its FP32 scale.  False
// (nothing launched) when it cannot run: no hipBLASLt (STRATA_HIPBLASLT_TUNING unset), a buffer it could not get, or no
// FP8 solution for a slice's shape - the caller takes the FP16 product then.
bool Gemm::f16_fp8(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta) {
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    auto* st = static_cast<HipLtState*>(hipblaslt_state_);
    if (fp8_failed_ || st == nullptr || W != scratch_ || N * K > scratch_elems_ || K % 16 != 0 || T <= 0 || N > INT_MAX ||
        K > INT_MAX)
        return false;
    if (ldy <= 0) ldy = N;
    const cudaStream_t s = (cudaStream_t) stream_;
    if (fp8_w_bytes_ < N * K) {
        if (fp8_w_) cudaFree(fp8_w_);
        fp8_w_ = nullptr;
        fp8_w_bytes_ = 0;
        if (cudaMalloc((void**) &fp8_w_, (size_t) (N * K)) != cudaSuccess) { (void) cudaGetLastError(); fp8_failed_ = true; return false; }
        fp8_w_bytes_ = N * K;
    }
    if (fp8_vec_rows_ < N) {
        if (fp8_vec_) cudaFree(fp8_vec_);
        fp8_vec_ = nullptr;
        fp8_vec_rows_ = 0;
        if (cudaMalloc((void**) &fp8_vec_, (size_t) (2 * N) * sizeof(float)) != cudaSuccess) { (void) cudaGetLastError(); fp8_failed_ = true; return false; }
        fp8_vec_rows_ = N;
    }
    if (fp8_amax_ == nullptr && cudaMalloc((void**) &fp8_amax_, 256) != cudaSuccess) { (void) cudaGetLastError(); fp8_failed_ = true; return false; }
    if (fp8_cache_ == nullptr) {
        fp8_cache_ = new Fp8Cache();
        std::fprintf(stderr, "prefill gemm: FP8 dense products on\n");
    }
    auto& cache = *static_cast<Fp8Cache*>(fp8_cache_);
    const int64_t R = std::min<int64_t>(T, (scratch_elems_ * 2) / K);   // activation rows of K bytes the scratch holds
    for (int64_t r0 = 0; r0 < T; r0 += R) {   // every slice's shape must have a plan before anything is launched
        const int64_t rows = std::min(R, T - r0);
        if (rows > INT_MAX || !fp8_plan(cache, *st, (int) rows, (int) N, (int) K, (int) ldy, beta).ok) return false;
    }
    float* wscale = fp8_vec_;
    float* alpha = fp8_vec_ + N;
    fp8_rows_kernel<<<(unsigned) N, 256, 0, s>>>(W, K, fp8_w_, wscale);
    uint8_t* x8 = reinterpret_cast<uint8_t*>(scratch_);
    for (int64_t r0 = 0; r0 < T; r0 += R) {
        const int64_t rows = std::min(R, T - r0), n = rows * K;
        const unsigned g = (unsigned) std::min<int64_t>((n + 255) / 256, 4096);
        cudaMemsetAsync(fp8_amax_, 0, 4, s);
        fp8_amax_kernel<<<g, 256, 0, s>>>(X + r0 * K, n, fp8_amax_);
        fp8_act_kernel<<<g, 256, 0, s>>>(X + r0 * K, n, fp8_amax_, x8);
        fp8_alpha_kernel<<<(unsigned) ((N + 255) / 256), 256, 0, s>>>(wscale, N, fp8_amax_, alpha);
        const Fp8Plan& p = fp8_plan(cache, *st, (int) rows, (int) N, (int) K, (int) ldy, beta);
        const hipblasStatus_t rs = hipblasLtMatmul(st->handle, p.op, alpha, fp8_w_, p.a, x8, p.b, &beta, Y + r0 * ldy, p.c,
                                                   Y + r0 * ldy, p.c, &p.algo, st->workspace, st->workspace_bytes, s);
        if (rs != HIPBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "prefill gemm: the FP8 product failed (status %d, t %lld n %lld k %lld)\n", (int) rs,
                         (long long) rows, (long long) N, (long long) K);
            std::exit(1);   // part of Y may be written: no silent fallback
        }
    }
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::fprintf(stderr, "prefill gemm: f16_fp8: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
#else
    (void) X; (void) W; (void) Y; (void) T; (void) N; (void) K; (void) ldy; (void) beta;
    return false;
#endif
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill
