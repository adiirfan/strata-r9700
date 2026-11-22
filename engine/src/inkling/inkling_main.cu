// src/inkling/inkling_main.cu - strata-inkling: Thinking Machines' Inkling 2 Mini (model_type inkling2_loop, and
// fine-tunes such as OpenUncensored's Inkling-2-Mini-Uncensored) from its own safetensors, the whole model on one 32 GB
// card.
//
//   strata-inkling --model DIR --ids 1,2,3|@FILE [--new N] [--top FILE] [--logits FILE] [--max-ctx N] [--serve]
//
// The model (transformers' modeling_inkling2_loop.py, pull request #51050, is the spec): 24 layers of hidden 2304, the
// middle twelve (6 .. 17) run three times a token with the same weights (48 layer visits), the second and third passes
// mixing in "messages" from the previous pass. Every layer: attention (24 query heads, 8 K/V heads of 128, no RoPE: a
// learned relative-position bias over the last 1024 positions, a per-token sink logit, q and k scaled to at most
// sqrt(512), v RMS-normed, K and V through a short causal convolution, the output gated by a sigmoid of the input), then
// 128 experts (8 a token, sigmoid router with a bias for the choice, weights renormalised times 8, gate and up softcapped)
// with no shared expert. Layers 5, 11, 17 and 23 see everything (their K / V convolution taps also depend on the token,
// their queries and relative states grow with log(position) past 8192); the others slide over 1024 positions. Each
// sub-layer's output goes through its own 4-tap convolution. Instead of a plain residual sum, the inputs of attention and
// of the experts are a learned softmax mix of the normed outputs of earlier 6-layer blocks ("attention residuals").
// The output head is untied, with a tanh softcap of 30.
//
// Weights: the experts stay as stored (NVFP4 for layers 0 .. 19, MXFP8 for 20 .. 23); the bf16 matrices become fp16 (the
// same numbers: the conversion is checked at load). Every sum is fp32; K / V are fp16.
#ifdef INKLING_BLT
// (the HIP shim maps CUDA shuffle spellings to Strata helpers; hipBLASLt's headers declare the native ones)
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <tuple>
#endif
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <type_traits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "strata-inkling: %s failed: %s (%s:%d)\n", #x, hipGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

constexpr int kH = 2304, kLayers = 24, kQH = 24, kKV = 8, kHD = 128, kQD = kQH * kHD, kKVD = kKV * kHD, kDRel = 16;
constexpr int kExt = 1024, kWin = 1024, kE = 128, kTopK = 8, kF = 1152, kVisits = 48, kLS = 6, kLE = 18, kPasses = 3;
constexpr int kGR = 64, kBlk = 6, kVocab = 200058, kVocabP = 212992;
constexpr float kEps = 1e-6f, kQKClip = 22.627416997969522f, kRelClip = 8.f, kGCap = 4.f, kUCap = 25.f, kRouteScale = 8.f;
constexpr float kOutCap = 30.f, kLogAlpha = 0.1f, kLogFloor = 8192.f;
// the attention input product's rows: from the query input q (3072), the output gate (3072), the relative states
// (384), the sink logits (24, padded to 32); from the key input k (1024) and its convolution taps (32, global layers);
// from the value input v (1024) and its taps (32)
constexpr int kRQ = 0, kRG = 3072, kRR = 6144, kRS = 6528, kRK = 6560, kRKP = 7584, kRV = 7616, kRVP = 8640, kNP = 8672;
constexpr int kSwaRing = 8192;   // the sliding layers' K/V slots: the window plus the largest prompt chunk
typedef __half kc_t;
inline bool is_global(int i) { return i % 6 == 5; }

// ---------------------------------------------------------------------------------------------------- helpers
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    return v;
}
// the sum of v over the block (blockDim.x a multiple of 32, at most 1024); every thread gets it
__device__ inline float block_sum(float v) {
    __shared__ float red[32];
    __shared__ float tot;
    v = warp_sum(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
        t = warp_sum(t);
        if (threadIdx.x == 0) tot = t;
    }
    __syncthreads();
    return tot;
}
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float softcapf(float x, float c) { return tanhf(x / c) * c; }
// OCP e4m3 (fn) to float
__device__ __forceinline__ float e4m3(uint32_t b) {
    const uint32_t e = (b >> 3) & 15, m = b & 7;
    float v = e ? ldexpf(1.f + m * 0.125f, int(e) - 7) : ldexpf(float(m), -9);
    if (e == 15 && m == 7) v = NAN;
    return (b & 0x80) ? -v : v;
}
__constant__ float c_e2m1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// ---------------------------------------------------------------------------------------------------- weights
struct W8 { const int8_t* q; const __half* d; };   // int8 rows, an fp16 scale a 32 values (k_mv8)
struct W4 { const uint8_t* q = nullptr; const __half* d = nullptr; };   // two values a byte, an fp16 scale a 32 values
struct Experts {   // one layer's 128 experts as stored
    bool nv = false;               // NVFP4 (two E2M1 a byte, e4m3 scale a 16, a factor an expert); else MXFP8 (e4m3, 2^(s-127) a 32)
    const uint8_t *w13 = nullptr, *s13 = nullptr, *w2 = nullptr, *s2 = nullptr;
    const float *f13 = nullptr, *f2 = nullptr;   // NVFP4: amax / denominator per expert
    size_t w13_row = 0, s13_row = 0, w2_row = 0, s2_row = 0;   // bytes a row
};

struct Layer {
    bool global = false, loop = false;
    const __half* wp = nullptr;      // kNP x kH: q, gate, rel, sink | k, k taps | v, v taps
    const __half* wo = nullptr;      // kH x kQD
    W8 wp8{}, wo8{};                 // their 8-bit copies (the writing path)
    W4 wp4{}, wo4{};                 // 4-bit copies (INKLING_ATTN4 / INKLING_ATTN4_O, off by default)
    const float *qscale, *rscale, *proj;            // 24 x 128, 24 x 16, 16 x 1024
    const float *kconv, *vconv, *aconv, *mconv;     // [C][4] taps by backward distance
    const float *norm_q, *norm_k, *norm_v, *norm_m;  // (sliding: one attn_norm for all three)
    const float *router, *rbias;                    // 128 x 2304 fp32, 128
    const float *attn_q = nullptr, *mlp_q = nullptr;  // 1 or 3 queries; 1
    int n_attn_q = 0;
    const __half* g_down[4] = {};   // the message gates (loop layers): q, k, v, mlp; 64 x 4608 each
    const float *g_forget[4] = {}, *g_gain[4] = {};  // 2304 x 64 each
    const __half* g_fg16[4] = {};                    // forget then gain rows as fp16 (4608 x 64), the prompt path
    Experts ex;
    const int8_t* shift = nullptr;   // MXFP8: per stored row (gate / up rows, then down rows) the power of two taken out
    const __half* router16 = nullptr; // the router as fp16 (the bf16 values exactly) for the prompt path
    const __half *pjh = nullptr, *pjl = nullptr;   // the relative-bias table transposed, hi / lo (k_pf_attn)
};

// ---------------------------------------------------------------------------------------------------- small kernels
// out[t] = rmsnorm(x[t]) * w (w null: unweighted); one block of 256 a row
__global__ void __launch_bounds__(256) k_rms(const float* x, int xs, const float* w, float* out, int os, uint16_t* o16 = nullptr,
                                             uint16_t* lo16 = nullptr) {
    const float* xr = x + size_t(blockIdx.x) * xs;
    float ss = 0;
    for (int i = threadIdx.x; i < kH; i += 256) ss += xr[i] * xr[i];
    const float r = rsqrtf(block_sum(ss) / kH + kEps);
    float* o = out + size_t(blockIdx.x) * os;
    for (int i = threadIdx.x; i < kH; i += 256) {
        const float v = xr[i] * r * (w ? w[i] : 1.f);
        o[i] = v;
        if (o16) {   // (fp16 copy for the products; lo16: the remainder, x = hi + lo)
            const __half h = __float2half(fminf(fmaxf(v, -65504.f), 65504.f));
            o16[size_t(blockIdx.x) * kH + i] = __half_as_ushort(h);
            if (lo16) lo16[size_t(blockIdx.x) * kH + i] = __half_as_ushort(__float2half(v - __half2float(h)));
        }
    }
}
// the token rows of the table, rms-normed times w; grid T
__global__ void __launch_bounds__(256) k_embed(const __half* tab, const int* ids, const float* w, float* out) {
    const __half* row = tab + size_t(ids[blockIdx.x]) * kH;
    float ss = 0;
    for (int i = threadIdx.x; i < kH; i += 256) { const float v = __half2float(row[i]); ss += v * v; }
    const float r = rsqrtf(block_sum(ss) / kH + kEps);
    for (int i = threadIdx.x; i < kH; i += 256) out[size_t(blockIdx.x) * kH + i] = __half2float(row[i]) * r * w[i];
}

// Attention residual: for row t, the items (n blocks, then the partial) are rms-normed, each query scores them
// (dot / sqrt(H)), softmax over the items, out[t][q] = the mix of the normed items. grid T, 256 threads.
struct Items { const float* p[6]; int n; };   // p[0 .. n-1]: the blocks, then the partial, each [T][kH]
// the attention-residual scores for row t: sc[q][i] = softmax_i(q . rms(x_i) / sqrt(H)) / rms-scale, rr[i]; all sums in one pass
template <int MQ>
__device__ __forceinline__ void mix_scores(const Items& it, const float* queries, int nq, size_t t, float (*sc)[6], float* rr) {
    constexpr int NA = 6 * (MQ + 1);
    __shared__ float red[8][NA];
    __shared__ float tot[NA];
    const int n = it.n;
    float acc[NA];
#pragma unroll
    for (int k = 0; k < NA; ++k) acc[k] = 0.f;
    for (int c = threadIdx.x; c < kH; c += 256) {
        float qv[MQ];
#pragma unroll
        for (int q = 0; q < MQ; ++q) qv[q] = q < nq ? queries[q * kH + c] : 0.f;
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            if (i >= n) break;
            const float v = it.p[i][t * kH + c];
            acc[i * (MQ + 1)] += v * v;
#pragma unroll
            for (int q = 0; q < MQ; ++q) acc[i * (MQ + 1) + 1 + q] += v * qv[q];
        }
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
#pragma unroll
    for (int k = 0; k < NA; ++k) {
        const float v = warp_sum(acc[k]);
        if (l == 0) red[w][k] = v;
    }
    __syncthreads();
    if (threadIdx.x < NA) {
        float a = 0;
        for (int ww = 0; ww < 8; ++ww) a += red[ww][threadIdx.x];
        tot[threadIdx.x] = a;
    }
    __syncthreads();
    if (threadIdx.x < nq) {
        const int q = threadIdx.x;
        float r[6], z[6], mx = -INFINITY;
        for (int i = 0; i < n; ++i) {
            r[i] = rsqrtf(tot[i * (MQ + 1)] / kH + kEps);
            z[i] = tot[i * (MQ + 1) + 1 + q] * r[i] * rsqrtf(float(kH));
            mx = fmaxf(mx, z[i]);
        }
        float s = 0;
        for (int i = 0; i < n; ++i) { z[i] = expf(z[i] - mx); s += z[i]; }
        for (int i = 0; i < n; ++i) sc[q][i] = z[i] / s * r[i];
    }
    __syncthreads();
}
__global__ void __launch_bounds__(256) k_resmix(Items it, const float* queries, int nq, float* out) {
    __shared__ float sc[4][6];
    const size_t t = blockIdx.x;
    const int n = it.n;
    mix_scores<4>(it, queries, nq, t, sc, nullptr);
    for (int c = threadIdx.x; c < kH; c += 256) {
        float xv[6];
        for (int i = 0; i < n; ++i) xv[i] = it.p[i][t * kH + c];
        for (int q = 0; q < nq; ++q) {
            float a = 0;
            for (int i = 0; i < n; ++i) a += sc[q][i] * xv[i];
            out[(t * nq + q) * kH + c] = a;
        }
    }
}

// k_resmix then rmsnorm(mix_q) * nw[q] in one: out[q] (fp32), and when o16[q]: fp16 copies (lo16[q]: the remainder).
// nq <= 3; grid T, 256 threads.
struct NormOut { const float* nw[3]; float* out[3]; uint16_t* o16[3]; uint16_t* lo16[3]; };
__global__ void __launch_bounds__(256) k_resmix_norm(Items it, const float* queries, int nq, NormOut no) {
    constexpr int PER = (kH + 255) / 256;   // 9
    __shared__ float sc[3][6];
    const size_t t = blockIdx.x;
    const int n = it.n;
    mix_scores<3>(it, queries, nq, t, sc, nullptr);
    float mix[3][PER], ssq[3] = {0, 0, 0};
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int c = threadIdx.x + j * 256;
        float xv[6];
        for (int i = 0; i < n; ++i) xv[i] = c < kH ? it.p[i][t * kH + c] : 0.f;
#pragma unroll
        for (int q = 0; q < 3; ++q) {
            float a = 0;
            if (q < nq) for (int i = 0; i < n; ++i) a += sc[q][i] * xv[i];
            mix[q][j] = a;
            ssq[q] += a * a;
        }
    }
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        if (q >= nq) break;
        const float r = rsqrtf(block_sum(ssq[q]) / kH + kEps);
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int c = threadIdx.x + j * 256;
            if (c >= kH) break;
            const float v = mix[q][j] * r * no.nw[q][c];
            no.out[q][t * kH + c] = v;
            if (no.o16[q]) {
                const __half h = __float2half(fminf(fmaxf(v, -65504.f), 65504.f));
                no.o16[q][t * kH + c] = __half_as_ushort(h);
                if (no.lo16[q]) no.lo16[q][t * kH + c] = __half_as_ushort(__float2half(v - __half2float(h)));
            }
        }
    }
}

// k_resmix / k_resmix_norm with a row held in registers: 576 threads, a float4 each (pointers 16-byte aligned)
template <int MQ, bool NORM>
__global__ void __launch_bounds__(576) k_mix4(Items it, const float* queries, int nq, float* out, NormOut no, int ystep = 0) {
    constexpr int NA = 6 * (MQ + 1), NW = 18;
    queries += size_t(blockIdx.y) * ystep; out += size_t(blockIdx.y) * ystep;   // grid y: several query sets over the same items (the loop messages of a pass)
    __shared__ float red[NW][NA];
    __shared__ float tot[NA];
    __shared__ float sc[MQ][6];
    const size_t t = blockIdx.x;
    const int n = it.n, c4 = threadIdx.x, w = threadIdx.x >> 5, l = threadIdx.x & 31;
    float4 xv[6], qv[MQ];
#pragma unroll
    for (int i = 0; i < 6; ++i) xv[i] = i < n ? ((const float4*) (it.p[i] + t * kH))[c4] : make_float4(0, 0, 0, 0);
#pragma unroll
    for (int q = 0; q < MQ; ++q) qv[q] = q < nq ? ((const float4*) (queries + q * kH))[c4] : make_float4(0, 0, 0, 0);
    auto dot4 = [](float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; };
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        float v = warp_sum(dot4(xv[i], xv[i]));
        if (l == 0) red[w][i * (MQ + 1)] = v;
#pragma unroll
        for (int q = 0; q < MQ; ++q) {
            v = warp_sum(dot4(xv[i], qv[q]));
            if (l == 0) red[w][i * (MQ + 1) + 1 + q] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < NA) {
        float a = 0;
        for (int ww = 0; ww < NW; ++ww) a += red[ww][threadIdx.x];
        tot[threadIdx.x] = a;
    }
    __syncthreads();
    if (threadIdx.x < nq) {
        const int q = threadIdx.x;
        float r[6], z[6], mx = -INFINITY;
        for (int i = 0; i < n; ++i) {
            r[i] = rsqrtf(tot[i * (MQ + 1)] / kH + kEps);
            z[i] = tot[i * (MQ + 1) + 1 + q] * r[i] * rsqrtf(float(kH));
            mx = fmaxf(mx, z[i]);
        }
        float sm = 0;
        for (int i = 0; i < n; ++i) { z[i] = expf(z[i] - mx); sm += z[i]; }
        for (int i = 0; i < n; ++i) sc[q][i] = z[i] / sm * r[i];
    }
    __syncthreads();
    float4 mix[MQ];
#pragma unroll
    for (int q = 0; q < MQ; ++q) {
        float4 a = make_float4(0, 0, 0, 0);
        if (q < nq)
#pragma unroll
            for (int i = 0; i < 6; ++i)
                if (i < n) { const float f = sc[q][i]; a.x += f * xv[i].x; a.y += f * xv[i].y; a.z += f * xv[i].z; a.w += f * xv[i].w; }
        mix[q] = a;
    }
    if (!NORM) {
#pragma unroll
        for (int q = 0; q < MQ; ++q) if (q < nq) ((float4*) (out + (t * nq + q) * kH))[c4] = mix[q];
        return;
    }
#pragma unroll
    for (int q = 0; q < MQ; ++q) {
        const float v = warp_sum(dot4(mix[q], mix[q]));
        if (l == 0) red[w][q] = v;
    }
    __syncthreads();
    if (threadIdx.x < MQ) {
        float a = 0;
        for (int ww = 0; ww < NW; ++ww) a += red[ww][threadIdx.x];
        tot[threadIdx.x] = rsqrtf(a / kH + kEps);
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < MQ; ++q) {
        if (q >= nq) break;
        const float r = tot[q];
        const float4 nw = ((const float4*) no.nw[q])[c4];
        const float4 v = make_float4(mix[q].x * r * nw.x, mix[q].y * r * nw.y, mix[q].z * r * nw.z, mix[q].w * r * nw.w);
        ((float4*) (no.out[q] + t * kH))[c4] = v;
        if (no.o16[q]) {
            const float vv[4] = {v.x, v.y, v.z, v.w};
            for (int k = 0; k < 4; ++k) {
                const __half hh = __float2half(fminf(fmaxf(vv[k], -65504.f), 65504.f));
                no.o16[q][t * kH + 4 * c4 + k] = __half_as_ushort(hh);
                if (no.lo16[q]) no.lo16[q][t * kH + 4 * c4 + k] = __half_as_ushort(__float2half(vv[k] - __half2float(hh)));
            }
        }
    }
}
// k_rms with a float4 a thread (576 threads; x rows 16-byte aligned)
__global__ void __launch_bounds__(576) k_rms4(const float* x, int xs, const float* w, float* out, int os, uint16_t* o16 = nullptr,
                                              uint16_t* lo16 = nullptr) {
    __shared__ float red[18];
    const int c4 = threadIdx.x, wv = threadIdx.x >> 5, l = threadIdx.x & 31;
    const float4 a = ((const float4*) (x + size_t(blockIdx.x) * xs))[c4];
    float ss = warp_sum(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
    if (l == 0) red[wv] = ss;
    __syncthreads();
    ss = 0;
    for (int i = 0; i < 18; ++i) ss += red[i];
    const float r = rsqrtf(ss / kH + kEps);
    const float4 ww = w ? ((const float4*) w)[c4] : make_float4(1, 1, 1, 1);
    const float4 v = make_float4(a.x * r * ww.x, a.y * r * ww.y, a.z * r * ww.z, a.w * r * ww.w);
    ((float4*) (out + size_t(blockIdx.x) * os))[c4] = v;
    if (o16) {
        const float vv[4] = {v.x, v.y, v.z, v.w};
        for (int k = 0; k < 4; ++k) {
            const __half h = __float2half(fminf(fmaxf(vv[k], -65504.f), 65504.f));
            o16[size_t(blockIdx.x) * kH + 4 * c4 + k] = __half_as_ushort(h);
            if (lo16) lo16[size_t(blockIdx.x) * kH + 4 * c4 + k] = __half_as_ushort(__float2half(vv[k] - __half2float(h)));
        }
    }
}

// Message gate: out = (1 - tanh(F g)) h + tanh(G g) m, g = D . [rms(h), rms(m)] (64). grid T, 256 threads.
// h, m, out rows have strides hs, ms, os.
__global__ void __launch_bounds__(256) k_msg_gate(const float* h, int hs, const float* m, int ms, const __half* D, const float* Fw,
                                                  const float* Gw, float* out, int os) {
    __shared__ float xs[2 * kH];
    __shared__ float g[kGR];
    const size_t t = blockIdx.x;
    const float* hr = h + t * hs;
    const float* mr = m + t * ms;
    float s1 = 0, s2 = 0;
    for (int c = threadIdx.x; c < kH; c += 256) { const float a = hr[c], b = mr[c]; xs[c] = a; xs[kH + c] = b; s1 += a * a; s2 += b * b; }
    const float r1 = rsqrtf(block_sum(s1) / kH + kEps), r2 = rsqrtf(block_sum(s2) / kH + kEps);
    for (int c = threadIdx.x; c < kH; c += 256) { xs[c] *= r1; xs[kH + c] *= r2; }
    __syncthreads();
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    for (int j = w; j < kGR; j += 8) {
        const __half* row = D + size_t(j) * 2 * kH;
        float a = 0;
        for (int c = l; c < 2 * kH; c += 32) a += __half2float(row[c]) * xs[c];
        a = warp_sum(a);
        if (l == 0) g[j] = a;
    }
    __syncthreads();
    float* o = out + t * os;
    for (int c = threadIdx.x; c < kH; c += 256) {
        float f = 0, gg = 0;
        for (int j = 0; j < kGR; ++j) { f += Fw[c * kGR + j] * g[j]; gg += Gw[c * kGR + j] * g[j]; }
        o[c] = (1.f - tanhf(f)) * hr[c] + tanhf(gg) * mr[c];
    }
}

// the message gate's 64 values: G[t][j] = D[j] . [rms(h), rms(m)]; grid (kGR / 8, T), a warp a row of D
__global__ void __launch_bounds__(256) k_gate_g(const float* h, int hs, const float* m, int ms, const __half* D, float* G) {
    const size_t t = blockIdx.y;
    const float* hr = h + t * hs;
    const float* mr = m + t * ms;
    float s1 = 0, s2 = 0;
    for (int c = threadIdx.x; c < kH; c += 256) { const float a = hr[c], b = mr[c]; s1 += a * a; s2 += b * b; }
    s1 = block_sum(s1);
    s2 = block_sum(s2);
    const float r1 = rsqrtf(s1 / kH + kEps), r2 = rsqrtf(s2 / kH + kEps);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, j = blockIdx.x * 8 + w;
    const __half* row = D + size_t(j) * 2 * kH;
    float a1 = 0, a2 = 0;
    for (int c = l * 8; c < kH; c += 256) {
        const uint4 p = *(const uint4*) (row + c), q = *(const uint4*) (row + kH + c);
        const __half2* ph = (const __half2*) &p;
        const __half2* qh = (const __half2*) &q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(ph[i]), g = __half22float2(qh[i]);
            a1 += f.x * hr[c + 2 * i] + f.y * hr[c + 2 * i + 1];
            a2 += g.x * mr[c + 2 * i] + g.y * mr[c + 2 * i + 1];
        }
    }
    const float a = warp_sum(a1 * r1 + a2 * r2);
    if (l == 0) G[t * kGR + j] = a;
}
// up to 4 gates in one launch (blockIdx.z): k_gate_g / k_gate_o for each job
struct GateJob { const float* h; int hs; const float* m; int ms; const __half* D; const float *Fw, *Gw; float* out; float* G; };
struct GateJobs { GateJob j[4]; };
__global__ void __launch_bounds__(256) k_gate_gn(GateJobs jobs) {
    const GateJob& J = jobs.j[blockIdx.z];
    const size_t t = blockIdx.y;
    const float* hr = J.h + t * J.hs;
    const float* mr = J.m + t * J.ms;
    float s1 = 0, s2 = 0;
    for (int c = threadIdx.x; c < kH; c += 256) { const float a = hr[c], b = mr[c]; s1 += a * a; s2 += b * b; }
    s1 = block_sum(s1);
    s2 = block_sum(s2);
    const float r1 = rsqrtf(s1 / kH + kEps), r2 = rsqrtf(s2 / kH + kEps);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, j = blockIdx.x * 8 + w;
    const __half* row = J.D + size_t(j) * 2 * kH;
    float a1 = 0, a2 = 0;
    for (int c = l * 8; c < kH; c += 256) {
        const uint4 p = *(const uint4*) (row + c), q = *(const uint4*) (row + kH + c);
        const __half2* ph = (const __half2*) &p;
        const __half2* qh = (const __half2*) &q;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(ph[i]), g = __half22float2(qh[i]);
            a1 += f.x * hr[c + 2 * i] + f.y * hr[c + 2 * i + 1];
            a2 += g.x * mr[c + 2 * i] + g.y * mr[c + 2 * i + 1];
        }
    }
    const float a = warp_sum(a1 * r1 + a2 * r2);
    if (l == 0) J.G[t * kGR + j] = a;
}
__global__ void __launch_bounds__(256) k_gate_on(GateJobs jobs) {
    const GateJob& J = jobs.j[blockIdx.z];
    const size_t t = blockIdx.y;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const float g0 = J.G[t * kGR + l], g1 = J.G[t * kGR + 32 + l];
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        const int c = blockIdx.x * 32 + w * 4 + rr;
        const float f = warp_sum(J.Fw[c * kGR + l] * g0 + J.Fw[c * kGR + 32 + l] * g1);
        const float gg = warp_sum(J.Gw[c * kGR + l] * g0 + J.Gw[c * kGR + 32 + l] * g1);
        if (l == 0) J.out[t * kH + c] = (1.f - tanhf(f)) * J.h[t * J.hs + c] + tanhf(gg) * J.m[t * J.ms + c];
    }
}
// The same two steps with more blocks at work (k_gate_gn ran 24 blocks for 1.8 MB, k_gate_on read a float a lane):
// k_gate_gp: each row of D in kGP pieces of 256 columns (grid (kGR / 8 x kGP, T, gates)), unnormed partial sums of the
// h and m halves into GP[gate][t][piece][j][2]; the first block of each (t, gate) also writes h's and m's rms factors.
// k_gate_op: g from the partials, then a warp 4 channels, each read as one float4 a lane (lanes 0-15 F's row, 16-31 Gw's).
constexpr int kGP = kH / 256, kGPS = kGP * kGR * 2 + 2;   // pieces; floats a (gate, row) in GP
__global__ void __launch_bounds__(256) k_gate_gp(GateJobs jobs, float* GP, int T) {
    const GateJob& J = jobs.j[blockIdx.z];
    const int t = blockIdx.y, part = blockIdx.x % kGP, jb = blockIdx.x / kGP;
    const float* hr = J.h + size_t(t) * J.hs;
    const float* mr = J.m + size_t(t) * J.ms;
    float* G = GP + (size_t(blockIdx.z) * T + t) * kGPS;
    if (blockIdx.x == 0) {
        float s1 = 0, s2 = 0;
        for (int c = threadIdx.x; c < kH; c += 256) { const float a = hr[c], b = mr[c]; s1 += a * a; s2 += b * b; }
        s1 = block_sum(s1);
        s2 = block_sum(s2);
        if (threadIdx.x == 0) { G[kGP * kGR * 2] = rsqrtf(s1 / kH + kEps); G[kGP * kGR * 2 + 1] = rsqrtf(s2 / kH + kEps); }
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, j = jb * 8 + w, c = part * 256 + l * 8;
    const __half* row = J.D + size_t(j) * 2 * kH;
    const uint4 p = *(const uint4*) (row + c), q = *(const uint4*) (row + kH + c);
    const float4 h0 = *(const float4*) (hr + c), h1 = *(const float4*) (hr + c + 4);
    const float4 m0 = *(const float4*) (mr + c), m1 = *(const float4*) (mr + c + 4);
    const float hv[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w}, mv[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
    const __half2* ph = (const __half2*) &p;
    const __half2* qh = (const __half2*) &q;
    float a1 = 0, a2 = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(ph[i]), g = __half22float2(qh[i]);
        a1 += f.x * hv[2 * i] + f.y * hv[2 * i + 1];
        a2 += g.x * mv[2 * i] + g.y * mv[2 * i + 1];
    }
    a1 = warp_sum(a1);
    a2 = warp_sum(a2);
    if (l == 0) { G[(part * kGR + j) * 2] = a1; G[(part * kGR + j) * 2 + 1] = a2; }
}
__global__ void __launch_bounds__(256) k_gate_op(GateJobs jobs, const float* GP, int T) {
    const GateJob& J = jobs.j[blockIdx.z];
    const int t = blockIdx.y;
    const float* G = GP + (size_t(blockIdx.z) * T + t) * kGPS;
    __shared__ float g[kGR];
    if (threadIdx.x < kGR) {
        float a1 = 0, a2 = 0;
#pragma unroll
        for (int pt = 0; pt < kGP; ++pt) { a1 += G[(pt * kGR + threadIdx.x) * 2]; a2 += G[(pt * kGR + threadIdx.x) * 2 + 1]; }
        g[threadIdx.x] = a1 * G[kGP * kGR * 2] + a2 * G[kGP * kGR * 2 + 1];
    }
    __syncthreads();
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, k4 = (l & 15) * 4;
    const float g0 = g[k4], g1 = g[k4 + 1], g2 = g[k4 + 2], g3 = g[k4 + 3];
    const float* M = l < 16 ? J.Fw : J.Gw;
    float4 v[4];
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) v[rr] = *(const float4*) (M + size_t(blockIdx.x * 32 + w * 4 + rr) * kGR + k4);
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        const int c = blockIdx.x * 32 + w * 4 + rr;
        float a = v[rr].x * g0 + v[rr].y * g1 + v[rr].z * g2 + v[rr].w * g3;
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) a += __shfl_xor(a, o, 32);   // (within each half of the warp)
        const float gg = __shfl(a, 16, 32);
        if (l == 0) J.out[size_t(t) * kH + c] = (1.f - tanhf(a)) * J.h[size_t(t) * J.hs + c] + tanhf(gg) * J.m[size_t(t) * J.ms + c];
    }
}
// out = (1 - tanh(F g)) h + tanh(Gw g) m; grid (kH / 32, T), a warp 4 channels
__global__ void __launch_bounds__(256) k_gate_o(const float* h, int hs, const float* m, int ms, const float* G, const float* Fw,
                                                const float* Gw, float* out, int os) {
    const size_t t = blockIdx.y;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const float g0 = G[t * kGR + l], g1 = G[t * kGR + 32 + l];
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        const int c = blockIdx.x * 32 + w * 4 + rr;
        const float f = warp_sum(Fw[c * kGR + l] * g0 + Fw[c * kGR + 32 + l] * g1);
        const float gg = warp_sum(Gw[c * kGR + l] * g0 + Gw[c * kGR + 32 + l] * g1);
        if (l == 0) out[t * os + c] = (1.f - tanhf(f)) * h[t * hs + c] + tanhf(gg) * m[t * ms + c];
    }
}

// y[t][r] = W[r] . x[t] for the rows of up to three segments, each with its own input; fp16 weights, fp32 everything else.
// 32 rows a block (8 warps of 4), grid (rows / 32, T). Segments start and end on multiples of 32.
struct MvSeg { const float* x[3]; int lo[3], hi[3]; int n; int xs; const float* nw[3] = {nullptr, nullptr, nullptr}; };   // nw: rms-norm the input with these weights as it is loaded (k_mv8)
// a single row's sub-layer output through its 4-tap convolution, plus the residual, the state moved on (k_conv_h for one
// row, done where the value is made): out[c] = res[c] + (sum_lag w[c][lag] x[-lag] + x) / sqrt(5)
struct ConvEpi { const float* w = nullptr; float* st = nullptr; const float* res = nullptr; float* out = nullptr; };
__device__ __forceinline__ void conv_epi(const ConvEpi& e, int c, float x) {
    float* st = e.st;
    const float s0 = st[c], s1 = st[kH + c], s2 = st[2 * kH + c];
    const float y = (e.w[c * 4] * x + e.w[c * 4 + 1] * s0 + e.w[c * 4 + 2] * s1 + e.w[c * 4 + 3] * s2 + x) * rsqrtf(5.f);
    e.out[c] = (e.res ? e.res[c] : 0.f) + y;
    st[2 * kH + c] = s1; st[kH + c] = s0; st[c] = x;
}
__global__ void __launch_bounds__(256) k_mv16(const __half* W, int K, MvSeg s, float* y, int ys) {
    extern __shared__ float xsh[];
    const int r0 = blockIdx.x * 32;
    int sg = -1;
    for (int j = 0; j < s.n; ++j) if (r0 >= s.lo[j] && r0 < s.hi[j]) sg = j;
    if (sg < 0) return;
    const float* x = s.x[sg] + size_t(blockIdx.y) * s.xs;
    for (int i = threadIdx.x; i < K; i += 256) xsh[i] = x[i];
    __syncthreads();
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    float a[4] = {0, 0, 0, 0};
    for (int c = l; c < K / 8; c += 32) {
        float xv[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) xv[i] = xsh[c * 8 + i];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const uint4 wv = *(const uint4*) (W + size_t(r0 + w * 4 + r) * K + c * 8);
            const __half2* hp = (const __half2*) &wv;
#pragma unroll
            for (int i = 0; i < 4; ++i) { const float2 f = __half22float2(hp[i]); a[r] += f.x * xv[2 * i] + f.y * xv[2 * i + 1]; }
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const float v = warp_sum(a[r]);
        if (l == 0) y[size_t(blockIdx.y) * ys + r0 + w * 4 + r] = v;
    }
}

// The writing path's 8-bit copy of a dense matrix (INKLING_W8, default on): int8 rows and an fp16 scale a 32 values
// (llama.cpp's Q8_0: d = max |w| / 127 of the 32, q = round(w / d)) - half the bytes of fp16 a token.
// RW rows a warp (a block 8 RW rows); the inputs in LDS padded (16 values then 4 spare): K * 5 bytes of LDS
template <int RW>
__global__ void __launch_bounds__(256) k_mv8(W8 W, int K, MvSeg s, float* y, int ys, ConvEpi epi = ConvEpi{}, W4 W4q = W4{}, int split4 = 0) {
    extern __shared__ float xsh[];
    const int r0 = blockIdx.x * 8 * RW;
    int sg = -1;
    for (int j = 0; j < s.n; ++j) if (r0 >= s.lo[j] && r0 < s.hi[j]) sg = j;
    if (sg < 0) return;
    const float* x = s.x[sg] + size_t(blockIdx.y) * s.xs;
    const float* nw = s.nw[sg];
    float ss = 0;
    for (int i = threadIdx.x; i < K; i += 256) { const float xv = x[i]; xsh[i + (i >> 4) * 4] = xv; ss += xv * xv; }
    if (nw) {
        const float r = rsqrtf(block_sum(ss) / K + kEps);
        for (int i = threadIdx.x; i < K; i += 256) xsh[i + (i >> 4) * 4] *= r * nw[i];
    }
    __syncthreads();
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    float a[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) a[r] = 0.f;
    for (int c = l; c < K / 16; c += 32) {   // 16 values a lane a step
        float xv[16];
#pragma unroll
        for (int i = 0; i < 16; i += 4) *(float4*) (xv + i) = *(const float4*) (xsh + c * 20 + i);
        if (r0 < split4) {   // INKLING_ATTN4: these rows from the 4-bit copy (half the bytes again)
#pragma unroll
            for (int r = 0; r < RW; ++r) {
                const size_t row = size_t(r0 + w * RW + r);
                const uint2 q = *(const uint2*) (W4q.q + row * (K / 2) + c * 8);
                const uint32_t qq[2] = {q.x, q.y};
                float t = 0;
#pragma unroll
                for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int b = 0; b < 8; ++b) t += float((int(qq[j] << (28 - 4 * b))) >> 28) * xv[j * 8 + b];
                a[r] += __half2float(W4q.d[row * (K / 32) + (c >> 1)]) * t;
            }
            continue;
        }
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const size_t row = size_t(r0 + w * RW + r);
            const uint4 q = *(const uint4*) (W.q + row * K + c * 16);
            const uint32_t qq[4] = {q.x, q.y, q.z, q.w};
            float t = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int b = 0; b < 4; ++b) t += float(int8_t(qq[j] >> (8 * b))) * xv[j * 4 + b];
            a[r] += __half2float(W.d[row * (K / 32) + (c >> 1)]) * t;
        }
    }
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        const float v = warp_sum(a[r]);
        if (l == 0) {
            y[size_t(blockIdx.y) * ys + r0 + w * RW + r] = v;
            if (epi.out) conv_epi(epi, r0 + w * RW + r, v);
        }
    }
}
// k_mv8 with K fixed at compile time and the (row, 16-value piece) pairs of a warp's RW rows dealt out to its lanes in
// one flat run: K 2304 has 144 pieces a row, which leaves half the lanes idle on k_mv8's last step; RW x 144 pieces dealt
// together divide evenly, and the fixed trip count lets the loads of several steps be in flight at once.
template <int RW, int K>
__global__ void __launch_bounds__(256) k_mv8f(W8 W, MvSeg s, float* y, int ys, ConvEpi epi = ConvEpi{}) {
    extern __shared__ float xsh[];
    const int r0 = blockIdx.x * 8 * RW;
    int sg = -1;
    for (int j = 0; j < s.n; ++j) if (r0 >= s.lo[j] && r0 < s.hi[j]) sg = j;
    if (sg < 0) return;
    const float* x = s.x[sg] + size_t(blockIdx.y) * s.xs;
    const float* nw = s.nw[sg];
    float ss = 0;
    for (int i = threadIdx.x; i < K; i += 256) { const float xv = x[i]; xsh[i + (i >> 4) * 4] = xv; ss += xv * xv; }
    if (nw) {
        const float r = rsqrtf(block_sum(ss) / K + kEps);
        for (int i = threadIdx.x; i < K; i += 256) xsh[i + (i >> 4) * 4] *= r * nw[i];
    }
    __syncthreads();
    constexpr int C = K / 16, N = RW * C, STEPS = (N + 31) / 32;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const size_t rb = size_t(r0 + w * RW);
    float a[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) a[r] = 0.f;
#pragma unroll 3
    for (int st = 0; st < STEPS; ++st) {
        const int i = l + 32 * st;
        if (N % 32 != 0 && i >= N) break;
        const int r = i / C, c = i - r * C;
        const size_t row = rb + r;
        const uint4 q = *(const uint4*) (W.q + row * K + c * 16);
        const float d = __half2float(W.d[row * (K / 32) + (c >> 1)]);
        float xv[16];
#pragma unroll
        for (int k = 0; k < 16; k += 4) *(float4*) (xv + k) = *(const float4*) (xsh + c * 20 + k);
        const uint32_t qq[4] = {q.x, q.y, q.z, q.w};
        float t = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int b = 0; b < 4; ++b) t += float(int8_t(qq[j] >> (8 * b))) * xv[j * 4 + b];
        const float v = d * t;
#pragma unroll
        for (int rr = 0; rr < RW; ++rr) a[rr] += rr == r ? v : 0.f;
    }
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        const float v = warp_sum(a[r]);
        if (l == 0) {
            y[size_t(blockIdx.y) * ys + r0 + w * RW + r] = v;
            if (epi.out) conv_epi(epi, r0 + w * RW + r, v);
        }
    }
}
// the head from its 8-bit copy, softcapped; grid (kVocabP / 32, T)
__global__ void __launch_bounds__(256) k_head8(W8 W, const float* x, float* logits) {
    __shared__ float xsh[kH];
    const int t = blockIdx.y, r0 = blockIdx.x * 32;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (r0 >= kVocab) {
        for (int i = threadIdx.x; i < 32; i += 256) logits[size_t(t) * kVocabP + r0 + i] = -INFINITY;
        return;
    }
    for (int i = threadIdx.x; i < kH; i += 256) xsh[i] = x[size_t(t) * kH + i];
    __syncthreads();
    float a[4] = {0, 0, 0, 0};
    for (int c = l; c < kH / 16; c += 32) {
        float xv[16];
#pragma unroll
        for (int i = 0; i < 16; i += 4) *(float4*) (xv + i) = *(const float4*) (xsh + c * 16 + i);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const size_t row = size_t(min(r0 + w * 4 + r, kVocab - 1));
            const uint4 q = *(const uint4*) (W.q + row * kH + c * 16);
            const uint32_t qq[4] = {q.x, q.y, q.z, q.w};
            float tt = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int b = 0; b < 4; ++b) tt += float(int8_t(qq[j] >> (8 * b))) * xv[j * 4 + b];
            a[r] += __half2float(W.d[row * (kH / 32) + (c >> 1)]) * tt;
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const float v = warp_sum(a[r]);
        const int row = r0 + w * 4 + r;
        if (l == 0) logits[size_t(t) * kVocabP + row] = row < kVocab ? softcapf(v, kOutCap) : -INFINITY;
    }
}

// The writing path's two-pass head (INKLING_HEAD4, default on): a 4-bit copy of the head (two values a byte, an fp16 scale
// a 32 values: half the 8-bit copy's bytes) gives every logit roughly, then only the tokens within kHeadMargin of the
// best are redone from the 8-bit copy (k_head_ref). The logits a sampler can pick are the 8-bit copy's exactly.
constexpr int kHeadRefB = kVocabP / 1024;
// grid (kVocabP / 32): raw (not softcapped) logits, and this block's largest into hmax
__global__ void __launch_bounds__(256) k_head4(W4 W, const float* x, float* logits, float* hmax) {
    __shared__ float xsh[kH];
    __shared__ float wm[8];
    const int r0 = blockIdx.x * 32;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (r0 >= kVocab) {
        for (int i = threadIdx.x; i < 32; i += 256) logits[r0 + i] = -INFINITY;
        if (threadIdx.x == 0) hmax[blockIdx.x] = -INFINITY;
        return;
    }
    for (int i = threadIdx.x; i < kH; i += 256) xsh[i] = x[i];
    __syncthreads();
    float a[4] = {0, 0, 0, 0};
    for (int c = l; c < kH / 16; c += 32) {
        float xv[16];
#pragma unroll
        for (int i = 0; i < 16; i += 4) *(float4*) (xv + i) = *(const float4*) (xsh + c * 16 + i);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const size_t row = size_t(min(r0 + w * 4 + r, kVocab - 1));
            const uint2 q = *(const uint2*) (W.q + row * (kH / 2) + c * 8);
            const uint32_t qq[2] = {q.x, q.y};
            float tt = 0;
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int b = 0; b < 8; ++b) tt += float((int(qq[j] << (28 - 4 * b))) >> 28) * xv[j * 8 + b];
            a[r] += __half2float(W.d[row * (kH / 32) + (c >> 1)]) * tt;
        }
    }
    float m = -INFINITY;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const float v = warp_sum(a[r]);
        const int row = r0 + w * 4 + r;
        const float o = row < kVocab ? v : -INFINITY;
        if (l == 0) logits[row] = o;
        m = fmaxf(m, o);
    }
    if (l == 0) wm[w] = m;
    __syncthreads();
    if (threadIdx.x == 0) { for (int i = 1; i < 8; ++i) m = fmaxf(m, wm[i]); hmax[blockIdx.x] = m; }
}
// grid (kHeadRefB), 256 threads, 1024 tokens a block: softcap every logit; the ones within margin of the best first
// redone from the 8-bit copy
__global__ void __launch_bounds__(256) k_head_ref(W8 W, const float* x, float* logits, const float* hmax, float margin) {
    __shared__ float xsh[kH];
    __shared__ float wm[8];
    __shared__ int list[1024];
    __shared__ int cnt;
    __shared__ float gm;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, r0 = blockIdx.x * 1024;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < kVocabP / 32; i += 256) m = fmaxf(m, hmax[i]);
    m = warp_max(m);
    if (l == 0) wm[w] = m;
    if (threadIdx.x == 0) cnt = 0;
    for (int i = threadIdx.x; i < kH; i += 256) xsh[i] = x[i];
    __syncthreads();
    if (threadIdx.x == 0) { for (int i = 1; i < 8; ++i) m = fmaxf(m, wm[i]); gm = m; }
    __syncthreads();
    const float cut = softcapf(gm, kOutCap) - margin;   // (in softcapped units: what a sampler sees)
    const float4 v4 = *(const float4*) (logits + r0 + threadIdx.x * 4);
    const float vv[4] = {v4.x, v4.y, v4.z, v4.w};
    float o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        o[j] = vv[j] == -INFINITY ? -INFINITY : softcapf(vv[j], kOutCap);
        if (o[j] >= cut) list[atomicAdd(&cnt, 1)] = r0 + threadIdx.x * 4 + j;
    }
    *(float4*) (logits + r0 + threadIdx.x * 4) = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    const int n = cnt;
    for (int k = w; k < n; k += 8) {
        const size_t row = size_t(list[k]);
        float a = 0;
        for (int c = l; c < kH / 16; c += 32) {
            float xv[16];
#pragma unroll
            for (int i = 0; i < 16; i += 4) *(float4*) (xv + i) = *(const float4*) (xsh + c * 16 + i);
            const uint4 q = *(const uint4*) (W.q + row * kH + c * 16);
            const uint32_t qq[4] = {q.x, q.y, q.z, q.w};
            float tt = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int b = 0; b < 4; ++b) tt += float(int8_t(qq[j] >> (8 * b))) * xv[j * 4 + b];
            a += __half2float(W.d[row * (kH / 32) + (c >> 1)]) * tt;
        }
        a = warp_sum(a);
        if (l == 0) logits[row] = softcapf(a, kOutCap);
    }
}

// After the input product: q heads scaled, clipped to kQKClip (global: times tau(pos)), then times 1 / sqrt(128); the
// relative states likewise (clip kRelClip), in place. grid (T), 256 threads.
__device__ __forceinline__ float log_tau(int pos) { return 1.f + kLogAlpha * logf(fmaxf(float(pos + 1) / kLogFloor, 1.f)); }
__global__ void __launch_bounds__(256) k_qrel(float* P, const float* qscale, const float* rscale, const int* state, int glob) {
    const int t = blockIdx.x, pos = state[0] + t;
    const float tau = glob ? log_tau(pos) : 1.f;
    float* p = P + size_t(t) * kNP;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    for (int h = w; h < kQH; h += 8) {
        float v[4], ss = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) { v[i] = p[kRQ + h * kHD + l + 32 * i] * qscale[h * kHD + l + 32 * i]; ss += v[i] * v[i]; }
        ss = warp_sum(ss);
        const float f = fminf(kQKClip / (sqrtf(ss) + 1e-6f), 1.f) * tau * 0.08838834764831845f;
#pragma unroll
        for (int i = 0; i < 4; ++i) p[kRQ + h * kHD + l + 32 * i] = v[i] * f;
        float r = l < kDRel ? p[kRR + h * kDRel + l] * rscale[h * kDRel + l] : 0.f;
        const float rn = warp_sum(r * r);
        r *= fminf(kRelClip / (sqrtf(rn) + 1e-6f), 1.f) * tau;
        if (l < kDRel) p[kRR + h * kDRel + l] = r;
    }
}

// A conversation's own buffers of one visit (--parallel). The kernels that read or write them take a table of these
// ([conversation][visit], already moved on to the visit) with rs (row -> conversation), and then every row is a single
// row of its own conversation: its position, its caches, its convolution states (moved on in place).
struct SlotRow { kc_t *K, *V; int8_t *K8, *V8; __half *KS, *VS; float *cs_k, *cs_v, *cs_a, *cs_m; int* state; };
__global__ void k_advance_rows(const SlotRow* tab, const int* rs, int n) {
    for (int r = 0; r < n; ++r) tab[size_t(rs[r]) * kVisits].state[0] += 1;
}
// K and V: the short convolution over the product's rows (taps by backward distance; global layers add the token's own
// taps, one set of 4 a head), plus the input, over sqrt(5) or sqrt(9); then k clipped to kQKClip, v rms-normed; into
// the caches at the row's slot. Rows before the chunk come from the convolution state st ([3][1024]: t-1, t-2, t-3).
// grid (8 heads, 2: k / v, T), 128 threads.
__global__ void __launch_bounds__(128) k_kvconv(const float* P, const float* kw, const float* vw, const float* kst, const float* vst,
                                                const int* state, int glob, int mask, kc_t* K, kc_t* V, int upd,
                                                const float* qscale = nullptr, const float* rscale = nullptr,
                                                int8_t* K8 = nullptr, int8_t* V8 = nullptr, __half* KS = nullptr, __half* VS = nullptr,
                                                const SlotRow* tab = nullptr, const int* rs = nullptr) {
    int tl = blockIdx.z;   // the row's place in its chunk (a conversation's row: alone)
    if (tab) {
        const SlotRow& s = tab[size_t(rs[blockIdx.z]) * kVisits];
        kst = s.cs_k; vst = s.cs_v; state = s.state; K = s.K; V = s.V; K8 = s.K8; V8 = s.V8; KS = s.KS; VS = s.VS;
        upd = 1; tl = 0;
    }
    if (blockIdx.y == 2) {   // (grid y 3: k_qrel's work for this K/V head's 3 query heads, warps 0 .. 2)
        const int t = blockIdx.z, pos = state[0] + tl, w = threadIdx.x >> 5, l = threadIdx.x & 31;
        if (w >= 3) return;
        const int h = blockIdx.x * 3 + w;
        const float tau = glob ? log_tau(pos) : 1.f;
        float* p = const_cast<float*>(P) + size_t(t) * kNP;
        float v[4], ss = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) { v[i] = p[kRQ + h * kHD + l + 32 * i] * qscale[h * kHD + l + 32 * i]; ss += v[i] * v[i]; }
        ss = warp_sum(ss);
        const float f = fminf(kQKClip / (sqrtf(ss) + 1e-6f), 1.f) * tau * 0.08838834764831845f;
#pragma unroll
        for (int i = 0; i < 4; ++i) p[kRQ + h * kHD + l + 32 * i] = v[i] * f;
        float r = l < kDRel ? p[kRR + h * kDRel + l] * rscale[h * kDRel + l] : 0.f;
        const float rn = warp_sum(r * r);
        r *= fminf(kRelClip / (sqrtf(rn) + 1e-6f), 1.f) * tau;
        if (l < kDRel) p[kRR + h * kDRel + l] = r;
        return;
    }
    const int h = blockIdx.x, isv = blockIdx.y, t = blockIdx.z, c = h * kHD + threadIdx.x;
    const int col = (isv ? kRV : kRK) + c, tcol = (isv ? kRVP : kRKP) + h * 4;
    const float* w = isv ? vw : kw;
    const float* st = isv ? vst : kst;
    float y = P[size_t(t) * kNP + col];
    float acc = 0;
#pragma unroll
    for (int lag = 0; lag < 4; ++lag) {
        const int tt = tl - lag;
        const float x = tt >= 0 ? P[size_t(t - lag) * kNP + col] : st[(lag - tl - 1) * kKVD + c];
        float tap = w[c * 4 + lag];
        if (glob) tap += P[size_t(t) * kNP + tcol + lag];
        acc += tap * x;
    }
    y = (acc + y) * (glob ? (1.f / 3.f) : rsqrtf(5.f));
    const float ss = block_sum(y * y);
    const float o = isv ? y * rsqrtf(ss / kHD + kEps) : y * fminf(kQKClip / (sqrtf(ss) + 1e-6f), 1.f);
    const int pos = state[0] + tl;
    const size_t slot = size_t(pos & mask) * kKVD + c;
    if (K8) {   // int8 with an fp16 scale a (row, head): the head's 128 values over their largest
        __shared__ float bm[4];
        const float am = warp_max(fabsf(o));
        if ((threadIdx.x & 31) == 0) bm[threadIdx.x >> 5] = am;
        __syncthreads();
        const float mx = fmaxf(fmaxf(bm[0], bm[1]), fmaxf(bm[2], bm[3]));
        const __half sh = __float2half(mx > 0.f ? mx / 127.f : 1.f);
        const float s = __half2float(sh);
        (isv ? V8 : K8)[slot] = int8_t(fminf(fmaxf(rintf(o / s), -127.f), 127.f));
        if (threadIdx.x == 0) (isv ? VS : KS)[size_t(pos & mask) * kKV + h] = sh;
    } else (isv ? V : K)[slot] = __float2half(o);
    if (upd) {   // (a single row: this thread alone reads its channel's state, so it moves the state on itself)
        float* sw = const_cast<float*>(st);
        const float s0 = sw[c], s1 = sw[kKVD + c];
        sw[2 * kKVD + c] = s1; sw[kKVD + c] = s0; sw[c] = P[size_t(t) * kNP + col];
    }
}
// out[t][c] = res[t][c] (if res) + ((sum_lag w[c][lag] x[t - lag][c]) + x[t][c]) / sqrt(5); C = kH; grid (kH / 256, T)
__global__ void __launch_bounds__(256) k_conv_h(const float* x, const float* w, const float* st, const float* res, float* out, int upd,
                                                const SlotRow* tab = nullptr, const int* rs = nullptr, int which = 0) {
    const int c = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    int tl = t;
    if (tab) {   // (a conversation's row: its own state, the attention's (0) or the experts' (1))
        const SlotRow& s = tab[size_t(rs[t]) * kVisits];
        st = which ? s.cs_m : s.cs_a; upd = 1; tl = 0;
    }
    float acc = 0;
#pragma unroll
    for (int lag = 0; lag < 4; ++lag) {
        const int tt = tl - lag;
        const float v = tt >= 0 ? x[size_t(t - lag) * kH + c] : st[(lag - tl - 1) * kH + c];
        acc += w[c * 4 + lag] * v;
    }
    const float xc = x[size_t(t) * kH + c];
    const float y = (acc + xc) * rsqrtf(5.f);
    out[size_t(t) * kH + c] = (res ? res[size_t(t) * kH + c] : 0.f) + y;
    if (upd) {   // (a single row: the state moves on here)
        float* sw = const_cast<float*>(st);
        const float s0 = sw[c], s1 = sw[kH + c];
        sw[2 * kH + c] = s1; sw[kH + c] = s0; sw[c] = xc;
    }
}
// the convolution states after a chunk of T rows: st[j] = x[T - 1 - j] (or the old st[j - T]); one thread a channel
struct ConvUpd { const float* x[4]; int xs[4]; float* st[4]; int C[4]; };
__global__ void k_conv_state(ConvUpd u, int T) {
    const int k = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    if (c >= u.C[k]) return;
    const int C = u.C[k];
    float* st = u.st[k];
    float nv[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) nv[j] = T - 1 - j >= 0 ? u.x[k][size_t(T - 1 - j) * u.xs[k] + c] : st[(j - T) * C + c];
#pragma unroll
    for (int j = 0; j < 3; ++j) st[j * C + c] = nv[j];
}

// Attention for rows t of a chunk (each its own position, keys up to it): block (K/V head kh, chunk c, row t), warp w
// its query head kh * 3 + w, lane l the dims l + 32 i. Scores q.k (q already over sqrt(128)) plus the relative bias
// r . proj[:, distance] for distances under 1024 (lanes 0 .. 15 add their r[l] proj[l][d] into the same warp sum).
// Each chunk's (max, sum, output) goes to part; k_attn_merge adds the chunks and the sink.
constexpr int kAttnChunks = 64, kMinChunk = 32, kMinChunk3 = 64;
constexpr int kLsaR = 32, kLsaWin = 1024, kLsaTop = 4096 /* the most pools a token may take; INKLING_LSA_TOP picks */, kLsaList = 4 + 4 * kLsaTop + kLsaWin + 8;   // (k_attn3: 64 pieces of at least 64 positions)
__global__ void __launch_bounds__(96) k_attn(const float* P, const float* proj, const kc_t* K, const kc_t* V, float* part,
                                             const int* state, int swa, int mask) {
    const int kh = blockIdx.x, c = blockIdx.y, t = blockIdx.z, w = threadIdx.x >> 5, l = threadIdx.x & 31, h = kh * 3 + w;
    const int pos = state[0] + t, lo = swa ? max(0, pos - (kWin - 1)) : 0, n = pos - lo + 1;
    const int per = max(kMinChunk, (n + kAttnChunks - 1) / kAttnChunks), a = lo + c * per, bnd = min(pos + 1, a + per);
    float q[4], o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { q[i] = P[size_t(t) * kNP + kRQ + h * kHD + l + 32 * i]; o[i] = 0.f; }
    const float rl = l < kDRel ? P[size_t(t) * kNP + kRR + h * kDRel + l] : 0.f;
    const float* pj = proj + (l & 15) * kExt;
    float m = -INFINITY, s = 0;
    for (int p = a; p < bnd; ++p) {
        const size_t r = size_t(p & mask) * kKVD + kh * kHD + l;
        const int dist = pos - p;
        float d = dist < kExt ? rl * pj[min(dist, kExt - 1)] : 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) d += q[i] * __half2float(K[r + 32 * i]);
        d = warp_sum(d);
        const float mx = fmaxf(m, d), cf = expf(m - mx), e = expf(d - mx);
        s = s * cf + e;
#pragma unroll
        for (int i = 0; i < 4; ++i) o[i] = o[i] * cf + e * __half2float(V[r + 32 * i]);
        m = mx;
    }
    float* Pp = part + ((size_t(t) * kQH + h) * kAttnChunks + c) * (kHD + 4);
    if (l == 0) { Pp[0] = m; Pp[1] = s; }
#pragma unroll
    for (int i = 0; i < 4; ++i) Pp[4 + l + 32 * i] = o[i];
}
// the same, 32 positions at a time: each lane scores one position (its K row against q in LDS, plus the bias), one
// max / sum a tile, then the V rows weighted (lanes over the head's dims). grid (kKV, kAttnChunks, T), 96 threads.
__global__ void __launch_bounds__(96) k_attn2(const float* P, const float* proj, const kc_t* K, const kc_t* V, float* part,
                                              const int* state, int swa, int mask) {
    constexpr int KP = kHD + 8;   // padded rows: lanes reading their own key row hit different banks
    __shared__ float qs[3][kHD];
    __shared__ __align__(16) __half ks[32][KP];
    __shared__ __align__(16) __half vs[32][kHD];
    const int kh = blockIdx.x, c = blockIdx.y, t = blockIdx.z, w = threadIdx.x >> 5, l = threadIdx.x & 31, h = kh * 3 + w;
    const int pos = state[0] + t, lo = swa ? max(0, pos - (kWin - 1)) : 0, n = pos - lo + 1;
    const int per = max(kMinChunk, (n + kAttnChunks - 1) / kAttnChunks), a = lo + c * per, bnd = min(pos + 1, a + per);
#pragma unroll
    for (int i = 0; i < 4; ++i) qs[w][l + 32 * i] = P[size_t(t) * kNP + kRQ + h * kHD + l + 32 * i];
    float rl[kDRel];
#pragma unroll
    for (int d = 0; d < kDRel; ++d) rl[d] = P[size_t(t) * kNP + kRR + h * kDRel + d];
    float m = -INFINITY, s = 0, o[4] = {0, 0, 0, 0};
    for (int p0 = a; p0 < bnd; p0 += 32) {
        const int cnt = min(32, bnd - p0);
        __syncthreads();   // (the last tile's reads done; qs written)
        // the tile's K and V rows into LDS: 32 rows x 16 pieces of 8 each, all loads issued together
#pragma unroll
        for (int j = 0; j < (32 * kHD / 8 + 95) / 96; ++j) {
            const int i = threadIdx.x + 96 * j;
            if (i < 32 * kHD / 8) {
                const int r = i >> 4, cc = (i & 15) * 8;
                if (r < cnt) {
                    const size_t base = size_t((p0 + r) & mask) * kKVD + kh * kHD + cc;
                    *(uint4*) &ks[r][cc] = *(const uint4*) (K + base);
                    *(uint4*) &vs[r][cc] = *(const uint4*) (V + base);
                }
            }
        }
        __syncthreads();
        const int p = p0 + l;
        const bool ok = l < cnt;
        float d = -INFINITY;
        if (ok) {
            float acc = 0;
#pragma unroll
            for (int i = 0; i < kHD / 8; ++i) {
                const uint4 kv = *(const uint4*) &ks[l][i * 8];
                const __half2* k2 = (const __half2*) &kv;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __half22float2(k2[j]);
                    acc += f.x * qs[w][i * 8 + 2 * j] + f.y * qs[w][i * 8 + 2 * j + 1];
                }
            }
            const int dist = pos - p;
            if (dist < kExt) {
                float b = 0;
#pragma unroll
                for (int dd = 0; dd < kDRel; ++dd) b += rl[dd] * proj[dd * kExt + dist];
                acc += b;
            }
            d = acc;
        }
        const float mx = fmaxf(m, warp_max(d)), cf = expf(m - mx), e = ok ? expf(d - mx) : 0.f;
        s = s * cf + warp_sum(e);
#pragma unroll
        for (int i = 0; i < 4; ++i) o[i] *= cf;
        for (int j = 0; j < cnt; ++j) {
            const float ej = __shfl(e, j, 32);
#pragma unroll
            for (int i = 0; i < 4; ++i) o[i] += ej * __half2float(vs[j][l + 32 * i]);
        }
        m = mx;
    }
    float* Pp = part + ((size_t(t) * kQH + h) * kAttnChunks + c) * (kHD + 4);
    if (l == 0) { Pp[0] = m; Pp[1] = s; }
#pragma unroll
    for (int i = 0; i < 4; ++i) Pp[4 + l + 32 * i] = o[i];
}
// Writing-path attention, one block a (K/V head, piece of the positions): 256 positions a tile scored at once (a
// thread a position, its K row against the 3 query heads, plus the bias near the end), one max / sum a tile, then the
// V rows weighted (a thread a dimension, two halves of the tile). K / V fp16, or (Q8) int8 with an fp16 scale a row and
// head. Partial results per head as k_attn's (m, l, o). grid (kKV, kAttnChunks, T), 256 threads.
template <bool Q8>
__global__ void __launch_bounds__(256) k_attn3(const float* P, const float* proj, const void* Kc, const void* Vc, const __half* KS,
                                               const __half* VS, float* part, const int* state, int swa, int mask,
                                               const int* lst = nullptr, const int* lcnt = nullptr,
                                               const SlotRow* tab = nullptr, const int* rowslot = nullptr) {
    __shared__ float qs[3][kHD];
    __shared__ float rs[3][kDRel];
    __shared__ float ps[3][256];
    __shared__ float red[8][6];
    __shared__ float ored[8][3][kHD];
    const int nch = gridDim.y;
    const int kh = blockIdx.x, c = blockIdx.y, t = blockIdx.z, tid = threadIdx.x, w = tid >> 5, l = tid & 31;
    if (tab) {
        const SlotRow& s = tab[size_t(rowslot[t]) * kVisits];
        state = s.state; KS = s.KS; VS = s.VS;
        Kc = Q8 ? (const void*) s.K8 : (const void*) s.K;
        Vc = Q8 ? (const void*) s.V8 : (const void*) s.V;
    }
    const int pos = state[0] + (tab ? 0 : t);
    // LSA: list entries 0 .. lc - 1 instead of positions lo .. pos
    const int lc = lcnt ? lcnt[kh] : -1;
    const int* L = lc >= 0 ? lst + size_t(kh) * kLsaList : nullptr;
    const int lo = L ? 0 : (swa ? max(0, pos - (kWin - 1)) : 0), n = L ? lc : pos - lo + 1, top = L ? lc : pos + 1;
    const int per = max(kMinChunk3, (n + nch - 1) / nch), a = lo + c * per, bnd = min(top, a + per);
    if (a >= bnd) return;   // (an empty piece: k_attn_merge3 reads only the pieces that hold positions)
    for (int i = tid; i < 3 * kHD; i += 256) qs[i / kHD][i % kHD] = P[size_t(t) * kNP + kRQ + kh * 3 * kHD + i];
    if (tid < 3 * kDRel) rs[tid / kDRel][tid % kDRel] = P[size_t(t) * kNP + kRR + kh * 3 * kDRel + tid];
    __syncthreads();
    float m[3] = {-INFINITY, -INFINITY, -INFINITY}, ls[3] = {0, 0, 0};
    float o[3][4] = {};
    for (int p0 = a; p0 < bnd; p0 += 256) {
        const int cnt = min(256, bnd - p0);
        float s0 = -INFINITY, s1 = -INFINITY, s2 = -INFINITY, vsc = 1.f;
        if (tid < cnt) {
            const int p = L ? L[p0 + tid] : p0 + tid;
            const size_t row = size_t(p & mask);
            float a0 = 0, a1 = 0, a2 = 0;
            if (Q8) {
                const uint4* kr = (const uint4*) ((const int8_t*) Kc + row * kKVD + kh * kHD);
#pragma unroll 2
                for (int i = 0; i < kHD / 16; ++i) {
                    const uint4 u = kr[i];
                    const uint32_t uu[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        const float kv = float(int8_t(uu[j >> 2] >> (8 * (j & 3))));
                        a0 += kv * qs[0][i * 16 + j]; a1 += kv * qs[1][i * 16 + j]; a2 += kv * qs[2][i * 16 + j];
                    }
                }
                const float ks = __half2float(KS[row * kKV + kh]);
                a0 *= ks; a1 *= ks; a2 *= ks;
                vsc = __half2float(VS[row * kKV + kh]);
            } else {
                const uint4* kr = (const uint4*) ((const __half*) Kc + row * kKVD + kh * kHD);
#pragma unroll 2
                for (int i = 0; i < kHD / 8; ++i) {
                    const uint4 u = kr[i];
                    const __half2* h2 = (const __half2*) &u;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = __half22float2(h2[j]);
                        const int dd = i * 8 + 2 * j;
                        a0 += f.x * qs[0][dd] + f.y * qs[0][dd + 1];
                        a1 += f.x * qs[1][dd] + f.y * qs[1][dd + 1];
                        a2 += f.x * qs[2][dd] + f.y * qs[2][dd + 1];
                    }
                }
            }
            const int dist = pos - p;
            if (dist < kExt)
                for (int dd = 0; dd < kDRel; ++dd) {
                    const float pj = proj[dd * kExt + dist];
                    a0 += rs[0][dd] * pj; a1 += rs[1][dd] * pj; a2 += rs[2][dd] * pj;
                }
            s0 = a0; s1 = a1; s2 = a2;
        }
        {
            const float v0 = warp_max(s0), v1 = warp_max(s1), v2 = warp_max(s2);
            if (l == 0) { red[w][0] = v0; red[w][1] = v1; red[w][2] = v2; }
        }
        __syncthreads();
        float cf[3];
        const float sv[3] = {s0, s1, s2};
#pragma unroll
        for (int hh = 0; hh < 3; ++hh) {
            float tm = red[0][hh];
            for (int ww = 1; ww < 8; ++ww) tm = fmaxf(tm, red[ww][hh]);
            const float mn = fmaxf(m[hh], tm);
            cf[hh] = expf(m[hh] - mn);
            m[hh] = mn;
            const float e = tid < cnt ? expf(sv[hh] - mn) : 0.f;
            ps[hh][tid] = e * vsc;
            const float su = warp_sum(e);
            if (l == 0) red[w][3 + hh] = su;
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < 3; ++hh) {
            float su = 0;
            for (int ww = 0; ww < 8; ++ww) su += red[ww][3 + hh];
            ls[hh] = ls[hh] * cf[hh] + su;
#pragma unroll
            for (int k = 0; k < 4; ++k) o[hh][k] *= cf[hh];
        }
        // V: warp w the positions w, w + 8, ..; lane l the dims 4 l .. 4 l + 3
#pragma unroll 4
        for (int j = w; j < cnt; j += 8) {
            const size_t row = size_t((L ? L[p0 + j] : p0 + j) & mask) * kKVD + kh * kHD + 4 * l;
            float v[4];
            if (Q8) {
                const uint32_t u = *(const uint32_t*) ((const int8_t*) Vc + row);
#pragma unroll
                for (int k = 0; k < 4; ++k) v[k] = float(int8_t(u >> (8 * k)));
            } else {
                const uint2 u = *(const uint2*) ((const __half*) Vc + row);
                const float2 f0 = __half22float2(*(const __half2*) &u.x), f1 = __half22float2(*(const __half2*) &u.y);
                v[0] = f0.x; v[1] = f0.y; v[2] = f1.x; v[3] = f1.y;
            }
            const float p0v = ps[0][j], p1v = ps[1][j], p2v = ps[2][j];
#pragma unroll
            for (int k = 0; k < 4; ++k) { o[0][k] += p0v * v[k]; o[1][k] += p1v * v[k]; o[2][k] += p2v * v[k]; }
        }
        __syncthreads();
    }
#pragma unroll
    for (int hh = 0; hh < 3; ++hh)
#pragma unroll
        for (int k = 0; k < 4; ++k) ored[w][hh][4 * l + k] = o[hh][k];
    __syncthreads();
    for (int i = tid; i < 3 * kHD; i += 256) {
        const int hh = i / kHD, d = i % kHD;
        float su = 0;
        for (int ww = 0; ww < 8; ++ww) su += ored[ww][hh][d];
        float* Pp = part + ((size_t(t) * kQH + kh * 3 + hh) * kAttnChunks + c) * (kHD + 4);
        Pp[4 + d] = su;
        if (d == 0) { Pp[0] = m[hh]; Pp[1] = ls[hh]; }
    }
}
// ---- LSA (latent sparse attention) for the writing path's global passes: an index of 4-position pools (the mean
// key, projected to kLsaR dims by a PCA map of the model's own keys), the best kLsaTop pools by max over the K/V head's
// 3 queries, plus sinks (pool 0) and every position from the last pool before the window on (the window covers the
// relative bias's 1024). The attention then runs on that position list.
// the pools rows p0 .. p0 + T - 1 touch: their mean keys projected; grid (T / 4 + 2, kKV), 128 threads
__global__ void __launch_bounds__(128) k_lsa_index(const int8_t* K8, const __half* KS, const float* U, __half* IDX, const int* state, int T) {
    __shared__ float kb[kHD];
    const int kh = blockIdx.y, d = threadIdx.x, p0 = state[0];
    const int pool = p0 / 4 + blockIdx.x, last = p0 + T - 1;
    if (pool * 4 > last) return;
    const int pe = min(pool * 4 + 3, last);
    float a = 0;
    for (int p = pool * 4; p <= pe; ++p) a += float(K8[size_t(p) * kKVD + kh * kHD + d]) * __half2float(KS[size_t(p) * kKV + kh]);
    kb[d] = a / float(pe - pool * 4 + 1);
    __syncthreads();
    if (d < kLsaR) {
        const float* u = U + size_t(kh) * kHD * kLsaR + d;
        float r = 0;
        for (int i = 0; i < kHD; ++i) r += kb[i] * u[i * kLsaR];
        IDX[(size_t(pool) * kKV + kh) * kLsaR + d] = __float2half(r);
    }
}
__device__ __forceinline__ unsigned lsa_key(float s) {
    const unsigned u = __float_as_uint(s);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// the position list of one K/V head (one row, the writing path); lcnt -1: full attention (history up to lsa_min)
__global__ void __launch_bounds__(1024) k_lsa_select(const float* P, const float* U, const __half* IDX, float* SC, int maxp, int* lst,
                                                     int* lcnt, const int* state, int lsa_min, int top) {
    __shared__ float qs[3][kHD];
    __shared__ float qp[3][kLsaR];
    __shared__ unsigned hist[4096];
    __shared__ unsigned wsum[32];
    __shared__ int thr, need, nsel, neq;
    const int kh = blockIdx.x, tid = threadIdx.x, pos = state[0];
    if (pos + 1 <= lsa_min) { if (tid == 0) lcnt[kh] = -1; return; }
    const int ws = pos - kLsaWin + 1, np = ws / 4;   // pools 1 .. np - 1 are candidates (pool 0: the sinks)
    for (int i = tid; i < 3 * kHD; i += 1024) qs[i / kHD][i % kHD] = P[kRQ + kh * 3 * kHD + i];
    for (int i = tid; i < 4096; i += 1024) hist[i] = 0;
    if (tid == 0) { nsel = 0; neq = 0; }
    __syncthreads();
    if (tid < 3 * kLsaR) {
        const int hh = tid / kLsaR, r = tid % kLsaR;
        const float* u = U + size_t(kh) * kHD * kLsaR + r;
        float a = 0;
        for (int i = 0; i < kHD; ++i) a += qs[hh][i] * u[i * kLsaR];
        qp[hh][r] = a;
    }
    __syncthreads();
    float* sc = SC + size_t(kh) * maxp;
    for (int pp = 1 + tid; pp < np; pp += 1024) {
        const __half2* x = (const __half2*) (IDX + (size_t(pp) * kKV + kh) * kLsaR);
        float a0 = 0, a1 = 0, a2 = 0;
#pragma unroll
        for (int r = 0; r < kLsaR / 2; ++r) {
            const float2 f = __half22float2(x[r]);
            a0 += f.x * qp[0][2 * r] + f.y * qp[0][2 * r + 1];
            a1 += f.x * qp[1][2 * r] + f.y * qp[1][2 * r + 1];
            a2 += f.x * qp[2][2 * r] + f.y * qp[2][2 * r + 1];
        }
        const float s = fmaxf(a0, fmaxf(a1, a2));
        sc[pp] = s;
        atomicAdd(&hist[lsa_key(s) >> 20], 1u);
    }
    __syncthreads();
    // the threshold bin: suffix counts from the top (thread t owns bins 4t .. 4t + 3), all in parallel
    // top 0: grows with the history (a 48th of its pools, 1024 at least): 1024 pools found 5-6 of 8 facts at 385K, 2048 all 8
    const int ncand = max(0, np - 1), K = min(min(top > 0 ? top : max(1024, ncand / 48), kLsaTop), ncand);
    unsigned c4[4], mine = 0;
    for (int j = 0; j < 4; ++j) { c4[j] = hist[4 * tid + j]; mine += c4[j]; }
    // exclusive suffix sum of `mine` over threads above
    const int w = tid >> 5, l = tid & 31;
    unsigned inc = mine;   // inclusive suffix within the warp (lanes l .. 31)
    for (int o = 1; o < 32; o <<= 1) { const unsigned v = __shfl_down(inc, o, 32); if (l + o < 32) inc += v; }
    if (l == 0) wsum[w] = inc;
    __syncthreads();
    unsigned above = inc - mine;
    for (int ww = w + 1; ww < 32; ++ww) above += wsum[ww];
    if (K > 0 && above < unsigned(K) && above + mine >= unsigned(K)) {
        unsigned acc = above;
        for (int j = 3; j >= 0; --j) {
            if (acc + c4[j] >= unsigned(K)) { thr = 4 * tid + j; need = K - int(acc); break; }
            acc += c4[j];
        }
    }
    if (K == 0 && tid == 0) { thr = 4096; need = 0; }
    __syncthreads();
    int* L = lst + size_t(kh) * kLsaList;
    if (tid < 4) L[tid] = tid;   // the sinks
    for (int pp = 1 + tid; pp < np; pp += 1024) {
        const int b = int(lsa_key(sc[pp]) >> 20);
        int slot = -1;
        if (b > thr) slot = atomicAdd(&nsel, 1);
        else if (b == thr && atomicAdd(&neq, 1) < need) slot = atomicAdd(&nsel, 1);
        if (slot >= 0) for (int i = 0; i < 4; ++i) L[4 + 4 * slot + i] = 4 * pp + i;
    }
    __syncthreads();
    const int base = 4 + 4 * nsel, w0 = max(4, 4 * np), nw = pos - w0 + 1;
    for (int j = tid; j < nw; j += 1024) L[base + j] = w0 + j;
    if (tid == 0) lcnt[kh] = base + nw;
}
// k_attn_merge for k_attn3: only the pieces that hold positions (the same split as k_attn3's)
__global__ void __launch_bounds__(128) k_attn_merge3(const float* part, const float* P, float* out, uint16_t* o16, int nch,
                                                     const int* state, int swa, const int* lcnt = nullptr,
                                                     const SlotRow* tab = nullptr, const int* rs = nullptr) {
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int lc = lcnt ? lcnt[h / 3] : -1;
    if (tab) state = tab[size_t(rs[t]) * kVisits].state;
    const int pos = state[0] + (tab ? 0 : t), lo = swa ? max(0, pos - (kWin - 1)) : 0, n = lc >= 0 ? lc : pos - lo + 1;
    const int per = max(kMinChunk3, (n + nch - 1) / nch), used = (n + per - 1) / per;
    const float* Pp = part + (size_t(t) * kQH + h) * kAttnChunks * (kHD + 4);
    const float sink = P[size_t(t) * kNP + kRS + h];
    float M = sink;
    for (int c = 0; c < used; ++c) M = fmaxf(M, Pp[c * (kHD + 4)]);
    float L = expf(sink - M), O = 0;
    for (int c = 0; c < used; ++c) {
        const float f = expf(Pp[c * (kHD + 4)] - M);
        L += Pp[c * (kHD + 4) + 1] * f;
        O += Pp[c * (kHD + 4) + 4 + i] * f;
    }
    const float g = sigm(P[size_t(t) * kNP + kRG + h * kHD + i]);
    const float r = O / L * g;
    out[size_t(t) * kQD + h * kHD + i] = r;
    if (o16) o16[size_t(t) * kQD + h * kHD + i] = __half_as_ushort(__float2half(r));
}
// out[t][h] = (sum of the chunks) / (their sums + e^sink), times sigmoid(gate); grid (24, T), 128 threads
__global__ void __launch_bounds__(128) k_attn_merge(const float* part, const float* P, float* out, uint16_t* o16, int nch = kAttnChunks) {
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const float* Pp = part + (size_t(t) * kQH + h) * kAttnChunks * (kHD + 4);
    const float sink = P[size_t(t) * kNP + kRS + h];
    float M = sink;
    for (int c = 0; c < nch; ++c) M = fmaxf(M, Pp[c * (kHD + 4)]);
    float L = expf(sink - M), O = 0;
    for (int c = 0; c < nch; ++c) {
        const float mc = Pp[c * (kHD + 4)];
        if (mc == -INFINITY) continue;
        const float f = expf(mc - M);
        L += Pp[c * (kHD + 4) + 1] * f;
        O += Pp[c * (kHD + 4) + 4 + i] * f;
    }
    const float g = sigm(P[size_t(t) * kNP + kRG + h * kHD + i]);
    const float r = O / L * g;
    out[size_t(t) * kQD + h * kHD + i] = r;
    if (o16) o16[size_t(t) * kQD + h * kHD + i] = __half_as_ushort(__float2half(r));
}

// Router: logits = x . R (fp32), scores sigmoid; the top 8 of score + bias (ties: the lower index); weights = their scores
// renormalised, times 8. grid T, 256 threads (8 warps of 16 experts).
__global__ void __launch_bounds__(256) k_router(const float* x, const float* R, const float* bias, int* idx, float* wts) {
    __shared__ float xs[kH];
    __shared__ float sc[kE];
    const size_t t = blockIdx.x;
    for (int i = threadIdx.x; i < kH; i += 256) xs[i] = x[t * kH + i];
    __syncthreads();
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    for (int e = w; e < kE; e += 8) {
        const float* r = R + size_t(e) * kH;
        float a = 0;
        for (int i = l; i < kH; i += 32) a += r[i] * xs[i];
        a = warp_sum(a);
        if (l == 0) sc[e] = sigm(a);
    }
    __syncthreads();
    if (threadIdx.x >= 32) return;
    float v[4], s[4];
    for (int j = 0; j < 4; ++j) { s[j] = sc[l * 4 + j]; v[j] = s[j] + bias[l * 4 + j]; }
    float pk[kTopK];
    int pi[kTopK];
    for (int k = 0; k < kTopK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int j = 0; j < 4; ++j) if (v[j] > bv) { bv = v[j]; bi = l * 4 + j; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(bv, o, 32);
            const int oi = __shfl_xor(bi, o, 32);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (l == bi / 4) v[bi % 4] = -INFINITY;
        pi[k] = bi;
        pk[k] = sc[bi];
    }
    float tot = 0;
    for (int k = 0; k < kTopK; ++k) tot += pk[k];
    if (l < kTopK) { idx[t * kTopK + l] = pi[l]; wts[t * kTopK + l] = pk[l] / (tot + 1e-20f) * kRouteScale; }
}

// the same, a warp an expert over kE / 8 blocks a row (grid (kE / 8, T)); the last block of a row to finish picks.
// sc: [T][kE] scratch, done: a zeroed counter a row (left zeroed)
__global__ void __launch_bounds__(256) k_router2(const float* x, const float* R, const float* bias, float* scb, unsigned* done,
                                                 int* idx, float* wts, const float* nw = nullptr) {
    __shared__ bool last;
    const int t = blockIdx.y, w = threadIdx.x >> 5, l = threadIdx.x & 31, e = blockIdx.x * 8 + w;
    const float4* xr = (const float4*) (x + size_t(t) * kH);
    const float4* r = (const float4*) (R + size_t(e) * kH);
    float a = 0, rn = 1.f;
    if (nw) {   // x rms-normed with nw on the way in
        const float4* nr = (const float4*) nw;
        float ss = 0;
        for (int i = threadIdx.x; i < kH / 4; i += 256) { const float4 q = xr[i]; ss += q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w; }
        rn = rsqrtf(block_sum(ss) / kH + kEps);
#pragma unroll 6
        for (int i = l; i < kH / 4; i += 32) {
            const float4 p = r[i], q = xr[i], g = nr[i];
            a += p.x * q.x * g.x + p.y * q.y * g.y + p.z * q.z * g.z + p.w * q.w * g.w;
        }
    } else {
#pragma unroll 6
        for (int i = l; i < kH / 4; i += 32) {
            const float4 p = r[i], q = xr[i];
            a += p.x * q.x + p.y * q.y + p.z * q.z + p.w * q.w;
        }
    }
    a = warp_sum(a) * rn;
    float* sc = scb + size_t(t) * kE;
    if (l == 0) sc[e] = sigm(a);
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        last = atomicAdd(done + t, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last || threadIdx.x >= 32) return;
    __threadfence();
    const volatile float* vs = sc;
    float v[4], s[4];
    for (int j = 0; j < 4; ++j) { s[j] = vs[l * 4 + j]; v[j] = s[j] + bias[l * 4 + j]; }
    float pk[kTopK];
    int pi[kTopK];
    for (int k = 0; k < kTopK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff, bj = 0;
        for (int j = 0; j < 4; ++j) if (v[j] > bv) { bv = v[j]; bi = l * 4 + j; bj = j; }
        float bs = s[bj];
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(bv, o, 32), os = __shfl_xor(bs, o, 32);
            const int oi = __shfl_xor(bi, o, 32);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; bs = os; }
        }
        if (l == bi / 4) v[bi % 4] = -INFINITY;
        pi[k] = bi;
        pk[k] = bs;
    }
    float tot = 0;
    for (int k = 0; k < kTopK; ++k) tot += pk[k];
    if (l < kTopK) { idx[t * kTopK + l] = pi[l]; wts[t * kTopK + l] = pk[l] / (tot + 1e-20f) * kRouteScale; }
    if (l == 0) done[t] = 0;
}

__global__ void __launch_bounds__(256) k_router3(const float* x, const float* R, const float* bias, float* scb, unsigned* done,
                                                 int* idx, float* wts, const float* nw = nullptr) {
    __shared__ bool last;
    // (k_router2 with two warps an expert, each half its row: 32 blocks of 4 experts instead of 16 of 8)
    __shared__ float part[8];
    const int t = blockIdx.y, w = threadIdx.x >> 5, l = threadIdx.x & 31, e = blockIdx.x * 4 + (w >> 1), hf = w & 1;
    const float4* xr = (const float4*) (x + size_t(t) * kH) + hf * (kH / 8);
    const float4* r = (const float4*) (R + size_t(e) * kH) + hf * (kH / 8);
    float a = 0, rn = 1.f;
    if (nw) {   // x rms-normed with nw on the way in
        const float4* nr = (const float4*) nw + hf * (kH / 8);
        const float4* xa = (const float4*) (x + size_t(t) * kH);
        float ss = 0;
        for (int i = threadIdx.x; i < kH / 4; i += 256) { const float4 q = xa[i]; ss += q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w; }
        rn = rsqrtf(block_sum(ss) / kH + kEps);
#pragma unroll
        for (int i = l; i < kH / 8; i += 32) {
            const float4 p = r[i], q = xr[i], g = nr[i];
            a += p.x * q.x * g.x + p.y * q.y * g.y + p.z * q.z * g.z + p.w * q.w * g.w;
        }
    } else {
#pragma unroll
        for (int i = l; i < kH / 8; i += 32) {
            const float4 p = r[i], q = xr[i];
            a += p.x * q.x + p.y * q.y + p.z * q.z + p.w * q.w;
        }
    }
    a = warp_sum(a);
    if (l == 0) part[w] = a;
    __syncthreads();
    float* sc = scb + size_t(t) * kE;
    if (threadIdx.x < 4) sc[blockIdx.x * 4 + threadIdx.x] = sigm((part[2 * threadIdx.x] + part[2 * threadIdx.x + 1]) * rn);
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        last = atomicAdd(done + t, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last || threadIdx.x >= 32) return;
    __threadfence();
    const volatile float* vs = sc;
    float v[4], s[4];
    for (int j = 0; j < 4; ++j) { s[j] = vs[l * 4 + j]; v[j] = s[j] + bias[l * 4 + j]; }
    float pk[kTopK];
    int pi[kTopK];
    for (int k = 0; k < kTopK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff, bj = 0;
        for (int j = 0; j < 4; ++j) if (v[j] > bv) { bv = v[j]; bi = l * 4 + j; bj = j; }
        float bs = s[bj];
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(bv, o, 32), os = __shfl_xor(bs, o, 32);
            const int oi = __shfl_xor(bi, o, 32);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; bs = os; }
        }
        if (l == bi / 4) v[bi % 4] = -INFINITY;
        pi[k] = bi;
        pk[k] = bs;
    }
    float tot = 0;
    for (int k = 0; k < kTopK; ++k) tot += pk[k];
    if (l < kTopK) { idx[t * kTopK + l] = pi[l]; wts[t * kTopK + l] = pk[l] / (tot + 1e-20f) * kRouteScale; }
    if (l == 0) done[t] = 0;
}

// The chosen experts' gate / up for row t: act[t][k][j] = softcap(g, 4) sigmoid(g) softcap(u, 25), g / u the dot of
// rows 2j / 2j + 1 (as stored: interleaved) with x[t]. grid (kF / 32, kTopK, T), 8 warps of 4 outputs.
__global__ void __launch_bounds__(256) k_moe_gu(Experts ex, const float* x, const int* idx, float* act) {
    __shared__ float xs[kH];
    const int t = blockIdx.z, k = blockIdx.y, e = idx[t * kTopK + k];
    for (int i = threadIdx.x; i < kH; i += 256) xs[i] = x[size_t(t) * kH + i];
    __syncthreads();
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    for (int jj = 0; jj < 4; ++jj) {
        const int j = blockIdx.x * 32 + w * 4 + jj;
        float gu[2];
        for (int s = 0; s < 2; ++s) {
            const size_t row = size_t(e) * 2 * kF + 2 * j + s;
            float a = 0;
            if (ex.nv) {
                const uint8_t* wr = ex.w13 + row * ex.w13_row;
                const uint8_t* sr = ex.s13 + row * ex.s13_row;
                for (int blk = l; blk < kH / 16; blk += 32) {
                    const float sc = e4m3(sr[blk]);
                    float b = 0;
                    for (int i = 0; i < 8; ++i) {
                        const uint32_t by = wr[blk * 8 + i];
                        b += c_e2m1[by & 15] * xs[blk * 16 + 2 * i] + c_e2m1[by >> 4] * xs[blk * 16 + 2 * i + 1];
                    }
                    a += sc * b;
                }
            } else {
                const uint8_t* wr = ex.w13 + row * ex.w13_row;
                const uint8_t* sr = ex.s13 + row * ex.s13_row;
                for (int blk = l; blk < kH / 32; blk += 32) {
                    const float sc = ldexpf(1.f, int(sr[blk]) - 127);
                    float b = 0;
                    for (int i = 0; i < 32; ++i) b += e4m3(wr[blk * 32 + i]) * xs[blk * 32 + i];
                    a += sc * b;
                }
            }
            a = warp_sum(a);
            gu[s] = ex.nv ? a * ex.f13[e] : a;
        }
        if (l == 0) act[(size_t(t) * kTopK + k) * kF + j] = softcapf(gu[0], kGCap) * sigm(gu[0]) * softcapf(gu[1], kUCap);
    }
}
// y[t][c] = sum_k wts[k] * (row c of expert k's down) . act[t][k]; grid (kH / 32, T), 8 warps of 4 rows
__global__ void __launch_bounds__(256) k_moe_down(Experts ex, const float* act, const int* idx, const float* wts, float* y) {
    __shared__ float as[kTopK * kF];
    const int t = blockIdx.y;
    for (int i = threadIdx.x; i < kTopK * kF; i += 256) as[i] = act[size_t(t) * kTopK * kF + i];
    __syncthreads();
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    for (int rr = 0; rr < 4; ++rr) {
        const int c = blockIdx.x * 32 + w * 4 + rr;
        float tot = 0;
        for (int k = 0; k < kTopK; ++k) {
            const int e = idx[t * kTopK + k];
            const size_t row = size_t(e) * kH + c;
            const float* xa = as + k * kF;
            float a = 0;
            if (ex.nv) {
                const uint8_t* wr = ex.w2 + row * ex.w2_row;
                const uint8_t* sr = ex.s2 + row * ex.s2_row;
                for (int blk = l; blk < kF / 16; blk += 32) {
                    const float sc = e4m3(sr[blk]);
                    float b = 0;
                    for (int i = 0; i < 8; ++i) {
                        const uint32_t by = wr[blk * 8 + i];
                        b += c_e2m1[by & 15] * xa[blk * 16 + 2 * i] + c_e2m1[by >> 4] * xa[blk * 16 + 2 * i + 1];
                    }
                    a += sc * b;
                }
            } else {
                const uint8_t* wr = ex.w2 + row * ex.w2_row;
                const uint8_t* sr = ex.s2 + row * ex.s2_row;
                for (int blk = l; blk < kF / 32; blk += 32) {
                    const float sc = ldexpf(1.f, int(sr[blk]) - 127);
                    float b = 0;
                    for (int i = 0; i < 32; ++i) b += e4m3(wr[blk * 32 + i]) * xa[blk * 32 + i];
                    a += sc * b;
                }
            }
            a = warp_sum(a);
            tot += wts[t * kTopK + k] * (ex.nv ? a * ex.f2[e] : a);
        }
        if (l == 0) y[size_t(t) * kH + c] = tot;
    }
}
// ---- the writing path's experts: 16 bytes a lane at a time (NVFP4: 32 values and 2 scales; MXFP8: 16 values, a
// scale), unpacked to fp16 by byte permutes (exact) and summed in fp32 against fp32 inputs in LDS
__device__ __forceinline__ float e4m3f(uint32_t b) {   // exact, sub-normals included
    const float v = __half2float(__ushort_as_half(uint16_t(((b & 0x7f) << 7) | ((b & 0x80) << 8))));
    return v * 256.f;
}
// one 16-byte piece of a row: its dot with x (the piece's place k0 in values), scales applied
template <bool NV>
__device__ __forceinline__ float ex_piece(const uint8_t* wrow, const uint8_t* srow, int u, const float* x) {
    const uint4 w = *(const uint4*) (wrow + size_t(u) * 16);
    if (NV) {
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
        float a0 = 0, a1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                const uint32_t t = (ww[j] >> (16 * hf)) & 0xffff;
                const uint32_t sp = (t & 0xf) | ((t & 0xf0) << 4) | ((t & 0xf00) << 8) | ((t & 0xf000) << 12);
                const uint32_t hb = __builtin_amdgcn_perm(0x46444240u, 0x3e3c3800u, sp & 0x07070707u) | ((sp & 0x08080808u) << 4);
                const float2 f0 = __half22float2(__builtin_bit_cast(__half2, __builtin_amdgcn_perm(0u, hb, 0x010c000cu)));
                const float2 f1 = __half22float2(__builtin_bit_cast(__half2, __builtin_amdgcn_perm(0u, hb, 0x030c020cu)));
                const float* xx = x + j * 8 + hf * 4;
                const float d = f0.x * xx[0] + f0.y * xx[1] + f1.x * xx[2] + f1.y * xx[3];
                if (j < 2) a0 += d; else a1 += d;
            }
        const uint16_t sc = *(const uint16_t*) (srow + u * 2);
        return a0 * e4m3f(sc & 0xff) + a1 * e4m3f(sc >> 8);
    } else {
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
        float a = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int b = 0; b < 4; ++b) a += e4m3f((ww[j] >> (8 * b)) & 0xff) * x[j * 4 + b];
        return a * ldexpf(1.f, int(srow[u >> 1]) - 127);
    }
}
// gate / up for one row: a warp does 2 outputs (4 stored rows: g0 u0 g1 u1), each lane 16-byte pieces spread over them;
// a block 8 warps = 16 outputs; grid (kF / 16, kTopK, T)
template <bool NV> struct XPad { static constexpr int P = NV ? 32 : 16, S = P + 4; };
template <bool NV> __device__ __forceinline__ int xpad(int i) { return i + (i / XPad<NV>::P) * 4; }
template <bool NV>
__global__ void __launch_bounds__(256) k_moe_gu2(Experts ex, const float* x, const int* idx, float* act, const float* nw = nullptr) {
    __shared__ float xs[kH / XPad<NV>::P * XPad<NV>::S];
    const int t = blockIdx.z, k = blockIdx.y, e = idx[t * kTopK + k];
    float ss = 0;
    for (int i = threadIdx.x; i < kH; i += 256) { const float v = x[size_t(t) * kH + i]; xs[xpad<NV>(i)] = v; ss += v * v; }
    if (nw) {
        const float r = rsqrtf(block_sum(ss) / kH + kEps);
        for (int i = threadIdx.x; i < kH; i += 256) xs[xpad<NV>(i)] *= r * nw[i];
    }
    __syncthreads();
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    constexpr int PR = NV ? kH / 32 : kH / 16;   // pieces a row (72 / 144)
    const int j0 = blockIdx.x * 16 + w * 2;
    const size_t row0 = size_t(e) * 2 * kF + 2 * j0;   // 4 consecutive stored rows
    const uint8_t* W = ex.w13 + row0 * ex.w13_row;
    const uint8_t* S = ex.s13 + row0 * ex.s13_row;
    float a[4] = {0, 0, 0, 0};
#pragma unroll
    for (int it = 0; it < (4 * PR) / 32; ++it) {
        const int u = l + 32 * it, r = u / PR, pc = u % PR;
        const float v = ex_piece<NV>(W + size_t(r) * ex.w13_row, S + size_t(r) * ex.s13_row, pc, xs + pc * XPad<NV>::S);
#pragma unroll
        for (int q = 0; q < 4; ++q) a[q] += r == q ? v : 0.f;
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) a[q] = warp_sum(a[q]);
    if (l < 2) {
        const float f = NV ? ex.f13[e] : 1.f;
        const float g = a[2 * l] * f, u = a[2 * l + 1] * f;
        act[(size_t(t) * kTopK + k) * kF + j0 + l] = softcapf(g, kGCap) * sigm(g) * softcapf(u, kUCap);
    }
}
// down: a warp one output row c over all 8 experts' rows (NVFP4: 8 x 36 pieces = 9 a lane); y = sum_k w_k f_e dot;
// grid (kH / 8, T)
template <bool NV, int RC = 1>
__global__ void __launch_bounds__(256) k_moe_down2(Experts ex, const float* act, const int* idx, const float* wts, float* y,
                                                   ConvEpi epi = ConvEpi{}) {
    __shared__ float as[kTopK * kF / XPad<NV>::P * XPad<NV>::S];
    __shared__ float mul[kTopK];
    __shared__ int es[kTopK];
    const int t = blockIdx.y;
    for (int i = threadIdx.x; i < kTopK * kF; i += 256) as[xpad<NV>(i)] = act[size_t(t) * kTopK * kF + i];
    if (threadIdx.x < kTopK) {
        const int e = idx[t * kTopK + threadIdx.x];
        es[threadIdx.x] = e;
        mul[threadIdx.x] = wts[t * kTopK + threadIdx.x] * (NV ? ex.f2[e] : 1.f);
    }
    __syncthreads();
    // RC output rows a warp (the act rows in LDS serve a block of 8 RC outputs: fewer blocks re-reading them)
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    constexpr int PR = NV ? kF / 32 : kF / 16;   // 36 / 72
#pragma unroll
    for (int rc = 0; rc < RC; ++rc) {
        const int c = (blockIdx.x * 8 + w) * RC + rc;
        float a = 0;
#pragma unroll
        for (int it = 0; it < (kTopK * PR) / 32; ++it) {
            const int u = l + 32 * it, k = u / PR, pc = u % PR;
            const size_t row = size_t(es[k]) * kH + c;
            a += mul[k] * ex_piece<NV>(ex.w2 + row * ex.w2_row, ex.s2 + row * ex.s2_row, pc, as + (k * PR + pc) * XPad<NV>::S);
        }
        a = warp_sum(a);
        if (l == 0) {
            y[size_t(t) * kH + c] = a;
            if (epi.out) conv_epi(epi, c, a);
        }
    }
}
// the output head over rows: logits[t][r] = softcap(W[r] . x[t], 30), padded rows -inf; grid (kVocabP / 32, T)
__global__ void __launch_bounds__(256) k_head(const __half* W, const float* x, float* logits) {
    __shared__ float xsh[kH];
    const int t = blockIdx.y, r0 = blockIdx.x * 32;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (r0 >= kVocab) {
        for (int i = threadIdx.x; i < 32; i += 256) logits[size_t(t) * kVocabP + r0 + i] = -INFINITY;
        return;
    }
    for (int i = threadIdx.x; i < kH; i += 256) xsh[i] = x[size_t(t) * kH + i];
    __syncthreads();
    float a[4] = {0, 0, 0, 0};
    for (int c = l; c < kH / 8; c += 32) {
        float xv[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) xv[i] = xsh[c * 8 + i];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = min(r0 + w * 4 + r, kVocab - 1);
            const uint4 wv = *(const uint4*) (W + size_t(row) * kH + c * 8);
            const __half2* hp = (const __half2*) &wv;
#pragma unroll
            for (int i = 0; i < 4; ++i) { const float2 f = __half22float2(hp[i]); a[r] += f.x * xv[2 * i] + f.y * xv[2 * i + 1]; }
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const float v = warp_sum(a[r]);
        const int row = r0 + w * 4 + r;
        if (l == 0) logits[size_t(t) * kVocabP + row] = row < kVocab ? softcapf(v, kOutCap) : -INFINITY;
    }
}
// the best of the logits (ties: the lower id) into state[1]; 64 blocks then the last one done
__global__ void __launch_bounds__(1024) k_argmax64(const float* lg, int n, int* state, float2* best, unsigned* done) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ bool last;
    const int per = (n + gridDim.x - 1) / gridDim.x, a = blockIdx.x * per, e = min(n, a + per);
    float v = -INFINITY;
    int i = 0;
    for (int j = a + threadIdx.x; j < e; j += 1024) if (lg[j] > v) { v = lg[j]; i = j; }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor(v, o, 32);
        const int oi = __shfl_xor(i, o, 32);
        if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = i; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 32; ++w) if (bv[w] > v || (bv[w] == v && bi[w] < i)) { v = bv[w]; i = bi[w]; }
        best[blockIdx.x] = make_float2(v, __int_as_float(i));
        __threadfence();
        last = atomicAdd(done, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last || threadIdx.x != 0) return;
    __threadfence();
    float bvv = -INFINITY;
    int bii = 0;
    for (int b = 0; b < (int) gridDim.x; ++b) {
        const float cv = ((volatile float*) best)[2 * b];
        const int ci = __float_as_int(((volatile float*) best)[2 * b + 1]);
        if (cv > bvv || (cv == bvv && ci < bii)) { bvv = cv; bii = ci; }
    }
    state[1] = bii;
    *done = 0;
}
__global__ void k_setpos(int* state, int p) { state[0] = p; }
__global__ void k_advance(int* state, int n) { state[0] += n; }
__global__ void k_tok_from_state(const int* state, int* ids) { ids[0] = state[1]; }

#include "inkling_prefill.cuh"
#include "inkling_draft.cuh"
#include "inkling_multi.cuh"

// ---------------------------------------------------------------------------------------------------- safetensors
struct StTensor { std::string dtype; std::vector<int64_t> shape; size_t off = 0, len = 0; };
struct SafeTensors {
    int fd = -1;
    const uint8_t* base = nullptr;
    size_t size = 0, data0 = 0;
    std::map<std::string, StTensor> t;
    explicit SafeTensors(const std::string& path) {
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { std::fprintf(stderr, "strata-inkling: cannot open %s\n", path.c_str()); std::exit(1); }
        struct stat sb;
        fstat(fd, &sb);
        size = size_t(sb.st_size);
        base = (const uint8_t*) mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) { std::fprintf(stderr, "strata-inkling: mmap failed\n"); std::exit(1); }
        uint64_t n = 0;
        std::memcpy(&n, base, 8);
        data0 = 8 + n;
        parse(std::string((const char*) base + 8, n));
    }
    void drop() const { madvise(const_cast<uint8_t*>(base), size, MADV_DONTNEED); }
    // the header: {"name": {"dtype": "..", "shape": [..], "data_offsets": [a, b]}, ...}
    void parse(const std::string& s) {
        size_t i = 0;
        auto ws = [&]() { while (i < s.size() && std::isspace((unsigned char) s[i])) ++i; };
        auto str = [&]() { ws(); std::string o; ++i; while (s[i] != '"') { if (s[i] == '\\') ++i; o += s[i++]; } ++i; return o; };
        auto num = [&]() { ws(); size_t e = i; while (e < s.size() && (std::isdigit((unsigned char) s[e]) || s[e] == '-')) ++e; const int64_t v = std::stoll(s.substr(i, e - i)); i = e; return v; };
        auto arr = [&]() { std::vector<int64_t> v; ws(); ++i; ws(); if (s[i] == ']') { ++i; return v; } for (;;) { v.push_back(num()); ws(); if (s[i++] == ']') break; } return v; };
        std::function<void()> skip = [&]() {   // any value
            ws();
            if (s[i] == '"') { str(); return; }
            if (s[i] == '{' || s[i] == '[') { const char cl = s[i] == '{' ? '}' : ']'; ++i; ws(); if (s[i] == cl) { ++i; return; }
                for (;;) { if (cl == '}') { str(); ws(); ++i; } skip(); ws(); if (s[i++] == cl) break; } return; }
            while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') ++i;
        };
        ws(); ++i;   // {
        for (;;) {
            ws();
            if (s[i] == '}') break;
            const std::string name = str();
            ws(); ++i;   // :
            ws();
            if (name == "__metadata__") skip();
            else {
                StTensor T;
                ++i;   // {
                for (;;) {
                    const std::string k = str();
                    ws(); ++i;
                    if (k == "dtype") T.dtype = str();
                    else if (k == "shape") T.shape = arr();
                    else if (k == "data_offsets") { auto v = arr(); T.off = size_t(v[0]); T.len = size_t(v[1] - v[0]); }
                    else skip();
                    ws();
                    if (s[i++] == '}') break;
                }
                t[name] = T;
            }
            ws();
            if (s[i] == ',') ++i;
        }
    }
    const StTensor* find(const std::string& n) const { auto it = t.find(n); return it == t.end() ? nullptr : &it->second; }
    const uint8_t* data(const StTensor& T) const { return base + data0 + T.off; }
};

inline float bf16f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }

// ---------------------------------------------------------------------------------------------------- the model
struct Model {
    std::unique_ptr<SafeTensors> st_;
    std::vector<Layer> L;
    const __half *embed = nullptr, *head = nullptr;
    const float *embed_norm, *out_norm, *out_q, *msg_q;   // msg_q: 12 x 4 x 2304 (q, k, v, mlp)
    int max_ctx, tc;                                       // context, prompt chunk (rows a pass)
    size_t loaded = 0, kv_bytes = 0, pf_bytes = 0, fp16_changed = 0;
    hipStream_t st;
    // per visit: K / V caches and the convolution states (k, v: 3 x 1024; attention out, expert out: 3 x 2304)
    kc_t *K[kVisits], *V[kVisits];
    int8_t *K8[kVisits], *V8[kVisits];
    __half *KS[kVisits], *VS[kVisits];
    bool kv8 = std::getenv("INKLING_KV8") ? std::atoi(std::getenv("INKLING_KV8")) != 0 : true;
    // LSA (INKLING_LSA=<map file>: 8 global passes x kKV x 128 x kLsaR fp32), on past INKLING_LSA_MIN (16384) positions
    bool lsa = false;
    int lsa_min = 16384, lsa_maxp = 0;
    float* LU[kVisits] = {};
    __half* LIDX[kVisits] = {};
    float* LSC = nullptr;
    int *LLST = nullptr, *LCNT = nullptr;
    int lsa_top = 0;         // pools a written token's K/V head takes; 0: by history length (INKLING_LSA_TOP, up to kLsaTop)
    bool lsa_pf = true;      // LSA on the prompt path's global passes too (INKLING_LSA_PF=0: writing only)
    int lsa_pf_top = 4096, lbcap = 0;   // pools a (64-query block, K/V head) (INKLING_LSA_PF_TOP); its list's slots
    __half* QP16 = nullptr;
    float* LBS = nullptr;
    int *LBL = nullptr, *LBC = nullptr;
    // DSpark speculation (INKLING_SPEC=1): the draft's weights, caches and buffers (inkling_draft.cuh)
    bool spec = std::getenv("INKLING_SPEC") && std::atoi(std::getenv("INKLING_SPEC")) != 0;
    bool force_pf = false, spec_feat = false, spec_rows = false;
    std::vector<__half> head_keep;
    struct DLayer { const __half *qkv, *kvc, *o, *gu, *dn; const float *in_ln, *post_ln, *dfl, *qn, *kn, *knc; };
    DLayer DLy[kDL];
    __half *DK[kDL], *DV[kDL];
    int dmask[kDL];
    const float *d_alpha = nullptr, *d_mask = nullptr, *d_fnorm = nullptr, *d_mdfl = nullptr, *d_rb = nullptr, *d_ws = nullptr, *d_fbias = nullptr;
    const __half *d_wrc = nullptr, *d_w1 = nullptr;
    W8 d_fold{};
    float *FEAT = nullptr, *DX = nullptr, *DRES = nullptr, *DN = nullptr, *DQKV = nullptr, *DQ = nullptr, *DOUT = nullptr, *DGU = nullptr;
    float *DHID = nullptr, *DKVC = nullptr, *DPART = nullptr, *DCTX = nullptr, *DMS = nullptr, *DLAT = nullptr, *DBASE = nullptr;
    float *DLOG = nullptr, *DEMB0 = nullptr, *DFR = nullptr, *SPST = nullptr;
    uint16_t *DH16 = nullptr, *DO16 = nullptr, *DA16 = nullptr, *DC16 = nullptr, *DMK16 = nullptr;
    int *DTOK = nullptr, *DGT = nullptr, *DID = nullptr, *DACC = nullptr, *DOUTT = nullptr, *hout = nullptr;
    float** CSP = nullptr;   // the conv states, [visit][4] (k_conv_state_put_all)
    int feat_last = 0;       // the FEAT row of the last token whose features are known
    MoeList* MLIST = nullptr;   // a few rows at once (inkling_multi.cuh): the experts' list and their outputs
    float* YP = nullptr;
    uint16_t* XP16 = nullptr;
    bool multi = !std::getenv("INKLING_MULTI") || std::atoi(std::getenv("INKLING_MULTI")) != 0;
    int kv_mask[kVisits];
    float *cs_k[kVisits], *cs_v[kVisits], *cs_a[kVisits], *cs_m[kVisits];
    // row buffers ([tc][...])
    float *EMB, *X5, *Y11[kPasses], *Y17[kPasses], *PP[2];
    float *MSGA = nullptr;   // the loop messages of all looped layers of a pass, made by one launch (one row)
    float *A, *MSG, *GQ, *GK, *GV, *XQ, *XK, *XV, *P, *PART, *O, *OP, *R, *MI, *XM, *ACT, *Y, *logits;
    float *gtmp;
    int *ids_d, *idx;
    float* wts;
    float2* best;
    unsigned* done;
    int* state;   // on the card: [0] the position of the next row, [1] the next token
    int* host_state;
    int pos = 0;
    hipGraphExec_t graph = nullptr, vgraph = nullptr;
    // Several conversations (--parallel N): each has its own K / V caches, convolution states, position and step
    // graph. The members above are the bound one's (bind); a step of several at once (step_multi) reads the others'
    // through `slots`. LSA is off with more than one.
    struct Slot {
        kc_t *K[kVisits], *V[kVisits];
        int8_t *K8[kVisits], *V8[kVisits];
        __half *KS[kVisits], *VS[kVisits];
        float *cs_k[kVisits], *cs_v[kVisits], *cs_a[kVisits], *cs_m[kVisits];
        int* state = nullptr;
        int pos = 0;
        hipGraphExec_t graph = nullptr;
        float* snap = nullptr;
    };
    std::vector<Slot> slots;
    int n_slots = 1, cur = 0;
    const std::vector<int>* mrows = nullptr;   // step_multi: row r is conversation (*mrows)[r]
    int *MTOK = nullptr, *mtok_h = nullptr;    // step_multi's tokens: [kMR] pairs (unused, token) on the card; the host's copy
    SlotRow* SLOTTAB = nullptr;                // [conversation][visit] on the card
    int* ROWSLOT = nullptr;                    // step_multi: row -> conversation
    hipGraphExec_t mgraph[kMR + 1] = {};       // step_multi's graph for each number of rows
    // the prompt path
    uint16_t *XQ16, *XK16, *XV16, *O16, *XMH, *ACT16;   // XMH: [2 tc][kH], hi rows then lo rows
    uint16_t *Z16, *G16;
    float *G64, *FGO;
    float* GP = nullptr;   // k_gate_gp's partial sums: 4 gates x 32 rows
    float *RL, *YD, *SCB;
    unsigned* rdone;
    int *src, *slot, *count, *bounds, *fill, *tiles, *n_tiles;
    int max_tiles = 0;
    int pf_min = 32;   // chunks of fewer rows go through the row kernels
#ifdef INKLING_BLT
    Blt blt;
#endif

    const StTensor* need(const std::string& name, const char* dtype) {
        const StTensor* T = st_->find("model.llm." + name);
        if (!T) { std::fprintf(stderr, "strata-inkling: %s is missing\n", name.c_str()); std::exit(1); }
        if (T->dtype != dtype) { std::fprintf(stderr, "strata-inkling: %s is %s, not %s\n", name.c_str(), T->dtype.c_str(), dtype); std::exit(1); }
        return T;
    }
    size_t numel(const StTensor* T) { size_t n = 1; for (auto d : T->shape) n *= size_t(d); return n; }
    template <class X> X* dev(const std::vector<X>& v) {
        void* d = nullptr;
        CK(hipMalloc(&d, v.size() * sizeof(X)));
        CK(hipMemcpy(d, v.data(), v.size() * sizeof(X), hipMemcpyHostToDevice));
        loaded += v.size() * sizeof(X);
        return (X*) d;
    }
    // an 8-bit copy (Q8_0) of fp16 rows
    W4 q4_of(const std::vector<__half>& w, size_t rows, size_t K) {
        std::vector<uint8_t> q(rows * K / 2);
        std::vector<__half> d(rows * (K / 32));
        for (size_t r = 0; r < rows; ++r)
            for (size_t b = 0; b < K / 32; ++b) {
                float mx = 0;
                for (int i = 0; i < 32; ++i) mx = std::max(mx, std::fabs(__half2float(w[r * K + b * 32 + i])));
                const float dd = mx / 7.f, id = dd > 0 ? 1.f / dd : 0.f;
                d[r * (K / 32) + b] = __float2half(dd);
                for (int i = 0; i < 32; i += 2) {
                    const int lo = int(std::lrintf(__half2float(w[r * K + b * 32 + i]) * id)), hi = int(std::lrintf(__half2float(w[r * K + b * 32 + i + 1]) * id));
                    q[(r * K + b * 32 + i) / 2] = uint8_t((lo & 15) | ((hi & 15) << 4));
                }
            }
        W4 o; o.q = dev(q); o.d = dev(d);
        return o;
    }
    W8 q8_of(const std::vector<__half>& w, size_t rows, size_t K) {
        std::vector<int8_t> q(rows * K);
        std::vector<__half> d(rows * (K / 32));
        for (size_t r = 0; r < rows; ++r)
            for (size_t b = 0; b < K / 32; ++b) {
                float mx = 0;
                for (int i = 0; i < 32; ++i) mx = std::max(mx, std::fabs(__half2float(w[r * K + b * 32 + i])));
                const float dd = mx / 127.f, id = dd > 0 ? 1.f / dd : 0.f;
                d[r * (K / 32) + b] = __float2half(dd);
                for (int i = 0; i < 32; ++i) q[r * K + b * 32 + i] = int8_t(std::lrintf(__half2float(w[r * K + b * 32 + i]) * id));
            }
        return W8{dev(q), dev(d)};
    }
    // (a test) rows r0 .. r1 of w put on the 8-bit grid in place, as q8_of would: which parts lose the most
    static void simq(std::vector<__half>& w, size_t r0, size_t r1, size_t K) {
        for (size_t r = r0; r < r1; ++r)
            for (size_t b = 0; b < K / 32; ++b) {
                float mx = 0;
                for (int i = 0; i < 32; ++i) mx = std::max(mx, std::fabs(__half2float(w[r * K + b * 32 + i])));
                const float dd = __half2float(__float2half(mx / 127.f)), id = dd > 0 ? 1.f / (mx / 127.f) : 0.f;
                for (int i = 0; i < 32; ++i) w[r * K + b * 32 + i] = __float2half(float(std::lrintf(__half2float(w[r * K + b * 32 + i]) * id)) * dd);
            }
    }
    // (a test, INKLING_SIMQ_BITS=4..7 [INKLING_SIMQ_BLK=16|32]) the same with fewer bits: what a smaller copy would cost
    static void simq_bits(std::vector<__half>& w, size_t r0, size_t r1, size_t K, int bits, int blk) {
        const float top = float((1 << (bits - 1)) - 1);
        for (size_t r = r0; r < r1; ++r)
            for (size_t b = 0; b < K / blk; ++b) {
                float mx = 0;
                for (int i = 0; i < blk; ++i) mx = std::max(mx, std::fabs(__half2float(w[r * K + b * blk + i])));
                const float dd = mx / top, id = dd > 0 ? 1.f / dd : 0.f;
                for (int i = 0; i < blk; ++i) w[r * K + b * blk + i] = __float2half(float(std::lrintf(__half2float(w[r * K + b * blk + i]) * id)) * dd);
            }
    }
    static void simq_any(std::vector<__half>& w, size_t r0, size_t r1, size_t K) {
        static const int bits = std::getenv("INKLING_SIMQ_BITS") ? std::atoi(std::getenv("INKLING_SIMQ_BITS")) : 8;
        static const int blk = std::getenv("INKLING_SIMQ_BLK") ? std::atoi(std::getenv("INKLING_SIMQ_BLK")) : 32;
        if (bits >= 8) simq(w, r0, r1, K); else simq_bits(w, r0, r1, K, bits, blk);
    }
    static bool simq_has(const char* part) {
        const char* e = std::getenv("INKLING_SIMQ");
        if (!e) return false;
        std::string s = std::string(",") + e + ",";
        return s.find(std::string(",") + part + ",") != std::string::npos || s.find(",all,") != std::string::npos;
    }
    bool w8 = true, w8h = true;   // 8-bit projections (wp / wo); 8-bit head
    int attn4_rows = 0;           // INKLING_ATTN4=<rows>: wp's first rows (q 3072, q+g 6144, all 8672) from a 4-bit copy when writing
    bool attn4_o = false;         // INKLING_ATTN4_O=1: wo from a 4-bit copy when writing
    W8 head8{};
    W4 head4{};          // the two-pass head's rough copy (INKLING_HEAD4=0: off)
    float* HMAX = nullptr;
    float head_margin = 12.f;
    // a bf16 or f32 tensor as fp32 values
    std::vector<float> f32(const std::string& name, size_t expect) {
        const StTensor* T = st_->find("model.llm." + name);
        if (!T) { std::fprintf(stderr, "strata-inkling: %s is missing\n", name.c_str()); std::exit(1); }
        const size_t n = numel(T);
        if (n != expect) { std::fprintf(stderr, "strata-inkling: %s has %zu values, not %zu\n", name.c_str(), n, expect); std::exit(1); }
        std::vector<float> v(n);
        const uint8_t* p = st_->data(*T);
        if (T->dtype == "BF16") for (size_t i = 0; i < n; ++i) { uint16_t b; std::memcpy(&b, p + 2 * i, 2); v[i] = bf16f(b); }
        else if (T->dtype == "F32") std::memcpy(v.data(), p, n * 4);
        else { std::fprintf(stderr, "strata-inkling: %s is %s\n", name.c_str(), T->dtype.c_str()); std::exit(1); }
        return v;
    }
    // bf16 rows into an fp16 array at row r0 (counting the values fp16 cannot hold exactly)
    void put16(std::vector<__half>& dst, size_t r0, const std::string& name, size_t rows, size_t cols) {
        const StTensor* T = need(name, "BF16");
        if (numel(T) != rows * cols) { std::fprintf(stderr, "strata-inkling: %s has %zu values\n", name.c_str(), numel(T)); std::exit(1); }
        const uint16_t* p = (const uint16_t*) st_->data(*T);
        for (size_t i = 0; i < rows * cols; ++i) {
            const float f = bf16f(p[i]);
            const __half h = __float2half(f);
            if (__half2float(h) != f) ++fp16_changed;
            dst[r0 * cols + i] = h;
        }
    }
    const uint8_t* up_raw(const std::string& name, const char* dtype, size_t expect_bytes) {
        const StTensor* T = need(name, dtype);
        if (T->len != expect_bytes) { std::fprintf(stderr, "strata-inkling: %s is %zu bytes, not %zu\n", name.c_str(), T->len, expect_bytes); std::exit(1); }
        void* d = nullptr;
        CK(hipMalloc(&d, T->len));
        CK(hipMemcpy(d, st_->data(*T), T->len, hipMemcpyHostToDevice));
        loaded += T->len;
        return (const uint8_t*) d;
    }
    // static conv1d weights ([C][1][4], tap j on x[t - 3 + j]) as taps by backward distance
    const float* conv_static(const std::string& name, int C) {
        auto w = f32(name, size_t(C) * 4);
        std::vector<float> o(size_t(C) * 4);
        for (int c = 0; c < C; ++c) for (int lag = 0; lag < 4; ++lag) o[c * 4 + lag] = w[c * 4 + (3 - lag)];
        return dev(o);
    }
    const float* conv_dynamic(const std::string& name) {   // static_w [4][1024]: tap by backward distance
        auto w = f32(name, 4 * kKVD);
        std::vector<float> o(4 * kKVD);
        for (int c = 0; c < kKVD; ++c) for (int lag = 0; lag < 4; ++lag) o[c * 4 + lag] = w[lag * kKVD + c];
        return dev(o);
    }

    bool all_logits;
    Model(const std::string& dir, int ctx, int chunk, bool every, int nslots = 1) : max_ctx(ctx), tc(chunk), n_slots(nslots), all_logits(every) {
        st_ = std::make_unique<SafeTensors>(dir + "/model-00001-of-00001.safetensors");
        if (const char* e = std::getenv("INKLING_W8")) { const int m = std::atoi(e); w8 = m == 1; w8h = m != 0; }   // 2: the head only
        if (const char* e = std::getenv("INKLING_ATTN4")) attn4_rows = w8 ? std::min(kNP, std::max(0, std::atoi(e))) / 32 * 32 : 0;
        if (const char* e = std::getenv("INKLING_ATTN4_O")) attn4_o = w8 && std::atoi(e) != 0;
        CK(hipStreamCreate(&st));
        {   // the token table and the head, fp16
            std::vector<__half> v(size_t(kVocab) * kH);
            const StTensor* T = need("embed.weight", "BF16");
            if (T->shape.size() != 2 || T->shape[1] != kH || T->shape[0] < kVocab) { std::fprintf(stderr, "strata-inkling: embed shape\n"); std::exit(1); }
            const uint16_t* p = (const uint16_t*) st_->data(*T);
            for (size_t i = 0; i < v.size(); ++i) { const float f = bf16f(p[i]); v[i] = __float2half(f); if (__half2float(v[i]) != f) ++fp16_changed; }
            embed = dev(v);
            const StTensor* U = need("unembed.weight", "BF16");
            if (U->shape.size() != 2 || U->shape[1] != kH || U->shape[0] < kVocab) { std::fprintf(stderr, "strata-inkling: unembed shape\n"); std::exit(1); }
            p = (const uint16_t*) st_->data(*U);
            for (size_t i = 0; i < v.size(); ++i) { const float f = bf16f(p[i]); v[i] = __float2half(f); if (__half2float(v[i]) != f) ++fp16_changed; }
            if (!w8h) head = dev(v);   // (the 8-bit copy replaces it)
            if (w8h) head8 = q8_of(v, kVocab, kH);
            if (w8h && !(std::getenv("INKLING_HEAD4") && std::atoi(std::getenv("INKLING_HEAD4")) == 0)) {
                head4 = q4_of(v, kVocab, kH);
                CK(hipMalloc(&HMAX, (kVocabP / 32) * 4));
                if (const char* e = std::getenv("INKLING_HEAD_MARGIN")) head_margin = float(std::atof(e));
            }
            if (spec) head_keep = v;   // (the draft's markov head is folded through it once the products are up)
        }
        embed_norm = dev(f32("embed_norm.weight", kH));
        out_norm = dev(f32("norm.weight", kH));
        out_q = dev(f32("attn_res.output_query", kH));
        {
            auto q = f32("middle_cycle_cross_pass_queries", 24 * kH), kv = f32("middle_cycle_cross_pass_kv_queries", 24 * kH);
            std::vector<float> m(12 * 4 * kH);
            for (int j = 0; j < 12; ++j)
                for (int c = 0; c < kH; ++c) {
                    m[(j * 4 + 0) * kH + c] = q[(2 * j) * kH + c];
                    m[(j * 4 + 1) * kH + c] = kv[(2 * j) * kH + c];
                    m[(j * 4 + 2) * kH + c] = kv[(2 * j + 1) * kH + c];
                    m[(j * 4 + 3) * kH + c] = q[(2 * j + 1) * kH + c];
                }
            msg_q = dev(m);
        }
        L.resize(kLayers);
        for (int i = 0; i < kLayers; ++i) {
            st_->drop();   // (the file pages read so far leave this process: a low host-memory peak)
            Layer& l = L[i];
            const std::string b = "layers." + std::to_string(i) + ".";
            l.global = is_global(i);
            l.loop = i >= kLS && i < kLE;
            {
                std::vector<__half> w(size_t(kNP) * kH, __float2half(0.f));
                put16(w, kRQ, b + "attn.wq_du.weight", kQD, kH);
                put16(w, kRG, b + "attn.gate_attn.weight", kQD, kH);
                put16(w, kRR, b + "attn.wr_du.weight", kQH * kDRel, kH);
                put16(w, kRS, b + "attn.attn_sink_proj.weight", kQH, kH);
                put16(w, kRK, b + "attn.wk_dv.weight", kKVD, kH);
                put16(w, kRV, b + "attn.wv_dv.weight", kKVD, kH);
                if (l.global) {
                    put16(w, kRKP, b + "attn.k_sconv.weight_proj", 32, kH);
                    put16(w, kRVP, b + "attn.v_sconv.weight_proj", 32, kH);
                }
                {
                    const char* nm[8] = {"q", "g", "r", "s", "k", "kp", "v", "vp"};
                    const int lo[8] = {kRQ, kRG, kRR, kRS, kRK, kRKP, kRV, kRVP}, hi[8] = {kRG, kRR, kRS, kRK, kRKP, kRV, kRVP, kNP};
                    for (int j = 0; j < 8; ++j) if (simq_has(nm[j])) simq_any(w, lo[j], hi[j], kH);
                }
                l.wp = dev(w);
                if (w8) l.wp8 = q8_of(w, kNP, kH);
                if (attn4_rows) l.wp4 = q4_of(w, kNP, kH);
                std::vector<__half> o(size_t(kH) * kQD);
                put16(o, 0, b + "attn.wo_ud.weight", kH, kQD);
                if (simq_has("o")) simq_any(o, 0, kH, kQD);
                l.wo = dev(o);
                if (w8) l.wo8 = q8_of(o, kH, kQD);
                if (attn4_o) l.wo4 = q4_of(o, kH, kQD);
            }
            l.qscale = dev(f32(b + "attn.q_l2_clip.scale", kQH * kHD));
            l.rscale = dev(f32(b + "attn.rel_l2_clip.scale", kQH * kDRel));
            {
                auto pj = f32(b + "attn.rel_logits_proj.proj", kDRel * kExt);
                l.proj = dev(pj);
                std::vector<__half> hh(size_t(kPjRows) * kDRel, __float2half(0.f)), hl = hh;
                for (int d = 0; d < kExt; ++d)
                    for (int j = 0; j < kDRel; ++j) {
                        const float v = pj[size_t(j) * kExt + d];
                        const __half a = __float2half(v);
                        hh[size_t(d + kPjOff) * kDRel + j] = a;
                        hl[size_t(d + kPjOff) * kDRel + j] = __float2half(v - __half2float(a));
                    }
                l.pjh = dev(hh);
                l.pjl = dev(hl);
            }
            if (l.global) {
                l.kconv = conv_dynamic(b + "attn.k_sconv.static_w");
                l.vconv = conv_dynamic(b + "attn.v_sconv.static_w");
                l.norm_q = dev(f32(b + "q_input_norm.weight", kH));
                l.norm_k = dev(f32(b + "k_input_norm.weight", kH));
                l.norm_v = dev(f32(b + "v_input_norm.weight", kH));
            } else {
                l.kconv = conv_static(b + "attn.k_sconv.weight", kKVD);
                l.vconv = conv_static(b + "attn.v_sconv.weight", kKVD);
                l.norm_q = l.norm_k = l.norm_v = dev(f32(b + "attn_norm.weight", kH));
            }
            l.aconv = conv_static(b + "attn_sconv.weight", kH);
            l.mconv = conv_static(b + "mlp_sconv.weight", kH);
            l.norm_m = dev(f32(b + "mlp_norm.weight", kH));
            l.router = dev(f32(b + "mlp.gate.weight", size_t(kE) * kH));
            l.rbias = dev(f32(b + "mlp.gate.bias", kE));
            {
                std::vector<__half> r(size_t(kE) * kH);
                put16(r, 0, b + "mlp.gate.weight", kE, kH);
                l.router16 = dev(r);
            }
            if (i > 0) {
                l.n_attn_q = l.global ? 3 : 1;
                l.attn_q = dev(f32("attn_res.attn_queries." + std::to_string(i - 1), size_t(l.n_attn_q) * kH));
            }
            l.mlp_q = dev(f32("attn_res.mlp_queries." + std::to_string(i), kH));
            if (l.loop) {
                const char* nm[4] = {"attn", "k", "v", "mlp"};
                for (int g = 0; g < 4; ++g) {
                    const std::string c = b + "controllers." + nm[g] + ".";
                    std::vector<__half> d(size_t(kGR) * 2 * kH);
                    put16(d, 0, c + "down_weight", kGR, 2 * kH);
                    l.g_down[g] = dev(d);
                    l.g_forget[g] = dev(f32(c + "forget_weight", size_t(kH) * kGR));
                    l.g_gain[g] = dev(f32(c + "gain_weight", size_t(kH) * kGR));
                    std::vector<__half> fg(size_t(2 * kH) * kGR);
                    put16(fg, 0, c + "forget_weight", kH, kGR);
                    put16(fg, kH, c + "gain_weight", kH, kGR);
                    l.g_fg16[g] = dev(fg);
                }
            }
            Experts& ex = l.ex;
            const std::string p = b + "mlp.experts.";
            ex.nv = st_->find("model.llm." + p + "w13_weight_amax") != nullptr;
            if (ex.nv) {
                ex.w13_row = kH / 2; ex.s13_row = kH / 16; ex.w2_row = kF / 2; ex.s2_row = kF / 16;
                ex.w13 = up_raw(p + "w13_weight", "U8", size_t(kE) * 2 * kF * ex.w13_row);
                ex.s13 = up_raw(p + "w13_weight_scale", "F8_E4M3", size_t(kE) * 2 * kF * ex.s13_row);
                ex.w2 = up_raw(p + "w2_weight", "U8", size_t(kE) * kH * ex.w2_row);
                ex.s2 = up_raw(p + "w2_weight_scale", "F8_E4M3", size_t(kE) * kH * ex.s2_row);
                for (int w = 0; w < 2; ++w) {
                    auto am = f32(p + (w ? "w2" : "w13") + "_weight_amax", kE);
                    auto dn = f32(p + (w ? "w2" : "w13") + "_weight_outer_scale_denominator", 1);
                    std::vector<float> f(kE);
                    for (int e = 0; e < kE; ++e) f[e] = float(double(am[e]) / double(dn[0]));
                    (w ? ex.f2 : ex.f13) = dev(f);
                }
            } else {
                ex.w13_row = kH; ex.s13_row = kH / 32; ex.w2_row = kF; ex.s2_row = kF / 32;
                ex.w13 = up_raw(p + "w13_weight", "F8_E4M3", size_t(kE) * 2 * kF * ex.w13_row);
                ex.s13 = up_raw(p + "w13_weight_scale", "U8", size_t(kE) * 2 * kF * ex.s13_row);
                ex.w2 = up_raw(p + "w2_weight", "F8_E4M3", size_t(kE) * kH * ex.w2_row);
                ex.s2 = up_raw(p + "w2_weight_scale", "U8", size_t(kE) * kH * ex.s2_row);
                // per row: the largest block scale s sets 2^(s - 121) aside, so that the row's values times
                // 2^-shift stay within fp16 (largest 448 * 2^6) for the prompt path's products
                std::vector<int8_t> sh(size_t(kE) * 2 * kF + size_t(kE) * kH);
                size_t wide = 0;
                for (int w = 0; w < 2; ++w) {
                    const uint8_t* S = st_->data(*need(p + (w ? "w2" : "w13") + "_weight_scale", "U8"));
                    const size_t rows = w ? size_t(kE) * kH : size_t(kE) * 2 * kF, per = w ? ex.s2_row : ex.s13_row;
                    for (size_t r = 0; r < rows; ++r) {
                        int mx = 0, mn = 255;
                        for (size_t j = 0; j < per; ++j) { mx = std::max<int>(mx, S[r * per + j]); mn = std::min<int>(mn, S[r * per + j]); }
                        if (mx - mn > 30) ++wide;
                        sh[(w ? size_t(kE) * 2 * kF : 0) + r] = int8_t(mx - 121);
                    }
                }
                if (wide) std::fprintf(stderr, "strata-inkling: layer %d: %zu MXFP8 rows span more than 2^30 (their smallest blocks round in the prompt path)\n", i, wide);
                l.shift = dev(sh);
            }
        }
        // K / V caches (global layers: the whole context; sliding: a ring) and convolution states, a set a conversation
        slots.resize(size_t(n_slots));
        for (int sl = 0; sl < n_slots; ++sl) {
        for (int v = 0; v < kVisits; ++v) {
            const int i = visit_layer(v);
            const size_t slots = is_global(i) ? size_t(max_ctx) : size_t(kSwaRing);
            kv_mask[v] = is_global(i) ? 0x7fffffff : kSwaRing - 1;
            if (is_global(i) && kv8) {   // the global passes' caches in 8 bits (INKLING_KV8, default on)
                K[v] = V[v] = nullptr;
                CK(hipMalloc(&K8[v], slots * kKVD)); CK(hipMalloc(&V8[v], slots * kKVD));
                CK(hipMalloc(&KS[v], slots * kKV * 2)); CK(hipMalloc(&VS[v], slots * kKV * 2));
                kv_bytes += 2 * slots * (kKVD + kKV * 2);
            } else {
                K8[v] = V8[v] = nullptr; KS[v] = VS[v] = nullptr;
                CK(hipMalloc(&K[v], slots * kKVD * sizeof(kc_t)));
                CK(hipMalloc(&V[v], slots * kKVD * sizeof(kc_t)));
                kv_bytes += 2 * slots * kKVD * sizeof(kc_t);
            }
            for (float** p : {&cs_k[v], &cs_v[v]}) { CK(hipMalloc(p, 3 * kKVD * 4)); CK(hipMemset(*p, 0, 3 * kKVD * 4)); }
            for (float** p : {&cs_a[v], &cs_m[v]}) { CK(hipMalloc(p, 3 * kH * 4)); CK(hipMemset(*p, 0, 3 * kH * 4)); }
        }
        CK(hipMalloc(&state, 8));
        CK(hipMemset(state, 0, 8));
        slot_put(slots[size_t(sl)]);
        }
        slot_get(slots[0]);
        {
            std::vector<SlotRow> tab(size_t(n_slots) * kVisits);
            for (int sl = 0; sl < n_slots; ++sl)
                for (int v = 0; v < kVisits; ++v) {
                    const Slot& s = slots[size_t(sl)];
                    tab[size_t(sl) * kVisits + v] = SlotRow{s.K[v], s.V[v], s.K8[v], s.V8[v], s.KS[v], s.VS[v], s.cs_k[v], s.cs_v[v], s.cs_a[v], s.cs_m[v], s.state};
                }
            CK(hipMalloc(&SLOTTAB, tab.size() * sizeof(SlotRow)));
            CK(hipMemcpy(SLOTTAB, tab.data(), tab.size() * sizeof(SlotRow), hipMemcpyHostToDevice));
            CK(hipMalloc(&ROWSLOT, kMR * 4));
            CK(hipMemset(ROWSLOT, 0, kMR * 4));
        }
        auto rows = [&](float** p, size_t per) { CK(hipMalloc(p, size_t(tc) * per * 4)); pf_bytes += size_t(tc) * per * 4; };
        for (float** p : {&EMB, &X5, &Y11[0], &Y11[1], &Y11[2], &Y17[0], &Y17[1], &Y17[2], &PP[0], &PP[1], &GQ, &GK, &GV, &XQ, &XK, &XV,
                          &OP, &R, &MI, &XM, &Y, &gtmp})
            rows(p, kH);
        rows(&A, 3 * kH);
        rows(&MSG, 4 * kH);
        CK(hipMalloc(&MSGA, size_t(kLE - kLS) * 4 * kH * 4));
        rows(&P, kNP);
        {   // the row kernels' attention parts: only chunks under pf_min rows use them (all with INKLING_PF_OLDATTN)
            const size_t r = std::getenv("INKLING_PF_OLDATTN") ? size_t(tc) : size_t(std::min(tc, 64));
            CK(hipMalloc(&PART, r * kQH * kAttnChunks * (kHD + 4) * 4));
            pf_bytes += r * kQH * kAttnChunks * (kHD + 4) * 4;
        }
        rows(&O, kQD);
        rows(&ACT, size_t(kTopK) * kF);
        {   // logits: one row, or every row of a chunk for the checks (--logits / --top)
            const size_t r = all_logits ? size_t(tc) : (spec || n_slots > 1) ? size_t(kDB) : 1;
            CK(hipMalloc(&logits, r * kVocabP * 4));
            pf_bytes += r * kVocabP * 4;
        }
        rows(&wts, kTopK);
        CK(hipMalloc(&idx, size_t(tc) * kTopK * 4));
        auto rows16 = [&](uint16_t** p, size_t per) { CK(hipMalloc(p, size_t(tc) * per * 2)); pf_bytes += size_t(tc) * per * 2; };
        rows16(&XQ16, kH); rows16(&XK16, kH); rows16(&XV16, kH); rows16(&O16, kQD); rows16(&XMH, 2 * kH);
        rows16(&ACT16, size_t(kTopK) * kF);
        rows(&RL, 2 * kE);
        rows16(&Z16, 2 * kH); rows16(&G16, kGR); rows(&G64, kGR); rows(&FGO, 2 * kH);
        CK(hipMalloc(&GP, size_t(4) * 32 * kGPS * 4));
        rows(&YD, size_t(kTopK) * kH);
        rows(&SCB, kE);
        CK(hipMalloc(&rdone, size_t(tc) * 4)); CK(hipMemset(rdone, 0, size_t(tc) * 4));
        max_tiles = (tc * kTopK + 63) / 64 + kE;
        for (int** p : {&src, &slot}) CK(hipMalloc(p, size_t(tc) * kTopK * 4));
        CK(hipMalloc(&count, kE * 4)); CK(hipMalloc(&fill, kE * 4)); CK(hipMalloc(&bounds, (kE + 1) * 4));
        CK(hipMalloc(&tiles, size_t(max_tiles) * 2 * 4)); CK(hipMalloc(&n_tiles, 4));
#ifdef INKLING_BLT
        blt.init();
        if (spec) init_draft(dir);
#endif
        if (const char* e = std::getenv("INKLING_PF_MIN")) pf_min = std::atoi(e);
        if (const char* mp = std::getenv("INKLING_LSA"); mp && *mp && kv8 && n_slots == 1) {
            FILE* f = std::fopen(mp, "rb");
            if (!f) { std::fprintf(stderr, "strata-inkling: cannot open the LSA map %s\n", mp); std::exit(1); }
            std::vector<float> u(size_t(kKV) * kHD * kLsaR);
            lsa_maxp = max_ctx / 4 + 2;
            for (int v = 0; v < kVisits; ++v) {
                if (!is_global(visit_layer(v))) continue;
                if (std::fread(u.data(), 4, u.size(), f) != u.size()) { std::fprintf(stderr, "strata-inkling: the LSA map is short\n"); std::exit(1); }
                LU[v] = dev(u);
                CK(hipMalloc(&LIDX[v], size_t(lsa_maxp) * kKV * kLsaR * 2));
                CK(hipMemset(LIDX[v], 0, size_t(lsa_maxp) * kKV * kLsaR * 2));
            }
            std::fclose(f);
            CK(hipMalloc(&LSC, size_t(kKV) * lsa_maxp * 4));
            CK(hipMalloc(&LLST, size_t(kKV) * kLsaList * 4));
            CK(hipMalloc(&LCNT, kKV * 4));
            if (const char* e = std::getenv("INKLING_LSA_MIN")) lsa_min = std::atoi(e);
            if (const char* e = std::getenv("INKLING_LSA_TOP")) lsa_top = std::min(kLsaTop, std::max(64, std::atoi(e)));
            if (const char* e = std::getenv("INKLING_LSA_PF")) lsa_pf = std::atoi(e) != 0;
            if (const char* e = std::getenv("INKLING_LSA_PF_TOP")) lsa_pf_top = std::atoi(e);
            if (lsa_pf) {
                const int nbm = (tc + 63) / 64;
                lbcap = (4 + 4 * lsa_pf_top + 15) & ~15;
                CK(hipMalloc(&QP16, size_t(tc) * kQH * kLsaR * 2));
                CK(hipMalloc(&LBS, size_t(nbm) * kKV * lsa_maxp * 4));
                CK(hipMalloc(&LBL, size_t(nbm) * kKV * lbcap * 4));
                CK(hipMalloc(&LBC, size_t(nbm) * kKV * 4));
            }
            lsa = true;
            std::fprintf(stderr, "strata-inkling: LSA on past %d positions (map %s)\n", lsa_min, mp);
        }
        CK(hipMalloc(&ids_d, size_t(tc) * 4));
        CK(hipMalloc(&MLIST, sizeof(MoeList)));
        CK(hipMalloc(&YP, size_t(kMR) * kTopK * kH * 4));
        CK(hipMalloc(&XP16, size_t(3) * kMR * kQD * 2));
        CK(hipMalloc(&best, 64 * sizeof(float2)));
        CK(hipMalloc(&done, 4));
        CK(hipMemset(done, 0, 4));
        CK(hipHostMalloc(&host_state, 8));
        CK(hipMalloc(&MTOK, kMR * 2 * 4));
        CK(hipHostMalloc(&mtok_h, kMR * 2 * 4));
    }

    static int visit_layer(int v) {
        if (v < kLS) return v;
        if (v < kLS + kPasses * (kLE - kLS)) return kLS + (v - kLS) % (kLE - kLS);
        return v - (kPasses - 1) * (kLE - kLS);
    }
    static int visit_pass(int v) { return (v >= kLS && v < kLS + kPasses * (kLE - kLS)) ? (v - kLS) / (kLE - kLS) : 0; }

    float* out_of(int v) {
        const int i = visit_layer(v), p = visit_pass(v);
        if (v == kLS - 1) return X5;
        if (i == 11 && v < kLS + kPasses * 12) return Y11[p];
        if (i == 17 && v < kLS + kPasses * 12) return Y17[p];
        return PP[v & 1];
    }
    float* in_of(int v) {
        const int i = visit_layer(v);
        if (v == 0) return EMB;
        if (i == kLS && v < kLS + kPasses * 12) return X5;
        if (i == kLE) return Y17[kPasses - 1];
        return out_of(v - 1);
    }
    // the block states a visit's layer sees (before its own, if it starts a block)
    std::vector<const float*> blocks_of(int v) {
        const int i = visit_layer(v), p = visit_pass(v);
        std::vector<const float*> b;
        if (i >= 1) b.push_back(EMB);
        if (i >= 7) b.push_back(X5);
        if (i >= 13) b.push_back(Y11[i >= kLE ? kPasses - 1 : p]);
        if (i >= 19) b.push_back(Y17[kPasses - 1]);
        return b;
    }

    static bool al16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }
    static bool old_mix() { static const bool o = std::getenv("INKLING_OLD_MIX") != nullptr; return o; }
    // the bound conversation's members into / out of its slot
    void slot_put(Slot& s) {
        for (int v = 0; v < kVisits; ++v) {
            s.K[v] = K[v]; s.V[v] = V[v]; s.K8[v] = K8[v]; s.V8[v] = V8[v]; s.KS[v] = KS[v]; s.VS[v] = VS[v];
            s.cs_k[v] = cs_k[v]; s.cs_v[v] = cs_v[v]; s.cs_a[v] = cs_a[v]; s.cs_m[v] = cs_m[v];
        }
        s.state = state; s.pos = pos; s.graph = graph; s.snap = snap;
    }
    void slot_get(const Slot& s) {
        for (int v = 0; v < kVisits; ++v) {
            K[v] = s.K[v]; V[v] = s.V[v]; K8[v] = s.K8[v]; V8[v] = s.V8[v]; KS[v] = s.KS[v]; VS[v] = s.VS[v];
            cs_k[v] = s.cs_k[v]; cs_v[v] = s.cs_v[v]; cs_a[v] = s.cs_a[v]; cs_m[v] = s.cs_m[v];
        }
        state = s.state; pos = s.pos; graph = s.graph; snap = s.snap;
    }
    int pos_of(int b) const { return b == cur ? pos : slots[size_t(b)].pos; }
    // conversation b becomes the one the single-row and prompt paths work on
    void bind(int b) {
        if (b == cur) return;
        slot_put(slots[size_t(cur)]);
        slot_get(slots[size_t(b)]);
        cur = b;
    }
    void rms(int T, const float* x, int xs, const float* w, float* out, int os, uint16_t* o16 = nullptr, uint16_t* lo16 = nullptr) {
        if (!old_mix() && xs % 4 == 0 && os % 4 == 0 && al16(x) && al16(out) && (!w || al16(w)))
            k_rms4<<<T, 576, 0, st>>>(x, xs, w, out, os, o16, lo16);
        else k_rms<<<T, 256, 0, st>>>(x, xs, w, out, os, o16, lo16);
    }
    bool mix_ok(const Items& it, const float* q, int nq) {
        if (old_mix()) return false;
        for (int i = 0; i < it.n; ++i) if (!al16(it.p[i])) return false;
        for (int j = 0; j < nq; ++j) if (!al16(q + size_t(j) * kH)) return false;
        return true;
    }
    void mix(const Items& it, const float* q, int nq, float* out, int T) {
        if (nq <= 4 && al16(out) && mix_ok(it, q, nq)) k_mix4<4, false><<<T, 576, 0, st>>>(it, q, nq, out, NormOut{});
        else k_resmix<<<T, 256, 0, st>>>(it, q, nq, out);
    }
    void mix_norm(const Items& it, const float* q, int nq, const NormOut& no, int T) {
        bool ok = nq <= 3 && mix_ok(it, q, nq);
        for (int j = 0; j < nq && ok; ++j) ok = al16(no.out[j]) && al16(no.nw[j]);
        if (ok) k_mix4<3, true><<<T, 576, 0, st>>>(it, q, nq, nullptr, no);
        else k_resmix_norm<<<T, 256, 0, st>>>(it, q, nq, no);
    }
    // a message gate over T rows: the per-row kernel, or (prompt path) three products
    void gate(const Layer& l, int g, const float* h, int hs, const float* m, int ms, float* out, int T, bool pf) {
#ifdef INKLING_BLT
        if (pf) {
            k_gate_in<<<T, 256, 0, st>>>(h, hs, m, ms, Z16);
            blt.run(l.g_down[g], kGR, 2 * kH, Z16, 2 * kH, T, G64, kGR, st);
            k_to16<<<T, 256, 0, st>>>(G64, kGR, kGR, G16, kGR);
            blt.run(l.g_fg16[g], 2 * kH, kGR, G16, kGR, T, FGO, 2 * kH, st);
            k_gate_out<<<T, 256, 0, st>>>(h, hs, m, ms, FGO, out, kH);
            return;
        }
#endif
        static const bool old_gate = std::getenv("INKLING_OLD_GATE") != nullptr;
        if (old_gate) { k_msg_gate<<<T, 256, 0, st>>>(h, hs, m, ms, l.g_down[g], l.g_forget[g], l.g_gain[g], out, kH); return; }
        static const bool gate2 = !std::getenv("INKLING_GATE2") || std::atoi(std::getenv("INKLING_GATE2")) != 0;
        if (gate2 && T <= 32) {
            GateJobs gj{};
            gj.j[0] = GateJob{h, hs, m, ms, l.g_down[g], l.g_forget[g], l.g_gain[g], out, nullptr};
            k_gate_gp<<<dim3(kGR / 8 * kGP, T, 1), 256, 0, st>>>(gj, GP, T);
            k_gate_op<<<dim3(kH / 32, T, 1), 256, 0, st>>>(gj, GP, T);
            return;
        }
        k_gate_g<<<dim3(kGR / 8, T), 256, 0, st>>>(h, hs, m, ms, l.g_down[g], G64);
        k_gate_o<<<dim3(kH / 32, T), 256, 0, st>>>(h, hs, m, ms, G64, l.g_forget[g], l.g_gain[g], out, kH);
    }
    void mv(const __half* W, const W8& W8q, int K, MvSeg s, int rows, float* y, int ys, int T, const W4& w4 = W4{}, int split4 = 0) {
        static const int rw = std::getenv("INKLING_MV_RW") ? std::atoi(std::getenv("INKLING_MV_RW")) : 0;
        if (w8 && multi && T > 1 && T <= kMR) {
            k_mprep<<<dim3(T, s.n), 256, 0, st>>>(s, K, XP16);
            with_rows(T, [&](auto NC) {
                constexpr int N = decltype(NC)::value;
                if (rows >= 8192) k_mv8m<4, N><<<rows / 32, 256, size_t(N) * K * 2, st>>>(W8q, K, s, XP16, y, ys);
                else k_mv8m<1, N><<<rows / 8, 256, size_t(N) * K * 2, st>>>(W8q, K, s, XP16, y, ys);
            });
        } else if (w8) {
            const int r = rw ? rw : (rows >= 8192 ? 4 : 1);
            static const bool mvf = !std::getenv("INKLING_MVF") || std::atoi(std::getenv("INKLING_MVF")) != 0;
            if (r == 4 && mvf && K == kH && split4 == 0 && std::getenv("INKLING_MVF_WP")) k_mv8f<4, kH><<<dim3(rows / 32, T), 256, K * 5, st>>>(W8q, s, y, ys);
            else if (r == 4) k_mv8<4><<<dim3(rows / 32, T), 256, K * 5, st>>>(W8q, K, s, y, ys, ConvEpi{}, w4, split4);
            else if (r == 2) k_mv8<2><<<dim3(rows / 16, T), 256, K * 5, st>>>(W8q, K, s, y, ys, ConvEpi{}, w4, split4);
            else k_mv8<1><<<dim3(rows / 8, T), 256, K * 5, st>>>(W8q, K, s, y, ys, ConvEpi{}, w4, split4);
        }
        else k_mv16<<<dim3(rows / 32, T), 256, K * 4, st>>>(W, K, s, y, ys);
    }

    // one layer visit over T rows at positions state[0] ..
    // (a test: INKLING_DUMP_FEAT=file appends the drafter's target features, tag then T rows of kH fp32)
    void dump_rows(int tag, const float* d, int T) {
        static const char* fn = std::getenv("INKLING_DUMP_FEAT");
        if (!fn) return;
        std::vector<float> h(size_t(T) * kH);
        CK(hipMemcpyAsync(h.data(), d, h.size() * 4, hipMemcpyDeviceToHost, st));
        CK(hipStreamSynchronize(st));
        if (FILE* f = std::fopen(fn, "ab")) { int hd[2] = {tag, T}; std::fwrite(hd, 4, 2, f); std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
    }
    void visit(int v, int T) {
        const int i = visit_layer(v), p = visit_pass(v);
        const Layer& l = L[i];
        const float* hin = in_of(v);
        if (v == 5 || v == 23 || v == 41) {
            dump_rows(v, hin, T);
            if (spec_feat) CK(hipMemcpyAsync(FEAT + size_t(v == 5 ? 0 : v == 23 ? 1 : 2) * tc * kH, hin, size_t(T) * kH * 4, hipMemcpyDeviceToDevice, st));
        }
        float* hout = out_of(v);
        const auto blocks = blocks_of(v);
        const bool msgs = l.loop && p > 0;
#ifdef INKLING_BLT
        const bool pf = T >= pf_min || force_pf;
#else
        const bool pf = false;
#endif
        const float* msg = MSG;
        if (msgs) {
            Items it{};
            it.p[0] = Y11[p - 1]; it.p[1] = Y17[p - 1]; it.n = 2;
            static const bool nomsgb = std::getenv("INKLING_NO_MSGBATCH") != nullptr;
            if (T == 1 && !nomsgb && mix_ok(it, msg_q, 4) && al16(MSGA)) {
                // both inputs are last pass's, the same for every looped layer: all their messages at the pass's first
                if (i == kLS) k_mix4<4, false><<<dim3(1, kLE - kLS), 576, 0, st>>>(it, msg_q, 4, MSGA, NormOut{}, 4 * kH);
                msg = MSGA + size_t(i - kLS) * 4 * kH;
            } else mix(it, msg_q + size_t(i - kLS) * 4 * kH, 4, MSG, T);
        }
        // attention inputs
        const float *aq = hin, *ak = hin, *av = hin;
        int astride = kH;
        static const bool unfused = std::getenv("INKLING_UNFUSED") != nullptr;
        const bool boundary = i % kBlk == 0;
        const bool fuse1 = T == 1 && w8 && !unfused;   // the convolutions inside the producing kernels (one row)
        bool normed = false;   // XQ / XK / XV already made (k_resmix_norm)
        if (i > 0 && !msgs && !unfused) {
            Items it{};
            for (size_t j = 0; j < blocks.size(); ++j) it.p[j] = blocks[j];
            it.p[blocks.size()] = hin;
            it.n = int(blocks.size()) + 1;
            NormOut no{};
            no.nw[0] = l.norm_q; no.nw[1] = l.norm_k; no.nw[2] = l.norm_v;
            no.out[0] = XQ; no.out[1] = XK; no.out[2] = XV;
            if (pf) { no.o16[0] = XQ16; no.o16[1] = XK16; no.o16[2] = XV16; }
            mix_norm(it, l.attn_q, l.n_attn_q, no, T);
            normed = true;
        } else if (i > 0) {
            Items it{};
            for (size_t j = 0; j < blocks.size(); ++j) it.p[j] = blocks[j];
            it.p[blocks.size()] = hin;
            it.n = int(blocks.size()) + 1;
            mix(it, l.attn_q, l.n_attn_q, A, T);
            astride = l.n_attn_q * kH;
            aq = A;
            ak = A + (l.global ? kH : 0);
            av = A + (l.global ? 2 * kH : 0);
        }
        bool same = !l.global && !msgs;
        static const bool nofuse2 = std::getenv("INKLING_NOFUSE2") != nullptr;
        const bool f2 = !pf && w8 && !nofuse2;   // round-2 fusions: batched gates, norms done by their consumers
        if (msgs) {
            if (f2) {
                GateJobs gj{};
                float* outs[3] = {GQ, GK, GV};
                const float* ins[3] = {aq, ak, av};
                for (int g = 0; g < 3; ++g)
                    gj.j[g] = GateJob{ins[g], astride, msg + g * kH, 4 * kH, l.g_down[g], l.g_forget[g], l.g_gain[g], outs[g], G64 + size_t(g) * kGR * 32};
                static const bool gate2 = !std::getenv("INKLING_GATE2") || std::atoi(std::getenv("INKLING_GATE2")) != 0;
                if (gate2 && T <= 32) {
                    k_gate_gp<<<dim3(kGR / 8 * kGP, T, 3), 256, 0, st>>>(gj, GP, T);
                    k_gate_op<<<dim3(kH / 32, T, 3), 256, 0, st>>>(gj, GP, T);
                } else {
                k_gate_gn<<<dim3(kGR / 8, T, 3), 256, 0, st>>>(gj);
                k_gate_on<<<dim3(kH / 32, T, 3), 256, 0, st>>>(gj);
                }
            } else {
                gate(l, 0, aq, astride, msg + 0 * kH, 4 * kH, GQ, T, pf);
                gate(l, 1, ak, astride, msg + 1 * kH, 4 * kH, GK, T, pf);
                gate(l, 2, av, astride, msg + 2 * kH, 4 * kH, GV, T, pf);
            }
            aq = GQ; ak = GK; av = GV; astride = kH;
        }
        const bool nol = f2 && !normed;   // the projection rms-norms its own inputs (XQ / XK / XV not made)
        if (!normed && !nol) rms(T, aq, astride, l.norm_q, XQ, kH, pf ? XQ16 : nullptr);
        if (!same && !normed && !nol) {
            rms(T, ak, astride, l.norm_k, XK, kH, pf ? XK16 : nullptr);
            rms(T, av, astride, l.norm_v, XV, kH, pf ? XV16 : nullptr);
        }
        if (pf) {
#ifdef INKLING_BLT
            if (same) blt.run(l.wp, kNP, kH, XQ16, kH, T, P, kNP, st);
            else {
                blt.run(l.wp, kRK, kH, XQ16, kH, T, P, kNP, st);
                blt.run(l.wp + size_t(kRK) * kH, l.global ? kRV - kRK : kKVD, kH, XK16, kH, T, P + kRK, kNP, st);
                blt.run(l.wp + size_t(kRV) * kH, l.global ? kNP - kRV : kKVD, kH, XV16, kH, T, P + kRV, kNP, st);
            }
#endif
        } else {
            MvSeg s{};
            s.x[0] = XQ; s.x[1] = same ? XQ : XK; s.x[2] = same ? XQ : XV;
            if (nol) {
                s.x[0] = aq; s.x[1] = same ? aq : ak; s.x[2] = same ? aq : av;
                s.nw[0] = l.norm_q; s.nw[1] = same ? l.norm_q : l.norm_k; s.nw[2] = same ? l.norm_q : l.norm_v;
            }
            s.lo[0] = 0; s.hi[0] = kRK;
            s.lo[1] = kRK; s.hi[1] = l.global ? kRV : kRKP;
            s.lo[2] = kRV; s.hi[2] = l.global ? kNP : kRVP;
            s.n = 3; s.xs = kH;
            mv(l.wp, l.wp8, kH, s, kNP, P, kNP, T, l.wp4, T == 1 ? attn4_rows : 0);
        }
        const int upd = T == 1;
        if (mrows)   // (rows of different conversations: each a single row of its own, its state moved on in place)
            k_kvconv<<<dim3(kKV, 3, T), 128, 0, st>>>(P, l.kconv, l.vconv, nullptr, nullptr, nullptr, l.global, kv_mask[v], nullptr, nullptr, 1,
                                                      l.qscale, l.rscale, nullptr, nullptr, nullptr, nullptr, SLOTTAB + v, ROWSLOT);
        else
        k_kvconv<<<dim3(kKV, 3, T), 128, 0, st>>>(P, l.kconv, l.vconv, cs_k[v], cs_v[v], state, l.global, kv_mask[v], K[v], V[v], upd,
                                                  l.qscale, l.rscale, K8[v], V8[v], KS[v], VS[v]);
        if (lsa && LIDX[v]) k_lsa_index<<<dim3(T / 4 + 2, kKV), 128, 0, st>>>(K8[v], KS[v], LU[v], LIDX[v], state, T);
        static const bool old_attn = std::getenv("INKLING_PF_OLDATTN") != nullptr;
        if (pf && !old_attn) {
            static const int qw_s = std::getenv("INKLING_ATT_QW") ? std::atoi(std::getenv("INKLING_ATT_QW")) : 4;
            static const int qw_g = std::getenv("INKLING_ATT_QWG") ? std::atoi(std::getenv("INKLING_ATT_QWG")) : 4;
            const int qw = l.global ? qw_g : qw_s;
            const bool q8 = K8[v] != nullptr;
            const void* Kc = q8 ? (const void*) K8[v] : (const void*) K[v];
            const void* Vc = q8 ? (const void*) V8[v] : (const void*) V[v];
            static const bool a32 = std::getenv("INKLING_PFA32") ? std::atoi(std::getenv("INKLING_PFA32")) != 0 : false;   // (off: only ~8% on the global passes, and two unexplained very slow 32K runs)
            if (a32 && qw == 4) {
                if (q8) k_pf_attn32<4, true><<<dim3((T + 63) / 64, kKV), 384, 0, st>>>(P, Kc, Vc, KS[v], VS[v], l.pjh, l.pjl, O, O16, pos, T, !l.global, kv_mask[v]);
                else k_pf_attn32<4, false><<<dim3((T + 63) / 64, kKV), 384, 0, st>>>(P, Kc, Vc, nullptr, nullptr, l.pjh, l.pjl, O, O16, pos, T, !l.global, kv_mask[v]);
            } else if (q8) {
                if (qw == 8) k_pf_attn<8, true><<<dim3((T + 127) / 128, kKV), 768, 0, st>>>(P, Kc, Vc, KS[v], VS[v], l.pjh, l.pjl, O, O16, pos, T, !l.global, kv_mask[v]);
                else if (lsa && lsa_pf && LIDX[v] && l.global && pos + T > lsa_min) {
                    const int nb = (T + 63) / 64;
                    k_lsa_qproj<<<T, 256, 0, st>>>(P, LU[v], QP16);
                    k_lsa_bsel<<<dim3(nb, kKV), 1024, 0, st>>>(QP16, LIDX[v], LBS, lsa_maxp, LBL, LBC, pos, T, lsa_pf_top, lsa_min, lbcap);
                    k_pf_attn<4, true><<<dim3(nb, kKV), 384, 0, st>>>(P, Kc, Vc, KS[v], VS[v], l.pjh, l.pjl, O, O16, pos, T, !l.global, kv_mask[v], LBL, LBC, lbcap);
                }
                else k_pf_attn<4, true><<<dim3((T + 63) / 64, kKV), 384, 0, st>>>(P, Kc, Vc, KS[v], VS[v], l.pjh, l.pjl, O, O16, pos, T, !l.global, kv_mask[v]);
            } else if (qw == 4) k_pf_attn<4, false><<<dim3((T + 63) / 64, kKV), 384, 0, st>>>(P, Kc, Vc, nullptr, nullptr, l.pjh, l.pjl, O, O16, pos, T, !l.global, kv_mask[v]);
            else if (qw == 8) k_pf_attn<8, false><<<dim3((T + 127) / 128, kKV), 768, 0, st>>>(P, Kc, Vc, nullptr, nullptr, l.pjh, l.pjl, O, O16, pos, T, !l.global, kv_mask[v]);
            else k_pf_attn<2, false><<<dim3((T + 31) / 32, kKV), 192, 0, st>>>(P, Kc, Vc, nullptr, nullptr, l.pjh, l.pjl, O, O16, pos, T, !l.global, kv_mask[v]);
        } else {
            static const bool old_attn = std::getenv("INKLING_OLD_ATTN") != nullptr;
            static const int attn_k = std::getenv("INKLING_ATTN_K") ? std::atoi(std::getenv("INKLING_ATTN_K")) : 3;
            const int nch = (attn_k == 3 && !l.global) ? 16 : kAttnChunks;
            const bool use_lsa = lsa && LIDX[v] && T == 1;
            if (mrows) {
                if (K8[v]) k_attn3<true><<<dim3(kKV, nch, T), 256, 0, st>>>(P, l.proj, nullptr, nullptr, nullptr, nullptr, PART, nullptr, !l.global, kv_mask[v],
                                                                            nullptr, nullptr, SLOTTAB + v, ROWSLOT);
                else k_attn3<false><<<dim3(kKV, nch, T), 256, 0, st>>>(P, l.proj, nullptr, nullptr, nullptr, nullptr, PART, nullptr, !l.global, kv_mask[v],
                                                                       nullptr, nullptr, SLOTTAB + v, ROWSLOT);
                k_attn_merge3<<<dim3(kQH, T), 128, 0, st>>>(PART, P, O, nullptr, nch, nullptr, !l.global, nullptr, SLOTTAB + v, ROWSLOT);
            } else {
            if (use_lsa) {
                k_lsa_select<<<kKV, 1024, 0, st>>>(P, LU[v], LIDX[v], LSC, lsa_maxp, LLST, LCNT, state, lsa_min, lsa_top);
                k_attn3<true><<<dim3(kKV, nch, T), 256, 0, st>>>(P, l.proj, K8[v], V8[v], KS[v], VS[v], PART, state, !l.global, kv_mask[v], LLST, LCNT);
            } else if (K8[v]) k_attn3<true><<<dim3(kKV, nch, T), 256, 0, st>>>(P, l.proj, K8[v], V8[v], KS[v], VS[v], PART, state, !l.global, kv_mask[v]);
            else if (old_attn) k_attn<<<dim3(kKV, kAttnChunks, T), 96, 0, st>>>(P, l.proj, K[v], V[v], PART, state, !l.global, kv_mask[v]);
            else if (attn_k == 2) k_attn2<<<dim3(kKV, kAttnChunks, T), 96, 0, st>>>(P, l.proj, K[v], V[v], PART, state, !l.global, kv_mask[v]);
            else k_attn3<false><<<dim3(kKV, nch, T), 256, 0, st>>>(P, l.proj, K[v], V[v], nullptr, nullptr, PART, state, !l.global, kv_mask[v]);
            if (K8[v] || attn_k == 3) k_attn_merge3<<<dim3(kQH, T), 128, 0, st>>>(PART, P, O, pf ? O16 : nullptr, nch, state, !l.global, use_lsa ? LCNT : nullptr);
            else k_attn_merge<<<dim3(kQH, T), 128, 0, st>>>(PART, P, O, pf ? O16 : nullptr, nch);
            }
        }
        if (pf) {
#ifdef INKLING_BLT
            blt.run(l.wo, kH, kQD, O16, kQD, T, OP, kH, st);
#endif
        } else {
            MvSeg so{};
            so.x[0] = O; so.lo[0] = 0; so.hi[0] = kH; so.n = 1; so.xs = kQD;
            if (fuse1) {
                ConvEpi e;
                e.w = l.aconv; e.st = cs_a[v]; e.res = boundary ? nullptr : hin; e.out = R;
                static const int worw = std::getenv("INKLING_WO_RW") ? std::atoi(std::getenv("INKLING_WO_RW")) : 2;   // (rows a warp)
                static const bool mvf = !std::getenv("INKLING_MVF") || std::atoi(std::getenv("INKLING_MVF")) != 0;
                if (mvf && !attn4_o) {
                    if (worw == 4) k_mv8f<4, kQD><<<dim3(kH / 32, 1), 256, kQD * 5, st>>>(l.wo8, so, OP, kH, e);
                    else if (worw == 1) k_mv8f<1, kQD><<<dim3(kH / 8, 1), 256, kQD * 5, st>>>(l.wo8, so, OP, kH, e);
                    else k_mv8f<2, kQD><<<dim3(kH / 16, 1), 256, kQD * 5, st>>>(l.wo8, so, OP, kH, e);
                } else if (worw == 2) k_mv8<2><<<dim3(kH / 16, 1), 256, kQD * 5, st>>>(l.wo8, kQD, so, OP, kH, e, l.wo4, attn4_o ? kH : 0);
                else if (worw == 4) k_mv8<4><<<dim3(kH / 32, 1), 256, kQD * 5, st>>>(l.wo8, kQD, so, OP, kH, e, l.wo4, attn4_o ? kH : 0);
                else k_mv8<1><<<dim3(kH / 8, 1), 256, kQD * 5, st>>>(l.wo8, kQD, so, OP, kH, e, l.wo4, attn4_o ? kH : 0);
            } else mv(l.wo, l.wo8, kQD, so, kH, OP, kH, T);
        }
        if (mrows) k_conv_h<<<dim3(kH / 256, T), 256, 0, st>>>(OP, l.aconv, nullptr, boundary ? nullptr : hin, R, 1, SLOTTAB + v, ROWSLOT, 0);
        else
        if (!fuse1) k_conv_h<<<dim3(kH / 256, T), 256, 0, st>>>(OP, l.aconv, cs_a[v], boundary ? nullptr : hin, R, upd);
        bool mnormed = false;   // XM (and XMH) already made
        {
            Items it{};
            for (size_t j = 0; j < blocks.size(); ++j) it.p[j] = blocks[j];
            int n = int(blocks.size());
            if (boundary) it.p[n++] = hin;
            it.p[n++] = R;
            it.n = n;
            if (!msgs && !unfused) {
                NormOut no{};
                no.nw[0] = l.norm_m; no.out[0] = XM;
                if (pf) { no.o16[0] = XMH; no.lo16[0] = XMH + size_t(T) * kH; }
                mix_norm(it, l.mlp_q, 1, no, T);
                mnormed = true;
            } else mix(it, l.mlp_q, 1, MI, T);
        }
        const float* mi = MI;
        if (msgs) {
            gate(l, 3, MI, kH, msg + 3 * kH, 4 * kH, GQ, T, pf);
            mi = GQ;
        }
        if (pf) {
#ifdef INKLING_BLT
            if (!mnormed) rms(T, mi, kH, l.norm_m, XM, kH, XMH, XMH + size_t(T) * kH);
            blt.run(l.router16, kE, kH, XMH, kH, 2 * T, RL, kE, st);
            k_pf_topk<<<(T + 7) / 8, 256, 0, st>>>(RL, T, l.rbias, idx, wts);
            if (const char* dr = std::getenv("INKLING_DUMP_ROUTES")) {   // (a test: visit, T, then T x 8 expert ids)
                std::vector<int> h(size_t(T) * kTopK);
                CK(hipMemcpyAsync(h.data(), idx, h.size() * 4, hipMemcpyDeviceToHost, st));
                CK(hipStreamSynchronize(st));
                if (FILE* f = std::fopen(dr, "ab")) { int hd[2] = {v, T}; std::fwrite(hd, 4, 2, f); std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
            }
            CK(hipMemsetAsync(count, 0, kE * 4, st));
            k_pf_count<<<(T * kTopK + 255) / 256, 256, 0, st>>>(idx, T * kTopK, count);
            // expert tiles of 64 or 128 rows (INKLING_ETILE; 64 measured faster: less of the last tile wasted)
            static const int et = std::getenv("INKLING_ETILE") ? std::atoi(std::getenv("INKLING_ETILE")) : 64;
            static const int edown = std::getenv("INKLING_EDOWN") ? std::atoi(std::getenv("INKLING_EDOWN")) : 0;
            const int R = et == 64 ? 64 : 128;
            k_pf_bounds<<<1, kE, 0, st>>>(count, bounds, fill, tiles, n_tiles, R);
            k_pf_scatter<<<(T * kTopK + 255) / 256, 256, 0, st>>>(idx, T * kTopK, bounds, fill, src, slot);
            const int nt = std::min(max_tiles, (T * kTopK + R - 1) / R + kE);
            auto run_ex = [&](auto nvt) {
                constexpr bool NV = decltype(nvt)::value;
                if (R == 64) {
                    k_moe_grp<true, NV, 1, 4, 8, 1><<<dim3(kF / 128, nt), 256, 0, st>>>(l.ex, l.shift, XMH, src, bounds, tiles, n_tiles, nullptr, ACT16);
                    k_moe_grp<false, NV, 2, 4, 8, 1><<<dim3(kH / 256, nt), 256, 0, st>>>(l.ex, l.shift, ACT16, nullptr, bounds, tiles, n_tiles, YD, nullptr);
                    return;
                }
                k_moe_grp<true, NV, 1, 8, 8, 1><<<dim3(kF / 128, nt), 256, 0, st>>>(l.ex, l.shift, XMH, src, bounds, tiles, n_tiles, nullptr, ACT16);
                if (edown == 1) k_moe_grp<false, NV, 1, 8, 8, 1><<<dim3(kH / 128, nt), 256, 0, st>>>(l.ex, l.shift, ACT16, nullptr, bounds, tiles, n_tiles, YD, nullptr);
                else if (edown == 2) k_moe_grp<false, NV, 2, 8, 8, 1><<<dim3(kH / 256, nt), 256, 0, st>>>(l.ex, l.shift, ACT16, nullptr, bounds, tiles, n_tiles, YD, nullptr);
                else k_moe_grp<false, NV, 2, 4, 8, 2><<<dim3(kH / 256, nt), 512, 0, st>>>(l.ex, l.shift, ACT16, nullptr, bounds, tiles, n_tiles, YD, nullptr);
            };
            if (l.ex.nv) run_ex(std::true_type{});
            else run_ex(std::false_type{});
            k_pf_combine<<<T, 256, 0, st>>>(YD, slot, wts, Y);
#endif
        } else {
            // (round 2: when the input is not normed yet, the router and gate / up kernels norm it as they load it)
            const float* xm = XM;
            const float* nwm = nullptr;
            if (!mnormed) {
                if (f2) { xm = mi; nwm = l.norm_m; }
                else rms(T, mi, kH, l.norm_m, XM, kH);
            }
            static const bool old_router = std::getenv("INKLING_OLD_ROUTER") != nullptr;
            if (old_router) k_router<<<T, 256, 0, st>>>(XM, l.router, l.rbias, idx, wts);
            else if (std::getenv("INKLING_ROUTER2")) k_router2<<<dim3(kE / 8, T), 256, 0, st>>>(xm, l.router, l.rbias, SCB, rdone, idx, wts, nwm);
            else k_router3<<<dim3(kE / 4, T), 256, 0, st>>>(xm, l.router, l.rbias, SCB, rdone, idx, wts, nwm);
            static const bool old_moe = std::getenv("INKLING_OLD_MOE") != nullptr;
            if (multi && T > 1 && T <= kMR && !old_moe) {
                k_moe_list<<<1, 64, 0, st>>>(idx, T, MLIST);
                MvSeg sx{};
                sx.x[0] = xm; sx.nw[0] = nwm; sx.lo[0] = 0; sx.hi[0] = kH; sx.n = 1; sx.xs = kH;
                k_mprep<<<dim3(T, 1), 256, 0, st>>>(sx, kH, XP16);
                if (l.ex.nv) {
                    k_moe_gum<true><<<dim3(kF / 16, T * kTopK), 256, 0, st>>>(l.ex, XP16, MLIST, ACT16);
                    k_moe_downm<true><<<dim3(kH / 32, T * kTopK), 256, 0, st>>>(l.ex, ACT16, MLIST, YP);
                } else {
                    k_moe_gum<false><<<dim3(kF / 16, T * kTopK), 256, 0, st>>>(l.ex, XP16, MLIST, ACT16);
                    k_moe_downm<false><<<dim3(kH / 32, T * kTopK), 256, 0, st>>>(l.ex, ACT16, MLIST, YP);
                }
                k_moe_combm<<<T, 256, 0, st>>>(YP, wts, Y);
            } else if (old_moe) {
                k_moe_gu<<<dim3(kF / 32, kTopK, T), 256, 0, st>>>(l.ex, XM, idx, ACT);
                k_moe_down<<<dim3(kH / 32, T), 256, 0, st>>>(l.ex, ACT, idx, wts, Y);
            } else {
                ConvEpi e;
                if (fuse1) { e.w = l.mconv; e.st = cs_m[v]; e.res = R; e.out = hout; }
                if (l.ex.nv) {
                    k_moe_gu2<true><<<dim3(kF / 16, kTopK, T), 256, 0, st>>>(l.ex, xm, idx, ACT, nwm);
                    static const int drc = std::getenv("INKLING_DOWN_RC") ? std::atoi(std::getenv("INKLING_DOWN_RC")) : 1;
                    if (drc == 2) k_moe_down2<true, 2><<<dim3(kH / 16, T), 256, 0, st>>>(l.ex, ACT, idx, wts, Y, e);
                    else if (drc == 4) k_moe_down2<true, 4><<<dim3(kH / 32, T), 256, 0, st>>>(l.ex, ACT, idx, wts, Y, e);
                    else k_moe_down2<true><<<dim3(kH / 8, T), 256, 0, st>>>(l.ex, ACT, idx, wts, Y, e);
                } else {
                    k_moe_gu2<false><<<dim3(kF / 16, kTopK, T), 256, 0, st>>>(l.ex, xm, idx, ACT, nwm);
                    k_moe_down2<false><<<dim3(kH / 8, T), 256, 0, st>>>(l.ex, ACT, idx, wts, Y, e);
                }
            }
        }
        static const bool old_moe2 = std::getenv("INKLING_OLD_MOE") != nullptr;
        if (mrows) {
            k_conv_h<<<dim3(kH / 256, T), 256, 0, st>>>(Y, l.mconv, nullptr, R, hout, 1, SLOTTAB + v, ROWSLOT, 1);
            return;
        }
        if (!(fuse1 && !old_moe2)) k_conv_h<<<dim3(kH / 256, T), 256, 0, st>>>(Y, l.mconv, cs_m[v], R, hout, upd);
        ConvUpd u{};
        u.x[0] = P + kRK; u.xs[0] = kNP; u.st[0] = cs_k[v]; u.C[0] = kKVD;
        u.x[1] = P + kRV; u.xs[1] = kNP; u.st[1] = cs_v[v]; u.C[1] = kKVD;
        u.x[2] = OP; u.xs[2] = kH; u.st[2] = cs_a[v]; u.C[2] = kH;
        u.x[3] = Y; u.xs[3] = kH; u.st[3] = cs_m[v]; u.C[3] = kH;
        if (spec_rows) k_conv_state_all<<<dim3(kH / 256, 4, T), 256, 0, st>>>(u, T, SPST + size_t(v) * kDB * 4 * 3 * kH);
        if (!upd) k_conv_state<<<dim3(kH / 256, 4), 256, 0, st>>>(u, T);
    }
    // the final mix, norm and head for rows [r0, r0 + n) of the last visit's output
    void finish(int r0, int n) {
        const float* hl = out_of(kVisits - 1);
        Items it{};
        const auto b = blocks_of(kVisits - 1);   // layer 23's blocks: EMB, X5, Y11, Y17
        for (size_t j = 0; j < b.size(); ++j) it.p[j] = b[j] + size_t(r0) * kH;
        it.p[b.size()] = hl + size_t(r0) * kH;
        it.n = int(b.size()) + 1;
        mix(it, out_q, 1, gtmp, n);
        rms(n, gtmp, kH, out_norm, XM, kH);
        dump_rows(48, XM, n);
        if (w8h && n > 1 && n <= kDB) {
            head_rows16(XM, n);
            with_rows(n, [&](auto NC) { k_head8m2<decltype(NC)::value><<<kVocabP / 32 / kHeadG, 256, 0, st>>>(head8, XP16, logits, 1); });
        }
        else if (w8h && head4.q && n == 1) {
            k_head4<<<kVocabP / 32, 256, 0, st>>>(head4, XM, logits, HMAX);
            k_head_ref<<<kHeadRefB, 256, 0, st>>>(head8, XM, logits, HMAX, head_margin);
        }
        else if (w8h) k_head8<<<dim3(kVocabP / 32, n), 256, 0, st>>>(head8, XM, logits);
        else k_head<<<dim3(kVocabP / 32, n), 256, 0, st>>>(head, XM, logits);
    }
    void head_rows16(const float* x, int n) {
        MvSeg sx{};
        sx.x[0] = x; sx.lo[0] = 0; sx.hi[0] = kH; sx.n = 1; sx.xs = kH;
        k_mprep<<<dim3(n, 1), 256, 0, st>>>(sx, kH, XP16);
    }
    void feat_final(int T) {
        Items it{};
        const auto b = blocks_of(kVisits - 1);
        for (size_t j = 0; j < b.size(); ++j) it.p[j] = b[j];
        it.p[b.size()] = out_of(kVisits - 1);
        it.n = int(b.size()) + 1;
        mix(it, out_q, 1, gtmp, T);
        rms(T, gtmp, kH, out_norm, FEAT + size_t(3) * tc * kH, kH);
    }
    // ---- DSpark
    void init_draft(const std::string& dir) {
        SafeTensors m(dir + "/mtp.safetensors");
        auto bf = [&](const std::string& n, size_t expect) {
            const StTensor* T = m.find(n);
            if (!T || T->len != expect * 2) { std::fprintf(stderr, "strata-inkling: mtp %s missing or wrong size\n", n.c_str()); std::exit(1); }
            return (const uint16_t*) m.data(*T);
        };
        auto h16 = [&](std::vector<__half>& dst, size_t r0, const std::string& n, size_t rows, size_t cols) {
            const uint16_t* p = bf(n, rows * cols);
            for (size_t i = 0; i < rows * cols; ++i) dst[r0 * cols + i] = __float2half(bf16f(p[i]));
        };
        auto f32v = [&](const std::string& n, size_t cnt) {
            const uint16_t* p = bf(n, cnt);
            std::vector<float> v(cnt);
            for (size_t i = 0; i < cnt; ++i) v[i] = bf16f(p[i]);
            return v;
        };
        for (int i = 0; i < kDL; ++i) {
            const std::string b = "layers." + std::to_string(i) + ".";
            DLayer& d = DLy[i];
            { std::vector<__half> w(size_t(kQD + 2 * kKVD) * kH);
              h16(w, 0, b + "self_attn.q_proj.weight", kQD, kH); h16(w, kQD, b + "self_attn.k_proj.weight", kKVD, kH);
              h16(w, kQD + kKVD, b + "self_attn.v_proj.weight", kKVD, kH); d.qkv = dev(w); }
            { std::vector<__half> w(size_t(2 * kKVD) * kH);
              h16(w, 0, b + "self_attn.k_proj_ctx.weight", kKVD, kH); h16(w, kKVD, b + "self_attn.v_proj_ctx.weight", kKVD, kH); d.kvc = dev(w); }
            { std::vector<__half> w(size_t(kH) * kQD); h16(w, 0, b + "self_attn.o_proj.weight", kH, kQD); d.o = dev(w); }
            { std::vector<__half> w(size_t(2 * kDF) * kH);
              h16(w, 0, b + "mlp.gate_proj.weight", kDF, kH); h16(w, kDF, b + "mlp.up_proj.weight", kDF, kH); d.gu = dev(w); }
            { std::vector<__half> w(size_t(kH) * kDF); h16(w, 0, b + "mlp.down_proj.weight", kH, kDF); d.dn = dev(w); }
            d.in_ln = dev(f32v(b + "input_layernorm.weight", kH)); d.post_ln = dev(f32v(b + "post_attention_layernorm.weight", kH));
            d.dfl = dev(f32v(b + "dflare_norm.weight", kH)); d.qn = dev(f32v(b + "self_attn.q_norm.weight", kHD));
            d.kn = dev(f32v(b + "self_attn.k_norm.weight", kHD)); d.knc = dev(f32v(b + "self_attn.k_norm_ctx.weight", kHD));
            const bool local = i != 1;
            const size_t slots = local ? kDRing : size_t(max_ctx) + kDB;
            dmask[i] = local ? kDRing - 1 : 0x7fffffff;
            CK(hipMalloc(&DK[i], slots * kKVD * 2)); CK(hipMalloc(&DV[i], slots * kKVD * 2));
            CK(hipMemset(DK[i], 0, slots * kKVD * 2)); CK(hipMemset(DV[i], 0, slots * kKVD * 2));
            kv_bytes += 2 * slots * kKVD * 2;
        }
        {   // fuse weights: a softmax over the 4 taps for each of the 4 users (3 layers, the markov head)
            auto fw = f32v("fuse_weight", 16);
            for (int r = 0; r < 4; ++r) {
                float mx = -1e30f, sm = 0;
                for (int c = 0; c < 4; ++c) mx = std::max(mx, fw[r * 4 + c]);
                for (int c = 0; c < 4; ++c) { fw[r * 4 + c] = std::exp(fw[r * 4 + c] - mx); sm += fw[r * 4 + c]; }
                for (int c = 0; c < 4; ++c) fw[r * 4 + c] /= sm;
            }
            d_alpha = dev(fw);
        }
        d_mask = dev(f32v("mask_embed", kH));
        d_fnorm = dev(f32v("norm.weight", kH));
        d_mdfl = dev(f32v("markov_head.dflare_norm.weight", kH));
        d_rb = dev(f32v("markov_head.rnn_proj.bias", 3 * kDR));
        {   // rnn_proj [768][2r + 2H]: the [state, prev] columns fp32, the [hidden, f_rnn] columns fp16 (a product)
            const uint16_t* p = bf("markov_head.rnn_proj.weight", size_t(3 * kDR) * (2 * kDR + 2 * kH));
            std::vector<float> ws(size_t(3 * kDR) * 2 * kDR);
            std::vector<__half> wc(size_t(3 * kDR) * 2 * kH);
            for (int r = 0; r < 3 * kDR; ++r) {
                const uint16_t* row = p + size_t(r) * (2 * kDR + 2 * kH);
                for (int c = 0; c < 2 * kDR; ++c) ws[size_t(r) * 2 * kDR + c] = bf16f(row[c]);
                for (int c = 0; c < 2 * kH; ++c) wc[size_t(r) * 2 * kH + c] = __float2half(bf16f(row[2 * kDR + c]));
            }
            d_ws = dev(ws); d_wrc = dev(wc);
        }
        {   // markov_w1 [padded vocab][256]
            const StTensor* T = m.find("markov_head.markov_w1.weight");
            const size_t rows = T->shape[0];
            std::vector<__half> w(rows * kDR);
            h16(w, 0, "markov_head.markov_w1.weight", rows, kDR);
            d_w1 = dev(w);
        }
        {   // the fold: head . hidden_out (vocab x 256) and head . its bias, on the card in fp32, then 8-bit
            auto wo = f32v("markov_head.hidden_out.weight", size_t(kH) * kDR);   // [kH][256]
            auto bo = f32v("markov_head.hidden_out.bias", kH);
            std::vector<__half> xt(size_t(kDR + 1) * kH);   // rows: hidden_out's columns, then its bias
            for (int t = 0; t < kDR; ++t) for (int k = 0; k < kH; ++k) xt[size_t(t) * kH + k] = __float2half(wo[size_t(k) * kDR + t]);
            for (int k = 0; k < kH; ++k) xt[size_t(kDR) * kH + k] = __float2half(bo[k]);
            __half* hd = dev(head_keep);
            __half* xd = dev(xt);
            float* y;
            CK(hipMalloc(&y, size_t(kDR + 1) * kVocab * 4));
            blt.run(hd, kVocab, kH, (const uint16_t*) xd, kH, kDR + 1, y, kVocab, st);
            CK(hipStreamSynchronize(st));
            std::vector<float> yh(size_t(kDR + 1) * kVocab);
            CK(hipMemcpy(yh.data(), y, yh.size() * 4, hipMemcpyDeviceToHost));
            CK(hipFree(y)); CK(hipFree(hd)); CK(hipFree(xd));
            std::vector<__half> fold(size_t(kVocab) * kDR);
            for (int v = 0; v < kVocab; ++v) for (int t = 0; t < kDR; ++t) fold[size_t(v) * kDR + t] = __float2half(yh[size_t(t) * kVocab + v]);
            d_fold = q8_of(fold, kVocab, kDR);
            std::vector<float> fb(kVocabP, 0.f);
            for (int v = 0; v < kVocab; ++v) fb[v] = yh[size_t(kDR) * kVocab + v];
            d_fbias = dev(fb);
            head_keep.clear(); head_keep.shrink_to_fit();
        }
        auto f = [&](float** p, size_t n) { CK(hipMalloc(p, n * 4)); pf_bytes += n * 4; };
        auto u = [&](uint16_t** p, size_t n) { CK(hipMalloc(p, n * 2)); pf_bytes += n * 2; };
        f(&FEAT, size_t(kDTaps) * tc * kH); f(&DX, kDB * kH); f(&DRES, kDB * kH); f(&DN, kDB * kH); f(&DQKV, kDB * (kQD + 2 * kKVD));
        f(&DQ, kDB * kQD); f(&DOUT, kDB * kH); f(&DGU, size_t(kDB) * 2 * kDF); f(&DHID, kDB * kH); f(&DKVC, size_t(tc) * 2 * kKVD);
        f(&DPART, size_t(kKV) * kDCh * 24 * (kHD + 4)); f(&DCTX, kDB * 3 * kDR); f(&DMS, kDR); f(&DLAT, kDR);
        f(&DBASE, size_t(kDB) * kVocabP); f(&DLOG, kVocabP); f(&DEMB0, kH); f(&DFR, kH);
        f(&SPST, size_t(kVisits) * kDB * 4 * 3 * kH);
        u(&DH16, kDB * kH); u(&DO16, kDB * kQD); u(&DA16, size_t(kDB) * kDF); u(&DC16, size_t(tc) * kH); u(&DMK16, kDB * 2 * kH);
        CK(hipMalloc(&DTOK, 16 * 4)); CK(hipMalloc(&DGT, 16 * 4)); CK(hipMalloc(&DID, 4));
        CK(hipMalloc(&DACC, 4 * 4)); CK(hipMalloc(&DOUTT, 16 * 4)); CK(hipHostMalloc((void**) &hout, 16 * 4, 0));
        std::vector<float*> csp(size_t(kVisits) * 4);
        for (int v = 0; v < kVisits; ++v) { csp[v * 4] = cs_k[v]; csp[v * 4 + 1] = cs_v[v]; csp[v * 4 + 2] = cs_a[v]; csp[v * 4 + 3] = cs_m[v]; }
        CK(hipMalloc(&CSP, csp.size() * sizeof(float*)));
        CK(hipMemcpy(CSP, csp.data(), csp.size() * sizeof(float*), hipMemcpyHostToDevice));
        std::fprintf(stderr, "strata-inkling: DSpark draft loaded (3 layers, block %d)\n", kDB);
    }
    // the draft's context K / V of FEAT rows r0 .. r0 + R - 1 (positions p0 ..)
    void draft_ctx(int r0, int R, int p0) {
        for (int i = 0; i < kDL; ++i) {
            k_dmix<<<R, 256, 0, st>>>(FEAT + size_t(r0) * kH, size_t(tc) * kH, d_alpha + i * 4, DLy[i].dfl, DC16, kH, nullptr);
            blt.run(DLy[i].kvc, 2 * kKVD, kH, DC16, kH, R, DKVC, 2 * kKVD, st);
            k_dkvctx<<<dim3(R, kKV), 128, 0, st>>>(DKVC, DLy[i].knc, p0, dmask[i], DK[i], DV[i]);
        }
    }
    // the same for the rows the last check took (DACC: first position, rows)
    void draft_ctx_dev() {
        for (int i = 0; i < kDL; ++i) {
            k_dmix<<<kDB, 256, 0, st>>>(FEAT, size_t(tc) * kH, d_alpha + i * 4, DLy[i].dfl, DC16, kH, nullptr, nullptr, DACC + 1);
            blt.run(DLy[i].kvc, 2 * kKVD, kH, DC16, kH, kDB, DKVC, 2 * kKVD, st);
            k_dkvctx<<<dim3(kDB, kKV), 128, 0, st>>>(DKVC, DLy[i].knc, 0, dmask[i], DK[i], DV[i], DACC);
        }
    }
    int spec_n() const { static const int nv = std::getenv("INKLING_SPEC_N") ? std::max(2, std::min(kDB, std::atoi(std::getenv("INKLING_SPEC_N")))) : 3; return nv; }   // (3: the best measured)
    // ns drafted tokens into DTOK[1 ..] after the anchor DTOK[0] at position state[0] (context: positions before it; the
    // last known features: FEAT row DACC[2])
    void draft_body(int ns) {
        k_embed<<<1, 256, 0, st>>>(embed, DTOK, embed_norm, DEMB0);
        k_dblock<<<dim3(kH / 256, kDB), 256, 0, st>>>(DEMB0, d_mask, DRES);
        for (int i = 0; i < kDL; ++i) {
            const DLayer& d = DLy[i];
            rms(kDB, DRES, kH, d.in_ln, DN, kH, DH16);
            blt.run(d.qkv, kQD + 2 * kKVD, kH, DH16, kH, kDB, DQKV, kQD + 2 * kKVD, st);
            k_dqkv<<<dim3(kDB, kQH + 2 * kKV), 128, 0, st>>>(DQKV, d.qn, d.kn, state, dmask[i], DQ, DK[i], DV[i]);
            const int nch = i != 1 ? (kWin + kDB + 63) / 64 : kDCh;
            k_dattn<<<dim3(kKV, nch), 256, 0, st>>>(DQ, DK[i], DV[i], state, i != 1, dmask[i], DPART);
            k_dattn_merge<<<kKV * 24, 128, 0, st>>>(DPART, nch, DO16);
            blt.run(d.o, kH, kQD, DO16, kQD, kDB, DOUT, kH, st);
            k_dadd<<<kDB * kH / 256, 256, 0, st>>>(DRES, DOUT, kDB * kH);
            rms(kDB, DRES, kH, d.post_ln, DN, kH, DH16);
            blt.run(d.gu, 2 * kDF, kH, DH16, kH, kDB, DGU, 2 * kDF, st);
            k_dsilu<<<dim3(kDF / 256, kDB), 256, 0, st>>>(DGU, DA16);
            blt.run(d.dn, kH, kDF, DA16, kDF, kDB, DOUT, kH, st);
            k_dadd<<<kDB * kH / 256, 256, 0, st>>>(DRES, DOUT, kDB * kH);
        }
        rms(kDB, DRES, kH, d_fnorm, DHID, kH);
        head_rows16(DHID + kH, ns);
        with_rows(ns, [&](auto NC) { k_head8m2<decltype(NC)::value><<<kVocabP / 32 / kHeadG, 256, 0, st>>>(head8, XP16, DBASE, 0); });
        k_dmix<<<1, 256, 0, st>>>(FEAT, size_t(tc) * kH, d_alpha + 12, d_mdfl, nullptr, 0, DFR, DACC + 2, nullptr);
        k_dmkin<<<dim3((2 * kH + 255) / 256, ns), 256, 0, st>>>(DHID, DFR, DMK16);
        blt.run(d_wrc, 3 * kDR, 2 * kH, DMK16, 2 * kH, ns, DCTX, 3 * kDR, st);
        for (int s = 0; s < ns; ++s) {
            k_dmstep<<<1, 3 * kDR, 0, st>>>(DCTX, d_rb, d_ws, d_w1, DMS, DLAT, DTOK, s);
            k_dfold<<<kVocabP / 32, 256, 0, st>>>(d_fold, DLAT, d_fbias, DBASE + size_t(s) * kVocabP, DLOG);
            k_argmax64<<<64, 1024, 0, st>>>(DLOG, kVocabP, DTOK + s, best, done);
        }
    }
    // (a test) the 7 drafts after anchor a at position q0, the last known features FEAT row feat_last
    std::vector<int> draft(int a, int q0) {
        const int acc[3] = {0, 0, feat_last};
        CK(hipMemcpyAsync(DACC, acc, 12, hipMemcpyHostToDevice, st));
        CK(hipMemcpyAsync(DTOK, &a, 4, hipMemcpyHostToDevice, st));
        k_setpos<<<1, 1, 0, st>>>(state, q0);
        draft_body(kDB - 1);
        std::vector<int> t(kDB);
        CK(hipMemcpyAsync(t.data(), DTOK, kDB * 4, hipMemcpyDeviceToHost, st));
        CK(hipStreamSynchronize(st));
        return std::vector<int>(t.begin() + 1, t.end());
    }
    // speculation from here: the anchor (the target's next token) is in state[1], its position pos; the draft's context
    // is complete and FEAT row feat_last holds the last token's features
    void spec_begin() {
        const int acc[3] = {pos, 0, feat_last};
        CK(hipMemcpyAsync(DACC, acc, 12, hipMemcpyHostToDevice, st));
        CK(hipMemcpyAsync(DTOK, state + 1, 4, hipMemcpyDeviceToDevice, st));
        k_setpos<<<1, 1, 0, st>>>(state, pos);
    }
    // one cycle (a graph): the last check's rows into the draft's context, the drafts, the check of [anchor, drafts]
    // through the row path, the verdict; returns the tokens written
    std::vector<int> spec_cycle() {
        const int nv = spec_n();
        static const bool ngraph = std::getenv("INKLING_NOGRAPH") != nullptr;
        auto body = [&]() {
            draft_ctx_dev();
            draft_body(nv - 1);
            CK(hipMemcpyAsync(ids_d, DTOK, nv * 4, hipMemcpyDeviceToDevice, st));
            spec_feat = true; spec_rows = true;
            k_embed<<<nv, 256, 0, st>>>(embed, ids_d, embed_norm, EMB);
            for (int v = 0; v < kVisits; ++v) visit(v, nv);
            feat_final(nv);
            finish(0, nv);
            spec_rows = false;
            k_argmax_rows<<<nv, 1024, 0, st>>>(logits, DGT);
            k_accept<<<1, 1, 0, st>>>(DTOK, DGT, nv, state, DACC, DOUTT);
            k_conv_state_put_all<<<dim3(kH / 256, 12, kVisits), 256, 0, st>>>(SPST, CSP, DACC);
            CK(hipMemcpyAsync(hout, DOUTT, (kDB + 1) * 4, hipMemcpyDeviceToHost, st));
        };
        static bool warmed = false;   // (the first cycle runs plainly: hipBLASLt times its methods outside a capture)
        if (ngraph || !warmed) { body(); warmed = true; }
        else {
            if (!vgraph) {
                hipGraph_t gr;
                CK(hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal));
                body();
                CK(hipStreamEndCapture(st, &gr));
                CK(hipGraphInstantiate(&vgraph, gr, nullptr, nullptr, 0));
                CK(hipGraphDestroy(gr));
            }
            CK(hipGraphLaunch(vgraph, st));
        }
        CK(hipStreamSynchronize(st));
        const int n = hout[0];
        pos += n;
        return std::vector<int>(hout + 1, hout + 1 + n);
    }
    // T rows (ids on the card) at position pos; logits of the last row (all rows: every)
    void run_rows(int T, bool every) {
        k_setpos<<<1, 1, 0, st>>>(state, pos);
        k_embed<<<T, 256, 0, st>>>(embed, ids_d, embed_norm, EMB);
        for (int v = 0; v < kVisits; ++v) visit(v, T);
        if (spec_feat) feat_final(T);   // the final normed rows of all T: the draft's 4th feature
        if (every) finish(0, T);
        else finish(T - 1, 1);
        pos += T;
        k_setpos<<<1, 1, 0, st>>>(state, pos);
    }
    void set_pos(int p) {   // (only back to 0 or forward: the convolution states are not rewound)
        pos = p;
        if (p == 0) {
            for (int v = 0; v < kVisits; ++v) {
                CK(hipMemsetAsync(cs_k[v], 0, 3 * kKVD * 4, st)); CK(hipMemsetAsync(cs_v[v], 0, 3 * kKVD * 4, st));
                CK(hipMemsetAsync(cs_a[v], 0, 3 * kH * 4, st)); CK(hipMemsetAsync(cs_m[v], 0, 3 * kH * 4, st));
            }
        }
        k_setpos<<<1, 1, 0, st>>>(state, p);
    }
    // a prompt piece through the chunked path; after(t) per row when every
    void prefill_chunk(const int* ids, int T, const std::function<void(int)>& after = nullptr) {
        CK(hipMemcpyAsync(ids_d, ids, size_t(T) * 4, hipMemcpyHostToDevice, st));
        run_rows(T, bool(after));
        if (after) {
            for (int t = 0; t < T; ++t) {
                CK(hipMemcpyAsync(logits, logits + size_t(t) * kVocabP, kVocabP * 4, hipMemcpyDeviceToDevice, st));
                after(t);
            }
        }
    }
    // the prompt with the draft's features and context K / V kept (speculation)
    void spec_prefill(const std::vector<int>& ids) {
        for (size_t i = 0; i < ids.size(); i += tc) {
            const int T = int(std::min<size_t>(tc, ids.size() - i));
            spec_feat = true; force_pf = !std::getenv("INKLING_SPEC_ROWPF");
            prefill_chunk(ids.data() + i, T);
            spec_feat = false; force_pf = false;
            draft_ctx(0, T, int(i));
            feat_last = T - 1;
        }
    }
    void prefill(const std::vector<int>& ids, const std::function<void(int)>& after = nullptr) {
        for (size_t i = 0; i < ids.size(); i += tc) {
            const int T = int(std::min<size_t>(tc, ids.size() - i));
            prefill_chunk(ids.data() + i, T, after);
        }
    }
    // the writing step: the token in state[1] at state[0]; logits after
    void step_kernels() {
        k_tok_from_state<<<1, 1, 0, st>>>(state, ids_d);
        k_embed<<<1, 256, 0, st>>>(embed, ids_d, embed_norm, EMB);
        for (int v = 0; v < kVisits; ++v) visit(v, 1);
        finish(0, 1);
        k_advance<<<1, 1, 0, st>>>(state, 1);
    }
    void step() {
        static const bool no_graph = std::getenv("INKLING_NOGRAPH") != nullptr;
        if (no_graph) step_kernels();
        else {
            if (!graph) {
                hipGraph_t g;
                CK(hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal));
                step_kernels();
                CK(hipStreamEndCapture(st, &g));
                CK(hipGraphInstantiate(&graph, g, nullptr, nullptr, 0));
                CK(hipGraphDestroy(g));
            }
            CK(hipGraphLaunch(graph, st));
        }
        ++pos;
    }
    // One token of each of the conversations bs (2 .. kMR, row r is bs[r]) through the model together: toks[r] is fed
    // at that conversation's position, logits row r is its next token's. Every weight is read once for all the rows;
    // the K / V caches, attention and convolution states are each conversation's own.
    void step_multi(const std::vector<int>& bs, const int* toks) {
        const int B = int(bs.size());
        slots[size_t(cur)].pos = pos;
        for (int r = 0; r < B; ++r) { mtok_h[r] = toks[r]; mtok_h[kMR + r] = bs[size_t(r)]; }
        CK(hipMemcpyAsync(ids_d, mtok_h, size_t(B) * 4, hipMemcpyHostToDevice, st));
        CK(hipMemcpyAsync(ROWSLOT, mtok_h + kMR, size_t(B) * 4, hipMemcpyHostToDevice, st));
        auto body = [&]() {   // (the same launches whichever conversations the rows are: a graph a row count)
            k_embed<<<B, 256, 0, st>>>(embed, ids_d, embed_norm, EMB);
            mrows = &bs;
            for (int v = 0; v < kVisits; ++v) visit(v, B);
            mrows = nullptr;
            finish(0, B);
            k_advance_rows<<<1, 1, 0, st>>>(SLOTTAB, ROWSLOT, B);
        };
        static const bool no_graph = std::getenv("INKLING_NOGRAPH") != nullptr;
        if (no_graph) body();
        else {
            if (!mgraph[B]) {
                hipGraph_t g;
                CK(hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal));
                body();
                CK(hipStreamEndCapture(st, &g));
                CK(hipGraphInstantiate(&mgraph[B], g, nullptr, nullptr, 0));
                CK(hipGraphDestroy(g));
            }
            CK(hipGraphLaunch(mgraph[B], st));
        }
        for (int r = 0; r < B; ++r) ++slots[size_t(bs[size_t(r)])].pos;
        pos = slots[size_t(cur)].pos;
    }
    // the greedy picks of step_multi's B rows (to the host)
    const int* argmax_rows(int B) {
        k_argmax_rows<<<B, 1024, 0, st>>>(logits, MTOK);
        CK(hipMemcpyAsync(mtok_h, MTOK, size_t(B) * 4, hipMemcpyDeviceToHost, st));
        CK(hipStreamSynchronize(st));
        return mtok_h;
    }
    // tokens one at a time through the writing path; after() per token
    void feed(const std::vector<int>& ids, const std::function<void()>& after = nullptr) {
        for (int t : ids) {
            CK(hipMemcpyAsync(ids_d, &t, 4, hipMemcpyHostToDevice, st));
            k_setpos<<<1, 1, 0, st>>>(state, pos);
            k_embed<<<1, 256, 0, st>>>(embed, ids_d, embed_norm, EMB);
            for (int v = 0; v < kVisits; ++v) visit(v, 1);
            finish(0, 1);
            ++pos;
            k_setpos<<<1, 1, 0, st>>>(state, pos);
            if (after) { CK(hipStreamSynchronize(st)); after(); }
        }
    }
    // the convolution states at the end of the current position, kept aside / put back (with the position)
    float* snap = nullptr;
    size_t snap_bytes() const { return size_t(kVisits) * 3 * (2 * kKVD + 2 * kH) * 4; }
    void snap_copy(bool save) {
        if (!snap) CK(hipMalloc(&snap, snap_bytes()));
        size_t off = 0;
        for (int v = 0; v < kVisits; ++v)
            for (auto [p, n] : {std::pair{cs_k[v], kKVD}, std::pair{cs_v[v], kKVD}, std::pair{cs_a[v], kH}, std::pair{cs_m[v], kH}}) {
                const size_t b = size_t(3) * n * 4;
                if (save) CK(hipMemcpyAsync((char*) snap + off, p, b, hipMemcpyDeviceToDevice, st));
                else CK(hipMemcpyAsync(p, (char*) snap + off, b, hipMemcpyDeviceToDevice, st));
                off += b;
            }
    }
    void save_snap() { snap_copy(true); }
    void restore_snap(int p) { snap_copy(false); pos = p; k_setpos<<<1, 1, 0, st>>>(state, p); }
    // before a server's first request: the hipBLASLt method search for each range of chunk sizes (a few seconds)
    void warm() {
#ifdef INKLING_BLT
        std::vector<int> d(tc, 13);
        for (int b = 256;; b *= 2) {
            const int T = std::min(b, tc);
            if (T >= pf_min) { set_pos(0); prefill_chunk(d.data(), T); }
            if (b >= tc) break;
        }
        CK(hipStreamSynchronize(st));
        set_pos(0);
#endif
    }
    int read_token() {
        CK(hipMemcpyAsync(host_state, state, 8, hipMemcpyDeviceToHost, st));
        CK(hipStreamSynchronize(st));
        return host_state[1];
    }
    int next_token() {
        k_argmax64<<<64, 1024, 0, st>>>(logits, kVocabP, state, best, done);
        CK(hipMemcpyAsync(host_state, state, 8, hipMemcpyDeviceToHost, st));
        CK(hipStreamSynchronize(st));
        return host_state[1];
    }
};

#include "inkling_serve.cuh"

}  // namespace

int main(int argc, char** argv) {
    setenv("GPU_MAX_HW_QUEUES", "2", 0);
    std::string path, ids_s, ids2_s, top_path, logits_path;
    int nslots = 1;
    int n_new = 0, ctx = 32768, chunk = 2048;
    bool serve_mode = false;
    std::vector<int> eos_ids = {200006};   // <|content_model_end_sampling|>
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--model") path = next();
        else if (a == "--ids") {
            ids_s = next();
            if (!ids_s.empty() && ids_s[0] == '@') {
                std::ifstream f(ids_s.substr(1));
                ids_s.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            }
        } else if (a == "--ids2") {   // (the several-conversations test: the odd conversations' prompt)
            ids2_s = next();
            if (!ids2_s.empty() && ids2_s[0] == '@') {
                std::ifstream f(ids2_s.substr(1));
                ids2_s.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            }
        } else if (a == "--parallel") nslots = std::max(1, std::min(kMR, std::stoi(next())));
        else if (a == "--new") n_new = std::stoi(next());
        else if (a == "--top") top_path = next();
        else if (a == "--logits") logits_path = next();
        else if (a == "--max-ctx" || a == "--max-context") ctx = std::stoi(next());
        else if (a == "--chunk") chunk = std::stoi(next());
        else if (a == "--serve") serve_mode = true;
        else if (a == "--eos-ids") {
            eos_ids.clear();
            std::stringstream es(next());
            for (std::string v; std::getline(es, v, ',');) eos_ids.push_back(std::stoi(v));
        }
        else if (a == "--vision") next();
        else if (a == "--vram-reserve-mib") next();
        else { std::fprintf(stderr, "usage: strata-inkling --model DIR --ids 1,2,3|@FILE [--new N] [--top FILE] [--logits FILE] [--max-ctx N]\n"); return 2; }
    }
    if (path.empty() || (ids_s.empty() && !serve_mode)) { std::fprintf(stderr, "strata-inkling: --model and --ids are needed\n"); return 2; }
    std::vector<int> ids;
    { std::stringstream ss(ids_s); for (std::string v; std::getline(ss, v, ',');) if (v.find_first_not_of(" \n\r\t") != std::string::npos) ids.push_back(std::stoi(v)); }
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int> ids2;
    { std::stringstream ss(ids2_s); for (std::string v; std::getline(ss, v, ',');) if (v.find_first_not_of(" \n\r\t") != std::string::npos) ids2.push_back(std::stoi(v)); }
    // --parallel N: N conversations, the context shared between them (each holds --max-ctx / N)
    Model m(path, ctx / nslots, chunk, !serve_mode && (!top_path.empty() || !logits_path.empty()), nslots);
    {
        size_t fr = 0, tot = 0;
        CK(hipMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-inkling: %.2f GB on the card in %.1f s (%zu bf16 values changed in fp16), %.2f GB free; K/V %.2f GB, row buffers %.2f GB\n",
                     m.loaded / 1e9, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), m.fp16_changed, fr / 1e9,
                     m.kv_bytes / 1e9, m.pf_bytes / 1e9);
    }

    if (const char* mb = std::getenv("INKLING_MBENCH")) {   // the few-rows expert kernels alone: U experts of R rows each
        int U = 30, Rr = 2;
        std::sscanf(mb, "%d,%d", &U, &Rr);
        MoeList h{};
        h.cnt = U;
        std::vector<int> idx(kMR * kTopK);
        for (int u = 0; u < U; ++u) {
            h.e[u] = (u * 37 + 5) % kE; h.nr[u] = Rr;
            for (int r = 0; r < Rr; ++r) { h.t[u][r] = r; h.k[u][r] = u % kTopK; }
        }
        for (int j = 0; j < kMR * kTopK; ++j) idx[j] = (j * 37 + 5) % kE;
        CK(hipMemcpy(m.MLIST, &h, sizeof(h), hipMemcpyHostToDevice));
        CK(hipMemcpy(m.idx, idx.data(), idx.size() * 4, hipMemcpyHostToDevice));
        const Experts& ex = m.L[0].ex;
        hipEvent_t e0, e1;
        CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        auto timeit = [&](const char* name, auto&& f) {
            for (int i = 0; i < 3; ++i) f();
            CK(hipEventRecord(e0, m.st));
            for (int i = 0; i < 50; ++i) f();
            CK(hipEventRecord(e1, m.st));
            CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1));
            std::fprintf(stderr, "%-10s %8.1f us\n", name, ms * 1000 / 50);
        };
        timeit("gum", [&]() { k_moe_gum<true><<<dim3(kF / 16, kMR * kTopK), 256, 0, m.st>>>(ex, m.XP16, m.MLIST, m.ACT16); });
        timeit("gum exact", [&]() { k_moe_gum<true><<<dim3(kF / 16, U), 256, 0, m.st>>>(ex, m.XP16, m.MLIST, m.ACT16); });
        timeit("downm", [&]() { k_moe_downm<true><<<dim3(kH / 32, kMR * kTopK), 256, 0, m.st>>>(ex, m.ACT16, m.MLIST, m.YP); });
        timeit("gu2 T=1", [&]() { k_moe_gu2<true><<<dim3(kF / 16, kTopK, 1), 256, 0, m.st>>>(ex, m.XM, m.idx, m.ACT, nullptr); });
        timeit("gu2 T=4", [&]() { k_moe_gu2<true><<<dim3(kF / 16, kTopK, 4), 256, 0, m.st>>>(ex, m.XM, m.idx, m.ACT, nullptr); });
        timeit("down2 T=1", [&]() { k_moe_down2<true><<<dim3(kH / 8, 1), 256, 0, m.st>>>(ex, m.ACT, m.idx, m.wts, m.Y, ConvEpi{}); });
        return 0;
    }
    if (serve_mode) { m.warm(); return nslots > 1 ? serve_multi(m, eos_ids, m.tc) : serve(m, eos_ids, m.tc); }
    if (nslots > 1) {   // (a test) every conversation reads the prompt (the odd ones --ids2's when given), then all write together, greedy
        std::vector<int> bs((size_t) nslots), tok((size_t) nslots);
        std::vector<std::vector<int>> out((size_t) nslots);
        if (!logits_path.empty()) {   // (the accuracy check: the ids fed to every conversation a token a step; the last row's logits)
            FILE* lf2 = std::fopen(logits_path.c_str(), "wb");
            std::vector<float> row(kVocab);
            for (int b = 0; b < nslots; ++b) { m.bind(b); m.set_pos(0); bs[size_t(b)] = b; }
            for (int t : ids) {
                for (int b = 0; b < nslots; ++b) tok[size_t(b)] = t;
                m.step_multi(bs, tok.data());
                CK(hipMemcpy(row.data(), m.logits + size_t(nslots - 1) * kVocabP, row.size() * 4, hipMemcpyDeviceToHost));
                std::fwrite(row.data(), 4, row.size(), lf2);
            }
            std::fclose(lf2);
            return 0;
        }
        const auto t1 = std::chrono::steady_clock::now();
        for (int b = 0; b < nslots; ++b) {
            m.bind(b);
            m.prefill((b & 1) && !ids2.empty() ? ids2 : ids);
            tok[size_t(b)] = m.next_token();
            bs[size_t(b)] = b;
            out[size_t(b)].push_back(tok[size_t(b)]);
        }
        const auto t2 = std::chrono::steady_clock::now();
        for (int n = 1; n < n_new; ++n) {
            m.step_multi(bs, tok.data());
            const int* g = m.argmax_rows(nslots);
            for (int b = 0; b < nslots; ++b) { tok[size_t(b)] = g[b]; out[size_t(b)].push_back(g[b]); }
        }
        CK(hipStreamSynchronize(m.st));
        const double ws = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
        for (int b = 0; b < nslots; ++b) {
            for (size_t j = 0; j < out[size_t(b)].size(); ++j) std::printf("%d%s", out[size_t(b)][j], j + 1 < out[size_t(b)].size() ? "," : "\n");
        }
        std::fprintf(stderr, "strata-inkling: %d conversations read in %.2f s, wrote %d each in %.2f s (%.2f per second in all, %.2f each)\n", nslots,
                     std::chrono::duration<double>(t2 - t1).count(), n_new, ws, double(nslots) * (n_new - 1) / std::max(ws, 1e-9),
                     (n_new - 1) / std::max(ws, 1e-9));
        return 0;
    }
    if (int(ids.size()) > ctx) { std::fprintf(stderr, "strata-inkling: the prompt (%zu tokens) does not fit the context (%d)\n", ids.size(), ctx); return 2; }
    if (int(ids.size()) + n_new > ctx) {
        n_new = ctx - int(ids.size());
        std::fprintf(stderr, "strata-inkling: writing at most %d tokens (the context is %d)\n", n_new, ctx);
    }
    std::vector<float> lg(kVocab);
    FILE* lf = logits_path.empty() ? nullptr : std::fopen(logits_path.c_str(), "wb");
    FILE* top = top_path.empty() ? nullptr : std::fopen(top_path.c_str(), "w");
    auto write_top = [&]() {
        CK(hipMemcpy(lg.data(), m.logits, lg.size() * 4, hipMemcpyDeviceToHost));
        if (lf) std::fwrite(lg.data(), 4, lg.size(), lf);
        if (!top) return;
        double mx = *std::max_element(lg.begin(), lg.end()), s = 0;
        for (float v : lg) s += std::exp(double(v) - mx);
        const double lse = mx + std::log(s);
        std::vector<int> o(lg.size());
        for (size_t i = 0; i < o.size(); ++i) o[i] = int(i);
        std::partial_sort(o.begin(), o.begin() + 5, o.end(), [&](int a, int b) { return lg[a] > lg[b]; });
        for (int j = 0; j < 5; ++j) std::fprintf(top, "%d:%.5f%s", o[j], lg[o[j]] - lse, j < 4 ? " " : "\n");
    };
    const auto t1 = std::chrono::steady_clock::now();
    if (std::getenv("INKLING_TOKWISE")) m.feed(ids, [&]() { if (lf || top) write_top(); });
    else if (lf || top) m.prefill(ids, [&](int) { write_top(); });
    else if (m.spec) m.spec_prefill(ids);
    else m.prefill(ids);
    if (const char* dk = std::getenv("INKLING_DUMP_K")) {   // (the global passes' keys, for tools/inkling/lsa_pca.py)
        CK(hipStreamSynchronize(m.st));
        const size_t n = ids.size();
        for (int v = 0; v < kVisits; ++v) {
            if (!m.K8[v]) continue;
            std::vector<int8_t> k(n * kKVD);
            std::vector<__half> s(n * kKV);
            CK(hipMemcpy(k.data(), m.K8[v], k.size(), hipMemcpyDeviceToHost));
            CK(hipMemcpy(s.data(), m.KS[v], s.size() * 2, hipMemcpyDeviceToHost));
            const std::string fn = std::string(dk) + "_v" + std::to_string(v) + ".bin";
            if (FILE* f = std::fopen(fn.c_str(), "wb")) { int hd[2] = {v, int(n)}; std::fwrite(hd, 4, 2, f); std::fwrite(k.data(), 1, k.size(), f); std::fwrite(s.data(), 2, s.size(), f); std::fclose(f); }
        }
    }
    CK(hipStreamSynchronize(m.st));
    const double read_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    const auto t2 = std::chrono::steady_clock::now();
    if (const char* tf = std::getenv("INKLING_SPEC_TF")) {   // the draft against a known continuation (one chunk): the sim's test
        const int plen = std::atoi(tf), N = int(ids.size());
        std::vector<int> h(kDB, 0);
        long acc = 0, cnt = 0;
        for (int p = plen + 1; p < N - kDB; ++p) {
            m.feat_last = p - 1;
            const auto d = m.draft(ids[p], p);
            int k = 0;
            while (k < kDB - 1 && d[k] == ids[p + 1 + k]) ++k;
            ++h[k]; acc += k; ++cnt;
            m.draft_ctx(p, kDB, p);   // (the block's own K / V back to the context's)
        }
        std::fprintf(stderr, "teacher-forced: %ld anchors, mean taken %.2f, hist", cnt, double(acc) / cnt);
        for (int x : h) std::fprintf(stderr, " %d", x);
        std::fprintf(stderr, "\n");
        return 0;
    }
    long cycles = 0, accepted = 0;
    std::vector<int> hist(kDB, 0);
    if (m.spec && n_new > 0) {   // speculation: the draft's 7 tokens checked 8 rows at a time (greedy)
        int a = m.next_token(), n = 0;
        auto put = [&](int t) { std::printf("%d%s", t, ++n < n_new ? "," : "\n"); std::fflush(stdout); };
        put(a);
        m.spec_begin();
        while (n < n_new) {
            if (m.pos + kDB > ctx) {   // (the end of the context: one row at a time)
                m.feed({a});
                a = m.next_token();
                put(a);
                continue;
            }
            static const bool sp = std::getenv("INKLING_SPEC_PROF") != nullptr;
            const auto c0 = std::chrono::steady_clock::now();
            const auto c1 = std::chrono::steady_clock::now();
            const auto out = m.spec_cycle();
            if (sp) {
                CK(hipStreamSynchronize(m.st));
                const auto c2 = std::chrono::steady_clock::now();
                std::fprintf(stderr, "cycle %.2f ms %.2f ms took %zu\n", std::chrono::duration<double, std::milli>(c1 - c0).count(),
                             std::chrono::duration<double, std::milli>(c2 - c1).count(), out.size());
            }
            ++cycles; accepted += long(out.size()) - 1; ++hist[out.size() - 1];
            for (int t : out) if (n < n_new) put(t);
            a = out.back();
        }
    } else
    for (int n = 0; n < n_new; ++n) {
        const int nxt = m.next_token();
        std::printf("%d%s", nxt, n + 1 < n_new ? "," : "\n");
        std::fflush(stdout);
        m.step();
    }
    CK(hipStreamSynchronize(m.st));
    const double write_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
    if (lf) std::fclose(lf);
    if (top) std::fclose(top);
    std::fprintf(stderr, "strata-inkling: read %zu in %.2f s (%.1f per second), wrote %d in %.2f s (%.2f per second)\n", ids.size(), read_s,
                 ids.size() / read_s, n_new, write_s, n_new / std::max(write_s, 1e-9));
    if (cycles) {
        std::fprintf(stderr, "strata-inkling: speculation: %ld checks, %.2f drafts taken a check (%.2f tokens), hist", cycles,
                     double(accepted) / cycles, 1.0 + double(accepted) / cycles);
        for (int h : hist) std::fprintf(stderr, " %d", h);
        std::fprintf(stderr, "\n");
    }
    return 0;
}
