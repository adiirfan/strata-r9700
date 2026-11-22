// src/gemma4/gemma4_main.cu - strata-gemma4: Google Gemma 4 26B-A4B (and fine-tunes such as TheDrummer's Orion) from its
// Q8_0 GGUF, the whole model on one 32 GB card.
//
//   strata-gemma4 --model FILE.gguf --ids 1,2,3|@FILE [--new N] [--top FILE] [--logits FILE] [--max-ctx N] [--serve]
//
// The model (llama.cpp src/models/gemma4.cpp is the spec): 30 layers, hidden 2816, embeddings times sqrt(2816), tied
// output head with a tanh softcap of 30. Attention: 16 query heads; five of every six layers slide over the last 1024
// positions with 8 K/V heads of 256 (NEOX RoPE base 1e4), every sixth sees everything with 2 K/V heads of 512 (base 1e6,
// only the first 64 of 256 frequency pairs turn: rope_freqs) and no V projection (V is K before its norm). Q and K are
// RMS-normed with weights, V without; the attention scale is 1. Feed-forward: a dense GELU MLP (2112) and 128 experts
// (top 8, GELU, 704, gate / up fused) side by side, each with its own norms; the router reads rms(x) / sqrt(2816) times
// a scale, softmax over all 128, the top 8 renormalised and times the expert's down scale; then a norm, the residual,
// and the layer's output scale.
//
// Weights stay in their stored values: Q8_0 repacked as int8 rows plus fp16 scales. K / V are fp16.
#include "strata/artifact/gguf_reader.hpp"

#ifdef GEMMA4_BLT
// (the HIP shim maps CUDA shuffle spellings to Strata helpers; hipBLASLt's headers declare the native ones)
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>
#include <tuple>
#endif

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "strata-gemma4: %s failed: %s (%s:%d)\n", #x, hipGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

constexpr int kHidden = 2816, kLayers = 30, kHeads = 16, kExperts = 128, kTopK = 8, kFe = 704, kFs = 2112, kWindow = 1024;
constexpr int kHdS = 256, kKvS = 8, kHdG = 512, kKvG = 2;
constexpr int kQkvS = kHeads * kHdS + 2 * kKvS * kHdS;   // 8192: q, k, v
constexpr int kQkvG = kHeads * kHdG + kKvG * kHdG;       // 9216: q, k (v is k)
constexpr int kQMax = kHeads * kHdG;                     // 8192
constexpr float kEps = 1e-6f, kCap = 30.f;
constexpr int kF32 = 0, kQ8_0 = 8;
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
__device__ __forceinline__ float gelu(float x) {   // tanh form, as ggml: 0.5 x (1 + tanh(u)) written as x / (1 + e^-2u)
    const float u = 0.7978845608028654f * x * (1.f + 0.044715f * x * x);   // (tanhf was a quarter of the expert kernel)
    return __fdividef(x, 1.f + exp2f(-2.8853900817779268f * u));
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

// x (n values) into LDS, rmsnorm(x) * nw when nw; 256 threads
__device__ inline void stage_in(float* xs, const float* x, const float* nw, int n) {
    float ss = 0;
    for (int i = threadIdx.x; i < n; i += 256) { const float v = x[i]; xs[i] = v; ss += v * v; }
    if (!nw) { __syncthreads(); return; }
    const float r = rsqrtf(block_sum(ss) / n + kEps);
    for (int i = threadIdx.x; i < n; i += 256) xs[i] = xs[i] * r * nw[i];
    __syncthreads();
}

// ---- Q8_0 as stored on the card: int8 rows (K bytes) and fp16 scales (K / 32 a row)
struct W8 { const int8_t* q; const __half* d; };
// a lane's share of R rows . x: lane takes 16-value pieces lane, lane + 32, ...
template <int R>
__device__ inline void q8_rows(W8 W, int K, size_t r0, const float* xs, int lane, float* acc) {
    const int nc = K / 16;
    for (int c = lane; c < nc; c += 32) {
        float xv[16];
#pragma unroll
        for (int i = 0; i < 16; i += 4) *(float4*) (xv + i) = *(const float4*) (xs + c * 16 + i);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint4 w = *(const uint4*) (W.q + (r0 + r) * K + c * 16);
            const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
            float t = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int b = 0; b < 4; ++b) t += float(int8_t(ww[j] >> (8 * b))) * xv[j * 4 + b];
            acc[r] += __half2float(W.d[(r0 + r) * (K / 32) + (c >> 1)]) * t;
        }
    }
}

// ---------------------------------------------------------------------------------------------------- kernels
constexpr int kMvRows = 32, kMvRpw = 4;   // a block: 8 warps of 4 rows
// y (rows) = W . x, x = rmsnorm(x) * nw when nw; 32 rows a block
__global__ void __launch_bounds__(256) k_mv8(W8 W, const float* x, const float* nw, int K, float* y) {
    extern __shared__ float xs[];
    stage_in(xs, x, nw, K);
    const int lane = threadIdx.x & 31, r0 = blockIdx.x * kMvRows + (threadIdx.x >> 5) * kMvRpw;
    float a[kMvRpw] = {};
    q8_rows<kMvRpw>(W, K, r0, xs, lane, a);
#pragma unroll
    for (int r = 0; r < kMvRpw; ++r) {
        const float v = warp_sum(a[r]);
        if (lane == 0) y[r0 + r] = v;
    }
}
// act[r] = gelu(gate_r . x) * (up_r . x), gate rows 0 .. F-1 and up rows F .. 2F-1 of W; x = rmsnorm(x) * nw
__global__ void __launch_bounds__(256) k_glu8(W8 W, int F, const float* x, const float* nw, float* act) {
    extern __shared__ float xs[];
    stage_in(xs, x, nw, kHidden);
    const int lane = threadIdx.x & 31, r0 = blockIdx.x * kMvRows + (threadIdx.x >> 5) * kMvRpw;
    float g[kMvRpw] = {}, u[kMvRpw] = {};
    q8_rows<kMvRpw>(W, kHidden, r0, xs, lane, g);
    q8_rows<kMvRpw>(W, kHidden, size_t(F) + r0, xs, lane, u);
#pragma unroll
    for (int r = 0; r < kMvRpw; ++r) {
        const float gv = warp_sum(g[r]), uv = warp_sum(u[r]);
        if (lane == 0) act[r0 + r] = gelu(gv) * uv;
    }
}
// the token's row of the (tied) Q8_0 table, times sqrt(hidden)
__global__ void k_embed(W8 T, const int* state, float* h) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= kHidden) return;
    const size_t r = size_t(state[1]);
    h[i] = __half2float(T.d[r * (kHidden / 32) + i / 32]) * float(T.q[r * kHidden + i]) * sqrtf(float(kHidden));
}

// RoPE angle pos * inv (in double, reduced to one turn) as cos / sin
__device__ __forceinline__ void rope_cs(int pos, double inv, float& c, float& s) {
    double a = double(pos) * inv;
    a -= 6.283185307179586 * rint(a * 0.15915494309189535);
    sincosf(float(a), &s, &c);
}
// After the q / k / v product of a writing step: q heads normed (weights qn) and turned, in place; k heads normed (kn)
// and turned, v heads normed (no weights), both into the cache at pos. grid nq + nkv, HD / 2 threads (thread j: the
// NEOX pair j, j + HD / 2). The global layers have no v: it is k before its norm (voff == koff).
template <int HD>
__global__ void __launch_bounds__(HD / 2) k_qk_dec(float* qkv, const float* qn, const float* kn, const double* inv, kc_t* K,
                                                   kc_t* V, const int* state, int kv_mask, int nkv, int koff, int voff) {
    const int b = blockIdx.x, j = threadIdx.x, pos = state[0];
    float c, s;
    rope_cs(pos, inv[j], c, s);
    if (b < kHeads) {
        float* x = qkv + b * HD;
        const float a = x[j], e = x[j + HD / 2];
        const float r = rsqrtf(block_sum(a * a + e * e) / HD + kEps);
        const float a1 = a * r * qn[j], e1 = e * r * qn[j + HD / 2];
        x[j] = a1 * c - e1 * s;
        x[j + HD / 2] = e1 * c + a1 * s;
    } else {
        const int kh = b - kHeads;
        const float* x = qkv + koff + kh * HD;
        const float* v = qkv + voff + kh * HD;
        const float a = x[j], e = x[j + HD / 2], va = v[j], ve = v[j + HD / 2];
        const float r = rsqrtf(block_sum(a * a + e * e) / HD + kEps);
        const float rv = rsqrtf(block_sum(va * va + ve * ve) / HD + kEps);
        const float a1 = a * r * kn[j], e1 = e * r * kn[j + HD / 2];
        const size_t row = size_t(pos & kv_mask) * nkv * HD + kh * HD;
        K[row + j] = __float2half(a1 * c - e1 * s);
        K[row + j + HD / 2] = __float2half(e1 * c + a1 * s);
        V[row + j] = __float2half(va * rv);
        V[row + j + HD / 2] = __float2half(ve * rv);
    }
}
// Attention for the writing step: block (K/V head kh, chunk c), warp w its query head kh * G + w, lane l the dims
// l + 32 i. Each chunk's (max, sum, output) goes to part; k_attn_merge adds the chunks.
constexpr int kAttnChunks = 64, kMinChunk = 16;   // (chunks: the most a layer kind uses, for the part buffer)
constexpr int kChunksG = 64, kChunksS = 16;
template <int HD, int G, int NC>
__global__ void __launch_bounds__(G * 32) k_attn_dec(const float* qkv, const kc_t* K, const kc_t* V, float* part,
                                                     const int* state, int swa, int kv_mask) {
    constexpr int NKV = kHeads / G, PER = HD / 32, KB = 2048 / HD;   // KB keys a step: their loads in flight together
    const int kh = blockIdx.x, c = blockIdx.y, w = threadIdx.x >> 5, l = threadIdx.x & 31, h = kh * G + w;
    const int pos = state[0], lo = swa ? max(0, pos - (kWindow - 1)) : 0, n = pos - lo + 1;
    const int per = max(kMinChunk, (n + NC - 1) / NC), a = lo + c * per, bnd = min(pos + 1, a + per);
    float q[PER], o[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) { q[i] = qkv[h * HD + l + 32 * i]; o[i] = 0.f; }
    float m = -INFINITY, s = 0;
    for (int p0 = a; p0 < bnd; p0 += KB) {
        float d[KB];
#pragma unroll
        for (int k = 0; k < KB; ++k) {
            const size_t r = size_t(min(p0 + k, bnd - 1) & kv_mask) * NKV * HD + kh * HD + l;
            float t = 0;
#pragma unroll
            for (int i = 0; i < PER; ++i) t += q[i] * __half2float(K[r + 32 * i]);
            d[k] = t;
        }
#pragma unroll
        for (int off = 16; off; off >>= 1)
#pragma unroll
            for (int k = 0; k < KB; ++k) d[k] += __shfl_xor(d[k], off, 32);
        float mx = m;
#pragma unroll
        for (int k = 0; k < KB; ++k) { if (p0 + k >= bnd) d[k] = -INFINITY; mx = fmaxf(mx, d[k]); }
        const float cf = expf(m - mx);
        s *= cf;
#pragma unroll
        for (int i = 0; i < PER; ++i) o[i] *= cf;
#pragma unroll
        for (int k = 0; k < KB; ++k) {
            const float e = expf(d[k] - mx);
            s += e;
            const size_t r = size_t(min(p0 + k, bnd - 1) & kv_mask) * NKV * HD + kh * HD + l;
#pragma unroll
            for (int i = 0; i < PER; ++i) o[i] += e * __half2float(V[r + 32 * i]);
        }
        m = mx;
    }
    float* P = part + (size_t(h) * NC + c) * (HD + 4);
    if (l == 0) { P[0] = m; P[1] = s; }
#pragma unroll
    for (int i = 0; i < PER; ++i) P[4 + l + 32 * i] = o[i];
}
template <int HD, int NC>
__global__ void k_attn_merge(const float* part, float* out) {   // HD threads a head
    const int h = blockIdx.x, t = threadIdx.x;
    const float* P = part + size_t(h) * NC * (HD + 4);
    float M = -INFINITY;
    for (int c = 0; c < NC; ++c) M = fmaxf(M, P[c * (HD + 4)]);
    float L = 0, O = 0;
    for (int c = 0; c < NC; ++c) {
        const float mc = P[c * (HD + 4)];
        if (mc == -INFINITY) continue;
        const float f = expf(mc - M);
        L += P[c * (HD + 4) + 1] * f;
        O += P[c * (HD + 4) + 4 + t] * f;
    }
    out[h * HD + t] = O / L;
}
// out = rmsnorm(x) * w + res (one block of 1024)
__global__ void __launch_bounds__(1024) k_norm_add(float* out, const float* x, const float* w, const float* res) {
    float ss = 0;
    for (int i = threadIdx.x; i < kHidden; i += 1024) ss += x[i] * x[i];
    const float r = rsqrtf(block_sum(ss) / kHidden + kEps);
    for (int i = threadIdx.x; i < kHidden; i += 1024) out[i] = x[i] * r * w[i] + res[i];
}
// router logits: rms(x) / sqrt(hidden) * scale . R[e]; 2 experts a block, 4 warps an expert (more blocks and more
// loads in flight: the fp32 router is 1.4 MB a layer, read by 32 blocks it took 10.7 us)
__global__ void __launch_bounds__(256) k_router(const float* x, const float* scale, const float* R, float* logits) {
    __shared__ float xs[kHidden];
    __shared__ float red[8];
    float ss = 0;
    for (int i = threadIdx.x; i < kHidden; i += 256) { const float v = x[i]; xs[i] = v; ss += v * v; }
    const float r = rsqrtf(block_sum(ss) / kHidden + kEps) * rsqrtf(float(kHidden));
    for (int i = threadIdx.x; i < kHidden; i += 256) xs[i] *= r * scale[i];
    __syncthreads();
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, e = blockIdx.x * 2 + (w >> 2), q = w & 3;
    const float* Re = R + size_t(e) * kHidden;
    float a = 0;
    // the expert's row in 4 quarters of 704 (176 float4), a quarter a warp
#pragma unroll 6
    for (int j = l; j < 176; j += 32) {
        const int i = q * 704 + j * 4;
        const float4 wv = *(const float4*) (Re + i);
        a += wv.x * xs[i] + wv.y * xs[i + 1] + wv.z * xs[i + 2] + wv.w * xs[i + 3];
    }
    a = warp_sum(a);
    if (l == 0) red[w] = a;
    __syncthreads();
    if (threadIdx.x < 2) logits[blockIdx.x * 2 + threadIdx.x] = red[threadIdx.x * 4] + red[threadIdx.x * 4 + 1] + red[threadIdx.x * 4 + 2] + red[threadIdx.x * 4 + 3];
}
// k_norm_add + k_glu8 + k_router in one launch (two launches and two kernels fewer a layer): every block forms
// aout = rmsnorm(tmp) * post_attn + h itself (block 0 also stores it for the experts and the layer's end); blocks
// 0 .. kFs / 32 - 1 are k_glu8's, the next kExperts / 2 k_router's. Same sums in the same order as the three kernels.
__global__ void __launch_bounds__(256) k_glu8_router(W8 W, const float* tmp, const float* post_attn, const float* h,
                                                      float* aout, const float* ffn_norm, float* act, const float* gate_scale,
                                                      const float* R, float* logits) {
    extern __shared__ float xs[];
    __shared__ float red[8];
    float ss = 0;
    for (int i = threadIdx.x; i < kHidden; i += 256) { const float v = tmp[i]; xs[i] = v; ss += v * v; }
    const float r1 = rsqrtf(block_sum(ss) / kHidden + kEps);
    float s2 = 0;
    for (int i = threadIdx.x; i < kHidden; i += 256) {
        const float a = xs[i] * r1 * post_attn[i] + h[i];
        xs[i] = a;
        s2 += a * a;
        if (blockIdx.x == 0) aout[i] = a;
    }
    const float tot2 = block_sum(s2);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (blockIdx.x < kFs / kMvRows) {
        const float r2 = rsqrtf(tot2 / kHidden + kEps);
        for (int i = threadIdx.x; i < kHidden; i += 256) xs[i] = xs[i] * r2 * ffn_norm[i];
        __syncthreads();
        const int r0 = blockIdx.x * kMvRows + w * kMvRpw;
        float g[kMvRpw] = {}, u[kMvRpw] = {};
        q8_rows<kMvRpw>(W, kHidden, r0, xs, lane, g);
        q8_rows<kMvRpw>(W, kHidden, size_t(kFs) + r0, xs, lane, u);
#pragma unroll
        for (int r = 0; r < kMvRpw; ++r) {
            const float gv = warp_sum(g[r]), uv = warp_sum(u[r]);
            if (lane == 0) act[r0 + r] = gelu(gv) * uv;
        }
        return;
    }
    const float r = rsqrtf(tot2 / kHidden + kEps) * rsqrtf(float(kHidden));
    for (int i = threadIdx.x; i < kHidden; i += 256) xs[i] *= r * gate_scale[i];
    __syncthreads();
    const int b = blockIdx.x - kFs / kMvRows, e = b * 2 + (w >> 2), q = w & 3;
    const float* Re = R + size_t(e) * kHidden;
    float a = 0;
#pragma unroll 6
    for (int j = lane; j < 176; j += 32) {
        const int i = q * 704 + j * 4;
        const float4 wv = *(const float4*) (Re + i);
        a += wv.x * xs[i] + wv.y * xs[i + 1] + wv.z * xs[i + 2] + wv.w * xs[i + 3];
    }
    a = warp_sum(a);
    if (lane == 0) red[w] = a;
    __syncthreads();
    if (threadIdx.x < 2) logits[b * 2 + threadIdx.x] = red[threadIdx.x * 4] + red[threadIdx.x * 4 + 1] + red[threadIdx.x * 4 + 2] + red[threadIdx.x * 4 + 3];
}
// softmax over the 128 logits, the top 8 (ties: the lower index), their probabilities renormalised and times the
// expert's down scale. One warp.
__device__ inline void topk_warp(const float* lg, const float* dscale, int* idx, float* wts) {
    const int lane = threadIdx.x & 31;
    float sc[4];
    for (int j = 0; j < 4; ++j) sc[j] = lg[lane * 4 + j];
    float mx = warp_max(fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[2], sc[3])));
    float picked[kTopK];
    int pi[kTopK];
    for (int k = 0; k < kTopK; ++k) {
        float v = -INFINITY;
        int i = 0x7fffffff;
        for (int j = 0; j < 4; ++j) if (sc[j] > v) { v = sc[j]; i = lane * 4 + j; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(v, o, 32);
            const int oi = __shfl_xor(i, o, 32);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
        }
        if (lane == i / 4) sc[i % 4] = -INFINITY;
        picked[k] = v; pi[k] = i;
    }
    float ex[kTopK], tot = 0;
    for (int k = 0; k < kTopK; ++k) { ex[k] = expf(picked[k] - mx); tot += ex[k]; }
    if (lane < kTopK) { idx[lane] = pi[lane]; wts[lane] = ex[lane] / tot * dscale[pi[lane]]; }
}
__global__ void k_topk(const float* lg, const float* dscale, int* idx, float* wts) { topk_warp(lg, dscale, idx, wts); }
// the chosen experts' gate / up: act[k][r] = gelu(gate . xn) * (up . xn), xn = rmsnorm(x) * nw. GU holds every expert's
// 1408 rows (gate 704, up 704) one after another; grid kTopK * kFe / 32
__global__ void __launch_bounds__(256) k_moe_gu(W8 GU, const float* x, const float* nw, const int* idx_in, float* act,
                                                const float* rl = nullptr, const float* dscale = nullptr, int* idx_out = nullptr,
                                                float* wts_out = nullptr) {
    // rl: the top 8 picked here (every block the same; block 0 stores them for k_moe_down / k_ffn_out): k_topk's launch saved
    extern __shared__ float xs[];
    __shared__ int sidx[kTopK];
    __shared__ float swts[kTopK];
    const int* idx = idx_in;
    if (rl) {
        if (threadIdx.x < 32) {
            topk_warp(rl, dscale, sidx, swts);
            if (blockIdx.x == 0 && threadIdx.x < kTopK) { idx_out[threadIdx.x] = sidx[threadIdx.x]; wts_out[threadIdx.x] = swts[threadIdx.x]; }
        }
        idx = sidx;
    }
    stage_in(xs, x, nw, kHidden);
    const int k = blockIdx.x / (kFe / kMvRows), r0 = (blockIdx.x % (kFe / kMvRows)) * kMvRows + (threadIdx.x >> 5) * kMvRpw;
    const int lane = threadIdx.x & 31;
    const size_t base = size_t(idx[k]) * 2 * kFe;
    float g[kMvRpw] = {}, u[kMvRpw] = {};
    q8_rows<kMvRpw>(GU, kHidden, base + r0, xs, lane, g);
    q8_rows<kMvRpw>(GU, kHidden, base + kFe + r0, xs, lane, u);
#pragma unroll
    for (int r = 0; r < kMvRpw; ++r) {
        const float gv = warp_sum(g[r]), uv = warp_sum(u[r]);
        if (lane == 0) act[k * kFe + r0 + r] = gelu(gv) * uv;
    }
}
// y[k] = down_k . act_k; D holds every expert's 2816 rows of 704; grid (kHidden / 32, kTopK)
__global__ void __launch_bounds__(256) k_moe_down(W8 D, const float* act, const int* idx, float* y) {
    __shared__ float xs[kFe];
    const int k = blockIdx.y;
    stage_in(xs, act + k * kFe, nullptr, kFe);
    const int lane = threadIdx.x & 31, r0 = blockIdx.x * kMvRows + (threadIdx.x >> 5) * kMvRpw;
    float a[kMvRpw] = {};
    q8_rows<kMvRpw>(D, kFe, size_t(idx[k]) * kHidden + r0, xs, lane, a);
#pragma unroll
    for (int r = 0; r < kMvRpw; ++r) {
        const float v = warp_sum(a[r]);
        if (lane == 0) y[k * kHidden + r0 + r] = v;
    }
}
// the shared MLP's down (blocks 0 .. kHidden / 32 - 1, k_mv8's) and the experts' down (the next kTopK * kHidden / 32,
// k_moe_down's) in one launch: independent products on different weights
__global__ void __launch_bounds__(256) k_down2(W8 Ws, const float* act_s, float* mlp, W8 D, const float* act, const int* idx, float* y) {
    __shared__ float xs[kFs];
    const int lane = threadIdx.x & 31;
    constexpr int NB = kHidden / kMvRows;
    if (blockIdx.x < NB) {
        stage_in(xs, act_s, nullptr, kFs);
        const int r0 = blockIdx.x * kMvRows + (threadIdx.x >> 5) * kMvRpw;
        float a[kMvRpw] = {};
        q8_rows<kMvRpw>(Ws, kFs, r0, xs, lane, a);
#pragma unroll
        for (int r = 0; r < kMvRpw; ++r) {
            const float v = warp_sum(a[r]);
            if (lane == 0) mlp[r0 + r] = v;
        }
        return;
    }
    const int b = blockIdx.x - NB, k = b / NB;
    stage_in(xs, act + k * kFe, nullptr, kFe);
    const int r0 = (b % NB) * kMvRows + (threadIdx.x >> 5) * kMvRpw;
    float a[kMvRpw] = {};
    q8_rows<kMvRpw>(D, kFe, size_t(idx[k]) * kHidden + r0, xs, lane, a);
#pragma unroll
    for (int r = 0; r < kMvRpw; ++r) {
        const float v = warp_sum(a[r]);
        if (lane == 0) y[k * kHidden + r0 + r] = v;
    }
}
// h = (rmsnorm(rmsnorm(mlp) * n1 + rmsnorm(sum_k w_k y_k) * n2) * nf + a) * scale (one block of 1024)
__global__ void __launch_bounds__(1024) k_ffn_out(float* h, const float* a, const float* mlp, const float* y, const float* wts,
                                                  const float* n1, const float* n2, const float* nf, const float* scale) {
    __shared__ float moe[kHidden], t[kHidden];
    float s1 = 0, s2 = 0;
    for (int i = threadIdx.x; i < kHidden; i += 1024) {
        float v = 0;
#pragma unroll
        for (int k = 0; k < kTopK; ++k) v += wts[k] * y[k * kHidden + i];
        moe[i] = v;
        s1 += mlp[i] * mlp[i];
        s2 += v * v;
    }
    const float r1 = rsqrtf(block_sum(s1) / kHidden + kEps), r2 = rsqrtf(block_sum(s2) / kHidden + kEps);
    float s3 = 0;
    for (int i = threadIdx.x; i < kHidden; i += 1024) {
        const float v = mlp[i] * r1 * n1[i] + moe[i] * r2 * n2[i];
        t[i] = v;
        s3 += v * v;
    }
    const float r3 = rsqrtf(block_sum(s3) / kHidden + kEps), sc = *scale;
    for (int i = threadIdx.x; i < kHidden; i += 1024) h[i] = (t[i] * r3 * nf[i] + a[i]) * sc;
}
__global__ void k_softcap(float* lg, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) lg[i] = tanhf(lg[i] / kCap) * kCap;
}

// the most likely next token into state[1], and the position moves on (state[0]): 64 blocks each take a slice, the
// last one to finish picks among their 64 bests
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
    state[0] += 1;
    *done = 0;
}

#include "gemma4_prefill.cuh"

#ifdef GEMMA4_BLT
// Q8_0 rows to fp16 (d * q rounded once: the same numbers q8_cvt gives the hand-written products), 8 values a thread
__global__ void __launch_bounds__(256) k_deq16(W8 W, size_t n8, uint4* __restrict__ out) {
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < n8; i += size_t(gridDim.x) * 256) {
        const pf_v8h h = q8_cvt(*(const uint2*) (W.q + i * 8), __half_as_ushort(W.d[i >> 2]));
        out[i] = __builtin_bit_cast(uint4, h);
    }
}
// The prompt path's dense products on hipBLASLt (fp16 in, fp32 sums and out): Y (T x N) = X (T x K) . W^T, W the fp16
// copy above. Per shape the heuristic's first 8 methods are timed once and the fastest kept (GEMMA4_BLT_TUNE=0: the
// first). Measured on the R9700 at 4096 tokens: 130-180 TFLOPs where the hand-written Q8 kernel reaches 80-93.
struct Blt {
    hipblasLtHandle_t h = nullptr;
    void* ws = nullptr;
    size_t wss = 32u << 20;
    struct Plan { hipblasLtMatmulDesc_t md; hipblasLtMatrixLayout_t la, lb, lc; hipblasLtMatmulAlgo_t algo; };
    std::map<std::tuple<int, int, int>, Plan> plans;           // per exact shape
    std::map<std::tuple<int, int, int>, hipblasLtMatmulAlgo_t> picks;   // per (N, K, token count rounded up to a power of 2)
#define BK(x) do { hipblasStatus_t e_ = (x); if (e_ != HIPBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "strata-gemma4: %s failed: %d (%s:%d)\n", #x, int(e_), __FILE__, __LINE__); std::exit(1); } } while (0)
    void init() { BK(hipblasLtCreate(&h)); CK(hipMalloc(&ws, wss)); }
    // the fastest of the heuristic's first few methods for this shape, timed on the real call
    hipblasLtMatmulAlgo_t pick(Plan& p, const __half* W, const uint16_t* X, float* Y, hipStream_t st) {
        const float al = 1.f, be = 0.f;
        hipblasLtMatmulPreference_t pref;
        BK(hipblasLtMatmulPreferenceCreate(&pref));
        BK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wss, sizeof wss));
        hipblasLtMatmulHeuristicResult_t res[6];
        int n = 0;
        BK(hipblasLtMatmulAlgoGetHeuristic(h, p.md, p.la, p.lb, p.lc, p.lc, pref, 6, res, &n));
        hipblasLtMatmulPreferenceDestroy(pref);
        if (n == 0) { std::fprintf(stderr, "strata-gemma4: hipBLASLt has no method for this product\n"); std::exit(1); }
        static const bool tune = !std::getenv("GEMMA4_BLT_TUNE") || std::atoi(std::getenv("GEMMA4_BLT_TUNE")) != 0;
        int best = 0;
        if (tune && n > 1) {
            hipEvent_t a, b;
            CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
            float bt = 1e30f;
            for (int i = 0; i < n; ++i) {
                if (hipblasLtMatmul(h, p.md, &al, W, p.la, X, p.lb, &be, Y, p.lc, Y, p.lc, &res[i].algo, ws, wss, st) != HIPBLAS_STATUS_SUCCESS) continue;
                CK(hipEventRecord(a, st));
                for (int r = 0; r < 2; ++r) hipblasLtMatmul(h, p.md, &al, W, p.la, X, p.lb, &be, Y, p.lc, Y, p.lc, &res[i].algo, ws, wss, st);
                CK(hipEventRecord(b, st));
                CK(hipEventSynchronize(b));
                float ms = 0;
                CK(hipEventElapsedTime(&ms, a, b));
                if (ms < bt) { bt = ms; best = i; }
            }
            CK(hipEventDestroy(a)); CK(hipEventDestroy(b));
        }
        return res[best].algo;
    }
    void run(const __half* W, int N, int K, const uint16_t* X, int T, float* Y, hipStream_t st) {
        const float al = 1.f, be = 0.f;
        auto it = plans.find({N, K, T});
        if (it == plans.end()) {
            Plan p;
            BK(hipblasLtMatmulDescCreate(&p.md, HIPBLAS_COMPUTE_32F, HIP_R_32F));
            const hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
            BK(hipblasLtMatmulDescSetAttribute(p.md, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
            BK(hipblasLtMatmulDescSetAttribute(p.md, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
            BK(hipblasLtMatrixLayoutCreate(&p.la, HIP_R_16F, K, N, K));   // W row-major N x K = column-major K x N
            BK(hipblasLtMatrixLayoutCreate(&p.lb, HIP_R_16F, K, T, K));   // X row-major T x K
            BK(hipblasLtMatrixLayoutCreate(&p.lc, HIP_R_32F, N, T, N));   // Y row-major T x N
            // one timing per power-of-2 range of token counts: a server's chunks end at any length
            int b = 256;
            while (b < T) b *= 2;
            auto pk = picks.find({N, K, b});
            bool ok = false;
            if (pk != picks.end()) {
                p.algo = pk->second;
                size_t need = 0;
                ok = hipblaslt_ext::matmulIsAlgoSupported(h, p.md, &al, p.la, p.lb, &be, p.lc, p.lc, p.algo, need) == HIPBLAS_STATUS_SUCCESS && need <= wss;
            }
            if (!ok) {
                p.algo = pick(p, W, X, Y, st);
                if (pk == picks.end()) picks.emplace(std::make_tuple(N, K, b), p.algo);
            }
            it = plans.emplace(std::make_tuple(N, K, T), p).first;
        }
        Plan& p = it->second;
        BK(hipblasLtMatmul(h, p.md, &al, W, p.la, X, p.lb, &be, Y, p.lc, Y, p.lc, &p.algo, ws, wss, st));
    }
};
#endif

// ---------------------------------------------------------------------------------------------------- the model
struct Layer {
    bool global = false;
    int hd = 0, nkv = 0, qkv_rows = 0;
    const float *attn_norm, *q_norm, *k_norm, *post_attn, *ffn_norm, *post1, *pre2, *post2, *post_ffn, *out_scale;
    const float *gate_scale, *router, *router_t, *dscale;
    const float *rnw, *rsc;                  // the router's input norm weights (scale / sqrt(hidden)); per expert 2^-s
    const __half *rh, *rl;                   // the router as hi + lo fp16, each row times 2^s
    W8 wqkv, wo, mlp_gu, mlp_down, gu, down;
    kc_t *K = nullptr, *V = nullptr;
    int kv_mask = 0x7fffffff;
};

struct Model {
    std::unique_ptr<strata::GgufFile> g;
    std::vector<Layer> L;
    W8 head{};                        // the token table, also the output head
    const float* out_norm = nullptr;
    double *inv_s = nullptr, *inv_g = nullptr;   // RoPE inverse frequencies: sliding (128), global (256)
    int vocab = 0, max_ctx;
    size_t loaded = 0, kv_bytes = 0;
    hipStream_t st;
    float *h, *qkv, *att, *tmp, *aout, *act_s, *mlp, *rl, *wts, *act, *ye, *part, *logits;
    int* idx;
    float2* best;
    unsigned* done;
    int pos = 0;
    int* state;        // on the card: [0] the position, [1] the token going in next
    int* host_state;
    hipGraphExec_t graph = nullptr;

    const strata::TensorInfo* need(const std::string& name, int type) {
        const strata::TensorInfo* t = g->find(name);
        if (!t) { std::fprintf(stderr, "strata-gemma4: %s is missing\n", name.c_str()); std::exit(1); }
        if (int(t->type) != type) { std::fprintf(stderr, "strata-gemma4: %s is %s\n", name.c_str(), t->type_name()); std::exit(1); }
        return t;
    }
    float* up_f32(const std::string& name, size_t expect = 0) {
        const strata::TensorInfo* t = need(name, kF32);
        if (expect && t->elements() != expect) { std::fprintf(stderr, "strata-gemma4: %s has %zu values\n", name.c_str(), size_t(t->elements())); std::exit(1); }
        void* d = nullptr;
        CK(hipMalloc(&d, t->elements() * 4));
        CK(hipMemcpy(d, g->tensor_data(*t), t->elements() * 4, hipMemcpyHostToDevice));
        loaded += t->elements() * 4;
        return (float*) d;
    }
    // Q8_0 tensors (rows of K; 3-D ones as all their rows) one after another into one int8 + fp16 pair
    W8 up_q8(const std::vector<std::string>& names, int K, size_t expect_rows) {
        size_t rows = 0;
        for (const auto& n : names) {
            const strata::TensorInfo* t = need(n, kQ8_0);
            if (int(t->shape[0]) != K) { std::fprintf(stderr, "strata-gemma4: %s has rows of %d, not %d\n", n.c_str(), int(t->shape[0]), K); std::exit(1); }
            rows += t->elements() / K;
        }
        if (rows != expect_rows) { std::fprintf(stderr, "strata-gemma4: %s has %zu rows, not %zu\n", names[0].c_str(), rows, expect_rows); std::exit(1); }
        int8_t* q = nullptr;
        __half* d = nullptr;
        CK(hipMalloc(&q, rows * K));
        CK(hipMalloc(&d, rows * (K / 32) * 2));
        const size_t nb = size_t(K / 32), piece = 16384;   // rows a copy
        std::vector<int8_t> hq(piece * K);
        std::vector<uint16_t> hd(piece * nb);
        size_t r0 = 0;
        for (const auto& n : names) {
            const strata::TensorInfo* t = need(n, kQ8_0);
            const uint8_t* src = g->tensor_data(*t);
            const size_t nr = t->elements() / K;
            for (size_t a = 0; a < nr; a += piece) {
                const size_t m = std::min(piece, nr - a);
                for (size_t r = 0; r < m; ++r)
                    for (size_t b = 0; b < nb; ++b) {
                        const uint8_t* B = src + ((a + r) * nb + b) * 34;
                        std::memcpy(&hd[r * nb + b], B, 2);
                        std::memcpy(&hq[r * K + b * 32], B + 2, 32);
                    }
                CK(hipMemcpy(q + (r0 + a) * K, hq.data(), m * K, hipMemcpyHostToDevice));
                CK(hipMemcpy(d + (r0 + a) * nb, hd.data(), m * nb * 2, hipMemcpyHostToDevice));
            }
            r0 += nr;
        }
        loaded += rows * K + rows * nb * 2;
        return W8{q, d};
    }
    double meta(const std::string& k) {
        const auto* v = g->get("gemma4." + k);
        if (!v || !v->is_num()) { std::fprintf(stderr, "strata-gemma4: gemma4.%s is missing\n", k.c_str()); std::exit(1); }
        return v->num();
    }

    Model(const std::string& path, int ctx) : max_ctx(ctx) {
        g = std::make_unique<strata::GgufFile>(path);
        const auto* arch = g->get("general.architecture");
        if (!arch || arch->s != "gemma4") { std::fprintf(stderr, "strata-gemma4: %s is not a gemma4 GGUF\n", path.c_str()); std::exit(1); }
        // the shape this program is written for
        bool ok = meta("block_count") == kLayers && meta("embedding_length") == kHidden && meta("attention.head_count") == kHeads &&
                  meta("expert_count") == kExperts && meta("expert_used_count") == kTopK && meta("expert_feed_forward_length") == kFe &&
                  meta("feed_forward_length") == kFs && meta("attention.sliding_window") == kWindow && meta("attention.key_length") == kHdG &&
                  meta("attention.key_length_swa") == kHdS && meta("rope.freq_base") == 1e6 && meta("rope.freq_base_swa") == 1e4 &&
                  meta("final_logit_softcapping") == kCap && std::fabs(meta("attention.layer_norm_rms_epsilon") - 1e-6) < 1e-9 &&
                  meta("embedding_length_per_layer_input") == 0;
        if (const auto* pat = g->get("gemma4.attention.sliding_window_pattern"))
            for (int i = 0; i < kLayers && ok; ++i) ok = (pat->items.at(i).u != 0) == !is_global(i);
        if (const auto* kv = g->get("gemma4.attention.head_count_kv"))
            for (int i = 0; i < kLayers && ok; ++i) ok = int(kv->items.at(i).u) == (is_global(i) ? kKvG : kKvS);
        if (!ok) { std::fprintf(stderr, "strata-gemma4: this GGUF is not the 26B-A4B shape\n"); std::exit(1); }
        vocab = int(need("token_embd.weight", kQ8_0)->shape[1]);
        if (vocab % kMvRows) { std::fprintf(stderr, "strata-gemma4: vocab %d\n", vocab); std::exit(1); }
        if (g->find("output.weight")) { std::fprintf(stderr, "strata-gemma4: an untied output head is not supported\n"); std::exit(1); }
        CK(hipStreamCreate(&st));
        L.resize(kLayers);
        for (int i = 0; i < kLayers; ++i) {
            Layer& l = L[i];
            const std::string b = "blk." + std::to_string(i) + ".";
            l.global = is_global(i);
            l.hd = l.global ? kHdG : kHdS;
            l.nkv = l.global ? kKvG : kKvS;
            l.qkv_rows = l.global ? kQkvG : kQkvS;
            l.attn_norm = up_f32(b + "attn_norm.weight", kHidden);
            l.q_norm = up_f32(b + "attn_q_norm.weight", l.hd);
            l.k_norm = up_f32(b + "attn_k_norm.weight", l.hd);
            l.post_attn = up_f32(b + "post_attention_norm.weight", kHidden);
            l.ffn_norm = up_f32(b + "ffn_norm.weight", kHidden);
            l.post1 = up_f32(b + "post_ffw_norm_1.weight", kHidden);
            l.pre2 = up_f32(b + "pre_ffw_norm_2.weight", kHidden);
            l.post2 = up_f32(b + "post_ffw_norm_2.weight", kHidden);
            l.post_ffn = up_f32(b + "post_ffw_norm.weight", kHidden);
            l.out_scale = up_f32(b + "layer_output_scale.weight", 1);
            l.gate_scale = up_f32(b + "ffn_gate_inp.scale", kHidden);
            l.router = up_f32(b + "ffn_gate_inp.weight", size_t(kHidden) * kExperts);
            {   // and transposed ([2816][128]) for the prompt path
                const float* f = (const float*) g->tensor_data(*need(b + "ffn_gate_inp.weight", kF32));
                std::vector<float> vt(size_t(kHidden) * kExperts);
                for (int e = 0; e < kExperts; ++e) for (int j = 0; j < kHidden; ++j) vt[size_t(j) * kExperts + e] = f[size_t(e) * kHidden + j];
                void* d = nullptr;
                CK(hipMalloc(&d, vt.size() * 4));
                CK(hipMemcpy(d, vt.data(), vt.size() * 4, hipMemcpyHostToDevice));
                l.router_t = (const float*) d;
                loaded += vt.size() * 4;
                // hi + lo fp16 of each row times 2^s (its largest value near 2^14)
                std::vector<__half> hh(size_t(kHidden) * kExperts), hl(hh.size());
                std::vector<float> sc(kExperts);
                for (int e = 0; e < kExperts; ++e) {
                    float mx = 0;
                    for (int j = 0; j < kHidden; ++j) mx = std::max(mx, std::fabs(f[size_t(e) * kHidden + j]));
                    const int ex = mx > 0 ? std::ilogb(mx) : 0;
                    sc[e] = std::ldexp(1.f, ex - 14);
                    for (int j = 0; j < kHidden; ++j) {
                        const float v = std::ldexp(f[size_t(e) * kHidden + j], 14 - ex);
                        const __half h = __float2half(v);
                        hh[size_t(e) * kHidden + j] = h;
                        hl[size_t(e) * kHidden + j] = __float2half(v - __half2float(h));
                    }
                }
                for (auto [dst, src] : {std::pair{&l.rh, &hh}, std::pair{&l.rl, &hl}}) {
                    void* dd = nullptr;
                    CK(hipMalloc(&dd, src->size() * 2));
                    CK(hipMemcpy(dd, src->data(), src->size() * 2, hipMemcpyHostToDevice));
                    *dst = (const __half*) dd;
                }
                void* dd = nullptr;
                CK(hipMalloc(&dd, kExperts * 4));
                CK(hipMemcpy(dd, sc.data(), kExperts * 4, hipMemcpyHostToDevice));
                l.rsc = (const float*) dd;
                loaded += hh.size() * 4 + kExperts * 4;
            }
            {   // scale / sqrt(hidden): the router's input as an rmsnorm weight
                const float* gs = (const float*) g->tensor_data(*need(b + "ffn_gate_inp.scale", kF32));
                std::vector<float> v(gs, gs + kHidden);
                for (float& x : v) x /= std::sqrt(float(kHidden));
                void* dd = nullptr;
                CK(hipMalloc(&dd, kHidden * 4));
                CK(hipMemcpy(dd, v.data(), kHidden * 4, hipMemcpyHostToDevice));
                l.rnw = (const float*) dd;
            }
            l.dscale = up_f32(b + "ffn_down_exps.scale", kExperts);
            if (l.global != !g->find(b + "attn_v.weight")) { std::fprintf(stderr, "strata-gemma4: layer %d's V projection is unexpected\n", i); std::exit(1); }
            if (l.global) l.wqkv = up_q8({b + "attn_q.weight", b + "attn_k.weight"}, kHidden, kQkvG);
            else l.wqkv = up_q8({b + "attn_q.weight", b + "attn_k.weight", b + "attn_v.weight"}, kHidden, kQkvS);
            l.wo = up_q8({b + "attn_output.weight"}, kHeads * l.hd, kHidden);
            l.mlp_gu = up_q8({b + "ffn_gate.weight", b + "ffn_up.weight"}, kHidden, 2 * kFs);
            l.mlp_down = up_q8({b + "ffn_down.weight"}, kFs, kHidden);
            l.gu = up_q8({b + "ffn_gate_up_exps.weight"}, kHidden, size_t(kExperts) * 2 * kFe);
            l.down = up_q8({b + "ffn_down_exps.weight"}, kFe, size_t(kExperts) * kHidden);
            // K / V: fp16; the sliding layers a ring of kSwaRing slots
            const size_t rows = !l.global && max_ctx > kSwaRing ? kSwaRing : max_ctx;
            if (!l.global && rows == size_t(kSwaRing)) l.kv_mask = kSwaRing - 1;
            const size_t w = size_t(l.nkv) * l.hd;
            CK(hipMalloc(&l.K, rows * w * 2));
            CK(hipMalloc(&l.V, rows * w * 2));
            kv_bytes += rows * w * 4;
        }
        head = up_q8({"token_embd.weight"}, kHidden, vocab);
        out_norm = up_f32("output_norm.weight", kHidden);
        {   // RoPE: sliding base 1e4 over 256 dims; global base 1e6 over 512 dims divided by rope_freqs
            const float* ff = (const float*) g->tensor_data(*need("rope_freqs.weight", kF32));
            std::vector<double> is(kHdS / 2), ig(kHdG / 2);
            for (int j = 0; j < kHdS / 2; ++j) is[j] = std::pow(1e4, -2.0 * j / kHdS);
            for (int j = 0; j < kHdG / 2; ++j) ig[j] = std::pow(1e6, -2.0 * j / kHdG) / double(ff[j]);
            CK(hipMalloc(&inv_s, is.size() * 8));
            CK(hipMalloc(&inv_g, ig.size() * 8));
            CK(hipMemcpy(inv_s, is.data(), is.size() * 8, hipMemcpyHostToDevice));
            CK(hipMemcpy(inv_g, ig.data(), ig.size() * 8, hipMemcpyHostToDevice));
        }
        auto f = [&](float*& p, size_t n) { CK(hipMalloc(&p, n * 4)); };
        f(h, kHidden); f(qkv, kQkvG); f(att, kQMax); f(tmp, kHidden); f(aout, kHidden); f(act_s, kFs); f(mlp, kHidden);
        f(rl, kExperts); f(wts, kTopK); f(act, kTopK * kFe); f(ye, kTopK * kHidden);
        f(part, size_t(kHeads) * kAttnChunks * (kHdG + 4)); f(logits, vocab);
        CK(hipMalloc(&idx, kTopK * 4));
        CK(hipMalloc(&best, 64 * sizeof(float2)));
        CK(hipMalloc(&done, 4)); CK(hipMemset(done, 0, 4));
        CK(hipMalloc(&state, 8)); CK(hipMemset(state, 0, 8));
        CK(hipHostMalloc(&host_state, 8, 0));
        pf_alloc(chunk());
        g.reset();
    }

    // layer i of a writing step (kernels only)
    void record_layer(int i) {
        Layer& l = L[i];
        k_mv8<<<l.qkv_rows / kMvRows, 256, kHidden * 4, st>>>(l.wqkv, h, l.attn_norm, kHidden, qkv);
        const int qd = kHeads * l.hd;
        if (l.global) {
            k_qk_dec<kHdG><<<kHeads + kKvG, kHdG / 2, 0, st>>>(qkv, l.q_norm, l.k_norm, inv_g, l.K, l.V, state, l.kv_mask, kKvG, qd, qd);
            k_attn_dec<kHdG, kHeads / kKvG, kChunksG><<<dim3(kKvG, kChunksG), 32 * kHeads / kKvG, 0, st>>>(qkv, l.K, l.V, part, state, 0, l.kv_mask);
            k_attn_merge<kHdG, kChunksG><<<kHeads, kHdG, 0, st>>>(part, att);
        } else {
            k_qk_dec<kHdS><<<kHeads + kKvS, kHdS / 2, 0, st>>>(qkv, l.q_norm, l.k_norm, inv_s, l.K, l.V, state, l.kv_mask, kKvS, qd, qd + kKvS * kHdS);
            k_attn_dec<kHdS, kHeads / kKvS, kChunksS><<<dim3(kKvS, kChunksS), 32 * kHeads / kKvS, 0, st>>>(qkv, l.K, l.V, part, state, 1, l.kv_mask);
            k_attn_merge<kHdS, kChunksS><<<kHeads, kHdS, 0, st>>>(part, att);
        }
        k_mv8<<<kHidden / kMvRows, 256, qd * 4, st>>>(l.wo, att, nullptr, qd, tmp);
        static const bool unfused = std::getenv("GEMMA4_UNFUSED");
        if (unfused) {
            k_norm_add<<<1, 1024, 0, st>>>(aout, tmp, l.post_attn, h);
            k_glu8<<<kFs / kMvRows, 256, kHidden * 4, st>>>(l.mlp_gu, kFs, aout, l.ffn_norm, act_s);
            k_mv8<<<kHidden / kMvRows, 256, kFs * 4, st>>>(l.mlp_down, act_s, nullptr, kFs, mlp);
            k_router<<<kExperts / 2, 256, 0, st>>>(aout, l.gate_scale, l.router, rl);
            k_topk<<<1, 32, 0, st>>>(rl, l.dscale, idx, wts);
            k_moe_gu<<<kTopK * kFe / kMvRows, 256, kHidden * 4, st>>>(l.gu, aout, l.pre2, idx, act);
        } else {
            k_glu8_router<<<kFs / kMvRows + kExperts / 2, 256, kHidden * 4, st>>>(l.mlp_gu, tmp, l.post_attn, h, aout, l.ffn_norm, act_s,
                                                                               l.gate_scale, l.router, rl);
            k_moe_gu<<<kTopK * kFe / kMvRows, 256, kHidden * 4, st>>>(l.gu, aout, l.pre2, nullptr, act, rl, l.dscale, idx, wts);
            k_down2<<<(1 + kTopK) * kHidden / kMvRows, 256, 0, st>>>(l.mlp_down, act_s, mlp, l.down, act, idx, ye);
        }
        if (unfused) k_moe_down<<<dim3(kHidden / kMvRows, kTopK), 256, 0, st>>>(l.down, act, idx, ye);
        k_ffn_out<<<1, 1024, 0, st>>>(h, aout, mlp, ye, wts, l.post1, l.post2, l.post_ffn, l.out_scale);
    }
    void record_head() {
        k_mv8<<<vocab / kMvRows, 256, kHidden * 4, st>>>(head, h, out_norm, kHidden, logits);
        k_softcap<<<(vocab + 255) / 256, 256, 0, st>>>(logits, vocab);
        k_argmax64<<<64, 1024, 0, st>>>(logits, vocab, state, best, done);
    }
    void dump_rows(const float* d, int rows) {   // (debugging) GEMMA4_DUMP_H=FILE: h after every layer, fp32
        static FILE* f = std::fopen(std::getenv("GEMMA4_DUMP_H"), "wb");
        std::vector<float> v(size_t(rows) * kHidden);
        CK(hipMemcpyAsync(v.data(), d, v.size() * 4, hipMemcpyDeviceToHost, st));
        CK(hipStreamSynchronize(st));
        std::fwrite(v.data(), 4, v.size(), f);
        std::fflush(f);
    }
    // one token through the model (state[1] going in); logits on the card, the most likely next token in state[1]
    void step() {
        if (pos >= max_ctx) { std::fprintf(stderr, "strata-gemma4: context full (%d)\n", max_ctx); std::exit(1); }
        static const bool no_graph = std::getenv("GEMMA4_NO_GRAPH") || std::getenv("GEMMA4_DUMP_H");
        if (no_graph) {
            k_embed<<<(kHidden + 255) / 256, 256, 0, st>>>(head, state, h);
            for (int i = 0; i < kLayers; ++i) {
                record_layer(i);
                if (std::getenv("GEMMA4_DUMP_H")) dump_rows(h, 1);
            }
            record_head();
        } else {
            if (!graph) {
                hipGraph_t gr;
                CK(hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal));
                k_embed<<<(kHidden + 255) / 256, 256, 0, st>>>(head, state, h);
                for (int i = 0; i < kLayers; ++i) record_layer(i);
                record_head();
                CK(hipStreamEndCapture(st, &gr));
                CK(hipGraphInstantiate(&graph, gr, nullptr, nullptr, 0));
                CK(hipGraphDestroy(gr));
            }
            CK(hipGraphLaunch(graph, st));
        }
        ++pos;
    }
    // the prompt's tokens one writing step each
    void feed(const std::vector<int>& toks, const std::function<void()>& after_each = nullptr) {
        int* pin = nullptr;
        CK(hipHostMalloc(&pin, std::max<size_t>(toks.size(), 1) * 4, 0));
        std::memcpy(pin, toks.data(), toks.size() * 4);
        for (size_t i = 0; i < toks.size(); ++i) {
            CK(hipMemcpyAsync(state + 1, pin + i, 4, hipMemcpyHostToDevice, st));
            step();
            if (after_each) after_each();
        }
        CK(hipStreamSynchronize(st));
        CK(hipHostFree(pin));
    }
    // ------------------------------------------------ the prompt path: chunks of up to chunk() tokens through all layers
    static int chunk() { static const int c = std::min(kSwaRing - kWindow, std::getenv("GEMMA4_CHUNK") ? std::atoi(std::getenv("GEMMA4_CHUNK")) : 2048); return c; }
    int tc = 0;
    float *pH, *pQKV, *pATT, *pTMP, *pA, *pX, *pGU, *pACTS, *pMLP, *pRL, *pW, *pACT, *pY, *pXS;
    uint16_t *pXH, *pXL;
    int *pIds, *pIdx, *pCount, *pBounds, *pFill, *pTiles, *pNT, *pSrc, *pSlot;
    int* pIdsHost = nullptr;
    size_t pf_bytes = 0;
    void pf_alloc(int T) {
        tc = T;
        const size_t S = size_t(T) * kTopK;
        auto f = [&](float*& p_, size_t n) { CK(hipMalloc(&p_, n * 4)); pf_bytes += n * 4; };
        auto fi = [&](int*& p_, size_t n) { CK(hipMalloc(&p_, n * 4)); pf_bytes += n * 4; };
        f(pH, size_t(T) * kHidden); f(pQKV, size_t(T) * kQkvG); f(pATT, size_t(T) * kQMax); f(pTMP, size_t(T) * kHidden);
        f(pA, size_t(T) * kHidden); f(pX, size_t(T) * kHidden); f(pGU, size_t(T) * 2 * kFs); f(pACTS, size_t(T) * kFs);
        f(pMLP, size_t(T) * kHidden); f(pRL, size_t(T) * kExperts); f(pW, S); f(pACT, S * kFe); f(pY, S * kHidden);
        f(pXS, std::max(size_t(T), S));
        const size_t xn = std::max(size_t(T) * kQMax, S * kFe);
        CK(hipMalloc(&pXH, xn * 2)); CK(hipMalloc(&pXL, xn * 2)); pf_bytes += xn * 4;
        fi(pIds, T); fi(pIdx, S); fi(pCount, kExperts); fi(pBounds, kExperts + 1); fi(pFill, kExperts);
        fi(pTiles, 2 * (S / 64 + kExperts + 1)); fi(pNT, 1); fi(pSrc, S); fi(pSlot, S);
        CK(hipHostMalloc(&pIdsHost, size_t(T) * 4, 0));
#ifdef GEMMA4_BLT
        if (blt_on()) {
            size_t mx = 0;
            for (const Layer& l : L) mx = std::max({mx, size_t(l.qkv_rows) * kHidden, size_t(kHidden) * kHeads * l.hd, size_t(2 * kFs) * kHidden});
            CK(hipMalloc(&pWF, mx * 2));
            pf_bytes += mx * 2 + blt.wss;
            blt.init();
        }
#endif
    }
#ifdef GEMMA4_BLT
    // dense products on hipBLASLt from GEMMA4_BLT_MIN tokens (default 256; 0 turns it off): below that, turning the
    // weights into fp16 costs more than the faster product saves
    static int blt_min() { static const int v = std::getenv("GEMMA4_BLT_MIN") ? std::atoi(std::getenv("GEMMA4_BLT_MIN")) : 256; return v; }
    static bool blt_on() { return blt_min() > 0 && !lo(); }
    Blt blt;
    __half* pWF = nullptr;
#endif
    void split(const float* x, const float* nw, int K, int rows) { k_split16<<<rows, 256, 0, st>>>(x, nw, K, pXH, pXL, pXS, lo()); }
    // GEMMA4_LO=1: the inputs of the prompt path's products as two fp16 numbers (hi + lo, near fp32) instead of one;
    // measured 2026-11-10: no closer to the float64 reference (both drift the same through router near-ties), 27% slower
    static bool lo() { static const bool v = std::getenv("GEMMA4_LO") && std::atoi(std::getenv("GEMMA4_LO")) == 1; return v; }
    // prompt attention: GEMMA4_FA=0 the first kernel (16 keys a step, V transposed through LDS), else k_pf_fa variants;
    // 2 (default) measured best on the R9700: global 4 heads x 16 queries x 2 slices, 16 keys a step (~60 TFLOPs - both
    // the K tile wait and V's own loads cost ~30% each; prefetching either spilled registers or gained nothing)
    // the attention / activation steps write the next product's fp16 input themselves (hi only, the new attention)
    uint16_t* H16() { return lo() || fa() == 0 ? nullptr : pXH; }
    // the routed experts (hi only): GEMMA4_GE=0 k_gemm8_grp, else k_gemm8_gb variants (tiles of ge_rows() rows)
    static int ge() { static const int v = std::getenv("GEMMA4_GE") ? std::atoi(std::getenv("GEMMA4_GE")) : 0; return lo() ? 0 : v; }
    static int ge_rows() { return ge() == 1 ? 64 : 128; }
    void experts_gu(Layer& l, int max_tiles) {
        switch (ge()) {
        case 1: k_gemm8_gb<true, 1, 4, 8, 1><<<dim3((kFe + 127) / 128, max_tiles), 256, 0, st>>>(l.gu, pXH, pSrc, pBounds, pTiles, pNT, nullptr, pXL); return;
        case 2: k_gemm8_gb<true, 1, 4, 4, 2><<<dim3((kFe + 63) / 64, max_tiles), 256, 0, st>>>(l.gu, pXH, pSrc, pBounds, pTiles, pNT, nullptr, pXL); return;
        case 3: k_gemm8_gb<true, 2, 4, 4, 2><<<dim3((kFe + 127) / 128, max_tiles), 256, 0, st>>>(l.gu, pXH, pSrc, pBounds, pTiles, pNT, nullptr, pXL); return;
        default: k_gemm8_gb<true, 1, 8, 8, 1><<<dim3((kFe + 127) / 128, max_tiles), 256, 0, st>>>(l.gu, pXH, pSrc, pBounds, pTiles, pNT, nullptr, pXL); return;
        }
    }
    void experts_down(Layer& l, int max_tiles) {
        switch (ge()) {
        case 1: k_gemm8_gb<false, 2, 4, 8, 1><<<dim3(kHidden / 256, max_tiles), 256, 0, st>>>(l.down, pXL, nullptr, pBounds, pTiles, pNT, pY, nullptr); return;
        case 2: k_gemm8_gb<false, 2, 4, 4, 2><<<dim3(kHidden / 128, max_tiles), 256, 0, st>>>(l.down, pXL, nullptr, pBounds, pTiles, pNT, pY, nullptr); return;
        case 3: k_gemm8_gb<false, 4, 4, 4, 2><<<dim3(kHidden / 256, max_tiles), 256, 0, st>>>(l.down, pXL, nullptr, pBounds, pTiles, pNT, pY, nullptr); return;
        default: k_gemm8_gb<false, 2, 8, 8, 1><<<dim3(kHidden / 256, max_tiles), 256, 0, st>>>(l.down, pXL, nullptr, pBounds, pTiles, pNT, pY, nullptr); return;
        }
    }
    static int fa() { static const int v = std::getenv("GEMMA4_FA") ? std::atoi(std::getenv("GEMMA4_FA")) : 2; return v; }
    void attn_global(Layer& l, int p0, int T) {
        switch (fa()) {
        case 0: k_pf_attn<kHdG, kKvG, 1, 2><<<dim3(kKvG, (T + 15) / 16), 32 * 8 * 1 * 2, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 0, l.kv_mask); return;
        case 2: k_pf_fa<kHdG, kKvG, 4, 1, 2, 16><<<dim3(kKvG * 2, (T + 15) / 16), 32 * 4 * 2, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 0, l.kv_mask, H16()); return;
        case 4: k_pf_fa<kHdG, kKvG, 2, 1, 2, 16><<<dim3(kKvG * 4, (T + 15) / 16), 32 * 2 * 2, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 0, l.kv_mask, H16()); return;
        case 5: k_pf_fa<kHdG, kKvG, 2, 1, 2, 32><<<dim3(kKvG * 4, (T + 15) / 16), 32 * 2 * 2, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 0, l.kv_mask, H16()); return;
        case 6: k_pf_fa<kHdG, kKvG, 4, 1, 2, 16><<<dim3(kKvG * 2, (T + 15) / 16), 32 * 4 * 2, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 0, l.kv_mask, H16()); return;
        case 7: k_pf_fa<kHdG, kKvG, 4, 1, 2, 16, true><<<dim3(kKvG * 2, (T + 15) / 16), 32 * 4 * 2, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 0, l.kv_mask, H16()); return;
        case 8: k_pf_fa<kHdG, kKvG, 8, 1, 2, 16, true><<<dim3(kKvG, (T + 15) / 16), 32 * 8 * 2, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 0, l.kv_mask, H16()); return;
        case 3: k_pf_fa<kHdG, kKvG, 2, 1, 4, 32><<<dim3(kKvG * 4, (T + 15) / 16), 32 * 2 * 4, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 0, l.kv_mask, H16()); return;
        default: k_pf_fa<kHdG, kKvG, 4, 1, 2, 32><<<dim3(kKvG * 2, (T + 15) / 16), 32 * 4 * 2, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 0, l.kv_mask, H16()); return;
        }
    }
    void attn_sliding(Layer& l, int p0, int T) {
        switch (fa()) {
        case 0: k_pf_attn<kHdS, kKvS, 4, 2><<<dim3(kKvS, (T + 63) / 64), 32 * 2 * 4 * 2, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 1, l.kv_mask); return;
        case 2: k_pf_fa<kHdS, kKvS, 2, 4, 1, 32><<<dim3(kKvS, (T + 63) / 64), 32 * 2 * 4, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 1, l.kv_mask, H16()); return;
        case 4: k_pf_fa<kHdS, kKvS, 2, 1, 1, 32><<<dim3(kKvS, (T + 15) / 16), 32 * 2, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 1, l.kv_mask, H16()); return;
        case 5: k_pf_fa<kHdS, kKvS, 2, 2, 1, 32><<<dim3(kKvS, (T + 31) / 32), 32 * 2 * 2, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 1, l.kv_mask, H16()); return;
        case 6: k_pf_fa<kHdS, kKvS, 2, 4, 1, 32><<<dim3(kKvS, (T + 63) / 64), 32 * 2 * 4, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 1, l.kv_mask, H16()); return;
        case 7: k_pf_fa<kHdS, kKvS, 2, 4, 1, 32, true><<<dim3(kKvS, (T + 63) / 64), 32 * 2 * 4, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 1, l.kv_mask, H16()); return;
        case 8: k_pf_fa<kHdS, kKvS, 2, 4, 1, 32, true><<<dim3(kKvS, (T + 63) / 64), 32 * 2 * 4, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 1, l.kv_mask, H16()); return;
        case 3: k_pf_fa<kHdS, kKvS, 2, 2, 1, 64><<<dim3(kKvS, (T + 31) / 32), 32 * 2 * 2, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 1, l.kv_mask, H16()); return;
        default: k_pf_fa<kHdS, kKvS, 2, 4, 2, 32><<<dim3(kKvS, (T + 63) / 64), 32 * 2 * 4 * 2, 0, st>>>(pQKV, l.qkv_rows, l.K, l.V, pATT, p0, T, 1, l.kv_mask, H16()); return;
        }
    }
    void gemm8(W8 W, int N, int K, int T, float* Y) {
#ifdef GEMMA4_BLT
        if (blt_on() && T >= blt_min()) {
            const size_t n8 = size_t(N) * K / 8;
            k_deq16<<<int(std::min<size_t>((n8 + 255) / 256, 4096)), 256, 0, st>>>(W, n8, (uint4*) pWF);
            blt.run(pWF, N, K, pXH, T, Y, st);
            return;
        }
#endif
        static const int var = std::getenv("GEMMA4_GEMM") ? std::atoi(std::getenv("GEMMA4_GEMM")) : 0;
        if (!lo()) {
            switch (var) {
            case 1: k_gemm8b<4, 4, 4, 2><<<dim3((N + 255) / 256, (T + 127) / 128), 256, 0, st>>>(W, N, K, pXH, pXS, T, Y, N); return;
            case 2: k_gemm8b<4, 4, 2, 2><<<dim3((N + 127) / 128, (T + 127) / 128), 128, 0, st>>>(W, N, K, pXH, pXS, T, Y, N); return;
            case 3: k_gemm8b<2, 4, 4, 2><<<dim3((N + 127) / 128, (T + 127) / 128), 256, 0, st>>>(W, N, K, pXH, pXS, T, Y, N); return;
            case 4: k_gemm8b<4, 2, 2, 4><<<dim3((N + 127) / 128, (T + 127) / 128), 256, 0, st>>>(W, N, K, pXH, pXS, T, Y, N); return;
            case 5: k_gemm8b<2, 4, 8, 1><<<dim3((N + 255) / 256, (T + 63) / 64), 256, 0, st>>>(W, N, K, pXH, pXS, T, Y, N); return;
            default: k_gemm8<false><<<dim3((N + 255) / 256, (T + 63) / 64), 256, 0, st>>>(W, N, K, pXH, pXL, pXS, T, Y, N); return;
            }
        }
        k_gemm8<true><<<dim3((N + 255) / 256, (T + 63) / 64), 256, 0, st>>>(W, N, K, pXH, pXL, pXS, T, Y, N); }
    // T tokens at positions pos .. pos + T - 1; afterwards the decode state is the next position and the most likely
    // next token (row_logits: every row's logits through the head, one row at a time, into the callback)
    void prefill_chunk(const int* ids, int T, const std::function<void(int)>& row_logits = nullptr) {
        if (pos + T > max_ctx) { std::fprintf(stderr, "strata-gemma4: context full (%d)\n", max_ctx); std::exit(1); }
        if (T > tc) { std::fprintf(stderr, "strata-gemma4: a chunk of %d (at most %d)\n", T, tc); std::exit(1); }
        const int p0 = pos, S = T * kTopK, max_tiles = S / 64 + kExperts + 1;
        std::memcpy(pIdsHost, ids, T * 4);
        CK(hipMemcpyAsync(pIds, pIdsHost, T * 4, hipMemcpyHostToDevice, st));
        k_pf_embed<<<T, 256, 0, st>>>(head, pIds, pH);
        for (int i = 0; i < kLayers; ++i) {
            Layer& l = L[i];
            const int qd = kHeads * l.hd;
            split(pH, l.attn_norm, kHidden, T);
            gemm8(l.wqkv, l.qkv_rows, kHidden, T, pQKV);
            if (l.global) {
                k_pf_qk<kHdG><<<dim3(kHeads + kKvG, T), kHdG / 2, 0, st>>>(pQKV, l.qkv_rows, l.q_norm, l.k_norm, inv_g, l.K, l.V, p0, l.kv_mask, kKvG, qd, qd);
                attn_global(l, p0, T);
            } else {
                k_pf_qk<kHdS><<<dim3(kHeads + kKvS, T), kHdS / 2, 0, st>>>(pQKV, l.qkv_rows, l.q_norm, l.k_norm, inv_s, l.K, l.V, p0, l.kv_mask, kKvS, qd, qd + kKvS * kHdS);
                attn_sliding(l, p0, T);
            }
            if (!H16()) split(pATT, nullptr, qd, T);
            gemm8(l.wo, kHidden, qd, T, pTMP);
            k_pf_norm_add<<<T, 256, 0, st>>>(pA, pTMP, l.post_attn, pH);
            // the shared MLP
            split(pA, l.ffn_norm, kHidden, T);
            gemm8(l.mlp_gu, 2 * kFs, kHidden, T, pGU);
            k_pf_glu<<<T, 256, 0, st>>>(pGU, pACTS, lo() ? nullptr : pXH);
            if (lo()) split(pACTS, nullptr, kFs, T);
            gemm8(l.mlp_down, kHidden, kFs, T, pMLP);
            // the experts
            if (std::getenv("GEMMA4_ROUTER_FMA")) {
                k_pf_rnorm<<<T, 256, 0, st>>>(pA, l.gate_scale, pX);
                k_pf_router<<<(T + kRtTok - 1) / kRtTok, 128, 0, st>>>(pX, l.router_t, T, pRL);
            } else {
                k_split16<<<T, 256, 0, st>>>(pA, l.rnw, kHidden, pXH, pXL, pXS, 1);
                k_pf_router_mm<<<(T + 63) / 64, 256, 0, st>>>(l.rh, l.rl, l.rsc, pXH, pXL, pXS, T, pRL);
            }
            CK(hipMemsetAsync(pCount, 0, kExperts * 4, st));
            k_pf_topk<<<(T + 7) / 8, 256, 0, st>>>(pRL, l.dscale, T, pIdx, pW, pCount);
            k_pf_bounds<<<1, kExperts, 0, st>>>(pCount, pBounds, pFill, pTiles, pNT, ge() ? ge_rows() : 64);
            k_pf_scatter<<<(S + 255) / 256, 256, 0, st>>>(pIdx, T, pBounds, pFill, pSrc, pSlot);
            split(pA, l.pre2, kHidden, T);
            if (lo()) k_gemm8_grp<true, true><<<dim3((kFe + 127) / 128, max_tiles), 256, 0, st>>>(l.gu, pXH, pXL, pXS, pSrc, pBounds, pTiles, pNT, pACT);
            else if (ge()) experts_gu(l, max_tiles);
            else k_gemm8_grp<true, false><<<dim3((kFe + 127) / 128, max_tiles), 256, 0, st>>>(l.gu, pXH, pXL, pXS, pSrc, pBounds, pTiles, pNT, pACT, pXL);
            if (lo()) split(pACT, nullptr, kFe, S);
            if (lo()) k_gemm8_grp<false, true><<<dim3(kHidden / 256, max_tiles), 256, 0, st>>>(l.down, pXH, pXL, pXS, nullptr, pBounds, pTiles, pNT, pY);
            else if (ge()) experts_down(l, max_tiles);
            else k_gemm8_grp<false, false><<<dim3(kHidden / 256, max_tiles), 256, 0, st>>>(l.down, pXL, pXL, pXS, nullptr, pBounds, pTiles, pNT, pY);   // (pXL: the activations in fp16)
            k_pf_ffn_out<<<T, 256, 0, st>>>(pH, pA, pMLP, pY, pSlot, pW, l.post1, l.post2, l.post_ffn, l.out_scale);
            if (std::getenv("GEMMA4_DUMP_H")) dump_rows(pH, T);
        }
        auto head_row = [&](int t) {
            k_mv8<<<vocab / kMvRows, 256, kHidden * 4, st>>>(head, pH + size_t(t) * kHidden, out_norm, kHidden, logits);
            k_softcap<<<(vocab + 255) / 256, 256, 0, st>>>(logits, vocab);
        };
        if (row_logits)
            for (int t = 0; t < T; ++t) {
                head_row(t);
                CK(hipStreamSynchronize(st));
                row_logits(t);
            }
        head_row(T - 1);
        const int st0 = p0 + T - 1;
        CK(hipMemcpyAsync(state, &st0, 4, hipMemcpyHostToDevice, st));
        k_argmax64<<<64, 1024, 0, st>>>(logits, vocab, state, best, done);
        CK(hipStreamSynchronize(st));
        CK(hipGetLastError());
        pos = p0 + T;
    }
    void prefill(const std::vector<int>& ids, const std::function<void(int)>& row_logits = nullptr) {
        for (size_t i = 0; i < ids.size(); i += tc)
            prefill_chunk(ids.data() + i, int(std::min<size_t>(tc, ids.size() - i)), row_logits);
    }
    int next_token() {
        CK(hipMemcpyAsync(host_state, state, 8, hipMemcpyDeviceToHost, st));
        CK(hipStreamSynchronize(st));
        return host_state[1];
    }
    // before a server's first request: the hipBLASLt method search for every range of chunk sizes (a few seconds)
    void warm() {
#ifdef GEMMA4_BLT
        if (!blt_on()) return;
        std::vector<int> d(tc, 2);
        for (int b = blt_min();; b *= 2) {
            const int T = std::min(b, tc);
            set_pos(0);
            prefill_chunk(d.data(), T);
            if (T == tc) break;
        }
        CK(hipStreamSynchronize(st));
        set_pos(0);
#endif
    }
    void set_pos(int p) {
        pos = p;
        CK(hipMemcpyAsync(state, &p, 4, hipMemcpyHostToDevice, st));
        CK(hipStreamSynchronize(st));
    }
};

#include "gemma4_serve.cuh"

}  // namespace

int main(int argc, char** argv) {
    setenv("GPU_MAX_HW_QUEUES", "2", 0);
    std::string path, ids_s, top_path, logits_path;
    int n_new = 0, ctx = 4096;
    bool serve_mode = false;
    std::vector<int> eos_ids = {106, 1};   // <turn|>, <eos>
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
        } else if (a == "--new") n_new = std::stoi(next());
        else if (a == "--top") top_path = next();
        else if (a == "--logits") logits_path = next();
        else if (a == "--max-ctx" || a == "--max-context") ctx = std::stoi(next());
        else if (a == "--serve") serve_mode = true;
        else if (a == "--eos-ids") {
            eos_ids.clear();
            std::stringstream es(next());
            for (std::string v; std::getline(es, v, ',');) eos_ids.push_back(std::stoi(v));
        }
        else if (a == "--vision") {}
        else if (a == "--vram-reserve-mib") next();
        else { std::fprintf(stderr, "usage: strata-gemma4 --model FILE.gguf --ids 1,2,3|@FILE [--new N] [--top FILE] [--logits FILE] [--max-ctx N]\n"); return 2; }
    }
    if (path.empty() || (ids_s.empty() && !serve_mode)) { std::fprintf(stderr, "strata-gemma4: --model and --ids are needed\n"); return 2; }
    std::vector<int> ids;
    { std::stringstream ss(ids_s); for (std::string v; std::getline(ss, v, ',');) if (v.find_first_not_of(" \n\r\t") != std::string::npos) ids.push_back(std::stoi(v)); }
    const auto t0 = std::chrono::steady_clock::now();
    Model m(path, ctx);
    {
        size_t fr = 0, tot = 0;
        CK(hipMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-gemma4: %.1f GB on the card in %.1f s, %.2f GB free; K/V for %d positions %.2f GB, prompt buffers %.2f GB\n",
                     m.loaded / 1e9, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), fr / 1e9, ctx, m.kv_bytes / 1e9,
                     m.pf_bytes / 1e9);
    }
    if (const char* pc = std::getenv("GEMMA4_PPL")) {   // perplexity as llama-perplexity counts it: chunks of n, the second half scored
        const int n = std::atoi(pc);
        std::vector<float> row(m.vocab);
        double nll = 0;
        long cnt = 0;
        for (size_t c0 = 0; c0 + n <= ids.size(); c0 += n) {
            m.set_pos(0);
            std::vector<int> ch(ids.begin() + c0, ids.begin() + c0 + n);
            ch[0] = 2;   // <bos> starts every chunk, as llama-perplexity does for a model that adds it
            int base = 0;
            if (std::getenv("GEMMA4_TOKWISE")) {   // (checking) the writing path, a token a step
                int j = 0;
                m.feed(ch, [&]() {
                    if (j >= n / 2 && j + 1 < n) {
                        CK(hipMemcpy(row.data(), m.logits, row.size() * 4, hipMemcpyDeviceToHost));
                        const double mx = *std::max_element(row.begin(), row.end());
                        double s = 0;
                        for (float v : row) s += std::exp(double(v) - mx);
                        nll += mx + std::log(s) - row[ch[j + 1]];
                        ++cnt;
                    }
                    ++j;
                });
                std::fprintf(stderr, "[%zu] %.4f\n", c0 / n + 1, std::exp(nll / cnt));
                continue;
            }
            for (size_t i = 0; i < ch.size(); i += m.tc) {
                const int T = int(std::min<size_t>(m.tc, ch.size() - i));
                m.prefill_chunk(ch.data() + i, T, [&](int t) {
                    const int j = base + t;
                    if (j < n / 2 || j + 1 >= n) return;
                    CK(hipMemcpy(row.data(), m.logits, row.size() * 4, hipMemcpyDeviceToHost));
                    const double mx = *std::max_element(row.begin(), row.end());
                    double s = 0;
                    for (float v : row) s += std::exp(double(v) - mx);
                    nll += mx + std::log(s) - row[ch[j + 1]];
                    ++cnt;
                });
                base += T;
            }
            std::fprintf(stderr, "[%zu] %.4f\n", c0 / n + 1, std::exp(nll / cnt));
        }
        std::printf("perplexity %.4f over %ld tokens\n", std::exp(nll / cnt), cnt);
        return 0;
    }
    if (serve_mode) { m.warm(); return serve(m, eos_ids, m.tc); }
    std::vector<float> lg(m.vocab);
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
    if (std::getenv("GEMMA4_WARM")) {   // (timing) read the prompt once untimed first: one-off set-up out of the numbers
        m.prefill(ids);
        CK(hipDeviceSynchronize());
        m.set_pos(0);
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (std::getenv("GEMMA4_TOKWISE")) m.feed(ids, [&]() { if (lf || top) write_top(); });
    else if (lf || top) m.prefill(ids, [&](int) { write_top(); });
    else m.prefill(ids);
    const double read_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    const auto t2 = std::chrono::steady_clock::now();
    // GEMMA4_TOPK=k: draw with the server's sampler (temperature 1, top_p 0.95, top_k k) instead of the greedy pick
    const int samp_k = std::getenv("GEMMA4_TOPK") ? std::atoi(std::getenv("GEMMA4_TOPK")) : 0;
    SampleParams spp;
    spp.top_k = samp_k; spp.top_p = 0.95f; spp.seed = 1;
    for (int n = 0; n < n_new; ++n) {
        if (samp_k > 0) {
            static FastSampler fs;
            if (!fs.cv) fs.init();
            if (std::getenv("GEMMA4_OLD_SAMPLER") && samp_k <= kTopKMax) k_sample_topk<<<1, 1024, 0, m.st>>>(m.logits, spp, n, m.state);
            else if (samp_k <= kTopKMax) fs.run(m.logits, spp, n, m.state, m.st);
            else k_sample<<<1, 1024, 0, m.st>>>(m.logits, spp, n, m.state);
        }
        const int nxt = m.next_token();
        std::printf("%d%s", nxt, n + 1 < n_new ? "," : "\n");
        std::fflush(stdout);
        m.step();
    }
    CK(hipStreamSynchronize(m.st));
    const double write_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
    if (lf) std::fclose(lf);
    if (top) std::fclose(top);
    std::fprintf(stderr, "strata-gemma4: read %zu in %.2f s (%.1f per second), wrote %d in %.2f s (%.2f per second)\n", ids.size(), read_s,
                 ids.size() / read_s, n_new, write_s, n_new / std::max(write_s, 1e-9));
    return 0;
}
