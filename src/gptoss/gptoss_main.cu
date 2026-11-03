// src/gptoss/gptoss_main.cu - strata-gptoss: OpenAI gpt-oss-120b (openai-moe GGUF: Q8_0 dense weights, MXFP4 experts)
// on one 32 GB card plus system RAM.
//
//   strata-gptoss --model FILE.gguf --ids 1,2,3|@FILE [--new N] [--top FILE] [--logits FILE] [--max-ctx N] [--serve]
//
// The model: 36 layers, hidden 2880. Attention: 64 query heads and 8 K/V heads of 64, q / k / v / o with biases, YaRN
// RoPE (base 150000, factor 32 over 4096, NEOX halves) on every layer, a learned "sink" logit per head in the softmax,
// the even layers a sliding window of 128 positions. Feed-forward: 128 experts, top 4 by router logit (with bias), their
// weights the softmax over those 4; an expert is gate / up 2880 -> 2880 and down 2880 -> 2880 with biases and gpt-oss's
// SwiGLU (gate <= 7, |up| <= 7, (up + 1) * gate * sigmoid(1.702 gate)).
//
// The experts (61 GB) do not fit the card: every layer keeps a set of them on the card (a plan, GPTOSS_PLAN, or the
// first GPTOSS_SLOTS of each layer) and the CPU computes the rest from a host copy while the card computes its own
// (strata-flint's hybrid mode). Weights stay in their stored values: Q8_0 repacked as int8 rows plus fp16 scales,
// MXFP4 as nibble rows plus scale bytes.
#include "strata/artifact/gguf_reader.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
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
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gptoss_cpu.hpp"

namespace {

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "strata-gptoss: %s failed: %s (%s:%d)\n", #x, hipGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

constexpr int kHidden = 2880, kLayers = 36, kHeads = 64, kKv = 8, kHd = 64, kQDim = kHeads * kHd, kKvDim = kKv * kHd;
constexpr int kQkv = kQDim + 2 * kKvDim;   // 5120
constexpr int kExperts = 128, kTopK = 4, kFf = 2880, kWindow = 128;
constexpr float kEps = 1e-5f;
constexpr int kF32 = 0, kQ8_0 = 8, kMXFP4 = 39;
// an MXFP4 matrix (2880 x 2880) in the engine's layout: the rows' nibble bytes, then the rows' scale bytes
constexpr size_t kMatQ = size_t(kFf) * (kHidden / 2), kMatE = size_t(kFf) * (kHidden / 32), kMatBytes = kMatQ + kMatE;
constexpr size_t kSlotBytes = 3 * kMatBytes;   // an expert: gate, up, down
constexpr int kSwaRing = 8192;                 // sliding layers' K/V slots: the window plus the largest prompt chunk
typedef __half kc_t;

// ---------------------------------------------------------------------------------------------------- helpers
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    return v;
}
__device__ __forceinline__ uint32_t perm(uint32_t a, uint32_t b, uint32_t sel) { return __builtin_amdgcn_perm(a, b, sel); }
__device__ __forceinline__ float e8_half(uint32_t e) {   // 2^(e - 128)
    return __uint_as_float(e < 2 ? (0x00200000u << e) : (e - 1) << 23);
}
__device__ __forceinline__ float swiglu_oai(float g, float u) {
    g = fminf(g, 7.f);
    u = fmaxf(-7.f, fminf(u, 7.f));
    return (u + 1.f) * g / (1.f + expf(-1.702f * g));
}

// x (n values) into LDS, rmsnorm(x) * nw when nw; 256 threads
__device__ inline void stage_in(float* xs, const float* x, const float* nw, int n) {
    for (int i = threadIdx.x; i < n; i += 256) xs[i] = x[i];
    __syncthreads();
    if (!nw) return;
    __shared__ float red[8];
    float ss = 0;
    for (int i = threadIdx.x; i < n; i += 256) ss += xs[i] * xs[i];
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float t = 0;
    for (int i = 0; i < 8; ++i) t += red[i];
    const float r = rsqrtf(t / n + kEps);
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += 256) xs[i] = xs[i] * r * nw[i];
    __syncthreads();
}

// ---- Q8_0 as stored on the card: int8 rows (K bytes) and fp16 scales (K / 32 a row)
struct W8 { const int8_t* q; const __half* d; };
// a lane's share of R rows . x: lane takes 16-value pieces lane, lane + 32, ...
template <int R>
__device__ inline void q8_rows(W8 W, int K, int r0, const float* xs, int lane, float* acc) {
    const int nc = K / 16;
    for (int c = lane; c < nc; c += 32) {
        float xv[16];
#pragma unroll
        for (int i = 0; i < 16; i += 4) *(float4*) (xv + i) = *(const float4*) (xs + c * 16 + i);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint4 w = *(const uint4*) (W.q + size_t(r0 + r) * K + c * 16);
            const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
            float t = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int b = 0; b < 4; ++b) t += float(int8_t(ww[j] >> (8 * b))) * xv[j * 4 + b];
            acc[r] += __half2float(W.d[size_t(r0 + r) * (K / 32) + (c >> 1)]) * t;
        }
    }
}
// ---- MXFP4 as stored by the engine: q rows of K / 2 bytes (a 32-value block: 16 bytes, low nibbles values 0-15, high
// 16-31), e rows of K / 32 scale bytes
struct M4 { const uint8_t* q; const uint8_t* e; };
__device__ inline M4 slot_mat(const uint8_t* slot, int m) { const uint8_t* b = slot + size_t(m) * kMatBytes; return M4{b, b + kMatQ}; }
// 4 nibbles (one per byte of u, low 4 bits each) to their doubled E2M1 values as floats
__device__ __forceinline__ void fp4x4(uint32_t u, float* f) {
    const uint32_t mag = perm(0x0C080604u, 0x03020100u, u & 0x07070707u);
    const uint32_t sg = (u & 0x08080808u) << 4;   // sign bits at bit 7 of each byte
    f[0] = __uint_as_float(__float_as_uint(float((mag >> 0) & 0xffu)) | ((sg & 0x80u) << 24));
    f[1] = __uint_as_float(__float_as_uint(float((mag >> 8) & 0xffu)) | ((sg & 0x8000u) << 16));
    f[2] = __uint_as_float(__float_as_uint(float((mag >> 16) & 0xffu)) | ((sg & 0x800000u) << 8));
    f[3] = __uint_as_float(__float_as_uint(float((mag >> 24) & 0xffu)) | (sg & 0x80000000u));
}
// one block (16 nibble bytes, scale byte e) . 32 values of x
__device__ __forceinline__ float mx_block(uint4 q, uint32_t e, const float* x) {
    const uint32_t qq[4] = {q.x, q.y, q.z, q.w};
    float t = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float lo[4], hi[4];
        fp4x4(qq[j], lo);
        fp4x4(qq[j] >> 4, hi);
        const float4 xa = *(const float4*) (x + j * 4), xb = *(const float4*) (x + 16 + j * 4);
        t += lo[0] * xa.x + lo[1] * xa.y + lo[2] * xa.z + lo[3] * xa.w + hi[0] * xb.x + hi[1] * xb.y + hi[2] * xb.z + hi[3] * xb.w;
    }
    return t * e8_half(e);
}
template <int R>
__device__ inline void mx_rows(M4 W, int r0, const float* xs, int lane, float* acc) {
    constexpr int nb = kHidden / 32;   // (every expert matrix has K = 2880)
    for (int b = lane; b < nb; b += 32)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint4 q = *(const uint4*) (W.q + size_t(r0 + r) * (kHidden / 2) + b * 16);
            acc[r] += mx_block(q, W.e[size_t(r0 + r) * nb + b], xs + b * 32);
        }
}

// ---------------------------------------------------------------------------------------------------- kernels
constexpr int kMvRows = 32, kMvRpw = 4;   // a block: 8 warps of 4 rows
// y (rows) = or += W . x (+ bias), x = rmsnorm(x) * nw when nw; 32 rows a block
__global__ void __launch_bounds__(256) k_mv8(W8 W, const float* bias, const float* x, const float* nw, int K, float* y, int acc_) {
    extern __shared__ float xs[];
    stage_in(xs, x, nw, K);
    const int lane = threadIdx.x & 31, r0 = blockIdx.x * kMvRows + (threadIdx.x >> 5) * kMvRpw;
    float a[kMvRpw] = {};
    q8_rows<kMvRpw>(W, K, r0, xs, lane, a);
#pragma unroll
    for (int r = 0; r < kMvRpw; ++r) {
        float v = warp_sum(a[r]);
        if (lane == 0) {
            if (bias) v += bias[r0 + r];
            y[r0 + r] = acc_ ? y[r0 + r] + v : v;
        }
    }
}

__global__ void k_embed(const uint8_t* table, const int* state, float* h) {   // a Q8_0 row of the GGUF (34-byte blocks)
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= kHidden) return;
    const uint8_t* B = table + size_t(state[1]) * (kHidden / 32) * 34 + (i / 32) * 34;
    h[i] = __half2float(*(const __half*) B) * float(int8_t(B[2 + i % 32]));
}

// Attention for the writing step: block (K/V head kh, chunk c) - warp w takes query head kh * 8 + w; lane l the dims l
// and l + 32 (a NEOX RoPE pair). The position's q and k are RoPE'd here, its k / v written to the cache by chunk 0's
// first warp, and its own key used from registers (rounded as the cache holds it). Each chunk's (max, sum, output)
// goes to part; k_attn_merge adds the chunks and the sink.
constexpr int kAttnChunks = 16, kMinChunk = 64, kPart = kHd + 4;
__global__ void __launch_bounds__(256) k_attn_dec(const float* qkv, kc_t* K, kc_t* V, const float2* rope, float* part,
                                                  const int* state, int swa, int kv_mask) {
    const int kh = blockIdx.x, c = blockIdx.y, w = threadIdx.x >> 5, l = threadIdx.x & 31, h = kh * 8 + w;
    const int pos = state[0], lo = swa ? max(0, pos - (kWindow - 1)) : 0, n = pos - lo + 1;
    const int per = max(kMinChunk, (n + kAttnChunks - 1) / kAttnChunks), a = lo + c * per, b = min(pos + 1, a + per);
    const float2 cs = rope[size_t(pos) * 32 + l];
    float q0 = qkv[h * kHd + l], q1 = qkv[h * kHd + l + 32];
    { const float t0 = q0 * cs.x - q1 * cs.y, t1 = q1 * cs.x + q0 * cs.y; q0 = t0 * 0.125f; q1 = t1 * 0.125f; }
    float k0 = qkv[kQDim + kh * kHd + l], k1 = qkv[kQDim + kh * kHd + l + 32];
    { const float t0 = k0 * cs.x - k1 * cs.y, t1 = k1 * cs.x + k0 * cs.y; k0 = t0; k1 = t1; }
    const __half hk0 = __float2half(k0), hk1 = __float2half(k1);
    const __half hv0 = __float2half(qkv[kQDim + kKvDim + kh * kHd + l]), hv1 = __float2half(qkv[kQDim + kKvDim + kh * kHd + l + 32]);
    if (c == 0 && w == 0) {
        const size_t r = size_t(pos & kv_mask) * kKvDim + kh * kHd;
        K[r + l] = hk0; K[r + l + 32] = hk1; V[r + l] = hv0; V[r + l + 32] = hv1;
    }
    float m = -INFINITY, s = 0, o0 = 0, o1 = 0;
    for (int p = a; p < b; ++p) {
        float kk0, kk1, vv0, vv1;
        if (p == pos) { kk0 = __half2float(hk0); kk1 = __half2float(hk1); vv0 = __half2float(hv0); vv1 = __half2float(hv1); }
        else {
            const size_t r = size_t(p & kv_mask) * kKvDim + kh * kHd;
            kk0 = __half2float(K[r + l]); kk1 = __half2float(K[r + l + 32]); vv0 = __half2float(V[r + l]); vv1 = __half2float(V[r + l + 32]);
        }
        const float d = warp_sum(q0 * kk0 + q1 * kk1);
        const float mn = fmaxf(m, d), cf = expf(m - mn), e = expf(d - mn);
        o0 = o0 * cf + e * vv0; o1 = o1 * cf + e * vv1; s = s * cf + e; m = mn;
    }
    float* P = part + (size_t(h) * kAttnChunks + c) * kPart;
    if (l == 0) { P[0] = m; P[1] = s; }
    P[4 + l] = o0; P[4 + l + 32] = o1;
}
__global__ void k_attn_merge(const float* part, const float* sinks, float* out) {   // 64 threads a head
    const int h = blockIdx.x, t = threadIdx.x;
    const float* P = part + size_t(h) * kAttnChunks * kPart;
    float M = sinks[h];
    for (int c = 0; c < kAttnChunks; ++c) M = fmaxf(M, P[c * kPart]);
    float L = expf(sinks[h] - M), O = 0;
    for (int c = 0; c < kAttnChunks; ++c) {
        const float mc = P[c * kPart];
        if (mc == -INFINITY) continue;
        const float f = expf(mc - M);
        L += P[c * kPart + 1] * f;
        O += P[c * kPart + 4 + t] * f;
    }
    out[h * kHd + t] = O / L;
}

// hybrid: the card and the CPU meet in mapped host memory (strata-flint's scheme): the router posts the layer's normed
// input and choice and raises in_seq; the CPU computes the experts the card does not hold into y and raises out_seq;
// k_moe_combine spins on it. No host call inside the step, so it stays one graph.
struct HyShared {
    float x[kHidden];
    float y[kHidden];
    int idx[kTopK];
    float w[kTopK];
    alignas(128) uint32_t in_seq;
    alignas(128) uint32_t out_seq;
    uint32_t has;
};

// router: xn = rmsnorm(h) * nw; logits = R . xn + rb (R bf16, 128 rows), one warp a row, 4 rows a block. Block 0 writes
// xn (and posts it); the last block to finish picks the top 4 (ties to the lower index), their softmax weights, each one's
// card slot (-1: the CPU's), and posts the choice (done: a counter it resets)
__global__ void __launch_bounds__(128) k_router(const float* h, const float* nw, const uint16_t* R, const float* rb,
                                                float* logits, float* xn, int* idx, float* wts, const int* slot, int* ridx,
                                                unsigned* done, HyShared* S, uint32_t* ctr) {
    __shared__ float xs[kHidden];
    __shared__ float red[4];
    __shared__ bool last;
    for (int i = threadIdx.x; i < kHidden; i += 128) xs[i] = h[i];
    __syncthreads();
    float ss = 0;
    for (int i = threadIdx.x; i < kHidden; i += 128) ss += xs[i] * xs[i];
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    const float rr = rsqrtf((red[0] + red[1] + red[2] + red[3]) / kHidden + kEps);
    __syncthreads();
    for (int i = threadIdx.x; i < kHidden; i += 128) xs[i] = xs[i] * rr * nw[i];
    __syncthreads();
    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < kHidden; i += 128) { xn[i] = xs[i]; if (S) S->x[i] = xs[i]; }
        __threadfence_system();
    }
    const int lane = threadIdx.x & 31, r = blockIdx.x * 4 + (threadIdx.x >> 5);
    float acc = 0;
    for (int i = lane * 2; i < kHidden; i += 64) {
        const uint32_t w = *(const uint32_t*) (R + size_t(r) * kHidden + i);
        acc += __uint_as_float(w << 16) * xs[i] + __uint_as_float(w & 0xffff0000u) * xs[i + 1];
    }
    acc = warp_sum(acc);
    if (lane == 0) logits[r] = acc + rb[r];
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(done, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    __shared__ float sc[kExperts], bv[4], picked[kTopK];
    __shared__ int bi[4];
    for (int i = threadIdx.x; i < kExperts; i += 128) sc[i] = ((volatile float*) logits)[i];
    __syncthreads();
    for (int k = 0; k < kTopK; ++k) {
        float v = sc[threadIdx.x];
        int i = threadIdx.x;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(v, o, 32);
            const int oi = __shfl_xor(i, o, 32);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
        }
        if (lane == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = i; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = bv[0];
            int j = bi[0];
            for (int w = 1; w < 4; ++w) if (bv[w] > b || (bv[w] == b && bi[w] < j)) { b = bv[w]; j = bi[w]; }
            idx[k] = j;
            picked[k] = b;
            sc[j] = -INFINITY;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float t = 0, e[kTopK];
        for (int k = 0; k < kTopK; ++k) { e[k] = expf(picked[k] - picked[0]); t += e[k]; }
        for (int k = 0; k < kTopK; ++k) {
            wts[k] = e[k] / t;
            ridx[k] = slot ? slot[idx[k]] : idx[k];
            if (S) { S->idx[k] = idx[k]; S->w[k] = e[k] / t; }
        }
        *done = 0;
        if (S) {
            __threadfence_system();
            const uint32_t s = *ctr + 1;
            *ctr = s;
            *(volatile uint32_t*) &S->in_seq = s;
            __threadfence_system();
        }
    }
}

// the chosen experts the card holds: act[k][r] = swiglu(gate . xn + bg, up . xn + bu); 32 rows (of gate and up) a
// block, grid kTopK * kFf / 32. slots: the layer's experts on the card (slot ridx[k]); biases indexed by expert
__global__ void __launch_bounds__(256) k_moe_gu(const float* xn, const uint8_t* slots, const int* ridx, const int* idx,
                                                const float* bg, const float* bu, float* act) {
    const int row = blockIdx.x * kMvRows, k = row / kFf;
    const int sl = ridx[k];
    if (sl < 0) return;
    extern __shared__ float xs[];
    stage_in(xs, xn, nullptr, kHidden);
    const int lane = threadIdx.x & 31, r0 = row % kFf + (threadIdx.x >> 5) * kMvRpw, e = idx[k];
    const uint8_t* S = slots + size_t(sl) * kSlotBytes;
    float g[kMvRpw] = {}, u[kMvRpw] = {};
    mx_rows<kMvRpw>(slot_mat(S, 0), r0, xs, lane, g);
    mx_rows<kMvRpw>(slot_mat(S, 1), r0, xs, lane, u);
#pragma unroll
    for (int r = 0; r < kMvRpw; ++r) {
        const float gv = warp_sum(g[r]), uv = warp_sum(u[r]);
        if (lane == 0) act[k * kFf + r0 + r] = swiglu_oai(gv + bg[size_t(e) * kFf + r0 + r], uv + bu[size_t(e) * kFf + r0 + r]);
    }
}
// y[k][r] = down_k . act_k + bd_k for the card's chosen experts; grid (kHidden / 32, kTopK)
__global__ void __launch_bounds__(256) k_moe_down(const float* act, const uint8_t* slots, const int* ridx, const int* idx,
                                                  const float* bd, float* y) {
    const int k = blockIdx.y, sl = ridx[k];
    if (sl < 0) return;
    extern __shared__ float xs[];
    stage_in(xs, act + k * kFf, nullptr, kFf);
    const int lane = threadIdx.x & 31, r0 = blockIdx.x * kMvRows + (threadIdx.x >> 5) * kMvRpw;
    float a[kMvRpw] = {};
    mx_rows<kMvRpw>(slot_mat(slots + size_t(sl) * kSlotBytes, 2), r0, xs, lane, a);
#pragma unroll
    for (int r = 0; r < kMvRpw; ++r) {
        const float v = warp_sum(a[r]);
        if (lane == 0) y[k * kHidden + r0 + r] = v + bd[size_t(idx[k]) * kHidden + r0 + r];
    }
}
// h += sum over the card's chosen experts of w_k y_k, plus the CPU's share (after waiting for it, when S)
__global__ void __launch_bounds__(256) k_moe_combine(float* h, const float* y, const int* ridx, const float* wts,
                                                     HyShared* S, const uint32_t* ctr) {
    __shared__ uint32_t has;
    if (threadIdx.x == 0) {
        has = 0;
        if (S) {
            const uint32_t s = *ctr;
            while (*(volatile uint32_t*) &S->out_seq < s) __builtin_amdgcn_s_sleep(1);
            __threadfence_system();
            has = *(volatile uint32_t*) &S->has;
        }
    }
    __syncthreads();
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= kHidden) return;
    float t = 0;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) if (ridx[k] >= 0) t += wts[k] * y[k * kHidden + i];
    if (has) t += ((volatile float*) S->y)[i];
    h[i] += t;
}

#include "gptoss_prefill.cuh"

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

// ---------------------------------------------------------------------------------------------------- the model
struct Layer {
    const float *attn_norm, *post_norm, *sinks, *bqkv, *bo, *rb;
    W8 wqkv, wo;
    const uint16_t* router;                  // bf16 (the checkpoint's fp32 values are all bf16-exact)
    const uint16_t* router_t;                // and transposed ([2880][128]) for the prompt path
    const float *bg, *bu, *bd;               // every expert's biases, on the card
    std::vector<float> bg_h, bu_h, bd_h;     // and in host memory for the CPU
    uint8_t* slots = nullptr;                // the experts the card holds (kSlotBytes each)
    int n_slots = 0;
    int* slot = nullptr;                     // expert -> card slot (-1: the CPU's), on the card
    std::vector<int> slot_h;
    // the prompt path: each expert's gate / up and down (its card slot, or its place in the staging buffers), and the
    // experts the card does not hold (copied to staging for every prompt chunk)
    const uint8_t **pf_gu = nullptr, **pf_d = nullptr;
    std::vector<int> missing;
    bool pf_dirty = false;                   // (the slots changed since the tables were made)
    std::vector<float> heat;                 // the adaptive tier: each expert's recent routing
    bool swa = false;
    kc_t *K = nullptr, *V = nullptr;
    int kv_mask = 0x7fffffff;
};

struct Model {
    std::unique_ptr<strata::GgufFile> g;
    std::vector<Layer> L;
    const uint8_t* embed = nullptr;   // the GGUF's Q8_0 token table, pinned host memory the card reads a row of
    const float* out_norm = nullptr;
    W8 head{};
    int vocab = 0, max_ctx;
    size_t loaded = 0, kv_bytes = 0;
    hipStream_t st;
    float2* rope = nullptr;           // [position][32]: cos, sin (times YaRN's 1 + 0.1 ln 32)
    float *h, *qkv, *att, *part, *logits, *rl, *wts, *xn, *act, *ye;
    int *idx, *ridx;
    float2* best;
    unsigned *done, *done2;
    int pos = 0;
    int* state;        // on the card: [0] the position, [1] the token going in next
    int* host_state;
    hipGraphExec_t graph = nullptr;
    // hybrid
    uint8_t* arena = nullptr;   // every expert, host side, in the card's layout: [layer][expert] slots
    size_t arena_bytes = 0;
    HyShared *hy_s = nullptr, *hy_s_dev = nullptr;
    uint32_t* hy_ctr = nullptr;
    uint32_t hy_seq = 0;
    size_t hy_cpu = 0, hy_card = 0;
    double hy_t_wait = 0, hy_t_cpu = 0;

    const strata::TensorInfo* need(const std::string& name, int type) {
        const strata::TensorInfo* t = g->find(name);
        if (!t) { std::fprintf(stderr, "strata-gptoss: %s is missing\n", name.c_str()); std::exit(1); }
        if (int(t->type) != type) { std::fprintf(stderr, "strata-gptoss: %s is %s\n", name.c_str(), t->type_name()); std::exit(1); }
        return t;
    }
    float* up_f32(const std::string& name) {
        const strata::TensorInfo* t = need(name, kF32);
        void* d = nullptr;
        CK(hipMalloc(&d, t->elements() * 4));
        CK(hipMemcpy(d, g->tensor_data(*t), t->elements() * 4, hipMemcpyHostToDevice));
        loaded += t->elements() * 4;
        return (float*) d;
    }
    // Q8_0 tensors (rows of K) one after another into one int8 + fp16 pair; returns the rows
    W8 up_q8(const std::vector<std::string>& names, int K, int* rows_out = nullptr) {
        size_t rows = 0;
        for (const auto& n : names) rows += need(n, kQ8_0)->shape[1];
        int8_t* q = nullptr;
        __half* d = nullptr;
        CK(hipMalloc(&q, rows * K));
        CK(hipMalloc(&d, rows * (K / 32) * 2));
        std::vector<int8_t> hq(rows * K);
        std::vector<uint16_t> hd(rows * (K / 32));
        size_t r0 = 0;
        for (const auto& n : names) {
            const strata::TensorInfo* t = need(n, kQ8_0);
            if (int(t->shape[0]) != K) { std::fprintf(stderr, "strata-gptoss: %s has rows of %d\n", n.c_str(), int(t->shape[0])); std::exit(1); }
            const uint8_t* src = g->tensor_data(*t);
            const size_t nr = t->shape[1], nb = size_t(K / 32);
            for (size_t r = 0; r < nr; ++r)
                for (size_t b = 0; b < nb; ++b) {
                    const uint8_t* B = src + (r * nb + b) * 34;
                    std::memcpy(&hd[(r0 + r) * nb + b], B, 2);
                    std::memcpy(&hq[(r0 + r) * K + b * 32], B + 2, 32);
                }
            r0 += nr;
        }
        CK(hipMemcpy(q, hq.data(), hq.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(d, hd.data(), hd.size() * 2, hipMemcpyHostToDevice));
        loaded += hq.size() + hd.size() * 2;
        if (rows_out) *rows_out = int(rows);
        return W8{q, d};
    }

    Model(const std::string& path, int ctx) : max_ctx(ctx) {
        g = std::make_unique<strata::GgufFile>(path);
        const auto* arch = g->get("general.architecture");
        if (!arch || arch->s != "gpt-oss") { std::fprintf(stderr, "strata-gptoss: %s is not a gpt-oss GGUF\n", path.c_str()); std::exit(1); }
        vocab = int(need("output.weight", kQ8_0)->shape[1]);
        CK(hipStreamCreate(&st));
        gptoss_cpu::init(std::getenv("GPTOSS_CPU_THREADS") ? std::atoi(std::getenv("GPTOSS_CPU_THREADS")) : 0);
        // the experts the card holds: a plan file (lines "layer expert expert ..."), else the first GPTOSS_SLOTS of
        // each layer, else as many as fit leaving GPTOSS_RESERVE_GB (default 3) free
        std::vector<std::vector<int>> plan(kLayers);
        if (const char* pf = std::getenv("GPTOSS_PLAN")) {
            FILE* f = std::fopen(pf, "r");
            if (!f) { std::fprintf(stderr, "strata-gptoss: cannot read the plan %s\n", pf); std::exit(1); }
            char line[1 << 16];
            while (std::fgets(line, sizeof line, f)) {
                std::stringstream ss(line);
                int li;
                if (!(ss >> li) || li < 0 || li >= kLayers) continue;
                for (int e; ss >> e;) if (e >= 0 && e < kExperts) plan[li].push_back(e);
            }
            std::fclose(f);
        }
        L.resize(kLayers);
        for (int i = 0; i < kLayers; ++i) {
            Layer& l = L[i];
            const std::string b = "blk." + std::to_string(i) + ".";
            l.swa = i % 2 == 0;
            l.attn_norm = up_f32(b + "attn_norm.weight");
            l.post_norm = up_f32(b + "post_attention_norm.weight");
            l.sinks = up_f32(b + "attn_sinks.weight");
            l.wqkv = up_q8({b + "attn_q.weight", b + "attn_k.weight", b + "attn_v.weight"}, kHidden);
            {
                float* bq = nullptr;
                CK(hipMalloc(&bq, kQkv * 4));
                const char* nm[3] = {"attn_q.bias", "attn_k.bias", "attn_v.bias"};
                const int off[3] = {0, kQDim, kQDim + kKvDim}, n[3] = {kQDim, kKvDim, kKvDim};
                for (int j = 0; j < 3; ++j) CK(hipMemcpy(bq + off[j], g->tensor_data(*need(b + nm[j], kF32)), n[j] * 4, hipMemcpyHostToDevice));
                l.bqkv = bq;
            }
            l.wo = up_q8({b + "attn_output.weight"}, kQDim);
            l.bo = up_f32(b + "attn_output.bias");
            {   // the router as bf16 (checked exact)
                const strata::TensorInfo* t = need(b + "ffn_gate_inp.weight", kF32);
                const float* f = (const float*) g->tensor_data(*t);
                std::vector<uint16_t> v(t->elements());
                for (size_t j = 0; j < v.size(); ++j) {
                    uint32_t u;
                    std::memcpy(&u, f + j, 4);
                    if (u & 0xffff) { std::fprintf(stderr, "strata-gptoss: %s is not bf16-exact\n", t->name.c_str()); std::exit(1); }
                    v[j] = uint16_t(u >> 16);
                }
                void* d = nullptr;
                CK(hipMalloc(&d, v.size() * 2));
                CK(hipMemcpy(d, v.data(), v.size() * 2, hipMemcpyHostToDevice));
                l.router = (const uint16_t*) d;
                loaded += v.size() * 2;
                std::vector<uint16_t> vt(v.size());
                for (int e = 0; e < kExperts; ++e) for (int j = 0; j < kHidden; ++j) vt[size_t(j) * kExperts + e] = v[size_t(e) * kHidden + j];
                CK(hipMalloc(&d, vt.size() * 2));
                CK(hipMemcpy(d, vt.data(), vt.size() * 2, hipMemcpyHostToDevice));
                l.router_t = (const uint16_t*) d;
                loaded += vt.size() * 2;
            }
            l.rb = up_f32(b + "ffn_gate_inp.bias");
            l.bg = up_f32(b + "ffn_gate_exps.bias");
            l.bu = up_f32(b + "ffn_up_exps.bias");
            l.bd = up_f32(b + "ffn_down_exps.bias");
            for (auto [v, nm] : {std::pair{&l.bg_h, "ffn_gate_exps.bias"}, std::pair{&l.bu_h, "ffn_up_exps.bias"}, std::pair{&l.bd_h, "ffn_down_exps.bias"}}) {
                const strata::TensorInfo* t = need(b + nm, kF32);
                const float* f = (const float*) g->tensor_data(*t);
                v->assign(f, f + t->elements());
            }
            // K / V: fp16; the sliding layers a ring of kSwaRing slots
            const size_t rows = l.swa && max_ctx > kSwaRing ? kSwaRing : max_ctx;
            if (l.swa && rows == size_t(kSwaRing)) l.kv_mask = kSwaRing - 1;
            CK(hipMalloc(&l.K, rows * kKvDim * 2));
            CK(hipMalloc(&l.V, rows * kKvDim * 2));
            kv_bytes += rows * kKvDim * 4;
        }
        {   // the token table stays in pinned host memory (0.6 GB): one 3 KB row a token over the bus
            const strata::TensorInfo* t = need("token_embd.weight", kQ8_0);
            const size_t bytes = t->shape[1] * (kHidden / 32) * 34;
            void* hp = nullptr;
            CK(hipHostMalloc(&hp, bytes, hipHostMallocMapped));
            std::memcpy(hp, g->tensor_data(*t), bytes);
            void* dp = nullptr;
            CK(hipHostGetDevicePointer(&dp, hp, 0));
            embed = (const uint8_t*) dp;
        }
        out_norm = up_f32("output_norm.weight");
        head = up_q8({"output.weight"}, kHidden);
        {   // YaRN (OpenAI's reference: the ramp ends not rounded), cos / sin per position in double precision
            const double base = 150000.0, factor = 32.0, ctx0 = 4096.0, bf = 32.0, bs = 1.0, dh = kHd / 2;
            const double conc = 0.1 * std::log(factor) + 1.0;
            double low = dh * std::log(ctx0 / (bf * 2 * M_PI)) / std::log(base), high = dh * std::log(ctx0 / (bs * 2 * M_PI)) / std::log(base);
            if (std::getenv("GPTOSS_LLAMA_YARN")) { low = std::max(0.0, std::floor(low)); high = std::min(kHd - 1.0, std::ceil(high)); }
            double inv[32];
            for (int j = 0; j < 32; ++j) {
                const double freq = std::pow(base, 2.0 * j / kHd);
                const double ramp = std::min(1.0, std::max(0.0, (j - low) / std::max(high - low, 0.001)));
                inv[j] = (1 / (factor * freq)) * ramp + (1 / freq) * (1 - ramp);
            }
            std::vector<float2> t(size_t(max_ctx) * 32);
            for (int p = 0; p < max_ctx; ++p)
                for (int j = 0; j < 32; ++j) {
                    const double a = p * inv[j];
                    t[size_t(p) * 32 + j] = make_float2(float(std::cos(a) * conc), float(std::sin(a) * conc));
                }
            CK(hipMalloc(&rope, t.size() * sizeof(float2)));
            CK(hipMemcpy(rope, t.data(), t.size() * sizeof(float2), hipMemcpyHostToDevice));
        }
        auto f = [&](float*& p, size_t n) { CK(hipMalloc(&p, n * 4)); };
        f(h, kHidden); f(qkv, kQkv); f(att, kQDim); f(part, size_t(kHeads) * kAttnChunks * kPart); f(logits, vocab);
        f(rl, kExperts); f(wts, kTopK); f(xn, kHidden); f(act, kTopK * kFf); f(ye, kTopK * kHidden);
        CK(hipMalloc(&idx, kTopK * 4)); CK(hipMalloc(&ridx, kTopK * 4));
        CK(hipMalloc(&best, 64 * sizeof(float2)));
        CK(hipMalloc(&done, 4)); CK(hipMemset(done, 0, 4));
        CK(hipMalloc(&done2, 4)); CK(hipMemset(done2, 0, 4));
        CK(hipMalloc(&state, 8)); CK(hipMemset(state, 0, 8));
        CK(hipHostMalloc(&host_state, 8, 0));
        pf_alloc(chunk());
        load_experts(plan);
        g.reset();
    }

    // every expert into the host arena (the card's layout), then each layer's chosen ones onto the card
    void load_experts(std::vector<std::vector<int>>& plan) {
        const auto t0 = std::chrono::steady_clock::now();
        arena_bytes = size_t(kLayers) * kExperts * kSlotBytes;
        void* a = mmap(nullptr, arena_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (a == MAP_FAILED) { std::fprintf(stderr, "strata-gptoss: no room for the %.1f GB expert arena\n", arena_bytes / 1e9); std::exit(1); }
        madvise(a, arena_bytes, MADV_HUGEPAGE);
        arena = (uint8_t*) a;
        const char* kinds[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
        constexpr size_t rowg = kHidden / 32 * 17;   // a GGUF MXFP4 row (90 blocks of 17 bytes)
        const int nth = std::max(1u, std::thread::hardware_concurrency());
        const int fd = open(g->path().c_str(), O_RDONLY);
        for (int i = 0; i < n_layers(); ++i) {   // (GPTOSS_NLAYERS for tests: the later layers' experts are left out)
            const uint8_t* src[3];
            for (int m = 0; m < 3; ++m) {
                const strata::TensorInfo* t = need("blk." + std::to_string(i) + "." + kinds[m], kMXFP4);
                if (t->shape[0] != kHidden || t->shape[1] != kFf || t->shape[2] != kExperts) { std::fprintf(stderr, "strata-gptoss: %s has an unexpected shape\n", t->name.c_str()); std::exit(1); }
                src[m] = g->tensor_data(*t);
            }
            std::vector<std::thread> th;
            for (int tt = 0; tt < nth; ++tt)
                th.emplace_back([&, tt] {
                    for (int e = tt; e < kExperts; e += nth)
                        for (int m = 0; m < 3; ++m) {
                            uint8_t* dst = arena + (size_t(i) * kExperts + e) * kSlotBytes + m * kMatBytes;
                            const uint8_t* s = src[m] + size_t(e) * kFf * rowg;
                            uint8_t* qd = dst;
                            uint8_t* ed = dst + kMatQ;
                            for (int r = 0; r < kFf; ++r) {
                                const uint8_t* B = s + size_t(r) * rowg;
                                for (int b = 0; b < kHidden / 32; ++b, B += 17, qd += 16) {
                                    *ed++ = B[0];
                                    std::memcpy(qd, B + 1, 16);
                                }
                            }
                        }
                });
            for (auto& t : th) t.join();
            // the layer's file pages leave the page cache: the arena (61 GB) and the file (63 GB) do not both fit in RAM,
            // and the kernel evicting by itself slows loading
            if (fd >= 0)
                for (int m = 0; m < 3; ++m) {
                    const strata::TensorInfo* t = need("blk." + std::to_string(i) + "." + kinds[m], kMXFP4);
                    const size_t len = size_t(kExperts) * kFf * rowg, pg = 4096;
                    const uintptr_t a0 = (uintptr_t(g->tensor_data(*t)) + pg - 1) & ~uintptr_t(pg - 1);
                    const uintptr_t a1 = (uintptr_t(g->tensor_data(*t)) + len) & ~uintptr_t(pg - 1);
                    if (a1 > a0) madvise((void*) a0, a1 - a0, MADV_DONTNEED);   // (our mapping's pages first: mapped ones stay cached)
                    posix_fadvise(fd, off_t(g->data_start() + t->offset), off_t(len), POSIX_FADV_DONTNEED);
                }
        }
        if (fd >= 0) close(fd);
        std::fprintf(stderr, "strata-gptoss: experts repacked into a %.1f GB host arena in %.1f s\n", arena_bytes / 1e9,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        if (!(std::getenv("GPTOSS_PIN") && std::atoi(std::getenv("GPTOSS_PIN")) == 0) && hipHostRegister(arena, arena_bytes, 0) != hipSuccess) {
            (void) hipGetLastError();
            std::fprintf(stderr, "strata-gptoss: could not pin the expert arena (prompts copy slower)\n");
        }
        // how many slots a layer: the plan's, or GPTOSS_SLOTS, or what fits
        bool have_plan = false;
        for (auto& p : plan) have_plan |= !p.empty();
        if (!have_plan) {
            int n = std::getenv("GPTOSS_SLOTS") ? std::atoi(std::getenv("GPTOSS_SLOTS")) : -1;
            if (n < 0) {
                size_t fr = 0, tot = 0;
                CK(hipMemGetInfo(&fr, &tot));
                const double reserve = (std::getenv("GPTOSS_RESERVE_GB") ? std::atof(std::getenv("GPTOSS_RESERVE_GB")) : 3.0) * 1e9;
                n = int(std::max(0.0, (double(fr) - reserve) / (double(kSlotBytes) * kLayers)));
            }
            n = std::min(n, kExperts);
            for (auto& p : plan) for (int e = 0; e < n; ++e) p.push_back(e);
        }
        size_t on_card = 0;
        for (int i = 0; i < kLayers; ++i) {
            Layer& l = L[i];
            if (i >= n_layers()) plan[i].clear();
            l.n_slots = int(plan[i].size());
            l.slot_h.assign(kExperts, -1);
            l.heat.assign(kExperts, 0.f);
            if (l.n_slots) {
                CK(hipMalloc(&l.slots, size_t(l.n_slots) * kSlotBytes));
                for (int j = 0; j < l.n_slots; ++j) {
                    l.slot_h[plan[i][j]] = j;
                    CK(hipMemcpy(l.slots + size_t(j) * kSlotBytes, arena + (size_t(i) * kExperts + plan[i][j]) * kSlotBytes, kSlotBytes, hipMemcpyHostToDevice));
                }
            }
            on_card += l.n_slots;
            loaded += size_t(l.n_slots) * kSlotBytes;
            CK(hipMalloc(&l.slot, kExperts * 4));
            CK(hipMemcpy(l.slot, l.slot_h.data(), kExperts * 4, hipMemcpyHostToDevice));
            pf_tables(i);
        }
        CK(hipHostMalloc((void**) &hy_s, sizeof(HyShared), hipHostMallocMapped));
        std::memset((void*) hy_s, 0, sizeof(HyShared));
        CK(hipHostGetDevicePointer((void**) &hy_s_dev, hy_s, 0));
        CK(hipMalloc(&hy_ctr, 4));
        CK(hipMemset(hy_ctr, 0, 4));
        std::fprintf(stderr, "strata-gptoss: %zu of %d experts on the card (%.1f GB), the rest on the CPU\n", on_card,
                     kLayers * kExperts, on_card * double(kSlotBytes) / 1e9);
    }

    static int n_layers() { static const int n = std::getenv("GPTOSS_NLAYERS") ? std::atoi(std::getenv("GPTOSS_NLAYERS")) : kLayers; return n; }

    // layer i of a writing step (kernels only)
    void record_layer(int i) {
        Layer& l = L[i];
        k_mv8<<<kQkv / kMvRows, 256, kHidden * 4, st>>>(l.wqkv, l.bqkv, h, l.attn_norm, kHidden, qkv, 0);
        k_attn_dec<<<dim3(kKv, kAttnChunks), 256, 0, st>>>(qkv, l.K, l.V, rope, part, state, l.swa, l.kv_mask);
        k_attn_merge<<<kHeads, kHd, 0, st>>>(part, l.sinks, att);
        k_mv8<<<kHidden / kMvRows, 256, kQDim * 4, st>>>(l.wo, l.bo, att, nullptr, kQDim, h, 1);
        k_router<<<kExperts / 4, 128, 0, st>>>(h, l.post_norm, l.router, l.rb, rl, xn, idx, wts, l.slot, ridx, done, hy_s_dev, hy_ctr);
        k_moe_gu<<<kTopK * kFf / kMvRows, 256, kHidden * 4, st>>>(xn, l.slots, ridx, idx, l.bg, l.bu, act);
        k_moe_down<<<dim3(kHidden / kMvRows, kTopK), 256, kFf * 4, st>>>(act, l.slots, ridx, idx, l.bd, ye);
        k_moe_combine<<<(kHidden + 255) / 256, 256, 0, st>>>(h, ye, ridx, wts, hy_s_dev, hy_ctr);
    }
    void record_head() {
        k_mv8<<<vocab / kMvRows, 256, kHidden * 4, st>>>(head, nullptr, h, out_norm, kHidden, logits, 0);
        k_argmax64<<<64, 1024, 0, st>>>(logits, vocab, state, best, done2);
    }
    // the CPU's half of layer i: wait for the card's post, compute the experts it does not hold, hand the sum back
    void hy_serve_layer(int i) {
        const Layer& l = L[i];
        volatile HyShared* S = hy_s;
        const uint32_t sq = ++hy_seq;
        const auto w0 = std::chrono::steady_clock::now();
        for (long spin = 0; S->in_seq < sq; ++spin) {
            __builtin_ia32_pause();
            if ((spin & 0xfffff) == 0xfffff && std::chrono::steady_clock::now() - w0 > std::chrono::seconds(30)) {
                std::fprintf(stderr, "strata-gptoss: the card never posted layer %d (%s)\n", i, hipGetErrorString(hipStreamQuery(st)));
                std::exit(1);
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        const auto w1 = std::chrono::steady_clock::now();
        hy_t_wait += std::chrono::duration<double>(w1 - w0).count();
        gptoss_cpu::Expert ex[kTopK];
        float w[kTopK];
        int n = 0;
        for (int k = 0; k < kTopK; ++k) {
            const int e = S->idx[k];
            L[i].heat[e] += heat_inc;
            if (l.slot_h[e] >= 0) { ++hy_card; continue; }
            missed.push_back({i, e});
            const uint8_t* s = arena + (size_t(i) * kExperts + e) * kSlotBytes;
            auto mat = [&](int m) { return gptoss_cpu::Mat{s + m * kMatBytes, s + m * kMatBytes + kMatQ}; };
            ex[n] = {mat(0), mat(1), mat(2), l.bg_h.data() + size_t(e) * kFf, l.bu_h.data() + size_t(e) * kFf, l.bd_h.data() + size_t(e) * kHidden};
            w[n++] = S->w[k];
        }
        hy_cpu += n;
        if (n) gptoss_cpu::moe((const float*) S->x, kHidden, kFf, n, ex, w, (float*) S->y);
        S->has = n > 0;
        hy_t_cpu += std::chrono::duration<double>(std::chrono::steady_clock::now() - w1).count();
        std::atomic_thread_fence(std::memory_order_release);
        S->out_seq = sq;
    }
    // (debugging) GPTOSS_DUMP_H=FILE: h after every layer, appended as fp32
    void dump_h() { dump_rows(h, 1); }
    // one token through the model (state[1] going in); logits on the card, the most likely next token in state[1]
    void step() {
        if (pos >= max_ctx) { std::fprintf(stderr, "strata-gptoss: context full (%d)\n", max_ctx); std::exit(1); }
        static const bool no_graph = std::getenv("GPTOSS_NO_GRAPH") || std::getenv("GPTOSS_DUMP_H") || n_layers() < kLayers;
        gptoss_cpu::set_active(true);
        if (no_graph) {
            k_embed<<<(kHidden + 255) / 256, 256, 0, st>>>(embed, state, h);
            for (int i = 0; i < n_layers(); ++i) {
                record_layer(i);
                hy_serve_layer(i);
                if (std::getenv("GPTOSS_DUMP_H")) dump_h();
            }
            record_head();
        } else {
            if (!graph) {
                hipGraph_t gr;
                CK(hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal));
                k_embed<<<(kHidden + 255) / 256, 256, 0, st>>>(embed, state, h);
                for (int i = 0; i < kLayers; ++i) record_layer(i);
                record_head();
                CK(hipStreamEndCapture(st, &gr));
                CK(hipGraphInstantiate(&graph, gr, nullptr, nullptr, 0));
                CK(hipGraphDestroy(gr));
            }
            CK(hipGraphLaunch(graph, st));
            for (int i = 0; i < kLayers; ++i) hy_serve_layer(i);
        }
        gptoss_cpu::set_active(false);
        ++pos;
        adapt_after_step();
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
    // ---- the adaptive tier (strata-ling's): the card's expert slots follow the routing. Each layer keeps its number of
    // slots; an expert the CPU computed that is now hotter than the layer's coldest resident one takes that one's slot.
    // heat decays (the increment grows 0.2% a token: a half-life of about 350 tokens). Writing: up to GPTOSS_SWAPS (2)
    // swaps after a token, copied from the arena. The prompt path: up to GPTOSS_SWAPS_PROMPT (32) a layer a chunk, copied
    // from the staging buffers on the card. GPTOSS_ADAPT=0 keeps the plan fixed.
    float heat_inc = 1.f;
    std::vector<std::pair<int, int>> missed;   // this token's CPU experts (layer, expert)
    size_t n_swaps = 0;
    static bool adapt_on() { static const bool on = !(std::getenv("GPTOSS_ADAPT") && std::atoi(std::getenv("GPTOSS_ADAPT")) == 0); return on; }
    void heat_tick(int tokens) {
        heat_inc *= std::pow(1.002f, float(tokens));
        if (heat_inc > 1e20f) {
            for (auto& l : L) for (auto& v : l.heat) v /= heat_inc;
            heat_inc = 1.f;
        }
    }
    int coldest(const Layer& l) const {
        int c = -1;
        for (int e = 0; e < kExperts; ++e) if (l.slot_h[e] >= 0 && (c < 0 || l.heat[e] < l.heat[c])) c = e;
        return c;
    }
    bool worth(const Layer& l, int e_in, int e_out) const { return l.heat[e_in] > 1.25f * l.heat[e_out] + 2.f * heat_inc; }
    void push_slots(Layer& l) {
        CK(hipMemcpyAsync(l.slot, l.slot_h.data(), kExperts * 4, hipMemcpyHostToDevice, st));
        CK(hipStreamSynchronize(st));   // (slot_h may change again before a pageable copy is done)
        l.pf_dirty = true;
    }
    void adapt_after_step() {
        heat_tick(1);
        if (!adapt_on() || missed.empty()) { missed.clear(); return; }
        static const int budget = std::getenv("GPTOSS_SWAPS") ? std::atoi(std::getenv("GPTOSS_SWAPS")) : 2;
        std::sort(missed.begin(), missed.end(), [&](const auto& a, const auto& b) { return L[a.first].heat[a.second] > L[b.first].heat[b.second]; });
        int done = 0;
        for (const auto& mm : missed) {
            if (done >= budget) break;
            Layer& l = L[mm.first];
            if (l.slot_h[mm.second] >= 0) continue;
            const int c = coldest(l);
            if (c < 0 || !worth(l, mm.second, c)) continue;
            const int sl = l.slot_h[c];
            CK(hipMemcpyAsync(l.slots + size_t(sl) * kSlotBytes, arena + (size_t(mm.first) * kExperts + mm.second) * kSlotBytes, kSlotBytes, hipMemcpyHostToDevice, st));
            l.slot_h[mm.second] = sl;
            l.slot_h[c] = -1;
            push_slots(l);
            ++n_swaps;
            ++done;
        }
        missed.clear();
    }

    // ------------------------------------------------ the prompt path: chunks of up to chunk() tokens through all layers
    static int chunk() { static const int c = std::min(kSwaRing - kWindow, std::getenv("GPTOSS_CHUNK") ? std::atoi(std::getenv("GPTOSS_CHUNK")) : 4096); return c; }
    int tc = 0;
    float *pH = nullptr, *pQKV, *pATT, *pXN, *pACT, *pY, *pXS, *pRL, *pW;
    uint16_t *pXH, *pXL;
    int *pIds, *pIdx, *pCount, *pBounds, *pFill, *pTiles, *pNT, *pSrc, *pSlot;
    int* pIdsHost = nullptr;
    int* pCountHost = nullptr;
    // a layer's experts on the card for the prompt path: gate / up of all 128 (stg_gu) and down (stg_d), copied from
    // the arena on their own stream while the card works (gate / up of the next layer during this one's down)
    uint8_t *stg_gu = nullptr, *stg_d = nullptr;
    hipStream_t cp = nullptr;
    hipEvent_t ev_gu_ready, ev_gu_free, ev_d_ready, ev_d_free;
    void pf_alloc(int T) {
        tc = T;
        const size_t S = size_t(T) * kTopK;
        auto f = [&](float*& p_, size_t n) { CK(hipMalloc(&p_, n * 4)); };
        auto fi = [&](int*& p_, size_t n) { CK(hipMalloc(&p_, n * 4)); };
        f(pH, size_t(T) * kHidden); f(pQKV, size_t(T) * kQkv); f(pATT, size_t(T) * kQDim); f(pXN, size_t(T) * kHidden);
        f(pACT, S * kFf); f(pY, S * kHidden); f(pXS, S); f(pRL, size_t(T) * kExperts); f(pW, S);
        CK(hipMalloc(&pXH, std::max(size_t(T) * kQDim, S * kFf) * 2));
        CK(hipMalloc(&pXL, std::max(size_t(T) * kQDim, S * kFf) * 2));
        fi(pIds, T); fi(pIdx, S); fi(pCount, kExperts); fi(pBounds, kExperts + 1); fi(pFill, kExperts);
        fi(pTiles, 2 * (S / 64 + kExperts + 1)); fi(pNT, 1); fi(pSrc, S); fi(pSlot, S);
        CK(hipHostMalloc(&pIdsHost, size_t(T) * 4, 0));
        CK(hipHostMalloc(&pCountHost, kExperts * 4, 0));
        CK(hipMalloc(&stg_gu, kExperts * 2 * kMatBytes));
        CK(hipMalloc(&stg_d, kExperts * kMatBytes));
        CK(hipStreamCreateWithFlags(&cp, hipStreamNonBlocking));
        for (hipEvent_t* e : {&ev_gu_ready, &ev_gu_free, &ev_d_ready, &ev_d_free}) CK(hipEventCreateWithFlags(e, hipEventDisableTiming));
        CK(hipEventRecord(ev_gu_free, st));
        CK(hipEventRecord(ev_d_free, st));
    }
    void copy_gu(int i) {
        CK(hipStreamWaitEvent(cp, ev_gu_free, 0));
        const std::vector<int>& mi = L[i].missing;
        for (size_t j = 0; j < mi.size(); ++j)
            CK(hipMemcpyAsync(stg_gu + j * 2 * kMatBytes, arena + (size_t(i) * kExperts + mi[j]) * kSlotBytes, 2 * kMatBytes, hipMemcpyHostToDevice, cp));
        CK(hipEventRecord(ev_gu_ready, cp));
    }
    void copy_d(int i) {
        CK(hipStreamWaitEvent(cp, ev_d_free, 0));
        const std::vector<int>& mi = L[i].missing;
        for (size_t j = 0; j < mi.size(); ++j)
            CK(hipMemcpyAsync(stg_d + j * kMatBytes, arena + (size_t(i) * kExperts + mi[j]) * kSlotBytes + 2 * kMatBytes, kMatBytes, hipMemcpyHostToDevice, cp));
        CK(hipEventRecord(ev_d_ready, cp));
    }
    // a layer's expert pointer tables for the prompt path (after its slots change)
    void pf_tables(int i) {
        Layer& l = L[i];
        std::vector<const uint8_t*> gu(kExperts), d(kExperts);
        l.missing.clear();
        for (int e = 0; e < kExperts; ++e) {
            if (l.slot_h[e] >= 0) {
                gu[e] = l.slots + size_t(l.slot_h[e]) * kSlotBytes;
                d[e] = gu[e] + 2 * kMatBytes;
            } else {
                gu[e] = stg_gu + l.missing.size() * 2 * kMatBytes;
                d[e] = stg_d + l.missing.size() * kMatBytes;
                l.missing.push_back(e);
            }
        }
        if (!l.pf_gu) { CK(hipMalloc(&l.pf_gu, kExperts * sizeof(void*))); CK(hipMalloc(&l.pf_d, kExperts * sizeof(void*))); }
        CK(hipMemcpy(l.pf_gu, gu.data(), kExperts * sizeof(void*), hipMemcpyHostToDevice));
        CK(hipMemcpy(l.pf_d, d.data(), kExperts * sizeof(void*), hipMemcpyHostToDevice));
    }
    void gemm8(W8 W, const float* bias, int N, int K, int T, float* Y, int ldy, int add) {
        k_gemm8<<<dim3((N + 255) / 256, (T + 63) / 64), 256, 0, st>>>(W, bias, N, K, pXH, pXL, pXS, T, Y, ldy, add);
    }
    // T tokens at positions pos .. pos + T - 1; afterwards the decode state is the next position and the most likely
    // next token (row_logits: every row's logits through the head, one row at a time, into the callback)
    void prefill_chunk(const int* ids, int T, const std::function<void(int)>& row_logits = nullptr) {
        if (pos + T > max_ctx) { std::fprintf(stderr, "strata-gptoss: context full (%d)\n", max_ctx); std::exit(1); }
        if (T > tc) { std::fprintf(stderr, "strata-gptoss: a chunk of %d (at most %d)\n", T, tc); std::exit(1); }
        const int p0 = pos, S = T * kTopK, max_tiles = S / 64 + kExperts + 1;
        std::memcpy(pIdsHost, ids, T * 4);
        CK(hipMemcpyAsync(pIds, pIdsHost, T * 4, hipMemcpyHostToDevice, st));
        k_pf_embed<<<T, 256, 0, st>>>(embed, pIds, pH);
        for (int i = 0; i < n_layers(); ++i) if (L[i].pf_dirty) { CK(hipStreamSynchronize(st)); pf_tables(i); L[i].pf_dirty = false; }
        copy_gu(0);
        for (int i = 0; i < n_layers(); ++i) {
            Layer& l = L[i];
            copy_d(i);
            k_split16<<<T, 256, 0, st>>>(pH, l.attn_norm, kHidden, pXH, pXL, pXS);
            gemm8(l.wqkv, l.bqkv, kQkv, kHidden, T, pQKV, kQkv, 0);
            k_pf_qkpost<<<dim3(kHeads + kKv, T), 32, 0, st>>>(pQKV, rope, l.K, l.V, p0, l.kv_mask);
            k_pf_attn<<<dim3(kKv, (T + 15) / 16), 256, 0, st>>>(pQKV, l.K, l.V, l.sinks, pATT, p0, T, l.swa, l.kv_mask);
            k_split16<<<T, 256, 0, st>>>(pATT, nullptr, kQDim, pXH, pXL, pXS);
            gemm8(l.wo, l.bo, kHidden, kQDim, T, pH, kHidden, 1);
            k_pf_norm<<<T, 256, 0, st>>>(pH, l.post_norm, pXN);
            k_pf_router<<<(T + 3) / 4, 128, 0, st>>>(pXN, l.router_t, l.rb, T, pRL);
            CK(hipMemsetAsync(pCount, 0, kExperts * 4, st));
            k_pf_topk<<<(T + 7) / 8, 256, 0, st>>>(pRL, T, pIdx, pW, pCount);
            // GPTOSS_ROUTE_DUMP=FILE: every token's 4 experts, per (chunk, layer) as int16 records [layer, T] then T x 4 ids
            static FILE* route_f = std::getenv("GPTOSS_ROUTE_DUMP") ? std::fopen(std::getenv("GPTOSS_ROUTE_DUMP"), "ab") : nullptr;
            if (route_f) {
                std::vector<int> ix(size_t(T) * kTopK);
                CK(hipMemcpyAsync(ix.data(), pIdx, ix.size() * 4, hipMemcpyDeviceToHost, st));
                CK(hipStreamSynchronize(st));
                std::vector<int16_t> rec(2 + ix.size());
                rec[0] = int16_t(i); rec[1] = int16_t(T);
                for (size_t j = 0; j < ix.size(); ++j) rec[2 + j] = int16_t(ix[j]);
                std::fwrite(rec.data(), 2, rec.size(), route_f);
                std::fflush(route_f);
            }
            k_pf_bounds<<<1, kExperts, 0, st>>>(pCount, pBounds, pFill, pTiles, pNT);
            // the adaptive tier: this chunk's routing counted; its hottest staged experts are promoted into the coldest
            // card slots right after the products that read them (gate / up, then down) - before the staging is reused
            std::vector<std::pair<int, int>> promote;   // (expert in, its staging index) -> the slot of the expert out
            std::vector<int> promote_slot;
            if (adapt_on()) {
                CK(hipMemcpyAsync(pCountHost, pCount, kExperts * 4, hipMemcpyDeviceToHost, st));
                CK(hipStreamSynchronize(st));
                for (int e = 0; e < kExperts; ++e) l.heat[e] += float(pCountHost[e]) * heat_inc;
                static const int budget = std::getenv("GPTOSS_SWAPS_PROMPT") ? std::atoi(std::getenv("GPTOSS_SWAPS_PROMPT")) : 32;
                std::vector<int> pend;
                for (size_t j = 0; j < l.missing.size(); ++j) if (pCountHost[l.missing[j]] > 0) pend.push_back(int(j));
                std::sort(pend.begin(), pend.end(), [&](int a, int b) { return l.heat[l.missing[a]] > l.heat[l.missing[b]]; });
                for (int j : pend) {
                    if (int(promote.size()) >= budget || !l.n_slots) break;
                    const int e = l.missing[j], c = coldest(l);
                    if (c < 0 || !worth(l, e, c)) break;
                    promote.push_back({e, j});
                    promote_slot.push_back(l.slot_h[c]);
                    l.slot_h[e] = l.slot_h[c];
                    l.slot_h[c] = -1;
                }
            }
            k_pf_scatter<<<(S + 255) / 256, 256, 0, st>>>(pIdx, T, pBounds, pFill, pSrc, pSlot);
            k_split16<<<T, 256, 0, st>>>(pXN, nullptr, kHidden, pXH, pXL, pXS);
            CK(hipStreamWaitEvent(st, ev_gu_ready, 0));
            k_gemm_mx_grp<true><<<dim3((kFf + 255) / 256, max_tiles), 256, 0, st>>>(l.pf_gu, l.bg, l.bu, pXH, pXL, pXS, pSrc, pBounds, pTiles, pNT, pACT);
            for (size_t j = 0; j < promote.size(); ++j)
                CK(hipMemcpyAsync(l.slots + size_t(promote_slot[j]) * kSlotBytes, stg_gu + size_t(promote[j].second) * 2 * kMatBytes, 2 * kMatBytes, hipMemcpyDeviceToDevice, st));
            CK(hipEventRecord(ev_gu_free, st));
            if (i + 1 < n_layers()) copy_gu(i + 1);
            k_split16<<<S, 256, 0, st>>>(pACT, nullptr, kFf, pXH, pXL, pXS);
            CK(hipStreamWaitEvent(st, ev_d_ready, 0));
            k_gemm_mx_grp<false><<<dim3((kHidden + 255) / 256, max_tiles), 256, 0, st>>>(l.pf_d, l.bd, nullptr, pXH, pXL, pXS, nullptr, pBounds, pTiles, pNT, pY);
            for (size_t j = 0; j < promote.size(); ++j)
                CK(hipMemcpyAsync(l.slots + size_t(promote_slot[j]) * kSlotBytes + 2 * kMatBytes, stg_d + size_t(promote[j].second) * kMatBytes, kMatBytes, hipMemcpyDeviceToDevice, st));
            if (!promote.empty()) { n_swaps += promote.size(); push_slots(l); }
            CK(hipEventRecord(ev_d_free, st));
            k_pf_combine<<<T, 256, 0, st>>>(pH, pY, pSlot, pW);
            if (std::getenv("GPTOSS_DUMP_H")) dump_rows(pH, T);
        }
        if (row_logits)
            for (int t = 0; t < T; ++t) {
                k_mv8<<<vocab / kMvRows, 256, kHidden * 4, st>>>(head, nullptr, pH + size_t(t) * kHidden, out_norm, kHidden, logits, 0);
                CK(hipStreamSynchronize(st));
                row_logits(t);
            }
        k_mv8<<<vocab / kMvRows, 256, kHidden * 4, st>>>(head, nullptr, pH + size_t(T - 1) * kHidden, out_norm, kHidden, logits, 0);
        const int st0 = p0 + T - 1;
        CK(hipMemcpyAsync(state, &st0, 4, hipMemcpyHostToDevice, st));
        k_argmax64<<<64, 1024, 0, st>>>(logits, vocab, state, best, done2);
        CK(hipStreamSynchronize(st));
        CK(hipGetLastError());
        pos = p0 + T;
        heat_tick(T);
    }
    void dump_rows(const float* d, int rows) {
        static FILE* f = std::fopen(std::getenv("GPTOSS_DUMP_H"), "wb");
        std::vector<float> v(size_t(rows) * kHidden);
        CK(hipMemcpyAsync(v.data(), d, v.size() * 4, hipMemcpyDeviceToHost, st));
        CK(hipStreamSynchronize(st));
        std::fwrite(v.data(), 4, v.size(), f);
        std::fflush(f);
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
    void set_pos(int p) {
        pos = p;
        CK(hipMemcpyAsync(state, &p, 4, hipMemcpyHostToDevice, st));
        CK(hipStreamSynchronize(st));
    }
};

#include "gptoss_serve.cuh"

}  // namespace

int main(int argc, char** argv) {
    setenv("GPU_MAX_HW_QUEUES", "2", 0);
    std::string path, ids_s, top_path, logits_path;
    int n_new = 0, ctx = 4096;
    bool serve_mode = false;
    std::vector<int> eos_ids = {200002, 200012};   // <|return|>, <|call|>
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
        else { std::fprintf(stderr, "usage: strata-gptoss --model FILE.gguf --ids 1,2,3|@FILE [--new N] [--top FILE] [--logits FILE] [--max-ctx N]\n"); return 2; }
    }
    if (path.empty() || (ids_s.empty() && !serve_mode)) { std::fprintf(stderr, "strata-gptoss: --model and --ids are needed\n"); return 2; }
    std::vector<int> ids;
    { std::stringstream ss(ids_s); for (std::string v; std::getline(ss, v, ',');) if (v.find_first_not_of(" \n\r\t") != std::string::npos) ids.push_back(std::stoi(v)); }
    const auto t0 = std::chrono::steady_clock::now();
    Model m(path, ctx);
    {
        size_t fr = 0, tot = 0;
        CK(hipMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-gptoss: %.1f GB on the card in %.1f s, %.2f GB free; K/V for %d positions %.2f GB\n", m.loaded / 1e9,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), fr / 1e9, ctx, m.kv_bytes / 1e9);
    }
    if (serve_mode) return serve(m, eos_ids, m.tc);
    if (const char* pc = std::getenv("GPTOSS_PPL")) {   // perplexity as llama-perplexity counts it: chunks of n, the second half scored
        const int n = std::atoi(pc);
        std::vector<float> row(m.vocab);
        double nll = 0;
        long cnt = 0;
        for (size_t c0 = 0; c0 + n <= ids.size(); c0 += n) {
            m.set_pos(0);
            const std::vector<int> ch(ids.begin() + c0, ids.begin() + c0 + n);
            int base = 0;
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
    std::vector<float> lg(m.vocab);
    FILE* lf = logits_path.empty() ? nullptr : std::fopen(logits_path.c_str(), "wb");
    FILE* top = top_path.empty() ? nullptr : std::fopen(top_path.c_str(), "w");
    auto write_top = [&](int nxt) {
        CK(hipMemcpy(lg.data(), m.logits, lg.size() * 4, hipMemcpyDeviceToHost));
        if (lf) std::fwrite(lg.data(), 4, lg.size(), lf);
        if (!top) return;
        (void) nxt;
        double mx = *std::max_element(lg.begin(), lg.end()), s = 0;
        for (float v : lg) s += std::exp(double(v) - mx);
        const double lse = mx + std::log(s);
        std::vector<int> o(lg.size());
        for (size_t i = 0; i < o.size(); ++i) o[i] = int(i);
        std::partial_sort(o.begin(), o.begin() + 5, o.end(), [&](int a, int b) { return lg[a] > lg[b]; });
        for (int j = 0; j < 5; ++j) std::fprintf(top, "%d:%.5f%s", o[j], lg[o[j]] - lse, j < 4 ? " " : "\n");
    };
    const auto t1 = std::chrono::steady_clock::now();
    if (std::getenv("GPTOSS_TOKWISE")) m.feed(ids, [&]() { if (lf || top) write_top(m.next_token()); });
    else if (lf || top) m.prefill(ids, [&](int) { write_top(0); });
    else m.prefill(ids);
    const double read_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    const auto t2 = std::chrono::steady_clock::now();
    m.hy_t_wait = m.hy_t_cpu = 0;
    m.hy_cpu = m.hy_card = 0;
    for (int n = 0; n < n_new; ++n) {
        const int nxt = m.next_token();
        std::printf("%d%s", nxt, n + 1 < n_new ? "," : "\n");
        std::fflush(stdout);
        m.step();
        if (lf || top) write_top(m.next_token());
    }
    CK(hipStreamSynchronize(m.st));
    const double write_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
    if (lf) std::fclose(lf);
    if (top) std::fclose(top);
    std::fprintf(stderr, "strata-gptoss: read %zu in %.2f s (%.1f per second), wrote %d in %.2f s (%.2f per second)\n", ids.size(), read_s,
                 ids.size() / read_s, n_new, write_s, n_new / std::max(write_s, 1e-9));
    if (n_new > 0)
        std::fprintf(stderr, "strata-gptoss: %.0f%% of the chosen experts on the card; a token: %.2f ms waiting for the card's post, %.2f ms on the CPU\n",
                     100.0 * m.hy_card / std::max<size_t>(1, m.hy_card + m.hy_cpu), m.hy_t_wait * 1e3 / n_new, m.hy_t_cpu * 1e3 / n_new);
    if (m.n_swaps) std::fprintf(stderr, "strata-gptoss: %zu experts moved onto the card\n", m.n_swaps);
    return 0;
}
