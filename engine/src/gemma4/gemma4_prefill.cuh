// src/gemma4/gemma4_prefill.cuh - strata-gemma4's prompt path: chunks of many tokens on the matrix units (gfx12 WMMA
// 16x16x16, fp16 in, fp32 sums). Inputs go in as two fp16 numbers a value (hi + lo, a row scaled by a power of two:
// near fp32); Q8_0 weights as d * q rounded once to fp16. Included by gemma4_main.cu.
#pragma once

typedef _Float16 pf_v8h __attribute__((ext_vector_type(8)));
typedef float pf_v8f __attribute__((ext_vector_type(8)));
// D = A . B + C in gfx12's wave32 layout: lane (l16, g = lane >> 4) holds A[l16][g * 8 + i], B[g * 8 + i][l16] and
// C/D[g * 8 + i][l16]
__device__ __forceinline__ pf_v8f pf_wmma(const pf_v8h& a, const pf_v8h& b, const pf_v8f& c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}
__device__ __forceinline__ void pf_sync() {   // LDS-only barrier
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// rows of x (T x K fp32; rmsnorm * nw first when nw) to hi / lo fp16 and the row's scale (a power of two)
__global__ void __launch_bounds__(256) k_split16(const float* x, const float* nw, int K, uint16_t* xh, uint16_t* xl, float* xs, int want_lo = 1) {
    const int t = blockIdx.x;
    const float* r = x + size_t(t) * K;
    __shared__ float red[8], redm[8];
    float ss = 0, mx = 0;
    for (int i = threadIdx.x; i < K; i += 256) {
        const float v = r[i];
        ss += v * v;
        mx = fmaxf(mx, fabsf(nw ? v * nw[i] : v));
    }
    ss = warp_sum(ss);
    mx = warp_max(mx);
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = ss; redm[threadIdx.x >> 5] = mx; }
    __syncthreads();
    float tot = 0, m = 0;
    for (int i = 0; i < 8; ++i) { tot += red[i]; m = fmaxf(m, redm[i]); }
    const float rn = nw ? rsqrtf(tot / K + kEps) : 1.f;
    m *= rn;
    const int e = m > 0.f ? ilogbf(m) : 0;
    // hi only: unscaled (the products' outputs need no fix-up, so hipBLASLt can make them); measured on 29K tokens of
    // text the largest input is under 4096, far inside fp16 - saturated in case
    const float up = want_lo ? ldexpf(1.f, 14 - e) : 1.f;
    for (int i = threadIdx.x; i < K; i += 256) {
        float v = r[i];
        if (nw) v = v * rn * nw[i];
        const float a = want_lo ? v * up : fminf(fmaxf(v, -65504.f), 65504.f);
        const __half hi = __float2half(a);
        const __half lo = __float2half(a - __half2float(hi));
        xh[size_t(t) * K + i] = __half_as_ushort(hi);
        if (want_lo) xl[size_t(t) * K + i] = __half_as_ushort(lo);
    }
    if (threadIdx.x == 0) xs[t] = want_lo ? ldexpf(1.f, e - 14) : 1.f;
}

// the 8 fp16 weights of a lane's A operand, values k .. k + 7 of a row
// (each int8 q as fp16 by byte permutes: q ^ 0x80 under the high byte 0x64 is the fp16 1152 + q, exactly; then
// packed subtract and multiply - the same values as (fp16) q * d, a third of the instructions)
typedef _Float16 pf_v2h __attribute__((ext_vector_type(2)));
__device__ __forceinline__ pf_v8h q8_cvt(uint2 q, uint16_t dbits) {
    const _Float16 d = __builtin_bit_cast(_Float16, dbits);
    const uint32_t x0 = q.x ^ 0x80808080u, x1 = q.y ^ 0x80808080u, c = 0x64646464u;
    const pf_v2h off = {(_Float16) -1152.f, (_Float16) -1152.f}, dd = {d, d};
    const pf_v2h h0 = (__builtin_bit_cast(pf_v2h, __builtin_amdgcn_perm(c, x0, 0x04010400u)) + off) * dd;
    const pf_v2h h1 = (__builtin_bit_cast(pf_v2h, __builtin_amdgcn_perm(c, x0, 0x04030402u)) + off) * dd;
    const pf_v2h h2 = (__builtin_bit_cast(pf_v2h, __builtin_amdgcn_perm(c, x1, 0x04010400u)) + off) * dd;
    const pf_v2h h3 = (__builtin_bit_cast(pf_v2h, __builtin_amdgcn_perm(c, x1, 0x04030402u)) + off) * dd;
    pf_v8h r;
    r[0] = h0[0]; r[1] = h0[1]; r[2] = h1[0]; r[3] = h1[1]; r[4] = h2[0]; r[5] = h2[1]; r[6] = h3[0]; r[7] = h3[1];
    return r;
}
// a lane's weights for one k step of 64 (4 fragments of 8 values and the step's 2 scales), loaded a step ahead
struct AFrag { uint2 q[4]; uint16_t d[2]; };
__device__ __forceinline__ void q8_fetch(W8 W, int K, size_t row, int k0, int g, AFrag& f) {
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) f.q[kk] = *(const uint2*) (W.q + row * K + k0 + kk * 16 + g * 8);
    f.d[0] = __half_as_ushort(W.d[row * (K / 32) + (k0 >> 5)]);
    f.d[1] = __half_as_ushort(W.d[row * (K / 32) + (k0 >> 5) + 1]);
}

// Y[t][o] = sum_k W[o][k] x[t][k] for T tokens, W Q8_0 (N x K): a block 8 warps side by side over 256 outputs (2 tiles
// of 16 each), 64 tokens (4 tiles), k 64 a step; weights straight into the operand registers, inputs through LDS
// (double-buffered), hi and lo both
template <bool LO>
__global__ void __launch_bounds__(256) k_gemm8(W8 W, int N, int K, const uint16_t* __restrict__ xh, const uint16_t* __restrict__ xl,
                                               const float* __restrict__ xs, int T, float* __restrict__ Y, int ldy) {
    constexpr int TA = 2, TB = 4, BN = TB * 16, BK = 64, LD = BK + 8, NHL = LO ? 2 : 1, NJ = NHL * BN * 8 / 256;
    __shared__ __align__(16) uint16_t sx[2][NHL][BN][LD];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15, g = lane >> 4;
    const int o0 = blockIdx.x * 256 + warp * TA * 16, t0 = blockIdx.y * BN;
    int wrow[TA];
#pragma unroll
    for (int a = 0; a < TA; ++a) wrow[a] = min(o0 + a * 16 + l16, N - 1);
    auto fetch = [&](int k0, uint4* r) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int i = tid + j * 256, hl = i / (BN * 8), t = (i >> 3) % BN, c = (i & 7) * 8;
            const uint16_t* src = hl ? xl : xh;
            r[j] = t0 + t < T ? *(const uint4*) (src + size_t(t0 + t) * K + k0 + c) : make_uint4(0, 0, 0, 0);
        }
    };
    auto put = [&](const uint4* r, int buf) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int i = tid + j * 256, hl = i / (BN * 8), t = (i >> 3) % BN, c = (i & 7) * 8;
            *(uint4*) &sx[buf][hl][t][c] = r[j];
        }
    };
    pf_v8f acc[TA][TB];
#pragma unroll
    for (int a = 0; a < TA; ++a)
#pragma unroll
        for (int b = 0; b < TB; ++b) acc[a][b] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    uint4 rb[4];
    AFrag af[TA], an[TA];
#pragma unroll
    for (int a = 0; a < TA; ++a) q8_fetch(W, K, wrow[a], 0, g, af[a]);
    fetch(0, rb);
    put(rb, 0);
    pf_sync();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K) {
            fetch(k0 + BK, rb);
#pragma unroll
            for (int a = 0; a < TA; ++a) q8_fetch(W, K, wrow[a], k0 + BK, g, an[a]);
        }
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            pf_v8h fa[TA];
#pragma unroll
            for (int a = 0; a < TA; ++a) fa[a] = q8_cvt(af[a].q[kk], af[a].d[kk >> 1]);
#pragma unroll
            for (int b = 0; b < TB; ++b) {
                const pf_v8h bh = *(const pf_v8h*) &sx[buf][0][b * 16 + l16][kk * 16 + g * 8];
                const pf_v8h bl = *(const pf_v8h*) &sx[buf][NHL - 1][b * 16 + l16][kk * 16 + g * 8];
#pragma unroll
                for (int a = 0; a < TA; ++a) { acc[a][b] = pf_wmma(fa[a], bh, acc[a][b]); if (LO) acc[a][b] = pf_wmma(fa[a], bl, acc[a][b]); }
            }
        }
        if (k0 + BK < K) {
            put(rb, buf ^ 1);
#pragma unroll
            for (int a = 0; a < TA; ++a) af[a] = an[a];
        }
        pf_sync();
        buf ^= 1;
    }
#pragma unroll
    for (int b = 0; b < TB; ++b) {
        const int t = t0 + b * 16 + l16;
        if (t >= T) continue;
        const float sc = xs[t];
#pragma unroll
        for (int a = 0; a < TA; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int o = o0 + a * 16 + g * 8 + i;
                if (o >= N) break;
                Y[size_t(t) * ldy + o] = acc[a][b][i] * sc;
            }
    }
}

// k_gemm8 with bigger warp tiles (no lo half): a warp TA x TB tiles of 16 (rows x tokens), WN x WT warps a block
// (block: WN TA 16 outputs x WT TB 16 tokens), k 64 a step, weights a step ahead in registers, inputs via LDS
template <int TA, int TB, int WN, int WT>
__global__ void __launch_bounds__(32 * WN * WT) k_gemm8b(W8 W, int N, int K, const uint16_t* __restrict__ xh,
                                                         const float* __restrict__ xs, int T, float* __restrict__ Y, int ldy) {
    constexpr int NT = 32 * WN * WT, BR = WN * TA * 16, BT = WT * TB * 16, BK = 64, LD = BK + 8, NJ = BT * 8 / NT;
    __shared__ __align__(16) uint16_t sx[2][BT][LD];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15, g = lane >> 4;
    const int wn = warp % WN, wt = warp / WN;
    const int o0 = blockIdx.x * BR + wn * TA * 16, t0 = blockIdx.y * BT, tw = wt * TB * 16;
    int wrow[TA];
#pragma unroll
    for (int a = 0; a < TA; ++a) wrow[a] = min(o0 + a * 16 + l16, N - 1);
    auto fetch = [&](int k0, uint4* r) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int i = tid + j * NT, t = i >> 3, c = (i & 7) * 8;
            r[j] = t0 + t < T ? *(const uint4*) (xh + size_t(t0 + t) * K + k0 + c) : make_uint4(0, 0, 0, 0);
        }
    };
    auto put = [&](const uint4* r, int buf) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int i = tid + j * NT, t = i >> 3, c = (i & 7) * 8;
            *(uint4*) &sx[buf][t][c] = r[j];
        }
    };
    pf_v8f acc[TA][TB];
#pragma unroll
    for (int a = 0; a < TA; ++a)
#pragma unroll
        for (int b = 0; b < TB; ++b) acc[a][b] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    uint4 rb[NJ];
    AFrag af[TA], an[TA];
#pragma unroll
    for (int a = 0; a < TA; ++a) q8_fetch(W, K, wrow[a], 0, g, af[a]);
    fetch(0, rb);
    put(rb, 0);
    pf_sync();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K) {
            fetch(k0 + BK, rb);
#pragma unroll
            for (int a = 0; a < TA; ++a) q8_fetch(W, K, wrow[a], k0 + BK, g, an[a]);
        }
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            pf_v8h fa[TA];
#pragma unroll
            for (int a = 0; a < TA; ++a) fa[a] = q8_cvt(af[a].q[kk], af[a].d[kk >> 1]);
#pragma unroll
            for (int b = 0; b < TB; ++b) {
                const pf_v8h bh = *(const pf_v8h*) &sx[buf][tw + b * 16 + l16][kk * 16 + g * 8];
#pragma unroll
                for (int a = 0; a < TA; ++a) acc[a][b] = pf_wmma(fa[a], bh, acc[a][b]);
            }
        }
        if (k0 + BK < K) {
            put(rb, buf ^ 1);
#pragma unroll
            for (int a = 0; a < TA; ++a) af[a] = an[a];
        }
        pf_sync();
        buf ^= 1;
    }
#pragma unroll
    for (int b = 0; b < TB; ++b) {
        const int t = t0 + tw + b * 16 + l16;
        if (t >= T) continue;
        const float sc = xs[t];
#pragma unroll
        for (int a = 0; a < TA; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int o = o0 + a * 16 + g * 8 + i;
                if (o < N) Y[size_t(t) * ldy + o] = acc[a][b][i] * sc;
            }
    }
}

// The routed experts on rows grouped by expert: row r of the grouped order reads input row src[r] (or r), tiles of
// 64 rows ((expert, first row) pairs). GU: gate rows e * 1408 + o and up rows e * 1408 + 704 + o of W (K 2816),
// Y = gelu(gate) * up (704 wide); otherwise W's rows e * 2816 + o (K 704), Y 2816 wide.
template <bool GU, bool LO>
__global__ void __launch_bounds__(256) k_gemm8_grp(W8 W, const uint16_t* __restrict__ xh, const uint16_t* __restrict__ xl,
                                                   const float* __restrict__ xs, const int* __restrict__ src,
                                                   const int* __restrict__ bounds, const int* __restrict__ tiles,
                                                   const int* __restrict__ n_tiles, float* __restrict__ Y,
                                                   uint16_t* __restrict__ Yh = nullptr) {
    constexpr int TA = GU ? 1 : 2, TB = 4, BN = TB * 16, BK = 64, LD = BK + 8, NM = GU ? 2 : 1, NHL = LO ? 2 : 1, NJ = NHL * BN * 8 / 256;
    constexpr int N = GU ? kFe : kHidden, K = GU ? kHidden : kFe;
    __shared__ __align__(16) uint16_t sx[2][NHL][BN][LD];
    __shared__ int srow[BN];
    const int tile = blockIdx.y;
    if (tile >= *n_tiles) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15, g = lane >> 4;
    const int e = tiles[2 * tile], r0 = bounds[e] + tiles[2 * tile + 1], rn = min(BN, bounds[e + 1] - r0);
    const int o0 = blockIdx.x * (TA * 128) + warp * TA * 16;
    if (tid < BN) srow[tid] = tid < rn ? (src ? src[r0 + tid] : r0 + tid) : -1;
    pf_sync();
    const size_t base = GU ? size_t(e) * 2 * kFe : size_t(e) * kHidden;
    size_t wrow[NM][TA];
#pragma unroll
    for (int mm = 0; mm < NM; ++mm)
#pragma unroll
        for (int a = 0; a < TA; ++a) wrow[mm][a] = base + mm * kFe + min(o0 + a * 16 + l16, N - 1);
    auto fetch = [&](int k0, uint4* r) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int i = tid + j * 256, hl = i / (BN * 8), t = (i >> 3) % BN, c = (i & 7) * 8;
            const uint16_t* sp = hl ? xl : xh;
            const int row = srow[t];
            r[j] = row >= 0 ? *(const uint4*) (sp + size_t(row) * K + k0 + c) : make_uint4(0, 0, 0, 0);
        }
    };
    auto put = [&](const uint4* r, int buf) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int i = tid + j * 256, hl = i / (BN * 8), t = (i >> 3) % BN, c = (i & 7) * 8;
            *(uint4*) &sx[buf][hl][t][c] = r[j];
        }
    };
    pf_v8f acc[NM][TA][TB];
#pragma unroll
    for (int mm = 0; mm < NM; ++mm)
#pragma unroll
        for (int a = 0; a < TA; ++a)
#pragma unroll
            for (int b = 0; b < TB; ++b) acc[mm][a][b] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    uint4 rb[4];
    AFrag af[NM][TA], an[NM][TA];
#pragma unroll
    for (int mm = 0; mm < NM; ++mm)
#pragma unroll
        for (int a = 0; a < TA; ++a) q8_fetch(W, K, wrow[mm][a], 0, g, af[mm][a]);
    fetch(0, rb);
    put(rb, 0);
    pf_sync();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K) {
            fetch(k0 + BK, rb);
#pragma unroll
            for (int mm = 0; mm < NM; ++mm)
#pragma unroll
                for (int a = 0; a < TA; ++a) q8_fetch(W, K, wrow[mm][a], k0 + BK, g, an[mm][a]);
        }
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            pf_v8h fa[NM][TA];
#pragma unroll
            for (int mm = 0; mm < NM; ++mm)
#pragma unroll
                for (int a = 0; a < TA; ++a) fa[mm][a] = q8_cvt(af[mm][a].q[kk], af[mm][a].d[kk >> 1]);
#pragma unroll
            for (int b = 0; b < TB; ++b) {
                const pf_v8h bh = *(const pf_v8h*) &sx[buf][0][b * 16 + l16][kk * 16 + g * 8];
                const pf_v8h bl = *(const pf_v8h*) &sx[buf][NHL - 1][b * 16 + l16][kk * 16 + g * 8];
#pragma unroll
                for (int mm = 0; mm < NM; ++mm)
#pragma unroll
                    for (int a = 0; a < TA; ++a) {
                        acc[mm][a][b] = pf_wmma(fa[mm][a], bh, acc[mm][a][b]);
                        if (LO) acc[mm][a][b] = pf_wmma(fa[mm][a], bl, acc[mm][a][b]);
                    }
            }
        }
        if (k0 + BK < K) {
            put(rb, buf ^ 1);
#pragma unroll
            for (int mm = 0; mm < NM; ++mm)
#pragma unroll
                for (int a = 0; a < TA; ++a) af[mm][a] = an[mm][a];
        }
        pf_sync();
        buf ^= 1;
    }
#pragma unroll
    for (int b = 0; b < TB; ++b) {
        const int t = b * 16 + l16;
        if (t >= rn) continue;
        const float sc = LO ? xs[srow[t]] : 1.f;   // (hi only: the inputs are unscaled)
#pragma unroll
        for (int a = 0; a < TA; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int o = o0 + a * 16 + g * 8 + i;
                if (o >= N) break;
                const float v = GU ? gelu(acc[0][a][b][i] * sc) * (acc[NM - 1][a][b][i] * sc) : acc[0][a][b][i] * sc;
                if (Yh) Yh[size_t(r0 + t) * N + o] = __half_as_ushort(__float2half(fminf(fmaxf(v, -65504.f), 65504.f)));
                else Y[size_t(r0 + t) * N + o] = v;
            }
    }
}

// k_gemm8_grp (hi only) with the warp grid of k_gemm8b: a warp TA x TB tiles of 16 (outputs x rows; GU: TA of gate and
// TA of up), WN x WT warps, a block WN TA 16 outputs x WT TB 16 rows of one expert (tiles of BT = WT TB 16 rows).
// GU writes gelu(gate) * up as fp16 (the down product's input), down writes fp32.
template <bool GU, int TA, int TB, int WN, int WT>
__global__ void __launch_bounds__(32 * WN * WT) k_gemm8_gb(W8 W, const uint16_t* __restrict__ xh, const int* __restrict__ src,
                                                          const int* __restrict__ bounds, const int* __restrict__ tiles,
                                                          const int* __restrict__ n_tiles, float* __restrict__ Y, uint16_t* __restrict__ Yh) {
    constexpr int NT = 32 * WN * WT, BR = WN * TA * 16, BT = WT * TB * 16, BK = 64, LD = BK + 8, NM = GU ? 2 : 1;
    constexpr int NJ = (BT * 8 + NT - 1) / NT;
    constexpr int N = GU ? kFe : kHidden, K = GU ? kHidden : kFe;
    __shared__ __align__(16) uint16_t sx[2][BT][LD];
    __shared__ int srow[BT];
    const int tile = blockIdx.y;
    if (tile >= *n_tiles) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15, g = lane >> 4;
    const int wn = warp % WN, wt = warp / WN;
    const int e = tiles[2 * tile], r0 = bounds[e] + tiles[2 * tile + 1], rn = min(BT, bounds[e + 1] - r0);
    const int o0 = blockIdx.x * BR + wn * TA * 16, tw = wt * TB * 16;
    for (int i = tid; i < BT; i += NT) srow[i] = i < rn ? (src ? src[r0 + i] : r0 + i) : -1;
    pf_sync();
    const size_t base = GU ? size_t(e) * 2 * kFe : size_t(e) * kHidden;
    size_t wrow[NM][TA];
#pragma unroll
    for (int mm = 0; mm < NM; ++mm)
#pragma unroll
        for (int a = 0; a < TA; ++a) wrow[mm][a] = base + mm * kFe + min(o0 + a * 16 + l16, N - 1);
    auto fetch = [&](int k0, uint4* r) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int i = tid + j * NT, t = i >> 3, c = (i & 7) * 8;
            const int row = i < BT * 8 ? srow[t] : -1;
            r[j] = row >= 0 ? *(const uint4*) (xh + size_t(row) * K + k0 + c) : make_uint4(0, 0, 0, 0);
        }
    };
    auto put = [&](const uint4* r, int buf) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int i = tid + j * NT, t = i >> 3, c = (i & 7) * 8;
            if (i < BT * 8) *(uint4*) &sx[buf][t][c] = r[j];
        }
    };
    pf_v8f acc[NM][TA][TB];
#pragma unroll
    for (int mm = 0; mm < NM; ++mm)
#pragma unroll
        for (int a = 0; a < TA; ++a)
#pragma unroll
            for (int b = 0; b < TB; ++b) acc[mm][a][b] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    uint4 rb[NJ];
    AFrag af[NM][TA], an[NM][TA];
#pragma unroll
    for (int mm = 0; mm < NM; ++mm)
#pragma unroll
        for (int a = 0; a < TA; ++a) q8_fetch(W, K, wrow[mm][a], 0, g, af[mm][a]);
    fetch(0, rb);
    put(rb, 0);
    pf_sync();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K) {
            fetch(k0 + BK, rb);
#pragma unroll
            for (int mm = 0; mm < NM; ++mm)
#pragma unroll
                for (int a = 0; a < TA; ++a) q8_fetch(W, K, wrow[mm][a], k0 + BK, g, an[mm][a]);
        }
        // the step's input fragments a k16 ahead of their products (each LDS read waited on just before its use
        // left the matrix units idle for the LDS latency 16 times a step)
        pf_v8h bf[2][TB];
#pragma unroll
        for (int b = 0; b < TB; ++b) bf[0][b] = *(const pf_v8h*) &sx[buf][tw + b * 16 + l16][g * 8];
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            if (kk + 1 < BK / 16)
#pragma unroll
                for (int b = 0; b < TB; ++b) bf[(kk + 1) & 1][b] = *(const pf_v8h*) &sx[buf][tw + b * 16 + l16][(kk + 1) * 16 + g * 8];
            pf_v8h fa[NM][TA];
#pragma unroll
            for (int mm = 0; mm < NM; ++mm)
#pragma unroll
                for (int a = 0; a < TA; ++a) fa[mm][a] = q8_cvt(af[mm][a].q[kk], af[mm][a].d[kk >> 1]);
#pragma unroll
            for (int b = 0; b < TB; ++b)
#pragma unroll
                for (int mm = 0; mm < NM; ++mm)
#pragma unroll
                    for (int a = 0; a < TA; ++a) acc[mm][a][b] = pf_wmma(fa[mm][a], bf[kk & 1][b], acc[mm][a][b]);
        }
        if (k0 + BK < K) {
            put(rb, buf ^ 1);
#pragma unroll
            for (int mm = 0; mm < NM; ++mm)
#pragma unroll
                for (int a = 0; a < TA; ++a) af[mm][a] = an[mm][a];
        }
        pf_sync();
        buf ^= 1;
    }
#pragma unroll
    for (int b = 0; b < TB; ++b) {
        const int t = tw + b * 16 + l16;
        if (t >= rn) continue;
#pragma unroll
        for (int a = 0; a < TA; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int o = o0 + a * 16 + g * 8 + i;
                if (o >= N) break;
                if (GU) {
                    const float v = gelu(acc[0][a][b][i]) * acc[NM - 1][a][b][i];
                    Yh[size_t(r0 + t) * N + o] = __half_as_ushort(__float2half(fminf(fmaxf(v, -65504.f), 65504.f)));
                } else {
                    Y[size_t(r0 + t) * N + o] = acc[0][a][b][i];
                }
            }
    }
}

// rmsnorm(x) * w + res per token (out may be res)
__global__ void __launch_bounds__(256) k_pf_norm_add(float* out, const float* x, const float* w, const float* res) {
    const size_t o = size_t(blockIdx.x) * kHidden;
    float ss = 0;
    for (int i = threadIdx.x; i < kHidden; i += 256) ss += x[o + i] * x[o + i];
    const float r = rsqrtf(block_sum(ss) / kHidden + kEps);
    for (int i = threadIdx.x; i < kHidden; i += 256) out[o + i] = x[o + i] * r * w[i] + res[o + i];
}
// the router's input per token: rms(x) / sqrt(hidden) * scale
__global__ void __launch_bounds__(256) k_pf_rnorm(const float* x, const float* scale, float* y) {
    const size_t o = size_t(blockIdx.x) * kHidden;
    float ss = 0;
    for (int i = threadIdx.x; i < kHidden; i += 256) ss += x[o + i] * x[o + i];
    const float r = rsqrtf(block_sum(ss) / kHidden + kEps) * rsqrtf(float(kHidden));
    for (int i = threadIdx.x; i < kHidden; i += 256) y[o + i] = x[o + i] * r * scale[i];
}
// router logits for 32 tokens a block (128 threads, one expert each), the inputs through LDS 128 values at a time:
// Rt the router transposed ([2816][128]). (The sums run over i in order, as before.)
constexpr int kRtTok = 32, kRtK = 128;
__global__ void __launch_bounds__(128) k_pf_router(const float* X, const float* Rt, int T, float* logits) {
    __shared__ float xs[kRtK][kRtTok + 1];
    const int t0 = blockIdx.x * kRtTok, e = threadIdx.x;
    float a[kRtTok] = {};
    for (int k0 = 0; k0 < kHidden; k0 += kRtK) {
        __syncthreads();
        for (int j = 0; j < kRtTok; ++j) xs[threadIdx.x][j] = t0 + j < T ? X[size_t(t0 + j) * kHidden + k0 + threadIdx.x] : 0.f;
        __syncthreads();
        for (int i = 0; i < kRtK; ++i) {
            const float w = Rt[size_t(k0 + i) * kExperts + e];
#pragma unroll
            for (int j = 0; j < kRtTok; ++j) a[j] += w * xs[i][j];
        }
    }
    for (int j = 0; j < kRtTok; ++j) if (t0 + j < T) logits[size_t(t0 + j) * kExperts + e] = a[j];
}
// router logits on the matrix units, near fp32: each expert's row as hi + lo fp16 (scaled by a power of two, rsc undoes
// it), the inputs as hi + lo (k_split16 with the router's norm weights), three products (hh, hl, lh). A block: 64
// tokens x 128 experts, warp w the experts 16 w .., four token tiles; operands straight from memory (L2)
__global__ void __launch_bounds__(256) k_pf_router_mm(const __half* __restrict__ Rh, const __half* __restrict__ Rl,
                                                      const float* __restrict__ rsc, const uint16_t* __restrict__ xh,
                                                      const uint16_t* __restrict__ xl, const float* __restrict__ xs, int T,
                                                      float* __restrict__ logits) {
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, g = lane >> 4, t0 = blockIdx.x * 64;
    const int e = w * 16 + l16;
    pf_v8f acc[4];
#pragma unroll
    for (int b = 0; b < 4; ++b) acc[b] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    for (int k0 = 0; k0 < kHidden; k0 += 16) {
        const pf_v8h ah = *(const pf_v8h*) (Rh + size_t(e) * kHidden + k0 + g * 8);
        const pf_v8h al = *(const pf_v8h*) (Rl + size_t(e) * kHidden + k0 + g * 8);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int t = t0 + b * 16 + l16;
            pf_v8h bh = {}, bl = {};
            if (t < T) { bh = *(const pf_v8h*) (xh + size_t(t) * kHidden + k0 + g * 8); bl = *(const pf_v8h*) (xl + size_t(t) * kHidden + k0 + g * 8); }
            acc[b] = pf_wmma(ah, bh, acc[b]);
            acc[b] = pf_wmma(ah, bl, acc[b]);
            acc[b] = pf_wmma(al, bh, acc[b]);
        }
    }
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int t = t0 + b * 16 + l16;
        if (t >= T) continue;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int ee = w * 16 + g * 8 + i;
            logits[size_t(t) * kExperts + ee] = acc[b][i] * xs[t] * rsc[ee];
        }
    }
}
// top 8 per token (a warp each, the writing path's rule); counts per expert
__global__ void k_pf_topk(const float* logits, const float* dscale, int T, int* idx, float* wts, int* count) {
    const int t = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (t >= T) return;
    topk_warp(logits + size_t(t) * kExperts, dscale, idx + t * kTopK, wts + t * kTopK);
    if (lane < kTopK) atomicAdd(count + idx[t * kTopK + lane], 1);
}
// bounds[e] = sum of counts before e; fill reset; tiles of 64 rows (one block of 128)
__global__ void k_pf_bounds(const int* count, int* bounds, int* fill, int* tiles, int* n_tiles, int rows) {
    __shared__ int sc[kExperts], st[kExperts];
    const int e = threadIdx.x;
    sc[e] = count[e];
    st[e] = (count[e] + rows - 1) / rows;
    __syncthreads();
    for (int off = 1; off < kExperts; off <<= 1) {
        const int a = e >= off ? sc[e - off] : 0, b = e >= off ? st[e - off] : 0;
        __syncthreads();
        sc[e] += a; st[e] += b;
        __syncthreads();
    }
    const int nt = (count[e] + rows - 1) / rows, base = st[e] - nt;
    bounds[e + 1] = sc[e];
    if (e == 0) bounds[0] = 0;
    fill[e] = 0;
    for (int i = 0; i < nt; ++i) { tiles[2 * (base + i)] = e; tiles[2 * (base + i) + 1] = i * rows; }
    if (e == kExperts - 1) *n_tiles = st[e];
}
__global__ void k_pf_scatter(const int* idx, int T, const int* bounds, int* fill, int* src, int* slot) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= T * kTopK) return;
    const int e = idx[i], p = bounds[e] + atomicAdd(fill + e, 1);
    src[p] = i / kTopK;
    slot[i] = p;
}
// per token: gelu(gate) * up of the shared MLP's product (GU rows: gate 2112, up 2112)
__global__ void k_pf_glu(const float* GU, float* act, uint16_t* acth = nullptr) {   // (acth: straight to fp16, unscaled)
    const size_t t = blockIdx.x;
    for (int i = threadIdx.x; i < kFs; i += 256) {
        const float v = gelu(GU[t * 2 * kFs + i]) * GU[t * 2 * kFs + kFs + i];
        if (acth) acth[t * kFs + i] = __half_as_ushort(__float2half(fminf(fmaxf(v, -65504.f), 65504.f)));
        else act[t * kFs + i] = v;
    }
}
// per token: h = (rmsnorm(rmsnorm(mlp) * n1 + rmsnorm(sum_k w_k Y[slot_k]) * n2) * nf + a) * scale
__global__ void __launch_bounds__(256) k_pf_ffn_out(float* H, const float* A, const float* MLP, const float* Y, const int* slot,
                                                    const float* wts, const float* n1, const float* n2, const float* nf, const float* scale) {
    __shared__ float moe[kHidden], tt[kHidden];
    const int t = blockIdx.x;
    const size_t o = size_t(t) * kHidden;
    float s1 = 0, s2 = 0;
    for (int i = threadIdx.x; i < kHidden; i += 256) {
        float v = 0;
#pragma unroll
        for (int k = 0; k < kTopK; ++k) v += wts[t * kTopK + k] * Y[size_t(slot[t * kTopK + k]) * kHidden + i];
        moe[i] = v;
        s1 += MLP[o + i] * MLP[o + i];
        s2 += v * v;
    }
    const float r1 = rsqrtf(block_sum(s1) / kHidden + kEps), r2 = rsqrtf(block_sum(s2) / kHidden + kEps);
    float s3 = 0;
    for (int i = threadIdx.x; i < kHidden; i += 256) {
        const float v = MLP[o + i] * r1 * n1[i] + moe[i] * r2 * n2[i];
        tt[i] = v;
        s3 += v * v;
    }
    const float r3 = rsqrtf(block_sum(s3) / kHidden + kEps), sc = *scale;
    for (int i = threadIdx.x; i < kHidden; i += 256) H[o + i] = (tt[i] * r3 * nf[i] + A[o + i]) * sc;
}
__global__ void k_pf_embed(W8 T, const int* ids, float* H) {
    const int t = blockIdx.x;
    const size_t r = size_t(ids[t]);
    for (int i = threadIdx.x; i < kHidden; i += 256)
        H[size_t(t) * kHidden + i] = __half2float(T.d[r * (kHidden / 32) + i / 32]) * float(T.q[r * kHidden + i]) * sqrtf(float(kHidden));
}
// q / k / v of T tokens at positions p0 + t (the writing path's k_qk_dec, a token a grid row); q in place, k / v into
// the cache. grid (nq + nkv, T), HD / 2 threads
template <int HD>
__global__ void __launch_bounds__(HD / 2) k_pf_qk(float* QKV, int ld, const float* qn, const float* kn, const double* inv, kc_t* K,
                                                  kc_t* V, int p0, int kv_mask, int nkv, int koff, int voff) {
    const int b = blockIdx.x, t = blockIdx.y, j = threadIdx.x, pos = p0 + t;
    float* row = QKV + size_t(t) * ld;
    float c, s;
    rope_cs(pos, inv[j], c, s);
    if (b < kHeads) {
        float* x = row + b * HD;
        const float a = x[j], e = x[j + HD / 2];
        const float r = rsqrtf(block_sum(a * a + e * e) / HD + kEps);
        const float a1 = a * r * qn[j], e1 = e * r * qn[j + HD / 2];
        x[j] = a1 * c - e1 * s;
        x[j + HD / 2] = e1 * c + a1 * s;
    } else {
        const int kh = b - kHeads;
        const float* x = row + koff + kh * HD;
        const float* v = row + voff + kh * HD;
        const float a = x[j], e = x[j + HD / 2], va = v[j], ve = v[j + HD / 2];
        const float r = rsqrtf(block_sum(a * a + e * e) / HD + kEps);
        const float rv = rsqrtf(block_sum(va * va + ve * ve) / HD + kEps);
        const float a1 = a * r * kn[j], e1 = e * r * kn[j + HD / 2];
        const size_t o = size_t(pos & kv_mask) * nkv * HD + kh * HD;
        K[o + j] = __float2half(a1 * c - e1 * s);
        K[o + j + HD / 2] = __float2half(e1 * c + a1 * s);
        V[o + j] = __float2half(va * rv);
        V[o + j + HD / 2] = __float2half(ve * rv);
    }
}
// Prompt attention: block (K/V head kh, 16 QW queries from q0): its G query heads x QW query tiles of 16 x SPLIT slices
// of the head dims, a warp each, share every key tile of 16 (K and V^T through LDS once). A warp's part of S^T = K . Q^T
// over its dims (lane: query l16, keys g * 8 + i) goes through LDS and each warp adds its head's / tile's SPLIT parts;
// online softmax per query; O^T += V^T . P^T for the warp's dims. Keys 0 .. (or the window) up to each query's position.
template <int HD, int NKV, int QW, int SPLIT>
__global__ void __launch_bounds__(32 * (kHeads / NKV) * QW * SPLIT) k_pf_attn(const float* __restrict__ QKV, int ld,
                                                 const kc_t* __restrict__ K, const kc_t* __restrict__ V, float* __restrict__ out,
                                                 int p0, int T, int swa, int kv_mask) {
    constexpr int G = kHeads / NKV, W = G * QW * SPLIT, NT = 32 * W, DW = HD / SPLIT, NS = DW / 16;
    __shared__ __align__(16) _Float16 ks[16][HD + 8];
    __shared__ __align__(16) _Float16 vt[HD][16 + 8];
    __shared__ float sp[W][16][17];
    const int kh = blockIdx.x, tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, g = lane >> 4;
    const int slice = w % SPLIT, qw = (w / SPLIT) % QW, h = kh * G + w / (SPLIT * QW);
    const int qb0 = blockIdx.y * 16 * QW, q0 = qb0 + qw * 16, qt = q0 + l16, qpos = p0 + qt, d0 = slice * DW;
    const bool qok = qt < T;
    pf_v8h qb[NS];
#pragma unroll
    for (int s = 0; s < NS; ++s)
#pragma unroll
        for (int i = 0; i < 8; ++i) qb[s][i] = (_Float16) (qok ? QKV[size_t(qt) * ld + h * HD + d0 + s * 16 + g * 8 + i] : 0.f);
    const int last = p0 + min(T, qb0 + 16 * QW) - 1;              // the block's last query position
    const int first = swa ? max(0, p0 + qb0 - (kWindow - 1)) : 0;
    const int my_last = p0 + min(T, q0 + 16) - 1, my_first = swa ? max(0, p0 + q0 - (kWindow - 1)) : 0;
    float m = -INFINITY, l = 0;
    pf_v8f o[NS];
#pragma unroll
    for (int j = 0; j < NS; ++j) o[j] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    for (int k0 = first & ~15; k0 <= last; k0 += 16) {
        for (int i = tid; i < 16 * HD / 8; i += NT) {   // the tile: 16 keys x HD of K and V
            const int r = i / (HD / 8), c = (i % (HD / 8)) * 8, p = k0 + r;
            const bool ok = p <= last;
            const size_t base = size_t(p & kv_mask) * NKV * HD + kh * HD + c;
            const uint4 kq = ok ? *(const uint4*) (K + base) : make_uint4(0, 0, 0, 0);
            const uint4 vq = ok ? *(const uint4*) (V + base) : make_uint4(0, 0, 0, 0);
            *(uint4*) &ks[r][c] = kq;
            const _Float16* vv = (const _Float16*) &vq;
#pragma unroll
            for (int x = 0; x < 8; ++x) vt[c + x][r] = vv[x];
        }
        __syncthreads();
        const bool live = k0 + 15 >= my_first && k0 <= my_last;   // (a warp whose queries see none of this tile skips it)
        if (live) {
            pf_v8f sc = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < NS; ++s) sc = pf_wmma(*(const pf_v8h*) &ks[l16][d0 + s * 16 + g * 8], qb[s], sc);
#pragma unroll
            for (int i = 0; i < 8; ++i) sp[w][g * 8 + i][l16] = sc[i];
        }
        __syncthreads();
        if (live) {
            float v[8], mx = -INFINITY;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int p = k0 + g * 8 + i;
                float tot = 0;
#pragma unroll
                for (int s = 0; s < SPLIT; ++s) tot += sp[w - slice + s][g * 8 + i][l16];
                const bool ok = qok && p <= qpos && (!swa || p > qpos - kWindow);
                v[i] = ok ? tot : -INFINITY;
                mx = fmaxf(mx, v[i]);
            }
            mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
            const float mn = fmaxf(m, mx), ref = mn == -INFINITY ? 0.f : mn;
            const float cf = expf(m - ref);
            m = mn;
            pf_v8h pb;
            float ps = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) { const float e = expf(v[i] - ref); ps += e; pb[i] = (_Float16) e; }
            l = l * cf + ps;
#pragma unroll
            for (int j = 0; j < NS; ++j) o[j] = pf_wmma(*(const pf_v8h*) &vt[d0 + j * 16 + l16][g * 8], pb, o[j] * cf);
        }
        __syncthreads();
    }
    l += __shfl_xor(l, 16, 32);
    if (!qok) return;
#pragma unroll
    for (int j = 0; j < NS; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) out[size_t(qt) * (kHeads * HD) + h * HD + d0 + j * 16 + g * 8 + i] = o[j][i] / l;
}

// k_pf_attn with KT keys a step, the K tile through LDS and V straight from the cache as the A operand: the matrix
// units' transposing load (global_load_tr_b128) hands each lane 8 keys of one V column, so V needs no LDS transpose.
// A block: HB query heads of one K/V head x QW tiles of 16 queries x SPLIT slices of the head dim (the score partials of
// the slices summed through LDS). Scores in log2 units (exp2).
typedef short pf_v8s __attribute__((ext_vector_type(8)));
template <int HD, int NKV, int HB, int QW, int SPLIT, int KT, bool VL = false>
__global__ void __launch_bounds__(32 * HB * QW * SPLIT) k_pf_fa(const float* __restrict__ QKV, int ld, const kc_t* __restrict__ K,
                                                               const kc_t* __restrict__ V, float* __restrict__ out, int p0, int T,
                                                               int swa, int kv_mask, uint16_t* __restrict__ outh) {
    constexpr int G = kHeads / NKV, W = HB * QW * SPLIT, NT = 32 * W, DW = HD / SPLIT, NS = DW / 16, NK = KT / 16;
    constexpr float kL2e = 1.4426950408889634f;
    __shared__ __align__(16) _Float16 ks[KT][HD + 8];
    __shared__ float sp[SPLIT > 1 ? W : 1][NK][8][32];
    // VL: the tile's V as ready A fragments (fragment f = key tile t, d tile j: 32 lanes x 16 bytes), loaded once a block
    __shared__ __align__(16) uint4 vs[VL ? NK * (HD / 16) : 1][32];
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, g = lane >> 4;
    const int slice = w % SPLIT, qw = (w / SPLIT) % QW, hb = w / (SPLIT * QW);
    const int kh = blockIdx.x / (G / HB), h = kh * G + (blockIdx.x % (G / HB)) * HB + hb;
    const int qb0 = blockIdx.y * 16 * QW, q0 = qb0 + qw * 16, qt = q0 + l16, qpos = p0 + qt, d0 = slice * DW;
    const bool qok = qt < T;
    const size_t rs = size_t(NKV) * HD;
    pf_v8h qb[NS];
#pragma unroll
    for (int s = 0; s < NS; ++s)
#pragma unroll
        for (int i = 0; i < 8; ++i) qb[s][i] = (_Float16) (qok ? QKV[size_t(qt) * ld + h * HD + d0 + s * 16 + g * 8 + i] : 0.f);
    const int last = p0 + min(T, qb0 + 16 * QW) - 1;
    const int first = swa ? max(0, p0 + qb0 - (kWindow - 1)) : 0;
    const int my_last = p0 + min(T, q0 + 16) - 1, my_first = swa ? max(0, p0 + q0 - (kWindow - 1)) : 0;
    // this lane's row and column of a transposed 16 x 16 load
    const int tr_r = (lane & 7) + 8 * (lane >> 4), tr_c = 8 * ((lane >> 3) & 1);
    float m = -INFINITY, l = 0;
    pf_v8f o[NS];
#pragma unroll
    for (int j = 0; j < NS; ++j) o[j] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    for (int k0 = first & ~(KT - 1); k0 <= last; k0 += KT) {
        for (int i = tid; i < KT * HD / 8; i += NT) {
            const int r = i / (HD / 8), c = (i % (HD / 8)) * 8, p = min(k0 + r, last);
            *(uint4*) &ks[r][c] = *(const uint4*) (K + size_t(p & kv_mask) * rs + kh * HD + c);
        }
        if (VL)
            for (int f = w; f < NK * (HD / 16); f += W) {
                const int t = f / (HD / 16), j = f % (HD / 16), p = min(k0 + t * 16 + tr_r, last);
                const pf_v8s vv = __builtin_amdgcn_global_load_tr_b128_v8i16(
                    (pf_v8s __attribute__((address_space(1)))*) (V + size_t(p & kv_mask) * rs + kh * HD + j * 16 + tr_c));
                vs[f][lane] = __builtin_bit_cast(uint4, vv);
            }
        __syncthreads();
        const bool live = k0 + KT - 1 >= my_first && k0 <= my_last;
        pf_v8f sc[NK];
        if (live) {
            // CH independent sums a tile (one chain of NS dependent products left the matrix units waiting on their
            // own latency: 1 chain runs at a fifth of the peak, 4 chains near it)
            constexpr int CH = NK >= 4 ? 1 : 4 / NK;
            pf_v8f part[NK][CH];
#pragma unroll
            for (int t = 0; t < NK; ++t)
#pragma unroll
                for (int c = 0; c < CH; ++c) part[t][c] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int s = 0; s < NS; ++s)
#pragma unroll
                for (int t = 0; t < NK; ++t) part[t][s % CH] = pf_wmma(*(const pf_v8h*) &ks[t * 16 + l16][d0 + s * 16 + g * 8], qb[s], part[t][s % CH]);
#pragma unroll
            for (int t = 0; t < NK; ++t) {
                sc[t] = part[t][0];
#pragma unroll
                for (int c = 1; c < CH; ++c) sc[t] += part[t][c];
                if (SPLIT > 1)
#pragma unroll
                    for (int i = 0; i < 8; ++i) sp[w][t][i][lane] = sc[t][i];
            }
        }
        if (SPLIT > 1) __syncthreads();
        if (live) {
            float mx = -INFINITY;
#pragma unroll
            for (int t = 0; t < NK; ++t)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float tot = sc[t][i];
                    if (SPLIT > 1) {
                        tot = 0;
#pragma unroll
                        for (int s = 0; s < SPLIT; ++s) tot += sp[w - slice + s][t][i][lane];
                    }
                    const int p = k0 + t * 16 + g * 8 + i;
                    const bool ok = qok && p <= qpos && (!swa || p > qpos - kWindow);
                    sc[t][i] = ok ? tot * kL2e : -INFINITY;
                    mx = fmaxf(mx, sc[t][i]);
                }
            mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
            const float mn = fmaxf(m, mx), ref = mn == -INFINITY ? 0.f : mn;
            const float cf = exp2f(m - ref);
            m = mn;
            float ps = 0;
            pf_v8h pb[NK];
#pragma unroll
            for (int t = 0; t < NK; ++t)
#pragma unroll
                for (int i = 0; i < 8; ++i) { const float e = exp2f(sc[t][i] - ref); ps += e; pb[t][i] = (_Float16) e; }
            l = l * cf + ps;
#pragma unroll
            for (int j = 0; j < NS; ++j) o[j] = o[j] * cf;
#pragma unroll
            for (int t = 0; t < NK; ++t) {
                if (VL) {
#pragma unroll
                    for (int j = 0; j < NS; ++j) o[j] = pf_wmma(__builtin_bit_cast(pf_v8h, vs[t * (HD / 16) + d0 / 16 + j][lane]), pb[t], o[j]);
                    continue;
                }
                const int p = min(k0 + t * 16 + tr_r, last);
                const kc_t* vrow = V + size_t(p & kv_mask) * rs + kh * HD + d0 + tr_c;
#pragma unroll
                for (int j = 0; j < NS; ++j) {
                    const pf_v8s vv = __builtin_amdgcn_global_load_tr_b128_v8i16((pf_v8s __attribute__((address_space(1)))*) (vrow + j * 16));
                    o[j] = pf_wmma(__builtin_bit_cast(pf_v8h, vv), pb[t], o[j]);
                }
            }
        }
        __syncthreads();
    }
    l += __shfl_xor(l, 16, 32);
    if (!qok) return;
#pragma unroll
    for (int j = 0; j < NS; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const size_t at = size_t(qt) * (kHeads * HD) + h * HD + d0 + j * 16 + g * 8 + i;
            if (outh) outh[at] = __half_as_ushort(__float2half(o[j][i] / l));   // (|o| <= max |v| = 16: no saturation needed)
            else out[at] = o[j][i] / l;
        }
}
