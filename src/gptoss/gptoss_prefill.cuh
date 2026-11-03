// src/gptoss/gptoss_prefill.cuh - strata-gptoss's prompt path: chunks of many tokens on the matrix units (gfx12 WMMA
// 16x16x16, fp16 in, fp32 sums). Inputs go in as two fp16 numbers a value (hi + lo, a row scaled by a power of two:
// near fp32); MXFP4 expert weights convert to fp16 exactly (every scale of this checkpoint is in fp16's range), Q8_0
// weights as d * q rounded once to fp16. Included by gptoss_main.cu.
#pragma once

typedef _Float16 pf_v8h __attribute__((ext_vector_type(8)));
typedef float pf_v8f __attribute__((ext_vector_type(8)));
typedef _Float16 pf_v2h __attribute__((ext_vector_type(2)));
// D = A . B + C in gfx12's wave32 layout: lane (l16, g = lane >> 4) holds A[l16][g * 8 + i], B[g * 8 + i][l16] and
// C/D[g * 8 + i][l16]
__device__ __forceinline__ pf_v8f pf_wmma(const pf_v8h& a, const pf_v8h& b, const pf_v8f& c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}
// LDS-only barrier
__device__ __forceinline__ void pf_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// rows of x (T x K fp32; rmsnorm * nw first when nw) to hi / lo fp16 and the row's scale (a power of two)
__global__ void __launch_bounds__(256) k_split16(const float* x, const float* nw, int K, uint16_t* xh, uint16_t* xl, float* xs) {
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
    const float up = ldexpf(1.f, 14 - e);
    for (int i = threadIdx.x; i < K; i += 256) {
        float v = r[i];
        if (nw) v = v * rn * nw[i];
        const float a = v * up;
        const __half hi = __float2half(a);
        const __half lo = __float2half(a - __half2float(hi));
        xh[size_t(t) * K + i] = __half_as_ushort(hi);
        xl[size_t(t) * K + i] = __half_as_ushort(lo);
    }
    if (threadIdx.x == 0) xs[t] = ldexpf(1.f, e - 14);
}
// rmsnorm(x) * nw, fp32 rows
__global__ void __launch_bounds__(256) k_pf_norm(const float* x, const float* nw, float* y) {
    const int t = blockIdx.x;
    __shared__ float red[8];
    float ss = 0;
    for (int i = threadIdx.x; i < kHidden; i += 256) ss += x[size_t(t) * kHidden + i] * x[size_t(t) * kHidden + i];
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0;
    for (int i = 0; i < 8; ++i) tot += red[i];
    const float r = rsqrtf(tot / kHidden + kEps);
    for (int i = threadIdx.x; i < kHidden; i += 256) y[size_t(t) * kHidden + i] = x[size_t(t) * kHidden + i] * r * nw[i];
}

// the 8 fp16 weights of a lane's A operand, values k .. k + 7 of a row
__device__ __forceinline__ pf_v8h q8_a8(W8 W, int K, int row, int k) {
    const uint2 q = *(const uint2*) (W.q + size_t(row) * K + k);
    const _Float16 d = __builtin_bit_cast(_Float16, W.d[size_t(row) * (K / 32) + (k >> 5)]);
    pf_v8h r;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        r[i] = (_Float16) float(int8_t(q.x >> (8 * i))) * d;
        r[4 + i] = (_Float16) float(int8_t(q.y >> (8 * i))) * d;
    }
    return r;
}
// MXFP4 matrix M (the engine's layout): 4 nibbles (bytes of u, low 4 bits) as fp16 times s
__device__ __forceinline__ void fp4h4(uint32_t u, _Float16 s, _Float16* o) {
    const uint32_t mag = perm(0x0C080604u, 0x03020100u, u & 0x07070707u);
    const uint32_t sg = (u & 0x08080808u) << 4;
    // byte b -> fp16 1024 + b, minus 1024 (exact), the sign, times the block's scale (exact: a power of two in range)
    const uint32_t c = 0x64646464u;
    const uint32_t h0 = perm(c, mag, 0x07010700u), h1 = perm(c, mag, 0x07030702u);
    const pf_v2h off = {(_Float16) -1024.f, (_Float16) -1024.f};
    const pf_v2h a = __builtin_bit_cast(pf_v2h, h0) + off, b = __builtin_bit_cast(pf_v2h, h1) + off;
    const _Float16 v[4] = {a[0], a[1], b[0], b[1]};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint16_t bits = __builtin_bit_cast(uint16_t, v[i]) | uint16_t(((sg >> (8 * i + 7)) & 1u) << 15);
        o[i] = __builtin_bit_cast(_Float16, bits) * s;
    }
}
__device__ __forceinline__ pf_v8h mx_a8(const uint8_t* M, int row, int k) {
    const int b = k >> 5, j0 = k & 31;
    const uint2 q = *(const uint2*) (M + size_t(row) * (kHidden / 2) + b * 16 + (j0 & 15));
    const uint32_t e = M[kMatQ + size_t(row) * (kHidden / 32) + b];
    const _Float16 s = (_Float16) e8_half(e);
    const int sh = j0 >= 16 ? 4 : 0;
    _Float16 o[8];
    fp4h4((q.x >> sh) & 0x0F0F0F0Fu, s, o);
    fp4h4((q.y >> sh) & 0x0F0F0F0Fu, s, o + 4);
    pf_v8h r;
#pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = o[i];
    return r;
}

// Y[t][o] (= or +=) sum_k W[o][k] x[t][k] + bias[o] for T tokens, W Q8_0 (N x K): a block 8 warps side by side over 256
// outputs (2 tiles of 16 each), 64 tokens (4 tiles), k 64 a step; weights straight into the operand registers, inputs
// through LDS (double-buffered), hi and lo both
__global__ void __launch_bounds__(256) k_gemm8(W8 W, const float* bias, int N, int K, const uint16_t* __restrict__ xh,
                                               const uint16_t* __restrict__ xl, const float* __restrict__ xs, int T,
                                               float* __restrict__ Y, int ldy, int add) {
    constexpr int TA = 2, TB = 4, BN = TB * 16, BK = 64, LD = BK + 8;
    __shared__ __align__(16) uint16_t sx[2][2][BN][LD];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15, g = lane >> 4;
    const int o0 = blockIdx.x * 256 + warp * TA * 16, t0 = blockIdx.y * BN;
    int wrow[TA];
#pragma unroll
    for (int a = 0; a < TA; ++a) wrow[a] = min(o0 + a * 16 + l16, N - 1);
    auto fetch = [&](int k0, uint4* r) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = tid + j * 256, hl = i >> 9, t = (i >> 3) & 63, c = (i & 7) * 8;
            const uint16_t* src = hl ? xl : xh;
            r[j] = t0 + t < T ? *(const uint4*) (src + size_t(t0 + t) * K + k0 + c) : make_uint4(0, 0, 0, 0);
        }
    };
    auto put = [&](const uint4* r, int buf) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = tid + j * 256, hl = i >> 9, t = (i >> 3) & 63, c = (i & 7) * 8;
            *(uint4*) &sx[buf][hl][t][c] = r[j];
        }
    };
    pf_v8f acc[TA][TB];
#pragma unroll
    for (int a = 0; a < TA; ++a)
#pragma unroll
        for (int b = 0; b < TB; ++b) acc[a][b] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    uint4 rb[4];
    fetch(0, rb);
    put(rb, 0);
    pf_sync();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K) fetch(k0 + BK, rb);
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            pf_v8h fa[TA];
#pragma unroll
            for (int a = 0; a < TA; ++a) fa[a] = q8_a8(W, K, wrow[a], k0 + kk * 16 + g * 8);
#pragma unroll
            for (int b = 0; b < TB; ++b) {
                const pf_v8h bh = *(const pf_v8h*) &sx[buf][0][b * 16 + l16][kk * 16 + g * 8];
                const pf_v8h bl = *(const pf_v8h*) &sx[buf][1][b * 16 + l16][kk * 16 + g * 8];
#pragma unroll
                for (int a = 0; a < TA; ++a) { acc[a][b] = pf_wmma(fa[a], bh, acc[a][b]); acc[a][b] = pf_wmma(fa[a], bl, acc[a][b]); }
            }
        }
        if (k0 + BK < K) put(rb, buf ^ 1);
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
                float* y = Y + size_t(t) * ldy + o;
                const float v = acc[a][b][i] * sc + (bias ? bias[o] : 0.f);
                *y = add ? *y + v : v;
            }
    }
}

// The routed experts on rows grouped by expert: row r of the grouped order reads input row src[r] (or r), tiles of
// 64 rows ((expert, first row) pairs). Wp[e]: expert e's matrices (on the card, or in the staging buffers). GU: gate
// (Wp[e]) and up (+ kMatBytes) both, Y = swiglu(gate + bg, up + bu); otherwise Y = down (Wp[e]) . x + bd. N = K = 2880.
template <bool GU>
__global__ void __launch_bounds__(256) k_gemm_mx_grp(const uint8_t* const* __restrict__ Wp, const float* b1, const float* b2,
                                                     const uint16_t* __restrict__ xh, const uint16_t* __restrict__ xl,
                                                     const float* __restrict__ xs, const int* __restrict__ src,
                                                     const int* __restrict__ bounds, const int* __restrict__ tiles,
                                                     const int* __restrict__ n_tiles, float* __restrict__ Y) {
    constexpr int TA = 2, TB = 4, BN = TB * 16, BK = 64, LD = BK + 8, NM = GU ? 2 : 1, N = kHidden, K = kHidden;
    __shared__ __align__(16) uint16_t sx[2][2][BN][LD];
    __shared__ int srow[BN];
    const int tile = blockIdx.y;
    if (tile >= *n_tiles) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15, g = lane >> 4;
    const int e = tiles[2 * tile], r0 = bounds[e] + tiles[2 * tile + 1], rn = min(BN, bounds[e + 1] - r0);
    const int o0 = blockIdx.x * 256 + warp * TA * 16;
    if (tid < BN) srow[tid] = tid < rn ? (src ? src[r0 + tid] : r0 + tid) : -1;
    pf_sync();
    const uint8_t* M[NM];
#pragma unroll
    for (int mm = 0; mm < NM; ++mm) M[mm] = Wp[e] + mm * kMatBytes;
    int wrow[TA];
#pragma unroll
    for (int a = 0; a < TA; ++a) wrow[a] = min(o0 + a * 16 + l16, N - 1);
    auto fetch = [&](int k0, uint4* r) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = tid + j * 256, hl = i >> 9, t = (i >> 3) & 63, c = (i & 7) * 8;
            const uint16_t* sp = hl ? xl : xh;
            const int row = srow[t];
            r[j] = row >= 0 ? *(const uint4*) (sp + size_t(row) * K + k0 + c) : make_uint4(0, 0, 0, 0);
        }
    };
    auto put = [&](const uint4* r, int buf) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = tid + j * 256, hl = i >> 9, t = (i >> 3) & 63, c = (i & 7) * 8;
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
    fetch(0, rb);
    put(rb, 0);
    pf_sync();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K) fetch(k0 + BK, rb);
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            pf_v8h fa[NM][TA];
#pragma unroll
            for (int mm = 0; mm < NM; ++mm)
#pragma unroll
                for (int a = 0; a < TA; ++a) fa[mm][a] = mx_a8(M[mm], wrow[a], k0 + kk * 16 + g * 8);
#pragma unroll
            for (int b = 0; b < TB; ++b) {
                const pf_v8h bh = *(const pf_v8h*) &sx[buf][0][b * 16 + l16][kk * 16 + g * 8];
                const pf_v8h bl = *(const pf_v8h*) &sx[buf][1][b * 16 + l16][kk * 16 + g * 8];
#pragma unroll
                for (int mm = 0; mm < NM; ++mm)
#pragma unroll
                    for (int a = 0; a < TA; ++a) {
                        acc[mm][a][b] = pf_wmma(fa[mm][a], bh, acc[mm][a][b]);
                        acc[mm][a][b] = pf_wmma(fa[mm][a], bl, acc[mm][a][b]);
                    }
            }
        }
        if (k0 + BK < K) put(rb, buf ^ 1);
        pf_sync();
        buf ^= 1;
    }
#pragma unroll
    for (int b = 0; b < TB; ++b) {
        const int t = b * 16 + l16;
        if (t >= rn) continue;
        const float sc = xs[srow[t]];
#pragma unroll
        for (int a = 0; a < TA; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int o = o0 + a * 16 + g * 8 + i;
                if (o >= N) break;
                const float v = GU ? swiglu_oai(acc[0][a][b][i] * sc + b1[size_t(e) * kFf + o], acc[NM - 1][a][b][i] * sc + b2[size_t(e) * kFf + o])
                                   : acc[0][a][b][i] * sc + b1[size_t(e) * kHidden + o];
                Y[size_t(r0 + t) * N + o] = v;
            }
    }
}

// router logits for 4 tokens a block (128 threads, one expert each): Rt the router transposed (bf16, [2880][128])
__global__ void __launch_bounds__(128) k_pf_router(const float* X, const uint16_t* Rt, const float* rb, int T, float* logits) {
    __shared__ float xs[4][kHidden];
    const int t0 = blockIdx.x * 4, e = threadIdx.x;
    for (int j = 0; j < 4; ++j)
        for (int i = threadIdx.x; i < kHidden; i += 128) xs[j][i] = t0 + j < T ? X[size_t(t0 + j) * kHidden + i] : 0.f;
    __syncthreads();
    float a[4] = {};
    for (int i = 0; i < kHidden; ++i) {
        const float w = __uint_as_float(uint32_t(Rt[size_t(i) * kExperts + e]) << 16);
#pragma unroll
        for (int j = 0; j < 4; ++j) a[j] += w * xs[j][i];
    }
    for (int j = 0; j < 4; ++j) if (t0 + j < T) logits[size_t(t0 + j) * kExperts + e] = a[j] + rb[e];
}
// top 4 per token (one warp a token, the writing path's tie rule), softmax weights; counts per expert
__global__ void k_pf_topk(const float* logits, int T, int* idx, float* wts, int* count) {
    const int t = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (t >= T) return;
    float sc[4];
    for (int j = 0; j < 4; ++j) sc[j] = logits[size_t(t) * kExperts + lane * 4 + j];
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
    for (int k = 0; k < kTopK; ++k) { ex[k] = expf(picked[k] - picked[0]); tot += ex[k]; }
    if (lane < kTopK) {
        idx[t * kTopK + lane] = pi[lane];
        wts[t * kTopK + lane] = ex[lane] / tot;
        atomicAdd(count + pi[lane], 1);
    }
}
// bounds[e] = sum of counts before e; fill reset; tiles of 64 rows (one block of 128)
__global__ void k_pf_bounds(const int* count, int* bounds, int* fill, int* tiles, int* n_tiles) {
    __shared__ int sc[kExperts], st[kExperts];
    const int e = threadIdx.x;
    sc[e] = count[e];
    st[e] = (count[e] + 63) / 64;
    __syncthreads();
    for (int off = 1; off < kExperts; off <<= 1) {
        const int a = e >= off ? sc[e - off] : 0, b = e >= off ? st[e - off] : 0;
        __syncthreads();
        sc[e] += a; st[e] += b;
        __syncthreads();
    }
    const int nt = (count[e] + 63) / 64, base = st[e] - nt;
    bounds[e + 1] = sc[e];
    if (e == 0) bounds[0] = 0;
    fill[e] = 0;
    for (int i = 0; i < nt; ++i) { tiles[2 * (base + i)] = e; tiles[2 * (base + i) + 1] = i * 64; }
    if (e == kExperts - 1) *n_tiles = st[e];
}
__global__ void k_pf_scatter(const int* idx, int T, const int* bounds, int* fill, int* src, int* slot) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= T * kTopK) return;
    const int e = idx[i], p = bounds[e] + atomicAdd(fill + e, 1);
    src[p] = i / kTopK;
    slot[i] = p;
}
// H[t] += sum_k w[t][k] * Y[slot[t][k]]
__global__ void k_pf_combine(float* H, const float* Y, const int* slot, const float* wts) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < kHidden; i += 256) {
        float s = 0;
        for (int k = 0; k < kTopK; ++k) s += wts[t * kTopK + k] * Y[size_t(slot[t * kTopK + k]) * kHidden + i];
        H[size_t(t) * kHidden + i] += s;
    }
}
__global__ void k_pf_embed(const uint8_t* table, const int* ids, float* H) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < kHidden; i += 256) {
        const uint8_t* B = table + size_t(ids[t]) * (kHidden / 32) * 34 + (i / 32) * 34;
        H[size_t(t) * kHidden + i] = __half2float(*(const __half*) B) * float(int8_t(B[2 + i % 32]));
    }
}
// RoPE on q (times 1/8, in place) and k at positions p0 + t; k and v into the cache. grid (kHeads + kKv, T), 32 threads
__global__ void k_pf_qkpost(float* QKV, const float2* rope, kc_t* K, kc_t* V, int p0, int kv_mask) {
    const int hb = blockIdx.x, t = blockIdx.y, l = threadIdx.x, pos = p0 + t;
    float* row = QKV + size_t(t) * kQkv;
    const float2 cs = rope[size_t(pos) * 32 + l];
    if (hb < kHeads) {
        float* q = row + hb * kHd;
        const float a = q[l], b = q[l + 32];
        q[l] = (a * cs.x - b * cs.y) * 0.125f;
        q[l + 32] = (b * cs.x + a * cs.y) * 0.125f;
    } else {
        const int kh = hb - kHeads;
        const float* k = row + kQDim + kh * kHd;
        const float* v = row + kQDim + kKvDim + kh * kHd;
        const float a = k[l], b = k[l + 32];
        const size_t r = size_t(pos & kv_mask) * kKvDim + kh * kHd;
        K[r + l] = __float2half(a * cs.x - b * cs.y);
        K[r + l + 32] = __float2half(b * cs.x + a * cs.y);
        V[r + l] = __float2half(v[l]);
        V[r + l + 32] = __float2half(v[l + 32]);
    }
}
// Prompt attention: block (K/V head kh, 16 queries t0 ..), warp w its query head kh * 8 + w. Key tiles of 16 go through
// LDS once for the 8 heads: S^T = K . Q^T (lane: query l16, keys g * 8 + i), online softmax per query, O^T += V^T . P^T.
// The sink joins the softmax at the end. Keys 0 .. (or the window) up to each query's own position.
__global__ void __launch_bounds__(256) k_pf_attn(const float* __restrict__ QKV, const kc_t* __restrict__ K, const kc_t* __restrict__ V,
                                                 const float* sinks, float* __restrict__ out, int p0, int T, int swa, int kv_mask) {
    __shared__ __align__(16) _Float16 ks[16][kHd + 8];
    __shared__ __align__(16) _Float16 vt[kHd][16 + 8];
    const int kh = blockIdx.x, q0 = blockIdx.y * 16, tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, g = lane >> 4;
    const int h = kh * 8 + w, qt = q0 + l16, qpos = p0 + qt;   // this lane's query
    const bool qok = qt < T;
    pf_v8h qb[4];
#pragma unroll
    for (int s = 0; s < 4; ++s)
#pragma unroll
        for (int i = 0; i < 8; ++i) qb[s][i] = (_Float16) (qok ? QKV[size_t(qt) * kQkv + h * kHd + s * 16 + g * 8 + i] : 0.f);
    const int last = p0 + min(T, q0 + 16) - 1;                 // the block's last query position
    const int first = swa ? max(0, p0 + q0 - (kWindow - 1)) : 0;
    const int a0 = first & ~15;
    float m = -INFINITY, l = 0;
    pf_v8f o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    for (int k0 = a0; k0 <= last; k0 += 16) {
        {   // the tile: 16 keys x 64 dims of K and V (256 threads, 4 values each)
            const int r = tid >> 4, c = (tid & 15) * 4, p = k0 + r;
            const bool ok = p <= last;
            const kc_t* kr = K + size_t(p & kv_mask) * kKvDim + kh * kHd + c;
            const kc_t* vr = V + size_t(p & kv_mask) * kKvDim + kh * kHd + c;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ks[r][c + i] = ok ? (_Float16) __half2float(kr[i]) : (_Float16) 0.f;
                vt[c + i][r] = ok ? (_Float16) __half2float(vr[i]) : (_Float16) 0.f;
            }
        }
        __syncthreads();
        pf_v8f sc = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int s = 0; s < 4; ++s) sc = pf_wmma(*(const pf_v8h*) &ks[l16][s * 16 + g * 8], qb[s], sc);
        float v[8], mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int p = k0 + g * 8 + i;
            const bool ok = qok && p <= qpos && (!swa || p > qpos - kWindow) && p >= 0;
            v[i] = ok ? sc[i] : -INFINITY;
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
        for (int j = 0; j < 4; ++j) o[j] = pf_wmma(*(const pf_v8h*) &vt[j * 16 + l16][g * 8], pb, o[j] * cf);
        __syncthreads();
    }
    l += __shfl_xor(l, 16, 32);
    if (!qok) return;
    const float sk = sinks[h], M = fmaxf(m, sk), f = expf(m - M), L = l * f + expf(sk - M);
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) out[size_t(qt) * kQDim + h * kHd + j * 16 + g * 8 + i] = o[j][i] * f / L;
}
