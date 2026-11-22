// src/inkling/inkling_prefill.cuh - strata-inkling's prompt path: chunks of many rows on the matrix units (gfx12 WMMA
// 16x16x16, fp16 in, fp32 sums). The dense products go to hipBLASLt (fp16 weights: the bf16 values exactly, fp16 inputs);
// the experts run grouped by expert in one kernel that unpacks NVFP4 / MXFP8 straight into the operand registers.
// Included by inkling_main.cu.
#pragma once
#ifndef PF_DIAG
#define PF_DIAG 0
#endif

typedef _Float16 pf_v8h __attribute__((ext_vector_type(8)));
typedef _Float16 pf_v2h __attribute__((ext_vector_type(2)));
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
__device__ __forceinline__ uint16_t h16(float v) { return __half_as_ushort(__float2half(fminf(fmaxf(v, -65504.f), 65504.f))); }

// fp32 rows (stride xs) to fp16 rows (stride os), C values a row; grid T
__global__ void __launch_bounds__(256) k_to16(const float* x, int xs, int C, uint16_t* o, int os) {
    const float* r = x + size_t(blockIdx.x) * xs;
    uint16_t* d = o + size_t(blockIdx.x) * os;
    for (int i = threadIdx.x; i < C; i += 256) d[i] = h16(r[i]);
}
// rows to fp16 hi and lo (x = hi + lo near fp32), for the router's products: [T][kH] each
__global__ void __launch_bounds__(256) k_split16(const float* x, uint16_t* hi, uint16_t* lo) {
    const size_t t = blockIdx.x;
    for (int i = threadIdx.x; i < kH; i += 256) {
        const float v = x[t * kH + i];
        const __half h = __float2half(v);
        hi[t * kH + i] = __half_as_ushort(h);
        lo[t * kH + i] = __half_as_ushort(__float2half(v - __half2float(h)));
    }
}

// ---------------------------------------------------------------------------------------------------- hipBLASLt
#ifdef INKLING_BLT
// Y (T x N, row stride ldy) = X (T x K fp16, row stride ldx) . W^T, W fp16 row-major (N x K, row stride K). Per shape the
// heuristic's first methods are timed once and the fastest kept (INKLING_BLT_TUNE=0: the first).
struct Blt {
    hipblasLtHandle_t h = nullptr;
    void* ws = nullptr;
    size_t wss = 32u << 20;
    struct Plan { hipblasLtMatmulDesc_t md; hipblasLtMatrixLayout_t la, lb, lc; hipblasLtMatmulAlgo_t algo; };
    std::map<std::tuple<int, int, int, int, int>, Plan> plans;
    std::map<std::tuple<int, int, int, int, int>, hipblasLtMatmulAlgo_t> picks;
#define BK(x) do { hipblasStatus_t e_ = (x); if (e_ != HIPBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "strata-inkling: %s failed: %d (%s:%d)\n", #x, int(e_), __FILE__, __LINE__); std::exit(1); } } while (0)
    void init() { BK(hipblasLtCreate(&h)); CK(hipMalloc(&ws, wss)); }
    hipblasLtMatmulAlgo_t pick(Plan& p, const void* W, const void* X, float* Y, hipStream_t st) {
        const float al = 1.f, be = 0.f;
        hipblasLtMatmulPreference_t pref;
        BK(hipblasLtMatmulPreferenceCreate(&pref));
        BK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wss, sizeof wss));
        hipblasLtMatmulHeuristicResult_t res[6];
        int n = 0;
        BK(hipblasLtMatmulAlgoGetHeuristic(h, p.md, p.la, p.lb, p.lc, p.lc, pref, 6, res, &n));
        hipblasLtMatmulPreferenceDestroy(pref);
        if (n == 0) { std::fprintf(stderr, "strata-inkling: hipBLASLt has no method for this product\n"); std::exit(1); }
        static const bool tune = !std::getenv("INKLING_BLT_TUNE") || std::atoi(std::getenv("INKLING_BLT_TUNE")) != 0;
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
    void run(const __half* W, int N, int K, const uint16_t* X, int ldx, int T, float* Y, int ldy, hipStream_t st) {
        const float al = 1.f, be = 0.f;
        auto key = std::make_tuple(N, K, T, ldx, ldy);
        auto it = plans.find(key);
        if (it == plans.end()) {
            Plan p;
            BK(hipblasLtMatmulDescCreate(&p.md, HIPBLAS_COMPUTE_32F, HIP_R_32F));
            const hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
            BK(hipblasLtMatmulDescSetAttribute(p.md, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
            BK(hipblasLtMatmulDescSetAttribute(p.md, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
            BK(hipblasLtMatrixLayoutCreate(&p.la, HIP_R_16F, K, N, K));
            BK(hipblasLtMatrixLayoutCreate(&p.lb, HIP_R_16F, K, T, ldx));
            BK(hipblasLtMatrixLayoutCreate(&p.lc, HIP_R_32F, N, T, ldy));
            int b = 256;
            while (b < T) b *= 2;
            auto pk = picks.find(std::make_tuple(N, K, b, ldx, ldy));
            bool ok = false;
            if (pk != picks.end()) {
                p.algo = pk->second;
                size_t need = 0;
                ok = hipblaslt_ext::matmulIsAlgoSupported(h, p.md, &al, p.la, p.lb, &be, p.lc, p.lc, p.algo, need) == HIPBLAS_STATUS_SUCCESS && need <= wss;
            }
            if (!ok) {
                p.algo = pick(p, W, X, Y, st);
                if (pk == picks.end()) picks.emplace(std::make_tuple(N, K, b, ldx, ldy), p.algo);
            }
            it = plans.emplace(key, p).first;
        }
        Plan& p = it->second;
        BK(hipblasLtMatmul(h, p.md, &al, W, p.la, X, p.lb, &be, Y, p.lc, Y, p.lc, &p.algo, ws, wss, st));
    }
};
#endif

// ---------------------------------------------------------------------------------------------------- experts
// e4m3 (OCP) byte to fp16 bits, exactly (sub-normals included): the byte's 7 value bits under fp16's are 2^-8 of it
__device__ __forceinline__ _Float16 e4m3_h(uint32_t b) {
    const _Float16 v = __builtin_bit_cast(_Float16, uint16_t(((b & 0x7f) << 7) | ((b & 0x80) << 8)));
    return v * (_Float16) 256.f;
}
// 8 NVFP4 values (4 bytes, the even value in the low nibble) times their e4m3 scale (fp16), as fp16 - exact: the
// E2M1 values' fp16 forms have a zero low byte, the high byte is a lookup by byte permute (00 38 3C 3E 40 42 44 46).
__device__ __forceinline__ pf_v8h nv_cvt(uint32_t w, _Float16 sc) {
    pf_v8h r;
#if PF_DIAG & 1   // (a timing test only: wrong values) no unpacking
    { const _Float16 c = (w & 1) ? sc : (_Float16) 0.f; return pf_v8h{c, c, c, c, c, c, c, c}; }
#endif
    const pf_v2h s2 = {sc, sc};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint32_t t = (w >> (16 * half)) & 0xffff;
        const uint32_t sp = (t & 0xf) | ((t & 0xf0) << 4) | ((t & 0xf00) << 8) | ((t & 0xf000) << 12);   // a nibble a byte
        const uint32_t hb = __builtin_amdgcn_perm(0x46444240u, 0x3e3c3800u, sp & 0x07070707u) | ((sp & 0x08080808u) << 4);
        const pf_v2h a = __builtin_bit_cast(pf_v2h, __builtin_amdgcn_perm(0u, hb, 0x010c000cu)) * s2;
        const pf_v2h b = __builtin_bit_cast(pf_v2h, __builtin_amdgcn_perm(0u, hb, 0x030c020cu)) * s2;
        r[4 * half + 0] = a[0]; r[4 * half + 1] = a[1]; r[4 * half + 2] = b[0]; r[4 * half + 3] = b[1];
    }
    return r;
}
// 8 e4m3 values (8 bytes) times a power of two (fp16) as fp16
__device__ __forceinline__ pf_v8h mx_cvt(uint2 q, _Float16 sc) {
    pf_v8h r;
    const uint32_t w[2] = {q.x, q.y};
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int b = 0; b < 4; ++b) r[4 * j + b] = e4m3_h((w[j] >> (8 * b)) & 0xff) * sc;
    return r;
}
// a lane's weights for one k step of 64: 4 fragments of 8 values (NVFP4: a word each and its scale byte; MXFP8: 8 bytes
// each and the step's 2 scale bytes)
// The k order inside a step of 64 is permuted (A and B alike, so the sums are unchanged): in sub-step kk, lane group g
// holds k = k0 + 32 g + 8 kk .. + 8, so a lane's weights for the step are 32 values in a row: one 16-byte load (NVFP4,
// and 2 scale bytes) or two (MXFP8, 1 scale byte).
struct EFrag { uint4 w[2]; uint8_t s[2]; };
template <bool NV>
__device__ __forceinline__ void ex_fetch(const Experts& ex, bool gu, size_t row, int k0, int g, EFrag& f) {
    const uint8_t* W = gu ? ex.w13 : ex.w2;
    const uint8_t* S = gu ? ex.s13 : ex.s2;
    const size_t wr = gu ? ex.w13_row : ex.w2_row, sr = gu ? ex.s13_row : ex.s2_row;
    const int k = k0 + g * 32;
    if (NV) {
        f.w[0] = *(const uint4*) (W + row * wr + k / 2);
        const uint16_t sc = *(const uint16_t*) (S + row * sr + k / 16);
        f.s[0] = sc & 0xff; f.s[1] = sc >> 8;
    } else {
        f.w[0] = *(const uint4*) (W + row * wr + k);
        f.w[1] = *(const uint4*) (W + row * wr + k + 16);
        f.s[0] = S[row * sr + k / 32];
    }
}
__device__ __forceinline__ uint32_t ef_word(const EFrag& f, int i) {
    const uint4& u = f.w[i >> 2];
    const int j = i & 3;
    return j == 0 ? u.x : j == 1 ? u.y : j == 2 ? u.z : u.w;
}
// MXFP8's 2^(s - 127) times 2^-shift[row] as fp16 (the row's outputs times 2^shift after)
__device__ __forceinline__ _Float16 mx_scale(uint8_t s, int shift) { return (_Float16) ldexpf(1.f, int(s) - 127 - shift); }

// The routed experts on rows grouped by expert (k_gemm8_gb of strata-gemma4 with NVFP4 / MXFP8 weights): row r of the
// grouped order reads input row src[r] (or r). GU: K 2304, the gate (stored row 2o) and up (2o + 1) of output o, act =
// softcap(g, 4) sigmoid(g) softcap(u, 25) as fp16 (the down product's input, 1152 wide); down: K 1152, Y 2304 wide fp32.
// A warp TA x TB tiles of 16 (outputs x rows; GU: TA of gate and TA of up), WN x WT warps, a block of one expert.
template <bool GU, bool NV, int TA, int TB, int WN, int WT>
__global__ void __launch_bounds__(32 * WN * WT) k_moe_grp(Experts ex, const int8_t* __restrict__ shift, const uint16_t* __restrict__ xh,
                                                         const int* __restrict__ src, const int* __restrict__ bounds,
                                                         const int* __restrict__ tiles, const int* __restrict__ n_tiles,
                                                         float* __restrict__ Y, uint16_t* __restrict__ Yh) {
    constexpr int NT = 32 * WN * WT, BR = WN * TA * 16, BT = WT * TB * 16, BK = 64, LD = BK + 8, NM = GU ? 2 : 1;
    constexpr int NJ = (BT * 8 + NT - 1) / NT;
    constexpr int N = GU ? kF : kH, K = GU ? kH : kF;
    __shared__ __align__(16) uint16_t sx[2][BT][LD];
    __shared__ int srow[BT];
    const int tile = blockIdx.y;
    if (tile >= *n_tiles) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, l16 = lane & 15, g = lane >> 4;
    const int wn = warp % WN, wt = warp / WN;
    const int e = tiles[2 * tile], r0 = bounds[e] + tiles[2 * tile + 1], rn = min(BT, bounds[e + 1] - r0);
    const int o0 = blockIdx.x * BR + wn * TA * 16, tw = wt * TB * 16;
    const int nb = min(TB, max(0, (rn - tw + 15) / 16));   // this warp's 16-row tiles that hold rows (the rest skipped)
    for (int i = tid; i < BT; i += NT) srow[i] = i < rn ? (src ? src[r0 + i] : r0 + i) : -1;
    pf_sync();
    size_t wrow[NM][TA];
    int sh[NM][TA];
#pragma unroll
    for (int mm = 0; mm < NM; ++mm)
#pragma unroll
        for (int a = 0; a < TA; ++a) {
            const int o = min(o0 + a * 16 + l16, N - 1);
            wrow[mm][a] = GU ? size_t(e) * 2 * kF + 2 * o + mm : size_t(e) * kH + o;
            sh[mm][a] = NV ? 0 : shift[wrow[mm][a] + (GU ? 0 : size_t(kE) * 2 * kF)];
        }
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
    EFrag af[NM][TA], an[NM][TA];
#pragma unroll
    for (int mm = 0; mm < NM; ++mm)
#pragma unroll
        for (int a = 0; a < TA; ++a) ex_fetch<NV>(ex, GU, wrow[mm][a], 0, g, af[mm][a]);
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
                for (int a = 0; a < TA; ++a)
#if PF_DIAG & 4   // (a timing test only) the weights not read again
                    an[mm][a] = af[mm][a];
#else
                    ex_fetch<NV>(ex, GU, wrow[mm][a], k0 + BK, g, an[mm][a]);
#endif
        }
        pf_v8h bf[2][TB];
#pragma unroll
        for (int b = 0; b < TB; ++b) if (b < nb) bf[0][b] = *(const pf_v8h*) &sx[buf][tw + b * 16 + l16][g * 32];
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            if (kk + 1 < BK / 16)
#pragma unroll
                for (int b = 0; b < TB; ++b) if (b < nb) bf[(kk + 1) & 1][b] =
#if PF_DIAG & 2   // (a timing test only) no LDS reload of the inputs
                    bf[kk & 1][b];
#else
                    *(const pf_v8h*) &sx[buf][tw + b * 16 + l16][g * 32 + (kk + 1) * 8];
#endif
            pf_v8h fa[NM][TA];
#pragma unroll
            for (int mm = 0; mm < NM; ++mm)
#pragma unroll
                for (int a = 0; a < TA; ++a)
                    fa[mm][a] = NV ? nv_cvt(ef_word(af[mm][a], kk), e4m3_h(af[mm][a].s[kk >> 1]))
                                   : mx_cvt(make_uint2(ef_word(af[mm][a], 2 * kk), ef_word(af[mm][a], 2 * kk + 1)),
                                            mx_scale(af[mm][a].s[0], sh[mm][a]));
#pragma unroll
            for (int b = 0; b < TB; ++b)
                if (b < nb)
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
    const float* fac = GU ? ex.f13 : ex.f2;
    const float fe = NV ? fac[e] : 1.f;
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
                // (the row shift of output o: lane l16 = o's position in the tile holds it; fetched again here)
                const size_t wr = GU ? size_t(e) * 2 * kF + 2 * o : size_t(e) * kH + o;
                if (GU) {
                    const float gs = NV ? fe : ldexpf(1.f, shift[wr]), us = NV ? fe : ldexpf(1.f, shift[wr + 1]);
                    const float gv = acc[0][a][b][i] * gs, uv = acc[1][a][b][i] * us;
                    Yh[size_t(r0 + t) * N + o] = h16(softcapf(gv, kGCap) * sigm(gv) * softcapf(uv, kUCap));
                } else {
                    Y[size_t(r0 + t) * N + o] = acc[0][a][b][i] * (NV ? fe : ldexpf(1.f, shift[size_t(kE) * 2 * kF + wr]));
                }
            }
    }
}

// counts per expert from the router's picks
__global__ void k_pf_count(const int* idx, int n, int* count) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) atomicAdd(count + idx[i], 1);
}
// bounds[e] = sum of counts before e; fill reset; tiles of `rows` rows (one block of 128)
__global__ void k_pf_bounds(const int* count, int* bounds, int* fill, int* tiles, int* n_tiles, int rows) {
    __shared__ int sc[kE], stl[kE];
    const int e = threadIdx.x;
    sc[e] = count[e];
    stl[e] = (count[e] + rows - 1) / rows;
    __syncthreads();
    for (int off = 1; off < kE; off <<= 1) {
        const int a = e >= off ? sc[e - off] : 0, b = e >= off ? stl[e - off] : 0;
        __syncthreads();
        sc[e] += a; stl[e] += b;
        __syncthreads();
    }
    const int nt = (count[e] + rows - 1) / rows, base = stl[e] - nt;
    bounds[e + 1] = sc[e];
    if (e == 0) bounds[0] = 0;
    fill[e] = 0;
    for (int i = 0; i < nt; ++i) { tiles[2 * (base + i)] = e; tiles[2 * (base + i) + 1] = i * rows; }
    if (e == kE - 1) *n_tiles = stl[e];
}
__global__ void k_pf_scatter(const int* idx, int n, const int* bounds, int* fill, int* src, int* slot) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const int e = idx[i], p = bounds[e] + atomicAdd(fill + e, 1);
    src[p] = i / kTopK;
    slot[i] = p;
}
// y[t] = sum_k wts[t][k] Yd[slot[t][k]] (the k order fixed)
__global__ void __launch_bounds__(256) k_pf_combine(const float* Yd, const int* slot, const float* wts, float* y) {
    const int t = blockIdx.x;
    for (int c = threadIdx.x; c < kH; c += 256) {
        float a = 0;
#pragma unroll
        for (int k = 0; k < kTopK; ++k) a += wts[t * kTopK + k] * Yd[size_t(slot[t * kTopK + k]) * kH + c];
        y[size_t(t) * kH + c] = a;
    }
}
// router from the products' logits (two row blocks: hi and lo inputs): the writing path's rule
__global__ void k_pf_topk(const float* lg, int T, const float* bias, int* idx, float* wts) {
    const int t = blockIdx.x * 8 + (threadIdx.x >> 5), l = threadIdx.x & 31;
    if (t >= T) return;
    float s[4], v[4];
    for (int j = 0; j < 4; ++j) {
        const int e = l * 4 + j;
        s[j] = sigm(lg[size_t(t) * kE + e] + lg[size_t(T + t) * kE + e]);
        v[j] = s[j] + bias[e];
    }
    float pk[kTopK];
    int pi[kTopK];
    for (int k = 0; k < kTopK; ++k) {
        float bv = -INFINITY, bs = 0;
        int bi = 0x7fffffff;
        for (int j = 0; j < 4; ++j) if (v[j] > bv) { bv = v[j]; bi = l * 4 + j; bs = s[j]; }
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
}

// ---------------------------------------------------------------------------------------------------- attention
// The relative-bias table transposed for the matrix units: rows = distance + kPjOff (zero outside 0 .. 1023), 16 values
// (proj[j][d]) a row, as hi and lo fp16 (hi + lo near fp32)
constexpr int kPjOff = 64, kPjRows = kPjOff + kExt + 32;
// Prompt attention (strata-gemma4's k_pf_attn with Inkling's extras): block (K/V head kh, 16 QW queries from q0): its 3
// query heads x QW query tiles of 16, a warp each, share every key tile of 16 (K and V^T through LDS once). A warp: S^T =
// K . Q^T (lane: query l16, keys g * 8 + i), plus the relative bias of the tile's 31 distances (two products of the
// transposed table with the queries' relative states, hi / lo, through LDS to skew them), online softmax per query
// starting from the sink, O^T += V^T . P^T. Out: o * sigmoid(gate), fp32 and fp16.
// 8 int8 K / V values times their row's scale as 8 fp16 (k_pf_attn on the 8-bit caches)
__device__ __forceinline__ uint4 deq8(const int8_t* p, float s) {
    const uint2 u = *(const uint2*) p;
    const uint32_t uu[2] = {u.x, u.y};
    __half h[8];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int b = 0; b < 4; ++b) h[4 * j + b] = __float2half(float(int8_t(uu[j] >> (8 * b))) * s);
#if PF_DIAG & 8   // (a timing test only: wrong values) no 8-bit unpacking
    return make_uint4(u.x & 0x03ff03ffu, u.y & 0x03ff03ffu, u.x & 0x03ff03ffu, u.y & 0x03ff03ffu);
#endif
    return *(const uint4*) h;
}
// ---- LSA for the prompt path's global passes (INKLING_LSA_PF, default on with INKLING_LSA): a position list a
// (64-query block, K/V head) - the sinks and the best `top` 4-position pools by the max over the block's queries (3 heads
// x 64) of the projected score, scored on the matrix units - and every position from the block's window start on
// (k_pf_attn reads the list's tiles, then the window as before).
// the chunk's queries projected to the map's kLsaR dims, fp16: QP[t][h][r]; grid (T), 256 threads
__global__ void __launch_bounds__(256) k_lsa_qproj(const float* __restrict__ P, const float* __restrict__ U, __half* __restrict__ QP) {
    __shared__ float qs[kQD];
    const int t = blockIdx.x;
    const float* prow = P + size_t(t) * kNP + kRQ;
    for (int i = threadIdx.x; i < kQD; i += 256) qs[i] = prow[i];
    __syncthreads();
    for (int o = threadIdx.x; o < kQH * kLsaR; o += 256) {
        const int h = o / kLsaR, r = o % kLsaR;
        const float* u = U + size_t(h / 3) * kHD * kLsaR + r;
        float a = 0;
        for (int i = 0; i < kHD; ++i) a += qs[h * kHD + i] * u[i * kLsaR];
        QP[(size_t(t) * kQH + h) * kLsaR + r] = __float2half(a);
    }
}
// grid (query blocks of 64, kKV), 1024 threads; lcnt -1: full attention for the block (little history before its window)
__global__ void __launch_bounds__(1024) k_lsa_bsel(const __half* __restrict__ QP, const __half* __restrict__ IDX, float* __restrict__ SC, int maxp,
                                                   int* __restrict__ lst, int* __restrict__ lcnt, int p0, int T, int top, int lsa_min, int cap) {
    __shared__ unsigned hist[4096];
    __shared__ unsigned wsum[32];
    __shared__ int thr, need, nsel, neq;
    const int kh = blockIdx.y, tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, g = lane >> 4;
    const int qb0 = blockIdx.x * 64, sid = blockIdx.x * kKV + kh;
    const int ws = max(0, p0 + qb0 - kLsaWin + 1) & ~15, np = ws / 4;   // pools 1 .. np - 1 are candidates
    if (p0 + qb0 + 1 <= lsa_min || np - 1 <= top) { if (tid == 0) lcnt[sid] = -1; return; }
    for (int i = tid; i < 4096; i += 1024) hist[i] = 0;
    if (tid == 0) { nsel = 0; neq = 0; }
    const pf_v8h zh = pf_v8h{0, 0, 0, 0, 0, 0, 0, 0};
    pf_v8h bq[12][2];   // B operand: QP[query l16 of the tile][head][kk * 16 + g * 8 + i]
    bool qv[4];
#pragma unroll
    for (int qt = 0; qt < 4; ++qt) {
        const int q = qb0 + qt * 16 + l16;
        qv[qt] = q < T;
#pragma unroll
        for (int hh = 0; hh < 3; ++hh)
#pragma unroll
            for (int kk = 0; kk < 2; ++kk)
                bq[qt * 3 + hh][kk] = qv[qt] ? *(const pf_v8h*) (QP + (size_t(q) * kQH + kh * 3 + hh) * kLsaR + kk * 16 + g * 8) : zh;
    }
    __syncthreads();
    float* sc = SC + size_t(sid) * maxp;
    for (int tl = w; tl * 16 < np; tl += 32) {
        const int pool = min(tl * 16 + l16, np - 1);
        const __half* ir = IDX + (size_t(pool) * kKV + kh) * kLsaR + g * 8;
        const pf_v8h a0 = *(const pf_v8h*) ir, a1 = *(const pf_v8h*) (ir + 16);
        float mx[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) mx[i] = -INFINITY;
#pragma unroll
        for (int j = 0; j < 12; ++j) {
            pf_v8f c = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
            c = pf_wmma(a0, bq[j][0], c);
            c = pf_wmma(a1, bq[j][1], c);
            const bool ok = qv[j / 3];
#pragma unroll
            for (int i = 0; i < 8; ++i) mx[i] = ok ? fmaxf(mx[i], c[i]) : mx[i];
        }
#pragma unroll
        for (int o = 8; o > 0; o >>= 1)
#pragma unroll
            for (int i = 0; i < 8; ++i) mx[i] = fmaxf(mx[i], __shfl_xor(mx[i], o, 32));
        if (l16 == 0)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int pp = tl * 16 + g * 8 + i;
                if (pp >= 1 && pp < np) { sc[pp] = mx[i]; atomicAdd(&hist[lsa_key(mx[i]) >> 20], 1u); }
            }
    }
    __syncthreads();
    const int ncand = np - 1, K = min(top, ncand);
    unsigned c4[4], mine = 0;
    for (int j = 0; j < 4; ++j) { c4[j] = hist[4 * tid + j]; mine += c4[j]; }
    unsigned inc = mine;   // inclusive suffix within the warp (lanes lane .. 31)
    for (int o = 1; o < 32; o <<= 1) { const unsigned v = __shfl_down(inc, o, 32); if (lane + o < 32) inc += v; }
    if (lane == 0) wsum[w] = inc;
    __syncthreads();
    unsigned above = inc - mine;
    for (int ww = w + 1; ww < 32; ++ww) above += wsum[ww];
    if (above < unsigned(K) && above + mine >= unsigned(K)) {
        unsigned acc = above;
        for (int j = 3; j >= 0; --j) {
            if (acc + c4[j] >= unsigned(K)) { thr = 4 * tid + j; need = K - int(acc); break; }
            acc += c4[j];
        }
    }
    __syncthreads();
    int* L = lst + size_t(sid) * cap;
    if (tid < 4) L[tid] = tid;   // the sinks
    for (int pp = 1 + tid; pp < np; pp += 1024) {
        const int b = int(lsa_key(sc[pp]) >> 20);
        int slot = -1;
        if (b > thr) slot = atomicAdd(&nsel, 1);
        else if (b == thr && atomicAdd(&neq, 1) < need) slot = atomicAdd(&nsel, 1);
        if (slot >= 0) for (int i = 0; i < 4; ++i) L[4 + 4 * slot + i] = 4 * pp + i;
    }
    __syncthreads();
    const int n = 4 + 4 * nsel, n16 = (n + 15) & ~15;
    for (int j = n + tid; j < n16; j += 1024) L[j] = -1;
    if (tid == 0) lcnt[sid] = n16;
}
template <int QW, bool Q8>
__global__ void __launch_bounds__(32 * 3 * QW) k_pf_attn(const float* __restrict__ P, const void* __restrict__ Kc, const void* __restrict__ Vc,
                                                       const __half* __restrict__ KS, const __half* __restrict__ VS,
                                                       const __half* __restrict__ pjh, const __half* __restrict__ pjl, float* __restrict__ out,
                                                       uint16_t* __restrict__ outh, int p0, int T, int swa, int kv_mask,
                                                       const int* __restrict__ lst = nullptr, const int* __restrict__ lcnt = nullptr, int lcap = 0) {
    constexpr int W = 3 * QW, NT = 32 * W, NS = kHD / 16;
    __shared__ int tp[16];   // (LSA) a list tile's positions
    __shared__ __align__(16) _Float16 ks[16][kHD + 8];
    __shared__ __align__(16) _Float16 vt[kHD][16 + 8];
    __shared__ float bm[W][32][17];
    // grid (query blocks, K/V heads): the blocks running together share a head's keys in the cache
    const int kh = blockIdx.y, tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, g = lane >> 4;
    const int qw = w % QW, h = kh * 3 + w / QW;
    const int qb0 = blockIdx.x * 16 * QW, q0 = qb0 + qw * 16, qt = q0 + l16, qpos = p0 + qt;
    const bool qok = qt < T;
    const float* prow = P + size_t(min(qt, T - 1)) * kNP;
    pf_v8h qb[NS];
#pragma unroll
    for (int s = 0; s < NS; ++s)
#pragma unroll
        for (int i = 0; i < 8; ++i) qb[s][i] = (_Float16) (qok ? prow[kRQ + h * kHD + s * 16 + g * 8 + i] : 0.f);
    pf_v8h rh, rl;   // B operand: r[query l16][g * 8 + i]
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float r = qok ? prow[kRR + h * kDRel + g * 8 + i] : 0.f;
        rh[i] = (_Float16) r;
        rl[i] = (_Float16) (r - float(rh[i]));
    }
    const int last = p0 + min(T, qb0 + 16 * QW) - 1;
    const int first = swa ? max(0, p0 + qb0 - (kWin - 1)) : 0;
    const int my_last = p0 + min(T, q0 + 16) - 1, my_first = swa ? max(0, p0 + q0 - (kWin - 1)) : 0;
    const float sink = qok ? prow[kRS + h] : 0.f;
    float m = sink, l = g == 0 ? 1.f : 0.f;
    pf_v8f o[NS];
#pragma unroll
    for (int j = 0; j < NS; ++j) o[j] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    // (LSA: a position list for this block and head - its tiles first, no bias, then the window from kstart on)
    const int lc = lst ? lcnt[blockIdx.x * kKV + kh] : -1;
    const int ntl = lc > 0 ? lc / 16 : 0;
    const int* L = lst ? lst + size_t(blockIdx.x * kKV + kh) * lcap : nullptr;
    const int kstart = lc >= 0 ? (max(0, p0 + qb0 - kLsaWin + 1) & ~15) : (first & ~15);
    for (int it = 0;; ++it) {
        const bool lt = it < ntl;
        const int k0 = kstart + (it - ntl) * 16;
        if (!lt && k0 > last) break;
        for (int i = tid; i < 16 * kHD / 8; i += NT) {
            const int r = i / (kHD / 8), c = (i % (kHD / 8)) * 8, p = lt ? L[it * 16 + r] : k0 + r;
            const bool ok = lt ? p >= 0 : p <= last;
            if (lt && c == 0) tp[r] = p;
            const size_t base = size_t(p & kv_mask) * kKVD + kh * kHD + c;
            uint4 kq = make_uint4(0, 0, 0, 0), vq = kq;
            if (ok) {
                if (Q8) {
                    const size_t sr = size_t(p & kv_mask) * kKV + kh;
                    kq = deq8((const int8_t*) Kc + base, __half2float(KS[sr]));
                    vq = deq8((const int8_t*) Vc + base, __half2float(VS[sr]));
                } else {
                    kq = *(const uint4*) ((const kc_t*) Kc + base);
                    vq = *(const uint4*) ((const kc_t*) Vc + base);
                }
            }
            *(uint4*) &ks[r][c] = kq;
            const _Float16* vv = (const _Float16*) &vq;
#pragma unroll
            for (int x = 0; x < 8; ++x) vt[c + x][r] = vv[x];
        }
        __syncthreads();
        const bool live = lt || (k0 + 15 >= my_first && k0 <= my_last);
        if (live) {
            pf_v8f sc = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0}, sc2 = sc;
#pragma unroll
            for (int s = 0; s < NS; s += 2) {
                sc = pf_wmma(*(const pf_v8h*) &ks[l16][s * 16 + g * 8], qb[s], sc);
                sc2 = pf_wmma(*(const pf_v8h*) &ks[l16][(s + 1) * 16 + g * 8], qb[s + 1], sc2);
            }
            sc += sc2;
            // the bias: distances dmin .. dmin + 31 (dmin = first query's position - k0 - 15)
            const int dmin = p0 + q0 - k0 - 15;
            const bool near = !lt && dmin < kExt;
            if (near) {
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const int row = dmin + t * 16 + l16 + kPjOff;   // >= 0: dmin >= -(16 QW + 15)
                    const pf_v8h ah = *(const pf_v8h*) (pjh + size_t(row) * kDRel + g * 8);
                    const pf_v8h al = *(const pf_v8h*) (pjl + size_t(row) * kDRel + g * 8);
                    pf_v8f mb = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
                    mb = pf_wmma(ah, rh, mb);
                    mb = pf_wmma(ah, rl, mb);
                    mb = pf_wmma(al, rh, mb);
#pragma unroll
                    for (int i = 0; i < 8; ++i) bm[w][t * 16 + g * 8 + i][l16] = mb[i];
                }
            }
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront", "local");
            float v[8], mx = -INFINITY;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int p = lt ? tp[g * 8 + i] : k0 + g * 8 + i;
                const float b = near ? bm[w][15 + l16 - g * 8 - i][l16] : 0.f;
                const bool ok = qok && (lt ? p >= 0 : (p <= qpos && (!swa || p > qpos - kWin)));
                v[i] = ok ? sc[i] + b : -INFINITY;
                mx = fmaxf(mx, v[i]);
            }
            mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
            const float mn = fmaxf(m, mx);
#if PF_DIAG & 16
            const float cf = fmaxf(0.f, 1.f + (m - mn) * 0.01f);
#else
            const float cf = expf(m - mn);
#endif
            m = mn;
            pf_v8h pb;
            float ps = 0;
#pragma unroll
#if PF_DIAG & 16   // (a timing test only: wrong values) no exp
            for (int i = 0; i < 8; ++i) { const float e = fmaxf(0.f, 1.f + (v[i] - mn) * 0.01f); ps += e; pb[i] = (_Float16) e; }
#else
            for (int i = 0; i < 8; ++i) { const float e = expf(v[i] - mn); ps += e; pb[i] = (_Float16) e; }
#endif
            l = l * cf + ps;
#pragma unroll
            for (int j = 0; j < NS; ++j) o[j] = pf_wmma(*(const pf_v8h*) &vt[j * 16 + l16][g * 8], pb, o[j] * cf);
        }
        __syncthreads();
    }
    l += __shfl_xor(l, 16, 32);
    if (!qok) return;
#pragma unroll
    for (int j = 0; j < NS; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = j * 16 + g * 8 + i;
            const float r = o[j][i] / l * sigm(prow[kRG + h * kHD + d]);
            const size_t at = size_t(qt) * kQD + h * kHD + d;
            out[at] = r;
            outh[at] = __half_as_ushort(__float2half(r));
        }
}

// k_pf_attn with key tiles of 32 (half the barriers and rescales a key)
template <int QW, bool Q8>
__global__ void __launch_bounds__(32 * 3 * QW) k_pf_attn32(const float* __restrict__ P, const void* __restrict__ Kc, const void* __restrict__ Vc,
                                                       const __half* __restrict__ KS, const __half* __restrict__ VS,
                                                       const __half* __restrict__ pjh, const __half* __restrict__ pjl, float* __restrict__ out,
                                                       uint16_t* __restrict__ outh, int p0, int T, int swa, int kv_mask) {
    constexpr int W = 3 * QW, NT = 32 * W, NS = kHD / 16;
    __shared__ __align__(16) _Float16 ks[32][kHD + 8];
    __shared__ __align__(16) _Float16 vt[kHD][32 + 8];
    __shared__ float bm[W][32][17];
    // grid (query blocks, K/V heads): the blocks running together share a head's keys in the cache
    const int kh = blockIdx.y, tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, g = lane >> 4;
    const int qw = w % QW, h = kh * 3 + w / QW;
    const int qb0 = blockIdx.x * 16 * QW, q0 = qb0 + qw * 16, qt = q0 + l16, qpos = p0 + qt;
    const bool qok = qt < T;
    const float* prow = P + size_t(min(qt, T - 1)) * kNP;
    pf_v8h qb[NS];
#pragma unroll
    for (int s = 0; s < NS; ++s)
#pragma unroll
        for (int i = 0; i < 8; ++i) qb[s][i] = (_Float16) (qok ? prow[kRQ + h * kHD + s * 16 + g * 8 + i] : 0.f);
    pf_v8h rh, rl;   // B operand: r[query l16][g * 8 + i]
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float r = qok ? prow[kRR + h * kDRel + g * 8 + i] : 0.f;
        rh[i] = (_Float16) r;
        rl[i] = (_Float16) (r - float(rh[i]));
    }
    const int last = p0 + min(T, qb0 + 16 * QW) - 1;
    const int first = swa ? max(0, p0 + qb0 - (kWin - 1)) : 0;
    const int my_last = p0 + min(T, q0 + 16) - 1, my_first = swa ? max(0, p0 + q0 - (kWin - 1)) : 0;
    const float sink = qok ? prow[kRS + h] : 0.f;
    float m = sink, l = g == 0 ? 1.f : 0.f;
    pf_v8f o[NS];
#pragma unroll
    for (int j = 0; j < NS; ++j) o[j] = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
    for (int k0 = first & ~31; k0 <= last; k0 += 32) {
        for (int i = tid; i < 32 * kHD / 8; i += NT) {
            const int r = i / (kHD / 8), c = (i % (kHD / 8)) * 8, p = k0 + r;
            const bool ok = p <= last;
            const size_t base = size_t(p & kv_mask) * kKVD + kh * kHD + c;
            uint4 kq = make_uint4(0, 0, 0, 0), vq = kq;
            if (ok) {
                if (Q8) {
                    const size_t sr = size_t(p & kv_mask) * kKV + kh;
                    kq = deq8((const int8_t*) Kc + base, __half2float(KS[sr]));
                    vq = deq8((const int8_t*) Vc + base, __half2float(VS[sr]));
                } else {
                    kq = *(const uint4*) ((const kc_t*) Kc + base);
                    vq = *(const uint4*) ((const kc_t*) Vc + base);
                }
            }
            *(uint4*) &ks[r][c] = kq;
            const _Float16* vv = (const _Float16*) &vq;
#pragma unroll
            for (int x = 0; x < 8; ++x) vt[c + x][r] = vv[x];
        }
        __syncthreads();
        const bool live = k0 + 31 >= my_first && k0 <= my_last;
        if (live) {
            pf_v8f scs[2];
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                pf_v8f sc = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0}, sc2 = sc;
#pragma unroll
                for (int s = 0; s < NS; s += 2) {
                    sc = pf_wmma(*(const pf_v8h*) &ks[u * 16 + l16][s * 16 + g * 8], qb[s], sc);
                    sc2 = pf_wmma(*(const pf_v8h*) &ks[u * 16 + l16][(s + 1) * 16 + g * 8], qb[s + 1], sc2);
                }
                scs[u] = sc + sc2;
            }
            float v[2][8], mx = -INFINITY;
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int kb = k0 + 16 * u;
                // the bias: distances dmin .. dmin + 31 (dmin = first query's position - kb - 15; >= -46 for a live warp)
                const int dmin = p0 + q0 - kb - 15;
                const bool near = dmin < kExt;
                if (near) {
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const int row = dmin + t * 16 + l16 + kPjOff;   // >= 0: dmin >= -(16 QW + 15)
                    const pf_v8h ah = *(const pf_v8h*) (pjh + size_t(row) * kDRel + g * 8);
                    const pf_v8h al = *(const pf_v8h*) (pjl + size_t(row) * kDRel + g * 8);
                    pf_v8f mb = pf_v8f{0, 0, 0, 0, 0, 0, 0, 0};
                    mb = pf_wmma(ah, rh, mb);
                    mb = pf_wmma(ah, rl, mb);
                    mb = pf_wmma(al, rh, mb);
#pragma unroll
                    for (int i = 0; i < 8; ++i) bm[w][t * 16 + g * 8 + i][l16] = mb[i];
                }
            }
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront", "local");
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int p = kb + g * 8 + i;
                    const float b = near ? bm[w][15 + l16 - g * 8 - i][l16] : 0.f;
                    const bool ok = qok && p <= qpos && (!swa || p > qpos - kWin);
                    v[u][i] = ok ? scs[u][i] + b : -INFINITY;
                    mx = fmaxf(mx, v[u][i]);
                }
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront", "local");
            }
            mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
            const float mn = fmaxf(m, mx);
#if PF_DIAG & 16
            const float cf = fmaxf(0.f, 1.f + (m - mn) * 0.01f);
#else
            const float cf = expf(m - mn);
#endif
            m = mn;
            pf_v8h pb[2];
            float ps = 0;
#pragma unroll
            for (int u = 0; u < 2; ++u)
#pragma unroll
                for (int i = 0; i < 8; ++i) { const float e = expf(v[u][i] - mn); ps += e; pb[u][i] = (_Float16) e; }
            l = l * cf + ps;
#pragma unroll
            for (int j = 0; j < NS; ++j) {
                o[j] = pf_wmma(*(const pf_v8h*) &vt[j * 16 + l16][g * 8], pb[0], o[j] * cf);
                o[j] = pf_wmma(*(const pf_v8h*) &vt[j * 16 + l16][16 + g * 8], pb[1], o[j]);
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
            const int d = j * 16 + g * 8 + i;
            const float r = o[j][i] / l * sigm(prow[kRG + h * kHD + d]);
            const size_t at = size_t(qt) * kQD + h * kHD + d;
            out[at] = r;
            outh[at] = __half_as_ushort(__float2half(r));
        }
}

// ---------------------------------------------------------------------------------------------------- message gates
// the gate's input for the products: [rms(h), rms(m)] as fp16 (T x 4608)
__global__ void __launch_bounds__(256) k_gate_in(const float* h, int hs, const float* m, int ms, uint16_t* z) {
    const size_t t = blockIdx.x;
    const float* hr = h + t * hs;
    const float* mr = m + t * ms;
    float s1 = 0, s2 = 0;
    for (int c = threadIdx.x; c < kH; c += 256) { s1 += hr[c] * hr[c]; s2 += mr[c] * mr[c]; }
    const float r1 = rsqrtf(block_sum(s1) / kH + kEps), r2 = rsqrtf(block_sum(s2) / kH + kEps);
    for (int c = threadIdx.x; c < kH; c += 256) { z[t * 2 * kH + c] = h16(hr[c] * r1); z[t * 2 * kH + kH + c] = h16(mr[c] * r2); }
}
// out = (1 - tanh(f)) h + tanh(g) m, f / g the products' rows (T x 4608: f then g)
__global__ void __launch_bounds__(256) k_gate_out(const float* h, int hs, const float* m, int ms, const float* fg, float* out, int os) {
    const size_t t = blockIdx.x;
    for (int c = threadIdx.x; c < kH; c += 256)
        out[t * os + c] = (1.f - tanhf(fg[t * 2 * kH + c])) * h[t * hs + c] + tanhf(fg[t * 2 * kH + kH + c]) * m[t * ms + c];
}
