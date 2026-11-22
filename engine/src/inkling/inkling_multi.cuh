// The writing path for a few rows at once (2 .. 8: the speculation check): every weight read once for all the rows.
// The rows sit in LDS as fp16 and the weights are unpacked to fp16 pairs (exact) for v_dot2_f32_f16, summed in fp32.
// Experts: a list of the distinct experts the rows chose, then one pass over each of them for all its rows.
#pragma once

constexpr int kMR = 8;   // rows at most
typedef _Float16 mh2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float mdot2(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(mh2, a), __builtin_bit_cast(mh2, b), c, false);
}
// bytes 2i, 2i + 1 of w (signed int8) as an fp16 pair, exact: 1024 + (b + 128) - 1152
__device__ __forceinline__ uint32_t i8h2(uint32_t w, int i) {
    const uint32_t v = (w >> (16 * i)) & 0xffff;
    const uint32_t p = ((v & 0xff) | ((v & 0xff00) << 8)) ^ 0x00800080u | 0x64006400u;
    const mh2 r = __builtin_bit_cast(mh2, p) - mh2{(_Float16) 1152.f, (_Float16) 1152.f};
    return __builtin_bit_cast(uint32_t, r);
}
// bytes 2i, 2i + 1 of w (e4m3) as an fp16 pair of the values / 256, exact
__device__ __forceinline__ uint32_t e4h2(uint32_t w, int i) {
    const uint32_t v = (w >> (16 * i)) & 0xffff;
    return ((v & 0x7f) << 7) | ((v & 0x80) << 8) | ((v & 0x7f00) << 15) | ((v & 0x8000) << 16);
}
__device__ __forceinline__ uint16_t f2h(float v) { return __half_as_ushort(__float2half(v)); }

// the rows' inputs as fp16 once (rms-normed first where the segment says): out [segment][n][K]; grid (n, segments)
__global__ void __launch_bounds__(256) k_mprep(MvSeg s, int K, uint16_t* out) {
    const int t = blockIdx.x, sg = blockIdx.y, n = gridDim.x;
    const float* x = s.x[sg] + size_t(t) * s.xs;
    const float* nw = s.nw[sg];
    float r = 1.f;
    if (nw) {
        float ss = 0;
        for (int i = threadIdx.x; i < K; i += 256) { const float v = x[i]; ss += v * v; }
        r = rsqrtf(block_sum(ss) / K + kEps);
    }
    uint16_t* o = out + (size_t(sg) * n + t) * K;
    for (int i = threadIdx.x; i < K; i += 256) o[i] = f2h(nw ? x[i] * r * nw[i] : x[i]);
}
// bytes from global to LDS, 16 a thread a step, the loads issued together
__device__ __forceinline__ void m_copy(const uint16_t* src, uint16_t* dst, int halves) {
    const uint4* s4 = (const uint4*) src;
    uint4* d4 = (uint4*) dst;
    const int n4 = halves / 8;
    int i = threadIdx.x;
    for (; i + 3 * 256 < n4; i += 4 * 256) {
        const uint4 a = s4[i], b = s4[i + 256], c = s4[i + 512], d = s4[i + 768];
        d4[i] = a; d4[i + 256] = b; d4[i + 512] = c; d4[i + 768] = d;
    }
    for (; i < n4; i += 256) d4[i] = s4[i];
}
// y[t][row] = W8[row] . x[t] for n rows; RW rows a warp; LDS n * K * 2 bytes; grid (rows / (8 RW))
template <int RW, int N>
__global__ void __launch_bounds__(256) k_mv8m(W8 W, int K, MvSeg s, const uint16_t* xp16, float* y, int ys) {
    extern __shared__ uint16_t xh[];
    const int r0 = blockIdx.x * 8 * RW;
    int sg = -1;
    for (int j = 0; j < s.n; ++j) if (r0 >= s.lo[j] && r0 < s.hi[j]) sg = j;
    if (sg < 0) return;
    m_copy(xp16 + size_t(sg) * N * K, xh, N * K);
    __syncthreads();
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    float a[RW][N];
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int t = 0; t < N; ++t) a[r][t] = 0.f;
    for (int c = l; c < K / 16; c += 32) {
        uint4 q[RW];
        float sc[RW];
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const size_t row = size_t(r0 + w * RW + r);
            q[r] = *(const uint4*) (W.q + row * K + c * 16);
            sc[r] = __half2float(W.d[row * (K / 32) + (c >> 1)]);
        }
#pragma unroll
        for (int t = 0; t < N; ++t) {
            const uint4* xp = (const uint4*) (xh + size_t(t) * K + c * 16);
            const uint4 x0 = xp[0], x1 = xp[1];
#pragma unroll
            for (int r = 0; r < RW; ++r) {
                float d = 0;
                d = mdot2(i8h2(q[r].x, 0), x0.x, d); d = mdot2(i8h2(q[r].x, 1), x0.y, d);
                d = mdot2(i8h2(q[r].y, 0), x0.z, d); d = mdot2(i8h2(q[r].y, 1), x0.w, d);
                d = mdot2(i8h2(q[r].z, 0), x1.x, d); d = mdot2(i8h2(q[r].z, 1), x1.y, d);
                d = mdot2(i8h2(q[r].w, 0), x1.z, d); d = mdot2(i8h2(q[r].w, 1), x1.w, d);
                a[r][t] += sc[r] * d;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int t = 0; t < N; ++t) {
            const float v = warp_sum(a[r][t]);
            if (l == 0) y[size_t(t) * ys + r0 + w * RW + r] = v;
        }
}

// the 8-bit head over N rows, one weight pass; raw (cap 0) or softcapped logits; grid kVocabP / 32 / kHeadG
constexpr int kHeadG = 4;   // 32-row groups a block
template <int N>
__global__ void __launch_bounds__(256) k_head8m2(W8 W, const uint16_t* x16, float* logits, int cap) {
    __shared__ uint16_t xh[N * kH];
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    m_copy(x16, xh, N * kH);
    __syncthreads();
    for (int g = 0; g < kHeadG; ++g) {
        const int r0 = (blockIdx.x * kHeadG + g) * 32;
        if (r0 >= kVocab) {
            for (int i = threadIdx.x; i < 32 * N; i += 256) logits[size_t(i / 32) * kVocabP + r0 + i % 32] = -INFINITY;
            continue;
        }
        float a[4][N];
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int t = 0; t < N; ++t) a[r][t] = 0.f;
        for (int c = l; c < kH / 16; c += 32) {
            uint4 q[4];
            float sc[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const size_t row = size_t(min(r0 + w * 4 + r, kVocab - 1));
                q[r] = *(const uint4*) (W.q + row * kH + c * 16);
                sc[r] = __half2float(W.d[row * (kH / 32) + (c >> 1)]);
            }
#pragma unroll
            for (int t = 0; t < N; ++t) {
                const uint4* xp = (const uint4*) (xh + t * kH + c * 16);
                const uint4 x0 = xp[0], x1 = xp[1];
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    float d = 0;
                    d = mdot2(i8h2(q[r].x, 0), x0.x, d); d = mdot2(i8h2(q[r].x, 1), x0.y, d);
                    d = mdot2(i8h2(q[r].y, 0), x0.z, d); d = mdot2(i8h2(q[r].y, 1), x0.w, d);
                    d = mdot2(i8h2(q[r].z, 0), x1.x, d); d = mdot2(i8h2(q[r].z, 1), x1.y, d);
                    d = mdot2(i8h2(q[r].w, 0), x1.z, d); d = mdot2(i8h2(q[r].w, 1), x1.w, d);
                    a[r][t] += sc[r] * d;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int t = 0; t < N; ++t) {
                const float v = warp_sum(a[r][t]);
                const int row = r0 + w * 4 + r;
                if (l == 0) logits[size_t(t) * kVocabP + row] = row < kVocab ? (cap ? softcapf(v, kOutCap) : v) : -INFINITY;
            }
    }
}
// host side: the row count as a template
template <class F> void with_rows(int n, F&& f) {
    switch (n) {
        case 1: f(std::integral_constant<int, 1>{}); break;
        case 2: f(std::integral_constant<int, 2>{}); break;
        case 3: f(std::integral_constant<int, 3>{}); break;
        case 4: f(std::integral_constant<int, 4>{}); break;
        case 5: f(std::integral_constant<int, 5>{}); break;
        case 6: f(std::integral_constant<int, 6>{}); break;
        case 7: f(std::integral_constant<int, 7>{}); break;
        default: f(std::integral_constant<int, 8>{}); break;
    }
}

// ---- experts for n rows: the distinct experts (ML.cnt of them), each with its rows (t) and their top-k places (k)
struct MoeList { int cnt; int e[kMR * kTopK]; int nr[kMR * kTopK]; int t[kMR * kTopK][kMR]; int k[kMR * kTopK][kMR]; };
// one block of 64 threads: entry j = (t, k) of idx
__global__ void __launch_bounds__(64) k_moe_list(const int* idx, int n, MoeList* L) {
    __shared__ int lead[kMR * kTopK], uid[kMR * kTopK];
    const int j = threadIdx.x, N = n * kTopK;
    const int e = j < N ? idx[j] : -1;
    int first = j, rank = 0, tot = 0;
    for (int i = 0; i < N; ++i) {
        const bool same = idx[i] == e;
        if (same && i < first) first = i;
        if (same && i < j) ++rank;
        if (same) ++tot;
    }
    lead[j] = j < N && first == j;
    __syncthreads();
    if (j == 0) {
        int c = 0;
        for (int i = 0; i < kMR * kTopK; ++i) { uid[i] = c; c += lead[i]; }
        L->cnt = c;
    }
    __syncthreads();
    if (j < N) {
        const int u = uid[first];
        if (first == j) { L->e[u] = e; L->nr[u] = tot; }
        L->t[u][rank] = j / kTopK;
        L->k[u][rank] = j % kTopK;
    }
}
// one 16-byte piece of an expert row unpacked to fp16 pairs: NVFP4 32 values (16 pairs, scales s0 for the first 16
// and s1), MXFP8 16 values / 256 (8 pairs, s0 = 256 * 2^(e - 127))
// (the piece's 16 bytes w and its scale bytes sc already loaded: ex_raw)
template <bool NV>
__device__ __forceinline__ void ex_raw(const uint8_t* wrow, const uint8_t* srow, int u, uint4& w, uint32_t& sc) {
    w = *(const uint4*) (wrow + size_t(u) * 16);
    sc = NV ? uint32_t(*(const uint16_t*) (srow + u * 2)) : uint32_t(srow[u >> 1]);
}
template <bool NV>
__device__ __forceinline__ void ex_unpack(uint4 w, uint32_t scw, uint32_t* wh, float& s0, float& s1) {
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
    if (NV) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                const uint32_t t = (ww[j] >> (16 * hf)) & 0xffff;
                const uint32_t sp = (t & 0xf) | ((t & 0xf0) << 4) | ((t & 0xf00) << 8) | ((t & 0xf000) << 12);
                const uint32_t hb = __builtin_amdgcn_perm(0x46444240u, 0x3e3c3800u, sp & 0x07070707u) | ((sp & 0x08080808u) << 4);
                wh[(j * 2 + hf) * 2] = __builtin_amdgcn_perm(0u, hb, 0x010c000cu);
                wh[(j * 2 + hf) * 2 + 1] = __builtin_amdgcn_perm(0u, hb, 0x030c020cu);
            }
        s0 = e4m3f(scw & 0xff);
        s1 = e4m3f(scw >> 8);
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) { wh[2 * j] = e4h2(ww[j], 0); wh[2 * j + 1] = e4h2(ww[j], 1); }
        s0 = s1 = ldexpf(256.f, int(scw) - 127);
    }
}
// the dot of an unpacked piece with 32 (NV) / 16 fp16 values at xp
template <bool NV>
__device__ __forceinline__ float ex_mdot(const uint32_t* wh, float s0, float s1, const uint16_t* xp) {
    const uint4* q = (const uint4*) xp;
    const uint4 a = q[0], b = q[1];
    float d0 = 0;
    d0 = mdot2(wh[0], a.x, d0); d0 = mdot2(wh[1], a.y, d0); d0 = mdot2(wh[2], a.z, d0); d0 = mdot2(wh[3], a.w, d0);
    d0 = mdot2(wh[4], b.x, d0); d0 = mdot2(wh[5], b.y, d0); d0 = mdot2(wh[6], b.z, d0); d0 = mdot2(wh[7], b.w, d0);
    if (!NV) return d0 * s0;
    const uint4 c = q[2], e = q[3];
    float d1 = 0;
    d1 = mdot2(wh[8], c.x, d1); d1 = mdot2(wh[9], c.y, d1); d1 = mdot2(wh[10], c.z, d1); d1 = mdot2(wh[11], c.w, d1);
    d1 = mdot2(wh[12], e.x, d1); d1 = mdot2(wh[13], e.y, d1); d1 = mdot2(wh[14], e.z, d1); d1 = mdot2(wh[15], e.w, d1);
    return d0 * s0 + d1 * s1;
}
// gate / up of every listed expert for its rows (4 at a time); act [t][k][kF] fp16; grid (kF / 16, n * kTopK).
// (the row count a template inside: no branches in the piece loop, so the loads can run ahead)
constexpr int kMG = 4;
template <bool NV, int NG>
__device__ __forceinline__ void gum_rows(const Experts& ex, const uint8_t* W, const uint8_t* S, const uint16_t* xh, float (&a)[4][kMG]) {
    constexpr int PR = NV ? kH / 32 : kH / 16, VP = NV ? 32 : 16;
    const int l = threadIdx.x & 31;
#pragma unroll
    for (int it = 0; it < (4 * PR) / 32; ++it) {
        const int uu = l + 32 * it, r = uu / PR, pc = uu % PR;
        uint4 raw;
        uint32_t scw;
        ex_raw<NV>(W + size_t(r) * ex.w13_row, S + size_t(r) * ex.s13_row, pc, raw, scw);
        uint32_t wh[16];
        float s0, s1;
        ex_unpack<NV>(raw, scw, wh, s0, s1);
#pragma unroll
        for (int t = 0; t < NG; ++t) {
            const float v = ex_mdot<NV>(wh, s0, s1, xh + t * kH + pc * VP);
#pragma unroll
            for (int q = 0; q < 4; ++q) a[q][t] += r == q ? v : 0.f;
        }
    }
}
template <bool NV>
__global__ void __launch_bounds__(256) k_moe_gum(Experts ex, const uint16_t* x16, const MoeList* L, uint16_t* act) {
    __shared__ uint16_t xh[kMG * kH];
    const int u = blockIdx.y;
    if (u >= L->cnt) return;
    const int e = L->e[u], nr = L->nr[u];
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const int j0 = blockIdx.x * 16 + w * 2;
    const size_t row0 = size_t(e) * 2 * kF + 2 * j0;
    const uint8_t* W = ex.w13 + row0 * ex.w13_row;
    const uint8_t* S = ex.s13 + row0 * ex.s13_row;
    const float f = NV ? ex.f13[e] : 1.f;
    for (int g0 = 0; g0 < nr; g0 += kMG) {
        const int ng = min(kMG, nr - g0);
        if (g0) __syncthreads();
        for (int r = 0; r < ng; ++r) m_copy(x16 + size_t(L->t[u][g0 + r]) * kH, xh + r * kH, kH);
        __syncthreads();
        float a[4][kMG];
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int t = 0; t < kMG; ++t) a[q][t] = 0.f;
        if (ng == 1) gum_rows<NV, 1>(ex, W, S, xh, a);
        else if (ng == 2) gum_rows<NV, 2>(ex, W, S, xh, a);
        else if (ng == 3) gum_rows<NV, 3>(ex, W, S, xh, a);
        else gum_rows<NV, 4>(ex, W, S, xh, a);
#pragma unroll
        for (int t = 0; t < kMG; ++t) {
            if (t >= ng) break;
            float s4[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) s4[q] = warp_sum(a[q][t]);
            if (l < 2) {
                const float gg = s4[2 * l] * f, uv = s4[2 * l + 1] * f;
                act[(size_t(L->t[u][g0 + t]) * kTopK + L->k[u][g0 + t]) * kF + j0 + l] =
                    f2h(softcapf(gg, kGCap) * sigm(gg) * softcapf(uv, kUCap));
            }
        }
    }
}
// down of every listed expert for its rows (4 at a time): yp [t][k][kH] (times the NVFP4 factor); a warp 4 outputs;
// grid (kH / 32, n * kTopK)
template <bool NV, int NG>
__device__ __forceinline__ void downm_rows(const Experts& ex, int e, const uint16_t* ah, float (&a)[4][kMG]) {
    constexpr int PR = NV ? kF / 32 : kF / 16, VP = NV ? 32 : 16;
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
#pragma unroll
    for (int it = 0; it < (4 * PR + 31) / 32; ++it) {
        const int uu = l + 32 * it, q = uu / PR, pc = uu % PR;
        const bool ok = q < 4;
        const size_t row = size_t(e) * kH + blockIdx.x * 32 + w * 4 + (ok ? q : 0);
        uint4 raw;
        uint32_t scw;
        ex_raw<NV>(ex.w2 + row * ex.w2_row, ex.s2 + row * ex.s2_row, ok ? pc : 0, raw, scw);
        uint32_t wh[16];
        float s0, s1;
        ex_unpack<NV>(raw, scw, wh, s0, s1);
#pragma unroll
        for (int t = 0; t < NG; ++t) {
            const float v = ex_mdot<NV>(wh, s0, s1, ah + t * kF + (ok ? pc : 0) * VP);
#pragma unroll
            for (int qq = 0; qq < 4; ++qq) a[qq][t] += q == qq ? v : 0.f;
        }
    }
}
template <bool NV>
__global__ void __launch_bounds__(256) k_moe_downm(Experts ex, const uint16_t* act, const MoeList* L, float* yp) {
    __shared__ uint16_t ah[kMG * kF];
    const int u = blockIdx.y;
    if (u >= L->cnt) return;
    const int e = L->e[u], nr = L->nr[u];
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const float f = NV ? ex.f2[e] : 1.f;
    for (int g0 = 0; g0 < nr; g0 += kMG) {
        const int ng = min(kMG, nr - g0);
        if (g0) __syncthreads();
        for (int r = 0; r < ng; ++r) m_copy(act + (size_t(L->t[u][g0 + r]) * kTopK + L->k[u][g0 + r]) * kF, ah + r * kF, kF);
        __syncthreads();
        float a[4][kMG];
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int t = 0; t < kMG; ++t) a[q][t] = 0.f;
        if (ng == 1) downm_rows<NV, 1>(ex, e, ah, a);
        else if (ng == 2) downm_rows<NV, 2>(ex, e, ah, a);
        else if (ng == 3) downm_rows<NV, 3>(ex, e, ah, a);
        else downm_rows<NV, 4>(ex, e, ah, a);
#pragma unroll
        for (int t = 0; t < kMG; ++t) {
            if (t >= ng) break;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float v = warp_sum(a[q][t]);
                if (l == 0) yp[(size_t(L->t[u][g0 + t]) * kTopK + L->k[u][g0 + t]) * kH + blockIdx.x * 32 + w * 4 + q] = v * f;
            }
        }
    }
}
// y[t] = sum_k wts[t][k] yp[t][k]; grid n
__global__ void __launch_bounds__(256) k_moe_combm(const float* yp, const float* wts, float* y) {
    const int t = blockIdx.x;
    for (int c = threadIdx.x; c < kH; c += 256) {
        float a = 0;
#pragma unroll
        for (int k = 0; k < kTopK; ++k) a += wts[t * kTopK + k] * yp[(size_t(t) * kTopK + k) * kH + c];
        y[size_t(t) * kH + c] = a;
    }
}
