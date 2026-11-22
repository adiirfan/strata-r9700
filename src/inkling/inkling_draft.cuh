// Inkling 2 Mini's DSpark draft (mtp.safetensors), from vLLM's vllm/models/inkling2/nvidia/dspark.py: an anchor plus 7
// mask rows through 3 small dense layers that attend to context K / V made from the target's features (the running
// partials entering visits 5, 23 and 41 and the final normed output), then a recurrent rank-256 head that corrects the
// 7 rows' logits left to right. Greedy drafts; the target checks all 8 rows in one pass (the prompt path).
#pragma once

constexpr int kDL = 3, kDB = 8, kDF = 9216, kDR = 256, kDTaps = 4, kDRing = 2048;   // layers, block, mlp, markov rank
constexpr float kDTheta = 1e6f;

// rms(sum_j alpha[j] F[j][row]) * nw as fp16 (and fp32); F: kDTaps planes of [rows][kH]
__global__ void __launch_bounds__(256) k_dmix(const float* F, size_t plane, const float* alpha, const float* nw, uint16_t* o16,
                                              int o16s, float* o32, const int* roff = nullptr, const int* cnt = nullptr) {
    if (cnt && int(blockIdx.x) >= *cnt) return;
    if (roff) F += size_t(*roff) * kH;
    const size_t r = blockIdx.x;
    float v[9], ss = 0;
#pragma unroll
    for (int j = 0; j < 9; ++j) {
        const int c = threadIdx.x + j * 256;
        float a = 0;
        if (c < kH)
#pragma unroll
            for (int t = 0; t < kDTaps; ++t) a += alpha[t] * F[t * plane + r * kH + c];
        v[j] = a;
        ss += a * a;
    }
    const float rr = rsqrtf(block_sum(ss) / kH + kEps);
#pragma unroll
    for (int j = 0; j < 9; ++j) {
        const int c = threadIdx.x + j * 256;
        if (c >= kH) break;
        const float y = v[j] * rr * nw[c];
        if (o16) o16[r * o16s + c] = __half_as_ushort(__float2half(y));
        if (o32) o32[r * kH + c] = y;
    }
}
// neox RoPE of one 128-wide head held by 128 threads (x[d] in, out), position pos
__device__ __forceinline__ float d_rope(const float* xs, int d, int pos) {
    const int i = d & 63;
    const float inv = powf(kDTheta, -float(2 * i) / float(kHD));
    float sn, cs;
    sincosf(float(pos) * inv, &sn, &cs);
    return d < 64 ? xs[d] * cs - xs[d + 64] * sn : xs[d] * cs + xs[d - 64] * sn;
}
// per-head rms * w (128 threads), the head's values in, normed in xs
__device__ __forceinline__ void d_headnorm(float* xs, int d, const float* w) {
    const float ss = block_sum(xs[d] * xs[d]);
    const float y = xs[d] * rsqrtf(ss / kHD + kEps) * w[d];
    __syncthreads();
    xs[d] = y;
    __syncthreads();
}
// context K / V of rows (positions p0 + r) from the kv product [rows][2048] (k then v) into the caches; grid (rows, kKV)
__global__ void __launch_bounds__(128) k_dkvctx(const float* KV, const float* knw, int p0, int mask, __half* Kc, __half* Vc,
                                                const int* pc = nullptr) {   // pc: {first position, rows} on the card
    __shared__ float xs[kHD];
    if (pc) { if (int(blockIdx.x) >= pc[1]) return; p0 = pc[0]; }
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x, pos = p0 + r;
    xs[d] = KV[size_t(r) * 2 * kKVD + h * kHD + d];
    __syncthreads();
    d_headnorm(xs, d, knw);
    const size_t slot = size_t(pos & mask) * kKVD + h * kHD + d;
    Kc[slot] = __float2half(d_rope(xs, d, pos));
    Vc[slot] = __float2half(KV[size_t(r) * 2 * kKVD + kKVD + h * kHD + d]);
}
// the block's q (normed, roped, over sqrt(128): Q [8][3072]) and k / v (into the caches at q0 + r); QKV [8][5120]
// grid (8, 24 + 8 + 8), 128 threads
__global__ void __launch_bounds__(128) k_dqkv(const float* QKV, const float* qnw, const float* knw, const int* qs, int mask, float* Q,
                                              __half* Kc, __half* Vc) {
    __shared__ float xs[kHD];
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x, pos = qs[0] + r;
    xs[d] = QKV[size_t(r) * (kQD + 2 * kKVD) + h * kHD + d];   // q heads, then k heads, then v heads: contiguous
    __syncthreads();
    if (h < kQH) {
        d_headnorm(xs, d, qnw);
        Q[size_t(r) * kQD + h * kHD + d] = d_rope(xs, d, pos) * 0.08838834764831845f;
    } else if (h < kQH + kKV) {
        d_headnorm(xs, d, knw);
        Kc[size_t(pos & mask) * kKVD + (h - kQH) * kHD + d] = __float2half(d_rope(xs, d, pos));
    } else {
        Vc[size_t(pos & mask) * kKVD + (h - kQH - kKV) * kHD + d] = __float2half(xs[d]);
    }
}
// the block's attention: 24 queries (8 rows x 3 heads) of K / V head kh over keys lo .. q0 + 7 (bidirectional inside
// the block), a piece of them a block; partials (m, l, o) per query. grid (kKV, nch), 256 threads.
constexpr int kDCh = 32;
__global__ void __launch_bounds__(256) k_dattn(const float* Q, const __half* Kc, const __half* Vc, const int* qpos, int local, int mask, float* part) {
    const int q0 = qpos[0], lo = local ? max(0, q0 - kWin) : 0;
    __shared__ float qs[24][kHD];
    __shared__ float S[24][64];
    __shared__ float mq[24], lq[24], cq[24];
    const int kh = blockIdx.x, c = blockIdx.y, nch = gridDim.y, tid = threadIdx.x, w = tid >> 5, l = tid & 31;
    const int n = q0 + kDB - lo, per = max(64, (n + nch - 1) / nch), a = lo + c * per, bnd = min(q0 + kDB, a + per);
    float* pp = part + (size_t(kh) * kDCh + c) * 24 * (kHD + 4);
    if (a >= bnd) { if (tid < 24) pp[tid * (kHD + 4)] = -INFINITY, pp[tid * (kHD + 4) + 1] = 0.f; return; }
    for (int i = tid; i < 24 * kHD; i += 256) {
        const int qi = i / kHD, d = i % kHD, r = qi / 3, hh = qi % 3;
        qs[qi][d] = Q[size_t(r) * kQD + (kh * 3 + hh) * kHD + d];
    }
    if (tid < 24) { mq[tid] = -INFINITY; lq[tid] = 0.f; }
    __syncthreads();
    const int d = tid & 127, half = tid >> 7;
    float o[12] = {};
    for (int p0 = a; p0 < bnd; p0 += 64) {
        const int cnt = min(64, bnd - p0);
        {   // scores: thread (position j, 6 queries)
            const int j = tid & 63, qg = tid >> 6;
            float acc[6] = {};
            if (j < cnt) {
                const uint4* kr = (const uint4*) (Kc + size_t((p0 + j) & mask) * kKVD + kh * kHD);
                for (int i = 0; i < kHD / 8; ++i) {
                    const uint4 u = kr[i];
                    const __half2* h2 = (const __half2*) &u;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float2 f = __half22float2(h2[e]);
#pragma unroll
                        for (int q = 0; q < 6; ++q) acc[q] += f.x * qs[qg * 6 + q][i * 8 + 2 * e] + f.y * qs[qg * 6 + q][i * 8 + 2 * e + 1];
                    }
                }
            }
#pragma unroll
            for (int q = 0; q < 6; ++q) S[qg * 6 + q][j] = j < cnt ? acc[q] : -INFINITY;
        }
        __syncthreads();
        // softmax update: warp w the queries 3w .. 3w + 2, lanes 2 positions each
#pragma unroll
        for (int q3 = 0; q3 < 3; ++q3) {
            const int q = 3 * w + q3;
            const float s0 = S[q][l], s1 = S[q][l + 32];
            const float mx = warp_max(fmaxf(s0, s1)), mn = fmaxf(mq[q], mx);
            const float e0 = expf(s0 - mn), e1 = expf(s1 - mn);
            S[q][l] = e0; S[q][l + 32] = e1;
            const float su = warp_sum(e0 + e1);
            if (l == 0) { const float cf = expf(mq[q] - mn); cq[q] = cf; lq[q] = lq[q] * cf + su; mq[q] = mn; }
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 12; ++q) o[q] *= cq[half * 12 + q];
        for (int j = 0; j < cnt; ++j) {
            const float v = __half2float(Vc[size_t((p0 + j) & mask) * kKVD + kh * kHD + d]);
#pragma unroll
            for (int q = 0; q < 12; ++q) o[q] += S[half * 12 + q][j] * v;
        }
        __syncthreads();
    }
#pragma unroll
    for (int q = 0; q < 12; ++q) pp[(half * 12 + q) * (kHD + 4) + 4 + d] = o[q];
    if (tid < 24) { pp[tid * (kHD + 4)] = mq[tid]; pp[tid * (kHD + 4) + 1] = lq[tid]; }
}
// the pieces added: out16 [8][3072]; grid (kKV * 24), 128 threads
__global__ void __launch_bounds__(128) k_dattn_merge(const float* part, int nch, uint16_t* out16) {
    const int kh = blockIdx.x / 24, qi = blockIdx.x % 24, d = threadIdx.x, r = qi / 3, hh = qi % 3;
    float M = -INFINITY;
    for (int c = 0; c < nch; ++c) M = fmaxf(M, part[((size_t(kh) * kDCh + c) * 24 + qi) * (kHD + 4)]);
    float L = 0, O = 0;
    for (int c = 0; c < nch; ++c) {
        const float* pp = part + ((size_t(kh) * kDCh + c) * 24 + qi) * (kHD + 4);
        if (pp[0] == -INFINITY) continue;
        const float f = expf(pp[0] - M);
        L += pp[1] * f;
        O += pp[4 + d] * f;
    }
    out16[size_t(r) * kQD + (kh * 3 + hh) * kHD + d] = __half_as_ushort(__float2half(O / L));
}
// silu(gate) * up as fp16; GU [8][2 kDF]
__global__ void k_dsilu(const float* GU, uint16_t* A16) {
    const int r = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    const float g = GU[size_t(r) * 2 * kDF + i], u = GU[size_t(r) * 2 * kDF + kDF + i];
    A16[size_t(r) * kDF + i] = __half_as_ushort(__float2half(g / (1.f + expf(-g)) * u));
}
__global__ void k_dadd(float* res, const float* x, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) res[i] += x[i];
}
// the block's input: row 0 the anchor's normed embedding (EMB0), rows 1 .. 7 the mask embedding
__global__ void k_dblock(const float* emb0, const float* mask_embed, float* X) {
    const int r = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    if (c < kH) X[size_t(r) * kH + c] = r == 0 ? emb0[c] : mask_embed[c];
}
// [hid (2304) | f_rnn (2304)] rows as fp16 for the markov context product; HID [8][kH] (rows 1 .. 7 used), FR [kH]
__global__ void k_dmkin(const float* HID, const float* FR, uint16_t* X16) {
    const int r = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    if (c >= 2 * kH) return;
    const float v = c < kH ? HID[size_t(r + 1) * kH + c] : FR[c - kH];
    X16[size_t(r) * 2 * kH + c] = __half_as_ushort(__float2half(v));
}
// one markov step s: raw = ctx[s] + bias + Ws [state, w1[prev]]; state, latent (tanh) out. 768 threads.
__global__ void __launch_bounds__(768) k_dmstep(const float* CTX, const float* rb, const float* Ws, const __half* w1, float* state,
                                                float* latent, const int* toks, int s) {
    __shared__ float in[2 * kDR];
    __shared__ float raw[3 * kDR];
    const int i = threadIdx.x, prev = toks[s];   // toks[0] the anchor, toks[s] the token before step s's
    if (i < kDR) in[i] = s == 0 ? 0.f : state[i];
    else if (i < 2 * kDR) in[i] = __half2float(w1[size_t(prev) * kDR + i - kDR]);
    __syncthreads();
    float a = CTX[size_t(s) * 3 * kDR + i] + rb[i];
    const float* wr = Ws + size_t(i) * 2 * kDR;
    for (int k = 0; k < 2 * kDR; ++k) a += wr[k] * in[k];
    raw[i] = a;
    __syncthreads();
    if (i < kDR) {
        const float g = 1.f / (1.f + expf(-raw[i])), cnd = tanhf(raw[kDR + i]);
        state[i] = g * in[i] + (1.f - g) * cnd;
        latent[i] = tanhf(raw[2 * kDR + i]);
    }
}
// logits = softcap(base + FOLD . latent + fbias, 30); FOLD int8 rows of 256 with fp16 scales a 32; grid kVocabP / 32
__global__ void __launch_bounds__(256) k_dfold(W8 F, const float* latent, const float* fbias, const float* base, float* logits) {
    __shared__ float ls[kDR];
    const int r0 = blockIdx.x * 32, w = threadIdx.x >> 5, l = threadIdx.x & 31;
    ls[threadIdx.x] = latent[threadIdx.x];
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        const int row = r0 + w * 4 + rr;
        if (row >= kVocab) { if (l == 0) logits[row] = -INFINITY; continue; }
        const uint2 q = *(const uint2*) (F.q + size_t(row) * kDR + l * 8);
        const uint32_t qq[2] = {q.x, q.y};
        float t = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) t += float(int8_t(qq[j >> 2] >> (8 * (j & 3)))) * ls[l * 8 + j];
        t *= __half2float(F.d[size_t(row) * (kDR / 32) + (l >> 2)]);
        t = warp_sum(t);
        if (l == 0) logits[row] = softcapf(base[row] + t + fbias[row], kOutCap);
    }
}
// the 8-bit head over up to 8 rows at once (rows as fp16 in LDS), raw (cap 0) or softcapped logits; grid kVocabP / 32
__global__ void __launch_bounds__(256) k_head8m(W8 W, const float* x, int n, float* logits, int cap) {
    __shared__ __half xsh[kDB][kH];
    const int r0 = blockIdx.x * 32, w = threadIdx.x >> 5, l = threadIdx.x & 31;
    for (int i = threadIdx.x; i < n * kH; i += 256) xsh[i / kH][i % kH] = __float2half(x[i]);
    __syncthreads();
    for (int rr = 0; rr < 4; ++rr) {
        const int row = r0 + w * 4 + rr;
        if (row >= kVocab) { if (l < n) logits[size_t(l) * kVocabP + row] = -INFINITY; continue; }
        float a[kDB] = {};
        for (int c = l; c < kH / 16; c += 32) {
            const uint4 q = *(const uint4*) (W.q + size_t(row) * kH + c * 16);
            const uint32_t qq[4] = {q.x, q.y, q.z, q.w};
            const float sc = __half2float(W.d[size_t(row) * (kH / 32) + (c >> 1)]);
            float wv[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) wv[j] = float(int8_t(qq[j >> 2] >> (8 * (j & 3)))) * sc;
            for (int t = 0; t < n; ++t) {
                float s = 0;
#pragma unroll
                for (int j = 0; j < 16; ++j) s += wv[j] * __half2float(xsh[t][c * 16 + j]);
                a[t] += s;
            }
        }
        for (int t = 0; t < n; ++t) {
            const float v = warp_sum(a[t]);
            if (l == 0) logits[size_t(t) * kVocabP + row] = cap ? softcapf(v, kOutCap) : v;
        }
    }
}
// argmax of each of n rows of logits into out[r]; grid n, 1024 threads
__global__ void __launch_bounds__(1024) k_argmax_rows(const float* lg, int* out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const float* x = lg + size_t(blockIdx.x) * kVocabP;
    float v = -INFINITY;
    int i = 0;
    for (int j = threadIdx.x; j < kVocab; j += 1024) if (x[j] > v) { v = x[j]; i = j; }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor(v, o, 32);
        const int oi = __shfl_xor(i, o, 32);
        if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = i; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 32; ++w) if (bv[w] > v || (bv[w] == v && bi[w] < i)) { v = bv[w]; i = bi[w]; }
        out[blockIdx.x] = i;
    }
}
// the conv states after each row j of a T-row pass (T <= 8): out [T][4][3][kH]
__global__ void k_conv_state_all(ConvUpd u, int T, float* out) {
    const int k = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x, j = blockIdx.z;
    if (c >= u.C[k] || j >= T) return;
    const int C = u.C[k];
    const float* st = u.st[k];   // (read before k_conv_state moves it on)
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int src = j - i;
        out[((size_t(j) * 4 + k) * 3 + i) * kH + c] = src >= 0 ? u.x[k][size_t(src) * u.xs[k] + c] : st[(i - j - 1) * C + c];
    }
}
// the conv states of every visit as after row j of the last checked block (SPST [visit][8][4][3][kH]); grid (kH / 256, 12, visits)
__global__ void k_conv_state_put_all(const float* in, float* const* csp, const int* acc) {
    const int k = blockIdx.y / 3, i = blockIdx.y % 3, v = blockIdx.z, c = blockIdx.x * 256 + threadIdx.x, j = acc[2];
    const int C = k < 2 ? kKVD : kH;
    if (c < C) csp[v * 4 + k][i * C + c] = in[(((size_t(v) * kDB + j) * 4 + k) * 3 + i) * kH + c];
}
// the check's verdict on the card: j drafts taken (leading matches of DTOK[1 + i] and the target's G[i], i < n - 1);
// out = {j + 1, tokens...}; acc = {first position, rows, last feature row} for the next cycle's context K / V; the next
// anchor (the target's own token after them) into DTOK[0]; the position moved on
__global__ void k_accept(int* toks, const int* G, int n, int* state, int* acc, int* out) {
    int j = 0;
    while (j < n - 1 && toks[1 + j] == G[j]) ++j;
    const int p = state[0];
    out[0] = j + 1;
    for (int i = 0; i < j; ++i) out[1 + i] = toks[1 + i];
    out[1 + j] = G[j];
    acc[0] = p; acc[1] = j + 1; acc[2] = j;
    toks[0] = G[j];
    state[0] = p + j + 1;
}
