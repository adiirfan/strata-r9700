// src/inkling/inkling_serve.cuh - strata-inkling --serve (from strata-inkling's): the Strata engine protocol (serve/server.py's StrataEngine) on
// stdin / stdout, with sampling and conversation reuse.
//
//   -> READY <max_context> stop
//   <- GEN <max_new> [key=value ...] <id,id,...>
//   -> RESUME <n>, PP <position> <prompt tokens> <ms> <tok/s> after each prompt chunk, T <id> per token,
//      DONE <generated> <prompt tokens> <prompt ms> <decode ms> <stop|length|cancel> 0 0 <reused> 0 0 0 0 0 <read> 0
//   <- STOP (ends the running request), QUIT
//
// Keys: temperature, top_p, top_k (0 = off), min_p, seed; absent temperature (or 0) is greedy. Other keys are read and
// ignored. Besides its K/V caches Inkling carries short-convolution states (the last 3 inputs of 4 convolutions in each
// of the 48 visits), which cannot be rewound: a request continues where the cache's tokens end when it extends them,
// else from the snapshot taken before the previous prompt's last token (a regenerated answer, a chat turn whose
// earlier answer is written back differently), else from the start.
#pragma once
#include <deque>
#include <poll.h>
#include <unistd.h>

struct SampleParams {
    float inv_temp = 1.f, top_p = 1.f, min_p = 0.f;
    int top_k = 0;
    unsigned long long seed = 0;
};
__device__ __forceinline__ uint64_t sp_mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
__device__ __forceinline__ float sp_uniform(uint64_t seed, uint64_t step, int what) {
    const uint64_t h = sp_mix(seed ^ sp_mix(step * 0x100000001b3ull + what));
    return float(h >> 40) * (1.f / 16777216.f);
}
constexpr int kVocabF = kVocabP, kSpChunk = (kVocabF + 1023) / 1024;
__device__ __forceinline__ float sp_bsum(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = sh[threadIdx.x];
        for (int o = 16; o > 0; o >>= 1) t += __shfl_xor(t, o, 32);
        if (threadIdx.x == 0) sh[32] = t;
    }
    __syncthreads();
    return sh[32];
}
__device__ __forceinline__ float sp_bmax(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = sh[threadIdx.x];
        for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor(t, o, 32));
        if (threadIdx.x == 0) sh[32] = t;
    }
    __syncthreads();
    return sh[32];
}
// one token from the logits: temperature, then the tokens above one cut (the largest of: top-k's k-th value, the
// smallest set reaching top_p of the mass, min_p of the best), drawn by inverse CDF; into state[1]
__global__ void __launch_bounds__(1024) k_sample(const float* l, SampleParams sp, unsigned long long step, int* state) {
    __shared__ float sh[33];
    __shared__ float part[1024];
    __shared__ int pick;
    __shared__ float base;
    const int tid = threadIdx.x, j0 = tid * kSpChunk, j1 = min(kVocabF, j0 + kSpChunk);
    float mx = -INFINITY;
    for (int j = j0; j < j1; ++j) mx = fmaxf(mx, l[j] * sp.inv_temp);
    const float M = sp_bmax(mx, sh);
    auto mass = [&](float s, float& cnt) {
        float m = 0.f, c = 0.f;
        for (int j = j0; j < j1; ++j) { const float x = l[j] * sp.inv_temp - M; if (x >= s) { m += __expf(x); c += 1.f; } }
        const float tm = sp_bsum(m, sh);
        cnt = sp_bsum(c, sh);
        return tm;
    };
    float cnt = 0.f;
    const float Z = mass(-INFINITY, cnt);
    float s = sp.min_p > 0.f ? __logf(sp.min_p) : -INFINITY;
    if (sp.top_k > 0 && sp.top_k < kVocabF) {
        float lo = -90.f, hi = 0.f;
        for (int it = 0; it < 26; ++it) {
            const float mid = 0.5f * (lo + hi);
            mass(mid, cnt);
            if (cnt >= float(sp.top_k)) lo = mid; else hi = mid;
        }
        s = fmaxf(s, lo);
    }
    if (sp.top_p < 1.f) {
        float lo = -90.f, hi = 0.f;
        for (int it = 0; it < 26; ++it) {
            const float mid = 0.5f * (lo + hi);
            if (mass(mid, cnt) >= sp.top_p * Z) lo = mid; else hi = mid;
        }
        s = fmaxf(s, lo);
    }
    const float K = mass(s, cnt);
    const float target = sp_uniform(sp.seed, step, 1) * K;
    float m = 0.f;
    for (int j = j0; j < j1; ++j) { const float x = l[j] * sp.inv_temp - M; if (x >= s) m += __expf(x); }
    part[tid] = m;
    __syncthreads();
    if (tid == 0) {
        float acc = 0.f, before = 0.f;
        int t = -1, lastnz = 0;
        for (int u = 0; u < 1024; ++u) {
            if (part[u] <= 0.f) continue;
            lastnz = u;
            if (t < 0 && acc + part[u] > target) { t = u; before = acc; }
            acc += part[u];
        }
        if (t < 0) { t = lastnz; before = acc - part[lastnz]; }
        base = before;
        pick = t;
    }
    __syncthreads();
    if (tid == pick) {
        float acc = base;
        int best = -1;
        for (int j = j0; j < j1; ++j) {
            const float x = l[j] * sp.inv_temp - M;
            if (x < s) continue;
            best = j;
            acc += __expf(x);
            if (acc > target) break;
        }
        state[1] = best;
    }
}

// The common case, top_k in 1..64: the k best logits are found once (each thread keeps its own best k, then k rounds
// of a block maximum over the threads' heads), and temperature, top_p and min_p act on those k only. The same cut and
// draw as k_sample, without its bisection passes over the whole vocabulary.
constexpr int kTopKMax = 64;
// the k best of a row into cv / ci (sorted, best first), the whole block of 1024
__device__ void sp_topk(const float* l, int k, float* cv, int* ci) {
    __shared__ float wv[32];
    __shared__ int wi[32], wt[32];
    const int tid = threadIdx.x, j0 = tid * kSpChunk, j1 = min(kVocabF, j0 + kSpChunk);
    // this thread's best k, sorted (k <= 64, kSpChunk values each)
    float bv[kTopKMax];
    int bi[kTopKMax];
    int nb = 0;
    for (int j = j0; j < j1; ++j) {
        const float v = l[j];
        if (nb == k && v <= bv[k - 1]) continue;
        int p = nb < k ? nb++ : k - 1;
        while (p > 0 && bv[p - 1] < v) { bv[p] = bv[p - 1]; bi[p] = bi[p - 1]; --p; }
        bv[p] = v; bi[p] = j;
    }
    int head = 0;
    for (int r = 0; r < k; ++r) {
        float v = head < nb ? bv[head] : -INFINITY;
        int i = head < nb ? bi[head] : 0x7fffffff, t = tid;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(v, o, 32);
            const int oi = __shfl_xor(i, o, 32), ot = __shfl_xor(t, o, 32);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; t = ot; }
        }
        if ((tid & 31) == 0) { wv[tid >> 5] = v; wi[tid >> 5] = i; wt[tid >> 5] = t; }
        __syncthreads();
        if (tid == 0) {
            float b = wv[0]; int bi_ = wi[0], bt = wt[0];
            for (int w = 1; w < 32; ++w) if (wv[w] > b || (wv[w] == b && wi[w] < bi_)) { b = wv[w]; bi_ = wi[w]; bt = wt[w]; }
            cv[r] = b; ci[r] = bi_; wt[0] = bt;
        }
        __syncthreads();
        if (tid == wt[0]) ++head;
        __syncthreads();
    }
}
// the kept tokens of a sorted top-k row: probabilities at the temperature, and how many survive top_p / min_p
__device__ inline int sp_keep(const float* cv, int k, const SampleParams& sp, float* p, float& kept) {
    float z = 0.f;
    for (int r = 0; r < k; ++r) { p[r] = cv[r] == -INFINITY ? 0.f : __expf((cv[r] - cv[0]) * sp.inv_temp); z += p[r]; }
    int keep = k;
    float acc = 0.f;
    for (int r = 0; r < k; ++r) {
        acc += p[r];
        if (sp.top_p < 1.f && acc >= sp.top_p * z) { keep = r + 1; break; }
    }
    if (sp.min_p > 0.f) for (int r = 1; r < keep; ++r) if (p[r] < sp.min_p * p[0]) { keep = r; break; }
    kept = 0.f;
    for (int r = 0; r < keep; ++r) kept += p[r];
    return keep;
}
__global__ void __launch_bounds__(1024) k_sample_topk(const float* l, SampleParams sp, unsigned long long step, int* state) {
    __shared__ float cv[kTopKMax];
    __shared__ int ci[kTopKMax];
    const int k = sp.top_k;
    sp_topk(l, k, cv, ci);
    if (threadIdx.x != 0) return;
    // cv is sorted, best first: softmax at the temperature, then the top_p and min_p cuts
    float p[kTopKMax], kept;
    const int keep = sp_keep(cv, k, sp, p, kept);
    const float target = sp_uniform(sp.seed, step, 1) * kept;
    float run = 0.f;
    int pick = ci[keep - 1];
    for (int r = 0; r < keep; ++r) { run += p[r]; if (run > target) { pick = ci[r]; break; } }
    state[1] = pick;
}

// The fast path for top_k 1..64 (k_sample_topk, one block over the whole vocabulary, was slow).
// Stage 1: 256 blocks each sort their 1024 logits (bitonic, in LDS) and keep their best 64. Stage 2: one block finds by
// bisection the value the k-th best of those 16,384 is at, gathers the ones at or above it, sorts them, and draws as
// k_sample_topk does (the same cut, the same draw).
constexpr int kCandBlocks = kVocabF / 1024;
__device__ inline bool sp_before(float av, int ai, float bv, int bi) { return av > bv || (av == bv && ai < bi); }
template <int N>
__device__ inline void sp_bitonic(float* v, int* ix) {   // sort N pairs, best first; blockDim.x threads
    for (int k = 2; k <= N; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                const int l = i ^ j;
                if (l <= i) continue;
                const bool up = (i & k) == 0;
                if (sp_before(v[l], ix[l], v[i], ix[i]) == up) {
                    const float tv = v[i]; v[i] = v[l]; v[l] = tv;
                    const int ti = ix[i]; ix[i] = ix[l]; ix[l] = ti;
                }
            }
            __syncthreads();
        }
}
__global__ void __launch_bounds__(256) k_samp_cand(const float* l, float* cv, int* ci) {
    __shared__ float v[1024];
    __shared__ int ix[1024];
    const int base = blockIdx.x * 1024;
    for (int i = threadIdx.x; i < 1024; i += 256) { v[i] = l[base + i]; ix[i] = base + i; }
    __syncthreads();
    sp_bitonic<1024>(v, ix);
    if (threadIdx.x < kTopKMax) { cv[blockIdx.x * kTopKMax + threadIdx.x] = v[threadIdx.x]; ci[blockIdx.x * kTopKMax + threadIdx.x] = ix[threadIdx.x]; }
}
__global__ void __launch_bounds__(1024) k_samp_pick(const float* cv, const int* ci, SampleParams sp, unsigned long long step, int* state) {
    constexpr int NC = kCandBlocks * kTopKMax, PER = NC / 1024, CAP = 256;
    __shared__ float sh[33];
    __shared__ float gv[CAP];
    __shared__ int gi[CAP];
    __shared__ int ng;
    const int k = sp.top_k, tid = threadIdx.x;
    float mine[PER];
    int mi[PER];
    float mx = -INFINITY, mn = INFINITY;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        mine[j] = cv[tid * PER + j]; mi[j] = ci[tid * PER + j];
        mx = fmaxf(mx, mine[j]);
        if (mine[j] > -INFINITY) mn = fminf(mn, mine[j]);
    }
    float hi = sp_bmax(mx, sh), lo = -sp_bmax(-mn, sh);
    // the largest t with at least k candidates >= t
    for (int it = 0; it < 30 && hi > lo; ++it) {
        const float mid = 0.5f * (lo + hi);
        float c = 0.f;
#pragma unroll
        for (int j = 0; j < PER; ++j) c += mine[j] >= mid;
        if (sp_bsum(c, sh) >= float(k)) lo = mid; else hi = mid;
    }
    if (tid == 0) ng = 0;
    for (int i = tid; i < CAP; i += 1024) { gv[i] = -INFINITY; gi[i] = 0x7fffffff; }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < PER; ++j)
        if (mine[j] >= lo) { const int p = atomicAdd(&ng, 1); if (p < CAP) { gv[p] = mine[j]; gi[p] = mi[j]; } }
    __syncthreads();
    sp_bitonic<CAP>(gv, gi);
    if (tid != 0) return;
    float p[kTopKMax], kept;
    const int keep = sp_keep(gv, k, sp, p, kept);
    const float target = sp_uniform(sp.seed, step, 1) * kept;
    float run = 0.f;
    int pick = gi[keep - 1];
    for (int r = 0; r < keep; ++r) { run += p[r]; if (run > target) { pick = gi[r]; break; } }
    state[1] = pick;
}
// the sampler's two stages (the candidate buffers live with the caller)
struct FastSampler {
    float* cv = nullptr;
    int* ci = nullptr;
    void init() { CK(hipMalloc(&cv, kCandBlocks * kTopKMax * 4)); CK(hipMalloc(&ci, kCandBlocks * kTopKMax * 4)); }
    void run(const float* logits, const SampleParams& sp, unsigned long long step, int* state, hipStream_t st) {
        k_samp_cand<<<kCandBlocks, 256, 0, st>>>(logits, cv, ci);
        k_samp_pick<<<1, 1024, 0, st>>>(cv, ci, sp, step, state);
    }
};

struct LineReader {
    std::string buf;
    bool eof = false;
    bool fill(int timeout_ms) {
        pollfd p{0, POLLIN, 0};
        if (poll(&p, 1, timeout_ms) <= 0) return false;
        char tmp[65536];
        const ssize_t n = read(0, tmp, sizeof tmp);
        if (n <= 0) { eof = true; return false; }
        buf.append(tmp, size_t(n));
        return true;
    }
    bool take(std::string& line) {
        const size_t e = buf.find('\n');
        if (e == std::string::npos) return false;
        line = buf.substr(0, e);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        buf.erase(0, e + 1);
        return true;
    }
    bool get(std::string& line) {
        while (!take(line)) { if (eof) return false; fill(-1); if (eof && buf.find('\n') == std::string::npos) return false; }
        return true;
    }
    bool poll_line(std::string& line) { if (take(line)) return true; fill(0); return take(line); }
};

static int serve(Model& m, const std::vector<int>& eos_ids, int chunk) {
    LineReader in;
    std::vector<int> live;                                         // the tokens whose K/V the cache holds, in order
    int snap_pos = -1;                                              // the snapshot's position (its tokens: live's first)
    unsigned long long step_no = 0;
    FastSampler fast;
    fast.init();
    std::printf("INFO engine=strata-inkling\n");
    std::printf("READY %d stop\n", m.max_ctx);
    std::fflush(stdout);
    auto is_eos = [&](int t) { return std::find(eos_ids.begin(), eos_ids.end(), t) != eos_ids.end(); };
    std::string line;
    while (in.get(line)) {
        if (line == "QUIT") break;
        if (line == "STOP" || line.empty()) continue;
        if (line.rfind("GEN ", 0) != 0) { std::printf("ERR expected: GEN <max_new> <id,id,...>\n"); std::fflush(stdout); continue; }
        std::vector<std::string> f;
        { std::stringstream ss(line); for (std::string w; ss >> w;) f.push_back(w); }
        if (f.size() < 3) { std::printf("ERR a request needs max_new and ids\n"); std::fflush(stdout); continue; }
        const int max_new = std::max(1, std::atoi(f[1].c_str()));
        float temperature = 0.f;
        SampleParams sp;
        bool seeded = false;
        size_t w = 2;
        for (; w + 1 < f.size(); ++w) {
            const size_t eq = f[w].find('=');
            if (eq == std::string::npos) break;
            const std::string k = f[w].substr(0, eq), v = f[w].substr(eq + 1);
            if (k == "temperature") temperature = std::strtof(v.c_str(), nullptr);
            else if (k == "top_p") sp.top_p = std::strtof(v.c_str(), nullptr);
            else if (k == "top_k") sp.top_k = std::atoi(v.c_str());
            else if (k == "min_p") sp.min_p = std::strtof(v.c_str(), nullptr);
            else if (k == "seed") { sp.seed = std::strtoull(v.c_str(), nullptr, 10); seeded = true; }
        }
        std::vector<int> ids;
        { std::stringstream ss(f.back()); for (std::string v; std::getline(ss, v, ',');) if (!v.empty()) ids.push_back(std::atoi(v.c_str())); }
        if (ids.empty() || w + 1 != f.size()) { std::printf("ERR the ids are missing\n"); std::fflush(stdout); continue; }
        bool bad = false;
        for (int t : ids) bad |= t < 0 || t >= kVocab;
        if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); std::fflush(stdout); continue; }
        if (int(ids.size()) + 2 > m.max_ctx) { std::printf("ERR the prompt (%zu tokens) does not fit the context (%d)\n", ids.size(), m.max_ctx); std::fflush(stdout); continue; }
        const bool sample = temperature > 0.f;
        sp.inv_temp = sample ? 1.f / temperature : 1.f;
        if (!seeded) sp.seed = (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();
        step_no = 0;
        const auto t0 = std::chrono::steady_clock::now();
        size_t common = 0;
        while (common < live.size() && common < ids.size() && live[common] == ids[common]) ++common;
        size_t reuse = 0;
        if (common == live.size() && common < ids.size()) reuse = common;                    // the cache's tokens, extended
        else if (snap_pos >= 0 && common >= size_t(snap_pos) && size_t(snap_pos) < ids.size()) reuse = size_t(snap_pos);
        // (the sliding layers' ring must still hold the window before reuse)
        if (reuse && live.size() > reuse + size_t(kSwaRing - kWin)) reuse = 0;
        if (reuse == live.size() && reuse) {}
        else if (reuse) m.restore_snap(int(reuse));
        else m.set_pos(0);
        live.resize(reuse);
        std::printf("RESUME %zu\n", reuse);
        std::fflush(stdout);
        bool cancelled = false;
        int read_n = 0;
        const size_t body = ids.size() - 1;                         // the last token goes after the snapshot
        for (size_t i = reuse; i < body;) {
            const size_t e = std::min(body, i + size_t(chunk));
            const std::vector<int> c(ids.begin() + i, ids.begin() + e);
            static const size_t feed_max = size_t(std::getenv("INKLING_FEED") ? std::atoi(std::getenv("INKLING_FEED")) : 4);
            if (c.size() > feed_max) m.prefill_chunk(c.data(), int(c.size()));
            else m.feed(c);
            live.insert(live.end(), c.begin(), c.end());
            read_n += int(c.size());
            i = e;
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("PP %d %zu %.1f %.1f\n", m.pos, ids.size(), ms, read_n / std::max(ms, 1e-3) * 1e3);
            std::fflush(stdout);
            std::string l2;
            if (i < body && in.poll_line(l2) && l2 == "STOP") { cancelled = true; break; }
        }
        if (cancelled) {
            const double prompt_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0 %d 0\n", ids.size(), prompt_ms, reuse, read_n);
            std::fflush(stdout);
            continue;
        }
        if (live.size() == body) { m.save_snap(); snap_pos = int(body); }
        m.feed({ids.back()});
        live.push_back(ids.back());
        ++read_n;
        const double prompt_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const auto t1 = std::chrono::steady_clock::now();
        int produced = 0;
        const char* finish = nullptr;
        auto pick = [&]() {
            if (sample && sp.top_k > 0 && sp.top_k <= kTopKMax) fast.run(m.logits, sp, step_no++, m.state, m.st);
            else if (sample) k_sample<<<1, 1024, 0, m.st>>>(m.logits, sp, step_no++, m.state);
            return sample ? m.read_token() : m.next_token();
        };
        int tok = pick();
        auto emit = [&](int t) {
            std::printf("T %d\n", t);
            ++produced;
            if (is_eos(t)) finish = "stop";
            else if (produced >= max_new || m.pos + 2 > m.max_ctx) finish = "length";
        };
        for (;;) {
            emit(tok);
            std::fflush(stdout);
            if (finish) break;
            std::string l2;
            if (in.poll_line(l2) && l2 == "STOP") { finish = "cancel"; break; }
            m.step();                                          // (state[1] is tok: the step reads it on the card)
            live.push_back(tok);
            tok = pick();
        }
        const double decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        std::printf("DONE %d %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0 %d 0\n", produced, ids.size(), prompt_ms, decode_ms, finish,
                    reuse, read_n);
        std::fflush(stdout);
    }
    return 0;
}

// --parallel N (READY ... multi=N): several conversations at once, in the protocol of src/dsv41's. GEN lines carry
// rid=<id>, and every line a request causes carries it after its keyword (T <rid> <id>, PP <rid> ..., DONE <rid> ...,
// ERR <rid> ...); STOP <rid> ends that request, a bare STOP every one. A request takes the idle conversation whose
// tokens its prompt continues the furthest (the one used longest ago among equals); with none idle it waits its turn.
// A prompt is read while the others wait; then every running request writes a token a step: one alone through the
// single-row step (its graph), two or more through Model::step_multi, every weight read once for all of them.
static int serve_multi(Model& m, const std::vector<int>& eos_ids, int chunk) {
    using clk = std::chrono::steady_clock;
    struct Conv { std::vector<int> live; int snap_pos = -1; bool busy = false; uint64_t used = 0; };
    struct Act {
        long long rid = 0;
        int slot = 0, max_new = 1, tok = -1, produced = 0, read_n = 0;
        bool sample = false, on_card = false, stop = false;
        SampleParams sp;
        unsigned long long step_no = 0;
        size_t n_ids = 0, reuse = 0;
        double prompt_ms = 0;
        clk::time_point t1;
    };
    const int N = m.n_slots;
    std::vector<Conv> conv((size_t) N);
    std::vector<Act> acts;
    std::deque<std::string> waiting;
    uint64_t use_clock = 0;
    long long reading = -1;      // the request whose prompt is being read
    bool reading_stop = false, quit = false;
    LineReader in;
    FastSampler fast;
    fast.init();
    std::printf("INFO engine=strata-inkling\n");
    std::printf("READY %d stop multi=%d\n", m.max_ctx, N);
    std::fflush(stdout);
    auto is_eos = [&](int t) { return std::find(eos_ids.begin(), eos_ids.end(), t) != eos_ids.end(); };
    auto rid_of = [](const std::string& line) {
        const size_t p = line.find(" rid=");
        return p == std::string::npos ? -1ll : std::atoll(line.c_str() + p + 5);
    };
    auto handle = [&](const std::string& line) {
        if (line.empty()) return;
        if (line == "QUIT") { quit = true; return; }
        if (line == "STOP") {
            for (Act& a : acts) a.stop = true;
            reading_stop = true;
            for (const std::string& w : waiting) std::printf("DONE %lld 0 0 0.0 0.0 cancel 0 0 0 0 0 0 0 0 0 0\n", rid_of(w));
            waiting.clear();
            return;
        }
        if (line.rfind("STOP ", 0) == 0) {
            const long long rid = std::atoll(line.c_str() + 5);
            for (Act& a : acts) if (a.rid == rid) a.stop = true;
            if (reading == rid) reading_stop = true;
            for (auto it = waiting.begin(); it != waiting.end(); ++it)
                if (rid_of(*it) == rid) { std::printf("DONE %lld 0 0 0.0 0.0 cancel 0 0 0 0 0 0 0 0 0 0\n", rid); waiting.erase(it); break; }
            return;
        }
        if (line.rfind("GEN ", 0) == 0) { waiting.push_back(line); return; }
        std::printf("ERR expected: GEN <max_new> [key=value ...] rid=<id> <id,id,...>\n");
    };
    auto drain = [&]() { std::string l; while (!quit && in.poll_line(l)) handle(l); };
    // how much of conversation c's cache a prompt keeps (serve()'s rule)
    auto reuse_of = [&](const Conv& c, const std::vector<int>& ids) {
        size_t common = 0;
        while (common < c.live.size() && common < ids.size() && c.live[common] == ids[common]) ++common;
        size_t reuse = 0;
        if (common == c.live.size() && common < ids.size()) reuse = common;
        else if (c.snap_pos >= 0 && common >= size_t(c.snap_pos) && size_t(c.snap_pos) < ids.size()) reuse = size_t(c.snap_pos);
        if (reuse && c.live.size() > reuse + size_t(kSwaRing - kWin)) reuse = 0;
        return reuse;
    };
    auto pick1 = [&](Act& a) {   // the bound conversation's next token from logits row 0
        if (a.sample && a.sp.top_k > 0 && a.sp.top_k <= kTopKMax) fast.run(m.logits, a.sp, a.step_no++, m.state, m.st);
        else if (a.sample) k_sample<<<1, 1024, 0, m.st>>>(m.logits, a.sp, a.step_no++, m.state);
        a.on_card = true;
        return a.sample ? m.read_token() : m.next_token();
    };
    // the first waiting request into an idle conversation: its prompt read, its first token picked
    auto admit = [&]() {
        while (!quit && !waiting.empty()) {
            int idle = 0;
            for (const Conv& c : conv) idle += !c.busy;
            if (!idle) return;
            const std::string line = waiting.front();
            waiting.pop_front();
            std::vector<std::string> f;
            { std::stringstream ss(line); for (std::string w; ss >> w;) f.push_back(w); }
            Act a;
            a.rid = rid_of(line);
            if (a.rid < 0) { std::printf("ERR a request needs rid=<id> with --parallel\n"); continue; }
            if (f.size() < 3) { std::printf("ERR %lld a request needs max_new and ids\n", a.rid); continue; }
            a.max_new = std::max(1, std::atoi(f[1].c_str()));
            float temperature = 0.f;
            bool seeded = false;
            size_t w = 2;
            for (; w + 1 < f.size(); ++w) {
                const size_t eq = f[w].find('=');
                if (eq == std::string::npos) break;
                const std::string k = f[w].substr(0, eq), v = f[w].substr(eq + 1);
                if (k == "temperature") temperature = std::strtof(v.c_str(), nullptr);
                else if (k == "top_p") a.sp.top_p = std::strtof(v.c_str(), nullptr);
                else if (k == "top_k") a.sp.top_k = std::atoi(v.c_str());
                else if (k == "min_p") a.sp.min_p = std::strtof(v.c_str(), nullptr);
                else if (k == "seed") { a.sp.seed = std::strtoull(v.c_str(), nullptr, 10); seeded = true; }
            }
            std::vector<int> ids;
            { std::stringstream ss(f.back()); for (std::string v; std::getline(ss, v, ',');) if (!v.empty()) ids.push_back(std::atoi(v.c_str())); }
            if (ids.empty() || w + 1 != f.size()) { std::printf("ERR %lld the ids are missing\n", a.rid); continue; }
            bool bad = false;
            for (int t : ids) bad |= t < 0 || t >= kVocab;
            if (bad) { std::printf("ERR %lld a token id is outside the vocabulary\n", a.rid); continue; }
            if (int(ids.size()) + 2 > m.max_ctx) { std::printf("ERR %lld the prompt (%zu tokens) does not fit the context (%d)\n", a.rid, ids.size(), m.max_ctx); continue; }
            a.sample = temperature > 0.f;
            a.sp.inv_temp = a.sample ? 1.f / temperature : 1.f;
            if (!seeded) a.sp.seed = (unsigned long long) clk::now().time_since_epoch().count();
            // the idle conversation that keeps the most; among equals the one used longest ago
            int best = -1;
            size_t best_reuse = 0;
            for (int b = 0; b < N; ++b) {
                if (conv[size_t(b)].busy) continue;
                const size_t r = reuse_of(conv[size_t(b)], ids);
                if (best < 0 || r > best_reuse || (r == best_reuse && conv[size_t(b)].used < conv[size_t(best)].used)) { best = b; best_reuse = r; }
            }
            Conv& c = conv[size_t(best)];
            a.slot = best;
            a.n_ids = ids.size();
            a.reuse = best_reuse;
            c.used = ++use_clock;
            const auto t0 = clk::now();
            m.bind(best);
            if (a.reuse == c.live.size() && a.reuse) {}
            else if (a.reuse) m.restore_snap(int(a.reuse));
            else m.set_pos(0);
            c.live.resize(a.reuse);
            reading = a.rid;
            reading_stop = false;
            const size_t body = ids.size() - 1;                     // the last token goes after the snapshot
            for (size_t i = a.reuse; i < body;) {
                const size_t e = std::min(body, i + size_t(chunk));
                const std::vector<int> ch(ids.begin() + i, ids.begin() + e);
                static const size_t feed_max = size_t(std::getenv("INKLING_FEED") ? std::atoi(std::getenv("INKLING_FEED")) : 4);
                if (ch.size() > feed_max) m.prefill_chunk(ch.data(), int(ch.size()));
                else m.feed(ch);
                c.live.insert(c.live.end(), ch.begin(), ch.end());
                a.read_n += int(ch.size());
                i = e;
                const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
                std::printf("PP %lld %d %zu %.1f %.1f\n", a.rid, m.pos, ids.size(), ms, a.read_n / std::max(ms, 1e-3) * 1e3);
                std::fflush(stdout);
                drain();
                if (reading_stop || quit) break;
            }
            reading = -1;
            if (reading_stop || quit) {
                c.snap_pos = -1;   // (the snapshot is of an earlier prompt's end, which the cache has now passed)
                const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
                std::printf("DONE %lld 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0 %d 0\n", a.rid, ids.size(), ms, a.reuse, a.read_n);
                std::fflush(stdout);
                continue;
            }
            if (c.live.size() == body) { m.save_snap(); c.snap_pos = int(body); }
            m.feed({ids.back()});
            c.live.push_back(ids.back());
            ++a.read_n;
            a.prompt_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            a.t1 = clk::now();
            a.tok = pick1(a);
            c.busy = true;
            acts.push_back(a);
        }
    };
    std::vector<int> bs, toks;
    // (INKLING_PSTAT=1: when the last request ends, the steps so far by how many conversations shared them)
    static const bool pstat = std::getenv("INKLING_PSTAT") != nullptr;
    long st_n[kMR + 1] = {};
    double st_ms[kMR + 1] = {}, read_ms = 0;
    while (!quit) {
        if (acts.empty() && waiting.empty()) {
            std::string l;
            if (!in.get(l)) break;
            handle(l);
        }
        drain();
        if (quit) break;
        const auto ta = clk::now();
        admit();
        read_ms += std::chrono::duration<double, std::milli>(clk::now() - ta).count();
        const auto ts = clk::now();
        const size_t nb = acts.size();
        // every running request's token out; the ones that end here leave
        for (size_t i = 0; i < acts.size();) {
            Act& a = acts[i];
            const char* finish = nullptr;
            if (a.stop) finish = "cancel";
            else {
                std::printf("T %lld %d\n", a.rid, a.tok);
                ++a.produced;
                if (is_eos(a.tok)) finish = "stop";
                else if (a.produced >= a.max_new || m.pos_of(a.slot) + 2 > m.max_ctx) finish = "length";
            }
            if (!finish) { ++i; continue; }
            const double decode_ms = std::chrono::duration<double, std::milli>(clk::now() - a.t1).count();
            std::printf("DONE %lld %d %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0 %d 0\n", a.rid, a.produced, a.n_ids, a.prompt_ms, decode_ms, finish,
                        a.reuse, a.read_n);
            conv[size_t(a.slot)].busy = false;
            acts.erase(acts.begin() + long(i));
        }
        std::fflush(stdout);
        if (acts.empty()) {
            if (pstat && nb) {
                std::fprintf(stderr, "strata-inkling: steps by conversations at once:");
                for (int k = 1; k <= kMR; ++k) if (st_n[k]) std::fprintf(stderr, " %d: %ld x %.2f ms;", k, st_n[k], st_ms[k] / st_n[k]);
                std::fprintf(stderr, " prompts %.0f ms\n", read_ms);
                for (int k = 0; k <= kMR; ++k) { st_n[k] = 0; st_ms[k] = 0; }
                read_ms = 0;
            }
            continue;
        }
        struct Tick { const clk::time_point t; long* n; double* ms; ~Tick() { ++*n; *ms += std::chrono::duration<double, std::milli>(clk::now() - t).count(); } }
            tick{ts, &st_n[acts.size()], &st_ms[acts.size()]};
        if (acts.size() == 1) {
            Act& a = acts[0];
            m.bind(a.slot);
            if (!a.on_card) CK(hipMemcpy(m.state + 1, &a.tok, 4, hipMemcpyHostToDevice));
            m.step();
            conv[size_t(a.slot)].live.push_back(a.tok);
            a.tok = pick1(a);
            continue;
        }
        bs.clear(); toks.clear();
        for (const Act& a : acts) { bs.push_back(a.slot); toks.push_back(a.tok); }
        m.step_multi(bs, toks.data());
        for (size_t r = 0; r < acts.size(); ++r) {
            Act& a = acts[r];
            conv[size_t(a.slot)].live.push_back(a.tok);
            const float* lg = m.logits + r * size_t(kVocabP);
            int* out = m.MTOK + 2 * r;                          // (the pick goes to out[1], as to a state)
            if (a.sample && a.sp.top_k > 0 && a.sp.top_k <= kTopKMax) fast.run(lg, a.sp, a.step_no++, out, m.st);
            else if (a.sample) k_sample<<<1, 1024, 0, m.st>>>(lg, a.sp, a.step_no++, out);
            else k_argmax64<<<64, 1024, 0, m.st>>>(lg, kVocabP, out, m.best, m.done);
        }
        CK(hipMemcpyAsync(m.mtok_h, m.MTOK, acts.size() * 8, hipMemcpyDeviceToHost, m.st));
        CK(hipStreamSynchronize(m.st));
        for (size_t r = 0; r < acts.size(); ++r) { acts[r].tok = m.mtok_h[2 * r + 1]; acts[r].on_card = false; }
    }
    return 0;
}
