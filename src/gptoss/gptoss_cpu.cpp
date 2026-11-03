// src/gptoss/gptoss_cpu.cpp - hybrid mode's CPU half for gpt-oss: the routed experts the card does not hold, from the
// engine's host arena (MXFP4: a 32-value block is 16 nibble bytes - low nibbles values 0-15, high 16-31 - and a scale
// byte 2^(e - 128) applied to the doubled E2M1 values), times fp32 inputs. AVX-512 (VBMI shuffle not needed: the
// 16-entry table fits SSSE3's byte shuffle).
#include "gptoss_cpu.hpp"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <pthread.h>
#include <sched.h>

namespace gptoss_cpu {
namespace {

inline float e8_half(uint8_t e) {   // 2^(e - 128) (ggml's e8m0 "half", for the doubled E2M1 values)
    uint32_t b = e < 2 ? (0x00200000u << e) : uint32_t(e - 1) << 23;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
// one row (nb blocks) . x
__attribute__((target("avx512f,avx512bw,fma"))) float dot512(const uint8_t* q, const uint8_t* e, const float* x, int nb) {
    const __m128i tbl = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i mask = _mm_set1_epi8(0x0F);
    __m512 acc = _mm512_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const __m128i v = _mm_loadu_si128((const __m128i*) (q + b * 16));
        const __m128i lo = _mm_shuffle_epi8(tbl, _mm_and_si128(v, mask));
        const __m128i hi = _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(v, 4), mask));
        __m512 t = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)), _mm512_loadu_ps(x + b * 32));
        t = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)), _mm512_loadu_ps(x + b * 32 + 16), t);
        acc = _mm512_fmadd_ps(_mm512_set1_ps(e8_half(e[b])), t, acc);
    }
    return _mm512_reduce_add_ps(acc);
}

void pin_to(int cpu) {
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

// A pool of n - 1 workers plus the calling thread (as strata-flint's): workers spin while a writing step is under
// way - a layer's CPU share is a few hundred microseconds - and sleep otherwise
class Pool {
public:
    explicit Pool(int n, const std::vector<int>& cores) : n_(n) {
        for (int i = 1; i < n_; ++i)
            th_.emplace_back([this, i, cores] {
                if (i < (int) cores.size()) pin_to(cores[i]);
                work(i);
            });
    }
    int size() const { return n_; }
    void set_active(bool a) {
        active_.store(a, std::memory_order_relaxed);
        if (a) cv_.notify_all();
    }
    template <class F> void run(F&& f) {
        job_ = [&f](int id) { f(id); };
        left_.store(n_ - 1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mu_);
            gen_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_all();
        job_(0);
        while (left_.load(std::memory_order_acquire) != 0) _mm_pause();
    }

private:
    void work(int id) {
        uint64_t seen = 0;
        for (;;) {
            while (gen_.load(std::memory_order_acquire) == seen) {
                if (active_.load(std::memory_order_relaxed)) { _mm_pause(); continue; }
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return gen_.load(std::memory_order_acquire) != seen || active_.load(std::memory_order_relaxed); });
            }
            seen = gen_.load(std::memory_order_acquire);
            job_(id);
            left_.fetch_sub(1, std::memory_order_release);
        }
    }
    int n_;
    std::vector<std::thread> th_;
    std::function<void(int)> job_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> left_{0};
    std::atomic<bool> active_{false};
    std::mutex mu_;
    std::condition_variable cv_;
};

Pool* g_pool = nullptr;
std::vector<int> g_cores;
cpu_set_t g_host_saved;
bool g_host_pinned = false;

// one logical processor per physical core (the first of its SMT siblings)
std::vector<int> physical_cores() {
    std::vector<int> out;
    std::set<std::string> seen;
    for (int c = 0; c < 1024; ++c) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
        if (!f) break;
        std::string sib;
        std::getline(f, sib);
        if (seen.insert(sib).second) out.push_back(c);
    }
    return out;
}

}  // namespace

void init(int threads) {
    if (g_pool) return;
    g_cores = physical_cores();
    if (threads <= 0) threads = g_cores.empty() ? int(std::max(1u, std::thread::hardware_concurrency() / 2)) : int(g_cores.size());
    g_pool = new Pool(threads, g_cores);
}
void set_active(bool a) {
    if (!g_pool) return;
    g_pool->set_active(a);
    if (a && !g_cores.empty() && !g_host_pinned) {
        g_host_pinned = pthread_getaffinity_np(pthread_self(), sizeof g_host_saved, &g_host_saved) == 0;
        if (g_host_pinned) pin_to(g_cores[0]);
    } else if (!a && g_host_pinned) {
        pthread_setaffinity_np(pthread_self(), sizeof g_host_saved, &g_host_saved);
        g_host_pinned = false;
    }
}

void moe(const float* x, int n_in, int n_ff, int n, const Expert* ex, const float* w, float* y) {
    static thread_local std::vector<float> actv;   // (the workers get plain pointers)
    actv.resize(size_t(n) * n_ff);
    float* act = actv.data();
    const int P = g_pool->size(), nbi = n_in / 32, nbf = n_ff / 32;
    const int rows1 = n * n_ff;
    g_pool->run([&](int id) {
        const int a = rows1 * id / P, z = rows1 * (id + 1) / P;
        for (int rr = a; rr < z; ++rr) {
            const int k = rr / n_ff, row = rr % n_ff;
            const Expert& E = ex[k];
            float g = dot512(E.gate.q + size_t(row) * (n_in / 2), E.gate.e + size_t(row) * nbi, x, nbi) + E.bg[row];
            float u = dot512(E.up.q + size_t(row) * (n_in / 2), E.up.e + size_t(row) * nbi, x, nbi) + E.bu[row];
            g = std::min(g, 7.f);
            u = std::max(-7.f, std::min(u, 7.f));
            act[rr] = (u + 1.f) * g / (1.f + std::exp(-1.702f * g));
        }
    });
    g_pool->run([&](int id) {
        const int a = n_in * id / P, z = n_in * (id + 1) / P;
        for (int row = a; row < z; ++row) {
            float t = 0;
            for (int k = 0; k < n; ++k) {
                const Expert& E = ex[k];
                t += w[k] * (dot512(E.down.q + size_t(row) * (n_ff / 2), E.down.e + size_t(row) * nbf, act + size_t(k) * n_ff, nbf) + E.bd[row]);
            }
            y[row] = t;
        }
    });
}

}  // namespace gptoss_cpu
