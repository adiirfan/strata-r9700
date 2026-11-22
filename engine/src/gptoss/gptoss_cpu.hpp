// src/gptoss/gptoss_cpu.hpp - strata-gptoss's CPU side: the routed experts the card does not hold (see gptoss_cpu.cpp)
#pragma once
#include <cstdint>

namespace gptoss_cpu {

// one MXFP4 matrix as the engine stores it: rows of n_in / 2 nibble bytes (q), and n_in / 32 scale bytes a row (e)
struct Mat { const uint8_t* q; const uint8_t* e; };
// a routed expert: gate / up (n_ff x n_in), down (n_in x n_ff), and their fp32 biases
struct Expert { Mat gate, up, down; const float *bg, *bu, *bd; };

// the worker threads (0: one per physical core); once
void init(int threads);
// workers spin while a writing step is under way, sleep otherwise
void set_active(bool active);
// y (n_in) = sum_k w[k] * (down_k(swiglu(gate_k x + bg_k, up_k x + bu_k)) + bd_k), x already normed; swiglu as
// gpt-oss: g = min(g, 7), u = clamp(u, -7, 7), (u + 1) * g * sigmoid(1.702 g)
void moe(const float* x, int n_in, int n_ff, int n, const Expert* ex, const float* w, float* y);

}  // namespace gptoss_cpu
