"""ref_gptoss.py: gpt-oss (openai-moe GGUF: Q8_0 dense, MXFP4 experts) in plain numpy, float64 sums, for checking
strata-gptoss. Follows OpenAI's reference implementation (gpt_oss/torch/model.py): YaRN without rounding the ramp
ends (--llama-yarn: llama.cpp's floor / ceil instead), learned sink logits, sliding window 128 on the even layers,
router top 4 then softmax over them, SwiGLU with alpha 1.702 and clamp 7 ((up + 1) * glu).

  ref_gptoss.py --model FILE.gguf --ids 1,2,3 [--top FILE] [--logits FILE] [--new N]

--top: per position the five most likely next tokens "id:logprob ..."; --logits: every position's logits (fp32);
--new N: then N greedy tokens (printed)."""
import argparse, math, sys
import numpy as np

sys.path.insert(0, "/home/debian/llama.cpp-gpp/gguf-py")
import gguf  # noqa: E402

KV_MXFP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float32)
H, NH, NKV, HD, NE, TOPK, WIN = 2880, 64, 8, 64, 128, 4, 128


def deq_q8_0(raw):  # (rows, nb * 34) uint8 -> (rows, nb * 32) float32
    b = raw.reshape(raw.shape[0], -1, 34)
    d = b[:, :, :2].copy().view(np.float16).astype(np.float32)
    q = b[:, :, 2:].view(np.int8).astype(np.float32)
    return (q * d).reshape(raw.shape[0], -1)


def deq_mxfp4(raw):  # (rows, nb * 17) -> (rows, nb * 32)
    b = raw.reshape(raw.shape[0], -1, 17)
    e = b[:, :, 0].astype(np.int32)
    d = np.where(e < 2, np.ldexp(1.0, -128 + e), np.ldexp(1.0, e - 128)).astype(np.float32)
    qs = b[:, :, 1:]
    lo, hi = KV_MXFP4[qs & 15], KV_MXFP4[qs >> 4]
    return (np.concatenate([lo, hi], axis=2) * d[:, :, None]).reshape(raw.shape[0], -1)


def rms(x, w, eps=1e-5):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def yarn(llama_style=False, base=150000.0, factor=32.0, ctx0=4096, beta_fast=32.0, beta_slow=1.0):
    dh = HD // 2
    freq = base ** (np.arange(0, HD, 2, dtype=np.float64) / HD)
    conc = 0.1 * math.log(factor) + 1.0
    low = dh * math.log(ctx0 / (beta_fast * 2 * math.pi)) / math.log(base)
    high = dh * math.log(ctx0 / (beta_slow * 2 * math.pi)) / math.log(base)
    if llama_style:
        low, high = max(0.0, math.floor(low)), min(HD - 1.0, math.ceil(high))
    ramp = np.clip((np.arange(dh) - low) / max(high - low, 0.001), 0, 1)
    mask = 1 - ramp   # 1: keep the original frequency (fast dims), 0: interpolate
    inv = (1 / (factor * freq)) * (1 - mask) + (1 / freq) * mask
    return inv, conc


def rope(x, pos, inv, conc):  # x (..., T, 64), pos (T,)
    ang = pos[:, None] * inv[None, :]
    c, s = np.cos(ang) * conc, np.sin(ang) * conc
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


class Model:
    def __init__(self, path, llama_yarn=False):
        self.r = gguf.GGUFReader(path)
        self.T = {t.name: t for t in self.r.tensors}
        self.inv, self.conc = yarn(llama_yarn)
        self.nl = int(self.r.fields["gpt-oss.block_count"].contents())
        self.cache = {}

    def raw(self, n):
        return np.asarray(self.T[n].data)

    def f32(self, n):
        return self.raw(n).astype(np.float64)

    def dense(self, n):  # Q8_0 matrix as float64 (cached per layer while it is in use)
        if n not in self.cache:
            self.cache[n] = deq_q8_0(self.raw(n)).astype(np.float64)
        return self.cache[n]

    def expert(self, li, kind, e):
        return deq_mxfp4(self.raw(f"blk.{li}.ffn_{kind}_exps.weight")[e]).astype(np.float64)

    def forward(self, ids, dump=None):
        ids = np.asarray(ids)
        T_ = len(ids)
        h = deq_q8_0(self.raw("token_embd.weight")[ids]).astype(np.float64)
        pos = np.arange(T_, dtype=np.float64)
        causal = np.tril(np.ones((T_, T_), bool))
        swa_m = causal & (np.arange(T_)[:, None] - np.arange(T_)[None, :] < WIN)
        for li in range(self.nl):
            b = f"blk.{li}."
            x = rms(h, self.f32(b + "attn_norm.weight"))
            q = x @ self.dense(b + "attn_q.weight").T + self.f32(b + "attn_q.bias")
            k = x @ self.dense(b + "attn_k.weight").T + self.f32(b + "attn_k.bias")
            v = x @ self.dense(b + "attn_v.weight").T + self.f32(b + "attn_v.bias")
            q = rope(q.reshape(T_, NH, HD).transpose(1, 0, 2), pos, self.inv, self.conc)   # (NH, T, HD)
            k = rope(k.reshape(T_, NKV, HD).transpose(1, 0, 2), pos, self.inv, self.conc)
            v = v.reshape(T_, NKV, HD).transpose(1, 0, 2)
            sinks = self.f32(b + "attn_sinks.weight")
            m = swa_m if li % 2 == 0 else causal
            att = np.empty((NH, T_, HD))
            for hh in range(NH):
                kh = hh // (NH // NKV)
                s = q[hh] @ k[kh].T * (1 / math.sqrt(HD))
                s = np.where(m, s, -np.inf)
                mx = np.maximum(s.max(axis=1), sinks[hh])
                p = np.exp(s - mx[:, None])
                den = p.sum(axis=1) + np.exp(sinks[hh] - mx)
                att[hh] = (p / den[:, None]) @ v[kh]
            att = att.transpose(1, 0, 2).reshape(T_, NH * HD)
            h = h + att @ self.dense(b + "attn_output.weight").T + self.f32(b + "attn_output.bias")
            x = rms(h, self.f32(b + "post_attention_norm.weight"))
            lg = x @ self.f32(b + "ffn_gate_inp.weight").T + self.f32(b + "ffn_gate_inp.bias")
            bg, bu, bd = (self.f32(b + f"ffn_{k_}_exps.bias") for k_ in ("gate", "up", "down"))
            y = np.zeros_like(h)
            for t in range(T_):
                top = np.argsort(-lg[t], kind="stable")[:TOPK]
                w = np.exp(lg[t, top] - lg[t, top].max())
                w /= w.sum()
                for e, we in zip(top, w):
                    g = self.expert(li, "gate", e) @ x[t] + bg[e]
                    u = self.expert(li, "up", e) @ x[t] + bu[e]
                    g = np.minimum(g, 7.0)
                    u = np.clip(u, -7.0, 7.0)
                    a = g / (1 + np.exp(-1.702 * g)) * (u + 1)
                    y[t] += we * (self.expert(li, "down", e) @ a + bd[e])
            h = h + y
            self.cache.clear()
            if dump is not None:
                dump.append(h.copy())
            print(f"layer {li} done", file=sys.stderr, flush=True)
        x = rms(h, self.f32("output_norm.weight"))
        W = deq_q8_0(self.raw("output.weight"))
        return x @ W.T.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--top")
    ap.add_argument("--logits")
    ap.add_argument("--dump-h", help="h after every layer (fp32, layers x T x 2880)")
    ap.add_argument("--new", type=int, default=0)
    ap.add_argument("--llama-yarn", action="store_true")
    a = ap.parse_args()
    ids = [int(s) for s in (open(a.ids[1:]).read() if a.ids.startswith("@") else a.ids).replace("\n", "").split(",") if s]
    m = Model(a.model, a.llama_yarn)
    dump = [] if a.dump_h else None
    lg = m.forward(ids, dump)
    if dump is not None:
        np.asarray(dump, np.float32).tofile(a.dump_h)
    if a.logits:
        lg.astype(np.float32).tofile(a.logits)
    if a.top:
        with open(a.top, "w") as f:
            for row in lg:
                lse = row.max() + np.log(np.exp(row - row.max()).sum())
                o = np.argsort(-row)[:5]
                f.write(" ".join(f"{i}:{row[i] - lse:.5f}" for i in o) + "\n")
    out = []
    for _ in range(a.new):
        nxt = int(np.argmax(lg[-1]))
        out.append(nxt)
        ids.append(nxt)
        lg = m.forward(ids)
    if out:
        print(",".join(map(str, out)))
    print("greedy next after the prompt:", int(np.argmax(lg[-1])) if not out else out[0], file=sys.stderr)


if __name__ == "__main__":
    main()
