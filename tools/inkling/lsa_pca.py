"""lsa_pca.py OUT.bin DUMP_PREFIX: the LSA map for strata-inkling from its own keys (INKLING_DUMP_K=DUMP_PREFIX): per
global pass (visits 5, 11, .., 47) and K/V head, the top 32 directions of the keys' second moment (uncentred: the
index must keep q . k), as [kKV][128][32] fp32, the passes in order."""
import struct, sys
import numpy as np

KV, HD, R = 8, 128, 32
out, pre = sys.argv[1], sys.argv[2]
with open(out, "wb") as f:
    for v in (5, 11, 17, 23, 29, 35, 41, 47):
        d = open(f"{pre}_v{v}.bin", "rb").read()
        _, n = struct.unpack_from("<2i", d, 0)
        k = np.frombuffer(d, np.int8, n * KV * HD, 8).reshape(n, KV, HD).astype(np.float32)
        s = np.frombuffer(d, np.float16, n * KV, 8 + n * KV * HD).reshape(n, KV, 1).astype(np.float32)
        k = k * s
        maps, kept = [], []
        for h in range(KV):
            x = k[:, h, :]
            m = x.T @ x / n
            w, u = np.linalg.eigh(m)
            maps.append(u[:, ::-1][:, :R])
            kept.append(w[::-1][:R].sum() / w.sum())
        np.stack(maps).astype(np.float32).tofile(f)
        print(f"visit {v}: {n} keys, energy kept by {R} dims per head: " + " ".join(f"{x:.2f}" for x in kept))
