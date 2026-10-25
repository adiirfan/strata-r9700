"""DeepSeek-V4.1's pictures (the deepseek41 engine reads them itself, with the vision tower in its pack).

Unlike the Qwen models, no separate encoder runs: the server only resizes each picture as DeepSeek's own
inference/image_processor.py does (plan_image_grid / load_image), writes the pixels into one `DSI1` file for the
request's GENI line, and puts each picture's span of image tokens into the prompt:

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

every position of it the image token (the engine checks the run and encodes the pixels).  The template knows no
pictures: a message's content blocks are joined with "\\n\\n" as DeepSeek's encoding.py does, each picture a
placeholder there (MARK, a private-use character, so a placeholder typed into a message stays text) that the
rendered prompt turns into the image token.
"""
from __future__ import annotations

import io
import math
import struct
from pathlib import Path

PLACEHOLDER = "<｜deepseek_image｜>"
MARK = "\U0010FF00"     # (clear of frontend's literal and control-token marks, which start at U+F0E01)


def pack_geometry(pack: Path) -> dict | None:
    """The vision keys of a pack's ds4.txt (None: the pack has no vision tower)."""
    try:
        lines = (Path(pack) / "ds4.txt").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    kv = dict(line.split(None, 1) for line in lines if " " in line)
    if int(kv.get("vision.num_hidden_layers", "0")) <= 0 or "image_token_id" not in kv:
        return None
    ratio = kv.get("vision.max_wh_ratio")
    return {"patch": int(kv.get("vision.patch_size", 14)), "merge": int(kv.get("vision.downsample_ratio", 3)),
            "min_pixels": int(kv.get("vision.min_pixels", 0)), "max_tokens": int(kv.get("vision.max_image_tokens", 1024)),
            "max_wh_ratio": float(ratio) if ratio not in (None, "", "0", "None") else None,
            "image_token": int(kv["image_token_id"])}


def from_args(args: list[str], loader=None, work_dir: Path | None = None):
    """The engine args' pack -> a Ds4Pictures, or None (no --pack, a pack without the tower, or --vision off)."""
    if "--vision" in args and args.index("--vision") + 1 < len(args) and args[args.index("--vision") + 1] == "off":
        return None
    if "--pack" not in args or args.index("--pack") + 1 >= len(args):
        return None
    geo = pack_geometry(Path(args[args.index("--pack") + 1]))
    return Ds4Pictures(geo, loader, work_dir) if geo else None


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


class Ds4Pictures:
    """Stands where the server keeps its image encoder (Service.vision) for a deepseek41 engine."""
    elastic = False
    elastic_mib = 0

    def __init__(self, geo: dict, loader=None, work_dir: Path | None = None):
        self.geo = geo
        self.image_token = geo["image_token"]
        self.loader = loader            # a source string -> its bytes (the server's Vision.load)
        self.dir = work_dir             # where a request's DSI1 file goes when /dev/shm has no room

    # (the encoder's life cycle: nothing runs apart from the engine)
    def alive(self) -> bool:
        return True

    def restart(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def stop(self) -> None:
        pass

    # ---- DeepSeek's image_processor.py
    def _llm_grid(self, h: int, w: int) -> tuple[int, int]:
        p, m = self.geo["patch"], self.geo["merge"]
        return math.ceil((h // p) / m), math.ceil((w // p) / m)

    def _solve_resize_ratio(self, height, width):
        p, m, n = self.geo["patch"], self.geo["merge"], self.geo["max_tokens"]
        r = height / width
        max_w_float = math.sqrt((n - 2) / r + 0.25) - 0.5
        max_h_float = max_w_float * r
        cell = p * m
        if max_w_float < 1.0:
            return (n - 2) // 2 * cell, cell
        if max_h_float < 1.0:
            return cell, (n - 3) * cell
        beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
        return math.floor(height * beta / p) * p, math.floor(width * beta / p) * p

    def plan(self, width: int, height: int) -> tuple[int, int, int, int]:
        """-> (n_llm_h, n_llm_w, best_height, best_width) for a picture of this size."""
        p, ratio = self.geo["patch"], self.geo["max_wh_ratio"]
        if ratio is not None and width > height * ratio:
            width = height * ratio
        if 0 < width * height < self.geo["min_pixels"]:
            k = (self.geo["min_pixels"] / (width * height)) ** 0.5
            width, height = int(width * k), int(height * k)
        bw, bh = math.ceil(width / p) * p, math.ceil(height / p) * p
        n_h, n_w = self._llm_grid(bh, bw)
        if num_image_tokens(n_h, n_w) > self.geo["max_tokens"]:
            bh, bw = self._solve_resize_ratio(height, width)
            n_h, n_w = self._llm_grid(bh, bw)
        return n_h, n_w, bh, bw

    def pixels(self, source) -> tuple[bytes, int, int, int]:
        """A picture (its bytes, or what `loader` reads: a data: URL, a URL already downloaded, a path) -> (RGB rows,
        width, height, its span's length)."""
        from PIL import Image, ImageOps
        data = source if isinstance(source, bytes) else self.loader(source)
        with Image.open(io.BytesIO(data)) as im:
            image = im.convert("RGB")
        n_h, n_w, bh, bw = self.plan(image.width, image.height)
        ratio = self.geo["max_wh_ratio"]
        if ratio is not None and image.width >= ratio * image.height:
            image = image.resize((bw, bh))
        else:
            image = ImageOps.pad(image, (bw, bh), color=(127, 127, 127))
        return image.tobytes(), bw, bh, num_image_tokens(n_h, n_w)

    def encode(self, source):
        """(a tool's picture checked before the prompt is built: one that cannot be read raises here)"""
        return self.pixels(source)


def flatten(messages: list[dict]) -> list[dict]:
    """Content block lists -> one text, blocks joined with "\\n\\n" and each picture MARK (encoding.py's
    process_image_messages); messages without a list come back as they are."""
    out = []
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list):
            parts = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") in ("image", "image_url"):
                    parts.append(MARK)
                elif b.get("type") == "text":
                    parts.append(b.get("text") or "")
            m = {**m, "content": "\n\n".join(parts)}
        out.append(m)
    return out


def expand(ids: list[int], token: int, spans: list[int]) -> tuple[list[int], list[int]]:
    """Each image token of `ids` (one a picture, in order) -> its span of `spans[k]` image tokens; -> (ids, the spans'
    starts)."""
    out, starts, k = [], [], 0
    for t in ids:
        if t != token:
            out.append(t)
            continue
        if k == len(spans):
            raise ValueError("the prompt and its images do not match")
        starts.append(len(out))
        out += [token] * spans[k]
        k += 1
    if k != len(spans):
        raise ValueError("the prompt and its images do not match")
    return out, starts


def write_dsi(path: Path, pics: list[tuple[bytes, int, int, int]], starts: list[int]) -> None:
    """The pictures as the engine's GENI file: int32 'DSI1', int32 count, per picture int64 start, int32 width,
    int32 height, the RGB rows."""
    with open(path, "wb") as f:
        f.write(b"DSI1" + struct.pack("<i", len(pics)))
        for (rgb, w, h, _), s in zip(pics, starts):
            f.write(struct.pack("<qii", s, w, h))
            f.write(rgb)
