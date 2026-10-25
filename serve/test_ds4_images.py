"""serve/ds4_images.py: DeepSeek-V4.1's picture plan, prompt spans and GENI file."""
import struct
import tempfile
import unittest
from pathlib import Path

from serve import ds4_images as DI

GEO = {"patch": 14, "merge": 3, "min_pixels": 295936, "max_tokens": 1024, "max_wh_ratio": None, "image_token": 129264}


class Plan(unittest.TestCase):
    def test_a_picture_that_fits_is_rounded_up_to_whole_patches(self):
        # 1024 x 701 -> 1036 x 714 pixels, a 25 x 17 token grid: 17 rows of 25 + a newline, + start and end
        self.assertEqual(DI.Ds4Pictures(GEO).plan(1024, 701), (17, 25, 714, 1036))

    def test_a_small_picture_grows_to_min_pixels(self):
        n_h, n_w, h, w = DI.Ds4Pictures(GEO).plan(100, 100)
        self.assertGreaterEqual(h * w, GEO["min_pixels"])

    def test_a_large_picture_fits_max_tokens(self):
        n_h, n_w, h, w = DI.Ds4Pictures(GEO).plan(4000, 3000)
        self.assertLessEqual(DI.num_image_tokens(n_h, n_w), GEO["max_tokens"])
        self.assertEqual((h % 14, w % 14), (0, 0))


class Prompt(unittest.TestCase):
    def test_flatten_joins_blocks_and_marks_pictures(self):
        m = [{"role": "user", "content": [{"type": "image", "source": "x"}, {"type": "text", "text": "what is it?"}]},
             {"role": "assistant", "content": "a cat"}]
        out = DI.flatten(m)
        self.assertEqual(out[0]["content"], DI.MARK + "\n\nwhat is it?")
        self.assertIs(out[1], m[1])

    def test_expand_gives_each_picture_its_span(self):
        t = GEO["image_token"]
        ids, starts = DI.expand([5, t, 6, t, 7], t, [3, 2])
        self.assertEqual(ids, [5, t, t, t, 6, t, t, 7])
        self.assertEqual(starts, [1, 5])
        with self.assertRaises(ValueError):
            DI.expand([5, t], t, [3, 2])

    def test_the_geni_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.dsi"
            DI.write_dsi(p, [(b"\x01" * 12, 2, 2, 8)], [3])
            b = p.read_bytes()
        self.assertEqual(b[:8], b"DSI1" + struct.pack("<i", 1))
        self.assertEqual(struct.unpack("<qii", b[8:24]), (3, 2, 2))
        self.assertEqual(b[24:], b"\x01" * 12)


if __name__ == "__main__":
    unittest.main()
