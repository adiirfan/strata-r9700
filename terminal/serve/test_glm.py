"""GLM-5.3-Flash (strata-glm) through the server: its pre-tokenizer, its chat template's thinking levels, forced tool
calls in its form, and its family defaults (no GPU, no pack, no model).

    python -m unittest serve.test_glm -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402
from serve import frontend  # noqa: E402
from serve.frontend import ChatTemplate, FlintParser, forced_call, to_glm_force  # noqa: E402
from serve.server import FAMILY_MODEL_NAMES, default_model_name, model_card_sampling  # noqa: E402

TEMPLATE = ROOT / "serve" / "glm5_chat_template.jinja"
TOOLS = [{"name": "get_weather", "description": "the weather in a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]


class Tokenizer(unittest.TestCase):
    def tok(self, pre):
        tokens = ["a", "b", "c", "ab", "abc", "1", "2", "3", "4", "12", "123"]
        return ST.Tokenizer(tokens, ["a b", "1 2", "12 3"], None, pre)

    def test_whole_vocabulary_piece_is_kept(self):
        # ignore_merges (glm4/glm5): "abc" is a vocabulary token, so it is emitted whole although no merge makes it
        t = self.tok("glm5")
        self.assertEqual(t.encode("abc"), [t.ids["abc"]])
        q = self.tok("qwen35")
        self.assertEqual(q.encode("abc"), [q.ids["ab"], q.ids["c"]])

    def test_digits_in_threes(self):
        t = self.tok("glm4")
        self.assertEqual(t.encode("1234"), [t.ids["123"], t.ids["4"]])


class Template(unittest.TestCase):
    def setUp(self):
        frontend.use_family("glm5")
        self.tpl = ChatTemplate(TEMPLATE)
        self.msgs = [{"role": "user", "content": "hi"}]

    def tearDown(self):
        frontend.use_family(None)

    def test_levels(self):
        self.assertTrue(self.tpl.glm_effort)
        for ours, glm in (("low", "Low"), ("medium", "High"), ("high", "High"), ("max", "Max"), (None, "Max")):
            out = self.tpl.render(self.msgs, reasoning_effort=ours) if ours else self.tpl.render(self.msgs)
            self.assertIn(f"Reasoning Effort: {glm}", out, ours)
            self.assertTrue(out.endswith("<think>"), ours)

    def test_thinking_off_closes_the_block(self):
        out = self.tpl.render(self.msgs, enable_thinking=False, reasoning_effort="high")
        self.assertIn("Reasoning Effort: Low", out)
        self.assertTrue(out.endswith("<think></think>"))

    def test_picture_markers(self):
        # what Service.prepare looks for in the GLM family: <|image|> right after <|begin_of_image|>
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "what is this?"}]}]
        self.assertIn("<|begin_of_image|><|image|><|end_of_image|>", self.tpl.render(msgs))

    def test_effort_words(self):
        self.assertEqual(frontend.effort_kwargs("high"), {"reasoning_effort": "high"})
        self.assertEqual(frontend.effort_kwargs("none"), {"enable_thinking": False})


class ToolCalls(unittest.TestCase):
    def test_forced(self):
        self.assertEqual(to_glm_force(forced_call({"type": "function", "function": {"name": "get_weather"}}, TOOLS)),
                         "<tool_call>get_weather")
        two = TOOLS + [{"name": "get_time", "parameters": {"type": "object", "properties": {}}}]
        self.assertEqual(to_glm_force(forced_call("required", two)), "<tool_call>")
        self.assertIsNone(to_glm_force(None))

    def test_forced_opening_then_the_model(self):
        p = FlintParser(thinking=False, tools=TOOLS, stream_tools=True)
        events = p.feed(to_glm_force(forced_call("required", TOOLS))) + \
            p.feed("<arg_key>city</arg_key><arg_value>Jakarta</arg_value></tool_call>") + p.finish("stop")
        calls = [e.call for e in events if e.kind == "tool_call"]
        self.assertEqual([(c.name, c.arguments) for c in calls], [("get_weather", {"city": "Jakarta"})])


class Family(unittest.TestCase):
    def test_name_and_sampling(self):
        self.assertIn("glm5", FAMILY_MODEL_NAMES)
        self.assertEqual(default_model_name({"arch": "glm5"}), "glm-5.3-flash")
        self.assertEqual(model_card_sampling("glm5", True), {"temperature": 1.0, "top_p": 0.95})


if __name__ == "__main__":
    unittest.main()
