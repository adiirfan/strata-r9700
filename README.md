# strata-r9700: DeepSeek-V4.1-Flash, GLM-5.3-Flash, Qwen3.8-Flash-Next, Nemotron 3.5 Bolt and Recursive Flint on one Radeon AI PRO R9700

This is [Strata](https://github.com/Niko1221/Strata) 0.2.4 (commit `96ca4383`) tuned for a 32 GB Radeon AI PRO R9700,
with four more models beside Strata's own. Every model works behind Strata's server, which offers OpenAI- and
Anthropic-compatible APIs and a web app. Strata's own README is [README-STRATA.md](README-STRATA.md).

- **Qwen3.8-Flash-Next** is Strata's own model. This repo adds the R9700 work from
  [strata-patches](https://github.com/adiirfan/strata-patches): 8-bit matrix maths for prompts, conversation memory that
  is pinned only as a conversation grows, and experts given back from RAM after a long prompt.
- **DeepSeek-V4.1-Flash**: Strata 0.2.4 carries its engine (`src/dsv41/`, from Project Maya). This repo tunes it for
  this card and a fast SSD and lets the server pass it pictures.
- **GLM-5.3-Flash**: Project Maya's GLM engine (v1.1.28), built here as `engine/strata-glm` (`src/glm/`), with the
  server, the tokenizer and the picture encoder taught GLM's ways, and its prompt path made several times faster on
  this card.
- **Nemotron 3.5 Bolt, Recursive Flint and Ling-3.0-flash** are programs of their own. (This repo was `strata-nemo`,
  built on an older Strata. Nemotron, with and without its drafter, and Flint give the same output, token for token, on
  0.2.4 as before. Ling builds, but its model was not on hand for these measurements.)

| Program | Model | Weights | Where they live | Context | Code |
| --- | --- | --- | --- | ---: | --- |
| `engine/strata-dsv41` | [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), [ISTA-DASLab's 2-bit GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) | 138 GB + 111 GB of Engram tables | card + RAM + SSD | 262,144 | `src/dsv41/`, `serve/ds4_images.py` |
| `engine/strata-glm` | [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) 321B-A18B, [Maya-M-Derisked IQ2_S GGUF](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF) (fewer blanket refusals), vision files from [the Maya release](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF) | 116 GB, of it 97 GB of experts | card + RAM (+ SSD with a cap) | 262,144 | `src/glm/`, `tools/glm/`, `tools/vision/glm5next*` |
| `engine/strata-r9700` | [Qwen3.8-Flash-Next](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF) (SC117's abliterated GSQ-RCO IQ3_S) | 79 GB, of it 50 GB of experts | card + RAM (+ SSD with a cap) | 262,144 | `src/core/`, `src/prefill/`, `src/program/` |
| `engine/strata-nemo` | [Nemotron 3.5 Bolt 74B-A9B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB), with the [LiLiCorr drafter](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr) | 24 GB | all on the card | 262,144 | `src/nemotron/` |
| `engine/strata-flint` | [Recursive Flint 40B-A2.5B](https://huggingface.co/RecursiveSI/flint), [Q4_1 GGUF](https://huggingface.co/RecursiveSI/flint-int4-gguf) | 26 GB | all on the card | 262,144 | `src/flint/` |
| `engine/strata-ling` | [Ling-3.0-flash-VL 124B-A5.5B](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL), [Q4_0 GGUF](https://huggingface.co/bartowski/Ling-3.0-flash-VL-GGUF) | 67 GB | card + RAM | 131,072 | `src/ling/` |

## Results

Every model was measured the same way, through the server, one after another on the same desktop:

1. **Strata's benchmark,** `tools/strata_bench.py --long`: three short chats, fresh 4K, 16K and 32K prompts, and a
   six-turn agent session with four tools. Each test three times; the table gives the median.
2. **A reading and writing sweep,** `tools/r9700/sweep_bench.py`: fresh prompts of 1K to 256K tokens, each answered once
   with up to 128 tokens and once with up to 1,024.

Both send the same request: streamed, greedy (temperature 0), thinking off. The prompts are real text, never repeated
(the sweep: Strata's README and docs; every prompt starts with a new number so no cache helps). Speeds are the engine's
own counts in tokens per second; first-token times are measured by the client.

**The setups:**
- **DeepSeek:** the leave-out setting at 0.20 and 24K prompt chunks (see [DeepSeek-V4.1-Flash](#deepseek-v41-flash)).
  *72 GB store*: a 72 GB RAM store, no cap (about 80 GB of RAM in all). *48 GB cap* and *32 GB cap*: the whole server
  held to that much RAM by the system (systemd `MemoryMax`), with a 38 GB and a 21 GB store.
- **GLM:** the leave-out setting at 0.12 on every layer and prompt chunks of up to 32K tokens (see
  [GLM-5.3-Flash](#glm-53-flash)). *All in RAM*: every expert on the card or in RAM bar a few on the SSD (about 83 GB in
  all). *48 GB cap* and *32 GB cap*: the whole server held to that much RAM, with a 36 GB and a 24 GB expert store.
- **Qwen:** *all in RAM*: every expert in RAM (about 55 GB). *32 GB cap* and *16 GB cap*: the whole server held to that
  much RAM, with 20 GB and 5 GB of experts in RAM and the rest read from the SSD.
- **Nemotron:** with its drafter. **Flint:** with the server's guessing ahead.

**Desktop:** AMD Ryzen 5 7500F (6 cores), 96 GB DDR5-5600, a 2 TB Viper VP4300L SSD (PCIe 4.0). Radeon AI PRO R9700
32 GB at a 230 W power limit (default 300 W). Debian 13, kernel 7.1.8, ROCm 10.2.0a20261009 (AMD TheRock).

### Strata's benchmark

| Test | DeepSeek, 72 GB store | DeepSeek, 48 GB cap | DeepSeek, 32 GB cap | GLM, all in RAM | GLM, 48 GB cap | GLM, 32 GB cap | Qwen, all in RAM | Qwen, 32 GB cap | Qwen, 16 GB cap | Nemotron | Flint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Chat: writing (tokens/s) | 17.9 | 15.3 | 12.0 | 28.4 | 22.1 | 17.8 | 78.3 | 60.1 | 40.9 | 124.4 | 238.0 |
| Chat: first token | 2.67 s | 3.28 s | 4.05 s | 1.35 s | 2.87 s | 4.05 s | 0.49 s | 0.57 s | 0.89 s | 0.12 s | 0.07 s |
| 4K prompt: reading (tokens/s) | 720 | 595 | 506 | 884 | 436 | 347 | 2,332 | 1,382 | 504 | 4,930 | 6,829 |
| 4K prompt: first token | 5.73 s | 6.94 s | 8.15 s | 4.61 s | 9.28 s | 11.66 s | 1.75 s | 2.93 s | 7.99 s | 0.84 s | 0.60 s |
| 16K prompt: reading (tokens/s) | 1,176 | 1,286 | 1,218 | 989 | 968 | 921 | 3,246 | 2,692 | 1,587 | 5,237 | 6,506 |
| 16K prompt: first token | 14.11 s | 12.90 s | 13.61 s | 19.07 s | 16.71 s | 17.60 s | 4.98 s | 6.00 s | 10.15 s | 3.16 s | 2.54 s |
| 32K prompt: reading (tokens/s) | 1,064 | 1,389 | 1,326 | 977 | 966 | 908 | 3,380 | 2,708 | 2,120 | 4,990 | 5,721 |
| 32K prompt: first token | 31.18 s | 23.89 s | 25.01 s | 35.77 s | 33.42 s | 35.54 s | 9.52 s | 11.87 s | 15.16 s | 6.62 s | 5.76 s |
| Agent session: prompt reused | 69% | 69% | 69% | 69% | 69% | 69% | 69% | 69% | 69% | 69% | 68% |
| Agent session: first token per turn | 9.89 s | 12.52 s | 15.01 s | 4.34 s | 9.32 s | 11.59 s | 1.60 s | 2.79 s | 7.55 s | 0.62 s | 0.46 s |
| Agent session: writing (tokens/s) | 12.5 | 12.9 | 11.7 | 25.8 | 21.0 | 17.4 | 76.6 | 71.6 | 58.9 | 97.5 | 206.5 |

- Nemotron's writing moves with how many of the drafter's guesses the model keeps (105-221 tokens/s over the chats).
- The agent session reuses 69% of each turn's prompt from the conversation cache in every setup, so its first-token
  time is the cost of the new part of each turn.

### Reading and writing by prompt size

"Reading" and "First token" are the mean of the two runs at that size. "Writing 128" and "Writing 1,024" are the
answers with up to 128 and 1,024 tokens. The prompt column gives the size asked for and, in brackets, the tokens the
model's own tokenizer made of it.

#### DeepSeek, 72 GB store

| Prompt | Reading (tokens/s) | First token | Writing 128 (tokens/s) | Writing 1,024 (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1K (1,067) | 120 | 8.9 s | 11.4 | 14.0 |
| 4K (4,212) | 419 | 10.0 s | 13.8 | 15.6 |
| 16K (14,833) | 837 | 18.7 s | 10.0 | 15.2 |
| 32K (30,660) | 1,153 | 27.1 s | 14.0 | 16.0 |
| 64K (63,685) | 1,092 | 58.5 s | 13.9 | 15.7 |
| 128K (123,616) | 1,088 | 114.3 s | 13.0 | 17.3 |
| 256K (231,917) | 1,127 | 208.8 s | 14.1 | 18.0 |

#### DeepSeek, 48 GB cap

| Prompt | Reading (tokens/s) | First token | Writing 128 (tokens/s) | Writing 1,024 (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1K (1,067) | 88 | 12.1 s | 11.8 | 14.4 |
| 4K (4,212) | 301 | 14.0 s | 12.5 | 15.1 |
| 16K (14,833) | 716 | 21.1 s | 12.4 | 15.8 |
| 32K (30,660) | 929 | 33.5 s | 12.5 | 16.1 |
| 64K (63,685) | 977 | 65.3 s | 13.4 | 16.5 |
| 128K (123,616) | 943 | 131.9 s | 12.4 | 15.8 |
| 256K (231,917) | 1,036 | 225.1 s | 12.5 | 16.5 |

#### DeepSeek, 32 GB cap

| Prompt | Reading (tokens/s) | First token | Writing 128 (tokens/s) | Writing 1,024 (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1K (1,067) | 74 | 14.3 s | 10.3 | 12.2 |
| 4K (4,212) | 264 | 15.9 s | 11.0 | 13.2 |
| 16K (14,833) | 670 | 22.3 s | 11.4 | 14.5 |
| 32K (30,660) | 871 | 35.4 s | 11.4 | 14.4 |
| 64K (63,685) | 956 | 67.0 s | 12.0 | 14.8 |
| 128K (123,616) | 829 | 150.0 s | 12.0 | 14.6 |
| 256K (231,917) | 1,024 | 227.7 s | 12.6 | 14.8 |

#### GLM, all in RAM

| Prompt | Reading (tokens/s) | First token | Writing 128 (tokens/s) | Writing 1,024 (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1K (1,076) | 287 | 3.9 s | 24.5 | 26.7 |
| 4K (4,176) | 843 | 5.1 s | 25.9 | 26.0 |
| 16K (14,724) | 991 | 15.0 s | 25.6 | 25.6 |
| 32K (30,537) | 978 | 31.4 s | 24.6 | 25.6 |
| 64K (63,822) | 951 | 69.8 s | 25.2 | 25.3 |
| 128K (123,356) | 902 | 149.5 s | 24.0 | 23.8 |
| 256K (231,868) | 818 | 296.7 s | 22.8 | 22.8 |

#### GLM, 48 GB cap

| Prompt | Reading (tokens/s) | First token | Writing 128 (tokens/s) | Writing 1,024 (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1K (1,075) | 117 | 9.3 s | 19.3 | 19.8 (721 written) |
| 4K (4,177) | 447 | 9.5 s | 19.4 | 20.5 |
| 16K (14,725) | 940 | 15.8 s | 17.7 | 20.0 |
| 32K (30,536) | 936 | 32.8 s | 17.9 | 19.0 |
| 64K (63,822) | 920 | 69.7 s | 18.0 | 19.0 |
| 128K (123,356) | 882 | 140.6 s | 16.6 | 18.7 |
| 256K (231,868) | 801 | 299.0 s | 16.1 | 15.6 (41 written) |

#### GLM, 32 GB cap

| Prompt | Reading (tokens/s) | First token | Writing 128 (tokens/s) | Writing 1,024 (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1K (1,075) | 91 | 11.9 s | 16.6 | 16.3 (824 written) |
| 4K (4,176) | 357 | 11.8 s | 15.3 | 16.6 |
| 16K (14,725) | 859 | 17.3 s | 15.8 | 15.5 |
| 32K (30,536) | 867 | 36.4 s | 15.6 | 15.6 |
| 64K (63,820) | 856 | 74.9 s | 15.1 | 15.9 |
| 128K (123,354) | 841 | 147.3 s | 14.2 | 15.1 |
| 256K (231,867) | 775 | 301.4 s | 11.9 | 14.2 |

#### Qwen, all in RAM

| Prompt | Reading (tokens/s) | First token | Writing 128 (tokens/s) | Writing 1,024 (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1K (1,077) | 722 | 1.5 s | 80.3 | 84.1 |
| 4K (4,121) | 2,521 | 1.7 s | 85.3 | 89.4 |
| 16K (14,852) | 3,135 | 4.8 s | 82.5 | 83.1 |
| 32K (30,625) | 3,348 | 9.2 s | 85.4 | 84.5 |
| 64K (65,452) | 3,372 | 19.5 s | 76.1 | 76.7 |
| 128K (125,056) | 3,217 | 39.0 s | 74.0 | 75.9 |
| 256K (231,117) | 2,866 | 80.8 s | 67.6 | 69.6 |

#### Qwen, 32 GB cap

| Prompt | Reading (tokens/s) | First token | Writing 128 (tokens/s) | Writing 1,024 (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1K (1,077) | 400 | 2.7 s | 67.9 | 74.1 |
| 4K (4,120) | 1,426 | 2.9 s | 76.9 | 80.4 |
| 16K (14,852) | 2,545 | 5.9 s | 76.6 | 78.5 |
| 32K (30,625) | 2,604 | 11.8 s | 80.2 | 81.0 |
| 64K (65,453) | 2,895 | 22.7 s | 74.4 | 74.2 |
| 128K (125,056) | 2,890 | 43.4 s | 71.7 | 69.9 |
| 256K (231,117) | 2,555 | 90.6 s | 67.4 | 65.9 |

#### Qwen, 16 GB cap

| Prompt | Reading (tokens/s) | First token | Writing 128 (tokens/s) | Writing 1,024 (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1K (1,077) | 147 | 7.3 s | 52.3 | 55.9 |
| 4K (4,121) | 528 | 7.8 s | 70.4 | 55.9 |
| 16K (14,852) | 1,496 | 10.0 s | 59.2 | 63.2 |
| 32K (30,625) | 2,019 | 15.3 s | 67.5 | 67.4 |
| 64K (65,453) | 2,266 | 29.0 s | 57.6 | 58.1 |
| 128K (125,055) | 2,261 | 55.4 s | 59.2 | 56.6 |
| 256K (231,116) | 2,017 | 114.8 s | 58.9 | 58.1 |

#### Nemotron

| Prompt | Reading (tokens/s) | First token | Writing 128 (tokens/s) | Writing 1,024 (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1K (1,045) | 3,294 | 0.3 s | 117.3 | 135.1 (901 written) |
| 4K (4,041) | 4,899 | 0.8 s | 103.8 | 121.6 (943 written) |
| 16K (14,693) | 5,002 | 3.0 s | 128.3 | 134.2 |
| 32K (30,032) | 4,815 | 6.3 s | 107.3 | 129.5 |
| 64K (64,665) | 4,370 | 14.9 s | 123.6 | 120.8 |
| 128K (122,962) | 3,699 | 33.4 s | 119.2 | 115.4 |
| 256K (226,804) | 2,895 | 79.2 s | 103.7 | 104.5 |

#### Flint

| Prompt | Reading (tokens/s) | First token | Writing 128 (tokens/s) | Writing 1,024 (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1K (1,067) | 4,583 | 0.2 s | 219.4 | 214.5 |
| 4K (4,210) | 6,792 | 0.6 s | 209.2 | 208.8 |
| 16K (14,798) | 6,535 | 2.3 s | 200.7 | 190.6 |
| 32K (30,582) | 5,738 | 5.4 s | 167.0 | 169.2 |
| 64K (63,613) | 4,549 | 14.1 s | 137.4 | 139.1 |
| 128K (123,235) | 3,296 | 37.6 s | 116.1 | 108.6 |
| 256K (233,206) | 2,198 | 106.5 s | 75.6 | 75.1 |

- **DeepSeek's writing hardly moves with the prompt's length:** what limits it is how many experts per word come from
  the SSD, not the context. Its prompts start slowly because every chunk streams the experts the card does not hold,
  about 110 GB, so a short prompt pays for one whole pass.
- **GLM writes faster than DeepSeek at every RAM size** (it leaves out light experts on every layer, and most of what
  it needs fits on the card and in RAM), and reads long prompts a little slower: its 34 KDA and 11 DSA layers cost more
  per token than DeepSeek's attention. With a cap its short prompts slow down like Qwen's, while from 16K on it
  loses little.
- **With less RAM,** Qwen's short prompts slow down most (the experts they need come from the SSD); long prompts share
  each pass over the experts among more tokens and lose less.

## DeepSeek-V4.1-Flash

DeepSeek-V4.1-Flash-1017 has 40 layers with 384 experts each, 6 used per token, plus two Engram layers (a lookup
table of 111 GB that each token reads 48 rows of) and a vision tower. The 2-bit GGUF's routed experts are 132 GB, about
8.6 MB an expert.

- **Where it lives:** the card holds the dense weights (5.9 GB), about 2,250 of the 15,360 experts and the working
  buffers. A RAM store (`--ssd-cache`) keeps the experts the conversation uses, read with unbuffered reads, and the card
  reads them in place over PCIe. Everything else is read from the SSD as it is needed. The engine needs about 9 GB of
  RAM beyond its store while loading.
- **Long context:** prompts of 64K-256K tokens with one planted sentence ("the vault access code for the archive room is
  MARIGOLD-7319") and a question about it at the end: each answered `MARIGOLD-7319`. Earlier builds, with a 72 GB store,
  also found such a code at 512K and at 1,048,576 tokens (`--max-context 1048576`, 30 minutes of reading).
- **Leaving out cold experts** (`STRATA_DS4_DISK_SKIP_W`, 0.20 here): an expert that is in neither VRAM nor RAM and
  whose routing weight is under 0.20 is left out of that token instead of read from the SSD, about 7% of the experts.
  With a 32 GB store, writing goes from about 13 to 18-19 tokens/s. Against the full model, the same next token 93.6%
  of the time (KL divergence 0.074); two runs of the full model agree 96.6% of the time (0.081), and a setting of 0.35
  drops to 90.0% (0.22). Unset, nothing is left out.
- **Why writing is SSD-bound:** each word uses about 240 experts. With little RAM, several of them a word are neither on
  the card nor in RAM and were not guessed early enough; the SSD gives about 5.4 GB/s for reads of this size, 2 ms an
  expert, and the next layer cannot start until they are in.
- **Pictures** go through Strata's server like the Qwen models'. The vision tower comes from DeepSeek's own checkpoint
  (shard 1 of 48, 0.97 GB, see [Build](#build)); the server resizes each picture as DeepSeek's `image_processor.py`
  does and the engine encodes it.

## GLM-5.3-Flash

GLM-5.3-Flash has 45 layers: 34 KDA layers (a recurrent attention with a fixed-size state) and 11 DSA layers (attention
over the 2,048 earlier positions its indexer picks), 64 heads, and 42 expert layers of 288 experts, 8 used per token,
beside a shared expert. The Maya-M-Derisked GGUF keeps its experts in 2-3 bits (97 GB, about 8 MB an expert) and the
rest in Q6_K (about 7 GB on the card).

**What this repo changes,** each measured on this desktop with fresh prompts through the server (all in RAM):

| Change | Chat: first token | 4K prompt | 16K prompt |
| --- | ---: | ---: | ---: |
| Maya's engine as it is | 7.0 s | 213 tok/s | 221 tok/s |
| FP32 products as two BF16 parts | 1.0 s | 757 | 798 |
| DSA attention: two head tiles a block, one FP16 pass | 1.0 s | 810 | 960 |
| Prompt experts: the fused kernel, gate/up in FP8 | 1.0 s | 860 | 1,015 |

- **FP32 products:** on this ROCm (10.2.0a20261009) an FP32 matrix product (rocBLAS sgemm, or GemmEx with FP32 operands)
  takes about 18 ms a call on gfx1201 whatever its size; the same product in BF16 takes 0.02-0.06 ms. GLM's prompt path
  ran its KDA projections, stream mixing, indexer and router through it. Each FP32 input is now split into two BF16
  parts (`STRATA_GLM_SGEMM_F32=1` keeps the FP32 product). DeepSeek's prompt path had one such call (its stream mixing):
  a short chat's first token there went from about 2.9 to 1.4 s (`DS4_HC_MIX_F32=1` keeps it).
- **DSA attention** (`STRATA_GLM_MLA_V3=0` and `STRATA_GLM_MLA_SPLIT=1` restore the original): each block loads a tile
  of stored keys once for 32 heads instead of 16, reads the next tile while it computes, and does each product once in
  FP16 instead of twice (high part + residual).
- **Prompt experts** (`STRATA_GLM_PREFILL_FUSED=0`, `STRATA_GLM_PREFILL_F8=0`): Maya's fused expert kernel had its RDNA4
  code but was only switched on for RDNA3; here it runs on the R9700 with its gate/up product on the FP8 matrix cores,
  as Qwen's `STRATA_PF_FUSED_F8`.
- **Output:** every change above was checked by teacher-forcing 128 tokens after a 1,920-token prompt
  (`STRATA_GLM_KL_DUMP`) against the unchanged engine: KL divergence 0.011-0.015, about the same as two runs of the unchanged
  engine (0.010) - the engine is not bit-for-bit repeatable run to run.
- **The server:** GLM's tokenizer (digits in threes, whole-vocabulary pieces kept; the same ids as llama.cpp on a
  30,000-character document), its chat template and thinking levels (low, high, max, or off), its tool calls, and its
  pictures (`<|begin_of_image|>` markers; the encoder gives GLM's picture tokens as one row, not a grid).

**Leaving out light experts** (`STRATA_GLM_MISS_SKIP_W`, `STRATA_GLM_MISS_SKIP_FROM`): an expert that is not on the card
and whose routing weight is under the setting is left out of that token instead of fetched from RAM or the SSD.
Writing 400 tokens, all in RAM, against the unchanged model (the KL test above; two runs of it give 0.010):

| Setting | Writing | Same next token | KL | Next-token loss |
| --- | ---: | ---: | ---: | ---: |
| off | 17.5 tok/s | 92% | 0.010 | 2.87 |
| 0.12 from layer 23 | 22.0 | 85% | 0.020 | 2.92 |
| 0.10 from layer 0 | 25.1 | 84% | 0.035 | 2.91 |
| **0.12 from layer 0** (the tables above) | **28.2** | 85% | 0.048 | 2.94 |
| 0.16 from layer 0 | 31.7 | 73% | 0.177 | 3.08 |

- **Why writing stops there:** every word reads all of GLM's always-used weights, about 6.9 GB (the attention layers in
  Q6_K), about 27 ms of the card's own work; the rest is experts copied from RAM. On this desktop the R9700 runs at
  PCIe 5.0 x8 (the second graphics card shares the processor's lanes), about 25 GB/s.
- **With a RAM cap** the engine's own sizing is stopped by the system while loading: set the expert store
  (`STRATA_GLM_RAM_GB`, with `STRATA_GLM_RAM_HEADROOM_GB=2`). The engine and the server need about 6.5 GB beside it, so
  a 48 GB cap takes a 36 GB store and a 32 GB cap a 24 GB one. The 32 GB setup read a 256K-token prompt at 782 tok/s.
- **Tried, no gain here:** the model's MTP drafts (forced, writing fell to 16 and 8 tok/s: each draft needs more
  experts over PCIe), prefetching the next layer's experts, more CPU threads, int8 stored keys, and the dense products
  in FP8 (`STRATA_GLM_PF_FP8=1`: the weights' conversion each batch costs what the faster product saves).

## Qwen3.8-Flash-Next

What the R9700 work adds, against Strata 0.2.4 with only its own switches, through `strata_bench.py`, 131,072-token
context (a separate run from the tables above, same desktop):

| Setup | | Chat writing | 4K prompt | 16K prompt | 32K prompt |
| --- | --- | ---: | ---: | ---: | ---: |
| All experts in RAM | 0.2.4 | 77.9 | 1,978 | 2,920 | 2,998 |
| | this repo | 82.2 | 2,414 | 3,302 | 3,460 |
| 32 GB cap, 23 GB of experts | 0.2.4 | does not load | | | |
| | 0.2.4 + the lazy conversation memory | 72.2 | 1,263 | 1,911 | 1,991 |
| | this repo | 70.2 | 2,073 | 2,887 | 2,871 |
| 16 GB cap, 7 GB of experts | 0.2.4 + the lazy conversation memory | 49.0 | 561 | 1,427 | 1,702 |
| | this repo | 53.6 | 595 | 1,807 | 2,186 |

- **Under a 32 GB cap, 0.2.4 is stopped by the system while loading:** it pins the conversation memory for the whole
  context at start. Here it is reserved, and pinned as a conversation grows (`STRATA_KV_HOST_LAZY`, on by default).
- **The switches** (all opt-in, set in the configs below): `STRATA_PF_FUSED_F8` (the prompt's experts on the FP8 matrix
  cores), `STRATA_PF_FP8` (its dense products in FP8), `STRATA_PF_DM16` (the experts' outputs in BF16),
  `STRATA_REFILL_ASYNC` (the card's experts come back while the answer starts), `STRATA_BUDGET_LEND_GIB` and
  `STRATA_REFILL_BATCH` (with a cap: a long prompt's borrowed experts come back from RAM, in batches),
  `STRATA_WARMUP_TOKENS` (one throwaway prompt before the server says it is ready).
- **Output:** the FP8 and BF16 switches change the numbers. On the older Strata, 7 questions about a 20,000-token
  document answered greedily kept 6 answers word for word and reworded one summary.
- **Left out** from strata-patches: the RTX 5080 laptop's low-RAM mode (0.2.4 has its own direct and overlapped reads)
  and the picture-detail ceiling.

## Nemotron 3.5 Bolt

- **Quality:** next-token perplexity 4.5656 on 8K tokens of the model's card and 7.5771 on 8K tokens of Strata's
  `docs/DETAILS.md`. The model card scores 4.5612 with every product kept in 16-bit, about 0.1% better.
- **Same answer with or without the drafter:** with the drafter, a worked-arithmetic answer is word for word the
  same as plain greedy writing.
- **Why reading slows down with length:** 80 of the model's 88 layers cost the same for every token. The 8 attention
  layers compare each new token with every one before it, so at 256K they take about half the reading time.

## Recursive Flint

- **Card memory at its peak:** 30.1-31.4 GB of the card's 32 GB (31.9 GB usable), or 28.4-29.7 GB with 8-bit keys.
  Nothing spills into system memory. The program itself holds 1.5 GB of RAM, mostly the 1.2 GB token table, which
  stays in RAM on purpose.
- **8-bit keys** (`FLINT_K8=1`, off by default) store the long-history layers' keys in 8 bits as well as their
  values. Writing at long context gets faster and 1.8 GB of card memory is freed, but the accuracy loss doubles (below).
- **Accuracy,** against Flint computed exactly in 64-bit floats (`tools/flint/ref_all.py`), over 512 positions of
  real text: KL divergence 0.008 with the default 8-bit values, 0.010 with a 16-bit cache, 0.018 with 8-bit keys too.
  The engine picks the same top token as the exact model 95.3% of the time (94.3% with 8-bit keys). For scale,
  rounding the layers' inputs to 8 bits, as llama.cpp does, costs 0.068 on a 300-token text.
- **Guessing ahead** (on in the server): when the text repeats something earlier in the conversation, the server
  guesses the next 1-3 tokens and checks them in one step, with the same output as plain greedy writing. On fresh
  text it guesses only when that pays. `FLINT_SPEC=0` turns it off.
- **Why writing slows down with length:** 14 of the model's 49 layers compare each new token with every token before
  it, and Flint spreads that attention widely, so every word reads the whole stored history: 1.9-2.8 GB at 128K. The
  engine reads it at about 87% of the card's memory bandwidth.

## Ling-3.0-flash

Not measured again here (its model was not on hand); the notes are from strata-nemo.

Ling-3.0-flash-VL is 124B parameters with 5.5B used per token: 42 layers, 35 of them a linear "delta" attention with a
running state and 7 of them full attention over a compressed history, and 512 experts a layer, 8 used per token. This
program reads text only; the picture reader is not wired in.

- **What sits where:** the card holds the attention, the shared expert, the output layer and 23 GB of the routed
  experts (26.0 GB in all). The rest of the experts stay in the GGUF, read through the file mapping, so it wants
  about 67 GB of free RAM to keep them out of the disk. Each layer's experts on the card and on the processor run at
  the same time, and the card adds the processor's share when it is ready.
- **The card's experts follow the conversation.** Which experts a text uses depends on the kind of text. A set picked
  from code and documentation covered 87% of what that text used, but only 37% of what everyday chat used; a set
  picked from half of the chat covered 79% of the other half. So the engine counts each layer's routing as it goes,
  and an expert used more than the layer's least-used one on the card takes its place: up to 2 after each written
  token, and up to 64 a layer for each prompt chunk, copied from where the prompt path already put them on the card.
  Over 24 chat questions, writing went from about 44 to about 70 tokens/s with it. `LING_ADAPT=0` keeps the starting
  set fixed.
- **Command line:** about 79 tokens/s writing after a 1,500-token code prompt, read at about 300 tokens/s. Each prompt
  chunk copies the experts it uses that the card lacks, which costs much the same for a short chunk as a long one, so
  longer prompts read faster (1,055 tokens/s at 4K through the server).
- **Accuracy,** against Ling computed exactly in 64-bit floats (`tools/ling/ref_ling.py`, dequantizing the same
  GGUF): KL divergence about 1e-11 on the writing path and 3e-8 on the prompt path, the same top token at every
  position. The prompt path's matrix-core products split each input into two 16-bit halves, so they stay this close.
- **Context:** 131,072 tokens, the model's own limit (its GGUF says so, and it carries no setting for stretching it
  further). All working memory is set aside at start: at 131,072 tokens 0.75 GB of the card stays free, at 65,536
  (the benchmark above) 1.8 GB.
- **Conversation reuse:** the delta layers' running state cannot be rewound to an earlier token, so the engine saves
  it one token before the end of each prompt (`LING_SNAPS`, 4 kept). A turn that extends the last one goes on from
  where it stopped; otherwise from the latest saved state inside the shared part.

## Build

Build against AMD TheRock ROCm 10 (`pip install "rocm[libraries,devel,device-gfx1201]"` from
`https://nightly.repo.amd.com/rocm/whl-next/`; the SDK folder is its `_rocm_sdk_devel`):

```sh
tools/r9700/build-r9700.sh <ROCm 10 SDK dir>    # writes engine/strata-r9700 (Qwen)
tools/r9700/build-dsv41.sh <ROCm 10 SDK dir>    # writes engine/strata-dsv41
tools/r9700/build-glm.sh <ROCm 10 SDK dir>      # writes engine/strata-glm
tools/r9700/build-vision-glm.sh <ROCm 10 SDK dir> <llama.cpp copy>   # GLM's picture encoder: build-vision-glm/bin/strata-vision-hip
tools/r9700/build-nemo.sh <ROCm 10 SDK dir>     # writes engine/strata-nemo
tools/r9700/build-flint.sh <ROCm 10 SDK dir>    # writes engine/strata-flint
tools/r9700/build-ling.sh <ROCm 10 SDK dir>     # writes engine/strata-ling
tools/cuda/build-flint.sh                       # Flint for NVIDIA: writes engine/strata-flint-cuda
```

**DeepSeek's pack:** the engine reads a folder that lists the GGUFs' tensors and adds the vision tower and the
Engram tables. Download the four `GSQ-RCO-IQ2_XXS` shards and `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` from
ISTA-DASLab, the first shard of Unsloth's `UD-Q8_K_XL` set (5.8 MB: the Engram hash settings) and, for pictures,
`model-00001-of-00048.safetensors` and `config.json` from DeepSeek's checkpoint into `<model dir>/vision`. Then:

```sh
python3 tools/dsv41/vision_pack.py --dir <model dir>/vision            # the vision tower as vision.bin
python3 tools/dsv41/pack_gguf.py --model <model dir> --shards <model dir>/*GSQ-RCO-IQ2_XXS-0000?-of-00004.gguf \
    --engram <model dir>/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf \
    --engram-meta <model dir>/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf \
    --vision <model dir>/vision --out <pack dir>
cp data/dsv41/profile-2.00.bin <pack dir>/expert-profile.bin           # which experts start on the card and in RAM
```

`tools/dsv41/download_gguf.json` lists every file with its size and checksum.

**GLM's pack:** download the three `GLM-5.3-Flash-Maya-M-Derisked-IQ2_S` shards into one folder and, for pictures,
`vision/mmproj-GLM-5.3-Flash-F16.gguf` and `vision/GLM-5.3-Flash-vocab.gguf` from the Maya release. The engine finds the
GGUFs at `<pack>/..`, so the pack goes beside them (llama.cpp's `gguf-py` from the checkout at the pinned commit):

```sh
STRATA_GGUF_PY=<llama.cpp>/gguf-py python3 tools/glm/iq_pack.py --compat-bf16 \
    --gguf <model dir>/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-00001-of-00003.gguf --out <model dir>/pack
```

## Run from the command line

The programs read and write token ids, not text. Turn a prompt into ids with the model's tokenizer (apply the chat
template yourself for a chat), for example:

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("<model dir>/tokenizer.json")
open("prompt_ids.txt", "w").write(",".join(map(str, tok.encode(text, add_special_tokens=False).ids)))
```

Then:

```sh
# Nemotron: first write the tensor list the model and the drafter each need
python3 tools/nemotron/manifest.py <model dir>
python3 tools/nemotron/manifest.py <drafter dir>
engine/strata-nemo --model <model dir> --ids @prompt_ids.txt --new 256 --max-ctx 262144
engine/strata-nemo --model <model dir> --ids @prompt_ids.txt --new 256 --max-ctx 262144 --mode spec --draft <drafter dir>
engine/strata-nemo --model <model dir> --ids @text_ids.txt --ppl --max-ctx 9000      # perplexity over a text

# DeepSeek (the settings of the server config below)
DS4_STORE_ZC=1 DS4_STORE_ZC_FRAC=75 DS4_AHEAD=1 STRATA_DS4_DISK_SKIP_W=0.20 DS4_CHUNK=24576 \
    engine/strata-dsv41 --pack <pack dir> --ids @prompt_ids.txt --n 256 --max-context 262144 --eos 1 \
    --cpu-threads 12 --ssd-cache 32

# Flint
engine/strata-flint --model flint-int4-q4_1.gguf --ids @prompt_ids.txt --new 256 --max-ctx 262144

# Ling: LING_PLAN names the experts the card starts with (below)
LING_PLAN=plan-23gb.txt engine/strata-ling --model Ling-3.0-flash-VL-Q4_0-00001-of-00002.gguf \
    --ids @prompt_ids.txt --new 256 --max-ctx 131072
```

The written token ids go to standard output, the speeds to standard error. `--ids` also takes the ids inline
(`--ids 1784,8961,1307`).

**Expert plans** (Ling, and Flint on a smaller card): run some text with `LING_ROUTE_DUMP=<file>` (Flint:
`FLINT_ROUTE_DUMP`) to record the routing, then pick the most used experts up to a size:

```sh
python3 tools/flint/make_plan.py plan-23gb.txt routes1.bin routes2.bin --model ling --experts 23GB
```

Without a plan, Ling starts with no experts on the card and the processor computes them all.

## Run through the server

Write the model's tokenizer folder, then a config for it:

```sh
python3 tools/nemotron/export_tokenizer.py <model dir> <tokenizer dir>                 # Nemotron
python3 tools/strata_tokenizer.py --gguf flint-int4-q4_1.gguf --out <dir>                # Flint: <dir>/tokenizer
python3 tools/strata_tokenizer.py --gguf Ling-3.0-flash-VL-Q4_0-00001-of-00002.gguf --out <dir>   # Ling
python -m serve.server --engine strata --config strata-ling.json --port 8080
```

```json
{
  "exe": "<repo>/engine/strata-ling",
  "args": ["--model", "<model dir>/Ling-3.0-flash-VL-Q4_0-00001-of-00002.gguf", "--max-context", "131072"],
  "tokenizer": "<dir>/tokenizer",
  "model_family": "ling",
  "model_name": "ling-3.0-flash",
  "lib_dirs": ["<ROCm 10 SDK dir>/lib", "<ROCm 10 SDK dir>/lib/rocm_sysdeps/lib", "<ROCm 10 SDK dir>/lib/llvm/lib"],
  "port": 8080,
  "backend": "hip",
  "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
  "env": {"LING_PLAN": "<model dir>/plan-23gb.txt"}
}
```

What changes for the other two:

| Field | Nemotron | Flint |
| --- | --- | --- |
| `exe` | `engine/strata-nemo` | `engine/strata-flint` (`strata-flint-cuda` on NVIDIA, with `"backend": "cuda"`) |
| `args` | `--model <model dir> --draft <drafter dir> --max-context 262144` | `--model <dir>/flint-int4-q4_1.gguf --max-context 262144` |
| `model_family` | `nemotron` | `flint` |
| `sampling` | temperature 1.0, top_p 0.95 | temperature 1.0, top_p 0.95, top_k 20 |
| `env` | | on a smaller card: `FLINT_PLAN` |

DeepSeek's config names its pack instead of a model family (its tokenizer and chat template are in the pack):

```json
{
  "exe": "<repo>/engine/strata-dsv41",
  "args": ["--pack", "<pack dir>", "--max-context", "262144", "--eos", "1", "--dspark", "auto", "--cpu-threads", "12",
           "--ssd-cache", "32"],
  "tokenizer": "<pack dir>/tokenizer",
  "arch": "deepseek41",
  "model_name": "deepseek-v4.1-flash",
  "lib_dirs": ["<ROCm 10 SDK dir>/lib", "<ROCm 10 SDK dir>/lib/rocm_sysdeps/lib", "<ROCm 10 SDK dir>/lib/llvm/lib"],
  "port": 8080,
  "backend": "hip",
  "engine_ready_s": 3600,
  "env": {"DS4_STORE_ZC": "1", "DS4_STORE_ZC_FRAC": "75", "DS4_AHEAD": "1", "STRATA_DS4_DISK_SKIP_W": "0.20",
          "DS4_CHUNK": "24576"}
}
```

- `--ssd-cache 32`: the RAM store in GB (the tables above: 72, 38 and 21). The engine needs about 9 GB more; under a
  RAM cap, keep the store that far below it.
- `DS4_STORE_ZC=1`, `DS4_STORE_ZC_FRAC=75`: the card reads the RAM store's experts in place over PCIe.
- `DS4_AHEAD=1`: each layer guesses the next one's experts and starts reading them early.
- `DS4_CHUNK=24576`: prompt chunks of 24,576 tokens (the default on a 28 GB card is 12,288). Long prompts read up to
  twice as fast; the larger buffers keep about 3 GB fewer experts on the card.

GLM's config names `"arch": "glm5"`, which picks its tokenizer rules, chat template, tool calls and thinking words; its
tokenizer is in the pack. The 48 GB and 32 GB setups add `"STRATA_GLM_RAM_GB": "36"` (or `"24"`) and
`"STRATA_GLM_RAM_HEADROOM_GB": "2"` to `env` and run the server under that cap (`systemd-run -p MemoryMax=48G ...`):

```json
{
  "exe": "<repo>/engine/strata-glm",
  "args": ["--glm-pack", "<model dir>/pack", "--max-context", "262144", "--expert-cache", "auto", "--prefill", "auto",
           "--kv-streaming", "on"],
  "tokenizer": "<model dir>/pack/tokenizer",
  "arch": "glm5",
  "model_family": "glm5",
  "model_name": "glm-5.3-flash",
  "reasoning_effort": "high",
  "lib_dirs": ["<ROCm 10 SDK dir>/lib", "<ROCm 10 SDK dir>/lib/rocm_sysdeps/lib", "<ROCm 10 SDK dir>/lib/llvm/lib"],
  "port": 8080,
  "backend": "hip",
  "env": {"STRATA_GLM_RESERVE_MB": "3072", "STRATA_GLM_RAM_HEADROOM_GB": "16", "STRATA_GLM_SPLIT": "0",
          "STRATA_GLM_PREFILL_SUB": "1024", "STRATA_GLM_PREFILL_MB": "8192",
          "STRATA_GLM_MISS_SKIP_W": "0.12", "STRATA_GLM_MISS_SKIP_FROM": "0"},
  "vision": {"exe": "<repo>/build-vision-glm/bin/strata-vision-hip", "mmproj": "<model dir>/vision/mmproj-GLM-5.3-Flash-F16.gguf",
             "model": "<model dir>/vision/GLM-5.3-Flash-vocab.gguf", "gpu": true}
}
```

Qwen's config, from Strata's own pack and MTP folder (see [README-STRATA.md](README-STRATA.md)); the *all in RAM*
setup above:

```json
{
  "exe": "<repo>/engine/strata-r9700",
  "args": ["--pack", "<pack dir>", "--native", "<model dir>/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf",
           "--ple-gguf", "<model dir>/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf",
           "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto", "--prefill", "auto:32768",
           "--spec", "4", "--spec-min-p", "0.5", "--mtp", "<mtp dir>", "--max-context", "262144", "--kv", "int8",
           "--kv-resident", "32768", "--vram-reserve-mib", "700"],
  "tokenizer": "<pack dir>/tokenizer",
  "port": 8080,
  "backend": "hip",
  "env": {"STRATA_HIPBLASLT_TUNING": "tools/hip/gfx1201-hipblaslt-100500.txt", "STRATA_HIP_WMMA": "1",
          "STRATA_SELECT_WMMA": "1", "STRATA_PF_FUSED": "1", "STRATA_ROUTE_RESIDENT": "1", "STRATA_UNBUFFERED_LOAD": "1",
          "HSA_USERPTR_FOR_PAGED_MEM": "0", "STRATA_PF_FP8": "1", "STRATA_PF_FUSED_F8": "1", "STRATA_PF_DM16": "1",
          "STRATA_REFILL_ASYNC": "1", "STRATA_WARMUP_TOKENS": "4096"}
}
```

With a RAM cap, add `"--mmap-experts", "--resident-budget-gib", "20"` (16 GB: `"5"`), `"--prompt-cache", "2",
"--prompt-cache-every", "0"` and `STRATA_PROMPT_DIRECT=1`, `STRATA_BUDGET_LEND_GIB=3` (16 GB: `1`),
`STRATA_REFILL_BATCH=32`, and start the server under the cap (`systemd-run --user -p MemoryMax=32G ...`). Leave room for
the conversation memory: 20 GB of experts under 32 GB holds a 256K conversation, 23 GB does not.

## How it works

**DeepSeek** (what this repo adds to Strata 0.2.4's engine):
- **Prompts on the matrix cores:** the attention and the sparse attention's indexer (its full scoring, 16 queries a
  block) run on gfx12's BF16 matrix cores; each chunk's dense weights are unpacked once to FP16.
- **The RAM store refills itself:** after a prompt it reads back its hottest experts in the background, oldest last,
  and waits whenever a word's own reads are queued. Experts that move to the card after a prompt are copied from the
  store instead of read again from the SSD.
- **Early reads only from the disk:** a layer's guessed experts are read ahead only when they come from the SSD; RAM
  ones are a copy once the layer has routed.
- **Fewer launches:** one launch rounds a layer's inputs to 8 bits for all its products, and the matrix library's
  kernels are loaded at start (their first call used to stall the first prompt for 2.6 s).
- **Fix:** a background refill could reuse a store slot a prompt chunk had looked up, which crashed the card
  (illegal memory access); the refill now stops before a chunked prompt.

**Nemotron:**
- **Everything on the card:** every weight stays in its stored format (NVFP4 state-update projections, IQ2_XXS
  experts, Q8_0 attention and output layer).
- **Reading** goes in chunks of 4,096 tokens: the state-update projections and the shared expert as 8-bit (FP8)
  matrix products, the routed experts unpacked straight into the matrix cores' registers, the state-update scan and
  the attention on the 16-bit matrix cores.
- **Writing** replays each step as a recorded graph. One token at a time, the chosen experts are read in one launch
  and multiplied in FP16 pairs, the router picks them in one launch, and the shared expert runs beside them on a
  second stream. The drafter's checks group the tokens' picks by expert, so each expert is read once.
- **The drafter** guesses 7 tokens a round. The engine checks only as many as recent rounds suggest will pay off, so
  hard text costs fewer wasted checks and easy text still checks all 7.

**Flint and Ling, with experts in RAM:**
- **One graph a token.** The card writes a layer's expert input and choices into memory the processor can see and
  raises a flag. The processor's workers (one pinned to each physical core) compute the experts the card does not
  hold, and a card kernel waits for their flag and adds the result. No call from the host is needed in the middle, so
  the whole step stays one recorded graph.
- **Prompts** copy the experts each chunk uses to the card and run them there as grouped matrix products.

**Ling:**
- **Weights are rearranged at load** into forms the card reads in whole 16- or 32-byte pieces (the 4-bit codes
  first, then the scales). Writing reads one 32-value block of a row per thread.
- **Full-attention layers** keep the compressed history (512 + 64 values a token) and read it once for all 32 heads
  per token.

**Switches:** most newer paths have an environment variable that turns the older one back on. The commit messages
name them.

## Limits

- The programs take and give token ids. The tokenizer, chat template and API are Strata's Python server (`serve/`).
- Nemotron's command line writes greedily only.
- DeepSeek's speed depends on the SSD and on how much RAM its store gets; a prompt costs at least one pass over the
  experts the card does not hold. Its leave-out setting trades some accuracy for speed; set it to 0 for the full model.
- DeepSeek, Qwen's R9700 switches, Nemotron and Ling are built and tested for the R9700 (gfx1201) only. Flint also
  builds for NVIDIA cards (`tools/cuda/build-flint.sh`).
- Ling reads text only, up to 131,072 tokens.

## License

The code is under Strata's MIT license (`LICENSE`). The models are not included; their own licenses apply.
