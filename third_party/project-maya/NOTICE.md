# Project Maya (maya-ds4), DeepSeek-V4.1 engine

The files under `src/dsv41/`, `include/strata/dsv41/`, `tests/dsv41/` and `tools/dsv41/` come from
https://github.com/mw00/project-maya (release v1.1.17, commit 8d9a9b5d), its DeepSeek engine `src/ds4`, `tests/ds4_*`,
`tools/ds4` and the kernel sources it builds on (`exl3`, `sampler`). Project Maya is itself built on Strata.

Copyright (c) 2026 Peasant Smith (mw00) and the Project Maya contributors, including needmorevram.
MIT License, the full text is in `LICENSE` beside this file. The notice stays with every copy.

Changes made while porting into Strata are listed in the git history of the `src/dsv41` directory
(the first commit of each file is the unmodified upstream copy).

Later changes taken from Project Maya releases v1.1.18 to v1.1.26 (the same approach re-written onto Strata's engine,
for example the long-prompt layer skip, the thinking budget, conversation slots) are covered by the same copyright
and permission notice, in `LICENSE` beside this file.

## Project Maya's GLM-5.3-Flash engine (strata-glm)

The files under `src/glm/` come from the same repository at release v1.1.28 (commit 6afa525b): its GLM-5.3-Flash
engine (`src` without `ds4`, `include`, `cmake`, `third_party/ggml`, its `CMakeLists.txt` and the GLM hipBLASLt tables
and tuner from `tools/hip`), and `serve/glm5_chat_template.jinja` from its `serve`. Same copyright and permission
notice, in `LICENSE` beside this file and in `src/glm/LICENSE`. The first commit of `src/glm` is the unmodified copy;
the changes made to build it inside Strata are in the git history after it.
