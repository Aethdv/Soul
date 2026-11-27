# <img src="assets/tetosing.png" alt="TetoSing" width="32" height="32" /> Soul

Soul is a contemporary alpha-beta chess engine written in Rust with a hand-crafted evaluation (HCE),
and the first engine built on [XorBoard](<https://aethdv.com/posts/xorboard/>).
Supports both the UCI and XBoard/CECP protocols, including full Chess960 (Fischer Random).

Instead of relying on NNUE, Soul pushes classical evaluation as far as it can go using SIMD accumulators
and a custom tuning framework built from scratch.

## Rating

Official ratings from [COPE Chess](<https://cope-chess.live/engines/soul>).\
Ranks shift over time and reflect the lists at the time of writing.\
Self-estimated ratings are calibrated via head-to-head matches against [Stash](<https://gitlab.com/mhouppin/stash-bot>).

| Version | Self-est. |     HCE    |   Rapid    |   Bullet   |    DFRC    |
| :---: | :---: | :---: | :---: | :---: | :---: |
| v0.6.0  | ~3141     | 2967 (#18) | 3169 (#81) | 3018 (#99) | 3008 (#54) |
| v0.5.0  | ~2946     |      -     |      -     |      -     |      -     |
| v0.4.0  | ~2782     |      -     |      -     |      -     |      -     |
| v0.3.0  | ~2490     |      -     |      -     |      -     |      -     |
| v0.2.0  | ~2275     |      -     |      -     |      -     |      -     |
| v0.1.0  | ~1660     |      -     |      -     |      -     |      -     |

## <img src="assets/rocket.gif" alt="Rocket" width="20" height="20" /> Quick Start

You will need `rustup` with the nightly toolchain and `make`.\
The nightly version is specified in `rust-toolchain.toml` and will be installed automatically by `rustup` on first build.

> [!WARNING]  
> The evaluation accumulator is SIMD throughout, so Soul strictly requires an **AVX2**-capable CPU (x86-64-v3 or newer). Older CPUs are not supported.

### Building

```bash
make pgo     # Profile-Guided Optimization (recommended)
make native  # Optimized specifically for your host CPU

make avx512
make avx2-bmi2
make avx2
```

## 🎮 Configuration (UCI / XBoard Options)

Soul supports the following UCI options (with equivalents exposed under XBoard where applicable):

|       Option       | Default |                                   Description                                     |
| :--- | :---: | :--- |
| `Hash`             | 16      | Transposition table size in MB                                                    |
| `Threads`          | 1       | Number of search threads                                                          |
| `Overhead`         | 10      | Move-time overhead buffer in ms                                                   |
| `UCI_ShowCurrMove` | true    | Show the current move being searched                                              |
| `UCI_ShowWDL`      | false   | Show Win/Draw/Loss estimates (Stockfish coefficients; inaccurate on Soul's scale) |
| `UCI_Chess960`     | false   | Enable Fischer Random Chess mode                                                  |

## <img src="assets/tetoplush.png" alt="TetoPlushie" width="22" height="22" /> License

AGPL-3.0-or-later. See [LICENSE](LICENSE) for details.

TL;DR:

**Permissions:**
- ✓ Free to use, study, and modify
- ✓ Can distribute original or modified versions
- ✓ Commercial use allowed

**Conditions:**
- Must provide source code to users who access it over a network
- Derivative works must be licensed under AGPL-3.0-or-later
- Changes must be documented
- Must include original copyright and license notice

**Limitations:**
- ✗ No warranty
- ✗ No liability

## <img src="assets/tetoheart.png" alt="TetoHeart" width="32" height="32" /> Special Thanks

In no particular order.

<img src="assets/gemheartgreen.gif" alt="GemHeartGreen" width="16" height="16" /> [Lofty](https://github.com/yukarichess): the author of [Yukari](https://github.com/yukarichess/Yukari), for being a patient and understanding friend. Also helped me understand XBoard.\
<img src="assets/gemheartpink.gif" alt="GemHeartPink" width="16" height="16" /> [Lily](https://github.com/87flowers): the author of [Rose](https://github.com/87flowers/Rose) and co-author of [Clockwork](https://github.com/official-clockwork/Clockwork), for always getting me. Check her [ByteBoard Attack Tables](<https://87flowers.com/byteboard-attack-tables-1/>) blog!\
<img src="assets/gemheartcyan.gif" alt="GemHeartCyan" width="16" height="16" /> [Styx](https://github.com/styxdoto): the co-author of [Reckless](https://github.com/codedeliveryservice/Reckless), for the help and sanity checks early on.
