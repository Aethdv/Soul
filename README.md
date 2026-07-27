# <img src="assets/tetosing.png" alt="TetoSing" width="32" height="32" /> Soul

Soul is a modern, hand-crafted evaluation (HCE) alpha-beta chess engine written in Rust.

Rather than relying on black-box Neural Networks, Soul pushes traditional algorithmic evaluation to its
absolute limit with SIMD acceleration, zero-cost abstractions, and a custom tuning framework built from scratch.

It supports the UCI and XBoard protocols, plus full Chess960 (Fischer Random)

## <img src="assets/rocket.gif" alt="Rocket" width="20" height="20" /> Quick Start

You will need `rustup` with the nightly toolchain and `make`.\
The nightly version is specified in `rust-toolchain.toml`\
and will be installed automatically by `rustup` on first build.

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

### Development

```bash
make debug   # Debug build
make test    # Run test suite
make clippy  # Lint with clippy
make format  # Format with rustfmt
```

## <img src="assets/thonk.gif" alt="ThonkTriangle" width="22" height="22" /> Tuning & Datasets

Soul uses a custom binary format (`.soul.zst`), a zstd-compressed nibble-array layout.

### Generating Data

Generate self-play data for evaluation tuning:

```bash
./soul genfens -t <N> -n <N>
```

*Inspect datasets or encode existing EPDs with the built-in utilities (`./soul dataset help`).*

### Evaluation Tuning

Tune the hand-crafted evaluation parameters with the Lion optimizer:

```bash
make evaltune
./eval --dataset data/data.soul.zst --epochs <N>
```

### Search Tuning

Tune search parameters with a separable, active CMA-ES with learning-rate adaptation:

```bash
make searchtune
./search --openings book.epd --epochs <N> --tc "40+0.4"
```

*(Results auto-save to `searchtune_checkpoint.json` and can be resumed with `--resume`.)*

## 🎮 Configuration (UCI / XBoard Options)

Soul exposes the following options via both protocols:

|       Option       | Default |                                   Description                                     |
|--------------------|---------|-----------------------------------------------------------------------------------|
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
- Derivative works must be licensed under AGPL-3.0
- Changes must be documented
- Must include original copyright and license notice

**Limitations:**
- ✗ No warranty
- ✗ No liability

## <img src="assets/tetoheart.png" alt="TetoHeart" width="32" height="32" /> In no particular order, Special Thanks

<img src="assets/gemheartgreen.gif" alt="GemHeartGreen" width="16" height="16" /> [Lofty](https://github.com/yukarichess): the author of [Yukari](https://github.com/yukarichess/Yukari), for being a patient and understanding friend. Also helped me understand XBoard.\
<img src="assets/gemheartpink.gif" alt="GemHeartPink" width="16" height="16" /> [Lily](https://github.com/87flowers): the author of [Rose](https://github.com/87flowers/Rose) and co-author of [Clockwork](https://github.com/official-clockwork/Clockwork), for always getting me. Check her [ByteBoard Attack Tables](<https://87flowers.com/byteboard-attack-tables-1/>) blog!\
<img src="assets/gemheartcyan.gif" alt="GemHeartCyan" width="16" height="16" /> [Styx](https://github.com/styxdoto): the co-author of [Reckless](https://github.com/codedeliveryservice/Reckless), for being a nice friend and helping minimize my loss of sanity.
