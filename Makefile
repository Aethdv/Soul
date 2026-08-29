EXE_NAME := soul
DEPTH    ?= 12

FEATURES ?=
CARGO_FEATURES = $(if $(FEATURES),--features $(FEATURES))

# Evaltune profiling dataset + epoch count.
ET_DATA   ?= data/big3.txt
ET_EPOCHS ?= 100

HAS_PGO := $(shell command -v cargo-pgo 2> /dev/null)
WIN_TARGET := x86_64-pc-windows-gnu
HAS_ZIG := $(shell command -v cargo-zigbuild 2> /dev/null)
HAS_WIN_STD := $(shell rustup target list --installed 2>/dev/null | grep -x $(WIN_TARGET))
COMMA := ,
RUST_HOST := $(shell rustc -vV | sed -n 's/host: //p')

# OpenBench passes CC=cargo (for C/C++ engines). Override it so cc-rs
# (used by evaltune's zstd-sys) finds a real C compiler instead of cargo.
override CC := cc

# If you want to build faster (specify your own number of threads);
# bash: export CARGO_ENCODED_RUSTFLAGS=$'-Ctarget-cpu=native\x1f-Zthreads=8'
# fish: set -gx CARGO_ENCODED_RUSTFLAGS (printf '%s\x1f%s' -Ctarget-cpu=native -Zthreads=8)

VERSION := $(shell awk -F'"' '/^version/ {print $$2; exit}' Cargo.toml)
ifeq ($(VERSION),)
    VERSION := unknown
endif

ifeq ($(OS),Windows_NT)
    EXE_EXT := .exe
else
    EXE_EXT :=
endif
EXE := $(EXE_NAME)$(EXE_EXT)
DEBUG_EXE := debug$(EXE_EXT)

.DEFAULT_GOAL := openbench

.PHONY: help debug release native bench v3 v4 pgo openbench clean \
        evaltune test oracle flops seefmt fmt clippy profile etprofile tools \
        datagen avx2 avx2-bmi2 avx512 corrstats movepicker storecost \
        windows win-avx2 win-avx2-bmi2 win-avx512

debug: ## Build for development
	@echo "Building debug..."
	@RUSTFLAGS="-C target-cpu=native" cargo build $(CARGO_FEATURES)
	@cp target/debug/$(EXE_NAME) $(DEBUG_EXE)
	@echo "Done: ./$(DEBUG_EXE)"

ifeq ($(and $(HAS_ZIG),$(HAS_WIN_STD)),)
define win_build
	@printf '\033[33mSkipping $(EXE_NAME)-$(2).exe: needs cargo-zigbuild and rustup target add $(WIN_TARGET)\033[0m\n'
endef
else
define win_build
	@echo "Building $(EXE_NAME)-v$(VERSION)-$(2).exe..."
	@RUSTFLAGS="$(1)" cargo zigbuild --release --quiet --target $(WIN_TARGET)
	@cp target/$(WIN_TARGET)/release/$(EXE_NAME).exe $(EXE_NAME)-v$(VERSION)-$(2).exe
	@echo "Done: ./$(EXE_NAME)-v$(VERSION)-$(2).exe"
endef
endif

release: avx2 avx2-bmi2 avx512 ## Build all release binaries at once

avx2: ## Build AVX2 + FMA (pre Zen-3)
	@echo "Building $(EXE_NAME)-v$(VERSION)-avx2..."
	@RUSTFLAGS="-C target-cpu=x86-64-v2 -C target-feature=+avx2,+fma" \
		cargo build --release --quiet --target $(RUST_HOST)
	@cp target/$(RUST_HOST)/release/$(EXE_NAME) $(EXE_NAME)-v$(VERSION)-avx2$(EXE_EXT)
	@echo "Done: ./$(EXE_NAME)-v$(VERSION)-avx2$(EXE_EXT)"

avx2-bmi2: ## Build AVX2 + BMI2 (Intel 2013+ / Zen-3+)
	@echo "Building $(EXE_NAME)-v$(VERSION)-avx2-bmi2..."
	@RUSTFLAGS="-C target-cpu=x86-64-v3" \
		cargo build --release --quiet --target $(RUST_HOST)
	@cp target/$(RUST_HOST)/release/$(EXE_NAME) $(EXE_NAME)-v$(VERSION)-avx2-bmi2$(EXE_EXT)
	@echo "Done: ./$(EXE_NAME)-v$(VERSION)-avx2-bmi2$(EXE_EXT)"

windows: win-avx2 win-avx2-bmi2 win-avx512 ## Cross-build every Windows release

win-avx2: ## Windows AVX2 + FMA
	$(call win_build,-C target-cpu=x86-64-v2 -C target-feature=+avx2$(COMMA)+fma,avx2)

win-avx2-bmi2: ## Windows AVX2 + BMI2
	$(call win_build,-C target-cpu=x86-64-v3,avx2-bmi2)

win-avx512: ## Windows AVX-512
	$(call win_build,-C target-cpu=x86-64-v4,avx512)

avx512: ## Build AVX-512 (Intel Rocket Lake/Server / Zen-4+)
	@echo "Building $(EXE_NAME)-v$(VERSION)-avx512..."
	@RUSTFLAGS="-C target-cpu=x86-64-v4" \
		cargo build --release --quiet --target $(RUST_HOST)
	@cp target/$(RUST_HOST)/release/$(EXE_NAME) $(EXE_NAME)-v$(VERSION)-avx512$(EXE_EXT)
	@echo "Done: ./$(EXE_NAME)-v$(VERSION)-avx512$(EXE_EXT)"

define pgo_build
	@echo "PGO Build $(1) (depth=$(DEPTH))"
	@cargo clean > /dev/null
	@cargo pgo clean > /dev/null
	@echo "Instrumenting..."
	@CC=cc RUSTFLAGS="-C target-cpu=native -C metadata=pgo" \
		cargo pgo build -- --quiet
	@echo "Training..."
	@LLVM_PROFILE_FILE="target/pgo-profiles/%p.profraw" \
		target/$(RUST_HOST)/release/$(EXE_NAME) bench $(DEPTH) > /dev/null
	@echo "Optimizing..."
	@CC=cc RUSTFLAGS="-C target-cpu=native -C metadata=pgo" \
		cargo pgo optimize build -- --quiet
	@cp target/$(RUST_HOST)/release/$(EXE_NAME) $(EXE)
	@echo "Done: ./$(EXE)"
endef

pgo: check-pgo ## PGO build (recommended)
	@$(call pgo_build,Standard)

native: ## Build optimized for your CPU
	@echo "Building native..."
	@RUSTFLAGS="-C target-cpu=native" \
		cargo build --release --quiet --target $(RUST_HOST) $(CARGO_FEATURES)
	@cp target/$(RUST_HOST)/release/$(EXE_NAME) $(EXE)
	@echo "Done: ./$(EXE)"

bench: ## Fast compile w/ bench
	@RUSTFLAGS="-C target-cpu=native" cargo build --profile quick --quiet
	@./target/quick/$(EXE_NAME) bench $(DEPTH)

tools: ## Native build with datagen, dataset and the measurement rigs
	@$(MAKE) --no-print-directory native FEATURES=datagen,rigs EXE=tools$(EXE_EXT)

datagen: ## Native build with self-play generation and the dataset pipeline
	@$(MAKE) --no-print-directory native FEATURES=datagen EXE=datagen$(EXE_EXT)

storecost: ## Price XorBoard against a build without it (RUNS=5)
	@python3 scripts/storecost.py $(RUNS)

corrstats: ## Native build with correction-history stats
	@echo "Building $(EXE_NAME)-corrstats..."
	@RUSTFLAGS="-C target-cpu=native" \
		cargo build --release --features corrstats --quiet
	@cp target/release/$(EXE_NAME) $(EXE)-corrstats
	@echo "Done: ./$(EXE)-corrstats"
	@./$(EXE)-corrstats bench $(DEPTH)

movepicker: ## Native build with move-picker quiet stats
	@echo "Building $(EXE_NAME)-movepicker..."
	@RUSTFLAGS="-C target-cpu=native" \
		cargo build --release --features mvpstats --quiet
	@cp target/release/$(EXE_NAME) $(EXE)-movepicker
	@echo "Done: ./$(EXE)-movepicker"
	@./$(EXE)-movepicker bench $(DEPTH)

v4: ## AVX512
	@echo "Building x86-64-v4..."
	@RUSTFLAGS="-C target-cpu=x86-64-v4" \
		cargo build --release --quiet --target $(RUST_HOST)
	@cp target/$(RUST_HOST)/release/$(EXE_NAME) $(EXE)
	@echo "Done: ./$(EXE)"

v3: ## AVX2 + BMI2
	@echo "Building x86-64-v3..."
	@RUSTFLAGS="-C target-cpu=x86-64-v3" \
		cargo build --release --quiet --target $(RUST_HOST)
	@cp target/$(RUST_HOST)/release/$(EXE_NAME) $(EXE)
	@echo "Done: ./$(EXE)"

# 2>/dev/null: HEADER_EVENT_DESC is the one section perf cannot read back from a file it wrote.
# --no-inline: inline expansion spends --max-stack on the thread-start preamble.
define perf_report
	@perf report --stdio --header-only 2>/dev/null > $(1)
	@printf '\n== Self time ==\n' >> $(1)
	@perf report --stdio --no-children -g none --percent-limit 0.4 >> $(1)
	@printf '\n== Call graph ==\n' >> $(1)
	@perf report --stdio --no-inline --children --max-stack 32 --percent-limit 1.0 >> $(1)
endef

profile: ## Generate CPU performance profile
	@echo "Building with debug symbols..."
	@RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes" \
		cargo build --profile profiling --quiet --features rigs
	@cp target/profiling/$(EXE_NAME) $(EXE)
	@echo "Recording profile..."
	@rm -f perf.data
	@perf record -g --call-graph fp -F 999 ./$(EXE) speedtest
	@echo "Generating profiling report..."
	@$(call perf_report,profile_data.txt)
	@printf '\nThe profiling report has been generated in profile_data.txt\n'
	@echo "Done: profile_data.txt"

etprofile: ## Generate CPU performance profile for evaltune (set ET_DATA / ET_EPOCHS)
	@echo "Building evaltune with debug symbols..."
	@RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes" \
		cargo build --profile profiling -p evaltuner --bin evaltune --quiet
	@cp target/profiling/evaltune eval$(EXE_EXT)
	@echo "Recording profile ($(ET_DATA), $(ET_EPOCHS) epochs)..."
	@rm -f perf.data
	@perf record -g --call-graph fp -F 999 ./eval$(EXE_EXT) -d $(ET_DATA) -e $(ET_EPOCHS) --seed 1
	@echo "Generating profiling report..."
	@$(call perf_report,evaltune_profile_data.txt)
	@printf '\nThe profiling report has been generated in evaltune_profile_data.txt\n'
	@echo "Done: evaltune_profile_data.txt"

openbench:
ifdef HAS_PGO
	@$(call pgo_build,OpenBench)
else
	@echo "cargo-pgo not found — installing it so the graded build is PGO, not native..."
	@cargo install cargo-pgo --quiet 2>/dev/null || true
	@if command -v cargo-pgo >/dev/null 2>&1; then \
		$(MAKE) --no-print-directory openbench; \
	else \
		echo "cargo-pgo unavailable — native build."; \
		CC=cc RUSTFLAGS="-C target-cpu=native" cargo build --release --quiet; \
		cp target/release/$(EXE_NAME) $(EXE); \
		echo "Done: ./$(EXE)"; \
	fi
endif

evaltune: ## Build the eval tuner
	@echo "Building evaltune..."
	@RUSTFLAGS="-C target-cpu=native" \
		cargo build --release -p evaltuner --bin evaltune --quiet --target $(RUST_HOST)
	@cp target/$(RUST_HOST)/release/evaltune eval$(EXE_EXT)
	@echo "Done: ./eval$(EXE_EXT)"

test: ## Run test suite
	@RUSTDOCFLAGS="-C target-cpu=native" RUSTFLAGS="-C target-cpu=native" cargo test --workspace -- --nocapture

oracle: ## Run the eval gradient oracle tests
	@RUSTFLAGS="-C target-cpu=native" cargo test --workspace --release oracle

flops: ## f64 ops the gradient costs per position, differenced under perf
	@FLOP_EVENT="$(FLOP_EVENT)" scripts/flops.sh

seefmt: ## Check formatting
	@cargo fmt --check

fmt: ## Auto-format with rustfmt
	@cargo fmt

clippy: ## Lint with Clippy (-D warnings, whole workspace + features)
	@RUSTFLAGS="-C target-cpu=native" cargo clippy --workspace --all-features --all-targets --quiet -- -D warnings
	@RUSTFLAGS="-C target-cpu=native" cargo clippy -p soul --all-targets --quiet -- -D warnings

clean: ## Remove all build artifacts
	@echo "Cleaning..."
	@cargo clean
	@rm -f $(EXE) $(DEBUG_EXE) ./search ./eval ./datagen ./tools $(EXE)-corrstats $(EXE)-movepicker
	@rm -f $(EXE_NAME)-v*-avx2* $(EXE_NAME)-v*-avx512*
	@rm -rf target/pgo-profiles
	@echo "Done"

check-pgo:
	@command -v cargo-pgo >/dev/null 2>&1 || (echo "\x1b[38;2;220;187;80mWarning: cargo-pgo is not installed. To run PGO builds, please install it via: cargo install cargo-pgo\x1b[0m" && exit 1)

# Pens match src/color.rs: LILAC titles, IVORY names, CORAL placeholders, ASH rule, HAZE prose.
help:
	@printf '\033[1;38;2;180;140;255mSoul Chess Engine\033[0m\n\n'
	@printf '\033[38;2;139;154;171mUsage:\033[0m make \033[38;2;224;105;100m<target>\033[0m\n\n'
	@printf '\033[38;2;180;140;255mTargets\033[0m\n'
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*?##/ { \
		printf "  \033[38;2;246;238;218m%-14s\033[38;2;118;112;104m-\033[38;2;139;154;171m%s\033[0m\n", $$1, $$2 \
	}' $(MAKEFILE_LIST)
