EXE_NAME := soul
DEPTH    ?= 12

# Evaltune profiling dataset + epoch count.
ET_DATA   ?= data/big3.txt
ET_EPOCHS ?= 100

HAS_PGO := $(shell command -v cargo-pgo 2> /dev/null)
RUST_HOST := $(shell rustc -vV | sed -n 's/host: //p')

# OpenBench passes CC=cargo (for C/C++ engines). Override it so cc-rs
# (used by zstd-sys) finds a real C compiler instead of cargo.
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

.PHONY: all help debug release native bench v3 v4 pgo openbench clean \
        evaltune searchtune test oracle seeformat format clippy profile etprofile \
        releases avx2 avx2-bmi2 avx512 corrstats

all: openbench

debug: ## Build for development
	@echo "Building debug..."
	@RUSTFLAGS="-C target-cpu=native" cargo build
	@cp target/debug/$(EXE_NAME) $(DEBUG_EXE)
	@echo "Done: ./$(DEBUG_EXE)"

releases: avx2 avx2-bmi2 avx512 ## Build all release binaries at once

avx2: ## build AVX2 + FMA (pre Zen-3)	
	@echo "Building $(EXE_NAME)-v$(VERSION)-avx2..."
	@RUSTFLAGS="-C target-cpu=x86-64-v2 -C target-feature=+avx2,+fma" \
		cargo build --release --quiet
	@cp target/release/$(EXE_NAME) $(EXE_NAME)-v$(VERSION)-avx2$(EXE_EXT)
	@echo "Done: ./$(EXE_NAME)-v$(VERSION)-avx2$(EXE_EXT)"

avx2-bmi2: ## Build AVX2 + BMI2 (Intel 2013+ / Zen-3+)
	@echo "Building $(EXE_NAME)-v$(VERSION)-avx2-bmi2..."
	@RUSTFLAGS="-C target-cpu=x86-64-v3" \
		cargo build --release --quiet
	@cp target/release/$(EXE_NAME) $(EXE_NAME)-v$(VERSION)-avx2-bmi2$(EXE_EXT)
	@echo "Done: ./$(EXE_NAME)-v$(VERSION)-avx2-bmi2$(EXE_EXT)"

avx512: ## Build AVX-512 (Intel Rocket Lake/Server / Zen-4+)
	@echo "Building $(EXE_NAME)-v$(VERSION)-avx512..."
	@RUSTFLAGS="-C target-cpu=x86-64-v4" \
		cargo build --release --quiet
	@cp target/release/$(EXE_NAME) $(EXE_NAME)-v$(VERSION)-avx512$(EXE_EXT)
	@echo "Done: ./$(EXE_NAME)-v$(VERSION)-avx512$(EXE_EXT)"

define pgo_build
	@echo "PGO Build $(1) (depth=$(DEPTH))"
	@cargo clean > /dev/null 2>&1
	@cargo pgo clean > /dev/null 2>&1
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
		cargo build --release --quiet
	@cp target/release/$(EXE_NAME) $(EXE)
	@echo "Done: ./$(EXE)"

bench: ## Fast compile w/ bench
	@RUSTFLAGS="-C target-cpu=native" cargo build --profile quick --quiet
	@./target/quick/$(EXE_NAME) bench $(DEPTH)

corrstats: ## Native build with correction-history stats
	@echo "Building $(EXE_NAME)-corrstats..."
	@RUSTFLAGS="-C target-cpu=native" \
		cargo build --release --features corrstats --quiet
	@cp target/release/$(EXE_NAME) $(EXE)-corrstats
	@echo "Done: ./$(EXE)-corrstats"
	@./$(EXE)-corrstats bench $(DEPTH)

v4: ## AVX512
	@echo "Building x86-64-v4..."
	@RUSTFLAGS="-C target-cpu=x86-64-v4" \
		cargo build --release --quiet
	@cp target/release/$(EXE_NAME) $(EXE)
	@echo "Done: ./$(EXE)"

v3: ## AVX2 + BMI2
	@echo "Building x86-64-v3..."
	@RUSTFLAGS="-C target-cpu=x86-64-v3" \
		cargo build --release --quiet
	@cp target/release/$(EXE_NAME) $(EXE)
	@echo "Done: ./$(EXE)"

profile: ## Generate CPU performance profile
	@echo "Building with debug symbols..."
	@RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes" \
		cargo build --profile profiling --quiet
	@cp target/profiling/$(EXE_NAME) $(EXE)
	@echo "Recording profile..."
	@rm -f perf.data
	@perf record -g --call-graph fp -F 999 ./$(EXE) speedtest
	@echo "Generating profiling report..."
	@perf report --stdio --header --inline --children --max-stack 15 --percent-limit 1.0 > profile_data.txt
	@echo "\nThe profiling report has been generated in profile_data.txt"
	@echo "Done: profile_data.txt"

etprofile: ## Generate CPU performance profile for evaltune (set ET_DATA / ET_EPOCHS)
	@echo "Building evaltune with debug symbols..."
	@RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes" \
		cargo build --profile profiling -p tuner --bin evaltune --quiet
	@cp target/profiling/evaltune eval$(EXE_EXT)
	@echo "Recording profile ($(ET_DATA), $(ET_EPOCHS) epochs)..."
	@rm -f perf.data
	@perf record -g --call-graph fp -F 999 ./eval$(EXE_EXT) -d $(ET_DATA) -e $(ET_EPOCHS) --seed 1
	@echo "Generating profiling report..."
	@perf report --stdio --header --inline --children --max-stack 15 --percent-limit 1.0 > evaltune_profile_data.txt
	@echo "\nThe profiling report has been generated in evaltune_profile_data.txt"
	@echo "Done: evaltune_profile_data.txt"

openbench:
ifdef HAS_PGO
	@$(call pgo_build,OpenBench)
else
	@echo "Building for OpenBench (cargo-pgo not found — native build)..."
	@CC=cc RUSTFLAGS="-C target-cpu=native" \
		cargo build --release --quiet
	@cp target/release/$(EXE_NAME) $(EXE)
	@echo "Done: ./$(EXE)"
endif

evaltune:
	@echo "Building evaltune..."
	@RUSTFLAGS="-C target-cpu=native" \
		cargo build --release -p tuner --bin evaltune --quiet
	@cp target/release/evaltune eval$(EXE_EXT)
	@echo "Done: ./eval$(EXE_EXT)"

searchtune:
	@echo "Building searchtune..."
	@RUSTFLAGS="-C target-cpu=native" \
		cargo build --release -p tuner --bin searchtune --features searchtune --quiet
	@cp target/release/searchtune search$(EXE_EXT)
	@echo "Done: ./search$(EXE_EXT)"

test: ## Run test suite
	@RUSTDOCFLAGS="-C target-cpu=native" RUSTFLAGS="-C target-cpu=native" cargo test --workspace -- --nocapture

oracle: ## Run the eval-tuner gradient oracle tests
	@RUSTFLAGS="-C target-cpu=native" cargo test -p tuner --release oracle

seeformat: ## Check formatting (no changes)
	@cargo fmt --check

format: ## Auto-format with rustfmt
	@cargo fmt

clippy: ## Lint with Clippy (-D warnings, whole workspace + features)
	@RUSTFLAGS="-C target-cpu=native" cargo clippy --workspace --all-features --quiet -- -D warnings

clean: ## Remove all build artifacts
	@echo "Cleaning..."
	@cargo clean
	@rm -f $(EXE) $(DEBUG_EXE) ./search ./eval $(EXE)-corrstats
	@rm -f $(EXE_NAME)-v*-avx2* $(EXE_NAME)-v*-avx512*
	@rm -rf target/pgo-profiles
	@echo "Done"

check-pgo:
	@command -v cargo-pgo >/dev/null 2>&1 || (echo "\x1b[33mWarning: cargo-pgo is not installed. To run PGO builds, please install it via: cargo install cargo-pgo\x1b[0m" && exit 1)

help:
	@printf '\033[1;38;2;180;140;255mSoul Chess Engine\033[0m\n\n'
	@printf 'Usage: make \033[38;2;255;215;0m<target>\033[0m\n\n'
	@printf 'Targets:\n'
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*?##/ { \
		printf "  \033[38;2;255;215;0m%-12s\033[0m \033[38;2;130;130;130m%s\033[0m\n", $$1, $$2 \
	}' $(MAKEFILE_LIST)
