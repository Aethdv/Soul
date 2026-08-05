#!/usr/bin/env bash
#
# What the gradient costs on top of the eval: one run of `measure_gradient_ops`
# per mode, differenced on a retired-FLOP counter and task-clock.
# Needs perf and jq.
#
# FLOP_EVENT defaults to AMD's counter. On Intel:
#   FLOP_EVENT=fp_arith_inst_retired.scalar_double scripts/flops.sh
set -euo pipefail

cd "$(dirname "$0")/.."

event=${FLOP_EVENT:-fp_ret_sse_avx_ops.all}

bin=$(RUSTFLAGS="-C target-cpu=native" cargo test --release -p soul --no-run --message-format=json 2>/dev/null \
    | jq -r 'select(.executable != null and .target.kind[0] == "lib") | .executable' | tail -1)

[[ -n $bin ]] || { echo "no lib test binary; cargo test --no-run built nothing" >&2; exit 1; }

# Four fields for one mode: retired ops, cycles, elapsed ms, positions.
#
# Cycles carry the comparison. Wall time on aeth's laptop drifts 10% with boost
# state alone, enough to read a thermal change as a code change, while cycles
# per position hold within about a percent.
measure() {
    for _ in $(seq "${SOUL_OPS_RUNS:-3}"); do
        SOUL_OPS_MODE=$1 perf stat -x, -e "$event",cycles,task-clock "$bin" measure_gradient_ops --ignored --nocapture 2>&1
    done | awk -F, -v event="$event" '
        $0 ~ event { ops = $1 }
        /,cycles/ { if (cyc == "" || $1 + 0 < cyc + 0) cyc = $1 }
        /task-clock/ { if (ms == "" || $1 + 0 < ms + 0) ms = $1 }
        # Not CSV, so this one splits on whitespace.
        /^positions/ { split($0, field, " "); positions = field[2] }
        END { print ops, cyc, ms, positions }'
}

read -r eval_ops eval_cyc eval_ms positions <<<"$(measure eval)"
read -r loss_ops loss_cyc loss_ms _ <<<"$(measure loss)"
read -r grad_ops grad_cyc grad_ms _ <<<"$(measure grad)"
read -r record_ops record_cyc record_ms _ <<<"$(measure record)"
read -r record_grad_ops record_grad_cyc record_grad_ms _ <<<"$(measure recordgrad)"

[[ ${positions:-0} -gt 0 && ${eval_ops:-0} -gt 0 ]] || {
    echo "no usable counts: check perf_event_paranoid and whether this CPU has $event" >&2
    exit 1
}

awk -v p="$positions" \
    -v a="$eval_ops" -v ya="$eval_cyc" -v ta="$eval_ms" \
    -v b="$loss_ops" -v yb="$loss_cyc" -v tb="$loss_ms" \
    -v c="$grad_ops" -v yc="$grad_cyc" -v tc="$grad_ms" \
    -v r="$record_ops" -v yr="$record_cyc" -v tr="$record_ms" \
    -v g="$record_grad_ops" -v yg="$record_grad_cyc" -v tg="$record_grad_ms" '
    function row(label, ops, cyc, ms, base_ops, base_cyc) {
        if (base_ops == 0) {
            printf "    %-13s %8.1f %8.0f %8.1f\n", label, ops / p, cyc / p, ms * 1e6 / p;
            return;
        }

        printf "    %-13s %8.1f %8.0f %8.1f   (%+.1f ops, %+.0f cyc)\n", label, ops / p, cyc / p, ms * 1e6 / p,
            (ops - base_ops) / p, (cyc - base_cyc) / p;
    }

    BEGIN {
        printf "%s positions, per position\n\n", p;

        printf "  board path%14s%9s%9s\n", "ops", "cyc", "ns";
        row("eval", a, ya, ta, 0, 0);
        row("+ loss", b, yb, tb, a, ya);
        row("+ gradient", c, yc, tc, a, ya);

        printf "\n  cached path, what an epoch runs\n";
        row("eval_record", r, yr, tr, 0, 0);
        row("+ gradient", g, yg, tg, r, yr);
    }'
