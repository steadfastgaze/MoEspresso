#!/bin/sh
# Require an exact ignored test to exist before running its opt-in gate.
set -eu

if [ "$#" -ne 2 ]; then
    printf '%s\n' 'Usage: test-ignored.sh lib|lib-release|lib-release-live|acceptance exact-test-name' >&2
    exit 2
fi
test_target=$1
test_name=$2
test_live=0
case "$test_target" in
    lib) set -- --lib ;;
    lib-release) set -- --release --lib ;;
    lib-release-live) set -- --release --lib; test_live=1 ;;
    acceptance) set -- --release --test acceptance ;;
    *) printf 'Unsupported test target: %s\n' "$test_target" >&2; exit 2 ;;
esac

test_listing=$(mktemp "${TMPDIR:-/tmp}/moespresso-test-list.XXXXXXXX")
if [ "$test_live" -eq 1 ]; then
    live_directory=$(mktemp -d "${TMPDIR:-/tmp}/moespresso-test-live.XXXXXXXX")
    live_start=$live_directory/start
    live_pid=
    live_pgid=
    live_launch_pending=0
    live_signal=TERM
    live_status=0

    stop_live() {
        if [ -n "$live_pgid" ]; then
            if kill -0 -- "-$live_pgid" 2>/dev/null; then
                kill -s "$1" -- "-$live_pgid" 2>/dev/null || :
                live_steps=0
                while kill -0 -- "-$live_pgid" 2>/dev/null && [ "$live_steps" -lt 20 ]; do
                    sleep 0.1
                    live_steps=$((live_steps + 1))
                done
                if kill -0 -- "-$live_pgid" 2>/dev/null; then
                    kill -KILL -- "-$live_pgid" 2>/dev/null || :
                fi
            fi
        elif [ -n "$live_pid" ]; then
            # No Cargo command starts before process-group verification.
            kill -TERM "$live_pid" 2>/dev/null || :
        fi
        if [ -n "$live_pid" ]; then wait "$live_pid" 2>/dev/null || :; fi
        live_pid=
        live_pgid=
    }

    cleanup_live() {
        live_exit=$?
        stop_live TERM
        rm -f "$test_listing" "$live_start"
        rmdir "$live_directory"
        return "$live_exit"
    }

    interrupt_live() {
        trap '' HUP INT TERM
        live_signal=$1
        live_status=$2
        # A trap between background launch and PID assignment defers cleanup
        # until that exact owned job has been claimed and verified.
        if [ "$live_launch_pending" -eq 1 ] && [ -z "$live_pid" ]; then return; fi
        stop_live "$live_signal"
        exit "$live_status"
    }

    trap cleanup_live EXIT
    trap 'interrupt_live HUP 129' HUP
    trap 'interrupt_live INT 130' INT
    trap 'interrupt_live TERM 143' TERM
    set -m
    case "$-" in
        *m*) ;;
        *) printf '%s\n' 'Live gates require shell process-group job control.' >&2; exit 2 ;;
    esac

    run_live() {
        rm -f "$live_start"
        live_launch_pending=1
        (
            trap - HUP INT TERM
            while [ ! -f "$live_start" ]; do sleep 0.05; done
            exec cargo test --locked "$@"
        ) &
        live_pid=$!
        live_launch_pending=0
        live_observed=$(ps -o pgid= -p "$live_pid") || live_observed=
        live_observed=$(printf '%s' "$live_observed" | tr -d '[:space:]')
        if [ "$live_observed" != "$live_pid" ]; then
            printf '%s\n' 'Live gate refused: the child does not own a dedicated process group.' >&2
            stop_live TERM
            exit 2
        fi
        live_pgid=$live_pid
        if [ "$live_status" -ne 0 ]; then
            stop_live "$live_signal"
            exit "$live_status"
        fi
        : > "$live_start"
        if wait "$live_pid"; then live_result=0; else live_result=$?; fi
        # A completed Cargo leader must not leave owned descendants running.
        stop_live TERM
        return "$live_result"
    }

    run_live "$@" "$test_name" -- --exact --ignored --list > "$test_listing"
else
    trap 'rm -f "$test_listing"' EXIT HUP INT TERM
    cargo test --locked "$@" "$test_name" -- --exact --ignored --list > "$test_listing"
fi
if ! rg -F -x -- "$test_name: test" "$test_listing" > /dev/null; then
    printf 'Required ignored test is absent: %s\n' "$test_name" >&2
    exit 1
fi
if [ "$test_live" -eq 1 ]; then
    run_live "$@" "$test_name" -- --exact --ignored --test-threads=1 --nocapture
else
    cargo test --locked "$@" "$test_name" -- --exact --ignored --test-threads=1 --show-output
fi
