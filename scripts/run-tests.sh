#!/usr/bin/env bash
# run-tests.sh — run the local unit suites and refuse to report a run that barely ran.
#
# Usage: scripts/run-tests.sh [extra `swift test` arguments...]
#        scripts/run-tests.sh --filter PromptCacheTests
#
# Why this exists: MLX aborts the test process when it cannot load its Metal library. On the
# pre-6.4 build layout that happens on the second test, and `swift test` can still exit 0
# after printing "Failed to load the default metallib" — the run looks like a short pass
# while ~270 tests never executed. `scripts/bootstrap_local_tests.sh` installs the metallib;
# this wrapper calls it and then checks the run actually executed the suite.
set -euo pipefail
cd "$(dirname "$0")/.."

# The suite is a few hundred tests (326 at the time of writing). The floor sits well below
# that so ordinary additions and removals never need a bump, but far above the one or two a
# broken metallib leaves behind.
MIN_TESTS="${MIN_TESTS:-200}"

bash scripts/bootstrap_local_tests.sh

LOG="$(mktemp -t swiftlm-tests.XXXXXX)"
trap 'rm -f "$LOG"' EXIT

set +e
swift test --skip-build --disable-swift-testing "$@" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

# XCTest prints "Executed N tests" per suite and once for the whole run; the largest number
# is the total for the run.
executed=$(grep -oE "Executed [0-9]+ tests" "$LOG" | grep -oE "[0-9]+" | sort -n | tail -1)
executed=${executed:-0}

if [ "$status" -ne 0 ]; then
    exit "$status"
fi

# The floor only makes sense for an unfiltered full run. A caller passing --filter is
# deliberately narrowing, and a dozen passing tests is a correct result there.
if [ "$#" -eq 0 ] && [ "$executed" -lt "$MIN_TESTS" ]; then
    cat >&2 <<EOF

error: ${executed} test(s) executed, expected at least ${MIN_TESTS}.

A run this short usually means MLX could not load its Metal library: the test process
died early while \`swift test\` still exited 0. Look for "Failed to load the default
metallib" in the output above, then re-run scripts/bootstrap_local_tests.sh.

If the suite genuinely shrank, set MIN_TESTS to the new floor.
EOF
    exit 1
fi

echo "=> ${executed} tests executed (floor ${MIN_TESTS})."
