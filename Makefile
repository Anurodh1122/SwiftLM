# Local entry points.
#
# `make test` is the supported way to run the unit suites. A bare `swift test` on a fresh
# checkout can die before the suite runs (MLX cannot find its Metal library) and still
# exit 0, which looks like a short green pass; scripts/run-tests.sh bootstraps the metallib
# and fails the run when too few tests executed.

.PHONY: test test-filter

test:
	./scripts/run-tests.sh

# make test-filter FILTER=PromptCacheTests
test-filter:
	@test -n "$(FILTER)" || { echo "usage: make test-filter FILTER=<XCTest filter>"; exit 2; }
	./scripts/run-tests.sh --filter "$(FILTER)"
