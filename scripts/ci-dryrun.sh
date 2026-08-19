#!/usr/bin/env bash

set -uo pipefail

if (( $# == 0 )); then
	echo "Usage: $0 COMMAND..." >&2
	exit 2
fi

output_dir="$(mktemp -d)"
trap 'rm -rf "$output_dir"' EXIT
stdout="$output_dir/stdout"
stderr="$output_dir/stderr"

echo "Running..."

for command in "$@"; do
	status=0
	bash -c "$command" >"$stdout" 2>"$stderr" || status=$?

	if [[ -s "$stderr" || $status -ne 0 ]]; then
		cat "$stdout"
	fi
	cat "$stderr" >&2

	if (( status != 0 )); then
		exit "$status"
	fi
done

echo "Passed."
