#!/usr/bin/env bash

set -uo pipefail

if (( $# == 0 )); then
	echo "Usage: $0 COMMAND..." >&2
	exit 2
fi

output_dir="$(mktemp -d)"
trap 'rm -rf "$output_dir"' EXIT

declare -a commands=("$@")
declare -a pids=()

echo "Running..."

for index in "${!commands[@]}"; do
	bash -c "${commands[$index]}" \
		>"$output_dir/$index.stdout" \
		2>"$output_dir/$index.stderr" &
	pids[$index]=$!
done

status=0
for index in "${!commands[@]}"; do
	command_status=0
	wait "${pids[$index]}" || command_status=$?
	stdout="$output_dir/$index.stdout"
	stderr="$output_dir/$index.stderr"

	if (( command_status == 0 )); then
		if [[ -s "$stderr" ]]; then
			cat "$stdout"
		fi
		cat "$stderr" >&2
		continue
	fi

	echo "Failed: ${commands[$index]}" >&2

	if [[ -s "$stdout" ]]; then
		echo "stdout:"
		cat "$stdout"
	fi
	if [[ -s "$stderr" ]]; then
		echo "stderr:" >&2
		cat "$stderr" >&2
	fi

	(( status == 0 )) && status=$command_status
done

(( status == 0 )) || exit "$status"

echo "Passed."
