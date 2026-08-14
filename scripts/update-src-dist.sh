#!/usr/bin/env bash

set -euo pipefail

source_branch="main"
target_branch="src-dist"
push=false

usage() {
  cat <<'EOF'
Usage: scripts/update-src-dist.sh [--push]

Rebuild src-dist from the contents and history of main:src/plutil. The files
under src/plutil become the root of src-dist; nothing else from main is kept.

Options:
  --push  Force-push the rebuilt branch to origin using --force-with-lease.
  -h, --help  Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --push)
      push=true
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

git rev-parse --verify --quiet "${source_branch}^{commit}" >/dev/null || {
  printf 'Branch %s does not exist.\n' "$source_branch" >&2
  exit 1
}

git cat-file -e "${source_branch}:src/plutil" || {
  printf '%s does not contain src/plutil.\n' "$source_branch" >&2
  exit 1
}

if git worktree list --porcelain | grep -Fxq "branch refs/heads/${target_branch}"; then
  printf 'Branch %s is checked out in a worktree; switch that worktree to another branch first.\n' \
    "$target_branch" >&2
  exit 1
fi

old_target=$(git rev-parse --verify --quiet "refs/heads/${target_branch}" || true)
split_commit=$(git subtree split --prefix=src/plutil "$source_branch")

if [[ -n "$old_target" ]]; then
  git update-ref "refs/heads/${target_branch}" "$split_commit" "$old_target"
else
  git update-ref "refs/heads/${target_branch}" "$split_commit"
fi

source_tree=$(git rev-parse "${source_branch}:src/plutil")
target_tree=$(git rev-parse "${target_branch}^{tree}")
if [[ "$source_tree" != "$target_tree" ]]; then
  printf 'Refusing to continue: %s does not exactly match %s:src/plutil.\n' \
    "$target_branch" "$source_branch" >&2
  exit 1
fi

printf 'Updated %s to %s from %s:src/plutil.\n' \
  "$target_branch" "$split_commit" "$source_branch"

if [[ "$push" == true ]]; then
  git push --force-with-lease "origin" "${target_branch}:${target_branch}"
fi
