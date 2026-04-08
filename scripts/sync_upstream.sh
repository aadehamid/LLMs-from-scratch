#!/usr/bin/env bash
set -euo pipefail

# Sync fork's main with upstream and rebase working branch.
# Usage: ./scripts/sync_upstream.sh [branch]
#   branch: the branch to rebase onto updated main (default: experiment)

BRANCH="${1:-experiment}"
CURRENT=$(git rev-parse --abbrev-ref HEAD)

echo ":: Fetching upstream..."
git fetch upstream

echo ":: Switching to main..."
git checkout main

echo ":: Merging upstream/main..."
git merge upstream/main

echo ":: Pushing main to origin..."
git push origin main

echo ":: Switching to $BRANCH..."
git checkout "$BRANCH"

echo ":: Rebasing $BRANCH onto main..."
if ! git rebase main; then
    echo "!! Rebase conflict — resolve manually, then run:"
    echo "   git rebase --continue"
    echo "   git push --force-with-lease origin $BRANCH"
    exit 1
fi

echo ":: Force-pushing $BRANCH to origin..."
git push --force-with-lease origin "$BRANCH"

# Return to original branch if different
if [ "$CURRENT" != "$BRANCH" ] && [ "$CURRENT" != "main" ]; then
    git checkout "$CURRENT"
fi

echo ":: Done. $BRANCH is rebased onto latest upstream/main."
