.PHONY: help sync-upstream sync-experiment status push save quick-sync branches

# Default target - show help
help:
	@echo "Git Workflow Commands:"
	@echo "  make sync-upstream    - Fetch upstream and update local main branch"
	@echo "  make sync-experiment  - Merge main into experiment branch"
	@echo "  make sync             - Full sync: update main from upstream, then sync experiment"
	@echo "  make status           - Show git status and branch info"
	@echo "  make push             - Push current branch to origin"
	@echo "  make save MSG='...'   - Quick commit and push (use: make save MSG='commit message')"
	@echo "  make branches         - Show all local and remote branches"
	@echo ""
	@echo "Common Workflows:"
	@echo "  1. Daily sync:        make sync"
	@echo "  2. Quick save work:   make save MSG='Added feature X'"
	@echo "  3. Check status:      make status"

# Fetch from upstream and update main branch
sync-upstream:
	@echo "==> Fetching updates from upstream..."
	git fetch upstream
	@echo "==> Switching to main branch..."
	git checkout main
	@echo "==> Merging upstream/main into local main..."
	git merge upstream/main --ff-only
	@echo "==> Pushing updated main to origin..."
	git push origin main
	@echo "✓ Main branch synced with upstream"

# Merge main into experiment branch
sync-experiment:
	@echo "==> Switching to experiment branch..."
	git checkout experiment
	@echo "==> Merging main into experiment..."
	git merge main -m "Sync experiment with main branch"
	@echo "==> Pushing experiment to origin..."
	git push origin experiment
	@echo "✓ Experiment branch synced with main"

# Full sync: upstream -> main -> experiment
sync: sync-upstream sync-experiment
	@echo ""
	@echo "✓ Full sync complete!"
	@echo "  - main: synced with upstream"
	@echo "  - experiment: synced with main"

# Show current status
status:
	@echo "==> Current branch:"
	@git branch --show-current
	@echo ""
	@echo "==> Git status:"
	@git status -s
	@echo ""
	@echo "==> Recent commits:"
	@git log --oneline -5
	@echo ""
	@echo "==> Remote tracking:"
	@git remote -v

# Push current branch
push:
	@echo "==> Pushing current branch to origin..."
	git push

# Quick save: add, commit, and push
save:
ifndef MSG
	$(error MSG is required. Usage: make save MSG='your commit message')
endif
	@echo "==> Adding all changes..."
	git add .
	@echo "==> Committing with message: $(MSG)"
	git commit -m "$(MSG)"
	@echo "==> Pushing to origin..."
	git push
	@echo "✓ Changes saved and pushed"

# Show all branches
branches:
	@echo "==> All branches:"
	@git branch -a

# Quick sync without verbose output
quick-sync:
	@git fetch upstream && \
	git checkout main && \
	git merge upstream/main --ff-only && \
	git push origin main && \
	git checkout experiment && \
	git merge main -m "Sync experiment with main" && \
	git push origin experiment && \
	echo "✓ Sync complete"
