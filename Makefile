# Makefile for EasyOCR project

# Variables
PROJECT_NAME = EasyOCR
REMOTE_HOST = tl
REMOTE_PATH = /project/lt200384-ff_bio/puem/ocr_list
LOCAL_PATH = ..

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  sync        - Sync project to remote server"
	@echo "  sync-dry    - Dry run sync (show what would be transferred)"
	@echo "  sync-fast   - Fast sync (skip checksum verification)"
	@echo "  clean       - Clean local cache and temporary files"
	@echo "  help        - Show this help message"

# Sync to remote server
.PHONY: sync
sync:
	@echo "Syncing $(PROJECT_NAME) to $(REMOTE_HOST):$(REMOTE_PATH)..."
	rsync -rvz --exclude-from=.rsync-exclude $(LOCAL_PATH)/$(PROJECT_NAME) $(REMOTE_HOST):$(REMOTE_PATH)
	@echo "Sync completed!"

# Dry run sync
.PHONY: sync-dry
sync-dry:
	@echo "Dry run sync to $(REMOTE_HOST):$(REMOTE_PATH)..."
	rsync -rvz --dry-run --exclude-from=.rsync-exclude $(LOCAL_PATH)/$(PROJECT_NAME) $(REMOTE_HOST):$(REMOTE_PATH)

# Fast sync (skip checksum verification)
.PHONY: sync-fast
sync-fast:
	@echo "Fast syncing $(PROJECT_NAME) to $(REMOTE_HOST):$(REMOTE_PATH)..."
	rsync -rvz --size-only --exclude-from=.rsync-exclude $(LOCAL_PATH)/$(PROJECT_NAME) $(REMOTE_HOST):$(REMOTE_PATH)
	@echo "Fast sync completed!"

# Clean local temporary files
.PHONY: clean
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	find . -type f -name "*.tmp" -delete
	@echo "Clean completed!"

# SSH into remote server
.PHONY: ssh
ssh:
	@echo "Connecting to $(REMOTE_HOST)..."
	ssh $(REMOTE_HOST)

# SSH and navigate to project directory
.PHONY: ssh-project
ssh-project:
	@echo "Connecting to $(REMOTE_HOST) and navigating to project..."
	ssh -t $(REMOTE_HOST) "cd $(REMOTE_PATH)/$(PROJECT_NAME) && bash"
