include .env
export

.PHONY: install
install:
	poetry install --no-root
