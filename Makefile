include .env
export

.PHONY: install
install:
	poetry install --no-root

.env:
	echo "" > .env
