include .env
export

.PHONY: install
install:
	poetry install
	pip install -e .

.env:
	echo "" > .env
