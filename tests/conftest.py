import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "validation: mark a test as a data validation test"
    )
