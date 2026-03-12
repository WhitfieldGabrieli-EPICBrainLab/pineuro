"""Smoke test: verify the package imports correctly."""


def test_import():
    import pineuro

    assert pineuro.__name__ == "pineuro"
