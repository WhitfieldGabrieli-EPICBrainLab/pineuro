"""Smoke test: verify the package imports correctly."""


def test_import():
    import pineuro  # noqa: F811

    assert pineuro.__name__ == "pineuro"
