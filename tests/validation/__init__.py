"""pineuro validation suite.

Scientific-correctness tests: synthetic data with known ground truth is run
through the *real* pipeline (real ANTsPy / nilearn / dcm2niix, no mocking of
the science) and the outputs are asserted against that ground truth.

This is distinct from the regression suite in ``tests/`` (which mocks the
science and only checks that functions run / refactors preserve behavior).

Opt-in: plain ``pytest`` skips these tests; run them with ``pytest -m
validation``. See ``tests/validation/README.md``.
"""
