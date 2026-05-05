"""Data layer: dataset acquisition, schema mapping, synthetic-data generation.

Submodules:

- ``cardiorisk.data.paths`` — repo-relative path constants.
- ``cardiorisk.data.fetch`` — download + checksum-pin UCI Heart Disease subsets
  (and optionally the Kaggle HFP combined CSV); idempotent re-runs.
- ``cardiorisk.data.combine`` — join the four UCI subsets into one DataFrame
  with the HFP 11-feature + target schema and a ``source`` LODO grouping
  column.
- ``cardiorisk.data.synthetic`` — deterministic synthetic-data generator for
  the test fixture; HFP-schema CSV.

Phase 2.2 will add ``cardiorisk.data.preprocess`` (imputation, encoding,
splitting) on top of these.
"""
