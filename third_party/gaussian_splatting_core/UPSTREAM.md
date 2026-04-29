## Upstream Source

This directory vendors a minimally modified copy of the upstream 3DGS implementation.

- Upstream repository: `https://github.com/graphdeco-inria/gaussian-splatting`
- Local source used for import: `/mnt/gxf/A_work/gaussian-splatting`
- Imported into this repository for integration with `lingbot-map`.

## Modification Policy

- Keep upstream behavior unchanged whenever possible.
- Concentrate integration changes in a small set of files:
  - `scene/dataset_readers.py`
  - `scene/__init__.py`
  - `arguments/__init__.py`
  - integration helper modules under `scene/`
- Avoid touching the optimizer core unless strictly required.
