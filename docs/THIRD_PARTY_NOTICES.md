# Third-Party Notices

This repository contains source code and other material derived from
third-party projects. Those components remain subject to their original
licenses.

## UniDriveVLA

- Project: UniDriveVLA
- Upstream repository: https://github.com/xiaomi-research/UniDriveVLA
- Upstream revision: `a93c175af893b35dc16618e659eca4d18bb1ec86`
- Copyright: 2026 Xiaomi Corporation
- License: Apache License 2.0
- License text: [`LICENSES/Apache-2.0.txt`](../LICENSES/Apache-2.0.txt)

The following local files contain material derived from UniDriveVLA:

- `data/UniDriveVLA_Data/tokens.txt`
  - Copied from UniDriveVLA's special-token configuration.
- `models/max_v1/prompt_template.py`
  - `NUSCENES_SYSTEM` is copied from UniDriveVLA's nuScenes system prompt.
- `tools/nuscenes/nuscenes_converter.py`
  - Ported from UniDriveVLA's nuScenes data converter.
  - Modified to remove the OpenMMLab runtime dependency and use standalone
    serialization, progress, and file helpers.
- `tools/nuscenes/utils/map_geometry.py`
  - Derived from UniDriveVLA's nuScenes map utility helpers.
  - Modified to remove unused dependencies and support standalone use.
- `tools/nuscenes/utils/nuscmap_extractor.py`
  - Derived from UniDriveVLA's nuScenes map extractor.
  - Modified to use the standalone local map-geometry helpers.
- `tools/nuscenes/utils/planning_eval.py`
  - Derived from UniDriveVLA's planning metrics, dataset helpers, and box
    utilities.
  - Modified to remove OpenMMLab runtime dependencies and integrate with the
    Max V1 inference and reporting flow.

No upstream `NOTICE` file was present in the referenced UniDriveVLA revision.
