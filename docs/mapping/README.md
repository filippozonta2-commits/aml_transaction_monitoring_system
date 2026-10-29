# AML data mapping workbook package

The three Excel workbooks delivered with this project are:
- `01_Source_to_Target_Mapping.xlsx`: transaction STTM and key/grain catalog.
- `02_Detection_and_Feature_Mapping.xlsx`: detector-stage mapping and frozen HOLDOUT evaluation.
- `03_Alert_Case_Lineage_Mapping.xlsx`: operational lineage and governance controls.

**Repository format:** This directory contains companion UTF-8 CSV mappings that GitHub can diff and review. Excel workbooks are generated separately and can be committed to this directory using the instructions supplied with the deliverables. The Excel binary files are **not** yet committed here.

Key distinction: Rule-Based V3 is the frozen 11-scenario baseline; XGBoost Detection V1 is the independent transaction-level ML detector. The four-scenario rolling experiment is not part of this mapping.

Source artifacts remain local under `data/` and `results/`. Mapping is documentation, not evidence that source data or pipelines are deployable.
