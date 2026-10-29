# AML data mapping workbook package

The three Excel workbooks delivered with this project are:
- `01_Source_to_Target_Mapping.xlsx`: transaction STTM and key/grain catalog.
- `02_Detection_and_Feature_Mapping.xlsx`: detector-stage mapping and frozen HOLDOUT evaluation.
- `03_Alert_Case_Lineage_Mapping.xlsx`: operational lineage and governance controls.

**Repository format:** The three Excel workbooks and companion UTF-8 CSV mappings are committed in this directory. CSVs support readable diffs; the workbooks provide review-friendly tables. These are mapping/documentation artifacts, **not independent proof of model training, HOLDOUT scoring, SIT/UAT execution or data lineage at runtime**.

Key distinction: Rule-Based V3 is the frozen 11-scenario baseline; XGBoost Detection V1 is the independent transaction-level ML detector. The four-scenario rolling experiment is not part of this mapping.

Source artifacts remain local under `data/` and `results/`. Mapping is documentation, not evidence that source data or pipelines are deployable.
