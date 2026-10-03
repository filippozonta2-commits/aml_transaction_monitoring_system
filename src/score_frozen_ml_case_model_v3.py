"""Score current V3/V1 HOLDOUT cases with the frozen DEVELOPMENT-selected XGBoost model.

Training uses DEVELOPMENT labels only. HOLDOUT labels are never loaded. Output is
an operational ranking alternative to the transparent scenario-based queue.
"""
from pathlib import Path
import argparse, json, numpy as np, pandas as pd
from xgboost import XGBClassifier
from case_model_frozen import MODEL_PARAMS, DROP_REDUNDANT

META={"CASE_ID","account_id","case_start","case_end","scenario_list","case_status","case_policy"}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--development-features",type=Path,default=Path("results/case_features/case_features_development.csv"))
 p.add_argument("--development-labels",type=Path,default=Path("results/investigator_queue/case_evaluation_labels.csv"))
 p.add_argument("--holdout-features",type=Path,default=Path("results/ml_case_v3/case_features_holdout.csv"))
 p.add_argument("--out",type=Path,default=Path("results/ml_case_v3"))
 a=p.parse_args()
 dev=pd.read_csv(a.development_features).merge(pd.read_csv(a.development_labels)[["CASE_ID","aml_positive"]],on="CASE_ID",validate="one_to_one")
 ho=pd.read_csv(a.holdout_features)
 dev["case_start"]=pd.to_datetime(dev.case_start); dev=dev.sort_values(["case_start","CASE_ID"]).reset_index(drop=True)
 cutoff=dev.loc[int(len(dev)*.70),"case_start"]; tr=dev[dev.case_start<cutoff].copy()
 features=[c for c in dev.columns if c not in META|{"aml_positive"}|set(DROP_REDUNDANT) and pd.api.types.is_numeric_dtype(dev[c]) and c!="account_id"]
 missing=[c for c in features if c not in ho.columns]
 for c in missing: ho[c]=0.0
 Xtr=tr[features].replace([np.inf,-np.inf],np.nan); Xho=ho[features].replace([np.inf,-np.inf],np.nan)
 med=Xtr.median(numeric_only=True); Xtr=Xtr.fillna(med).fillna(0); Xho=Xho.fillna(med).fillna(0)
 y=tr.aml_positive.astype(int); scale=max((len(y)-y.sum())/max(y.sum(),1),1.0)
 params=dict(MODEL_PARAMS); params["scale_pos_weight"]=scale
 model=XGBClassifier(**params); model.fit(Xtr,y)
 prob=model.predict_proba(Xho)[:,1]
 out=ho[["case_id","subject_id"]].copy(); out["ml_probability"]=prob
 out["ml_score"]=(100*prob).round(2); out["ml_rank"]=out.ml_probability.rank(method="first",ascending=False).astype(int)
 out=out.sort_values("ml_rank")
 imp=pd.DataFrame({"feature":features,"importance":model.feature_importances_}).sort_values("importance",ascending=False)
 a.out.mkdir(parents=True,exist_ok=True); out.to_csv(a.out/"ml_case_scores.csv",index=False); imp.to_csv(a.out/"ml_feature_importance.csv",index=False)
 with open(a.out/"ml_model_contract.json","w") as f: json.dump({"model":"xgboost","training":"DEVELOPMENT only","temporal_cutoff":str(cutoff),"features":features,"missing_holdout_features_filled_zero":missing,"holdout_labels_used":False},f,indent=2)
 print("\n=== FROZEN ML CASE PRIORITIZATION — HOLDOUT SCORING ===")
 print(f"Training cases: {len(tr):,} | scored HOLDOUT cases: {len(out):,} | predictors: {len(features)}")
 print(out.head(15).to_string(index=False)); print("\nTop feature importance:"); print(imp.head(12).to_string(index=False))
 print("\nHOLDOUT AML labels NOT used. Scores are prioritization support, not AML determinations.")
 print(f"Saved: {a.out}")
if __name__=="__main__":main()
