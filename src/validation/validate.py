import os
import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

class DataValidator:
    """
    Automated data quality validation and quarantine layer.
    Validates schemas, data types, value boundaries, nulls, duplicates, and referential integrity.
    """
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.raw_dir = self.config["paths"]["raw_data_dir"]
        self.errors_dir = self.config["paths"]["errors_data_dir"]
        self.reports_dir = self.config["paths"]["reports_dir"]
        
        os.makedirs(self.errors_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        self.min_rating = float(self.config["validation"].get("min_rating", 1.0))
        self.max_rating = float(self.config["validation"].get("max_rating", 5.0))
        self.current_timestamp = int(datetime.now().timestamp())

    def validate(self):
        """
        Executes automated validation rules on raw users, items, and interactions.
        Quarantines invalid rows to data/errors/ and writes data_validation_report.md.
        """
        print("[Validation] Starting automated data validation checks...")

        users_path = os.path.join(self.raw_dir, "users.csv")
        items_path = os.path.join(self.raw_dir, "items.csv")
        interactions_path = os.path.join(self.raw_dir, "interactions.csv")

        users_df = pd.read_csv(users_path)
        items_df = pd.read_csv(items_path)
        interactions_df = pd.read_csv(interactions_path)

        validation_report = {
            "execution_timestamp": datetime.now().isoformat(),
            "status": "PASSED_WITH_QUARANTINE",
            "summary": {},
            "checks": []
        }

        # -------------------------------------------------------------
        # 1. User Validation
        # -------------------------------------------------------------
        user_errors = []
        # Missing values check
        user_nulls = users_df[users_df[["user_id", "gender", "age"]].isnull().any(axis=1)]
        if not user_nulls.empty:
            user_errors.append(user_nulls.assign(error_reason="Missing required user fields"))

        # Age bounds check
        invalid_age = users_df[(users_df["age"] <= 0) | (users_df["age"] > 120)]
        if not invalid_age.empty:
            user_errors.append(invalid_age.assign(error_reason="Age out of valid range (1-120)"))

        # Duplicates check
        user_dups = users_df[users_df.duplicated(subset=["user_id"], keep=False)]
        if not user_dups.empty:
            user_errors.append(user_dups.assign(error_reason="Duplicate user_id"))

        if user_errors:
            invalid_users_df = pd.concat(user_errors).drop_duplicates(subset=["user_id"])
            valid_users_df = users_df[~users_df["user_id"].isin(invalid_users_df["user_id"])].copy()
            invalid_users_df.to_csv(os.path.join(self.errors_dir, "invalid_users.csv"), index=False)
        else:
            invalid_users_df = pd.DataFrame()
            valid_users_df = users_df.copy()

        # -------------------------------------------------------------
        # 2. Item Validation
        # -------------------------------------------------------------
        item_errors = []
        item_nulls = items_df[items_df[["item_id", "title"]].isnull().any(axis=1)]
        if not item_nulls.empty:
            item_errors.append(item_nulls.assign(error_reason="Missing required item fields"))

        item_dups = items_df[items_df.duplicated(subset=["item_id"], keep=False)]
        if not item_dups.empty:
            item_errors.append(item_dups.assign(error_reason="Duplicate item_id"))

        if item_errors:
            invalid_items_df = pd.concat(item_errors).drop_duplicates(subset=["item_id"])
            valid_items_df = items_df[~items_df["item_id"].isin(invalid_items_df["item_id"])].copy()
            invalid_items_df.to_csv(os.path.join(self.errors_dir, "invalid_items.csv"), index=False)
        else:
            invalid_items_df = pd.DataFrame()
            valid_items_df = items_df.copy()

        # -------------------------------------------------------------
        # 3. Interactions Validation
        # -------------------------------------------------------------
        interaction_errors = []

        # A. Missing values check
        null_mask = interactions_df[["user_id", "item_id", "rating", "timestamp"]].isnull().any(axis=1)
        if null_mask.any():
            bad_nulls = interactions_df[null_mask].assign(error_reason="Missing values in interaction record")
            interaction_errors.append(bad_nulls)
            validation_report["checks"].append({
                "check_name": "Missing Values Check",
                "target": "interactions",
                "failed_records": int(null_mask.sum()),
                "status": "FAILED_QUARANTINED"
            })
        else:
            validation_report["checks"].append({
                "check_name": "Missing Values Check",
                "target": "interactions",
                "failed_records": 0,
                "status": "PASSED"
            })

        # B. Duplicate interaction check (user_id, item_id)
        dup_mask = interactions_df.duplicated(subset=["user_id", "item_id"], keep="first")
        if dup_mask.any():
            bad_dups = interactions_df[dup_mask].assign(error_reason="Duplicate user-item interaction")
            interaction_errors.append(bad_dups)
            validation_report["checks"].append({
                "check_name": "Duplicate Check",
                "target": "interactions",
                "failed_records": int(dup_mask.sum()),
                "status": "FAILED_QUARANTINED"
            })
        else:
            validation_report["checks"].append({
                "check_name": "Duplicate Check",
                "target": "interactions",
                "failed_records": 0,
                "status": "PASSED"
            })

        # C. Invalid rating check (must be within [min_rating, max_rating])
        rating_mask = (interactions_df["rating"] < self.min_rating) | (interactions_df["rating"] > self.max_rating)
        # Note: fillna false to avoid double counting nulls
        rating_mask = rating_mask.fillna(False)
        if rating_mask.any():
            bad_ratings = interactions_df[rating_mask].assign(error_reason=f"Rating out of valid range [{self.min_rating}, {self.max_rating}]")
            interaction_errors.append(bad_ratings)
            validation_report["checks"].append({
                "check_name": "Rating Boundary Check",
                "target": "interactions",
                "failed_records": int(rating_mask.sum()),
                "status": "FAILED_QUARANTINED"
            })
        else:
            validation_report["checks"].append({
                "check_name": "Rating Boundary Check",
                "target": "interactions",
                "failed_records": 0,
                "status": "PASSED"
            })

        # D. Invalid timestamp check (cannot be future timestamp or non-positive)
        # Add small buffer for slight clock skews
        future_limit = self.current_timestamp + (30 * 86400)
        time_mask = (interactions_df["timestamp"] <= 0) | (interactions_df["timestamp"] > future_limit)
        time_mask = time_mask.fillna(False)
        if time_mask.any():
            bad_times = interactions_df[time_mask].assign(error_reason="Timestamp is in the future or non-positive")
            interaction_errors.append(bad_times)
            validation_report["checks"].append({
                "check_name": "Timestamp Validity Check",
                "target": "interactions",
                "failed_records": int(time_mask.sum()),
                "status": "FAILED_QUARANTINED"
            })
        else:
            validation_report["checks"].append({
                "check_name": "Timestamp Validity Check",
                "target": "interactions",
                "failed_records": 0,
                "status": "PASSED"
            })

        # E. Referential Integrity Checks
        # User referential integrity
        valid_user_set = set(valid_users_df["user_id"])
        ref_user_mask = ~interactions_df["user_id"].isin(valid_user_set)
        ref_user_mask = ref_user_mask & (~null_mask)
        if ref_user_mask.any():
            bad_user_ref = interactions_df[ref_user_mask].assign(error_reason="Referential Integrity: user_id not found in users catalog")
            interaction_errors.append(bad_user_ref)
            validation_report["checks"].append({
                "check_name": "User Referential Integrity",
                "target": "interactions",
                "failed_records": int(ref_user_mask.sum()),
                "status": "FAILED_QUARANTINED"
            })
        else:
            validation_report["checks"].append({
                "check_name": "User Referential Integrity",
                "target": "interactions",
                "failed_records": 0,
                "status": "PASSED"
            })

        # Item referential integrity
        valid_item_set = set(valid_items_df["item_id"])
        ref_item_mask = ~interactions_df["item_id"].isin(valid_item_set)
        ref_item_mask = ref_item_mask & (~null_mask)
        if ref_item_mask.any():
            bad_item_ref = interactions_df[ref_item_mask].assign(error_reason="Referential Integrity: item_id not found in items catalog")
            interaction_errors.append(bad_item_ref)
            validation_report["checks"].append({
                "check_name": "Item Referential Integrity",
                "target": "interactions",
                "failed_records": int(ref_item_mask.sum()),
                "status": "FAILED_QUARANTINED"
            })
        else:
            validation_report["checks"].append({
                "check_name": "Item Referential Integrity",
                "target": "interactions",
                "failed_records": 0,
                "status": "PASSED"
            })

        # Compile quarantined interactions
        if interaction_errors:
            all_invalid_interactions = pd.concat(interaction_errors).drop_duplicates(subset=["user_id", "item_id", "timestamp"])
            invalid_indices = all_invalid_interactions.index.unique()
            valid_interactions_df = interactions_df.drop(index=invalid_indices, errors="ignore").copy()
            # Also ensure no remaining bad users/items
            valid_interactions_df = valid_interactions_df[
                valid_interactions_df["user_id"].isin(valid_user_set) &
                valid_interactions_df["item_id"].isin(valid_item_set) &
                valid_interactions_df["rating"].between(self.min_rating, self.max_rating) &
                (valid_interactions_df["timestamp"] <= future_limit) &
                (valid_interactions_df["timestamp"] > 0)
            ].drop_duplicates(subset=["user_id", "item_id"])

            invalid_interactions_path = os.path.join(self.errors_dir, "invalid_interactions.csv")
            all_invalid_interactions.to_csv(invalid_interactions_path, index=False)
        else:
            valid_interactions_df = interactions_df.copy()
            all_invalid_interactions = pd.DataFrame()

        # Update summary statistics
        total_raw = len(interactions_df)
        total_valid = len(valid_interactions_df)
        total_quarantined = len(all_invalid_interactions)
        error_rate = (total_quarantined / total_raw) * 100 if total_raw > 0 else 0.0

        validation_report["summary"] = {
            "total_raw_interactions": total_raw,
            "total_valid_interactions": total_valid,
            "total_quarantined_interactions": total_quarantined,
            "quarantine_error_rate_pct": round(error_rate, 2),
            "total_valid_users": len(valid_users_df),
            "total_valid_items": len(valid_items_df),
            "quarantined_users_count": len(invalid_users_df),
            "quarantined_items_count": len(invalid_items_df)
        }

        # Save JSON report
        json_report_path = os.path.join(self.reports_dir, "data_validation_report.json")
        with open(json_report_path, "w") as f:
            json.dump(validation_report, f, indent=2)

        # Generate human-readable Markdown validation report
        md_report_path = os.path.join(self.reports_dir, "data_validation_report.md")
        self._write_markdown_report(md_report_path, validation_report, all_invalid_interactions)

        print(f"[Validation] Automated validation complete:")
        print(f"  - Total Raw Interactions: {total_raw:,}")
        print(f"  - Quarantined Anomalous Records: {total_quarantined:,} ({error_rate:.2f}%)")
        print(f"  - Clean Valid Records Passed: {total_valid:,}")
        print(f"  - Quarantined files saved to: {self.errors_dir}/")
        print(f"  - Validation report generated at: {md_report_path}")

        return {
            "valid_users_df": valid_users_df,
            "valid_items_df": valid_items_df,
            "valid_interactions_df": valid_interactions_df,
            "report": validation_report
        }

    def _write_markdown_report(self, path: str, report: dict, invalid_df: pd.DataFrame):
        summary = report["summary"]
        checks = report["checks"]
        
        md_content = f"""# Automated Data Validation & Quality Audit Report

**Generated At:** {report["execution_timestamp"]}  
**Status:** `{report["status"]}`  
**Pipeline Stage:** Data Ingestion & Validation Gate

---

## 1. Executive Summary

| Metric | Count | Percentage |
| :--- | :--- | :--- |
| **Total Raw Interactions Ingested** | {summary["total_raw_interactions"]:,} | 100.0% |
| **Clean Valid Interactions Passed** | {summary["total_valid_interactions"]:,} | {100.0 - summary["quarantine_error_rate_pct"]:.2f}% |
| **Quarantined Error Records** | {summary["total_quarantined_interactions"]:,} | {summary["quarantine_error_rate_pct"]:.2f}% |
| **Valid Active Users** | {summary["total_valid_users"]:,} | - |
| **Valid Catalog Items** | {summary["total_valid_items"]:,} | - |

---

## 2. Validation Check Matrix

| Check Name | Target Entity | Quarantined Records | Status | Audit Description |
| :--- | :--- | :--- | :--- | :--- |
"""
        for c in checks:
            badge = "✅ PASS" if c["status"] == "PASSED" else "⚠️ QUARANTINED"
            md_content += f"| **{c['check_name']}** | `{c['target']}` | {c['failed_records']:,} | {badge} | Automated gate check |\n"

        md_content += """
---

## 3. Quarantined Error Distribution & Root Causes

All anomalous records that failed validation constraints were immediately isolated into the error dataset (`data/errors/`) to prevent downstream model bias and pipeline crashes.

### Sample of Quarantined Records
"""
        if not invalid_df.empty:
            sample = invalid_df.head(10)[["user_id", "item_id", "rating", "timestamp", "error_reason"]]
            md_content += "\n" + sample.to_markdown(index=False) + "\n"
        else:
            md_content += "\n*No quarantined records found.*\n"

        md_content += """
---

## 4. Referential Integrity & Boundary Verification

1. **User Referential Integrity:** Verified that every user in interactions exists in the validated user catalog. Orphaned IDs were routed to `data/errors/invalid_interactions.csv`.
2. **Item Referential Integrity:** Verified that every item in interactions exists in the validated item catalog.
3. **Rating Boundaries:** Ensured ratings are strictly within [1.0, 5.0].
4. **Timestamp Sanity:** Filtered out negative or future timestamps (e.g. year 2035 test injections).
5. **Deduplication:** Dropped duplicate `(user_id, item_id)` interaction pairs.
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_content)

if __name__ == "__main__":
    validator = DataValidator()
    validator.validate()
