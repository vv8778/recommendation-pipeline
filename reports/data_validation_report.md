# Automated Data Validation & Quality Audit Report

**Generated At:** 2026-11-07T17:54:51.942073  
**Status:** `PASSED_WITH_QUARANTINE`  
**Pipeline Stage:** Data Ingestion & Validation Gate

---

## 1. Executive Summary

| Metric | Count | Percentage |
| :--- | :--- | :--- |
| **Total Raw Interactions Ingested** | 28,861 | 100.0% |
| **Clean Valid Interactions Passed** | 28,265 | 97.94% |
| **Quarantined Error Records** | 594 | 2.06% |
| **Valid Active Users** | 1,000 | - |
| **Valid Catalog Items** | 500 | - |

---

## 2. Validation Check Matrix

| Check Name | Target Entity | Quarantined Records | Status | Audit Description |
| :--- | :--- | :--- | :--- | :--- |
| **Missing Values Check** | `interactions` | 116 | ⚠️ QUARANTINED | Automated gate check |
| **Duplicate Check** | `interactions` | 62 | ⚠️ QUARANTINED | Automated gate check |
| **Rating Boundary Check** | `interactions` | 115 | ⚠️ QUARANTINED | Automated gate check |
| **Timestamp Validity Check** | `interactions` | 115 | ⚠️ QUARANTINED | Automated gate check |
| **User Referential Integrity** | `interactions` | 116 | ⚠️ QUARANTINED | Automated gate check |
| **Item Referential Integrity** | `interactions` | 116 | ⚠️ QUARANTINED | Automated gate check |

---

## 3. Quarantined Error Distribution & Root Causes

All anomalous records that failed validation constraints were immediately isolated into the error dataset (`data/errors/`) to prevent downstream model bias and pipeline crashes.

### Sample of Quarantined Records

|   user_id |   item_id |   rating |   timestamp | error_reason                         |
|----------:|----------:|---------:|------------:|:-------------------------------------|
|       271 |       279 |      nan |  1734692987 | Missing values in interaction record |
|       572 |       111 |      nan |  1759946678 | Missing values in interaction record |
|        53 |        42 |      nan |  1710141255 | Missing values in interaction record |
|       886 |       158 |      nan |  1736869096 | Missing values in interaction record |
|       279 |       257 |      nan |  1751008207 | Missing values in interaction record |
|       957 |       143 |      nan |  1718399091 | Missing values in interaction record |
|       271 |       483 |      nan |  1736415941 | Missing values in interaction record |
|         3 |       443 |      nan |  1742631220 | Missing values in interaction record |
|       549 |       141 |      nan |  1727414247 | Missing values in interaction record |
|       271 |       233 |      nan |  1707797273 | Missing values in interaction record |

---

## 4. Referential Integrity & Boundary Verification

1. **User Referential Integrity:** Verified that every user in interactions exists in the validated user catalog. Orphaned IDs were routed to `data/errors/invalid_interactions.csv`.
2. **Item Referential Integrity:** Verified that every item in interactions exists in the validated item catalog.
3. **Rating Boundaries:** Ensured ratings are strictly within [1.0, 5.0].
4. **Timestamp Sanity:** Filtered out negative or future timestamps (e.g. year 2035 test injections).
5. **Deduplication:** Dropped duplicate `(user_id, item_id)` interaction pairs.
