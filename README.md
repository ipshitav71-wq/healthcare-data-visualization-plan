# Reducing 30-Day Hospital Readmissions: Analytical Report (Simulated Data)

A hypothetical healthcare analytics project that shows how routine hospital data can be turned into evidence for strategy and policy decisions.

**Author:** Ipshita Verma

> **Important:** All data are **simulated**. No real patients, hospitals or organisations are represented. The results demonstrate a method and must not be used as real-world estimates.

## What this project does

1. Simulates a four-hospital discharge dataset (about 25,000 records, Sep 2024 - Aug 2026) with realistic problems such as duplicates and missing values.
2. Cleans the data and reports every exclusion.
3. Analyses trends (control charts), drivers (logistic regression), and the role of 7-day follow-up.
4. Builds and validates a readmission risk model (temporal split, calibration, decision curve) and audits it for fairness.
5. Estimates the possible impact of restoring follow-up and turns the findings into recommendations.

## Folder structure

| Path | Contents |
|---|---|
| `docs/` | Full report (`.docx` and `.pdf`) |
| `analysis.py` | One script: simulation, cleaning, analysis, model, figures, results |
| `data/` | Simulated raw extract and cleaned analytic cohort (CSV) |
| `images/` | Figures used in the report |
| `results.json` | Every number reported, saved by the script |

## How to reproduce

```bash
pip install -r requirements.txt
python analysis.py
```

Random seeds are fixed, so the script recreates the same data, tables and figures.

## Preview

![Control charts](images/f1_trend.png)
![Follow-up and readmission](images/f3_followup.png)

## Tools

Python (pandas, SciPy, statsmodels, scikit-learn, matplotlib).
