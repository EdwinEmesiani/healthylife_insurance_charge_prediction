---
name: "Monitoring / Drift issue"
about: "Report anomalies in model or data drift metrics"
title: "[DRIFT] Short summary"
labels: ["monitoring", "drift", "mlops"]
assignees: []
---

## Type of drift / monitoring issue

- [ ] Model drift (predictions vs training targets)
- [ ] Data drift (feature distributions changing)
- [ ] Logging issue (missing logs / incorrect format)
- [ ] Dashboard / analysis notebook problem

## Summary

Describe what drift or monitoring issue you’re seeing.

## Evidence

- Any plots (histograms, KDEs, etc.)?
- Any metrics (mean/variance shifts, KS tests, etc.)?
- Screenshots or snippets from logs / notebooks.

## When did this start?

Approximate time window or dataset segment where you observed it.

## Potential Impact

What could this drift affect?  
(e.g. “Charges underpriced for high-BMI non-smokers”, “More young customers than in training data”).

## Suggested Actions (if any)

- [ ] Retrain model
- [ ] Update thresholds
- [ ] Update monitoring code
- [ ] Investigate data pipeline

## Additional Context

Links to relevant runs, PRs, or Space builds.
