# AIRGUARD experiment protocol (pre-registration draft)

## Primary question
Can a conservative anomaly-screening system detect station-specific irregularities while preserving genuine regional pollution episodes?

## Output classes
1. **Potential measurement anomaly** — flag for human review; never delete automatically.
2. **Potential regional pollution event** — preserve elevated readings when comparable stations show compatible changes.
3. **Uncertain / abstain** — insufficient or contradictory evidence.

These are screening labels, not diagnoses of physical sensor condition.

## Data gate (must pass before model training)
- Source, publisher, download date, licence and file hashes recorded.
- Station identifiers and station history understood.
- Timestamp convention and cadence assessed.
- Pollutant units and scale understood by station.
- Missing codes and duplicate-key behaviour documented.
- Enough temporal coverage for blocked evaluation.
- Spatial corroboration used only if the network has enough geographically comparable stations.

If spatial comparability is not supportable, remove spatial claims and run a temporal-only study.

## Synthetic fault scenarios
Inject faults only into copies of the observed data:
- isolated spikes;
- stuck/repeated values;
- gradual drift;
- abrupt step changes;
- missing intervals.

Create separate genuine-event preservation scenarios with simultaneous increases at multiple comparable stations, including high values. High magnitude alone must never imply sensor failure.

## Methods
A. Rolling median + rolling MAD baseline.
B. Transparent rule-based checks: repeated values, abrupt changes, missingness, and conditional spatial disagreement.
C. Random Forest on engineered features, only if justified by data volume and labels.

Prefer the simplest method that performs adequately. Do not add a model merely for appearance.

## Features
- Temporal: rolling median/MAD, lagged values, rate of change.
- Stability: run length, rolling variance, missingness.
- Spatial: deviations from nearby comparable stations, only where justified by geography, timestamp alignment and (if available) meteorology.
- Metadata: pollutant, station and instrument details only where documented.

For real-time claims, features must not use future observations.

## Splitting and leakage controls
- Chronological blocks; never random row splits as the primary evaluation.
- Fit all learned transformations on training data only.
- Keep a final untouched test period.
- Use distinct fault seeds and parameter ranges for train/validation/test.
- Hold out fault patterns or parameter ranges for generalization tests.
- Station-held-out testing only if enough stations exist; otherwise disclose the limitation.
- Never use synthetic labels to construct input features.

## Metrics
- Precision, recall, F1 and PR-AUC for injected faults.
- False-positive rate on clean observations.
- Recall by fault type.
- Event-preservation rate on genuine-event simulations.
- Abstention rate and coverage-risk trade-off.
Report results per fault type and across multiple random seeds.

## Claim discipline
- Synthetic tests support claims only about tested synthetic patterns.
- Provider quality flags support agreement claims, not necessarily ground truth.
- Maintenance records or reference-monitor comparisons may provide independent evidence.
- Without independent real fault labels, real-world sensor-fault accuracy remains unvalidated.
