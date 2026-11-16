# AIRGUARD

### Hybrid Machine Learning for Air-Quality Data Reliability

**Veritas AI International High School Hackathon 2026 — Climate Track**

AIRGUARD is an experimental machine-learning framework that screens urban air-quality measurements for potentially unreliable patterns using unsupervised learning, robust statistics, and temporal analysis.

Rather than automatically deleting unusual measurements or declaring them faulty, AIRGUARD combines complementary detection signals into an interpretable review queue for further investigation.

---

## The Problem

Air-quality monitoring networks generate large volumes of environmental observations. However, unusual measurements can have very different causes.

A sudden PM2.5 increase could represent a genuine pollution episode. Repeated identical readings over several days could indicate a measurement, communication, or data-processing issue.

A detector that treats every statistical outlier as an error risks suppressing genuine environmental events.

AIRGUARD investigates how multiple complementary detection methods can help identify suspicious patterns while preserving uncertainty.

## How It Works

AIRGUARD currently combines three primary approaches.

### 1. Isolation Forest

An unsupervised machine-learning model identifies observations that differ from learned feature distributions.

The framework investigates both:

- A global model trained across monitoring stations.
- Station-specific models trained separately for individual stations.

The model uses seven engineered features:

- Measured PM2.5 value
- Past rolling median
- Past rolling median absolute deviation
- Robust deviation
- Absolute change
- Change rate
- Repeated-value run length

The global model uses 200 estimators, a contamination parameter of 0.01, and a fixed random seed of 42.

### 2. Robust MAD Detection

A past-only rolling median and median absolute deviation provide a robust statistical baseline.

This method helps identify observations that deviate substantially from recent historical measurements.

The baseline uses a 24-hour historical window and a robust deviation threshold of 6. The implementation uses past observations rather than future measurements when evaluating historical observations.

### 3. Temporal Flatline Detection

AIRGUARD identifies consecutive identical hourly measurements.

The current detector uses a minimum threshold of 12 consecutive identical hourly readings to identify potential flatline episodes.

These patterns are screening candidates, not confirmed sensor failures.

## Unified Review Queue

The outputs of the detection methods are consolidated into a review queue.

Current categories include:

| Category | Meaning |
|---|---|
| MULTI_SIGNAL | Multiple detection methods flagged the observation |
| FLATLINE_PATTERN | A prolonged repeated-value pattern was identified |
| STATISTICAL_OUTLIER | A statistical or machine-learning detector flagged the observation |
| NO_FLAG | No detector flagged the observation |

The review queue preserves detector signals instead of automatically removing observations.

## Dataset

The project uses publicly available Mumbai PM2.5 monitoring observations from January–December 2024.

**Primary source:** [XKDR India Air Quality Database](https://airquality.xkdr.org/)

| Property | Value |
|---|---:|
| Hourly observations | 65,757 |
| Monitoring stations in full dataset | 9 |
| Engineered ML features | 7 |
| Training observations | 43,409 |
| Testing observations | 20,540 |

The chronological evaluation split is:

- Training: through 31 August 2024.
- Testing: from 1 September 2024 onward.

The test period contains eight stations with observations available for evaluation.

The project also inspects additional Sion observations from OpenCity. These observations share source lineage with the primary dataset and are not treated as independent validation. Their hourly aggregation reproduced 8,244 of 8,244 complete matching hours within a difference of less than 0.01.

## Real-Data Observations

The flatline detector identified 13 episodes across the 2024 dataset.

Two illustrative examples:

**Sion — September 2024**

A channel recorded exactly 18.00 across 70 consecutive hourly observations between 20 September and 22 September.

**Kurla — October 2024**

A channel recorded exactly 17.56 across 75 consecutive hourly observations between 21 October and 24 October.

These episodes are flagged for investigation. The available measurements alone do not establish whether the cause was a physical sensor fault, a communication issue, or another data-path problem.

## Held-Out Review Results

The unified review queue was evaluated on the 20,540-observation chronological test period.

| Result | Count |
|---|---:|
| Review candidates | 717 |
| Review-candidate share | 3.49% |
| Isolation Forest flags | 272 |
| Robust MAD flags | 250 |
| Flatline-pattern observations | 239 |
| Multi-signal observations | 44 |

These are screening outputs, not confirmed erroneous measurements. The 44 multi-signal observations represent agreement between multiple detection methods.

## Synthetic Fault Validation

Real-world sensor-failure labels are not available for every observation. Therefore, AIRGUARD also uses controlled synthetic fault injection to evaluate detector behaviour.

Three fault types were tested:

1. Isolated spikes.
2. Short-duration flatlines.
3. Longer stuck-at measurements.

Eight events of each type were injected, producing 24 synthetic events.

### Preliminary Results

| Detector | Isolated spikes | Short flatlines | Stuck-at events |
|---|---:|---:|---:|
| Global Isolation Forest | 25% | 0% | 25% |
| Station-specific Isolation Forest | 75% | 0% | 37.5% |
| Robust MAD | 100% | 0% | 0% |
| Flatline detector | 0% | 100% | 100% |

These values represent event-level recall in this particular controlled experiment. Each fault type was tested with only eight events.

The experiment uses baseline-clean injection windows and excludes injected observations and subsequent feature-contamination periods from the untouched-observation denominator.

The untouched flag rate is not a true false-positive rate because untouched observations do not have verified ground-truth labels.

The results demonstrate that different methods detect different fault patterns. They do not establish generalized real-world detection accuracy.

## Model Comparison and Reproducibility

Global and station-specific Isolation Forest models were compared on the same chronological test period.

| Model | Test observations flagged |
|---|---:|
| Global Isolation Forest | 272 |
| Station-specific Isolation Forest | 416 |

The station-specific model changes the distribution of flags across stations. A higher number of flags does not by itself imply better detection.

A reproducibility check independently reconstructed the global model evaluation pipeline and confirmed:

- All 20,540 test observations matched.
- All seven model features matched.
- Global model predictions matched exactly.
- Maximum anomaly-score difference was 0.0.

## Repository Structure

```text
AIRGUARD/
│
├── data/
│   └── raw/
│       ├── xkdr/
│       └── opencity/
│
├── scripts/
│   ├── train_airguard.py
│   ├── analyze_temporal_baseline.py
│   ├── detect_temporal_anomalies.py
│   ├── detect_flatline_episodes.py
│   ├── build_review_queue.py
│   ├── compare_station_models.py
│   ├── check_model_reproducibility.py
│   └── validate_synthetic_faults.py
│
└── reports/
    ├── temporal_baseline/
    ├── pm25_value_audit/
    └── airguard_ml/
```

Additional data-inspection and analysis scripts are included in `scripts/`.

Raw datasets are not committed to the repository. Generated reports and evaluation outputs are stored under `reports/`.

## Reproducibility

Install the dependencies listed in the repository's dependency file:

```bash
python -m pip install -r requirements.txt
```

Obtain the 2024 Mumbai PM2.5 data from [XKDR](https://airquality.xkdr.org/) and place the monthly CSV files under:

```text
data/raw/xkdr/2024/
```

Place the supplementary Sion dataset under:

```text
data/raw/opencity/
```

Run the analysis scripts from the repository root. For example:

```bash
python scripts/train_airguard.py
python scripts/analyze_temporal_baseline.py
python scripts/detect_flatline_episodes.py
python scripts/build_review_queue.py
python scripts/validate_synthetic_faults.py
python scripts/compare_station_models.py
python scripts/check_model_reproducibility.py
```

The synthetic validation results are recorded in:

```text
reports/airguard_ml/synthetic_fault_validation.csv
```

The unified review queue is generated using:

```bash
python scripts/build_review_queue.py
```

Existing reports are experimental outputs from the current analysis. Running every script requires the relevant original datasets and dependencies.

## Limitations

AIRGUARD is a research prototype, not a certified sensor-diagnostics system.

Current limitations include:

- No comprehensive ground-truth labels for real sensor failures.
- Synthetic faults cannot represent every real-world failure mechanism.
- Genuine pollution events can trigger anomaly detectors.
- Repeated values may arise from sensor behaviour, data pipelines or other causes.
- Station-specific models may behave differently from global models.
- Results from Mumbai stations may not generalize to other monitoring networks.
- Detection thresholds require further investigation.

AIRGUARD should therefore be understood as a screening and prioritization framework, not an automatic sensor-failure diagnosis system.

## Future Work

Potential extensions include:

- Neural reconstruction-based anomaly detection.
- Evaluation on independently labelled sensor-failure datasets.
- Cross-city validation.
- Improved temporal and spatial modelling.
- More extensive threshold-sensitivity experiments.

Any additional model should be evaluated against the existing detectors before being incorporated into the review queue.

## Technology and Credits

- Python
- NumPy
- pandas
- scikit-learn
- XKDR public Mumbai air-quality data
- OpenCity data for additional Sion inspection
- ChatGPT and Codex for AI-assisted development

Dataset and library references should be linked to their original sources in the final submission.

## Project Information

**Project:** AIRGUARD  
**Track:** Climate  
**Developer:** Rishav Anand Kumar Jha  
**Institution:** Shree L. R. Tiwari Junior College of Science and Commerce  
**Hackathon:** Veritas AI International High School Hackathon 2026

---

*AIRGUARD does not assume that unusual measurements are errors. It identifies patterns worth investigating and preserves the distinction between an anomaly, a potential data-quality issue, and a confirmed sensor failure.*