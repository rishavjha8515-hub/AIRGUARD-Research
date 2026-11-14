# AIRGUARD real-data intake checklist

## Provenance
- [ ] Publisher and original source URL
- [ ] Download date/time (IST)
- [ ] Licence and attribution requirements
- [ ] File names, byte sizes and SHA-256 hashes
- [ ] Original files preserved unchanged

## Inventory
- [ ] File format, row/column counts
- [ ] Observation schema and data dictionary
- [ ] Station metadata and station history
- [ ] Pollutants and units per station
- [ ] First/last timestamps per station and pollutant

## Quality
- [ ] Timestamp parse failures and timezone convention
- [ ] Cadence distribution and gaps
- [ ] Missingness and sentinel tokens
- [ ] Identical and conflicting duplicate keys
- [ ] Non-numeric values
- [ ] Negative and extreme values (flag only; do not delete)
- [ ] Unit spelling vs actual unit conversion distinguished
- [ ] Coordinates, precision and station identity ambiguity
- [ ] Quality flags, maintenance and calibration metadata

## Spatial feasibility
- [ ] Station coordinates validated
- [ ] Distances calculated with coordinate precision caveat
- [ ] Nearby station overlap quantified
- [ ] Correlation after removing common diurnal patterns
- [ ] Meteorology considered where available
- [ ] No automatic assumption that correlation means same airmass

## Decision
- Temporal anomaly screening: GO / LIMITED / NO-GO
- Spatial corroboration: GO / LIMITED / NO-GO
- Independent fault-label validation: YES / NO
- Overall: GO / CONDITIONAL GO / NO-GO
- Reasons and unresolved issues:
