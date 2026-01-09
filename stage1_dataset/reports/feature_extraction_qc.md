# Feature Extraction QC Report
## Run Summary
### Input / Output Paths
- **Input CSV**: `/home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv`
- **Trials parquet**: `trials.parquet`
- **Features parquet**: `features.parquet`

### Row Counts
- Initial trials loaded: 3022
- Trials after filters: 65545

### Configuration Snapshot
```yaml
detection:
  k_std: 4.5
  min_duration_s: 0.05
  min_peak_over_threshold: 0.0
features:
  export_cols:
  - per
  - latency_s
  - duration_s
  - baseline_mean
  - baseline_std
  - threshold
  - peak
  - auc_pos_s
filters:
  drop_if_non_reactive_flag: false
  drop_if_tracking_flagged: false
  max_nan_frac: 0.2
  non_reactive_flag_col: non_reactive_flag
  tracking_flag_col: tracking_flagged
paths:
  input_csv: /home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv
  model_predictions_csv: /home/ramanlab/Documents/cole/Data/Opto/Combined/model_predictions.csv
  output_dir: stage1_dataset/data
  protocol_map_csv: /home/ramanlab/Documents/cole/VSCode/fly-olfactory-learning-model/stage1_dataset/data/protocol_map.csv
  qc_dir: stage1_dataset/data/qc_plots
  reports_dir: stage1_dataset/reports
  testing_csv: /home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv
  training_csv: /home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide_training.csv
protocol:
  cs_col: cs_type
  odor_col: odor_name
  odor_off_s_col: odor_off_s
  odor_on_s_col: odor_on_s
  reward_col: reward
qc:
  dist_cols:
  - latency_s
  - duration_s
  - auc_pos_s
  - peak
  - per
  n_random_trials: 30
  stratify_cols:
  - phase
  - trial_type
  - odor_name
run:
  combine_train_test: true
  name: stage1_wide_csv_v1
  random_seed: 1337
schema:
  fly_id_format: '{dataset}::{fly}'
  phase_rules:
    testing_contains:
    - test
    training_contains:
    - train
  pulse_idx_regex: (\d+)\s*$
  required_cols:
  - dataset
  - fly
  - fly_number
  - trial_type
  - trial_label
trace:
  clip_max: null
  clip_min: null
  fill_method: none
  fps_col: fps
  trace_prefix: dir_val_
windowing:
  analysis_frac: 1.0
  baseline_frac: 0.2
```

### Required Columns Check
- Required columns in input: ['dataset', 'fly', 'fly_number', 'trial_type', 'trial_label']

### QC Artifacts Generated
- **dist_latency_s**: `dist_latency_s.png`
- **dist_duration_s**: `dist_duration_s.png`
- **dist_auc_pos_s**: `dist_auc_pos_s.png`
- **dist_peak**: `dist_peak.png`
- **dist_per**: `dist_per.png`
- **random_traces_dir**: `qc_plots`
