# Closed-Loop Learning Overview

This project targets a closed-loop algorithm (not hardware implants):

1. **Sense**: proboscis behavior now; imaging later.
2. **Infer**: estimate learning/plasticity state z_t online.
3. **Actuate**: adapt stimulus (odor/light) to enhance/restore learning.
4. **Adapt**: update z_t during the experiment (learning in the forward pass).

## Stage order: A → C → B

- **Stage A**: behavior-only closed-loop inference
- **Stage C**: imaging-constrained inference
- **Stage B**: model-driven actuation

## Current focus
Stage 1 (dataset builder) enables offline analysis of behavior traces for feature extraction and protocol standardization.
