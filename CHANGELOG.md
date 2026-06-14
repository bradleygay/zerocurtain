# Changelog


## [revision-round] - 2026-06-13
### Documentation
- Added docs/REPRODUCIBILITY.md: two-model architecture, canonical file designation,
  Figure 5 mapping, metric/count-taxonomy provenance, duration-ceiling rationale,
  SI table numbering, and visualization policy.
### Code clarity (no behavioral change)
- part4_transfer_learning.py: corrected the duration-clamp comment to distinguish the
  raw log1p-space numerical guard from the 4,500 h downstream QC ceiling.
- zero_curtain_ml_model_discover.py: removed vestigial transformers/torchvision imports
  (unused; avoids implying an undisclosed pretrained-transformer dependency).
### Notes
- No model weights, hyperparameters, or reported metrics changed.
- Deployed model: zero_curtain_ml_model_discover.py; transfer driver: part4_transfer_learning.py.
