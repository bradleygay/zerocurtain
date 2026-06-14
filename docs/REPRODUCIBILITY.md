# GeoCryoAI — Reproducibility & Manuscript Crosswalk

Companion to "Resolving Circumarctic Zero-Curtain Phenomena with AI-Integrated
Earth Observations" (Scientific Reports, COMMSENV-25-5162-T).

## Two-model architecture (do not conflate)
1. Base model (teacher-forced on PINSZC) — src/part2_geocryoai/zero_curtain_ml_model_discover.py
   - Heads: intensity nn.Sigmoid() [0,1]; duration nn.ReLU() (hours); extent nn.ReLU() (meters)
   - Fusion: Liquid(128) + Spatial U-Net(d_model//8 = 32) + Physics(d_model//8 = 32) = 192 -> Linear(192, 256)
   - ~1,084,106 parameters; d_model=256, liquid_hidden=128, 8 attention heads
   - This is the configuration depicted in Figure 5, Panel A.
2. Transfer model (deployed; domain generalization to PIRSZC) — scripts/part4_transfer_learning.py
   - Heads: intensity Sigmoid [0,1]; duration & extent Softplus in log1p space
   - log1p forward transform; expm1 back-transform to physical units
   - Raw log-space numerical guard within the network; the 4,500-hour physical-plausibility
     ceiling is applied DOWNSTREAM as a quality-control mask, not inside the network.
   - Produces the deployed/mapped predictions.

Canonical references: model = zero_curtain_ml_model_discover.py; transfer driver = part4_transfer_learning.py.
zero_curtain_ml_model.py (prototype) and zero_curtain_ml_model_update.py (lazy-loading fork) differ only
in device configuration, data-loader parameters, memory handling, and I/O paths — no architectural
(head/fusion/parameter) divergence — and are retained as archival variants.

## Figure 5 mapping
- Panel A boxes <-> base-model modules; head activations Sigmoid/ReLU/ReLU.
- Panel B <-> training/convergence curves; SHAP gradient-magnitude feature importance.
- Diagram generator: geocryoai_visualization.py (standalone; not in this repo tree).

## Reported metrics (deployed transfer run, n = 239,636)
- Detection accuracy 0.964 (95% CI [0.962, 0.966]); recall/TPR 0.923; FPR 0.019; intensity percentile RMSE 0.104
- Duration RMSE 630.72 h; MAE 118.33 h; NSE 0.847; median 96 h; skew 9.82; kurtosis 97.82
- Spatial extent MAE 0.135 m; RMSE 0.278 m; bias 0.003 m
- Thermodynamic compliance 0.82 (282,327 / 345,033); energy 0.98 / temperature 0.95 / moisture 0.89
- Independent validation (GTN-P/CALM): 94.2% within 95% CI, n = 268 sites; site-based CV < 2%
- Transfer degradation (PIRSZC 2015-2024): accuracy 95.1%; duration RMSE 687.45 h; extent MAE 0.142 m; 12% QC-rejected
- ROC-AUC is undefined for the deployed single-threshold detector and is intentionally not reported.

## Count taxonomy (kept distinct)
- 54,418,117 candidate screening pool  !=  54,408,431 ablation event set
- 345,033 detected -> 282,327 physics-compliant
- PIRSZC pool 18,602,831 -> 12,533,447 retained after physical filtering
- 2,418,851 transfer candidate detections -> 239,636 validated/mapped
- 62.71 million deduplicated in-situ measurements

## Duration ceiling
- ~4,488 h latent-heat thermodynamic maximum (Outcalt et al., 1990)
- ~4,320 h longest field-verified GTN-P borehole event
- 4,500 h rounded operational QC ceiling (downstream mask)
- 6,570 h pre-truncation event-merging extremum (excluded)

## Supplementary table numbering (canonical)
- S2.3 masking sensitivity (Moran's I 0.344 -> 0.1635, z = 1170.10, k = 8)
- S2.4 hyperparameters (sequence_length=90, batch_size=128, 70/15/15 split)
- S2.5 ablation (baseline MLP 81.6%, -11.8 pp; physics removal 95.0% / compliance 94.0 -> 87.1)
- S3.5 Mann-Kendall (PINSZC 1891-2024 significant; PIRSZC/Transfer 2015-2024 non-significant)
- S3.6 statistical power (>= 30 yr required)

## Visualization / colormap policy
- No Gaussian smoothing. Interpolation: RBF + thin-plate-spline, volume-preserving (area-to-point) redistribution.
- No pcolormesh for gridded fields; LinearSegmentedColormap; no viridis/cividis/RdBu for SQ figures.
