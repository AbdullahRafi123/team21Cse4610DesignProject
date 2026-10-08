# Expanded CIFAR-10 / LightDP-FL measured report

Training images: 50,000; test images: 10,000. Main clients: 50; records/client: 1000.
Main rounds completed: 8/8; sweep rounds: 3/3.
Model: fixed public ResNet-18 at 224 pixels + trainable CNN branch/head (28,362 parameters). Not end-to-end ResNet training.
No DP accuracy: 65.72%; Vanilla local noise adding: 65.09%; LightDP: 65.86%.
LightDP minus vanilla local noise adding: +0.77 percentage points.
Normal actual epsilon: 6.000; LightDP actual epsilon: 6.000, delta=1e-05.
Measured LightDP/Normal training-time ratio: 0.996. Below 1 means faster.
Measured LightDP/Normal GPU peak allocation ratio: 1.000. Below 1 means lower memory.
SMPC+DP accuracy (ideal secure aggregation): 65.72%. Cryptographic runtime excluded from training timing.
Lightweight claim: small float uploads and no per-round dropout mask recovery. Normal independent DP may be cheaper.
Paillier sample: 1.6461s for 16 elements; projected full P encryption: 2917.89s/client. Projection is NOT measured full HE-FL.
See sweep chart for epsilon 3/6/9, client count, colluders, stragglers, clip, LR and non-IID comparisons. Short sweeps are screening evidence.
Combined fixed-partition DP-only runs epsilon bound: 33.247. Excludes changed partitions and all no-DP outputs.
Reconstruction results apply only to this fixed-label, strong auxiliary-information attack. Do not claim impossible reconstruction.
Paper-style comparison: see 07_paper_style_epsilon_comparison.png for epsilon 3/6/9.
Exact no-DP linear probe: see 00_exact_pixel_reconstruction.png. Hybrid-model reconstruction is a separate experiment.
Cached results preserve original measured training times; this session wall time includes cache load/export.
Total actual runtime: 4.70 minutes; shared feature preparation: 3.48 minutes.