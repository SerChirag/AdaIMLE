# ICLR rebuttal measurement scripts

Paths inside the scripts are hard-coded for the cs-apex machine; edit them before running elsewhere.

| Script | What it measures |
|---|---|
| `sample_common.py` | Shared setup: generator + eq-vae-ema decoder, fp32/bf16, eager/compiled |
| `bench_sampling.py` | Sampling throughput per stage (generator, VAE, end-to-end), eager vs compiled, fp32 vs bf16 |
| `bench_e2e.py` | End-to-end throughput (compiled bf16 generator + each VAE decode mode) |
| `gen_vae_precision.py` | 50k samples decoded with fp32 and bf16 VAE from the same latents, for the FID check |
| `dino_feats.py` | DINOv2-L features (FD-Loss paper protocol, uses its `frechet_distance/repr_models.py`) of a sample folder, sharded over torchrun ranks |
| `dino_merge_fd.py` | Merges the shards, saves features, computes FD-DINOv2 against the FD-Loss ImageNet reference stats |
| `dino_prc.py` | DINOv2 features of the 10k ADM/VIRTUAL reference images, then k=3 precision/recall |
| `safe_unpack_refstats.py` | Unpacks FD-Loss `paper_ref_stats.pkl` with a numpy-only unpickler |
| `make_grid.py` | Uncurated 16x8 sample grid |

Results (Nov 2026; FD target for the FD-loss model = full-ImageNet FID reference statistics):

| Model | FID | FD-DINOv2 | DINOv2 P / R |
|---|---|---|---|
| distill iter-56910 EMA (start) | 2.869 | 45.9 | |
| FD-loss iter-60000 | 0.831 | 53.8 | 0.878 / 0.450 |
| iMF-M/2 (1 NFE, cfg 10.5) | 2.27 (JiT ref) | 147.9 | |

Throughput on one L40S: generator 410 img/s, generator + bf16 VAE decode 147 img/s (fp32 decode 82 img/s).
