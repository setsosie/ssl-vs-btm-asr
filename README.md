# ssl-vs-btm-asr

**Is it self-supervised pretraining or the Branch-Train-Merge pipeline that
carries low-resource multilingual ASR?**

A clean, multi-seed ablation on the [XEUS](https://arxiv.org/abs/2407.00837)
encoder. We compare three training conditions across 3 / 16 / 64 languages and
on five held-out Indic languages the model never saw:

| Arm | Init | Pipeline | Question it answers |
|-----|------|----------|---------------------|
| **A. SSL-only** | XEUS SSL | per-language fine-tune | value of SSL alone |
| **B. BTM-on-SSL** | XEUS SSL | phase 0 → experts → merge | does BTM add anything on top of SSL? |
| **C. BTM-from-scratch** | random | phase 0 → experts → merge | does BTM work *without* SSL? |

Everything is run with **5 seeds**, reported as **mean ± std** with bootstrap
CIs and paired-permutation tests. All hyperparameters are auto-recorded next to
every result.

**Data.** Held-out Indic transfer uses **OpenSLR** (open, auto-fetched from the
HF Hub). The training scales use **Common Voice 25**, which since October 2025 is
distributed only via [Mozilla Data Collective](https://commonvoice.mozilla.org/en/datasets)
(account + ToS, no longer on the HF Hub). Download + extract CV25 and set
`CV_ROOT` so `$CV_ROOT/<lang>/{train,dev,test}.tsv` exist — the loader reads that
standard layout (`src/svb/data/commonvoice_local.py`).

## Quickstart

```bash
uv sync                                   # install
bash scripts/download_data.sh             # fetch public data (OpenSLR, CV)
svb run --arm A_ssl --scale 3 --seed 0    # one run, end to end
make exp ARM=A_ssl SCALE=3                # all seeds for one arm/scale
make aggregate SCALE=3                     # mean±std table for a scale
make tables figures                        # regenerate paper tables/figures
```

You need the XEUS SSL checkpoint (`espnet/xeus` on the HF Hub) for arms A and B;
set its path via `XEUS_CHECKPOINT` or `configs/base.yaml`. Arm C needs no
checkpoint (random init).

## Reproducibility

- One YAML per run; `svb` dumps `resolved_config.yaml` + `env.json` (git SHA,
  library versions, GPU) beside every `results.json`.
- Fixed seeds in `configs/seeds.yaml`; official corpus test splits only.
- Paper tables in `tables/` are generated from the result JSONs, not hand-typed.

## Layout

See `src/svb/` — `model/` (XEUS + CTC head), `train/`, `btm/`, `merge/`,
`eval/`, `stats/`. The standalone XEUS encoder (`model/xeus_standalone.py`) is a
fork-free reimplementation — no ESPnet dependency.

## Status

Core code complete; 15 CPU smoke tests pass (vocab, merge identities,
sign-election, stats, collate, config). **Next (on the GPU server):**

1. `uv sync` (full deps incl. torch+CUDA, torchaudio, datasets, soundfile).
2. `export XEUS_CHECKPOINT=/path/to/xeus_checkpoint.pth` (`espnet/xeus`).
3. `export CV_ROOT=/path/to/common_voice_25` (downloaded from Mozilla Data
   Collective); `make data` to verify split sizes.
4. First real run: `svb run --arm A_ssl --scale 3 --seed 0 --config configs/base.yaml`
   — validates the XEUS 577M load + train→eval→transfer end to end.
5. Finalize `configs/scales/64.yaml` before the 64-lang tier.

## ESPnet (planned)

`model/xeus_standalone.py` is a fork-free reimplementation of the XEUS encoder so
the checkpoint loads with PyTorch alone. A bespoke reimplementation is a
correctness risk — an earlier version had E-Branchformer forward deviations from
the reference. We plan to migrate the load + forward to the **reference ESPnet
implementation** behind the same `load_xeus_encoder` interface (keeping the
standalone path as a no-ESPnet fallback), so results rest on the canonical model.
See issue tracker.

## Acknowledgements

- **XEUS** — Chen et al., *Towards Robust Speech Representation Learning for
  Thousands of Languages* (arXiv:2407.00837); weights at `espnet/xeus`. All
  credit for the model and weights belongs to its authors; this repo only loads
  and fine-tunes it.
- **ESPnet** — the speech toolkit XEUS is built with.
- Merge strategies follow Ilharco et al. (Task Arithmetic), Yadav et al. (TIES),
  and Yu et al. (DARE).
- The XEUS loader is adapted from the author's FLAIME project.
