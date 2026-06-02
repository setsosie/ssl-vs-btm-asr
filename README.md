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
every result. Data is **public only** (OpenSLR + Common Voice).

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
2. Set `XEUS_CHECKPOINT` to the `espnet/xeus` checkpoint.
3. `huggingface-cli login` + accept Common Voice 17 terms; `make data`.
4. First real run: `svb run --arm A_ssl --scale 3 --seed 0 --config configs/base.yaml`
   — validates the XEUS 577M load + train→eval→transfer end to end.
5. Finalize `configs/scales/64.yaml` before the 64-lang tier.
