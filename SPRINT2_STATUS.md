# Sprint 2 — État et Configuration

## Objectif

Reproduire la baseline DDSP-Piano (JAES 2023, Renault et al.) à 16 kHz sur MAESTRO v3.
Critère de sortie : MR-STFT test ≤ 2.0.

## Architecture implémentée

```
piano_neuronal/s2_baseline/
  config.py              # Hyperparams (batch=32, lr=0.003, bf16, early stop)
  dataset.py             # PyTorch Dataset (midi_pairs.h5, precompute+cache)
  midi_encoding.py       # MIDI events → conditioning (n_frames, n_synths, 2)
  model.py               # PianoModel DAG complet
  loss.py                # MR-STFT (6 résolutions), ReverbReg, InharmonicityLoss
  train.py               # bf16 AMP, warmup, early stop, torch.compile, GPU log
  evaluate.py            # Évaluation test set, audio samples
  run_s2.py              # CLI : --local-debug, --smoke-test, --resume
  modules/
    core.py              # Primitives DDSP (resample, fft_convolve, scale_fn)
    harmonic_oscillator.py
    inharm_synth.py       # MultiInharmonic (2 substrings, B inharmonique)
    filtered_noise.py     # DynamicSizeFilteredNoise
    reverb.py             # MultiInstrumentReverb (IR apprise)
    z_encoder.py          # OneHotZEncoder (1 instrument)
    note_release.py       # Extension de release (vectorisé)
    context_network.py    # FiLMContextNetwork
    monophonic_network.py # MonophonicDeepNetwork
    inharm_model.py       # InharmonicityNetwork (B paramétrique)
    detuner.py             # Detuner (2 strings unison)
    parallelizer.py        # Batch × polyphonie merge/unmerge
scripts/
  setup_vast.sh          # Setup instance Vast.ai (RTX 4090)
```

## Paramètres d'entraînement

| Paramètre | Valeur | Notes |
|-----------|--------|-------|
| Sample rate | 16 kHz | DDSP-Piano baseline |
| Frame rate | 250 Hz | |
| Batch size | 32 | Configurable (64 sur 48 Go) |
| Learning rate | 0.003 | Warmup 500 steps |
| Optimiseur | Adam | lr=0.003, clip_norm=5.0 |
| Precision | bf16 | Natif sur RTX 4090/Blackwell |
| Early stopping | patience=8 epochs | Cible 30 epochs max |
| torch.compile | oui | Fallback sans si échoue |
| Checkpoint | tous les 500 steps | Reprise automatique |

## GPU cible

- **RTX 4090** (24 Go VRAM, 83 TFLOPS) — Vast.ai à ~0,82 $/h
- Template : PyTorch (Vast) — CUDA 12.9

## Procédure en 3 étapes

### A. Debug local (3070 Ti, gratuit)
```bash
python -m piano_neuronal.s2_baseline.run_s2 --local-debug
```
batch=2, 3 steps + 1 val — vérifie shapes, NaN, checkpoint

### B. Smoke test (RTX 4090 louée, ~2 min)
```bash
python -m piano_neuronal.s2_baseline.run_s2 --smoke-test
```
5 steps + 1 val + 1 checkpoint — vérifie GPU, bf16, compile

### C. Run 30 epochs (RTX 4090 louée, ~10h)
```bash
python -m piano_neuronal.s2_baseline.run_s2 --batch-size 32 --lr 0.003 --epochs 30 --patience 8 --ckpt-every 500
```
Budget estimé : **~15 $** (12.5h × 0,82 $/h + marge)

## Critères de sortie

1. MR-STFT test ≤ 2.0
2. Loss décroissante 50+ steps, sans NaN
3. 10 échantillons test reconnaissables comme piano
4. Tests unitaires passés
5. Shapes corrects : audio (B, 48000), conditioning (B, 750, 16, 2)

## Dépendances Sprint 1

- `data_output/midi_pairs.h5` (125 Go) — 23 438 paires MIDI-audio
- `data_output/manifest.parquet` — splits train/val/test

## Statut

- **Code** : ✅ Implémenté (11 modules)
- **Debug local** : ⏳ À faire
- **Smoke test** : ⏳ À faire
- **Entraînement** : ⏳ À faire
- **Évaluation** : ⏳ À faire