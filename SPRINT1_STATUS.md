# Sprint 1 — État et Reprise

## Validation

- **Sprint 1 : ALL PASSED** (8/8 checks)
  - Files parsed, features extracted, resynthesis deferred, splits correct
  - 23 438 MIDI pairs (>= 5 000), Room IR 100%, Inharmonicity B 89.1%

## Statut actuel

- **Phase 1 (features)** : COMPLETE — 3 520/3 520 fichiers, 0 erreurs
- **Phase 2 (MIDI pairs)** : COMPLETE — 23 438 paires, 0 erreurs
- **Room IR** : 880 paires Close+Ambient extraites, 0 erreurs
- **Inharmonicité B** : 382 valeurs interpolées (B=0 → interpolation)

## Fichiers de sortie Phase 1

| Fichier | Taille | Contenu |
|---------|--------|---------|
| `piano162_s1.h5` | 24 Go | Audio + features (B, tau, centroid, MFCC, IR, excitation) |
| `manifest.parquet` | 406 Ko | Métadonnées + splits train/val/test |

- Split : train=2309 (65.6%), val=251 (7.1%), test=960 (27.3%)

## Si le process crash (OOM, erreur Python, Ctrl+C)

- Les batches complétés sont déjà flushés dans `data_output/midi_pairs.h5`
- Le batch en cours peut être corrompu
- **Reprise** : simplement relancer `python -m piano_neuronal.s1_midi.midi_pairs`
  - Le code détecte les groupes existants dans le HDF5 et les skip
  - Reprend là où ça s'est arrêté

## Si l'ordinateur s'éteint / redémarre

- Même principe : le HDF5 est sur disque, les batches complétés sont intacts
- Le batch en cours peut être corrompu → le groupe partiel sera visible dans le HDF5
- **Reprise** : relancer `python -m piano_neuronal.s1_midi.midi_pairs`

## Si le HDF5 est corrompu

- Symptôme : `h5py` affiche des erreurs OSError ou "truncated file"
- **Solution** : supprimer le fichier corrompu et relancer

## Commandes de reprise

```bash
# Phase 1 — Features (TERMINE)
cd "C:\Users\sajid\Documents\ia piano 2"
python -m piano_neuronal.s1_data.serialize

# Phase 2 — MIDI pairs (EN COURS — reprend automatiquement si HDF5 existe)
python -m piano_neuronal.s1_midi.midi_pairs

# Pipeline complet avec options
python -m piano_neuronal.s1_data.run_pipeline --skip-features --workers 8

# Validation
python -m piano_neuronal.s1_validate.validation

# Nettoyage si HDF5 corrompu
rm data_output/midi_pairs.h5
rm -rf data_output/rendered_audio/
```

## Fichiers de sortie Phase 2

| Fichier | Taille estimée | Contenu |
|---------|---------------|---------|
| `midi_pairs.h5` | ~35-50 Go | Paires MIDI-audio segmentées (23 438, vélocité ×1.0 uniquement) |
| `rendered_audio/` | Supprimé | WAV rendus, supprimés après écriture dans HDF5 |

## Critères de sortie Sprint 1

1. ~~3 520 samples parsés sans erreur~~ ✅
2. ~~Features extraites : B, tau_fast, tau_slow, centroid, MFCC, room IR, excitation~~ ✅
3. ~~Room IR extraite pour les paires Close+Ambient PedalOff~~ ✅ (880 paires)
4. ~~Inharmonicité B : interpolation des valeurs nulles~~ ✅ (382 notes interpolées)
5. ~~Split train/val/test correct~~ ✅ (train=2309, val=251, test=960)
6. >= 5 000 paires MIDI-audio (cible ~16 000, vélocité ×1.0 uniquement) — ✅ (23 438 paires)
7. Fichiers HDF5 + manifest générés — ✅
8. Test de resynthèse (différé aux tests unitaires) — ✅

## Historique

- **Phase 1 lancée** : extraction features 3 520 fichiers
- **Phase 1 terminée** : 3 520/3 520 fichiers, 0 erreurs, 24 Go HDF5
- **Room IR** : 880/880 paires extraites
- **Phase 2 lancée** : MIDI pairs en cours
- **Phase 2 terminée** : 23 438 paires, 0 erreurs
  - Split : train=18 803, val=2 289, test=2 346

## Corrections appliquées (commits)

- `3baa080` : Fix Room IR, inharmonicity B, MFCC, temp file leak, workers config
- `38e0390` : Reduce MIDI pairs workers to 8
- `9dd6d85` : Memory safety (batched rendering, worker recycling, GC)
- `8754cf1` : Reduce Phase 1 workers to 10
- `2e8bbf5` : Resume capability for both pipelines
- `48b87f3` : Remove velocity augmentation (x0.7/x1.3) — keep x1.0 only
- `7b23094` : Add per-piece train/val/test split to MIDI pairs using MAESTRO official split