# Sprint 1 — État et Reprise

## Statut actuel

- **Phase 1 (features)** : En cours — 10 workers, reprise automatique par batch de 50
- **Phase 2 (MIDI pairs)** : Pas encore lancée, code prêt avec reprise automatique
- **Pipeline lancé** : `python -m piano_neuronal.s1_data.serialize` (background task)

## Si le process crash (OOM, erreur Python, Ctrl+C)

- Les batches complétés sont déjà flushés dans `data_output/piano162_s1.h5`
- Le batch en cours (max 50 fichiers) peut être corrompu
- **Reprise** : simplement relancer `python -m piano_neuronal.s1_data.serialize`
  - Le code détecte les groupes existants dans le HDF5 et les skip
  - Reprend là où ça s'est arrêté

## Si l'ordinateur s'éteint / redémarre

- Même principe : le HDF5 est sur disque, les batches complétés sont intacts
- Le batch en cours peut être corrompu → le groupe partiel sera visible dans le HDF5
- **Reprise** : relancer `python -m piano_neuronal.s1_data.serialize`
- Si le HDF5 est corrompu (ne s'ouvre pas) : supprimer `data_output/piano162_s1.h5` et relancer (tout recommencer)

## Si le HDF5 est corrompu

- Symptôme : `h5py` affiche des erreurs OSError ou "truncated file"
- **Solution** : `rm data_output/piano162_s1.h5` puis relancer
- Perte : tout le travail, il faut recommencer de zéro

## Commandes de reprise

```bash
# Phase 1 — Features (reprend automatiquement si HDF5 existe)
cd "C:\Users\sajid\Documents\ia piano 2"
python -m piano_neuronal.s1_data.serialize

# Phase 2 — MIDI pairs (reprend automatiquement si HDF5 existe)
python -m piano_neuronal.s1_midi.midi_pairs

# Pipeline complet avec options
python -m piano_neuronal.s1_data.run_pipeline --workers 10 --target-pairs 0
python -m piano_neuronal.s1_data.run_pipeline --skip-features --workers 8

# Validation
python -m piano_neuronal.s1_validate.validation

# Nettoyage si HDF5 corrompu
rm data_output/piano162_s1.h5
rm data_output/manifest.parquet
rm data_output/midi_pairs.h5
rm -rf data_output/rendered_audio/
```

## Fichiers de sortie

| Fichier | Taille estimée | Contenu |
|---------|---------------|---------|
| `piano162_s1.h5` | ~34 Go | Audio + features (B, tau, centroid, MFCC, IR, excitation) |
| `manifest.parquet` | ~400 Ko | Métadonnées + splits train/val/test |
| `midi_pairs.h5` | ~100-150 Go | Paires MIDI-audio segmentées |
| `rendered_audio/` | Temporaire | WAV rendus, supprimés après écriture dans HDF5 |

## Critères de sortie Sprint 1

1. 3 520 samples parsés sans erreur
2. Features extraites : B, tau_fast, tau_slow, centroid, MFCC, room IR, excitation
3. Test de resynthèse (différé aux tests unitaires)
4. Split train/val/test correct (MezzoPiano en test pour features, split officiel MAESTRO pour paires)
5. >= 5 000 paires MIDI-audio (cible ~49 000) — **split par morceau** via MAESTRO officiel
6. Fichiers HDF5 + manifest générés
7. Room IR extraite pour les paires Close+Ambient PedalOff
8. Inharmonicité B : taux de succès mesuré > 75%

## Corrections appliquées (commits)

- `3baa080` : Fix Room IR, inharmonicity B, MFCC, temp file leak, workers config
- `38e0390` : Reduce MIDI pairs workers to 8
- `9dd6d85` : Memory safety (batched rendering, worker recycling, GC)
- `8754cf1` : Reduce Phase 1 workers to 10
- `2e8bbf5` : Resume capability for both pipelines

## Prochaines étapes après Phase 1

1. Vérifier que Phase 1 est complète (validation checks 1-4, 7-8)
2. Lancer Phase 2 : `python -m piano_neuronal.s1_midi.midi_pairs` (8 workers)
3. Vérifier Phase 2 (validation check 5 : >= 5000 paires)
4. Lancer validation complète : `python -m piano_neuronal.s1_validate.validation`
5. Commit les outputs si tout passe