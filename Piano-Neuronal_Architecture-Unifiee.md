# Piano Neuronal Nouvelle Génération — Document d'Architecture Unifié

**Remplacement du dataset « Piano in 162 » (Steinway Model B) par un moteur neuronal hybride temps réel**

*Document de consolidation — fusionne le rapport de faisabilité (familles neuronales + blueprint), le rapport STN (Doc 1) et le rapport hybride physique (Doc 2). Structuré par axe A / B, avec protocoles de test.*

---

## 0. Décision en une page

Les trois sources convergent sur l'**ossature logicielle** (DDSP comme cadre, RTNeural/ANIRA pour l'inférence RT-safe, JUCE pour le natif, Emscripten/WASM + AudioWorklet + SharedArrayBuffer pour le web). La divergence porte uniquement sur le **cœur de synthèse** — et cette divergence tombe exactement sur les deux axes du cahier des charges.

| | **Axe A — Qualité / Poids** | **Axe B — Fidélité maximale** |
|---|---|---|
| **Priorité** | Tenir 128 voix + mobile/web + <10 ms + poids minimal | Restituer *ce* Steinway à l'identique, latence/poids relâchés |
| **Cœur retenu** | **Hybride physique** : Neural Hammer one-shot → guide d'ondes/modal → FDN (profil Pianoteq) | **Additif STN** : DDSP-Piano + module transitoire DCT + refiner Vocos |
| **Pourquoi** | Le DSP récursif est frugal ; coût quasi indépendant de la polyphonie | L'additif matche le **spectre cible mesuré**, seul chemin vers la fidélité au dataset |
| **Poids cible** | < 10 Mo | < 200 Mo |
| **Latence cible** | < 10 ms natif / 15–30 ms web (best effort) | < 20 ms (refiner toléré hors live critique) |
| **Risque principal** | **Dérive timbrale** : un waveguide sonne « plausible », pas « ce piano » → mitigé par matching spectral sur le dataset | **Goulot CPU additif** (7 680 oscillateurs à 128 voix) → mitigé par distillation + polyphonie interne bornée |

**Tranchant central de la décision :** l'additif (Axe B) optimise une *loss spectrale contre l'enregistrement réel* → il converge vers le timbre exact de Piano in 162. Le waveguide/modal (Axe A) résout un *problème inverse* (caler des paramètres physiques sur un son cible) notoirement difficile → il gagne sur le CPU mais doit être recalé empiriquement pour ne pas sonner « générique » comme Pianoteq.

---

## 1. Le dataset cible (rappel compact)

- **Piano in 162** par **Ivy Audio** (Simon Dalzell) — Steinway Model B (~2,11 m).
- **96 kHz / 24 bits**, FLAC ; **≈ 3 520 échantillons** ; **~4,7 Go compressé**, **~5,9–14 Go brut** selon la mesure.
- Matrice : **88 touches × 5 couches de vélocité** (pp, p, mp, mf, f) **× 2 round-robins × 2 micros** (Close + Ambient) **× pédale ON/OFF**.
- Micros : deux paires **Rode NT5** (ambiance couloir + proximité intérieure).
- **Failles structurelles à corriger** par le neuronal : (a) velocity stepping (5 couches pour 127 valeurs MIDI), (b) absence de vraie résonance sympathique (2⁸⁸ états d'étouffoirs impossibles à échantillonner), (c) décroissance figée à deux stades (Weinreich : prompt soundboard decay + aftersound découplé).
- **Licence : redistribution des samples interdite.** → contrainte d'ingénierie ET juridique (voir §9).

---

## 2. Architecture unifiée — le triptyque commun

Les deux axes partagent la **même topologie en trois strates**. Ils diffèrent uniquement par l'implémentation de la Strate 2 (le générateur de corde).

```
                    Événements MIDI (pitch, vélocité continue, pédale CC64/CC66)
                                          │
                                          ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │  STRATE 1 — RÉSEAU D'EXCITATION  (« Neural Hammer », inférence ONE-SHOT) │
   │  MLP/GRU léger → vecteur d'excitation 50–100 ms                          │
   │  Entrées : pitch[21..108], vélocité∈[0,1], état d'amortissement corde    │
   │  Exécuté 1× par note-on → retire la pression de calcul continu           │
   └────────────────────────────────────┬───────────────────────────────────┘
                                          │ excitation large-bande
                                          ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │  STRATE 2 — GÉNÉRATEUR DE CORDE     (DIFFÈRE SELON L'AXE)                │
   │  • Axe A : Guide d'ondes / synthèse modale DSP (récursif, frugal)        │
   │  • Axe B : Banc additif STN (Sines + Transients DCT + Noise filtré)      │
   │  Communs : inharmonicité B(note,vél), double decay vertical/horizontal   │
   └────────────────────────────────────┬───────────────────────────────────┘
                                          │ N voix
                                          ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │  STRATE 3 — RÉSONATEUR GLOBAL  (couplage trichord + FDN + IR soundboard) │
   │  Mixage des voix → Conv1D couplage cordes → FDN (matrice Householder/    │
   │  Hadamard) piloté par pédale → convolution partitionnée IR Ambient       │
   │  Sorties stéréo [Close, Ambient]                                         │
   └────────────────────────────────────┬───────────────────────────────────┘
                                          ▼
                              Sortie audio 48 kHz (A) / 96 kHz (B)
```

**Justification de la ségrégation des tâches :** le Deep Learning capture *l'âme timbrale* (impact chaotique du marteau, spectre dépendant de la vélocité) en une inférence one-shot ; le DSP pur prolonge la résonance (decay, sustain, sympathie) sur des dizaines de secondes à coût CPU quasi nul. C'est le même principe qui permet à Pianoteq de tenir ~250 voix en 1–3 % CPU pour 50 Mo.

---

## 3. Spécification Axe A — « DDSP-Hybride Lite »

**Objectif : 128 voix < 10 % CPU sur machine moyenne, < 10 Mo, mobile + web.**

| Élément | Spécification |
|---|---|
| **Strate 1** | MLP dense `[pitch, vél, état] → excitation`, ≤ 50 k params, inférence inline RTNeural (<0,3 ms), pas d'async |
| **Strate 2** | Guide d'ondes Karplus-Strong étendu **OU** synthèse modale paramétrique ; dispersion via filtres all-pass calibrés sur B mesuré ; double decay par filtre de perte bi-bande |
| **Strate 3** | FDN 8–16 lignes, matrice Householder, filtres d'absorption accordés sur l'IR soundboard ; pédale = modulation lissée des coefficients d'absorption |
| **Fréquence audio** | 48 kHz |
| **Polyphonie réelle** | 128 voix DSP indépendantes (SoA + auto-vectorisation) + voice stealing harmonique vers FDN |
| **Backbone neuronal** | Partagé, coût **indépendant de la polyphonie** (1 inférence/note-on) |
| **Compression** | Distillation enseignant→élève + élagage itératif (Lottery Ticket) jusqu'à 90 % sparsité |
| **Poids total** | < 10 Mo (modèle + IR courtes) |

**Recalage anti-dérive timbrale (critique) :** extraire de Piano in 162, par note : coefficient d'inharmonicité B (pics STFT haute résolution), IR soundboard (déconvolution de Wiener sur release/ambient), enveloppe d'excitation (résidu des 50 premières ms), T60 (enveloppe de l'aftersound). Ces paramètres mesurés *initialisent et contraignent* le waveguide, au lieu de le laisser converger vers un piano générique.

**Challenger à évaluer (poids minimal absolu) :** **Piano-SSM** (modèle d'espace d'état, ~270 k params, raw audio 44,1 kHz). Avantage : un seul modèle pour tout le clavier, complexité linéaire. Limite : résonance sympathique implicite/instable, pas de spatialisation par voix. À benchmarker comme variante Axe A, pas comme défaut.

---

## 4. Spécification Axe B — « DDSP-STN Pro + Refiner »

**Objectif : indiscernabilité à l'écoute vs l'enregistrement original ; latence/poids secondaires.**

| Élément | Spécification |
|---|---|
| **Strate 1** | GRU léger, excitation conditionnée (pitch, vél continue, point de frappe) |
| **Strate 2 — Sines** | Synthèse additive différentiable, ~60 partiels/voix, inharmonicité B(note,vél) physiquement contrainte, double polarisation (δf vélocité-dépendant), partiels fantômes par sommation FM |
| **Strate 2 — Transients** | CNN dans le domaine **DCT** → IDCT cohérent en phase (élimine pré-écho ; restitue l'impact métallique/boisé du Steinway) |
| **Strate 2 — Noise** | Bruit blanc gaussien → filtre temps-variable fréquentiel (MLP) → reconstruction Overlap-Add iFFT |
| **Strate 2 — Couplage** | Conv1D trichord (excitation sympathique des cordes adjacentes) |
| **Strate 3** | FDN différentiable + FIR différentiable + convolution partitionnée IR Ambient mesurée |
| **Étage post** | **Vocos** ou **iSTFTNet2** refiner conditionné, distillé 1-step (polish perceptuel, +20–30 ms tolérés) |
| **Fréquence audio** | 96 kHz, stéréo native [Close, Ambient] |
| **Polyphonie** | P interne 32 + routing ; batching GPU sur desktop ; rendu proactif si besoin |
| **Poids total** | < 200 Mo |

**Mitigation du goulot additif :** à 128 voix × 60 partiels = 7 680 oscillateurs/échantillon, l'additif sature un CPU faible. Mitigations cumulables : (1) borner la polyphonie additive interne à 32 puis projeter, (2) distiller vers Axe A au-delà du seuil de masquage, (3) batching SIMD/GPU strict en SoA, (4) couper les partiels sous le seuil psychoacoustique par note.

---

## 5. Stack de déploiement multiplateforme

| Plateforme | Format modèle | Runtime inférence | Hôte audio | Latence réaliste |
|---|---|---|---|---|
| Windows / Linux x86 | ONNX FP16 | RTNeural (inline) / ONNX Runtime via ANIRA | JUCE → VST3 | 3–5 ms |
| macOS (Intel / Apple Silicon) | Core ML + ONNX fallback | Core ML / RTNeural | JUCE → AU/VST3 | 2–3 ms |
| iOS / iPadOS | Core ML compilé | Core ML + AVAudioEngine | AUv3 | 3–8 ms |
| Android | TFLite/LiteRT (NNAPI/GPU) ou ONNX Mobile | TFLite | Oboe / AAudio | 8–15 ms |
| Web | ONNX INT8 quantifié | ONNX Runtime Web (WASM SIMD + WebGPU/WebNN) | Web Audio API + AudioWorklet | **15–30 ms** (cf. §9) |

**Discipline temps réel (boucle audio) :** zéro allocation dynamique (pas de `new`/`malloc`/`std::vector` extensible), zéro mutex (structures lock-free SPSC), I/O déportée en thread basse priorité, mémoire SoA pré-allouée. **Inférence Strate 1 inline** plutôt qu'async : pour un MLP one-shot <0,5 ms, bloquer le thread audio sur un sémaphore (approche ANIRA async) réintroduit la dépendance de synchro que la discipline RT interdit.

**Web spécifique :** code C++ → WASM (Emscripten, SIMD 128 bits, `wasm-strip` → ~6 Mo) ; exécution dans `AudioWorkletGlobalScope` (quantum 128 samples = 2,9 ms) ; communication thread principal ↔ worklet via **SharedArrayBuffer** en ring buffer SPSC wait-free (`Atomics.store`/`load`) — jamais `postMessage` (génère du GC). Serveur **doit** envoyer les en-têtes COOP + COEP (isolation cross-origin) pour déverrouiller SharedArrayBuffer.

---

## 6. Pipeline d'entraînement commun (4 phases)

1. **Préparation data** — FLAC 96 kHz → 48 kHz (A) / 96 kHz (B), sérialisation TFRecord ; annotations dérivées du nom de fichier `(note 21–108, couche vélocité, round-robin, micro, pédale)` ; vélocité normalisée continue ∈ [0,1].
2. **Extraction de features** (pour le recalage Axe A et les loss de référence) : B par STFT haute-rés, IR soundboard par Wiener, enveloppe d'excitation (résidu 50 ms), T60, centroïde spectral, MFCC.
3. **MIDI synthétique** — re-rendre MAESTRO v3 / GiantMIDI via Sforzando (SFZ) → paires `(MIDI, audio_cible)` polyphoniques pour apprendre couplage + pédale + sympathie.
4. **Entraînement** :
   - Phase 1 : notes isolées (3 520 samples), conditionnement (pitch, vélocité).
   - Phase 2 : MIDI synthétique (couplage, pédale).
   - Phase 3 (Axe B) : fine-tuning adversarial (discriminateur Multi-Period HiFi-GAN-style).
   - Phase 4 : distillation Axe B → Axe A (audio distillation souple + control distillation stricte sur paramètres DSP).

**Fonctions de perte :**
- **MR-STFT** multi-fenêtres (2048/1024/512/256/128/64) — grandes fenêtres = précision partiels graves + inharmonicité ; petites = transitoires sans pré-écho.
- **Perte DCT transitoire** (Axe B) — coefficients DCT prédits vs transitoire extrait par filtrage soustractif.
- **Perte vélocité hybride** — BCE (activité note) + MSE (niveau), pour ne pas pénaliser le silence en note-off.
- **Régularisation latente** (continuité vélocité) — contraindre l'espace latent à rester lisse pour interpoler les vélocités jamais enregistrées (ex. MIDI 71 entre mp=60 et f=80).

---

## 7. PROTOCOLES DE TEST

> Trois familles de tests s'appliquent aux deux axes : **(I) qualité perceptuelle**, **(II) fidélité objective**, **(III) performance système**. Les seuils Go/No-Go diffèrent par axe.

### 7.1 Test I — Qualité perceptuelle (écoute)

**I.a — ABX double aveugle vs enregistrement original**
- Stimuli : 30 notes isolées (couvrant graves/médiums/aigus × pp/mp/f) + 10 extraits polyphoniques (MIDI MAESTRO rendus par le moteur vs par le sampler Piano in 162).
- Protocole : ≥ 15 auditeurs (dont ≥ 5 pianistes/ingénieurs son), interface ABX, ≥ 20 essais/auditeur, casque calibré.
- Métrique : **taux de discrimination correcte**. 50 % = indiscernable ; > 75 % = différence audible nette.

**I.b — MOS (Mean Opinion Score) absolu**
- Échelle 1–5 sur naturel, attaque, timbre, résonance. Comparer moteur vs sample vs (référence haute) Pianoteq.

**Critères Go/No-Go :**

| | Axe A | Axe B |
|---|---|---|
| ABX vs original | ≤ 70 % discrimination (différence tolérée mais légère) | **≤ 60 %** (quasi-indiscernable) |
| MOS naturel | ≥ 3,8 / 5 | ≥ 4,3 / 5 |
| MOS ≥ celui du sample sur attaque | toléré −0,3 | **requis ≥** |

→ Si Axe A échoue ABX > 75 %, escalader vers hybride sample + résiduel neuronal. Si Axe B échoue, renforcer le refiner / l'adversarial.

### 7.2 Test II — Fidélité objective (métriques sans écoute)

Calculées sur le set de test (notes non vues à l'entraînement, et vélocités interpolées).

| Métrique | Cible Axe A | Cible Axe B | Ce qu'elle vérifie |
|---|---|---|---|
| **MR-STFT distance** (vs original) | ≤ seuil baseline DDSP-Piano | ≤ 50 % du seuil A | Fidélité spectrale globale |
| **Erreur sur B (inharmonicité)** | < 5 % par note | < 2 % | Timbre métallique correct |
| **Centroïde spectral** Δ | < 8 % | < 4 % | Brillance perçue |
| **MFCC distance (DTW)** | faible | très faible | Enveloppe spectrale |
| **Erreur T60 (decay)** | < 15 % | < 8 % | Décroissance réaliste |
| **Continuité vélocité** : monotonie + absence de saut spectral entre vél. n et n+1 | **zéro discontinuité** | **zéro discontinuité** | Élimination du velocity stepping |
| **Pré-écho transitoire** (énergie avant t₀ d'attaque) | < −40 dB | < −60 dB | Cohérence de phase de l'attaque |

**Test spécifique « réponse vélocité continue » (cœur du projet) :** balayer la vélocité de 1 à 127 par pas de 1 sur 5 notes ; tracer centroïde spectral et loudness en fonction de la vélocité ; **vérifier l'absence de marche d'escalier** (dérivée bornée, pas de discontinuité aux frontières des 5 couches d'origine). C'est le test qui valide l'argument de vente principal vs le sampling.

**Test spécifique « résonance sympathique » :** jouer une note, pédale enfoncée, vérifier l'apparition de l'excitation des cordes harmoniquement liées (analyse spectrale du halo) ; comparer qualitativement au comportement physique attendu (le sample figé en est incapable → pas de référence directe, donc évaluation par plausibilité + écoute experte).

### 7.3 Test III — Performance système (le verrou du cahier des charges)

Bancs à exécuter sur **3 cibles** : desktop référence (Apple M1 / i5 récent), mobile milieu de gamme (ex. Snapdragon 8 Gen 2), navigateur (Chrome WebGPU).

| KPI | Méthode de mesure | Cible Axe A | Cible Axe B |
|---|---|---|---|
| **Latence end-to-end** | Note-on physique → premier échantillon audio (mesure loopback matériel) | < 10 ms natif ; 15–30 ms web | < 20 ms |
| **Polyphonie soutenue** | Glissando + pédale, montée jusqu'à saturation ; compter voix avant 1er dropout | ≥ 128 sans dropout | ≥ 128 (desktop) |
| **CPU @ 128 voix** | Profiler sur le thread audio, charge moyenne + pics | **< 10 % (M1) / < 30 % (mobile) / < 60 % (web)** | < 5 % (M3 desktop) |
| **Poids binaire + modèle** | Taille du `.vst3`/`.wasm`/`.tflite` livré | **< 10 Mo** | < 200 Mo |
| **RAM résidente @ 128 voix** | RSS du process | < 50 Mo | < 500 Mo |
| **Xruns / glitches** | Compteur d'underruns sur 10 min de jeu intensif | **0** | 0 |
| **Temps de chargement** | Cold start jusqu'à première note jouable | < 1 s (web) | < 5 s |
| **Inférence Strate 1** | Chrono RTNeural par note-on | < 0,5 ms | < 1 ms |

**Critère Go/No-Go système (bloquant) :** un build qui ne tient pas `128 voix sans xrun ET CPU sous cible ET latence sous cible` sur la cible la plus contrainte (mobile pour A, desktop pour B) est rejeté avant tout test perceptuel.

### 7.4 Tableau de bord récapitulatif des KPI

| KPI | Baseline sample | Étalon Pianoteq | **Axe A (cible)** | **Axe B (cible)** |
|---|---|---|---|---|
| Poids | ~6 Go | 50 Mo | **< 10 Mo** | < 200 Mo |
| CPU / 128 voix | ~très faible (disque) | 1–3 % | **< 10 %** | < 5 % |
| Latence natif | < 10 ms | < 10 ms | **< 10 ms** | < 20 ms |
| Latence web | N/A | N/A | **15–30 ms** | N/A |
| Fidélité au dataset | exacte (figée) | générique | bonne, recalée | **quasi-exacte** |
| Continuité vélocité | ❌ 5 paliers | ✅ | **✅ continue** | **✅ continue** |
| Résonance sympathique | ⚠️ crossfade | ✅ | ✅ FDN | ✅ FDN + couplage |
| Multiplateforme (incl. web) | ❌ | ⚠️ (pas web) | **✅** | ⚠️ (desktop premium) |

---

## 8. Plan d'exécution (90 jours, 6 sprints)

| Sprint | Livrable | Test de sortie |
|---|---|---|
| S1 (sem. 1–2) | Pipeline data + extraction features (B, IR, T60) + MIDI synthétique | 5 000+ paires (MIDI, audio) ; features validées |
| S2 (3–4) | Baseline DDSP-Piano reproduite (16 kHz MAESTRO) | MR-STFT ≥ papier JAES 2023 |
| S3 (5–6) | **Axe B** sur Piano in 162 (STN + vélocité continue) | Test II complet + ABX préliminaire |
| S4 (7–8) | **Axe A** (waveguide recalé + FDN) + distillation depuis B | Test III natif (128 voix, <10 ms, <10 % CPU) |
| S5 (9–10) | Builds TFLite + Core ML (iOS/Android) | Test III mobile (<30 % CPU) |
| S6 (11–12) | Build ONNX Runtime Web + AudioWorklet + COOP/COEP | Démo web jouable (15–30 ms) + ABX final |

---

## 9. Garde-fous et points à verrouiller

1. **Latence web < 10 ms = irréaliste.** Le quantum AudioWorklet (2,9 ms) est correct, mais buffer de sortie + OS empilent 15–30 ms en pratique. Cible web = best-effort 15–30 ms, à documenter explicitement auprès des parties prenantes.

2. **« Polyphonie infinie » via transfert au FDN** = illusion de masquage psychoacoustique, valable en pratique mais elle **ne préserve pas** le pitch/decay propre de la voix volée. À présenter comme tel, jamais comme polyphonie réelle additionnelle.

3. **Inférence inline > async** pour la Strate 1. Bloquer le thread audio sur un sémaphore (ANIRA async) pour attendre un autre thread viole la discipline RT ; pour un MLP <0,5 ms, l'inférence RTNeural inline est plus sûre et plus simple.

4. **Piano-SSM « 10,1 µs »** = délai I/O algorithmique, **pas** la latence système (dominée par le buffer, 3–10 ms). Ne pas confondre dans les specs.

5. **Licence IP (absent des deux PDF d'origine, critique).** Redistribution des samples Piano in 162 interdite. Un modèle qui **mémorise** les samples bruts peut être juridiquement assimilé à une redistribution. Mitigations : régularisation forte, augmentation, vérification empirique de non-reconstruction à l'identique, et **accord écrit avec Ivy Audio / Simon Dalzell** avant tout usage commercial. Alternative propre : ré-enregistrer un Steinway Model B sous licence claire.

6. **Pianoteq reste l'étalon d'efficacité** (50 Mo, 1–3 % CPU, ~250 voix). La valeur ajoutée différenciante du moteur neuronal n'est pas de battre cette efficacité, mais de **capturer automatiquement le timbre d'un piano enregistré précis** (ici le Steinway de Piano in 162) sans modélisation physique manuelle exhaustive, et de pouvoir se ré-entraîner sur un autre piano/lieu.

7. **Chiffres de compression aspirationnels.** « Distillation 100× » et « sparsité 99 % » sont des bornes hautes de la littérature, non prouvées sur ce cas. À valider par le Test III, avec repli sur des taux plus conservateurs si la qualité chute (le pruning one-shot dégrade dès 50 % — d'où l'élagage itératif Lottery Ticket).

8. **Aucun test d'écoute n'existe encore** sur cette architecture appliquée à Piano in 162 ; toutes les cibles de qualité sont extrapolées. Le **Test I.a (ABX) dès le Sprint 3** est le jalon de vérité.

9. **« Machine moyenne » à définir contractuellement.** Le KPI « 128 voix sans saturer le CPU d'une machine moyenne » n'a de sens qu'avec une cible matérielle nommée (les cibles du Test III servent de définition opérationnelle).

---

## 10. Sources clés (consolidées des trois rapports)

- Renault, Mignot, Roebel — **DDSP-Piano**, DAFx 2022 / JAES 71(9) 2023 (IRCAM/STMS) ; code : github.com/lrenault/ddsp-piano
- Simionato et al. — **Sines, Transient, Noise Neural Modeling of Piano Notes**, arXiv 2409.06513 / Frontiers 2024
- Simionato et al. — **Physics-informed differentiable method for piano modeling**, Frontiers Signal Processing 2023/2024
- Berendes et al. — **Towards Differentiable Piano Synthesis based on Physical Modeling**, ISMIR 2023 (AudioLabs Erlangen)
- Dallinger — **Piano-SSM: Raw Audio Piano Synthesis with Structured State Space Models**, TU Wien 2025 ; github.com/domdal/piano-ssm
- Engel et al. — **DDSP: Differentiable Digital Signal Processing**, ICLR 2020 (Google Magenta)
- Caillon & Esling — **RAVE**, arXiv 2111.05011 (IRCAM ACIDS) ; latence streaming ~85–93 ms (Caspe et al. JAES 2025 / arXiv 2503.11562)
- Chowdhury — **RTNeural**, arXiv 2106.03037 (CCRMA Stanford)
- Schulz & Ackva — **ANIRA**, arXiv 2506.12665 (ADC23)
- Siuzdak — **Vocos**, ICLR 2024 ; **iSTFTNet2** (NTT, Interspeech 2023) ; **MS-Wavehax** (Interspeech 2025)
- ONNX Runtime Web (WASM SIMD + WebGPU/WebNN) ; Emscripten ; JUCE
- Ivy Audio — **Piano in 162** (sfzinstruments.github.io/pianos/in_162/)
