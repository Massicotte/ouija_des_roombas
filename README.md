# 🤖💥 Ouija des Roumbas

**UNE seule commande, UNE seule fenêtre** : la vidéo en direct à gauche, tous les
réglages à droite qui agissent **immédiatement** (calibration, mots, explosion,
collisions) — rien à relancer, jamais.

```bash
python main.py
```

## Le principe (faisable, simple, robuste)

| Étape | Technique |
|---|---|
| 1. Grille virtuelle | colonnes/lignes réglables en live |
| 2. Angle de caméra | **homographie** — tu cliques les 4 coins **directement sur la vidéo** |
| 3. Tracking des 2 Roombas | **ArUco** (marqueurs imprimés) → identités A/B garanties, zéro faux positif |
| 4. Collision | distance en cases + cooldown anti-spam |
| 5. Explosion 💥 | animation pixel-art (sprite) jouée **au point exact de collision**, taille réglable en live |

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 0) Imprimer les marqueurs (une seule fois)

```bash
python make_markers.py
```

→ `markers/marker_A_id0.png` et `marker_B_id1.png` à imprimer (8 cm de côté) et
coller bien à plat sur chaque Roomba. Roumba A = id 0, Roumba B = id 1.

### 1) Lancer l'application

```bash
python main.py
```

Tout est dans la même fenêtre, **en live** :

- **Vidéo (gauche)** : le flux caméra avec les 2 robots trackés, les explosions,
  le compteur de collisions. C'est AUSSI ici que tu cliques les 4 coins pour
  calibrer la grille (après avoir pressé le bouton 📍).
- **Grille** : compteurs colonnes/lignes + bouton "📍 Calibrer les 4 coins" —
  l'homographie se met à jour instantanément, la vidéo continue de tourner.
- **Mots** : ajoute/supprime des mots (mot + case de départ + direction H/V) en
  live — la case de collision affiche le mot couvert.
- **Explosion 💥** : slider de taille (x0.2→x5.0) et vitesse, appliqués à la
  prochaine explosion sans redémarrage.
- **Collisions** : liste des cases/mots touchés + bouton reset + compteur.
- **Réglages** : enregistrement vidéo on/off, distance de collision, sauvegarde.

### Sorties

- `captures/session_*.mp4` — la vidéo complète (explosions incluses)
- `collisions.csv` — historique des impacts (timestamp, case, mot)

## Réglages fins

Tout se règle **dans l'application**. Valeurs par défaut dans `config.py`,
sauvegardes dans `grid_config.json` (ignoré par git).

| Clé | Effet |
|---|---|
| `cam_index` | 0 = webcam par défaut |
| `aruco_dict` / `robot_a_id` / `robot_b_id` | marqueurs et ids des 2 Roombas |
| `collision_dist_cells` | distance de collision en cases (0.5 ≈ cases voisines) |
| `explosion_scale` / `explosion_fps` | taille / vitesse de l'animation 💥 |

## Limites connues

- **Marqueurs bien à plat** : un marqueur gondolé se détecte moins bien. 8 cm de
  côté est un bon compromis vu de haut.
- Si un marqueur est caché, la collision n'est évaluée que quand les DEUX sont
  visibles — pas de faux positif.

## Structure

```
main.py            # L'APPLICATION UNIQUE : vidéo + calibration + mots + explosion + collisions, tout en live
make_markers.py    # génère les 2 marqueurs ArUco à imprimer (une fois)
aruco_detect.py    # détection ArUco des 2 Roombas (uniquement ids A et B)
assets/            # sprite-sheet de l'animation d'explosion (pixel-art)
config.py          # config partagée, sauvegardée dans grid_config.json
roomba_tracker.py  # (optionnel) version standalone sans interface — laissé pour référence
calibrate.py       # (optionnel) calibration séparée — remplacé par main.py
```
