# 🤖💥 Ouija des Rouombas — version KISS

**Un fichier. Une boucle. Zéro thread.**

```bash
pip install -r requirements.txt   # opencv + numpy
python main.py
```

## Comment ça marche

À chaque frame, `main.py` fait 7 choses dans l'ordre, sans magie :

1. **Lit la caméra** (fallback DSHOW → MSMF → défaut sous Windows)
2. **Détecte 2 taches de couleur** : rouge = Roumba A, bleu = Roumba B (seuils HSV réglables en live)
3. **Ajuste l'image** (saturation / luminosité) — l'affichage seulement, pas la détection
4. **Dessine la grille** (4 coins calibrés au clic, ordre libre) + les **mots** (1 mot = 1 case, police adaptée)
5. **Dessine les traces** de chaque robot dans sa couleur
6. **Détecte la collision** (distance en cases + cooldown réglable) → **explosion pixel-art** au point exact + enregistrement du mot touché
7. **Affiche** dans la fenêtre, sans déformation (letterbox)

## L'interface (tout à droite, panneau défilable à la molette)

| Panneau | Réglages |
|---|---|
| **Caméra** | index + réouverture à chaud |
| **Grille** | colonnes/lignes + bouton 📍 calibrer les 4 coins |
| **Mots** | ajout/suppression (mot + case, ex: OUIJA + B3) |
| **Détection couleur 🎨** | saturation min / luminosité min / aire min + effacer traces |
| **Image 🖼** | saturation / luminosité / taille explosion / délai collisions |
| **Collisions 💥** | liste des mots touchés + reset + compteur |

Tout est **sauvegardé automatiquement** dans `grid_config.json`.

## Sorties

- `collisions.csv` — chaque collision (heure, case, mot)

## Matériel

Colle un **sticker rouge** sur le Roomba A et un **bleu** sur le B.
Si la détection accroche mal : monte la lumière, joue les sliders 🎨,
ou utilise des autocollants plus grands / plus saturés.
