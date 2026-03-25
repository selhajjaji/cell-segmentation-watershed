# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 12:43:22 2025

@author: m
"""

# -*- coding: utf-8 -*-
"""
Segmentation automatique de cellules (Sujet #1)


Partie A  : filtrage / prétraitement
Partie B  : watershed + visualisation des marqueurs
Partie C  : analyse quantitative (aire + intensités R/G/B)
"""

# ============================================================
# Imports
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from skimage.data import cells3d as c3d
from skimage.color import rgb2gray, label2rgb
from skimage.filters import sobel, threshold_otsu
from skimage import exposure
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed, mark_boundaries
from skimage.feature import peak_local_max

from scipy import ndimage as ndi
import scipy.ndimage

# ============================================================
# Partie A – Filtrage de l'image couleur et prétraitement
# ============================================================

# Image couleur construite à partir de cells3d

cell_img = np.stack(
    (c3d()[20, 1], c3d()[36, 1] / 2, c3d()[30, 0]),
    axis=2
).astype(float)

# Normalisation pour l'affichage
cell_img = cell_img / cell_img.max()

# ---------- A.0 : Visualisation des canaux couleur ----------
R_channel = cell_img[:, :, 0]
G_channel = cell_img[:, :, 1]
B_channel = cell_img[:, :, 2]

plt.figure(figsize=(14, 4))
plt.subplot(1, 4, 1)
plt.title("Image originale")
plt.imshow(cell_img)
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Canal R")
plt.imshow(R_channel, cmap="Reds")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Canal G")
plt.imshow(G_channel, cmap="Greens")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Canal B")
plt.imshow(B_channel, cmap="Blues")
plt.axis("off")

plt.tight_layout()
plt.show()

# ---------- A.1 : Conversion en niveaux de gris + lissage ----------
image_gris = rgb2gray(cell_img)

# Filtre gaussien pour réduire le bruit
image_gaussien = scipy.ndimage.gaussian_filter(image_gris, sigma=1.5)

# Égalisation d’histogramme pour améliorer le contraste
image_eq = exposure.equalize_hist(image_gaussien)

# ---------- A.2 : Gradient Sobel + seuillage ----------
# Gradient (magnitude) – structure des contours
image_grad = sobel(image_gaussien)

# Seuillage global (Otsu) sur l’image égalisée
seuil = threshold_otsu(image_eq)
image_seuil = image_eq > seuil

# ---------- A.3 : Remplissage des trous + morphologie ----------
# Remplissage des trous dans les cellules
image_rempli = scipy.ndimage.binary_fill_holes(image_seuil)

# Élément structurant "cercle"
cercle = np.array([[0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0]])

# Fermeture puis ouverture
image_fermeture = scipy.ndimage.binary_closing(
    image_rempli, structure=cercle
)
image_ouverture = scipy.ndimage.binary_opening(
    image_fermeture, structure=cercle
)

# Suppression des petits objets isolés (bruit)
mask_cells = remove_small_objects(image_ouverture, min_size=50)

# ---------- A.4 : Figure récapitulative du pipeline ----------
plt.figure(figsize=(12, 10))

plt.subplot(3, 3, 1)
plt.title("Image originale")
plt.imshow(cell_img)
plt.axis("off")

plt.subplot(3, 3, 2)
plt.title("Image en gris")
plt.imshow(image_gris, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.title("Image lissée (Gaussien)")
plt.imshow(image_gaussien, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 4)
plt.title("Gradient (Sobel)")
plt.imshow(image_grad, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.title("Image seuillée (Otsu)")
plt.imshow(image_seuil, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 6)
plt.title("Image remplie")
plt.imshow(image_rempli, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 7)
plt.title("Après fermeture")
plt.imshow(image_fermeture, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 8)
plt.title("Après ouverture")
plt.imshow(image_ouverture, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 9)
plt.title("Masque final (nettoyé)")
plt.imshow(mask_cells, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

# ============================================================
# Partie B – Segmentation complète par watershed
# ============================================================

# B.1 Distance transform sur le masque binaire
distance = ndi.distance_transform_edt(mask_cells)

# Lissage léger pour stabiliser les maxima
distance_smooth = ndi.gaussian_filter(distance, sigma=1)

# B.2 Maxima locaux (marqueurs pour le watershed)
coords = peak_local_max(
    distance_smooth,
    labels=mask_cells,
    min_distance=15,         # à ajuster si besoin
    exclude_border=False
)

# Création de l'image de marqueurs
markers = np.zeros_like(distance, dtype=int)
for i, (r, c) in enumerate(coords, start=1):
    markers[r, c] = i

# B.3 Watershed
labels_ws = watershed(-distance_smooth, markers, mask=mask_cells)

# Image en couleurs aléatoires pour visualiser les régions
cells_random_color = label2rgb(
    labels_ws, bg_label=0, bg_color=(0, 0, 0)
)

# ---------- B.4 : Figure ----------
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.title("Masque final (entrée watershed)")
plt.imshow(mask_cells, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Distance transform lissée")
plt.imshow(distance_smooth, cmap="magma")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Segmentation (couleurs aléatoires)")
plt.imshow(cells_random_color)
plt.axis("off")

plt.tight_layout()
plt.show()

# ---------- B.5 : NOUVELLES FIGURES – marqueurs + contours ----------

# 1) Maxima locaux sur la distance transform
plt.figure(figsize=(8, 6))
plt.title("Distance transform + maxima locaux (marqueurs)")
plt.imshow(distance_smooth, cmap="magma")
if len(coords) > 0:
    plt.scatter(coords[:, 1], coords[:, 0],
                s=20, facecolors="none", edgecolors="white", linewidths=1.5)
plt.axis("off")
plt.tight_layout()
plt.show()

# 2) Contours de segmentation sur l'image originale
plt.figure(figsize=(6, 6))
plt.title("Contours de segmentation sur l'image originale")
# mark_boundaries attend une image float dans [0,1]
image_with_boundaries = mark_boundaries(cell_img, labels_ws)
plt.imshow(image_with_boundaries)
plt.axis("off")
plt.tight_layout()
plt.show()

# ============================================================
# Partie C – Analyse quantitative des résultats
# ============================================================

# C.1 — Nombre total de cellules segmentées
num_cells = labels_ws.max()
print("Nombre total de cellules détectées :", num_cells)

# C.2 et C.3 — Aire + intensités moyennes R/G/B par cellule
cell_stats = []

for label_id in range(1, num_cells + 1):
    mask = (labels_ws == label_id)
    if not np.any(mask):
        continue

    # Aire en pixels
    aire = int(mask.sum())

    # Valeurs de chaque canal pour cette cellule
    R_vals = R_channel[mask]
    G_vals = G_channel[mask]
    B_vals = B_channel[mask]

    R_mean = float(R_vals.mean())
    G_mean = float(G_vals.mean())
    B_mean = float(B_vals.mean())

    cell_stats.append({
        "Cellule": label_id,
        "Aire_pixels": aire,
        "R_mean": R_mean,
        "G_mean": G_mean,
        "B_mean": B_mean
    })

df_cells = pd.DataFrame(cell_stats)

# C.3 (suite) — Couleur dominante
def couleur_dominante(row):
    m = max(row["R_mean"], row["G_mean"], row["B_mean"])
    if m == row["R_mean"]:
        return "Plus rouge"
    elif m == row["G_mean"]:
        return "Plus verte"
    else:
        return "Plus bleue"

df_cells["Couleur_dominante"] = df_cells.apply(couleur_dominante, axis=1)

print("\nAperçu du tableau récapitulatif des cellules :")
print(df_cells.head())

# ============================================================
# C.4 — Visualisations statistiques
# ============================================================

# 1) Histogramme des aires
plt.figure(figsize=(6, 4))
plt.hist(df_cells["Aire_pixels"], bins=10, edgecolor="black")
plt.title("Distribution des tailles de cellules")
plt.xlabel("Aire (pixels)")
plt.ylabel("Nombre de cellules")
plt.tight_layout()
plt.show()

# 2) Nuage de points R_mean vs G_mean, coloré par dominante
couleur_map = {
    "Plus rouge": "red",
    "Plus verte": "green",
    "Plus bleue": "blue"
}
colors = [couleur_map[c] for c in df_cells["Couleur_dominante"]]

plt.figure(figsize=(7, 6))
plt.scatter(df_cells["R_mean"], df_cells["G_mean"], c=colors)
plt.title("Intensité moyenne R vs G par cellule")
plt.xlabel("R_mean")
plt.ylabel("G_mean")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# 3) Diagramme en barres du nombre de cellules par couleur dominante
counts = df_cells["Couleur_dominante"].value_counts()

plt.figure(figsize=(5, 4))
plt.bar(counts.index, counts.values)
plt.title("Répartition des cellules par couleur dominante")
plt.xlabel("Catégorie")
plt.ylabel("Nombre de cellules")
plt.tight_layout()
plt.show()
