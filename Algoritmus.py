import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Inštalácia a stiahnutie modelov
!pip install torch torchvision opencv-python numpy matplotlib -q
!git clone https://github.com/magicleap/SuperPointPretrainedNetwork.git
!git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
!wget -nc https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth -P SuperPointPretrainedNetwork/

import sys
sys.path.append('SuperPointPretrainedNetwork')
sys.path.append('SuperGluePretrainedNetwork')
from SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend
from SuperGluePretrainedNetwork.models.superglue import SuperGlue
import torch
import os

# ==========================================
# 1. INICIALIZÁCIA MODELOV (Iba raz pre zrýchlenie)
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nPoužité zariadenie pre modely: {device}")

print("Načítavam SuperPoint model...")
superpoint = SuperPointFrontend(
    weights_path='SuperPointPretrainedNetwork/superpoint_v1.pth',
    nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=(device == 'cuda')
)

print("Načítavam SuperGlue model...")
superglue = SuperGlue({'weights': 'indoor', 'sinkhorn_iterations': 50, 'match_threshold': 0.2}).to(device)
superglue.eval()

# ==========================================
# 2. NAHRÁVANIE OBRÁZKOV A SLUČKA
# ==========================================
print("\nNahrajte jeden alebo viac obrázkov:")
uploaded = files.upload()

# Slučka pre každý nahratý obrázok
for image_path in uploaded.keys():
    print("\n" + "="*80)
    print(f"SPRACOVÁVAM OBRÁZOK: {image_path}")
    print("="*80)

    # Načítanie pôvodného obrázka
    img_full_bgr = cv2.imread(image_path)

    # ------------------------------------------
    # RESIZE A KONVERZIA NA PNG (Goldilocks pravidlo: ~800x600)
    # ------------------------------------------
    h, w = img_full_bgr.shape[:2]
    scale = min(800/w, 600/h)

    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img_full_bgr = cv2.resize(img_full_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w, new_h = w, h

    base_name = os.path.splitext(image_path)[0]
    png_path = f"{base_name}_compressed.png"
    cv2.imwrite(png_path, img_full_bgr)
    print(f"Obrázok bol upravený na rozlíšenie {new_w}x{new_h} a uložený ako: {png_path}")

    img_full_rgb = cv2.cvtColor(img_full_bgr, cv2.COLOR_BGR2RGB)

    # ------------------------------------------
    # INTERAKTÍVNE ROZDELENIE
    # ------------------------------------------
    split_orientation = input("Sú zrkadlo a objekt vedľa seba (zadajte 'V') alebo nad sebou (zadajte 'H')? [V/H/Skip]: ").strip().upper()

    if split_orientation == 'SKIP':
        print("Preskakujem tento obrázok...")
        continue

    plt.figure(figsize=(12, 8))
    plt.imshow(img_full_rgb)
    plt.title(f'[{png_path}] Nájdite Y súradnicu pre rozdelenie' if split_orientation == 'H' else f'[{png_path}] Nájdite X súradnicu pre rozdelenie')
    plt.xlabel('X súradnica')
    plt.ylabel('Y súradnica')
    plt.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.show()

    if split_orientation == 'H':
        try:
            split_y = int(input("Zadajte Y súradnicu pre horizontálne rozdelenie (napr. 300): "))
        except ValueError:
            split_y = img_full_bgr.shape[0] // 2
            print(f"Neplatný vstup. Rozdeľujem presne v strede: Y = {split_y}")

        print(f"Rozdeľujem na Y = {split_y} a otáčam o 90 stupňov doprava...")
        img_rotated_bgr = cv2.rotate(img_full_bgr, cv2.ROTATE_90_CLOCKWISE)
        new_split_x = img_full_bgr.shape[0] - split_y

        img_mirror_bgr = img_rotated_bgr[:, :new_split_x]
        img_object_bgr = img_rotated_bgr[:, new_split_x:]
    else:
        try:
            split_x = int(input("Zadajte X súradnicu pre vertikálne rozdelenie (napr. 400): "))
        except ValueError:
            split_x = img_full_bgr.shape[1] // 2
            print(f"Neplatný vstup. Rozdeľujem presne v strede: X = {split_x}")

        print(f"Obrázok úspešne rozdelený na X = {split_x}!")
        img_mirror_bgr = img_full_bgr[:, :split_x]
        img_object_bgr = img_full_bgr[:, split_x:]

    # ------------------------------------------
    # SPRACOVANIE A EXTRAKCIA
    # ------------------------------------------
    mirror_rgb = cv2.cvtColor(img_mirror_bgr, cv2.COLOR_BGR2RGB)
    object_rgb = cv2.cvtColor(img_object_bgr, cv2.COLOR_BGR2RGB)

    gray_mirror_uint8 = cv2.cvtColor(img_mirror_bgr, cv2.COLOR_BGR2GRAY)
    gray_object_uint8 = cv2.cvtColor(img_object_bgr, cv2.COLOR_BGR2GRAY)

    gray_mirror = gray_mirror_uint8.astype(np.float32) / 255.0
    gray_object = gray_object_uint8.astype(np.float32) / 255.0

    h_m, w_m = gray_mirror.shape
    h_o, w_o = gray_object.shape

    object_flipped = cv2.flip(gray_object, 1)
    object_flipped_rgb = cv2.flip(object_rgb, 1)

    print("\nExtrahujem kľúčové body (SuperPoint)...")
    pts_mirror, desc_mirror, _ = superpoint.run(gray_mirror)
    pts_flipped, desc_flipped, _ = superpoint.run(object_flipped)

    kp_mirror = pts_mirror[:2, :].T
    kp_flipped = pts_flipped[:2, :].T

    kpts0 = torch.from_numpy(kp_mirror).float()[None].to(device)
    kpts1 = torch.from_numpy(kp_flipped).float()[None].to(device)
    desc0 = torch.from_numpy(desc_mirror).float()[None].to(device)
    desc1 = torch.from_numpy(desc_flipped).float()[None].to(device)
    scores0 = torch.from_numpy(pts_mirror[2, :]).float()[None].to(device)
    scores1 = torch.from_numpy(pts_flipped[2, :]).float()[None].to(device)

    print("Hľadám zhody (SuperGlue)...")
    with torch.no_grad():
        pred = superglue({
            'keypoints0': kpts0, 'keypoints1': kpts1, 'descriptors0': desc0, 'descriptors1': desc1,
            'scores0': scores0, 'scores1': scores1,
            'image0': torch.zeros(1, 1, h_m, w_m).to(device), 'image1': torch.zeros(1, 1, h_o, w_o).to(device),
        })

    matches0 = pred['matches0'][0].cpu().numpy()
    mscores = pred['matching_scores0'][0].cpu().numpy()

    valid = matches0 > -1
    valid_indices = np.where(valid)[0]
    matched_indices1 = matches0[valid]

    class SimpleMatch:
        def __init__(self, queryIdx, trainIdx, distance):
            self.queryIdx, self.trainIdx, self.distance = queryIdx, trainIdx, distance

    good_matches = [SimpleMatch(q, t, 1.0 - mscores[q]) for q, t in zip(valid_indices, matched_indices1)]

    if len(good_matches) < 5:
        print("❌ Nenašiel sa dostatok zhôd pre tento obrázok!")
        continue

    src_pts = np.float32([kp_mirror[m.queryIdx] for m in good_matches])
    dst_pts_flipped = np.float32([kp_flipped[m.trainIdx] for m in good_matches])

    # RANSAC so striktným prahom
    best_H, best_mask = cv2.findHomography(src_pts, dst_pts_flipped, cv2.RANSAC, 10.0)

    if best_H is None:
        print("❌ RANSAC nenašiel konzistentnú homografiu pre tento obrázok.")
        continue

    inliers = best_mask.ravel().astype(bool)
    src_inliers = src_pts[inliers]
    dst_flipped_inliers = dst_pts_flipped[inliers]

    dst_original_inliers = np.column_stack([w_o - 1 - dst_flipped_inliers[:, 0], dst_flipped_inliers[:, 1]])

    canvas_width = w_m + w_o
    canvas_height = max(h_m, h_o)

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:h_m, :w_m] = mirror_rgb
    canvas[:h_o, w_m:] = object_rgb

    # VIZUALIZÁCIA (Spojené inliers)
    plt.figure(figsize=(12, 6))
    plt.imshow(canvas)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(src_inliers)))
    lines = []
    for i, (src, dst) in enumerate(zip(src_inliers, dst_original_inliers)):
        x1, y1 = src
        x2, y2 = dst[0] + w_m, dst[1]
        plt.plot([x1, x2], [y1, y2], color=colors[i], linewidth=1.5, alpha=0.7)
        plt.plot(x1, y1, 'o', color=colors[i], markersize=6, markeredgecolor='white', markeredgewidth=1)
        plt.plot(x2, y2, 'o', color=colors[i], markersize=6, markeredgecolor='white', markeredgewidth=1)
        lines.append(((x1, y1), (x2, y2)))
    plt.title(f'Spojenie pre {len(src_inliers)} geometricky konzistentných zhôd')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Výpočet priesečníkov
    def line_intersection(line1, line2):
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2
        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if abs(denom) < 1e-10: return None
        x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return (x, y)

    all_intersections = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            pt = line_intersection(lines[i], lines[j])
            if pt:
                all_intersections.append(pt)

    if len(all_intersections) < 3:
        print("❌ Nedostatok priesečníkov na analýzu.")
        continue

    all_intersections = np.array(all_intersections)
    filtered = all_intersections.copy()

    # ==========================================
    # 3. ROBUSTNÁ PRIESTOROVÁ FILTRÁCIA (IQR)
    # ==========================================
    for _ in range(3):
        if len(filtered) < 5:
            break

        # Extrakcia súradníc
        x_coords = filtered[:, 0]
        y_coords = filtered[:, 1]

        # Výpočet IQR hraníc pre os X
        q1_x, q3_x = np.percentile(x_coords, [25, 75])
        iqr_x = q3_x - q1_x
        h_dolna_x = q1_x - (1.5 * iqr_x)
        h_horna_x = q3_x + (1.5 * iqr_x)

        # Výpočet IQR hraníc pre os Y
        q1_y, q3_y = np.percentile(y_coords, [25, 75])
        iqr_y = q3_y - q1_y
        h_dolna_y = q1_y - (1.5 * iqr_y)
        h_horna_y = q3_y + (1.5 * iqr_y)

        # Definovanie "Bezpečnej zóny"
        valid_x = (x_coords >= h_dolna_x) & (x_coords <= h_horna_x)
        valid_y = (y_coords >= h_dolna_y) & (y_coords <= h_horna_y)
        safe_zone_mask = valid_x & valid_y

        # Aplikácia masky
        filtered = filtered[safe_zone_mask]

    # ==========================================
    # 4. ZHLUKOVANIE (DBSCAN) A ŠTATISTIKA
    # ==========================================
    if len(filtered) >= 10:
      scaled = StandardScaler().fit_transform(filtered)
      neigh = NearestNeighbors(n_neighbors=min(5, len(filtered)-1)).fit(scaled)
      dists, _ = neigh.kneighbors(scaled)
      eps_auto = np.median(np.sort(dists[:,-1])) * 1.5

      labels = DBSCAN(eps=eps_auto, min_samples=min(3, len(filtered)//5)).fit_predict(scaled)

      clusters = {}
      unique_labels = set(labels)
      for lbl in unique_labels:
          if lbl != -1:
              points_in_cluster = np.sum(labels == lbl)
              clusters[lbl] = points_in_cluster

      if clusters:
          best_label = max(clusters, key=clusters.get)
          main_cluster = filtered[labels == best_label]
        else:
            main_cluster = filtered
    else:
        main_cluster = filtered

    # Výpočet čistej priestorovej štatistiky
    if len(main_cluster) > 0:
        mean_x = np.mean(main_cluster[:,0])
        mean_y = np.mean(main_cluster[:,1])

        spatial_spread = np.sqrt(np.std(main_cluster[:,0])**2 + np.std(main_cluster[:,1])**2)

        median_x = np.median(main_cluster[:,0])
        median_y = np.median(main_cluster[:,1])
        iqr_spread = np.hypot(np.percentile(main_cluster[:,0], 75) - np.percentile(main_cluster[:,0], 25),
                              np.percentile(main_cluster[:,1], 75) - np.percentile(main_cluster[:,1], 25)) / 2
    else:
        mean_x, mean_y = np.mean(filtered, axis=0) if len(filtered) > 0 else (0,0)
        spatial_spread = np.sqrt(np.std(filtered[:,0])**2 + np.std(filtered[:,1])**2) if len(filtered) > 0 else 0
        median_x, median_y = mean_x, mean_y
        iqr_spread = 0

    # VIZUALIZÁCIA GEOMETRIE
    plt.figure(figsize=(12, 6))
    for line in lines:
        plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 'gray', alpha=0.1, linewidth=1)

    plt.scatter(all_intersections[:,0], all_intersections[:,1], c='lightgray', s=15, alpha=0.4, label='Priesečníky')
    plt.scatter(filtered[:,0], filtered[:,1], c='orange', s=25, alpha=0.5, label='Filtrované priesečníky')
    plt.scatter(main_cluster[:,0], main_cluster[:,1], c='red', s=50, alpha=0.8, edgecolors='white', label=f'Hlavný zhluk ({len(main_cluster)})')
    plt.scatter(mean_x, mean_y, c='gold', s=300, marker='*', edgecolors='black', label='Priestorový stred')
    plt.scatter(median_x, median_y, c='lime', s=200, marker='P', edgecolors='black', label='Stred (Medián)')
    plt.axvline(x=w_m, color='cyan', linestyle='--', linewidth=2, alpha=0.8, label='Hranica zrkadla')

    margin_x = max(canvas_width * 0.15, 200)
    margin_y = max(canvas_height * 0.15, 200)

    if len(main_cluster) > 0:
        x_min_plot = min(-margin_x, np.min(main_cluster[:, 0]) - margin_x)
        x_max_plot = max(canvas_width + margin_x, np.max(main_cluster[:, 0]) + margin_x)
        y_min_plot = min(-margin_y, np.min(main_cluster[:, 1]) - margin_y)
        y_max_plot = max(canvas_height + margin_y, np.max(main_cluster[:, 1]) + margin_y)
    else:
        x_min_plot = min(-margin_x, np.min(filtered[:, 0]) - margin_x) if len(filtered) > 0 else -margin_x
        x_max_plot = max(canvas_width + margin_x, np.max(filtered[:, 0]) + margin_x) if len(filtered) > 0 else canvas_width + margin_x
        y_min_plot = min(-margin_y, np.min(filtered[:, 1]) - margin_y) if len(filtered) > 0 else -margin_y
        y_max_plot = max(canvas_height + margin_y, np.max(filtered[:, 1]) + margin_y) if len(filtered) > 0 else canvas_height + margin_y

    plt.xlim(x_min_plot, x_max_plot)
    plt.ylim(y_max_plot, y_min_plot)
    plt.xlabel('X súradnica')
    plt.ylabel('Y súradnica')
    plt.title(f'Analýza konvergencie pre {png_path}\nPriestorový rozptyl: {spatial_spread:.1f} px | IQR rozptyl: {iqr_spread:.1f} px')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 5. FINÁLNY VERDIKT
    # ==========================================
    avg_spread = (spatial_spread + iqr_spread) / 2

    print("\n" + "="*60)
    print(" VÝSLEDOK ANALÝZY GEOMETRIE")
    print("="*60)

    if avg_spread < 40.0:
            print(" ✅ VÝBORNÁ KONVERGENCIA: Autentická geometria")
            print("    Lúče odrazu vykazujú mimoriadne presnú zbiehavosť do jedného")
            print("    matematického bodu. Takto nízky rozptyl je charakteristický")
            print("    pre reálne fotografie s kvalitným rovným zrkadlom.")
            
        elif avg_spread < 130.0:
            print(" ✓ PRIRODZENÁ KONVERGENCIA: Pravdepodobne reálna fotografia")
            print("    Geometria zrkadla je prijateľná a zachováva 3D perspektívu.")
            print("    Zaznamenaný rozptyl je v medziach normy a býva spôsobený")
            print("    bežným skreslením šošovky, drobným šumom alebo textúrou.")
            
        elif avg_spread < 200.0:
            print(" ⚠️ SLABÁ KONVERGENCIA: Výrazné nezrovnalosti (Šedá zóna)")
            print("    Priesečníky lúčov sú príliš rozptýlené, čo narúša fyzikálny")
            print("    model odrazu. Môže ísť o AI generovaný obraz, silne zakrivené")
            print("    zrkadlo, alebo mäkký/členitý materiál pôvodného objektu.")
            
        else:
            print(" ❌ ŽIADNA KONVERGENCIA: Fyzikálne nemožná geometria")
            print("    Čiary sa zbiehajú úplne chaoticky a netvoria spoločný úbežník.")
            print("    Zákony optiky tu zlyhali, čo je s veľmi vysokou pravdepodobnosťou")
            print("    dôkazom toho, že ide o syntetický (AI generovaný) fake obraz.")

    print("-" * 60)
    print(f" Priestorový rozptyl: {spatial_spread:.1f} px")
    print(f" IQR rozptyl: {iqr_spread:.1f} px")
    print(f" Priemerný rozptyl: {avg_spread:.1f} px")
    print("="*60 + "\n")
