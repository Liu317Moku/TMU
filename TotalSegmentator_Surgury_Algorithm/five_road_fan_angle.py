# five_road_suture_front1cm_Yfilter.py
import SimpleITK as sitk
import numpy as np
import pandas as pd
import csv
import os
import re
from sklearn.neighbors import NearestNeighbors

# ------------------------------
# Config
# ------------------------------
SEG_PATH = "/home/sandy0317/TotalSegmentator/dataset/output/final_result9/intracerebral_hemorrhage.nii.gz"
FORWARD_DIR = "/home/sandy0317/CranialCTProcessing/point_dataset/coronal_forward_points_0118_10151132"
OUTPUT_SINGLE_CSV = "/home/sandy0317/TotalSegmentator/point_dataset/118_12110210_alg5_sector_output.csv"
OUTPUT_CANDIDATES_CSV = "/home/sandy0317/TotalSegmentator/point_dataset/118_12110210_alg5_sector_candidates.csv"

# sector params
max_elev_deg = 40
max_azim_deg = 35
elev_step = 8
azim_step = 8
ray_start = 10
ray_end = 80
ray_step = 2
min_spacing = 3.0
k_normal = 30
w_dist = 0.6
w_angle = 0.4

# ------------------------------
# Helpers
# ------------------------------
def read_point_cloud_from_dir(forward_dir):
    all_points, layer_labels, source_files = [], [], []
    for file in sorted(os.listdir(forward_dir)):
        if file.endswith(".csv"):
            match = re.search(r'forward_(\d+)mm', file)
            layer_mm = int(match.group(1)) if match else 0
            csv_path = os.path.join(forward_dir, file)
            df = pd.read_csv(csv_path)
            pts = df[['x','y','z']].to_numpy()
            all_points.append(pts)
    return np.vstack(all_points)

def local_normal_pca(points, query_point, k=30):
    if points.shape[0] < 3:
        raise ValueError("點雲數量不足以估算法向")
    nbrs = NearestNeighbors(n_neighbors=min(k, points.shape[0])).fit(points)
    _, idx = nbrs.kneighbors(query_point.reshape(1, -1))
    neighborhood = points[idx[0]]
    X = neighborhood - neighborhood.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, np.argmin(eigvals)]
    return normal / np.linalg.norm(normal)

def write_single_output(output_csv, centroid, entry_point, info_dict):
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label","x","y","z"])
        w.writerow(["hematoma_centroid", *centroid])
        w.writerow(["outer_skull", *entry_point])
        w.writerow([])
        w.writerow(["metadata","value"])
        for k,v in info_dict.items():
            w.writerow([k, v])
    print(f"✅ 寫出結果到: {output_csv}")

# ------------------------------
# main
# ------------------------------
img = sitk.ReadImage(SEG_PATH)
arr = sitk.GetArrayFromImage(img)
coords = np.argwhere(arr > 0)
if coords.shape[0] == 0:
    raise ValueError("Segmentation 內沒有血腫區域")

centroid_voxel = coords.mean(axis=0)
centroid_lps = img.TransformContinuousIndexToPhysicalPoint(centroid_voxel[::-1])
centroid = np.array([-centroid_lps[0], -centroid_lps[1], centroid_lps[2]])
print("🩸 血腫質心 (RAS):", centroid)

all_points = read_point_cloud_from_dir(FORWARD_DIR)
if all_points.size == 0:
    raise ValueError("FORWARD_DIR 沒有點雲")
print(f"🔢 讀取顱骨點數: {all_points.shape[0]}")

# ------------------------------
# 篩選 Y > -100 (冠狀顱縫前 1 公分)
# ------------------------------
filtered_points = all_points[all_points[:,1] > -60]  #調整y軸
if filtered_points.size == 0:
    raise RuntimeError("沒有符合 Y > -100 的點")
print(f"🔹 篩選後點數: {filtered_points.shape[0]}")

nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(filtered_points)

hemisphere = "left" if centroid[0] < 0 else "right"

found = []
candidates = []

# ------------------------------
# sector search
# ------------------------------
for theta in range(0, max_elev_deg+1, elev_step):
    th = np.radians(theta)
    if hemisphere == "left":
        phi_vals = range(-max_azim_deg, 1, azim_step)
    else:
        phi_vals = range(0, max_azim_deg+1, azim_step)
    for phi in phi_vals:
        ph = np.radians(phi)
        direction = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
        direction = direction / (np.linalg.norm(direction)+1e-12)

        for L in range(ray_start, ray_end+1, ray_step):
            query = centroid + direction * float(L)
            _, idx = nbrs.kneighbors(query.reshape(1, -1), n_neighbors=1)
            skull_p = filtered_points[idx[0][0]]

            if any(np.linalg.norm(skull_p - f) < min_spacing for f in found):
                continue

            found.append(skull_p)
            try:
                ln = local_normal_pca(filtered_points, skull_p, k=k_normal)
                if ln[2] < 0: ln = -ln
                path = centroid - skull_p
                plen = np.linalg.norm(path)
                if plen == 0:
                    angle = 90.0
                else:
                    angle = float(np.degrees(np.arccos(np.clip(np.dot(path/plen, ln), -1.0, 1.0))))
            except:
                angle = 90.0

            dist = float(np.linalg.norm(skull_p - centroid))
            candidates.append({"point":skull_p, "dist_mm":dist, "angle_deg":angle, "theta":theta, "phi":phi})
            break

if len(candidates) == 0:
    raise RuntimeError("Sector search 無候選 (Y > -100)")

# ------------------------------
# ranking by normalized distance+angle
# ------------------------------
ds = np.array([c["dist_mm"] for c in candidates])
as_ = np.array([c["angle_deg"] for c in candidates])
dn = (ds - ds.min())/(ds.max()-ds.min()+1e-8)
an = as_/180.0
scores = w_dist * dn + w_angle * an
best_idx = int(np.argmin(scores))
best = candidates[best_idx]

# write candidates CSV
with open(OUTPUT_CANDIDATES_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x","y","z","dist_mm","angle_deg","theta","phi"])
    for c in candidates:
        x,y,z = c["point"]
        writer.writerow([x,y,z,c["dist_mm"],c["angle_deg"],c["theta"],c["phi"]])
print(f"🗂 已輸出候選點到: {OUTPUT_CANDIDATES_CSV}")

# write best
info = {"method":"sector_search","dist_mm":best["dist_mm"], "angle_deg":best["angle_deg"], "theta":best["theta"], "phi":best["phi"]}
write_single_output(OUTPUT_SINGLE_CSV, centroid, best["point"], info)

print("────────────────────────────")
print("最佳候選 (point RAS):", best["point"])
print(f"distance: {best['dist_mm']:.2f} mm, angle: {best['angle_deg']:.1f}°, theta: {best['theta']}, phi: {best['phi']}")
print("────────────────────────────")
