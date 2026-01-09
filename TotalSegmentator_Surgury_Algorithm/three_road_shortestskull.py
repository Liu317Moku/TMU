# algorithm_3_shortest_skull_distance.py
import SimpleITK as sitk
import numpy as np
import csv
import os
import re
from sklearn.neighbors import NearestNeighbors

# Config 
SEG_PATH = "/home/sandy0317/TotalSegmentator/dataset/output/final_result9/intracerebral_hemorrhage.nii.gz"
FORWARD_DIR = "/home/sandy0317/CranialCTProcessing/point_dataset/coronal_forward_points_0118_10151132"
OUTPUT_SINGLE_CSV = "/home/sandy0317/TotalSegmentator/point_dataset/118_12110149_alg3_min_dist_output.csv"
OUTPUT_CANDIDATES_CSV = "/home/sandy0317/TotalSegmentator/point_dataset/118_12110149_alg3_min_dist_candidates.csv"

# params
k_normal = 30
forward_threshold_y = -75  # mm: 只考慮前方 y > -100 的點

# Helpers
def read_point_cloud_from_dir(forward_dir):
    all_points, layer_labels, source_files = [], [], []
    for file in sorted(os.listdir(forward_dir)):
        if file.endswith(".csv"):
            match = re.search(r'forward_(\d+)mm', file)
            layer_mm = int(match.group(1)) if match else 0
            csv_path = os.path.join(forward_dir, file)
            with open(csv_path, newline="") as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)
                for row_idx, row in enumerate(reader):
                    if len(row) >= 3:
                        try:
                            pt = np.array([float(row[0]), float(row[1]), float(row[2])])
                            all_points.append(pt)
                            layer_labels.append(layer_mm)
                            source_files.append(f"{csv_path}:{row_idx+2}")
                        except:
                            continue
    return np.array(all_points), np.array(layer_labels), np.array(source_files)

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

# main
img = sitk.ReadImage(SEG_PATH)
arr = sitk.GetArrayFromImage(img)
coords = np.argwhere(arr > 0)
if coords.shape[0] == 0:
    raise ValueError("Segmentation 內沒有血腫")
centroid_voxel = coords.mean(axis=0)
centroid_lps = img.TransformContinuousIndexToPhysicalPoint(centroid_voxel[::-1])
centroid_ras = np.array([-centroid_lps[0], -centroid_lps[1], centroid_lps[2]])
print("🩸 血腫質心 (RAS):", centroid_ras)

all_points, layer_labels, source_files = read_point_cloud_from_dir(FORWARD_DIR)
if all_points.size == 0:
    raise ValueError("沒有顱骨點雲")

# 只保留前方 y > forward_threshold_y 的點
mask_forward = all_points[:,1] > forward_threshold_y
if not np.any(mask_forward):
    mask_forward = np.ones(all_points.shape[0], dtype=bool)

cand_points = all_points[mask_forward]
cand_labels = layer_labels[mask_forward]
cand_sources = source_files[mask_forward]

# 計算與血腫質心距離
dists = np.linalg.norm(cand_points - centroid_ras, axis=1)
best_idx = int(np.argmin(dists))
entry_point = cand_points[best_idx]
entry_layer = cand_labels[best_idx] if cand_labels.size>0 else None
entry_src = cand_sources[best_idx] if cand_sources.size>0 else None
dist_mm = float(dists[best_idx])

# 計算局部法向量與角度（可選）
try:
    ln = local_normal_pca(all_points, entry_point, k=k_normal)
    if ln[2] < 0: ln = -ln
    path = centroid_ras - entry_point
    path_unit = path / np.linalg.norm(path)
    angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(path_unit, ln), -1.0, 1.0))))
except:
    angle_deg = None

# 輸出候選點 CSV
with open(OUTPUT_CANDIDATES_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x","y","z","layer_mm","dist_mm","source"])
    for p,lab,src,d in zip(cand_points, cand_labels, cand_sources, dists):
        x,y,z = p
        w.writerow([x,y,z,lab,float(d),src])
print(f"🗂 已輸出候選點到: {OUTPUT_CANDIDATES_CSV}")

# 輸出最佳點
info = {"method":"min_skull_distance_forward", "dist_mm":dist_mm, "angle_deg": angle_deg, "source": entry_src}
write_single_output(OUTPUT_SINGLE_CSV, centroid_ras, entry_point, info)

print("────────────────────────────")
print("最佳候選 (point RAS):", entry_point)
print(f"distance: {dist_mm:.2f} mm, angle: {angle_deg if angle_deg is not None else 'NA'}")
print("────────────────────────────")
