# algorithm_1_vertical_projection.py (血腫正上方垂直最近點版)
import SimpleITK as sitk
import numpy as np
import csv
import os
import re

# Config
SEG_PATH = "/home/sandy0317/TotalSegmentator/dataset/output/final_result9/intracerebral_hemorrhage.nii.gz"
FORWARD_DIR = "/home/sandy0317/CranialCTProcessing/point_dataset/coronal_forward_points_0118_10151132"
OUTPUT_SINGLE_CSV = "/home/sandy0317/TotalSegmentator/point_dataset/118_12101657_alg1_vertical_output.csv"
OUTPUT_CANDIDATES_CSV = "/home/sandy0317/TotalSegmentator/point_dataset/118_12101657_alg1_vertical_candidates.csv"

min_height_above = 0.0  # 血腫上方最小距離，可設為0

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

# 1) 血腫質心
img = sitk.ReadImage(SEG_PATH)
arr = sitk.GetArrayFromImage(img)
coords = np.argwhere(arr>0)
if coords.shape[0] == 0:
    raise ValueError("Segmentation 內沒有血腫區域")
centroid_voxel = coords.mean(axis=0)
centroid_lps = img.TransformContinuousIndexToPhysicalPoint(centroid_voxel[::-1])
centroid_ras = np.array([-centroid_lps[0], -centroid_lps[1], centroid_lps[2]])
print("🩸 血腫質心 (RAS):", centroid_ras)

# 2) 前方顱骨點
all_points, layer_labels, source_files = read_point_cloud_from_dir(FORWARD_DIR)
if all_points.size == 0:
    raise ValueError("沒有讀到任何顱骨點")
print(f"🔢 讀到顱骨點總數: {all_points.shape[0]}")

# 前 40% Y 最大點
frac_front = 0.40
N = all_points.shape[0]
n_front = max(1, int(np.round(N * frac_front)))
sorted_idx_by_y = np.argsort(all_points[:,1])
front_idx = sorted_idx_by_y[-n_front:]
front_mask = np.zeros(N, dtype=bool)
front_mask[front_idx] = True

all_points_front = all_points[front_mask]
layer_labels_front = layer_labels[front_mask]
source_files_front = source_files[front_mask]
print(f"👉 前方顱骨點: {all_points_front.shape[0]} 個")

# 3) 血腫正上方最近點（垂直入路）
mask = (
    (all_points_front[:,2] > centroid_ras[2] + min_height_above) &
    (all_points_front[:,1] > -60)   # ⭐ 加入 Y < 90 的篩選條件
)
pts = all_points_front[mask]
labs = layer_labels_front[mask]
srcs = source_files_front[mask]

if pts.shape[0] == 0:
    raise ValueError("血腫正上方沒有顱骨點，請檢查點雲")

# 計算水平距離 (X/Y) 與血腫
dx = pts[:,0] - centroid_ras[0]
dy = pts[:,1] - centroid_ras[1]
dist_xy = np.sqrt(dx*dx + dy*dy)
best_idx = np.argmin(dist_xy)
best_point = pts[best_idx]

# 將所有血腫正上方點作為候選 CSV
candidates = []
for i, p in enumerate(pts):
    candidates.append({
        "point": p,
        "layer": labs[i],
        "src": srcs[i],
        "dist_mm": float(np.linalg.norm(p - centroid_ras)),
        "angle_deg": float(np.degrees(np.arccos(np.clip((centroid_ras - p)[2] / np.linalg.norm(centroid_ras - p), -1,1)))),
        "radius_used": "vertical_only"
    })

best = {
    "point": best_point,
    "layer": labs[best_idx],
    "src": srcs[best_idx],
    "dist_mm": float(np.linalg.norm(best_point - centroid_ras)),
    "angle_deg": 0.0,
    "radius_used": "vertical_only"
}

# 4) 輸出
with open(OUTPUT_CANDIDATES_CSV,"w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["x","y","z","layer","dist_mm","angle_deg","radius_used","source"])
    for c in candidates:
        x,y,z = c["point"]
        w.writerow([x,y,z,c["layer"],c["dist_mm"],c["angle_deg"],c["radius_used"],c["src"]])
print(f"🗂 已輸出候選點到: {OUTPUT_CANDIDATES_CSV}")

with open(OUTPUT_SINGLE_CSV,"w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["label","x","y","z"])
    w.writerow(["hematoma_centroid",*centroid_ras])
    w.writerow(["outer_skull",*best["point"]])
    w.writerow([])
    w.writerow(["metadata","value"])
    w.writerow(["dist_mm",best["dist_mm"]])
    w.writerow(["angle_deg",best["angle_deg"]])
    w.writerow(["radius_used",best["radius_used"]])
    w.writerow(["source",best["src"]])
print(f"✅ 已寫入最終入路: {OUTPUT_SINGLE_CSV}")

print("────────────────────────────")
print("最佳候選 (point):", best["point"])
print(f"distance = {best['dist_mm']:.2f} mm, angle = {best['angle_deg']:.1f}°")
print("────────────────────────────")
