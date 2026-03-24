# algorithm_1_vertical_projection.py (血腫正上方垂直最近點版)
import SimpleITK as sitk
import numpy as np
import csv
import os
import re

# Config
SEG_PATH = "/home/TotalSegmentator/dataset/output/final_result9/intracerebral_hemorrhage.nii.gz"
FORWARD_DIR = "/home/CranialCTProcessing/point_dataset/coronal_forward_points"
OUTPUT_SINGLE_CSV = "/home/TotalSegmentator/point_dataset/alg1_vertical_output.csv"
OUTPUT_CANDIDATES_CSV = "/home/TotalSegmentator/point_dataset/alg1_vertical_candidates.csv"

# Helpers
def read_point_cloud_from_dir(forward_dir):
    all_points, layer_labels, source_files = [], [], []
    for file in sorted(os.listdir(forward_dir)):
        if file.endswith(".csv"):
            match = re.search(r'forward_(\d+)mm', file)
            with open(csv_path, newline="") as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)
                for row_idx, row in enumerate(reader):
                    if len(row) >= 3:
                        try:
                            pt = np.array([float(row[0]), float(row[1]), float(row[2])])
                            all_points.append(pt)
                        except:
                            continue
    return np.array(all_points), np.array(layer_labels)

#血腫質心
img = sitk.ReadImage(SEG_PATH)
arr = sitk.GetArrayFromImage(img)
coords = np.argwhere(arr>0)
centroid_voxel = coords.mean(axis=0)
centroid_lps = img.TransformContinuousIndexToPhysicalPoint(centroid_voxel[::-1])

#前方顱骨點
all_points, layer_labels, source_files = read_point_cloud_from_dir(FORWARD_DIR)
print(f"讀到顱骨點總數: {all_points.shape[0]}")

# 前 40% Y 最大點
frac_front = 0.40
N = all_points.shape[0]
n_front = max(1, int(np.round(N * frac_front)))
front_mask = np.zeros(N, dtype=bool)

all_points_front = all_points[front_mask]
layer_labels_front = layer_labels[front_mask]
source_files_front = source_files[front_mask]
print(f"前方顱骨點: {all_points_front.shape[0]} 個")


mask = (
    (all_points_front[:,2] > centroid_ras[2] + min_height_above) &
    (all_points_front[:,1] > -60) 
)

if pts.shape[0] == 0:
    raise ValueError("血腫正上方沒有顱骨點，請檢查點雲")

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

#輸出
with open(OUTPUT_CANDIDATES_CSV,"w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["x","y","z","layer","dist_mm","angle_deg","radius_used","source"])
    for c in candidates:
        x,y,z = c["point"]
        w.writerow([x,y,z,c["layer"],c["dist_mm"],c["angle_deg"],c["radius_used"],c["src"]])
print(f" 已輸出候選點到: {OUTPUT_CANDIDATES_CSV}")

with open(OUTPUT_SINGLE_CSV,"w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["label","x","y","z"])
    w.writerow(["hematoma_centroid",*centroid_ras])
    w.writerow(["outer_skull",*best["point"]])
    w.writerow(["metadata","value"])
    w.writerow(["dist_mm",best["dist_mm"]])
    w.writerow(["angle_deg",best["angle_deg"]])
    w.writerow(["radius_used",best["radius_used"]])
    w.writerow(["source",best["src"]])
print(f" 已寫入最終入路: {OUTPUT_SINGLE_CSV}")

print("最佳候選 (point):", best["point"])
print(f"distance = {best['dist_mm']:.2f} mm, angle = {best['angle_deg']:.1f}°")
