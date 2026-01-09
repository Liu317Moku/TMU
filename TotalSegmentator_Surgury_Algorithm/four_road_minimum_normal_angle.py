import SimpleITK as sitk
import numpy as np
import csv
import os
import re
import time
from sklearn.neighbors import NearestNeighbors

# ------------------------------
# 路徑設定
# ------------------------------
seg_path = "/home/sandy0317/TotalSegmentator/dataset/output/final_result9/intracerebral_hemorrhage.nii.gz"
forward_dir =  "/home/sandy0317/CranialCTProcessing/point_dataset/coronal_forward_points_0118_10151132"
coronal_csv = "/home/sandy0317/CranialCTProcessing/point_dataset/coronal_suture_pointline/output_combined0118/coronal_suture_resampled.csv"
output_csv = "/home/sandy0317/TotalSegmentator/point_dataset/118_12110157_surgical_entry_points_final.csv"
candidates_csv = "/home/sandy0317/TotalSegmentator/point_dataset/118_12110157_surgical_entry_candidates_final.csv"

# ------------------------------
# 工具函式
# ------------------------------
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

def local_normal_pca_with_nbrs(points, query_point, nbrs, k=30):
    """Use pre-fitted NearestNeighbors object for local PCA normal estimation"""
    _, idx = nbrs.kneighbors(query_point.reshape(1, -1), n_neighbors=min(k, points.shape[0]))
    neighborhood = points[idx[0]]
    X = neighborhood - neighborhood.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, np.argmin(eigvals)]
    return normal / np.linalg.norm(normal)

# ------------------------------
# Step 1: 血腫質心 (RAS)
# ------------------------------
seg_img = sitk.ReadImage(seg_path)
seg_array = sitk.GetArrayFromImage(seg_img)
coords = np.argwhere(seg_array > 0)
if coords.shape[0] == 0:
    raise ValueError("Segmentation 內沒有血腫區域！")
centroid_voxel = coords.mean(axis=0)
centroid_lps = seg_img.TransformContinuousIndexToPhysicalPoint(centroid_voxel[::-1])
centroid_ras = np.array([-centroid_lps[0], -centroid_lps[1], centroid_lps[2]])
print("🩸 血腫質心 (RAS):", centroid_ras)

# ------------------------------
# Step 2: 顱骨點雲
# ------------------------------
all_points, layer_labels, source_files = read_point_cloud_from_dir(forward_dir)
if all_points.size == 0:
    raise ValueError("資料夾中沒有有效顱骨點！")
print(f"🔢 讀取顱骨點數: {all_points.shape[0]}")

# === 建立 NearestNeighbors 索引
k_normal = 30
print("⏳ 建立全域鄰居索引中...")
t0 = time.time()
nbrs_global = NearestNeighbors(n_neighbors=min(k_normal, all_points.shape[0]), n_jobs=-1)
nbrs_global.fit(all_points)
print(f"✅ NearestNeighbors 建立完成，耗時 {time.time()-t0:.1f} 秒")

# ------------------------------
# Step 3: 冠狀縫
# ------------------------------
suture_pts = []
with open(coronal_csv, newline="") as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        try:
            suture_pts.append([float(row[0]), float(row[1]), float(row[2])])
        except:
            continue
suture_pts = np.array(suture_pts)
if suture_pts.shape[0] < 10:
    raise ValueError("冠狀縫點不足！")
suture_center = suture_pts.mean(axis=0)

# estimate plane
cov = np.cov(suture_pts - suture_center, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
suture_normal = eigvecs[:, np.argmin(eigvals)]
if suture_normal[1] < 0:
    suture_normal = -suture_normal
A, B, C = suture_normal
D = -np.dot(suture_normal, suture_center)
dist_to_plane = A * all_points[:,0] + B * all_points[:,1] + C * all_points[:,2] + D
front_mask = dist_to_plane > 0
front_points = all_points[front_mask]
front_labels = layer_labels[front_mask]
front_sources = source_files[front_mask]
print(f"✅ 前方顱骨點數 (供 fallback): {front_points.shape[0]}")

# ------------------------------
# Step 4: 垂直投影候選 (XY距離 + Z高 + 前方 + y>-80)
# ------------------------------
vertical_radii = [5.0,10.0,20.0,40.0]
min_height_above = 5.0
w_dist, w_angle = 0.6, 0.4
candidates = []
centroid_xy = centroid_ras[[0,1]]

for r in vertical_radii:
    dx = all_points[:,0] - centroid_xy[0]
    dy = all_points[:,1] - centroid_xy[1]
    dist_xy = np.sqrt(dx*dx + dy*dy)
    mask = (dist_xy <= r) & (all_points[:,2] >= centroid_ras[2] + min_height_above) & (dist_to_plane > 0) & (all_points[:,1] > -80)
    pts = all_points[mask]
    labs = layer_labels[mask]
    srcs = source_files[mask]
    if pts.shape[0] > 0:
        print(f"🔎 半徑 {r}mm 找到 {pts.shape[0]} 個候選點")
        for i, p in enumerate(pts):
            if i % 100 == 0:
                print(f"  處理 {i}/{pts.shape[0]} ...")
            d = np.linalg.norm(p - centroid_ras)
            try:
                ln = local_normal_pca_with_nbrs(all_points, p, nbrs_global, k=k_normal)
                if ln[2] < 0: ln = -ln
            except:
                ln = None
            path = centroid_ras - p
            plen = np.linalg.norm(path)
            if plen == 0: continue
            path_unit = path / plen
            angle = np.degrees(np.arccos(np.clip(np.dot(path_unit, ln), -1.0, 1.0))) if ln is not None else 90.0
            candidates.append({"point": p, "layer": labs[i], "src": srcs[i],
                               "dist_mm": d, "angle_deg": angle, "radius_used": r})
        break

# ------------------------------
# Step 5: fallback 使用 V 區或所有前方顱骨點 (y>-80)
# ------------------------------
if len(candidates) == 0:
    print("⚠️ 垂直投影沒找到候選，使用前方顱骨 fallback")
    mask_fallback = front_points[:,1] > -70
    pts_all = front_points[mask_fallback]
    labs_all = front_labels[mask_fallback]
    srcs_all = front_sources[mask_fallback]

    max_candidates = 500
    distances_to_centroid = np.linalg.norm(pts_all - centroid_ras, axis=1)
    sorted_idx = np.argsort(distances_to_centroid)[:max_candidates]
    pts = pts_all[sorted_idx]
    labs = labs_all[sorted_idx]
    srcs = srcs_all[sorted_idx]

    for i, p in enumerate(pts):
        if i % 50 == 0:
            print(f"  fallback 計算 {i}/{pts.shape[0]} ...")
        d = np.linalg.norm(p - centroid_ras)
        try:
            ln = local_normal_pca_with_nbrs(all_points, p, nbrs_global, k=k_normal)
            if ln[2] < 0: ln = -ln
        except:
            ln = None
        path = centroid_ras - p
        plen = np.linalg.norm(path)
        if plen == 0: continue
        path_unit = path / plen
        angle = np.degrees(np.arccos(np.clip(np.dot(path_unit, ln), -1.0, 1.0))) if ln is not None else 90.0
        candidates.append({
            "point": p, "layer": labs[i], "src": srcs[i],
            "dist_mm": d, "angle_deg": angle, "radius_used": "V_zone_front_points"
        })

# ------------------------------
# Step 6: 選最佳候選
# ------------------------------
if len(candidates) == 0:
    raise ValueError("所有策略均找不到候選點！")

d_list = np.array([c["dist_mm"] for c in candidates])
a_list = np.array([c["angle_deg"] for c in candidates])
d_norm = (d_list - d_list.min()) / (d_list.max() - d_list.min() + 1e-8)
a_norm = a_list / 180.0
scores = w_dist * d_norm + w_angle * a_norm
best_idx = int(np.argmin(scores))
best = candidates[best_idx]

# ------------------------------
# Step 7: 輸出 CSV
# ------------------------------
with open(candidates_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "z", "layer_mm", "dist_mm", "angle_deg", "radius_used", "source"])
    for c in candidates:
        x, y, z = c["point"]
        writer.writerow([x, y, z, c["layer"], c["dist_mm"], c["angle_deg"], c["radius_used"], c["src"]])
print(f"🗂 候選點清單: {candidates_csv}")

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "x", "y", "z"])
    writer.writerow(["hematoma_centroid", *centroid_ras])
    writer.writerow(["outer_skull", *best["point"]])
    writer.writerow([])
    writer.writerow(["metadata", "value"])
    writer.writerow(["chosen_distance_mm", best["dist_mm"]])
    writer.writerow(["chosen_angle_deg", best["angle_deg"]])
    writer.writerow(["chosen_radius_used", best["radius_used"]])
    writer.writerow(["chosen_source", best["src"]])
print(f"✅ 最終入路: {output_csv}")
