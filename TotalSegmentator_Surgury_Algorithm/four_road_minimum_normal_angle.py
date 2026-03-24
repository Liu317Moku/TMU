import SimpleITK as sitk
import numpy as np
import csv
import os
import re
import time
from sklearn.neighbors import NearestNeighbors

#路徑設定
seg_path = "/home/TotalSegmentator/dataset/intracerebral_hemorrhage.nii.gz"
forward_dir =  "/home/CranialCTProcessing/point_dataset/coronal_forward_points"
coronal_csv = "/home/CranialCTProcessing/point_dataset/coronal_suture_pointline/coronal_suture_resampled.csv"
output_csv = "/home/TotalSegmentator/point_dataset/urgical_entry_points_final.csv"
candidates_csv = "/home/TotalSegmentator/point_dataset/surgical_entry_candidates_final.csv"

#工具函式
def read_point_cloud_from_dir(forward_dir):
    all_points, layer_labels, source_files = [], [], []
    for file in sorted(os.listdir(forward_dir)):
        if file.endswith(".csv"):
            csv_path = os.path.join(forward_dir, file)
            with open(csv_path, newline="") as csvfile:
                reader = csv.reader(csvfile)
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
    idx = nbrs.kneighbors(query_point.reshape(1, -1), n_neighbors=min(k, points.shape[0]))
    neighborhood = points[idx[0]]
    X = neighborhood.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, np.argmin(eigvals)]
    return normal / np.linalg.norm(normal)

#血腫質心 (RAS)
seg_img = sitk.ReadImage(seg_path)
seg_array = sitk.GetArrayFromImage(seg_img)
centroid_voxel = coords.mean(axis=0)
centroid_lps = seg_img.TransformContinuousIndexToPhysicalPoint(centroid_voxel[::-1])
centroid_ras = np.array([-centroid_lps[0], -centroid_lps[1], centroid_lps[2]])
print("血腫質心 (RAS):", centroid_ras)

# 顱骨點雲
all_points, layer_labels, source_files = read_point_cloud_from_dir(forward_dir)
print(f"讀取顱骨點數: {all_points.shape[0]}")

#建立 NearestNeighbors 索引
k_normal = 30
print(" 建立全域鄰居索引中...")
nbrs_global = NearestNeighbors(n_neighbors=min(k_normal, all_points.shape[0]), n_jobs=-1)
print(f" NearestNeighbors 建立完成，耗時 {time.time()-t0:.1f} 秒")

#冠狀縫
suture_pts = []
with open(coronal_csv, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            suture_pts.append([float(row[0]), float(row[1]), float(row[2])])
        except:
            continue
suture_pts = np.array(suture_pts)
suture_center = suture_pts.mean(axis=0)

# estimate plane
cov = np.cov(suture_pts - suture_center, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
suture_normal = eigvecs[:, np.argmin(eigvals)]
front_mask = dist_to_plane > 0
front_points = all_points[front_mask]
front_labels = layer_labels[front_mask]
front_sources = source_files[front_mask]
print(f"前方顱骨點數 (供 fallback): {front_points.shape[0]}")

#垂直投影候選 (XY距離 + Z高 + 前方 + y>-80)
centroid_xy = centroid_ras[[0,1]]

for r in vertical_radii:
    dx = all_points[:,0] - centroid_xy[0]
    dy = all_points[:,1] - centroid_xy[1]
    dist_xy = np.sqrt(dx*dx + dy*dy)
    pts = all_points[mask]
    labs = layer_labels[mask]
    srcs = source_files[mask]
    if pts.shape[0] > 0:
        print(f"半徑 {r}mm 找到 {pts.shape[0]} 個候選點")
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
            candidates.append({"point": p, "layer": labs[i], "src": srcs[i],
                               "dist_mm": d, "angle_deg": angle, "radius_used": r})
        break

if len(candidates) == 0:
    print("垂直投影沒找到候選，使用前方顱骨 fallback")
    pts_all = front_points[mask_fallback]
    labs_all = front_labels[mask_fallback]

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
        angle = np.degrees(np.arccos(np.clip(np.dot(path_unit, ln), -1.0, 1.0))) if ln is not None else 90.0
        candidates.append({
            "point": p, "layer": labs[i], "src": srcs[i],
            "dist_mm": d, "angle_deg": angle, "radius_used": "V_zone_front_points"
        })

#選最佳候選

if len(candidates) == 0:
    raise ValueError("所有策略均找不到候選點！")

d_list = np.array([c["dist_mm"] for c in candidates])
a_list = np.array([c["angle_deg"] for c in candidates])
d_norm = (d_list - d_list.min()) / (d_list.max() - d_list.min() + 1e-8)
a_norm = a_list / 180.0

#輸出 CSV
with open(candidates_csv, "w", newline="") as f:
    writer.writerow(["x", "y", "z", "layer_mm", "dist_mm", "angle_deg", "radius_used", "source"])
    for c in candidates:
        x, y, z = c["point"]
print(f"候選點清單: {candidates_csv}")

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
print(f"最終入路: {output_csv}")
