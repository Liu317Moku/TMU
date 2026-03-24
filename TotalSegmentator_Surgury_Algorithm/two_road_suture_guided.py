import SimpleITK as sitk
import numpy as np
import csv
import os
import re
from sklearn.neighbors import NearestNeighbors

#Paths
seg_path = "/home/TotalSegmentator/dataset/output/final_result9/intracerebral_hemorrhage.nii.gz"
forward_dir =  "/home/CranialCTProcessing/point_dataset/coronal_forward_points"
coronal_csv = "/home/CranialCTProcessing/point_dataset/coronal_suture_pointline/coronal_suture_resampled.csv"
output_csv = "/home/TotalSegmentator/point_dataset/surgical_entry_points_final.csv"

# Debug options
save_candidates_csv = False
candidates_csv_path = "/home/TotalSegmentator/point_dataset/candidates_debug.csv"

# Parameters for "shrunk" V-region + top-K processing
# 這組參數比之前更保守：側向 lateral 範圍小、upward 也較保守
upward_offsets = [8.0, 10.0, 12.0]        # mm above suture_center
lateral_limits = [30.0, 40.0, 50.0]      # lateral distance from midline
mid_excludes = [5.0, 8.0, 10.0]          # exclude 中線附近的距離
y_window_height = 8.0                    # y 範圍高度（縮小，原本是 10）
top_k_for_angle = 500                    # 只對 top K 最近候選計算局部法向與角度
angle_weight_mm = 6.0                    # 角度懲罰權重（可調）

# Utility: estimate local surface normal using PCA on k-nearest neighbors
def local_normal_pca(points, query_point, k=30):
    """
    points: (N,3) pointcloud (assumed RAS)
    query_point: (3,)
    returns: normal vector (unit) with arbitrary sign
    """
    if points.shape[0] < 3:
        raise ValueError("點雲數量不足以估算法向")
    nbrs = NearestNeighbors(n_neighbors=min(k, points.shape[0]), algorithm='auto').fit(points)
    idx = nbrs.kneighbors(query_point.reshape(1, -1))
    neighborhood = points[idx[0]]
    X = neighborhood.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    eigvals= np.linalg.eigh(cov)
    normal = normal / np.linalg.norm(normal)
    return normal

#compute hematoma centroid (RAS)
seg_img = sitk.ReadImage(seg_path)
seg_array = sitk.GetArrayFromImage(seg_img)  # z,y,x (voxel index order)
coords = np.argwhere(seg_array > 0)
if coords.shape[0] == 0:
    raise ValueError("Segmentation 內沒有血腫區域！")

centroid_voxel = coords.mean(axis=0)  # in array index order (z,y,x)
centroid_lps = seg_img.TransformContinuousIndexToPhysicalPoint(centroid_voxel[::-1])  # (x,y,z) in LPS
centroid_ras = np.array([-centroid_lps[0], -centroid_lps[1], centroid_lps[2]])
print("血腫質心 (RAS):", centroid_ras)

#read all outer skull points (assume CSVs contain RAS)
all_points_list, layer_labels, source_files = [], [], []

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
                        all_points_list.append(pt)
                        layer_labels.append(layer_mm)
                        source_files.append(f"{csv_path}:{row_idx+2}")
                    except:
                        continue

all_points = np.array(all_points_list)
if all_points.size == 0:
    raise ValueError("資料夾中沒有任何有效的外層顱骨點！")

print(f"讀取到 {all_points.shape[0]} 顆外層顱骨點")

#read coronal suture and fit plane
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
    raise ValueError("冠狀縫點不足，請確認輸入檔案！")

suture_center = np.mean(suture_pts, axis=0)
X = suture_pts - suture_center
cov = np.cov(X, rowvar=False)
eigvecs = eigvecs[:, order]
suture_normal = eigvecs[:, 2]

print(f"冠狀縫法向量: {suture_normal}")
print(f"冠狀縫中心: {suture_center}")

#filter skull points in front of the suture plane
front_mask = dist_to_plane > 0
filtered_points_init = all_points[front_mask]

# try flip normal if nothing found
if filtered_points_init.shape[0] == 0:
    print("找不到位於冠狀縫法向正側的點，嘗試翻轉法向重試。")
    front_mask = dist_to_plane > 0
    filtered_points_init = all_points[front_mask]

if filtered_points_init.shape[0] == 0:
    raise ValueError("沒有位於冠狀縫前方的顱骨點！請檢查座標系方向或輸入點雲。")

print(f" 篩出 {filtered_points_init.shape[0]} 個在冠狀縫前方的顱骨點 (初始)")

#shrunk V-region 收集候選（較保守）
midline_x = np.median(suture_pts[:, 0])
print(f" 血腫位於 X={centroid_ras[0]:.2f}, Y={centroid_ras[1]:.2f}, Z={centroid_ras[2]:.2f}")

y_min_all, y_max_all = pts_all[:, 1].min(), pts_all[:, 1].max()
print(f" 前方顱骨 Y 範圍: {y_min_all:.1f} ~ {y_max_all:.1f}")

candidate_pts_list = []

for upward in upward_offsets:
    for lateral in lateral_limits:
        for mid_ex in mid_excludes:
            y_min_limit = suture_center[1] + upward
            y_max_limit = y_min_limit + y_window_height
            # lateral windows (左右)
            right_mask = (pts_all[:, 0] > midline_x + mid_ex) & (pts_all[:, 0] < midline_x + lateral)
            left_mask  = (pts_all[:, 0] < midline_x - mid_ex) & (pts_all[:, 0] > midline_x - lateral)
            region_mask = up_mask & (right_mask | left_mask)
            cand_count = np.count_nonzero(region_mask)
            print(f"嘗試 upward={upward} lateral={lateral} mid_ex={mid_ex} -> Y∈[{y_min_limit:.1f},{y_max_limit:.1f}] 候選={cand_count}")
                
# 合併所有候選（如果有）
if len(candidate_pts_list) > 0:
    pts = np.vstack(candidate_pts_list)
    print(f" 共收集到 {pts.shape[0]} 個 V 區候選點 (全參數組合)")
else:
    pts = None
    print(" 未能在任何參數組合中找到 V 區候選點")

# optional: save candidates for debug
if save_candidates_csv and (pts is not None):
    with open(candidates_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x","y","z","layer_mm","source"])
        for i,p in enumerate(pts):
            w.writerow([p[0], p[1], p[2]])
    print(f" 已輸出候選點至 {candidates_csv_path}")


#選點策略：
#    - 若有 V 區候選：先用距離排序，取 top_k_for_angle 最近的 few hundred
#      再對這些計算 local normal + angle，並用 score = distance + angle_weight * (angle_deg / 90)
#      最小 score 為最終選點（angle 被歸一化到 0..1）
#    - 若沒有候選：回退到前方最近點
def write_output_and_print(entry_point, entry_layer, entry_src, centroid, angle_deg=None, dist_mm=None, params=None):
    print("────────────────────────────")
    print(f" 建議外層顱骨點 (RAS): {entry_point}")
    print(f" 層距: {entry_layer} mm")
    if dist_mm is not None:
        print(f" 與血腫質心距離: {dist_mm:.2f} mm")
    if angle_deg is not None:
        print(f" 與局部顱骨法向夾角: {angle_deg:.1f} °")
    print(f" 來源檔案: {entry_src}")
    print("────────────────────────────")
    # 輸出 CSV (寫入更多欄位)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "x", "y", "z"])
        writer.writerow(["hematoma_centroid", *centroid])
        writer.writerow(["outer_skull", *entry_point])
        writer.writerow([])
        writer.writerow(["metadata", "value"])
        if params is not None:
            writer.writerow(["chosen_parameters", params])
        if dist_mm is not None:
            writer.writerow(["distance_mm", dist_mm])
        if angle_deg is not None:
            writer.writerow(["angle_deg_local_normal", angle_deg])
    print(f"已輸出結果至: {output_csv}")

if pts is None or pts.shape[0] == 0:
    print("自動參數嘗試未找到紅色 V 區候選點，將退回到「前方最近點」作為後備。")
    distances_all = np.linalg.norm(pts_all - centroid_ras, axis=1)
    min_idx_all = np.argmin(distances_all)

    try:
        ln = local_normal_pca(all_points, closest_point, k=30)
        path_vec = (centroid_ras - closest_point)
        angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(path_vec, ln)), -1.0, 1.0)))
    except Exception as e:
        print(" 無法估算局部法向:", e)

    write_output_and_print(centroid_ras, angle_deg=angle, dist_mm=closest_distance, params=None)

else:
    # 先以距離排序並取 top-K（避免對 100w 點計算法向）
    top_k = min(top_k_for_angle, pts.shape[0])
    top_idx = sort_idx[:top_k]

    top_labs = labs[top_idx]
    top_dists = distances[top_idx]

    best_score = float("inf")
    best_i = None
    best_angle = None

    for i_local, p in enumerate(top_pts):
        try:
            ln = local_normal_pca(all_points, p, k=30)
            path_vec = (centroid_ras - p)
            angle_deg = np.degrees(np.arccos(np.abs(cosang)))  # 0..180
        except Exception as e:
            angle_deg = 90.0

        # normalize angle 0..1 by 90deg (90deg = 1.0)

        if score < best_score:
            best_i = i_local


    closest_layer = top_labs[best_i]
    closest_distance = top_dists[best_i]
    angle_deg = best_angle

    params_dict = {
        "upward_options": upward_offsets,
        "lateral_options": lateral_limits,
        "mid_exclude_options": mid_excludes,
        "y_window_height": y_window_height,
        "top_k_for_angle": top_k_for_angle,
        "angle_weight_mm": angle_weight_mm
    }
    write_output_and_print(angle_deg=angle_deg, dist_mm=closest_distance, params=params_dict)
