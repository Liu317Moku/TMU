import SimpleITK as sitk
import numpy as np
import os
import csv
from sklearn.decomposition import PCA

# --- 設定 ---
input_path = "/home/sandy0317/CranialCTProcessing/label_dataset/coronal_suture/output_final_0106_11062357/coronal_suture_result.mha"
output_dir = "point_dataset/coronal_suture_pointline/output_combined0106"
spacing_mm = 1.0   # 方法2: 每 1 mm 取 1 點

os.makedirs(output_dir, exist_ok=True)

# --- 讀取影像 ---
image = sitk.ReadImage(input_path)
mask_array = sitk.GetArrayFromImage(image)  # shape: (z, y, x)

# --- 提取所有非零點 ---
coords_voxel = np.argwhere(mask_array > 0)
if len(coords_voxel) < 10:
    raise ValueError("冠狀縫點數太少，請檢查 segmentation。")
print(f"✅ 找到 {len(coords_voxel)} 個非零點 (冠狀縫 voxel)")

# --- 基本資訊 ---
spacing = np.array(image.GetSpacing())
origin = np.array(image.GetOrigin())
direction = np.array(image.GetDirection()).reshape(3, 3)

# --- voxel ➜ physical ➜ RAS (方法1) ---
def voxel_to_ras(coords_voxel, image):
    points = []
    for voxel_z, voxel_y, voxel_x in coords_voxel:
        voxel_coord = np.array([voxel_x, voxel_y, voxel_z])  # (x,y,z)
        physical_lps = origin + direction @ (voxel_coord * spacing)
        physical_ras = [-physical_lps[0], -physical_lps[1], physical_lps[2]]
        points.append(physical_ras)
    return points

# --- 方法2: 等距 resample ---
def resample_side_points(coords_voxel_side, spacing_mm):
    if len(coords_voxel_side) < 2:
        return []

    # PCA 找主軸
    pca = PCA(n_components=1)
    pca.fit(coords_voxel_side)
    main_direction = pca.components_[0]
    center_point = np.mean(coords_voxel_side, axis=0)

    # 排序
    projections = np.dot(coords_voxel_side - center_point, main_direction)
    sorted_indices = np.argsort(projections)
    sorted_coords = coords_voxel_side[sorted_indices]

    # voxel -> physical (LPS)
    physical_lps = []
    for voxel_z, voxel_y, voxel_x in sorted_coords:
        voxel_coord = np.array([voxel_x, voxel_y, voxel_z])
        pt_lps = origin + direction @ (voxel_coord * spacing)
        physical_lps.append(pt_lps)
    physical_lps = np.array(physical_lps)

    # 累積長度
    segment_lens = np.linalg.norm(np.diff(physical_lps, axis=0), axis=1)
    cumlen = np.insert(np.cumsum(segment_lens), 0, 0.0)
    max_len = cumlen[-1]
    target_distances = np.arange(0, max_len, spacing_mm)

    # 插值取樣
    sampled_points = []
    for td in target_distances:
        idx = np.searchsorted(cumlen, td)
        if idx == 0:
            pt = physical_lps[0]
        elif idx >= len(cumlen):
            pt = physical_lps[-1]
        else:
            t = (td - cumlen[idx-1]) / (cumlen[idx] - cumlen[idx-1])
            pt = physical_lps[idx-1] * (1-t) + physical_lps[idx] * t
        pt_ras = [-pt[0], -pt[1], pt[2]]  # 轉 RAS
        sampled_points.append(pt_ras)
    return sampled_points

# --- 方法1: 全部點 ---
points_all = voxel_to_ras(coords_voxel, image)

# --- 方法2: 左右各自取樣 ---
mid_x = mask_array.shape[2] // 2
left_coords = coords_voxel[coords_voxel[:, 2] < mid_x]
right_coords = coords_voxel[coords_voxel[:, 2] >= mid_x]
points_left = resample_side_points(left_coords, spacing_mm)
points_right = resample_side_points(right_coords, spacing_mm)
points_resampled = points_left + points_right

# --- 儲存 CSV ---
csv_all = os.path.join(output_dir, "coronal_suture_allpoints.csv")
csv_resampled = os.path.join(output_dir, "coronal_suture_resampled.csv")

def save_csv(points, path):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'z'])
        for pt in points:
            writer.writerow([f"{pt[0]:.4f}", f"{pt[1]:.4f}", f"{pt[2]:.4f}"])

save_csv(points_all, csv_all)
save_csv(points_resampled, csv_resampled)

print(f"✅ 方法1: {len(points_all)} 點已輸出到 {csv_all}")
print(f"✅ 方法2: {len(points_resampled)} 點已輸出到 {csv_resampled}")
