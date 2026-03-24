import SimpleITK as sitk
import numpy as np
import os
import csv
from sklearn.decomposition import PCA

#  參數設定
input_dir = "/home/CranialCTProcessing/label_dataset/one_at_five"
output_dir = "point_dataset/"
spacing_mm = 1.0  # 每 1 mm 取 1 點
os.makedirs(output_dir, exist_ok=True)

# 共用函式
def voxel_to_ras(coords_voxel, image):
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    points = []
    for voxel_z, voxel_y, voxel_x in coords_voxel:
        voxel_coord = np.array([voxel_x, voxel_y, voxel_z])
        pt_lps = origin + direction @ (voxel_coord * spacing)
        pt_ras = [-pt_lps[0], -pt_lps[1], pt_lps[2]]
        points.append(pt_ras)
    return points


def resample_side_points(coords_voxel_side, spacing_mm, image):
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    direction = np.array(image.GetDirection()).reshape(3, 3)

    # PCA 主軸
    pca = PCA(n_components=1)
    pca.fit(coords_voxel_side)
    main_dir = pca.components_[0]
    center = np.mean(coords_voxel_side, axis=0)

    # 排序
    proj = np.dot(coords_voxel_side - center, main_dir)
    sorted_coords = coords_voxel_side[sorted_idx]

    # voxel -> physical (LPS)
    pts_lps = []
    for voxel_z, voxel_y, voxel_x in sorted_coords:
        voxel_coord = np.array([voxel_x, voxel_y, voxel_z])
        pt = origin + direction @ (voxel_coord * spacing)
    pts_lps = np.array(pts_lps)

    # 累積距離
    seg_len = np.linalg.norm(np.diff(pts_lps, axis=0), axis=1)
    max_len = cumlen[-1]
    targets = np.arange(0, max_len, spacing_mm)

    # 插值取樣
    sampled_pts = []
    for td in targets:
        idx = np.searchsorted(cumlen, td)
        if idx == 0:
            pt = pts_lps[0]
        else:
            t = (td - cumlen[idx-1]) / (cumlen[idx] - cumlen[idx-1])
            pt = pts_lps[idx-1] * (1 - t) + pts_lps[idx] * t
    return sampled_pts


def save_csv(points, out_path):
    with open(out_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])
        for pt in points:
            writer.writerow([f"{pt[0]:.4f}", f"{pt[1]:.4f}", f"{pt[2]:.4f}"])


# 批次處理
all_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".mha")])
print(f" 偵測到 {len(all_files)} 個影像檔案可處理")

for fname in all_files:
    input_path = os.path.join(input_dir, fname)
    print(f"\n 處理中: {fname}")

    image = sitk.ReadImage(input_path)
    coords_voxel = np.argwhere(mask_array > 0)

    if len(coords_voxel) < 10:
        print(f" {fname} 非零點太少，略過。")
        continue

    mid_x = mask_array.shape[2] // 2
    left_coords = coords_voxel[coords_voxel[:, 2] < mid_x]
    right_coords = coords_voxel[coords_voxel[:, 2] >= mid_x]

    points_left = resample_side_points(left_coords, spacing_mm, image)
    points_right = resample_side_points(right_coords, spacing_mm, image)
    points_all = points_left + points_right

    # 儲存
    out_csv = os.path.join(output_dir, fname.replace(".mha", "_points.csv"))
    save_csv(points_all, out_csv)
    print(f" {fname} 轉換完成，共 {len(points_all)} 點，輸出至 {out_csv}")

print("\n 全部影像轉換完成！")
