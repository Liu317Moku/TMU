import SimpleITK as sitk
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


#路徑設定
skull_path = "/home/CranialCTProcessing/label_dataset/seg_z_adjusted.mha"
suture_path = "/home/CranialCTProcessing/label_dataset/coronal_suture_result.mha"
output_dir = "/home/CranialCTProcessing/label_dataset/one_at_five"
os.makedirs(output_dir, exist_ok=True)

# 參數設定
down_weight = 0.4
neighbor_k = 200      # 局部 PCA 鄰域大小（越大越平滑）
exclude_suture_th = 0.5  # 排除冠狀縫本體距離 (mm)
max_range_mm = 50        # 向外延伸距離
step_mm = 1              # 每層厚度 1 mm

#讀取 segmentation
print("讀取影像中...")
skull_img = sitk.ReadImage(skull_path)
suture_img = sitk.ReadImage(suture_path)
skull_arr = sitk.GetArrayFromImage(skull_img)
suture_arr = sitk.GetArrayFromImage(suture_img)

spacing = np.array(skull_img.GetSpacing())
origin = np.array(skull_img.GetOrigin())
direction = np.array(skull_img.GetDirection()).reshape(3, 3)
print("spacing:", spacing)

#  index → physical
def index_to_physical(idxs):
    return origin + (idxs @ (np.diag(spacing) @ direction.T))

suture_inds = np.argwhere(suture_arr > 0)
if len(suture_inds) == 0:
    raise RuntimeError("冠狀顱縫 segmentation 為空")

#  局部 PCA 計算每點法向
print("計算局部法向中...")
nbrs = NearestNeighbors(n_neighbors=neighbor_k).fit(suture_coords)

normals = np.zeros_like(suture_coords)
for i, neigh_idx in enumerate(indices):
    local_pts = suture_coords[neigh_idx]
    pca = PCA(n_components=3)
    n = n + down_weight * np.array([0, 0, -1])  # 下偏修正
    n /= np.linalg.norm(n)

print("修正法向方向中...")
nearest = NearestNeighbors(n_neighbors=1).fit(suture_coords)
nearest_norm = normals[idx[:, 0]]

sample_idx = np.random.choice(len(skull_coords), size=min(2000, len(skull_coords)), replace=False)
proj = np.sum((inside_points - nearest_suture[sample_idx]) * nearest_norm[sample_idx], axis=1)
mean_proj = np.mean(proj)

if mean_proj > 0:
    normals = -normals
    print("法向方向翻轉（改為往顱外）")
else:
    print("法向方向正確（往顱外）")

#  計算 skull 每點到顱縫的投影距離
print("計算顱骨距離中...")
nearest = NearestNeighbors(n_neighbors=1).fit(suture_coords)
dist, idx = nearest.kneighbors(skull_coords)
nearest_suture = suture_coords[idx[:, 0]]

#  儲存每層 mask（細分1mm）
print("開始輸出距離層...")

valid_mask = proj > exclude_suture_th
proj_valid = proj[valid_mask]
skull_inds_valid = skull_inds[valid_mask]

def save_mask(sel_idx, path):
    mask = np.zeros_like(skull_arr, dtype=np.uint8)
    mask[tuple(sel_idx.T)] = 1
    img = sitk.GetImageFromArray(mask)
    
for (a, b) in ranges_mm:
    sel = (proj_valid >= a) & (proj_valid < b)
    if np.sum(sel) == 0:
        continue
    save_mask(skull_inds_valid[sel], path)
    print(f" {a}-{b} mm 輸出完成 ({np.sum(sel)} voxels)")

#  額外輸出 1–50 mm 的整合檔
print("輸出整合檔 (1–50mm)...")
if np.sum(sel_total) > 0:
    path_total = os.path.join(output_dir, "skull_from_suture_1-50mm.mha")
    save_mask(skull_inds_valid[sel_total], path_total)
    print(f" 1–50 mm 整合檔輸出完成 ({np.sum(sel_total)} voxels)")
else:
    print(" 沒有符合 1–50 mm 範圍的點")

print("所有層已輸出完成:", output_dir)

#  可視化 (局部法向)
sample = suture_coords[::300]
sample_n = normals[::300]
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=2, alpha=0.4)
ax.quiver(sample[:, 0], sample[:, 1], sample[:, 2],
          sample_n[:, 0], sample_n[:, 1], sample_n[:, 2],
          length=5, color='r', alpha=0.5)
plt.title("冠狀顱縫局部法向 (已自動修正方向)")
plt.show()
