import SimpleITK as sitk
import numpy as np
import os
from scipy.ndimage import (
    binary_dilation, binary_closing, binary_fill_holes, distance_transform_edt
)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


# 參數設定
ct_path = "/home/sandy0317/CranialCTProcessing/dataset/input/0106.mha"
seg_path = "/home/sandy0317/CranialCTProcessing/dataset/output/final_data_0106.mha"
output_dir = "label_dataset/coronal_suture/output_final_0106_11062357"
os.makedirs(output_dir, exist_ok=True)

coronal_pairs = [(1, 3), (2, 4)]  
z_scale_threshold = 1.2  


# 讀取影像
print("讀取影像中...")
ct_img = sitk.ReadImage(ct_path)
seg_img = sitk.ReadImage(seg_path)
print("原始 CT spacing:", ct_img.GetSpacing())
print(" Seg spacing:", seg_img.GetSpacing())


# 函式：僅調整 Z 軸
def resample_adjust_z(image, z_scale_thresh=1.2, is_label=False):
    orig_spacing = image.GetSpacing()
    z_spacing = orig_spacing[2]

    if z_spacing / xy_mean > z_scale_thresh:
        new_z = xy_mean
        print(f" Z spacing {z_spacing:.3f} 過大，調整為 {new_z:.3f}")
    else:
        new_z = z_spacing
        print(f"Z spacing {z_spacing:.3f} 保持原值")

    new_spacing = (orig_spacing[0], orig_spacing[1], new_z)
    new_size = [int(round(orig_size[i] * (orig_spacing[i] / new_spacing[i]))) for i in range(3)]
    print(f" 新影像尺寸: {new_size}, spacing: {new_spacing}")

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(image)


#調整 CT 與 Segmentation Z 軸
print("\n調整 CT Z 軸...")
ct_out_path = os.path.join(output_dir, "ct_z_adjusted.mha")
sitk.WriteImage(ct_iso, ct_out_path)
print(f" 已輸出 CT：{ct_out_path}")

print("\n 調整 Segmentation Z 軸...")
seg_out_path = os.path.join(output_dir, "seg_z_adjusted.mha")
sitk.WriteImage(seg_iso, seg_out_path)
print(f" 已輸出 Segmentation：{seg_out_path}")


#  冠狀縫偵測
seg_arr = sitk.GetArrayFromImage(seg_iso)
result_array = np.zeros_like(seg_arr, dtype=np.uint8)

for (a, b) in coronal_pairs:
    region_a = (seg_arr == a)
    region_b = (seg_arr == b)
    suture_closed = binary_closing(suture_region, iterations=2)
    suture_dilated = binary_dilation(suture_closed, iterations=2)
    suture_filled = binary_fill_holes(suture_dilated)

#  儲存冠狀縫結果
suture_img = sitk.GetImageFromArray(result_array)
suture_out_path = os.path.join(output_dir, "coronal_suture_result.mha")
sitk.WriteImage(suture_img, suture_out_path)
print(f" 冠狀縫結果已輸出至: {suture_out_path}")

#  PCA 主軸分析
print("\n 執行 PCA 主軸分析...")
coords = np.column_stack(np.nonzero(result_array))

if len(coords) > 0:
    pca = PCA(n_components=3)

    print(" PCA 主軸方向（第一主成分）:")
    print(pca.components_[0])
    print(" 各軸解釋變異比例:", pca.explained_variance_ratio_)

    # 3D 視覺化 (冠狀縫 + 主軸線)
    mean_point = pca.mean_
    direction = pca.components_[0]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[::10, 2], coords[::10, 1], coords[::10, 0], s=1, alpha=0.3)
    ax.quiver(mean_point[2], mean_point[1], mean_point[0],
              direction[2], direction[1], direction[0],
              length=50, color='r', linewidth=3)
    ax.set_title("PCA 主軸與冠狀縫點雲")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
else:
    print(" 冠狀縫區域為空，無法進行 PCA 分析。")

#  中軸切片檢查 (灰階顯示)
mid_slice = result_array.shape[0] // 2
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("CT (Z mid)")
plt.imshow(sitk.GetArrayFromImage(ct_iso)[mid_slice, :, :], cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Segmentation (Z mid)")
plt.imshow(seg_arr[mid_slice, :, :], cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Coronal Suture (Z mid)")
plt.imshow(result_array[mid_slice, :, :], cmap='gray')
plt.axis('off')

plt.show()
