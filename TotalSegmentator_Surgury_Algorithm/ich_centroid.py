import SimpleITK as sitk
import numpy as np

# 讀取 segmentation 影像 (二值化的血腫分割結果) 
seg_path = "/home/sandy0317/TotalSegmentator/dataset/output/final_result5/intracerebral_hemorrhage.nii.gz"
seg_img = sitk.ReadImage(seg_path)

# 轉成 numpy array 
seg_array = sitk.GetArrayFromImage(seg_img)  # (z, y, x)

# 找出血腫的座標點 
coords = np.argwhere(seg_array > 0)  # 只取血腫區域的點

if coords.shape[0] == 0:
    print("Segmentation 內沒有血腫區域！")
else:
    # 計算 3D 質心 (以 voxel index 為單位)
    centroid_voxel = coords.mean(axis=0)  # (z, y, x)

    # 轉換為物理座標 (mm, LPS 系統)
    centroid_physical_lps = seg_img.TransformContinuousIndexToPhysicalPoint(centroid_voxel[::-1])

    # LPS → RAS 轉換
    x_lps, y_lps, z_lps = centroid_physical_lps
    centroid_physical_ras = (-x_lps, -y_lps, z_lps)

    print("血腫質心 (voxel index):", centroid_voxel)         # voxel index
    print("血腫質心 (LPS mm):", centroid_physical_lps)       # ITK / DICOM 座標
    print("血腫質心 (RAS mm):", centroid_physical_ras)       # 3D Slicer 座標

#TotalSegmentator -i dataset/input/0131.nii.gz -o dataset/output/final_result --task cerebral_bleed  執行腦出血切割