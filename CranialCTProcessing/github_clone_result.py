
import SimpleITK as sitk
import DataProcessing, ModelConfiguration
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(BASE_DIR, "dataset/input/0106.mha")
output_path = os.path.join(BASE_DIR, "dataset/output/final_data_0106.mha")
ct = sitk.ReadImage(input_path)


mask = DataProcessing.CreateBoneMask(ct)

ct_resampled = DataProcessing.ResampleAndMaskImage(ct, mask)

model = ModelConfiguration.adaptModel('Model.dat', ModelConfiguration.getDevice())

data = ModelConfiguration.adaptData(ct_resampled, ModelConfiguration.getDevice())

landmarks, bone_labels = ModelConfiguration.runModel(model, ct_resampled, mask, data)

sitk.WriteImage(bone_labels, output_path)

print(landmarks.GetPoints())