import os
import shutil
import re
from collections import defaultdict

def organize_images(input_root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(r"slice_output_(\d+)_t(\d+)_(axial|coronal|sagittal)_(con|inc)\.png")
    sample_dict = defaultdict(dict)

    for subject in os.listdir(input_root_dir):
        subject_path = os.path.join(input_root_dir, subject)
        if not os.path.isdir(subject_path):
            continue
        if not subject.startswith("sub-"):
            continue

        slice_dir = os.path.join(subject_path, "slices_out")
        if not os.path.exists(slice_dir):
            continue

        for fname in os.listdir(slice_dir):
            match = pattern.match(fname)
            if match:
                run, time, view, label = match.groups()
                # ✅ subject명을 포함한 고유 key 사용
                key = f"{subject}_{run}_{time}"
                if 'views' not in sample_dict[key]:
                    sample_dict[key]['views'] = {}
                sample_dict[key]['views'][view] = os.path.join(slice_dir, fname)
                sample_dict[key]['label'] = label

    sample_count = 0
    for key, sample in sample_dict.items():
        views = sample.get('views', {})
        if len(views) != 3:
            continue  # axial, coronal, sagittal 모두 있어야 함

        label = sample['label']
        sample_count += 1
        sample_name = f"sample_{sample_count:04d}"
        sample_folder = os.path.join(output_dir, label, sample_name)
        os.makedirs(sample_folder, exist_ok=True)

        for view, src_path in views.items():
            dst_path = os.path.join(sample_folder, f"{view}.png")
            shutil.copy(src_path, dst_path)

    print(f"✅ 총 {sample_count}개의 샘플이 정리되었습니다.")

organize_images(input_root_dir=".", output_dir="sorted_dataset")