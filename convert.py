import os
import shutil
import csv

def convert(input_dir="sorted_dataset", output_dir="data"):
    sample_output_dir = os.path.join(output_dir, "samples")
    os.makedirs(sample_output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "labels.csv")
    
    label_map = {"con": 0, "inc": 1}
    rows = []
    
    for label_name in ["con", "inc"]:
        class_dir = os.path.join(input_dir, label_name)
        if not os.path.isdir(class_dir):
            continue
        
        for sample_name in sorted(os.listdir(class_dir)):
            sample_path = os.path.join(class_dir, sample_name)
            if not os.path.isdir(sample_path):
                continue
            
            sample_id = sample_name
            for view in ["axial", "coronal", "sagittal"]:
                src = os.path.join(sample_path, f"{view}.png")
                dst = os.path.join(sample_output_dir, f"{sample_id}_{view}.png")
                if os.path.exists(src):
                    shutil.copy(src, dst)
                else:
                    print(f"누락: {src}")
            
            rows.append([sample_id, label_map[label_name]])
    
    # CSV 파일 저장
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'label'])  # 헤더
        for sample_id, label in rows:
            writer.writerow([sample_id, label])  # 각각 따로 쓰기

    
    print(f"✅ 변환 완료: {len(rows)}개 샘플이 portable_dataset에 저장되었습니다.")

convert()
        