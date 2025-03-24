import os

def count_png_files_in_slices(root_dir):
    for subject in sorted(os.listdir(root_dir)):
        if not subject.startswith("sub-"):
            continue  # 'data', 'src' 등은 무시

        slice_dir = os.path.join(root_dir, subject, "slices_out")
        if not os.path.exists(slice_dir):
            print(f"⚠️ {subject}: slices_out 폴더 없음")
            continue

        png_count = len([f for f in os.listdir(slice_dir) if f.endswith('.png')])
        print(f"{subject}/slices_out/: {png_count}개 PNG 파일")

# 사용 예시
count_png_files_in_slices(".")
