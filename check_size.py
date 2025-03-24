from PIL import Image
import os

def check_image_sizes(image_root_dir):
    size_counts = {}

    for root, _, files in os.walk(image_root_dir):
        for fname in files:
            if fname.endswith('.png'):
                path = os.path.join(root, fname)
                try:
                    with Image.open(path) as img:
                        size = img.size  # (width, height)
                        size_counts[size] = size_counts.get(size, 0) + 1
                except Exception as e:
                    print(f"⚠️ {path} 열기 실패: {e}")

    print("📏 이미지 크기별 개수:")
    for size, count in sorted(size_counts.items()):
        print(f"  - {size}: {count}개")

check_image_sizes('sorted_dataset')
