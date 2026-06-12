"""얼굴 검출 및 전처리.

입력 프레임에서 MTCNN으로 얼굴 영역을 검출하고,
모델 입력 규격에 맞게 크롭·리사이즈한다.
"""

import os
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
from tqdm import tqdm
import torch

device = 'cuda:0'
print("Using device:", device)

BASE_DIR = "/workspace/shared/Wassup11-8"
OUTPUT_DIR = "/workspace/shared/cropped_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mtcnn = MTCNN(image_size=224, margin=20, device=device, keep_all=False)

for split in ['train', 'valid', 'test']:
    split_input_dir = os.path.join(BASE_DIR, split)
    split_output_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(split_output_dir, exist_ok=True)

    for class_name in os.listdir(split_input_dir):
        class_input_dir = os.path.join(split_input_dir, class_name)
        class_output_dir = os.path.join(split_output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        for img_name in tqdm(os.listdir(class_input_dir), desc=f"{split}/{class_name}"):
            img_path = os.path.join(class_input_dir, img_name)
            output_path = os.path.join(class_output_dir, img_name)

            try:
                img = Image.open(img_path).convert('RGB')
                face = mtcnn(img)

                if face is not None:
                    face = face.detach().cpu()
                    face = (face - face.min()) / (face.max() - face.min() + 1e-5)
                    face_img = transforms.ToPILImage()(face)
                    face_img.save(output_path)
                else:
                    continue

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

print("새 데이터셋 경로:", OUTPUT_DIR)
