"""실시간 표정 인식 데모

웹캠 또는 이미지 파일에서 표정 5클래스(anger, closed, happy, panic, sadness) 분류

사용법:
    python demo_emotion.py --webcam
    python demo_emotion.py --image path/to/face.jpg

가중치(models/effemotenet_infer.pt)는 GitHub Releases에서 다운로드
facenet-pytorch 있으면 MTCNN으로 얼굴 크롭, 없으면 프레임 전체 사용
"""

import argparse
import time

import torch
from PIL import Image
from torchvision import transforms

from models.effemotenet_infer import CLASS_NAMES, INPUT_SIZE, add_sobel_channel, load_model

DEFAULT_WEIGHTS = "models/effemotenet_infer.pt"

# 학습 때 전처리랑 동일하게: 300x300 리사이즈 -> 텐서 -> Y-Sobel 4번째 채널 추가
TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    add_sobel_channel,
])


def get_face_cropper(device):
    # facenet-pytorch 있으면 MTCNN 크롭 함수 반환, 없으면 None
    try:
        from facenet_pytorch import MTCNN
    except ImportError:
        print("[안내] facenet-pytorch 미설치 -> 얼굴 크롭 없이 전체 프레임 사용")
        return None

    mtcnn = MTCNN(image_size=224, margin=20, device=device, keep_all=False)

    def crop(pil_img):
        box, _ = mtcnn.detect(pil_img)
        if box is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in box[0]]
        return pil_img.crop((max(x1, 0), max(y1, 0), x2, y2))

    return crop


@torch.no_grad()
def predict(model, pil_img, device):
    x = TRANSFORM(pil_img).unsqueeze(0).to(device)
    t0 = time.perf_counter()
    probs = model(x).softmax(dim=1)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    conf, idx = probs.max(dim=0)
    return CLASS_NAMES[idx.item()], conf.item(), elapsed_ms


def run_image(model, image_path, device):
    pil_img = Image.open(image_path).convert("RGB")
    cropper = get_face_cropper(device)
    if cropper is not None:
        face = cropper(pil_img)
        if face is not None:
            pil_img = face
        else:
            print("[안내] 얼굴 못 찾음 -> 이미지 전체 사용")
    label, conf, ms = predict(model, pil_img, device)
    print(f"예측: {label} (confidence {conf:.3f}, 추론 {ms:.1f}ms)")


def run_webcam(model, device):
    import cv2

    cropper = get_face_cropper(device)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없음")
    print("q 키로 종료")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cropper is not None:
            face = cropper(pil_img)
            if face is not None:
                pil_img = face
        label, conf, ms = predict(model, pil_img, device)
        text = f"{label} {conf:.2f} ({ms:.0f}ms)"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2)
        cv2.imshow("EffEmoteNet demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="EffEmoteNet 표정 인식 데모")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--webcam", action="store_true", help="웹캠 실시간 데모")
    group.add_argument("--image", type=str, help="이미지 파일 1장 분류")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                        help=f"가중치 경로 (기본: {DEFAULT_WEIGHTS})")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model = load_model(args.weights, device=device)

    if args.webcam:
        run_webcam(model, device)
    else:
        run_image(model, args.image, device)


if __name__ == "__main__":
    main()
