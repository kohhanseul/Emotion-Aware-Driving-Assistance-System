"""학습 체크포인트에서 추론용 슬림 가중치를 추출한다.

학습에 사용한 클래스(EffEmoteNetSmall)는 forward에서 쓰지 않는 모듈
(res_block1~3, fc1~4)까지 등록되어 있어, 저장된 체크포인트(약 520MB)에
사용하지 않는 파라미터 약 80M개가 함께 들어 있다.
이 스크립트는 실제 추론 경로에서 쓰는 파라미터(약 56M개, 약 215MB)만
남긴 가중치 파일을 만든다.

사용법:
    python scripts/export_inference_weights.py <원본.pt> <출력.pt>
예:
    python scripts/export_inference_weights.py effemote.pt models/effemotenet_infer.pt
"""

import sys

import torch

# 추론 forward에서 실제로 사용되는 최상위 모듈들
ACTIVE_PREFIXES = (
    "conv1.", "bn1.", "conv2.", "bn2.", "conv3.", "bn3.",
    "attention1.", "se.", "mbconv_block.", "head.",
)


def main(src_path, dst_path):
    state = torch.load(src_path, map_location="cpu", weights_only=False)
    slim = {k: v for k, v in state.items() if k.startswith(ACTIVE_PREFIXES)}

    dropped = len(state) - len(slim)
    kept_params = sum(v.numel() for v in slim.values())
    print(f"추출: {len(slim)}개 텐서 유지 ({kept_params/1e6:.1f}M 파라미터), {dropped}개 제거")

    torch.save(slim, dst_path)

    # 검증: 추론 모델에 strict 로드가 되는지 확인
    sys.path.insert(0, ".")
    from models.effemotenet_infer import EffEmoteNet

    model = EffEmoteNet()
    model.load_state_dict(torch.load(dst_path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 4, 300, 300))
    print(f"검증 완료: strict 로드 OK, forward OK, 출력 shape {tuple(out.shape)}")
    print(f"저장: {dst_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
