"""EffEmoteNet 추론 전용 모듈.

학습 코드(effeemotnet_model.py)와 달리 데이터 로딩 등 부작용 없이
import만으로 사용할 수 있는 추론 전용 정의입니다.

- 입력: 4채널 (RGB + Y-Sobel 엣지 채널), 300x300
- 출력: 5클래스 (anger, closed, happy, panic, sadness)
- 가중치: models/effemotenet_infer.pt (GitHub Releases에서 다운로드)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ["anger", "closed", "happy", "panic", "sadness"]
INPUT_SIZE = 300


class MBConvBlock(nn.Module):
    """EfficientNet의 역병목(Inverted-Bottleneck) 합성곱 블록."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.expand:
            return self.block(x)
        return x + self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 채널 어텐션."""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """채널 어텐션 + 공간 어텐션."""

    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = x * self.ca(x)
        return out * self.sa(out)


class EffEmoteNet(nn.Module):
    """최종 선정 모델 (학습 코드의 EffEmoteNetSmall과 동일 구조).

    conv stem(3층) → CBAM → SE → MBConv x8 → head.
    추론에 사용되는 파라미터 약 56M개.
    """

    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.attention1 = CBAM(256)
        self.se = SEBlock(256)

        k, e = 3, 6  # kernel size, expand ratio (EfficientNet B3 기준 depth scaling)
        self.mbconv_block = nn.Sequential(
            MBConvBlock(256, 512, k, 2, e),
            MBConvBlock(512, 512, k, 1, e),
            MBConvBlock(512, 512, k, 1, e),
            MBConvBlock(512, 512, k, 1, e),
            MBConvBlock(512, 1024, k, 2, e),
            MBConvBlock(1024, 1024, k, 1, e),
            MBConvBlock(1024, 1024, k, 1, e),
            MBConvBlock(1024, 1024, k, 1, e),
        )

        self.dropout1 = nn.Dropout(0.2)
        self.head = nn.Sequential(
            nn.Conv2d(1024, 2048, 1, 1, 0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.attention1(x)
        x = self.se(x)
        x = self.mbconv_block(x)
        return self.head(x)


def add_sobel_channel(img_tensor):
    """(3,H,W) RGB 텐서에 Y-Sobel 엣지 채널을 붙여 (4,H,W)로 만든다.

    감은 눈의 'ㅡ' 형태 수평 엣지를 강조하기 위해 Y방향 Sobel만 사용.
    """
    r, g, b = img_tensor[0:1], img_tensor[1:2], img_tensor[2:3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    sobel_y = torch.tensor(
        [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], dtype=torch.float32
    ).view(1, 1, 3, 3)
    edge = F.conv2d(gray.unsqueeze(0), sobel_y, padding=1).squeeze(0)
    edge = torch.clamp(torch.abs(edge) / 4.0, 0.0, 1.0)
    return torch.cat([img_tensor, edge], dim=0)


def load_model(weights_path, device="cpu"):
    """가중치를 로드한 eval 모드 모델을 반환한다."""
    model = EffEmoteNet()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)  # strict=True: 키가 하나라도 어긋나면 실패
    model.to(device).eval()
    return model
