"""EffEmoteNet 모델 정의.

EfficientNet의 MBConv 블록을 기반으로,
Y-Sobel 엣지 채널과 CBAM(채널·공간 어텐션)을 결합한
실시간 운전자 감정 분류 모델(5클래스).

구성:
    - MBConvBlock: 파라미터 효율적인 역병목 합성곱 블록
    - CBAM: 채널 + 공간 어텐션 모듈
    - EffEmoteNet: 전체 분류 네트워크
"""

#라이브러리 불러오기
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics.classification import MulticlassConfusionMatrix

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConvBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio#확장된 채널 수(bottleneck 안쪽 채널)
        self.expand = in_channels != out_channels#in_channels != out_channels일 때만 블록 출력에 residual을 더하지 않음
        self.block = nn.Sequential(
            # Pointwise Convolution(1x1)_채널 확장
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # Depthwise Convolution# 채널별 따로
            #groups=hidden_dim → Depthwise conv (채널 하나당 별도 필터)
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # Pointwise Convolution Linear_채널 줄이
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
   #self.expand == False이면 skip connection (입력 + 출력) 사용(in/out채널 같을 때 residual이 들어가는 구조)
    def forward(self, x):
        if self.expand:
            return self.block(x)
        else:
            return x + self.block(x)

#Squeeze-and-Excitation 채널 어텐션(채널별 중요도 학습)
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#(B, C, H, W) → (B, C, 1, 1)→ 각 채널의 글로벌 평균(공간 정보 압축 = Squeeze)
        self.fc = nn.Sequential(#두 번의 Linear로 채널 중요도 벡터 계산
            nn.Linear(in_channels, in_channels // reduction, bias=False),#reduction: 중간 차원 축소 비율
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

#3×3 Conv 두 번 + BatchNorm + ReLU + Skip Connection(Shortcut)→ 깊어져도 gradient가 잘 흐르게 해줌
class ResidualBlock(nn.Module):#ResNet 기본 블록
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        #첫 번째 Conv: stride가 2이면 다운샘플링 역할도 수행
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)#
        self.bn1 = nn.BatchNorm2d(out_ch)
        #두 번째 Conv: 항상 stride=1, 특징 추출
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:#입력과 출력의 크기나 채널 수가 다를 때
            self.shortcut = nn.Sequential(#1×1 Conv로 맞춰서 더해주는 부분
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

#평균 풀링뿐 아니라 최대 풀링도 같이 사용,두 결과를 더해서 채널 중요도 계산
#평균 + 최대 풀링 → 서로 다른 통계정보
class ChannelAttention(nn.Module):#CBAM의 채널 어텐션 부분
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #1x1 Conv 두 번 → SEBlock의 Linear와 비슷한 역할(채널 축소/확장)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    #평균/최대 기반 채널 중요도를 합산한 뒤 Sigmoid
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)#shape: (B, C, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        #kernel_size=7, padding=3 → 7×7 필터로 넓은 주변 문맥을 보되,padding 3을 줘서 입력과 같은 H×W 크기 유지
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)#평균맵 / 최대맵 두 개를 concat 할 거라서 입력채널2
        self.sigmoid = nn.Sigmoid()#0~1 사이의 가중치 맵

    def forward(self, x):
      #각 위치(픽셀)에 대해 모든 채널의 평균값 → “전체 특징의 평균적 반응”
        avg_out = torch.mean(x, dim=1, keepdim=True)#dim=1 → 채널 방향으로 평균, keepdim=True → shape를 (B, 1, H, W)로 유지
        max_out, _ = torch.max(x, dim=1, keepdim=True)#각 위치에서 가장 강하게 반응한 채널의 값만 뽑기
        x_cat = torch.cat([avg_out, max_out], dim=1)#avg,max두가지를 채널방향으로 합치기(x_cat: (B, 2, H, W))
        x_out = self.conv1(x_cat)#각 위치의 ‘중요도’를 1채널로 요약
        return self.sigmoid(x_out)

#채널 어텐션: 어떤 채널이 중요한지
#공간 어텐션: 어느 위치가 중요한지
class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = x * self.ca(x)#채널 어텐션
        out = out * self.sa(out)#공간 어텐션
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)



class EfficientNet(nn.Module):  #EfficientNet B0
    def __init__(self, num_classes=5):
        super(EfficientNet, self).__init__()


        self.conv1 = nn.Conv2d(4, 32, 3, 2, 1, bias=False) # 입력 이미지를 RGB+흑백 4채널로 했으므로 여기도 4채널, 그 중 하나는 sobel filter 고정
        self.stem = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.attention1 = CBAM(32)

        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, 3, 1, 1),
            MBConvBlock(16, 24, 3, 2, 6),
            MBConvBlock(24, 40, 5, 2, 6),
            MBConvBlock(40, 80, 3, 2, 6),
            MBConvBlock(80, 112, 5, 1, 6),
            MBConvBlock(112, 192, 5, 2, 6),
            MBConvBlock(192, 320, 3, 1, 6)
        )

        self.attention2 = CBAM(320)

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.attention1(x)
        x = self.blocks(x)
        x = self.attention2(x)
        x = self.head(x)
        return x

class EfficientNet_B1(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNet_B1, self).__init__()

        # stem: 동일
        self.conv1 = nn.Conv2d(4, 32, 3, 2, 1, bias=False)
        self.stem = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.attention1 = CBAM(32)

        # EfficientNet B1 block 반복 수 적용
        self.blocks = nn.Sequential(
    # 1. 32 → 16 (1 block)
            MBConvBlock(32, 16, 3, 1, 1),

            MBConvBlock(16, 24, 3, 2, 6),
            *[MBConvBlock(24, 24, 3, 1, 6) for _ in range(2)],

            MBConvBlock(24, 40, 5, 2, 6),
            *[MBConvBlock(40, 40, 5, 1, 6) for _ in range(2)],

            MBConvBlock(40, 80, 3, 2, 6),
            *[MBConvBlock(80, 80, 3, 1, 6) for _ in range(3)],

            MBConvBlock(80, 112, 5, 1, 6),
            *[MBConvBlock(112, 112, 5, 1, 6) for _ in range(3)],

            MBConvBlock(112, 192, 5, 2, 6),
            *[MBConvBlock(192, 192, 5, 1, 6) for _ in range(4)],

            MBConvBlock(192, 320, 3, 1, 6),
            MBConvBlock(320, 320, 3, 1, 6),
        )

        self.attention2 = CBAM(320)

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.attention1(x)
        x = self.blocks(x)
        x = self.attention2(x)
        x = self.head(x)
        return x

class EfficientNet_B2(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNet_B2, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 3, 2, 1, bias=False)
        self.stem = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.attention1 = CBAM(32)

        self.blocks = nn.Sequential(
            # 32 → 16 (2 blocks)
            MBConvBlock(32, 16, 3, 1, 1),
            MBConvBlock(16, 16, 3, 1, 1),

            # 16 → 24 (3 blocks)
            MBConvBlock(16, 24, 3, 2, 6),
            MBConvBlock(24, 24, 3, 1, 6),
            MBConvBlock(24, 24, 3, 1, 6),

            # 24 → 40 (3 blocks)
            MBConvBlock(24, 40, 5, 2, 6),
            MBConvBlock(40, 40, 5, 1, 6),
            MBConvBlock(40, 40, 5, 1, 6),

            # 40 → 80 (4 blocks)
            MBConvBlock(40, 80, 3, 2, 6),
            MBConvBlock(80, 80, 3, 1, 6),
            MBConvBlock(80, 80, 3, 1, 6),
            MBConvBlock(80, 80, 3, 1, 6),

            # 80 → 112 (5 blocks)
            MBConvBlock(80, 112, 5, 1, 6),
            MBConvBlock(112, 112, 5, 1, 6),
            MBConvBlock(112, 112, 5, 1, 6),
            MBConvBlock(112, 112, 5, 1, 6),
            MBConvBlock(112, 112, 5, 1, 6),

            # 112 → 192 (5 blocks)
            MBConvBlock(112, 192, 5, 2, 6),
            MBConvBlock(192, 192, 5, 1, 6),
            MBConvBlock(192, 192, 5, 1, 6),
            MBConvBlock(192, 192, 5, 1, 6),
            MBConvBlock(192, 192, 5, 1, 6),

            # 192 → 320 (2 blocks)
            MBConvBlock(192, 320, 3, 1, 6),
            MBConvBlock(320, 320, 3, 1, 6)
        )

        self.attention2 = CBAM(320)

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.attention1(x)
        x = self.blocks(x)
        x = self.attention2(x)
        x = self.head(x)
        return x

class EfficientNet_B3(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNet_B3, self).__init__()

        self.conv1 = nn.Conv2d(4, 40, 3, 2, 1, bias=False)
        #self.conv1 = nn.Conv2d(3, 40, 3, 2, 1, bias=False)

        self.stem = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(40),
            nn.ReLU6(inplace=True)
        )

        self.attention1 = CBAM(40)

        self.blocks = nn.Sequential(
            # 40 → 24 (2 blocks)
            MBConvBlock(40, 24, 3, 1, 1),
            MBConvBlock(24, 24, 3, 1, 1),

            # 24 → 32 (3 blocks)
            MBConvBlock(24, 32, 3, 2, 6),
            MBConvBlock(32, 32, 3, 1, 6),
            MBConvBlock(32, 32, 3, 1, 6),

            # 32 → 48 (3 blocks)
            MBConvBlock(32, 48, 5, 2, 6),
            MBConvBlock(48, 48, 5, 1, 6),
            MBConvBlock(48, 48, 5, 1, 6),

            # 48 → 96 (5 blocks)
            MBConvBlock(48, 96, 3, 2, 6),
            MBConvBlock(96, 96, 3, 1, 6),
            MBConvBlock(96, 96, 3, 1, 6),
            MBConvBlock(96, 96, 3, 1, 6),
            MBConvBlock(96, 96, 3, 1, 6),

            # 96 → 136 (5 blocks)
            MBConvBlock(96, 136, 5, 1, 6),
            MBConvBlock(136, 136, 5, 1, 6),
            MBConvBlock(136, 136, 5, 1, 6),
            MBConvBlock(136, 136, 5, 1, 6),
            MBConvBlock(136, 136, 5, 1, 6),

            # 136 → 232 (6 blocks)
            MBConvBlock(136, 232, 5, 2, 6),
            MBConvBlock(232, 232, 5, 1, 6),
            MBConvBlock(232, 232, 5, 1, 6),
            MBConvBlock(232, 232, 5, 1, 6),
            MBConvBlock(232, 232, 5, 1, 6),
            MBConvBlock(232, 232, 5, 1, 6),

            # 232 → 384 (2 blocks)
            MBConvBlock(232, 384, 3, 1, 6),
            MBConvBlock(384, 384, 3, 1, 6)
        )

        self.attention2 = CBAM(384)

        self.head = nn.Sequential(
            nn.Conv2d(384, 1536, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1536),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1536, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.attention1(x)
        x = self.blocks(x)
        x = self.attention2(x)
        x = self.head(x)
        return x

class EfficientNet_B4(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNet_B4, self).__init__()

        # 입력 4채널 그대로 유지
        self.conv1 = nn.Conv2d(4, 48, 3, 2, 1, bias=False)

        self.stem = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(48),
            nn.ReLU6(inplace=True)
        )

        self.attention1 = CBAM(48)

        self.blocks = nn.Sequential(
            # 48 → 28 (2 blocks)
            MBConvBlock(48, 28, 3, 1, 1),
            MBConvBlock(28, 28, 3, 1, 1),

            # 28 → 32 (4 blocks)
            MBConvBlock(28, 32, 3, 2, 6),
            MBConvBlock(32, 32, 3, 1, 6),
            MBConvBlock(32, 32, 3, 1, 6),
            MBConvBlock(32, 32, 3, 1, 6),

            # 32 → 56 (4 blocks)
            MBConvBlock(32, 56, 5, 2, 6),
            MBConvBlock(56, 56, 5, 1, 6),
            MBConvBlock(56, 56, 5, 1, 6),
            MBConvBlock(56, 56, 5, 1, 6),

            # 56 → 112 (6 blocks)
            MBConvBlock(56, 112, 3, 2, 6),
            MBConvBlock(112, 112, 3, 1, 6),
            MBConvBlock(112, 112, 3, 1, 6),
            MBConvBlock(112, 112, 3, 1, 6),
            MBConvBlock(112, 112, 3, 1, 6),
            MBConvBlock(112, 112, 3, 1, 6),

            # 112 → 160 (6 blocks)
            MBConvBlock(112, 160, 5, 1, 6),
            MBConvBlock(160, 160, 5, 1, 6),
            MBConvBlock(160, 160, 5, 1, 6),
            MBConvBlock(160, 160, 5, 1, 6),
            MBConvBlock(160, 160, 5, 1, 6),
            MBConvBlock(160, 160, 5, 1, 6),

            # 160 → 272 (7 blocks)
            MBConvBlock(160, 272, 5, 2, 6),
            MBConvBlock(272, 272, 5, 1, 6),
            MBConvBlock(272, 272, 5, 1, 6),
            MBConvBlock(272, 272, 5, 1, 6),
            MBConvBlock(272, 272, 5, 1, 6),
            MBConvBlock(272, 272, 5, 1, 6),
            MBConvBlock(272, 272, 5, 1, 6),

            # 272 → 448 (2 blocks)
            MBConvBlock(272, 448, 3, 1, 6),
            MBConvBlock(448, 448, 3, 1, 6)
        )

        self.attention2 = CBAM(448)

        self.head = nn.Sequential(
            nn.Conv2d(448, 1792, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1792),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1792, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.attention1(x)
        x = self.blocks(x)
        x = self.attention2(x)
        x = self.head(x)
        return x

class EfficientNet_B5(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNet_B5, self).__init__()

        self.conv1 = nn.Conv2d(4, 56, 3, 2, 1, bias=False)

        self.stem = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(56),
            nn.ReLU6(inplace=True)
        )

        self.attention1 = CBAM(56)

        self.blocks = nn.Sequential(
            # 56 → 32 (3 blocks)
            MBConvBlock(56, 32, 3, 1, 1),
            MBConvBlock(32, 32, 3, 1, 1),
            MBConvBlock(32, 32, 3, 1, 1),

            # 32 → 40 (4 blocks)
            MBConvBlock(32, 40, 3, 2, 6),
            MBConvBlock(40, 40, 3, 1, 6),
            MBConvBlock(40, 40, 3, 1, 6),
            MBConvBlock(40, 40, 3, 1, 6),

            # 40 → 64 (4 blocks)
            MBConvBlock(40, 64, 5, 2, 6),
            MBConvBlock(64, 64, 5, 1, 6),
            MBConvBlock(64, 64, 5, 1, 6),
            MBConvBlock(64, 64, 5, 1, 6),

            # 64 → 128 (7 blocks)
            MBConvBlock(64, 128, 3, 2, 6),
            MBConvBlock(128, 128, 3, 1, 6),
            MBConvBlock(128, 128, 3, 1, 6),
            MBConvBlock(128, 128, 3, 1, 6),
            MBConvBlock(128, 128, 3, 1, 6),
            MBConvBlock(128, 128, 3, 1, 6),
            MBConvBlock(128, 128, 3, 1, 6),

            # 128 → 192 (7 blocks)
            MBConvBlock(128, 192, 5, 1, 6),
            MBConvBlock(192, 192, 5, 1, 6),
            MBConvBlock(192, 192, 5, 1, 6),
            MBConvBlock(192, 192, 5, 1, 6),
            MBConvBlock(192, 192, 5, 1, 6),
            MBConvBlock(192, 192, 5, 1, 6),
            MBConvBlock(192, 192, 5, 1, 6),

            # 192 → 320 (8 blocks)
            MBConvBlock(192, 320, 5, 2, 6),
            MBConvBlock(320, 320, 5, 1, 6),
            MBConvBlock(320, 320, 5, 1, 6),
            MBConvBlock(320, 320, 5, 1, 6),
            MBConvBlock(320, 320, 5, 1, 6),
            MBConvBlock(320, 320, 5, 1, 6),
            MBConvBlock(320, 320, 5, 1, 6),
            MBConvBlock(320, 320, 5, 1, 6),

            # 320 → 512 (2 blocks)
            MBConvBlock(320, 512, 3, 1, 6),
            MBConvBlock(512, 512, 3, 1, 6)
        )

        self.attention2 = CBAM(512)

        self.head = nn.Sequential(
            nn.Conv2d(512, 2048, 1, 1, 0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.attention1(x)
        x = self.blocks(x)
        x = self.attention2(x)
        x = self.head(x)
        return x

class EffEmoteNet(nn.Module):
    def __init__(self):
        super(EffEmoteNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.attention1 = CBAM(256)

        self.se = SEBlock(256)

        #EffnetB3 기준으로 Depth Scaling
        k = 3 # kernal size
        e = 6 # expand ratio

        self.mbconv_block = nn.Sequential(

            # ==========================================
            # EfficientNet B3 Depth Scaling: 4 Layers
            # ==========================================

            MBConvBlock(in_channels=256, out_channels=512, kernel_size=k, stride=2, expand_ratio=e),

            MBConvBlock(in_channels=512, out_channels=512, kernel_size=k, stride=1, expand_ratio=e),
            MBConvBlock(in_channels=512, out_channels=512, kernel_size=k, stride=1, expand_ratio=e),
            MBConvBlock(in_channels=512, out_channels=512, kernel_size=k, stride=1, expand_ratio=e),
            MBConvBlock(in_channels=512, out_channels=1024, kernel_size=k, stride=2, expand_ratio=e),

            MBConvBlock(in_channels=1024, out_channels=1024, kernel_size=k, stride=1, expand_ratio=e),
            MBConvBlock(in_channels=1024, out_channels=1024, kernel_size=k, stride=1, expand_ratio=e),
            MBConvBlock(in_channels=1024, out_channels=1024, kernel_size=k, stride=1, expand_ratio=e),
        )

        self.res_block1 = ResidualBlock(256, 512, stride=2)
        self.res_block2 = ResidualBlock(512, 1024, stride=2)
        self.res_block3 = ResidualBlock(1024, 2048, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, 5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        #x = self.attention1(x) # cbam 추가
        x = self.se(x)

        #x = self.res_block1(x)
        #x = self.res_block2(x)
        x = self.mbconv_block(x)
        x = self.res_block3(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout2(x)
        x = self.fc4(x)
        return x

class EffEmoteNetSmall(nn.Module):
    def __init__(self):
        super(EffEmoteNetSmall, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.attention1 = CBAM(256)

        self.se = SEBlock(256)

        #EffnetB3 기준으로 Depth Scaling
        k = 3 # kernal size
        e = 6 # expand ratio

        self.mbconv_block = nn.Sequential(

            # ==========================================
            # EfficientNet B3 Depth Scaling: 4 Layers
            # ==========================================

            MBConvBlock(in_channels=256, out_channels=512, kernel_size=k, stride=2, expand_ratio=e),

            MBConvBlock(in_channels=512, out_channels=512, kernel_size=k, stride=1, expand_ratio=e),
            MBConvBlock(in_channels=512, out_channels=512, kernel_size=k, stride=1, expand_ratio=e),
            MBConvBlock(in_channels=512, out_channels=512, kernel_size=k, stride=1, expand_ratio=e),
            MBConvBlock(in_channels=512, out_channels=1024, kernel_size=k, stride=2, expand_ratio=e),

            MBConvBlock(in_channels=1024, out_channels=1024, kernel_size=k, stride=1, expand_ratio=e),
            MBConvBlock(in_channels=1024, out_channels=1024, kernel_size=k, stride=1, expand_ratio=e),
            MBConvBlock(in_channels=1024, out_channels=1024, kernel_size=k, stride=1, expand_ratio=e),
        )

        self.res_block1 = ResidualBlock(256, 512, stride=2)
        self.res_block2 = ResidualBlock(512, 1024, stride=2)
        self.res_block3 = ResidualBlock(1024, 2048, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, 5)

        self.head = nn.Sequential(
            nn.Conv2d(1024, 2048, 1, 1, 0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(2048, 5)
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
        x = self.attention1(x) # cbam 추가
        x = self.se(x)

        #x = self.res_block1(x)
        #x = self.res_block2(x)
        x = self.mbconv_block(x)
        #x = self.res_block3(x)
        x = self.head(x)

        return x

class ResEmoteNet(nn.Module):
    def __init__(self):
        super(ResEmoteNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.attention1 = CBAM(256)

        self.se = SEBlock(256)

        self.res_block1 = ResidualBlock(256, 512, stride=2)
        self.res_block2 = ResidualBlock(512, 1024, stride=2)
        self.res_block3 = ResidualBlock(1024, 2048, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, 5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.attention1(x) # cbam 추가
        x = self.se(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout2(x)
        x = self.fc4(x)
        return x

#조명 정보를 줄이고, 얼굴 모양 특징을 유지(야간운전자 데이터 대체가)
class AddGrayChannel:
    def __call__(self, img_tensor):
        # img_tensor shape: (3, H, W)

        r = img_tensor[0:1]
        g = img_tensor[1:2]
        b = img_tensor[2:3]

        # Y = 0.299 R + 0.587 G + 0.114 B
        gray = 0.299 * r + 0.587 * g + 0.114 * b  # shape: (1, H, W)
        # Concat → (4, H, W)
        img_4ch = torch.cat([img_tensor, gray], dim=0)
        return img_4ch

class AddSobelChannel:
    def __call__(self, img_tensor):
        # img_tensor shape: (3, H, W)

        r = img_tensor[0:1]
        g = img_tensor[1:2]
        b = img_tensor[2:3]

        # Y = 0.299 R + 0.587 G + 0.114 B
        gray = 0.299 * r + 0.587 * g + 0.114 * b  # shape: (1, H, W)
        sobel2_y = torch.tensor([ #감은 눈의 edge는 ㅡ 모양이기 떄문에 y
            [1., 2., 1.],
            [0., 0., 0.],
            [-1., -2., -1.]
        ], dtype=torch.float32)

        sobel2_y = sobel2_y.view(1, 1, 3, 3)
        sobeled = F.conv2d(gray.unsqueeze(0), sobel2_y, padding=1)
        sobeled = sobeled.squeeze(0)
        sobeled = torch.abs(sobeled)
        sobeled = torch.clamp(sobeled / 4.0, 0.0, 1.0)

        # Concat → (4, H, W)
        img_4ch = torch.cat([img_tensor, sobeled], dim=0)
        return img_4ch

class AddCannyChannel:
    def __call__(self, img_tensor):
        # img_tensor shape: (3, H, W)

        r = img_tensor[0:1]
        g = img_tensor[1:2]
        b = img_tensor[2:3]

        # Y = 0.299 R + 0.587 G + 0.114 B
        gray = 0.299 * r + 0.587 * g + 0.114 * b  # shape: (1, H, W)
        sobel2_y = torch.tensor([ #감은 눈의 edge는 ㅡ 모양이기 떄문에 y
            [1., 2., 1.],
            [0., 0., 0.],
            [-1., -2., -1.]
        ], dtype=torch.float32)

        sobel2_y = sobel2_y.view(1, 1, 3, 3)
        sobeled = F.conv2d(gray.unsqueeze(0), sobel2_y, padding=1)
        sobeled = sobeled.squeeze(0)
        sobeled = torch.abs(sobeled)
        sobeled = torch.clamp(sobeled / 4.0, 0.0, 1.0)

        # Concat → (4, H, W)
        img_4ch = torch.cat([img_tensor, sobeled], dim=0)
        return img_4ch

import torch
from torch.utils.data import Dataset

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root=root, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # transform을 거친 CPU 텐서 (3,H,W)와 레이블을 반환합니다.
        img, label = self.dataset[idx]

        return img, label

BASE_DIR = "/workspace/shared/cropped_dataset_nopain"

TRAIN_IMG_DIR = f"{BASE_DIR}/train"
VAL_IMG_DIR   = f"{BASE_DIR}/valid"
TEST_IMG_DIR  = f"{BASE_DIR}/test"
base_transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    #transforms.Resize((240, 240)),
    #transforms.Resize((260, 260)),
    transforms.Resize((300, 300)),
    #transforms.Resize((380, 380)),
    #transforms.Resize((460, 460)),
    transforms.ToTensor(),
    #AddGrayChannel()#그레이스케일
    AddSobelChannel()#소벨필터
])

# grey scale 채널 하나 추가해서 넣기

train_dataset = FaceDataset(root=TRAIN_IMG_DIR, transform=base_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, num_workers=8, shuffle=True)

val_dataset = FaceDataset(root=VAL_IMG_DIR, transform=base_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, num_workers=8, shuffle=True)

test_dataset = FaceDataset(root=TEST_IMG_DIR, transform=base_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=8, shuffle=True)

images, labels = next(iter(train_loader))
print(images.shape)  # (64, 4, 224, 224)

from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import matplotlib.pyplot as plt

num_classes = 5
metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

from torchsummary import summary

modeltest = ResEmoteNet().to(device)
summary(modeltest, (4, 300, 300), batch_size=1)
