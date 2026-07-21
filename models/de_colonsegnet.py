import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) 블록은 채널 간의 상호의존성을 명시적으로 모델링하여
    유용한 채널의 특징은 강조하고 불필요한 채널은 억제합니다.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 입력 특징 맵에 채널별 가중치를 곱해 스케일링을 적용합니다.
        return x * self.fc(x)

class ResidualBlock(nn.Module):
    """
    Residual 블록은 입력을 출력에 더해주는 숏컷(Shortcut) 경로를 제공하여
    경사 소실 문제를 방지하고 더 깊은 모델 학습을 돕습니다.
    여기에 SE 블록을 결합하여 특징 추출 능력을 향상시킵니다.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.se = SEBlock(out_channels)
        
        # 입력 채널과 출력 채널이 다른 경우 차원을 맞춰주기 위해 1x1 Convolution을 적용합니다.
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        out = out + self.shortcut(x)
        return self.relu(out)

class EncoderBlock(nn.Module):
    """
    ColonSegNet 인코더 블록은 피처 맵의 크기를 축소하면서 고차원 특징을 추출합니다.
    1. 첫 번째 Residual 블록 수행
    2. Stride 2인 Convolution을 통한 첫 번째 크기 축소 (1/2) 및 피처 맵 추출
    3. 두 번째 Residual 블록 수행
    4. MaxPool2d를 이용한 두 번째 크기 축소 (1/2)
    디코더의 skip connection으로 사용하기 위해 중간 특징 맵(x1, x3)을 리스트로 반환합니다.
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.res1 = ResidualBlock(in_channels, out_channels)
        self.strided_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.res2 = ResidualBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.res1(x)              # 해상도 유지
        x2 = self.strided_conv(x1)     # 해상도 1/2로 축소
        x3 = self.res2(x2)             # 해상도 축소 상태 유지
        p = self.pool(x3)              # 해상도 1/2로 추가 축소
        return p, [x1, x3]

class ASPP(nn.Module):
    """
    ASPP (Atrous Spatial Pyramid Pooling) 블록은 확장 비율(dilation rate)이 서로 다른
    다중 병렬 Atrous Conv 레이어를 통해 다양한 수용 영역(receptive field)의 피처를 결합합니다.
    작고 큰 다양한 크기의 폴립을 포착하는 데 탁월한 효과가 있습니다.
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        # 1x1 Convolution 분기
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Dilation rate 6 분기
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Dilation rate 12 분기
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Dilation rate 18 분기
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Global Average Pooling 분기 (전역 컨텍스트 보존)
        self.global_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 5개 채널 출력을 Concat한 뒤 입력 차원으로 복원하는 1x1 Conv
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        out1 = self.aspp1(x)
        out2 = self.aspp2(x)
        out3 = self.aspp3(x)
        out4 = self.aspp4(x)
        
        # 전역 평균 풀링 결과를 보간하여 원래 크기로 변환
        out_g = self.global_avg(x)
        out_g = F.interpolate(out_g, size=(h, w), mode='bilinear', align_corners=False)
        
        # 병렬 특징 결합 후 채널 축소 프로젝트 적용
        out = torch.cat([out1, out2, out3, out4, out_g], dim=1)
        return self.project(out)

class ColonSegNet(nn.Module):
    """
    단일 ColonSegNet 구조입니다. 
    두 개의 인코더 블록, ASPP Bottleneck, 그리고 스킵 연결을 이용해 특징을 
    복원하는 업샘플링 디코더로 구성됩니다.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(ColonSegNet, self).__init__()
        # 인코더부
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 256)
        
        # 바틀넥 (ASPP)
        self.aspp = ASPP(256, 256)
        
        # 디코더부 (ConvTranspose2d와 ResidualBlock을 통한 특징 복원)
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(512, 256) # 256(up) + 256(x3_256)
        
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(512, 256) # 256(up) + 256(x1_256)
        
        self.up3 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(128, 64)  # 64(up) + 64(x3_64)
        
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(128, 64)  # 64(up) + 64(x1_64)
        
        # 최종 세그멘테이션 맵 추출 (1x1 Convolution)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 인코더 특징 추출
        p1, skips1 = self.enc1(x)      # skips1 = [x1_64 (352x352), x3_64 (176x176)], p1 = (88x88)
        p2, skips2 = self.enc2(p1)     # skips2 = [x1_256 (88x88), x3_256 (44x44)], p2 = (22x22)
        
        # 바틀넥 다중 해상도 피처링
        b = self.aspp(p2)              # (22x22)
        
        # 디코더 역합성곱 및 스킵 결합
        u1 = self.up1(b)
        c1 = torch.cat([u1, skips2[1]], dim=1) # (44x44, 256 + 256)
        d1 = self.dec1(c1)
        
        u2 = self.up2(d1)
        c2 = torch.cat([u2, skips2[0]], dim=1) # (88x88, 256 + 256)
        d2 = self.dec2(c2)
        
        u3 = self.up3(d2)
        c3 = torch.cat([u3, skips1[1]], dim=1) # (176x176, 64 + 64)
        d3 = self.dec3(c3)
        
        u4 = self.up4(d3)
        c4 = torch.cat([u4, skips1[0]], dim=1) # (352x352, 64 + 64)
        d4 = self.dec4(c4)
        
        logits = self.final_conv(d4)
        return logits

class DE_ColonSegNet(nn.Module):
    """
    DE-ColonSegNet (Dual-Encoder ColonSegNet) 모델입니다.
    첫 번째 인코더 네트워크(net1)가 러프한 예측 마스크를 생성하면,
    이를 원본 입력 이미지와 concat하여 두 번째 네트워크(net2)에 제공함으로써
    세부 병변 정보 검출 정확도를 극대화합니다.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(DE_ColonSegNet, self).__init__()
        # 첫 번째 ColonSegNet (3채널 -> 1채널)
        self.net1 = ColonSegNet(in_channels=in_channels, out_channels=out_channels)
        # 두 번째 ColonSegNet (3채널 + 1채널(예측 마스크) -> 1채널)
        self.net2 = ColonSegNet(in_channels=in_channels + out_channels, out_channels=out_channels)

    def forward(self, x):
        # 1. 1단계 예측 logits 연산
        logits1 = self.net1(x)
        pred1 = torch.sigmoid(logits1)
        
        # 2. 원본 이미지와 1단계 예측 맵을 채널 축으로 결합 (4채널 입력 생성)
        x2 = torch.cat([x, pred1], dim=1)
        
        # 3. 2단계 최종 예측 logits 연산
        logits2 = self.net2(x2)
        
        # 학습 루프에서 보조 손실(Auxiliary Loss)을 적용하기 위해 두 예측을 모두 반환
        return logits2, logits1

if __name__ == "__main__":
    # 단위 성능/동작 검증용
    model = DE_ColonSegNet(in_channels=3, out_channels=1)
    test_tensor = torch.randn(1, 3, 352, 352)
    out2, out1 = model(test_tensor)
    print(f"Input shape        : {test_tensor.shape}")
    print(f"Stage 1 output shape: {out1.shape}")
    print(f"Stage 2 output shape: {out2.shape}")
    print("DE-ColonSegNet 아키텍처 모델 구성 성공!")
