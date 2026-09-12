# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. 단일 이미지에서 2개의 증강 뷰를 생성하는 클래스
class SimCLRTransform:
    def __init__(self, size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

# 2. ResNet18 백본 + MLP 투영 헤드 구조
class SimCLRModel(nn.Module):
    def __init__(self, out_dim=128):
        super(SimCLRModel, self).__init__()
        # 가중치 없이 뼈대만 생성 (라벨 없는 학습)
        self.encoder = resnet18(weights=None)
        in_features = self.encoder.fc.in_features  # 512
        self.encoder.fc = nn.Identity()  # FC 계층 제거하여 512차원 특징 h 추출

        # Projection Head: 512 -> 512 -> 128
        self.projector = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

# 3. NT-Xent (InfoNCE) 대조 손실 함수
def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    
    # L2 정규화
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # 2N 벡터 결합 및 코사인 유사도 행렬 계산 [2N, 2N]
    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T) / temperature
    
    # 자기 자신과의 유사도는 마스킹
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
    similarity_matrix.masked_fill_(mask, -9e15)
    
    # 양성 쌍 타깃 인덱스 (i의 짝은 i + batch_size)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z_i.device),
        torch.arange(0, batch_size, device=z_i.device)
    ], dim=0)
    
    return F.cross_entropy(similarity_matrix, labels)

def main():
    # 연산 장치 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[*] Using device: {device}")

    # 기존 데이터셋 경로 재활용 확인
    data_dir = '../vision-resnet-study/data' if os.path.exists('../vision-resnet-study/data') else './data'
    download_flag = not os.path.exists(data_dir)
    print(f"[*] Dataset location: {data_dir}")

    # 데이터 로더 (라벨 y는 학습에 전혀 쓰이지 않음)
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=SimCLRTransform(size=32),
        download=download_flag
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    model = SimCLRModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 맛보기 5 에폭 학습 루프
    epochs = 5
    epoch_losses = []

    print("[*] Starting Self-Supervised Contrastive Learning (SimCLR)...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for (view1, view2), _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()
            _, z_i = model(view1)
            _, z_j = model(view2)

            loss = nt_xent_loss(z_i, z_j, temperature=0.5)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"[*] Epoch {epoch+1} Complete - NT-Xent Contrastive Loss: {avg_loss:.4f}")

    # 결과 디렉터리 및 가중치 저장
    os.makedirs('weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 지도학습 분류기에서 재사용할 '순수 백본(Encoder)'만 저장
    torch.save(model.encoder.state_dict(), 'weights/simclr_backbone.pth')
    print("[+] Backbone encoder weights saved to weights/simclr_backbone.pth")

    # 손실 하강 곡선 저장
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', color='tab:purple')
    plt.title('SimCLR Contrastive Loss')
    plt.xlabel('Epoch')
    plt.ylabel('NT-Xent Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/simclr_loss.png')
    print("[+] Loss curve saved to results/simclr_loss.png")

if __name__ == '__main__':
    main()