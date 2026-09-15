# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. 2개의 강인한 뷰를 생성하는 SimCLR 증강 파이프라인
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

# 2. ResNet18 백본 + MLP 프로젝터 구조
class SimCLRModel(nn.Module):
    def __init__(self, out_dim=128):
        super(SimCLRModel, self).__init__()
        self.encoder = resnet18(weights=None)
        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()  # 512차원 특징 h 직접 출력

        self.projector = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

# 3. NT-Xent 대조 손실 함수
def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T) / temperature

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
    similarity_matrix.masked_fill_(mask, -9e15)

    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z_i.device),
        torch.arange(0, batch_size, device=z_i.device)
    ], dim=0)

    return F.cross_entropy(similarity_matrix, labels)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Upgraded SimCLR Training on: {device}")

    data_dir = '../vision-resnet-study/data' if os.path.exists('../vision-resnet-study/data') else './data'
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=SimCLRTransform(size=32),
        download=False
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    model = SimCLRModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 20 에폭 확장 & Cosine Annealing 스케줄러 결합
    epochs = 20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    epoch_losses = []

    print("[*] Starting 20-Epoch Representation Scaling...")
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

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(f"[*] Epoch {epoch+1}/{epochs} Complete - Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    os.makedirs('weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 20에폭으로 진화된 백본 가중치 갱신
    torch.save(model.encoder.state_dict(), 'weights/simclr_backbone.pth')
    print("[+] Upgraded backbone saved to weights/simclr_backbone.pth")

    # 수렴 그래프 저장
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', color='tab:purple')
    plt.title('SimCLR Contrastive Loss (20 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('NT-Xent Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/simclr_loss.png')
    print("[+] Loss curve updated at results/simclr_loss.png")

if __name__ == '__main__':
    main()