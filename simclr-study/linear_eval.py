# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from tqdm import tqdm

class LinearProbingModel(nn.Module):
    def __init__(self, backbone_path, num_classes=10):
        super(LinearProbingModel, self).__init__()
        # 1. ResNet18 백본 로드 및 최종 FC 제거
        self.encoder = resnet18(weights=None)
        in_features = self.encoder.fc.in_features  # 512
        self.encoder.fc = nn.Identity()

        # 2. SimCLR로 학습된 가중치 로드
        self.encoder.load_state_dict(torch.load(backbone_path, map_location='cpu'))

        # 3. 백본 가중치 동결 (Freeze): 역전파 그라디언트 차단
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 4. 단일 선형 계층 분류기 추가 (학습 대상)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        logits = self.fc(features)
        return logits

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Evaluation Device: {device}")

    # 평가용 표준 정규화 파이프라인
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    data_dir = '../vision-resnet-study/data' if os.path.exists('../vision-resnet-study/data') else './data'
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    model = LinearProbingModel('weights/simclr_backbone.pth', num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    # 선형 계층(fc) 파라미터만 옵티마이저에 전달
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-2, weight_decay=1e-4)

    epochs = 5
    print("[*] Training Linear Classifier on Frozen Representations...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 검증 정확도 측정
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        print(f"[*] Epoch {epoch+1} - Loss: {train_loss/len(trainloader):.4f} | Test Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()