# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Visualizing Representations on: {device}")

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

    # 1. 백본 모델 복원 및 가중치 로드
    model = resnet18(weights=None)
    model.fc = nn.Identity()  # 512차원 특징 벡터 h 직접 출력

    weights_path = 'weights/simclr_backbone.pth'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[-] Checkpoint not found at: {weights_path}")
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 2. 테스트 데이터셋 로더 (2,000개 샘플 추출)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    data_dir = '../vision-resnet-study/data' if os.path.exists('../vision-resnet-study/data') else './data'
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    print("[*] Extracting 512-dim feature vectors from SimCLR backbone...")
    features_list = []
    labels_list = []
    total_samples = 0
    max_samples = 2000

    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Feature Extraction"):
            inputs = inputs.to(device)
            feats = model(inputs)  # [Batch, 512]
            
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
            total_samples += inputs.size(0)

            if total_samples >= max_samples:
                break

    features = np.concatenate(features_list, axis=0)[:max_samples]
    labels = np.concatenate(labels_list, axis=0)[:max_samples]

    # 3. t-SNE 2차원 임베딩 축소
    print(f"[*] Running t-SNE on {features.shape[0]} samples (512D -> 2D)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(features)

    # 4. 2차원 산점도 시각화 및 저장
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('tab10')

    for idx, class_name in enumerate(classes):
        mask = (labels == idx)
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[cmap(idx)],
            label=class_name,
            alpha=0.65,
            edgecolors='none',
            s=25
        )

    plt.title('SimCLR Latent Space 2D Projection (t-SNE, 5 Epochs)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=11)
    plt.ylabel('t-SNE Dimension 2', fontsize=11)
    plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    output_path = 'results/tsne_simclr.png'
    plt.savefig(output_path, dpi=200)
    print(f"[+] t-SNE scatter plot successfully saved to {output_path}!")

if __name__ == '__main__':
    main()