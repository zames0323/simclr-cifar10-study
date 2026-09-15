# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Running Image Retrieval on: {device}")

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

    # 1. SimCLR 사전학습 백본 복원
    model = resnet18(weights=None)
    model.fc = nn.Identity()  # 512차원 특징 h 직접 출력

    weights_path = 'weights/simclr_backbone.pth'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[-] Checkpoint not found at: {weights_path}")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 2. 테스트 데이터셋 로더 (특징 뱅크 구축용 2,500개 추출)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    data_dir = '../vision-resnet-study/data' if os.path.exists('../vision-resnet-study/data') else './data'
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    print("[*] Extracting feature bank embeddings...")
    feature_list = []
    raw_images = []
    labels_list = []
    total_samples = 0
    max_samples = 2500

    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Building Bank"):
            inputs_dev = inputs.to(device)
            feats = model(inputs_dev)  # [B, 512]
            feats = F.normalize(feats, dim=1)  # L2 정규화 (코사인 유사도 연산 준비)

            feature_list.append(feats.cpu())
            raw_images.append(inputs)
            labels_list.append(labels)
            total_samples += inputs.size(0)

            if total_samples >= max_samples:
                break

    feature_bank = torch.cat(feature_list, dim=0)[:max_samples]  # [N, 512]
    all_images = torch.cat(raw_images, dim=0)[:max_samples]      # [N, 3, 32, 32]
    all_labels = torch.cat(labels_list, dim=0)[:max_samples]     # [N]

    # 3. 쿼리 이미지 5종 선택 (비행기, 자동차, 고양이, 사슴, 배)
    target_classes = [0, 1, 3, 4, 8]  # airplane, automobile, cat, deer, ship
    query_indices = []
    for cls_idx in target_classes:
        match = (all_labels == cls_idx).nonzero(as_tuple=True)[0]
        if len(match) > 0:
            query_indices.append(match[0].item())

    # 4. 코사인 유사도 기반 Top-5 이웃 탐색
    # sim = Q @ Bank.T (이미 L2 정규화되어 행렬곱이 곧 코사인 유사도)
    query_feats = feature_bank[query_indices]  # [5, 512]
    sim_matrix = torch.matmul(query_feats, feature_bank.T)  # [5, N]

    topk_sims, topk_indices = torch.topk(sim_matrix, k=6, dim=1)  # 0번은 자기 자신, 1~5번이 Top-5 이웃

    # 5. 이미지 역정규화 설정
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 1, 3)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 1, 3)

    def unnorm(img_tensor):
        np_img = img_tensor.permute(1, 2, 0).cpu().numpy()
        np_img = np_img * std + mean
        return np.clip(np_img, 0.0, 1.0)

    # 6. 그리드 시각화 (5행 x 6열: Query + Top 1~5)
    os.makedirs('results', exist_ok=True)
    fig, axes = plt.subplots(len(query_indices), 6, figsize=(14, 11))

    for row_idx, q_idx in enumerate(query_indices):
        q_label = classes[all_labels[q_idx].item()]
        
        # 쿼리 이미지 출력
        axes[row_idx, 0].imshow(unnorm(all_images[q_idx]))
        axes[row_idx, 0].set_title(f"[Query]\n{q_label}", fontsize=10, fontweight='bold', color='navy')
        axes[row_idx, 0].axis('off')

        # Top-5 유사 이미지 출력
        for rank in range(1, 6):
            neighbor_idx = topk_indices[row_idx, rank].item()
            neighbor_sim = topk_sims[row_idx, rank].item()
            neighbor_label = classes[all_labels[neighbor_idx].item()]

            # 같은 클래스면 초록색, 다른 클래스면 주황색 표시
            title_color = 'forestgreen' if neighbor_label == q_label else 'darkorange'

            axes[row_idx, rank].imshow(unnorm(all_images[neighbor_idx]))
            axes[row_idx, rank].set_title(f"Rank {rank} ({neighbor_sim:.2f})\n{neighbor_label}", 
                                         fontsize=9, color=title_color)
            axes[row_idx, rank].axis('off')

    plt.suptitle("SimCLR Unsupervised Latent Space: Top-5 Image Retrieval", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    out_path = 'results/simclr_retrieval.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"[+] Retrieval visualization successfully saved to {out_path}!")

if __name__ == '__main__':
    main()