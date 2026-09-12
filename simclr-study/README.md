# ? Mini-SimCLR: Self-Supervised Representation Learning on CIFAR-10

정답 라벨($y$) 없이 데이터 증강(Augmentation)과 대조 손실(NT-Xent Loss)만을 활용하여 시각적 특징 표현(Representation)을 학습하는 SimCLR(Simple Framework for Contrastive Learning) 미니 파이프라인 구현 및 선형 평가(Linear Evaluation) 저장소입니다.

## ? 파이프라인 구조
- **Backbone:** ResNet18 (Random Initialization, No Pretrained Weights)
- **Data Augmentations (Views):** RandomResizedCrop + ColorJitter(p=0.8) + RandomGrayscale(p=0.2) + HorizontalFlip
- **Projection Head:** 2-Layer MLP ($512 \rightarrow 512 \rightarrow 128$)
- **Contrastive Objective:** NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss ($\tau = 0.5$)
- **Linear Evaluation (Probing):** 사전학습된 백본 가중치를 완전히 동결(Freeze)한 후, 최상단에 단일 Linear Layer(`nn.Linear(512, 10)`)만 추가하여 분류 성능 측정

---

## ? 실험 결과 (Linear Probing Benchmark)

| 실험 구분 | 사전학습 에폭 | 백본 상태 | Linear Probing 에폭 | 검증 정확도 (Accuracy) | 비고 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Guess** | - | - | - | **10.00%** | 무작위 확률 베이스라인 |
| **SimCLR (Mini-run)** | 5 | Frozen | 5 | **36.76%** | 라벨 없는 대조 학습만으로 유의미한 군집 형성 |

![SimCLR Loss Curve](results/simclr_loss.png)

### ? 핵심 분석
- **표현 학습의 실증:** ImageNet 가중치 없이 5에폭(Batch 128)만 수행했음에도 무작위 확률(10%) 대비 **+26.76%p** 높은 분별력을 확보함.
- **수렴 검증:** 초기 이론적 손실값($\approx 5.54$)에서 4.31까지 진동 없이 하강하며 Negative Pairs 간의 반발력과 Positive Pairs 간의 응집력이 정상적으로 작동함을 규명함.