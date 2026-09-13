# ? Mini-SimCLR: Self-Supervised Representation Learning on CIFAR-10

정답 라벨($y$) 없이 데이터 증강(Augmentation)과 대조 손실(NT-Xent Loss)만을 활용하여 시각적 특징 표현(Representation)을 학습하는 SimCLR(Simple Framework for Contrastive Learning) 파이프라인 구현, 선형 평가(Linear Probing) 및 t-SNE 잠재 공간 시각화 저장소입니다.

## ? 파이프라인 구조
- **Backbone:** ResNet18 (Random Initialization, No Pretrained Weights)
- **Data Augmentations (Views):** RandomResizedCrop + ColorJitter(p=0.8) + RandomGrayscale(p=0.2) + HorizontalFlip
- **Projection Head:** 2-Layer MLP ($512 \rightarrow 512 \rightarrow 128$)
- **Contrastive Objective:** NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss ($\tau = 0.5$)
- **Linear Evaluation (Probing):** 사전학습된 백본 가중치를 완전히 동결(Freeze)한 후, 최상단에 단일 Linear Layer(`nn.Linear(512, 10)`)만 추가하여 분류 성능 측정

---

## ? 정량적 평가 (Linear Probing Benchmark)

| 실험 구분 | 사전학습 에폭 | 백본 상태 | Linear Probing 에폭 | 검증 정확도 (Accuracy) | 비고 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Guess** | - | - | - | **10.00%** | 무작위 확률 베이스라인 |
| **SimCLR (Mini-run)** | 5 | Frozen | 5 | **36.76%** | 라벨 없는 대조 학습만으로 유의미한 분별력 확보 |

<p align="center">
  <img src="results/simclr_loss.png" width="600" alt="SimCLR Loss Curve">
</p>

### ? 학습 수렴 및 표현력 분석
- **표현 학습의 실증:** ImageNet 사전 가중치 없이 백지상태에서 5에폭(Batch 128)만 수행했음에도 무작위 확률(10%) 대비 **+26.76%p** 높은 분별력을 확보함.
- **수렴 검증:** 초기 이론적 손실값($-\ln(1/255) \approx 5.54$)에서 4.31까지 안정적으로 감쇄하며, Negative Pairs 간의 반발력과 Positive Pairs 간의 응집력이 정상 작동함을 확인.

---

## ? 정성적 평가: t-SNE 잠재 공간 2차원 투영 (Latent Space Visualization)

사전학습된 ResNet18 백본에서 추출한 512차원 특징 벡터($h$) 2,000개를 t-SNE 알고리즘을 통해 2차원 평면으로 축소 투영하여 잠재 표현의 군집 양상을 분석했습니다.

<p align="center">
  <img src="results/tsne_simclr.png" width="700" alt="SimCLR Latent Space t-SNE">
</p>

### ? 잠재 공간 분석 (Qualitative Insights)
1. **거시적 시각 범주 양극화 (Macro Clustering):**
   - **우측 영역 ($X > 0$):** `airplane`, `automobile`, `ship`, `truck` 등 금속성 구조물 및 탈것(Vehicle) 계열이 집중적으로 응집됨.
   - **좌측 영역 ($X < 0$):** `cat`, `dog`, `deer`, `bird`, `frog` 등 유기체/동물(Animal) 계열 데이터가 반대편으로 뚜렷하게 분리됨.
2. **자기지도 표현 학습의 유효성 검증:**
   - 정답 라벨($y$)을 전혀 참조하지 않았음에도, 데이터 증강과 대조 손실만으로 인공물과 생명체의 시각적 질감 및 기하학적 본질을 스스로 파악하여 공간상에 분리 배치함을 입증함.
   - 무작위 모델에서 나타나는 산란 노이즈(White Noise) 형태를 탈피하여 클래스 간 의미론적 거리 관계를 형성하기 시작함.

---

## ? 실행 방법

```bash
# 1. 의존성 패키지 설치
pip install torch torchvision matplotlib scikit-learn tqdm

# 2. SimCLR 자기지도 사전학습 (가중치 저장: weights/simclr_backbone.pth)
python3 train_simclr.py

# 3. 동결 백본 기반 선형 평가 (Linear Probing)
python3 linear_eval.py

# 4. 잠재 공간 t-SNE 2차원 시각화
python3 tsne_visualize.py