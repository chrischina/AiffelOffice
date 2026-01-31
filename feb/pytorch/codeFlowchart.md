코드 실행 순서 상세 설명
🔄 실행 흐름 다이어그램
┌─────────────────┐
│   1. 라이브러리   │ torch, pandas, nn
│      임포트      │ optim, DataLoader
└────────┬────────┘
         ↓
┌─────────────────┐
│   2. CustomDataset│ CSV 데이터 로드 클래스 정의
│      클래스       │ __init__, __getitem__, __len__
└────────┬────────┘
         ↓
┌─────────────────┐
│   3. CustomModel │  신경망 모델 정의
│      클래스       │ Linear(3,1) + Sigmoid
└────────┬────────┘
         ↓
┌─────────────────┐
│   4. 데이터       │ binary.csv 로드 및
│   로드 및 분할    │ 8:1:1 비율로 분할
└────────┬────────┘
         ↓
┌─────────────────┐
│   5. DataLoader │ 배치별 데이터로더 생성
│      생성       │ train(64), val(4), test(4)
└────────┬────────┘
         ↓
┌─────────────────┐
│   6. 모델 설정    │ device, model, criterion
│                 │ optimizer 초기화
└────────┬────────┘
         ↓
┌─────────────────┐
│   7. 훈련 루프    │ 10,000 에포크
│                 │ 역전파 및 파라미터 업데이트
└────────┬────────┘
         ↓
┌─────────────────┐
│   8. 검증 단계    │ 훈련된 모델로 예측
│                 │ 0.5 기준 이진 분류
└─────────────────┘
📋 단계별 상세 설명
STEP 1: 라이브러리 임포트
import torch  # 핵심 PyTorch 라이브러리
import pandas as pd  # 데이터 조작을 위한 라이브러리
from torch import nn, optim  # 신경망 모듈, 최적화 알고리즘
from torch.utils.data import Dataset, DataLoader, random_split  # 데이터 처리 유틸
✅ 결과: 딥러닝에 필요한 모든 도구가 메모리에 로드됨
---
STEP 2: CustomDataset 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)  # CSV 파일 읽기
        self.x1 = df.iloc[:, 0].values  # 첫 번째 컬럼
        self.x2 = df.iloc[:, 1].values  # 두 번째 컬럼  
        self.x3 = df.iloc[:, 2].values  # 세 번째 컬럼
        self.y = df.iloc[:, 3].values   # 네 번째 컬럼 (정답)
📊 데이터 구조 예시:
binary.csv:
┌─────┬─────┬─────┬─────┐
│ x1  │ x2  │ x3  │  y  │
├─────┼─────┼─────┼─────┤
│ 1.2 │ 2.3 │ 0.5 │ 1   │
│ 0.8 │ 1.1 │ 1.2 │ 0   │
└─────┴─────┴─────┴─────┘
✅ 결과: CSV 데이터가 PyTorch가 사용할 수 있는 형태로 변환됨
---
STEP 3: CustomModel 아키텍처
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(3, 1),    # [x1,x2,x3] · [w1,w2,w3] + b = z
            nn.Sigmoid()        # σ(z) = 1/(1+e^(-z)) → 0~1 사이 값
        )
🧠 신경망 구조:
입력: [x1, x2, x3]
    ↓
Linear: [w1,w2,w3]·[x1,x2,x3] + b = z
    ↓  
Sigmoid: σ(z) = 출력 (0~1)
✅ 결과: 3→1 구조의 간단한 신경망이 메모리에 정의됨
---
STEP 4: 데이터 로드 및 분할
dataset = CustomDataset("binary.csv")  # 전체 데이터 로드
dataset_size = len(dataset)             # 전체 샘플 수
# 8:1:1 비율로 분할
train_size = int(dataset_size * 0.8)    # 80%
validation_size = int(dataset_size * 0.1) # 10%
test_size = dataset_size - train_size - validation_size # 10%
train_dataset, validation_dataset, test_dataset = random_split(dataset, 
    [train_size, validation_size, test_size], torch.manual_seed(4))
📊 데이터 분할 예시 (100개 데이터라고 가정):
전체 데이터 (100개)
    ├── 훈련 데이터 (80개)
    ├── 검증 데이터 (10개)  
    └── 테스트 데이터 (10개)
✅ 결과: 데이터가 훈련/검증/테스트 세트로 분할됨
---
STEP 5: DataLoader 생성
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=4)
test_dataloader = DataLoader(test_dataset, batch_size=4)
🔄 배치 처리:
훈련 데이터 (80개)
    ├── Batch 1: [샘플1~64]
    └── Batch 2: [샘플65~80]
검증 데이터 (10개)  
    ├── Batch 1: [샘플1~4]
    ├── Batch 2: [샘플5~8]
    └── Batch 3: [샘플9~10]
✅ 결과: 미니배치 단위로 데이터를 처리할 수 있게 준비됨
---
STEP 6: 모델 및 최적화 설정
device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU/CPU 선택
model = CustomModel().to(device)                         # 모델을 device로 이동
criterion = nn.BCELoss().to(device)                      # 손실함수 설정
optimizer = optim.SGD(model.parameters(), lr=0.0001)      # 최적화 알고리즘 설정
⚙️ 설정된 하이퍼파라미터:
- 디바이스: GPU 사용 가능 시 자동 선택
- 손실함수: Binary Cross Entropy Loss
- 최적화: Stochastic Gradient Descent
- 학습률: 0.0001 (매우 작음 → 느리지만 안정적인 학습)
✅ 결과: 훈련에 필요한 모든 컴포넌트가 준비됨
---
STEP 7: 훈련 루프 (10,000 에포크)
for epoch in range(10000):  # 전체 데이터 10,000번 반복
    cost = 0.0
    
    for x, y in train_dataloader:  # 각 배치마다
        x = x.to(device)  # GPU로 이동
        y = y.to(device)
        
        output = model(x)           # 순전파: 예측
        loss = criterion(output, y) # 손실 계산
        
        optimizer.zero_grad()       # 기울기 초기화
        loss.backward()             # 역전파: 기울기 계산  
        optimizer.step()            # 파라미터 업데이트
        
        cost += loss                # 배치 손실 누적
    
    cost = cost / len(train_dataloader)  # 평균 손실
    
    if (epoch + 1) % 1000 == 0:  # 1000 에포크마다 출력
        print(f"Epoch : {epoch+1}, Cost : {cost:.3f}")
🔄 한 에포크의 상세 과정:
1. 배치 1: [샘플1~64] → 예측 → 손실계산 → 파라미터 업데이트
2. 배치 2: [샘플65~80] → 예측 → 손실계산 → 파라미터 업데이트  
3. 에포크 종료 → 평균 손실 계산
4. 다음 에포크 시작...
✅ 결과: 모델 파라미터가 점진적으로 최적화됨
---
STEP 8: 검증 단계
with torch.no_grad():  # 기울기 계산 비활성화 (메모리 절약)
    model.eval()        # 평가 모드로 설정
    
    for x, y in validation_dataloader:
        outputs = model(x)  # 예측 수행
        
        # 0.5 이상이면 True (1), 미만이면 False (0)
        predictions = outputs >= torch.FloatTensor([0.5]).to(device)
🎯 예측 과정:
모델 출력: [0.85, 0.67, 0.62, 0.65]
          ↓
0.5 기준 비교: [True, True, True, True]  → [1, 1, 1, 1]
✅ 결과: 훈련된 모델의 성능을 검증 데이터로 확인
---
📈 전체 실행 시간 순서 요약
시작
  ↓
[1] 라이브러리 로드 (0.1초)
  ↓  
[2] Dataset 클래스 정의 (0.001초)
  ↓
[3] Model 클래스 정의 (0.001초)
  ↓
[4] CSV 데이터 로드 및 분할 (0.01초)
  ↓
[5] DataLoader 생성 (0.001초)
  ↓
[6] 모델/옵티마이저 초기화 (0.01초)
  ↓
[7] 훈련 루프 10,000 에포크 (수 분 ~ 수십 분)
  ↓
[8] 검증 단계 (0.01초)
  ↓
종료
