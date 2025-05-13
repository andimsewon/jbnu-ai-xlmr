# 🧠 JBNU AI Competition 2025 - XLM-RoBERTa Text Classifier

이 프로젝트는 2025 전북대학교 AI 경진대회 과제인
\*\*"주어진 텍스트가 사람(human)이 작성한 것인지, 생성형 AI가 작성한 것인지 분류하는 문제"\*\*를 해결하기 위한 텍스트 분류기입니다.

---

## 📁 프로젝트 구조

```
jbnu-ai-xlmr/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── models/
│   └── best_model.pt
├── output/
│   └── submission.csv
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
```

---

## ⚙️ 환경 설정

### 🐍 Python 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 📦 데이터 파일 위치

`data/` 폴더에 아래 세 파일을 넣어주세요:

* `train.csv` : 학습용 (id, text, label)
* `test.csv` : 테스트용 (id, text)
* `sample_submission.csv` : 제출 양식 예시

---

## 🚀 실행 방법

### 🔧 모델 학습

```bash
cd src
python train.py
```

* `data/train.csv`를 사용하여 XLM-RoBERTa 모델 학습
* 가장 성능이 좋은 모델은 `models/best_model.pt`에 저장됨

### 📤 예측 및 제출 파일 생성

```bash
python predict.py
```

* `data/test.csv`에 대한 예측 결과를 `output/submission.csv`로 저장

### 💻 Colab 환경에서 실행할 경우

```python
!pip install -r requirements.txt
!python src/train.py
!python src/predict.py
```

* Google Drive와 연동하여 `data/`, `models/`, `output/` 디렉토리를 마운트한 뒤 실행하세요.

---

## 🧪 하이퍼파라미터 튜닝

`src/train.py`에서 다음과 같은 하이퍼파라미터를 자유롭게 수정하여 성능을 개선할 수 있습니다:

* `epochs`: 학습 반복 횟수 (기본값: 3)
* `batch_size`: 배치 크기 (기본값: 16)
* `learning_rate`: 학습률 (기본값: 2e-5)
* `max_len`: 토큰 최대 길이 (기본값: 128)

```python
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
```

추가로:

* Stratified K-Fold Cross Validation 도입 가능
* F1-score 기준의 모델 저장 로직 추가 권장
* 데이터 전처리/증강을 통한 성능 향상 여지 있음

---

## 📊 제출 파일 형식

* 헤더 포함 CSV
* 총 6,514개 행
* `id`, `label` 두 열 포함
* 예시:

```csv
id,label
1,0
2,1
3,0
...
```

---

## 🧠 모델 정보

* **모델**: `xlm-roberta-base` (다국어 BERT 기반)
* **문제 유형**: 이진 분류 (0 = 사람 작성, 1 = AI 작성)
* **평가지표**: F1-Score (macro)

---

## 👨‍💻 개발자

* GitHub: [andimsewon](https://github.com/andimsewon)
* Repository: [jbnu-ai-xlmr](https://github.com/andimsewon/jbnu-ai-xlmr)

---

🔥 전북대 AI 경진대회, 대상 가자! 🔥
