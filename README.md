# 🧠 JBNU AI Competition 2025 - XLM-RoBERTa Text Classifier

이 프로젝트는 2025 전북대학교 AI 경진대회의 과제인 **"텍스트가 인간이 작성했는지, 생성형 AI가 작성했는지 분류하는 문제"** 를 해결하기 위해 구축된 다국어 텍스트 분류기입니다.

---

## 📌 프로젝트 구조

jbnu-ai-xlmr/
├── data/
│ ├── train.csv # 학습 데이터
│ ├── test.csv # 테스트 데이터
│ └── sample_submission.csv # 제출 형식 예시
├── models/
│ └── best_model.pt # 저장된 최적 모델
├── output/
│ └── submission.csv # 예측 결과 (제출용)
├── src/
│ ├── dataset.py # 데이터셋 클래스
│ ├── model.py # XLM-RoBERTa 분류 모델 정의
│ ├── train.py # 학습 코드
│ └── predict.py # 예측 코드
├── requirements.txt # 필요 라이브러리
└── README.md

yaml
복사
편집

---

## ⚙️ 사전 준비

1. **Python 환경 설치** (권장: Python 3.8 이상)
2. **필수 라이브러리 설치**

```bash
pip install -r requirements.txt
📁 데이터 준비
아래 3개의 파일을 data/ 디렉토리에 위치시켜야 합니다:

train.csv: 학습용 데이터 (id, text, label 포함)

test.csv: 테스트 데이터 (id, text 포함)

sample_submission.csv: 제출 파일 형식 예시

파일 포맷은 대회에서 제공한 기준을 그대로 따라야 합니다.

🚀 실행 방법
1. 모델 학습
bash
복사
편집
cd src
python train.py
train.csv를 불러와 XLM-RoBERTa 모델을 학습합니다.

검증 손실이 가장 낮은 모델을 ../models/best_model.pt로 저장합니다.

2. 예측 및 제출 파일 생성
bash
복사
편집
python predict.py
test.csv의 모든 텍스트에 대해 예측을 수행하고,

../output/submission.csv 파일을 생성합니다.

출력 형식:

csv
복사
편집
id,label
1,0
2,1
...
📊 평가 지표
F1-Score (macro)
대회에서는 정확도보다 정밀도와 재현율의 조화 평균인 F1-score가 핵심입니다.

✅ 기타 참고 사항
모델은 xlm-roberta-base를 사용하여 다국어 텍스트에 강건합니다.

추후 개선 아이디어:

Stratified K-Fold로 성능 향상

F1-score 기반 모델 저장

데이터 전처리 고도화

하이퍼파라미터 튜닝

📮 제출 예시
제출 파일은 반드시 아래 형식이어야 합니다:

python-repl
복사
편집
id,label
1,0
2,1
...
총 6,514개의 행 (header 제외), label ∈ {0, 1}

🙋‍♀️ 만든 사람
GitHub: andimsewon

프로젝트 링크: github.com/andimsewon/jbnu-ai-xlmr

🧡 즐거운 모델링 되세요! F1 스코어 대박 나시길!
