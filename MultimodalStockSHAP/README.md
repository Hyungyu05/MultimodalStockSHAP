# MultimodalStockSHAP

**Multimodal Stock Prediction with SHAP Explainability**

정형 데이터(OHLCV)와 비정형 데이터(뉴스 감성)를 결합한 LSTM 주가 예측 + SHAP 분석

## 🎯 특징

- **멀티모달 입력**: OHLCV(5) + FinBERT Sentiment(1) = 6 features
- **Many-to-One LSTM**: 시계열 입력 → 스칼라 출력 (다음날 종가)
- **SHAP 분석**: Feature별 예측 기여도 규명
- **Hydra 설정 관리**: YAML로 모든 파라미터 관리

## 📦 스크립트 (터미널)
1. API 키 설정
export POLYGON_API_KEY="your_key" if cmd : set POLYGON_API_KEY=your_key

2. 기본 실행
python prepare_data.py
python train.py

3. 설정 변경
python train.py data.ticker=AAPL data.window_size=20
python train.py model.hidden_size=128 training.epochs=100
