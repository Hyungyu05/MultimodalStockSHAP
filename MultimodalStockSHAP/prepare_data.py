"""
데이터 수집 및 전처리
- Polygon.io에서 주가 + 뉴스 데이터 수집
- FinBERT 감성 분석
- LSTM 입력 형태로 전처리
- pkl 파일로 저장

실행: python prepare_data.py
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import pickle
import os
from src.data_pipeline import PolygonDataPipeline


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    데이터 수집 및 전처리 (한 번만 실행)
    """
    print("=" * 70)
    print("📡 Data Collection & Preprocessing")
    print("=" * 70)
    print(f"📊 Ticker: {cfg.data.ticker}")
    print(f"📅 Period: {cfg.data.start_date} ~ {cfg.data.end_date}")
    print(f"🪟 Window Size: {cfg.data.window_size}")
    print(f"📰 News Limit: {cfg.data.news_limit}")
    print("=" * 70)
    
    # API 키 확인
    if not cfg.api_key:
        raise ValueError(
            "❌ POLYGON_API_KEY not found!\n"
            "Please set: set POLYGON_API_KEY=your_key"
        )
    
    # 데이터 파이프라인 실행
    pipeline = PolygonDataPipeline(cfg)
    X, y, scaler, df_merged = pipeline.prepare_lstm_data()
    
    # 저장 파일명 생성
    output_file = f"data_{cfg.data.ticker}_{cfg.data.start_date}_{cfg.data.end_date}.pkl"
    
    print("\n" + "=" * 70)
    print("💾 Saving Data")
    print("=" * 70)
    
    # 저장
    with open(output_file, 'wb') as f:
        pickle.dump({
            'X': X,
            'y': y,
            'scaler': scaler,
            'df_merged': df_merged,
            'config': OmegaConf.to_container(cfg, resolve=True)
        }, f)
    
    print(f"\n✅ Data saved to: {output_file}")
    print(f"   📊 X shape: {X.shape}")
    print(f"   📈 y shape: {y.shape}")
    print(f"   📋 Features: {X.shape[2]}")
    print(f"   📅 Samples: {len(X)}")
    
    # 통계 정보
    print("\n" + "=" * 70)
    print("📊 Data Statistics")
    print("=" * 70)
    
    feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment_Avg', 'News_Count']
    print(f"   Features ({len(feature_names)}): {', '.join(feature_names)}")
    
    if 'Sentiment_Avg' in df_merged.columns and 'News_Count' in df_merged.columns:
        print(f"\n   📰 News Statistics:")
        print(f"      - Days with news: {(df_merged['News_Count'] > 0).sum()}")
        print(f"      - Avg sentiment: {df_merged['Sentiment_Avg'].mean():.4f}")
        print(f"      - Avg news/day: {df_merged['News_Count'].mean():.2f}")
    
    print("\n" + "=" * 70)
    print("✅ Data preparation complete!")
    print(f"   Next step: python train.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
