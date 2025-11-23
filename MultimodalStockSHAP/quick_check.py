# quick_check.py - 최소한의 확인만
import hydra
from omegaconf import DictConfig
from src.data_pipeline import PolygonDataPipeline

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pipeline = PolygonDataPipeline(cfg)
    df_news, df_count = pipeline.fetch_news_with_volume()
    
    print(f"✅ 뉴스 수집: {len(df_news)}개")
    print(f"📅 발행일: {len(df_count)}일")
    print(f"\n첫 3개 뉴스:")
    print(df_news[['Date', 'Text']].head(3))

if __name__ == "__main__":
    main()
