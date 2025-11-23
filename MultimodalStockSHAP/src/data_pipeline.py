import os
import pandas as pd
import torch
import torch.nn.functional as F
from polygon import RESTClient
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class PolygonDataPipeline:
    def __init__(self, cfg):
        """Hydra config 기반 초기화"""
        self.cfg = cfg
        self.ticker = cfg.data.ticker
        self.start_date = cfg.data.start_date
        self.end_date = cfg.data.end_date
        self.news_limit = cfg.data.news_limit
        self.window_size = cfg.data.window_size
        self.api_key = cfg.api_key
        
        self.client = RESTClient(self.api_key)
        
        # FinBERT 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert').to(self.device)
        self.model.eval()

    def fetch_prices(self):
        """주가 데이터 수집 (OHLCV + Change)"""
        print(f"\n[1/3] Fetching Price Data for {self.ticker}...")
        aggs = []
        try:
            req_start = (pd.to_datetime(self.start_date) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            
            for a in self.client.list_aggs(
                ticker=self.ticker,
                multiplier=1,
                timespan="day",
                from_=req_start,
                to=self.end_date,
                limit=50000
            ):
                aggs.append({
                    'Date': pd.to_datetime(a.timestamp, unit='ms').date(),
                    'Open': a.open,
                    'High': a.high,
                    'Low': a.low,
                    'Close': a.close,
                    'Volume': a.volume
                })
        except Exception as e:
            print(f"Error fetching prices: {e}")
        
        df = pd.DataFrame(aggs)
        if not df.empty:
            df = df.set_index('Date').sort_index()
            df = df.loc[pd.to_datetime(self.start_date).date() : pd.to_datetime(self.end_date).date()]
        
        return df

    def fetch_news_with_volume(self):
        """
        뉴스 데이터 수집 (개수 정보 + 샘플링)
        - News_Count: 각 날짜의 실제 뉴스 개수 (모두 셈)
        - Sentiment: 날짜당 최대 max_per_day개 샘플링하여 분석
        """
        print(f"\n[2/3] Fetching News Data for {self.ticker}...")
        
        max_per_day = 10  # 날짜당 최대 분석 개수 (조절 가능)
        news_list = []
        news_count_dict = {}  # 날짜별 총 뉴스 개수 저장
        
        import time
        
        try:
            news_iter = self.client.list_ticker_news(
                ticker=self.ticker,
                published_utc_gte=self.start_date,
                published_utc_lte=self.end_date,
                limit=100,
                sort='published_utc',
                order='asc'
            )
            
            total_collected = 0
            
            for item in news_iter:
                if total_collected >= self.news_limit:
                    break
                
                text = item.description if item.description else item.title
                if not text:
                    continue
                
                date = pd.to_datetime(item.published_utc).date()
                
                # 1. 날짜별 개수 카운팅 (모든 뉴스)
                news_count_dict[date] = news_count_dict.get(date, 0) + 1
                
                # 2. 날짜당 최대 개수 체크
                current_date_count = len([n for n in news_list if n['Date'] == date])
                
                # 3. 날짜당 max_per_day개까지만 실제 저장 (샘플링)
                if current_date_count < max_per_day:
                    news_list.append({
                        'Date': date,
                        'Text': text
                    })
                
                total_collected += 1
                
                # Rate Limit 회피
                if total_collected % 5 == 0:
                    print(f"  -> Processed {total_collected} news, {len(news_list)} sampled...")
                    time.sleep(12)
                    
        except Exception as e:
            print(f"Error fetching news: {e}")
        
        # DataFrame 생성
        df_news = pd.DataFrame(news_list)
        
        # 뉴스 개수 정보 추가
        df_count = pd.DataFrame([
            {'Date': date, 'News_Count': count} 
            for date, count in news_count_dict.items()
        ])
        
        print(f"\n  📊 Total news found: {sum(news_count_dict.values())}")
        print(f"  📝 Sampled for analysis: {len(news_list)}")
        print(f"  📅 Unique dates: {len(news_count_dict)}")
        
        return df_news, df_count


    def calculate_sentiment(self, text_list):
        """
        FinBERT 감성 분석 (-1 ~ 1 스칼라)
        Core Logic 5.1: positive=prob, negative=-prob, neutral=0
        """
        print("\n[3/3] Calculating Sentiment Scores...")
        modified_scores = []
        
        batch_size = self.cfg.data.get('batch_size', 32)
        
        for i in tqdm(range(0, len(text_list), batch_size)):
            batch_texts = text_list[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=1)
            
            for prob in probabilities:
                # ProsusAI/finbert: {0: 'positive', 1: 'negative', 2: 'neutral'}
                pos_prob = prob[0].item()
                neg_prob = prob[1].item()
                label_idx = torch.argmax(prob).item()
                
                # Modified_Score: -1 ~ 1 스칼라
                if label_idx == 0:      # Positive
                    modified_scores.append(pos_prob)
                elif label_idx == 1:    # Negative
                    modified_scores.append(-neg_prob)
                else:                   # Neutral
                    modified_scores.append(0.0)
        
        return modified_scores

    def prepare_lstm_data(self):
        """
        LSTM용 데이터 준비 (Sentiment + News_Count 포함)
        """
        # 1. 주가 데이터
        df_price = self.fetch_prices()
        
        # 2. 뉴스 데이터 (개수 정보 포함)
        df_news, df_count = self.fetch_news_with_volume()
        
        if df_news.empty:
            print("⚠️ No news found. Using sentiment=0, count=0")
            df_price_reset = df_price.reset_index()
            df_price_reset['Sentiment'] = 0.0
            df_price_reset['News_Count'] = 0
        else:
            # 3. 감성 분석 (샘플링된 뉴스만)
            sentiments = self.calculate_sentiment(df_news['Text'].tolist())
            df_news['Sentiment'] = sentiments
            
            # 4. 날짜별 감성 평균
            df_sentiment_avg = df_news.groupby('Date')['Sentiment'].mean().reset_index()
            df_sentiment_avg.columns = ['Date', 'Sentiment_Avg']
            
            # 5. 주가 + 감성 + 뉴스개수 병합
            df_price_reset = df_price.reset_index()
            df_merged = pd.merge(df_price_reset, df_sentiment_avg, on='Date', how='left')
            df_merged = pd.merge(df_merged, df_count, on='Date', how='left')
            
            # 뉴스 없는 날 처리
            df_merged['Sentiment_Avg'] = df_merged['Sentiment_Avg'].fillna(0.0)
            df_merged['News_Count'] = df_merged['News_Count'].fillna(0).astype(int)
            
            df_price_reset = df_merged
        
        # 6. Feature 선택: OHLCV + Sentiment + News_Count (7개!)
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment_Avg', 'News_Count']
        data = df_price_reset[feature_cols].values
        
        # 7. Scaling
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        
        # 8. Sliding Window
        X, y = [], []
        for i in range(len(data_scaled) - self.window_size):
            X.append(data_scaled[i : i + self.window_size])
            y.append(data_scaled[i + self.window_size, 3])  # Close Price
        
        X = np.array(X)  # (samples, window_size, 7)
        y = np.array(y)
        
        print(f"\n✅ LSTM Data Ready: X.shape={X.shape}, y.shape={y.shape}")
        print(f"   Features: {feature_cols}")
        
        return X, y, scaler, df_price_reset
