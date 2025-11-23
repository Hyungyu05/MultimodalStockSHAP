"""
모델 학습 및 평가
- 저장된 데이터 로드
- LSTM 모델 학습
- 예측 시각화
- SHAP 분석

실행: python train.py
"""

import hydra
from omegaconf import DictConfig
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from src.model import LSTMModel
from src.utils import set_seed


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Multimodal Stock Prediction + SHAP Explanation
    """
    print("=" * 70)
    print(f"🚀 {cfg.project_name}")
    print(f"📊 Ticker: {cfg.data.ticker}")
    print(f"📅 Period: {cfg.data.start_date} ~ {cfg.data.end_date}")
    print(f"🪟 Window Size: {cfg.data.window_size}")
    print("=" * 70)
    
    # Seed 고정
    set_seed(cfg.training.seed)
    
    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    # ===== 1. 데이터 로드 =====
    print("\n" + "=" * 70)
    print("📦 STEP 1: Load Data")
    print("=" * 70)
    
    data_file = f"data_{cfg.data.ticker}_{cfg.data.start_date}_{cfg.data.end_date}.pkl"
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"\n❌ Data file not found: {data_file}\n"
            f"   Please run first: python prepare_data.py"
        )
    
    print(f"\n  📂 Loading from: {data_file}")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    X = data['X']
    y = data['y']
    scaler = data['scaler']
    df_merged = data['df_merged']
    
    print(f"  ✅ Loaded successfully!")
    print(f"     X shape: {X.shape}")
    print(f"     y shape: {y.shape}")
    
    # Train/Test Split
    split_idx = int(len(X) * (1 - cfg.training.test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n  📊 Train: {X_train.shape}, Test: {X_test.shape}")
    
    # DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).to(device),
        torch.FloatTensor(y_train).to(device)
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True
    )
    
    # ===== 2. 모델 구축 =====
    print("\n" + "=" * 70)
    print("🏗️  STEP 2: Model Setup")
    print("=" * 70)
    
    model = LSTMModel(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        output_size=cfg.model.output_size,
        dropout=cfg.model.dropout
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    print(f"\n  📐 Model Architecture:")
    print(f"     Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"     Input Shape: (batch, {cfg.data.window_size}, {cfg.model.input_size})")
    print(f"     Output: Scalar (Next Day Close Price)")
    
    # ===== 3. 학습 =====
    print("\n" + "=" * 70)
    print("🎯 STEP 3: Training")
    print("=" * 70)
    
    model.train()
    train_losses = []
    
    for epoch in range(cfg.training.epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{cfg.training.epochs}], Loss: {avg_loss:.6f}")
    
    # ===== 4. 평가 및 예측 시각화 =====
    print("\n" + "=" * 70)
    print("📊 STEP 4: Evaluation & Prediction Visualization")
    print("=" * 70)
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        
        test_pred = model(X_test_tensor)
        test_loss = criterion(test_pred, y_test_tensor).item()
        
        # CPU로 이동
        y_test_np = y_test_tensor.cpu().numpy()
        test_pred_np = test_pred.cpu().numpy()
    
    # 4-1. 예측 시각화
    print("\n  📊 Generating prediction plots...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 상단: 시계열
    axes[0].plot(y_test_np, label='Actual', color='blue', linewidth=2, alpha=0.7)
    axes[0].plot(test_pred_np, label='Predicted', color='red', linewidth=2, alpha=0.7, linestyle='--')
    axes[0].set_title(f'Stock Price Prediction - {cfg.data.ticker} (Test Set)', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time Steps', fontsize=12)
    axes[0].set_ylabel('Normalized Close Price', fontsize=12)
    axes[0].legend(loc='best', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 하단: Scatter
    axes[1].scatter(y_test_np, test_pred_np, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    axes[1].plot([y_test_np.min(), y_test_np.max()], 
                 [y_test_np.min(), y_test_np.max()], 
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_title('Actual vs Predicted (Scatter)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Actual Value', fontsize=12)
    axes[1].set_ylabel('Predicted Value', fontsize=12)
    axes[1].legend(loc='best', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    correlation = np.corrcoef(y_test_np, test_pred_np)[0, 1]
    axes[1].text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                 transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    pred_output_path = f"prediction_{cfg.data.ticker}.png"
    plt.savefig(pred_output_path, dpi=300)
    print(f"  💾 Prediction plot saved: {pred_output_path}")
    
    # 4-2. 메트릭 계산
    mae = mean_absolute_error(y_test_np, test_pred_np)
    r2 = r2_score(y_test_np, test_pred_np)
    rmse = np.sqrt(test_loss)
    
    print("\n  📏 Prediction Metrics:")
    print("  " + "-" * 60)
    print(f"  MSE (Mean Squared Error)   : {test_loss:.6f}")
    print(f"  RMSE (Root MSE)            : {rmse:.6f}")
    print(f"  MAE (Mean Absolute Error)  : {mae:.6f}")
    print(f"  R² Score                   : {r2:.6f}")
    print(f"  Correlation                : {correlation:.6f}")
    print("  " + "-" * 60)
    
    # 4-3. 학습 곡선
    print("\n  📈 Generating training curve...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, color='steelblue', linewidth=2)
    plt.title(f'Training Loss Curve - {cfg.data.ticker}', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_output_path = f"training_loss_{cfg.data.ticker}.png"
    plt.savefig(loss_output_path, dpi=300)
    print(f"  💾 Training loss plot saved: {loss_output_path}")
    
    # ===== 5. SHAP 분석 =====
    print("\n" + "=" * 70)
    print("🔍 STEP 5: SHAP Explainability")
    print("=" * 70)

    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            out = self.model(x)
            if out.dim() == 1:
                out = out.unsqueeze(-1)
            return out

    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()

    background = torch.FloatTensor(X_train[:100]).to(device)
    test_samples = torch.FloatTensor(X_test[:50]).to(device)

    explainer = shap.GradientExplainer(wrapped_model, background)

    print("\n  🔄 Computing SHAP values...")
    shap_values = explainer.shap_values(test_samples)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    feature_importance = np.abs(shap_values).mean(axis=(0, 1))
    feature_importance = np.array(feature_importance).flatten()

    if X.shape[2] == 7:
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment_Avg', 'News_Count']
    else:
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment']

    print("\n  ✅ Feature Importance (SHAP):")
    print("  " + "-" * 60)
    for name, imp in zip(feature_names, feature_importance):
        imp_value = float(imp.item()) if hasattr(imp, 'item') else float(imp)
        bar = "█" * int(imp_value * 100)
        print(f"  {name:15s} | {imp_value:.4f} {bar}")
    print("  " + "-" * 60)

    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, feature_importance, color='steelblue', alpha=0.8)
    plt.title(f'Feature Importance - {cfg.data.ticker} ({cfg.data.start_date} ~ {cfg.data.end_date})', 
            fontsize=14, fontweight='bold')
    plt.ylabel('Mean |SHAP Value|', fontsize=12)
    plt.xlabel('Features', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    output_path = f"shap_{cfg.data.ticker}.png"
    plt.savefig(output_path, dpi=300)
    print(f"\n  💾 SHAP plot saved: {output_path}")


    
    # ===== 6. 모델 저장 =====
    if cfg.get('save_model', True):
        model_path = f"model_{cfg.data.ticker}_{cfg.data.start_date}_{cfg.data.end_date}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"\n  💾 Model saved: {model_path}")
    
    print("\n" + "=" * 70)
    print("✅ Training & Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
