import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from torchvision import transforms
from tqdm import tqdm

# ==========================================
# 1. 參數設定 (黃金比例)
# ==========================================
CSV_PATH = "isic-2024-challenge/train-metadata.csv"
IMAGE_FOLDER = "isic-2024-challenge/train-image/image"

CONFIG = {
    "model_name": "tf_efficientnetv2_b0",
    "img_size": 224,
    "batch_size": 32,      
    "epochs": 10,           # 跑 10 輪，保證收斂
    "lr": 1e-4,            
    "num_workers": 0,      
    # 關鍵策略：每一輪只跑 3000 步 (約 9.6萬張圖)，而不是跑完 40萬張
    # 這樣一輪只要 15~20 分鐘，總共約 3 小時搞定
    "steps_per_epoch": 3000, 
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==========================================
# 2. 安全檢查
# ==========================================
def check_image_loading(df, folder):
    print(f"[{time.strftime('%H:%M:%S')}] 檢查圖片讀取...")
    sample_ids = df.sample(3)['isic_id'].values
    for img_id in sample_ids:
        path = os.path.join(folder, f"{img_id}.jpg")
        try:
            Image.open(path).convert('RGB')
        except:
            print(f"❌ 錯誤：無法讀取 {path}，請檢查路徑！")
            exit()
    print("✅ 檢查通過。")

# ==========================================
# 3. Dataset
# ==========================================
class ISICDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, f"{row['isic_id']}.jpg")
        try:
            image = Image.open(path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(row['target'], dtype=torch.float)

# ==========================================
# 4. 主程式
# ==========================================
def main():
    print(f"[{time.strftime('%H:%M:%S')}] === 啟動智慧全量訓練 (Smart Full Training) ===")
    
    # 1. 讀取
    try: df = pd.read_csv(CSV_PATH, low_memory=False)
    except: print("❌ 無法讀取 CSV"); return
    
    check_image_loading(df, IMAGE_FOLDER)
    
    # 2. 切分 (使用全部資料)
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['target'], random_state=42)
    
    # 3. 設定 Weighted Sampler (核心)
    print(f"[{time.strftime('%H:%M:%S')}] 設定加權採樣器...")
    count_pos = len(train_df[train_df['target'] == 1])
    count_neg = len(train_df[train_df['target'] == 0])
    weight_pos = 1.0 / count_pos
    weight_neg = 1.0 / count_neg
    
    sample_weights = np.where(train_df['target'] == 1, weight_pos, weight_neg)
    sample_weights = torch.from_numpy(sample_weights).double()
    
    # 這裡定義「一輪」要跑多少張圖： batch_size * steps_per_epoch
    # 雖然沒跑完 40萬張，但 sampler 會從 40萬張裡面隨機撈，且保證正負樣本平衡
    epoch_len = CONFIG['batch_size'] * CONFIG['steps_per_epoch']
    sampler = WeightedRandomSampler(sample_weights, num_samples=epoch_len, replacement=True)
    
    print(f"   -> 每一輪將訓練: {epoch_len} 張圖片 (從 {len(train_df)} 張全量池中採樣)")

    # 4. DataLoader
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(ISICDataset(train_df, IMAGE_FOLDER, train_tfm), 
                              batch_size=CONFIG['batch_size'], sampler=sampler, num_workers=CONFIG['num_workers'])
    # 驗證集驗證 2048 張就夠看分數了
    val_loader = DataLoader(ISICDataset(val_df.iloc[:2048], IMAGE_FOLDER, val_tfm), 
                            batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

    # 5. 模型
    model = timm.create_model(CONFIG['model_name'], pretrained=True, num_classes=1).to(CONFIG['device'])
    opt = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    crit = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    # 6. 訓練
    train_hist, val_hist = [], []
    best_auc = 0.5
    
    for ep in range(CONFIG['epochs']):
        model.train(); ep_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{CONFIG['epochs']}")
        for x, y in pbar:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device']).unsqueeze(1)
            opt.zero_grad()
            with autocast(): loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            ep_loss += loss.item(); pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_hist.append(ep_loss/len(train_loader))
        
        model.eval(); val_loss = 0; preds, targs = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device']).unsqueeze(1)
                val_loss += crit(model(x), y).item()
                preds.extend(torch.sigmoid(model(x)).cpu().numpy()); targs.extend(y.cpu().numpy())
        
        try: auc = roc_auc_score(targs, preds)
        except: auc = 0.5
        val_hist.append(val_loss/len(val_loader))
        
        print(f"   Epoch {ep+1} -> Val Loss: {val_hist[-1]:.4f} | AUC: {auc:.4f}")
        if auc > best_auc: best_auc = auc; torch.save(model.state_dict(), "best_model.pth")

    # 7. 畫圖與報告
    print(f"[{time.strftime('%H:%M:%S')}] 生成最終報告...")
    preds_bin = [1 if p>0.5 else 0 for p in preds]
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1); plt.plot(train_hist, label='Train'); plt.plot(val_hist, label='Val'); plt.legend(); plt.title('Loss')
    plt.subplot(1,3,2); fpr,tpr,_=roc_curve(targs, preds); plt.plot(fpr,tpr,label=f'AUC={auc:.4f}'); plt.plot([0,1],[0,1],'--'); plt.legend(); plt.title('ROC')
    plt.subplot(1,3,3); cm=confusion_matrix(targs, preds_bin); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues'); plt.title('Confusion Matrix')
    plt.tight_layout(); plt.savefig('final_report_smart.png')
    
    print("\n" + "="*40)
    print(f"AUC:       {auc:.4f}")
    print(f"Accuracy:  {accuracy_score(targs, preds_bin):.4f}")
    print(classification_report(targs, preds_bin, target_names=['Benign', 'Malignant'], zero_division=0))
    print("="*40 + "\n✅ 訓練完成！")

if __name__ == '__main__': main()