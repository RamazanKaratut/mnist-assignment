import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import numpy as np

# --- 1. Veri Hazırlama ---
def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader

# --- 2. Dinamik Model Tanımı ---
class DynamicMLP(nn.Module):
    def __init__(self, activation_name='ReLU', dropout_rate=0.0):
        super().__init__()
        self.flatten = nn.Flatten()
        
        if activation_name == 'ReLU':
            self.act = nn.ReLU()
        elif activation_name == 'Tanh':
            self.act = nn.Tanh()
        elif activation_name == 'LeakyReLU':
            self.act = nn.LeakyReLU()
        else:
            raise ValueError("Bilinmeyen aktivasyon fonksiyonu")
            
        self.layers = nn.Sequential(
            nn.Linear(28*28, 512),
            self.act,
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            self.act,
            nn.Dropout(dropout_rate),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

# --- 3. Eğitim ve Test Fonksiyonları ---
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), 100 * correct / total

def test(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return total_loss / len(loader), 100 * correct / total

# --- 4. Deney Yöneticisi ---
def run_experiment(config_name, activation, dropout, weight_decay, epochs=10):
    print(f"\nÇalışıyor: {config_name} (Act: {activation}, Drop: {dropout}, L2: {weight_decay})")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DynamicMLP(activation_name=activation, dropout_rate=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    
    train_loader, test_loader = get_data()
    
    train_acc_final = 0
    test_acc_final = 0
    
    for epoch in range(epochs):
        _, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        _, test_acc = test(model, test_loader, criterion, device)
        train_acc_final = train_acc
        test_acc_final = test_acc
        
    print(f"Sonuç -> Train Acc: {train_acc_final:.2f}%, Test Acc: {test_acc_final:.2f}%")
            
    return {
        'Deney Adı': config_name,
        'Aktivasyon': activation,
        'Dropout': dropout,
        'L2': weight_decay,
        'Train Acc': train_acc_final,
        'Test Acc': test_acc_final,
        'Gap (Train-Test)': train_acc_final - test_acc_final
    }

# --- 5. Otomatik Analiz Fonksiyonu ---
def print_detailed_analysis(df):
    print("\n" + "="*60)
    print("DETAYLI KARŞILAŞTIRMA VE SONUÇ RAPORU")
    print("="*60)

    # 1. L2 Analizi
    base = df[df['Deney Adı'] == 'Baseline (No Reg)'].iloc[0]
    l2_only = df[df['Deney Adı'] == 'L2 Only (1e-4)'].iloc[0]
    print("\n1. L2 REGULARIZATION ETKİSİ:")
    print(f"- L2 Yokken Gap: {base['Gap (Train-Test)']:.2f}% (Test Acc: {base['Test Acc']:.2f}%)")
    print(f"- L2 Varken Gap: {l2_only['Gap (Train-Test)']:.2f}% (Test Acc: {l2_only['Test Acc']:.2f}%)")
    print("-> YORUM: L2 regülarizasyonu ağırlıkları kısıtlayarak ezberlemeyi azaltır. Gap değeri düşmüşse model daha iyi genelleştirme yapmıştır.")

    # 2. Aktivasyon Analizi
    relu = df[df['Deney Adı'] == 'Act: ReLU (Base)'].iloc[0]
    tanh = df[df['Deney Adı'] == 'Act: Tanh'].iloc[0]
    leaky = df[df['Deney Adı'] == 'Act: LeakyReLU'].iloc[0]
    print("\n2. AKTİVASYON FONKSİYONLARI KARŞILAŞTIRMASI:")
    print(f"- ReLU Test Başarısı: {relu['Test Acc']:.2f}%")
    print(f"- Tanh Test Başarısı: {tanh['Test Acc']:.2f}%")
    print(f"- LeakyReLU Test Başarısı: {leaky['Test Acc']:.2f}%")
    print("-> YORUM: Genellikle ReLU ve LeakyReLU, türevleri sayesinde (Vanishing Gradient'i önleyerek) Tanh'dan daha iyi sonuç verir.")

    # 3. Dropout Analizi
    drop_01 = df[df['Deney Adı'] == 'Drop: 0.1'].iloc[0]
    drop_02 = df[df['Deney Adı'] == 'Drop: 0.2'].iloc[0]
    drop_03 = df[df['Deney Adı'] == 'Drop: 0.3'].iloc[0]
    drop_05 = df[df['Deney Adı'] == 'Drop: 0.5'].iloc[0]
    print("\n3. DROPOUT ORANLARI KARŞILAŞTIRMASI:")
    print(f"- Dropout Yok (Baseline) Gap: {base['Gap (Train-Test)']:.2f}% (Test Acc: {base['Test Acc']:.2f}%)")
    print(f"- Dropout 0.1 Gap: {drop_01['Gap (Train-Test)']:.2f}% (Test Acc: {drop_01['Test Acc']:.2f}%)")
    print(f"- Dropout 0.2 Gap: {drop_02['Gap (Train-Test)']:.2f}% (Test Acc: {drop_02['Test Acc']:.2f}%)")
    print(f"- Dropout 0.3 Gap: {drop_03['Gap (Train-Test)']:.2f}% (Test Acc: {drop_03['Test Acc']:.2f}%)")
    print(f"- Dropout 0.5 Gap: {drop_05['Gap (Train-Test)']:.2f}% (Test Acc: {drop_05['Test Acc']:.2f}%)")
    print("-> YORUM: Dropout oranı arttıkça Gap (Train-Test farkı) azalır. Ancak çok yüksek olması (örn: 0.5) test başarısını düşürebilir (Underfitting).")

    # 4. En İyi Generalization
    print("\n4. EN İYİ GENEL (GENERALIZATION) PERFORMANSI:")
    best_gen = df.loc[df['Gap (Train-Test)'].idxmin()]
    best_acc = df.loc[df['Test Acc'].idxmax()]
    
    print(f"-> En düşük Train-Test Farkı (En iyi Genelleştirme): '{best_gen['Deney Adı']}' (Gap: {best_gen['Gap (Train-Test)']:.2f}%)")
    print(f"-> En Yüksek Test Başarısı: '{best_acc['Deney Adı']}' (Test Acc: {best_acc['Test Acc']:.2f}%)")
    print("\nÖZET: İyi bir model hem yüksek Test Acc oranına sahip olmalı hem de Train-Test Gap'i olabildiğince düşük olmalıdır.")

# --- 6. Ana Döngü ---
def main():
    results = []
    
    # 1. Baseline ve L2 Analizi (Dropout yok)
    results.append(run_experiment("Baseline (No Reg)", "ReLU", 0.0, 0.0))
    results.append(run_experiment("L2 Only (1e-4)", "ReLU", 0.0, 1e-4))
    
    # 2. Aktivasyon Analizi (Standart 0.2 dropout ve 1e-4 L2 ile adil yarış)
    results.append(run_experiment("Act: ReLU (Base)", "ReLU", 0.2, 1e-4))
    results.append(run_experiment("Act: Tanh", "Tanh", 0.2, 1e-4))
    results.append(run_experiment("Act: LeakyReLU", "LeakyReLU", 0.2, 1e-4))
    
    # 3. Dropout Oranları Analizi (L2 yok, sadece dropout etkisini izole etmek için)
    results.append(run_experiment("Drop: 0.1", "ReLU", 0.1, 0.0))
    results.append(run_experiment("Drop: 0.2", "ReLU", 0.2, 0.0))
    results.append(run_experiment("Drop: 0.3", "ReLU", 0.3, 0.0))
    results.append(run_experiment("Drop: 0.5", "ReLU", 0.5, 0.0))
    
    # Tablo ve Rapor
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("TÜM DENEY SONUÇLARI (TABLO)")
    print("="*60)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.sort_values(by='Test Acc', ascending=False).to_string(index=False))
    
    df.to_csv('mnist_experiments_results.csv', index=False)
    
    # Detaylı Analizi Çalıştır
    print_detailed_analysis(df)

if __name__ == "__main__":
    main()