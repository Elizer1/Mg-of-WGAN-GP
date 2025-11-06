import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import grad
import os

# ====== 1. 读取并预处理原始数据 ======
path = r'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_database_clean_Ductility1.xlsx'
df = pd.read_excel(path)
df.fillna(0, inplace=True)
if 'Sum' in df.columns:
    df = df.drop(columns=['Sum'])

# 列名
element_cols = [c for c in df.columns if c not in ('Coded','Ductility')]

# 拆分数据
X_elems   = df[element_cols].values.astype(np.float32)
coded_raw = df[['Coded']].values.astype(np.float32)
y_raw     = df['Ductility'].values.reshape(-1,1).astype(np.float32)

# 合并特征：元素列 + Coded
X = np.hstack([X_elems, coded_raw])
y = y_raw

# 归一化
scaler_X = MinMaxScaler(); X_norm = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler(); y_norm = scaler_y.fit_transform(y)

# 构造 Dataset & DataLoader
class AlloyDataset(TensorDataset):
    def __init__(self, X, y):
        super().__init__(torch.tensor(X), torch.tensor(y))

dataset = AlloyDataset(X_norm, y_norm)
loader  = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

# ====== 2. 定义 WGAN-GP 模型 ======
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feat_dim   = X_norm.shape[1]
data_dim   = feat_dim + 1
latent_dim = 100
lambda_gp  = 10
n_critic   = 5

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, feat_dim)
        self.fc4 = nn.Linear(latent_dim, 1)
    def forward(self, z):
        h     = torch.relu(self.fc1(z))
        h     = torch.relu(self.fc2(h))
        feat  = self.fc3(h)
        duct  = self.fc4(z)
        return torch.cat([feat, duct], dim=1)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 1),
        )
    def forward(self, x):
        return self.net(x)

G     = Generator().to(device)
D     = Critic().to(device)
opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5,0.9))
opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5,0.9))

def gradient_penalty(D, real, fake):
    bs    = real.size(0)
    alpha = torch.rand(bs,1,device=device).expand_as(real)
    interp = (alpha*real + (1-alpha)*fake).requires_grad_(True)
    d_interp = D(interp)
    grads = grad(d_interp, interp,
                 grad_outputs=torch.ones_like(d_interp),
                 create_graph=True, retain_graph=True)[0]
    norm = grads.view(bs,-1).norm(2,dim=1)
    return lambda_gp * ((norm-1)**2).mean()

# ====== 3. 训练 WGAN-GP ======
epochs = 1000
for epoch in range(1, epochs+1):
    for real_x, real_y in loader:
        real_x = real_x.to(device)
        real_y = real_y.to(device)
        real   = torch.cat([real_x, real_y], dim=1)
        # Critic 多步
        for _ in range(n_critic):
            z      = torch.randn(real.size(0), latent_dim, device=device)
            fake   = G(z).detach()
            loss_D = D(fake).mean() - D(real).mean() + gradient_penalty(D, real, fake)
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()
        # Generator 一步
        z      = torch.randn(real.size(0), latent_dim, device=device)
        loss_G = -D(G(z)).mean()
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs}  D_loss={loss_D.item():.4f}  G_loss={loss_G.item():.4f}")

# ====== 4. 生成合成样本 & 后处理 ======
G.eval()
n_synth = X_norm.shape[0] * 5
with torch.no_grad():
    Z     = torch.randn(n_synth, latent_dim, device=device)
    synth = G(Z).cpu().numpy()

Xsynth = synth[:, :feat_dim]
duct   = synth[:, feat_dim]
Xsynth = scaler_X.inverse_transform(Xsynth)
duct   = scaler_y.inverse_transform(duct.reshape(-1,1)).flatten()
# 非负 & 整数化 Coded
Xsynth = np.maximum(Xsynth, 0.0)
duct   = np.maximum(duct, 0.0)
iC     = feat_dim - 1
Cs     = np.clip(np.round(Xsynth[:, iC]), 1, 6).astype(int)
Xsynth[:, iC] = Cs
# Mg 约束 & 元素总和 100%
iMg    = element_cols.index('Mg')
M      = np.clip(Xsynth[:, iMg], 65, 100)
Xsynth[:, iMg] = M
idx_oth  = [i for i in range(len(element_cols)) if i != iMg]
sum_oth  = Xsynth[:, idx_oth].sum(axis=1); sum_oth[sum_oth==0] = 1.0
scale    = (100 - M) / sum_oth
Xsynth[:, idx_oth] = (Xsynth[:, idx_oth].T * scale).T
# Ductility 截断 [0,100]%
duct = np.clip(duct, 0, 100)

# 合并 DataFrame
df_gen = pd.DataFrame(
    np.hstack([Xsynth, duct.reshape(-1,1)]),
    columns=element_cols + ['Coded','Ductility']
)

# ====== 5. 保存 ======
out_dir  = r'C:/Users/Administrator/Desktop/cleans'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'alloy_augmented_WGAN_GP_Ductility.xlsx')
df_all   = pd.concat([df, df_gen], ignore_index=True)
df_all.to_excel(out_path, index=False)
print(f"已生成并保存：{out_path}")