#LightGBM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from lightgbm import LGBMRegressor

# 1. 读取数据
data_path = r'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_database_clean_UTS2.xlsx'
data = pd.read_excel(data_path)
data.fillna(0, inplace=True)
if 'Sum' in data.columns:
    data.drop(columns=['Sum'], inplace=True)

# 2. 特征与目标
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al',
    'Ca', 'Zr', 'Ag', 'Ho', 'Mn', 'Y', 'Gd', 'Cu',
    'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga', 'Fe',
    'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'UTS(MPa)'
X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 先划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. 训练集和测试集分别做缺失值填充+标准化（仅用训练集参数fit）
numeric_features = [col for col in features_columns if col != 'Coded']

num_imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_train_numeric = pd.DataFrame(
    num_imputer.fit_transform(X_train[numeric_features]),
    columns=numeric_features, index=X_train.index
)
X_test_numeric = pd.DataFrame(
    num_imputer.transform(X_test[numeric_features]),
    columns=numeric_features, index=X_test.index
)
X_train_numeric_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_numeric),
    columns=numeric_features, index=X_train.index
)
X_test_numeric_scaled = pd.DataFrame(
    scaler.transform(X_test_numeric),
    columns=numeric_features, index=X_test.index
)

# Coded特征
X_train_coded = X_train[['Coded']].fillna(X_train['Coded'].mode()[0]).astype(int)
X_test_coded = X_test[['Coded']].fillna(X_train['Coded'].mode()[0]).astype(int)

# 合并
X_train_processed = pd.concat([X_train_coded.reset_index(drop=True), X_train_numeric_scaled.reset_index(drop=True)], axis=1)
X_test_processed = pd.concat([X_test_coded.reset_index(drop=True), X_test_numeric_scaled.reset_index(drop=True)], axis=1)

# -------- 新增多项式特征（degree=2） -------------
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_processed)
X_test_poly = poly.transform(X_test_processed)

# 5. LightGBM建模
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
    objective='regression'
)
model.fit(X_train_poly, y_train)

y_pred_test = model.predict(X_test_poly)
y_pred_train = model.predict(X_train_poly)

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("LightGBM UTS（多项式特征）模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 6. 可视化预测结果
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_test, alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         linestyle='--', color='red', label='Perfect Fit')
plt.xlabel('Actual UTS (MPa)')
plt.ylabel('Predicted UTS (MPa)')
plt.title(f'LightGBM UTS: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

# 7. 保存训练集和测试集的实际值与预测值到桌面
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_lgbm.xlsx'), index=False)
test_results_df.to_excel(os.path.join(desktop_path, 'test_actual_vs_predicted_lgbm.xlsx'), index=False)

print('训练集和测试集预测结果已保存到桌面。')
# 计算训练集 R²
r2_train = r2_score(y_train, y_pred_train)

# 输出训练集 R²
print(f"训练集 R² Score: {r2_train:.4f}")



#GBDT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# 1. 读取数据
data_path = r'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_database_clean_UTS2.xlsx'
data = pd.read_excel(data_path)
data.fillna(0, inplace=True)
if 'Sum' in data.columns:
    data.drop(columns=['Sum'], inplace=True)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al',
    'Ca', 'Zr', 'Ag', 'Ho', 'Mn', 'Y', 'Gd', 'Cu',
    'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga', 'Fe',
    'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'UTS(MPa)'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
numeric_features = [col for col in features_columns if col != 'Coded']
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(
    num_imputer.fit_transform(X[numeric_features]),
    columns=numeric_features
)

scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(
    scaler.fit_transform(X_numeric),
    columns=numeric_features
)

# "Coded"列为float
X_cat = X[['Coded']].fillna(X['Coded'].mode()[0])
X_cat['Coded'] = X_cat['Coded'].astype(float)

X_processed = pd.concat([X_cat, X_numeric_scaled], axis=1)

# 4. 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 定义 GBDT 回归模型
model = GradientBoostingRegressor(
    n_estimators=350,
    learning_rate=0.07,
    max_depth=8,
    subsample=0.85,
    random_state=42
)

# 6. 训练模型
model.fit(X_train, y_train)

# 7. 预测与评估
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("GBDT UTS 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"训练集 R² Score: {r2_score(y_train, y_pred_train):.4f}")

# 8. 可视化预测结果
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_test, alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         linestyle='--', color='red', label='Perfect Fit')
plt.xlabel('Actual UTS (MPa)')
plt.ylabel('Predicted UTS (MPa)')
plt.title(f'GBDT UTS: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

# 9. 保存训练集和测试集的实际值与预测值到桌面
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')

train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_gbdt.xlsx'), index=False)

test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
test_results_df.to_excel(os.path.join(desktop_path, 'test_actual_vs_predicted_gbdt.xlsx'), index=False)

print('训练集和测试集预测结果已保存到桌面。')



#XGBoost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# 1. 读取数据
data_path = r'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_database_clean_UTS2.xlsx'
data = pd.read_excel(data_path)
data.fillna(0, inplace=True)
if 'Sum' in data.columns:
    data.drop(columns=['Sum'], inplace=True)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al',
    'Ca', 'Zr', 'Ag', 'Ho', 'Mn', 'Y', 'Gd', 'Cu',
    'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga', 'Fe',
    'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'UTS(MPa)'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
numeric_features = [col for col in features_columns if col != 'Coded']
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(
    num_imputer.fit_transform(X[numeric_features]),
    columns=numeric_features
)

scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(
    scaler.fit_transform(X_numeric),
    columns=numeric_features
)

# "Coded" 必须是 float 型，不要字符串，XGBoost不支持字符串
X_cat = X[['Coded']].fillna(X['Coded'].mode()[0])
X_cat['Coded'] = X_cat['Coded'].astype(float)

X_processed = pd.concat([X_cat, X_numeric_scaled], axis=1)

# 4. 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 定义 XGBoost 回归模型
model = XGBRegressor(
    n_estimators=350,
    max_depth=8,
    learning_rate=0.07,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1,
    verbosity=1
)

# 6. 训练模型
model.fit(X_train, y_train)

# 7. 预测与评估
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("XGBoost UTS 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"训练集 R² Score: {r2_score(y_train, y_pred_train):.4f}")

# 8. 可视化预测结果
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_test, alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         linestyle='--', color='red', label='Perfect Fit')
plt.xlabel('Actual UTS (MPa)')
plt.ylabel('Predicted UTS (MPa)')
plt.title(f'XGBoost UTS: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

# 9. 保存训练集和测试集的实际值与预测值到桌面
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')

# 保存训练集数据
train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_xgb.xlsx'), index=False)

# 保存测试集数据
test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
test_results_df.to_excel(os.path.join(desktop_path, 'test_actual_vs_predicted_xgb.xlsx'), index=False)

print('训练集和测试集预测结果已保存到桌面。')



#Stacking
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# 1. 读取数据
data_path = r'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_database_clean_UTS2.xlsx'
data = pd.read_excel(data_path)
data.fillna(0, inplace=True)
if 'Sum' in data.columns:
    data.drop(columns=['Sum'], inplace=True)

# 2. 特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al',
    'Ca', 'Zr', 'Ag', 'Ho', 'Mn', 'Y', 'Gd', 'Cu',
    'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga', 'Fe',
    'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'UTS(MPa)'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
numeric_features = [col for col in features_columns if col != 'Coded']
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(
    num_imputer.fit_transform(X[numeric_features]),
    columns=numeric_features
)

scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(
    scaler.fit_transform(X_numeric),
    columns=numeric_features
)

# "Coded"列为float
X_cat = X[['Coded']].fillna(X['Coded'].mode()[0])
X_cat['Coded'] = X_cat['Coded'].astype(float)

X_processed = pd.concat([X_cat, X_numeric_scaled], axis=1)

# 4. 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 堆叠模型：底层 RandomForest+XGBoost，顶层GBDT
base_estimators = [
    ('rf', RandomForestRegressor(n_estimators=180, max_depth=10, random_state=42, n_jobs=-1)),
    ('xgb', XGBRegressor(n_estimators=180, max_depth=10, random_state=42, n_jobs=-1))
]
meta_estimator = GradientBoostingRegressor(
    n_estimators=120, learning_rate=0.06, max_depth=6, random_state=42
)
model = StackingRegressor(
    estimators=base_estimators,
    final_estimator=meta_estimator,
    passthrough=True,        # 原始特征一起传给元学习器
    n_jobs=-1
)

# 6. 训练堆叠模型
model.fit(X_train, y_train)

# 7. 预测与评估
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print("Stacking (RF+XGB->GBDT) UTS 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"训练集 R² Score: {r2_score(y_train, y_pred_train):.4f}")

# 8. 可视化预测结果
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_test, alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         linestyle='--', color='red', label='Perfect Fit')
plt.xlabel('Actual UTS (MPa)')
plt.ylabel('Predicted UTS (MPa)')
plt.title(f'Stacking UTS: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

# 9. 保存训练集和测试集的实际值与预测值到桌面
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_stacking.xlsx'), index=False)
test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
test_results_df.to_excel(os.path.join(desktop_path, 'test_actual_vs_predicted_stacking.xlsx'), index=False)

print('训练集和测试集预测结果已保存到桌面。')
print("\n测试集实际值与预测值预览：")
print(test_results_df.head(10))



#CatBoost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor

# 1. 读取数据（请确保路径正确）
data_path = r'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_database_clean_UTS2.xlsx'
data = pd.read_excel(data_path)
data.fillna(0, inplace=True)
if 'Sum' in data.columns:
    data.drop(columns=['Sum'], inplace=True)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al',
    'Ca', 'Zr', 'Ag', 'Ho', 'Mn', 'Y', 'Gd', 'Cu',
    'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga', 'Fe',
    'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'UTS(MPa)'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
# 对数值特征（除 "Coded" 外）进行缺失值填充和标准化
numeric_features = [col for col in features_columns if col != 'Coded']
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(
    num_imputer.fit_transform(X[numeric_features]),
    columns=numeric_features
)

scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(
    scaler.fit_transform(X_numeric),
    columns=numeric_features
)

# 对类别特征 "Coded" 进行缺失值填充，并转换为字符串
X_cat = X[['Coded']].fillna(X['Coded'].mode()[0])
X_cat['Coded'] = X_cat['Coded'].astype(str)

# 合并处理后的特征
X_processed = pd.concat([X_cat, X_numeric_scaled], axis=1)

# 4. 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 定义 CatBoostRegressor 模型
cat_features_list = ['Coded']
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.08,
    depth=7,
    random_seed=42,
    loss_function='RMSE',
    verbose=100,
    early_stopping_rounds=50
)

# 6. 训练模型，指定类别特征
model.fit(
    X_train, y_train,
    cat_features=cat_features_list,
    eval_set=(X_test, y_test)
)

# 7. 使用训练好的模型进行预测与评估
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("CatBoost UTS 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 8. 可视化预测结果
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_test, alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         linestyle='--', color='red', label='Perfect Fit')
plt.xlabel('Actual UTS (MPa)')
plt.ylabel('Predicted UTS (MPa)')
plt.title(f'CatBoost UTS: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

# 9. 保存训练集和测试集的实际值与预测值到桌面
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')

# 保存训练集数据
train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_catboost.xlsx'), index=False)

# 保存测试集数据
test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
test_results_df.to_excel(os.path.join(desktop_path, 'test_actual_vs_predicted_catboost.xlsx'), index=False)

print('训练集和测试集预测结果已保存到桌面。')
# 训练完模型、预测完后
y_pred_train = model.predict(X_train)

print(f"训练集 R² Score: {r2_score(y_train, y_pred_train):.4f}")