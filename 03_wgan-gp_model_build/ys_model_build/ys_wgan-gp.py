#LightGBM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMRegressor

# 1. 读取数据
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_augmented_WGAN_GP_YS10.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'YS(MPa)'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X), columns=features_columns)

scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=features_columns)

X_processed = X_numeric_scaled

# 4. 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 定义 LightGBM 回归模型
model = LGBMRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# 6. 训练模型
model.fit(X_train, y_train)

# 7. 训练集和测试集预测
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 8. 模型性能评估
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("LightGBM 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 9. 保存训练集和测试集实际值与预测值到桌面
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')

# 保存训练集
train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_lgbm.xlsx'), index=False)

# 保存测试集
test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
test_results_df.to_excel(os.path.join(desktop_path, 'test_actual_vs_predicted_lgbm.xlsx'), index=False)

print('训练集和测试集预测结果已保存到桌面。')

# 10. 可视化预测结果（测试集）
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'LightGBM Regression: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()
# 训练完模型、预测完后
y_pred_train = model.predict(X_train)

print(f"训练集 R² Score: {r2_score(y_train, y_pred_train):.4f}")



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
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_augmented_WGAN_GP_YS10.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'YS(MPa)'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X), columns=features_columns)

scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=features_columns)

X_processed = X_numeric_scaled

# 4. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 定义 GBDT 回归模型
model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

# 6. 训练模型
model.fit(X_train, y_train)

# 7. 训练集和测试集预测
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 8. 模型性能评估
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("GBDT 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 9. 保存训练集和测试集实际值与预测值到桌面
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

# 10. 可视化预测结果（测试集）
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'GBDT Regression: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()
# 计算训练集 R²
r2_train = r2_score(y_train, y_pred_train)

# 输出训练集 R²
print(f"训练集 R² Score: {r2_train:.4f}")



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
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_augmented_WGAN_GP_YS10.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'YS(MPa)'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X), columns=features_columns)

scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=features_columns)

X_processed = X_numeric_scaled

# 4. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 定义 XGBoost 回归模型
model = XGBRegressor(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.11,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# 6. 训练模型
model.fit(X_train, y_train)

# 7. 训练集和测试集预测
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 8. 模型性能评估
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("XGBoost 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 9. 保存训练集和测试集实际值与预测值到桌面
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')

# 保存训练集
train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_xgb.xlsx'), index=False)

# 保存测试集
test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
test_results_df.to_excel(os.path.join(desktop_path, 'test_actual_vs_predicted_xgb.xlsx'), index=False)

print('训练集和测试集预测结果已保存到桌面。')

# 10. 可视化预测结果（测试集）
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'XGBoost Regression: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()
# 计算训练集 R²
r2_train = r2_score(y_train, y_pred_train)

# 输出训练集 R²
print(f"训练集 R² Score: {r2_train:.4f}")



#Stacking
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# 1. 读取数据
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_augmented_WGAN_GP_YS10.xlsx'
data = pd.read_excel(data_path)

# 2. 特征与目标
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'YS(MPa)'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X), columns=features_columns)
scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=features_columns)
X_processed = X_numeric_scaled

# 4. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 定义基模型和融合模型
estimators = [
    ('lgbm', LGBMRegressor(
        n_estimators=100, max_depth=7, learning_rate=0.35,
        subsample=0.8, colsample_bytree=0.9, random_state=42, n_jobs=-1
    )),
    ('xgb', XGBRegressor(
        n_estimators=100, max_depth=7, learning_rate=0.35,
        subsample=0.8, colsample_bytree=0.9, random_state=42, n_jobs=-1
    )),
    ('gbdt', GradientBoostingRegressor(
        n_estimators=100, max_depth=7, learning_rate=0.35,
        subsample=0.9, random_state=42
    ))
]

stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    n_jobs=-1,
    passthrough=False
)

# 6. 训练Stacking模型
stacking.fit(X_train, y_train)

# 7. 预测
y_pred_train = stacking.predict(X_train)
y_pred_test = stacking.predict(X_test)

# 8. 评估
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("Stacking 集成模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 9. 保存训练集和测试集实际值与预测值到桌面
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')

# 保存训练集
train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_stacking.xlsx'), index=False)

# 保存测试集
test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
test_results_df.to_excel(os.path.join(desktop_path, 'test_actual_vs_predicted_stacking.xlsx'), index=False)

print('训练集和测试集预测结果已保存到桌面。')

# 10. 可视化预测结果（测试集）
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Stacking Regression: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()
# 计算训练集 R²
r2_train = r2_score(y_train, y_pred_train)

# 输出训练集 R²
print(f"训练集 R² Score: {r2_train:.4f}")



#CatBoost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor

# 1. 读取数据
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_augmented_WGAN_GP_YS10.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'YS(MPa)'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X), columns=features_columns)

scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=features_columns)

X_processed = X_numeric_scaled

# 4. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 定义 CatBoost 回归模型
model = CatBoostRegressor(
    iterations=300,
    depth=10,
    learning_rate=0.11,
    subsample=0.8,
    random_seed=42,
    loss_function='RMSE',
    verbose=100
)

# 6. 训练模型
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# 7. 训练集和测试集预测
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 8. 模型性能评估
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("CatBoost 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 9. 保存训练集和测试集实际值与预测值到桌面
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')

# 保存训练集
train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_catboost.xlsx'), index=False)

# 保存测试集
test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
test_results_df.to_excel(os.path.join(desktop_path, 'test_actual_vs_predicted_catboost.xlsx'), index=False)

print('训练集和测试集预测结果已保存到桌面。')

# 10. 可视化预测结果（测试集）
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'CatBoost Regression: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()
# 计算训练集 R²
r2_train = r2_score(y_train, y_pred_train)

# 输出训练集 R²
print(f"训练集 R² Score: {r2_train:.4f}")