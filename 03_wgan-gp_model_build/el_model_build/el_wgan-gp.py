import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMRegressor

# 1. 读取数据
data_path = r'C:\Users\Administrator\Desktop\清除数据的数据库\alloy_augmented_WGAN_GP_Ductility.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'Ductility'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
numeric_features = [col for col in features_columns]
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), columns=numeric_features)

scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_features)

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

# 7. 预测与评估
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("LightGBM Ductility 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 8. 可视化预测结果
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Ductility')
plt.ylabel('Predicted Ductility')
plt.title(f'LightGBM Ductility: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

# 9. 保存模型到桌面
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
model_path = os.path.join(desktop_path, "LightGBM_Ductility.pkl")
joblib.dump(model, model_path)
print(f"LightGBM Ductility 最佳模型已保存至: {model_path}")

# 10. 保存训练集和测试集实际值与预测值到桌面
train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_lgbm_ductility.xlsx'), index=False)

test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
test_results_df.to_excel(os.path.join(desktop_path, 'test_actual_vs_predicted_lgbm_ductility.xlsx'), index=False)

print('训练集和测试集预测结果已保存到桌面。')
# 训练完模型、预测完后
y_pred_train = model.predict(X_train)

print(f"训练集 R² Score: {r2_score(y_train, y_pred_train):.4f}")



#GBDT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# 1. 读取数据
data_path = r'C:\Users\Administrator\Desktop\清除数据的数据库\alloy_augmented_WGAN_GP_Ductility.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'Ductility'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
numeric_features = [col for col in features_columns]
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), columns=numeric_features)

scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_features)

X_processed = X_numeric_scaled

# 4. 划分数据集（80% 训练，20% 测试）
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

# 7. 预测与评估
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("GBDT Ductility 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 8. 可视化预测结果
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Ductility')
plt.ylabel('Predicted Ductility')
plt.title(f'GBDT Ductility: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

# 9. 保存模型到桌面
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
model_path = os.path.join(desktop_path, "GBDT_Ductility.pkl")
joblib.dump(model, model_path)
print(f"GBDT Ductility 最佳模型已保存至: {model_path}")

# 10. 保存训练集和测试集实际值与预测值到桌面
train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_gbdt_ductility.xlsx'), index=False)

test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
test_results_df.to_excel(os.path.join(desktop_path, 'test_actual_vs_predicted_gbdt_ductility.xlsx'), index=False)

print('训练集和测试集预测结果已保存到桌面。')
# 计算训练集 R²
r2_train = r2_score(y_train, y_pred_train)

# 输出训练集 R²
print(f"训练集 R² Score: {r2_train:.4f}")



#XGBoost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

# 1. 读取数据
data_path = r'C:\Users\Administrator\Desktop\清除数据的数据库\alloy_augmented_WGAN_GP_Ductility.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'Ductility'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
numeric_features = [col for col in features_columns]
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), columns=numeric_features)

scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_features)

X_processed = X_numeric_scaled

# 4. 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 定义 XGBoost 回归模型
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# 6. 训练模型
model.fit(X_train, y_train)

# 7. 预测与评估
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("XGBoost Ductility 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 8. 可视化预测结果
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Ductility')
plt.ylabel('Predicted Ductility')
plt.title(f'XGBoost Ductility: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

# 9. 保存模型到桌面
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
model_path = os.path.join(desktop_path, "XGBoost_Ductility.pkl")
joblib.dump(model, model_path)
print(f"XGBoost Ductility 最佳模型已保存至: {model_path}")

# 10. 保存训练集和测试集实际值与预测值到桌面
train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_xgb_ductility.xlsx'), index=False)

test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
test_results_df.to_excel(os.path.join(desktop_path, 'test_actual_vs_predicted_xgb_ductility.xlsx'), index=False)

print('训练集和测试集预测结果已保存到桌面。')
# 计算训练集 R²
r2_train = r2_score(y_train, y_pred_train)

# 输出训练集 R²
print(f"训练集 R² Score: {r2_train:.4f}")



#Stacking
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 1. 读取数据
data_path = r'C:\Users\Administrator\Desktop\清除数据的数据库\alloy_augmented_WGAN_GP_Ductility.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'Ductility'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
numeric_features = [col for col in features_columns]
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), columns=numeric_features)

scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_features)

X_processed = X_numeric_scaled

# 4. 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 定义基学习器和融合器
estimators = [
    ('lgbm', LGBMRegressor(
        n_estimators=300, max_depth=7, learning_rate=0.3,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )),
    ('xgb', XGBRegressor(
        n_estimators=300, max_depth=7, learning_rate=0.3,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )),
    ('gbdt', GradientBoostingRegressor(
        n_estimators=300, max_depth=7, learning_rate=0.3,
        subsample=0.8, random_state=42
    ))
]
stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    n_jobs=-1,
    passthrough=False
)

# 6. 训练 Stacking 集成模型
stacking.fit(X_train, y_train)

# 7. 训练集和测试集预测
y_pred_train = stacking.predict(X_train)
y_pred_test = stacking.predict(X_test)

# 8. 模型性能评估
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("Stacking 集成模型 Ductility 性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 9. 可视化预测结果
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Ductility')
plt.ylabel('Predicted Ductility')
plt.title(f'Stacking Regression (Ductility): Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

# 10. 保存模型到桌面
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
model_path = os.path.join(desktop_path, "Stacking_Ductility.pkl")
joblib.dump(stacking, model_path)
print(f"Stacking Ductility 最佳模型已保存至: {model_path}")

# 11. 保存训练集和测试集实际值与预测值到桌面
train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_stacking_ductility.xlsx'), index=False)

test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
test_results_df.to_excel(os.path.join(desktop_path, 'test_actual_vs_predicted_stacking_ductility.xlsx'), index=False)

print('训练集和测试集预测结果已保存到桌面。')
# 计算训练集 R²
r2_train = r2_score(y_train, y_pred_train)

# 输出训练集 R²
print(f"训练集 R² Score: {r2_train:.4f}")



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

# 1. 读取数据
data_path = r'C:\Users\Administrator\Desktop\清除数据的数据库\alloy_augmented_WGAN_GP_Ductility.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'Ductility'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
numeric_features = [col for col in features_columns]
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), columns=numeric_features)

scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_features)

X_processed = X_numeric_scaled

# 4. 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 定义 CatBoost 回归模型
model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.25,
    depth=6,
    random_seed=42,
    loss_function='RMSE',
    verbose=100,
    early_stopping_rounds=50
)

# 6. 训练模型
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# 7. 预测与评估
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("CatBoost Ductility 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 8. 可视化预测结果
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Ductility')
plt.ylabel('Predicted Ductility')
plt.title(f'CatBoost Ductility: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

# 9. 保存模型到桌面
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
model_path = os.path.join(desktop_path, "CatBoost_Ductility.pkl")
joblib.dump(model, model_path)
print(f"CatBoost Ductility 最佳模型已保存至: {model_path}")

# 10. 保存训练集和测试集实际值与预测值到桌面
train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_catboost_ductility.xlsx'), index=False)

test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
test_results_df.to_excel(os.path.join(desktop_path, 'test_actual_vs_predicted_catboost_ductility.xlsx'), index=False)

print('训练集和测试集预测结果已保存到桌面。')
# 计算训练集 R²
r2_train = r2_score(y_train, y_pred_train)

# 输出训练集 R²
print(f"训练集 R² Score: {r2_train:.4f}")