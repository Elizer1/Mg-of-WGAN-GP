#LightGBM
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

# 1. 读取数据（请确保路径正确）
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_augmented_WGAN_GP_UTS2.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'UTS(MPa)'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
numeric_features = [col for col in features_columns if col != 'Coded']
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), columns=numeric_features)
scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_features)

# 类别特征 "Coded"，做 one-hot
X_coded = pd.get_dummies(X['Coded'].fillna(X['Coded'].mode()[0]).astype(int), prefix='Coded')

# 合并所有特征
X_processed = pd.concat([X_numeric_scaled, X_coded], axis=1)

# 4. 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 定义 LightGBM 回归模型
model = LGBMRegressor(
    n_estimators=1000,
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

print("LightGBM 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 8. 可视化预测结果（测试集）
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

# 9. 定义桌面路径
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')

# 10. 保存训练集和测试集实际值与预测值到桌面
train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_lgbm.xlsx'), index=False)

test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
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
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# 1. 读取数据
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_augmented_WGAN_GP_UTS2.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'UTS(MPa)'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
numeric_features = [col for col in features_columns if col != 'Coded']
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), columns=numeric_features)
scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_features)

# 类别特征 "Coded"，做 one-hot
X_coded = pd.get_dummies(X['Coded'].fillna(X['Coded'].mode()[0]).astype(int), prefix='Coded')

# 合并所有特征
X_processed = pd.concat([X_numeric_scaled, X_coded], axis=1)

# 4. 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 定义 GBDT 回归模型
model = GradientBoostingRegressor(
    n_estimators=1000,
    max_depth=7,
    learning_rate=0.05,
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

print("GBDT 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 8. 可视化预测结果（测试集）
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

# 9. 定义桌面路径
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')

# 10. 保存训练集和测试集实际值与预测值到桌面
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
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_augmented_WGAN_GP_UTS2.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'UTS(MPa)'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
numeric_features = [col for col in features_columns if col != 'Coded']
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), columns=numeric_features)
scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_features)

# 类别特征 "Coded"，做 one-hot
X_coded = pd.get_dummies(X['Coded'].fillna(X['Coded'].mode()[0]).astype(int), prefix='Coded')

# 合并所有特征
X_processed = pd.concat([X_numeric_scaled, X_coded], axis=1)

# 4. 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 定义 XGBoost 回归模型
model = XGBRegressor(
    n_estimators=1000,
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

print("XGBoost 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 8. 可视化预测结果（测试集）
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

# 9. 定义桌面路径
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')

# 10. 保存训练集和测试集实际值与预测值到桌面
train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_xgb.xlsx'), index=False)

test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
test_results_df.to_excel(os.path.join(desktop_path, 'test_actual_vs_predicted_xgb.xlsx'), index=False)

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
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# 1. 读取数据
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_augmented_WGAN_GP_UTS2.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'UTS(MPa)'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
numeric_features = [col for col in features_columns if col != 'Coded']
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), columns=numeric_features)
scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_features)

# 类别特征 "Coded"，做 one-hot
X_coded = pd.get_dummies(X['Coded'].fillna(X['Coded'].mode()[0]).astype(int), prefix='Coded')

# 合并所有特征
X_processed = pd.concat([X_numeric_scaled, X_coded], axis=1)

# 4. 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 定义基模型和融合器
estimators = [
    ('lgbm', LGBMRegressor(
        n_estimators=1000, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )),
    ('xgb', XGBRegressor(
        n_estimators=1000, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )),
    ('gbdt', GradientBoostingRegressor(
        n_estimators=1000, max_depth=7, learning_rate=0.05,
        subsample=0.8, random_state=42
    ))
]
stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    n_jobs=-1,
    passthrough=False
)

# 6. 训练 Stacking 模型
stacking.fit(X_train, y_train)

# 7. 训练集和测试集预测
y_pred_train = stacking.predict(X_train)
y_pred_test = stacking.predict(X_test)

# 8. 模型性能评估
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("Stacking 集成模型 UTS 性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 9. 可视化预测结果
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual UTS (MPa)')
plt.ylabel('Predicted UTS (MPa)')
plt.title(f'Stacking UTS: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

# 10. 保存模型到桌面
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
model_path = os.path.join(desktop_path, "Stacking_UTS.pkl")
joblib.dump(stacking, model_path)
print(f"Stacking UTS 最佳模型已保存至: {model_path}")

# 11. 保存训练集和测试集实际值与预测值到桌面
train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_stacking_uts.xlsx'), index=False)

test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
test_results_df.to_excel(os.path.join(desktop_path, 'test_actual_vs_predicted_stacking_uts.xlsx'), index=False)

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

# 1. 读取数据（请确保路径正确）
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_augmented_WGAN_GP_UTS2.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'UTS(MPa)'

X = data[features_columns].copy()
y = data[target_column].copy()

# 3. 预处理
# 数值特征（除 "Coded" 外）缺失值填充和标准化
numeric_features = [col for col in features_columns if col != 'Coded']
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), columns=numeric_features)
scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_features)

# 类别特征 "Coded" 缺失值填充并转为字符串
X_cat = X[['Coded']].fillna(X['Coded'].mode()[0])
X_cat['Coded'] = X_cat['Coded'].astype(str)

# 合并特征
X_processed = pd.concat([X_cat, X_numeric_scaled], axis=1)

# 4. 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 定义 CatBoostRegressor 模型
cat_features_list = ['Coded']
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=7,
    random_seed=42,
    loss_function='RMSE',
    verbose=100,
    early_stopping_rounds=50
)

# 6. 训练模型，指定类别特征
model.fit(X_train, y_train, cat_features=cat_features_list, eval_set=(X_test, y_test))

# 7. 预测与评估
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("CatBoost 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 8. 可视化预测结果（测试集）
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



# 9. 定义桌面路径
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')

# 10. 保存训练集和测试集实际值与预测值到桌面
train_results_df = pd.DataFrame({
    'Actual Train': y_train,
    'Predicted Train': y_pred_train
})
train_results_df.to_excel(os.path.join(desktop_path, 'train_actual_vs_predicted_catboost.xlsx'), index=False)

test_results_df = pd.DataFrame({
    'Actual Test': y_test,
    'Predicted Test': y_pred_test
})
test_results_df.to_excel(os.path.join(desktop_path, 'test_actual_vs_predicted_catboost.xlsx'), index=False)

print('训练集和测试集预测结果已保存到桌面。')
# 训练完模型、预测完后
y_pred_train = model.predict(X_train)

print(f"训练集 R² Score: {r2_score(y_train, y_pred_train):.4f}")