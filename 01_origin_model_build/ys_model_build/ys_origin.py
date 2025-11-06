#LightGBM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

# 1. 读取数据
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_database_clean_YS1.xlsx'
data = pd.read_excel(data_path)

# 2. 去除目标的异常值
ys_mean = data['YS(MPa)'].mean()
ys_std = data['YS(MPa)'].std()
filtered_data = data[(data['YS(MPa)'] > ys_mean - 3*ys_std) & (data['YS(MPa)'] < ys_mean + 3*ys_std)]

# 3. 特征和目标
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'YS(MPa)'
X = filtered_data[features_columns].copy()
y = filtered_data[target_column].copy()

# 4. 预处理
numeric_features = [col for col in features_columns if col != 'Coded']
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), columns=numeric_features)
scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_features)
X_cat = X[['Coded']].fillna(X['Coded'].mode()[0])
X_cat['Coded'] = X_cat['Coded'].astype(int)
X_processed = pd.concat([X_cat.reset_index(drop=True), X_numeric_scaled.reset_index(drop=True)], axis=1)

# 5. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 6. 建模
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
    objective='regression'
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("LightGBM 去异常值后模型性能:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 7. 特征重要性
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features_sorted = X_train.columns[indices]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(15), importances[indices[:15]], align='center')
plt.xticks(range(15), features_sorted[:15], rotation=45)
plt.tight_layout()
plt.show()

# 8. 预测结果可视化
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'LightGBM Regression (去异常值): Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

# 9. 保存数据库（训练集和测试集预测结果）到桌面
# 计算训练集的预测结果
y_pred_train = model.predict(X_train)

# 构造保存训练集预测结果的 DataFrame
train_results_df = pd.DataFrame({
    'Actual Train': y_train.values,
    'Predicted Train': y_pred_train
})

# 构造保存测试集预测结果的 DataFrame
test_results_df = pd.DataFrame({
    'Actual Test': y_test.values,
    'Predicted Test': y_pred
})

# 定义保存文件的路径（桌面）
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
train_file = os.path.join(desktop_path, 'Train_Prediction_LightGBM_YS.csv')
test_file = os.path.join(desktop_path, 'Test_Prediction_LightGBM_YS.csv')

# 保存 CSV 文件
train_results_df.to_csv(train_file, index=False)
test_results_df.to_csv(test_file, index=False)

print("训练集预测结果已保存至:", train_file)
print("测试集预测结果已保存至:", test_file)
# 计算训练集 R²
r2_train = r2_score(y_train, y_pred_train)

# 输出训练集 R²
print(f"训练集 R² Score: {r2_train:.4f}")



#GBDT
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
from sklearn.impute import SimpleImputer

# 1. 读取数据（请确保路径正确）
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_database_clean_YS1.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量列名
features_columns = ['Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
                    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
                    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi']
target_column = 'YS(MPa)'

# 分离特征和目标变量
X = data[features_columns]
y = data[target_column]

# 3. 对 "Coded" 进行 One-Hot 编码
X = pd.get_dummies(X, columns=['Coded'], prefix='Coded')

# 4. 缺失值处理：使用均值填充
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 5. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 采用较简约的参数搜索范围（类似之前表现较好的参数设置）
param_grid = {
    'n_estimators': [300, 500, 700],
    'max_depth': [10, 20, 30],
    'learning_rate': [0.05, 0.1]
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)
print("开始简约参数网格搜索（GBDT + One-Hot 编码）...")
grid_search.fit(X_train, y_train)
print("网格搜索完成。")

# 输出最佳参数
best_params = grid_search.best_params_
print(f"最佳参数: {best_params}")

# 7. 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 8. 评估模型性能
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print("模型性能评估 (GBDT + One-Hot 编码):")
print(f"RMSE: {rmse:.2f} MPa")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")

# 9. 可视化预测结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'GBDT Regression: Actual vs Predicted\nR-squared: {r2:.4f}')
plt.legend()
plt.tight_layout()
plt.show()



#XGBoost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# 1. 读取数据（请确保路径正确）
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_database_clean_YS1.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量列名
features_columns = ['Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
                    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
                    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi']
target_column = 'YS(MPa)'

# 分离特征和目标变量
X = data[features_columns]
y = data[target_column]

# 3. 对 "Coded" 进行 One-Hot 编码
X = pd.get_dummies(X, columns=['Coded'], prefix='Coded')

# 4. 缺失值处理：使用均值填充
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 5. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 设置 XGBoost 模型参数搜索范围
param_grid = {
    'n_estimators': [300, 500, 700],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 使用 GridSearchCV 搜索最佳参数（设置 objective 为 reg:squarederror）
grid_search = GridSearchCV(
    XGBRegressor(random_state=42, objective='reg:squarederror'),
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)
print("开始网格搜索（XGBoost + One-Hot 编码）...")
grid_search.fit(X_train, y_train)
print("网格搜索完成。")

# 输出最佳参数
best_params = grid_search.best_params_
print(f"最佳参数: {best_params}")

# 7. 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 8. 评估模型性能
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print("模型性能评估 (XGBoost + One-Hot 编码):")
print(f"RMSE: {rmse:.2f} MPa")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")

# 9. 可视化预测结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'XGBoost Regression: Actual vs Predicted\nR-squared: {r2:.4f}')
plt.legend()
plt.tight_layout()
plt.show()



#Stacking
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. 读取数据（请确保路径正确）
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_database_clean_YS1.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'YS(MPa)'

X = data[features_columns]
y = data[target_column]

# 3. 构建预处理器：对 "Coded" 列采用 One-Hot 编码，对其他数值列用均值填充后再标准化
categorical_features = ['Coded']
numeric_features = [col for col in features_columns if col not in categorical_features]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 4. 定义基模型（基学习器）与元模型
estimators = [
    ('rf', RandomForestRegressor(random_state=42, n_estimators=300, max_depth=20)),
    ('xgb', XGBRegressor(random_state=42, objective='reg:squarederror', n_estimators=300, max_depth=5, learning_rate=0.1)),
    ('gb', GradientBoostingRegressor(random_state=42, n_estimators=300, max_depth=5, learning_rate=0.1))
]

final_estimator = Ridge(alpha=1.0)

stacking_reg = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=5,
    n_jobs=-1
)

# 5. 构建完整 Pipeline：预处理 -> 堆叠回归
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('stacking', stacking_reg)
])

# 6. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. 训练模型
pipeline.fit(X_train, y_train)

# 8. 预测与评估
y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print("Stacking Ensemble 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 9. 可视化预测结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red', linestyle='--', label='Perfect Fit'
)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Stacking Ensemble: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()



#CatBoost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor

# 1. 读取数据（请确保路径正确）
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_database_clean_YS1.xlsx'
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
# 对数值特征（除 "Coded" 外）进行缺失值填充和标准化
numeric_features = [col for col in features_columns if col != 'Coded']
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), columns=numeric_features)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_features)

# 对类别特征 "Coded" 进行缺失值填充，并转为字符串（CatBoost 可自动处理类别变量）
X_cat = X[['Coded']].fillna(X['Coded'].mode()[0])
X_cat['Coded'] = X_cat['Coded'].astype(str)

# 合并处理后的特征
X_processed = pd.concat([X_cat, X_numeric_scaled], axis=1)

# 4. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. 定义 CatBoostRegressor 模型
# CatBoost 可以自动处理类别特征（此处指定 'Coded' 列）
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

# 6. 训练模型（指定类别特征）
model.fit(X_train, y_train, cat_features=cat_features_list, eval_set=(X_test, y_test))

# 7. 预测与评估
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("CatBoost 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 8. 可视化预测结果
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'CatBoost Regression: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()
