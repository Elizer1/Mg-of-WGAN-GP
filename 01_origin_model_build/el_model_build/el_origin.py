#LightGBM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

# 1. 读取数据
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_database_clean_Ductility1.xlsx'
data = pd.read_excel(data_path)

# 2. 去除目标异常值
ys_mean = data['Ductility'].mean()
ys_std = data['Ductility'].std()
filtered_data = data[(data['Ductility'] > ys_mean - 3*ys_std) & (data['Ductility'] < ys_mean + 3*ys_std)]

# 3. 特征和目标
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'Ductility'
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

print("LightGBM 基线模型性能:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 7. 结果可视化
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Ductility')
plt.ylabel('Predicted Ductility')
plt.title(f'LightGBM Regression: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

# 8. 保存训练集和测试集预测结果到桌面
y_pred_train = model.predict(X_train)
train_results_df = pd.DataFrame({
    'Actual Train': y_train.values,
    'Predicted Train': y_pred_train
})
test_results_df = pd.DataFrame({
    'Actual Test': y_test.values,
    'Predicted Test': y_pred
})

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
train_file = os.path.join(desktop_path, 'Train_Prediction_LightGBM_Ductility.csv')
test_file = os.path.join(desktop_path, 'Test_Prediction_LightGBM_Ductility.csv')
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
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

# 1. 读取数据（请确保路径正确）
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_database_clean_Ductility1.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = ['Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
                    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
                    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi']
target_column = 'Ductility'  # 假设目标列是延伸率 (Ductility)

X = data[features_columns]
y = data[target_column]

# 3. 预先填充缺失值
# 对数值特征采用均值填充
for col in features_columns:
    if X[col].dtype in [np.float64, np.int64]:
        X[col] = X[col].fillna(X[col].mean())

# 对类别特征 'Coded' 使用众数填充（并转换为字符串）
X['Coded'] = X['Coded'].fillna(X['Coded'].mode()[0]).astype(str)

# 4. 对 'Coded' 列进行 One-Hot 编码
# 使用 drop_first=True 避免虚拟变量陷阱
X = pd.get_dummies(X, columns=['Coded'], drop_first=True)

# 检查是否还有缺失值
if X.isnull().sum().sum() > 0:
    print("警告：仍存在缺失值")
else:
    print("所有特征已填充，数据无缺失。")

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.1, 0.15, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 0.95, 1.0]
}

# 7. 定义评估指标（R² 得分）
from sklearn.metrics import make_scorer
r2 = make_scorer(r2_score)

# 8. 使用 GridSearchCV 进行参数搜索（5折交叉验证）
grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
                           param_grid=param_grid,
                           scoring=r2,
                           cv=5,
                           n_jobs=-1,
                           verbose=1)
print("开始 GridSearchCV 进行 GradientBoostingRegressor 参数调优...")
grid_search.fit(X_train, y_train)
print("GridSearchCV 完成。")
print("最佳参数：", grid_search.best_params_)

# 9. 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 10. 模型评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2_score_value = r2_score(y_test, y_pred)

print("GradientBoostingRegressor 模型性能评估:")
print(f"R² Score: {r2_score_value:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 11. 可视化预测结果
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Ductility')
plt.ylabel('Predicted Ductility')
plt.title(f'GradientBoostingRegressor: Actual vs Predicted (R² = {r2_score_value:.4f})')
plt.legend()
plt.tight_layout()



# 13. 保存数据库（训练集和测试集预测结果）到桌面
# 计算训练集的预测结果
y_pred_train = best_model.predict(X_train)

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

# 定义保存文件的路径
train_file = os.path.join(desktop_path, 'Train_Prediction_GBDT_Ductility.csv')
test_file = os.path.join(desktop_path, 'Test_Prediction_GBDT_Ductility.csv')

# 保存 CSV 文件
train_results_df.to_csv(train_file, index=False)
test_results_df.to_csv(test_file, index=False)

print("训练集预测结果已保存至:", train_file)
print("测试集预测结果已保存至:", test_file)
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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. 读取数据（请确保路径正确）
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_database_clean_Ductility1.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'Ductility'
X = data[features_columns]
y = data[target_column]

# 3. 数据预处理：
# 数值特征：均值填充 + 标准化；
# 类别特征 "Coded"：最频繁值填充 + One-Hot 编码。
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

# 4. 构建完整 Pipeline：预处理 -> XGBRegressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42, objective='reg:squarederror'))
])

# 5. 定义参数搜索范围（细调 XGBRegressor 参数）
param_grid = {
    'regressor__n_estimators': [300, 400, 500],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__max_depth': [3, 4, 5, 6],
    'regressor__subsample': [0.8, 0.9, 1.0],
    'regressor__colsample_bytree': [0.8, 1.0]
}

# 6. 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 使用 GridSearchCV 进行参数搜索（5折交叉验证）
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=1, verbose=2)
print("开始 GridSearchCV 进行 XGBRegressor 参数调优...")
grid_search.fit(X_train, y_train)
print("GridSearchCV 完成。")
print("最佳参数：", grid_search.best_params_)

# 8. 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 9. 模型评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2_val = r2_score(y_test, y_pred)

print("XGBRegressor 模型性能评估:")
print(f"R² Score: {r2_val:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 10. 可视化预测结果
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Ductility')
plt.ylabel('Predicted Ductility')
plt.title(f'XGBRegressor: Actual vs Predicted (R² = {r2_val:.4f})')
plt.legend()
plt.tight_layout()

# 保存图像到桌面（Windows环境）
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
plot_path = os.path.join(desktop_path, "XGB_Ductility_Prediction.png")
plt.savefig(plot_path)
print(f"预测图像已保存至: {plot_path}")
plt.show()

# 11. 保存最佳模型到桌面
model_path = os.path.join(desktop_path, "XGB_Ductility_Model.pkl")
joblib.dump(best_model, model_path)
print("最佳模型已保存至:", model_path)

# 12. 保存数据库（训练集和测试集的预测结果）到桌面
y_pred_train = best_model.predict(X_train)

train_results_df = pd.DataFrame({
    'Actual Train': y_train.values,
    'Predicted Train': y_pred_train
})
test_results_df = pd.DataFrame({
    'Actual Test': y_test.values,
    'Predicted Test': y_pred
})

train_file = os.path.join(desktop_path, 'Train_Prediction_XGB_Ductility.csv')
test_file = os.path.join(desktop_path, 'Test_Prediction_XGB_Ductility.csv')

train_results_df.to_csv(train_file, index=False)
test_results_df.to_csv(test_file, index=False)

print("训练集预测结果已保存至:", train_file)
print("测试集预测结果已保存至:", test_file)
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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. 读取数据（请确保路径正确）
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_database_clean_Ductility1.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'Ductility'  # 请确保目标列名称与文件中一致

X = data[features_columns]
y = data[target_column]

# 3. 数据预处理：
# 数值特征（除 'Coded' 外）：均值填充 + StandardScaler；
# 类别特征 'Coded'：最频繁值填充 + One-Hot 编码。
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

# 4. 定义基模型：
rf = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
gbdt = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42)
cat = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05, verbose=0, random_seed=42)

base_estimators = [
    ('rf', rf),
    ('gbdt', gbdt),
    ('cat', cat)
]

# 定义元模型：使用 Ridge
meta_model = Ridge(alpha=1.0)

stacking_reg = StackingRegressor(
    estimators=base_estimators,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

# 5. 构建完整 Pipeline：预处理 -> StackingRegressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('stacking', stacking_reg)
])

# 6. 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 使用 GridSearchCV 对元模型参数进行微调（例如调整 Ridge 的 alpha）
param_grid = {
    'stacking__final_estimator__alpha': [0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=1, verbose=2)
print("开始 GridSearchCV 进行 StackingRegressor 参数调优...")
grid_search.fit(X_train, y_train)
print("GridSearchCV 完成。")
print("最佳参数：", grid_search.best_params_)

# 8. 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 9. 模型评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Stacking Ensemble (RF, GBDT, CatBoost) 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 10. 可视化预测结果，并保存图像到桌面
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Ductility')
plt.ylabel('Predicted Ductility')
plt.title(f'Stacking Ensemble (RF, GBDT, CatBoost): Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()

# 获取桌面路径（Windows环境）
desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')

# 保存预测图像到桌面（保存前调用 savefig，然后再显示）
plot_path = os.path.join(desktop_path, "stacking_ductility_prediction.png")
plt.savefig(plot_path)
print(f"预测图像已保存至: {plot_path}")

plt.show()

# 11. 保存最佳模型到桌面
model_path = os.path.join(desktop_path, "best_stacking_model_rf_gbdt_catboost.pkl")
joblib.dump(best_model, model_path)
print(f"最佳模型已保存至: {model_path}")

# 12. 保存数据库（即保存训练集和测试集的预测结果）到桌面
y_pred_train = best_model.predict(X_train)

train_results_df = pd.DataFrame({
    'Actual Train': y_train.values,
    'Predicted Train': y_pred_train
})
test_results_df = pd.DataFrame({
    'Actual Test': y_test.values,
    'Predicted Test': y_pred
})

train_file = os.path.join(desktop_path, 'train_actual_vs_predicted_stacking_ductility.csv')
test_file = os.path.join(desktop_path, 'test_actual_vs_predicted_stacking_ductility.csv')

train_results_df.to_csv(train_file, index=False)
test_results_df.to_csv(test_file, index=False)

print(f"训练集预测结果已保存至: {train_file}")
print(f"测试集预测结果已保存至: {test_file}")
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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. 读取数据（请确保路径正确）
data_path = 'C:/Users/Administrator/Desktop/清除数据的数据库/alloy_database_clean_Ductility1.xlsx'
data = pd.read_excel(data_path)

# 2. 定义特征和目标变量
# 假设特征列与之前一致，目标列为 "Ductility"
features_columns = [
    'Coded', 'Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag',
    'Ho', 'Mn', 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga',
    'Fe', 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'
]
target_column = 'Ductility'  # 请确保目标列名称与文件中一致

X = data[features_columns]
y = data[target_column]

# 3. 数据预处理
# 对数值特征（除 'Coded' 外）：均值填充 + 标准化
numeric_features = [col for col in features_columns if col != 'Coded']
num_imputer = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_features]), columns=numeric_features)
scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=numeric_features)

# 对类别特征 "Coded"：缺失值填充（众数）并转换为字符串
cat_imputer = SimpleImputer(strategy='most_frequent')
X_cat = pd.DataFrame(cat_imputer.fit_transform(X[['Coded']]), columns=['Coded'])
X_cat['Coded'] = X_cat['Coded'].astype(str)

# 合并处理后的数值和类别特征
X_processed = pd.concat([X_cat, X_numeric_scaled], axis=1)

# 4. 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 5. 定义 CatBoostRegressor 模型
model = CatBoostRegressor(
    loss_function='RMSE',
    verbose=100,
    early_stopping_rounds=50
)

# 6. 定义参数搜索范围
param_grid = {
    'iterations': [800, 1000],
    'learning_rate': [0.03, 0.05, 0.07],
    'depth': [6, 7, 8],
    'l2_leaf_reg': [1, 3, 5]
}

# 7. 使用 GridSearchCV 进行参数调优
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=1, verbose=2)
print("开始 GridSearchCV 调优 CatBoostRegressor...")
grid_search.fit(X_train, y_train, cat_features=['Coded'])
print("GridSearchCV 完成。")
print("最佳参数：", grid_search.best_params_)

# 8. 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 9. 模型评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("CatBoostRegressor 模型性能评估:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} MPa")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# 新增：保存预测结果与模型
desktop_path = 'C:/Users/Administrator/Desktop/'  # 请确认此路径正确

# 保存训练集预测结果
y_pred_train = best_model.predict(X_train)
train_results_df = pd.DataFrame({
    'Actual Train': y_train.values,
    'Predicted Train': y_pred_train
})
train_file = os.path.join(desktop_path, 'train_actual_vs_predicted_catboost_ductility.csv')
train_results_df.to_csv(train_file, index=False)

# 保存测试集预测结果
test_results_df = pd.DataFrame({
    'Actual Test': y_test.values,
    'Predicted Test': y_pred
})
test_file = os.path.join(desktop_path, 'test_actual_vs_predicted_catboost_ductility.csv')
test_results_df.to_csv(test_file, index=False)

print("训练集和测试集的预测结果已保存到桌面。")

# 保存最佳模型到桌面
model_file = os.path.join(desktop_path, 'best_catboost_ductility_model.joblib')
joblib.dump(best_model, model_file)
print("最佳模型已保存到桌面。")

# 10. 可视化预测结果
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Ductility')
plt.ylabel('Predicted Ductility')
plt.title(f'CatBoostRegressor: Actual vs Predicted (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()
# 计算训练集 R²
r2_train = r2_score(y_train, y_pred_train)

# 输出训练集 R²
print(f"训练集 R² Score: {r2_train:.4f}")