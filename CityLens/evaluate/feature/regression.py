import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
model_name = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
model_name_full = model_name.replace("/", "_")
csv_file = f"/data5/liutianhui/UrbanSensing/data/health/US_health_mental_mlp_task_{model_name_full}.csv"
df = pd.read_csv(csv_file)

# 13个图像特征列
features = [
    "Person", "Bike", "Heavy Vehicle", "Light Vehicle", "Façade", "Window & Opening",
    "Road", "Sidewalk", "Street Furniture", "Greenery - Tree",
    "Greenery - Grass & Shrubs", "Sky", "Nature"
]

# 目标列
target = "reference"

# 丢弃有缺失值的行
df = df.dropna(subset=features + [target])

# 分割数据
X = df[features].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# # 创建 MLP 回归模型
# model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
#                      max_iter=500, random_state=42)

# # 模型训练
# model.fit(X_train_scaled, y_train)

# # 预测
# y_pred = model.predict(X_test_scaled)

# 5 折交叉验证的 Lasso 回归
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

# 使用最佳 alpha 进行预测
y_pred = lasso_cv.predict(X_test_scaled)

# 评估结果
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"测试集均方误差（MSE）: {mse:.4f}")
print(f"R²分数: {r2:.4f}")

