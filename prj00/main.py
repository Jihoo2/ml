# 개요
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import pipeline
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X = data.data
y = data.target

X = pd.DataFrame(X, columns=data.feature_names)
y = pd.DataFrame(y, columns=["target"])

# DateFrame으로 만들어서 현재 경로에 california_housing.csv이름으로 저작
df = X.copy()
df["target"] = y
df.to_csv("california_housing.csv", index=False)
# 데이터 정보 학인 = 다음 2개 반드시 포함
# shape
# describe


print(df.shape)
df.describe().to_csv("california_housing_desc.csv", index=False)

#
# 결측치 확인
# -칼럼별 결측치 갯수 파악할것
print(df.isna().sum())

# 데이터 탐색
# -가격의 분포를 히스토그램으로 확인
# df.hist()
df.hist(figsize=(12, 8), grid=False, bins=20)

sns.heatmap(df.corr(), annot=True ,cmap="coolwarm")

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


pipeline=make_pipeline(StandardScaler(),LinearRegression())
pipeline.fit(X_train,y_train)
#
# scaler = StandardScaler()
# X_train_S = scaler.fit_transform(X_train)
# X_test_s = scaler.transform(X_test)
#
#
# m = LinearRegression()
# m.fit(X_train_S, y_train)
y_pred = pipeline.predict(X_test)



r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse =mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)

print("r2 :",r2)
print("mae :",mae)
print("rmse :",rmse)



