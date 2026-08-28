'''
머신러닝<- 전처리가 매우중요
전처리가 모델의 성능을 결정해줌
전처리할때 결측지,중복,인코딩,파생변수 사이즈,시계열

'''
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

#결측치 처리
# df= pd.DataFrame({
#     "age":[20,30,np.nan,40,50],
#     "score":[100,np.nan,80,70,np.nan],
# })
# print(df)
#
# df["age"]=df["age"].fillna(df["age"].mean())
#
# # x=df["age"]
# # x.iloc[2] =x.mean()
# imputer=SimpleImputer(strategy="median").set_output(transform="pandas")
# # imputer.fit(df[["age","score"]])
# result=imputer.fit_transform(df)
# print(result)

#스케일링
# X = np.array([[1.0, 100.0],
#               [2.0, 300.0],
#               [3.0, 500.0],
#               [4.0, 700.0]])
# scaler = StandardScaler()
# #knn 거리기반으로 동작 알고리즘엔 스케일링이반드시 필요하다
# #값이 너무 크면 큰값이 값을 다잡아먹어버려서
#
# x_scaled=scaler.fit_transform(X)
# print(x_scaled)


#범주형 인코딩(순위x)
# df = pd.DataFrame({"color": ["red", "green", "blue", "green"]})
#
# result=pd.get_dummies(df,columns=["color"],drop_first=True)
# print(result)

#범주형 인코팅(순위o)
df = pd.DataFrame({"size": ["소", "대", "중", "소"]})

df['size'] = df['size'].map({
    "소":1,
    "중":2,
    "대":3
})

print(df)

df = pd.DataFrame({
    "size": ["소", "대", "중", "소"],
    "grade" : ["Gold", "Bronze", "Silver", "Bronze"]
})
enc = OrdinalEncoder(categories=[
    ["소", "중", "대"],
    ["Bronze", "Silver", "Gold"]
])
df[["size","grade"]]=enc.fit_transform(df[["size", "grade"]])
print(df)

#파생칼럼 ,중복제거,시계열 인덱싱

# 누수없이 전처리
#데이터준비
X,y = load_iris(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=10
)
#전처리
scaler= StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#모델
m=KNeighborsClassifier(n_neighbors=3) # 반드시 홀수여야만함 짝수는동률이나와서

#파이프라인(전처리 + 모델)
#sk가 해주는 전처리를 여기서 묶어줄 수 있음
pipe = make_pipeline(scaler,m)

#학습
pipe.fit(X_train,y_train)
#예측
y_pred = pipe.predict(X_test)
#평가
print(y_pred)
accuracy_score=accuracy_score(y_test,y_pred)

















