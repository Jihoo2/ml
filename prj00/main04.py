# # 모델 평가 및 검증
# # 교차검증 / 지표 / 과적합
#
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification, load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, \
    mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from main import rmse

#
# X, y = load_breast_cancer(return_X_y=True)
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42
# )
#
# for depth in range(1, 11):
#
#     m = DecisionTreeClassifier(random_state=42,max_depth=depth)
#     m.fit(X_train, y_train)
#     y_pred_train = m.predict(X_train)
#     y_pred_test = m.predict(X_test)
#
#     acc_score_train = accuracy_score(y_train, y_pred_train)
#     acc_score_test = accuracy_score(y_test, y_pred_test)
#
#     print(f"max_depth={depth} : 훈련 {acc_score_train}점 /테스트 {acc_score_test}점")

#==== f-fold교차 검증 ====
# X,y =load_breast_cancer(return_X_y=True)
#
# m= LogisticRegression(max_iter=5000)
# m.fit(X,y)
#
# result = cross_val_score(m,X,y,cv = 5 ,scoring="accuracy")
# print(result.mean())
# print(result.std())

 #==== f-fold교차 검증 ====stratified
# X,y =load_breast_cancer(return_X_y=True)
#
# m= LogisticRegression(max_iter=5000)
# m.fit(X,y)
#
# skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
# result = cross_val_score(m,X,y,cv = skf ,scoring="accuracy")
# print(result.mean())
# print(result.std())

#==== f-fold교차 검증 ==== 모든 지표 한번에
# X,y =load_breast_cancer(return_X_y=True)
# m= LogisticRegression(max_iter=4000)
# scores = ["accuracy","precision","recall","f1"]
# result = cross_validate(m,X,y,cv = 5 ,scoring="scoring")
# for k in result:
#     print(f"{k}:{result[k]}")

#=== 불균형 데이터는 정확도로 판단하면 안됨 ===
# X,y =make_classification(
#     n_samples=1000,
#     n_features=6,
#     n_informative=4,
#     n_classes=3,
#     weights=[0.9,0.05,0.05],
#     random_state=42,
# )
#
# y_pred = np.zeros_like(y)
# acc_score = accuracy_score(y,y_pred)
# print("acc_score:",acc_score)
#
# cm =confusion_matrix(y,y_pred)
# print("cm:",cm)
#=== 혼동 행렬 ===

# X, y = load_breast_cancer(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42
# )
# m = LogisticRegression(max_iter=5000)
# m.fit(X_train, y_train)
# y_pred = m.predict(X_test)
# cm = confusion_matrix(y_test,y_pred)
# print(cm)
# cr = classification_report(y_test,y_pred)
# print(cr)
#=== 회귀 모델 ===
X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

m = LinearRegression()

m.fit(X_train, y_train)

y_pred = m.predict(X_test)

# 평가 지표
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse =np.sqrt(mse)

print(f"R²   : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")


