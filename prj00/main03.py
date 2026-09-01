# #지도학습
# import numpy as np
# from sklearn.datasets import load_breast_cancer, load_iris
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
#
### ===로지스틱 회귀
# #X,y=load_breast_cancer(return_X_y=True)
# X,y=load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train_s = scaler.transform(X_train)
# X_test_s = scaler.transform(X_test)
#
# m=LogisticRegression(max_iter=500)
# m.fit(X_train_s, y_train)
# y_pred=m.predict(X_test_s)
# acc_score = accuracy_score(y_test,y_pred)
# print("acc_score:", acc_score)
#
# proba=m.predict_proba(X_test_s)
# print("proba:",np.round(proba,3))
# print("proba:",type(proba))


# from sklearn.datasets import load_wine
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
#
#
# #===knn ===
# X,y=load_wine(return_X_y=True)
#
# # 학습용 / 테스트용 데이터 분리
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=0
# )
#
# # 스케일링
# scaler = StandardScaler()
#
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
#
# # KNN 모델
# m = KNeighborsClassifier(n_neighbors=5)
#
#
# m.fit(X_train, y_train)
# y_pred = m.predict(X_test)
# m.score(X_test, y_test)
# acc_score =accuracy_score(y_test,y_pred)
# print("acc_score: ",acc_score)



# from sklearn.datasets import load_wine
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# from sklearn.svm import SVC
#
# #===knn ===
# X,y=load_wine(return_X_y=True)
#
# # 학습용 / 테스트용 데이터 분리
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=0
# )
#
# # 스케일링
# scaler = StandardScaler()
#
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# #svm
# m= SVC(kernel="rbf")
# m.fit(X_train, y_train)
# y_pred = m.predict(X_test)
# m.score(X_test, y_test)
# acc_score = accuracy_score(y_test,y_pred)
# print("acc_score: ",acc_score)

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# 모델별 성능비고
# X ,y  =load_iris(return_X_y=True)
X, y =load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

models = {
    "로지스틱" : make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),
    "KNN": make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5)),
    "SVM": make_pipeline(StandardScaler(), SVC())
}

for model_name,model_pipline in models.items():
    model_pipline.fit(X_train, y_train)
    y_pred = model_pipline.predict(X_test)
    acc_score =accuracy_score(y_test, y_pred)
    print(model_name + "acc_score:",acc_score)
