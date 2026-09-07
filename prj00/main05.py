#트리, 앙상블
from matplotlib.testing.jpl_units import rad
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

#===결정트리 ===
# iris = load_iris()
# X ,y= iris.data , iris.target
# X_train,X_test,y_train,y_test = train_test_split(
#     X,y,test_size = 0.2,
#     random_state = 42,
# )
#
# m= DecisionTreeClassifier(max_depth=3,random_state=42)
# m.fit(X_train,y_train)
# y_pred = m.predict(X_test)
#
# acc_score = accuracy_score(y_test,y_pred)
# print("acc_score : " ,acc_score)
#
# result = export_text(m,feature_names = iris.feature_names)
#
# print(result)

#=== 앙상블 ===배깅
# iris = load_iris()
# X,y = iris.data , iris.target
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
#
# m= RandomForestClassifier(n_estimators=100,random_state=42,max_depth=3)
# m.fit(X_train,y_train)
# y_pred = m.predict(X_test)
# acc_score = accuracy_score(y_test,y_pred)
#
# result = classification_report(y_test,y_pred)
# print(result)

#=== 앙상블 ===부스팅
cancer = load_breast_cancer()
X,y =cancer.data,cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

m=GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

m.fit(X_train, y_train)
y_pred = m.predict(X_test)

result = classification_report(y_test,y_pred)
print(result)