import numpy as np
from sklearn.linear_model import LinearRegression

# 지도학습
#y= 3x+2
X=np.linspace(0,10,10)
y= X * 3 + 2
X = X.reshape(-1,1)


m =LinearRegression()
m.fit(X,y)

print("m.coef_:",m.coef_)
print("m.intercept_:",m.intercept_)