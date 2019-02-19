import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Loading the dataset
data = pd.read_csv("headbrain.csv")

#Dividing Testing and Training data
X = data['Head Size(cm^3)']
Y = data['Brain Weight(grams)']


# Calculate Mean
mean_x= np.mean(X)
mean_y= np.mean(Y)

m=len(X)

#Calculate Co-efficients
numer=0
denom=0
for i in range(m):
    numer= numer + (X[i]-mean_x)*(Y[i]-mean_y)
    denom = denom + (X[i]-mean_x)**2

b1= numer/denom
b0= mean_y - (b1* mean_x)
print("Co-efficients of Best fit Line=",b1,"and ",b0)

max_x = np.max(X)
min_x = np.min(Y)


x=np.linspace(min_x,max_x,1000)
y=b0 + b1*x

#Plotting
plt.scatter(X,Y,color='black',label='Scatter Plot')
plt.plot(x,y,color='blue',linewidth=3,label='Regression Line')

plt.xticks(())
plt.yticks(())
plt.legend()
plt.show()


#Calculating Root Mean Squared Error
mse = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    mse += (Y[i] - y_pred) ** 2
mse = np.sqrt(mse/m)
print("Root Mean Squared Error =",mse)

#Calculate r-square
ss_t = 0
ss_r = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)
print("Variance score=",r2)

