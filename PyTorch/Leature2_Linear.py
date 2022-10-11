import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x_data = [1,2,3]
y_data = [2,4,6]

def forward(x):
    return x * w 

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

w_list = []
mse_list = []
for w in np.arange(0,4.0,0.1):
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print(x_val, y_val, y_pred_val, loss_val)
    print('MSE', l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)

plt.plot(w_list,mse_list)
plt.ylabel("Loss")
plt.xlabel("w")
plt.show()




# 计算该方程的损失函数：y = x * w + b （三维)
# 同理计算三维损失函数， 并且画出3D图：因为这个时候 loss是由 w 和 b 同时决定的
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

x_data = [1,2,3]
y_data = [2,4,6]

def equation(x, w, b):
    y_head = x * w + b 
    return y_head

def loss_cal(y_head, y_true):
    loss = (y_head - y_true) * (y_head - y_true)
    return loss
w_coll = []
mse_coll = []
b_coll = []
w_value = np.arange(0, 4.0, 0.1)
b_value = np.arange(0, 4.0, 0.1)

for w in w_value:
    print("weight is : ", w)
    for b in b_value:
        print("b is :", b)
        total_loss = []
        for x, y in zip(x_data, y_data):
            y_head = equation(x,w, b)
            loss = loss_cal(y_head, y)
            total_loss.append(loss)
        mse = np.cumsum(total_loss)[-1] / 3 
        print("MSE is :\n",mse, "\n")
        w_coll.append(w)
        b_coll.append(b)
        mse_coll.append(mse)

pd.DataFrame(w_coll)
pd.DataFrame(mse_coll)
pd.DataFrame(b_coll)

fig = plt.figure(figsize = (8,8))
ax = plt.axes(projection='3d')
ax.grid()

ax.scatter(w_coll, b_coll, mse_coll)
ax.set_title('3D Parametric Plot')

# Set axes label
ax.set_xlabel('w_coll', labelpad=20)
ax.set_ylabel('b_coll', labelpad=20)
ax.set_zlabel('mse_coll', labelpad=20)

plt.show()


