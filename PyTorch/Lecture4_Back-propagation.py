# 如何在PyTorch中构建模型？
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 3.0, 4.0]
w = torch.tensor([1.0]) #w就一个值就是1.0
w.requires_grad = True #他是需要计算梯度的

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)", 4, forward(4).item())
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        #注意损失值为l.item()； 如果想计算所有损失值的和就用sum += l.item()
        print("\t grad:" , x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()#把梯度中的数据全部清零
    print("progress:", epoch, l.item(),"\n")
print("predict (after training)", 4, forward(4).item())






# Experience first time torch model 
import torch 
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = torch.tensor([1.0])
w.requires_grad = True
def forward(x):
    return x * w

def loss(x, y):
    loss = (forward(x) - y) ** 2
    return loss

print("Before training:", 4, forward(4).item())
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print("\t grad:", w.grad.item())
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()
    print("\t progress:\n", "epoch:",epoch, "loss:", l.item(), "\n")
print("predict (after training)", 4, forward(4).item())







