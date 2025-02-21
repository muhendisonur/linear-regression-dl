import torch

# f = w * x + b
# f = 5 * x

X = torch.randn(5, dtype=torch.float16)
Y = X * 5

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return x * w

def loss(y, y_pred):
    return ((y - y_pred)**2).mean() # MSE

n_epoch = 200
lr = 0.01

for epoch in range(n_epoch):
    y_pred = forward(X)
    l = loss(Y, y_pred)
    l.backward()

    with torch.no_grad():
        w -= lr * w.grad
    w.grad.zero_()
    if((epoch+1) % 10 == 0):
        print(f"Epoch: {epoch+1}, loss: {l.item():.2f}, weight: {w.item():.2f}")

print(f"Y: {Y} \nY_pred: {forward(X)}")