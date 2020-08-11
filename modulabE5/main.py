from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

df = load_diabetes()
df_x = df.data
df_y = df.target

x_train, y_train , x_test, y_test = train_test_split(df_x,df_y,test_size = 12, random_state = 1)

def model(x, w, b):
    y = sum((w * x).T) + b
    return y

def loss(x, w, b, y):
    predictions = model(x, w, b)
    L = MSE(predictions, y)
    return L

def gradient(x, w, b, y):
    dw = (loss(x, w + 0.0001, b, y) - loss(x, w, b, y)) / 0.0001
    db = (loss(x, w, b + 0.0001, y) - loss(x, w, b, y)) / 0.0001
    return dw, db

def MSE(a, b):
    mse = ((a - b) ** 2).mean() 
    return mse
losses = []

w,b = 3,3
LEARNING_RATE=0.001

for i in range(1, 100001):
    dW, db = gradient(x_train, w, b, x_test)
    w -= LEARNING_RATE * dW
    b -= LEARNING_RATE * db
    L = loss(x_train, w, b, x_test)
    losses.append(L)
    if i % 10 == 0:
        print('Iteration %d : Loss %0.4f' % (i, L))