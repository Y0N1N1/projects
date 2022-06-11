import csv

import math

from sympy import Matrix


train_x, test_x = [], []

train_y, test_y = [], []

with open('train.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader: 
        train_x.append(row[1:-1])
        train_y.append(row[0])
    
with open('test.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader: 
        test_x.append(row[1:-1]) 
        test_y.append(row[0])

# normalize
def norm255(x):
    return float(x)/255

new_train_x, new_test_x = [], []
for i in train_x:
    x_new = [norm255(x) for x in i]
    x_new.append(0.0)
    new_train_x.append(x_new)
    
for i in test_x:
    x_new = [norm255(x) for x in i]
    x_new.append(0.0)
    new_test_x.append(x_new)

# add 784th element ^^^ too


train_x, test_x = new_train_x, new_test_x

# turn ys from numbers to probability distributions

new_train_y, new_test_y = [], []

def OneHotEncoding(x):
    nn = []
    for a in range(0,int(i)):
        nn.append(0)
    nn.append(1)
    for a in range(int(i), 9):
        nn.append(0)
    return nn

# one hot encoding is already done here \/
for i in train_y:
    new_train_y.append(OneHotEncoding(i))
for i in test_y:
    new_test_y.append(OneHotEncoding(i))

train_y, test_y = new_train_y, new_test_y

""" test if the datasets work
print(test_x[0])
print(test_y[0])
print(train_x[0])
print(train_y[0])
"""

import random
l1 = []
l2 = []
b1 = []
b2 = []

def InitParameters():
    for i in range(300):
        x = []
        for j in range(784):
            x.append(random.uniform(-1, 1))
        l1.append(x)
    for i in range(10):
        x = []
        for j in range(300):
            x.append(random.uniform(-1, 1))
        l2.append(x)
    for i in range(300):
        b1.append(random.uniform(-1, 1))
    for i in range(10):
        b2.append(random.uniform(-1, 1))



"""
print(l1)
print(l2)
"""
# 784 - 300 - 10
# [300x784], [10x300]

def MatrixFloatProduct(m, f):
    return [[i*f for i in m[n]] for n, j in enumerate(m)]

"""
Test:

print("Expected: [[5, 10],[15, 20]], Actual: ", MatrixFloatProduct([[1, 2],[3, 4]], 5))
"""

def TransposeMatrix(matrix):
    new_m = [[] for m, j in enumerate(matrix[0])]
    for n1, i1 in enumerate(matrix):
        for m1, j1 in enumerate(i1):
            new_m[m1].append(j1)
    return new_m

"""
print(TransposeMatrix([[2, 1],[6, 2]]))
print(TransposeMatrix([[2, 1, 22121],[6, 2, -234], [100, 200, 300]]))
"""

def MatrixVectorProduct(m, v):
    out = []
    for i in m:
        ff = 0
        for n,j in enumerate(i):
            ff +=float(j)*float(v[n])
        out.append(ff)
    return out
#test cases

"""
print(MatrixVectorProduct([[2, 3, -4],[11, 8, 7],[2, 5, 3]], [3, 7, 5]))
# dim changing matrix:
print(MatrixVectorProduct([[5, 2, 6], [1, -25 ,3]], [3, 1, 6]))
"""

# bias vectors

e = 2.71828182845904523536028747135266249775724709369995

def Sigmoid(x):
    def Single(a):
        try: return (1/(1+pow(e, -(a)))) 
        except OverflowError: return (1 if a>0 else 0)
    return [Single(i) for i in x]

def SigmoidGrad(x):
    def Single(a):
        try: return pow(e, -a)/((1+pow(e, -a))*(1+pow(e, -a)))
        except OverflowError: return 0
    return [Single(i) for i in x]


"""
tests for sigmoid:
Sigmoid(2) -> 0.8807971
Sigmoid(4) -> 0.9820138
Sigmoid(-2) -> 0.1192029
Sigmoid(222) -> 1
Sigmoid(-4234) -> 0

print(Sigmoid([2]))
print(Sigmoid([4]))
print(Sigmoid([-2]))
print(Sigmoid([222]))
print(Sigmoid([-4234]))

tests for sigmoid derivatives:

print("Expected: [0.25], Actual: ", SigmoidDerivate([0]))
print("Expected: [0.104994], Actual: ", SigmoidDerivate([2]))
print("Expected: [0.0176627], Actual: ", SigmoidDerivate([4]))
print("Expected: [0.104994], Actual: ", SigmoidDerivate([-2]))
print("Expected: [3.86034E-97], Actual: ", SigmoidDerivate([222]))
print("Expected: [1.57458E-1839], Actual: ", SigmoidDerivate([-4234]))
"""

def VectorAddition(x, y):
    return [i+y[n] for n,i in enumerate(x)]


def VectorTensorProduct(x, y):
    out = []
    for i in x: out.append([i*j for j in y])
    return out



"""
Test for Vector Tensor Product:

print("Expected: [[10, -6],[15, -9]], Actual: ", VectorTensorProduct([2, 3],[5, -3]))
"""

def Softmax(x):
    tot = sum([pow(e, i) for i in x])
    return [(pow(e, i))/tot for i in x]

def SoftmaxGrad(x):
    out = []
    a = sum([pow(e, i) for i in x])
    for i in x:
        try: out.append((a*pow(e, i))/((a+pow(e, i))*(a+pow(e, i))))
        except ZeroDivisionError: out.append((a*pow(e, i))/(0.0001+(a+pow(e, i))*(a+pow(e, i))))
    return out

""" test softmax:
print("Expected: [0.211941, 0.576116, 0.211941], Actual:", Softmax([2, 3, 2]))
print("Expected: [0.0000000008, 0.9999999992, 0], Actual:", Softmax([32, 53, 12]))
"""

def ProbabilityVectorToResult(x):
    return x.index(max(x))

def LossSquared(x, y):
    return sum([(i-y[n])*(i-y[n]) for n, i in enumerate(x)])

def LossSquaredGrad(x, y):
    return [2*(i-y[n]) for n, i in enumerate(x)]
"""
test loss:
print("Expected: 0, Actual: ", LossSquared([0, 0, 2],[0, 0, 2]))
print("Expected: 2, Actual: ", LossSquared([0, 1, 2],[1, 0, 2]))
print("Expected: 1.25, Actual: ", LossSquared([3, 0.5, 2],[2, 1, 2]))
print("Expected: 54014, Actual: ", LossSquared([35, 5, 232],[2, 0, 2]))

"""

def VectorScalarMultiplication(v, s):
    return [i*s for i in v]

def NVectorsAddition(list_of_vectors):
    return VectorAddition(list_of_vectors[0],NVectorsAddition(list_of_vectors[1:])) if len(list_of_vectors) > 1 else list_of_vectors[0]

"""
Test NVectorsAddition:

print("Expected: [3, 4, 2, 2, 5], Actual: ", NVectorsAddition([[1, 2, -3, 0, 5],[1, 1, 3, 0.5, -5],[1, 1, 2, 1.5, 5]]))

"""

def MatrixAddition(m1, m2):
    return [VectorAddition(i, m2[n]) for n, i in enumerate(m1)]

"""
test matrix addition:

print("Expected: [[2, 254],[-2, 2554]], Actual: ", MatrixAddition([[1, 0], [-234, 5108]], [[1,  254], [232, -2554]]))
"""

def NMatricesAddition(list_of_matrices):
    return MatrixAddition(list_of_matrices[0], NMatricesAddition(list_of_matrices[1:])) if len(list_of_matrices) > 1 else list_of_matrices[0]

"""
test n matrix addition:

print("Expected: [[5, 255],[23, 2512]], Actual: ", NMatricesAddition([[1, 0], [-234, 5108]], [[1,  254], [232, -2554]], [[3, 1],[25, -42]]))
"""

def AverageVectors(list_of_vectors):
    return VectorScalarMultiplication(NVectorsAddition(list_of_vectors), 1/len(list_of_vectors))

def AverageMatrix(list_of_matrices):
    return(MatrixFloatProduct(NMatricesAddition(list_of_matrices), 1/len(list_of_matrices)))


def VectorComponentMultiplication(x, y):
    return [i*y[n] for n, i in enumerate(x)]

# note these forward functions are only used for testing or applying the net cuz when training you need to compute stuff that isn't outputted on this function.
# net structure
def NeuralNetForward(x):
    out = x
    #first
    out =  Sigmoid(VectorAddition(MatrixVectorProduct(l1, out), b1))
    #second
    out =  Softmax(VectorAddition(MatrixVectorProduct(l2, out), b2))
    return out

def NeuralNetForwardError(x, y):
    return LossSquared(NeuralNetForward(x), y)

#print(NeuralNetForward(train_x[0]))

# get grads
def Grads(o1, y):
    #first
    o2 = VectorAddition(MatrixVectorProduct(l1, o1), b1)
    a2 =  Sigmoid(o2)
    #second
    o3 = VectorAddition(MatrixVectorProduct(l2, a2), b2)
    a3 =  Softmax(o3)
    err = LossSquared(a3, y)
    errg = LossSquaredGrad(a3, y)
    # kronecker deltas 
    k3 = VectorComponentMultiplication(errg, SoftmaxGrad(o3))
    k2 = VectorComponentMultiplication(MatrixVectorProduct(TransposeMatrix(l2), k3), SigmoidGrad(o2))
    b2Grad = k3
    b1Grad = k2
    l2Grad = VectorTensorProduct(k3, a2)
    l1Grad = VectorTensorProduct(k2, o1)
    return b2Grad, b1Grad, l2Grad, l1Grad, a3, err

def UpdateParameters(learning_rate, b2Grad, b1Grad, l2Grad, l1Grad):
    global l1
    global b1
    global l2
    global b2
    l1 = MatrixAddition(l1, MatrixFloatProduct(l1Grad, -learning_rate))
    b1 = VectorAddition(b1, VectorScalarMultiplication(b1Grad, -learning_rate))  
    l2 = MatrixAddition(l2, MatrixFloatProduct(l2Grad, -learning_rate))
    b2 = VectorAddition(b2, VectorScalarMultiplication(b2Grad, -learning_rate))  

learning_rate = 0.5

def BackProp(o1, y):
    UpdateParameters(learning_rate, Grads(o1, y))

def BatchData(batch_size):
    batched_train_x = []
    train_x_temp = train_x
    train_y_temp = train_y
    for i in range(len(train_x) // batch_size):
        batch = []
        for j in range(batch_size):
            batch.append(train_x_temp[0])
            train_x_temp.pop(0)
        batched_train_x.append(batch)
    batched_train_y = []
    for i in range(len(train_y) // batch_size):
        batch = []
        for j in range(batch_size):
            batch.append(train_y_temp[0])
            train_y_temp.pop(0)
        batched_train_y.append(batch)
    # match both lists
    return [[i, batched_train_y[n]] for n, i in enumerate(batched_train_x)]

# print(BatchData(2)[0])

def VectorAvg(x):
    return sum(x)/len(x)

def AnswerFromSoftmax(x):
    return x.index(max(x))

def Train(batch_size, epochs):
    for epoch in range(epochs):
        accuracy = []
        error = []
        # batch data (train_x, test_x, train_y, test_y)
        data = BatchData(batch_size)
        # averaged grads
        for batch in range(len(data)):
            b2Grads, b1Grads, l2Grads, l1Grads = [], [], [], []
            this_batch = data[0]
            for i in range(batch_size):
                b2Grad, b1Grad, l2Grad, l1Grad, result, err = Grads(this_batch[0][i], this_batch[1][i])
                b2Grads.append(b2Grad)
                b1Grads.append(b1Grad)
                l2Grads.append(l2Grad)
                l1Grads.append(l1Grad)
                # print(" | ", b2Grad, " | ", b1Grad, " | ", l2Grad, " | ", l1Grad, " | ", result, " | ", err)
                accuracy.append(1 if this_batch[1][i] == OneHotEncoding(AnswerFromSoftmax(result)) else 0)
                error.append(err)
            data.pop(0)
            b2Grad, b1Grad, l2Grad, l1Grad = AverageVectors(b2Grads), AverageVectors(b1Grads), AverageMatrix(l2Grads), AverageMatrix(l1Grads)
            UpdateParameters(learning_rate, b2Grad, b1Grad, l2Grad, l1Grad)
            # visualizing
            print(batch, "th batch accuracy: ", VectorAvg(accuracy)*100, "%, error: ", VectorAvg(error))
            if batch != (len(train_x) // batch_size):
                accuracy.clear()
                error.clear()
        print("Epoch: ", epoch, ", Accuracy: ", accuracy, "%, Error: ", error)

InitParameters()
Train(25, 3)

def Test():
    out = []
    pass
