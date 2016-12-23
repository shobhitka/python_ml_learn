import sys
import os
import scipy as sp
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

data = sp.genfromtxt("/home/skumar40/sandbox/personal/machine-learning/book/1400OS_Code/1400OS_01_Codes/data/web_traffic.tsv", delimiter="\t")
print ("Before Cleaning: %s" % str(data.shape))

x = data[:,0]
y = data[:,1]

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

print("After Cleaning x, y : %s, %s" % (str(x.shape), str(y.shape)))

def plot_data(x, y):
    plt.scatter(x,y)
    plt.title("Web traffic over the last month")
    plt.xlabel("Time")
    plt.ylabel("HIts/hour")

    plt.xticks([w*7*24 for w in range(5)], ['week %i'%w for w in range(5)])

    plt.autoscale(tight=True)
    plt.grid(True)

def error(f, x, y):
    return sp.sum((f(x) - y)**2)

def plot_model_order_n(n, x, y):
    fpn = sp.polyfit(x, y, n)
    fn = sp.poly1d(fpn)
    print("Order %d model error: %f" % (fn.order, error(fn, x, y)))

    fx = sp.linspace(0, x[-1], 1000)

    if n == 1:
        linecolor="red"
    elif n == 2:
        linecolor="black"
    elif n == 3:
        linecolor="green"
    elif n == 10:
        linecolor="violet"
    elif n == 100:
        linecolor="cyan"
    else:
        linecolor="orange"

    plt.plot(fx, fn(fx), color=linecolor, linewidth=4, label=str(fn.order))
    plt.legend(loc="upper left")
    plt.autoscale(tight=True)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.grid(True)

#plot_data(x, y)
#plot_model_order_n(1, x, y)
#plot_model_order_n(2, x, y)
#plot_model_order_n(3, x, y)
#plot_model_order_n(10, x, y)
#plot_model_order_n(100, x, y)

#plt.show()

# Use inflection point and analyze separately
inflection = 3.5 * 7 * 24 # inflection point in hours
xi = x[:inflection] # data before inflection
yi = y[:inflection]

xa = x[inflection:] # data after inflection
ya = y[inflection:]

# Linear model func before and after inflection
fi = sp.poly1d(sp.polyfit(xi, yi, 1))
fa = sp.poly1d(sp.polyfit(xa, ya, 1))

plot_data(x, y)
print("Error after inflection point from base data\n");
for i in 1,2,3,10,100 :
    fn = sp.poly1d(sp.polyfit(x, y, i))
    print("Order %d model error: %f" % (fn.order, error(fn, xa, ya)))

#plt.show()
print("Total inflection error for %f" % (error(fi, xi, yi) + error(fa, xa, ya)))

# Separating training and test data; only using data after inflection point
fraction = 0.3
index = int(fraction * len(xa))

shuffled = sp.random.permutation(list(range(len(xa))))
test = sorted(shuffled[:index])
train = sorted(shuffled[index:])

fat1 = sp.poly1d(sp.polyfit(xa[train], ya[train], 1))
fat2 = sp.poly1d(sp.polyfit(xa[train], ya[train], 2))
fat3 = sp.poly1d(sp.polyfit(xa[train], ya[train], 3))
fat10 = sp.poly1d(sp.polyfit(xa[train], ya[train], 10))
fat100 = sp.poly1d(sp.polyfit(xa[train], ya[train], 100))

print("Test data errors for data after inflection point")
for f in [fat1, fat2, fat3, fat10, fat100 ]:
    print("Order %d model error: %f" % (f.order, error(f, xa[test], ya[test])))

# Winner function is order 2 after taking data from inflection point
# plot it
plot_data(x, y)
plot_model_order_n(2, xa[train], ya[train])
plt.show()

# Finding roots for the fat2 solution
# roots are the values that solve f(x) = 0
# We need solution to f(x) = Ax^2 + Bx + c where y = 100000
# or in other words we can solve for f(x) - 100000 = 0
# second parameter is the starting point for the solution
# start with last hour from which we have data
from scipy.optimize import fsolve
reached_max = fsolve(fat2 - 100000, len(x)) / (7 * 24)
print("100000 hits/hour are expetced at week %f" % reached_max[0])
