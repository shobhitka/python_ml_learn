import scipy as sp
import matplotlib.pyplot as plt

data = sp.genfromtxt("/home/skumar40/sandbox/personal/machine-learning/book/1400OS_Code/1400OS_01_Codes/data/web_traffic.tsv", delimiter="\t")
print "Before Cleaning"
print data.shape

x = data[:,0]
y = data[:,1]

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

print("After Cleaning x, y")
print x.shape, y.shape

def plot_data(x, y):
    plt.scatter(x,y)
    plt.title("Web traffic over the last month")
    plt.xlabel("Time")
    plt.ylabel("HIts/hour")

    plt.xticks([w*7*24 for w in range(5)], ['week %i'%w for w in range(5)])

    plt.autoscale(tight=True)
    plt.grid()

def error(f, x, y):
    return sp.sum((f(x) - y)**2)

def plot_model_linear(x, y):
    # Polyfit with linear model: order=1
    fp1 = sp.polyfit(x, y, 1)
    print ("Model parameters: %s" % fp1)

    f1 = sp.poly1d(fp1)
    print (error(f1, x, y))

    fx = sp.linspace(0, x[-1], 1000)
    plt.plot(fx, f1(fx), color="red", linewidth=4)
    plt.legend(["d=%i" % f1.order], loc="upper left")

plot_data(x, y)
plot_model_linear(x, y)
plt.show()
