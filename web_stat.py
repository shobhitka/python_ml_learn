import scipy as sp
import matplotlib.pyplot as plt

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
    plt.grid()

def error(f, x, y):
    return sp.sum((f(x) - y)**2)

def plot_model_order_n(n, x, y):
    fpn = sp.polyfit(x, y, n)
    fn = sp.poly1d(fpn)
    print("Order %d model error: %f" % (n, error(fn, x, y)))

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

    plt.plot(fx, fn(fx), color=linecolor, linewidth=4)
    plt.legend(["d=%i" % fn.order], loc="upper left")

plot_data(x, y)
plot_model_order_n(1, x, y)
plot_model_order_n(2, x, y)

plt.show()
