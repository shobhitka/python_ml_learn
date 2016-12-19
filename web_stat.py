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

plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("HIts/hour")

plt.xticks([w*7*24 for w in range(5)], ['week %i'%w for w in range(5)])

plt.autoscale(tight=True)
plt.grid()
plt.show()


