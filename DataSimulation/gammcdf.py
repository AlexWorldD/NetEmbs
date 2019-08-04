from scipy.special import gamma
from scipy import stats
import numpy as np

import matplotlib.pyplot as plt


def C(a, b, b1, n):
    prod = 1
    for i in range(n):
        prod *= (b1 / (b[i])) ** (a[i])
    return prod


def dk(k, a, b, b1, n):
    if k == 0:
        return 1
    d_k = 0
    for i in np.arange(1, k + 1):
        d_k += i * g(i, a, b, b1, n) * dk(k - i, a, b, b1, n)

    d_k = (1 / k) * d_k
    return d_k


def g(k, a, b, b1, n):
    s = 0
    for i in np.arange(1, n + 1):
        s += (a[i - 1] * (1 - b1 / (b[i - 1])) ** k) / k
    return s


def rho(a):
    return sum(a)


def pdf(Y, a, b, b1, n):
    val = 0
    Cs = C(a, b, b1, n)
    for k in np.arange(0, 10):
        # print "dk: %.2f" % dk(k, a, b, b1, n)
        # print "Y**(ss):%.2f" % Y**(rho(a)+k-1)
        # print "exp: %.10f" % np.exp(-Y/b1)
        # print gamma(rho(a)+k)*b1**(rho(a)*k)
        val += (dk(k, a, b, b1, n) * Y ** (rho(a) + k - 1) * np.exp(-Y / b1)) / (gamma(rho(a) + k) * b1 ** (rho(a) + k))
    return Cs * val


def cdf(w, a, b, b1, n):
    val = 0
    Cs = C(a, b, b1, n)
    for k in np.arange(0, 10):
        val2 = 0.0
        step_size = 0.0001
        for dw in np.arange(0, w, step_size):
            val2 += (dw ** (rho(a) + k - 1) * np.exp(-dw / b1)) / (gamma(rho(a) + k) * b1 ** (rho(a) + k)) * step_size
        val += dk(k, a, b, b1, n) * val2

    return Cs * val


################ NOTE ###########
# theory is based in Y=X_1 + ..+ X_n
# multiply every alpha by its weight to obtain
# Y = w_1*X_1 + ... + w_n*X_n
# with w as the weights 
#
#	Goal is to reconstruct the gamma(1,1) distribution from weighted components
#	Then by re-weighting, i.e. to risk areas, we can construct a different gamma distribution such that the beta is smaller than with the original construction
#
#


# bs = [1,  1]
# b1 = min(bs)

# ass = [.5*1., .5*1.]

# x 	= np.linspace(1E-6, 5, 100)
# ys 	= [cdf(i, ass, bs, b1, len(bs)) for i in x]
# ys1 = [pdf(i, ass, bs, b1, len(bs)) for i in x]

# max_ys = max(ys1)

# ys1 = [i*1./max_ys for i in ys1]

# counter = 0
# for value in ys:
# 	if value > 0.96:
# 		print "case 1;For x: %.10f with cdf value: %.10f" %(x[counter], value)
# 		break;

# 	counter += 1

# # print "X: 3.2189 gives  CDF: %.10f" % cdf(3.2189, ass, bs, b1, len(bs))

# plt.plot(x, ys, x, np.repeat(0.96, len(x)), x, ys1, '-.')
# plt.title("Gamma CDF")
# plt.show()


print "One moderate risk area (50 percent of the total) and high risk area"

bs2 = [1. / 102, 1. / 130]
b12 = min(bs2)

ass = [.5 * 1., .5 * 2.]

x = np.linspace(1E-6, 0.2, 200)
ys = [cdf(i, ass, bs2, b12, len(bs2)) for i in x]
ys1 = [pdf(i, ass, bs2, b12, len(bs2)) for i in x]

max_ys = max(ys1)

ys1 = [i * 1. / max_ys for i in ys1]

counter = 0
for value in ys:
    if value > 0.96:
        print "case 2; For x: %.10f with cdf value: %.10f" % (x[counter], value)
        break

    counter += 1

# print "X: 3.2189 gives  CDF: %.10f" % cdf(3.2189, ass, bs, b1, len(bs))

plt.plot(x, ys, x, np.repeat(0.96, len(x)), x, ys1, '-.')
plt.title("Gamma CDF")
plt.show()

# dist1 = stats.gamma(1, loc=0, scale=1./255)
# ys1 = dist1.pdf(x) 

# plt.plot(x, ys, '-', x, ys1, '--')
# plt.legend(['PDF sum', 'pdf gamma'])
# plt.title("Y = X + X, with X ~ gamma(1,1) vs Y-gamma")
# plt.show()

# dist2 = stats.gamma(ass[1], 0, scale=1./bs[1])

# dist3 = stats.gamma(1,scale=1./1.0)
# dist4 = stats.gamma(1,scale=1./1.1)

# ys = [pdf(i, ass, bs, b1, len(bs)) for i in x]
# ys1 = dist1.pdf(x) 
# ys2 = dist2.pdf(x) 
# ys3 = dist3.pdf(x)
# ys4 = dist4.pdf(x)

# # plt.plot(x, ys3, x, ys4)
# # plt.legend(["b=1/1.0", "b=1/1.1"])
# # plt.show()

# plt.plot(x,ys, '-' , x, ys1, '--',  x, ys2, '-.', x, ys3, ':')
# plt.legend(["GT","G1","G2","G"])
# plt.show()
