import pickle
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

file = open("PE_a_radii.pkl",'rb')
dic_a = pickle.load(file)
file.close()

file = open("PE_b_radii.pkl",'rb')
dic_b = pickle.load(file)
file.close()

v_vanilla = np.mean(np.array(dic_a['vanilla']), axis=0)
# print(np.std(np.array(dic_a['vanilla']), axis=0))

d = {}
for key in dic_a.keys():
    # print(key)
    vec = np.array(dic_a[key])
    dist = []
    for v in vec:
        dist.append(LA.norm(v-v_vanilla))
    d[key] = [np.mean(dist), np.std(dist)]
print(d)

d = {}
for key in dic_b.keys():
    if key not in ['vanilla']:
        # print(key)
        vec = np.array(dic_b[key])
        dist = []
        for v in vec:
            dist.append(LA.norm(v-v_vanilla))
        d[key] = [np.mean(dist), np.std(dist)]
print(d)


x = [.01, .001, 1e-5, 0.]

# y_r2 = [.24, .024, 2.4e-4, 0]
# e_r2 = [0., 0., 0., 0]

y_r2 = [7.5, .8, 8e-3, 0]
e_r2 = [0., 0., 0., 0]

y_robust = [1.8, .18, 1e-3, 0]
e_robust = e_r2

fontsize = 16
plt.xlabel("Radius of transition uncertainty set", fontsize=fontsize)
plt.ylabel("Distance to the optimal vanilla value function", fontsize=fontsize)
# plt.xticks(x, ["3x3", "5x5", "10x10", "100x100"], fontsize=fontsize)
plt.errorbar(x, y_r2, yerr=e_r2, linestyle='-', fmt = ".k", ecolor='g', capthick=1, capsize = 5, label='R2 MPI', color = 'g')
plt.errorbar(x, y_robust, yerr=e_robust, linestyle='--', fmt = ".k", ecolor='r', capthick=2, capsize = 5, label='Robust MPI', color='r')
plt.ylim(0, max(y_r2)+ .2)
plt.yticks(fontsize=fontsize)
plt.legend()
plt.show()




