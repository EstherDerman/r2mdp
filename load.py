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


x = [.01, .001, 1e-5, 0.]  # radius

y_r2_a = [.24, .024, 2.4e-4, 0]  # distance norm to nominal for each run
e_r2_a = [0., 0., 0., 0]  # std of distance norm
y_robust_a = [.24, .024, 2.4e-4, 0]
e_robust_a = [0., 0., 0., 0]


y_r2_b = [7.5, .8, 8e-3, 0]
e_r2_b = [0., 0., 0., 0]
y_robust_b = [1.8, .18, 1e-3, 0]
e_robust_b = e_r2_b

font_size = 16

plt.xlabel("Radius of reward uncertainty set", fontsize=font_size)
plt.ylabel("Distance to the optimal vanilla value function", fontsize=font_size)
plt.errorbar(x, y_r2_a, yerr=e_r2_a, linestyle='-', fmt = ".k", ecolor='g', capthick=1, capsize = 5, label='R2 MPI', color = 'g')
plt.errorbar(x, y_robust_a, yerr=e_robust_a, linestyle='--', fmt = ".k", ecolor='r', capthick=2, capsize = 5, label='Robust MPI', color='r')
plt.ylim(0, max(y_r2_a) + .2)
plt.yticks(fontsize=font_size)
plt.legend()
plt.show()

plt.xlabel("Radius of transition uncertainty set", fontsize=font_size)
plt.ylabel("Distance to the optimal vanilla value function", fontsize=font_size)
plt.errorbar(x, y_r2_b, yerr=e_r2_b, linestyle='-', fmt = ".k", ecolor='g', capthick=1, capsize = 5, label='R2 MPI', color = 'g')
plt.errorbar(x, y_robust_b, yerr=e_robust_b, linestyle='--', fmt = ".k", ecolor='r', capthick=2, capsize = 5, label='Robust MPI', color='r')
plt.ylim(0, max(y_r2_b)+ .2)
plt.yticks(fontsize=font_size)
plt.legend()
plt.show()




