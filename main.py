import random
import numpy as np
import argparse
import time
import pickle

from my_envs.gridworld import GridWorldEnv
from utils import utils
import planning
import robust_planning
import reg_planning


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=4, help='number of partial evaluations in MPI. Set k=0 for running a '
                                                         'VI algorithm, and big k (˜10) for PI')
    parser.add_argument('--theta', type=float, default=0.001, help="threshold error for convergence criterion")
    parser.add_argument('--discount_factor', type=float, default=0.9, help='discount factor')
    parser.add_argument('--alpha', type=float, default=1e-3, help='level uncertainty for reward model')
    parser.add_argument('--beta', type=float, default=1e-5, help='level uncertainty for transition model')
    options = parser.parse_args()
    return options


args = get_args()
num_seeds = 5
env = GridWorldEnv(shape=(5, 5), p=0.9)


print("---------- Policy evaluation: check-up ----------------------")
dic_PE = {'vanilla': []}
for i in range(num_seeds):
    np.random.seed(i)
    random.seed(i)
    print("Seed number: ", i)
    nums = np.random.randint(env.nA, size=env.nS)
    policy = np.ones((env.nS, env.nA)) * .25  # evaluate uniform policy

    # start0 = time.monotonic()
    # v0, cc0 = planning.policy_eval(env, policy, theta=args.theta, discount_factor=args.discount_factor)
    # t0 = time.monotonic() - start0
    # dic_PE['vanilla'].append(v0)

    for a in [0.]:
        for b in [1e-2, 1e-3, 1e-5, 0.]:
            args.alpha = a
            args.beta = b
            for k in ['r2_', 'robust_']:
                dic_PE[k + 'a_' + str(a) + 'b_' + str(b)] = []

            start1 = time.monotonic()
            v1, cc1 = reg_planning.reg_policy_eval(env, policy, theta=args.theta, discount_factor=args.discount_factor,
                                                   alpha=args.alpha, beta=args.beta)
            t1 = time.monotonic()-start1
            dic_PE['r2_a_' + str(a) + 'b_' + str(b)].append(v1)

            start2 = time.monotonic()
            v2, cc2 = robust_planning.robust_policy_eval(env, policy, theta=args.theta, discount_factor=args.discount_factor,
                                                   alpha=args.alpha, beta=args.beta)
            t2 = time.monotonic()-start2
            dic_PE['robust_a_' + str(a) + 'b_' + str(b)].append(v2)

print(dic_PE)
file = open('PE_b_radii.pkl', 'wb')
pickle.dump(dic_PE, file)
file.close()

#
# dic_MPI = {'vanilla': [], 'r2': [], 'robust': [], 'robust_iter': []}
# print("---------- MPI, R2-MPI and robust MPI ----------------------")
# for i in range(num_seeds):
#     np.random.seed(i)
#     random.seed(i)
#
#     env = GridWorldEnv(shape=(5, 5), p=0.9)
#
#     print("Seed number: ", i)
#     start0 = time.monotonic()
#     policy0, v0, iters0 = planning.modified_policy_iteration(env, k=args.k, theta=args.theta, discount_factor=args.discount_factor)
#     t0 = time.monotonic() - start0
#     dic_MPI['vanilla'].append(t0)
#     # print("MPI [p=0.9] converges in ", iters0, " iters and ", time.monotonic()-start0, " seconds")
#     # print(v0)
#     # p0 = np.reshape(np.argmax(policy0, axis=1), env.shape)
#     # utils.print_policy(p0)
#     # print("-----------------------------------------------")
#
#     start1 = time.monotonic()
#     policy1, v1, iters1 = reg_planning.reg_modified_policy_iteration(env, k=args.k, theta=args.theta, discount_factor=args.discount_factor,
#                                                          alpha=args.alpha, beta=args.beta)
#     t1 = time.monotonic() - start1
#     dic_MPI['r2'].append(t1)
#     # print("R2-MPI [p=0.9] converges in ", iters1, " iters and ", time.monotonic()-start1, " seconds")
#     # print(v1)
#     # p1 = np.reshape(np.argmax(policy1, axis=1), env.shape)
#     # utils.print_policy(p1)
#     # print("-----------------------------------------------")
#     #
#     start2 = time.monotonic()
#     policy2, v2, iters2 = robust_planning.robust_modified_policy_iteration(env, k=args.k, theta=args.theta,
#                                                                discount_factor=args.discount_factor,
#                                                                alpha=args.alpha, beta=args.beta)
#     t2 = time.monotonic() - start2
#     dic_MPI['robust'].append(t2)
#     dic_MPI['robust_iter'].append(iters2)
#     # print("Robust-MPI [p=0.9] converges in ", iters2, " iters and ", time.monotonic()-start2, " seconds")
#     # print(v2)
#     # p2 = np.reshape(np.argmax(policy2, axis=1), env.shape)
#     # utils.print_policy(p2)
#     # print("-----------------------------------------------")
# print(dic_MPI)
# print(dic_MPI['robust_iter'])
# file = open('MPI_m=4.pkl', 'wb')
# pickle.dump(dic_MPI, file)
# file.close()
