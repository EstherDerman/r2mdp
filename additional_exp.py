import random
import numpy as np
import argparse
import time
import pickle

from my_envs.gridworld import GridWorldEnv
import planning
import robust_planning
import reg_planning
from config import get_args


args = get_args()
env = GridWorldEnv(shape=(5, 5), p=0.9)


print("---------- Policy evaluation: check-up ----------------------")
dic_PE = {'vanilla': []}

for i in range(args.num_seeds):
    np.random.seed(i)
    random.seed(i)
    print("Seed number: ", i)
    nums = np.random.randint(env.nA, size=env.nS)
    policy = np.ones((env.nS, env.nA)) * .25  # evaluate uniform policy

    start0 = time.monotonic()
    v0, cc0 = planning.policy_eval(env, policy, theta=args.theta, discount_factor=args.discount_factor)
    t0 = time.monotonic() - start0
    dic_PE['vanilla'].append(v0)

    for a in [0.]:  # [1e-2, 1e-3, 1e-5, 0.]:
        for b in [1e-2, 1e-3, 1e-5, 0.]:  # [0.]
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
file = open('PE_b_radii.pkl', 'wb')  # file = open('PE_a_radii.pkl', 'wb')
pickle.dump(dic_PE, file)
file.close()
