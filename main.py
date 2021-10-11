import random
import numpy as np
import time
import pickle

from my_envs.gridworld import GridWorldEnv
from utils import utils
import planning
import robust_planning
import reg_planning
from config import get_args


args = get_args()
env = GridWorldEnv(shape=(5, 5), p=0.9)

dic_PE = {'vanilla': [], 'r2': [], 'robust': [], 'robust_iter': []}
dic_MPI = {'vanilla': [], 'r2': [], 'robust': [], 'robust_iter': []}


for i in range(args.num_seeds):
    running_seed = "Seed number " + str(i)

    print("---------- Policy evaluation: check-up ----------------------", running_seed)
    np.random.seed(i)
    random.seed(i)
    nums = np.random.randint(env.nA, size=env.nS)
    policy = np.ones((env.nS, env.nA)) * .25  # evaluate uniform policy

    start0 = time.monotonic()
    v0, cc0 = planning.policy_eval(env, policy, theta=args.theta, discount_factor=args.discount_factor)
    t0 = time.monotonic() - start0
    dic_PE['vanilla'].append(t0)

    start1 = time.monotonic()
    v1, cc1 = reg_planning.reg_policy_eval(env, policy, theta=args.theta, discount_factor=args.discount_factor,
                                           alpha=args.alpha, beta=args.beta)
    dic_PE['r2'].append(t0)

    start2 = time.monotonic()
    v2, cc2 = robust_planning.robust_policy_eval(env, policy, theta=args.theta, discount_factor=args.discount_factor,
                                           alpha=args.alpha, beta=args.beta)
    dic_PE['robust'].append(t0)

    print("---------- MPI, R2-MPI and robust MPI ----------------------", running_seed)
    start0 = time.monotonic()
    policy0, v0, iters0 = planning.modified_policy_iteration(env, k=args.k, theta=args.theta, discount_factor=args.discount_factor)
    t0 = time.monotonic() - start0
    dic_MPI['vanilla'].append(t0)
    print("MPI [p=0.9] converges in ", iters0, " iters and ", t0, " seconds to optimal value ", v0)

    p0 = np.reshape(np.argmax(policy0, axis=1), env.shape)
    utils.print_policy(p0)
    print("-----------------------------------------------")

    start1 = time.monotonic()
    policy1, v1, iters1 = reg_planning.reg_modified_policy_iteration(env, k=args.k, theta=args.theta, discount_factor=args.discount_factor,
                                                                     alpha=args.alpha, beta=args.beta)
    t1 = time.monotonic() - start1
    dic_MPI['r2'].append(t1)
    print("R2-MPI [p=0.9] converges in ", iters1, " iters and ", t1, " seconds to optimal value ", v1)
    p1 = np.reshape(np.argmax(policy1, axis=1), env.shape)
    utils.print_policy(p1)
    print("-----------------------------------------------")

    start2 = time.monotonic()
    policy2, v2, iters2 = robust_planning.robust_modified_policy_iteration(env, k=args.k, theta=args.theta,
                                                                           discount_factor=args.discount_factor,
                                                                           alpha=args.alpha, beta=args.beta)
    t2 = time.monotonic() - start2
    dic_MPI['robust'].append(t2)
    print("Robust-MPI [p=0.9] converges in ", iters2, " iters and ", t2, " seconds to optimal value ", v2)

    p2 = np.reshape(np.argmax(policy2, axis=1), env.shape)
    utils.print_policy(p2)
    print("-----------------------------------------------")

dic = {'PE': dic_PE, 'MPI': dic_MPI}
file = open('output.pkl', 'wb')
pickle.dump(dic, file)
file.close()
