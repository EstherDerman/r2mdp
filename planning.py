import numpy as np
from numpy import linalg as LA


def policy_eval(env, policy, theta, discount_factor, k=10000000, init=None):
    v = np.zeros(env.nS) if init is None else init
    cc = 0
    for i in range(k):
        value_fc = np.zeros(env.nS)
        for s in range(env.nS):
            r_pi = np.dot(policy[s, :], env.Rmat[s, :])
            pv = np.dot(env.Pmat[s, :, :].T, v)
            p_pi = np.dot(pv, policy[s, :])
            value_fc[s] = r_pi + discount_factor * p_pi
        delta = LA.norm(value_fc - v, np.inf)
        v[:] = value_fc
        cc += 1
        if delta < theta:
            break
    return v, cc


def modified_policy_iteration(env, k, theta, discount_factor):
    v = np.zeros(env.nS)
    threshold = (theta * (1 - discount_factor))/(2 * discount_factor)
    counter = 0
    while True:
        q = np.zeros([env.nS, env.nA])
        for a in range(env.nA):
            q[:, a] = env.Rmat[:, a] + discount_factor * np.dot(env.Pmat[:, :, a], v)
        greedy_v = np.max(q, -1)
        best_action = np.argmax(q, -1)
        policy = np.eye(env.nA)[best_action]
        # print(LA.norm(v - greedy_v, np.inf))
        if LA.norm(v - greedy_v, np.inf) <= threshold:
            return policy, greedy_v, counter
        else:
            # print(policy)
            v, cc = policy_eval(env, policy, theta=theta, discount_factor=discount_factor, k=k,  init=greedy_v)
            counter += 1
