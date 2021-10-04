import numpy as np
from numpy import linalg as LA
import picos as pic


def robust_policy_eval(env, policy, theta, discount_factor, alpha, beta, k=10000000, init=None):
    v = np.zeros(env.nS) if init is None else init
    cc = 0
    for i in range(k):
        value_fc = np.zeros(env.nS)
        for s in range(env.nS):
            o = pic.Problem()
            r = o.add_variable("r", env.nA, vtype='continuous')
            p = o.add_variable("p", (env.nA, env.nS), vtype='continuous')
            o.add_constraint(p * ([1]*env.nS) == 1)
            o.add_constraint(p >= 0)
            o.add_constraint(pic.norm(r - env.Rmat[s, :]) <= alpha)
            o.add_constraint(pic.norm(p - env.Pmat[s, :, :].T) <= beta)
            o.set_objective("min", (policy[s, :] | r) + discount_factor * (p * v | policy[s, :]))
            o.solve(verbose=0)
            value_fc[s] = o.obj_value()
            r2_val = np.dot(policy[s, :], env.Rmat[s, :]) \
            + discount_factor * np.dot(np.dot(env.Pmat[s, :, :].T, v), policy[s, :]) \
            - alpha * LA.norm(policy[s, :]) - beta * LA.norm(policy[s, :]) * discount_factor * LA.norm(v)
            # print("Robust at s: ", value_fc[s], " -- R2 at s: ", r2_val)
        delta = LA.norm(value_fc - v, np.inf)
        v[:] = value_fc
        cc += 1
        if delta < theta:
            break
    return v, cc


def robust_modified_policy_iteration(env, k, theta, discount_factor, alpha, beta):
    v = np.zeros(env.nS)
    threshold = (theta * (1 - discount_factor)) / (2 * discount_factor)
    counter = 0
    while True:
        q = np.zeros([env.nS, env.nA])
        for a in range(env.nA):
            for s in range(env.nS):
                o = pic.Problem()
                r = o.add_variable("r", 1, vtype='continuous')
                p = o.add_variable("p", int(env.nS), vtype='continuous')
                o.add_constraint(pic.sum(p) == 1)
                o.add_constraint(p >= 0)
                o.add_constraint(pic.norm(r - env.Rmat[s, a]) <= alpha)
                o.add_constraint(pic.norm(p - env.Pmat[s, :, a]) <= beta)
                o.set_objective("min",  r + discount_factor * (p | v))
                o.solve(verbose=0)
                q[s, a] = o.obj_value()
        greedy_v = np.max(q, -1)
        best_action = np.argmax(q, -1)
        policy = np.eye(env.nA)[best_action]
        # print(LA.norm(v - greedy_v, np.inf))
        if LA.norm(v - greedy_v, np.inf) <= threshold:
            return policy, greedy_v, counter
        else:
            v, cc = robust_policy_eval(env, policy, theta=theta, discount_factor=discount_factor,
                                       alpha=alpha, beta=beta, k=k, init=greedy_v)
            counter += 1


#
# def robust_modified_policy_iteration(env, k=1, theta=0.001, discount_factor=0.9, alpha=.02, beta=.01):
#     policy_stable = False
#     np.random.seed(1)
#     nums = np.random.randint(env.nA, size=env.nS)
#     policy = np.eye(env.nA)[nums]
#     v_true, _ = robust_policy_eval(env, policy, theta=theta, discount_factor=discount_factor, alpha=alpha, beta=beta)
#     v, cc = robust_policy_eval(env, policy, k=k, theta=theta, discount_factor=discount_factor, alpha=alpha, beta=beta)
#     iters = cc
#     iters_list = [iters]
#     # bot_left = [v_true[env.nS-env.shape[1]]]
#     # bot_right = [v_true[env.nS-1]]
#     while True:
#         chosen_action = np.argmax(policy, -1)
#         q = np.zeros([env.nS, env.nA])
#         for a in range(env.nA):
#             for s in range(env.nS):
#                 o = pic.Problem()
#                 r = o.add_variable("r", 1, vtype='continuous')
#                 # print(env.nS, type(int(env.nS)))
#                 p = o.add_variable("p", int(env.nS), vtype='continuous')
#                 o.add_constraint(pic.sum(p) == 1)
#                 o.add_constraint(p >= 0)
#                 o.add_constraint(pic.norm(r - env.Rmat[s, a]) <= alpha)
#                 o.add_constraint(pic.norm(p - env.Pmat[s, :, a]) <= beta)
#                 o.set_objective("min",  r + discount_factor * (p | v))
#                 o.solve(verbose=0)
#                 q[s, a] = o.obj_value()
#         best_action = np.argmax(q, -1)
#         if np.all(chosen_action == best_action):
#             policy_stable = True
#         policy[:] = np.eye(env.nA)[best_action]
#         v, cc = robust_policy_eval(env, policy, k=k, theta=theta, discount_factor=discount_factor, init=v,
#                                    alpha=alpha, beta=beta)
#         v_true, _ = robust_policy_eval(env, policy, theta=theta, discount_factor=discount_factor, init=v_true,
#                                        alpha=alpha, beta=beta)
#         # bot_left.append(v_true[env.nS-env.shape[1]])
#         # bot_right.append(v_true[env.nS-1])
#         iters += cc + 1
#         iters_list.append(iters)
#         if policy_stable:
#             return policy, v_true, iters_list
#                 # , bot_left, bot_right
