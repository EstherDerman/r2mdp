import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=1, help='number of partial evaluations in MPI. Set k=0 for running a '
                                                         'VI algorithm, and big k (˜10) for PI')
    parser.add_argument('--theta', type=float, default=0.001, help="threshold error for convergence criterion")
    parser.add_argument('--discount_factor', type=float, default=0.9, help='discount factor')
    parser.add_argument('--alpha', type=float, default=1e-3, help='level uncertainty for reward model')
    parser.add_argument('--beta', type=float, default=1e-5, help='level uncertainty for transition model')
    parser.add_argument('--num_seeds', type=int, default=5, help='number of running seeds')
    options = parser.parse_args()
    return options
