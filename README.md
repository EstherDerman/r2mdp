# R2-MPI

Implementation of the DP algorithms from the NeurIPS 2021 paper:
**Twice regularized MDPs and the equivalence between
robustness and regularization**

The code is run on a 5x5 grid, where the agent has to reach a goal state to get maximal reward.
The file requirements.txt contains the modules necessary for running the code. 
These may be installed using "pip install". 

## Repository structure
The environment is in my_envs directory. 

Directory utils contains a method visualizing a policy over the grid.

Planning algorithms are in their respective scripts:
- planning - contains standard, non-regularized policy evaluation and MPI
- robust_planning - contains robust policy evaluation and robust-MPI
- reg_planning - contains R2-policy evaluation and R2-MPI

The script 'main' executes all planning methods and outputs optimal value function and policy for each approach.
Script 'additional_exp' executes additional experiments from the Appendix. 

## Further details

Parameters are set through the script 'config'. 
For ease of implementation, only one parameter alpha/beta 
determines the size of the uncertainty set (resp. the regularization level). 

This means that for each state-action pair, we set the same uncertainty level. 
As such, the uncertainty set is (s,a) rectangular and
it suffices to search over deterministic policies for both robust and R2-MPI. 