
def print_policy(pi):
    actions_display = {3: '\u2190', 2: '\u2193', 1: '\u2192', 0: '\u2191'}
    n = len(pi)
    for row in range(n):
        for col in range(n):
            print(actions_display[pi[row, col]], end=' ')
        print()
    return
