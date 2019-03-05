import pstats
with open('stats.txt', 'w') as stream:
    pstats.Stats('profile.txt', stream=stream).strip_dirs().sort_stats("time").print_stats()
