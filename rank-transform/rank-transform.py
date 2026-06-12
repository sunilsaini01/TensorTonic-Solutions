def rank_transform(values):
    """
    Replace each value with its average rank.
    """
    x = sorted(values)

    rank_map = {}

    i = 0
    n = len(x)

    while i < n:

        j = i

        while j < n and x[j] == x[i]:
            j += 1

        avg_rank = ((i + 1) + j) / 2.0

        rank_map[x[i]] = avg_rank

        i = j

    return [float(rank_map[x]) for x in values]
    