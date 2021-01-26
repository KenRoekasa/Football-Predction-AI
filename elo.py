def expectedProbability(Ra, Rb):
    c = 10
    d = 400
    return 1 / 1 + pow(c, ((Rb - Ra) / d))


def elo(Ra, Rb, a_score, b_score):
    K = 32

    Pb = expectedProbability(Ra, Rb)

    Pa = expectedProbability(Rb, Ra)

    if a_score > b_score:

        Ra = Ra + K * (1 - Pa)

        Rb = Rb + K * (0 - Pb)

    elif a_score == b_score:
        Ra = Ra + K * (0.5 - Pa)

        Rb = Rb + K * (0.5 - Pb)

    else:
        Ra = Ra + K * (0 - Pa)

        Rb = Rb + K * (1 - Pb)

    print("%d , %d" % (Ra, Rb))


elo(1000, 500, 3, 1)
elo(500, 1000, 1, 3)
