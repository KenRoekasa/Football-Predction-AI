
def expected_probability(Ra, Rb):
    c = 10
    d = 400
    return 1 / (1 + pow(c, ((Rb - Ra) / d)))


def elo(Ra, Rb, a_score, b_score):
    K = 10

    Pb = expected_probability(Ra, Rb)

    Pa = expected_probability(Rb, Ra)

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

    return round(Ra), round(Rb)


if __name__ == "__main__":
    Ra, Rb = elo(400, 100, 3, 3)
    print(Ra)
