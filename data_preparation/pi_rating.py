import math


def weighted_error(e):
    c = 3
    return c * math.log(1 + e, 10)


def pi_rating(Rah, Raa, Rbh, Rba, a_score, b_score):
    lambda_rate = 0.035
    gamma = 0.7
    b = 10
    c = 3

    goal_difference = a_score - b_score

    # Calculate the expected goal differences
    expected_goal_difference_a = pow(b, (math.fabs(Rah) / c)) - 1

    expected_goal_difference_b = (pow(b, (math.fabs(Rba) / c)) - 1)

    if Rba < 0 :
        expected_goal_difference_b = -expected_goal_difference_b

    # Calculate the expected goal difference
    predicted_goal_difference = expected_goal_difference_a - expected_goal_difference_b

    e = math.fabs(predicted_goal_difference - goal_difference)

    if predicted_goal_difference < goal_difference:
        weighted_e_a = weighted_error(e)
    else:
        weighted_e_a = -weighted_error(e)

    if predicted_goal_difference > goal_difference:
        weighted_e_b = weighted_error(e)
    else:
        weighted_e_b = -weighted_error(e)

        # update home team's rating
    Rah_new = Rah + weighted_e_a * lambda_rate
    Raa_new = Raa + (Rah_new - Rah) * gamma

    # update away team's rating
    Rba_new = Rba + weighted_e_b * lambda_rate
    Rbh_new = Rbh + (Rba_new - Rba) * gamma

    print("%f, %f, %f,%f" % (Rah_new, Raa_new, Rba_new, Rbh_new))

    return Rah_new, Raa_new, Rba_new, Rbh_new


if __name__ == "__main__":
    Rah, Raa, Rbh, Rba = pi_rating(1.6, 0.4, 0.3, 1.2, 4, 1)
    print(Rah)
    print(Raa)
    print(Rbh)
    print(Rba)
