from pi_rating_gen import pi_rating_gen
from elo_gen import elo_gen
import sys


if __name__ == '__main__':

    if len(sys.argv) == 4:
        elo_gen(sys.argv[1], sys.argv[2])
        pi_rating_gen(sys.argv[1], sys.argv[3])

    else:
        print("Invalid arguments elo_gen.py [csvfile] [elorating json file name] [pirating json file name]")