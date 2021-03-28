import urllib.request, json
from urllib.error import HTTPError

import pandas as pd
from pandas.io.json import json_normalize


def json_parser(id):
    # print(id)
    api_link = "https://api.sofascore.com/api/v1/event/%d/statistics" % id
    try:
        with urllib.request.urlopen(api_link) as url:
            data = json.loads(url.read().decode())
            input = {}
            data = data['statistics'][0]['groups']
            for d in data:
                f = d['statisticsItems']
                for s in f:
                    name = s['name']
                    input.update({'home ' + name: s['home'], 'away ' +name: s['away']})

    except HTTPError:
        input = {}
        print("can't find " + str(api_link))
    return input



if __name__ == "__main__":
    # execute only if run as a script
    print(json_parser(3965805))
