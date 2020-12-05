import urllib.request, json


def json_parser(id):
    api_link = "https://api.sofascore.com/api/v1/event/%d/statistics" % id
    with urllib.request.urlopen(api_link) as url:
        data = json.loads(url.read().decode())
        data = data['statistics'][0]['groups']
        possession = data[0]['statisticsItems']
        possession_arr = []
        for p in possession:
            possession_arr.append(p['home'])
            possession_arr.append(p['away'])
        print("possession_arr " + str(possession_arr))
        shots = data[1]['statisticsItems']
        shot_data = []
        for s in shots:
            shot_data.append(s['home'])
            shot_data.append(s['away'])
        print("shot data " + str(shot_data))
        tvdata = data[2]['statisticsItems']
        tv_data_arr = []
        for tv in tvdata:
            tv_data_arr.append(tv['home'])
            tv_data_arr.append(tv['away'])
        print("tv_data_arr data " + str(tv_data_arr))
        shots_extra = data[3]['statisticsItems']
        shots_extra_arr = []
        for s in shots_extra:
            shots_extra_arr.append(s['home'])
            shots_extra_arr.append(s['away'])
        print("shots_extra_arr data " + str(shots_extra_arr))
        passes = data[4]['statisticsItems']
        passes_data = []
        for p in passes:
            passes_data.append(p['home'])
            passes_data.append(p['away'])
        print("passes_data data " + str(passes_data))

        duels = data[5]['statisticsItems']
        duels_data = []
        for d in duels:
            duels_data.append(d['home'])
            duels_data.append(d['away'])
        print("duels_data data " + str(duels_data))

        defending = data[6]['statisticsItems']
        defending_arr = []
        for d in defending:
            defending_arr.append(d['home'])
            defending_arr.append(d['away'])
        print("defending_arr data " + str(defending_arr))

        csv_data = possession_arr + shot_data + tv_data_arr + shots_extra_arr + passes_data + duels_data + defending_arr
        return csv_data
