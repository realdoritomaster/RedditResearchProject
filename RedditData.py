# https://api.pushshift.io

import requests
import json
import time
from functools import reduce

def get_pretty_print(json_object):
    return json.dumps(json_object, sort_keys=True, indent=4, separators=(',', ': '))

def write_to_file(file, content):
    f = open(file, "a")
    f.truncate(0); # Clear data.json file
    f.write(get_pretty_print(content))
    f.close()

    #open and read the file after the appending:
    f = open(file, "r")
    # print(f.read())

class RedditCollection:

    def get_pushshift_data(data_type, **kwargs):
        base_url = f"https://api.pushshift.io/reddit/search/{data_type}/"
        payload = kwargs

        try:
            request = requests.get(base_url, params=payload, timeout=10)
            delay = request.elapsed.total_seconds()
            print(delay)
            print(request.status_code)

            time.sleep(delay+1)
            request.raise_for_status()


            if request.status_code != 204:
                return request.json()
            return [{"data": ""}]

        except Exception as inst:
            print(inst)
            time.sleep(10)
            return False


    data_type="comment"     # request comments
    query="nasdaq||google||apple||tesla||microsoft"          # Add your query
    duration_increment=5          # timeframe in days
    size=250                 # maximum comments
    batch_size=250
    num_batches=1000
    sort_type="score"       # Sort by score
    aggs="subreddit"
    subreddit="stocks,wallstreetbets,investing"

    data = [
        {
        "data": []
        }
    ]

    batches_per_file = 200

    x=0
    while x < num_batches:
        temp_data = []
        temp_data = (get_pushshift_data(data_type=data_type,
                       q=query,
                       after=f"{duration_increment*(x+1)}d",
                       before=f"{0+(duration_increment*x)}d",
                       size=size,
                       sort_type=sort_type,
                       subreddit=subreddit)
                       )
        if (temp_data == False):
            continue
        print("batch #" + str(x+1))
        print("before: " + f"{0+(duration_increment*x)}d")
        print("after: " + f"{duration_increment*(x+1)}d")
        print()
        data[0]["data"].append(temp_data["data"])
        x+=1
        if x % batches_per_file == 0:
            write_to_file(f"data{batches_per_file/x}.json", data)
            data = [
                {
                "data": []
                }
            ]

    print("# of batches: " + str(len( [i for i in data[0]["data"]] ) ))
    print("# of comments: " + str(reduce(lambda count, l: count + len(l), data[0]["data"], 0)))


CollectR = RedditCollection()
CollectR()
