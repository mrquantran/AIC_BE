import time
import json
import random
import os

# Generate test data
data = [{"id": i, "value": random.random()} for i in range(1000000)]

def write_json():
    start = time.time()
    with open("data.json", "w") as f:
        # write something
        json.dump(data, f)
    
    end = time.time()
    json_time = end - start

    return json_time

if __name__ == "__main__":
    json_time = write_json()