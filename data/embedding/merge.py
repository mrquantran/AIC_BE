import json
import os

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Function to merge two dictionaries
def merge_json(json1, json2):
    merged = {**json1, **json2}
    return merged


# Function to save the merged JSON data to a file
def save_json(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    # Load the two JSON files
    # json1 = load_json("./global2imgpath.json")
    json1 = load_json(os.path.join(os.path.dirname(__file__), "global2imgpath.json"))
    json2 = load_json(os.path.join(os.path.dirname(__file__), "global2imgpath_batch2.json"))

    # Merge the JSON data
    merged_json = merge_json(json1, json2)

    # Save the merged JSON to a new file
    save_json("merged_json.json", merged_json)

    print("JSON files merged successfully into 'merged_json.json'.")
