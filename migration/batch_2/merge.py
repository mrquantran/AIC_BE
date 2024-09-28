import json
import os

# Function to read a JSON file from the provided filepath
def read_json(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {filepath} is not a valid JSON file.")
        return None


# Function to write a dictionary to a JSON file
def write_json(filepath, data):
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"File saved to {filepath}")
    except Exception as e:
        print(f"Error saving file {filepath}: {e}")


# Function to convert indices in JSON 2 and merge with JSON 1
def merge_jsons(json1, json2, output_filepath):
    # Get the last index from JSON 1 and convert it to an integer
    last_index_json1 = int(list(json1.keys())[-1])

    # Create a new dictionary for json_2_converted
    json_2_converted = {}

    # Adjust indices in JSON 2 to start after the last index of JSON 1
    for key, value in json2.items():
        new_index = (
            last_index_json1 + 1 + int(key)
        )  # Convert key to int and shift the index
        json_2_converted[str(new_index)] = value

    # Merge JSON 1 and json_2_converted
    merged_json = {**json1, **json_2_converted}

    # Write the merged dictionary to a new file
    write_json(output_filepath, merged_json)

    # print the number of items in the merged dictionary
    print(f"Number of items in JSON 1: {len(json1)}")
    print(f"Number of items in JSON 2: {len(json2)}")
    print(f"Number of items in the merged dictionary: {len(merged_json)} = {len(json1)} + {len(json2)} = {len(json1) + len(json2)}")
    print("Merging completed successfully.")


# Main function to read files and call the merging function
def main():

    # Filepaths (you can adjust these paths as needed)
    json1_filepath = os.path.join(
        os.path.dirname(__file__), "inference_results_numbered.json"
    )
    json2_filepath = os.path.join(
        os.path.dirname(__file__), "inference_results_numbered_part_2.json"
    )
    output_filepath = os.path.join(
        os.path.dirname(__file__), "inference_results_numbered_output.json"
    )

    # Step 1: Read JSON 1
    json1 = read_json(json1_filepath)
    if json1 is None:
        return

    # Step 2: Read JSON 2
    json2 = read_json(json2_filepath)
    if json2 is None:
        return

    # Step 3-5: Merge JSON 1 and JSON 2, and save the result
    merge_jsons(json1, json2, output_filepath)


# Run the main function
if __name__ == "__main__":
    main()
