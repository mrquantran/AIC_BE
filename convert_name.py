import json
import re
import os


def migrate_json(input_file, output_file):
    # Read the input JSON file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Create a new dictionary to store the migrated data
    migrated_data = {}

    # Regular expression to extract group, video, and frame number
    pattern = r"/L(\d+)/kaggle/working/L\d+/V(\d+)/(\d+)\.webp"

    for key, value in data.items():
        # Extract group, video, and frame number using regex
        match = re.search(pattern, value)
        if match:
            group = int(match.group(1))
            video = int(match.group(2))
            frame = int(match.group(3))

            # Create the new value in the format "group/video/frame"
            new_value = f"{group}/{video}/{frame}"

            # Add to the migrated data
            migrated_data[key] = new_value

    # Write the migrated data to the output JSON file
    with open(output_file, "w") as f:
        json.dump(migrated_data, f, indent=2)

    print(f"Migration complete. Output written to {output_file}")


# Example usage
input_file = os.path.join(os.path.dirname(__file__), "./global2imgpath_batch2.json")

output_file = "output_batch2.json"

migrate_json(input_file, output_file)
