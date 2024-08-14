import os
import re

def rename_files_in_folder(folder_path):
    # Walk through each file in the directory and its subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is a .webp file
            if file.endswith(".webp"):
                # Remove leading zeros using regex
                new_file_name = re.sub(r"^0+", "", file)

                # Construct full file paths
                old_file_path = os.path.join(root, file)
                new_file_path = os.path.join(root, new_file_name)

                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {old_file_path} to {new_file_path}")


# Specify the root directory where you want to start renaming files
root_directory = os.path.join(os.path.dirname(__file__), "../data/images")
rename_files_in_folder(root_directory)
