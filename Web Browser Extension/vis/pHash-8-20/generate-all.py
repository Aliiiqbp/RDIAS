import os
import json


def add_all_key_to_json(directory):
    """
    Process all JSON files in the given directory that end with 'count.json',
    add an "All" key with aggregated values, and save the updated files
    with '_all.json' suffix.

    :param directory: Path to the directory containing JSON files.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('count.json'):
            # Construct the full file path
            file_path = os.path.join(directory, filename)

            # Read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Compute the "All" key
            all_key = {}
            for transformation, values in data.items():
                for key, value in values.items():
                    all_key[key] = all_key.get(key, 0) + value

            # Add the sorted "All" key to the data
            data["All"] = {key: all_key[key] for key in sorted(all_key, key=int)}

            # Save the updated JSON file with a new name
            new_filename = filename.replace('.json', '_all.json')
            new_file_path = os.path.join(directory, new_filename)
            with open(new_file_path, 'w') as new_file:
                json.dump(data, new_file, indent=4)

            print(f"Processed and saved: {new_file_path}")


# Example usage
# Replace '/path/to/directory' with the path to your directory containing the JSON files
for i in range(9, 21):
    add_all_key_to_json(str(i))
