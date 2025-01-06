import tkinter as tk
from tkinter import filedialog
import c2pa
import json
import os


def extract_filename(file_path):
    # Use os.path.basename to get the last part of the path, which is the filename
    filename = os.path.basename(file_path)
    return filename


def remove_last_dot(input_string):
    if "." in input_string:
        # Find the last occurrence of dot in the string
        last_dot_index = input_string.rindex(".")

        # Remove the last dot and remaining characters
        result = input_string[:last_dot_index]
        return result
    else:
        return input_string


def keep_last_dot(input_string):
    if "." in input_string:
        # Find the last occurrence of dot in the string
        last_dot_index = input_string.rindex(".")

        # Keep the last dot and remaining characters
        result = input_string[last_dot_index:]
        return result
    else:
        return input_string


def manifest_binding(private_key_file, certificate_file,
                     manifest_file, source_image_file,
                     destination_folder, embedding_algorithm):
    source_image_file_name = extract_filename(source_image_file)
    print("C2PA version ==", c2pa.version())
    private_key = open(private_key_file, "rb").read()
    certificate = open(certificate_file, "rb").read()
    sign_info = c2pa.SignerInfo(embedding_algorithm, certificate, private_key, "http://timestamp.digicert.com")

    f = open(manifest_file)
    manifest_json = json.load(f)
    result = c2pa.sign_file(source=source_image_file,
                            dest=destination_folder + '/' + remove_last_dot(source_image_file_name)
                                 + "_manifest_bounded" + keep_last_dot(source_image_file_name),
                            manifest=json.dumps(manifest_json),
                            signer_info=sign_info,
                            data_dir=destination_folder)

    json_store = c2pa.read_file(
        destination_folder + '/' + remove_last_dot(source_image_file_name) + "_manifest_bounded" +
        keep_last_dot(source_image_file_name), destination_folder)
    with open(destination_folder + '/' + remove_last_dot(source_image_file_name) + "_manifest.json", 'w') as f:
        json.dump(json_store, f)


def on_manifest_binding_click():
    # Get values from the input fields
    private_key_file = private_key_entry.get()
    certificate_file = certificate_entry.get()
    manifest_file = manifest_entry.get()
    source_image_file = source_image_entry.get()
    destination_folder = destination_folder_entry.get()
    embedding_algorithm = embedding_algorithm_entry.get()

    manifest_binding(private_key_file, certificate_file, manifest_file, source_image_file, destination_folder,
                     embedding_algorithm)


def browse_file(entry_widget):
    file_path = filedialog.askopenfilename()
    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, file_path)


def browse_folder(entry_widget):
    folder_path = filedialog.askdirectory()
    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, folder_path)


# Create the main window
root = tk.Tk()
root.title("Manifest Binding Tool")

# Create and place input fields
# Create and place input fields for private key file
private_key_label = tk.Label(root, text="Private Key File:")
private_key_label.grid(row=0, column=0, sticky="e")
private_key_entry = tk.Entry(root, width=40)
private_key_entry.grid(row=0, column=1)
private_key_button = tk.Button(root, text="Browse", command=lambda: browse_file(private_key_entry))
private_key_button.grid(row=0, column=2)

# Create and place input fields for certificate file
certificate_label = tk.Label(root, text="Certificate File:")
certificate_label.grid(row=1, column=0, sticky="e")
certificate_entry = tk.Entry(root, width=40)
certificate_entry.grid(row=1, column=1)
certificate_button = tk.Button(root, text="Browse", command=lambda: browse_file(certificate_entry))
certificate_button.grid(row=1, column=2)

# Create and place input fields for manifest file
manifest_label = tk.Label(root, text="Manifest File:")
manifest_label.grid(row=2, column=0, sticky="e")
manifest_entry = tk.Entry(root, width=40)
manifest_entry.grid(row=2, column=1)
manifest_button = tk.Button(root, text="Browse", command=lambda: browse_file(manifest_entry))
manifest_button.grid(row=2, column=2)

# Create and place input fields for source image file
source_image_label = tk.Label(root, text="Source Image File:")
source_image_label.grid(row=3, column=0, sticky="e")
source_image_entry = tk.Entry(root, width=40)
source_image_entry.grid(row=3, column=1)
source_image_button = tk.Button(root, text="Browse", command=lambda: browse_file(source_image_entry))
source_image_button.grid(row=3, column=2)

# Create and place input fields for destination folder
destination_folder_label = tk.Label(root, text="Destination Folder:")
destination_folder_label.grid(row=4, column=0, sticky="e")
destination_folder_entry = tk.Entry(root, width=40)
destination_folder_entry.grid(row=4, column=1)
destination_folder_button = tk.Button(root, text="Browse", command=lambda: browse_folder(destination_folder_entry))
destination_folder_button.grid(row=4, column=2)

# Create and place the Embedding Algorithm input field
embedding_algorithm_label = tk.Label(root, text="Embedding Algorithm:")
embedding_algorithm_label.grid(row=5, column=0, sticky="e")
embedding_algorithm_entry = tk.Entry(root, width=40)
embedding_algorithm_entry.grid(row=5, column=1)

# Create and place the Manifest Binding button
manifest_binding_button = tk.Button(root, text="Manifest Binding", command=on_manifest_binding_click)
manifest_binding_button.grid(row=6, column=0, columnspan=3, pady=10)

# Create and place the result label
result_label = tk.Label(root, text="")
result_label.grid(row=7, column=0, columnspan=3)

# Start the main event loop
root.mainloop()
