import os
import requests

classes = ['cat', 'house', 'tree']
base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
save_path = "data/quickdraw_data"

os.makedirs(save_path, exist_ok=True)

for cls in classes:
    file_name = f"{cls}.npy"
    url = base_url + file_name
    file_path = os.path.join(save_path, file_name)

    if not os.path.exists(file_path):
        print(f"Downloading {file_name}...")
        r = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(r.content)
    else:
        print(f"{file_name} already exists.")
