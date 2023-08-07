import os
import h5py
import numpy as np
from PIL import Image

def extract_hdf5_item(name, item, output_folder):
    if isinstance(item, h5py.Dataset):
        if item.dtype == np.dtype('O'):
            output_path = os.path.join(output_folder, name.replace('/', '_') + '.txt')
            with open(output_path, 'wb') as f:
                f.write(item[:])
        elif item.dtype == np.dtype('uint8'):
            output_path = os.path.join(output_folder, name.replace('/', '_') + '.png')
            image = Image.fromarray(item[:])
            image.save(output_path)
    elif isinstance(item, h5py.Group):
        sub_folder = os.path.join(output_folder, name.replace('/', '_'))
        os.makedirs(sub_folder, exist_ok=True)
        item.visititems(lambda subname, subitem: extract_hdf5_item(subname, subitem, sub_folder))

def extract_hdf5_to_folder(file_path, output_folder):
    hdf5_file = h5py.File(file_path, 'r')
    hdf5_file.visititems(lambda name, item: extract_hdf5_item(name, item, output_folder))
    hdf5_file.close()

if __name__ == "__main__":
    file_path = 'path_to_your_file.h5'
    output_folder = 'output_folder'
    os.makedirs(output_folder, exist_ok=True)
    extract_hdf5_to_folder(file_path, output_folder)
