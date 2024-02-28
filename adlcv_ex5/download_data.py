import os
import gdown
import zipfile
import shutil

os.makedirs('data', exist_ok=True)
data_url = 'https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG'
if not os.path.exists('data/nerf_synthetic.zip'):
    print('Downloading scences...')
    if not gdown.download(data_url, output='data/nerf_synthetic.zip', quiet=False):
        print("Download failed :(")
        print("Please download the file manually and place it as:")
        print("\tdata/nerf_synthetic.zip")
        print("and then re-reun this script.")
        exit()
print('Extracting scenes...')

with zipfile.ZipFile('data/nerf_synthetic.zip', 'r') as zip_file:
    zip_file.extractall('data/')

shutil.rmtree('data/__MACOSX')
shutil.rmtree('data/nerf_synthetic/hotdog')
shutil.rmtree('data/nerf_synthetic/materials')
os.remove('data/nerf_synthetic.zip')
print('Done.')