import urllib.request
import zipfile
import os
from tqdm import tqdm

# Download the GloVe word vectors
url = "http://nlp.stanford.edu/data/glove.6B.zip"
output = "glove.6B.zip"
def download_progress(block_num, block_size, total_size):
    progress = block_num * block_size / total_size * 100
    print(f"\rDownloading: {progress:.2f}%", end='')

urllib.request.urlretrieve(url, output, reporthook=download_progress)

# Unzip the file
# Add a progress bar to the unzip process
class ZipFileWithProgress(zipfile.ZipFile):
    def extractall(self, path=None, members=None, pwd=None):
        if members is None:
            members = self.namelist()
        total = len(members)
        
        with tqdm(total=total, unit='file') as pbar:
            for member in members:
                self.extract(member, path, pwd)
                pbar.update(1)

# Use the custom ZipFileWithProgress class
with ZipFileWithProgress(output, 'r') as zip_ref:
    zip_ref.extractall("glove.6B")

# List the extracted files
print(os.listdir("glove.6B"))