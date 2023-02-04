# importing the required module
import os
import shutil
import clustering


# assign directory
directory = 'testPatient'

list_images = []
list_filename = []

if os.path.exists('Slices') and os.path.isdir('Slices'):
    shutil.rmtree('Slices')
os.makedirs('Slices')

if os.path.exists('Clusters') and os.path.isdir('Clusters'):
    shutil.rmtree('Clusters')
os.makedirs('Clusters')

for filename in os.listdir(directory):
    # checking if it is a file
    if filename.endswith("_thresh.png"):
        list_images.append(filename)

for item in list_images:
    list_filename.append(item.split('.')[0])

for item in list_filename:
    clustering.brain_slice_extraction(item, directory)

print("====== COMPLETED ======")



