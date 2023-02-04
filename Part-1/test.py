# importing the required module
import os
import shutil
import brainExtraction


# assign directory
directory = 'testPatient'

list_images = []
list_filename = []

if os.path.exists('Boundaries') and os.path.isdir('Boundaries'):
    shutil.rmtree('Boundaries')
os.makedirs('Boundaries')

if os.path.exists('Slices') and os.path.isdir('Slices'):
    shutil.rmtree('Slices')
os.makedirs('Slices')

for filename in os.listdir(directory):
    # checking if it is a file
    if filename.endswith("_thresh.png"):
        list_images.append(filename)

for item in list_images:
    list_filename.append(item.split('.')[0])

for item in list_filename:
    brainExtraction.brain_slice_extraction(item, directory)
    brainExtraction.brain_boundary_extraction(item)


print("====== COMPLETED ======")



