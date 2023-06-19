import os
import importlib
import cv2
import numpy as np
import glob
import argparse
import random
import shutil

__import__("ffmpeg_extract")
from ffmpeg_extract import extractf

parser = argparse.ArgumentParser()
parser.add_argument("--movie_name", required=True, help="Filename of the Movie")
args = parser.parse_args()

movie_name = args.movie_name

#create frame folder
os.makedirs("frames", exist_ok=True)
os.makedirs("canny", exist_ok=True)
os.makedirs("merged", exist_ok=True)
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)
os.makedirs("validate", exist_ok=True)


folder = "frames"
folder = folder + "/"
extractf(movie_name, folder)

def auto_canny(image, sigma=0.33):
    # Compute the median of the single channel pixel intensities
    v = np.median(image)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

# Read in each image in a list
images = [cv2.imread(file, 1) for file in glob.glob(folder+"/*.png")]


# Iterate through each image, perform edge detection, and save image
number = 0
for image in images:

    rgb = image
#    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = auto_canny(gray)
    cv2.imwrite('canny/canny_{}.png'.format(number), canny)
    #convert to same color space, otherwise concat will not work
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    im_h = cv2.hconcat([rgb, canny])
    cv2.imwrite('merged/merged_{}.png'.format(number), im_h)
    number += 1

# list of merged files
merged_files = glob.glob("merged/*.png")

for i in random.sample(merged_files, round(len(merged_files)*0.8)):
    #copy 80% to validation 
    shutil.copy(i,"validate/")

for i in random.sample(merged_files, round(len(merged_files)*0.8)):
    #copy 80% to train 
    shutil.move(i,"train/")

#move 20% of the folder to /test
merged_files = glob.glob("merged/*.png")

for i in merged_files:
    #copy 80% to train 
    shutil.move(i,"test/")

shutil.rmtree("frames")
shutil.rmtree("canny")
shutil.rmtree("merged")

