import os
import subprocess
import importlib
import cv2
import numpy as np
import glob
import argparse
import shutil
from PIL import Image
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--movie_name", required=True, help="Filename of the Movie")
args = parser.parse_args()

movie_name = args.movie_name
movie_name = str(movie_name)



generator = tf.keras.models.load_model(r"generator_model.h5", compile=False)



#create folders
os.makedirs("frames", exist_ok=True)
os.makedirs("movie_canny", exist_ok=True)
os.makedirs("predict", exist_ok=True)
os.makedirs("scale", exist_ok=True)

###########
## EXTRACT FRAMES 

#get info from video
def movie_info(movie_name):
    cap = cv2.VideoCapture(movie_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    crop = width - height
    extractf(movie_name)
    return

  
## ffmpeg movie extract to frames

def extractf(movie_name):  
    command = "ffmpeg -i {movie_name}  -r 30 frames/frames_%05d.png".format(movie_name=movie_name)
    subprocess.call(command,shell=True)
    print(movie_name+" extracted")
    return

###########
### CANNY

def auto_canny(image, sigma=0.33):
    # Compute the median of the single channel pixel intensities
    v = np.median(image)
    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

# Read in each image in a list

def batch_canny():
    for image in glob.glob("frames/*.png"):
        filename = os.path.basename(image)
        print("canny detection", filename)
        img = cv2.imread(image, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny = auto_canny(gray)
        canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
        cv2.imwrite('movie_canny/{}'.format(filename), canny)
    return

def batch_generator():
    for inp in glob.glob("movie_canny/*.png"):
        filename = os.path.basename(inp)

        #load picture and build compatible array 
        inpic = tf.io.read_file(inp)
        inpic = tf.image.decode_jpeg(inpic)
        inpic = tf.cast(inpic, tf.float32)
        inpic = (inpic / 127.5) - 1
        inpic = tf.expand_dims(inpic, axis=0)
        #generate
        img = generator(inpic)
        #write image
        img = tf.squeeze(img)
        pil_img = tf.keras.preprocessing.image.array_to_img(img)
        filename = os.path.splitext(filename)[0]+".jpg"
        print("GAN prediction", filename)
        pil_img.save('predict/'+filename,'JPEG')
    return

####
##SUPER RES

def superres():
    for inp in glob.glob(r'\*.jpg'):
        filename = os.path.basename(inp)
        img = Image.open(inp)
        img = img.resize(size=(img.size[0]*2, img.size[1]*2), resample=Image.BICUBIC)
        lr_img = np.array(img)
        rdn = RDN(weights='noise-cancel')
        sr_img = rdn.predict(lr_img)
        pic = Image.fromarray(sr_img)
        pic.save(r'\'+filename,'JPEG')
    return

## ffmpeg movie extract to frames
def demo_movie(movie_name):  
    command = "ffmpeg -i predict/frames_%05d.jpg -c:v libx264 -r 30 preview_{movie_name}".format(movie_name=movie_name)
    subprocess.call(command,shell=True)
    return
    

def main():
    movie_info(movie_name)
    batch_canny()
    batch_generator()
#    superres()
    demo_movie(movie_name)

main()
