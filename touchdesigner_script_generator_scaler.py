# me - this DAT
# scriptOp - the OP which is cooking
#
# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
#
## Python Program to integrate Pix2Pix Models in Touchdesigner and scale them up to 1024x1024
## by Alexander König 2021
#
import numpy as np
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential, Model, load_model, save_model

pix2pix_model = "/save_model"
sr_model = "/SuperResolution/ESPCN_x4.pb"

#load ML-Models
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(sr_model)
sr.setModel("espcn",4)
generator = load_model(pix2pix_model , compile=False)

def onSetupParameters(scriptOp):
	page = scriptOp.appendCustomPage('Custom')
	p = page.appendFloat('Valuea', label='Value A')
	p = page.appendFloat('Valueb', label='Value B')

def batch_generator():
    ## NODE to NUMPY
    image = op('/project1/AI_OUT').numpyArray(delayed=True, writable=True)
    ## NORMALIZE
    image = image*255
    ## OPENCV COLOR CONVERSIONS
    image = cv2.cvtColor(np.uint8(image * 255.), cv2.COLOR_BGR2GRAY)
    ## CANNY FILTER
    image = cv2.Canny(image, 50, 200)
    ## RECONVERSION
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.expand_dims(image, axis=0)


    ## generate from CANNY-IMAGE
    img = generator(image, training=True)

    ## write image
    pil_img = tf.squeeze(img)
    pil_img = (pil_img * 127.5) + 1
    pil_img = tf.keras.preprocessing.image.array_to_img(pil_img)
    pil_img.save('generated.png','PNG')

    ## super resolution
    img = cv2.imread('generated.png')
    res_img = sr.upsample(img)
    cv2.imwrite('/image.png',res_img)

    #reload Touchdesigner Op
    op('/project1/AI_IN').par.reloadpulse.pulse()

# called whenever custom pulse parameter is pushed
def onPulse(par):
    return

def onCook(scriptOp):
#    scriptOp.clear()
    batch_generator()
