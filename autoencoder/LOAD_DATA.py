import tensorflow as tf 
import matplotlib as plt
import numpy as np
import matplotlib.image as mpimg

#Defining functions to get class, image IDs from our TSV files and convert JPEG files to arrays
def get_IDs(f_path):
  FULL_ID = []
  # open .tsv file
  with open(f_path) as f: 
    # Read data line by line
    for line in f:     
      l=line.split('\t')
      # append list to ans
      FULL_ID.append(l)
  CLASS_ID=[]
  IMAGE_ID=[]
  for i in range(len(FULL_ID)):
      CLASS_ID.append(FULL_ID[i][0][:9])
      IMAGE_ID.append(FULL_ID[i][0])
  return CLASS_ID, IMAGE_ID


#Get Image Data from folder with images
def get_image_data(d_path, I_IDs,dim=(32,32)):
    DATA=[]
    for IDs in range(len(I_IDs)):
        image = mpimg.imread(f'{d_path}{I_IDs[IDs]}.JPEG')
        if image.shape[-1] != 3:
          image = tf.concat([tf.expand_dims(image,-1), tf.expand_dims(image,-1), tf.expand_dims(image,-1)], axis=-1)
        image = tf.image.resize(image, dim)
        image = tf.math.round(image)
        image = tf.cast(image,tf.uint8)
        image = tf.expand_dims(image,0)
        DATA.append(image)
    DATA=tf.concat(DATA, axis = 0)
    #print(DATA.shape)
    return DATA