import numpy as np
import keras.backend as K
import tensorflow as tf
import numpy as np
import pickle 
import operator
import pandas as pd
from threading import Thread
from tensorflow.keras.models import load_model
from keras.layers import Embedding
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict



def image_feature_extractor(img_path):
	img = image.load_img(img_path,target_size=(224,224))
	x = image.img_to_array(img)
	x = np.expand_dims(x,axis=0)
	x = preprocess_input(x)
	IMG_SHAPE = (224, 224, 3)
	model = tf.keras.applications.VGG16()
	last_layer = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('fc2').output)
	img_fet = last_layer.predict(x)

	return img_fet


def process_sentence(sentence):
	questions=sentence
	with open('data/tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
	train_question_tokenized = tokenizer.texts_to_sequences(questions)
	questions = pad_sequences(train_question_tokenized, maxlen = 25) 
	return questions


def predict(img_feat, ques_feat,model,int_to_answers):
	x=[img_feat,ques_feat]
	historty=model.predict(x)[0]
	answers = np.argsort(historty[:1000])
	top_answers = int_to_answers()
	answer=top_answers[answers[-1]]

	return answer




