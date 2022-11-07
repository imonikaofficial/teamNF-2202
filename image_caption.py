##imports
import sys
import os
##1:
import string
from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
##2:
from pickle import load
##3:
from numpy import array
import tensorflow
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import add
from keras.callbacks import ModelCheckpoint
##4:
from numpy import argmax
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from pickle import HIGHEST_PROTOCOL

##Global vars
container_directory = "test_data/"
#print(container_directory)
DEBUG_PRINTING = False
string_identifier = "###"
DEBUG_TEST = False

def debug_print(message):
        if DEBUG_PRINTING:
                print(message)

##Functions
##1:
# extract features from each photo in the directory
def extract_features(directory):
        # load the model
        model = VGG16()
        # re-structure the model
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        # summarize
        #print(model.summary())
        # extract features from each photo
        features = dict()
        count = 0
        for name in listdir(directory):
                # load an image from file
                filename = directory + '/' + name
                image = load_img(filename, target_size=(224, 224))
                # convert the image pixels to a numpy array
                image = img_to_array(image)
                # reshape data for the model
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                # prepare the image for the VGG model
                image = preprocess_input(image)
                # get features
                feature = model.predict(image, verbose=0)
                # get image id
                image_id = name.split('.')[0]
                # store feature
                features[image_id] = feature
                #print('>%s' % name)
                count += 1
                if count%1000 == 0:
                        debug_print("processed:", count)
        return features

# load doc into memory
def load_doc(filename):
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text

# load a pre-defined list of photo identifiers
def load_set(filename):
        doc = load_doc(filename)
        dataset = list()
        # process line by line
        for line in doc.split('\n'):
                # skip empty lines
                if len(line) < 1:
                        continue
                # get the image identifier
                identifier = line.split('.')[0]
                dataset.append(identifier)
        return set(dataset)

# extract descriptions for images
def load_descriptions(doc):
        mapping = dict()
        # process lines
        for line in doc.split('\n'):
                # split line by white space
                tokens = line.split()
                if len(line) < 2:
                        continue
                # take the first token as the image id, the rest as the description
                image_id, image_desc = tokens[0], tokens[1:]
                # remove filename from image id
                image_id = image_id.split('.')[0]
                # convert description tokens back to string
                image_desc = ' '.join(image_desc)
                # create the list if needed
                if image_id not in mapping:
                        mapping[image_id] = list()
                # store description
                mapping[image_id].append(image_desc)
        return mapping

def clean_descriptions(descriptions):
        # prepare translation table for removing punctuation
        table = str.maketrans('', '', string.punctuation)
        for key, desc_list in descriptions.items():
                for i in range(len(desc_list)):
                        desc = desc_list[i]
                        # tokenize
                        desc = desc.split()
                        # convert to lower case
                        desc = [word.lower() for word in desc]
                        # remove punctuation from each token
                        desc = [w.translate(table) for w in desc]
                        # remove hanging 's' and 'a'
                        desc = [word for word in desc if len(word)>1]
                        # remove tokens with numbers in them
                        desc = [word for word in desc if word.isalpha()]
                        # store as string
                        desc_list[i] =  ' '.join(desc)

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
        # build a list of all description strings
        all_desc = set()
        for key in descriptions.keys():
                [all_desc.update(d.split()) for d in descriptions[key]]
        return all_desc

# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
        lines = list()
        for key, desc_list in descriptions.items():
                for desc in desc_list:
                        lines.append(key + ' ' + desc)
        data = '\n'.join(lines)
        file = open(filename, 'w')
        file.write(data)
        file.close()


# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
        # load document
        doc = load_doc(filename)
        descriptions = dict()
        for line in doc.split('\n'):
                # split line by white space
                tokens = line.split()
                # split id from description
                image_id, image_desc = tokens[0], tokens[1:]
                # skip images not in the set
                if image_id in dataset:
                        # create list
                        if image_id not in descriptions:
                                descriptions[image_id] = list()
                        # wrap description in tokens
                        desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
                        # store
                        descriptions[image_id].append(desc)
        return descriptions

##2:
# load photo features
def load_photo_features(filename, dataset):
        # load all features
        all_features = load(open(filename, 'rb'))
        # filter features
        features = {k: all_features[k] for k in dataset}
        return features

##3:

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
        all_desc = list()
        for key in descriptions.keys():
                [all_desc.append(d) for d in descriptions[key]]
        return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
        lines = to_lines(descriptions)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer


# calculate the length of the description with the most words
def max_length(descriptions):
        lines = to_lines(descriptions)
        return max(len(d.split()) for d in lines)

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo):
        X1, X2, y = list(), list(), list()
        # walk through each description for the image
        for desc in desc_list:
                # encode the sequence
                seq = tokenizer.texts_to_sequences([desc])[0]
                # split one sequence into multiple X,y pairs
                for i in range(1, len(seq)):
                        # split into input and output pair
                        in_seq, out_seq = seq[:i], seq[i]
                        # pad input sequence
                        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                        # encode output sequence
                        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                        # store
                        X1.append(photo)
                        X2.append(in_seq)
                        y.append(out_seq)
        return array(X1), array(X2), array(y)

# define the captioning model
def define_model(vocab_size, max_length):
        # feature extractor model
        #inputs1 = Input(shape=(4096,)) #Original
        inputs1 = Input(shape=(1000,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        # sequence model
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        # decoder model
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        # tie it together [image, seq] [word]
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        # summarize model
        #print(model.summary())
        return model

#Below code is used to progressively load the batch of data
# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length):
        # loop for ever over images
        while 1:
                for key, desc_list in descriptions.items():
                        # retrieve the photo feature
                        photo = photos[key][0]
                        in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
                        yield [[in_img, in_seq], out_word]

# map an integer to a word
def word_for_id(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
                if index == integer:
                        return word
        return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length1):
        # seed the generation process
        in_text = 'startseq'
        # iterate over the whole length of the sequence
        for i in range(max_length1):
                # integer encode input sequence
                sequence = tokenizer.texts_to_sequences([in_text])[0]
                # pad input
                sequence = pad_sequences([sequence], maxlen=max_length1)
                # predict next word
                yhat = model.predict([photo,sequence], verbose=0)
                # convert probability to integer
                yhat = argmax(yhat)
                # map integer to word
                word = word_for_id(yhat, tokenizer)
                # stop if we cannot map the word
                if word is None:
                        break
                # append as input for generating the next word
                in_text += ' ' + word
                # stop if we predict the end of the sequence
                if word == 'endseq':
                        break
        return in_text

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length1):
        actual, predicted = list(), list()
        # step over the whole set
        for key, desc_list in descriptions.items():
                # generate description
                yhat = generate_desc(model, tokenizer, photos[key], max_length1)
                # store actual and predicted
                references = [d.split() for d in desc_list]
                actual.append(references)
                predicted.append(yhat.split())
        # calculate BLEU score
        print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
        print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
        print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
        print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))



def test(epoch_count, dataset_directory, training_dataset, token_dataset, testing_dataset=None):
        global container_directory, train,\
               train_descriptions, train_features,\
               vocabulary, vocab_size,\
               max_length1, DEBUG_TEST

        if DEBUG_TEST:
                try:
                        td = testing_dataset
                        if container_directory not in testing_dataset:
                                td = container_directory + testing_dataset
                        with open(td,"r") as f:
                                pass
                except:
                        print("testing dataset absent, testing will not be performed.")
                        DEBUG_TEST = False
        
        debug_print("start")
        # extract features from all images
        #directory = container_directory + 'Flicker8k_Dataset_small'
        directory = container_directory + dataset_directory
        #directory = 'processing\\important_frames'
        features = extract_features(directory)

        debug_print('   ----2202 training model for video frames----\n  ')
        debug_print('              ---Extract Images---\n                ')
        debug_print('Extracted Features: %d' % len(features))
        # save to file
        dump(features, open(container_directory + 'features.pkl', 'wb'))

        debug_print("features file created.")
        #input()

        # filename = 'Flickr8k_text/Flickr8k.token.txt'
        filename = token_dataset
        if container_directory not in filename:
                filename = container_directory + filename
        # load descriptions
        doc = load_doc(filename)
        # parse descriptions
        descriptions = load_descriptions(doc)
        debug_print('Loaded: %d ' % len(descriptions))
        # clean descriptions
        clean_descriptions(descriptions)
        # summarize vocabulary
        vocabulary = to_vocabulary(descriptions)
        debug_print('Vocabulary Size: %d' % len(vocabulary))
        # save to file
        save_descriptions(descriptions, container_directory + 'descriptions.txt')

        debug_print("Descriptions loaded.")
        #input()

        # load training dataset (6K)
        filename = training_dataset
        if container_directory not in filename:
                filename = container_directory + filename
                         
        train = load_set(filename)
        debug_print('Dataset: %d' % len(train))
        # descriptions
        train_descriptions = load_clean_descriptions(container_directory + 'descriptions.txt', train)
        debug_print('Descriptions: train=%d' % len(train_descriptions))
        # photo features
        train_features = load_photo_features(container_directory + 'features.pkl', train)
        debug_print('Photos: train=%d' % len(train_features))

        # prepare tokenizer
        tokenizer = create_tokenizer(train_descriptions)
        # get vocabulary size
        vocab_size = len(tokenizer.word_index) + 1
        debug_print('Vocabulary Size: %d' % vocab_size)
        # determine the maximum sequence length
        max_length1 = max_length(train_descriptions)
        debug_print('Description Length: %d' % max_length1)
        
        debug_print("tokeniser, descriptions, features loaded.")

        with open(container_directory + "max_length.txt", "w+") as f:
                f.write(str(max_length1))
        #train the model
        model = define_model(vocab_size, max_length1)
        # train the model, run epochs manually and save after each epoch
        epochs = epoch_count
        steps = len(train_descriptions)
        debug_print(steps)
        debug_print(max_length1)
        min_loss = None
        chosen_model = 0
        for i in range(epochs):
                # create the data generator
                generator = data_generator(train_descriptions, train_features, tokenizer, max_length1)
                # fit for one epoch
                history = model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=0)
                # save model
                debug_print('model_' + str(i) + '.h5 ' + "created")
                if min_loss == None:
                        min_loss = history.history['loss']
                        chosen_model = i

                else:
                        if min_loss > history.history['loss']:
                                min_loss = history.history['loss']
                                chosen_model = i
                        else:
                                pass
                debug_print(min_loss)
                model.save(container_directory + 'model_' + str(i) + '.h5')

        # prepare test set
        debug_print("models created.")

        if DEBUG_TEST:
                # load test set
                filename = testing_dataset
                if container_directory not in filename:
                         filename = container_directory + filename
                #filename = container_directory + 'Flickr_8k.testImages_small.txt'
                test = load_set(filename)
                debug_print('Dataset: %d' % len(test))
                # descriptions
                test_descriptions = load_clean_descriptions(container_directory + 'descriptions.txt', test)
                debug_print('Descriptions: test=%d' % len(test_descriptions))
                # photo features
                test_features = load_photo_features(container_directory + 'features.pkl', test)
                debug_print('Photos: test=%d' % len(test_features))

                # load the model which has minimum loss, in this case it was model_18
                filename = container_directory + 'model_'+str(chosen_model)+'.h5'
                model = load_model(filename)
                # evaluate model
                evaluate_model(model, test_descriptions, test_features, tokenizer, max_length1)
                debug_print("model evaluated.")
        
        with open(container_directory + 'tokenizer.pkl', 'wb') as handle:
                dump(tokenizer, handle, protocol=HIGHEST_PROTOCOL)

        with open(container_directory + "chosen_model.txt", "w+") as f:
                f.write('model_'+str(chosen_model)+'.h5')

        # extract features from each photo in the directory
        
def extract_features_for_image(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature


def generate_caption(image):
        # load the tokenizer
        tokenizer = load(open(container_directory + 'tokenizer.pkl', 'rb'))
        # pre-define the max sequence length (from training)
        max_length1 = 0
        chosen_model = ""
        try:
                with open(container_directory + "max_length.txt", "r") as f:
                        max_length1 = int(f.read())
        except:
                max_length1 = 34

        try:
                with open(container_directory + "chosen_model.txt", "r") as f:
                        chosen_model = f.read()
        except:
                chosen_model = "model_0.h5"
        #print(max_length1)
        #print(chosen_model)
                

                
        # load the model
        model = load_model(container_directory + chosen_model)
        # load and prepare the photograph
        photo = extract_features_for_image(image)
        # generate description
        description = generate_desc(model, tokenizer, photo, max_length1)
        debug_print(description)

        #Remove startseq and endseq
        query = description
        stopwords = ['startseq','endseq']
        querywords = query.split()

        resultwords  = [word for word in querywords if word.lower() not in stopwords]
        result = ' '.join(resultwords)

        print(string_identifier + result + string_identifier)


def open_test(data):
        data_to_open = data
        if container_directory not in data:
                data_to_open = container_directory + data
        try:
                with open(data_to_open,"r") as f:
                        pass
                return 1
        except:
                return 0
                
def validate_trainset(argvs):
        if not argvs[2].isdigit():
               return 0
        filenames = []

        listdir_data = []
        try:
                if container_directory not in argvs[3]:
                        direct = container_directory + argvs[3]
                listdir_data = os.listdir(direct)
        except:
                return 0

        for i in range(4,6):
                if not open_test(argvs[i]):
                        return 0
                
        if len(argvs) > 6:
                if not open_test(argvs[6]):
                        return 0
                testset = argvs[6]
                if container_directory not in testset:
                        testset = container_directory + testset
                with open(testset,"r") as f:
                        data = f.read().strip("\n")
                        filenames += data.split("\n")

        testset = argvs[4]
        if container_directory not in testset:
                testset = container_directory + testset
        with open(testset,"r") as f:
                data = f.read().strip("\n")
                filenames += data.split("\n")
        filenames.sort()
        listdir_data.sort()
        for i in filenames:
                if i not in listdir_data:
                        return 0
                
        return 1
      
def main():
        debug_print(sys.argv)
        execution_type = -1
        ex_type = {"caption":0, "train_small":1, "train_large":2, "train_custom":3}

        if sys.argv[1] in ex_type:
                execution_type = ex_type[sys.argv[1]]
        else:
                print("No such option exists.")
                return
        
        if execution_type == 1:
                DEBUG_TEST = True
                test(5, 'Flicker8k_Dataset_small', 'Flickr_8k.trainImages_small.txt', 'Flickr8k.token_small.txt', 'Flickr_8k.testImages_small.txt')
        elif execution_type == 2:
                DEBUG_TEST = True
                test(20, 'Flicker8k_Dataset', 'Flickr_8k.trainImages.txt', 'Flickr8k.token.txt', 'Flickr_8k.testImages.txt')
        elif execution_type == 3:
                if len(sys.argv) < 5:
                        print("not enough arguments.")
                        return 0
                if not validate_trainset(sys.argv):
                        print("invalid arguments.")
                        return 0
                if len(sys.argv) > 6:
                        DEBUG_TEST = True
                        test(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
                else:
                        test(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
        
        else:
                generate_caption(sys.argv[2])
#python test.py train_custom 5 Flicker8k_Dataset_small Flickr_8k.trainImages_small.txt Flickr8k.token_small.txt Flickr_8k.testImages_small.txt

main()
