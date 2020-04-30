import io
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import preprocess_sentence

class DataLoader:
    """
    Data-loader to load to input and target dataset
    """

    def __init__(self, batch_size, input_lang_path, target_lang_path, tokenizer_inp, tokenizer_tar, input_lang,
                 target_lang, shuffle=True):
        self.BATCH_SIZE = batch_size
        self.input_lang_path = input_lang_path
        self.target_lang_path = target_lang_path
        self.tokenizer_inp = tokenizer_inp
        self.tokenizer_tar = tokenizer_tar
        self.BUFFER_SIZE = 20000  # buffer size to shuffle
        self.shuffle = shuffle
        self.input_lang = input_lang
        self.target_lang = target_lang
        self.initialize()

    def initialize(self):

        input_data = io.open(self.input_lang_path, encoding='UTF-8').read().strip().split('\n')
        input_data = [preprocess_sentence(x, self.input_lang) for x in input_data]
        input_data = self.tokenizer_inp.texts_to_sequences(input_data)
        input_data = pad_sequences(input_data, padding='post', maxlen=120)

        output_data = io.open(self.target_lang_path, encoding='UTF-8').read().strip().split('\n')
        output_data = [preprocess_sentence(x, self.target_lang) for x in output_data]
        output_data = self.tokenizer_tar.texts_to_sequences(output_data)
        output_data = pad_sequences(output_data, padding='post', maxlen=120)

        # aligned_sentences_tar is required for evaluation (ignore for training)
        if self.shuffle:
            self.data_loader = tf.data.Dataset.from_tensor_slices(
                (input_data, output_data)).shuffle(
                self.BUFFER_SIZE).batch(
                self.BATCH_SIZE).prefetch(
                tf.data.experimental.AUTOTUNE)
        else:
            self.data_loader = tf.data.Dataset.from_tensor_slices(
                (input_data, output_data)).batch(
                self.BATCH_SIZE).prefetch(
                tf.data.experimental.AUTOTUNE)

    def get_data_loader(self):
        '''
        Returns: ``tf.data.Dataset`` object
        '''
        return self.data_loader
