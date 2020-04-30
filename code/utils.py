import os
import json
import unicodedata
import numpy as np
import tensorflow as tf
import re
import subprocess
import tensorflow_datasets as tfds
from transformer import Transformer, CustomSchedule


def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def compute_bleu(pred_file_path: str, target_file_path: str, print_all_scores: bool):
    """
    :param pred_file_path: the file path that contains the predictions
    :param target_file_path: the file path that contains the targets (also called references)
    :param print_all_scores: if True, will print one score per example
    :return: None
    """
    out = subprocess.run(["sacrebleu", "--input", pred_file_path, target_file_path, '--tokenize',
                          'none', '--sentence-level', '--score-only'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = out.stdout.split(b'\n')
    if print_all_scores:
        print('\n'.join(lines[:-1]))
    else:
        scores = [float(x) for x in lines[:-1]]
        print('final avg bleu score: {:.2f}'.format(sum(scores) / len(scores)))
        return sum(scores) / len(scores)


def load_file(path):
    """
    :param path: json file path to load
    :return: json dictionary
    """
    assert os.path.isfile(path), f"invalid config file: {path}"
    with open(path, "r") as fd:
        return json.load(fd)


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', str(s))
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference: https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def get_tokenizers(examples, user_config, VOCAB_SIZE, shuffle=True):
    BUFFER_SIZE = 20000
    BATCH_SIZE = user_config["transformer_batch_size"]

    train_examples, val_examples = examples['train'], examples['validation']

    tokenizer_en = tf.keras.preprocessing.text.Tokenizer(filters=' ', lower=False, num_words=VOCAB_SIZE)
    tokenizer_pt = tf.keras.preprocessing.text.Tokenizer(filters=' ', lower=False, num_words=VOCAB_SIZE)

    en_data_train, pt_data_train, en_data_val, pt_data_val = [], [], [], []
    for pt, en in train_examples:
        en_ = preprocess_sentence(en.numpy().decode("utf-8"))
        pt_ = preprocess_sentence(pt.numpy().decode("utf-8"))
        en_data_train.append(en_)
        pt_data_train.append(pt_)

    for pt, en in val_examples:
        en_ = preprocess_sentence(en.numpy().decode("utf-8"))
        pt_ = preprocess_sentence(pt.numpy().decode("utf-8"))
        en_data_val.append(en_)
        pt_data_val.append(pt_)

    tokenizer_en.fit_on_texts(en_data_train + en_data_val)
    tokenizer_pt.fit_on_texts(pt_data_train + pt_data_val)

    tensor_en_train = encode_and_pad(en_data_train, tokenizer_en)
    tensor_pt_train = encode_and_pad(pt_data_train, tokenizer_pt)
    tensor_en_val = encode_and_pad(en_data_val, tokenizer_en)
    tensor_pt_val = encode_and_pad(pt_data_val, tokenizer_pt)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (tensor_pt_train, tensor_en_train)).shuffle(
        BUFFER_SIZE).batch(
        BATCH_SIZE).cache().prefetch(
        tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (tensor_pt_val, tensor_en_val)).batch(
        BATCH_SIZE).cache().prefetch(
        tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, tokenizer_en, tokenizer_pt


def encode_and_pad(data, tokenizer):
    tensor = tokenizer.texts_to_sequences(data)
    return tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen=100)


def get_dataset_and_tokenizer(user_config):
    VOCAB_SIZE = 20000

    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                   with_info=True,
                                   as_supervised=True)

    train_dataset, val_dataset, tokenizer_en, tokenizer_pt = get_tokenizers(examples, user_config, VOCAB_SIZE,
                                                                            shuffle=True)

    return train_dataset, val_dataset, tokenizer_en, tokenizer_pt


def load_transformer_model(user_config):
    """
    load transformer model and latest checkpoint to continue training
    """
    input_vocab_size = 20000
    target_vocab_size = 20000

    pretrained_weights_inp = None
    pretrained_weights_tar = None

    # custom learning schedule
    learning_rate = CustomSchedule(user_config["transformer_model_dimensions"])
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    transformer_model = Transformer(user_config["transformer_num_layers"],
                                    user_config["transformer_model_dimensions"],
                                    user_config["transformer_num_heads"],
                                    user_config["transformer_dff"],
                                    input_vocab_size,
                                    target_vocab_size,
                                    en_input=input_vocab_size,
                                    fr_target=target_vocab_size,
                                    rate=user_config["transformer_dropout_rate"],
                                    weights_inp=pretrained_weights_inp,
                                    weights_tar=pretrained_weights_tar)

    ckpt = tf.train.Checkpoint(transformer=transformer_model,
                               optimizer=optimizer)

    checkpoint_path = user_config["transformer_checkpoint_path"]
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored from path {}'.format(ckpt_manager.latest_checkpoint))

    return transformer_model, optimizer, ckpt_manager
