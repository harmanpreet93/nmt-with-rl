import os
import io
import json
import unicodedata

from transformer import Transformer, CustomSchedule
import numpy as np
import tensorflow as tf
import pickle
import re
import subprocess
import tensorflow_datasets as tfds


def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

def compute_bleu(pred_file_path: str, target_file_path: str, print_all_scores: bool):
    """

    Args:
        pred_file_path: the file path that contains the predictions.
        target_file_path: the file path that contains the targets (also called references).
        print_all_scores: if True, will print one score per example.

    Returns: None

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
    load json file
    param path: json file path to load
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
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def get_tokenizers(examples, user_config, VOCAB_SIZE, shuffle):
    BUFFER_SIZE = 20000
    BATCH_SIZE = user_config["transformer_batch_size"]

    train_examples, val_examples = examples['train'], examples['validation']

    tokenizer_en = tf.keras.preprocessing.text.Tokenizer(filters=' ', lower=False, num_words=VOCAB_SIZE)
    tokenizer_pt = tf.keras.preprocessing.text.Tokenizer(filters=' ', lower=False, num_words=VOCAB_SIZE)

    en_data_train, pt_data_train, en_data_val, pt_data_val = [], [], [], []
    for pt, en in train_examples:
        en_ = preprocess_sentence(en.numpy().decode("utf-8"))
        pt_ = preprocess_sentence(pt.numpy().decode("utf-8"))
        tokenizer_en.fit_on_texts(en_)
        tokenizer_pt.fit_on_texts(pt_)
        en_data_train.append(en_)
        pt_data_train.append(pt_)

    for pt, en in val_examples:
        en_ = preprocess_sentence(en.numpy().decode("utf-8"))
        pt_ = preprocess_sentence(pt.numpy().decode("utf-8"))
        tokenizer_en.fit_on_texts(en_)
        tokenizer_pt.fit_on_texts(pt_)
        en_data_val.append(en_)
        pt_data_val.append(pt_)

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


def tokenize(aligned_lang, unaligned_lang, num_words):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' ', lower=False, num_words=num_words)
    lang_tokenizer.fit_on_texts(aligned_lang + unaligned_lang)
    tensor = lang_tokenizer.texts_to_sequences(aligned_lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


def preprocess_sentence_(w, lang, aligned=True, add_special_tag=True):
    w = w.strip()
    if lang == "en" and not aligned:
        # This part is required only for english unaligned samples in word2vec
        # w = unicode_to_ascii(w.lower())
        # Removing everything except(letters)
        w = re.sub(r"[^a-z]+", " ", w)
    if lang == "fr" and not aligned:
        # Adding space with punctuation for easy split.
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
    w = w.strip()
    # currently keeping these tags for both type of data
    if add_special_tag:
        w = '<start> ' + w + ' <end>'
    return w


def load_tokenizers(user_config):
    """
        load pickled tokenizers for input and target language
    """
    if user_config["inp_language"] == "en":
        tokenizer_inp_path = "../tokenizers/tokenizer_en.pkl"
        tokenizer_tar_path = "../tokenizers/tokenizer_fr.pkl"
    else:
        tokenizer_inp_path = "../tokenizers/tokenizer_fr.pkl"
        tokenizer_tar_path = "../tokenizers/tokenizer_en.pkl"

    tokenizer_tar = pickle.load(open(tokenizer_inp_path, "rb"))
    tokenizer_inp = pickle.load(open(tokenizer_tar_path, "rb"))

    return tokenizer_inp, tokenizer_tar


# def load_tokenizers(inp_language, target_language, user_config):
#     """
#     load pre-trained tokenizer for input and target language
#     """
#
#     pretrained_tokenizer_path_inp = user_config["tokenizer_path_{}".format(inp_language)]
#     pretrained_tokenizer_path_tar = user_config["tokenizer_path_{}".format(target_language)]
#
#     tokenizer_inp = Tokenizer(inp_language, pretrained_tokenizer_path_inp,
#                               max_length=user_config["max_length_{}".format(inp_language)])
#     tokenizer_tar = Tokenizer(target_language, pretrained_tokenizer_path_tar,
#                               max_length=user_config["max_length_{}".format(target_language)])
#
#     return tokenizer_inp, tokenizer_tar


def load_transformer_model(user_config, tokenizer_inp, tokenizer_tar):
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


def create_mix_dataset(synthetic_data_path_lang1, true_data_path_lang1, true_unaligned_data_path_lang2,
                       true_data_path_lang2, num_of_times_to_add_true_data: int):
    """
    Mix back-translated dataset with aligned dataset in some pre-defined ratio
    Used during iterative back-translation to maintain certain ratio of true_data:syn_data
    """
    assert num_of_times_to_add_true_data > 0

    synthetic_data_lang1 = io.open(synthetic_data_path_lang1).read().strip().split('\n')
    true_aligned_data_lang1 = io.open(true_data_path_lang1).read().strip().split('\n')
    true_unaligned_data_lang2 = io.open(true_unaligned_data_path_lang2).read().strip().split('\n')
    true_aligned_data_lang2 = io.open(true_data_path_lang2).read().strip().split('\n')

    new_data_lang1, new_data_lang2 = synthetic_data_lang1, true_unaligned_data_lang2
    for _ in range(num_of_times_to_add_true_data):
        new_data_lang1 += true_aligned_data_lang1
        new_data_lang2 += true_aligned_data_lang2

    shuffle_together = list(zip(new_data_lang1, new_data_lang2))
    np.random.shuffle(shuffle_together)
    new_data_lang1, new_data_lang2 = zip(*shuffle_together)

    return list(new_data_lang1), list(new_data_lang2)
