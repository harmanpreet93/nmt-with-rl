import argparse
import utils
from data_loader import DataLoader
from generate_model_predictions import sacrebleu_metric, compute_bleu
import tensorflow as tf
import os
import json
import sacrebleu
from transformer import create_masks
import tensorflow_probability as tfp
import numpy as np
from time import time


# from functools import wraps
# def timeit(f):
#     @wraps(f)
#     def wrap(*args, **kw):
#         ts = time()
#         result = f(*args, **kw)
#         te = time()
#         print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
#         return result
#
#     return wrap


def get_bleu_score(sys, refs):
    """
    :param sys: sentence 1
    :param refs: sentence 2
    :return: bleu score
    """
    bleu = sacrebleu.corpus_bleu(sys, [refs])
    # TODO: bleu score 0 if length less than 3
    return bleu.score


def get_sample_sent(sent, greedy=True):
    if greedy:
        sent = tf.argmax(sent, axis=-1)
        return sent, None
    else:
        sent_dist = tfp.distributions.Categorical(sent)  # batch X seq-len
        sent_sample = sent_dist.sample()
        sent_log_prob = sent_dist.log_prob(sent_sample)
        return sent_sample, sent_log_prob


def get_rl_loss(real, pred, tokenizer_tar):
    sample_sents, log_probs = get_sample_sent(pred, greedy=False)
    greedy_sents, _ = get_sample_sent(pred, greedy=True)

    sampled_out = tokenizer_tar.sequences_to_texts(sample_sents.numpy())
    greedy_out = tokenizer_tar.sequences_to_texts(greedy_sents.numpy())
    real_out = tokenizer_tar.sequences_to_texts(real.numpy())

    sample_reward = get_bleu_score(sampled_out, real_out)
    baseline_reward = get_bleu_score(greedy_out, real_out)

    rl_loss = -(sample_reward - baseline_reward) * log_probs
    batch_reward = np.array(sample_reward).mean()

    return rl_loss, batch_reward


# Since the target sequences are padded, it is important
# to apply a padding mask when calculating the loss.
def loss_function(real, pred, loss_object, tokenizer_tar, pad_token_id, use_rl=True):
    """Calculates total loss containing cross entropy with padding ignored.
      Args:
        real: Tensor of size [batch_size, length_logits, vocab_size]
        pred: Tensor of size [batch_size, length_labels]
        loss_object: Cross entropy loss
        tokenizer_tar: tokenizer
        pad_token_id: Pad token id to ignore
      Returns:
        A scalar float tensor for loss.
    """
    mask = tf.math.logical_not(tf.math.equal(real, pad_token_id))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    if use_rl:
        rl_loss, batch_reward = get_rl_loss(real, pred, tokenizer_tar)
        rl_loss *= mask
        combined_loss = lambda_DL * loss_ + lambda_RL * rl_loss
    else:
        combined_loss = loss_
        batch_reward = 0
    return tf.reduce_sum(combined_loss) / tf.reduce_sum(mask), batch_reward


def train_step(model, loss_object, optimizer, inp, tar,
               train_loss, train_accuracy, tokenizer_tar, pad_token_id, use_rl=True):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions, _ = model(inp, tar_inp,
                               True,
                               enc_padding_mask,
                               combined_mask,
                               dec_padding_mask)

        loss, batch_reward = loss_function(tar_real, predictions, loss_object, tokenizer_tar, pad_token_id, use_rl)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)
    return batch_reward


def val_step(model, loss_object, inp, tar,
             val_loss, val_accuracy, tokenizer_tar, pad_token_id, use_rl=True):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    predictions, _ = model(inp, tar_inp,
                           False,
                           enc_padding_mask,
                           combined_mask,
                           dec_padding_mask)
    loss, batch_reward = loss_function(tar_real, predictions, loss_object, tokenizer_tar, pad_token_id, use_rl)

    val_loss(loss)
    val_accuracy(tar_real, predictions)
    return batch_reward


def compute_bleu_score(transformer_model, dataset, user_config, tokenizer_tar, epoch):
    inp_language = user_config["inp_language"]
    target_language = user_config["target_language"]
    checkpoint_path = user_config["transformer_checkpoint_path"]
    val_aligned_path_tar = user_config["val_data_path_{}".format(target_language)]
    pred_file_path = "../log/log_{}_{}/".format(inp_language, target_language) + checkpoint_path.split('/')[
        -1] + "_epoch-" + str(epoch) + "_prediction_{}.txt".format(target_language)

    sacrebleu_metric(transformer_model, pred_file_path, None,
                     tokenizer_tar, dataset,
                     max_length=120)
    print("-----------------------------")
    compute_bleu(pred_file_path, val_aligned_path_tar, print_all_scores=False)
    print("-----------------------------")

    # append checkpoint and score to file name for easy reference
    new_path = "../log/log_{}_{}/".format(inp_language, target_language) + checkpoint_path.split('/')[
        -1] + "_epoch-" + str(epoch) + "_prediction_{}".format(target_language) + ".txt"
    # append score and checkpoint name to file_name
    os.rename(pred_file_path, new_path)
    print("Saved translated prediction at {}".format(new_path))


def do_training(user_config):
    inp_language = user_config["inp_language"]
    target_language = user_config["target_language"]

    print("\n****Training model from {} to {}****\n".format(inp_language, target_language))
    print("****Using RL: {}****".format(user_config["user_RL"]))

    print("****Loading tokenizers****")
    # load pre-trained tokenizer
    tokenizer_inp, tokenizer_tar = utils.load_tokenizers()

    print("****Loading train dataset****")
    # train data loader
    train_aligned_path_inp = user_config["train_data_path_{}".format(inp_language)]
    train_aligned_path_tar = user_config["train_data_path_{}".format(target_language)]
    train_dataloader = DataLoader(user_config["transformer_batch_size"],
                                  train_aligned_path_inp,
                                  train_aligned_path_tar,
                                  tokenizer_inp,
                                  tokenizer_tar,
                                  inp_language,
                                  target_language,
                                  True)
    train_dataset = train_dataloader.get_data_loader()

    print("****Loading val dataset****")
    # val data loader
    val_aligned_path_inp = user_config["val_data_path_{}".format(inp_language)]
    val_aligned_path_tar = user_config["val_data_path_{}".format(target_language)]
    val_dataloader = DataLoader(user_config["transformer_batch_size"] * 2,  # for fast validation increase batch size
                                val_aligned_path_inp,
                                val_aligned_path_tar,
                                tokenizer_inp,
                                tokenizer_tar,
                                inp_language,
                                target_language,
                                False)
    val_dataset = val_dataloader.get_data_loader()

    # define loss and accuracy metrics
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    train_batch_reward = []
    val_batch_reward = []
    user_rl = user_config['user_RL']
    pad_token_id = 0

    print("****Loading transformer model****")
    # load model and optimizer
    transformer_model, optimizer, ckpt_manager = \
        utils.load_transformer_model(user_config, tokenizer_inp, tokenizer_tar)

    epochs = user_config["transformer_epochs"]
    print("\nTraining model now...")
    for epoch in range(epochs):
        print()
        start = time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        # inp -> english, tar -> french
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_batch_reward.append(train_step(transformer_model, loss_object, optimizer, inp, tar,
                                                 train_loss, train_accuracy, tokenizer_tar,
                                                 pad_token_id=pad_token_id, use_rl=user_rl))

            if batch % 50 == 0:
                print('Train: Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
                if user_rl:
                    print("Train reward {}".format(np.mean(train_batch_reward)))

        print("After {} epochs".format(epoch + 1))
        print('Train Loss: {:.4f}, Train Accuracy: {:.4f}'.format(train_loss.result(), train_accuracy.result()))

        # inp -> english, tar -> french
        for (batch, (inp, tar)) in enumerate(val_dataset):
            val_batch_reward.append(val_step(transformer_model, loss_object, inp, tar,
                                             val_loss, val_accuracy, tokenizer_tar,
                                             pad_token_id=pad_token_id, use_rl=user_rl))
        print('Val Loss: {:.4f}, Val Accuracy: {:.4f}'.format(val_loss.result(), val_accuracy.result()))
        if user_rl:
            print("Val reward {}".format(np.mean(val_batch_reward)))

        print('Time taken for training epoch {}: {} secs'.format(epoch + 1, time() - start))

        # evaluate and save model every x-epochs
        if (epoch + 1) % 5 == 0 and user_config["compute_bleu"]:
            ckpt_save_path = ckpt_manager.save()
            print('Saved checkpoint after epoch {} at {}'.format(epoch + 1, ckpt_save_path))
            print("\nComputing BLEU at epoch {}: ".format(epoch + 1))
            compute_bleu_score(transformer_model, val_dataset, user_config, tokenizer_tar, epoch + 1)


def main():
    global lambda_DL
    global lambda_RL
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file containing training parameters", type=str)
    args = parser.parse_args()
    user_config = utils.load_file(args.config)
    seed = user_config["random_seed"]
    utils.set_seed(seed)

    lambda_DL = user_config["lambda_dl"]
    lambda_RL = user_config["lambda_rl"]
    print(json.dumps(user_config, indent=2))
    do_training(user_config)


if __name__ == "__main__":
    main()
