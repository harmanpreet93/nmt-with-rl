import argparse
import datetime
import utils
from generate_model_predictions import generate_and_save_translations
import tensorflow as tf
import os
import json
import sacrebleu
from transformer import create_masks
import tensorflow_probability as tfp
import numpy as np
from time import time


def get_bleu_score(sys, refs):
    """
    :param sys: sentence 1
    :param refs: sentence 2
    :return: bleu score
    """
    # Note: bleu score 0 if length less than 3
    bleu_scores = [sacrebleu.corpus_bleu([sys[i]], [[refs[i]]]).score for i in range(len(sys))]
    return np.array(bleu_scores)


def get_sample_sent(sent, greedy=True):
    if greedy:
        sent = tf.argmax(sent, axis=-1)
        return sent, None
    else:
        sent_dist = tfp.distributions.Categorical(sent)  # batch X seq-len
        sent_sample = sent_dist.sample()
        sent_log_prob = sent_dist.log_prob(sent_sample)
        return sent_sample, sent_log_prob


def rl_loss_fn(real_seq, pred_seq, tokenizer_tar, pad_token_id):
    sample_seq, log_probs = get_sample_sent(pred_seq, greedy=False)
    greedy_seq, _ = get_sample_sent(pred_seq, greedy=True)

    sampled_sents = tokenizer_tar.sequences_to_texts(sample_seq.numpy())
    greedy_sents = tokenizer_tar.sequences_to_texts(greedy_seq.numpy())
    real_sents = tokenizer_tar.sequences_to_texts(real_seq.numpy())

    sampled_sents = [x.split("<end>")[0] for x in sampled_sents]
    greedy_sents = [x.split("<end>")[0] for x in greedy_sents]
    real_sents = [x.split("<end>")[0] for x in real_sents]

    sample_reward = get_bleu_score(sampled_sents, real_sents)
    baseline_reward = get_bleu_score(greedy_sents, real_sents)

    mask = tf.math.logical_not(tf.math.equal(real_seq, pad_token_id))
    mask = tf.cast(mask, dtype=log_probs.dtype)
    log_probs *= mask
    log_probs = tf.reduce_sum(log_probs, axis=1) / tf.reduce_sum(mask, axis=1)  # avg on seq level
    rl_loss = (sample_reward - baseline_reward) * log_probs  # batch_size * 1
    rl_loss = tf.reduce_mean(rl_loss)  # avg on batch level

    return rl_loss, np.mean(sample_reward)


# Since the target sequences are padded, it is important
# to apply a padding mask when calculating the losses
def cross_entropy_loss_fn(real, pred, loss_object, pad_token_id):
    """Calculates total loss containing cross entropy with padding ignored.
      Args:
        real: Tensor of size [batch_size, length_logits, vocab_size]
        pred: Tensor of size [batch_size, length_labels]
        loss_object: Cross entropy loss
        pad_token_id: Pad token id to ignore
      Returns:
        A scalar float tensor for loss.
    """
    mask = tf.math.logical_not(tf.math.equal(real, pad_token_id))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def loss_function(real, pred, loss_object, tokenizer_tar, pad_token_id, use_rl=True):
    mle_loss = cross_entropy_loss_fn(real, pred, loss_object, pad_token_id)

    if use_rl:
        rl_loss_, batch_reward = rl_loss_fn(real, pred, tokenizer_tar, pad_token_id)
        combined_loss = lambda_DL * mle_loss + lambda_RL * rl_loss_
    else:
        combined_loss = mle_loss
        rl_loss_ = 0
        batch_reward = 0

    return combined_loss, mle_loss, rl_loss_, batch_reward


def train_step(model, loss_object, optimizer, inp, tar, train_accuracy,
               tokenizer_tar, pad_token_id, use_rl=True):
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

        combined_loss, loss_, rl_loss, batch_reward = loss_function(tar_real, predictions,
                                                                    loss_object, tokenizer_tar,
                                                                    pad_token_id, use_rl)

    gradients = tape.gradient(combined_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy(tar_real, predictions)
    return combined_loss, loss_, rl_loss, batch_reward


def val_step(model, loss_object, inp, tar,
             val_accuracy, tokenizer_tar, pad_token_id, use_rl=True):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    predictions, _ = model(inp, tar_inp,
                           False,
                           enc_padding_mask,
                           combined_mask,
                           dec_padding_mask)
    combined_loss, loss_, rl_loss, batch_reward = loss_function(tar_real, predictions,
                                                                loss_object, tokenizer_tar,
                                                                pad_token_id, use_rl)

    val_accuracy(tar_real, predictions)
    return combined_loss, loss_, rl_loss, batch_reward


def compute_bleu_score(transformer_model, dataset, user_config, tokenizer_tar, epoch):
    inp_language = user_config["inp_language"]
    target_language = user_config["target_language"]
    checkpoint_path = user_config["transformer_checkpoint_path"]
    val_aligned_path_tar = user_config["val_data_path_{}".format(target_language)]
    pred_file_path = "../logs/log_{}_{}/".format(inp_language, target_language) + checkpoint_path.split('/')[
        -1] + "_epoch-" + str(epoch) + "_prediction_{}.txt".format(target_language)

    generate_and_save_translations(transformer_model, pred_file_path,
                                   tokenizer_tar, dataset,
                                   max_length=120)

    print("-----------------------------")
    scores = utils.compute_bleu(pred_file_path, val_aligned_path_tar, print_all_scores=False)
    print("-----------------------------")

    # append checkpoint and score to file name for easy reference
    new_path = "../logs/log_{}_{}/".format(inp_language, target_language) + checkpoint_path.split('/')[
        -1] + "_epoch-" + str(epoch) + "_prediction_{}_".format(target_language) + "{:.3f}.txt".format(scores)
    # append score and checkpoint name to file_name
    os.rename(pred_file_path, new_path)
    print("Saved translated prediction at {}".format(new_path))

    return scores


def do_training(user_config):
    inp_language = user_config["inp_language"]
    target_language = user_config["target_language"]
    print("\n****Training model from {} to {}****".format(inp_language, target_language))
    print("****Using RL: {}****".format(user_config["user_RL"]))
    print("****Tensroboard Logging: {}****".format(user_config["tensorboard_logging"]))
    print("****Loading tokenizers and datasets****")

    train_dataset, val_dataset, tokenizer_tar, tokenizer_inp = utils.get_dataset_and_tokenizer(user_config)

    # define loss and accuracy metrics
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_rl_loss = tf.keras.metrics.Mean(name='train_rl_loss')
    train_mle_loss = tf.keras.metrics.Mean(name='train_mle_loss')
    train_reward = tf.keras.metrics.Mean(name='train_reward')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_rl_loss = tf.keras.metrics.Mean(name='val_rl_loss')
    val_mle_loss = tf.keras.metrics.Mean(name='val_mle_loss')
    val_reward = tf.keras.metrics.Mean(name='val_reward')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    user_rl = user_config['user_RL']
    PAD_TOKEN_ID = 0

    if user_config["tensorboard_logging"]:
        model_name = user_config["transformer_checkpoint_path"].split("/")[-1]
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = '../logs/gradient_tape/' + model_name + "/" + current_time + '/train'
        test_log_dir = '../logs/gradient_tape/' + model_name + "/" + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    print("****Loading transformer model****")
    # load model and optimizer
    transformer_model, optimizer, ckpt_manager = utils.load_transformer_model(user_config)

    start_epoch = 21  # manually setting epoch number for best DL model

    epochs = user_config["transformer_epochs"]
    total_steps = 50000 // user_config["transformer_batch_size"] + 1
    print("\nTraining model now...")
    for epoch in range(start_epoch, epochs):
        print()
        start = time()
        train_loss.reset_states()
        train_rl_loss.reset_states()
        train_mle_loss.reset_states()
        train_reward.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_rl_loss.reset_states()
        val_mle_loss.reset_states()
        val_reward.reset_states()
        val_accuracy.reset_states()

        # inp -> english, tar -> french
        for (batch, (inp, tar)) in enumerate(train_dataset):
            combined_loss, loss_, rl_loss, batch_reward = train_step(transformer_model, loss_object,
                                                                     optimizer, inp, tar,
                                                                     train_accuracy, tokenizer_tar,
                                                                     pad_token_id=PAD_TOKEN_ID, use_rl=user_rl)

            train_loss(combined_loss)
            train_mle_loss(loss_)
            train_rl_loss(rl_loss)
            train_reward(batch_reward)

            if batch % 50 == 0:
                print('Train: Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
                if user_rl:
                    print("Train: MLE Loss: {}, RL Loss: {} Reward: {}".format(train_mle_loss.result(),
                                                                               train_rl_loss.result(),
                                                                               train_reward.result()))

            if batch % 10 == 0:
                if user_config["tensorboard_logging"]:
                    with train_summary_writer.as_default():
                        tf.summary.scalar('Combined Loss', train_loss.result(), step=total_steps * epoch + batch)
                        tf.summary.scalar('MLE Loss', train_mle_loss.result(), step=total_steps * epoch + batch)
                        tf.summary.scalar('RL Loss', train_rl_loss.result(), step=total_steps * epoch + batch)
                        tf.summary.scalar('Average Reward', train_reward.result(), step=total_steps * epoch + batch)
                        tf.summary.scalar('Accuracy', train_accuracy.result(), step=total_steps * epoch + batch)

        print("After {} epochs".format(epoch + 1))
        print('Train Loss: {:.4f}, Train Accuracy: {:.4f}'.format(train_loss.result(), train_accuracy.result()))

        # inp -> english, tar -> french
        for (batch, (inp, tar)) in enumerate(val_dataset):
            combined_loss, loss_, rl_loss, batch_reward = val_step(transformer_model, loss_object,
                                                                   inp, tar,
                                                                   val_accuracy, tokenizer_tar,
                                                                   pad_token_id=PAD_TOKEN_ID,
                                                                   use_rl=user_rl)
            val_loss(combined_loss)
            val_mle_loss(loss_)
            val_rl_loss(rl_loss)
            val_reward(batch_reward)

        print('Val Loss: {:.4f}, Val Accuracy: {:.4f}'.format(val_loss.result(), val_accuracy.result()))
        if user_rl:
            print("Val: MLE Loss: {}, RL Loss: {} Reward: {}".format(val_mle_loss.result(),
                                                                     val_rl_loss.result(),
                                                                     val_reward.result()))

        if user_config["tensorboard_logging"]:
            with test_summary_writer.as_default():
                tf.summary.scalar('Combined Loss', val_loss.result(), step=(epoch + 1))
                tf.summary.scalar('MLE Loss', val_mle_loss.result(), step=(epoch + 1))
                tf.summary.scalar('RL Loss', val_rl_loss.result(), step=(epoch + 1))
                tf.summary.scalar('Average Reward', val_reward.result(), step=(epoch + 1))
                tf.summary.scalar('Accuracy', val_accuracy.result(), step=(epoch + 1))

        print('Time taken for training epoch {}: {} secs'.format(epoch + 1, time() - start))

        # evaluate and save model every x-epochs
        if (epoch) % 5 == 0 and user_config["compute_bleu"]:
            ckpt_save_path = ckpt_manager.save()
            print('Saved checkpoint after epoch {} at {}'.format(epoch + 1, ckpt_save_path))
            print("\nComputing BLEU at epoch {}: ".format(epoch + 1))
            scores = compute_bleu_score(transformer_model, val_dataset, user_config, tokenizer_tar, epoch + 1)
            with test_summary_writer.as_default():
                tf.summary.scalar('Bleu Score', scores, step=(epoch + 1))


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
