import argparse
from data_loader import DataLoader
from transformer import create_masks
import utils
import tensorflow as tf
import os
import json

"""Evaluate"""


def sacrebleu_metric(model, pred_file_path, tokenizer_tar, dataset, max_length):
    with open(pred_file_path, "w", buffering=1) as f_pred:
        # evaluation faster in batches
        for batch, (inp_seq, _) in enumerate(dataset):
            print("Evaluating batch {}".format(batch))
            translated_batch = translate_batch(model, inp_seq, tokenizer_tar, max_length)
            for pred in translated_batch:
                f_pred.write(pred.strip() + "\n")


def translate_batch(model, inp, tokenizer_tar, max_length):
    output, _ = evaluate_batch(model, inp, tokenizer_tar, max_length)
    pred_sentences = tokenizer_tar.sequences_to_texts(output.numpy())
    pred_sentences = [x.split("<end>")[0].replace("<start>", "").strip() for x in pred_sentences]
    return pred_sentences


def evaluate_batch(model, inputs, tokenizer_tar, max_length):
    encoder_input = tf.convert_to_tensor(inputs)
    decoder_input = tf.expand_dims([tokenizer_tar.word_index["<start>"]] * inputs.shape[0], axis=1)
    output = decoder_input
    attention_weights = None

    for _ in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = model(encoder_input,
                                               output,
                                               False,
                                               enc_padding_mask,
                                               combined_mask,
                                               dec_padding_mask)


        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # print("predictions: ", predicted_id)

        # return the result if the predicted_id is equal to the end token
        if (predicted_id == tokenizer_tar.word_index["<end>"]).numpy().all():
            return output, attention_weights
            # return tf.squeeze(output, axis=0), attention_weights

        # concatenate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return output, attention_weights


def do_evaluation(user_config, input_file_path, target_file_path, pred_file_path):
    inp_language = user_config["inp_language"]
    target_language = user_config["target_language"]

    print("\n****Evaluating model from {} to {}****\n".format(inp_language, target_language))

    print("****Loading Sub-Word Tokenizers****")
    # load pre-trained tokenizer
    tokenizer_inp, tokenizer_tar = utils.load_tokenizers()

    print("****Initializing DataLoader****")
    # data loader
    test_dataloader = DataLoader(user_config["transformer_batch_size"],
                                 input_file_path,
                                 target_file_path,
                                 tokenizer_inp,
                                 tokenizer_tar,
                                 inp_language,
                                 target_language,
                                 False)
    test_dataset = test_dataloader.get_data_loader()

    print("****Loading transformer model****")
    # load model and optimizer
    transformer_model, optimizer, ckpt_manager = \
        utils.load_transformer_model(user_config, tokenizer_inp, tokenizer_tar)

    print("****Generating Translations****")
    sacrebleu_metric(transformer_model,
                     pred_file_path,
                     tokenizer_tar,
                     test_dataset,
                     tokenizer_tar.MAX_LENGTH
                     )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file containing training parameters", type=str, required=True)
    parser.add_argument("--input_file_path", help="File to generate translations for", type=str, required=True)
    parser.add_argument("--pred_file_path", help="Path to save predicted translations", type=str, required=True)
    parser.add_argument("--target_file_path",
                        help="Path to save true translations. If you already have true translations, "
                             "don't pass anything. Else this will overwrite file.",
                        type=str, default=None)
    args = parser.parse_args()

    assert os.path.isfile(args.input_file_path), f"invalid input file: {args.input_file_path}"
    if args.target_file_path is not None:
        assert os.path.isfile(args.target_file_path), f"invalid target file: {args.target_file_path}"

    user_config = utils.load_file(args.config)
    print(json.dumps(user_config, indent=2))
    seed = user_config["random_seed"]
    utils.set_seed(seed)

    # generate translations
    do_evaluation(user_config,
                  args.input_file_path,
                  None,
                  args.pred_file_path)

    if args.target_file_path is not None:
        print("\nComputing bleu score now...")
        # compute bleu score
        utils.compute_bleu(args.pred_file_path, args.target_file_path, print_all_scores=False)
    else:
        print("\nNot predicting bleu as --target_file_path was not provided")


if __name__ == "__main__":
    main()
