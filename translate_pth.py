# Translate sentences from a .pth dataset file.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
#
# Usage:
#     python translate_pth.py --input source_data.pth \
#     --exp_name translate \
#     --src_lang en --tgt_lang fr \
#     --model_path trained_model.pth --output_path output
#

import os
import io
import sys
import argparse
import torch

from src.utils import AttrDict
from src.utils import bool_flag, initialize_exp, to_cuda
from src.data.dictionary import Dictionary
from src.data.dataset import Dataset
from src.model.transformer import TransformerModel

from src.fp16 import network_to_half


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate sentences")

    # main parameters
    parser.add_argument("--input", type=str, help="Input dataset file")

    parser.add_argument("--dump_path", type=str, default="./dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--fp16", type=bool_flag, default=False, help="Run model with float16")

    # model / output paths
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    parser.add_argument("--output_path", type=str, default="", help="Output path")

    # parser.add_argument("--max_vocab", type=int, default=-1, help="Maximum vocabulary size (-1 to disable)")
    # parser.add_argument("--min_count", type=int, default=0, help="Minimum vocabulary count")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")


    # beam search (for MT only)
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--early_stopping", type=bool_flag, default=False,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # batch parameters
    parser.add_argument("--bptt", type=int, default=256,
                        help="Sequence length")
    parser.add_argument("--max_len", type=int, default=100,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
    parser.add_argument("--tokens_per_batch", type=int, default=-1,
                        help="Number of tokens per batch")

    # dataset subset
    parser.add_argument("--subset_start", type=int, default=None)
    parser.add_argument("--subset_end", type=int, default=None)

    return parser


def main(params):

    # initialize the experiment
    logger = initialize_exp(params)

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    reloaded = torch.load(params.model_path)
    model_params = AttrDict(reloaded['params'])
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

    # update dictionary parameters
    for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
        setattr(params, name, getattr(model_params, name))

    # build dictionary / build encoder / build decoder / reload weights
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    encoder = TransformerModel(model_params, dico, is_encoder=True, with_output=True).cuda().eval()
    decoder = TransformerModel(model_params, dico, is_encoder=False, with_output=True).cuda().eval()
    encoder.load_state_dict(reloaded['encoder'])
    decoder.load_state_dict(reloaded['decoder'])
    params.src_id = model_params.lang2id[params.src_lang]
    params.tgt_id = model_params.lang2id[params.tgt_lang]

    # float16
    if params.fp16:
        assert torch.backends.cudnn.enabled
        encoder = network_to_half(encoder)
        decoder = network_to_half(decoder)

    input_data = torch.load(params.input)
    eval_dataset = Dataset(input_data["sentences"], input_data["positions"], params)

    if params.subset_start is not None:
        assert params.subset_end
        eval_dataset.select_data(params.subset_start, params.subset_end)

    eval_dataset.remove_empty_sentences()
    eval_dataset.remove_long_sentences(params.max_len)

    n_batch = 0

    out = io.open(params.output_path, "w", encoding="utf-8")
    inp_dump = io.open(os.path.join(params.dump_path, "input.txt"), "w", encoding="utf-8")
    logger.info("logging to {}".format(os.path.join(params.dump_path, 'input.txt')))

    with open(params.output_path, "w", encoding="utf-8") as out:

        for batch in eval_dataset.get_iterator(shuffle=False):
            n_batch += 1

            (x1, len1) = batch
            input_text = convert_to_text(x1, len1, input_data["dico"], params)
            inp_dump.write("\n".join(input_text))
            inp_dump.write("\n")

            langs1 = x1.clone().fill_(params.src_id)

            # cuda
            x1, len1, langs1 = to_cuda(x1, len1, langs1)

            # encode source sentence
            enc1 = encoder("fwd", x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)

            # generate translation - translate / convert to text
            max_len = int(1.5 * len1.max().item() + 10)
            if params.beam_size == 1:
                generated, lengths = decoder.generate(enc1, len1, params.tgt_id, max_len=max_len)
            else:
                generated, lengths = decoder.generate_beam(
                    enc1, len1, params.tgt_id, beam_size=params.beam_size,
                    length_penalty=params.length_penalty,
                    early_stopping=params.early_stopping,
                    max_len=max_len)

            hypotheses_batch = convert_to_text(generated, lengths, input_data["dico"], params)

            out.write("\n".join(hypotheses_batch))
            out.write("\n")

            if n_batch % 100 == 0:
                logger.info("{} batches processed".format(n_batch))

    out.close()
    inp_dump.close()


def convert_to_text(batch, lengths, dico, params):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == params.eos_index).sum() == bs
    assert (batch == params.eos_index).sum() == 2 * bs
    sentences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sentences.append(" ".join(words))
    return sentences


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    assert os.path.isfile(params.model_path)
    assert params.src_lang != '' and params.tgt_lang != '' and params.src_lang != params.tgt_lang
    assert params.output_path and not os.path.isfile(params.output_path)

    # translate
    with torch.no_grad():
        main(params)
