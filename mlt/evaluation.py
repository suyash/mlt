"""
NOTE: this script __has__ to run in 1.14, since T2T beam_search does not support 2.0.
"""

import json

import numpy as np
import tensorflow as tf
import tf_sentencepiece as tfs
from tensor2tensor.utils.beam_search import beam_search

from .third_party import compute_bleu


def predict(model,
            inputs,
            inpf,
            tarf,
            bos_id,
            eos_id,
            beam_size,
            vocab_size,
            alpha=1.0,
            decode_length=40):
    """
    inputs: already int encoded set of inputs, [batch_size, ?], tf.int32
    """

    batch_size = inputs.shape[0]
    initial_ids = [bos_id] * batch_size

    enc_input = tf.expand_dims(inputs, 1)
    enc_input = tf.tile(enc_input, [1, beam_size, 1])
    enc_input = tf.reshape(enc_input, [batch_size * beam_size, -1])

    def symbols_to_logits(ids):
        logits = model([
            enc_input,
            tf.tile(tf.expand_dims(inpf, 0), [tf.shape(ids)[0], 1]),
            ids,
            tf.tile(tf.expand_dims(tarf, 0), [tf.shape(ids)[0], 1]),
        ])

        logits = logits[0][:, -1, :]
        return logits

    x = beam_search(symbols_to_logits,
                    initial_ids,
                    beam_size,
                    decode_length,
                    vocab_size,
                    alpha=alpha,
                    eos_id=eos_id)

    ids = x[0]
    probs = x[1]

    return ids, probs


def predict_batch(sess,
                  src,
                  model,
                  src_model_file,
                  tar_model_file,
                  src_offset,
                  tar_offset,
                  srcf,
                  tarf,
                  vocab_size,
                  single_vocab_size=8192,
                  batch_size=60):
    """
    sess: tf.Session
    src: list of strings
    model: tf.keras.Model
    """
    t = len(src)

    ans = []

    for i in range(t // batch_size):
        print(i)

        start = i * batch_size
        end = start + batch_size
        inp = src[start:end]

        a = tfs.encode(inp,
                       model_file=src_model_file,
                       add_bos=True,
                       add_eos=True)[0]

        if src_offset > 0:
            a_mask = tf.cast(tf.not_equal(a, 0), tf.int32) * src_offset
            a = a + a_mask

        ids, probs = predict(
            model=model,
            inputs=a,
            inpf=tf.constant(srcf),
            tarf=tf.constant(tarf),
            bos_id=tar_offset + 1,
            eos_id=tar_offset + 2,
            beam_size=5,
            vocab_size=vocab_size,
            alpha=1.0,
        )

        mask = tf.cast(tf.not_equal(ids, 0), tf.int32)
        seq_len = tf.reduce_sum(mask, axis=-1)

        if tar_offset > 0:
            ids = ids + mask * -tar_offset

        probs = tf.math.exp(probs)

        ids_, seq_len_ = sess.run([ids, seq_len])

        for cids, cseqlen in zip(list(ids_), list(seq_len_)):
            fids = tf.cast(
                tf.logical_and(tf.greater(cids, 0),
                               tf.less(cids, single_vocab_size)),
                tf.int32) * cids
            decoded = sess.run(
                tfs.decode(fids, cseqlen, model_file=tar_model_file))
            ans.append(decoded)

    return ans


def evaluate(sess,
             model,
             src_file,
             tar_file,
             out_file,
             src_model_file,
             tar_model_file,
             src_offset,
             tar_offset,
             srcf,
             tarf,
             vocab_size,
             single_vocab_size=8192,
             batch_size=100):
    with tf.io.gfile.GFile(src_file) as f:
        src = json.load(f)

    preds = predict_batch(
        sess,
        src=src,
        model=model,
        src_model_file=src_model_file,
        tar_model_file=tar_model_file,
        src_offset=src_offset,
        tar_offset=tar_offset,
        srcf=srcf,
        tarf=tarf,
        vocab_size=vocab_size,
        single_vocab_size=single_vocab_size,
        batch_size=batch_size,
    )

    preds = [[x.decode("utf-8") for x in s] for s in preds]

    with tf.io.gfile.GFile(out_file, "w") as f:
        json.dump(preds, f, ensure_ascii=False)

    with tf.io.gfile.GFile(tar_file) as f:
        tar = json.load(f)

    tar = tar[:len(preds)]

    references = [[s.split(" ") for s in x] for x in preds]
    translations = [s.split(" ") for s in tar]

    b1, _, _, _, _, _ = compute_bleu(references, translations, max_order=1)
    b2, _, _, _, _, _ = compute_bleu(references, translations, max_order=2)
    b3, _, _, _, _, _ = compute_bleu(references, translations, max_order=3)
    b4, _, _, _, _, _ = compute_bleu(references, translations, max_order=4)

    return b1, b2, b3, b4
