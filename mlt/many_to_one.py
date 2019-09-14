import os

from absl import app, flags, logging
import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard  # pylint: disable=import-error
from tensorflow.keras.layers import Input  # pylint: disable=import-error
import tensorflow_datasets as tfds
import tf_sentencepiece as tfs

from .losses import SparseSoftmaxCrossentropyWithMaskedPadding
from .schedules import CustomSchedule
from .transformer import TransformerWithTiedEmbedding


def prepare_datasets(batch_size, dataset_size=500000):
    en_fr = tfds.load("para_crawl/enfr_plain_text",
                      as_supervised=True,
                      split=tfds.Split.TRAIN,
                      as_dataset_kwargs=dict(shuffle_files=True),
                      data_dir=flags.FLAGS.tfds_data_dir)
    fr_en = en_fr.map(lambda a, b: (b, a))

    en_de = tfds.load("para_crawl/ende_plain_text",
                      as_supervised=True,
                      split=tfds.Split.TRAIN,
                      as_dataset_kwargs=dict(shuffle_files=True),
                      data_dir=flags.FLAGS.tfds_data_dir)
    de_en = en_de.map(lambda a, b: (b, a))

    en_es = tfds.load("para_crawl/enes_plain_text",
                      as_supervised=True,
                      split=tfds.Split.TRAIN,
                      as_dataset_kwargs=dict(shuffle_files=True),
                      data_dir=flags.FLAGS.tfds_data_dir)
    es_en = en_es.map(lambda a, b: (b, a))

    en_it = tfds.load("para_crawl/enit_plain_text",
                      as_supervised=True,
                      split=tfds.Split.TRAIN,
                      as_dataset_kwargs=dict(shuffle_files=True),
                      data_dir=flags.FLAGS.tfds_data_dir)
    it_en = en_it.map(lambda a, b: (b, a))

    val_fr_en = fr_en.take(1000)
    train_fr_en = fr_en.skip(1000).take(dataset_size)

    val_de_en = de_en.take(1000)
    train_de_en = de_en.skip(1000).take(dataset_size)

    val_es_en = es_en.take(1000)
    train_es_en = es_en.skip(1000).take(dataset_size)

    val_it_en = it_en.take(1000)
    train_it_en = it_en.skip(1000).take(dataset_size)

    with tf.io.gfile.GFile(flags.FLAGS.en_model_file, "rb") as f:
        en_model_proto = f.read()

    with tf.io.gfile.GFile(flags.FLAGS.de_model_file, "rb") as f:
        de_model_proto = f.read()

    with tf.io.gfile.GFile(flags.FLAGS.fr_model_file, "rb") as f:
        fr_model_proto = f.read()

    with tf.io.gfile.GFile(flags.FLAGS.es_model_file, "rb") as f:
        es_model_proto = f.read()

    with tf.io.gfile.GFile(flags.FLAGS.it_model_file, "rb") as f:
        it_model_proto = f.read()

    en_offset = tf.constant(0)
    fr_offset = tfs.piece_size(model_proto=en_model_proto)
    de_offset = fr_offset + tfs.piece_size(model_proto=fr_model_proto)
    es_offset = de_offset + tfs.piece_size(model_proto=de_model_proto)
    it_offset = es_offset + tfs.piece_size(model_proto=es_model_proto)

    train_de_en = encode_sentencepiece(train_de_en,
                                       a_model_proto=de_model_proto,
                                       b_model_proto=en_model_proto,
                                       a_offset=de_offset,
                                       b_offset=en_offset)

    val_de_en = encode_sentencepiece(val_de_en,
                                     a_model_proto=de_model_proto,
                                     b_model_proto=en_model_proto,
                                     a_offset=de_offset,
                                     b_offset=en_offset)

    train_fr_en = encode_sentencepiece(train_fr_en,
                                       a_model_proto=fr_model_proto,
                                       b_model_proto=en_model_proto,
                                       a_offset=fr_offset,
                                       b_offset=en_offset)

    val_fr_en = encode_sentencepiece(val_fr_en,
                                     a_model_proto=fr_model_proto,
                                     b_model_proto=en_model_proto,
                                     a_offset=fr_offset,
                                     b_offset=en_offset)

    train_es_en = encode_sentencepiece(train_es_en,
                                       a_model_proto=es_model_proto,
                                       b_model_proto=en_model_proto,
                                       a_offset=es_offset,
                                       b_offset=en_offset)

    val_es_en = encode_sentencepiece(val_es_en,
                                     a_model_proto=es_model_proto,
                                     b_model_proto=en_model_proto,
                                     a_offset=es_offset,
                                     b_offset=en_offset)

    train_it_en = encode_sentencepiece(train_it_en,
                                       a_model_proto=it_model_proto,
                                       b_model_proto=en_model_proto,
                                       a_offset=it_offset,
                                       b_offset=en_offset)

    val_it_en = encode_sentencepiece(val_it_en,
                                     a_model_proto=it_model_proto,
                                     b_model_proto=en_model_proto,
                                     a_offset=it_offset,
                                     b_offset=en_offset)

    train_de_en = train_de_en.filter(lambda a, b: tf.logical_and(
        tf.size(a) < (flags.FLAGS.seq_len + 3),
        tf.size(b) < (flags.FLAGS.seq_len + 3)))

    train_fr_en = train_fr_en.filter(lambda a, b: tf.logical_and(
        tf.size(a) < (flags.FLAGS.seq_len + 3),
        tf.size(b) < (flags.FLAGS.seq_len + 3)))

    train_es_en = train_es_en.filter(lambda a, b: tf.logical_and(
        tf.size(a) < (flags.FLAGS.seq_len + 3),
        tf.size(b) < (flags.FLAGS.seq_len + 3)))

    train_it_en = train_it_en.filter(lambda a, b: tf.logical_and(
        tf.size(a) < (flags.FLAGS.seq_len + 3),
        tf.size(b) < (flags.FLAGS.seq_len + 3)))

    # de: 0, fr: 1, es: 2, it: 3

    train_de_en = train_de_en.map(
        lambda a, b: ((a, [1.0, 0.0, 0.0, 0.0], b[:-1], [1.0]), b[1:]))
    val_de_en = val_de_en.map(
        lambda a, b: ((a, [1.0, 0.0, 0.0, 0.0], b[:-1], [1.0]), b[1:]))

    train_fr_en = train_fr_en.map(
        lambda a, b: ((a, [0.0, 1.0, 0.0, 0.0], b[:-1], [1.0]), b[1:]))
    val_fr_en = val_fr_en.map(
        lambda a, b: ((a, [0.0, 1.0, 0.0, 0.0], b[:-1], [1.0]), b[1:]))

    train_es_en = train_es_en.map(
        lambda a, b: ((a, [0.0, 0.0, 1.0, 0.0], b[:-1], [1.0]), b[1:]))
    val_es_en = val_es_en.map(
        lambda a, b: ((a, [0.0, 0.0, 1.0, 0.0], b[:-1], [1.0]), b[1:]))

    train_it_en = train_it_en.map(
        lambda a, b: ((a, [0.0, 0.0, 0.0, 1.0], b[:-1], [1.0]), b[1:]))
    val_it_en = val_it_en.map(
        lambda a, b: ((a, [0.0, 0.0, 0.0, 1.0], b[:-1], [1.0]), b[1:]))

    train_data = train_de_en.concatenate(train_fr_en).concatenate(
        train_es_en).concatenate(train_it_en)
    val_data = val_de_en.concatenate(val_fr_en).concatenate(
        val_es_en).concatenate(val_it_en)

    train_data = train_data.cache().shuffle(
        flags.FLAGS.shuffle_buffer_size).padded_batch(
            batch_size,
            padded_shapes=(((-1, ), (-1, ), (-1, ), (-1, )), (-1, )))
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE).repeat()

    val_data = val_data.padded_batch(batch_size,
                                     padded_shapes=(((-1, ), (-1, ), (-1, ),
                                                     (-1, )), (-1, )))

    return train_data, val_data


def encode_sentencepiece(dataset, a_model_proto, b_model_proto, a_offset,
                         b_offset):
    return dataset.map(lambda a, b: (
        tfs.encode(tf.expand_dims(a, 0),
                   model_proto=a_model_proto,
                   add_bos=True,
                   add_eos=True)[0][0] + a_offset,
        tfs.encode(tf.expand_dims(b, 0),
                   model_proto=b_model_proto,
                   add_bos=True,
                   add_eos=True)[0][0] + b_offset,
    ))


def main(_):
    strategy = tf.distribute.MirroredStrategy()

    logging.info("Number of Devices: %d", strategy.num_replicas_in_sync)
    validation_steps = flags.FLAGS.validation_steps // strategy.num_replicas_in_sync
    batch_size = flags.FLAGS.batch_size_per_replica * strategy.num_replicas_in_sync

    train_data, val_data = prepare_datasets(batch_size, dataset_size=62500)

    with strategy.scope():
        num_enc_factors = 4
        num_dec_factors = 1

        vocab_size = 8192 * 5

        src = Input((None, ), dtype="int32", name="src")
        srcf = Input((num_enc_factors, ), dtype="float32", name="srcf")
        tar = Input((None, ), dtype="int32", name="tar")
        tarf = Input((num_dec_factors, ), dtype="float32", name="tarf")

        o, _, _, _ = TransformerWithTiedEmbedding(
            num_layers=flags.FLAGS.num_layers,
            num_enc_factors=num_enc_factors,
            num_dec_factors=num_dec_factors,
            norm_axis=-1 if flags.FLAGS.normalization == "layer" else 1,
            d_model=flags.FLAGS.d_model,
            num_heads=flags.FLAGS.num_heads,
            d_ff=flags.FLAGS.d_ff,
            vocab_size=vocab_size,
            dropout_rate=flags.FLAGS.dropout_rate,
        )(src, srcf, tar, tarf)

        model = Model(inputs=[src, srcf, tar, tarf], outputs=o)

        if flags.FLAGS.initial_model_weights != None:
            logging.info("Loading Weights from %s",
                         flags.FLAGS.initial_model_weights)
            model.load_weights(flags.FLAGS.initial_model_weights)

        learning_rate = CustomSchedule(d_model=flags.FLAGS.d_model,
                                       initial_steps=flags.FLAGS.initial_steps)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = SparseSoftmaxCrossentropyWithMaskedPadding(mask_val=0)

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        callbacks = []

        if flags.FLAGS.tensorboard:
            callbacks.append(TensorBoard(log_dir=flags.FLAGS["job-dir"].value))

        if flags.FLAGS.best_checkpoints:
            callbacks.append(
                ModelCheckpoint(filepath=os.path.join(
                    flags.FLAGS["job-dir"].value, "best"),
                                monitor="loss",
                                save_weights_only=True,
                                save_best_only=True,
                                verbose=1))

    model.fit(train_data,
              epochs=flags.FLAGS.epochs,
              steps_per_epoch=flags.FLAGS.steps_per_epoch,
              validation_data=val_data,
              validation_steps=validation_steps,
              verbose=flags.FLAGS.fit_verbose,
              callbacks=callbacks)

    # tf.keras.experimental.export_saved_model(
    #     model,
    #     os.path.join(flags.FLAGS["job-dir"].value, "model"),
    #     custom_objects={"CustomSchedule": CustomSchedule})

    model.save_weights(
        os.path.join(flags.FLAGS["job-dir"].value, "model_weights"))


if __name__ == "__main__":
    print(tf.__version__)

    app.flags.DEFINE_enum("normalization", "layer", ["layer", "instance"],
                          "normalization")
    app.flags.DEFINE_integer("num_layers", 6, "num_layers")
    app.flags.DEFINE_integer("d_model", 512, "d_model")
    app.flags.DEFINE_integer("num_heads", 8, "num_heads")
    app.flags.DEFINE_integer("d_ff", 2048, "d_ff")
    app.flags.DEFINE_float("dropout_rate", 0.1, "dropout_rate")
    app.flags.DEFINE_integer("seq_len", 40, "seq_len")
    app.flags.DEFINE_integer("batch_size_per_replica", 60,
                             "batch_size_per_replica")
    app.flags.DEFINE_integer("shuffle_buffer_size", 40000,
                             "shuffle_buffer_size")
    app.flags.DEFINE_integer("epochs", 35, "epochs")
    app.flags.DEFINE_integer("fit_verbose", 1, "fit_verbose")
    app.flags.DEFINE_integer("steps_per_epoch", 500, "steps_per_epoch")
    app.flags.DEFINE_integer("validation_steps", 20, "validation_steps")
    app.flags.DEFINE_integer("initial_steps", 0, "initial_steps")
    app.flags.DEFINE_string("initial_model_weights", None,
                            "initial_model_weights")
    app.flags.DEFINE_string("tfds_data_dir", "~/tensorflow_datasets",
                            "tfds_data_dir")
    app.flags.DEFINE_string(
        "en_model_file",
        "sentencepiece/para_crawl/ende_plain_text/models/unigram/8192/a.model",
        "en_model_file")
    app.flags.DEFINE_string(
        "fr_model_file",
        "sentencepiece/para_crawl/enfr_plain_text/models/unigram/8192/b.model",
        "fr_model_file")
    app.flags.DEFINE_string(
        "de_model_file",
        "sentencepiece/para_crawl/ende_plain_text/models/unigram/8192/b.model",
        "de_model_file")
    app.flags.DEFINE_string(
        "es_model_file",
        "sentencepiece/para_crawl/enes_plain_text/models/unigram/8192/b.model",
        "es_model_file")
    app.flags.DEFINE_string(
        "it_model_file",
        "sentencepiece/para_crawl/enit_plain_text/models/unigram/8192/b.model",
        "it_model_file")
    app.flags.DEFINE_boolean("tensorboard", False, "tensorboard")
    app.flags.DEFINE_boolean("best_checkpoints", False, "best_checkpoints")
    app.flags.DEFINE_string("job-dir", "runs/one_to_one/test", "job")
    app.run(main)
