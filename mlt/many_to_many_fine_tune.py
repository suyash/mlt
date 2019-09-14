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


def prepare_datasets(batch_size, dataset_size=25000):
    en_fr = tfds.load("para_crawl/enfr_plain_text",
                      as_supervised=True,
                      split=tfds.Split.TRAIN,
                      as_dataset_kwargs=dict(shuffle_files=True),
                      data_dir=flags.FLAGS.tfds_data_dir)
    en_de = tfds.load("para_crawl/ende_plain_text",
                      as_supervised=True,
                      split=tfds.Split.TRAIN,
                      as_dataset_kwargs=dict(shuffle_files=True),
                      data_dir=flags.FLAGS.tfds_data_dir)

    fr_pt = tfds.load("ted_hrlr_translate/fr_to_pt",
                      as_supervised=True,
                      as_dataset_kwargs=dict(shuffle_files=True),
                      data_dir=flags.FLAGS.tfds_data_dir)
    pt_en = tfds.load("ted_hrlr_translate/pt_to_en",
                      as_supervised=True,
                      as_dataset_kwargs=dict(shuffle_files=True),
                      data_dir=flags.FLAGS.tfds_data_dir)

    train_en_fr = en_fr.take(dataset_size)
    train_fr_en = en_fr.skip(dataset_size).take(dataset_size).map(
        lambda a, b: (b, a))

    train_en_de = en_de.take(dataset_size)
    train_de_en = en_de.skip(dataset_size).take(dataset_size).map(
        lambda a, b: (b, a))

    train_pt_en = pt_en[tfds.Split.TRAIN]
    train_en_pt = pt_en[tfds.Split.TRAIN].map(lambda a, b: (b, a))

    val_pt_en = pt_en[tfds.Split.VALIDATION]

    train_fr_pt = fr_pt[tfds.Split.TRAIN]
    train_pt_fr = fr_pt[tfds.Split.TRAIN].map(lambda a, b: (b, a))

    val_fr_pt = fr_pt[tfds.Split.VALIDATION]

    with tf.io.gfile.GFile(flags.FLAGS.encoding_model_file, "rb") as f:
        encoding_model_proto = f.read()

    train_en_fr = train_en_fr.map(lambda a, b:
                                  (tfs.encode(tf.expand_dims(a, 0),
                                              model_proto=encoding_model_proto,
                                              add_bos=True,
                                              add_eos=True)[0][0],
                                   tfs.encode(tf.expand_dims(b, 0),
                                              model_proto=encoding_model_proto,
                                              add_bos=True,
                                              add_eos=True)[0][0]))

    train_fr_en = train_fr_en.map(lambda a, b:
                                  (tfs.encode(tf.expand_dims(a, 0),
                                              model_proto=encoding_model_proto,
                                              add_bos=True,
                                              add_eos=True)[0][0],
                                   tfs.encode(tf.expand_dims(b, 0),
                                              model_proto=encoding_model_proto,
                                              add_bos=True,
                                              add_eos=True)[0][0]))

    train_en_de = train_en_de.map(lambda a, b:
                                  (tfs.encode(tf.expand_dims(a, 0),
                                              model_proto=encoding_model_proto,
                                              add_bos=True,
                                              add_eos=True)[0][0],
                                   tfs.encode(tf.expand_dims(b, 0),
                                              model_proto=encoding_model_proto,
                                              add_bos=True,
                                              add_eos=True)[0][0]))

    train_de_en = train_de_en.map(lambda a, b:
                                  (tfs.encode(tf.expand_dims(a, 0),
                                              model_proto=encoding_model_proto,
                                              add_bos=True,
                                              add_eos=True)[0][0],
                                   tfs.encode(tf.expand_dims(b, 0),
                                              model_proto=encoding_model_proto,
                                              add_bos=True,
                                              add_eos=True)[0][0]))

    train_pt_en = train_pt_en.map(lambda a, b:
                                  (tfs.encode(tf.expand_dims(a, 0),
                                              model_proto=encoding_model_proto,
                                              add_bos=True,
                                              add_eos=True)[0][0],
                                   tfs.encode(tf.expand_dims(b, 0),
                                              model_proto=encoding_model_proto,
                                              add_bos=True,
                                              add_eos=True)[0][0]))

    train_en_pt = train_en_pt.map(lambda a, b:
                                  (tfs.encode(tf.expand_dims(a, 0),
                                              model_proto=encoding_model_proto,
                                              add_bos=True,
                                              add_eos=True)[0][0],
                                   tfs.encode(tf.expand_dims(b, 0),
                                              model_proto=encoding_model_proto,
                                              add_bos=True,
                                              add_eos=True)[0][0]))

    val_pt_en = val_pt_en.map(lambda a, b:
                              (tfs.encode(tf.expand_dims(a, 0),
                                          model_proto=encoding_model_proto,
                                          add_bos=True,
                                          add_eos=True)[0][0],
                               tfs.encode(tf.expand_dims(b, 0),
                                          model_proto=encoding_model_proto,
                                          add_bos=True,
                                          add_eos=True)[0][0]))

    train_pt_fr = train_pt_fr.map(lambda a, b:
                                  (tfs.encode(tf.expand_dims(a, 0),
                                              model_proto=encoding_model_proto,
                                              add_bos=True,
                                              add_eos=True)[0][0],
                                   tfs.encode(tf.expand_dims(b, 0),
                                              model_proto=encoding_model_proto,
                                              add_bos=True,
                                              add_eos=True)[0][0]))

    train_fr_pt = train_fr_pt.map(lambda a, b:
                                  (tfs.encode(tf.expand_dims(a, 0),
                                              model_proto=encoding_model_proto,
                                              add_bos=True,
                                              add_eos=True)[0][0],
                                   tfs.encode(tf.expand_dims(b, 0),
                                              model_proto=encoding_model_proto,
                                              add_bos=True,
                                              add_eos=True)[0][0]))

    val_fr_pt = val_fr_pt.map(lambda a, b:
                              (tfs.encode(tf.expand_dims(a, 0),
                                          model_proto=encoding_model_proto,
                                          add_bos=True,
                                          add_eos=True)[0][0],
                               tfs.encode(tf.expand_dims(b, 0),
                                          model_proto=encoding_model_proto,
                                          add_bos=True,
                                          add_eos=True)[0][0]))

    train_en_fr = train_en_fr.filter(lambda a, b: tf.logical_and(
        tf.size(a) < (flags.FLAGS.seq_len + 3),
        tf.size(b) < (flags.FLAGS.seq_len + 3)))

    train_fr_en = train_fr_en.filter(lambda a, b: tf.logical_and(
        tf.size(a) < (flags.FLAGS.seq_len + 3),
        tf.size(b) < (flags.FLAGS.seq_len + 3)))

    train_en_de = train_en_de.filter(lambda a, b: tf.logical_and(
        tf.size(a) < (flags.FLAGS.seq_len + 3),
        tf.size(b) < (flags.FLAGS.seq_len + 3)))

    train_de_en = train_de_en.filter(lambda a, b: tf.logical_and(
        tf.size(a) < (flags.FLAGS.seq_len + 3),
        tf.size(b) < (flags.FLAGS.seq_len + 3)))

    train_fr_pt = train_fr_pt.filter(lambda a, b: tf.logical_and(
        tf.size(a) < (flags.FLAGS.seq_len + 3),
        tf.size(b) < (flags.FLAGS.seq_len + 3)))

    train_pt_fr = train_pt_fr.filter(lambda a, b: tf.logical_and(
        tf.size(a) < (flags.FLAGS.seq_len + 3),
        tf.size(b) < (flags.FLAGS.seq_len + 3)))

    train_en_pt = train_en_pt.filter(lambda a, b: tf.logical_and(
        tf.size(a) < (flags.FLAGS.seq_len + 3),
        tf.size(b) < (flags.FLAGS.seq_len + 3)))

    train_pt_en = train_pt_en.filter(lambda a, b: tf.logical_and(
        tf.size(a) < (flags.FLAGS.seq_len + 3),
        tf.size(b) < (flags.FLAGS.seq_len + 3)))

    # en: 0, fr: 1, de: 2, pt: 3

    train_en_fr = train_en_fr.map(lambda a, b: (
        (a, [1.0, 0.0, 0.0, 0.0], b[:-1], [0.0, 1.0, 0.0, 0.0]), b[1:]))
    train_fr_en = train_fr_en.map(lambda a, b: (
        (a, [0.0, 1.0, 0.0, 0.0], b[:-1], [1.0, 0.0, 0.0, 0.0]), b[1:]))

    train_en_de = train_en_de.map(lambda a, b: (
        (a, [1.0, 0.0, 0.0, 0.0], b[:-1], [0.0, 0.0, 1.0, 0.0]), b[1:]))
    train_de_en = train_de_en.map(lambda a, b: (
        (a, [0.0, 0.0, 1.0, 0.0], b[:-1], [1.0, 0.0, 0.0, 0.0]), b[1:]))

    train_fr_pt = train_fr_pt.map(lambda a, b: (
        (a, [0.0, 1.0, 0.0, 0.0], b[:-1], [0.0, 0.0, 0.0, 1.0]), b[1:]))
    train_pt_fr = train_pt_fr.map(lambda a, b: (
        (a, [0.0, 0.0, 0.0, 1.0], b[:-1], [0.0, 1.0, 0.0, 0.0]), b[1:]))

    train_en_pt = train_en_pt.map(lambda a, b: (
        (a, [1.0, 0.0, 0.0, 0.0], b[:-1], [0.0, 0.0, 0.0, 1.0]), b[1:]))
    train_pt_en = train_pt_en.map(lambda a, b: (
        (a, [0.0, 0.0, 0.0, 1.0], b[:-1], [1.0, 0.0, 0.0, 0.0]), b[1:]))

    val_fr_pt = val_fr_pt.map(lambda a, b: (
        (a, [0.0, 1.0, 0.0, 0.0], b[:-1], [0.0, 0.0, 0.0, 1.0]), b[1:]))

    val_pt_en = val_pt_en.map(lambda a, b: (
        (a, [0.0, 0.0, 0.0, 1.0], b[:-1], [1.0, 0.0, 0.0, 0.0]), b[1:]))

    train_data = train_en_fr.concatenate(train_fr_en).concatenate(
        train_en_de).concatenate(train_de_en).concatenate(
            train_fr_pt).concatenate(train_pt_fr).concatenate(
                train_en_pt).concatenate(train_pt_en)

    val_data = val_fr_pt.concatenate(val_pt_en)

    train_data = train_data.cache()
    train_data = train_data.shuffle(flags.FLAGS.shuffle_buffer_size)
    train_data = train_data.padded_batch(batch_size,
                                         padded_shapes=(((-1, ), (-1, ),
                                                         (-1, ), (-1, )),
                                                        (-1, )))
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
    train_data = train_data.repeat()

    val_data = val_data.padded_batch(batch_size,
                                     padded_shapes=(((-1, ), (-1, ), (-1, ),
                                                     (-1, )), (-1, )))

    return train_data, val_data


def main(_):
    strategy = tf.distribute.MirroredStrategy()

    logging.info("Number of Devices: %d", strategy.num_replicas_in_sync)
    validation_steps = flags.FLAGS.validation_steps // strategy.num_replicas_in_sync
    batch_size = flags.FLAGS.batch_size_per_replica * strategy.num_replicas_in_sync

    train_data, val_data = prepare_datasets(batch_size, dataset_size=50000)

    with strategy.scope():
        num_enc_factors = 4
        num_dec_factors = 4

        vocab_size = 40960

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
    app.flags.DEFINE_string("encoding_model_file",
                            "sentencepiece/en_de_fr_pt_shared/encoding.model",
                            "en_model_file")
    app.flags.DEFINE_boolean("tensorboard", False, "tensorboard")
    app.flags.DEFINE_boolean("best_checkpoints", False, "best_checkpoints")
    app.flags.DEFINE_string("job-dir", "runs/one_to_one/test", "job")
    app.run(main)
