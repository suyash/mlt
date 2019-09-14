import math

from tensorflow.keras.layers import Add, Dense, Dropout, Embedding, Permute, Reshape  # pylint: disable=import-error

from .layers import Attention, ConditionalNormalization, Gelu, MultiplyConstant, PaddingMask, PaddingAndLookaheadMask, PositionalEncoding


class TransformerWithTiedEmbedding:
    """
    A transformer modified to take weights of num_factors as inputs,
    as well as a common embedding layer for both inputs and targets.
    """
    def __init__(self,
                 num_layers,
                 num_enc_factors,
                 num_dec_factors,
                 norm_axis,
                 d_model,
                 num_heads,
                 d_ff,
                 vocab_size,
                 dropout_rate,
                 scope="transformer"):
        self.embedding = Embedding(input_dim=vocab_size,
                                   output_dim=d_model,
                                   name="%s/embedding" % scope)

        self.encoder = Encoder(num_layers=num_layers,
                               num_factors=num_enc_factors,
                               norm_axis=norm_axis,
                               d_model=d_model,
                               num_heads=num_heads,
                               d_ff=d_ff,
                               dropout_rate=dropout_rate,
                               scope="%s/encoder" % scope)

        self.decoder = Decoder(num_layers=num_layers,
                               num_factors=num_dec_factors,
                               norm_axis=norm_axis,
                               d_model=d_model,
                               num_heads=num_heads,
                               d_ff=d_ff,
                               dropout_rate=dropout_rate,
                               scope="%s/decoder" % scope)

        self.final_layer = Dense(vocab_size,
                                 activation=None,
                                 name="%s/dense" % scope)

        self.padding_mask = PaddingMask(name="%s/padding_mask" % scope)
        self.lookahead_mask = PaddingAndLookaheadMask(
            name="%s/lookahead_mask" % scope)

    def __call__(self, inputs, inp_factors, target, targ_factors):
        padding_mask = self.padding_mask(inputs)
        lookahead_mask = self.lookahead_mask(target)

        enc_output, enc_attention = self.encoder(self.embedding(inputs),
                                                 inp_factors, padding_mask)

        dec_output, dec_attention, enc_dec_attention = self.decoder(
            self.embedding(target), targ_factors, enc_output, lookahead_mask,
            padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, enc_attention, dec_attention, enc_dec_attention


class Encoder:
    def __init__(self,
                 num_layers,
                 num_factors,
                 norm_axis,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout_rate,
                 scope="encoder"):
        self.d_model = d_model
        self.num_layers = num_layers
        self.scope = scope
        self.pos_encoding = PositionalEncoding(d_model,
                                               name="%s/positional_encoding" %
                                               scope)

        self.enc_layers = [
            EncoderLayer(num_factors=num_factors,
                         norm_axis=norm_axis,
                         d_model=d_model,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         dropout_rate=dropout_rate,
                         scope="%s/encoder_layer_%d" % (scope, i))
            for i in range(num_layers)
        ]

        self.dropout = Dropout(dropout_rate, name="%s/dropout" % self.scope)

    def __call__(self, x, factors, padding_mask):
        x = MultiplyConstant(math.sqrt(self.d_model),
                             name="%s/multiply" % self.scope)(x)
        x = Add(name="%s/add" % self.scope)([x, self.pos_encoding(x)])
        x = self.dropout(x)

        enc_attention_weights = {}

        for i in range(self.num_layers):
            x, enc_attention = self.enc_layers[i](x, factors, padding_mask)
            enc_attention_weights["layer_%d" % i] = enc_attention

        return x, enc_attention_weights


class Decoder:
    def __init__(self,
                 num_layers,
                 num_factors,
                 norm_axis,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout_rate,
                 scope="decoder"):
        self.d_model = d_model
        self.num_layers = num_layers
        self.scope = scope

        self.pos_encoding = PositionalEncoding(d_model,
                                               name="%s/positional_encoding" %
                                               scope)

        self.dec_layers = [
            DecoderLayer(num_factors=num_factors,
                         norm_axis=norm_axis,
                         d_model=d_model,
                         num_heads=num_heads,
                         d_ff=d_ff,
                         dropout_rate=dropout_rate,
                         scope="%s/decoder_layer_%d" % (scope, i))
            for i in range(num_layers)
        ]

        self.dropout = Dropout(dropout_rate, name="%s/dropout" % self.scope)

    def __call__(self, x, factors, enc_output, lookahead_mask, padding_mask):
        x = MultiplyConstant(math.sqrt(self.d_model),
                             name="%s/multiply" % self.scope)(x)
        x = Add(name="%s/add" % self.scope)([x, self.pos_encoding(x)])
        x = self.dropout(x)

        dec_attention_weights = {}
        enc_dec_attention_weights = {}

        for i in range(self.num_layers):
            x, dec_attention, enc_dec_attention = self.dec_layers[i](
                x, factors, enc_output, lookahead_mask, padding_mask)

            dec_attention_weights["layer_%d" % i] = dec_attention
            enc_dec_attention_weights["layer_%d" % i] = enc_dec_attention

        return x, dec_attention_weights, enc_dec_attention_weights


class EncoderLayer:
    def __init__(self,
                 num_factors,
                 norm_axis,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout_rate,
                 scope="encoder_layer"):
        self.scope = scope

        self.mha1 = MultiHeadAttention(d_model,
                                       num_heads,
                                       scope="%s/multi_head_attention_1" %
                                       scope)
        self.ffn = PointwiseFeedForwardNetwork(
            d_model, d_ff, scope="%s/pointwise_feed_forward_network" % scope)

        self.norm1 = ConditionalNormalization(num_factors=num_factors,
                                              axis=norm_axis,
                                              epsilon=1e-6,
                                              name="%s/norm_1" % scope)
        self.norm2 = ConditionalNormalization(num_factors=num_factors,
                                              axis=norm_axis,
                                              epsilon=1e-6,
                                              name="%s/norm_2" % scope)

        self.dropout1 = Dropout(dropout_rate, name="%s/dropout_1" % scope)
        self.dropout2 = Dropout(dropout_rate, name="%s/dropout_2" % scope)

    def __call__(self, x, factors, padding_mask):
        out1, enc_enc_attention = self.mha1(x, x, x, padding_mask)
        out1 = self.dropout1(out1)
        x = Add(name="%s/add_1" % self.scope)([x, out1])
        x = self.norm1([x, factors])

        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)
        x = Add(name="%s/add_2" % self.scope)([x, ffn_output])
        x = self.norm2([x, factors])

        return x, enc_enc_attention


class DecoderLayer:
    def __init__(self,
                 num_factors,
                 norm_axis,
                 d_model,
                 num_heads,
                 d_ff,
                 dropout_rate,
                 scope="decoder_layer"):
        self.scope = scope

        self.mha1 = MultiHeadAttention(d_model,
                                       num_heads,
                                       scope="%s/multi_head_attention_1" %
                                       scope)
        self.mha2 = MultiHeadAttention(d_model,
                                       num_heads,
                                       scope="%s/multi_head_attention_2" %
                                       scope)
        self.ffn = PointwiseFeedForwardNetwork(
            d_model, d_ff, scope="%s/pointwise_feed_forward_network" % scope)

        self.norm1 = ConditionalNormalization(num_factors=num_factors,
                                              axis=norm_axis,
                                              epsilon=1e-6,
                                              name="%s/norm_1" % scope)
        self.norm2 = ConditionalNormalization(num_factors=num_factors,
                                              axis=norm_axis,
                                              epsilon=1e-6,
                                              name="%s/norm_2" % scope)
        self.norm3 = ConditionalNormalization(num_factors=num_factors,
                                              axis=norm_axis,
                                              epsilon=1e-6,
                                              name="%s/norm_3" % scope)

        self.dropout1 = Dropout(dropout_rate, name="%s/dropout_1" % scope)
        self.dropout2 = Dropout(dropout_rate, name="%s/dropout_2" % scope)
        self.dropout3 = Dropout(dropout_rate, name="%s/dropout_3" % scope)

    def __call__(self, x, factors, enc_output, lookahead_mask, padding_mask):
        out1, dec_dec_attention = self.mha1(x, x, x, lookahead_mask)
        out1 = self.dropout1(out1)
        x = Add(name="%s/add_1" % self.scope)([x, out1])
        x = self.norm1([x, factors])

        out2, enc_dec_attention = self.mha2(x, enc_output, enc_output,
                                            padding_mask)
        out2 = self.dropout2(out2)
        x = Add(name="%s/add_2" % self.scope)([x, out2])
        x = self.norm2([x, factors])

        ffn_output = self.ffn(x)
        ffn_output = self.dropout3(ffn_output)
        x = Add(name="%s/add_3" % self.scope)([x, ffn_output])
        x = self.norm3([x, factors])

        return x, dec_dec_attention, enc_dec_attention


class MultiHeadAttention:
    def __init__(self, d_model, num_heads, scope="multi_head_attention"):
        assert d_model % num_heads == 0

        self.wq = Dense(d_model, name="%s/dense_q" % scope)
        self.wk = Dense(d_model, name="%s/dense_k" % scope)
        self.wv = Dense(d_model, name="%s/dense_v" % scope)

        self.reshapeq = Reshape((-1, num_heads, d_model // num_heads),
                                name="%s/reshape_q" % scope)
        self.reshapek = Reshape((-1, num_heads, d_model // num_heads),
                                name="%s/reshape_k" % scope)
        self.reshapev = Reshape((-1, num_heads, d_model // num_heads),
                                name="%s/reshape_v" % scope)

        self.transposeq = Permute((2, 1, 3), name="%s/transpose_q" % scope)
        self.transposek = Permute((2, 1, 3), name="%s/transpose_k" % scope)
        self.transposev = Permute((2, 1, 3), name="%s/transpose_v" % scope)

        self.reshape_output = Reshape((-1, d_model),
                                      name="%s/reshape_output" % scope)

        self.transpose_output = Permute((2, 1, 3),
                                        name="%s/transpose_output" % scope)

        self.dense = Dense(d_model, name="%s/dense" % scope)

        self.attention = Attention(name="%s/attention" % scope)

    def __call__(self, q, k, v, mask):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.reshapeq(q)
        k = self.reshapek(k)
        v = self.reshapev(v)

        q = self.transposeq(q)
        k = self.transposek(k)
        v = self.transposev(v)

        x, attention_weights = self.attention([q, k, v, mask])

        x = self.transpose_output(x)
        x = self.reshape_output(x)
        x = self.dense(x)

        return x, attention_weights


class PointwiseFeedForwardNetwork:
    def __init__(self, d_model, d_ff, scope="pointwise_feed_forward_network"):
        self.dense_1 = Dense(d_ff, activation=None, name="%s/dense_1" % scope)
        self.act = Gelu(name="%s/gelu" % scope)
        self.dense_2 = Dense(d_model,
                             activation=None,
                             name="%s/dense_2" % scope)

    def __call__(self, x):
        x = self.dense_1(x)
        x = self.act(x)
        x = self.dense_2(x)
        return x
