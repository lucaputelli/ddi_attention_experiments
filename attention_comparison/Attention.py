
from keras import backend as K, initializers, regularizers, constraints
from kerasenginetopology import Layer
import numpy as np

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if Kbackend() == 'tensorflow':
        return Ksqueeze(Kdot(x, Kexpand_dims(kernel)), axis=-1)
    else:
        return Kdot(x, kernel)


class CandidateContextAttention(Layer):

    def __init__(self,candidate,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        selfsupports_masking = True
        selfinit = initializersget('glorot_uniform')

        selfW_regularizer = regularizersget(W_regularizer)
        selfu_regularizer = regularizersget(u_regularizer)
        selfb_regularizer = regularizersget(b_regularizer)

        selfW_constraint = constraintsget(W_constraint)
        selfu_constraint = constraintsget(u_constraint)
        selfb_constraint = constraintsget(b_constraint)

        selfcandidate = Ktfconstant(candidate, dtype=Ktffloat32)
        selfbias = bias
        super(CandidateContextAttention, self)__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        selfW = selfadd_weight((input_shape[-1], input_shape[-1],),
                                 initializer=selfinit,
                                 name='{}_W'format(selfname),
                                 regularizer=selfW_regularizer,
                                 constraint=selfW_constraint)
        if selfbias:
            selfb = selfadd_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'format(selfname),
                                     regularizer=selfb_regularizer,
                                     constraint=selfb_constraint)
        super(CandidateContextAttention, self)build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, selfW)

        if selfbias:
            uit += selfb

        uit = Ktanh(uit)
        ait = dot_product(uit, selfcandidate)

        a = Kexp(ait)

        # apply mask after the exp will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= Kcast(mask, Kfloatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's A workaround is to add a very small positive number ε to the sum
        # a /= Kcast(Ksum(a, axis=1, keepdims=True), Kfloatx())
        a /= Kcast(Ksum(a, axis=1, keepdims=True) + Kepsilon(), Kfloatx())

        a = Kexpand_dims(a)
        weighted_input = x * a
        return weighted_input
        # return Ksum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class CandidateAttention(Layer):

    def __init__(self, candidate : nparray, **kwargs):
        selfcandidate = Ktfconstant(candidate, dtype=Ktffloat32)
        selffeature_dims = candidateshape[1]
        super(CandidateAttention, self)__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        selfbuilt = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        dot_score = Ktfmultiply(x, selfcandidate)
        scalar_product = Ksum(dot_score, axis=2)
        dot_score = Ktfdivide(scalar_product, selffeature_dims)
        # norm_x = Ktfnorm(x)
        # norm_c1 = Ktfnorm(selfcandidate)
        # cos_score = Ktfdivide(dot_score, norm_x)
        # cos_score = Ktfdivide(cos_score, norm_c1)
        a = Ksoftmax(dot_score)
        # a2 = Ksoftmax(cos_score)
        # selfa = a1 + a2 / 2
        selfcandidate_embedding = Ktfmultiply(x, Kexpand_dims(a, 2))
        # selfcandidate_embedding = Kdot(Ktfexpand_dims(selfa,0), x)
        # print(candidate_embeddingshape)
        return selfcandidate_embedding

    # def compute_output_shape(self, input_shape):
        # return selfcandidate_embeddingshape


class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data
        Supports Masking
        Follows the work of Raffel et al [https://arxivorg/abs/151208756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`
        # Output shape
            2D tensor with shape: `(samples, features)`
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True
        The dimensions are inferred based on the output shape of the RNN


        Note: The layer has been tested with Keras 1x

        Example:
        
            # 1
            modeladd(LSTM(64, return_sequences=True))
            modeladd(Attention())
            # next add a Dense layer (for classification/regression) or whatever

            # 2 - Get the attention scores
            hidden = LSTM(64, return_sequences=True)(words)
            sentence, word_scores = Attention(return_attention=True)(hidden)

        """
        selfsupports_masking = True
        selfreturn_attention = return_attention
        selfinit = initializersget('glorot_uniform')

        selfW_regularizer = regularizersget(W_regularizer)
        selfb_regularizer = regularizersget(b_regularizer)

        selfW_constraint = constraintsget(W_constraint)
        selfb_constraint = constraintsget(b_constraint)

        selfbias = bias
        super(Attention, self)__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        selfW = selfadd_weight((input_shape[-1],),
                                 initializer=selfinit,
                                 name='{}_W'format(selfname),
                                 regularizer=selfW_regularizer,
                                 constraint=selfW_constraint)
        if selfbias:
            selfb = selfadd_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'format(selfname),
                                     regularizer=selfb_regularizer,
                                     constraint=selfb_constraint)
        else:
            selfb = None

        selfbuilt = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, selfW)

        if selfbias:
            eij += selfb

        eij = Ktanh(eij)

        a = Kexp(eij)

        # apply mask after the exp will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= Kcast(mask, Kfloatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's A workaround is to add a very small positive number ε to the sum
        # a /= Kcast(Ksum(a, axis=1, keepdims=True), Kfloatx())
        a /= Kcast(Ksum(a, axis=1, keepdims=True) + Kepsilon(), Kfloatx())

        weighted_input = x * Kexpand_dims(a)

        result = Ksum(weighted_input, axis=1)

        if selfreturn_attention:
            return [result, a]
        return result
        # return weighted_input

    def compute_output_shape(self, input_shape):
        if selfreturn_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]
