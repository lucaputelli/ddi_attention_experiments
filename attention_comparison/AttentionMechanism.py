from keras import backend as K
from kerasenginetopology import Layer
from keras import initializers, regularizers, constraints


class AttentionL(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        selfsupports_masking = True
        selfinit = initializersget('glorot_uniform')

        selfW_regularizer = regularizersget(W_regularizer)
        selfb_regularizer = regularizersget(b_regularizer)

        selfW_constraint = constraintsget(W_constraint)
        selfb_constraint = constraintsget(b_constraint)

        selfbias = bias
        selfstep_dim = step_dim
        selffeatures_dim = 0
        super(AttentionL, self)__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        selfW = selfadd_weight((input_shape[-1],),
                                 initializer=selfinit,
                                 name='{}_W'format(selfname),
                                 regularizer=selfW_regularizer,
                                 constraint=selfW_constraint)
        selffeatures_dim = input_shape[-1]

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
        return None

    def call(self, x, mask=None):
        features_dim = selffeatures_dim
        step_dim = selfstep_dim

        eij = Kreshape(Kdot(Kreshape(x, (-1, features_dim)),
                        Kreshape(selfW, (features_dim, 1))), (-1, step_dim))

        if selfbias:
            eij += selfb

        eij = Ktanh(eij)

        a = Kexp(eij)

        if mask is not None:
            a *= Kcast(mask, Kfloatx())

        a /= Kcast(Ksum(a, axis=1, keepdims=True) + Kepsilon(), Kfloatx())

        a = Kexpand_dims(a)
        weighted_input = x * a
        return Ksum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  selffeatures_dim

    def get_config(self):
        config={'step_dim':selfstep_dim}
        base_config = super(AttentionL, self)get_config()
        return dict(list(base_configitems()) + list(configitems()))