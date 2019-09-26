from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Conv2D, Embedding, Concatenate, Reshape
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing import text

class CLCNN_model(object):
  def __init__(self, input_shape, output_shape, embed_size=128, filter_sizes=(1,2,3,4,5), filter_num=64):
    vec_shape, num_shape, cat_shape = input_shape
    
    """ input layer """
    in_vecs = Input(shape=(vec_shape, ), name='vec_input')
    emb = Embedding(0xffff, embed_size)(in_vecs)
    emb_ex = Reshape((vec_shape, embed_size, 1))(emb)
    
    in_nums = Input(shape=(num_shape,), name='num_input')
    
    in_cats = Input(shape=(cat_shape,), name='cat_input') 
    
    """ convolution layer """
    vec_convs = []
    for filter_size in filter_sizes:
      conv = Conv2D(filter_num, (filter_size, embed_size), activation='relu')(emb_ex)
      pool = MaxPooling2D(pool_size=(vec_shape - filter_size + 1, 1))(conv)
      vec_convs.append(pool)
    
    # full convination
    convs = Concatenate()(vec_convs)
    convs = Reshape((filter_num * len(filter_sizes),))(convs)
    
    concat = Concatenate()([convs, in_nums, in_cats])
    fc1 = Dense(128, activation='relu')(concat)
    bn1 = BatchNormalization()(fc1)
    do1 = Dropout(0.5)(bn1)
    fc2 = Dense(64, activation='relu')(do1)
    bn2 = BatchNormalization()(fc2)
    do2 = Dropout(0.5)(bn2)

    # output layer
    outputs = Dense(1, activation='softmax', name='output')(do2)
    self.model = Model(inputs=[in_vecs, in_nums, in_cats], outputs=outputs)
    self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['acc'])
  
  def get_model(self):
    return self.model

clcnn = CLCNN_model()
model = clcnn.get_model()
model.summary()