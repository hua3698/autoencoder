import math
from library.common import *
from library.functions import *

my_callbacks = [
    tensorflow.keras.callbacks.EarlyStopping(patience=100,monitor = 'loss',mode='min',min_delta=0.0001,restore_best_weights=True),
    tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,min_delta=0.0001,mode='min')
]

my_callbacks2 = [
    tensorflow.keras.callbacks.EarlyStopping(patience=50,monitor = 'accuracy',mode='max',min_delta=0.0001,restore_best_weights=True)
]

def train_ae_210(x_train, x_test, size):

    input_dim = x_train.shape[1]
    input_layer = Input(shape = (input_dim, ))

    encoded = BatchNormalization()(input_layer)
    encoded = Dense(math.ceil(input_dim*0.9), activation='relu')(encoded)

    encoded = BatchNormalization()(encoded)
    encoded_end = Dense(math.ceil(input_dim*0.8), activation='relu')(encoded)

    decoded = BatchNormalization()(encoded_end)
    decoded = Dense(math.ceil(input_dim*0.9), activation='relu')(decoded)

    decoded = BatchNormalization()(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    # 訓練
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x_train, x_train,
            epochs=100,
            batch_size=size,
            shuffle=True,
            callbacks=my_callbacks,
            validation_data=(x_test, x_test))

    encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)

    x_train_encoded = encoder_model.predict(x_train)
    x_test_encoded = encoder_model.predict(x_test)

    return x_train_encoded,x_test_encoded

def train_ae_220(x_train, x_test, size):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(math.ceil(input_dim*0.6), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='mse')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=size,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_ae_230(x_train, x_test, size):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.7), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(math.ceil(input_dim*0.4), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='mse')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=size,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_ae_240(x_train, x_test, size):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(math.ceil(input_dim*0.2), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='mse')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=size,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_ae_310(x_train, x_test, size):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.9), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(math.ceil(input_dim*0.7), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='mse')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=size,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_ae_320(x_train, x_test, size):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(math.ceil(input_dim*0.4), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='mse')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=size,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_ae_330(x_train, x_test, size):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.7), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.4), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(math.ceil(input_dim*0.1), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.4), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='mse')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=size,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_ae_410(x_train, x_test, size):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.9), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.7), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(math.ceil(input_dim*0.6), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='mse')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=size,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_ae_420(x_train, x_test, size):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.4), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(math.ceil(input_dim*0.2), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.4), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='mse')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=size,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_ae_510(x_train, x_test, size):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.9), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.7), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(math.ceil(input_dim*0.5), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='mse')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=size,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_ae_610(x_train, x_test, size):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.9), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.7), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.5), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(math.ceil(input_dim*0.4), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.5), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='mse')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=size,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_ae_710(x_train, x_test, size):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.9), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.7), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.5), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.4), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(math.ceil(input_dim*0.3), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.4), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.5), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(math.ceil(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='mse')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=size,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded