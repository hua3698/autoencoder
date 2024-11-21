
from library.common import *
from library.functions import *
from tensorflow.keras import losses
import math

my_callbacks = [
    tensorflow.keras.callbacks.EarlyStopping(patience=100,monitor = 'loss',mode='min',min_delta=0.0001,restore_best_weights=True),
    tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,min_delta=0.0001,mode='min')
]

my_callbacks2 = [
    tensorflow.keras.callbacks.EarlyStopping(patience=50,monitor = 'accuracy',mode='max',min_delta=0.0001,restore_best_weights=True)
]

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# 自定義 loss layer
class VAELossLayer(Layer):
    def __init__(self, input_dim, **kwargs):
        self.input_dim = input_dim
        super(VAELossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        input_layer, output_layer, z_mean, z_log_var = inputs
        mse = losses.MeanSquaredError()
        reconstruction_loss = self.input_dim * mse(input_layer, output_layer)
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        total_loss = K.mean(reconstruction_loss + kl_loss)
        self.add_loss(total_loss)
        return output_layer

def train_vae_210(x_train, x_test, size):
    input_dim = x_train.shape[1]
    latent_dim = math.ceil(input_dim * 0.8)
    
    # Encoder layers
    input_layer = Input(shape=(input_dim,))
    encoded = BatchNormalization()(input_layer)
    encoded = Dense(math.ceil(input_dim * 0.9), activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    
    z_mean = Dense(latent_dim)(encoded)
    z_log_var = Dense(latent_dim)(encoded)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder layers
    decoded = BatchNormalization()(z)
    decoded = Dense(math.ceil(input_dim * 0.9), activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    # 添加自定義 loss layer
    vae_output = VAELossLayer(input_dim)([input_layer, output_layer, z_mean, z_log_var])

    # 建立模型
    vae = Model(input_layer, vae_output)
    
    # 編譯和訓練
    vae.compile(optimizer='adam')
    vae.fit(x_train, x_train,
            epochs=100,
            batch_size=size,
            shuffle=True,
            callbacks=my_callbacks,
            validation_data=(x_test, x_test))

    # 編碼器
    vae_encoder = Model(input_layer, z_mean)
    x_train_encoded = vae_encoder.predict(x_train)
    x_test_encoded = vae_encoder.predict(x_test)

    return x_train_encoded, x_test_encoded

def train_vae_220(x_train, x_test, size):
    input_dim = x_train.shape[1]
    latent_dim = math.ceil(input_dim * 0.6)
    
    # Encoder layers
    input_layer = Input(shape=(input_dim,))
    encoded = BatchNormalization()(input_layer)
    encoded = Dense(math.ceil(input_dim * 0.8), activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    
    z_mean = Dense(latent_dim)(encoded)
    z_log_var = Dense(latent_dim)(encoded)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder layers
    decoded = BatchNormalization()(z)
    decoded = Dense(math.ceil(input_dim * 0.8), activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    # 添加自定義 loss layer
    vae_output = VAELossLayer(input_dim)([input_layer, output_layer, z_mean, z_log_var])

    # 建立模型
    vae = Model(input_layer, vae_output)
    
    # 編譯和訓練
    vae.compile(optimizer='adam')
    vae.fit(x_train, x_train,
            epochs=100,
            batch_size=size,
            shuffle=True,
            callbacks=my_callbacks,
            validation_data=(x_test, x_test))

    # 編碼器
    vae_encoder = Model(input_layer, z_mean)
    x_train_encoded = vae_encoder.predict(x_train)
    x_test_encoded = vae_encoder.predict(x_test)

    return x_train_encoded, x_test_encoded

def train_vae_230(x_train, x_test, size):
    input_dim = x_train.shape[1]
    latent_dim = math.ceil(input_dim * 0.4)
    
    # Encoder layers
    input_layer = Input(shape=(input_dim,))
    encoded = BatchNormalization()(input_layer)
    encoded = Dense(math.ceil(input_dim * 0.7), activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    
    z_mean = Dense(latent_dim)(encoded)
    z_log_var = Dense(latent_dim)(encoded)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder layers
    decoded = BatchNormalization()(z)
    decoded = Dense(math.ceil(input_dim * 0.7), activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    # 添加自定義 loss layer
    vae_output = VAELossLayer(input_dim)([input_layer, output_layer, z_mean, z_log_var])

    # 建立模型
    vae = Model(input_layer, vae_output)
    
    # 編譯和訓練
    vae.compile(optimizer='adam')
    vae.fit(x_train, x_train,
            epochs=100,
            batch_size=size,
            shuffle=True,
            callbacks=my_callbacks,
            validation_data=(x_test, x_test))

    # 編碼器
    vae_encoder = Model(input_layer, z_mean)
    x_train_encoded = vae_encoder.predict(x_train)
    x_test_encoded = vae_encoder.predict(x_test)

    return x_train_encoded, x_test_encoded

def train_vae_240(x_train, x_test, size):
    input_dim = x_train.shape[1]
    latent_dim = math.ceil(input_dim * 0.2)
    
    # Encoder layers
    input_layer = Input(shape=(input_dim,))
    encoded = BatchNormalization()(input_layer)
    encoded = Dense(math.ceil(input_dim * 0.6), activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    
    z_mean = Dense(latent_dim)(encoded)
    z_log_var = Dense(latent_dim)(encoded)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder layers
    decoded = BatchNormalization()(z)
    decoded = Dense(math.ceil(input_dim * 0.6), activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    # 添加自定義 loss layer
    vae_output = VAELossLayer(input_dim)([input_layer, output_layer, z_mean, z_log_var])

    # 建立模型
    vae = Model(input_layer, vae_output)
    
    # 編譯和訓練
    vae.compile(optimizer='adam')
    vae.fit(x_train, x_train,
            epochs=100,
            batch_size=size,
            shuffle=True,
            callbacks=my_callbacks,
            validation_data=(x_test, x_test))

    # 編碼器
    vae_encoder = Model(input_layer, z_mean)
    x_train_encoded = vae_encoder.predict(x_train)
    x_test_encoded = vae_encoder.predict(x_test)

    return x_train_encoded, x_test_encoded


def train_vae_310(x_train, x_test, size):
    input_dim = x_train.shape[1]

    latent_dim = math.ceil(input_dim*0.7)
    enc_mean = Dense(latent_dim)
    enc_log_var = Dense(latent_dim)

    input_layer = Input(shape=(input_dim,))

    encoded = BatchNormalization()(input_layer)
    encoded = Dense(math.ceil(input_dim*0.9), activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(math.ceil(input_dim*0.8), activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    
    z_mean = enc_mean(encoded)
    z_log_var = enc_log_var(encoded)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    decoded = BatchNormalization()(z)
    decoded = Dense(math.ceil(input_dim*0.8), activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(math.ceil(input_dim*0.9), activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    vae = Model(input_layer, output_layer)
    vae.add_loss(VAELossLayer()([input_layer, z_mean, z_log_var]))

    vae.compile(optimizer='adam')
    vae.fit(x_train, x_train,
        epochs=100,
        batch_size=size,
        shuffle=True,
        callbacks=my_callbacks,
        validation_data=(x_test, x_test))

    vae_encoder = Model(vae.input, z_mean)
    x_train_encoded = vae_encoder.predict(x_train)
    x_test_encoded = vae_encoder.predict(x_test)

    return x_train_encoded, x_test_encoded

def train_vae_320(x_train, x_test, size):
    input_dim = x_train.shape[1]

    latent_dim = math.ceil(input_dim*0.4)
    enc_mean = Dense(latent_dim)
    enc_log_var = Dense(latent_dim)

    input_layer = Input(shape=(input_dim,))

    encoded = BatchNormalization()(input_layer)
    encoded = Dense(math.ceil(input_dim*0.8), activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(math.ceil(input_dim*0.6), activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    
    z_mean = enc_mean(encoded)
    z_log_var = enc_log_var(encoded)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    decoded = BatchNormalization()(z)
    decoded = Dense(math.ceil(input_dim*0.6), activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(math.ceil(input_dim*0.8), activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    vae = Model(input_layer, output_layer)
    vae.add_loss(VAELossLayer()([input_layer, z_mean, z_log_var]))

    vae.compile(optimizer='adam')
    vae.fit(x_train, x_train,
        epochs=100,
        batch_size=size,
        shuffle=True,
        callbacks=my_callbacks,
        validation_data=(x_test, x_test))

    vae_encoder = Model(vae.input, z_mean)
    x_train_encoded = vae_encoder.predict(x_train)
    x_test_encoded = vae_encoder.predict(x_test)

    return x_train_encoded, x_test_encoded

def train_vae_330(x_train, x_test, size):
    input_dim = x_train.shape[1]

    latent_dim = math.ceil(input_dim*0.1)
    enc_mean = Dense(latent_dim)
    enc_log_var = Dense(latent_dim)

    input_layer = Input(shape=(input_dim,))

    encoded = BatchNormalization()(input_layer)
    encoded = Dense(math.ceil(input_dim*0.7), activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(math.ceil(input_dim*0.4), activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    
    z_mean = enc_mean(encoded)
    z_log_var = enc_log_var(encoded)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    decoded = BatchNormalization()(z)
    decoded = Dense(math.ceil(input_dim*0.4), activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(math.ceil(input_dim*0.7), activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    vae = Model(input_layer, output_layer)
    vae.add_loss(VAELossLayer()([input_layer, z_mean, z_log_var]))

    vae.compile(optimizer='adam')
    vae.fit(x_train, x_train,
        epochs=100,
        batch_size=size,
        shuffle=True,
        callbacks=my_callbacks,
        validation_data=(x_test, x_test))

    vae_encoder = Model(vae.input, z_mean)
    x_train_encoded = vae_encoder.predict(x_train)
    x_test_encoded = vae_encoder.predict(x_test)

    return x_train_encoded, x_test_encoded

def train_vae_410(x_train, x_test, size):
    input_dim = x_train.shape[1]

    latent_dim = math.ceil(input_dim*0.6)
    enc_mean = Dense(latent_dim)
    enc_log_var = Dense(latent_dim)

    input_layer = Input(shape=(input_dim,))

    encoded = BatchNormalization()(input_layer)
    encoded = Dense(math.ceil(input_dim*0.9), activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(math.ceil(input_dim*0.8), activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(math.ceil(input_dim*0.7), activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    
    z_mean = enc_mean(encoded)
    z_log_var = enc_log_var(encoded)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    decoded = BatchNormalization()(z)
    decoded = Dense(math.ceil(input_dim*0.7), activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(math.ceil(input_dim*0.8), activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(math.ceil(input_dim*0.9), activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    vae = Model(input_layer, output_layer)
    vae.add_loss(VAELossLayer()([input_layer, z_mean, z_log_var]))

    vae.compile(optimizer='adam')
    vae.fit(x_train, x_train,
        epochs=100,
        batch_size=size,
        shuffle=True,
        callbacks=my_callbacks,
        validation_data=(x_test, x_test))

    vae_encoder = Model(vae.input, z_mean)
    x_train_encoded = vae_encoder.predict(x_train)
    x_test_encoded = vae_encoder.predict(x_test)

    return x_train_encoded, x_test_encoded

def train_vae_420(x_train, x_test, size):

  input_dim = x_train.shape[1]

  latent_dim = math.ceil(input_dim*0.2)
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)

  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(math.ceil(input_dim*0.4), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  z_mean = enc_mean(encoded)
  z_log_var = enc_log_var(encoded)
  z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.4), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  vae = Model(input_layer, output_layer)

  #loss
  reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)

  vae.compile(optimizer='adam', loss='mse')
  vae.fit(x_train, x_train,
      epochs=100,
      batch_size=size,
      shuffle=True,
      callbacks=my_callbacks,
      validation_data=(x_test, x_test))

  vae_encoder = Model(vae.input, z_mean)
  x_train_encoded = vae_encoder.predict(x_train)
  x_test_encoded = vae_encoder.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_vae_510(x_train, x_test, size):

  input_dim = x_train.shape[1]

  latent_dim = math.ceil(input_dim*0.5)
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)

  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.9), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.7), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(math.ceil(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  z_mean = enc_mean(encoded)
  z_log_var = enc_log_var(encoded)
  z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  vae = Model(input_layer, output_layer)

  #loss
  reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)

  vae.compile(optimizer='adam', loss='mse')
  vae.fit(x_train, x_train,
      epochs=100,
      batch_size=size,
      shuffle=True,
      callbacks=my_callbacks,
      validation_data=(x_test, x_test))

  vae_encoder = Model(vae.input, z_mean)
  x_train_encoded = vae_encoder.predict(x_train)
  x_test_encoded = vae_encoder.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_vae_610(x_train, x_test, size):

  input_dim = x_train.shape[1]

  latent_dim = math.ceil(input_dim*0.4)
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)

  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.9), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(math.ceil(input_dim*0.7), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(math.ceil(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(math.ceil(input_dim*0.5), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  z_mean = enc_mean(encoded)
  z_log_var = enc_log_var(encoded)
  z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.5), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  vae = Model(input_layer, output_layer)

  #loss
  reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)

  vae.compile(optimizer='adam', loss='mse')
  vae.fit(x_train, x_train,
      epochs=100,
      batch_size=size,
      shuffle=True,
      callbacks=my_callbacks,
      validation_data=(x_test, x_test))

  vae_encoder = Model(vae.input, z_mean)
  x_train_encoded = vae_encoder.predict(x_train)
  x_test_encoded = vae_encoder.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_vae_710(x_train, x_test, size):

  input_dim = x_train.shape[1]

  latent_dim = math.ceil(input_dim*0.3)
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)

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
  encoded = Dense(math.ceil(input_dim*0.4), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  z_mean = enc_mean(encoded)
  z_log_var = enc_log_var(encoded)
  z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.4), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.5), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(math.ceil(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  vae = Model(input_layer, output_layer)

  #loss
  reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)

  vae.compile(optimizer='adam', loss='mse')
  vae.fit(x_train, x_train,
      epochs=100,
      batch_size=size,
      shuffle=True,
      callbacks=my_callbacks,
      validation_data=(x_test, x_test))

  vae_encoder = Model(vae.input, z_mean)
  x_train_encoded = vae_encoder.predict(x_train)
  x_test_encoded = vae_encoder.predict(x_test)

  return x_train_encoded,x_test_encoded
