import tensorflow as tf

def UnetConv2D(x, units,ispool):
    if(ispool):
        x = tf.keras.layers.MaxPooling2D(padding="same")(x)
    x = tf.keras.layers.Conv2D(units,3,padding="same",activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(units,3,padding="same",activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def build_model(shape, num):
    input = tf.keras.layers.Input(shape)
    x1 = UnetConv2D(input, 64, False)
    x2 = UnetConv2D(x1, 128, True)
    x3 = UnetConv2D(x2, 256, True)
    x4 = UnetConv2D(x3, 512, True)
    x5 = UnetConv2D(x4, 1024, True)
    x6 = tf.keras.layers.Conv2DTranspose(512,2,strides=2,padding="same",activation="relu")(x5)
    x6 = tf.keras.layers.BatchNormalization()(x6)
    x7 = tf.concat([x4,x6], axis=-1)
    x7 = UnetConv2D(x7, 512, False)
    x7 = tf.keras.layers.Conv2DTranspose(256,2,strides=2,padding="same",activation="relu")(x7)
    x7 = tf.keras.layers.BatchNormalization()(x7)
    x8 = tf.concat([x3,x7], axis=-1)
    x8 = UnetConv2D(x8, 256, False)
    x8 = tf.keras.layers.Conv2DTranspose(128,2,strides=2,padding="same",activation="relu")(x8)
    x8 = tf.keras.layers.BatchNormalization()(x8)
    x9 = tf.concat([x2,x8], axis=-1)
    x9 = UnetConv2D(x9, 128, False)
    x9 = tf.keras.layers.Conv2DTranspose(64,2,strides=2,padding="same",activation="relu")(x9)
    x9 = tf.keras.layers.BatchNormalization()(x9)
    x10 = tf.concat([x1,x9], axis=-1)  
    x10 = UnetConv2D(x10, 64, False)
    output = tf.keras.layers.Conv2D(num,1,padding="same",activation="softmax")(x10)
    return tf.keras.Model(inputs=input,outputs=output)
