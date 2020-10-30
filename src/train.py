import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()

def build_model():
	model = tf.keras.models.Sequential([
	  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  	  tf.keras.layers.Dense(128,activation='relu'),
  	  tf.keras.layers.Dense(10, activation='softmax')
	])

	model.compile(
	    loss='sparse_categorical_crossentropy',
	    optimizer=tf.keras.optimizers.Adam(0.001),
	    metrics=['accuracy'], 
	)
	
	return model

def train(ds_train):

	model = build_model()

	model.fit(
	    ds_train,
	    epochs=6,
	    verbose=1,
	)

	return model

