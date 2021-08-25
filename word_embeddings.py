import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras import Input
from tensorflow.data import Dataset
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


model = Sequential()
model.add(Embedding(input_dim=100, output_dim=2))
model.compile('rmsprop', 'mse')

input_array = np.array([[6, 7, 5], [1, 7, 5], [1, 1, 1]])
output_array = model.predict(input_array)



# vocabulary 
vocabulary = Dataset.from_tensor_slices(['dog','his','is','old','two','years']) 
# vectorization 
vectorize_layer = TextVectorization(max_tokens=10000, output_mode= 'int', output_sequence_length=3) 
vectorize_layer.adapt(vocabulary.batch(64)) 

# create model that uses the vectorise text layer 
model = Sequential() 
model.add(Input(shape = (1,), dtype = tf.string)) 
model.add(vectorize_layer) 
input_data = [
    ['his dog is two years old'], 
    ['my dog is three years old'],
    ['he has a cat']
]

model.predict(input_data)







