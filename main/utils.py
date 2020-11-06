from tensorflow.keras.models import model_from_json
import tensorflow as tf
import os 
from tensorflow.compat.v1.keras.backend import set_session

JSONpath = os.path.join(os.path.dirname(__file__), 'models', 'mnist.json') #!!check file dir
MODELpath = os.path.join(os.path.dirname(__file__), 'models', 'mnist.h5') #!!check file dir

#session = tf.Session()

def init():
    #load json file
    graph = tf.compat.v1.get_default_graph()
    session = tf.compat.v1.Session()
    set_session(session)
    with graph.as_default():
        json_file = open(JSONpath, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
    
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(MODELpath)
        print('model loaded from disk')
        loaded_model.compile(loss='categorical_crossentropy', optimizer = 'adam', 
                             metrics=['accuracy'])
        
    return loaded_model, graph, session