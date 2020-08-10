# pip install --upgrade pip
# pip install tensorflow
# import tensorflow as tf

# --------------------------------------------------
#    CONSTANTES
# --------------------------------------------------

# valor1 = tf.constant(2)
# valor2 = tf.constant(3)

# print(type(valor1))

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))