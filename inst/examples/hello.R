library(tensorflow)

sess = tf.Session()

hello <- tf.constant('Hello, TensorFlow!')
sess$run(hello)

a <- tf.constant(10)
b <- tf.constant(32)
sess$run(a + b)
