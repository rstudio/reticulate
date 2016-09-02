library(tensorflow)

sess = session()

hello <- constant('Hello, TensorFlow!')
run(sess, hello)

a <- constant(10)
b <- constant(32)
run(sess, a + b)
