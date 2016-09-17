Using the TensorFlow API from R
================

Introduction
------------

The **tensorflow** package provides access to the entire TensorFlow API from within R. The top-level entry point to the API is the `tf` object which is exported from the **tensorflow** package. For example, here is a simple "Hello, World" script:

``` r
library(tensorflow)

sess = tf$Session()

hello <- tf$constant('Hello, TensorFlow!')
sess$run(hello)

a <- tf$constant(10L)
b <- tf$constant(32L)
sess$run(a + b)
```

The `tf` object represents the main `tensorflow` module, and includes a wide variety of functions and classes which you can read more more about in the [TensorFlow API](https://www.tensorflow.org/api_docs/python/) documentation (see also the [Getting Help](#getting-help) section below which describes accessing help for the TensorFlow API directly within the RStudio IDE editor and console).

The TensorFlow API also includes many sub-modules, including the `tensorflow.nn` module which provides specialized functions for neural networks and the `tensorlow.contrib.learn` and `tensorflow.contrib.slim` modules which define higher-level APIs for TensorFlow. You can access these modules the same way that functions are accessed (via the `$` operator). For example:

``` r
# Call the conv2d function within the tensorflow.nn module
tf$nn$conv2d(x, W, strides=c(1L, 1L, 1L, 1L), padding='SAME')

# Access the tensorflow.contrib.learn module
learn <- tf$contrib$learn
```

Simple Example
--------------

Here's a simple example of using R to define a TensorFlow model:

``` r
library(tensorflow)

# Create 100 phony x, y data points, y = x * 0.1 + 0.3
x_data <- runif(100, min=0, max=1)
y_data <- x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
W <- tf$Variable(tf$random_uniform(shape(1L), -1.0, 1.0))
b <- tf$Variable(tf$zeros(shape(1L)))
y <- W * x_data + b

# Minimize the mean squared errors.
loss <- tf$reduce_mean((y - y_data) ^ 2)
optimizer <- tf$train$GradientDescentOptimizer(0.5)
train <- optimizer$minimize(loss)

# Launch the graph and initialize the variables.
sess = tf$Session()
sess$run(tf$initialize_all_variables())

# Fit the line (Learns best fit is W: 0.1, b: 0.3)
for (step in 1:201) {
  sess$run(train)
  if (step %% 20 == 0)
    cat(step, "-", sess$run(W), sess$run(b), "\n")
}
```

If you've seen TensorFlow code written in Python you'll recognize this code as nearly identical save for some minor syntactic differences (e.g. the use of `$` rather than `.` as an object delimiter). You'll also note that R numeric vectors and random distribution functions are used, whereas Python code would typically use their NumPy equivalents.

The remainder of this article describes the core patterns used for interacting with the TensorFlow API from R,

Numeric Types
-------------

The TensorFlow API is more strict about numeric types than is customary in R (which often automatically casts from integer to float and vice-versa as necessary). Many TensorFlow function parameters require integers (e.g. for tensor dimensions) and in those cases it's important to use an R integer literal (e.g. `1L`). Here's an example of specifying the `strides` parameter for a 4-dimensional tensor using integer literals:

``` r
tf$nn$conv2d(x, W, strides=c(1L, 1L, 1L, 1L), padding='SAME')
```

Here's another example of using integer literals when defining a set of integer flags:

``` r
flags$DEFINE_integer('max_steps', 2000L, 'Number of steps to run trainer.')
flags$DEFINE_integer('hidden1', 128L, 'Number of units in hidden layer 1.')
flags$DEFINE_integer('hidden2', 32L, 'Number of units in hidden layer 2.')
```

Lists
-----

Some TensorFlow APIs call for lists of a numeric type. Typically you can use the `c` function (as illustrated above) to create lists of numeric types. However, there are a couple of special cases (mostly involving specifying the shapes of tensors) where you may need to create a numeric list with an embedded `NULL` or a numeric list with only a single item. In those cases you'll want to use the `list` function rather than `c` in order to force the argument to be treated as a list rather than a scalar, and to ensure that `NULL` elements are preserved. For example:

``` r
x <- tf$placeholder(tf$float32, list(NULL, 784L))
W <- tf$Variable(tf$zeros(list(784L, 10L)))
b <- tf$Variable(tf$zeros(list(10L)))
```

Tensor Shapes
-------------

This need to use `list` rather than `c` is very common for shape arguments (since they are often of one dimension and in the case of placeholders often have a `NULL` dimension). For these cases there is a `shape` function which you can use to make the calling syntax a bit more more clear. For example, the above code could be re-written as:

``` r
x <- tf$placeholder(tf$float32, shape(NULL, 784L))
W <- tf$Variable(tf$zeros(shape(784L, 10L)))
b <- tf$Variable(tf$zeros(shape(10L)))
```

Tensor Indexes
--------------

Tensor indexes within the TensorFlow API are 0-based (rather than 1-based as R vectors are). This typically comes up when specifying the dimension of a tensor to operate on (e.g with a function like `tf$reduce_mean` or `tf$argmax`). The first dimension of a tensor is specified as `0L`, the second `1L`, and so on. For example:

``` r
# call tf$reduce_mean on the second dimension of the specified tensor
cross_entropy <- tf$reduce_mean(
  -tf$reduce_sum(y_ * tf$log(y_conv), reduction_indices=1L)
)

# call tf$argmax on the second dimension of the specified tensor
correct_prediction <- tf$equal(tf$argmax(y_conv, 1L), tf$argmax(y_, 1L))
```

Dictionaries
------------

Some TensorFlow APIs accept dictionaries as arguments, the most common of which is the `feed_dict` argument which feeds training and test data to various functions. If a dictionary is keyed by character string you can simply pass an R named list. However, if a dictionary is keyed by another object type like a tensor (as `feed_dict` is) then you should use the `dict` function rather than a named list. For example:

``` r
sess$run(train_step, feed_dict = dict(x = batch_xs, y_ = batch_ys))
```

The `x` and `y_` variables in the above example are tensor placeholders which are substituted for by the specified training data.

With Contexts
-------------

The TensorFlow API includes several functions that yield scoped execution contexts (i.e. blocks of code which execute code on enter and exit, for example to set a default or to open and close a resource).

The R `with` generic function can be used with TensorFlow objects that define a scoped execution context. For example:

``` r
with(tf$name_scope('input'), {
  x <- tf$placeholder(tf$float32, shape(NULL, 784L), name='x-input')
  y_ <- tf$placeholder(tf$float32, shape(NULL, 10L), name='y-input')
})
```

In this case the `tf$name_scope` execution context will automatically pre-pend `"input/"` to the specified names, so they will become `"input/x-input"` and `"input/y-input"` respectively.

It is sometimes convenient to gain access to the execution context via an R object. For this purpose there is also a custom `%as%` operator defined, for example:

``` r
with(tf$Session() %as% sess, {
  sess$run(hello)
})
```

The `sess` variable will be assigned from the execution context and be available only within the expression passed to `with`.

Getting Help
------------

As you use TensorFlow from R you'll want to get help on the various functions and classes available within the API. If you are running the vary latest [Preview Release](https://www.rstudio.com/products/rstudio/download/preview/) (v1.0.18 or later) of RStudio IDE you can get code completion and inline help for the TensorFlow API within RStudio. For example:

<img src="images/completion-functions.png" style="margin-bottom: 15px; border: solid 1px #cccccc;" width="804" height="260" />

Inline help is also available for function parameters:

<img src="images/completion-params.png" style="margin-bottom: 15px; border: solid 1px #cccccc;" width="804" height="177" />

You can press the F1 key while viewing inline help (or whenever your cursor is over a TensorFlow API symbol) and you will be navigated to the location of that symbol's help within the [TensorFlow API](https://www.tensorflow.org/api_docs/python/) documentation.

### API Documentation

The [TensorFlow API](https://www.tensorflow.org/api_docs/python/) reference documents the Python API for TensorFlow. Since Python and R are so syntactically similar the documentation is also quite readable for users of R.

Python data types in the TensorFlow API map to R as follows:

| Python            | R                      | Examples                                    |
|-------------------|------------------------|---------------------------------------------|
| Scalar            | Single-element vector  | `1`, `1L`, `TRUE`, `"foo"`                  |
| List              | Multi-element vector   | `c(1.0, 2.0, 3.0)`, `c(1L, 2L, 3L)`         |
| Tuple             | List of multiple types | `list(1L, TRUE, "foo")`                     |
| Dict              | Named list or `dict`   | `list(a = 1L, b = 2.0)`, `dict(x = x_data)` |
| NumPy ndarray     | Matrix/Array           | `matrix(c(1,2,3,4), nrow = 2, ncol = 2)`    |
| None, True, False | NULL, TRUE, FALSE      | `NULL`, `TRUE`, `FALSE`                     |

If you keep these mappings along with the above guidelines on numeric types and tensor shapes/indexes in mind then TensorFlow documentation, tutorials, and examples that use Python will be very easy to apply to the use of the R API.
