
#' @export
gradient_descent_optimizer <- function(learning_rate, use_locking=FALSE,
                                       name="GradientDescent") {
  tf <- tf_import()
  tf$train$GradientDescentOptimizer(learning_rate,
                                    use_locking = use_locking,
                                    name = name)
}


#' @export
optimizer_minimize <- function(optimizer, loss, global_step=NULL, var_list=NULL,
                               gate_gradients=1L, aggregation_method=NULL,
                               colocate_gradients_with_ops=FALSE, name=NULL,
                               grad_loss=NULL) {
  optimizer$minimize(loss,
                     global_step = global_step,
                     var_list = var_list,
                     gate_gradients = as.integer(gate_gradients),
                     aggregation_method = aggregation_method,
                     colocate_gradients_with_ops = colocate_gradients_with_ops,
                     name = name,
                     grad_loss = grad_loss)
}

