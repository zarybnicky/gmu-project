kernel void descend_slow(
    double rate,
    double momentum,
    double regulariser,
    constant double *weights,
    constant double *gradient,
    constant double *last,
    global double *outputWeights,
    global double *outputMomentum
) {
  int i = get_global_id(0);
  double momentum_ = momentum * last[i] - rate * gradient[i];
  outputMomentum[i] = momentum_;
  outputWeights[i] = mad(weights[i], 1 - rate * regulariser, momentum_);
}

kernel void fcnn_forward(
    const int m,
    const int n,
    constant double *a,
    constant double *x,
    constant double *b,
    global double *y
) {
  const int i = get_global_id(0);
  double acc = 0;
  for (unsigned k = 0; k < n; k++) {
    acc = mad(a[i + m * k], x[k], acc);
  }
  y[i] = acc + b[i];
}

kernel void fcnn_update(
    int rows,
    int cols,
    double rate,
    double momentum,
    double regulariser,
    global double *weightsVec,
    global double *weightsMat,
    constant double *gradientVec,
    constant double *gradientMat,
    global double *lastVec,
    global double *lastMat
) {
  int i = get_global_id(0);
  double momentum_ = momentum * lastVec[i] - rate * gradientVec[i];
  lastVec[i] = momentum_;
  weightsVec[i] = (rate * regulariser - 1) * weightsVec[i] + momentum_;
  for (unsigned j = 0; j < cols; j++) {
    double momentum_ = momentum * lastMat[i + rows * j] - rate * gradientMat[i + rows * j];
    lastMat[i + rows * j] = momentum_;
    weightsMat[i + rows * j] = (rate * regulariser - 1) * weightsMat[i + rows * j] + momentum_;
  }
}

kernel void fcnn_backward(
    int rows,
    int cols,
    constant double *dEdy,
    constant double *x,
    constant double *wN,
    global double *mm,
    global double *dWs
) {
  int i = get_global_id(0);

  // let mm' = dEdy `outer` x
  if (i < rows) {
    const double d = dEdy[i];
    for (unsigned j = 0; j < cols; j++) {
      mm[i * rows + j] = d * x[j];
    }
  }

  // dWs  = tr wN #> dEdy
  if (i < cols) {
    double acc = 0;
    for (unsigned j = 0; j < rows; j++) {
      acc = mad(wN[i + rows * j], dEdy[j], acc);
    }
    dWs[i] = acc;
  }
}
