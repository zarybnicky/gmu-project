kernel void doubleArray(global float *in, global float *out) {
  int i = get_global_id(0);
  out[i] = 2 * in[i];
}

#define actflag 0

typedef struct {
  float output;
  float delta;
  int numWeights;
  float weights[397];
} Node;
typedef struct {
  int numNodes;
  Node nodes[399];
} Layer;

float inline sigmoid(float x)
{
  if (x < -100)
    return 0;
  if (x > 100)
    return 1;
  return 1 / (1 + exp(-x));
}
float inline devsigmoid(float x) { return x * (1 - x); }
float inline    devtanh(float x) { return 1 - x * x; }
float inline       relu(float x) { return x < 0 ? 0 : x; }
float inline    devrelu(float x) { return x < 0 ? 0 : 1; }

kernel void compout(global Layer *layer, global Layer *prevLayer, int softflag)
{
  const int n = get_global_size(0);
  const int i = get_global_id(0);
  float t = 0;
  for (int j = 0; j < layer->nodes[i].numWeights; j++) {
    t += layer->nodes[i].weights[j] * prevLayer->nodes[j].output;
  }
  t += 0.1; //bias
  if (softflag == 0) {
    switch (actflag) {
    case 0: layer->nodes[i].output = sigmoid(t); break;
    case 1: layer->nodes[i].output = tanh(t); break;
    case 2: layer->nodes[i].output = relu(t); break;
    }
  } else {
    layer->nodes[i].output = t;
  }
}

kernel void softmax(global Layer *layer){
  const int i = get_global_id(0);
  float expsum = 0;
  for (int j = 0; j < layer->numNodes; j++) {
    expsum += exp(layer->nodes[j].output);
  }
  layer->nodes[i].output = exp(layer->nodes[i].output) / expsum;
}

kernel void backprophid(global Layer *layer, global Layer *prevLayer, global Layer *nextLayer, float a)
{
  const int n = get_global_size(0);
  const int i = get_global_id(0);

  float delta = 0;
  for (int j = 0; j != nextLayer->numNodes; j++) {
    delta += nextLayer->nodes[j].delta * nextLayer->nodes[j].weights[i];
  }
  switch (actflag) {
  case 0: delta *= devsigmoid(layer->nodes[i].output); break;
  case 1: delta *= devtanh(layer->nodes[i].output); break;
  case 2: delta *= devrelu(layer->nodes[i].output); break;
  }
  layer->nodes[i].delta = delta;
  for (int j = 0; j != layer->nodes[i].numWeights; j++) {
    layer->nodes[i].weights[j] -= a * delta * prevLayer->nodes[j].output;
  }
}


kernel void backpropout(global Node* nodes, global Node *prevnodes, global float* targets, float a, int softflag)
{
  const int n = get_global_size(0);
  const int i = get_global_id(0);
  float delta=0;
  if (softflag == 1) {
    delta = nodes[i].output - targets[i];
  }
  else{
    switch (actflag) {
    case 0: delta = (nodes[i].output - targets[i]) * devsigmoid(nodes[i].output); break;
    case 1: delta = (nodes[i].output - targets[i]) * devtanh(nodes[i].output); break;
    case 2: delta = nodes[i].output - targets[i] * devrelu(nodes[i].output); break;
    }
  }
  for (int j = 0; j !=nodes[i].numWeights; j++) {
    nodes[i].weights[j] -= a*delta*prevnodes[j].output;
  }
  nodes[i].delta = delta;
}
