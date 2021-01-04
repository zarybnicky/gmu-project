#define actflag 0// 0:sigmoid , 1:tanh ,2:relu

typedef struct Node {
  int numWeights;
  float weights[1200];
  float output;
  float delta;
} Node;

typedef struct Filter {
  float weights[49];
  float bias;
} Filter;

typedef struct Layer {
  int numNodes;
  Node nodes[1200];
} Layer;

typedef struct ConvLayer {
  int numFilters;
  Filter filters[10];
} ConvLayer;

float inline    sigmoid(float x) { return 1 / (1 + exp(-x)); }
float inline devsigmoid(float x) { return (x*(1-x)); }
float inline      mtanh(float x) { return tanh(x); }
float inline    devtanh(float x) { return (1-x*x); }
float inline       relu(float x) { return x < 0 ? 0 : x; }
float inline    devrelu(float x) { return x < 0 ? 0 : 1; }

kernel void compout(global Node *nodes, global Node *prevnodes, int softflag) {
  const int n = get_global_size(0);
  const int i = get_global_id(0);
  float t = 0;
  for (int j = 0; j < nodes[i].numWeights; j++) {
    t += nodes[i].weights[j] * prevnodes[j].output;
  }
  t += 0.1; //bias
  if (softflag == 0) {
    switch (actflag) {
    case 0: nodes[i].output = sigmoid(t);break;
    case 1: nodes[i].output = mtanh(t);break;
    case 2: nodes[i].output = relu(t);break;
    }
  } else {
    nodes[i].output = t;
  }
}

kernel void softmax(local Node *nodes, int nodesnum) {
  const int i = get_local_id(0);
  float expsum = 0;
  for (int j = 0; j < nodesnum; j++) {
    expsum += exp(nodes[j].output);
  }
  nodes[i].output=exp(nodes[i].output)/expsum;
}

kernel void backprophid(global Node *nodes, global Node *prevnodes, global Node *nextnodes, int nextnumNodes, float a) {
  const int n = get_global_size(0);
  const int i = get_global_id(0);
  nodes[i].delta = 0;
  for (int j = 0; j !=nextnumNodes; j++) {
    nodes[i].delta += nextnodes[j].delta * nextnodes[j].weights[i];
  }
  switch (actflag) {
  case 0: nodes[i].delta *= devsigmoid(nodes[i].output); break;
  case 1: nodes[i].delta *= devtanh(nodes[i].output); break;
  case 2: nodes[i].delta *= devrelu(nodes[i].output); break;
  }
  for (int j = 0; j != nodes[i].numWeights; j++) {
    nodes[i].weights[j] -= a * nodes[i].delta * prevnodes[j].output;
  }
}

kernel void backpropout(global Node* nodes, global Node *prevnodes, global float *targets, float a, int softflag) {
  const int n = get_global_size(0);
  const int i = get_global_id(0);
  nodes[i].delta = nodes[i].output - targets[i];
  if (softflag != 1) {
    switch (actflag) {
    case 0: nodes[i].delta *= devsigmoid(nodes[i].output) ;break;
    case 1: nodes[i].delta *= devtanh(nodes[i].output); break;
    case 2: nodes[i].delta *= devrelu(nodes[i].output); break;
    }
  }
  for (int j = 0; j != nodes[i].numWeights; j++) {
    nodes[i].weights[j] -= a * nodes[i].delta * prevnodes[j].output;
  }
}

kernel void convolve(global float *image, global Filter *filters, global float *featMap, int filterWidth, int inWidth, int featmapdim) {
  const int x = get_global_id(0);//cols
  const int y = get_global_id(1);//rows
  const int z = get_global_id(2);//filters

  float sum = 0;
  for (int r = 0; r < filterWidth; r++) {
    for (int c = 0; c < filterWidth; c++) {
      sum += filters[z].weights[c * filterWidth + r] * image[x + c + inWidth * (y + r)];
    }
  }
  sum += filters[z].bias;
  switch(actflag){
  case 0: featMap[x + y * featmapdim + z * featmapdim * featmapdim] = sigmoid(sum);break;
  case 1: featMap[x + y * featmapdim + z * featmapdim * featmapdim] = mtanh(sum);break;
  case 2: featMap[x + y * featmapdim + z * featmapdim * featmapdim] = relu(sum);break;
  }
}

kernel void pooling(global float *prevfeatMap, global float *poolMap, global int *indexes, int Width, int pooldim) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);
  float max = 0;
  int index = 0;
  for (int r = 0; r < 2; r++) {
    for (int c = 0; c < 2; c++) {
      if (prevfeatMap[(y + c) * Width * z + x + r] > max) {
        max = prevfeatMap[(y + c) * Width * z + x + r];
        index = c * 2 + r;
      }
    }
  }
  poolMap[x + y * pooldim + z * pooldim * pooldim] = max;
  indexes[x + y * pooldim + z * pooldim * pooldim] = index;
}

kernel void deltas(global Node *nodes, global Node *nextnodes,global float *deltas, global int *indexes, int dim, int nextnumNodes, int pooldim) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);
  int i = x + y * pooldim + z * pooldim * pooldim;
  float delta = 0;
  for (int j = 0; j != nextnumNodes; j++) {
    delta += nextnodes[j].delta * nextnodes[j].weights[i];
  }
  switch(actflag){
  case 0: delta *= devsigmoid(nodes[i].output); break;
  case 1: delta *= devtanh(nodes[i].output); break;
  case 2: delta *= devrelu(nodes[i].output); break;
  }
  for (int r = 0; r < 2; r++) {
    for (int c = 0; c < 2; c++) {
      if ((c * 2 + r) == indexes[i]) {
        deltas[2 * x + r + (2 * y + c) * dim + z * dim * dim] = delta;
      }
    }
  }
}

kernel void rotatemat(global float *source, global float *destin, int dim) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  destin[x + dim * y] = source[dim - x + dim * (dim - y)];
}

kernel void backpropcnn(global float *featMap, global float *deltas, global Filter *filters, int featmapdim, int imagedim, int filterdim, float a, global float *Image) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);
  float sum = 0;
  for (int r = 0; r < featmapdim; r++) {
    for (int c = 0; c < featmapdim; c++) {
      sum += deltas[c + r * featmapdim + z * featmapdim * featmapdim]*Image[x + r + imagedim * (y + c)];
    }
  }
  filters[z].weights[x + filterdim * y] -= a * sum;
}

kernel void cnntoFcnn(global float *poolMap, global Node *nodes, int dim, int mapindex) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);
  nodes[x + y * dim + z * dim * dim].output = poolMap[x + y * dim + z * dim * dim];
}
