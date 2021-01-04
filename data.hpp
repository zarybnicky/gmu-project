#ifndef LAYER_H
#define LAYER_H

#include "include.hpp"

typedef struct {
  int numberOfWeights;
  float weights[1200];
  float output;
  float delta;
} Node;
typedef struct {
  float weights[49];
  float bias;
} Filter;
typedef struct {
  int numOfNodes;
  Node nodes[1200];
} Layer;
typedef struct {
  int numOfFilters;
  Filter filters[10];
} ConvLayer;

Layer *layer(int numberOfNodes, int numberOfWeights);
ConvLayer *convlayer(int numberOfFilters, int filtdim);

#endif //LAYER_H
