# Oxford 102 Flower Recognition

Classification of flowers in a dataset with large number of similar classes.

## Getting started

Required packages can be found in requirements.txt

### Cyclical learning rate
CNN models are trained using cyclical learning rate.

To automatically find the best learning rate range:
```
python trainmodel.py BATCH_SIZE MODEL OPTIMIZER 1 CROP CYCLICAL_METHOD CYCLICAL_STEP;
```

This produces a chart in lrfinder/output, that defines the lower and the upper bound 
that must be set in lrfinder/configadam.py or lrfinder/configsgd.py, depending on the optimizer used to train the model.

### Model training

```
python trainmodel.py BATCH_SIZE MODEL OPTIMIZER 0 CROP CYCLICAL_METHOD CYCLICAL_STEP;
```

### Parameters value
MODEL can be either [resnet, densenet, efficientnet]

OPTIMIZER can be either [sgd, adam]

CROP can be either [none, random]

CYCLICAL_METHOD can be either [triangular, triangular2]

## Visualization

Saliency maps and grad-cam example as shown in the report can be obtained running.
```
python visualization/visualize.py
```
This will produce the exact images inside visualization/output.

## Feature extraction


