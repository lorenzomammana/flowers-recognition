# Oxford 102 Flower Recognition

Classification of flowers in a dataset with large number of similar classes.
The same dataset division defined in the original work is used:
* 1020 training images
* 1020 validation images
* 6149 test images
## Getting started

Required packages can be found in requirements.txt

The code should only work on Python 3.

### Cyclical learning rate
CNN models are trained using the approach described in "Cyclical Learning Rates for Training Neural Networks" (L.N. Smith, 2017).

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

### Best results
| Model          | Method      | Step | lr range    | batch | crop | test acc |
|----------------|-------------|------|-------------|-------|------|----------|
| Resnet18       | Triangular  | 2    | [1e-6,5e-4] | 64    | No   | 0.9187   |
| Densenet121    | Triangular  | 4    | [1e-5,1e-3] | 64    | No   | 0.9455   |
| EfficientnetB4 | Triangular2 | 4    | [1e-5,1e-3] | 8     | No   | 0.9719   |

We are able to obtain state of the art performance without using segmented images or overly complexed methods!
## Visualization

Saliency maps and grad-cam example as shown in the report can be obtained running.
```
python visualization/visualize.py
```
This will produce the exact images inside visualization/output.

## Feature extraction
To use the network as a feature extract it's first necessary to save the features on file using:
```
python featext/extract_features.py
```

After that it's possible to train the MLP on top with or without cyclical learning rate:
```
python featext/trainautolr.py
python featext/train.py
```
