# Lecture 2

### Setting Hyperparameters
1. Cross-Validation: Split data into folds,try each fold as validation and average the results

# Lecture 3

## Linear classification

#### Linear score function
- give each class score from x * weights

#### Loss function(svm v.s. softmax)
 - hinge is more robust or give no more loss since one score is wrong compared to cross-entropy loss(softmax), in comparision cross-entropy loss is more sensitive to wrong score
 - softmax function may lead to Numeric stability (exponential blow up),
#### Regularization
 - L2 to minimize weights, to make the curves simple(Occam’s Razor)

## Optimization

#### Gradient Descent
- mini batch (stochastic) gradient descent is faster 
- Learning rate, begin with high rate, then decrease 

# Lecture 4

##  BP Algo

### Computational Graph
- Intuition it's impossible to compute the derivatives of weights in terms of neural netwroks cause it's too complex, you need computational graph
- graph consists of gates, gate is a function with inputs and output

### Back Propagation
- key
    - chain rule in derivatives
    - the gate model, out = func(in)
- Patterns in backward flow
    - add:gradient distributor
    - max:gradient router
   
- Gradients add at branches

### Gate Implemetation

```
#single gate
class Gate():
    self.x
    self.y
    def forward(inputs):
        return func(iuputs)
    
    def backward(dz):
        return [dx, dy]
```

### Jacobian Matrix
- input and output are always vectors
- for every element in the output vertor , compute the corresponding derivative for every element in input, so the size would be len(output) * len(input), a giant matrix

### Forward & Backforward as programming rules
- even in shallow model!!!

## Neural Networks

### Activation functions
- ReLU
- ....

### Why Layers & fully connected 
- code can be vectorized

### Layers & sizes v.s. regularization
- use regularization to avoid overfit instead of simplifying the whole architeture

### Interpretation, insight
- the hidden layers represent the raw data in a new space, which can be linearly classified

# Lecture 5 Training Neural Networks,Part 1

## history breakthrough:pre-training

## Activation Functions
**key:gradient flow**
-  sigmoid function 
    - gradient vanish, sigmoid function has 0 derivatives when x < -5 or x > 5, so input must be zero-centered
    - output not zero-centerd

- tanh : same as sigmoid but one, the output is zero centerd

- ReLU
     - converge much faster, compute efficiently,
     - kill gradient if x < 0
     - dead neuron, the input is alway small than 0, then no update to weights

- Leaky ReLU/Parametric ReLU:f(x) = max(ax,x)

- Maxout:max(w1\*x+b1,w2\*x+b2),double weights number

- suggest: use ReLU


## Data preprocessing
- common in machine learning
    - original data -> zero centered data -> normalized data(zero centered data divide its std)
    - original data -> decorrelated data(使用PCA降维，去相关,  diagonal covariance matrix) -> whitened data(covariance matrix is the identity matrix)
  
- in images
    - center only

## Weight Initialization
- random guassian numbers * 0.01 : the output would be  close to 0, gradients would therefore be small
- random guassian numbers * 1 : the output would be -1 and 1, so gradients are basically 0
- Xavier initialization : 
    - tanh
    - divide 2  in relu, because relu half the output variance

## Batch Normalization (for x not weights) after every layer
- Improves gradient flow through the network
- Allows higher learning rates
- Reduces the strong dependence on initialization
- Acts as a form of regularization in a funny way, and slightly reduces the need for dropout, maybe
- note that at test time mean & std are not computed while forward passing, they are computed & stored at train time

## Babysitting the Learning Process
1. preprocess the data
2. choose the architecture
3. double check the loss by turning on or off the regularization
4. make sure the model can overfit on small dataset(like 20)
5. binary search parameters, like learning rate,make sure not two low or high

## Hyperparameter Optimization
- in a loop optimize parameters in uniformly random log space 
- no grid search, use random search above instead
- Track the ratio of weight updates / weight magnitudes:want this to be somewhere around 0.001 or so

# Lecture 6 Training Neural Networks, Part 2

- problems;if no activation functinos ,then the whole networks is only linear

## Parameter updates
- Momentum update
- Nesterov Momentum update,
- Nesterov Accelerated Gradient
 - there is no bad local minimun, they have basically the same loss
- AdaGrad update
- RMSProp update
- Adam update
    - combination of Momentunm & AdaGrad
    - bias correction prevent initial zero for m&v(params)
    
    
## Learning rate
- exponential decay
- 1/t decay

## Second order optimization methods
- Quasi-Newton methods (BGFS most popular)
- L-BFGS (Limited memory BFGS)

### Inpratice
- adam : good choice
- L-BFGS(when able to afford full batch)

## Ensembles
- average results of different models(see the slides)
    - can also get a small boost from averaging multiple model checkpoints of a single model. 
    - keep track of (and use at test time) a running average parameter vector
    
## Regularization (dropout)
- at training time randomly set some neurons to zero in the forward pass
- Forces the network to have a redundant representation.
- Dropout is training a large ensemble of models (that share parameters).
- at test time or predicting
    - average multiple predictions with dropout while cmputing
    - no drop out at test time, remeber to scale  
- More common: “Inverted dropout”, test time is unchanged

## Gradient checking
- see the notes 

# Lecture 7 ConvNets

 - layer(3 dims) ->filters(squah to layer with 1 depth) ->new layer ->
 - stride, convolve speed
 - zero-padding with P = (F-1)/2. (will preserve size spatially)
 ```
 Common settings of various sizes:
K = (powers of 2, e.g. 32, 64, 128, 512)
- F = 3, S = 1, P = 1
- F = 5, S = 1, P = 2
- F = 5, S = 2, P = ? (whatever fits)
- F = 1, S = 1, P = 0 (change the depth)
 ```
 - pooling layer, downsampling the activation map, depth remain the same
     - max pooling

# Lecture 8 spatial localization Object detection

- localization , singel object
- Object detection , multiple obect

## localization, divide the task into classification & localization
- Localization as Regression
    - After conv layers or last fc layer of classification model(VGG ResNet),  attach new fully-connected “regression head” to the network to and train the head as regression model
    -output of regression head: Class agnostic v.s. Class specific, 4 numbers (one box)  v.s.C x 4 numbers (one box per class)
- Sliding Window: Overfeat
    - aggregate over different windows
    - speed up computation by transforming the final fc layer to  n\*1\*1 conv layer (see the slides)

## detection
- Detection as Classification, too many scales & positions
- Histogram of Oriented Gradients ,  Deformable Parts Models (CNN)
- Region Proposals: Selective Search, use EdgeBoxes
- R-CNN
    - compute boxes by Region Proposals method first,  warp to CNN input size
    - change the final fc layer of conv nets to adjust the num of classes of your task 
    - extra features using CNN
    - train shallow model to classify region features (is a correct region or not)
    - boxes regression ???
- datasets : ms-coco
- Object Detection: Evaluation
- Fast R-CNN
    - use region proposals on (Region of Interest Pooling) feature map computed by conv nets
    - Region of Interest Pooling key : Project region proposal onto conv feature map
    - neck : traditional region proposal cost too much time at testing time 

- Faster R-CNN
    - Region Proposal Network (RPN) : on the top of feature map, as a con layer
    - One network, four losses see the slides
        - RPN classification (anchor good / bad) ???? how is this evaluate
        - RPN regression (anchor -> proposal) ???? how is this evaluate
        - Fast R-CNN classification (over classes) 
        - Fast R-CNN regression (proposal -> box)
- YOLO: You Only Look Once, Detection as Regression, weired method???

# Lecture Note 9 Understanding and Visualizing Convolutional Neural Networks

### filters visualization 
- visualize filters (weights)

### last layer
- using Nearest Neighbors on the last layer of  many images
- t-SNE, Dimensionality Reduction, subject the final layer of image to 2 dims

### visualizing activations

### Occlusion experiments
- Mask part of the image before feeding to CNN, draw heatmap of probability at each mask location

### saliency maps
- Compute gradient of (unnormalized) class score with respect to image pixels, take absolute value and max over RGB channels
- one way to segment

### Intermediate Features via (guided) backprop
- Compute gradient of neuron value with respect to image pixels
- guided relu back prop, kill negative gradient, or influence

### Visualizing CNN features: Gradient Ascent
- update the image (all zeros at the begining) to maximize the neuron ouput

### fooling ConvNets
- pose optimization over input image to maximize any class score
- above strategy is effective at changing the prediction but totally not on original data, the image changes little
- result of linear classifier can be changed easily even if the overall data changes little

### deepdream
- Choose an image and a layer in a CNN; repeat:
    1. Forward: compute activations at chosen layer
    2. Set gradient of chosen layer equal to its activation
    3. Backward: Compute gradient on image
    4. Update image

### feature inversion
- Given a CNN feature vector for an image, find a new image that:
    - Matches the given feature vector
    - “looks natural” (image prior regularization)

### Texture Synthesis
- Nearest Neighbor copying
- Gram Matrix(how to compute? in ppt P54-57)


# Lecture Note 10 RNN

- flexibility: one | many to one | many 
- RNN
    - timestep
    - same computational graph every timestep
    - every cell take two inputs, last timestep output & last layer output
    - how everything computed, ppt p22
    - important to clarify how to back prop through the cell 
    - basically you can feed in any text with order RNN, tex, code
 
- Image Captioning
    - take the feature ouput of ConvNet as input as the input of hidden layer in the first timestep  only
    - take the output of this time step (a word) as the input of next time step
    - there wil be an token \<END\> indicating the end of caption
- training a model vanilla
    - iterate over inputs, forward pass
    - inverselty iterate to back prop, accmulate gradients!!
    - iteration length is limited, things are done in epoch - sequence length
   
- image captioning with attention
    - generate distribution over L locations along with word at the same time
    - turn locations into weighted features 
    - take weighted features & word as the input of next timestep
    
- visual question answearing: RNN with attention, see 17 sildes p88
- lstm, see the sildes for detailed formula
    - similiar to ResNet
    - hidden cell have to variables, h & c 
    - additive interactions improve gradient flow, addition equally distribute gradient
    
- gradient flow in RNN
     - vanilla type may result in gradients explode or vanish due to iteration in the one epoch
     - use gradient clipping to control explosion
     - use lstm to control vanishing


# Lecture Note 11 ConvNets in Pratice

## Data augmentation
1. Horizontal flips
2. Random crops/scales, different between training & testing see slides 19
3. color jitter, use pca, see slides 21
4. get creative, translation, rotation

- data augmentatin similiar to dropout
- useful for small datasets

## Transfer learning(less data)
- more data more layers, vice versa
- use frozen part networks as feature extractor, data tranformer(forward pass once to produce new form of data)
- fine tune method
    1. learning rate tip, see slides p30
    2. stage fine tune which is tune last layer first, then tune last few layers, because the gradients is too large at the last layer(?)
- dataset size v.s. dataset difference matrix, see slides p34

## All about convolutions

### stacking
- stacking make one neuron can see more areas
- small filters, more nonlinearity, less paras, less computation
- bottleneck sandwich, more nonlinearity, less params, less compute

### computation
- im2col, turn conv into matrix, see slides
- Convolution Theorem, Fast Fourier Transform, slow with small filters
- Strassen's Algorithm, real fast?

## Implementation Details

### bottleneck
- cpu-gpu communication
    - dataprefetch + augment 
    - turn pics into one giant raw byte stream because pics may store in different places

### floating point precision
- 32bit to save memory & speed up computing
- 16bit faster
- 10bit forward & 12bit backward
- 1 bit ?

# Lecture Note 12 Software Packages 

## caffe
- no need to write code
- have things done in pro file, just assign everything, the input data, the layers, the parameters, the archtecture, 
- datastructure
    - Blob: Stores data and derivatives
    - Layer: Transforms bottom blobs to top blobs (the same as the layer coded in assignments)
    - Net: Many layers; computes gradients via forward / backward
    - Solver: Uses gradients to update weights

## Torch
- modules in lua, like packages in python
- modules are easy to use, there are tensors, nn (layers stack), etc

## Tensorflow
- computational graph is key
- code the symbolic computation procedure, then run the real computation
- auto derive gradient

# Lecture Note 13

## Segmentation

### Semantic Segmentation
- method 1
    - take small patch of photo, run through cnn to get classification of the central pixel
    - iterate over the photo
- method 2
    - run 'fully convolutional' network & get all pixels classification at once
- multi-scale
- refinement, rnn & cnn ?
- learnable upsampling,  'deconvolution'

### Instance Segmentation
- SDS, region proposal
- Hypercolumns
- Cascades ???

## Attention Models

- Soft or Hard Attention for Captioning
- Soft Attention for Translation
- Soft Attention for Everything
- Attending to Arbitrary Regions
 
# Lecture Note 14 Unsupervised Learning

## Autoencoder
- train, encoder - decoder
- take encoder as feature extractor

## Variational Autoencoder
- not understood at all

## Generative Adversarial Nets

# Lecture Note 15 Invited talk from Jeff Dean

## Word Embedding
- distance means similiar meaning
- direction has interesting feature, for country &  capital, the vector between than is similiar

## traning speed up
- model parallelism
- data parallelism


















