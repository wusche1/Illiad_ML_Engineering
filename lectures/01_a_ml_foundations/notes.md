Topics to cover:
 - Training loop
   - quickly going thorugh forward pass/backward pass
   - gradient descent
   - talk about the stochasiticiy: we want to descend on the real loss, but limited data at every step only ever gives an approximation
   - what the optimizer does
 - How tensors are represented in python / pytorch
   - looking at some basic pytorch operations, like casting a tensor, multiplying, doing an operation...
   - go through einops and notation
   - talk about batching, bath diemsion, data loading
   - concept of a computational tree and autograd, how the grads are implmented and accessable in pytorch
   x here, taking 25 min for the first exercise
   - moving tensor to/from devices. speedups. this is the limiting factor on model size/batch size
 - Loss functions
   - example of classificaiton tasks, talking about optimial scoring funcitons
   - example of human raters / RL loss function 
   - say a few words about sparsity
   - talk about test loss/ train loss and over/underfitting
 - going throught the different parts of:
  - what is a parameter/ activation/ hyperparameter
  - talk about different optimizers, get the intuition for momentum and RMLprop
  x have another 25 min practival of them implementing the optimizers
 - talk about arcitectures
  - activation funcitons as nonlinearities
  - the general concept that in DL we just repeat cirtain themes again and again
  - the central resutls taht nns are universal, and you can approximate any smoooth funciotn wiht a 1 layer NN
  - talk about over and underparametrasation and its connection to over/underfitting
  - arcitecture is there to make use of symetries
  - looking at:
   - convolutional neural networks for images
   - rough look at transformers for text
   - concept of a resudial stream against vanishing gradients
  x small exercise here of implmening some simple arcitecure
- Optimizing hyerparameters
 - hyperparameter sweeps
 - scaling laws
 x ply around either with a wandb setup or with the website to tune the spiral hyperparameter until Lunch, then share results 
