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
 

Detailed slide by slide notes

fiest general notes: i exect most of you to alreadyknow most of the things we will talk about, this is just a talk to close potential gaps of knowledge, and bring us up to speed

I will go throught material somewhat quicly, if you have notes, quesitons, please interrupt

--
concepts in machine learning:
 - data (some input to a task we want an AI to do)
 - a model (the ai itself, gets data and outpus something, like an action or a classification)
 - a loss function (some way for us to quantify, how well the model did)
 - an optimizer (the programme that optimizes the model, to have do the task better)
--
 python code snippet, that show us an training loop implemented in pytorch:
  
for x in dataset
zerograd
forward
loss
backward
step


i explain the differnt lines here

--
some more terminiology:
  - the numbers that represtne the model itself, are called parameters or weights
  - the numbers that are created as we feed the data through the netowrk are called activation

--
a line about the concept of a the batch, we do not really see the loss landscape, just approximate it.
equation about batching, talking abotu batch size

--
we will now take a closer look at how all of these things are implemented, but first, let us have a refresher look at pytorch

the central element of consideration here are tensors (code example of difning a tensor)
 multidimensial arrays of numbers

we use these to represent: data, parameters, activations, output, and loss ...
by convention, everything except the parameters, has as the first diemiton the batch, so we can represnet many of them at the same time, and do a batched computation

--

 we have some handy shortnands for defning how to manipulate tehm
example: elementwise exponential
multipluing via @
einstein notation (exampl from einsum)
with ewuation to show how this is equivalent to einstein notation

--

(this is a file that builds up bit by bit)
they keep track of a computaitional graph (example wither we use different tensors to constuct a third)
then, when we have some number we want to take the gradient realtive to (we sum up the final tensor)
we can easily do this backward command (we show the final numner.backward)
and the rest of the computational graph, can analytically calculate thier gradient to that number
and now we can see each tensors gradient relative to that numner (we show that tensor.gradient is something)

--
Exercise 1, put a text "exercise 1" on screen with the clickable link to the text and also add a qr code to the exercise

--

Lets look at the loss function: how do we calculate how good a model is at a thin?
depends on the thing. for something like determin the position of the ball in the picutre, we could use abs x y distance
whenever the AI tries to predict soemthing discrete, like a class a pcicure belogns to, or the next word, we use Negatvie log likelyhodd, or in generalized form: cross entropy loss
minimizing this quantitiy makes our model well calibrated, we can interpret its output numebrs as probabiliteis it asignes
this is, when we can immediatly verify the output: called Supervised Learning
then there is also unsupervised

--
we do not always have the 'correct label' of something, sometimes we just know a pattern all outputs hsould have
example: raging how good text is, (one number per text), when all we have is a set of human preferences (I like A better then B) for a sparese sample of pairwise texts 
now we cant say, what the right answer has a particular property: for the human rated paris, the one that was preferred should be rated 
so we have a loss, that is lower when the differnce is in the correct directions
show the formula and cite @christiano2023a

--
in other occations, we do not know at all how a correct output might look like, but we can identify a correct output when we see one, like in chess. In that case we are in RL, and our loss funcitons look more complicated. as an example: we could jsut say that the loss funciotn on an output is negative the score of the model we trained with the method on the previous slide, though this is nor really how it is done

--
Optimizers:

in the learning loop, we have a gradient, but in what direction do we go for the next step

__
stochasit gradient descent
show the fromula, and name the hyperparameters

i talk about what it does

--
picture of a descent into a valey, with too small and too big learning rate

--
momentum
formula with named hyperparameters

--
show it 'rolling over' a local minimum

--
Adam
show equations with hyperparameters

__
exercise slide with links and stuff as before

--
models

--
example slide of an MLP

 - the ida, that we can represent a lot of things in more dimenstions and taht is why we dont just fit a funciotn from input to otuptu, but have a hidden layer in between
 if we just have input outptu connect, we an only rerpresent linear funciton
 - the hidden layer with hte nonlinearity lets us express more funcitons
 - in thoery we can express all funcitons that way (cite universality result)
however, this would not be efficient

arcitecural themes:

--

depth: we can mor easily comput ethings, if we comput thigns after each other, in muleiple sptes: for example when classifying a dog, it is helpfull to have a an early layer identifying ears and a a snoutk and later layers identfying ahead
- this is why, we usually stack arcitecrual themes a lot of times: models ahve layers that mosly look the same, but do differnt things
- we see an example code block stacking multultiple MLPs
--
resdiaul connections:
 - vanishinng graidens
 - chain rule often makes gradints go to zero
 - so we need side connection, to let all layers start learning someting 
 - we see a code bloack of the ML_ with a resdiaal conection, also a sketch of the siddual stream as an arrow in the enter and blcoks feeding in and out
--

using symetires in the data:
for example when looking at a pixel, each pixel might be a dog-ear. to learn this we compare it to its neughtbouse and see if that look slike a dog ear
- but this operation is tranlation ivaritant, so we do not need to learn it anew for every  point in the input picture
- we need a arcitecture that also has this symetire: convolutional nueral nets
- matrix defningion of convolutional symetire and a gif, showing how it works

--
transformers
similar, in text we have a 1d symetire: each word is the similr, but to understand words we not just need to look at its neighbours, but also at long-range conneciton, hence the transofmer, wich is the dominant arcitecure now, we will look at this in the afternoon;

gif of a transformer, acting on tokens in a sentence at the same time

--
exjrcise 3, same slide format

--
hyperparamter optimisation

any property that is not set by gradient descent is set by the experimenter
 -model arcitecure
 -batchisze, epochs...
 -momentum, lr and other optimizer properties

to find out how to se them we do:
- some of them we just have once found otu what good numebrs are, and keep using them, like momentum = 0.9 
- some of them, we just run a bunch of times, and see wich is best (typically learningn rate)
- some of them, we have empirical scaling laws, that can be cheaply measured in small experiments, and keep skaling to bigger expeirmetns (like numbers of parameters in a tranformer doing NLP) cite the chinchilla paper and include the central plot

