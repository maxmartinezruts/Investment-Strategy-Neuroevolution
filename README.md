# Investment-Strategy-Neuroevolution

Author: Max Martinez Ruts
Note: Some explanations of this file are shared with my other project 'Neuroevolution of Augmented Topologies'


## Basic Idea
Use neuroevolution to optimize Neural Networks that can identify succesfull strategies to invest in the stock market.

The idea is to create a generation of different strategies and testing the strategies on different domains of the
stock market value for each generation. For every single generation, the strategies that lead to the highest capitals
after being trained on the same sample of the market are the ones that are going to be carried to the next generation
with subtle modifications. As an example, in the Figure below, strategy leading to the brown capital would be carried to the subsequent generation.  

<p align="center">
  <img src="capitalvstime.PNG" width="350" alt="accessibility text">
</p>

Figure: Capital records of all strategies being tested 


The NN in charge of determining the policy of the strategy take as inputs the nth derivatives of the last 10 values of
the sample taken and outputs the percentage the current capital to invest in the stock market.


# Neuroevolution
Neuroevolution is the application of genetic algorithms to optimize weights and biases in a fixed topology
neural network. Artificial neural networks (ANNs) are usually optimized using gradient techniques with
backpropagation. Neuroevolution, alternatively, considers the use of genetic algorithms to optimize neural
networks by using a reinforcement system where multiple ANNs are evaluated under a fitness score. ANNs
with superior scores are reproduced more often, creating a tendency on the set of ANNs to evolve towards
superior and thereby optimal ANNs.

# Artificial Neural Networks
An Artificial Neural Network (ANN) is a computing system inspired by biological neural networks. Its
structure is formed by connected nodes. Nodes can be seen as of biological neurons and the connections can
be seen as of biological synapses, which can transmit and process signals from one node to another.
This model allows any information to be processed while traveling from one part of the ANN to another.
The information is to be processed by scaling and shifting the incoming value by the use of weights and
biases, and subsequently applying an activation transform to the resulting value in order to introduce nonlinearities to the system. Thereby, the only values that determine how the information is to be processed
are the weights and biases, which ultimately determine the behavior of the ANN.

Commonly, ANNs are structured in layers. Layered structures organize the nodes in different layers.
By doing so, connections can only be present from nodes of one layer to nodes of subsequent layers, which
causes the information to travel in an acyclic, unidirectional manner; from the input layer to the output layer.
The complexity of ANN goes beyond the scope of this study. From now on, therefore, an ANN will be
treated as a processing function; a function processing an input to obtain an output. 

## Neural Network Topology
The topology of an ANN reefer to its structural organization of nodes and connections. Topology of ANNs
can contribute to their performance. Having described the scenario in Experiment Description section, the input and output layers can already be determined. The input layer consists of various nodes - 1st derivative, 2nd derivative, 3rd derivative...-. The output layer
consists of one node containing the information on the percentage the current capital to invest in the stock market. An extra hidden layer consisting of 6 nodes is also present to and depth - and therefore complexity - to the neural network. In
order to account for non-linearities within the input variables to predict the system’s behavior, the activation
function of the hidden and output layers are sigmoid functions, (used for probabilistic examples, as it outputs
values from 0 to 1). The outputs .
Figure 'Network Topology' displays the topology of the ANN to be used.

The neural network can, therefore, be seen as a function that inputs the ith derivatives of the stock market at the last opening, and outputs an investment decision (the percentage of the capital to invest in the stock market).
There are several methods to optimize an ANN depending on the behavior and the utility of the ANN.


<p align="center">
  <img src="ANNstrategy.png" width="350" alt="accessibility text">
</p>
Figure: Network Topology

# Genetic Algorithms
Genetic Algorithms (GAs) are inspired by Darwins theory of biological evolution. GAs differ from the
rest of optimization methods by the fact that a set of solutions is maintained rather than a singular one.
GAs are formed by a set of genetically diverse specimens, being evaluated under a certain fitness function.
Each specimen inherits a likelihood of survival, proportional to its evaluated fitness score. This behavior
is often referred to as survival of the fittest, the basic argument on which natural selection is based. This
behavior allows species to evolve, as specimens which survive longer are also the ones with more chances
to reproduce similar copies of themselves, thereby inducing a tendency on the population of subsequent
generations towards fitter specimens.

GAs are proved to be very effective in many real world and engineering optimization tasks such as electromagnetic system design and aerodynamic design. The reason why it has become widely used and
effective is due to its simplistic implementation in object-oriented programming. A parallelism is present
between biological evolution and GAs, populations being sets, specimens being objects, genes being weights
and biases and brains being ANNs. One could think of this parallelism by comparing the biological approach:

”Natural selection invokes a tendency on succeeding generations to produce populations with specimens
having smarter brains as a consequence of the evolution of their DNA. Such improvement leads to a general
rise on their performance”

With the programmatic approach:

”Genetic algorithms invokes a tendency on succeeding generations to produce sets with objects having
optimal neural networks as a consequence of the evolution of their weights and biases. Such improvement
leads to a general rise in their fitness score.”

## Selection
Selection is the determination process to decide which specimens will leave a successor on the next generation and which specimens will disappear without a successor. GAs use a different variety of selection
methods. Most of them, however, have a peculiarity in common; they tend to select fit specimens, based on
the behavior known as survival of the fittest.

## Selection method used
The selection method used for this specific scenario is based on the principle of survival of the fittest. The
population size is set constant to N throughout all generations. After all specimens of the population are
evaluated under their fitness scores, an algorithm is applied to decide which of the specimens will partake
in the reproduction process. This algorithm assigns a probability of reproduction equivalent to its fitness
score, therefore leaving fitter specimens with higher chances to take part in the reproduction process.
The following block of code illustrates how a progenitor for a newly generated specimen is selected.
Note that the method only works under the condition that the total score generated by all the population
is equal to 0. Therefore, the fitness scores have to be previously normalized such that the sum of all scores is 1:

function pick parent<br/>
.....i ← 0<br/>
.....r ← random(0, 1)<br/>
.....while r > 0 do<br/>
..........r ← r − evaluate fitness(specimens[i])<br/>
..........i ← i + 1<br/>
.....end while<br/>
.....i ← i − 1<br/>
.....return specimens[i]<br/>
end function<br/>

## Reproduction
In nature, reproduction is the process of generating new specimens sharing similar genetic information to
the one of their progenitors. Multiple reproduction methods are present in nature, and all of them have a
particularity in common; they are designed to ensure diversity of a species. Genetic diversity is achieved
by DNA alternation, which can be developed by many different processes. Two of the most common are
crossover and mutation.

### Crossover
Crossover is the ability for some specimens to reproduce a hybridized genetic copy of themselves. Crossover
is, therefore, a feature that allows the species to diversify. Crossover is commonly thought as sexual reproduction, which involves the union of a male and a female specimen. Although nature mostly experiences sexual reproduction with two progenitors, a crossover method can be designed where multiple progenitors are involved in the contribution of genes to the successor. However, in the experiment driven, bi-parent reproduction is chosen. A programmatic approach to deal with crossover is to simply create a hybrid genotype
from the predecessors.

As genotypes are composed by the weights and biases of the ANN, which are arranged as matrices, a
hybrid genotype can be generated by simply creating a weighted average of the individual genotypes of each
progenitor. The following block of code pictures a programmatic approach to achieve crossover.

function crossover<br/>
.....genotype1 ← pick parent<br/>
.....genotype2 ← pick parent<br/>
.....genotype ← (genotype1 + genotype2)/2<br/>
.....return genotype<br/>
end function<br/>

As the genetic information needed to model the ANN is composed by four different genotypes (weights hidden,
biases hidden, weights output, biases output), the crossover method will be applied to each of these four
genotypes.

### Mutation
Mutation is the capability for a specimen of modifying its genotype, providing the population the ability to
diversify. Diversity is key-driving in GAs, as it allows a population to have a broad genotype domain, which
expands the search domain, thereby increasing the possibilities from emerging new species with a unique set
of genes that drive speciments to increase their performance.

If diversification was nonexistent, a population would tend to converge to a single species. Certainly, it
would be the best species generated at the moment, but the new species would not tend to evolve, as all
individuals would tend to the fittest specimen in the population without the opportunity to generate specimens with distinct genotypes. It is therefore meaningful to define a parameter that represents the likelihood
for a gene of a specimen to be cloned from its progenitor. Such a parameter is referred to as mutation rate;
defined as the likelihood for a gene to be mutated.

If the mutation rate is set high, the population tends to diversify. However, the successors might not carry
enough genetic information from their progenitors, and thereby some of the genes that made the progenitor
a fit specimen could be lost. Similarly, if the mutation rate is set low, the progenitors will carry most of the
genetic information of theirs successors but this could lead to a poor tendency to evolve, as the diversity
of the species might be too low. Another relevant parameter is the mutation magnitude. When a gene of
the specimen is mutated, it can either be redefined as a random gene or as a variation of the original gene.
In the last case, a key parameter is the mutation magnitude; defined as the magnitude of how much is the
original gene varied.

The following block of code uses a random mutation magnitude, which can contribute to increasing the
diversity of the population, as it adds a random parameter to the method.

function mutation(genotypes, mutation rate)<br/>
.....for all gene in genotype do<br/>
..........r ← random(0, 1)<br/>
..........if r < mutation rate then<br/>
...............gene ← gene + random normal<br/>
..........end if<br/>
...end for<br/>

# Backpropagation vs GAs for ANN optimization
In Neural-Network optimization, backpropagation is the most common approach. Backpropagation is a gradient search technique. Such technique is based on dtermining the optimal direction in the domain space
(weights and biases) such that the loss is minimized, by the use of the chain rule. The loss function is defined
as the distance between the output of the neural network and the desired output. To use backpropagation
it is, therefore, necessary to label the outpus, as the loss function is the distance from the labeled outputs
to the resulting outputs This limits the use of backpropagation to supervised learning, as it is not possible
to optimize the neural network if the solution (output) is not known in advance. Moreover, gradient descent
techniques guarantee the best solution in the region of the starting point. Obtaining a global solution is
therefore dependent on the choice of initial starting values.

GAs tend to produce global solutions (see [6]). Specimens in a global solution are fitter than specimens
in a local solution, thereby inducing a tendency on the population towards global solutions. Due to this
behavior, GAs typically outperform backpropagation for very non-linear problems as described in [4], as
gradient descent tends to produce local solutions due to the loss function being hilly among the search
space. In GAs, instead, if the population size is kept high and the diversity of the population is high enough
such that a broad space can be covered, a global solution is encountered. Another essential advantage
over backpropagation is that GAs can be used for unsupervised learning and reinforcement learning as described in [2], as it is not based on a loss function (dependent on the desired output) but on a fitness function
which is rather dependent on the performance achieved by processing the input (state) to an output (action).

The previous reasoning leaves GAs to be the best fit for the problem scenario, as the solution to the
problem requires reinforcement learning and that the domain space is rather non-linear. But why is the case
that this scenario requires reinforcement learning?

The proposed problem lies in a scenario where the neural network has as inputs (ith derivatives) and output
(percentage to invest in stock market). However, given the inputs, the best output cannot be directly determined, as it is difficult to determine
if the action will produce a good or a bad result in order to achieve an increase in capital. In other words, a loss function
cannot be determined. This is because an the amount it is invested in the stock market won't directly determine the success of the policy. The success will be rather achieved by a sequence of good investment decisions. Therefore
singular good decision won't have a direct result on the stability of the object, but rather a chaotic
one. It is for this reason why an output cannot be directly compared to the desired output, precluding
the determination of a loss function and therefore precluding supervised learning. A similar situation is
encountered in chess. One can not train a supervised neural network by determining which is the best move
to do in a certain situation, as the effectuation of a move will not have an instant result on the game but
rather a chaotic one that will affect the result of the game in the future.

