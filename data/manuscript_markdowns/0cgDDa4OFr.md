Sourcerer: Sample-based Maximum Entropy Source
Distribution Estimation

Julius Vetter†,1,2,∗

Guy Moss†,1,2,∗

Cornelius Schröder1,2

Richard Gao1,2

Jakob H. Macke1,2,3,∗

1Machine Learning in Science, Excellence Cluster Machine Learning, University of Tübingen
2Tübingen AI Center
3Department Empirical Inference, Max Planck Institute for Intelligent Systems
Tübingen, Germany
†Equal contribution.

Abstract

Scientific modeling applications often require estimating a distribution of param-
eters consistent with a dataset of observations—an inference task also known as
source distribution estimation. This problem can be ill-posed, however, since
many different source distributions might produce the same distribution of data-
consistent simulations. To make a principled choice among many equally valid
sources, we propose an approach which targets the maximum entropy distribu-
tion, i.e., prioritizes retaining as much uncertainty as possible. Our method is
purely sample-based—leveraging the Sliced-Wasserstein distance to measure the
discrepancy between the dataset and simulations—and thus suitable for simulators
with intractable likelihoods. We benchmark our method on several tasks, and
show that it can recover source distributions with substantially higher entropy than
recent source estimation methods, without sacrificing the fidelity of the simulations.
Finally, to demonstrate the utility of our approach, we infer source distributions
for parameters of the Hodgkin-Huxley model from experimental datasets with
hundreds of single-neuron measurements. In summary, we propose a principled
method for inferring source distributions of scientific simulator parameters while
retaining as much uncertainty as possible.

1

Introduction

In many scientific and engineering disciplines, mathematical and computational simulators are used
to gain mechanistic insights. A common challenge is to identify parameter settings of such simulators
that make their outputs compatible with a set of empirical observations. For example, by finding a
distribution of parameters that, when passed through the simulator, produces a distribution of outputs
that matches that of the empirical dataset of observations.

Suppose we have a stochastic simulator with input parameters θ and output x, which allows us to
generate samples from the forward model p(x|θ) (which is usually intractable). We have acquired
a dataset D = {x1, ..., xn} of observations with empirical distribution po(x), and want to identify

∗{firstname.secondname}@uni-tuebingen.de
Code available at https://github.com/mackelab/sourcerer

38th Conference on Neural Information Processing Systems (NeurIPS 2024).

a distribution q(θ) over parameters that, once passed through the simulator, yields a “pushforward”
distribution of simulations q#(x) = (cid:82) p(x|θ)q(θ)dθ that is indistinguishable from the empirical
distribution. This setting is known by different names in different disciplines, for example as unfolding
in high energy physics [10], stochastic inverse problems in various disciplines [7], population
of models in electrophysiology [30] and population inference in gravitational wave astronomy
[55]. Adopting the terminology of Vandegar et al. [58], we refer to this task as source distribution
estimation.

A common approach to source distribution estimation is empirical Bayes [51, 15]. Empirical Bayes
uses hierarchical models in which each observation is modeled as arising from different parameters
p(xi|θi). The hyper-parameters of the prior (and thus the source qϕ) are found by optimizing the
(cid:82) p(xi|θ)qϕ(θ)dθ over ϕ. Empirical Bayes has been successfully
marginal likelihood p(D) = (cid:81)
i
applied to a range of applications [31, 32, 55]. However, empirical Bayes is typically not applicable
to models with intractable likelihoods, which is usually the case for scientific simulators. Using
surrogate models for such likelihoods, empirical Bayes has been extended to increasingly more
complicated parameterizations ϕ of the source distribution, including neural networks [59, 58].

A more general issue, however, is that the source
distribution problem can often be ill-posed with-
out the introduction of a hyper-prior or other
regularization principles, as also noted in Van-
degar et al. [58]: Distinct source distributions
q(θ) can give rise to the same data distribu-
tion q#(x) when pushed through the simula-
tor p(x|θ) (Fig. 1, illustrative example in Ap-
pendix A.7).

We here propose to use the maximum entropy
principle, i.e., choosing the “maximum igno-
rance” distribution within a class of distribu-
tions to resolve the ill-posedness of the source
distribution problem [19, 24]. The maximum
entropy principle formalizes the notion that a
good choice for distributions should “assume
less”. It has been applied to specific source dis-
tribution estimation problems in scientific disci-
plines such as cosmology [23] and high-energy
physics [10].

Figure 1: Maximum entropy source distribu-
tion estimation. Given an observed dataset D =
{x1, . . . , xn} from some data distribution po(x),
the source distribution estimation problem is to
find the parameter distribution q(θ) that reproduces
po(x) when passed through the simulator p(x|θ),
i.e. q#(x) = (cid:82) p(x|θ)q(θ)dθ = po(x) for all x.
This problem can be ill-posed, as there might be
more than one distinct source distribution. We
resolve this by targeting the maximum entropy dis-
tribution, which is unique.

Our contributions We introduce Sourcerer, a
general method for source distribution estima-
tion, providing two key innovations: First, we
target the maximum entropy source distribution to obtain a well-posed problem, thereby increasing
the entropy of the estimated source distributions at no cost to their fidelity. Second, we use general
distance metrics between distributions, in particular the Sliced-Wasserstein distance, instead of maxi-
mizing the marginal likelihood as in empirical Bayes. This allows evaluation of the objective using
only samples from differentiable simulators, removing the requirement to have tractable likelihoods.
We validate our method on multiple tasks, including tasks with high-dimensional observation space,
which are challenging for likelihood-based methods. Finally, we apply our method to estimate the
source distribution over the mechanistic parameters of the Hodgkin-Huxley model from a large
(∼ 1000 samples) dataset of electrophysiological recordings.

2 Methods

We formulate the source distribution estimation problem in terms of the maximum entropy principle.
The (differential) entropy H(p) of a distribution p(θ) is defined as

(cid:90)

H(p) = −

p(θ) log p(θ)dθ.

(1)

2

simulatorp(x|θ)Goal: source estimationsource2.1 Data-consistency and regularized objective

For a given distribution q(θ) and a simulator with (possibly intractable) likelihood p(x|θ), the
pushforward of q is given by q#(x) = (cid:82) p(x|θ)q(θ)dθ. The distribution q(θ) is a source distribution
if its pushfoward matches the observed data distribution po(x), that is, q# = po almost everywhere.
Equivalently, given a distance metric D(·, ·) between probability distributions P (X ) over the data
space X , a source distribution q is one which satisfies D(q#, po) = 0. In general, for a given
distribution of observations po(x) and likelihood p(x|θ), the source distribution problem is ill-posed
as there are possibly many different source distributions. The maximum entropy principle can be
employed to resolve this ill-posedness:
Proposition 2.1. Let Q = {q|q# = po} be the set of source distributions for a given likelihood p(x|θ)
and data distribution po. Suppose that Q is non-empty and compact. Then q∗ = arg maxq∈Q H(q)
exists and is unique.

This proposition follows from the fact that the set of source distributions is convex and that the
(differential) entropy H(q) is a strictly concave functional. See Appendix A.7 for a proof and
additional assumptions.

Proposition 2.1 suggests to solve the constrained
optimization problem

max
ϕ

H(qϕ)

s.t. D(q#

ϕ , po) = 0,

(2)

where qϕ is some parametric family of distribu-
tions.

Practically, however, a solution might not exist,
for example due to simulator misspecification.
Furthermore, even if a solution exists, it is diffi-
cult to obtain since we only have a fixed number
of samples from po and can thus only estimate
D(q#
ϕ , po). We therefore propose a regularized
approximation of Eq. (2) and solve

max
ϕ

λH(qϕ) − (1 − λ) log(D(q#

ϕ , po)) (3)

Figure 2: Overview of Sourcerer. Given a source
distribution q(θ), we sample θ ∼ q and simulate
using p(x|θ) to obtain samples from the pushfor-
ward distribution q#(x) = (cid:82) p(x|θ)q(θ)dθ. We
maximize the entropy of the source distribution
q(θ) while regularizing with a Sliced-Wasserstein
distance (SWD) term between the pushforward of
q# and the data distribution po(x) (Eq. (3)). Θ and
X in top right corner of boxes denote parameter
space and data/observation space, respectively.

instead, where λ is a parameter determining the
strength of the data-consistency term and the
logarithm is added for numerical stability. This regularized objective is related to the Lagrangian
relaxation of Eq. (2), where now log D(q#, po) ≤ log ϵ for some ϵ > 0 and the dual variable is
(1 − λ)/λ.

For λ → 1, the loss in Eq. (3) is dominated by the entropy term, and for λ → 0 by the data-
consistency term. We apply ideas from constrained optimization and reinforcement learning [49, 4, 1]
and use a dynamical schedule during training. We initialize training with λt=1 = 1, and decay this
value linearly to a final value λt=T = λ > 0 over the course of training. This dynamical schedule
encourages the variational source model to first explore high-entropy distributions, and later increase
consistency with the data between high-entropy distributions. Pseudocode and details of the schedule
in Appendix A.3.

2.2 Reference distribution

For many tasks, there is an additional constraint in terms of a reference distribution p(θ). For example,
in the Bayesian inference framework, it is common to have a prior distribution p(θ), encoding existing
knowledge about the parameters θ from previous studies. In such cases, a distribution with higher
entropy than p(θ), even if it is a source distribution, is not always desirable. We therefore adapt our
objective function in Eq. (3) to minimize the Kullback-Leibler (KL) divergence between the source
q(θ) and the reference p(θ):

min
ϕ

λDKL(q||p) + (1 − λ) log(D(q#, po)).

(4)

3

simulatep(x|θ)Maximize, regularizeentropy H(q)mismatch SWDcompareSWD(   |   )data distributionpo(x)The KL divergence term can be rewritten as DKL(q||p) = −H(q) + H(q, p), where H(q, p) =
− (cid:82) log(p(θ))q(θ)dθ is the cross-entropy between q and p. Thus, provided we can evaluate the
density p(θ), we can obtain a sample-based estimate of the loss in Eq. (4). In our work, we consider
p(θ) to be the uniform distribution over some bounded domain BΘ (and hence the maximum entropy
distribution on this domain). This “box prior” is often used as the naive estimate from literature
observations in inference studies. More specifically, in this case, H(q, p) = −1/|BΘ|, where |BΘ|
is the volume of BΘ. Therefore, it is independent of q, and hence minimizing the KL divergence is
equivalent to maximizing H(q) on BΘ. In the case where p(θ) is non-uniform (e.g., Gaussian) the
cross-entropy term regularizes the loss by penalizing large q(θ) when p(θ) is small.

2.3 Sliced-Wasserstein as a distance metric

We are free to choose any distance metric D(·, ·) for the loss function Eq. (4). In this work, we use
the fast, sample-based, and differentiable Sliced-Wasserstein distance (SWD) [6, 27, 42] of order
two. The SWD is defined as the expected value of the one-dimensional Wasserstein distance between
the projections of the distribution onto uniformly random directions u on the unit sphere Sd−1 in Rd.
More precisely, the SWD is defined as

SWDm(p, q) = Eu∼U (Sd−1)[Wm(pu, qu)] ,
(5)
where pu is the one-dimensional distribution with samples u⊤x for x ∼ p(x), and Wm is the one-
dimensional Wasserstein distance of order m. In the empirical setting, where we are given n samples
each from pu and qu respectively, the one-dimensional Wasserstein distance is computed from the
order statistics as

Wm(pu, qu) =

||x(i)

p − x(i)

q ||m
m

,

(6)

(cid:32) n
(cid:88)

(cid:33)1/m

i=1

where x(i)
p denotes the i-th order statistic of the samples from pu (and similarly for x(i)
q ), and
|| · ||m denotes the Lm distance on R [47]. The time complexity of computing the sample-based
one-dimensional Wasserstein distance is thus the time complexity of computing the order statistics,
which is O(n log n) in the number of datapoints n [6]. This is significantly faster than computing the
multi-dimensional Wasserstein distance (O(n3), 29), or the commonly used Sinkhorn algorithm for
approximating the Wasserstein distance (O(n2) 47). While the SWD is not the same as the multi-
dimensional Wasserstein distance, it is still a valid metric on the space of probability distributions. In
particular, the SWD converges quickly with rate O(

n) to its true value [41, 42].

√

2.4 Differentiable simulators and surrogates

Our method only requires that sampling from the simulator p(x|θ) is a differentiable operation. In
practice, however, many simulators do not satisfy this property. For such simulators, we first train a
surrogate model. In particular, our method can make use of surrogates that model the likelihood only
implicitly. Such surrogate models can be easier to train and evaluate in practice. This is a distinct
requirement from likelihood-based approaches such as Vandegar et al. [58], which require that the
likelihood p(x|θ) can be evaluated explicitly and is differentiable. This means that our sample-based
approach can be readily applied to a larger set of simulators than likelihood-based approaches.

2.5 Source model and entropy estimation

In this work we use neural samplers as proposed in Vandegar et al. [58] to parameterize a source model
qϕ. These samplers employ unconstrained neural network architectures (in our case a multi-layer
perceptron) to transform a random sample from z ∈ N (0, I) into a sample from qϕ. While neural
samplers do not have a tractable likelihood, they are faster to evaluate than models with tractable
likelihoods. Furthermore, by using unconstrained network architectures, neural samplers are flexible
and additional constraints (e.g., symmetry, monotonicity) are easy to introduce.

To use likelihood-free source parameterizations, we require a purely sample-based estimator for the
entropy H(qϕ). This can be done using the Kozachenko-Leonenko entropy estimator [28, 3], which
is based on a nearest-neighbor density estimate. We use the Kozachenko-Leonenko estimator in this
work for its simplicity, but note that sample-based entropy estimation is an active area of research,
and other choices are possible [48]. Details about the Kozachenko-Leonenko estimator can be found
in Appendix A.6.

4

Figure 3: Results for the source estimation benchmark. (a) Original and estimated source and
corresponding pushforward for the differentiable IK simulator (λ = 0.35). The estimated source
has higher entropy than the original source that was used to generate the data. The observations
(simulated with parameters from the original source) and simulations (simulated with parameters
from the estimated source) match. (b) Performance of our approach for all four benchmark tasks (TM,
IK, SLCP, GM) using both the original (differentiable) simulators, and learned surrogates. Source
estimation is performed without (NA) and with entropy regularization for different choices of λ. For
all cases, mean C2ST accuracy between observations and simulations (lower is better) as well as
the mean entropy of estimated sources (higher is better) over five runs are shown together with the
standard deviation. The gray line at λ = 0.35 (λ = 0.062 for GM) indicates our choice of final λ for
the numerical benchmark results (Table 1).

3 Experiments

To evaluate the data-consistency and entropy of source distributions estimated by Sourcerer, we
benchmark our method against Neural Empirical Bayes (NEB) [58], a state-of-the-art approach to
source distribution estimation. The benchmark comparison is performed on four source distribution
estimation tasks including three presented in Vandegar et al. [58]. We then demonstrate the advantage
of Sourcerer in the case of differentiable simulators with a high-dimensional data domain, where
likelihood-based empirical Bayes approaches would require training a likelihood surrogate. Finally,
we use Sourcerer to estimate the source distribution for a Hodkgin-Huxley simulator of single-neuron
voltage dynamics from a large dataset of experimental electrophysiological recordings. For all
tasks except the Hodgkin-Huxley task (where the observed dataset is experimentally measured), we
generate two datasets of observations of equal size from the same reference source distribution. The
first is used to train the source model, and the second is used to evaluate the quality of the learned
source.

3.1 Source Estimation Benchmark

Benchmark tasks The source estimation benchmark contains four simulators: two moons (TM),
inverse kinematics (IK), simple likelihood complex posterior (SLCP), and Gaussian Mixture (GM)
(details about simulators and source distributions are in Appendix A.2). Notably, all four simulators
are differentiable. Therefore, we can evaluate our method directly on the simulator as well as trained
surrogates. For all four simulators, source estimation is performed on a synthetic dataset of 10000
observations that were generated by sampling from a pre-defined original source distribution and
evaluating the resulting pushforward distribution using the corresponding simulator. The quality of
the estimated source distributions is measured using a classifier two sample test (C2ST) [33] between
the observations and simulations from the source. We also report the entropy of the estimated sources.
Given two sources with the same C2ST accuracy, the higher entropy source is preferable. We compare

5

Table 1: Numerical benchmark results for Sourcerer. We show the mean and standard deviation
over five runs for differentiable simulators and surrogates of Sourcerer on the benchmark tasks, and
compare to NEB. All approaches achieve C2ST accuracies close to 50%. For the Sliced-Wasserstein-
based approach, the entropies of the estimated sources are substantially higher (bold) with the entropy
regularization (λ = 0.35 for TM, IK, SLCP, λ = 0.062 for GM, gray line in Fig. 3).

Method

TM

IK

SLCP

GM

C2ST acc.
Entropy

C2ST acc.
Entropy

C2ST acc.
Entropy

C2ST acc.
Entropy

Sourcerer
Sim. (with reg.)

Sourcerer
Sim. (w/o reg.)

Sourcerer
Sur. (with reg.)

Sourcerer
Sur. (w/o reg.)

NEB

0.51 (0.004)
1.26 (0.022)

0.51 (0.002)
3.75 (0.066)

0.53 (0.005)
9.81 (0.039)

0.5 (0.008)
1.0 (0.198)

0.51 (0.005)
1.59 (0.246)

0.53 (0.006)
7.23 (0.052)

0.51 (0.003)
1.21 (0.054)

0.51 (0.005)
3.78 (0.022)

0.55 (0.003)
9.74 (0.039)

0.51 (0.006)
1.02 (0.162)

0.53 (0.005)
1.13 (0.093)

0.51 (0.01)
1.7 (0.165)

0.6 (0.014)
0.82 (0.712)

0.59 (0.017)
6.76 (0.302)

0.53 (0.006)
7.56 (0.097)

0.51 (0.005)
-1.12 (0.083)

0.5 (0.006)
-1.25 (0.106)

0.54 (0.006)
-0.36 (0.095)

0.55 (0.005)
-2.19 (0.212)

0.52 (0.004)
-1.5 (0.052)

to the NEB estimator with the same parameterization of the source model and 1024 Monte Carlo
samples to estimate the marginal likelihood (details in Appendix A.3).

Benchmark performance We first check whether minimizing the Sliced-Wasserstein distance
without any entropy regularization finds good source distributions. This corresponds to the case λ = 0
in Eq. (3) without any decay. In this way, we compare the data-consistency objective in Eq. (4) to the
NEB objective of maximizing the marginal likelihood. We find that for the differentiable simulators,
the Sliced-Wasserstein-based approach is able to find good source distributions with C2ST accuracies
close to 50% for all benchmark tasks (Fig. 3, labeled NA). This also applies when we use surrogate
models to generate the pushforward distributions. In particular, the quality of the estimated source
distributions matches those found by NEB (Table 1).

We then apply entropy regularization as defined in Eq. (3) for all benchmark tasks. The entropy of the
estimated sources is drastically increased without any cost in the quality of the simulations (Fig. 3b).
While C2ST accuracy remains close to 50% across all benchmark tasks, the entropy of estimated
sources is substantially higher than that of sources estimated with NEB, or when minimizing only
the data-consistency term (Table 1). We also explore the dependence of the results on the final
regularization strength λ (Fig. 3b). We observe a sharp trade-off: above a critical value of λ, the
SWD term becomes too weak, and the fidelity of the simulations rapidly declines. However, below
this critical value of λ, the results are robust relative to λ: the estimated sources produce simulations
that match the observations, and have comparable entropy.

Additionally, for both IK and SLCP simulators, the entropy of the sources estimated by our method is
higher than the entropy of the original source distribution (Fig. 3a and Fig. A7) despite the simulations
and observations being indistinguishable from each other (C2ST accuracy: 50%). This does not
contradict our approach: The original source distribution just happens not to be the maximum entropy
source for these simulators.

We also investigate the robustness of our approach to the choice of the differentiable, sample-
based distance by repeating all experiments for these benchmark tasks using the Maximum Mean
Discrepancy (MMD, 22) and find comparable results (Fig. A4). Finally, we demonstrate (Fig. A5) the
robustness of our approach for small dataset sizes by repeating the Two Moons task with (N = 100)
observations (as opposed to 10000), and for high-dimensional parameter spaces by repeating the
Gaussian Mixture task with D = 25 dimensions (as opposed to 2).

3.2 High-dimensional observations: Lotka-Volterra and SIR

Since our method is sample-based and does not require likelihoods, it is possible to estimate sources
by back-propagating through the differentiable simulators directly. This is advantageous especially for
simulators with high-dimensional outputs, as we no longer require to first train a surrogate likelihood
model, which can be challenging when faced with high-dimensional data such as time series. Here, we

6

Figure 4: Source estimation on differentiable simulators. For both the deterministic SIR model (a)
and probabilistic Lotka-Volterra model (b), the Sliced-Wasserstein distance (lower is better) between
observations and simulations as well as entropy of estimated sources (higher is better) for different
choices of λ and without the entropy regularization (NA) are shown. Mean and standard deviation
are computed over five runs.

highlight this capability of our method by estimating source distributions for two high-dimensional,
differentiable simulators: The Lotka-Volterra model and the SIR (Susceptible, Infectious, Recovered)
model. The Lotka-Volterra model is used to model the density of two populations, predators and prey.
The SIR model is commonly used in epidemiology to model the spread of disease in a population
(details about both models and source distributions in Appendix A.2). Compared to the benchmark
tasks in Sec. 3.1, the dimensionality of the data space is much larger: Both the Lotka-Volterra and
the SIR model are simulated for 50 time points resulting in a 100 and 50 dimensional time series,
respectively.

Furthermore, to show that unlike NEB (which maximizes the marginal likelihood), our sample-based
approach is applicable to deterministic simulators, we use a deterministic version of the SIR model
with no observation noise. Similarly to the benchmark tasks, we define a source, and simulate 10000
observations using samples from this source to define a synthetic dataset on which to perform source
distribution estimation. Here, we directly evaluate the quality of the estimated source distributions
using the Sliced-Wasserstein distance. We compare this distance to the minimum expected distance,
which is the distance between simulations of different sets of samples from the same original source.
For a comparison with NEB, we train surrogate models with a reduced dimensionality and again
compute C2ST accuracies and entropies of the estimated sources (see Appendix A.5 and Fig. A3 for
details on surrogate training and pushforward plots).

Source estimation for the deterministic SIR model Our method is able to estimate a good source
distribution for the deterministic SIR model: The Sliced-Wasserstein distance between simulations
and observations is close to the minimum expected distance (Fig. 4a). In contrast to the benchmark
tasks, estimating sources with entropy regularization does not lead to an increase in entropy for the
SIR model, and the quality of the estimated source remains constant for various choices of λ. A
possible explanation for this is that there is no degeneracy in the parameter space of the deterministic
simulator, and there exists only one source distribution.

Source estimation for the probabilistic Lotka-Volterra model For the probabilistic Lotka-
Volterra model, our method is also capable of estimating source distributions. As for the SIR model,
the Sliced-Wasserstein distance between simulations and observations is close to the minimum
expected distance (Fig. 4b). However, unlike the SIR model, estimating the source with entropy
regularization yields a large increase in entropy compared to when not using the regularization. For
the Lotka-Volterra model, our method yields a substantially higher entropy at no additional cost in
terms of source quality.

When using the surrogate models with reduced dimensionality to estimate the source distributions, we
find that Sourcerer achieves better C2ST accuracies than NEB. Furthermore, for the Lotka-Volterra
model, the entropy regularization again leads to a substantial increase in the entropy of the estimated
sources (Table 2). In summary, the experiments on the SIR and Lotka-Volterra models show that our
approach is able to scale to higher dimensional problems and can use gradients of complex simulators
to estimate source distributions directly from a set of observations.

7

Table 2: Numerical results for the SIR and Lotka-Volterra model We show the mean and
standard deviation over five runs for differentiable simulators and surrogates of Sourcerer on the
high-dimensional SIR and Lotka-Volterra (LV) models, and compare to NEB. For the comparison
with NEB, we train the required surrogate models with reduced dimensionality (25 dimensions instead
of 50 or 100). Sourcerer achieves C2ST accuracies close to 50%. For NEB, the C2ST accuracies
are worse. For the LV model, the entropies of the estimated sources are higher with the entropy
regularization (λ = 0.015 for SIR, λ = 0.125 for LV).

Method

Sourcerer
Sim. (with reg.)

Sourcerer
Sim. (w/o reg.)

Sourcerer
Sur. (with reg.)

Sourcerer
Sur. (w/o reg.)

NEB

SIR

LV

C2ST acc.
Entropy

C2ST acc.
Entropy

0.56 (0.013)
-2.3 (0.079)

0.57 (0.009)
0.29 (0.017)

0.56 (0.015)
-2.37 (0.169)

0.52 (0.001)
-1.34 (0.087)

0.55 (0.005)
-2.29 (0.076)

0.55 (0.005)
-2.5 (0.05)

0.76 (0.024)
-0.63 (0.174)

0.56 (0.005)
0.34 (0.05)

0.54 (0.009)
-1.01 (0.13)

0.62 (0.011)
-1.28 (0.073)

3.3 Estimating source distributions for a single-compartment Hodgkin-Huxley model

Single-compartment Hodgkin-Huxley simulator and summary statistics The single-
compartment Hodgkin-Huxley model consists of a system of coupled ordinary differential equations
simulating different ion channels in a neuron. We use the simulator described in Bernaerts et al. [2]
with 13 parameters. In data space, we use five commonly used summary statistics of the observed
and simulated spike trains. These are the (log of the) number of spikes, the mean of the resting
potential, and the mean, variance and skewness of the voltage during external current stimulation.
As the internal noise in the simulator has little effect on the summary statistics, we train a simple
multi-layer perceptron as surrogate on 106 simulations. The parameters used to generate these
training simulations were sampled from a uniform distribution that was used as the prior in Bernaerts
et al. [2] (details on simulator, choice of surrogate and the surrogate training in Appendix A.9).

Using this surrogate, we estimate source distributions from a real-world dataset of electrophysiological
recordings. The dataset [52] consists of 1033 electrophysiological recordings from the mouse motor
cortex. In general, parameter inference for Hodgkin-Huxley models can be challenging as models
are often misspecified [56, 2]. Thus, estimating the source distribution for this task is useful for
downstream inference tasks, as the prior knowledge gained can significantly constrain the parameters
of interest.

Source estimation for the Hodgkin-Huxley model On visual inspection, simulations from the
estimated source look similar to the original recordings (all observations spike at least once, spikes
have similar magnitudes) and show none of the unrealistic properties (e.g., spiking before the stimulus
is applied) that can be observed in some of the box uniform prior simulations (Fig. 5a). This match is
also confirmed by the distribution of summary statistics, which match closely between simulations
and observations (Fig. 5b). Furthermore, our method achieves good C2ST accuracy of ≈ 61% for
different choices of λ (Fig. 5d), as well as a small Sliced-Wasserstein distance of ≈ 0.08 in the
standardized space of summary statistics (Fig. 5e). While the source estimated without entropy
regularization also achieves good fidelity, its entropy is significantly lower than any of the source
distributions estimated with entropy regularization (Fig. 5d/e, example source distribution in Fig. 5c,
full source in Fig. A11).

Overall, these results demonstrate the importance of estimating source distributions using the entropy
regularization, especially on real-world datasets: Estimating the source distribution without any
entropy regularization can introduce severe bias, since the estimated source may ignore entire regions
of the parameter space. In this example, the parameter space of the single-compartment Hodgkin-
Huxley model is known to be highly degenerate, and a given observation can be generated by multiple
parameter configurations [14, 39].

8

Figure 5: Source estimation for the single-compartment Hodgkin-Huxley model. (a) Example
voltage traces of the real observations of the motor cortex dataset, simulations from the estimated
source (λ = 0.25), and samples from the uniform distribution used to train the surrogate. (b) 1D
and 2D marginals for three of the five summary statistics used to perform source estimation. (c)
1D and 2D marginal distributions of the estimated source for three of the 13 simulator parameters.
(d) and (e) C2ST accuracy and Sliced-Wasserstein distance (lower is better) as well as entropy of
estimated sources (higher is better) for different choices of λ including λ = 0.25 (gray line) and
without entropy regularization (NA). Mean and standard deviation over five runs are shown.

4 Related Work

Neural Empirical Bayes High-dimensional source distributions have been estimated through
variational approximations to the empirical Bayes problem. Louppe et al. [34] train a generative
adversarial network (GAN) [20] qψ to approximate the source. The use of a discriminator to compute
an implicit distance makes this approach purely sample-based as well. In order to find the optimal ψ∗
of the true data-generating process, they augment the adversarial loss with a small entropy penalty on
the source qψ. This penalty encourages low entropy, point mass distributions, which is the opposite
of our approach. Vandegar et al. [58] take an empirical Bayes approach, and use normalizing flows
for both the variational approximation of the source and as a surrogate for the likelihood p(x|θ). This
allows for direct regression on the marginal likelihood, as all likelihoods can be computed directly.
Finally, the empirical Bayes problem is also known as “unfolding” in the particle physics literature
[10], “population inference” in gravitational wave astronomy [55], and “population of models” in
electrophysiology [30]. Approaches have been developed to identify the source distribution, including
classical approaches that seek to increase the entropy of the learned sources [50].

Simulation-Based Inference The use of variational surrogates of the likelihood of a simulator with
intractable likelihood is known as Neural Likelihood Estimation in the simulation-based inference
(SBI) literature [60, 45, 36, 11]. In neural posterior estimation [44, 35, 21], an amortized posterior
density estimate is learned, which can be applied to evaluate the posterior of a single observation
xi ∈ D, if a prior distribution p(θ) is already known. An intuitive but incorrect approach to source
distribution estimation would be to take the average posterior distribution over the observations D,

Gn(θ) =

1
n

n
(cid:88)

i=1

p(θ|xi).

(7)

The average posterior does not always (and typically does not) converge to a source distribution in the
infinite data limit, as shown for simple examples in Appendix A.8. Intuitively, the average posterior
becomes a worse approximation of a source distribution for simulators that have broader likelihoods.
Instead, SBI can be seen as a downstream task of source distribution estimation; once a prior has
been learned from the dataset of observations with source estimation, the posterior can be estimated
for each new observation individually.

9

Generalized Bayesian Inference Another field related to source estimation is Generalized Bayesian
Inference (GBI) [5, 40, 26]. GBI performs distance-based inference, as opposed to targeting the
exact Bayesian posterior. Similarly to our work, the distance function used in GBI can be arbitrarily
chosen for different tasks. However, GBI is used for single-parameter inference tasks, as opposed to
the source distribution estimation task considered in this work. Similarly, Bayesian non-parametric
methods [43, 38, 12] learn a posterior directly on the data space which can then be used to sample
from a posterior distribution over the parameter space.

5 Summary and Discussion

In this work, we introduced Sourcerer as a method to estimate source distributions of simulator
parameters given datasets of observations. This is a common problem setting across a range of
scientific and engineering disciplines. Our method has several advantages: first, we employ a
maximum entropy approach, improving reproducibility of the learned source, as the maximum
entropy source distribution is unique while the traditional source distribution estimation problem
can be ill-posed. Second, our method allows for sample-based optimization. In contrast to previous
likelihood-based approaches, this scales more readily to higher dimensional problems, and can be
applied to simulators without a tractable likelihood. We demonstrated the performance of our approach
across a diverse suite of tasks, including deterministic and probabilistic simulators, differentiable
simulators and surrogate models, low- and high-dimensional observation spaces, and a contemporary
scientific task of estimating a source distribution for the single-compartment Hodgkin-Huxley model
from a dataset of electrophysiological recordings. Throughout our experiments, we have consistently
found that our approach yields higher entropy sources without reducing the fidelity of simulations
from the learned source.

Limitations
In this work, we used the Sliced-Wasserstein distance (and MMD) for the data-
consistency term between simulations and observations. In practice, different distance metrics can
lead to different estimated sources, depending on its sensitivity to different features. While our
method is compatible with any sample-based differentiable distance metric between two distributions,
there is still an onus on the practitioner to carefully select a reasonable distance metric for the data at
hand. For example, in some cases, it might be appropriate to use a combination of several distance
metrics for different modalities of the data. Similarly, there is a dependence on the final regularization
strength λ. Principled methods for defining the regularization strength are desirable, though as we
demonstrate, our results are robust to a large range of λ.

In addition, the method requires a differentiable simulator, which in practice may require the training
of a surrogate model, for example, when dealing with a (partially) discrete simulator. While this
is a common requirement for simulation-based methods, this could present a challenge for some
applications. Finally, in our work, we enforce the maximum entropy principle on the entire (parameter)
source distribution. In practice, for example when constructing prior distributions for Bayesian
inference, there are other choices, such as the Jeffrey’s prior [9].

Acknowledgements

This work was funded by the German Research Foundation (DFG) under Germany’s Excellence
Strategy – EXC number 2064/1 – 390727645 and SFB 1233 ’Robust Vision’ (276693517). This work
was co-funded by the German Federal Ministry of Education and Research (BMBF): Tübingen AI
Center, FKZ: 01IS18039A and the European Union (ERC, DeepCoMechTome, 101089288). Views
and opinions expressed are however those of the author(s) only and do not necessarily reflect those
of the European Union or the European Research Council. Neither the European Union nor the
granting authority can be held responsible for them. JV is supported by the AI4Med-BW graduate
program. JV and GM are members of the International Max Planck Research School for Intelligent
Systems (IMPRS-IS). We would like to thank Jonas Beck, Sebastian Bischoff, Michael Deistler,
Manuel Glöckler, Jaivardhan Kapoor, Auguste Schulz, and all members of Mackelab for feedback
and discussion throughout the project.

10

References

[1] Zafarali Ahmed, Nicolas Le Roux, Mohammad Norouzi, and Dale Schuurmans. Understanding
the impact of entropy on policy optimization. In International conference on machine learning,
2019.

[2] Yves Bernaerts, Michael Deistler, Pedro J Goncalves, Jonas Beck, Marcel Stimberg, Federico
Scala, Andreas S Tolias, Jakob H Macke, Dmitry Kobak, and Philipp Berens. Combined
statistical-mechanistic modeling links ion channel genes to physiology of cortical neuron types.
bioRxiv, 2023.

[3] Thomas B. Berrett, Richard J. Samworth, and Ming Yuan. Efficient multivariate entropy

estimation via k-nearest neighbour distances. The Annals of Statistics, 2019.

[4] D.P. Bertsekas and W. Rheinboldt. Constrained Optimization and Lagrange Multiplier Methods.

Computer science and applied mathematics. Elsevier Science, 2014.

[5] Pier Giovanni Bissiri, Chris C Holmes, and Stephen G Walker. A general framework for
updating belief distributions. Journal of the Royal Statistical Society Series B: Statistical
Methodology, 2016.

[6] Nicolas Bonneel, Julien Rabin, Gabriel Peyré, and Hanspeter Pfister. Sliced and Radon Wasser-

stein barycenters of measures. Journal of Mathematical Imaging and Vision, 2015.

[7] T. Butler, J. Jakeman, and T. Wildey. Combining push-forward measures and bayes’ rule
to construct consistent solutions to stochastic inverse problems. SIAM Journal on Scientific
Computing, 2018.

[8] E.K.P. Chong, W.S. Lu, and S.H. Zak. An Introduction to Optimization: With Applications to

Machine Learning. Wiley, 2023.

[9] Guido Consonni, Dimitris Fouskakis, Brunero Liseo, and Ioannis Ntzoufras. Prior Distributions

for Objective Bayesian Analysis. Bayesian Analysis, 2018.

[10] G. Cowan. Statistical Data Analysis. Oxford science publications. Clarendon Press, 1998.

[11] Kyle Cranmer, Johann Brehmer, and Gilles Louppe. The frontier of simulation-based inference.

Proceedings of the National Academy of Sciences, 2019.

[12] Charita Dellaporta, Jeremias Knoblauch, Theodoros Damoulas, and François-Xavier Briol.
Robust Bayesian inference for simulator-based models via the MMD posterior bootstrap. In
International Conference on Artificial Intelligence and Statistics. PMLR, 2022.

[13] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using Real NVP.

In International Conference on Learning Representations, 2017.

[14] Gerald M Edelman and Joseph A Gally. Degeneracy and complexity in biological systems.

Proceedings of the National Academy of Sciences, 2001.

[15] Bradley Efron and Carl Morris. Limiting the risk of Bayes and empirical Bayes estimators, part

ii: The empirical Bayes case. Journal of the American Statistical Association, 1972.

[16] Philip E. Gill, Walter Murray, and Margaret H. Wright. Practical Optimization. Society for

Industrial and Applied Mathematics, 2019.

[17] Manuel Glöckler, Michael Deistler, and Jakob H. Macke. Adversarial robustness of amortized

Bayesian inference. In International Conference on Machine Learning, 2023.

[18] Pedro J Gonçalves, Jan-Matthis Lueckmann, Michael Deistler, Marcel Nonnenmacher, Kaan
Öcal, Giacomo Bassetto, Chaitanya Chintaluri, William F Podlaski, Sara A Haddad, Tim P
Vogels, et al. Training deep neural density estimators to identify mechanistic models of neural
dynamics. Elife, 2020.

[19] I. J. Good. Maximum entropy for hypothesis formulation, especially for multidimensional

contingency tables. Annals of Mathematical Statistics, 1963.

11

[20] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil
Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural
Information Processing Systems, 2014.

[21] David S. Greenberg, Marcel Nonnenmacher, and Jakob H. Macke. Automatic posterior trans-
formation for likelihood-free inference. In International Conference on Machine Learning,
2019.

[22] A Gretton, KM. Borgwardt, MJ. Rasch, B Schölkopf, and Alexander Smola. A kernel two-

sample test. Journal of Machine Learning Research, 2012.

[23] Will Handley and Marius Millea. Maximum-entropy priors with derived parameters in a

specified distribution. Entropy, 2018.

[24] Edwin T. Jaynes. Prior probabilities. IEEE Transactions on Systems Science and Cybernetics,

1968.

[25] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint

arXiv:1412.6980, 2014.

[26] Jeremias Knoblauch, Jack Jewson, and Theodoros Damoulas. An optimization-centric view on
bayes’ rule: Reviewing and generalizing variational inference. Journal of Machine Learning
Research, 2022.

[27] Soheil Kolouri, Kimia Nadjahi, Umut Simsekli, Roland Badeau, and Gustavo Rohde. Gener-
alized Sliced Wasserstein distances. In Advances in Neural Information Processing Systems,
2019.

[28] L. Kozachenko and N. Leonenko. A statistical estimate for the entropy of a random vector.

Problems of Information Transmission, 1987.

[29] H. W. Kuhn. The Hungarian method for the assignment problem. Naval Research Logistics

Quarterly, 1955.

[30] Brodie A. J. Lawson, Christopher C. Drovandi, Nicole Cusimano, Pamela Burrage, Blanca
Rodriguez, and Kevin Burrage. Unlocking data sets by calibrating populations of models to
data density: A study in atrial electrophysiology. Science Advances, 2018.

[31] Tai Sing Lee and David Mumford. Hierarchical Bayesian inference in the visual cortex. J. Opt.

Soc. Am. A, 2003.

[32] Ning Leng, John A. Dawson, James A. Thomson, Victor Ruotti, Anna I. Rissman, Bart M. G.
Smits, Jill D. Haag, Michael N. Gould, Ron M. Stewart, and Christina Kendziorski. EBSeq:
an empirical bayes hierarchical model for inference in RNA-seq experiments. Bioinformatics,
2013.

[33] David Lopez-Paz and Maxime Oquab. Revisiting classifier two-sample tests. In International

Conference on Learning Representations, 2017.

[34] Gilles Louppe, Joeri Hermans, and Kyle Cranmer. Adversarial variational optimization of non-
differentiable simulators. In International Conference on Artificial Intelligence and Statistics,
2019.

[35] Jan-Matthis Lueckmann, Pedro J Goncalves, Giacomo Bassetto, Kaan Öcal, Marcel Nonnen-
macher, and Jakob H Macke. Flexible statistical inference for mechanistic models of neural
dynamics. In Advances in Neural Information Processing Systems, 2017.

[36] Jan-Matthis Lueckmann, Giacomo Bassetto, Theofanis Karaletsos, and Jakob H. Macke.
In Proceedings of The 1st Symposium

Likelihood-free inference with emulator networks.
on Advances in Approximate Bayesian Inference, 2019.

[37] Jan-Matthis Lueckmann, Jan Boelts, David Greenberg, Pedro Goncalves, and Jakob Macke.
Benchmarking simulation-based inference. In International Conference on Artificial Intelligence
and Statistics, 2021.

12

[38] Simon Lyddon, Chris C. Holmes, and Stephen G. Walker. General Bayesian updating and the

loss-likelihood bootstrap. Biometrika, 2017.

[39] Eve Marder and Adam L Taylor. Multiple models to capture the variability in biological neurons

and networks. Nature neuroscience, 2011.

[40] Takuo Matsubara, Jeremias Knoblauch, François-Xavier Briol, and Chris J Oates. Robust
generalised Bayesian inference for intractable likelihoods. Journal of the Royal Statistical
Society Series B: Statistical Methodology, 2022.

[41] Kimia Nadjahi, Alain Durmus, Umut Simsekli, and Roland Badeau. Asymptotic guarantees
for learning generative models with the sliced-wasserstein distance. In Advances in Neural
Information Processing Systems. Curran Associates, Inc., 2019.

[42] Kimia Nadjahi, Alain Durmus, Lénaïc Chizat, Soheil Kolouri, Shahin Shahrampour, and Umut
Simsekli. Statistical and topological properties of sliced probability divergences. In Advances
in Neural Information Processing Systems, 2020.

[43] Peter Orbanz and Yee Whye Teh. Bayesian nonparametric models. Encyclopedia of Machine

Learning, 2010.

[44] George Papamakarios and Iain Murray. Fast ϵ-free inference of simulation models with Bayesian
conditional density estimation. In Advances in Neural Information Processing Systems, 2016.

[45] George Papamakarios, David C. Sterratt, and Iain Murray. Sequential neural likelihood: Fast
likelihood-free inference with autoregressive flows. In International Conference on Artificial
Intelligence and Statistics, 2018.

[46] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan,
Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative
style, high-performance deep learning library. Advances in neural information processing
systems, 32, 2019.

[47] Gabriel Peyré and Marco Cuturi. Computational optimal transport. Found. Trends Mach. Learn.,

2018.

[48] Georg Pichler, Pierre Colombo, Malik Boudiaf, Günther Koliander, and Pablo Piantanida. A
differential entropy estimator for training neural networks. In International Conference on
Machine Learning, 2022.

[49] John Platt and Alan Barr. Constrained differential optimization. In Neural Information Process-

ing Systems, 1987.

[50] Marcel Reginatto, Paul Goldhagen, and Sonja Neumann. Spectrum unfolding, sensitivity
analysis and propagation of uncertainties with the maximum entropy deconvolution code
MAXED. Nuclear Instruments and Methods in Physics Research Section A: Accelerators,
Spectrometers, Detectors and Associated Equipment, 2002.

[51] Herbert E. Robbins. An empirical bayes approach to statistics. In Breakthroughs in Statistics:

Foundations and basic theory, 1956.

[52] Federico Scala, Dmitry Kobak, Matteo Bernabucci, Yves Bernaerts, Cathryn René Cadwell,
Jesus Ramon Castro, Leonard Hartmanis, Xiaolong Jiang, Sophie Laturnus, Elanine Miranda,
et al. Phenotypic variation of transcriptomic cell types in mouse motor cortex. Nature, 2021.

[53] Scott A Sisson, Yanan Fan, and Mark M Tanaka. Sequential monte carlo without likelihoods.

Proceedings of the National Academy of Sciences, 2007.

[54] Alvaro Tejero-Cantero, Jan Boelts, Michael Deistler, Jan-Matthis Lueckmann, Conor Durkan,
Pedro J. Gonçalves, David S. Greenberg, and Jakob H. Macke. sbi: A toolkit for simulation-
based inference. Journal of Open Source Software, 2020.

[55] Eric Thrane and Colm Talbot. An introduction to Bayesian inference in gravitational-wave
astronomy: Parameter estimation, model selection, and hierarchical models. Publications of the
Astronomical Society of Australia, 2019.

13

[56] Nicholas Tolley, Pedro LC Rodrigues, Alexandre Gramfort, and Stephanie Jones. Methods
and considerations for estimating parameters in biophysically detailed neural models with
simulation based inference. bioRxiv, 2023.

[57] Pravin M. Vaidya. An O(n log n) algorithm for the all-nearest-neighbors problem. Discrete &

Computational Geometry, 1989.

[58] Maxime Vandegar, Michael Kagan, Antoine Wehenkel, and Gilles Louppe. Neural empirical
Bayes: Source distribution estimation and its applications to simulation-based inference. In
International Conference on Artificial Intelligence and Statistics, 2020.

[59] Yixin Wang, Andrew C. Miller, and David M. Blei. Comment: Variational Autoencoders as

Empirical Bayes. Statistical Science, 2019.

[60] Simon N. Wood. Statistical inference for noisy nonlinear ecological dynamic systems. Nature,

2010.

[61] Omry Yadan. Hydra - a framework for elegantly configuring complex applications. Github,

2019. URL https://github.com/facebookresearch/hydra.

14

A Appendix

A.1 Software and data

We use PyTorch [46] for the source distribution estimation and hydra [61] to track all configurations.
Code to reproduce results is available at https://github.com/mackelab/sourcerer.

A.2 Simulators and sources

Here we provide a definition of the four benchmark tasks Two Moons (TM), Inverse Kinematics
(IK), Simple Likelihood Complex Posterior (SLCP) and Gaussian Mixture (GM), as well as the two
high-dimensional simulators, the SIR and Lotka-Volterra model. We also describe the original source
distribution used to generate the synthetic observations, and the bounds of the reference uniform
distribution on the parameters.

A.2.1 Two moons simulator

Dimensionality
Bounded domain
Original source

Simulator

References

x ∈ R2, θ ∈ R2
[−5, 5]2
θ ∼ U([−1, 1]2)

(cid:21)

x|θ =

(cid:20)r cos(α) + 0.25
r sin(α)

√
2
√
2
where α ∼ U (−π/2, π/2), r ∼ N (0.1, 0.012).
Vandegar et al. [58], Lueckmann et al. [37]

(cid:20) −|θ1 + θ2|/
(−θ1 + θ2)/

+

(cid:21)
,

A.2.2

Inverse Kinematics simulator

Dimensionality
Bounded domain
Original source
Simulator

References

x ∈ R2, θ ∈ R4
[−π, π]4
θ ∼ N (0, Diag( 1
x1 = θ1 + l1 sin(θ2 + ϵ) + l2 sin(θ2 + θ3 + ϵ) + l3 sin(θ2 + θ3 + θ4 + ϵ),
x2 = l1 cos(θ2 + ϵ) + l2 cos(θ2 + θ3 + ϵ) + l3 cos(θ2 + θ3 + θ4 + ϵ),
where l1 = l2 = 0.5, l3 = 1.0 and ϵ ∼ N (0, 0.000172).
Vandegar et al. [58]

4 , 1

4 , 1

2 , 1

4 ))

A.2.3 SLCP simulator

Dimensionality
Bounded domain
Original source
Simulator

References

x ∈ R8, θ ∈ R5
[−5, 5]5
θ ∼ U([−3, 3]5)
x|θ = (x1, . . . , x4), xi ∼ N (mθ, Sθ),

where mθ =

(cid:21)
, Sθ =

(cid:20)θ1
θ2

(cid:20) s2
1
ρs1s2

ρs1s2
s2
2

(cid:21)
, s1 = θ2

3, s2 = θ2

4, ρ = tanh θ5.

Vandegar et al. [58], Lueckmann et al. [37]

A.2.4 Gaussian mixture simulator

Dimensionality
Bounded domain
Original source
Simulator
References

x ∈ R2, θ ∈ R2
[−5, 5]2
θ ∼ U([0.5, 1]2)
x|θ ∼ 0.5N (x|θ, I) + 0.5N (x|θ, 0.01 · I).
Sisson et al. [53]

15

A.2.5 SIR model

Dimensionality
Bounded domain
Original source
Simulator

References

x ∈ R50, θ ∈ R2
[0.001, 3]2
β ∼ LogN ormal(log(0.4), 0.5) γ ∼ LogN ormal(log(0.125), 0.2)
x|θ = (x1, . . . , x50), where xi = Ii/N equally spaced and I is
simulated from dS
N , dI
with initial values S = N − 1, I = 1, R = 0 and N = 106.
Lueckmann et al. [37]

N − γI, dR

dt = −β SI

dt = β SI

dt = γI

A.2.6 Lotka-Volterra model

Dimensionality
Bounded domain
Original source

Simulator

References

x ∈ R100, θ ∈ R4
[0.1, 3]4
θ′ ∼ N (0, 0.52)4, pushed through θ = f (θ′) = exp(σ(θ′)),
where σ is the sigmoid function.
x|θ = (xX
1 , . . . , xY
where xX
and X, Y are simulated from dX
with initial values X = Y = 1.
Glöckler et al. [17]

i ∼ N (Y, 0.052) equally spaced,
dt = αX − βXY , dY

1 , . . . , xX
50, xY
i ∼ N (X, 0.052), xY

50),

dt = −γY + δXY

A.3 Pseudocode and details on source estimation for benchmark tasks

Pseudocode for Sourcerer is provided in Algorithm 1.

For both the benchmark tasks and high dimensional simulators, sources were estimated from 10000
synthetic observations that were generated by simulating samples from an original previously defined
source.

For the benchmark tasks, we used T = 500 linear decay steps from λt=0 to λt=T = λ and optimized
the source model using the Adam optimizer with a learning rate of 10−4 and weight decay of 10−5.
The two high dimensional simulators were optimized with a higher learning rate of 10−3 and T = 50
linear decay steps. In both cases, early stopping was performed when the overall loss in Eq. (4) did
not improve over a set number of training iterations.

As a baseline, we compare to Neural Empirical Bayes (NEB) as described in Vandegar et al. [58].
Specifically, we use the biased estimator with 1024 samples per observation (L1024), which are used
to compute the Monte Carlo integral. Unlike our Sliced-Wasserstein-based approach, NEB does
not operate on the whole dataset of observations directly but attempts to maximize the marginal
likelihood per observation and thus uses part of the observations as a validation set. To ensure a
fair comparison, we increased the number of observations to 11112 for all NEB experiments, which
results in a training dataset of 10000 observations when using 10% as a validation set. For training,
we again used the Adam optimizer (learning rate 10−4, weight decay 10−5, training batch size 128).

A.4 Source model

Throughout all our experiments, we use neural samplers as the source models [58]. The sampler
architecture is a three-layer multi-layer perceptron with dimension of 100, ReLU activations and batch
normalization as our source model. Samples are generated by drawing a sample s ∼ N (0, I) from
the standard multivariate Gaussian and then (non-linearly) transforming s with the neural network.

A.5 Surrogates for the benchmark tasks

We follow Vandegar et al. [58] and train RealNVP flows [13] as surrogates for the four benchmark
tasks. For all benchmark tasks, the RealNVP surrogates have a flow length of 8 layers with a hidden
dimension of 50.

Surrogates for the benchmark tasks were trained using the Adam optimizer [25] on 15000 samples
and simulator evaluations from the uniform distribution over the bounded domain (learning rate 10−4,
weight decay 5 · 10−5, training batch size 256). In addition, 20% of the data was used for validation.

16

Algorithm 1: Sourcerer
Inputs: Source model qϕ constrained on the bounded domain BΘ, observed dataset
D = {x1, ..., xn} ∼ po(x), differentiable model p(x|θ) to draw samples from (simulator or
surrogate), number of samples m to estimate entropy, regularization schedule λt=1, ..., λt=T .
Outputs: Trained source model qϕ(θ).

t ← 0;
while not converged do
θ1, . . . , θn ∼ qϕ(θ) ;
x′
i ∼ p(x|θi) ;
θ′
1, . . . , θ′
m ∼ qϕ(θ) ;
λ ← λt=t if t ≤ T else λt=T ;
L ← λH({θ′
ϕ ← ϕ − Adam(∇ϕL) ;
t ← t + 1

1, . . . , θ′

return qϕ

# sample parameters for pushforward
# sample pushforward
# sample parameters for entropy estimation
# schedule lambda
# compute loss
# update source model

1, . . . , x′

n}) ;

m} + (1 − λ)D({x1, . . . xn}, {x′

To train surrogate models for the SIR and Lotka-Volterra model, we first reduce the simulator
dimension in observation space to 25 in both cases. Additionally, we add a small amount of
independent Gaussian noise (N (X, 0.012)) to the output of the SIR simulator to avoid training
the normalizing flow surrogate with simulations from a deterministic likelihood. We then use 106
simulations to train and validate (20% validation set) both surrogate models, again using the Adam
optimizer (learning rate 5 · 10−4, weight decay 5 · 10−5, training batch size 256).

A.6 Kozachenko-Leonenko entropy estimator

Our use of neural samplers requires us to use a sample-based estimate of (differential) entropy, since
no tractable likelihood is available (see Sec. 2.5).
We use the Kozachenko-Leonenko estimator [28, 3] for a set of samples {θi}n
p(θ) ∈ P (Θ), given by

i=1 from a distribution

H(qϕ) ≈

d
m

(cid:34) n
(cid:88)

i=1

(cid:35)

log(di)

− g(k) + g(n) + log(Vd),

(8)

where di is the distance of θi from its k-th nearest neighbor in {θj}j̸=i, d is the dimensionality of Θ,
m is the number of non-zero values of di, g is the digamma function, and Vd is the volume of the unit
ball using the same distance measure as used to compute the distances di.

The Kozachenko-Leonenko estimator is differentiable and can be used for gradient-based optimization.
The all-pairs nearest neighbor problem can be efficiently solved in O(n log n) [57]. In practice,
we find all nearest neighbors by computing all pairwise distances on a fixed number of samples.
Throughout all experiments, 512 source distribution samples were used to estimate the entropy during
training.

A.7 Uniqueness of maximum entropy source distribution

Here, we prove the uniqueness of the maximum entropy source distribution (Proposition 2.1). First,
however, we demonstrate for a simple example that the source distribution without the maximum
entropy condition is not unique.

Example of non-uniqueness Consider the (deterministic) simulator x = f (θ) = |θ|. Further
assume that our observed distribution is the uniform distribution p(x) = U(x; a, b), where 0 < a < b.
Due the symmetry of f , the source distribution p(θ) for the observed distribution p(x) is not
unique. Any convex combination of form αu1(θ) + (1 − α)u2, where u1(θ) = U(θ; −b, −a) and
u2(θ) = U(θ; a, b) and α ∈ [0, 1] provides a source distribution. The maximum entropy source
distribution is unique and is attained if both distributions are weighted equally with α = 0.5.

17

Proof of Proposition 2.1 First, let us state Proposition 2.1 in full:
Let Θ ⊂ RdΘ and X ⊂ RdX be the parameter and observation spaces, respectively. Suppose that Θ
is compact. Let P(Θ) ⊂ L1(Θ) and P(X ) ⊂ L1(X ) be the set of probability measures on Θ and X
respectively. Let Q = {q|q# = po almost everywhere } ⊂ P(Θ) be the set of source distributions
for a given likelihood p(x|θ) and data distribution po ∈ P(X ). Suppose that Q is non-empty and
compact (in the L1 norm topology). Then q∗ = arg maxq∈Q H(q) exists and is unique.
First, by the compactness assumption on Θ, the (differential) entropy of all q ∈ P (Θ) is bounded
above (by the entropy of the uniform distribution on Θ), and so in particular it is finite. By the
compactness assumption on Q, the entropy achieves its supremum of Q, that is, there exists a q∗ such
that H(q∗) = arg maxq∈Q H(q). To show that q∗ is unique (up to L1-null sets), it is sufficient to
show two results: (1) that the set Q is a convex set, and (2) that entropy is strictly concave. In this
case, if we have two distinct suprema q∗
2 is a valid
source distribution with higher entropy, causing a contradiction. For the remainder of this proof, we
let q1 and q2 be two distinct source distributions. Their convex combination q = αq1 + (1 − α)q2,
α ∈ [0, 1] is a valid probability distribution supported on both of the supports of q1 and q2.

2, then any convex combination of q∗

1 and q∗

1, q∗

(1) Sources distributions are closed under convex combination: q is also a source distribution, since

(cid:90)

q#(x) =

p(x|θ) · (αq1(θ) + (1 − α)q2(θ))dθ

(cid:90)

= α

p(x|θ)q1(θ)dθ + (1 − α)

(cid:90)

p(x|θ)q2(θ)dθ

= αpo(x) + (1 − α)po(x) = po(x).

(2) Entropy is (strictly) concave: the entropy of q satisfies

(cid:90)

(cid:90)

H(q) = −

≥ −

(αq1(θ) + (1 − α)q2(θ)) · log(αq1(θ) + (1 − α)q2(θ))dθ

[αq1(θ) log(q1(θ)) + (1 − α)q2(θ) log(q2(θ))]dθ

= αH(q1) + (1 − α)H(q2),

(9)

(10)

where we used the fact that the function f (x) = x log x is convex on [0, ∞), and hence −f is concave.
Furthermore, f (x) is strictly convex on [0, ∞), so for any θ ∈ Θ, the equality of the integrands

αq1(θ) + (1 − α)q2(θ)) log(αq1(θ) + (1 − α)q2(θ)) = αq1(θ) log(q1(θ) + (1 − α)q2(θ) log(q2(θ)
(11)

holds if and only if α ∈ {0, 1} or q1(θ) = q2(θ). Since q1 and q2 are assumed distinct, that is, it holds
q1(θ) ̸= q2(θ) on a positive measure set, the integral equality in Eq. (10) only holds if α ∈ {0, 1},
and thus entropy is strictly concave, which concludes our proof.

□

Regularized regression as an approximation to constrained optimization In practice, we ap-
proximate the optimization problem in Eq. (2) with the regularized regression objective in Eq. (3).
As a result, we cannot use the result of Proposition 2.1 to guarantee the uniqueness of our solution.
However, the dynamic schedule approach to λ we use in our work (see Appendix A.3) is similar to
the penalty method of approximating solutions to constrained optimization tasks [16, 8]. Future work
could use this connection to apply theoretical knowledge of constrained optimization in the source
distribution estimation setting.

A.8 Examples related to the average posterior distribution

In general, the average posterior distribution is not a source distribution. The average posterior distribu-
tion is defined in Eq. (7). The infinite data limit is given by Gn(θ) n→∞−−−−→ G(θ) = (cid:82) p(θ|x)po(x)dx.

18

Here, we provide two examples, one based on coin flips, and one based on a Gaussian bimodal
likelihood to illustrate this point.

Coin-flip example Consider the classical coin flip example, where the probability of heads (H)
follows a Bernoulli distribution with parameter θ. The source distribution estimation problem for this
setting would consist of the outcomes of flipping n distinct coins, with potentially different values θi.
Proposition A.1. Suppose we have a Beta prior distribution on the Bernoulli parameter θ ∼
Beta(α, β) with parameters α = β = 1, and that the empirical measurements consist of 70% heads,
i.e.:

po(x) =

(cid:26)0.7
0.3

x = H
x = T

Then the average posterior G(θ) = (cid:82) p(θ|x)po(x)dx is not a source distribution for po(x).

Proof: Since the Beta distribution is the conjugate prior for the Bernoulli likelihood, the single-
observation posteriors are known to be p(θ|x = H) = Beta(2, 1) and p(θ|x = T) = Beta(1, 2).
Hence, the average posterior is

G(θ) = 0.3 · Beta(1, 2) + 0.7 · Beta(2, 1).

(12)

However, the ratio of heads observed when pushing this distribution through the Bernoulli simulator
is

G#(x = H) =

θ[0.3 · Beta(θ; 1, 2) + 0.7 · Beta(θ; 2, 1)]dθ

(cid:90) 1

0
(cid:90) 1

=

(cid:20)

θ

0.3

0
(cid:90) 1

0

= 2

1 − θ
B(1, 2)

+ 0.7

θ
B(2, 1)

(cid:21)

dθ

(13)

[0.3θ(1 − θ) + 0.7θ2]dθ

(cid:12)
1
(cid:12)
(cid:12)
(cid:12)
0
where we have used the fact that the Beta function takes the values B(1, 2) = B(2, 1) = 1/2.
Therefore, the pushforward of the average posterior distribution does not recover the correct ratio of
heads, and so it is not a source distribution.

≈ 0.567 ̸= 0.7,

= 0.3θ2 +

0.4θ3

2
3

Gaussian bimodal example As another illustrative example to show the differences between
average posterior and estimated source, we consider a one-dimensional, bimodal Gaussian likelihood
given by x|θ ∼ 0.5N (x|θ − 1, 0.32) + 0.5N (x|θ + 1, 0.32) and the source N (θ|0, 0.252). We use
the sbi package [54] and perform neural posterior estimation with the uniform prior θ ∼ U([−5, 5])
to obtain the average posterior and compare it to the source estimated with our approach.

While the estimated source matches the original source closely, the average posterior is visibly
different and substantially broader (Fig. A1). As expected, this difference persists when sampling
from the average posterior and estimated source to simulate from the likelihood. The pushforward
distributions in data space of the original and estimated source match, while the one of the average
posterior is again substantially different (Fig. A1).

Additional average posteriors (in comparison to original and estimated source distributions) for the
Two Moons and Gaussian mixture are shown in Fig. A6.

A.9 Details on source estimation for the single-compartment Hodgkin-Huxley model

We use the simulators as described in Bernaerts et al. [2] for our source estimation. This work
provides a uniform prior over a specified box domain, which we use as the reference distribution for
source estimation. Since the simulator parameters live on different orders of magnitude, we transform
the original m-dimensional box domain to the [−1, 1]m cube. Note that this transformation does not
affect the maximum entropy source distribution. This is because this scaling results in a constant
term added to the (differential) entropy. More specifically, for a random variable X (associated with

19

Figure A1: Failure of the average posterior as a source distribution for the bimodal likelihood example.
Each of the individual posteriors is bimodal, resulting in an average posterior with 3 modes (left), the
secondary modes produce observations which are not observed in the data distribution when pushed
through the likelihood (right), and should not be part of the source distribution.

its probability density p(x)), the (differential) entropy of X scaled by a (diagonal) scaling matrix D
and shifted by a vector c is given by

H(DX + c) = H(X) + log(det D).

(14)

The surrogate is trained on 106 parameter-simulation pairs produced by sampling parameters from the
uniform distribution and simulating with the sampled parameters. We do not use the simulated traces
directly, but instead compute 5 commonly used summary statistics [2, 18]. These are the number of
spikes k transformed by a log(k + 3) transformation (ensuring it is defined in the case of k = 0),
the mean of the resting potential, and the first three moments (mean, variance, and skewness) of the
voltage during the stimulation.

As our surrogate, we choose a deterministic multi-layer perceptron, because we found that the internal
noise has almost no noticeable effect on the summary statistics, so that the likelihood p(x|θ) is
essentially a point function. We are able to make this choice because the sample based nature of
our source distribution estimation approach is less sensitive to sharp likelihood functions, whereas
likelihood-based approaches could struggle with such problems.

The multi-layer perceptron (MLP) surrogate has 3 layers with a hidden dimension of 256. ReLU
activations and batch normalization were used. Training of the MLP was done with Adam (learning
rate 5 · 10−4, weight decay 10−5, training batch size 4096). Again, 20% of the data were used for
validation.

A.10 Computational Resources

All numerical experiments reported in this work were performed on GPU using an NVIDIA A100
GPU. A single source estimation run for a benchmark task using the Sourcerer approach (for one
value of λ) took approx. 30 seconds. In comparison, learning the source using NEB for the same task
took approx. 2 minutes (see Table A1). A source estimation run for Sourcerer on the high-dimensional
tasks took approx. 10 min. When the observations are high-dimensional, training a surrogate (if
required) makes up the majority of the computational cost. For the Hodgkin-Huxley task, training a
surrogate took approx. 20 minutes, after which estimating the source distribution with Sourcerer took
approx. 30 seconds.

20

Table A1: Wall-clock runtime comparison between Sourcerer and NEB. Time in seconds mea-
sured on an Nvidia A100 GPU. Average and standard deviation are shown over 5 runs. For all
three settings (Sourcerer with and without entropy regularization, NEB), surrogate models for the
benchmark simulators were used. Sourcerer converges noticeably faster than the NEB baseline.

Method

Sur. (w/o reg.)

Sur. (with reg.)

NEB

TM
IK
SLCP
GM

29.4 (8.5)
28.5 (6.9)
71.7 (12.8)
26.6 (5.4)

63.9 (10.1)
66.7 (10.0)
53.1 (12.2)
46.2 (9.2)

145.2 (13.9)
116.8 (22.6)
91.6 (9.9)
98.5 (15.5)

A.11 Supplementary figures

Figure A2: Extended results for source distribution estimation on the benchmark tasks (Fig. 3) for
different choices of λ. In addition to the C2ST accuracy and entropy, here the Sliced-Wasserstein
distance (SWD) between the observations and the pushforward distribution of the estimated source is
shown. Mean and standard deviation were computed over five runs.

21

Figure A3: Extended results for source distribution estimation on the differentiable SIR and Lotka-
Volterra models (Fig. 4). In addition to the Sliced-Wasserstein distance (SWD), the C2ST accuracy
between the observations and the pushforward distribution of the the estimated source is shown.
Despite the high-dimensional data space of the simulators (50 and 100 dimensions), the estimated
sources achieve a good C2ST accuracy (below 60%) for various choices of λ. Mean and standard
deviation were computed over five runs. Additionally, percentile values of all samples computed
per time point between simulations (simulated with parameters from the estimated source) and
observations (simulated with parameters from the original source) closely match.

Figure A4: Sourcerer with Maximum Mean Discrepancy (MMD) as the differentiable, sample-based
distance. We use MMD with an RBF kernel and the median distance heuristic for selecting the
kernel length scale. Source estimation is performed without (NA) and with entropy regularization for
different choices of λ. For these tasks, MMD produces similar results to the previously used SWD
(Fig. 3b). These results show that Sourcerer is compatible with other sample-based, differentiable
distances other than the SWD. For all cases, mean C2ST accuracy between observations and simula-
tions (lower is better) as well as the mean entropy of estimated sources (higher is better) over five
runs are shown together with the standard deviation.

22

Figure A5: Experiments with less observations and higher-dimensional sources. Source estimation
without (NA) and with entropy regularization for different choices of λ. For the Two Moons task,
the number of observations was reduced from 10000 to 100. For the Gaussian Mixture task, the
dimensionality was increased from 2 to 25. These results show that Sourcerer is robust to small
datasets of observations, and can estimate high-dimensional source distributions. For all cases, mean
C2ST accuracy between observations and simulations (lower is better) as well as the mean entropy of
estimated sources (higher is better) over five runs are shown together with the standard deviation.

Figure A6: Original and estimated sources distributions as well as average posterior distribution for
Two Moons and Gaussian Mixture simulator with uniform prior θ ∼ U([−5, 5]2). For simulators for
which the likelihood is unimodal and narrow, such as the Two Moons simulator, the average posterior
can be a good approximation of a source distribution. However, for simulators where the likelihood
is broader, such as the Gaussian Mixture simulator, the average posterior is too broad, and does not
reproduce the data distribution po well, when compared to estimates of source distributions.

23

Figure A7: Original and estimated source distributions for the benchmark SLCP simulator. The
estimated source has higher entropy than the original source.

24

Figure A8: Original and estimated source distributions for the SIR and Lotka-Volterra model. For the
Lotka-Volterra model, the estimated source has higher entropy than the original source.

25

Figure A9: 50 random example traces produced by sampling from the estimated source and simulating
with the Hodgkin-Huxley model.

26

Figure A10: 50 random example traces produced by sampling from the uniform distribution over the
box domain and simulating with the Hodgkin-Huxley model.

27

Figure A11: Estimated sources using for Hodgkin-Huxley task with the entropy regularization
(λ = 0.25) and without the entropy regularization. Without, many viable parameter settings are
missed, which would have significant downstream effects if the learned source distribution is used as
a prior distribution for inference tasks.

28

NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: We demonstrate in Table 1 our claim that we achieve source distributions
with higher entropy than a state-of-the-art comparison, and show results in Fig. 4 and
Fig. 5 that our method recovers source distributions on high dimensional tasks and the
electrophysiological data, respectively.
Guidelines:

• The answer NA means that the abstract and introduction do not include the claims

made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how

much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals

are not attained by the paper.

2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: We clearly mark the limitations discussion in Sec. 5.
Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that

the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The authors
should reflect on how these assumptions might be violated in practice and what the
implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms

and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to

address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?

29

Answer: [Yes]

Justification: Proposition 2.1 is stated with a full set of assumptions and a complete proof in
Appendix A.7.

Guidelines:

• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-

referenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a short
proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented

by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We provide pseudocode of our method in Algorithm 1. We provide full
details of the architecture of the source model and surrogates in Appendices A.4 and A.5,
respectively.

Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken

to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general. releasing code and data is often
one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case
of a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submis-
sions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how

to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe

the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct
the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.

30

5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: We use public data from existing work which we reference for the elec-
trophysiological dataset. The code necessary to reproduce our results is available at
https://github.com/mackelab/sourcerer.

Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/

public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).

• The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized

versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the

paper) is recommended, but including URLs to data and code is permitted.

6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?

Answer: [Yes]

Justification: We provide full details on training the source model in Appendix A.3, A.4,
A.5, A.6 and A.9.

Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail

that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental

material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The numerical results in Table 1 are reported with estimated standard devia-
tions, and the figures include error bars showing the standard deviation over an independent
set of runs with different random seeds.

Guidelines:

• The answer NA means that the paper does not include experiments.

31

• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula,

call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error

of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).

• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.

8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes]
Justification: We specify the computational resources used in our numerical experiments in
Appendix A.10. We provide a breakdown of the approximate computation time for each of
the experiments performed in this work.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster,

or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual

experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).

9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: We confirm that this work conform with all aspects of the NeurIPS Code of
Ethics.
Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a

deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-

eration due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [NA]

32

Justification: Our work is fundamental in that we develop a new approach to solving the
source distribution estimation problem. We do not develop new classes of models, nor do
we apply our approach to problems with societal implications. We do not foresee any direct
or indirect misuse of this work.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal

impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
generate deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).

11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification: This work does not involve models that have a high risk of misuse.
Guidelines:

• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors

should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.

12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]
Justification: We use a dataset of electrophysiological recordings from Scala et al. [52],
which we cite in the main text.
Guidelines:

• The answer NA means that the paper does not use existing assets.

33

• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a

URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of

service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of

the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to

the asset’s creators.

13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]
Justification: The public repository contains the code to reproduce our results, along with
necessary documentation. It is licensed under the MIT license.
Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose

asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either

create an anonymized URL or include an anonymized zip file.

14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [NA]
Justification: This work does not involve crowdsourcing nor research with human subjects.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with

human subjects.

• Including this information in the supplemental material is fine, but if the main contribu-
tion of the paper involves human subjects, then as much detail as possible should be
included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
or other labor should be paid at least the minimum wage in the country of the data
collector.

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human

Subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification: This work does not involve crowdsourcing nor research with human subjects.

34

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with

human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if

applicable), such as the institution conducting the review.

35

