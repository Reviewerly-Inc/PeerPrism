Published as a conference paper at ICLR 2022

META LEARNING LOW RANK COVARIANCE FACTORS
FOR ENERGY-BASED DETERMINISTIC UNCERTAINTY

Jeffrey Ryan Willette1, Hae Beom Lee1, Juho Lee1,2, & Sung Ju Hwang1,2
KAIST1, AITRICS2
{jwillette,haebeom.lee,juholee,sjhwang82}@kaist.ac.kr

ABSTRACT

Numerous recent works utilize bi-Lipschitz regularization of neural network layers
to preserve relative distances between data instances in the feature spaces of each
layer. This distance sensitivity with respect to the data aids in tasks such as
uncertainty calibration and out-of-distribution (OOD) detection. In previous works,
features extracted with a distance sensitive model are used to construct feature
covariance matrices which are used in deterministic uncertainty estimation or OOD
detection. However, in cases where there is a distribution over tasks, these methods
result in covariances which are sub-optimal, as they may not leverage all of the
meta information which can be shared among tasks. With the use of an attentive set
encoder, we propose to meta learn either diagonal or diagonal plus low-rank factors
to efﬁciently construct task speciﬁc covariance matrices. Additionally, we propose
an inference procedure which utilizes scaled energy to achieve a ﬁnal predictive
distribution which is well calibrated under a distributional dataset shift.

1

INTRODUCTION

Accurate uncertainty in predictions (calibration) lies at the heart of being able to trust decisions
made by deep neural networks (DNNs). However, DNNs can be miscalibrated when given out-of-
distribution (OOD) test examples (Ovadia et al., 2019; Guo et al., 2017). Hein et al. (2019) show that
the problem can arise from ReLU non-linearities introducing linear polytopes into decision boundaries
which lead to arbitrary high conﬁdence regions outside of the domain of the training data. Another
series of works (van Amersfoort et al., 2021; Liu et al., 2020a; Mukhoti et al., 2021; van Amersfoort
et al., 2021) link the problem to feature collapse, whereby entire regions of feature space collapse into
singularities which then inhibits the ability of a downstream function to differentiate between points
in the singularity, thereby destroying any information which could be used to differentiate them.
When these collapsed regions include areas of OOD data, the model loses any ability to differentiate
between in-distribution (ID) and OOD data.

A solution to prevent feature collapse is to impose bi-Lipschitz regularization into the network,
enforcing both an upper and lower Lipschitz bound on each function operating in feature space
(van Amersfoort et al., 2021; Liu et al., 2020a), preventing feature collapse. Such features from bi-
Lipschitz regualarized extractors are then used to improve downstream tasks such as OOD detection
or uncertainty quantiﬁcation. Broadly speaking, previous works have done this by constructing
covariance matrices from the resulting features in order to aid in uncertainty quantiﬁcation (Liu et al.,
2020a; Van Amersfoort et al., 2020) or OOD detection (Mukhoti et al., 2021). Intuitively, features
from a Lipschitz regularized extractor make for more expressive covariances, due to the preservation
of identifying information within different features.

However, empirical covariance estimation is limited when there are few datapoints on hand, such as in
few-shot learning. A key aspect of meta-learning is to learn meta-knowledge over a task distribution,
but as we show, empirical covariance estimation methods are not able to effectively encode such
knowledge, even when the features used to calculate the covariance come from a meta-learned feature
extractor (see Figure 6). As a result, the empirical covariance matrices are not expressive given
limited data and thus the model loses its ability to effectively adapt feature covariances to each task.

Another obstacle, highlighted by Mukhoti et al. (2021), is that plain softmax classiﬁers cannot
accurately model epistemic uncertainties. We identify a contributing factor to this, which is the shift

1

Published as a conference paper at ICLR 2022

(a) ProtoDDU

(b) Protonet

(c) ProtoSNGP

(d) Protonet

(e) Proto Mahalanobis

(f) Proto Mahalanobis

(g) Proto Mahalanobis

(h) Proto Mahalanobis

Figure 1: Top row: Examples of the learned entropy surface of baseline networks. Bottom row: our Proto
Mahalanobis models. Each pixel in the the background color represents the entropy given to that coordinate in
input space. Baseline networks exhibit high conﬁdence in areas where there has been no evidence, leading to
higher calibration error when presented with OOD data.

invariance property of the softmax function. Speciﬁcally, even if an evaluation point comes from an
OOD area and is assigned low logit values (high energy), this alone is insufﬁcient for a well calibrated
prediction. Small variations in logit values can lead to arbitrarily conﬁdent predictions due to the
shift invariance. From the perspective of Prototypical Networks (Snell et al., 2017), we highlight this
problem in Figure 3, although it applies to linear softmax classiﬁers as well.

In the following work, we ﬁrst propose a method of meta-learning class-speciﬁc covariance matrices
that is transferable across the task distribution. Speciﬁcally, we meta-learn a function that takes a set
of class examples as an input and outputs a class-speciﬁc covariance matrix which is in the form of
either a diagonal or diagonal plus low-rank factors. By doing so, the resulting covariance matrices
remain expressive even with limited amounts of data. Further, in order to tackle the limitation
caused by the shift invariance property of the softmax function, we propose to use scaled energy
to parameterize a logit-normal softmax distribution which leads to better calibrated softmax scores.
We enforce its variance to increase as the minimum energy increases, and vice versa. In this way,
the softmax prediction can become progressively more uniform between ID and OOD data, after
marginalizing the logit-normal distribution (see example in Figure 1).

By combining those two components, we have an inference procedure which achieves a well calibrated
probabilistic model using a deterministic DNN. Our contributions are as follows:

• We show that existing approaches fail to generalize to the meta-learning setting.

• We propose a meta learning framework which predicts diagonal or low-rank covariance

factors as a function of a support set.

• We propose an energy-based inference procedure which leads to better calibrated uncertainty

on OOD data.

2 RELATED WORK

Mahalanobis Distance. Mahalanobis distance has been used in previous works for OOD detection
(Lee et al., 2018) which also showed that there is a connection between softmax classiﬁers and
Gaussian discriminant analysis, and that the representation space in the latent features of DNN’s
provides for an effective multivariate Gaussian distribution which can be more useful in constructing
class conditional Gaussian distributions than the output space of the softmax classiﬁer. The method
outlined in Lee et al. (2018) provides a solid groundwork for our method, which also utilizes

2

Accuracy: 1.00NLL 0.00ENT ID/OOD: 0.00 / 0.01ECE ID/OOD 0.00 / 0.50traintestAccuracy: 1.00NLL 0.00ENT ID/OOD: 0.01 / 0.01ECE ID/OOD 0.00 / 0.50traintestAccuracy: 0.97NLL 0.25ENT ID/OOD: 0.50 / 0.50ECE ID/OOD 0.19 / 0.28traintestAccuracy: 0.81NLL 0.54ENT ID/OOD: 0.37 / 0.16ECE ID/OOD 0.08 / 0.83traintestAccuracy: 1.00NLL 0.04ENT ID/OOD: 0.07 / 0.55ECE ID/OOD 0.03 / 0.20traintestAccuracy: 0.96NLL 0.10ENT ID/OOD: 0.12 / 0.55ECE ID/OOD 0.04 / 0.19traintestAccuracy: 0.87NLL 0.38ENT ID/OOD: 0.43 / 1.14ECE ID/OOD 0.08 / 0.44traintestEnergytraintestPublished as a conference paper at ICLR 2022

(a) ProtoSNGP

(b) ProtoDDU
Figure 2: Comparison between covariances learned in SNGP (Liu et al., 2020a), DDU (Mukhoti et al., 2021) and
Proto Mahalanobis (Ours) in the few shot setting (half-moons 2-way/5-shot). Covariance generated from SNGP
are close to a multiple of the identity matrix, while that of ProtoMahalanobis contains signiﬁcant contributions
from off-diagonal elements.

(c) Proto Mahalanobis

Mahalanobis distance in the latent space, and adds a deeper capability to learn meta concepts which
can be shared over a distribution of tasks.

Post Processing. We refer to post-processing as any method which applies some function after
training and before inference in order to improve the test set performance. In the calibration literature,
temperature scaling (Guo et al., 2017) is a common and effective post-processing method. As the
name suggests, temperature scaling scales the logits by a constant (temperature) before applying
the softmax function. The temperature is tuned such that the negative log-likelihood (NLL) on a
validation set is minimized. Previous works which utilize covariance (Lee et al., 2018; Mukhoti
et al., 2021; Liu et al., 2020a) have also applied post-processing methods to construct latent feature
covariance matrices after training. While effective for large single tasks, these post-processing
methods make less expressive covariances in the meta learning setting, as demonstrated in Figure 1.

Bi-Lipschitz Regularization. Adding a regularizer to enforce functional smoothness of a DNN is
a useful tactic in stabilizing the training of generative adversarial networks (GANs) (Miyato et al.,
2018; Arjovsky et al., 2017), improving predictive uncertainty (Liu et al., 2020a; Van Amersfoort
et al., 2020), and aiding in OOD detection (Mukhoti et al., 2021). By imposing a smoothness
constraint on the network, distances which are semantically meaningful w.r.t. the feature manifold
can be preserved in the latent representations, allowing for downstream tasks (such as uncertainty
estimation) to make use of the preserved information. (Van Amersfoort et al., 2020) showed that
without this regularization, a phenomena known as feature collapse can map regions of feature space
onto singularities (Huang et al., 2020), where previously distinct features become indistinguishable.
For both uncertainty calibration and OOD detection, feature collapse can map OOD features onto the
same feature spaces as ID samples, adversely affecting both calibration and OOD separability.

Meta Learning. The goal of meta learning (Schmidhuber, 1987; Thrun & Pratt, 1998) is to
leverage shared knowledge which may apply across a distribution of tasks. In the few shot learning
scenario, models leverage general meta-knowledge gained through episodic training over a task
distribution (Vinyals et al., 2016; Ravi & Larochelle, 2017), which allows for effective adaptation and
inference on a task which may contain only limited amounts of data during inference. The current
meta-learning approaches are roughly categorized into metric-based (Vinyals et al., 2016; Snell et al.,
2017) or optimization-based approaches (Finn et al., 2017; Nichol et al., 2018). In this work, our
model utilizes a metric-based approach as they are closely related to generative classiﬁers, which
have been shown to be important for epistemic uncertainty (Mukhoti et al., 2021).

3 APPROACH

i=1 and a query set Q = {(xi, yi)}Nq

We start by introducing a task distribution p(τ ) which randomly generates tasks containing a support
set S = {(˜xi, ˜yi)}Ns
i=1. Then, given randomly sampled task
τ = (S, Q), we meta-learn a generative classiﬁer that can estimate the class-wise distribution of
query examples, p(x|y = c, S) conditioned on the support set S, for each class c = 1, . . . , C. A
generative classiﬁer is a natural choice in our setting due to fact that it utilizes feature space densities
which has been shown to be a requirement for accurate epistemic uncertainty prediction (Mukhoti
et al., 2021). Under the class-balanced scenario p(y = 1) = · · · = p(y = C) we can easily predict

3

Published as a conference paper at ICLR 2022

the class labels as follows.

p(y = c|x, S) =

p(x|y = c, S)
c(cid:48)=1 p(x|y = c(cid:48), S)

(cid:80)C

.

(1)

3.1 LIMITATIONS OF EXISTING GENERATIVE CLASSIFIERS

Possibly one of the simplest forms of deep generative classiﬁer is Prototypical Networks (Snell
et al., 2017). In Protonets we assume a deep feature extractor fθ that embeds x to a common metric
space such that z = fθ(x). We then explicitly model the class-wise distribution p(z|y = c, S) of the
embedding z instead of the raw input x. Under the assumption of a regular exponential family distri-
bution for pθ(z|y = c, S) and a Bregman divergence d such as Euclidean or Mahalanobis distance,
we have pθ(z|y = c, S) ∝ exp(−d(z, µc)) (Snell et al., 2017), where µc = 1
fθ(˜x) is the
|Sc|
class-wise embedding mean computed from Sc, the set of examples from class c. In Protonets, d is
squared Euclidean distance, resulting in the following likelihood of the query embedding z = fθ(x)
in the form of a softmax function.

˜x∈Sc

(cid:80)

pθ(y = c|z, S) =

exp(−(cid:107)z − µc(cid:107)2)
c(cid:48)=1 exp(−(cid:107)z − µc(cid:48)(cid:107)2)

(cid:80)C

.

(2)

1. Limitations of ﬁxed or empirical covariance. Unfortunately, Eq. (2) cannot capture a nontrivial
class-conditional distribution structure, as Euclidean distance in Eq. (2) is equivalent to Mahalanobis
distance with ﬁxed covariance I for all classes, such that pθ(z|y = c, S) = N (z; µc, I). For this
reason, many-shot models such as SNGP (Liu et al., 2020b) and DDU (Mukhoti et al., 2021) calculate
empirical covariances from data after training to aid in uncertainty quantiﬁcation. However, such
empirical covariance estimations are limited especially when the dataset size is small. If we consider
the few-shot learning scenario where we have only a few training examples for each class, empirical
covariances can provide unreliable estimates of the true class covariance. Unreliable covariance leads
to poor estimation of Mahalanobis distances and therefore unreliable uncertainty estimation.

2. Shift invariant property of softmax and OOD cal-
ibration. Another critical limitation of Eq. (2) is that
it produces overconﬁdent predictions in areas distant
from the class prototypes. The problem can arise from
the shift invariance property of the softmax function
σ(ω) = eω/ (cid:80)
with ω denoting the logits, such
that σ(ω + s) = eω+s/ (cid:80)
ω(cid:48) eω(cid:48)
= σ(ω) for any shift s. More speciﬁcally,
suppose we have two classes c = 1, 2, and z moves along the line extrapolating the prototypes µ1
and µ2 such that z = µ1 + c(µ2 − µ1) for c ≤ 0 or c ≥ 1. Then, we can easily derive the following
equality based on the shift invariant property of the softmax function:

Figure 3: (cid:107)µ2 − µ1(cid:107) remains the same
while z travels along the line, making a pre-
diction with unnecessarily low entropy.

ω(cid:48) eω(cid:48)+s = eω/ (cid:80)

ω(cid:48) eω(cid:48)

pθ(y = 1|z, S) =

1
1 + exp(±(cid:107)µ2 − µ1(cid:107))

(3)

where ± corresponds to the sign of c. Note that the expression is invariant to the value of c except for
its sign. Therefore, even if z is OOD, residing somewhere distant from the prototypes µ1 and µ2
with extreme values of c, we still have equally conﬁdent predictions. See Figure 3 for illustration.

3.2 META-LEARNING OF THE CLASS-WISE COVARIANCE

In order to remedy the limitations of empirical covariance, and capture a nontrivial structure of the
class-conditional distribution even with a small support set, we propose to meta-learn the class-wise
covariances over p(τ ). Speciﬁcally, we meta-learn a set encoder gφ that takes a class set Sc as
input and outputs a covariance matrix corresponding to the density p(z|y = c, S), for each class
c = 1, . . . , C. We expect gφ to encode shared meta-knowledge gained through episodic training over
tasks from p(τ ), which, as we will demonstrate in section 4, ﬁlls a key shortcoming of applying
existing methods such as DDU (Mukhoti et al., 2021) and SNGP (Liu et al., 2020a). We denote the
set-encoder gφ for each class c as

Λc, Φc = gφ(Zc),

Zc = {˜z − µc|˜z = fθ(˜x) and ˜x ∈ Sc}.

(4)

4

𝝁!𝝁"𝒛Published as a conference paper at ICLR 2022

Algorithm 1 Proto Mahalanobis – Training

Algorithm 2 Proto Mahalanobis – Inference

Sample a task τ = (S, Q)
for c = 1 to C do
(cid:80)
µc ← 1
˜x∈Sc
|Sc|
Λc, Φc ← gφ(Zc)
Σc ← Λc + ΦcΦ(cid:62)
c
Compute Σ−1
and |Σc|

1: Input: Task distribution p(τ ), initial θ and φ
2: Output: Meta-learned θ and φ
3: while not converged do
4:
5:
6:
7:
8:
9:
10:
11:
12:
13: end while

end for
Lτ ← 1
|Q|
(θ, φ) ← (θ, φ) − α∇θ,φLτ

fθ(˜x)

(cid:80)

c

(x,y)∈Q − log pθ,φ(y|E[ω]) (cid:46) Eq. 10

fθ(˜x)

1: Input: Task τ , meta-learned θ and φ
2: for c = 1 to C do
(cid:80)
µc ← 1
3:
˜x∈Sc
|Sc|
Λc, Φc ← gφ(Zc)
4:
Σc ← Λc + ΦcΦ(cid:62)
5:
c
Compute Σ−1
and |Σc|
6:
7: end for
8: Eval. pθ,φ(y|z, S) for (y, x) ∈ Q (cid:46) Eq. 11

(cid:46) Eq. 4
(cid:46) Eq. 5
(cid:46) Eq. 8,9

c

(cid:46) Eq. 4
(cid:46) Eq. 5
(cid:46) Eq. 8,9

where Λc ∈ Rd×d is a diagonal matrix and Φc ∈ Rd×r is a rank-r matrix. Now, instead of the identity
covariance matrix or empirical covariance estimation, we have the meta-learnable covariance matrix
consisting of the strictly positive diagonal and low-rank component for each class c = 1, . . . , C.

Σc = Λc + ΦcΦ(cid:62)
c .

(5)

It is easy to see that Σc is a valid positive semi-deﬁnite covariance matrix for positive Λc. Note
that the covariance becomes diagonal when r = 0. A natural choice for gφ is the Set Transformer
(Lee et al., 2019) which models pairwise interactions between elements of the input set, an implicit
requirement for covariance matrices.

Now, we let pθ,φ(z|y = c, S) = N (z; µc, Σc). From Bayes’ rule (see Appendix A.1), we compute
the predictive distribution in the form of softmax function as follows,

pθ,φ(y = c|z, S) =

pθ,φ(z|y = c, S)
c(cid:48)=1 pθ,φ(z|y = c(cid:48), S)

(cid:80)C

=

exp(− 1
c(cid:48)=1 exp(− 1

(cid:80)C

2 (z − µc)(cid:62)Σ−1

c (z − µc) − 1

2 log |Σc|)

2 (z − µc(cid:48))(cid:62)Σ−1

c(cid:48) (z − µc(cid:48)) − 1

2 log |Σc(cid:48)|)

(6)

(7)

Covariance inversion and log-determinant. Note that the logit of the softmax function in Eq. (7)
involves the inverse covariance Σ−1
and the log-determinant log |Σc|. In contrast to both DDU and
SNGP which propose to calculate and invert an empirical feature covariance during post-processing,
the meta-learning setting requires that this inference procedure be performed on every iteration during
meta-training, which may be cumbersome if a full O(d3) inversion is to be performed. Therefore, we
utilize the matrix determinant lemma (Ding & Zhou, 2007) and the Sherman-Morrison formula in the
following recursive forms for both the inverse and the log determinant in Equation 7.

c

(Σi + Φi+1Φ(cid:62)

i+1)−1

i+1 = Σ−1

i −

det(Σi + Φi+1Φ(cid:62)

i+1)i+1 = (1 + Φ(cid:62)

i Φi+1Φ(cid:62)
i+1Σ−1

i+1Σ−1
i Φi+1

i

Σ−1
1 + Φ(cid:62)
i+1Σ−1

i Φi+1)det(Σi)

(8)

(9)

3.3 OUT-OF-DISTRIBUTION CALIBRATION WITH SCALED ENERGY

Next, in order to tackle the overconﬁdence problem caused softmax shift invariance (Figure 3), we pro-
pose incorporating a positive constrained function of energy h(E) = max((cid:15), − 1
c exp(−Ec)),
with temperature T , into the predictive distribution. Energy has been used for OOD detection (Liu
et al., 2020b) and density estimation (Grathwohl et al., 2019), and the success of energy in these tasks
implies that it can be used to calibrate the predictive distribution (example in Figure 1h). Results in
Grathwohl et al. (2019) show improvements in calibration, but their training procedure requires a
full input space generative model during training, adding unwanted complexity if the end goal does
not require input space generation. Our method makes use of our logit values ω = (ω1, . . . , ωC)
to parameterize the mean of a logit-normal distribution with the variance given by h(E). In this

T log (cid:80)

5

Published as a conference paper at ICLR 2022

Table 1: OOD ECE on models trained on variations of the Omniglot and MiniImageNet datasets. The OOD
distribution for these models are random classes from the test set which are not present in the support set.

Omniglot OOD Class ECE ↓

MiniImageNet OOD Class ECE ↓

Model

5-way 5-shot

5-way 1-shot

20-way 5-shot

20-way 1-shot

5-way 1-shot

5-way 5-shot

MAML
Reptile
Protonet
Protonet-SN
ProtoDDU
ProtoSNGP
Ours (Diag)
Ours (Rank 1)

63.14±0.67
48.01±0.76
68.50±0.69
69.43±0.57
69.16±0.63
65.39±0.64
33.95±0.98
33.19±0.94

53.90±0.77
41.84±0.98
67.64±0.63
67.67±0.70
66.61±1.15
60.22±0.61
40.52±0.68
39.62±2.02

56.60±5.98
46.31±0.30
77.58±0.37
77.84±0.44
78.14±0.19
76.90±0.72
40.00±0.23
40.04±0.40

48.39±1.09
35.62±0.49
72.07±0.63
72.36±0.58
71.39±0.74
68.16±0.40
50.39±1.84
49.28±1.21

29.00±0.67
29.86±0.73
33.23±1.20
33.24±2.14
35.31±2.09
34.38±1.21
17.19±1.80
18.78±1.72

42.43±0.51
38.35±0.93
47.06±1.30
46.76±1.40
46.82±1.28
45.84±0.81
32.22±3.12
34.44±0.64

way, the logit-normal distribution variance rises in conjunction with the energy magnitude, making
predictions more uniform over the simplex for higher magnitude energies.

pθ,φ(ωc|z, S) = N (ωc; ˜µc, ˜σ), where ˜µc = −

˜σ = −

1
2
1
T

(z − µc)(cid:62)Σ−1

c (z − µc) −

1
2

log |Σc|,

(cid:88)

log

c(cid:48)

exp (cid:0)−(z − µc(cid:48))(cid:62)Σ−1

c(cid:48) (z − µc(cid:48))(cid:1)

(10)

Intuitively, h(E) is dominated by minc(|Ec|) thereby acting as a soft approximation to the minimum
energy magnitude (shortest Mahalanobis distance), which only becomes large when the energy is
high for all classes represented in the logits. Then, the predictive distribution becomes

pθ,φ(y = c|z, S) =

(cid:90)

p(y = c|ω)pθ,φ(ω|z, S)dω

≈

1
M

M
(cid:88)

m=1

c

exp(ω(m)
)
c(cid:48) exp(ω(m)

c(cid:48)

(cid:80)

, ω(m)

c ∼ p(ωc|z, S).

)

(11)

(12)

(cid:80)

Meta-training At training time, we do not sample ω and use the simple deterministic approx-
imation pθ,φ(y|z, S) ≈ pθ,φ(y|E[ω]). Therefore, the loss for each task becomes Lτ (θ, φ) =
(x,y)∈Q − log pθ,φ(y|E[ω]). We then optimize θ and φ by minimizing the expected loss
1
|Q|
Ep(τ )[Lτ (θ, φ)] over the task distribution p(τ ) via episodic training.
Inference with equation 11 can still beneﬁt from temperature scaling of ˜σ in 10.
Energy scaling.
Therefore, in order properly scale the variance to avoid underconﬁdent ID performance, we tune the
temperature parameter T after training. Speciﬁcally, we start with T = 1 and iteratively increase T
by 1 until ED[− log p(y|z, S)] ≤ ED[− log p(y|E[ω])], where − log p(y|E[ω]) is the NLL evaluated
by using only the deterministic logits E[ω].

3.4 SPECTRAL NORMALIZATION.

Lastly, we enforce a bi-Lipschitz regularization fθ by employing both residual connections and
spectral normalization on the weights (Liu et al., 2020a), such that Equation 13 is satisﬁed. Using
features Z, the calculation of covariance (Z − µc)(Z − µc)(cid:62) and the subsequent mean and variance
of 10 both implicitly utilize distance, therefore we require bi-Lipschitz regularization of fθ. We
choose spectral normalization via the power iteration method, also known as the Von Mises Iteration
(Mises & Pollaczek-Geiringer, 1929), due to its low memory and computation overhead as compared
to second order methods such as gradient penalties (Arjovsky et al., 2017). Speciﬁcally, for features
at hidden layer h(·), at depth l, and for some constants α1, α2, for all zi and zj, we enforce:

α1||z(l)

i − z(l)

j || ≤ ||h(z(l−1)

i

) − h(z(l−1)

j

)|| ≤ α2||z(l)

i − z(l)

j ||.

(13)

4 EXPERIMENTS

The goal of our experimental evaluation is to answer the following questions. 1) What is the beneﬁt
of each component of our proposed model? 2) Does gφ produce more expressive covariances than

6

Published as a conference paper at ICLR 2022

Figure 4: ECE results for all models on different variants of the Omniglot dataset. ProtoMahalanobis models
show comparable in distribution ECE while signiﬁcantly improving ECE over the baselines on corrupted
instances from the dataset.

empirical features? 3) How does the ID/OOD calibration and accuracy compare with other popular
baseline models?

Datasets. For few shot learning, we evaluate our model on both the Omniglot (Lake et al., 2015)
and MiniImageNet (Vinyals et al., 2017) datasets. We utilize corrupted versions (Omniglot-C and
MiniImageNet-C) which consists of 17 corruptions at 5 different intensities (Hendrycks & Dietterich,
2019). We follow the precedent set by Snell et al. (2017) and test Omniglot for 1000 random episodes
and MiniImageNet for 600 episodes. For corruption experiments, the support set is uncorrupted, and
corruption levels 0-5 are used as the query set (0 being the uncorrupted query set). We also experiment
with multiple toy datasets which include half-moons, and concentric circles for binary classiﬁcation
and random 2D multivariate Gaussian distributions for multiclass classiﬁcation (Figure 1). On the toy
datasets, we create task distributions by sampling random tasks with biased support sets, applying
random class shufﬂing and varying levels of noise added to each task. Randomly biasing each task
ensures that no single task contains information from the whole distribution and therefore, the true
distribution must be meta-learned through the episodic training over many such tasks. For a detailed
explanation of the exact toy dataset task creation procedure, see the appendix section A.2.

Baselines. We compare our model against Protonets (Snell et al., 2017), A spectral normalized
version of Protonets (Protonet-SN), MAML (Finn et al., 2017), Reptile (Nichol et al., 2018), and

7

Test12345Shift Intensity: Omniglot-C (5-way/5-shot)0.00.10.20.30.40.5ECEMethodProtoDDUProtonetProtonetSNProtoSNGPMAMLReptileOurs (Diag)Ours (Rank-1)Test12345Shift Intensity: Omniglot-C (5-way/1-shot)0.00.10.20.30.40.5ECETest12345Shift Intensity: Omniglot-C (20-way/5-shot)0.00.20.40.6ECETest12345Shift Intensity: Omniglot-C (20-way/1-shot)0.00.10.20.30.40.50.6ECEPublished as a conference paper at ICLR 2022

Figure 5: ECE for different variants of the MiniImageNet dataset. ProtoMahalanobis models show improved
ECE on corrupted data instances while maintaining comparable performance on in-distribution data.

straightforward few-shot/protonet adaptations of Spectral Normalized Neural Gaussian Processes
(ProtoSNGP) (Liu et al., 2020a) and Deep Deterministic Uncertainty (ProtoDDU) (Mukhoti et al.,
2021). These models represent a range of both metric based, gradient based, and covariance based
meta learning algorithms. All baseline models are temperature scaled after training, with the
temperature parameter optimized via LBFGS for 50 iterations with a learning rate of 0.001. This
follows the temperature scaling implementation from Guo et al. (2017).

Calibration Error. We provide results for Expected Calibration Error (ECE) (Guo et al., 2017) on
various types of OOD data in Figures 4 and 5 as well as Table 1. Accuracy and NLL are reported
in Appendix A.8. Meta learning generally presents a high correlation between tasks, but random
classes from different tasks which are not in the current support set S should still be treated as
OOD. In Table 1 we provide results where the query set Q consists of random classes not in S.
ProtoMahalanobis models perform the best in every case except for Omniglot 20-way/1-shot, where
Reptile showed the lowest ECE. The reason for this can be seen in Figure 4, where Reptile shows
poor ID performance relative to all other models. Under-conﬁdence on ID data can lead to better
conﬁdence scores on OOD data, even though the model is poorly calibrated. Likewise we also
evaluate our models on Omniglot-C and MiniImageNet-C in Figures 4 and 5. As the corruption
intensity increases, ProtoMahalanobis models exhibit lower ECE in relation to baseline models while
maintaining competitive ID performance. Overall, Reptile shows the strongest calibration of baseline
models although it can be underconﬁdent on ID data as can be seen in Figure 4.

In our experiments, transductive batch normalization used in MAML/Reptile led to suboptimal results,
as the normalization statistics depend on the query set which is simultaneously passed through the
network. Passing a large batch of corrupted/uncorrupted samples caused performance degradation on
ID data and presented an unrealistic setting. We therefore utilized the normalization scheme proposed
by Nichol et al. (2018) which creates batch normalization statistics based on the whole support set
plus a single query instance.

Eigenvalue Distribution. In Figure 6, we evaluate the effectiveness of meta learning the low rank
covariance factors with gφ by analyzing the eigenvalue distribution of both empirical covariance
from DDU/SNGP and the encoded covariance from gφ (Equation 5). The empirically calculated
covariances exhibit lower diversity in eigenvalues, which implies that the learned Gaussian distribution
is more spherical and uniform for every class. ProtoMahalanobis models, on the other hand, exhibit a
more diverse range of eigenvalues, leading to non-trivial ellipsoid distributions. We also note that in
addition to more diverse range of eigenvalues, the differences between the distributions of each class
in S are also ampliﬁed in ProtoMahalanobis models, indicating a class speciﬁc variation between

8

Test12345Shift Intensity: MiniImageNet-C (5-way/5-shot)0.00.10.20.30.4ECEMethodProtoDDUProtonetProtonetSNProtoSNGPMAMLReptileOurs (Diag)Ours (Rank-1)Test12345Shift Intensity: MiniImageNet-C (5-way/1-shot)0.00.10.20.30.4ECEPublished as a conference paper at ICLR 2022

(a) Ours Σ−1 Eigenvals

(b) ProtoDDU Σ−1 Eigenvals

(c) ProtoSNGP Σ−1 Eigenvals

Figure 6: Precision matrix eigenvalue distribution for various meta learning model variants. A diverse
distribution of eigenvalues which varies by class, indicates a class speciﬁc, non-spherical Gaussian distribution
is learned. Data comes from Omniglot 5-way/5-shot experiments.

learned covariance factors. Extra ﬁgures are reported in the Appendix A.7, where it can be seen that
the eigenvalue distribution becomes less diverse for ProtoMahalanobis models in the one-shot setting.

Architectures. For both Omniglot and MiniImageNet experiments, we utilize a 4 layer convolutional
neural network with 64 ﬁlters, followed by BatchNorm and ReLU nonlinearities. Each of the four
layers is followed by a max-pooling layer which results in a vector embedding of size 64 for Omniglot
and 1600 for MiniImageNet. Exact architectures can be found in Appendix A.9. Protonet-like models
use BatchNorm with statistics tracked over the training set, and MAML-like baselines use Reptile
Norm (Nichol et al., 2018). As spectral normalized models require residual connections to maintain
the lower Lipschitz bound in equation 13, we add residual connections to the CNN architecture in all
Protonet based models.

4.1

IMPLEMENTATION DETAILS

ProtoSNGP & ProtoDDU Both ProtoSNGP and ProtoDDU baselines are adapted to meta learning
by using the original backbone implementation plus the addition of a positive constrained meta
parameter for the ﬁrst diagonal term in Equation 8 which is shared among all classes. This provides
meta knowledge and a necessary ﬁrst step in applying the recursive formula for inversion to make
predictions on each query set seen during during training.

Covariance Encoder gφ. We utilize the Set Transformer (Lee et al., 2019), as the self-attention
performed by the transformer is an expressive means to encode pairwise information between inputs.
We initialize the seeds in the pooling layers (PMA), with samples from N (0, 1). We do not use
any spectral normalization in gφ, as it should be sufﬁcient to only require that the input to the
encoder is composed of geometry preserving features. Crucially, we remove the residual connection
Q + σ(QK (cid:62))V as we found that this led to the pooling layer ignoring the inputs and outputting an
identical covariance for each class in each task. In the one-shot case, we skip the centering about the
centroid in Equation 4 because it would place all class centroids at the origin.

5 CONCLUSION

It is widely known that DNNs can be miscalibrated for OOD data. We have shown that existing
covariance based uncertainty quantiﬁcation methods fail to calibrate well when given a limited
amounts of data for class-speciﬁc covariance construction for meta learning. In this work, we have
proposed a novel method which meta-learns a diagonal or diagonal plus low rank covariance matrix
which can be used for downstream tasks such as uncertainty calibration. Additionally, we have
proposed an inference procedure and energy tuning scheme which can overcome miscalibration
due to the shift invariance property of softmax. We further enforce bi-Lipschitz regularization of
neural network layers to preserve relative distances between data instances in the feature spaces. We
validated our methods on both synthetic data and two benchmark few-shot learning datasets, showing
that the ﬁnal predictive distribution of our method is well calibrated under a distributional dataset
shift when compared with relevant baselines.

9

01230246810Countclass012340.000.250.500.751.000246810Countclass0123405010015020002468Countclass01234Published as a conference paper at ICLR 2022

6 ACKNOWLEDGEMENTS

This work was supported by the Institute of Information & communications Technology Planning
& Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2019-0-00075, Artiﬁcial
Intelligence Graduate School Program(KAIST)), the Engineering Research Center Program through
the National Research Foundation of Korea (NRF) funded by the Korean Government MSIT (NRF-
2018R1A5A1059921), the Institute of Information & communications Technology Planning &
Evaluation (IITP) grant funded by the Korea government (MSIT) No. 2021-0-02068 (Artiﬁcial
Intelligence Innovation Hub), and the National Research Foundation of Korea (NRF) funded by the
Ministry of Education (NRF2021R1F1A1061655).

REFERENCES

Bruno Andreis, Jeffrey Willette, Juho Lee, and Sung Ju Hwang. Mini-batch consistent slot set encoder

for scalable set encoding. arXiv preprint arXiv:2103.01615, 2021.

Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein gan, 2017.

Jiu Ding and Aihui Zhou. Eigenvalues of rank-one updated matrices with some applications. Applied

Mathematics Letters, 20(12):1223–1226, 2007.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of
deep networks. In International Conference on Machine Learning, pp. 1126–1135. PMLR, 2017.

Will Grathwohl, Kuan-Chieh Wang, Jörn-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi,
and Kevin Swersky. Your classiﬁer is secretly an energy based model and you should treat it like
one. arXiv preprint arXiv:1912.03263, 2019.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural

networks, 2017.

Matthias Hein, Maksym Andriushchenko, and Julian Bitterwolf. Why relu networks yield high-
conﬁdence predictions far away from the training data and how to mitigate the problem, 2019.

Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common

corruptions and perturbations. arXiv preprint arXiv:1903.12261, 2019.

Haiwen Huang, Zhihan Li, Lulu Wang, Sishuo Chen, Bin Dong, and Xinyu Zhou. Feature space

singularity for out-of-distribution detection. arXiv preprint arXiv:2011.14654, 2020.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint

arXiv:1412.6980, 2014.

Brenden M Lake, Ruslan Salakhutdinov, and Joshua B Tenenbaum. Human-level concept learning

through probabilistic program induction. Science, 350(6266):1332–1338, 2015.

Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. Set trans-
former: A framework for attention-based permutation-invariant neural networks. In International
Conference on Machine Learning, pp. 3744–3753. PMLR, 2019.

Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple uniﬁed framework for detecting
out-of-distribution samples and adversarial attacks. arXiv preprint arXiv:1807.03888, 2018.

Jeremiah Zhe Liu, Zi Lin, Shreyas Padhy, Dustin Tran, Tania Bedrax-Weiss, and Balaji Lakshmi-
narayanan. Simple and principled uncertainty estimation with deterministic deep learning via
distance awareness. arXiv preprint arXiv:2006.10108, 2020a.

Weitang Liu, Xiaoyun Wang, John D Owens, and Yixuan Li. Energy-based out-of-distribution

detection. arXiv preprint arXiv:2010.03759, 2020b.

RV Mises and Hilda Pollaczek-Geiringer. Praktische verfahren der gleichungsauﬂösung. ZAMM-
Journal of Applied Mathematics and Mechanics/Zeitschrift für Angewandte Mathematik und
Mechanik, 9(1):58–77, 1929.

10

Published as a conference paper at ICLR 2022

Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for

generative adversarial networks. arXiv preprint arXiv:1802.05957, 2018.

Jishnu Mukhoti, Andreas Kirsch, Joost van Amersfoort, Philip H. S. Torr, and Yarin Gal. Deterministic
neural networks with appropriate inductive biases capture epistemic and aleatoric uncertainty,
2021.

Alex Nichol, Joshua Achiam, and John Schulman. On ﬁrst-order meta-learning algorithms, 2018.

Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin, Joshua V
Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model’s uncertainty?
evaluating predictive uncertainty under dataset shift. arXiv preprint arXiv:1906.02530, 2019.

Massimiliano Patacchiola, Jack Turner, Elliot J. Crowley, and Amos Storkey. Bayesian meta-learning
for the few-shot setting via deep kernels. In Advances in Neural Information Processing Systems,
2020.

Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In ICLR, 2017.

Jürgen Schmidhuber. Evolutionary principles in self-referential learning, or on learning how to learn:

the meta-meta-... hook. PhD thesis, Technische Universität München, 1987.

Jake Snell, Kevin Swersky, and Richard S Zemel. Prototypical networks for few-shot learning. arXiv

preprint arXiv:1703.05175, 2017.

Sebastian Thrun and Lorien Pratt (eds.). Learning to Learn. Kluwer Academic Publishers, Norwell,

MA, USA, 1998. ISBN 0-7923-8047-9.

Joost Van Amersfoort, Lewis Smith, Yee Whye Teh, and Yarin Gal. Uncertainty estimation using a
single deep deterministic neural network. In International Conference on Machine Learning, pp.
9690–9700. PMLR, 2020.

Joost van Amersfoort, Lewis Smith, Andrew Jesson, Oscar Key, and Yarin Gal. Improving determin-

istic uncertainty estimation in deep learning for classiﬁcation and regression, 2021.

Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching Networks for One

Shot Learning. In NIPS, 2016.

Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, and Daan Wierstra. Match-

ing networks for one shot learning, 2017.

Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov, and

Alexander Smola. Deep sets. arXiv preprint arXiv:1703.06114, 2017.

11

Published as a conference paper at ICLR 2022

A APPENDIX

A.1 LOSS DERIVATION (EQUATION 7)

The full derivation of Equation 7 can be achieved by ﬁrst applying Bayes’ Rule, assuming a simple
uniform prior over the class labels, p(yi|xi) can be proportionately expressed as,

p(yi|xi) =

p(xi|yk)p(yk)
p(xi)

∝ p(xi|yk)p(yk)

(14)

In which case the objective of the model becomes raising the class conditional p(xi|yi), while
simultaneously lowering p(xi|yj) ∀ j (cid:54)= i. This is in fact equivalent to a softmax + cross entropy
loss over the class conditional densities which are output from our model. In the softmax case,
maximizing p(yi|xi) for a given class can be done by,

p(yi|xi) =

ezi
z(cid:48) ez(cid:48)

(cid:80)

(15)

Which them implies that the loss to be minimized is the following, commonly known as the negative
log likelihood of the data, or the empirical cross entropy between the true data distribution and the
predictive distribution of the model.

LN LL = ED[− log p(y|x)]
N
(cid:88)

− log p(yi|xi)

=

1
N

i=0

N
(cid:88)

(cid:16)

i=0
(cid:90)

=

1
N

LCE = −

≈

=

1
N

1
N

− zj + log

(cid:17)

exp(z(cid:48)
j)

i

(cid:88)

z(cid:48)
j

px(x) log pθ(y|x)dx

x
N
(cid:88)

− log pθ(yi|xi)

i=0

N
(cid:88)

(cid:16)

i=0

− zj + log

(cid:17)

exp(z(cid:48)
j)

i

(cid:88)

z(cid:48)
j

(16)

In our case, assuming a uniform prior over the classes, we can analogously formulate the loss as,

L = ED[− log p(x|y)p(y)]

=

=

=

1
N

1
N

1
N

N
(cid:88)

(cid:16)

i=0

N
(cid:88)

(cid:16)

i=0

N
(cid:88)

(cid:16)

i=0

− log p(xi|yk)p(yk) + log

(cid:17)

i|yk)p(yk)

(cid:88)

p(x(cid:48)

x(cid:48)

− log p(xi|yk) − log p(yk) + log p(yk) + log

(cid:17)

p(x(cid:48)

i|yk)

(cid:88)

x(cid:48)

(17)

− log p(xi|yk) + log

(cid:88)

p(x(cid:48)

(cid:17)

i|yk)

x(cid:48)

A.2 TOY DATASETS

To add bias to each samples task from our 2D toy datasets, we ﬁrst randomly choose an axis (X or Y)
for each class and then slice the datapoints in half randomly. We then sample the support set from the

12

Published as a conference paper at ICLR 2022

chosen biased subset and leave the rest of the remaining points for the query set. Each sampled task
calculates the mean and variance from the support set, which are then used to normalize all instances
in S and Q.

Dataset

N-Way K-Shot

Circles
Moons
Gaussians

2
2
10

5
5
10

A.2.1 META MOONS

For the Meta Moons dataset, we randomly invert the classes to make sure that the class indices appear
in a random order for each task. We add a random amount of Gaussian noise to each moon with a
uniform standard deviation in the range of (0, 0.25].

Figure 7: Random task samples from the Meta Moons dataset.

A.2.2 META CIRCLES

For the Meta Circles dataset, we randomly invert the order of the classes so that the inner circle and
the outer circle are not guaranteed to appear in the same order on every task. We inject a random
amount of Gaussian noise into the data, with a uniformly random standard deviation in the range of
(0, 0.25]. We also randomly choose the scale factor between the size of the inner circle and the outer
circle, which is uniformly random in the range of (0, 0.8]

Figure 8: Random task samples from the Meta Circles dataset.

A.2.3 META GAUSSIANS

The task construction of the Meta Gaussians dataset requires that we construct random positive
semideﬁnite covariance matrices for each class. We ﬁrst uniformly sample N 2 × 2 matrices in
the range U (−1, 1) and perform a QR decomposition to extract orthonormal matrices Q. We then
sample a random diagonal D ∼ U (0, 1), and construct the ﬁnal matrix as QDQ(cid:62) which is positive

13

Published as a conference paper at ICLR 2022

Table 2: Accuracy, ECE, NLL, and OOD AUPR for different n-way k-shot classiﬁcation problems on the
Omniglot-C dataset which contains 17 different corruptions at 5 different intensity levels. MAML/Reptile both
utilize ‘Reptile Norm’ instead of transductive BatchNorm

Accuracy ↑

NLL ↓

Model

5-way 5-shot

5-way 1-shot

20-way 5-shot

20-way 1-shot

5-way 5-shot

5-way 1-shot

20-way 5-shot

20-way 1-shot

MAML
Reptile
Protonet
Protonet-SN
ProtoDDU
ProtoSNGP

Ours (Diag)
Ours (Rank 1)
Ours (Rank 2)
Ours (Rank 4)
Ours (Rank 8)

MAML
Reptile
Protonet
Protonet-SN
ProtoDDU
ProtoSNGP

Ours (Diag)
Ours (Rank 1)
Ours (Rank 2)
Ours (Rank 4)
Ours (Rank 8)

65.02±17.94
61.29±18.61
60.91±19.13
60.08±19.47
60.31±19.19
59.18±19.66

60.77±19.12
59.88±19.55
59.68±19.61
60.28±19.51
59.69±19.69

28.10±15.25
24.85±13.12
34.08±17.02
35.12±17.46
27.84±14.19
29.26±14.53

5.87±2.58
6.10±2.87
6.62±3.09
7.10±3.40
7.05±3.42

64.72±16.96
60.01±17.63
58.15±19.68
57.64±19.89
58.03±19.57
57.03±19.91

57.71±19.88
58.15±19.59
56.61±20.21
59.14±19.32
58.21±19.63

48.17±23.93
46.55±23.75
46.29±25.21
46.19±25.22
45.75±25.33
46.49±25.12

45.98±25.31
45.47±25.57
45.53±25.52
45.87±25.44
45.93±25.39

ECE ↓

19.96±11.74
19.69±10.29
35.69±17.62
36.27±17.85
33.79±16.61
28.70±14.07

11.15±4.93
11.81±4.96
12.38±5.11
11.17±4.95
12.89±5.60

33.64±17.35
26.33±13.33
43.57±20.89
43.75±20.89
37.62±18.16
40.42±19.46

9.81±4.36
11.30±5.48
11.49±5.12
10.87±4.75
10.61±4.84

44.35±22.27
43.42±22.13
43.11±25.55
43.47±25.44
43.90±25.22
44.37±24.85

43.05±25.59
43.12±25.57
43.28±25.45
42.38±25.90
43.53±25.42

27.95±13.12
19.98±9.77
43.47±20.69
42.06±20.10
40.83±19.55
36.43±17.36

21.99±9.65
20.26±8.68
21.42±9.45
22.90±10.14
20.56±9.22

3.658±2.421
2.300±1.345
6.526±3.800
7.189±4.261
10.945±7.143
2.534±1.302

1.010±0.476
1.020±0.481
1.045±0.495
1.068±0.510
1.055±0.501

0.602±0.060
0.617±0.081
0.867±0.169
0.869±0.169
0.675±0.083
0.875±0.172

0.869±0.169
0.869±0.169
0.867±0.169
0.868±0.169
0.868±0.169

1.528±0.869
1.571±0.796
6.539±4.029
6.541±3.960
10.428±7.186
2.015±0.967

1.205±0.547
1.312±0.595
1.314±0.594
1.263±0.580
1.329±0.613

5.962±3.822
3.937±2.357
10.706±5.689
11.200±6.168
18.014±9.740
6.409±3.196

2.466±1.141
2.541±1.212
2.561±1.194
2.528±1.168
2.486±1.156

OOD AUPR

0.653±0.075
0.672±0.098
0.853±0.163
0.857±0.164
0.552±0.044
0.855±0.163

0.855±0.164
0.850±0.162
0.853±0.163
0.851±0.162
0.851±0.162

0.440±0.032
0.646±0.087
0.876±0.172
0.875±0.171
0.647±0.074
0.878±0.173

0.876±0.172
0.874±0.171
0.874±0.171
0.874±0.171
0.876±0.172

3.668±1.873
2.864±1.380
9.119±4.890
8.446±4.590
17.039±9.935
4.151±1.982

3.854±1.775
3.564±1.624
3.787±1.760
4.019±1.877
3.474±1.607

0.484±0.098
0.720±0.113
0.863±0.166
0.862±0.166
0.579±0.062
0.858±0.164

0.863±0.166
0.863±0.166
0.862±0.166
0.864±0.167
0.863±0.166

semi-deﬁnite. This leads to the distribution of each class being an elliptical multivariate Gaussian
distribution.

Figure 9: Random task samples from the Meta Gaussians dataset.

A.3 EXTRA RESULTS

We provide extra results on the MiniImageNet-C and Omniglot-C dataset here. Tables 2 and 3 contain
results averaged over the whole corrupted dataset, including the natural test set and all 5 levels of
corruption

A.4 SET ENCODING RELATED WORKS

Set encoding functions require special end-to-end design considerations such as obeying permutation
invariance w.r.t. the input set f ({X1, X2, ..., Xn}) = f ({Xπ(1), Xπ(2), ..., Xπ(n)}) for any random
permutation of indices π(.). Likewise, the intermediate latent representations must satisfy permutation
equivariance such that f ({Xπ(1), Xπ(2), ..., Xπ(n)}) = {fπ(1)(X), fπ(2)(X), ..., fπ(1)(X)}.

Deepsets (Zaheer et al., 2017) ﬁrst proposed basic adaptations of linear and convolutional neural
networks which obey the above required properties and have the addition of a sum decomposable
(permutation invariant) pooling function and decoder to match the requirements of the given task.
As sets can have complex interactions between elements, it may be beneﬁcial to model pairwise
interactions between set elements. The Set Transformer (Lee et al., 2019) uses a transformer
architecture with self attention to model such pairwise interactions between set elements. As

14

Published as a conference paper at ICLR 2022

Table 3: Accuracy, ECE, NLL, and AUPR for different n-way k-shot classiﬁcation problems on the
MiniImageNet-C dataset which contains 17 different corruptions at 5 different intensity levels. MAML/Reptile
both utilize ‘Reptile Norm’ instead of transductive BatchNorm

Model

5-way 1-shot

5-way 5-shot

5-way 1-shot

5-way 5-shot

Accuracy ↑

NLL ↓

MAML
Reptile
Protonet
Protonet-SN
ProtoDDU
ProtoSNGP

Ours (Diag)
Ours (Rank 1)
Ours (Rank 2)
Ours (Rank 4)
Ours (Rank 8)

MAML
Reptile
Protonet
Protonet-SN
ProtoDDU
ProtoSNGP

Ours (Diag)
Ours (Rank 1)
Ours (Rank 2)
Ours (Rank 4)
Ours (Rank 8)

31.94±7.85
33.07±7.85
33.43±8.49
32.79±8.56
33.62±8.92
33.56±8.65

33.21±8.68
33.19±8.45
33.03±8.54
32.52±8.45
32.41±8.55

39.30±12.92
39.18±12.84
41.35±14.03
40.95±14.21
41.46±14.35
41.20±13.66

40.69±13.66
40.89±13.90
40.90±13.87
41.24±13.84
40.33±13.77

1.720±0.242
1.580±0.150
1.801±0.323
1.836±0.341
1.906±0.465
1.699±0.267

1.556±0.158
1.575±0.171
1.571±0.174
1.591±0.191
1.581±0.175

1.748±0.462
1.581±0.339
2.123±0.798
2.112±0.802
2.180±0.932
1.889±0.614

1.630±0.423
1.699±0.484
1.659±0.456
1.696±0.486
1.644±0.442

ECE ↓

AUPR ↑

19.62±9.02
12.32±3.84
20.78±8.98
21.66±9.28
22.70±9.69
20.30±7.28

8.13±3.72
9.02±3.83
9.27±4.02
9.87±5.29
9.27±4.38

24.03±11.29
17.60±8.98
27.75±13.62
27.99±13.90
27.13±13.71
25.96±12.19

15.57±7.32
17.27±8.42
16.27±7.33
16.71±7.85
15.93±7.50

0.536±0.083
0.756±0.131
0.637±0.094
0.629±0.085
0.530±0.041
0.636±0.080

0.636±0.092
0.625±0.087
0.632±0.087
0.629±0.085
0.637±0.092

0.628±0.106
0.749±0.124
0.579±0.071
0.572±0.068
0.620±0.061
0.653±0.089

0.578±0.065
0.574±0.067
0.581±0.064
0.569±0.058
0.574±0.069

Table 4: Accuracy, ECE, NLL, and AUPR for different n-way k-shot classiﬁcation problems on the Omniglot
dataset. All metrics are measured on the natural test set except AUPR/AUROC which is measured using random
classes which are different from the classes in the support set. Our model maintains competitive performance for
ID data on all metrics. MAML/Reptile both utilize ‘Reptile Norm’ instead of transductive BatchNorm

Accuracy ↑

NLL ↓

Model

5-way 5-shot

5-way 1-shot

20-way 5-shot

20-way 1-shot

5-way 5-shot

5-way 1-shot

20-way 5-shot

20-way 1-shot

MAML
Reptile
Protonet
Protonet-SN
ProtoDDU
ProtoSNGP

Ours (Diag)
Ours (Rank 1)
Ours (Rank 2)
Ours (Rank 4)
Ours (Rank 8)

MAML
Reptile
Protonet
Protonet-SN
ProtoDDU
ProtoSNGP

Ours (Diag)
Ours (Rank 1)
Ours (Rank 2)
Ours (Rank 4)
Ours (Rank 8)

99.51±0.06
98.55±0.07
99.65±0.02
99.67±0.04
99.70±0.05
99.65±0.07

99.64±0.06
99.63±0.06
99.62±0.06
99.66±0.04
99.64±0.02

0.05±0.03
1.64±0.08
0.09±0.02
0.09±0.04
0.07±0.02
0.09±0.04

1.07±0.12
1.06±0.09
1.07±0.10
1.00±0.11
1.03±0.08

96.55±0.23
95.72±0.38
98.24±0.16
98.26±0.12
98.37±0.11
98.23±0.08

98.21±0.23
98.21±0.12
98.30±0.23
98.42±0.17
98.35±0.16

1.06±0.12
3.95±0.41
0.54±0.09
0.51±0.17
0.42±0.15
0.15±0.04

2.02±0.12
2.14±0.30
2.11±0.38
2.00±0.45
1.88±0.17

97.96±0.28
96.50±0.07
99.29±0.05
99.26±0.06
99.28±0.05
99.23±0.06

99.26±0.01
99.29±0.03
99.30±0.06
99.28±0.07
99.32±0.04

ECE ↓

1.39±0.87
3.21±0.11
0.19±0.04
0.21±0.03
0.14±0.03
0.14±0.05

1.13±0.05
1.14±0.05
1.13±0.08
1.16±0.06
1.10±0.10

91.97±0.27
90.95±0.47
97.47±0.09
97.51±0.16
97.54±0.16
97.41±0.13

97.49±0.09
97.61±0.14
97.56±0.10
97.56±0.17
97.63±0.16

4.95±0.68
8.73±0.12
0.35±0.09
0.39±0.11
0.33±0.13
0.20±0.04

1.58±0.26
1.71±0.27
1.47±0.28
1.52±0.12
1.47±0.11

0.015±0.002
0.054±0.002
0.013±0.002
0.013±0.003
0.010±0.002
0.012±0.003

0.020±0.002
0.020±0.002
0.020±0.002
0.019±0.002
0.019±0.001

0.856±0.006
0.831±0.019
0.994±0.001
0.994±0.000
0.482±0.003
0.994±0.001

0.994±0.001
0.994±0.001
0.994±0.001
0.994±0.000
0.994±0.000

0.104±0.006
0.150±0.010
0.059±0.007
0.061±0.007
0.058±0.010
0.054±0.004

0.064±0.005
0.067±0.005
0.064±0.010
0.060±0.005
0.059±0.004

0.078±0.015
0.142±0.002
0.027±0.006
0.029±0.006
0.027±0.005
0.029±0.006

0.032±0.002
0.031±0.002
0.031±0.003
0.032±0.003
0.030±0.003

OOD AUPR ↑

0.799±0.010
0.813±0.015
0.977±0.001
0.977±0.002
0.475±0.004
0.977±0.003

0.976±0.003
0.976±0.002
0.977±0.004
0.977±0.002
0.978±0.002

0.622±0.023
0.591±0.002
0.990±0.001
0.990±0.000
0.496±0.002
0.989±0.001

0.990±0.000
0.990±0.001
0.990±0.001
0.990±0.001
0.990±0.001

0.289±0.013
0.365±0.015
0.087±0.008
0.086±0.011
0.085±0.010
0.085±0.006

0.089±0.006
0.086±0.007
0.087±0.007
0.088±0.007
0.084±0.008

0.578±0.007
0.579±0.003
0.974±0.001
0.975±0.002
0.496±0.002
0.972±0.002

0.974±0.001
0.974±0.001
0.975±0.002
0.975±0.001
0.974±0.001

15

Published as a conference paper at ICLR 2022

Table 5: Accuracy, NLL, ECE, and AUPR for different n-way k-shot classiﬁcation problems on the Mini-
ImageNet dataset. All metrics are measured on the natural test set except AUPR/AUROC which is measured
using random classes which are different from the classes in the support set. Our model maintains competitive
performance for ID data on all metrics. MAML/Reptile both utilize ‘Reptile Norm’ instead of transductive
BatchNorm

Model

5-way 1-shot

5-way 5-shot

5-way 1-shot

5-way 5-shot

Accuracy ↑

NLL ↓

MAML
Reptile
Protonet
Protonet-SN
ProtoDDU
ProtoSNGP

Ours (Diag)
Ours (Rank 1)
Ours (Rank 2)
Ours (Rank 4)
Ours (Rank 8)

MAML
Reptile
Protonet
Protonet-SN
ProtoDDU
ProtoSNGP

Ours (Diag)
Ours (Rank 1)
Ours (Rank 2)
Ours (Rank 4)
Ours (Rank 8)

46.13±1.19
47.79±1.21
48.61±0.91
47.47±0.90
49.57±0.53
49.55±0.90

48.31±0.39
48.57±0.96
48.08±0.99
47.76±0.62
48.91±0.87

64.71±0.50
62.89±0.88
67.57±0.55
68.03±0.79
68.31±0.59
66.89±0.88

66.12±1.76
66.54±0.66
67.17±0.56
66.73±0.37
66.58±1.67

1.297±0.015
1.297±0.023
1.245±0.019
1.279±0.017
1.246±0.006
1.232±0.012

1.277±0.023
1.267±0.013
1.271±0.021
1.274±0.020
1.272±0.039

0.921±0.017
0.967±0.017
0.832±0.008
0.820±0.016
0.816±0.016
0.841±0.020

0.887±0.044
0.859±0.016
0.853±0.017
0.868±0.010
0.873±0.038

ECE ↓

AUPR ↑

3.39±1.09
4.65±0.76
5.62±1.08
6.77±1.81
7.33±1.44
7.21±1.26

9.40±1.82
8.09±1.60
7.50±2.30
7.57±2.45
10.03±3.64

2.79±0.45
1.64±0.23
4.09±0.86
3.22±0.96
3.46±0.55
4.06±0.84

8.32±2.46
6.77±0.53
7.92±1.34
8.02±2.19
8.80±1.53

0.508±0.009
0.517±0.006
0.609±0.010
0.608±0.011
0.473±0.006
0.621±0.007

0.607±0.011
0.605±0.007
0.610±0.008
0.603±0.005
0.609±0.010

0.544±0.005
0.542±0.003
0.596±0.006
0.602±0.002
0.479±0.003
0.687±0.004

0.602±0.013
0.611±0.008
0.611±0.003
0.609±0.005
0.606±0.005

transformers have a quadratic complexity w.r.t. input set length, it may not be possible to process a
large set with a transformer and maintain permutation invariance, if the set will not ﬁt into memory.
Therefore, recent works have also further explored how to make an attentive set encoder which
can process sets in batches (Andreis et al., 2021) while maintaining the above requirements of set
functions.

For our model, we chose to use the set transformer architecture, as it models pairwise interactions
between elements which is an implicit requirement of construction a Gaussian covariance matrix.
Therefore, it has the proper inductive biases needed to satisfy our requirement of predicting low rank
covariance factors given an input set of features.

A.5 EXTRA TOY RESULTS

In Figures 10, 11, 12, 13, 14, 15, and 16 we provide extra qualitative results on toy dataset covariances
and entropy surfaces.
In Tables 6, 7, and 8 we provide tabular results of all toy experiments,
showcasing the differences between in-distribution data and random uniform OOD noise.

Model

Protonet
ProtonetSN
Proto DDU
Proto SNGP

Ours (Diag)
Ours (Rank-1)
Ours (Rank-2)
Ours (Rank-4)
Ours (Rank-8)
Ours (Rank-16)
Ours (Rank-32)
Ours (Rank-64)

Accuracy ↑

97.02±1.60
97.31±1.54
96.04±2.74
97.22±1.19

96.82±1.09
96.86±1.42
96.90±1.55
96.90±1.15
96.69±1.36
96.73±1.28
96.73±1.28
96.73±1.55

In Distribiution
NLL ↓

ECE ↓

ECE ↓

Out of Distribiution
AUPR ↑

AUROC ↑

0.171±0.110
0.140±0.088
0.158±0.077
0.138±0.046

0.167±0.056
0.157±0.049
0.162±0.042
0.157±0.033
0.171±0.047
0.161±0.057
0.170±0.042
0.159±0.045

2.21±1.40
2.38±1.38
2.43±1.04
4.81±2.63

5.09±1.18
4.21±1.77
5.13±1.39
4.69±1.32
4.74±1.48
3.80±1.87
4.49±1.01
4.43±0.99

48.16±0.91
48.40±0.64
49.43±0.37
45.74±4.19

15.66±2.65
20.60±2.24
17.74±2.76
18.49±2.62
19.75±3.53
18.47±2.30
18.22±3.26
17.90±2.49

0.999±0.000
0.999±0.000
0.976±0.001
0.995±0.001

0.999±0.000
0.999±0.000
0.999±0.000
0.999±0.000
0.999±0.000
0.999±0.000
0.999±0.000
0.999±0.000

0.931±0.012
0.932±0.006
0.119±0.013
0.684±0.064

0.937±0.006
0.934±0.008
0.939±0.006
0.937±0.006
0.935±0.007
0.939±0.007
0.939±0.005
0.938±0.005

Table 6: Tabular results from the meta-moons toy experiment

16

Published as a conference paper at ICLR 2022

(a) Meta Circles (ProtoMahalanobisFC). From left to right: entropy surface, covariances for class 1-2

(b) Meta Gaussians (ProtoMahalanobisFC) From left to right: entropy surface, covariances for class 1-2

Figure 10: ProtoMahalanobisFC model performance on the meta-moons and meta-circles toy datasets.

Model

Protonet
ProtonetSN
Proto DDU
Proto SNGP

Ours (Diag)
Ours (Rank-1)
Ours (Rank-2)
Ours (Rank-4)
Ours (Rank-8)
Ours (Rank-16)
Ours (Rank-32)
Ours (Rank-64)

Accuracy ↑

89.12±3.92
88.93±4.12
90.45±2.67
89.63±3.30

90.61±3.54
91.01±2.65
91.31±2.87
90.96±2.76
90.99±2.52
91.01±2.88
90.45±3.22
90.83±2.99

In Distribiution
NLL ↓

ECE ↓

ECE ↓

Out of Distribiution
AUPR ↑

AUROC ↑

0.343±0.091
0.342±0.098
0.267±0.061
0.323±0.063

0.279±0.075
0.271±0.062
0.269±0.064
0.272±0.068
0.268±0.065
0.264±0.065
0.267±0.069
0.264±0.064

4.33±0.83
4.23±0.71
3.52±0.95
7.39±2.71

7.24±1.57
7.12±1.00
7.59±1.34
6.68±0.56
6.74±1.02
6.96±1.23
6.35±0.27
6.79±0.83

81.84±0.78
82.42±0.87
84.48±0.41
62.60±6.57

42.55±0.98
42.77±1.62
43.13±1.90
43.33±1.80
43.14±1.49
42.93±1.94
42.89±1.48
42.67±1.67

0.999±0.000
0.999±0.000
0.969±0.001
0.999±0.000

0.999±0.000
0.999±0.000
0.999±0.000
0.999±0.000
0.999±0.000
0.999±0.000
0.999±0.000
0.999±0.000

0.950±0.009
0.951±0.008
0.175±0.003
0.916±0.014

0.954±0.005
0.954±0.005
0.955±0.004
0.955±0.004
0.955±0.004
0.955±0.005
0.956±0.004
0.956±0.004

Table 7: Tabular results from the meta-Gaussians toy experiment

Model

Protonet
ProtonetSN
Proto DDU
Proto SNGP

Ours (Diag)
Ours (Rank-1)
Ours (Rank-2)
Ours (Rank-4)
Ours (Rank-8)
Ours (Rank-16)
Ours (Rank-32)
Ours (Rank-64)

Accuracy ↑

94.45±3.18
94.53±2.49
95.02±1.79
94.49±2.09

94.24±3.85
94.08±4.62
94.53±4.27
94.12±4.68
94.00±4.69
94.12±4.40
93.84±4.66
94.16±4.61

In Distribiution
NLL ↓

ECE ↓

ECE ↓

Out of Distribiution
AUPR ↑

AUROC ↑

0.195±0.106
0.185±0.098
0.165±0.088
0.192±0.071

0.215±0.139
0.214±0.158
0.192±0.148
0.209±0.158
0.194±0.134
0.205±0.148
0.193±0.141
0.196±0.146

3.49±1.67
3.03±1.58
3.62±2.27
6.05±3.56

4.11±0.62
4.34±1.46
3.54±1.52
4.27±1.83
3.80±1.45
4.12±1.93
3.48±1.54
3.46±1.31

49.19±0.25
49.17±0.21
48.87±0.18
45.14±3.08

14.64±4.32
19.04±8.21
18.22±3.72
19.61±4.52
19.37±4.42
20.59±5.47
20.50±5.56
19.53±6.30

1.000±0.000
1.000±0.000
0.972±0.001
0.992±0.001

1.000±0.000
1.000±0.000
1.000±0.000
1.000±0.000
1.000±0.000
1.000±0.000
1.000±0.000
1.000±0.000

0.952±0.007
0.952±0.007
0.072±0.008
0.683±0.055

0.954±0.011
0.953±0.013
0.954±0.013
0.954±0.013
0.955±0.014
0.955±0.013
0.954±0.014
0.955±0.014

Table 8: Tabular results from the meta-circles toy experiment

17

Accuracy: 1.00NLL 0.04ENT ID/OOD: 0.07 / 0.55ECE ID/OOD 0.03 / 0.20traintestAccuracy: 0.96NLL 0.10ENT ID/OOD: 0.12 / 0.55ECE ID/OOD 0.04 / 0.19traintestPublished as a conference paper at ICLR 2022

Figure 11: ProtoMahalanobisFC model performance on the meta Gaussians toy dataset. From the top
left: Entropy surface, covariances for clases 1-10

(a) Meta Circles (DDU). From left to right: entropy surface (distance), entropy surface (softmax sample),
covariances for class 1-2

(b) Meta Moons (DDU). From left to right: entropy surface (distance), entropy surface (softmax sample),
covariances for class 1-2

Figure 12: Proto DDU model performance on the two toy meta learning datasets.

18

Accuracy: 0.87NLL 0.38ENT ID/OOD: 0.43 / 1.14ECE ID/OOD 0.08 / 0.44traintestAccuracy: 1.00NLL 0.00ENT ID/OOD: 0.00 / 0.01ECE ID/OOD 0.00 / 0.50traintestAccuracy: 1.00NLL 0.19ENT ID/OOD: 0.46 / 0.50ECE ID/OOD 0.18 / 0.30traintestAccuracy: 0.97NLL 0.09ENT ID/OOD: 0.07 / 0.02ECE ID/OOD 0.02 / 0.50traintestAccuracy: 0.97NLL 0.25ENT ID/OOD: 0.50 / 0.53ECE ID/OOD 0.18 / 0.25traintestPublished as a conference paper at ICLR 2022

Figure 13: Proto DDU model performance on the Meta Gaussians dataset. From the top left: Entropy
surface (distance), entropy surface (softmax sample), covariances for clases 1-10

Figure 14: Protonet model performance on the three meta-toy datasets. From left to right: Meta-
Circles, Meta-Moons, Meta-Gaussians.

(a) Meta Circles (SNGPProtoFC). From left to right: entropy surface (softmax sample), entropy surface (distance),
covariances for class 1-2

(b) Meta Moons (SNGPProtoFC). From left to right: entropy surface (softmax sample), entropy surface (distance),
covariances for class 1-2

Figure 15: SNGPProtoFC model performance on the meta-moons and meta-circles toy datasets.

19

Accuracy: 0.84NLL 0.44ENT ID/OOD: 0.36 / 0.16ECE ID/OOD 0.08 / 0.83traintestAccuracy: 0.82NLL 1.84ENT ID/OOD: 2.26 / 2.29ECE ID/OOD 0.64 / 0.04traintestAccuracy: 1.00NLL 0.00ENT ID/OOD: 0.01 / 0.01ECE ID/OOD 0.00 / 0.50traintestAccuracy: 0.96NLL 0.16ENT ID/OOD: 0.03 / 0.05ECE ID/OOD 0.04 / 0.49traintestAccuracy: 0.81NLL 0.54ENT ID/OOD: 0.37 / 0.16ECE ID/OOD 0.08 / 0.83traintestAccuracy: 0.99NLL 0.21ENT ID/OOD: 0.47 / 0.47ECE ID/OOD 0.18 / 0.32traintestAccuracy: 0.99NLL 0.03ENT ID/OOD: 0.04 / 0.04ECE ID/OOD 0.01 / 0.49traintestAccuracy: 0.97NLL 0.25ENT ID/OOD: 0.50 / 0.50ECE ID/OOD 0.19 / 0.28traintestAccuracy: 0.97NLL 0.10ENT ID/OOD: 0.18 / 0.19ECE ID/OOD 0.05 / 0.44traintestPublished as a conference paper at ICLR 2022

Figure 16: SNGPProtoFC model performance on the meta-Gaussians toy dataset. From the top left:
Entropy surface (softmax sample), entropy surface (distance), covariances for clases 1-10

A.6 FURTHER IMPLEMENTATION DETAILS

In order to constrain the diagonal Λ of Proto Mahalanobis models
Positive Diagonal Constraint
(Equation 5) to be positive as mentioned in Section 3.2, we utilize a truncated sigmoid function
Λ = max(0.1, σ(z)). We truncate the values in order to avoid extreme values during the inversion.

SNGP & DDU Both SNGP (Liu et al., 2020a) and DDU (Mukhoti et al., 2021) were originally
designed under the assumption that an entire dataset would be used in the ﬁnal pass to construct a
feature covariance matrix. Given that few-shot-learning contains a limited number of samples for
each task, we compose the feature covariance as a diagonal + low-rank factor Λ + ΦΦ(cid:62), where Λ is
a positive constrained (via softplus) meta learned parameter. Λ can be seen as a shrinkage estimation
(δΛ + (1 − δ)ΦΦ(cid:62)) for low sample size, with a meta learned mixing coefﬁcient δ.

In order to extend SNGP to work in the few shot learning scenario under the prototypical network
Snell et al. (2017) framework, we had to modify the original algorithm by replacing the last linear
layer with the embedding layer and centroids used by prototypical networks. Empirically, we found
that using the SNGP logit-normal inference procedure led to a severe performance decrease, therefore
our results utilized Mahalanobis distance instead.

OOD AUPR/AUROC In order to evaluate the OOD AUPR/AUROC metrics in the supplementary
tables, we utilize the method proposed by Liu et al. (2020b). Speciﬁcally, we use the total energy in
the logits log (cid:80)

i exp(zi) as the score when evaluating AUPR/AUROC.

Optimizers All models are trained with the Adam (Kingma & Ba, 2014) optimizer

A.7 ADDITIONAL EIGENVALUE DISTRIBUTIONS

The eigenvalue distributions highlighted in section 4 exhibit the most diverse case of eigenvalues.
However, the eigenvalues of ProtoMahalanobis precision matrices become less diverse in the one-shot
setting which is also where we are unable to mean center the respective features by class.

20

Accuracy: 0.83NLL 1.84ENT ID/OOD: 2.27 / 2.29ECE ID/OOD 0.65 / 0.03traintestAccuracy: 0.82NLL 0.48ENT ID/OOD: 0.48 / 0.70ECE ID/OOD 0.07 / 0.69traintestPublished as a conference paper at ICLR 2022

Figure 18: Accuracy boxplots for different variations of the Omniglot dataset

Figure 17: From left to right: covariance, precision, and eigenvalue distribution for ProtoMahalanobis
precision matrix on Omniglot 20-way/1-shot (left) and 20-way/5-shot (right) experiments.

21

Test12345Shift Intensity: Omniglot-C (5-way/5-shot)0.40.50.60.70.80.91.0AccuracyMethodProtoDDUProtonetProtonetSNProtoSNGPMAMLReptileOurs (Diag)Ours (Rank-1)Test12345Shift Intensity: Omniglot-C (5-way/1-shot)0.40.50.60.70.80.91.0AccuracyTest12345Shift Intensity: Omniglot-C (20-way/5-shot)0.20.40.60.81.0AccuracyTest12345Shift Intensity: Omniglot-C (20-way/1-shot)0.20.40.60.81.0Accuracy024681001020304050Countclass0123456789101112131415161718190.00.51.01.52.02.53.00246810Countclass012345678910111213141516171819Published as a conference paper at ICLR 2022

Figure 19: NLL boxplots for different variations of the Omniglot dataset

22

Test12345Shift Intensity: Omniglot-C (5-way/5-shot)05101520NLLMethodProtoDDUProtonetProtonetSNProtoSNGPMAMLReptileOurs (Diag)Ours (Rank-1)Test12345Shift Intensity: Omniglot-C (5-way/1-shot)05101520NLLTest12345Shift Intensity: Omniglot-C (20-way/5-shot)0510152025NLLTest12345Shift Intensity: Omniglot-C (20-way/1-shot)051015202530NLLPublished as a conference paper at ICLR 2022

Figure 20: Accuracy boxplots for different variations of the MiniImageNet dataset

Figure 21: NLL boxplots for different variations of the MiniIMageNet dataset

23

Test12345Shift Intensity: MiniImageNet-C (5-way/5-shot)0.30.40.50.60.7AccuracyMethodProtoDDUProtonetProtonetSNProtoSNGPMAMLReptileOurs (Diag)Ours (Rank-1)Test12345Shift Intensity: MiniImageNet-C (5-way/1-shot)0.250.300.350.400.450.50AccuracyTest12345Shift Intensity: MiniImageNet-C (5-way/5-shot)1234NLLMethodProtoDDUProtonetProtonetSNProtoSNGPMAMLReptileOurs (Diag)Ours (Rank-1)Test12345Shift Intensity: MiniImageNet-C (5-way/1-shot)1.52.02.53.0NLLPublished as a conference paper at ICLR 2022

Table 9: Convolutional architecture used for MAML/Reptile Omniglot

Layers

Conv2d(1, 64, pad=1, stride=2) → BatchNorm(reptilenorm=True) → ReLU
Conv2d(64, 64, pad=1, stride=2) → BatchNorm(reptilenorm=True) → ReLU
Conv2d(64, 64, pad=1, stride=2) → BatchNorm(reptilenorm=True) → ReLU
Conv2d(64, 64, pad=1, stride=2) → BatchNorm(reptilenorm=True) → ReLU
AveragePool(2)
FC(64, nway)

Table 10: Convolutional architecture used for MAML/Reptile MiniImageNet. Reptile uses 64 ﬁlters
instead of 32.

Layers

Conv2d(1, 32, pad=1, stride=1) → BatchNorm(reptilenorm=True) → ReLU → MaxPool2d(2)
Conv2d(32, 32, pad=1, stride=2) → BatchNorm(reptilenorm=True) → ReLU → MaxPool2d(2)
Conv2d(32, 32, pad=1, stride=2) → BatchNorm(reptilenorm=True) → ReLU → MaxPool2d(2)
Conv2d(32, 32, pad=1, stride=2) → BatchNorm(reptilenorm=True) → ReLU → MaxPool2d(2)
Flatten
FC(1600, nway)

A.8 ADDITIONAL BOXPLOT RESULTS

Figures 18, 20 show extra boxplot results for accuracy while Figures 19, 21 show negative log
likelihood.

A.9 ARCHITECTURE DETAILS

Tables 9, and 10 show the backbone architectures for MAML/Reptile or Omniglot and MiniImageNet
respectively. Table 11 shows the backbone architecture for all Protonet based models.

A.10 RUNTIME ANALYSIS

In Tables 12, and 13 we provide a runtime analysis of different variants of our models and baselines.
Linear models are evaluated by using the mean and standard deviations from 50 iterations of both
training and inference on the MetaMoons dataset. Convolutional models are likewise evaluated on
50 iterations of the Omniglot dataset. All models were evaluated on a single GeForce GTX 1080 Ti
GPU. SNGP/DDU also utilize the matrix inversion outlined in Equation 8.

Mahalanobis models show slightly better (Linear) or similar (CNN) latency to SNGP/DDU for
diagonal and rank-1 variants. Latency increases as the rank goes higher due to more factors and more
iterations required for inversion and log-determinant calculations. Comparing Protonet, Protonet-SN,
and other variants which need to construct a covariance, we can see that constructing the covariance
matrix adds a cost which is roughly equivalent to spectral normalization.

Table 11: Convolutional architecture used for Protonet Models. Plain Protonets use no spectral
normalization

Layers

SpectralNorm(Conv2d(1, 64, pad=1, stride=1), residual=True, c=3) → BatchNorm() → ReLU → Dropout() → AveragePool2d(2)
SpectralNorm(Conv2d(1, 64, pad=1, stride=1), residual=True, c=3) → BatchNorm() → ReLU → Dropout() → AveragePool2d(2)
SpectralNorm(Conv2d(1, 64, pad=1, stride=1), residual=True, c=3) → BatchNorm() → ReLU → Dropout() → AveragePool2d(2)
SpectralNorm(Conv2d(1, 64, pad=1, stride=1), residual=True, c=3) → BatchNorm() → ReLU → Dropout() → AveragePool2d(2)
Flatten()
FC(features, nway)

24

Published as a conference paper at ICLR 2022

Model

Train Iteration (ms) Eval Iteration (ms)

ProtoMahalanobis-FC diag
ProtoMahalanobis-FC Rank-1
ProtoMahalanobis-FC Rank-5
ProtoMahalanobis-FC Rank-10
ProtoDDU-FC
ProtoSNGP-FC
Protonet-FC
Protonet-FC SN

10.33±0.40
10.96±0.37
13.06±0.44
15.65±0.35
11.86±0.44
12.32±0.38
2.83±0.39
6.78±0.34

2.95±0.20
3.11±0.22
3.84±0.24
4.63±0.20
3.97±0.21
4.02±0.22
0.86±0.09
1.91±0.08

Table 12: Runtime analysis of linear variants of models

Model

Train Iteration (ms) Eval Iteration (ms)

ProtoMahalanobis Diag
ProtoMahalanobis Rank-1
ProtoMahalanobis Rank-5
ProtoMahalanobis Rank-10
ProtoDDU
ProtoSNGP
Protonet
Protonet SN

11.84±0.69
12.35±0.75
15.04±0.69
17.72±0.71
11.39±0.81
11.43±0.65
3.55±0.24
8.03±0.40

3.61±0.35
3.85±0.31
4.55±0.36
5.42±0.45
3.80±0.29
3.68±0.30
1.14±0.20
2.42±0.18

Table 13: Runtime analysis of CNN variants of models

A.11 FURTHER EIGENVALUE EXPERIMENTS

In Table 14, we perform further experiments and analysis into the behavior of the low rank covariance
encoder outlined in Section 3, we analyze the signiﬁcance of the eigenvalues of the precision matrix.
In this experiment, we ﬁrst obtain the predicted precision matrix and perform an eigendecomposition
A = QΛQ−1 ∈ RN ×N . We then construct a set of alternate precision matrices S = {A(cid:48)}N
i=1, where
each set element is a recomposition A(cid:48)
i has one eigenvalue reset to 1. We then
compute the ﬁnal Accuracy, NLL, and ECE once for each matrix in S. If the predicted eigenvalues
are due to arbitrary error or noise, then we would expect to see that the test statistics would arbitrarily
improve for some precision matrices in S.

iQ−1, where Λ(cid:48)

i = QΛ(cid:48)

Instead, in Table 14 we see that the precision matrix which is predicted from the Set Transformer
gives the best results on the test set in all cases, showing that all of the predicted values are necessary
for the given solution. This experiment utilizes Omniglot 5-way/5-shot and the ProtoMahalanobis
Rank-1 variant.

25

Published as a conference paper at ICLR 2022

Figure 22: Extra results comparing to Deep Kernel Transfer (Patacchiola et al., 2020) on the Omniglot
dataset. In our experiments, DKT showed a large variance in performance between tasks. In the
5-way/1-shot case, calibration on corrupted data comes at the expense of underconﬁdence on in
distribution data.

26

Test12345Shift Intensity: Omniglot-C (5-way/5-shot)0.40.60.81.0AccuracyMethodDKT-cosDKT-bncosOurs (Diag)Ours (Rank-1)Test12345Shift Intensity: Omniglot-C (5-way/5-shot)0.000.050.100.150.200.250.30ECETest12345Shift Intensity: Omniglot-C (5-way/1-shot)0.40.50.60.70.80.91.0AccuracyTest12345Shift Intensity: Omniglot-C (5-way/1-shot)0.00.10.20.30.40.5ECEPublished as a conference paper at ICLR 2022

Σ−1 Matrix
predicted level 0 Σ−1
modiﬁed level 0 Σ−1
predicted level 1 Σ−1
modiﬁed level 1 Σ−1
predicted level 2 Σ−1
modiﬁed level 2 Σ−1
predicted level 3 Σ−1
modiﬁed level 3 Σ−1
predicted level 4 Σ−1
modiﬁed level 4 Σ−1
predicted level 5 Σ−1
modiﬁed level 5 Σ−1

Accuracy

NLL

ECE

better%

99.55±0.04
97.98±0.23

0.02±0.00
0.09±0.01

63.43±1.49
58.44±2.11

0.97±0.05
1.30±0.17

1.21±0.16
3.31±0.60

5.22±0.87
9.06±1.73

100%/100%/100%
0%/0%/0%

100%/100%/100%
0%/0%/0%

56.31±1.59
51.17±2.12

1.14±0.04
1.54±0.37

6.92±1.43
11.92±2.04

100%/100%/100%
0%/0%/0%

52.45±1.33
46.39±1.76

1.21±0.04
1.72±0.74

6.62±1.45
13.51±2.03

100%/100%/100%
0%/0%/0%

45.07±1.04
39.70±1.36

1.37±0.02
1.97±1.13

8.34±1.26
15.85±2.11

100%/100%/100%
0%/0%/0%

40.73±0.70
36.54±0.91

1.46±0.02
2.09±1.42

10.25±1.16
17.34±2.06

100%/100%/100%
0%/0%/0%

Table 14: Analyzing the predicted precision matrix against a set of modiﬁed precision matrices with
perturbed eigenvalues. The predicted precision matrix performs better in every instance, showing that
the precision matrix is not arbitrary. This data comes from Omniglot 5-way/5-shot and utilizes the
ProtoMahalanobis Rank-1 variant

27

