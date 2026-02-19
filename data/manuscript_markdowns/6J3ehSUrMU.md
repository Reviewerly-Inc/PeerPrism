Published as a conference paper at ICLR 2024

PRINCIPLED FEDERATED DOMAIN ADAPTATION:
GRADIENT PROJECTION AND AUTO-WEIGHTING

Enyi Jiang∗
enyij2@illinois.edu
UIUC

Yibo Jacky Zhang∗
yiboz@stanford.edu
Stanford

Sanmi Koyejo
sanmi@cs.stanford.edu
Stanford

ABSTRACT

Federated Domain Adaptation (FDA) describes the federated learning (FL) setting
where source clients and a server work collaboratively to improve the performance
of a target client where limited data is available. The domain shift between the
source and target domains, coupled with limited data of the target client, makes FDA
a challenging problem, e.g., common techniques such as federated averaging and
fine-tuning fail due to domain shift and data scarcity. To theoretically understand
the problem, we introduce new metrics that characterize the FDA setting and
a theoretical framework with novel theorems for analyzing the performance of
server aggregation rules. Further, we propose a novel lightweight aggregation
rule, Federated Gradient Projection (FedGP), which significantly improves the
target performance with domain shift and data scarcity. Moreover, our theory
suggests an auto-weighting scheme that finds the optimal combinations of the
source and target gradients. This scheme improves both FedGP and a simpler
heuristic aggregation rule. Extensive experiments verify the theoretical insights
and illustrate the effectiveness of the proposed methods in practice.

1

INTRODUCTION

Federated learning (FL) is a distributed machine learning paradigm that aggregates clients’ models
on the server while maintaining data privacy (McMahan et al., 2017). FL is particularly interesting
in real-world applications where data heterogeneity and insufficiency are common issues, such
as healthcare settings. For instance, a small local hospital may struggle to train a generalizable
model independently due to insufficient data, and the domain divergence from other hospitals further
complicates the application of FL. A promising framework for addressing this problem is Federated
Domain Adaptation (FDA), where source clients collaborate with the server to enhance the model
performance of a target client (Peng et al., 2020). FDA presents a considerable hurdle due to two
primary factors: (i) the domain shift existing between source and target domains, and (ii) the scarcity
of data in the target domain.

Recent works have studied these challenges of domain shift and limited data in federated settings.
Some of these works aim to minimize the impacts of distribution shifts between clients (Wang et al.,
2019; Karimireddy et al., 2020; Xie et al., 2020b), e.g., via personalized federated learning (Deng
et al., 2020; Li et al., 2021; Collins et al., 2021; Marfoq et al., 2022). However, these studies
commonly presume that all clients possess ample data, an assumption that may not hold for small
hospitals in a cross-silo FL setting. Data scarcity challenges can be a crucial bottleneck in real-world
scenarios, e.g., small hospitals lack data – whether labeled or unlabeled. In the special (and arguably
less common) case where the target client has access to abundant unlabeled data, Unsupervised
Federated Domain Adaptation (UFDA) (Peng et al., 2020; Feng et al., 2021; Wu & Gong, 2021) may
be useful. Despite existing work, there remains an under-explored gap in the literature addressing
both challenges, namely domain shift and data scarcity, coexist.

To fill the gap, this work directly approaches the two principal challenges associated with FDA,
focusing on carefully designing server aggregation rules, i.e., mechanisms used by the server to
combine updates across source and target clients within each global optimization loop. We focus on

∗Equal contribution

1

Published as a conference paper at ICLR 2024

aggregation rules as they are easy to implement – requiring only the server to change its operations
(e.g., variations of federated averaging (McMahan et al., 2017)), and thus have become the primary
target of innovation in the federated learning literature. In brief, our work is motivated by the question:

How does one define a “good” FDA aggregation rule?

To our best understanding, there are no theoretical foundations that
systematically examine the behaviors of various federated aggrega-
tion rules within the context of the FDA. Therefore, we introduce
a theoretical framework that establishes two metrics to characterize
the FDA settings and employ them to analyze the performance of
FDA aggregation rules. The proposed metrics characterize (i) the
divergence between source and target domains and (ii) the level of
training data scarcity in the target domain. Leveraging the proposed
theoretical framework, we propose and analyze two aggregation
approaches. The first is a simple heuristic FedDA, a simple convex
combination of source and target gradients. Perhaps surprisingly, we
discover that even noisy gradients, computed using the limited data
of the target client, can still deliver a valuable signal. The second
is a novel filtering-based gradient projection method, FedGP. This
method is designed to extract and aggregate beneficial components
of the source gradients with the assistance of the target gradient, as
depicted in Figure 1. FedGP calculates a convex combination of the target gradient and its positive
projection along the direction of source gradients. Intriguingly, using a generalization analysis on the
target domain, our theoretical framework unravels why FedGP may outperform FedDA– specifically,
we find that performing the projection operation before the convex combination is crucial.

Figure 1: FedGP filters out
the negative source gradients
(colored in red) and convexly
combines gT and its projec-
tions to to direction of the
remaining source gradients
(green ones).

Importantly, our theoretical framework suggests the optimal weights for combining source and target
gradients, leading to auto-weighted versions of both FedGP and FedDA. In particular, we find
that the under-performing FedDA is significantly improved by using auto-weighting – enough to
be competitive with FedGP, demonstrating the value of our theory. Across extensive datasets, we
demonstrate that FedGP, as well as the auto-weighted FedDA and FedGP, outperforms personalized
FL and UFDA baselines. Our code is at https://github.com/jackyzyb/AutoFedGP.

Summary of Contributions. Our contributions are both theoretical and practical, addressing the
FDA problem through federated aggregation in a principled way.

• We introduce a theoretical framework understanding and analyzing the performance of FDA
aggregation rules, inspired by two challenges existing in FDA. Our theories provide a principled
response to the question: How do we define a “good” FDA aggregation rule?

• We propose FedGP as an effective solution to the FDA challenges of substantial domain shifts and

limited target data.

• Our theory determines the optimal weight parameter for aggregation rules, FedDA and FedGP.

This auto-weighting scheme leads to further performance improvements.

• Extensive experiments illustrate that our theory is predictive of practice. The proposed meth-
ods outperform personalized FL and UFDA baselines on real-world ColoredMNIST, VLCS,
TerraIncognita, and DomainNet datasets.

2 THE PROBLEM OF FEDERATED DOMAIN ADAPTATION

We begin with a general definition of the problem of Federated Domain Adaptation and, subsequently
a review of related literature in the field.

Notation. Let D be a data domain1 on a ground set Z. In our supervised setting, a data point
z ∈ Z is the tuple of input and output data2 We denote the loss function as ℓ : Θ × Z → R+
where the parameter space is Θ = Rm; an m-dimensional Euclidean space. The population loss is
ℓD(θ) := Ez∼Dℓ(θ, z), where Ez∼D is the expectation w.r.t. D. Let (cid:98)D be a finite sample dataset

1In this paper, the terms distribution and domain are used interchangeably.
2For example, let x be the inputs and y be the targets, then z = (x, y).

2

Published as a conference paper at ICLR 2024

drawn from D, then ℓ
z∈ (cid:98)D ℓ(θ, z), where | (cid:98)D| = n is the size of the dataset. We use
[N ] := {1, 2, . . . , N }. By default, ⟨·, ·⟩, and ∥ · ∥ denote the Euclidean inner product and Euclidean
norm, respectively.

(cid:98)D(θ) := 1
| (cid:98)D|

(cid:80)

In FDA, there are N source clients with their respective source domains {DSi}i∈[N ] and a target
client with the target domain DT . For ∀i ∈ [N ], (cid:98)DSi denotes the ith source client dataset, and (cid:98)DT
denotes the target client dataset. We focus on the setting where | (cid:98)DT | is relatively small. In standard
federated learning, all clients collaborate to learn a global model orchestrated by the server, i.e, clients
cannot communicate directly, and all information is shared from/to the server. In contrast, FDA uses
the same system architecture to improve a single client’s performance.
Definition 2.1 (Aggregation for Federated Domain Adaptation (FDA)). The FDA problem is a
federated learning problem where all clients collaborate to improve the global model for a target
domain. The global model is trained by iteratively updating the global model parameter

θ ← θ − µ · Aggr({∇ℓ

(θ)}i∈[N ], ∇ℓ

(θ)),

(cid:98)DSi
where ∇ℓ is the gradient, µ is the step size. We seek an aggregation strategy Aggr(·) such that after
training, the global model parameter θ minimizes the target domain population loss function ℓDT (θ).
Note that we allow the aggregation function Aggr(·) to depend on the iteration index.

(cid:98)DT

There are a number of challenges that make FDA difficult to solve. First, the amount of labeled
data in the target domain is typically limited, which makes it difficult to learn a generalizable model.
Second, the source and target domains have different data distributions, which can lead to a mismatch
between the features learned by the source and target models. Moreover, the model must be trained
in a privacy-preserving manner where local data cannot be shared.

2.1 RELATED WORK

Data heterogeneity, personalization and label deficiency in FL. Distribution shifts between clients
remain a crucial challenge in FL. Current work often focuses on improving the aggregation rules:
Karimireddy et al. (2020) use control variates and Xie et al. (2020b) cluster the client weights to
correct the drifts among clients. More recently, there are works (Deng et al., 2020; Li et al., 2021;
Collins et al., 2021; Marfoq et al., 2022) concentrating on personalized federated learning by finding a
better mixture of local/global models and exploring shared representation. Further, recent works have
addressed the label deficiency problem with self-supervision or semi-supervision for personalized
models (Jeong et al., 2020; He et al., 2021; Yang et al., 2021). To our knowledge, all existing work
assumes sufficient data for all clients - nevertheless, the performance of a client with data deficiency
and large shifts may become unsatisfying (Table 1). Compared to related work on personalized FL,
our method is more robust to data scarcity on the target client.

Unsupervised federated domain adaptation. There is a considerable amount of recent work on
unsupervised federated domain adaptation (UFDA), with recent highlights in utilizing adversarial
networks (Saito et al., 2018; Zhao et al., 2018), knowledge distillation (Nguyen et al., 2021), and
source-free methods (Liang et al., 2020). Peng et al. (2020); Li et al. (2020) is the first to extend
MSDA into an FL setting; they apply adversarial adaptation techniques to align the representations
of nodes. More recently, in KD3A (Feng et al., 2021) and COPA (Wu & Gong, 2021), the server
with unlabeled target samples aggregates the local models by learning the importance of each source
domain via knowledge distillation and collaborative optimization. Their work assumes abundant data
without labels in the target domain, while small hospitals usually do not have enough data. Also,
training with unlabeled data every round is computationally expensive. Compared to their work,
we study a more important challenge where the data (not just the labels) are scarce. We show our
approaches achieve superior performance using substantially less target data on various benchmarks.

Using additional gradient information in FL. Model updates in each communication round
may provide valuable insights into client convergence directions. This idea has been explored for
robustness in FL, particularly with untrusted clients. For example, Zeno++ (Xie et al., 2020a) and
FlTrust (Cao et al., 2021) leverage the additional gradient computed from a small clean training dataset
on the server to compute the scores of candidate gradients for detecting the malicious adversaries.
Differently, our work focuses on a different task of improving the performance of the target domain
with auto-weighted aggregation rules that utilize the gradient signals from all clients.

3

Published as a conference paper at ICLR 2024

3 A THEORETICAL FRAMEWORK FOR ANALYZING AGGREGATION RULES

FOR FDA

This section introduces a general framework and a theoretical analysis of aggregation rules for
federated domain adaptation.

Additional Notation and Setting. We use additional notation to motivate a functional view of FDA.
Let gD : Θ → Θ with gD(θ) := ∇ℓD(θ). Given a distribution π on the parameter space Θ, we define
an inner product ⟨gD, gD′⟩π = Eθ∼π[⟨gD(θ), gD′(θ)⟩]. We interchangeably denote π as both the
distribution and the probability measure. The inner product induces the Lπ-norm on gD as ∥gD∥π :=
(cid:112)Eθ∼π∥gD(θ)∥2. With the Lπ-norm, we define the Lπ space as {g : Θ → Θ | ∥g∥π < ∞}. Given
an aggregation rule Aggr(·), we denote (cid:98)gAggr(θ) = Aggr({gDSi
(θ)). Note that we do
not care about the generalization on the source domains, and therefore, for the theoretical analysis,
we can view DS = (cid:98)DS without loss of generality. Throughout our theoretical analysis, we make the
following standard assumption about (cid:98)DT , and we use the hat symbol, (cid:98)·, to emphasize that a random
variable is associated with the sampled target domain dataset (cid:98)DT .
Assumption 3.1. We assume the target domain’s local dataset (cid:98)DT = {zi}i∈[n] consists of n i.i.d.
samples from its underlying target domain distribution DT . Note that this implies E
] = gDT .

i=1, g

(θ)}N

(cid:98)DT

[g

(cid:98)DT

(cid:98)DT

Observing Definition 2.1, we can see intuitively that a good aggregation rule should have (cid:98)gAggr be
“close” to the ground-truth target domain gradient gDT . From a functional view, we need to measure
the distance between the two functions. We choose the Lπ-norm, formally stated in the following.
Definition 3.2 (Delta Error of an aggregation rule Aggr(·)). We define the following squared error
term to measure the closeness between (cid:98)gAggr and gDT , i.e.,

∆2

Aggr := E

∥gDT − (cid:98)gAggr∥2
π.

(cid:98)DT

The distribution π characterizes where to measure the gradient difference in the parameter space.

The Delta error ∆2
Aggr is crucial in theoretical analysis and algorithm design due to its two main bene-
fits. First, it indicates the performance of an aggregation rule. Second, it reflects fundamental domain
properties that are irrelevant to the aggregation rules applied. Thus, the Delta error disentangles these
two elements, allowing in-depth analysis and algorithm design.

Concretely, for the first benefit: one expects an aggregation rule with a small Delta error to converge
better as measured by the population target domain loss function gradient ∇ℓDT .
Theorem 3.3 (Convergence and Generalization). For any probability measure π over the parameter
space, and an aggregation rule Aggr(·) with step size µ > 0. Given target domain sampled dataset
(cid:98)DT , update the parameter for T steps by θt+1 := θt − µ(cid:98)gAggr(θt). Assume the gradient ∇ℓ(θ, z)
and (cid:98)gAggr(θ) are γ
γ and a small
enough ϵ > 0, with probability at least 1 − δ we have

2 -Lipschitz in θ such that θt → (cid:98)θAggr. Then, given step size µ ≤ 1

∥∇ℓDT (θT )∥2 ≤

(cid:18)(cid:113)

1
δ2

Cϵ · ∆2

Aggr + O(ϵ)

(cid:19)2

+ O

(cid:19)

(cid:18) 1
T

+ O(ϵ),

[1/π(Bϵ((cid:98)θAggr))]2 and Bϵ((cid:98)θAggr) ⊂ Rm is the ball with radius ϵ centered at (cid:98)θAggr.
where Cϵ = E
The Cϵ measures how well the probability measure π covers where the optimization goes, i.e., (cid:98)θAggr.

(cid:98)DT

Interpretation. The left-hand side reveals the convergence quality of the optimization with respect
to the true target domain loss. As we can see, a smaller Delta error indicates better convergence and
generalization. In addition, we provide an analysis of a single gradient step in Theorem A.1, showing
similar properties of the Delta error. In the above theorem, π is arbitrary, allowing for its appropriate
choice to minimize the Cϵ. Ideally, π would accurately cover where the model parameters are after
optimization. We take this insight in the design of an auto-weighting algorithm, to be discussed later.

The behavior of an aggregation rule should vary with the degree and nature of source-target domain
shift and the data sample quality in the target domain. This suggests the necessity of their formal
characterizations for further in-depth analysis. Given a source domain DS, we can measure its
distance to the target domain DT as the Lπ-norm distance between gDS and the target domain model
ground-truth gradient gDT , hence the following definition.

4

Published as a conference paper at ICLR 2024

Definition 3.4 (Lπ Source-Target Domain Distance). Given a source domain DS, its distance to the
target domain DT is defined as

dπ(DS, DT ) := ∥gDT − gDS ∥π.

This proposed metric dπ has some properties inherited from the norm, including: (i. symme-
try) dπ(DS, DT ) = dπ(DT , DS); (ii, triangle inequality) For any data distribution D we have
dπ(DS, DT ) ≤ dπ(DS, D) + dπ(DT , D); (iii. zero property) For any D we have dπ(D, D) = 0.

To formalize the target domain sample quality, we again measure the distance between (cid:98)DT and DT .
Thus, its mean squared error characterizes how the sample size affects the target domain variance.
Definition 3.5 (Lπ Target Domain Variance). Given the target domain DT and dataset (cid:98)DT =
{zi}i∈[n] where zi ∼ DT is sampled i.i.d., the target domain variance is defined as

π( (cid:98)DT ) := E
σ2

Ez∼DT ∥gDT − ∇ℓ(·, z)∥2
π(z) is the variance of a single sampled gradient function ∇ℓ(·, z).

∥gDT − g

∥2
π =

(cid:98)DT

(cid:98)DT

π =:

where σ2

1
n

1
n

σ2
π(z),

Taken together, our exposition shows the second benefit of the Delta error: it decomposes into a mix
of the target-source domain shift dπ(DS, DT ) and the target domain variance σ2
π( (cid:98)DT ) for at least a
wide range of aggregation rules (including our FedDA and FedGP).
Theorem 3.6 (∆2
Aggr Decomposition Theorem). Consider any aggregation rule Aggr(·) in the
form of (cid:98)gAggr = 1
], i.e., the aggregation rule is defined by a mapping
FAggr : Lπ × Lπ → Lπ. If FAggr is affine w.r.t. to its first argument (i.e., the target gradient function),
and ∀g ∈ Lπ : FAggr[g, g] = g, and the linear mapping associated with FAggr has its eigenvalue
bounded in [λmin, λmax], then for any source and target distributions {DSi}i∈[N ], DT , (cid:98)DT we have
∆2

i∈[N ] FAggr[g

, gDSi

, where

(cid:80)

(cid:80)

(cid:98)DT

N

Aggr ≤ 1
N

i∈[N ] ∆2

Aggr,DSi

∆2

Aggr,DSi

≤ max{λ2

max, λ2

min} ·

σ2
π(z)
n

+ max{(1 − λmax)2, (1 − λmin)2} · dπ(DSi, DT )2.

Interpretation. The implications of this theorem are significant: first, it predicts how effective an
aggregation rule would be, which we use to compare FedGP vs. FedDA. Second, given an estimate
of the domain distance dπ(DSi, DT ) and the target domain variance σ2
π( (cid:98)DT ), we can optimally select
hyper-parameters for the aggregation operation, a process we name the auto-weighting scheme.

With the relevant quantities defined, we can describe an alternative definition of FDA, useful for our
analysis, which answers the pivotal question of how do we define a "good" aggregation rule.
Definition 3.7 (An Error-Analysis Definition of FDA Aggregation). Given the target domain variance
σ2
π( (cid:98)DT ) and source-target domain distances ∀i ∈ [N ] : dπ(DSi, DT ), the problem of FDA is to find
a good strategy Aggr(·) such that its Delta error ∆2

Aggr is minimized.

These definitions give a powerful framework for analyzing and designing aggregation rules:

• given an aggregation rule, we can derive its Delta error and see how it would perform given an
FDA setting (as characterized by the target domain variance and the source-target distances);

• given an FDA setting, we can design aggregation rules to minimize the Delta error.

4 METHODS: GRADIENT PROJECTION AND THE AUTO-WEIGHTING SCHEME

To start, we may try two simple methods, i.e., only using the target gradient and only using a
source gradient (e.g., the ith source domain). The Delta error of these baseline aggregation rules is
straightforward. By definition, we have that

∆2

(cid:98)DT only = σ2

π( (cid:98)DT ),

and ∆2

DSi

only = d2

π(DSi, DT ).

This immediate result demonstrates the usefulness of the proposed framework: if (cid:98)gAggr only uses the
target gradient then the error is the target domain variance; if (cid:98)gAggr only uses a source gradient then

5

Published as a conference paper at ICLR 2024

the error is the corresponding source-target domain bias. Therefore, a good aggregation method must
strike a balance between the bias and variance, i.e., a bias-variance trade-off, and this is precisely
what we will design our auto-weighting mechanism to do. Next, we propose two aggregation methods
and then show how their auto-weighting can be derived.

4.1 THE AGGREGATION RULES: FE DDA AND FE DGP

A straightforward way to combine the source and target gradients is to convexly combine them, as
defined in the following.
Definition 4.1 (FedDA). For each source domains i ∈ [N ], let βi ∈ [0, 1] be the weight that balances
between the ith source domain and the target domain. The FedDA aggregation operation is

FedDA({gDSi

(θ)}N

i=1, g

(cid:98)DT

(θ)) =

1
N

N
(cid:88)

(cid:16)

i=1

(1 − βi)g

(cid:98)DT

(θ) + βigDSi

(cid:17)

(θ)

.

Let us examine the Delta error of FedDA.
Theorem 4.2. Consider FedDA. Given the target domain (cid:98)DT and N source domains DS1 , . . . , DSN ,
we have ∆2

(cid:80)N

FedDA ≤ 1
N

i=1 ∆2
∆2

FedDA,Si

FedDA,Si

, where
= (1 − βi)2σ2

π( (cid:98)DT ) + β2

i d2

π(DSi, DT ).

(1)

2 , we have ∆2

Therefore, we can see the benefits of combining the source and target domains. For example, with
βi = 1
attains the upper
bound in Theorem 3.6 with λmax = λmin = 1 − βi. This hints that there may be other aggregation
rules that can do better, as we shown in the following.

π(DSi , DT ). We note that ∆2

π( (cid:98)DT ) + 1

FedDA,Si

FedDA,Si

4 σ2

4 d2

= 1

Intuitively, due to domain shift, signals from source domains may not always be relevant. Inspired
by the filtering technique in Byzantine robustness of FL (Xie et al., 2020a), we propose Federated
Gradient Projection (FedGP). This method refines and combines beneficial components of the source
gradients, aided by the target gradient, by gradient projection and filtering out unfavorable ones.
Definition 4.3 (FedGP). For each source domains i ∈ [N ], let βi ∈ [0, 1] be the weight that balances
between ith source domain and the target domain. The FedGP aggregation operation is

FedGP({gDSi

(θ)}N

i=1, g

(cid:98)DT

(θ)) =

1
N

N
(cid:88)

(cid:16)

i=1

(1 − βi)g

(cid:98)DT

(θ) + βiProj+(g

(cid:98)DT

(θ)|gDSi

(θ))

(cid:17)

.

where Proj+(g
(cid:98)DT
tion that projects g

(θ)|gDSi
(cid:98)DT

(θ)) = max{⟨g

(θ), gDSi
(θ) to the positive direction of gDSi

(cid:98)DT

(θ)⟩, 0}gDSi
(θ).

(θ)/∥gDSi

(θ)∥2 is the opera-

We first derive the Delta error of FedGP, and compare it to that of FedDA.
Theorem 4.4 (Informal Version). Consider FedGP. Given the target domain (cid:98)DT and N source
(cid:80)N
domains DS1, . . . , DSN , we have ∆2

, where

FedGP ≤ 1
N

FedGP,Si

(cid:18)

∆2

FedGP,Si

≈

(1 − βi)2 +

2βi − β2
i
m

i=1 ∆2
(cid:19)

π( (cid:98)DT ) + β2
σ2

i ¯τ 2d2

π(DSi , DT ),

(2)

In the above equation, m is the model dimension and ¯τ 2 = Eπ[τ (θ)2] ∈ [0, 1] where τ (θ) is the
sin(·) value of the angle between gDS (θ) and gDT (θ) − gDS (θ).

We note the above theorem is the approximated version of Theorem A.5 where the derivation is
non-trivial. The approximations are mostly done in analog to a mean-field analysis, which are detailed
in Appendix A.4.

Interpretation. Comparing the Delta error of FedGP (equation 2) and that of FedDA (equation 1),
we can see that FedGP is more robust to large source-target domain shift dπ(DSi, DT ) given ¯τ < 1.
This aligns with our motivation of FedGP which filters out biased signals from the source domain.
Moreover, our theory reveals a surprising benefit of FedGP as follows. Note that
≈ β2

i (1 − ¯τ 2)d2
In practice, the model dimension m ≫ 1 while the ¯τ 2 < 1, thus we can expect FedGP to be mostly
better than FedDA with the same weight βi. With that said, we move on the auto-weighting scheme.

π(DSi, DT ) − 2βi−β2

m σ2

π( (cid:98)DT ).

− ∆2

FedGP,Si

FedDA,Si

∆2

i

6

Published as a conference paper at ICLR 2024

4.2 THE AUTO-WEIGHTING FE DGP AND FE DDA

Naturally, the above analysis implies a good choice of weighting parameters for either of the methods.
For each source domains Si, we can solve for the optimal βi that minimize the corresponding Delta
errors, i.e., ∆2
(equation 2) for FedGP. Note that for
∆2
m ≈ 0 given the high dimensionality of our models. Since either
of the Delta errors is quadratic in βi, they enjoy closed-form solutions:

we can safely view 2βi−β2

(equation 1) for FedDA and ∆2

FedDA,Si

FedDA,Si

FedGP,Si

i

βFedDA
i

=

σ2
π( (cid:98)DT )
π(DSi, DT ) + σ2
d2

π( (cid:98)DT )

,

βFedGP
i

=

σ2
π( (cid:98)DT )

¯τ 2d2

π(DSi, DT ) + σ2

π( (cid:98)DT )

.

(3)

π( (cid:98)DT ), d2

The exact values of σ2
π(DSi, DT ), ¯τ 2 are unknown, since they would require knowing the
ground-truth target domain gradient gDT . Fortunately, using only the available training data, we can
efficiently obtain unbiased estimators for those values, and accordingly obtain estimators for the best
βi. The construction of the estimators is non-trivial and is detailed in Appendix A.5.

The proposed methods are summarized in Algorithm 1, and detailed in Appendix C. During one
round of computation, the target domain client does B local model updates with B batches of data.
j=1 to estimate the optimal βi, where gj
In practice, we use these intermediate local updates {gj
}B
(cid:98)DT
stands for the local model update using the jth batch of data. In other words, we choose π to be the
empirical distribution of the model parameters encountered along the optimization path, aligning
with Theorem 3.3’s suggestion for an ideal π.

(cid:98)DT

We observe that FedGP, quite remarkably, is robust to the choice of β: simply choosing β = 0.5
is good enough for most of the cases as observed in our experiments. On the other hand, although
FedDA is sensitive to the choice of β, the auto-weighted procedure significantly improves the
performance for FedDA, demonstrating the usefulness of our theoretical framework.

Algorithm 1 FDA: Gradient Projection and the Auto-Weighting Scheme

Input: N source domains DS = {DSi}N
i=1, target
client CT , server S; number of rounds R; aggregation rule Aggr; whether to use auto_weight.
Initialize global model h(0)
for r = 1, 2, ..., R do

i=1, target domain DT ; N source clients {CSi}N

global. Default {βi}N

i=1 ← {0.5}N

i=1.

for source domain client CSi in {CSi}N
← h(r−1)

Initialize local model h(r)
Si

i=1 do

end for
Target domain client CT initialize h(r)
Server S computes model updates gT ← h(r)
if auto_weight then

global, optimize h(r)
Si
global, optimizes h(r)
T − h(r−1)

on DSi, send h(r)
Si
T on DT , send h(r)
− h(r−1)

global and gSi ← h(r)
Si

T ← h(r−1)

to server S.

T to server S.
global for i ∈ [N ].

CT sends intermediate local model updates {gj
S estimates of {dπ(DSi, DT )}N
S updates {βi}N

i=1 according to (3).

(cid:98)DT
i=1, ¯τ 2 and σπ( (cid:98)DT ) using {gSi}N

}B
j=1 to S.

i=1 and {gj
(cid:98)DT

}B
j=1.

end if
S updates the global model as h(r)

global ← h(r−1)

global + Aggr({gSi}N

i=1, gT , {βi}N

i=1).

end for

5 EXPERIMENTS

In this section, we present and discuss the results of real dataset experiments with controlled domain
shifts (Section 5.1) and real-world domain shifts (Section 5.2). Ablation studies on target data
scarcity and visualizations are available in Appendix C.5 & C.8. Synthetic data experiments
verifying our theoretical insights are presented in Appendix B. In Appendix C.3, we show our
methods surpass UFDA and Domain Generalization (DG) methods on PACS (Li et al., 2017), Office-
Home (Venkateswara et al., 2017), and DomainNet (Peng et al., 2019). Implementation details and
extended experiments can be found in the appendix.

7

Published as a conference paper at ICLR 2024

5.1 SEMI-SYNTHETIC DATASET EXPERIMENTS WITH VARING SHIFTS

(a) Fashion-MNIST noisy features

(b) CIFAR-10 noisy features

(c) Fashion-MNIST label shifts

Figure 2: The impact of changing domain shifts with noisy features or label shifts.

Datasets, models, and methods. We create controlled distribution shifts by adding different levels
of feature noise and label shifts to Fashion-MNIST (Xiao et al., 2017) and CIFAR-10 (Krizhevsky
et al., 2009) datasets, adapting from the Non-IID benchmark (Li et al., 2022) with the following
two settings: 1) Noisy features: We add Gaussian noise levels of std = (0.2, 0.4, 0.6, 0.8) to input
images of the target client of two datasets, to create various degrees of shifts between source and
target domains. 2) Label shifts: We split the Fashion-MNIST into two sets with 3 and 7 classes,
respectively, denoted as D1 and D2. DS = η portion from D1 and (1 − η) portion from D2, DT
= (1 − η) portion from D1 and η portion from D2 with η = [0.45, 0.30, 0.15, 0.10, 0.05, 0.00]. In
addition, we use a CNN model architecture. We set the communication round R = 50 and the local
update epoch to 1, with 10 clients (1 target, 9 source clients) in the system. We compare the following
methods: Source Only (only averaging the source gradients), Finetune_Offline (fine-tuning
locally after source-only training), FedDA (β = 0.5), FedGP(β = 0.5), and their auto-weighted
versions FedDA_Auto and FedGP_Auto, the Oracle (supervised training with all target data (DT ))
and Target Only (only use target gradient ( (cid:98)DT )). More details can be found in Appendix C.7.

Auto-weighted methods and FedGP keep a better trade-off between bias and variance. As
shown in Figure 2, when the source-target shifts grow bigger, FedDA, Finetune_Offline, and
Source Only degrade more severely compared with auto-weighted methods, FedGP and Target Only.
We find that auto-weighted methods and FedGP outperform other baselines in most cases, being
less sensitive to changing shifts. In addition, auto-weighted FedDA manages to achieve a significant
improvement compared with the fixed weight FedDA, with a competitive performance compared with
FedGP_Auto, while FedGP_Auto generally has the best accuracy compared with other methods,
which coincides with the theoretical findings. Full experiment results can be found in Appendix C.7.

5.2 REAL DATASET EXPERIMENTS WITH REAL-WORLD SHIFTS

Datasets, models, baselines, and implementations. We use the Domainbed (Gulrajani & Lopez-Paz,
2020a) benchmark with multiple domains, with realistic shifts between source and target clients.
We conduct experiments on three datasets: ColoredMNIST (Arjovsky et al., 2019), VLCS (Fang
et al., 2013), TerraIncognita (Beery et al., 2018) datasets. We randomly sampled 0.1% samples of
ColoredMNIST, and 5% samples of VLCS and TerraIncognita for their respective target domains.
The task is classifying the target domain. We use a CNN model for ColoredMNIST, ResNet-
18 He et al. (2016) for VLCS and TerraIncognita. For baselines in particular, in addition to the
methods in Section 5.1, we compare (1) personalization baselines: FedAvg, Ditto (Li et al., 2021),
FedRep (Collins et al., 2021), APFL (Deng et al., 2020), and KNN-per (Marfoq et al., 2022); (2)
UFDA methods: KD3A (Feng et al., 2021) (current SOTA): note that our proposed methods use few
percentages of target data, while UFDA here uses 100% unlabeled target data; (3) DG method: we
report the best DG performance in DomainBed (Gulrajani & Lopez-Paz, 2020b). For each dataset,
we test the target accuracy of each domain using the left-out domain as the target and the rest as
source domains. More details and full results are in Appendix C.2.

Our methods consistently deliver superior performance. Table 1 reveals that our auto-weighted
methods outperform others in all cases, and some of their accuracies approach/outperform the
corresponding upper bound (Oracle). The auto-weighted scheme improves FedDA significantly.

8

0.00.10.20.30.40.50.60.70.8Noise Level0.450.500.550.600.650.700.750.800.850.90Target AccuracyFashion-MNIST Noisy FeaturesFinetune_OfflineFedDAFedGPFedDA_AutoFedGP_AutoTarget_OnlyOracle0.20.30.40.50.60.70.8Noise Level0.450.500.550.600.650.700.75Target AccuracyCIFAR-10 Noisy FeaturesFinetune_OfflineFedDAFedGPFedDA_AutoFedGP_AutoTarget_OnlyOracle0.100.150.200.250.300.350.400.45Eta0.760.780.800.820.840.860.880.900.92Target AccuracyFashion-MNIST Label ShiftsFinetune_OfflineFedDAFedGPFedDA_AutoFedGP_AutoTarget_OnlyOraclePublished as a conference paper at ICLR 2024

Interestingly, we observe that FedGP, even with default fixed betas (β = 0.5), achieves competitive
results. Our methods surpass personalized FL, UFDA, and DG baselines by significant margins.

Domains
Source Only
FedDA
FedGP
FedDA_Auto
FedGP_Auto
Target Only
FedAvg
Ditto
FedRep
APFL
KNN-per
KD3A (100% data)
Best DG
Oracle

ColoredMNIST (0.1%)

VLCS (5%)

L

V

+80%

+90%

S
69.1(2.0)
76.7(0.9)
78.7(1.3)
83.7(2.3)

-90% Avg
56.8(0.8) 62.4(1.8) 27.8(0.8) 49.0
60.5(2.5) 65.1(1.3) 33.0(3.2) 52.9
83.7(9.9) 74.4(4.4) 89.8(0.5) 82.4
85.3(5.7) 73.1(7.3) 88.9(1.1) 82.7
86.2(4.5) 76.5(7.3) 89.6(0.4) 84.1
85.6(4.8) 73.5(3.0) 87.1(3.4) 82.1
48.7
50.9
44.5
45.2
49.1
49.3
40.7
90.0(0.4) 80.3(0.4) 90.0(0.5) 86.8 100.0(0.0) 72.7(2.5) 78.7(1.4)

Avg
72.6
60.7(1.8) 70.2(2.0)
79.5
68.2(1.4) 75.3(1.5)
80.7
71.1(1.2) 73.6(3.1)
73.1(1.3) 78.4(1.5)
83.8
73.2(1.8) 78.6(1.5) 83.47(2.5) 83.8
78.7
68.9(1.9) 72.3(1.7)
75.7
75.0
73.0
61.2
78.4
79.0
78.7
83.5

C
90.5(5.3)
97.7(0.5)
99.4(0.3)
99.8(0.2)
99.9(0.2)
97.8(1.5)
96.3
95.9
91.1
68.6
97.8
99.6
96.9

76.0(1.9)
68.7
66.1
70.1
49.9
74.4
80.5
78.7
82.7(1.1)

72.0
71.3
36.4
61.6
67.6
73.1
62.1

63.2
62.2
65.4
43.8
67.5
65.2
49.9

69.8
70.5
70.3
65.4
75.7
78.1
73.3

68.0
67.5
60.4
61.0
65.9
63.3
65.7

10.9
19.3
31.7
30.2
12.4
9.7
10.0

Terre (5%)
Avg
37.5
64.7
71.2
74.6
74.4
67.2
30.0
28.8
20.9
52.7
43.1
39.0
48.7
93.1

Table 1: Target domain test accuracy (%) on ColoredMNIST, VLCS, and DomainNet.

5.3 ABLATION STUDY AND DISCUSSION

The effect of source-target balancing weight β. We run FedDA and FedGP with varying β on
Fashion-MNIST, CIFAR10, and Colored-MNIST, as shown in Figure 3. In most cases, we observe
FedGP outperforming FedDA, and FedDA being more sensitive to the varying β values, suggesting
that FedDA_Auto can choose optimal enough β. Complete results are in Appendix C.6.

Effectiveness of projection and filtering. We show the effectiveness of gradient projection and filter-
ing in Fashion-MNIST and CIFAR-10 noisy feature experiments in Table 4. Compared with FedDA,
which does not perform projection and filtering, projection achieves a large (15%) performance gain,
especially when the shifts are larger. Further, we generally get a 1% − 2% gain via filtering.

Noise level
N/A
w proj
FedGP

0.4
58.60/54.67
69.51/64.04
71.09/65.28

Fashion-MNIST / CIFAR10
0.6
50.13/49.77
65.46/62.06
68.01/63.29

0.8
45.51/47.08
60.70/60.91
62.22/61.59

(a) Fashion-MNIST
(0.4 noise level)

(b) CIFAR-10
(0.4 noise level)

(c) ColoredMNIST
(target: +90%)

Figure 4: Ablation study on projection
and filtering.

Figure 3: The effect of β on FedDA and FedGP.

Discussion. We analyze the computational and communication cost for our proposed methods
in Appendix C.10 and C.4, where we show how the proposed aggregation rules, especially the
auto-weighted operation, can be implemented efficiently. Moreover, we observe in our experiments
that Finetune_Offline is sensitive to its pre-trained model (obtained via FedAvg), highlighting
the necessity of a deeper study of the relation between personalization and adaptation. Lastly,
although different from UFDA and semi-supervised domain adaptation (SSDA) settings, which use
unlabeled samples (Saito et al., 2019; Kim & Kim, 2020), we conduct experiments comparing them
in Appendix C.3 on DomainNet and C.12 for SSDA. Our auto-weighted methods have better or
comparable performance across domains, especially for large shift cases.

6 CONCLUSION

We provide a theoretical framework that first formally defines the metrics to connect FDA settings
with aggregation rules. We propose FedGP, a filtering-based aggregation rule via gradient projection,
and develop the auto-weighted scheme that dynamically finds the best weights - both significantly
improve the target performances and outperform various baselines. In the future, we plan to extend
the current framework to perform FDA simultaneously on several source/target clients, explore the
relationship between personalization and adaptation, as well as devise stronger aggregation rules.

9

betaAcc5254565850.00.20.40.60.81.0FedDA FedGPbetaAcc5254565850.00.20.40.60.81.0FedDA FedGPbetaAcc50607080901000.00.20.40.60.81.0FedDA (+90%)FedGP (+90%)Published as a conference paper at ICLR 2024

ACKNOWLEDGMENTS

This work is partially supported by NSF III 2046795, IIS 1909577, CCF 1934986, NIH
1R01MH116226-01A, NIFA award 2020-67021-32799, the Alfred P. Sloan Foundation, and Google
Inc.

REFERENCES

Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization.

arXiv preprint arXiv:1907.02893, 2019.

Sara Beery, Grant Van Horn, and Pietro Perona. Recognition in terra incognita. In Proceedings of the

European conference on computer vision (ECCV), pp. 456–473, 2018.

Xiaoyu Cao, Minghong Fang, Jia Liu, and Neil Gong. Fltrust: Byzantine-robust federated learning

via trust bootstrapping. In Proceedings of NDSS, 2021.

Liam Collins, Hamed Hassani, Aryan Mokhtari, and Sanjay Shakkottai. Exploiting shared represen-
tations for personalized federated learning. In International Conference on Machine Learning, pp.
2089–2099. PMLR, 2021.

Li Deng. The mnist database of handwritten digit images for machine learning research. IEEE Signal

Processing Magazine, 29(6):141–142, 2012.

Yuyang Deng, Mohammad Mahdi Kamani, and Mehrdad Mahdavi. Adaptive personalized federated

learning. arXiv preprint arXiv:2003.13461, 2020.

Jean Ogier Du Terrail, Samy-Safwan Ayed, Edwige Cyffers, Felix Grimberg, Chaoyang He, Regis
Loeb, Paul Mangold, Tanguy Marchand, Othmane Marfoq, Erum Mushtaq, et al. Flamby: Datasets
and benchmarks for cross-silo federated learning in realistic healthcare settings. In NeurIPS,
Datasets and Benchmarks Track, 2022.

Chen Fang, Ye Xu, and Daniel N Rockmore. Unbiased metric learning: On the utilization of multiple
datasets and web images for softening bias. In Proceedings of the IEEE International Conference
on Computer Vision, pp. 1657–1664, 2013.

Haozhe Feng, Zhaoyang You, Minghao Chen, Tianye Zhang, Minfeng Zhu, Fei Wu, Chao Wu, and
Wei Chen. Kd3a: Unsupervised multi-source decentralized domain adaptation via knowledge
In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International
distillation.
Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp.
3274–3283. PMLR, 18–24 Jul 2021.

Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. In International

Conference on Learning Representations, 2020a.

Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. arXiv preprint

arXiv:2007.01434, 2020b.

Chaoyang He, Zhengyu Yang, Erum Mushtaq, Sunwoo Lee, Mahdi Soltanolkotabi, and Salman
Avestimehr. Ssfl: Tackling label deficiency in federated learning via personalized self-supervision.
arXiv preprint arXiv:2110.02470, 2021.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition,
pp. 770–778, 2016.

Wonyong Jeong, Jaehong Yoon, Eunho Yang, and Sung Ju Hwang. Federated semi-supervised
learning with inter-client consistency & disjoint learning. In International Conference on Learning
Representations, 2020.

10

Published as a conference paper at ICLR 2024

Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and
Ananda Theertha Suresh. SCAFFOLD: Stochastic controlled averaging for federated learning.
In Hal Daumé III and Aarti Singh (eds.), Proceedings of the 37th International Conference on
Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 5132–5143.
PMLR, 13–18 Jul 2020.

Taekyung Kim and Changick Kim. Attract, perturb, and explore: Learning a feature alignment
network for semi-supervised domain adaptation. In Computer Vision–ECCV 2020: 16th European
Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIV 16, pp. 591–607. Springer,
2020.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint

arXiv:1412.6980, 2014.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy M Hospedales. Deeper, broader and artier domain
generalization. In Proceedings of the IEEE international conference on computer vision, pp.
5542–5550, 2017.

Qinbin Li, Yiqun Diao, Quan Chen, and Bingsheng He. Federated learning on non-iid data silos: An

experimental study. In IEEE International Conference on Data Engineering, 2022.

Tian Li, Shengyuan Hu, Ahmad Beirami, and Virginia Smith. Ditto: Fair and robust federated learning
through personalization. In International Conference on Machine Learning, pp. 6357–6368. PMLR,
2021.

Xiaoxiao Li, Yufeng Gu, Nicha Dvornek, Lawrence H Staib, Pamela Ventola, and James S Duncan.
Multi-site fmri analysis using privacy-preserving federated learning and domain adaptation: Abide
results. Medical Image Analysis, 65:101765, 2020.

Jian Liang, Dapeng Hu, and Jiashi Feng. Do we really need to access the source data? Source
hypothesis transfer for unsupervised domain adaptation.
In Hal Daumé III and Aarti Singh
(eds.), Proceedings of the 37th International Conference on Machine Learning, volume 119 of
Proceedings of Machine Learning Research, pp. 6028–6039. PMLR, 13–18 Jul 2020.

Othmane Marfoq, Giovanni Neglia, Richard Vidal, and Laetitia Kameni. Personalized federated
learning through local memorization. In International Conference on Machine Learning, pp.
15070–15092. PMLR, 2022.

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas.
Communication-Efficient Learning of Deep Networks from Decentralized Data. In Aarti Singh
and Jerry Zhu (eds.), Proceedings of the 20th International Conference on Artificial Intelligence
and Statistics, volume 54 of Proceedings of Machine Learning Research, pp. 1273–1282. PMLR,
20–22 Apr 2017.

Tuan Nguyen, Trung Le, He Zhao, Quan Hung Tran, Truyen Nguyen, and Dinh Phung. Most:
Multi-source domain adaptation via optimal transport for student-teacher learning. In Uncertainty
in Artificial Intelligence, pp. 225–235. PMLR, 2021.

Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching
for multi-source domain adaptation. In Proceedings of the IEEE/CVF international conference on
computer vision, pp. 1406–1415, 2019.

Xingchao Peng, Zijun Huang, Yizhe Zhu, and Kate Saenko. Federated adversarial domain adaptation.

In International Conference on Learning Representations, 2020.

Daniel Rothchild, Ashwinee Panda, Enayat Ullah, Nikita Ivkin, Ion Stoica, Vladimir Braverman,
Joseph Gonzalez, and Raman Arora. Fetchsgd: Communication-efficient federated learning with
sketching. In International Conference on Machine Learning, pp. 8253–8265. PMLR, 2020.

Kuniaki Saito, Kohei Watanabe, Yoshitaka Ushiku, and Tatsuya Harada. Maximum classifier
discrepancy for unsupervised domain adaptation. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pp. 3723–3732, 2018.

11

Published as a conference paper at ICLR 2024

Kuniaki Saito, Donghyun Kim, Stan Sclaroff, Trevor Darrell, and Kate Saenko. Semi-supervised

domain adaptation via minimax entropy. ICCV, 2019.

Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman Panchanathan. Deep
hashing network for unsupervised domain adaptation. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pp. 5018–5027, 2017.

Hongyi Wang, Mikhail Yurochkin, Yuekai Sun, Dimitris Papailiopoulos, and Yasaman Khazaeni. Fed-
erated learning with matched averaging. In International Conference on Learning Representations,
2019.

Guile Wu and Shaogang Gong. Collaborative optimization and aggregation for decentralized domain
generalization and adaptation. In Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV), pp. 6484–6493, October 2021.

Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking

machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.

Cong Xie, Sanmi Koyejo, and Indranil Gupta. Zeno++: Robust fully asynchronous SGD. In Proceed-
ings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of
Machine Learning Research, pp. 10495–10503. PMLR, 13–18 Jul 2020a.

Ming Xie, Guodong Long, Tao Shen, Tianyi Zhou, Xianzhi Wang, Jing Jiang, and Chengqi Zhang.

Multi-center federated learning, 2020b.

Dong Yang, Ziyue Xu, Wenqi Li, Andriy Myronenko, Holger R Roth, Stephanie Harmon, Sheng
Xu, Baris Turkbey, Evrim Turkbey, Xiaosong Wang, et al. Federated semi-supervised learning for
covid region segmentation in chest ct using multi-national data from china, italy, japan. Medical
image analysis, 70:101992, 2021.

Han Zhao, Shanghang Zhang, Guanhang Wu, José MF Moura, Joao P Costeira, and Geoffrey J Gordon.
Adversarial multiple source domain adaptation. Advances in neural information processing systems,
31, 2018.

12

Published as a conference paper at ICLR 2024

Appendix Contents

• Section A: Supplementary Theoretical Results

– A.1: Proof of Theorem 3.3
– A.2: Proof of Theorem 3.6
– A.3: Proof of Theorem 4.2
– A.4: Proof of Theorem 4.4
– A.5: Additional Discussion of the Auto-weighting Method

• Section B: Synthetic Data Experiments
• Section C: Supplementary Experiment Information

– C.1: Algorithm Outlines for Federated Domain Adaptation
– C.2: Real-World Experiment Implementation Details and Results
– C.3: Additional Results on DomainBed Datasets
– C.4: Auto-Weighted Methods: Implementation Details, Time and

Space Complexity

– C.5: Visualization of Auto-Weighted Betas Values on Real-World

Distribution Shifts

– C.6: Additional Experiment Results on Varying Static Weights (β)
– C.7: Semi-Synthetic Experiment Settings, Implementation, and Results
– C.8: Additional Ablation Study Results
– C.9: Implementation Details of FedGP
– C.10: Gradient Projection Method’s Time and Space Complexity
– C.11: Additional Experiment Results on Fed-Heart
– C.12: Comparison with the Semi-Supervised Domain Adaptation

(SSDA) Method

A SUPPLEMENTARY THEORETICAL RESULTS

In this section, we provide theoretical results that are omitted in the main paper due to space
limitations. Specifically, in the first four subsections, we provide proofs of our theorems. Further, in
subsection A.5 we present an omitted discussion for our auto-weighting method, including how the
estimators are constructed.

A.1 PROOF OF THEOREM 3.3

We first prove the following theorem where we study what happens when we do one step of optimiza-
tion. While requiring minimal assumptions, this theorem shares the same intuition as Theorem 3.3
regarding the importance of the Delta error.
Theorem A.1. Consider model parameter θ ∼ π and an aggregation rule Aggr(·) with step size
µ > 0. Define the updated parameter as

Assuming the gradient ∇ℓ(θ, z) is γ-Lipschitz in θ for any z, and let the step size µ ≤ 1

γ , we have

θ+ := θ − µ(cid:98)gAggr(θ).

(cid:98)DT ,θ[ℓDT (θ+) − ℓDT (θ)] ≤ − µ
E

2 (∥gDT ∥2

π − ∆2

Aggr).

Proof. Given any distribution data D, we first prove that ∇ℓD is also γ-Lipschitz as below. For
∀θ1, θ2 ∈ Θ:

∥∇ℓD(θ1) − ∇ℓD(θ2)∥ = ∥Ez∼D[∇ℓ(θ1, z) − ∇ℓ(θ2, z)]∥
≤ Ez∼D∥∇ℓ(θ1, z) − ∇ℓ(θ2, z)∥
≤ Ez∼Dγ∥θ1 − θ2∥
= γ∥θ1 − θ2∥.

(Jensen’s inequality)
(∇ℓ(·, z) is γ-Lipschitz)

13

Published as a conference paper at ICLR 2024

Therefore, we know that ℓDT is γ-smooth. Conditioned on a θ and a (cid:98)DT , and apply the definition of
smoothness we have

ℓDT (θ+) − ℓDT (θ) ≤ ⟨∇ℓDT (θ), θ+ − θ⟩ +

∥θ+ − θ∥2

γ
2
∥µ(cid:98)gAggr(θ)∥2

γ
2

= −⟨∇ℓDT (θ), µ(cid:98)gAggr(θ)⟩ +
= −µ⟨∇ℓDT (θ), (cid:98)gAggr(θ) − ∇ℓDT (θ) + ∇ℓDT (θ)⟩

γµ2
2

+

∥(cid:98)gAggr(θ) − ∇ℓDT (θ) + ∇ℓDT (θ)∥2
= −µ(⟨∇ℓDT (θ), (cid:98)gAggr(θ) − ∇ℓDT (θ)⟩ + ∥∇ℓDT (θ)∥2)

+

γµ2
2

(∥(cid:98)gAggr(θ) − ∇ℓDT (θ)∥2 + ∥∇ℓDT (θ)∥2 + 2⟨(cid:98)gAggr(θ) − ∇ℓDT (θ), ∇ℓDT (θ)⟩)

= (µ − γµ2)(⟨∇ℓDT (θ), ∇ℓDT (θ) − (cid:98)gAggr(θ)⟩)
− µ)∥∇ℓDT (θ)∥2 +

+ (

γµ2
2

γµ2
2

∥(cid:98)gAggr(θ) − ∇ℓDT (θ)∥2

≤ (µ − γµ2) · ∥∇ℓDT (θ)∥ · ∥(cid:98)gAggr(θ) − ∇ℓDT (θ)∥

(Cauchy–Schwarz inequality)

+ (

γµ2
2
µ − γµ2
2

≤

γµ2
2

− µ)∥∇ℓDT (θ)∥2 +

γµ2
2

∥(cid:98)gAggr(θ) − ∇ℓDT (θ)∥2.

(cid:0)∥∇ℓDT (θ)∥2 + ∥(cid:98)gAggr(θ) − ∇ℓDT (θ)∥2(cid:1)

− µ)∥∇ℓDT (θ)∥2 +

+ (
(cid:0)∥∇ℓDT (θ)∥2 − ∥(cid:98)gAggr(θ) − ∇ℓDT (θ)∥2(cid:1) .

∥(cid:98)gAggr(θ) − ∇ℓDT (θ)∥2.

= −

µ
2

γµ2
2

(AM-GM inequality)

(4)

Additionally, note that the above two inequalities stand because: the step size µ ≤ 1
µ − γµ2 = µ2( 1

(cid:98)DT ,θ on both sides gives

µ − γ) ≥ 0. Taking the expectation E
µ
2

(cid:16)

E
(cid:98)DT ,θ[∥∇ℓDT (θ)∥2] − E

E
(cid:98)DT ,θ[ℓDT (θ+) − ℓDT (θ)] ≤ −

(cid:98)DT ,θ[∥(cid:98)gAggr(θ) − ∇ℓDT (θ)∥2]

(cid:17)

γ and thus

Note that we denote gDT = ∇ℓDT . Thus, with the Lπ norm notation we have

µ
E
(cid:98)DT ,θ[ℓDT (θ+) − ℓDT (θ)] ≤ −
2
Aggr = E

Finally, by Definition 3.2 we can see ∆2

(cid:16)
∥gDT ∥2

π − E

(cid:98)DT

∥(cid:98)gAggr − gDT ∥2

π

(cid:17)

.

∥(cid:98)gAggr − gDT ∥2

π, which concludes the proof.

(cid:98)DT

Theorem A.2 (Theorem 3.3 Restated). For any probability measure π over the parameter space, and
an aggregation rule Aggr(·) with step size µ > 0. Given target dataset (cid:98)DT , update the parameter
for T steps as

θt+1 := θt − µ(cid:98)gAggr(θt).

Assume the gradient ∇ℓ(θ, z) and (cid:98)gAggr(θ) is γ
step size µ ≤ 1

γ and a small enough ϵ > 0, with probability at least 1 − δ we have

2 -Lipschitz in θ such that θt → (cid:98)θAggr. Then, given

∥∇ℓDT (θT )∥2 ≤

(cid:18)(cid:113)

1
δ2

Cϵ · ∆2

Aggr + O(ϵ)

(cid:19)2

+ O

(cid:19)

(cid:18) 1
T

+ O(ϵ),

(cid:104)

(cid:105)2

(cid:98)DT

1/π(Bϵ((cid:98)θAggr))

where Cϵ = E

and Bϵ((cid:98)θAggr) ⊂ Rm is the ball with radius ϵ centered at
(cid:98)θAggr. The Cϵ measures how well the probability measure π covers where the optimization goes, i.e.,
(cid:98)θAggr.

Proof. We prove this theorem by starting from a seemingly mysterious place. However, its meaning
is assured to be clear as we proceed.

14

Published as a conference paper at ICLR 2024

Denote random function (cid:98)f : Rm → R+ as

(cid:98)f (θ) = ∥(cid:98)gAggr(θ) − ∇ℓDT (θ)∥,
where the randomness comes from (cid:98)DT . Note that (cid:98)f is γ-Lipschitz by assumption. Now we consider
Bϵ((cid:98)θAggr) ⊂ Rm, i.e., the ball with radius ϵ centered at (cid:98)θAggr. Then, by γ-Lipschitzness we have

(5)

Eθ∼π (cid:98)f (θ) =

≥

(cid:90)

(cid:90)

(cid:98)f (θ) dπ(θ)

( (cid:98)f ((cid:98)θAggr) − γϵ) dπ(θ)

Bϵ((cid:98)θAggr)

= ( (cid:98)f ((cid:98)θAggr) − γϵ)π(Bϵ((cid:98)θAggr)).

(cid:98)f ((cid:98)θAggr) ≤

1
π(Bϵ((cid:98)θAggr))

· Eθ∼π (cid:98)f (θ) + O(ϵ).

Therefore,

Taking expectation w.r.t. (cid:98)DT on both sides, we have
(cid:34)

1
π(Bϵ((cid:98)θAggr))

(cid:35)
· Eθ∼π (cid:98)f (θ)

+ O(ϵ)

(cid:98)DT

(cid:98)DT (cid:98)f ((cid:98)θAggr) ≤ E
E
(cid:118)
(cid:117)
(cid:117)
(cid:116)E

≤

(cid:98)DT

(cid:35)2

(cid:34)

1
π(Bϵ((cid:98)θAggr))

(cid:104)

· E

(cid:98)DT

Eθ∼π (cid:98)f (θ)

(cid:105)2

+ O(ϵ)

(Cauchy-Schwarz)

(cid:114)

(cid:114)

(cid:113)

=

≤

=

Cϵ · E

(cid:98)DT

(cid:104)

Eθ∼π (cid:98)f (θ)

(cid:105)2

+ O(ϵ)

Cϵ · E

(cid:98)DT

Eθ∼π

(cid:104)

(cid:105)2

(cid:98)f (θ)

+ O(ϵ)

Cϵ · ∆2

Aggr + O(ϵ)

(by definition of Cϵ)

(Jensen’s inequality)

Therefore, by Markov’s inequality, with probability at least 1 − δ we have a sampled dataset (cid:98)DT such
that

1
δ
Conditioned on such event, we proceed on to the optimization part.

E
(cid:98)DT (cid:98)f ((cid:98)θAggr) ≤

(cid:98)f ((cid:98)θAggr) ≤

Cϵ · ∆2

1
δ

(cid:113)

Aggr + O(ϵ/δ)

(6)

Note that Theorem A.1 characterizes how the optimization works for one gradient update. Therefore,
for any time step t = 0, . . . , T − 1, we can apply (4) which only requires the Lipschitz assumption:

ℓDT (θt+1) − ℓDT (θt) ≤ −

µ
2

(cid:0)∥∇ℓDT (θt)∥2 − ∥(cid:98)gAggr(θt) − ∇ℓDT (θt)∥2(cid:1) .

On both sides, summing over t = 0, . . . , T − 1 gives

ℓDT (θT ) − ℓDT (θ0) ≤ −

(cid:32)T −1
(cid:88)

∥∇ℓDT (θt)∥2 −

T −1
(cid:88)

∥(cid:98)gAggr(θt) − ∇ℓDT (θt)∥2

.

(cid:33)

µ
2

t=0
Dividing both sides by T , and with regular algebraic manipulation we derive

t=0

1
T

T −1
(cid:88)

t=0

∥∇ℓDT (θt)∥2 ≤

2
µT

(ℓDT (θ0) − ℓDT (θT )) +

1
T

T −1
(cid:88)

t=0

∥(cid:98)gAggr(θt) − ∇ℓDT (θt)∥2.

Note that we assume the loss function ℓ : Θ × Z → R+ is non-negative (described at the beginning
of Section 2). Thus, we have

1
T

T −1
(cid:88)

t=0

∥∇ℓDT (θt)∥2 ≤

2ℓDT (θ0)
µT

+

1
T

T −1
(cid:88)

t=0

∥(cid:98)gAggr(θt) − ∇ℓDT (θt)∥2.

(7)

15

Published as a conference paper at ICLR 2024

Note that we assume given (cid:98)DT we have θt → (cid:98)θAggr. Therefore, for any ϵ > 0 there exist Tϵ such that

∀t > Tϵ : ∥θt − (cid:98)θAggr∥ < ϵ.

(8)

This implies that ∀t > Tϵ:

µ∥(cid:98)gAggr(θt)∥ = ∥θt+1 − (cid:98)θAggr + (cid:98)θAggr − θt∥ ≤ ∥θt+1 − (cid:98)θAggr∥ + ∥(cid:98)θAggr − θt∥ < 2ϵ.

(9)

Moreover, (8) also implies ∀t1, t2 > Tϵ:

∥∇ℓDT (θt1) − ∇ℓDT (θt2)∥ ≤ γ∥θt1 − θt2 ∥

< 2ϵ.

(γ-Lipschitzness)
(10)

The above inequality means that {∇ℓDT (θt)}t is a Cauchy sequence.
Now, let’s get back to (7). For ∀T > Tϵ we have

1
T

T −1
(cid:88)

t=0

∥∇ℓDT (θt)∥2 ≤

2ℓDT (θ0)
µT

+

1
T

Tϵ−1
(cid:88)

t=0

∥(cid:98)gAggr(θt) − ∇ℓDT (θt)∥2 +

1
T

T −1
(cid:88)

t=Tϵ

∥(cid:98)gAggr(θt) − ∇ℓDT (θt)∥2

= O

= O

≤ O

= O

= O

≤ O

(cid:19)

(cid:19)

(cid:19)

(cid:19)

(cid:19)

(cid:19)

(cid:18) 1
T

(cid:18) 1
T

(cid:18) 1
T

(cid:18) 1
T

(cid:18) 1
T

(cid:18) 1
T

= O

(cid:19)

(cid:18) 1
T

+

+

+

+

1
T

1
T

1
T

1
T

T −1
(cid:88)

t=Tϵ

T −1
(cid:88)

t=Tϵ

∥(cid:98)gAggr(θt) − ∇ℓDT (θt)∥2

∥(cid:98)gAggr(θt) − (cid:98)gAggr((cid:98)θAggr) + (cid:98)gAggr((cid:98)θAggr) − ∇ℓDT (θt)∥2

T −1
(cid:88)

(cid:16)

t=Tϵ

T −1
(cid:88)

(cid:16)

t=Tϵ

∥(cid:98)gAggr(θt) − (cid:98)gAggr((cid:98)θAggr)∥ + ∥(cid:98)gAggr((cid:98)θAggr) − ∇ℓDT (θt)∥

(cid:17)2

(triangle inequality)

O(ϵ) + ∥(cid:98)gAggr((cid:98)θAggr) − ∇ℓDT (θt)∥

(cid:17)2

(by (9))

+ O(ϵ) +

+ O(ϵ) +

1
T

1
T

T −1
(cid:88)

(cid:16)

t=Tϵ

T −1
(cid:88)

(cid:16)

t=Tϵ

∥(cid:98)gAggr((cid:98)θAggr) − ∇ℓDT ((cid:98)θAggr) + ∇ℓDT ((cid:98)θAggr) − ∇ℓDT (θt)∥

(cid:17)2

∥(cid:98)gAggr((cid:98)θAggr) − ∇ℓDT ((cid:98)θAggr)∥ + O(ϵ)

(cid:17)2

+ O(ϵ) + ∥(cid:98)gAggr((cid:98)θAggr) − ∇ℓDT ((cid:98)θAggr)∥2

(by (10))

(11)

Then, we can continue with what we have done at the beginning of the proof of this theorem:

(11) = O

≤ O

(cid:19)

(cid:19)

(cid:18) 1
T
(cid:18) 1
T

+ O(ϵ) + f ((cid:98)θAggr)2

+ O(ϵ) +

(cid:113)

(cid:18) 1
δ

Cϵ · ∆2

Aggr + O(ϵ/δ)

(cid:19)2

(by (5))

(by (6))

Therefore, combining the above we finally have: for ∀T > Tϵ with probability at least 1 − δ,

1
T

T −1
(cid:88)

t=0

∥∇ℓDT (θt)∥2 ≤ O

(cid:19)

(cid:18) 1
T

+ O(ϵ) +

(cid:18)(cid:113)

1
δ2

16

Cϵ · ∆2

Aggr + O(ϵ)

(cid:19)2

(12)

Published as a conference paper at ICLR 2024

To complete the proof, let us investigate the left hand side.

1
T

1
T

1
T

T −1
(cid:88)

t=Tϵ

T −1
(cid:88)

t=Tϵ

T −1
(cid:88)

t=Tϵ

Tϵ−1
(cid:88)

∥∇ℓDT (θt)∥2 +

1
T

T −1
(cid:88)

t=Tϵ

∥∇ℓDT (θt)∥2

1
T

T −1
(cid:88)

t=0

∥∇ℓDT (θt)∥2 =

1
T

= O

≥ O

t=0
(cid:18) 1
T

(cid:19)

+

(cid:19)

(cid:18) 1
T

+

∥∇ℓDT (θt)∥2

(cid:0)∥∇ℓDT (θt) − ∇ℓDT (θT )∥ − ∥∇ℓDT (θT )∥(cid:1)2

+

= O

= O

(cid:19)

(cid:19)

(cid:18) 1
T
(cid:18) 1
T

(cid:0)O(ϵ) + ∥∇ℓDT (θT )∥(cid:1)2

+ O(ϵ) + ∥∇ℓDT (θT )∥2.

(triangle inequality)

(by (10))

(13)

Combining (12) and (13), we finally have
(cid:19)

∥∇ℓDT (θT )∥2 ≤ O

(cid:18) 1
T

+ O(ϵ) +

(cid:18)(cid:113)

1
δ2

Cϵ · ∆2

Aggr + O(ϵ)

(cid:19)2

,

which completes the proof.

A.2 PROOF OF THEOREM 3.6

N

(cid:80)

i∈[N ] FAggr[g

Theorem A.3 (Theorem 3.6 Restated). Consider any aggregation rule Aggr(·) in the form of
(cid:98)gAggr = 1
], i.e., the aggregation rule is defined by a mapping FAggr :
Lπ × Lπ → Lπ. If FAggr is affine w.r.t. to its first argument (i.e., the target gradient function),
and ∀g ∈ Lπ : FAggr[g, g] = g, and the linear mapping associated with FAggr has its eigenvalue
bounded in [λmin, λmax], then for any source and target distributions {DSi}i∈[N ], DT , (cid:98)DT we have
∆2

, gDSi

, where

(cid:80)

(cid:98)DT

Aggr ≤ 1
N

i∈[N ] ∆2

Aggr,DSi

∆2

Aggr,DSi

≤ max{λ2

max, λ2

min} ·

σ2
π(z)
n

+ max{(1 − λmax)2, (1 − λmin)2} · dπ(DSi, DT )2

Proof. Let’s begin from the definition of the Delta error.

∆2

Aggr = E

(cid:98)DT

FAggr[g

(cid:98)DT

, gDSi

]

(cid:13)
2
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
π

, gDSi

]

(cid:17)

(cid:13)
2
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
π

gDT − FAggr[g

(cid:98)DT

i∈[N ]
(cid:13)
(cid:13)
(cid:13)gDT − FAggr[g

(cid:98)DT

E

(cid:98)DT

π

(cid:88)

1
N

∥gDT − (cid:98)gAggr∥2
(cid:13)
(cid:13)
(cid:13)
gDT −
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:88)

1
N

(cid:88)

(cid:16)

i∈[N ]

i∈[N ]

(cid:88)

i∈[N ]

∆2

Aggr,DSi

,

= E

(cid:98)DT

= E

(cid:98)DT

≤

=

1
N

1
N

, gDSi

]

(cid:13)
2
(cid:13)
(cid:13)
π

(Jensen’s Inequality)

(14)

where we denote

∆2

Aggr,DSi

:= E

(cid:98)DT

(cid:13)
(cid:13)
(cid:13)gDT − FAggr[g

(cid:98)DT

, gDSi

]

(cid:13)
2
(cid:13)
(cid:13)
π

,

17

Published as a conference paper at ICLR 2024

We continue on upper bounding this term. Noting that FAggr[·, ·] is affine in its first argument, we
can denote

¯g := E

(cid:98)DT

FAggr[g

(cid:98)DT

, gDSi

] = FAggr[E

(cid:98)DT

[g

(cid:98)DT

], gDSi

] = FAggr[gDT , gDSi

].

(15)

Therefore,

∆2

Aggr,DSi

= E

(cid:98)DT

(cid:13)
(cid:13)
(cid:13)gDT − ¯g + ¯g − FAggr[g

(cid:98)DT

, gDSi

]

(cid:13)
2
(cid:13)
(cid:13)
π

= ∥gDT − ¯g∥2

π + E

(cid:98)DT

= ∥gDT − ¯g∥2

π + E

(cid:98)DT

= ∥gDT − ¯g∥2

π + E

(cid:98)DT

= ∥gDT − ¯g∥2

π + E

(cid:98)DT

(cid:13)
(cid:13)
(cid:13)¯g − FAggr[g
(cid:13)
(cid:13)
(cid:13)¯g − FAggr[g
(cid:13)
(cid:13)
(cid:13)¯g − FAggr[g
(cid:13)
(cid:13)
(cid:13)¯g − FAggr[g

(cid:98)DT

(cid:98)DT

(cid:98)DT

(cid:98)DT

, gDSi

, gDSi

, gDSi

, gDSi

(cid:13)
2
(cid:13)
]
(cid:13)
π
(cid:13)
2
(cid:13)
]
(cid:13)
π
(cid:13)
2
(cid:13)
]
(cid:13)
π
(cid:13)
2
(cid:13)
]
(cid:13)
π

+ 2E

(cid:98)DT

⟨gDT − ¯g, ¯g − FAggr[g

(cid:98)DT

, gDSi

]⟩π

+ 2⟨gDT − ¯g, ¯g − E

(cid:98)DT

FAggr[g

(cid:98)DT

, gDSi

]⟩π

+ 2⟨gDT − ¯g, ¯g − ¯g⟩π

(by (15))

.

(16)

The above derivation is based on properties of inner product. Next, we deal with the two terms in
(16). Denote G : Lπ → Lπ as the linear mapping associated with the affine mapping FAggr[·, gDSi
].
By definition, ∀g1, g2 ∈ Lπ:

FAggr[g1, gDSi

] − FAggr[g2, gDSi

] = G[g1] − G[g2] = G[g1 − g2].

(17)

For the first term in (16), we can do a similar trick as the following.

∥gDT − ¯g∥2

− ¯g(cid:13)
2
(cid:13)
π

(cid:13)gDT − gDSi
(cid:13)gDT − gDSi
(cid:13)gDT − gDSi
(cid:13)gDT − gDSi
(cid:13)(I − G)[gDSi

π = (cid:13)
+ gDSi
= (cid:13)
+ gDSi
= (cid:13)
+ FAggr[gDSi
= (cid:13)
+ G[gDSi
= (cid:13)
− gDT ](cid:13)
2
(cid:13)
π
≤ max{(1 − λmax)2, (1 − λmin)2}∥gDSi
− gDT ∥2
π
= max{(1 − λmax)2, (1 − λmin)2}dπ(DSi, DT )2.

](cid:13)
2
− FAggr[gDT , gDSi
(cid:13)
π
] − FAggr[gDT , gDSi
, gDSi
− gDT ](cid:13)
2
(cid:13)
π

](cid:13)
2
(cid:13)
π

(by (15))

(18)

(by (17))

(I stands for identity map)

where (18) is by the identity assumption of FAggr. This assumption is valid in then sense that: if all
of the inputs to an federated aggregation rule are the same thing, the aggregation rule should output
the same thing.

This upper bounds the first term in (16), and lets move on to the second term in (16) as the following.

E

(cid:98)DT

(cid:13)
(cid:13)
(cid:13)¯g − FAggr[g

(cid:98)DT

, gDSi

]

(cid:13)
2
(cid:13)
(cid:13)
π

= E

(cid:98)DT

= E

(cid:98)DT

(cid:13)
(cid:13)
(cid:13)FAggr[gDT , gDSi
(cid:13)
(cid:13)
2
(cid:13)
(cid:13)
(cid:13)G[gDT − g
]
(cid:13)
π

(cid:98)DT

] − FAggr[g

(cid:98)DT

, gDSi

]

(cid:13)
2
(cid:13)
(cid:13)
π

(by (15))

(by (17))

≤ max{λ2

max, λ2

min}E

(cid:98)DT

(cid:13)
(cid:13)
(cid:13)gDT − g

(cid:98)DT

(cid:13)
2
(cid:13)
(cid:13)
π

= max{λ2

max, λ2

min}σ2

π( (cid:98)DT ).

(by Definition 3.5)

Combining the above, we finally have

∆2

Aggr,DSi

= (16) ≤ max{λ2

max, λ2

min}σ2

π( (cid:98)DT ) + max{(1 − λmax)2, (1 − λmin)2}dπ(DSi, DT )2.

Putting (14) and the above inequality together, and noting that σ2
proof is complete.

π( (cid:98)DT ) = σ2

π(z)
n , we can see that the

18

Published as a conference paper at ICLR 2024

A.3 PROOF OF THEOREM 4.2

Next, we prove the following theorem.
Theorem A.4 (Theorem 4.2 Restated). Consider FedDA. Given the target domain (cid:98)DT and N source
domains DS1, . . . , DSN .

∆2

FedDA ≤

1
N

N
(cid:88)

i=1

∆2

FedDA,Si

, where ∆2

FedDA,Si

= (1 − βi)2σ2

π( (cid:98)DT ) + β2

i d2

π(DSi, DT )

is the Delta error when only considering DSi as the source domain.

Proof. Recall

Thus,

(cid:98)gFedDA =

(cid:16)

N
(cid:88)

i=1

1
N

(1 − βi)g

(cid:98)DT

+ βigDSi

(cid:17)

.

∆2

FedDA = E

(cid:98)DT

∥gDT − (cid:98)gFedDA∥2

π = E

(cid:98)DT

(cid:13)
(cid:13)gDT −

(cid:16)

N
(cid:88)

i=1

1
N

(1 − βi)g

(cid:98)DT

+ βigDSi

(cid:17) (cid:13)
2
(cid:13)
π

(cid:16)

N
(cid:88)

i=1

1
N

gDT − (1 − βi)g

(cid:98)DT

+ βigDSi

(cid:17) (cid:13)
2
(cid:13)
π

E

(cid:98)DT

(cid:13)
(cid:13)gDT − (1 − βi)g

(cid:98)DT

+ βigDSi

(cid:13)
2
(cid:13)
π

= E

(cid:98)DT

(cid:13)
(cid:13)

≤

N
(cid:88)

i=1

1
N

(19)

)∥2
π

E

(cid:98)DT

(cid:13)
(cid:13)gDT − (1 − βi)g
= (1 − βi)2E

where the inequality is derived by Jensen’s inequality. Next, we prove for each i ∈ [N ]:
(cid:13)
2
π = E
+ βigDSi
(cid:13)
π + β2
∥2
⟨gDT − g
⟩π
(cid:98)DT
π(DS, DT ) + 2(1 − βi)βi⟨E
i d2
i d2
π(DSi, DT )

(cid:98)DT
∥gDT − g
+ 2(1 − βi)βiE
π( (cid:98)DT ) + β2
π( (cid:98)DT ) + β2

∥(1 − βi)(gDT − g
∥2
π
, gDT − gDSi

i ∥gDT − gDSi

[gDT − g

(cid:98)DT

(cid:98)DT

(cid:98)DT

(cid:98)DT

(cid:98)DT

(cid:98)DT

(cid:98)DT

) + βi(gDT − gDSi

], gDT − gDSi

⟩π

= (1 − βi)2σ2
= (1 − βi)2σ2
= ∆2
.

FedDA,Si

Plugging the above equation into (19) gives the theorem.

A.4 PROOF OF THEOREM 4.4

Theorem A.5 (Theorem 4.4 Rigorously). Consider FedGP. Given the target domain (cid:98)DT and N
source domains DS1, . . . , DSN .

∆2

FedGP ≤

1
N

N
(cid:88)

i=1

∆2

FedGP,Si

,

where

∆2

FedGP,Si

= (1 − βi)2σ2
i )E
+ (2βi − β2
+ 2βi(1 − βi)E
+ β2
i

E

θ, (cid:98)DT

E

π( (cid:98)DT ) + β2
i
θ, (cid:98)DT (cid:98)δi(θ)⟨g

θ, (cid:98)DT

(cid:98)DT

[(cid:98)δi(θ)τ 2

i (θ)∥gDT (θ) − gDSi

(θ)∥2]

(θ) − gDT (θ), uDSi
(θ)⟩ · ⟨g

(θ)⟩2

(θ) − gDT (θ), uDSi

(θ)⟩]

(cid:98)DT

[(cid:98)δi(θ)⟨gDT (θ), gDSi
θ, (cid:98)DT
[(1 − (cid:98)δi(θ))∥gDT (θ)∥2].

In the above equation, (cid:98)δi(θ) := 1(⟨g
the condition is satisfied. τi(θ) := ∥gDT (θ) sin ρi(θ)∥
(θ)∥.
gDT (θ). Moreover, uDSi

∥gDSi
(θ)/∥gDSi

(θ) := gDSi

(cid:98)DT

(θ)⟩ > 0) is the indicator function and it is 1 if

(θ), gDSi
(θ)−gDT (θ)∥ where ρi(θ) is the angle between gDSi

(θ) and

19

Published as a conference paper at ICLR 2024

Proof. Recall that

(cid:98)gFedGP(θ) =

(cid:16)

N
(cid:88)

i=1

1
N

where we denote

(1 − βi)g

(cid:98)DT

(θ) + βiProj+(g

(cid:98)DT

(θ)|gDSi

(θ))

(cid:17)

=

N
(cid:88)

i=1

1
N (cid:98)gFedGP,DSi

(θ),

(cid:98)gFedGP,DSi

(θ) :=

(cid:16)

(1 − βi)g

(cid:98)DT

(θ) + βiProj+(g

(cid:98)DT

(θ)|gDSi

(θ))

(cid:17)

.

Then, we can derive:
∆2

FedGP = E

(cid:98)DT

π = E

(cid:98)DT ,θ∥gDT (θ) − (cid:98)gFedGP(θ)∥2

∥gDT − (cid:98)gFedGP∥2
N
(cid:88)

= E

(cid:98)DT ,θ∥gDT (θ) −

i=1

1
N (cid:98)gFedGP,DSi

(θ)∥2

= E

(cid:98)DT ,θ∥

N
(cid:88)

i=1

1
N

(gDT (θ) − (cid:98)gFedGP,DSi

(θ))∥2

≤

N
(cid:88)

i=1

1
N

E
(cid:98)DT ,θ∥gDT (θ) − (cid:98)gFedGP,DSi

(θ)∥2

(20)

(θ)∥2 = ∆2

FedGP,Si

. First, we simplify the notation.

(cid:98)DT

(θ), gDSi
(θ), uDSi

⟩, 0}gDSi (θ)/∥gDSi
(θ).
(θ)⟩uDSi

(θ)∥2

where the inequality is derived by Jensen’s inequality.
Next, we show E
Recall the definition of (cid:98)gFedGP,DSi
(θ) = (1 − βi)g

(cid:98)DT ,θ∥gDT (θ) − (cid:98)gFedGP,DSi

(θ) + βi max{⟨g

is that

(cid:98)gFedGP,DSi

(cid:98)DT

= (1 − βi)g

(cid:98)DT

(θ) + βi(cid:98)δi(θ)⟨g

(cid:98)DT

Let us fix θ and then further simplify the notation by denoting

(θ)

ˆv := g

(cid:98)DT
v := gDT (θ)
u := uDSi
(θ)
ˆδ := ˆδi(θ)

Therefore, we have (cid:98)gFedGP,DSi
Therefore, with the simplified notation,

(θ) = (1 − βi)ˆv + βi

ˆδ⟨ˆv, u⟩u.

E

(cid:98)DT

∥gDT (θ) − (cid:98)gFedGP,DSi

(θ)∥2 = E

∥(1 − βi)ˆv + βi
(cid:98)DT
∥βi(ˆδ⟨ˆv, u⟩u − v) + (1 − βi)(ˆv − v)∥2

ˆδ⟨ˆv, u⟩u − v∥2

= E

(cid:98)DT

= E

= β2
i

(cid:98)DT
E

∥βi

ˆδ⟨ˆv − v, u⟩u + βi(ˆδ⟨v, u⟩u − v) + (1 − βi)(ˆv − v)∥2
[∥ˆδ⟨v, u⟩u − v∥2] + (1 − βi)2E
[ˆδ⟨ˆv − v, u⟩2] + β2
i

E

(cid:98)DT

(cid:98)DT

[∥ˆv − v∥2]

(cid:98)DT
[ˆδ⟨v, u⟩⟨u, ˆv − v⟩]

[ˆδ⟨ˆv − v, u⟩2] + 2βi(1 − βi)E

(cid:98)DT

(cid:98)DT

+ 2βi(1 − βi)E
− 2β2
i
= (2βi − β2

E[ˆδ(1 − ˆδ)⟨ˆv − v, u⟩⟨v, u⟩]
[ˆδ⟨ˆv − v, u⟩2] + β2
E
i )E
i
(cid:124)

(cid:98)DT

+ 2βi(1 − βi)E

(cid:98)DT

[ˆδ⟨v, u⟩⟨u, ˆv − v⟩] − 2β2
i

(expanding the squared norm)
[∥ˆv − v∥2]

+(1 − βi)2E

(cid:98)DT

(cid:98)DT

[∥ˆδ⟨v, u⟩u − v∥2]
(cid:125)

(cid:123)(cid:122)
A
E[ˆδ(1 − ˆδ)⟨ˆv − v, u⟩⟨v, u⟩]
,
(cid:124)
(cid:123)(cid:122)
(cid:125)
B

where the last equality is by merging the similar terms. Next, we deal with the terms A and B.
Let us start from the term B. Noting that ˆδ is either 0 or 1, we can see that ˆδ(1 − ˆδ) = 0. Therefore,
B = 0.

20

Published as a conference paper at ICLR 2024

As for the term A, noting that ˆδ2 = ˆδ, expanding the squared term we have:

A = E

(cid:98)DT

= E

(cid:98)DT

= E

(cid:98)DT

[∥ˆδ⟨v, u⟩u − v∥2] = E
[ˆδ∥⟨v, u⟩u − v∥2] + E
[ˆδ∥⟨v, u⟩u − v∥2] + E

(cid:98)DT

(cid:98)DT

(cid:98)DT

[∥ˆδ(⟨v, u⟩u − v) − (1 − ˆδ)v∥2]
[(1 − ˆδ)∥v∥2] − 2E
[(1 − ˆδ)∥v∥2].

(cid:98)DT

[ˆδ(1 − ˆδ)⟨⟨v, u⟩u − v, v⟩]

Therefore, combining the above, we have

E

(cid:98)DT

∥gDT (θ) − (cid:98)gFedGP,DSi

(θ)∥2 = (1 − βi)2E
(cid:98)DT
i )E
+ (2βi − β2
+ 2βi(1 − βi)E

[∥ˆv − v∥2] + β2
i
[ˆδ⟨ˆv − v, u⟩2]
[ˆδ⟨v, u⟩⟨u, ˆv − v⟩] + β2
i

(cid:98)DT

(cid:98)DT

E

(cid:98)DT

[ˆδ∥⟨v, u⟩u − v∥2]

E

(cid:98)DT

[(1 − ˆδ)∥v∥2].

Let us give an alternative form for ∥⟨v, u⟩u − v∥2, as we aim to connect this term to ∥u − v∥ which
would become the domain-shift. Note that ∥⟨v, u⟩u − v∥ is the distance between v and its projection
to u. We can see that ∥⟨v, u⟩u − v∥ = ∥u − v∥ ∥⟨v,u⟩u−v∥
∥u−v∥ = ∥u − v∥τi(θ). Therefore, plugging this
in, we have

E

(cid:98)DT

∥gDT (θ) − (cid:98)gFedGP,DSi

(θ)∥2 = (1 − βi)2E
(cid:98)DT
i )E
+ (2βi − β2
+ 2βi(1 − βi)E

[∥ˆv − v∥2] + β2
i
[ˆδ⟨ˆv − v, u⟩2]
[ˆδ⟨v, u⟩⟨u, ˆv − v⟩] + β2
i

(cid:98)DT

(cid:98)DT

E

(cid:98)DT

[ˆδ∥u − v∥2τ 2

i (θ)]

E

(cid:98)DT

[(1 − ˆδ)∥v∥2]

Writing the abbreviations u, v, ˆv, ˆδ into their original forms and taking expectation over θ on the both
side we can derive:

E
(cid:98)DT ,θ∥gDT (θ) − (cid:98)gFedGP,DSi
i )E
+ (2βi − β2
+ 2βi(1 − βi)E
+ β2
i

E

= ∆2

FedGP,Si

θ, (cid:98)DT
.

(θ)∥2 = (1 − βi)2σ2

π( (cid:98)DT ) + β2
i

E

[(cid:98)δi(θ)τ 2

i (θ)∥gDT (θ) − gDSi

(θ)∥2]

θ, (cid:98)DT
(θ)⟩2

θ, (cid:98)DT (cid:98)δi(θ)⟨g

(cid:98)DT

(θ) − gDT (θ), uDSi
(θ)⟩ · ⟨g

[(cid:98)δi(θ)⟨gDT (θ), gDSi
θ, (cid:98)DT
[(1 − (cid:98)δi(θ))∥gDT (θ)∥2]

(θ) − gDT (θ), uDSi

(θ)⟩]

(cid:98)DT

Combining the above equation with (20) concludes the proof.

Approximations. As we can see, the Delta error of FedGP is rather complicated at its precise form.
However, reasonable approximation can be done to extract the useful components from it which
would help us to derive its auto-weighted version. In the following, we show how we derive the
approximated Delta error for FedGP, leading to what we present in (2).

First, we consider an approximation which is analogous to a mean-field approximation, i.e., ignoring
the cross-terms in the expectation of a product. Fixing (cid:98)δi(θ) = ¯δ and τi(θ) = ¯τ , i.e., their expectations.
This approximation is equivalent to assuming (cid:98)δ(θ), τ (θ) can be viewed as independent random
variables. This results in the following.

∆2

FedGP,DSi

≈ (1 − βi)2σ2
+ (2βi − β2

π( (cid:98)DT ) + β2
i
i )¯δE

⟨g

θ, (cid:98)DT

¯δ¯τ 2dπ(DSi, DT )2
(θ) − gDT (θ), uDSi
(cid:98)DT
(θ)⟩2 is the variance of g

(cid:98)DT

(cid:98)DT

⟨g

(θ) − gDT (θ), uDSi

The term E
uDSi
(θ).
We consider a further approximation based on the following intuition: consider a zero-mean random
vector ˆv ∈ Rm with i.i.d. entries ˆvj for j ∈ [m]. After projecting the random vector to a fixed
unit vector u ∈ Rm, the projected variance is E⟨ˆv, u⟩2 = E((cid:80)m
j =

(θ) when projected to a direction

j=1 ˆvjuj)2 = (cid:80)m

j ]u2

E[ˆv2

j=1

(cid:98)DT

(θ)⟩2 + β2

i (1 − ¯δ)∥gDT ∥2
π.

21

Published as a conference paper at ICLR 2024

(cid:80)m

j=1

σ2(ˆv)
m u2

j = σ2(ˆv)/m, i.e., the variance becomes much smaller. Therefore, knowing that the
(θ) − gDT (θ) is

parameter space Θ is m-dimensional, combined with the approximation that g
element-wise i.i.d., we derive a simpler (approximate) result.

(cid:98)DT

∆2

FedGP,DSi

(cid:16)

≈

(1 − βi)2 + (2βi−β2

i )¯δ

m

(cid:17)

π( (cid:98)DT ) + β2
σ2
i

¯δ¯τ 2dπ(DSi, DT )2 + β2

i (1 − ¯δ)∥gDT ∥2
π.

In fact, this approximated form can already be used for deriving the auto-weighted version for FedGP,
as it is quadratic in βi and all of the terms can be estimated. However, we find that in practice ¯δ ≈ 1,
and thus simply setting ¯δ = 1 makes little impact on ∆2
. Therefore, for simplicity, we
choose ¯δ = 1 as an approximation, which results in

FedGP,DSi

∆2

FedGP,DSi

(cid:16)

≈

(1 − βi)2 + (2βi−β2
i )

m

(cid:17)

π( (cid:98)DT ) + β2
σ2

i ¯τ 2dπ(DSi, DT )2.

This gives the result shown in (2).

Although many approximations are made, we observe in our experiments that the auto-weighted
scheme derived upon this is good enough to improve FedGP.

A.5 ADDITIONAL DISCUSSION OF THE AUTO-WEIGHTING METHOD AND FE DGP

In this sub-section, we show how we estimate the optimal βi for both FedDA and FedGP. Moreover,
we discuss the intuition behind why FedGP with a fixed β = 0.5 is fairly good in many cases.

In order to compute the β for each methods, as shown in Section 4.2, we need to estimate the
following three quantities: σ2
π(DSi, DT ), and ¯τ 2d2
π(DSi, DT ). In the following, we derive
unbiased estimators for each of the quantities, followed by a discussion of the choice of π.

π( (cid:98)DT ), d2

The essential technique is to divide the target domain dataset (cid:98)DT into many pieces, serving as samples.
Concretely, say we randomly divide (cid:98)DT into B parts of equal size, denoting (cid:98)DT = ∪B
T (without
loss of generality we may assume | (cid:98)DT | can be divided by B). This means

j=1 (cid:98)Dj

g
(cid:98)DT

=

1
B

B
(cid:88)

j=1

g
(cid:98)Dj
T

.

(21)

(cid:98)Dj
T

Note that each g

is a sample of dataset formed by | (cid:98)DT |/B data points. We denote (cid:98)DT,B as the
corresponding random variable (i.e., a dataset of | (cid:98)DT |/B sample points sampled i.i.d. from DT ).
Since we assume each data points in (cid:98)DT is sampled i.i.d. from DT , we have

E[g

(cid:98)DT ,B

] = E[g

(cid:98)Dj
T

] = E[g

(cid:98)DT

] = gDT .

(22)

Therefore, we may view (cid:98)Dj
Estimator of σ2

π( (cid:98)DT ).

T as i.i.d. samples of (cid:98)DT,B, which we can used for our estimation.

First, for σ2

π( (cid:98)DT ), we can derive that

π( (cid:98)DT ) := E∥gDT − g
σ2

(cid:98)DT

π = E∥gDT −
∥2

1
B

B
(cid:88)

j=1

g
(cid:98)Dj
T

∥2
π

=

=

=

1
B2

1
B2

E∥

B
(cid:88)

j=1

(gDT − g

)∥2
π

(cid:98)Dj
T

B
(cid:88)

j=1

E∥gDT − g

(cid:98)Dj
T

∥2
π

1
B

σ2
π( (cid:98)DT,B)

22

(by (22))

(by (22))

Published as a conference paper at ICLR 2024

Since we have B i.i.d. samples of (cid:98)DT,B, we can use their sample variance, denoted as (cid:98)σ2
an unbiased estimator for its variance σ2

π( (cid:98)DT,B). Concretely, the sample variance is

π( (cid:98)DT,B), as

(cid:98)σ2
π( (cid:98)DT,B) =

1
B − 1

B
(cid:88)

j=1

∥g

(cid:98)Dj
T

− g

(cid:98)DT

∥2
π.

Note that, as shown in (21), g
is an unbiased estimator of the variance, i.e.,

(cid:98)DT

is the sample mean. It is a known statistical fact that sample variance

E[(cid:98)σ2
Combining the above, our estimator for σ2

π( (cid:98)DT ) is

π( (cid:98)DT,B)] = σ2

π( (cid:98)DT,B).

(cid:98)σ2
π( (cid:98)DT ) =

1
B (cid:98)σ2

π( (cid:98)DT,B) =

1
(B − 1)B

B
(cid:88)

j=1

∥g

(cid:98)Dj
T

− g

(cid:98)DT

∥2
π.

Estimator of d2
For d2

π(DSi, DT ).

π(DSi, DT ), we adopt a similar approach. We first apply the following trick:

d2
π(DSi, DT ) = ∥gDSi
= E∥gDSi
= E∥gDSi

− gDT ∥2

π = ∥gDSi
− gDT + gDT − g
π − σ2
∥2

− g

(cid:98)DT ,B

(cid:98)DT ,B
π( (cid:98)DT,B),

π + E∥gDT − g
− gDT ∥2
π − E∥gDT − g
∥2

(cid:98)DT ,B
∥2
π

(cid:98)DT ,B

π − E∥gDT − g
∥2

(cid:98)DT ,B

∥2
π

(23)

(24)

where (23) is due to that E[g

(cid:98)DT ,B

] = gDT and therefore the inner product term

E[⟨gDSi

− gDT , gDT − g

(cid:98)DT ,B

⟩π] = ⟨gDSi

− gDT , E[gDT − g

(cid:98)DT ,B

]⟩π = 0.

This means

E∥gDSi

− gDT + gDT − g

(cid:98)DT ,B

∥2
π = ∥gDSi

− gDT ∥2

π + E∥gDT − g

(cid:98)DT ,B

∥2
π.

We can see that (24) has an unbiased estimator as the following

(cid:98)d2
π(DSi, DT ) =





1
B

B
(cid:88)

j=1

∥gDSi

− g

(cid:98)Dj
T

∥2
π


 − (cid:98)σ2

π( (cid:98)DT,B)

Estimator of ¯τ 2d2
π(DSi , DT ).
Finally, it left to find an estimator for ¯τ 2d2
directly, but not ¯τ 2 and d2
Concretely, from Theorem 4.4 we can see its original form (with δi = 1) is
(θ)∥2] = Eθ∼π∥gDT (θ) − ⟨gDT (θ), uDSi
(θ)∥.

i (θ)∥gDT (θ) − gDSi
(θ)/∥gDSi
(θ) = gDSi

Eθ∼π[τ 2

π(DSi, DT )
π(DSi, DT ) separately, is because our aim in finding an unbiased estimator.

π(DSi, DT ). Note that we estimate ¯τ 2d2

(θ)⟩uDSi

(θ)∥2,

(25)

where uDSi
Denote gT S, (cid:98)gT S, (cid:98)gj

T S : Θ → Θ as the following:

gT S(θ) := gDT (θ) − ⟨gDT (θ), uDSi
(cid:98)gT S(θ) := g
(cid:98)gj
T S(θ) := g
Therefore, we have the following two equations:

(cid:98)DT ,B
(θ), uDSi

(θ) − ⟨g

(θ) − ⟨g

(cid:98)DT ,B

(cid:98)Dj
T

(cid:98)Dj
T

(θ), uDSi

(θ)⟩uDSi

(θ)
(θ)⟩uDSi
(θ).

(θ)⟩uDSi

(θ)

(25) = ∥gT S∥2
π

E[(cid:98)gT S] = E[(cid:98)gj

T S] = gT S.

23

Published as a conference paper at ICLR 2024

This means we can use our samples (cid:98)Dj
ingly.

T to compute samples of (cid:98)gj

T S, and then estimate (25) accord-

Applying the same trick as before, we have
π = ∥gT S∥2

(25) = ∥gT S∥2

π + E∥(cid:98)gT S − gT S∥2

π − E∥(cid:98)gT S − gT S∥2

π

π − E∥(cid:98)gT S − gT S∥2

π

= E∥gT S + (cid:98)gT S − gT S∥2
= E∥(cid:98)gT S∥2

π − E∥(cid:98)gT S − gT S∥2
π.
π can be estimated unbiasedly by 1
B

Note that E∥(cid:98)gT S∥2
the variance of (cid:98)gT S and thus can be estimated unbiasedly by the sample variance.
Putting all together, the estimator of ¯τ 2d2
(25)) is

j=1 ∥(cid:98)gj

T S∥2

(cid:80)B

π(DSi, DT ) (rigorously speaking, the unbiased estimator of

π. Moreover, E∥(cid:98)gT S − gT S∥2

π is





1
B

B
(cid:88)

j=1

∥(cid:98)gj

T S∥2

π





 −



1
B − 1

B
(cid:88)

j=1

∥(cid:98)gj

T S −



T S∥2
(cid:98)gk
π



1
B

B
(cid:88)

k=1

Therefore, we have the unbiased estimators of σ2
we can computed estimated optimal βFedDA

π( (cid:98)DT ), d2
and βFedGP
i

π(DSi, DT ), and ¯τ 2d2
according to section 4.2.

i

π(DSi, DT ). Then,

The choice of π.
The distribution π characterizes where in the parameter space Θ we want to measure σ2
d2
π(DSi, DT ), and the Delta errors.
In practice, we have different ways to choose π. For example, in our synthetic experiment, we simply
choose π to be the point mass of the initialization model parameter. It turns out the Delta errors
computed at initialization are pretty accurate in predicting the final test results, as shown in Figure 5.
For the more realistic cases, we choose π to be the empirical distribution of parameters along the
optimization path. This means that we can simply take the local updates, computed by batches of
data, as (cid:98)Dj
T and estimate β accordingly. Detailed implementation is shown in Section C.4.
Intuition of why FedGP is more robust to the choice of β.

π( (cid:98)DT ),

To have an intuition about why FedGP is more robust to the choice of β compared FedDA, we
examine how varying β affects their Delta errors.

Recall that

∆2

FedDA,Si

∆2

FedGP,Si

= (1 − βi)2σ2
≈ (1 − βi)2σ2

π( (cid:98)DT ) + β2
π( (cid:98)DT ) + β2

i d2
i ¯τ 2d2

π(DSi , DT )

π(DSi, DT ).

Now suppose we change βi → β′
less than the change in ∆2
FedDA.

FedDA,Si

i, and since 0 ≤ ¯τ 2 ≤ 1, we can see that the change in ∆2
is
. In other words, FedGP should be more robust to varying β than

FedGP,Si

B SYNTHETIC DATA EXPERIMENTS IN DETAILS

The synthetic data experiment aims to bridge the gap between theory and practice by verifying our
theoretical insights. Specifically, we generate various source and target datasets and compute the
corresponding source-target domain distance dπ(DS, DT ) and target domain variance σπ( (cid:98)DT ). We
aim to verify if our theory is predictive of practice.

In this experiment, we use one-hidden-layer neural networks with sigmoid activation. We generate 9
datasets D1, . . . , D9 each consisting of 5000 data points as the following. We first generate 5000
samples x ∈ R50 from a mixture of 10 Gaussians. The ground truth target is set to be the sum of
100 radial basis functions, the target has 10 samples. We control the randomness and deviation of
the basis function to generate datasets with domain shift. As a result, D2 to D9 have an increasing
domain shift compared to D1. We take D1 as the target domain. We subsample (uniformly) 9 datasets

24

Published as a conference paper at ICLR 2024

(a)

(b)

(c)

Figure 5: Given specific source-target domain distance and target domain variance: (a) shows which
aggregation method has the smallest Delta error; (b)&(c) present which aggregation method actually
achieves the best test result. In (a)&(b), FedDA and FedGP use a fixed β = 0.5. In (c), FedDA
and FedGP adopt the auto-weighted scheme. Observations: Comparing (a) and (b), we can see
that the prediction from the Delta errors, computed at initialization, mostly track the actual test
performance after training. Comparing (b) and (c), we can see that FedDA is greatly improved
with the auto-weighted scheme. Moreover, we can see that FedGP with a fixed β = 0.5 is good
enough for most of the cases. These observations demonstrate the practical utility of our theoretical
framework.

2/(2σi)2

1, . . . , (cid:98)D9

1 has the smallest target

1 from D1 with decreasing number of subsamples. As a result, (cid:98)D1

(cid:98)D1
domain variance, and (cid:98)D9
1 has the largest.
Dataset. Denote a radial basis function as ϕi(x) = e−∥x−µi∥2
, and we set the target ground
truth to be the sum of M = 100 basis functions as y = (cid:80)M
i=1 ϕi, where each entry of the parameters
are sampled once from U (−0.5, 0.5). We set the dimension of x to be 50, and the dimension
of the target to be 10. We generate N = 5000 samples of x from a Gaussian mixture formed
by 10 Gaussians with different centers but the same covariance matrix Σ = I. The centers are
sampled randomly from U (−0.5, 0.5)n. We use the ground truth target function y(·) to derive the
corresponding data y for each x. That is, we want our neural networks to approximate y(·) on the
Gaussian mixture.
Methods. For each pair of (Di, (cid:98)Dj
1) where i, j = 1, . . . , 9 from the 81 pairs of datasets. We compute
the source-target domain distance dπ(Di, (cid:98)Dj
1) with π being
the point mass on only the initialization parameter. We then train the 2-layer neural network with
different aggregation strategies on (Di, (cid:98)Dj
1). Given the pair of datasets, we identify which strategies
have the smallest Delta error and the best test performance on the target domain. We report the
average results of three random trials.

1) and target domain variance σπ( (cid:98)Dj

As interpreted in Figure 5, the results verify that (i) the Delta error indicates the actual test result, and
(ii) the auto-weighted strategy, which minimizes the estimated Delta error, is effective in practice.

C SUPPLEMENTARY EXPERIMENT INFORMATION

In this section, we provide the algorithms, computational and communication cost analysis, additional
experiment details, and additional results on semi-synthetic and real-world datasets.

C.1 DETAILED ALGORITHM OUTLINES FOR FEDERATED DOMAIN ADAPTATION

As illustrated in Algorithm 1, for one communication round, each source client CSi performs
supervised training on its data distribution DSi and uploads the weights to the server. Then the server
computes and shuffles the source gradients, sending them to the target client CT . On CT , it updates its
parameter using the available target data. After that, the target client updates the global model using
aggregation rules (e.g. FedDA, FedGP, and their auto-weighted versions) and sends the model to the
server. The server then broadcasts the new weight to all source clients, which completes one round.

25

0.000.020.040.060.08Source-Target Domain Distance0.000.020.040.060.08Target Domain VarianceMethod with the Smallest Delta ErrorFedGPFedDATarget OnlySource Only0.000.020.040.060.08Source-Target Domain Distance0.000.020.040.060.08Target Domain VarianceMethod with the Best Test ResultFedGPFedDATarget OnlySource Only0.000.020.040.060.08Source-Target Domain Distance0.000.020.040.060.08Target Domain VarianceMethod with the Best Test ResultFedGP (auto)FedDA (auto)Target OnlySource OnlyPublished as a conference paper at ICLR 2024

C.2 REAL-WORLD EXPERIMENT IMPLEMENTATION DETAILS AND RESULTS

Implementation details We conduct experiments on three datasets: Colored-MNIST (Arjovsky
et al., 2019) (a dataset derived from MNIST (Deng, 2012) but with spurious features of colors),
VLCS (Fang et al., 2013) (four datasets with five categories of bird, car, chair, dog, and person),
and TerraIncognita (Beery et al., 2018) (consists of 57,868 images across 20 locations, each labeled
with one of 15 classes) datasets. The source learning rate is 1e−3 for Colored-MNIST and 5e−5
for VLCS and TerraIncognita datasets. The target learning rate is set to 1
5 of the source learning
rate. For source domains, we split the training/testing data with a 20% and 80% split. For the target
domain, we use a fraction (0.1% for Colored-MNIST, 5% for VLCS, 5% for TerraIncognita) of the
80% split of training data to compute the target gradient. We report the average of the last 5 epochs
of the target accuracy on the test split of the data across 5 trials. For Colored-MNIST, we use a CNN
model with four convolutional and batch-norm layers. For the other two datasets, we use pre-trained
ResNet-18 (He et al., 2016) models for training. Apart from that, we use the cross-entropy loss as the
criterion and apply the Adam (Kingma & Ba, 2014) optimizer. For initialization, we train 2 epochs
for the Colored-MNIST, VLCS datasets, as well as 5 epochs for the TerraIncognita dataset. We run
the experiment with 5 random seeds and report the average accuracies over five trials with variances.
Also, we set the total round R = 50 and the local update epoch to 1.

Personalized FL benchmark We adapt the code from Marfoq et al. (2022) to test the performances
of different personalization baselines. We report the personalization performance on the target domain
with limited data. We train for 150 epochs for each personalization method with the same learning
rate as our proposed methods. For APFL, the mixing parameter is set to 0.5. Ditto’s penalization
parameter is set to 1. For knnper, the number of neighbors is set to 10. We reported the highest
accuracy across grids of weights and capacities by evaluating pre-trained FedAvg for knnper.
Besides, we found out the full training of DomainNet is too time-consuming for personalized
baselines.

Full results of all real-world datasets with error bars As shown in Table 2 (Colored-MNIST),
Table 3 (VLCS), and Table 4 (TerraIncognita), auto-weighted methods mostly achieve the best
performance compared with personalized benchmarks and other FDA baselines, across various
target domains and on average. Also, FedGP with a fixed weight β = 0.5 has a comparable
performance compared with the auto-weighted versions. FedDA with auto weights greatly improves
the performance of the fixed-weight version.

Domains
Source Only
Finetune_Offline
FedDA
FedGP
FedDA_Auto
FedGP_Auto
Target Only
Oracle

+90%
56.82 (0.80)
66.58 (4.93)
60.49 (2.54)
83.68 (9.94)
85.29 (5.71)
86.18 (4.54)
85.60 (4.78)
89.94 (0.38)

Colored-MNIST

+80%
62.37 (1.75)
69.09 (2.62)
65.07 (1.26)
74.41 (4.40)
73.13 (7.32)
76.49 (7.29)
73.54 (2.98)
80.32 (0.44)

-90%
27.77 (0.82)
53.86 (6.89)
33.04 (3.15)
89.76 (0.48)
88.83(1.06)
89.62(0.43)
87.05 (3.41)
89.99 (0.54)

Avg
48.99
63.18
52.87
82.42
82.72
84.10
82.06
86.75

Table 2: Target domain test accuracy (%) on Colored-MNIST with varying target domains.

C.3 ADDITIONAL RESULTS ON DOMAINBED DATASETS

In this sub-section, we show (1) the performances of our methods compared with the SOTA unsuper-
vised FDA (UFDA) methods on DomainNet and (2) additional results on PACS (Li et al., 2017) and
Office-Home (Venkateswara et al., 2017) datasets.

Implementation We randomly sampled 15% target domain samples of PACS, Office-Home, and
DomainNet for our methods, while FADA and KD3A use 100% unlabeled data on the target domain.
To have a fair comparison, we use the same model architecture ResNet-50 for all methods across
three datasets. Additionally, for DomainNet, we train FedDA, FedGP and auto-weighted methods

26

Published as a conference paper at ICLR 2024

Domains
Source Only
Finetune_Offline
FedDA
FedGP
FedDA_Auto
FedGP_Auto
Target Only
Oracle

C
90.49 (5.34)
96.65 (3.68)
97.72 (0.46)
99.43 (0.30)
99.78 (0.19)
99.93 (0.16)
97.77 (1.45)
100.00 (0.00)

L
60.65 (1.83)
68.22 (2.26)
68.17 (1.42)
71.09 (1.24)
73.08 (1.31)
73.22 (1.81)
68.88 (1.86)
72.72 (2.53)

VLCS

V
70.24 (1.97)
74.34 (0.83)
75.27 (1.45)
73.65 (3.10)
78.41 (1.51)
78.62 (1.54)
72.29 (1.73)
78.65 (1.38)

S
69.10 (1.99)
74.66 (2.58)
76.68 (0.91)
78.70 (1.31)
83.73 (2.27)
83.47 (2.46)
76.00 (1.89)
82.71 (1.07)

Avg
72.62
78.47
79.46
80.72
83.75
83.81
78.74
83.52

Table 3: Target domain test accuracy (%) on VLCS with varying target domains.

Source Only
Finetune_Offline
FedDA
FedGP
FedDA_Auto
FedGP_Auto
Target Only
FedAvg
Ditto (Li et al., 2021)
FedRep (Collins et al., 2021)
APFL (Deng et al., 2020)
KNN-per (Marfoq et al., 2022)
Oracle

L100
54.62 (4.45)
77.45 (3.88)
77.24 (2.22)
81.46 (1.28)
82.35 (1.91)
81.85 (2.30)
78.85 (1.86)
38.46
44.30
45.27
65.16
42.20
96.41 (0.18)

L38
31.39 (3.13)
75.22 (5.46)
69.21 (1.83)
77.75 (1.55)
80.77 (1.43)
80.50 (1.60)
74.25 (2.52)
21.24
12.57
6.15
64.42
39.86
95.01 (0.28)

TerraIncognita
L43
36.85 (2.80)
61.16 (4.07)
58.55 (3.37)
64.18 (2.86)
68.42 (2.66)
68.43 (2.08)
58.96 (3.50)
39.76
40.16
21.13
38.97
50.00
91.98 (1.17)

L46
27.15 (1.21)
58.89 (7.95)
53.78 (1.74)
61.56 (1.94)
66.87 (2.03)
66.64 (1.52)
56.90 (3.09)
20.54
17.98
10.89
42.33
40.21
89.04 (0.93)

Avg
37.50
68.18
64.70
71.24
74.60
74.36
67.24
30.00
28.75
20.86
52.72
43.07
93.11

Table 4: Target domain test accuracy (%) on TerraIncognita with varying target domains.

with 20 global epochs and 1 local epochs; for the other two datasets, we train our methods for 50
epochs. For the auto-weighted methods, we notice that it converges faster than their fixed-weight
counterpart. Therefore, we set the learning rate ratio to be 0.25 to prevent overfitting for DomainNet
and PACS. For Office-Home, we set the rate ratio to be 0.1 on A domain and 0.5 on other domains.
For DomainNet, the target batch size is set to 64 for auto-weighted methods and 16 for FedDA and
FedGP; the source batch size is set to 128. For PACS, the target batch size is 4 for auto-weighted
methods and 16 for FedDA and FedGP; the source batch size is 32. For Office-Home, the target
batch size is 64 for auto-weighted methods and 16 for FedDA and FedGP; the source batch size is
32. Also, we initialize 2 epochs for DomainNet, PACS, and domain A of Office-Home; for other
domains in Office-Home, we perform the initialization for 5 epochs. Also, we use a learning rate of
5e−5 for source clients. The learning rate of the target client is set to 1
5 of the source learning rate.

Comparison with UFDA methods We note that our methods work differently than UFDA methods.
I.e., we consider the case where the target client possesses only limited labeled data, while the UFDA
case assumes the existence of abundant unlabeled data on the target domain. As shown in Table 5,
FedGP outperforms UFDA baselines using ResNet-101 in most domains and on average with a
significant margin. Especially for quick domain, when the source-target domain shift is large (Source
Only has a much lower performance compared with Oracle), our methods significantly improve the
accuracy, which suggests our methods can achieve a better bias-variance trade-off under the condition
when the shift is large. However, we observe that the estimated betas from the auto-weighted scheme
seem to become less accurate and more expensive to compute with a larger dataset, which potentially
leads to its slightly worsened performance compared with FedGP. In future work, we will keep
exploring how to speed up and better estimate the bias and variances terms for larger datasets.

PACS results As shown in Table 6, we compare our methods with the SOTA Domain Generalization
(DG) methods on the PACS dataset. The results show that our FedDA, FedGP and their auto-weighted
versions are able to outperform the DG method with significant margins.

27

Published as a conference paper at ICLR 2024

Domains
Source Only
FADA (100%)
KD3A (100%) (ResNet-50)
FedDA (ResNet-50)
FedGP (ResNet-50)
FedDA_auto (ResNet-50)
FedGP_auto (ResNet-50)
Best DG
Oracle

clip
52.1
59.1
63.7
67.1
64.0
62.0
62.2
59.2
69.3

info
23.1
21.7
15.4
26.7
26.6
27.8
27.7
19.9
34.5

paint
47.7
47.9
53.5
56.1
56.8
56.7
56.8
47.4
66.3

quick
13.3
8.8
11.5
33.8
51.1
50.9
50.7
14.0
66.8

real
60.7
60.8
65.4
67.1
71.3
68.1
68.4
59.8
80.1

sketch Avg
40.6
46.5
41.5
50.4
43.8
53.4
51.1
55.7
53.7
52.3
53.1
53.3
53.5
53.2
41.8
50.4
63.0
60.7

Table 5: Target domain test accuracy (%) on DomainNet. FedGP and auto-weighted methods
generally outperform UFDA methods with significant margins by using 15% of the data.

Domains
FedDA
FedGP
FedDA_Auto
FedGP_Auto
Best DG

A
92.6
94.4
94.2
94.2
87.8

C
89.1
92.2
90.9
93.7
81.8

P
97.4
97.6
96.6
97.3
97.4

S
89.2
88.9
89.6
88.3
82.1

Avg
92.0
93.3
92.8
93.4
87.2

Table 6: Target domain test accuracy (%) on PACS by using 15% data samples. We see FedGP
and auto-weighted methods outperformed DG methods with significant margins.

Office-Home results As shown in Table 7, we compare our methods with the SOTA DG methods on
the Office-Home dataset. The results show that our FedDA, FedGP and their auto-weighted versions
are able to outperform the DG method with significant margins. We found that the auto-weighting
versions of FedDA and FedGP generally outperform their fixed-weight counterparts. We notice
that FedDA where the fixed weight choice of β = 0.5 is surprisingly good on Office-Home. We
observe that on Office-Home, the source-only baseline sometimes surpasses the Oracle performance
on the target domain. Thus, we conjecture that the fixed weight β = 0.5 happens to be a good choice
for FedDA on Office-Home, while the noisy target domain data interferes with the auto-weighting
mechanism. Nevertheless, the auto-weighted FedGP still shows improvement over its fixed-weight
version.

Domains
Source Only
FedDA
FedGP
FedDA_Auto
FedGP_Auto
Oracle
Best DG

A
50.9
67.9
63.8
67.4
66.1
70.9
64.5

C
66.1
68.2
65.7
65.6
64.5
58.5
54.8

P
74.5
82.6
81.0
82.2
82.1
87.4
76.6

R
76.2
78.6
74.4
75.8
74.9
75.0
78.1

Avg
66.9
74.3
71.2
72.7
71.9
73.0
68.5

Table 7: Target domain test accuracy (%) on Office-Home by using 15% data samples. We see when
source-target shifts are small, FedDA works surprisingly well; FedGP_Auto manages to improve
FedGP.

C.4 AUTO-WEIGHTED METHODS: IMPLEMENTATION DETAILS, TIME AND SPACE

COMPLEXITY

Implementation As outlined in Algorithm 1, we compute the source gradients and target gradients
locally when performing local updates. For the target client, during local training, it optimizes its
model with B batches of data, and hence it computes B gradients (effectively local updates) during
this round. After this round of local optimization, the target client receives the source gradients
(source local updates) from the server. Note that the source clients do not need to remember nor send
its updates for every batch (as illustrated in Algorithm 1), but simply one model update per source

28

Published as a conference paper at ICLR 2024

client and we can compute the average model update divided by the number of batches. Also, because
of the different learning rates for the source and target domains, we align the magnitudes of the
gradients using the learning rate ratio. Then, on the target client, it computes {dπ(DSi, DT )}N
i=1 and
σπ( (cid:98)DT ) using {gSi}N
values for all source domains per round, as shown in Section 4.2, which we use for aggregation.

}B
j=1. Our theory suggests we can find the best weights {βi}N

i=1 and {gj
(cid:98)DT

i=1

We discovered the auto-weighted versions (FedDA_Auto and FedGP_Auto) usually have a quicker
convergence rate compared with static weights, we decide to use a smaller learning rate to train,
in order to prevent overfitting easily. In practice, we generally decrease the learning rate by some
factors after the initialization stage. For Colored-MNIST, we use factors of (0.1, 0.5, 0.25) for
domains (+90%, +80%, −90%). For domains of the other two datasets, we use the same factor of
0.25. Apart from that, we set the target domain batch size to 2, 4, 8 for Colored-MNIST, VLCS, and
TerraIncognita, respectively.

The following paragraphs discuss the extra time, space, and communication costs needed for running
the auto-weighted methods.

Extra time cost To compute a more accurate estimation of target variance, we need to use a smaller
batch size for the training on the target domain (more batches lead to more accurate estimation),
which will increase the training time for the target client during each round. Moreover, we need to
compute the auto weights each time when we perform the aggregation. The extra time cost in this
computation is linear in the number of batches.

Extra space cost
target client. I.e., the target client needs to store extra B model updates.

In each round, the target client needs to store the batches of gradients on the

Extra communication cost Since the aggregation is performed on the server, the extra communica-
tion would be sending the B model updates from the target client to the central server.

Discussion We choose to designate the target client to estimate the optimal β value as we assume
the cross-silo setting where the number of source clients is relatively small. In other scenarios where
the number of source clients is more than the number of batches used on the target client, one may
choose to let the global server compute the auto weights as well as do the aggregation, i.e., the target
client sends its batches of local updates to the global server, which reduces the communication cost
and shifts the computation task to the global server. If one intends to avoid the extra cost induced
by the auto-weighting method, we note that the FedGP with a fixed β = 0.5 can be good enough,
especially when the source-target domain shift is big. To reduce the extra communication cost of
the target client sending extra model updates to the server, we leave it to be an interesting future
work (Rothchild et al., 2020).

C.5 VISUALIZATION OF AUTO-WEIGHTED BETAS VALUES ON REAL-WORLD DISTRIBUTION

SHIFTS

Figure 6 and Figure 7 show the curves of auto weights β for each source domain with varying target
domains on the Colored-MNIST dataset, using FedDA and FedGP aggregation rules respectively.
Similarly, Figure 8 and Figure 9 are on the VLCS dataset; Figure 10 and Figure 11 are on the
TerraIncognita dataset. From the results, we observe FedDA_Auto usually has smaller β values
compared with FedGP_Auto. FedDA_Auto has drastically different ranges of weight choice de-
pending on the specific target domain (from 1e−3 to 1e−1), while FedGP_Auto usually has weights
around 1e−1. Also, FedDA_Auto has different weights for each source domain while interestingly,
FedGP_Auto has the almost same weights for each source domain for various experiments. Ad-
ditionally, the patterns of weight change are unclear - in most cases, they have an increasing or
increasing-then-decreasing pattern.

C.6 ADDITIONAL EXPERIMENT RESULTS ON VARYING STATIC WEIGHTS (β)

From the Table 8, 9, 10, and 11, we see FedGP is less sensitive to the choice of the weight parameter
β, enjoying a wider choice range of values, compared with FedDA. Additionally, under fixed weight
conditions, FedGP generally outperforms FedDA in most cases.

29

Published as a conference paper at ICLR 2024

(a) +90%

(b) +80%

(c) -90%

Figure 6: FedDA_Auto Colored-MNIST

(a) +90%

(b) +80%

(c) -90%

Figure 7: FedGP_Auto Colored-MNIST

(a) V

(b) L

(c) C

(d) S

Figure 8: FedDA_Auto VLCS

(a) V

(b) L

(c) C

(d) S

Figure 9: FedGP_Auto VLCS

(a) L100

(b) L38

(c) L43

(d) L46

Figure 10: FedDA_Auto TerraIncognita

C.7 SEMI-SYNTHETIC EXPERIMENT SETTINGS, IMPLEMENTATION, AND RESULTS

In this sub-section, we empirically explore the impact of different extents of domain shifts on FedDA,
FedGP, and their auto-weighted version. To achieve this, we conduct a semi-synthetic experiment,

30

01020304050Epoch0.010.020.030.040.05beta+80%-90%01020304050Epoch0.020.040.060.080.100.120.14beta+90%-90%01020304050Epoch0.0020.0040.0060.0080.010beta+90%+80%01020304050Epoch0.040.060.080.100.120.140.160.18beta+80%-90%01020304050Epoch0.10.20.30.40.50.60.7beta+90%-90%01020304050Epoch0.010.020.030.040.05beta+90%+80%01020304050Epoch0.0000.0020.0040.0060.0080.0100.012betaCLS01020304050Epoch0.0050.0100.0150.0200.025betaCSV01020304050Epoch0.0000.0050.0100.0150.0200.025betaLSV01020304050Epoch0.0020.0040.0060.0080.0100.0120.014betaCLV01020304050Epoch0.10.20.30.40.5betaCLS01020304050Epoch0.00.10.20.30.40.5betaCSV01020304050Epoch0.00.20.40.60.8betaLSV01020304050Epoch0.050.100.150.200.250.300.350.400.45betaCLV01020304050Epoch0.010.020.030.040.050.060.07betaL38L43L4601020304050Epoch0.00500.00750.01000.01250.01500.01750.02000.0225betaL100L43L4601020304050Epoch0.010.020.030.040.050.060.070.080.09betaL100L38L4601020304050Epoch0.010.020.030.040.050.06betaL100L38L43Published as a conference paper at ICLR 2024

(a) L100

(b) L38

(c) L43

(d) L46

Figure 11: FedGP_Auto TerraIncognita

FedDA
FedGP

0
61.21
61.21

0.2
61.10
63.82

0.4
59.15
64.30

0.6
49.90
65.31

0.8
29.80
64.80

1.0
17.65
38.48

Table 8: The effect of β on FedDA and FedGP on CIFAR-10 dataset with 0.4 noise level.

-90%
FedDA
FedGP

0
84.41
84.41

0.2
73.55
88.96

0.4
54.19
89.50

0.6
35.16
89.95

0.8
31.16
90.03

1.0
27.59
9.85

Table 9: The effect of β on FedDA and FedGP on Colored-MNIST -90% domain.

+90% 0
FedDA
FedGP

88.96
88.96

0.2
82.37
89.98

0.4
71.32
89.76

0.6
62.03
89.84

0.8
58.77
90.18

1.0
55.23
69.47

Table 10: The effect of β on FedDA and FedGP on Colored-MNIST +90% domain.

+80% 0
FedDA
FedGP

73.66
73.66

0.2
75.12
73.61

0.4
70.21
74.39

0.6
65.14
79.32

0.8
61.84
80.11

1.0
61.32
76.67

Table 11: The effect of β on FedDA and FedGP on Colored-MNIST +80% domain.

where we manipulate the extent of domain shifts by adding different levels of Gaussian noise (noisy
features) and degrees of class imbalance (label shifts). We show that the main impact comes from the
shifts between target and source domains instead of the shifts between source domains themselves.

Datasets and models We create the semi-synthetic distribution shifts by adding different levels
of feature noise and label shifts to Fashion-MNIST (Xiao et al., 2017) and CIFAR-10 (Krizhevsky
et al., 2009) datasets, adapting from the Non-IID benchmark (Li et al., 2022). For the model, we use
a CNN model architecture consisting of two convolutional layers and three fully-connected layers.
We set the communication round R = 50 and the local update epoch to 1, with 10 clients (1 target, 9
source clients) in the system.

Baselines We compare the following methods: Source Only: we only use the source gradients
by averaging. Finetune_Offline: we perform the same number of 50 epochs of fine-tuning after
FedAvg. FedDA (β = 0.5): a convex combination with a middle trade-off point of source and target
gradients. FedGP (β = 0.5): a middle trade-off point between source and target gradients with
gradient projection. Target Only: we only use the target gradient (β = 0). Oracle: a fully supervised
training on the labeled target domain serving as the upper bound.

Implementation For the experiments on the Fashion-MNIST dataset, we set the source learning
rate to be 0.01 and the target learning rate to 0.05. For CIFAR-10, we use a 0.005 source learning
rate and a 0.0025 learning rate. The source batch size is set to 64 and the target batch size is 16. We

31

01020304050Epoch0.050.100.150.200.250.300.350.400.45betaL38L43L4601020304050Epoch0.000.050.100.150.200.250.300.35betaL100L43L4601020304050Epoch0.050.100.150.200.250.300.350.40betaL100L38L4601020304050Epoch0.050.100.150.200.250.300.35betaL100L38L43Published as a conference paper at ICLR 2024

partition the data to clients using the same procedure described in the benchmark (Li et al., 2022).
We use the cross-entropy loss as the criterion and apply the Adam (Kingma & Ba, 2014) optimizer.

Setting 1: Noisy features We add different levels of Gaussian noise to the target domain to control
the source-target domain differences. For the Fashion-MNIST dataset, we add Gaussian noise levels
of std = (0.2, 0.4, 0.6, 0.8) to input images of the target client, to create various degrees of shifts
between source and target domains. The task is to predict 10 classes on both source and target clients.
For the CIFAR-10 dataset, we use the same noise levels, and the task is to predict 4 classes on both
source and target clients. We use 100 labeled target samples for Fashion-MNIST and 10% of the
labeled target data for the CIFAR-10 dataset.

Setting 2: Label shifts We split the Fashion-MNIST into two sets with 3 and 7 classes, respectively,
denoted as D1 and D2. A variable η ∈ [0, 0.5] is used to control the difference between source and
target clients by defining DS = η portion from D1 and (1 − η) portion from D2, DT = (1 − η) portion
from D1 and η portion from D2. When η = 0.5, there is no distribution shift, and when η → 0, the
shifts caused by label shifts become more severe. We use 15% labeled target samples for the target
client. We test on cases with η = [0.45, 0.30, 0.15, 0.10, 0.05, 0.00].

Auto-weighted methods and FedGP maintain a better trade-off between bias and variance
Table 12 and Table 13 display the performance trends of compared methods versus the change of
source-target domain shifts. In general, when the source-target domain difference grows bigger,
FedDA, Finetune_Offline, and FedAvg degrade more severely compared with auto-weighted methods,
FedGP and Target Only. We find that auto-weighted methods and FedGP outperform other baselines
in most cases, showing a good ability to balance bias and variances under various conditions and being
less sensitive to changing shifts. For the label shift cases, the target variance decreases as the domain
shift grows bigger (easier to predict with fewer classes). Therefore, auto-weighted methods, FedGP
as well as Target Only surprisingly achieve higher performance with significant shift cases. In addition,
auto-weighted FedDA manages to achieve a significant improvement compared with the fixed weight
FedDA, with a competitive performance compared with FedGP_Auto, while FedGP_Auto generally
has the best accuracy compared with other methods, which coincides with the synthetic experiment
results.

Connection with our theoretical insights
Interestingly, we see that when the shift is relatively
small (η = 0.45 and 0 noise level for Fashion-MNIST), FedAvg and FedDA both outperform FedGP.
Compared with what we have observed from our theory (Figure 5), adding increasing levels of noise
can be regarded as going from left to right on the x-axis and when the shifts are small, we probably
will get into an area where FedDA is better. When increasing the label shifts, we are increasing
the shifts and decreasing the variances simultaneously, we go diagonally from the top-left to the
lower-right in Figure 5, where we expect FedAvg is the best when we start from a small domain
difference.

Tables of noisy features and label shifts experiments Table 12 and Table 13 contain the full
results. We see that FedGP, FedDA_Auto and FedGP_Auto methods obtain the best accuracy
under various conditions; FedGP_Auto outperforms the other two in most cases, which confirms the
effectiveness of our weight selection methods suggested by the theory.

Impact of extent of shifts between source clients
In addition to the source-target domain dif-
ferences, we also experiment with different degrees of shifts within source clients. To control the
extent of shifts between source clients, we use a target noise = 0.4 with [3, 5, 7, 9] labels available.
From the results, we discover the shifts between source clients themselves have less impact on the
target domain performance. For example, the 3-label case (a bigger shift) generally outperforms the
5-label one with a smaller shift. Therefore, we argue that the source-target shift serves as the main
influencing factor for the FDA problem.

C.8 ADDITIONAL ABLATION STUDY RESULTS

The effect of target gradient variances We conduct experiments with increasing numbers of target
samples (decreased target variances) with varying noise levels [0.2, 0.4, 0.6] on Fashion-MNIST and

32

Published as a conference paper at ICLR 2024

Target noise
Source Only
Finetune_Offline
FedDA_0.5
FedGP_0.5
FedGP_1
FedDA_Auto
FedGP_Auto
Target Only
Oracle

Fashion-MNIST

0.2
25.49
48.26
69.73
75.09
77.03
77.03
75.09
70.59
82.53

0.4
18.55
40.15
58.6
71.09
71.67
72.68
71.46
66.03
81.20

0.6
16.71
36.64
50.13
68.01
63.71
67.69
67.53
61.26
75.60

0
83.94
81.39
86.41
76.33
79.40
78.25
76.19
74.00
82.00

0.8
14.99
33.71
45.51
62.22
54.18
62.85
62.93
57.82
72.60

CIFAR-10
0.2
0.4
17.61
20.48
56.80
66.31
54.67
62.25
65.28
66.40
20.75
21.46
64.09
65.83
65.26
67.02
60.25
60.69
70.12
73.61

0.6
16.44
52.10
49.77
63.29
19.31
62.29
63.12
59.38
69.22

0.8
16.27
49.37
47.08
61.59
18.26
60.39
61.30
59.03
68.50

Table 12: Target domain test accuracy (%) by adding feature noise to Fashion-MNIST and CIFAR-
10 datasets using different aggregation rules.

η
Source Only
Finetune_Offline
FedDA_0.5
FedGP_0.5
FedGP_1
FedDA_Auto
FedGP_Auto
Target only
Oracle

0.45
83.97
79.84
82.44
82.97
77.41
83.94
84.68
81.05
87.68

0.3
79.71
80.21
80.85
83.24
73.12
83.91
84.14
82.44
88.06

0.15
69.15
83.13
77.50
85.97
62.54
86.50
86.72
84.00
90.56

0.1
59.90
85.43
76.51
88.72
53.56
89.45
89.58
88.02
91.9

0.05
52.51
89.63
68.26
91.89
27.62
91.87
92.03
89.80
93.46

0
0.00
33.25
59.56
98.71
0.00
98.51
98.53
98.32
98.73

Table 13: Target domain test accuracy (%) by adding feature noise to the Fashion-MNIST dataset
using different aggregation rules.

CIFAR-10 datasets with 10 clients in the system. We compare two aggregation rules FedGP and
FedDA, as well as their auto-weighted versions. The results are shown in Table 15 (fixed FedDA
and FedGP) and Table 16 (auto-weighted FedDA and FedGP). When the number of available target
samples increases, the target performance also improves. For the static weights, we discover that
FedGP can predict quite well even with a small number of target samples, especially when the target
variance is comparatively small. For auto-weighted FedDA and FedGP, we find they usually have
higher accuracy compared with FedGP, which further confirms our auto-weighted scheme is effective
in practice. Also, we observe that sometimes FedDA_Auto performs better than FedGP_Auto (e.g.
on the Fashion-MNIST dataset) and sometimes vice versa (e.g. on the CIFAR-10 dataset). We
hypothesize that since the estimation of variances for FedGP is an approximation instead of the
equal sign, it is possible that FedDA_Auto can outperform FedGP_Auto in some cases because of
more accurate estimations of the auto weights β. Also, we notice auto-weighted scheme seems to
improve the performance more when the target variance is smaller with more available samples and
the source-target shifts are relatively small. In addition, we compare our methods with FedAvg, using
different levels of data scarcity. We show our methods consistently outperform FedAvg across all
cases, which further confirms the effectiveness of our proposed methods.

Number of labels
Source Only
Finetune_Offline
FedDA_0.5
FedGP_0.5
FedGP_1
Target_Only
Oracle

9
11.27
37.79
55.01
68.50
65.27
63.06
81.2

7
18.81
64.75
57.9
68.64
59.07
65.76
81.2

3
25.54
68.69
54.16
64.43
26.37
61.8
81.2

5
34.11
65.41
61.62
66.59
39.86
63.65
81.2

Table 14: Target domain test accuracy (%) on label shifts with [3,5,7,9] labels available on Fashion-
MNIST dataset with target noise = 0.4 and 100 target labeled samples.

33

Published as a conference paper at ICLR 2024

Noise level

0.2

0.4

0.6

Fashion-MNIST

CIFAR-10

FedAvg

FedDA

FedGP

FedAvg

FedDA

FedGP

FedAvg

FedDA

FedGP

100
200
500
1000

5%
10%
15%

75.98
76.50
75.55
76.12

24.50
22.86
23.20

69.73
72.07
76.59
77.92

62.24
62.25
59.16

75.09
74.21
78.41
78.68

64.21
65.92
65.97

59.36
60.20
58.74
62.33

21.25
22.35
23.25

58.60
58.59
65.34
68.26

46.89
54.67
56.93

71.09
70.93
74.07
75.17

63.57
65.39
65.11

49.94
48.56
47.90
50.81

19.42
18.60
17.88

50.13
52.67
54.97
59.16

47.56
49.77
51.83

68.01
70.31
70.52
71.63

61.39
63.67
63.73

Table 15: Target domain test accuracy (%) by adding feature noise=0.2, 0.4, 0.6 on the Fashion-
MNIST and CIFAR-10 datasets with different numbers of available target samples using fixed
weights, in comparison with FedAvg. We see our methods generally are more robust than FedAvg
with significant improvements.

Noise level

Fashion-MNIST

CIFAR-10

FedDA_Auto
79.04
79.74
79.48
80.23
63.04
65.72
66.57

100
200
500
1000
5%
10%
15%

0.2

FedGP_Auto
75.45
76.74
78.65
79.91
65.62
67.41
67.56

FedDA_Auto
72.21
74.30
75.21
76.75
60.79
64.43
65.4

0.4

FedGP_Auto
71.93
72.96
74.55
76.35
62.84
65.17
65.92

FedDA_Auto
66.16
69.27
71.40
73.16
60.02
62.25
63.36

0.6

FedGP_Auto
67.47
69.04
70.72
73.16
60.47
62.94
63.14

Table 16: Target domain test accuracy (%) by adding feature noise=0.2, 0.4, 0.6 on the Fashion-
MNIST and CIFAR-10 datasets with different numbers of available target samples using auto weights.

C.9

IMPLEMENTATION DETAILS OF FE DGP

To implement fine-grained projection for the real model architecture, we compute the cosine similarity
between one source client gradient gi and the target gradient gT for each layer of the model with a
threshold of 0. In addition, we align the magnitude of the gradients according to the number of tar-
get/source samples, batch sizes, local updates, and learning rates. In this way, we implement FedGP
by projecting the target gradient towards source directions. We show the details of implementing
static and auto-weighted versions of FedGP in the following two paragraphs.

T

≃ h(r)
Si

− h(r−1)

T ≃ h(r)

T − h(r−1)

global and G(r)

Static-weighted FedGP implementation Specifically, we compute the model updates from source
and target clients as G(r)
, respectively. In our real training
Si
process, because we use different learning rates, and training samples for source and target clients,
we need to align the magnitude of model updates. We first align the model updates from source
clients to the target client and combine the projection results with the target updates. We use lrT and
lrS to denote the target and source learning rates; batchsizeT and batchsizeS are the batch sizes
for target and source domains, respectively; nl is the labeled sample size on target client and ni is
the sample size for source client CSi; rS is the rounds of local updates on source clients. The total
gradient projection PGP from all source clients {G(r)
}N
i=1 projected on the target direction GT could
Si
be computed as follows. We use L to denote all layers of current model updates. ni denotes the
number of samples trained on source client CSi, which is adapted from FedAvg (McMahan et al.,
2017) to redeem data imbalance issue. Hence, we normalize the gradient projections according to the
number of samples. Also, (cid:83)L

l∈L concatenates the projected gradients of all layers.

PGP =


GP

(cid:18)(cid:16)

h

(cid:91)

N
(cid:88)

l∈L

i=1

(r)
Si

− h

(r−1)
global

(cid:17)l

(cid:16)

h

,

(r)
T − h

(r−1)
T

(cid:17)l(cid:19)

·

ni
(cid:80)N
i ni

·

nl
batchsizeT
ni
batchsizeS

·

lrT
lrS

·

1

rS

(cid:16)

h

·

(r)
Si

− h

(r−1)
global

(cid:17)





Lastly, a hyper-parameter β is used to incorporate target update GT into PGP to have a more stable
performance. The final target model weight h(r)

T at round r is thus expressed as:

T = h(r−1)
h(r)

T

+ (1 − β) · PGP + β · GT

34

Published as a conference paper at ICLR 2024

Auto-weighted FedGP Implementation For auto-weighted scheme for FedGP, we compute a
dynamic weight βi for each source domain DSi per communication round. With a set of pre-computed
{βi}N
i=1 weight values, the weighted projected gradients for a certain epoch can be expressed as
follows:

PGP = (cid:83)

l∈L

(cid:80)N

i=1


GP

(cid:18)(cid:16)

h

(r)
Si

− h

(r−1)
global

(cid:17)l

(cid:16)

h

,

(r)
T − h

(r−1)
T

(cid:17)l(cid:19)

·

ni·(1−βi)

(cid:80)N
i

ni

·

nl
batchsizeT
ni
batchsizeS

·

lrT
lrS

· 1
rS

(cid:16)

h

·

(r)
Si

− h

(r−1)
global

(cid:17)





Similarly, we need to incorporate target update GT into PGP . The final target model weight h(r)
round r is thus expressed as:

T at

T = h(r−1)
h(r)

T

+ PGP +

N
(cid:88)

i=1

ni · βi
(cid:80)N
i ni

· GT

C.10 GRADIENT PROJECTION METHOD’S TIME AND SPACE COMPLEXITY

Time complexity: Assume the total parameter is m and we have l layers. To make it simpler,
assume each layer has an average of m
l parameters. Computing cosine similarity for all layers of one
source client is O(( m
l )2 · l) = O(m2/l). We have N source clients so the total time cost for GP is
O(N · m2/l).

Space complexity: The extra memory cost for GP (computing cosine similarity) is O(1) per client for
storing the current cosine similarity value. In a real implementation, the whole process of projection is
fast, with around 0.023 seconds per call needed for N = 10 clients of Fashion-MNIST experiments
on the NVIDIA TITAN Xp hardware with GPU available.

C.11 ADDITIONAL EXPERIMENT RESULTS ON FED-HEART

As a showcase of a more realistic healthcare setting, we show the performances of our methods
compared with personalized baselines on the Fed-Heart dataset from FLamby (Du Terrail et al., 2022).
We randomly sample 20% data for the 0, 1, 3 centers and 100% data for the 2 center since there are
only 30 samples on the target domain. As shown in Table 17, our methods generally outperform
other baselines with large margins. KNN-per (Marfoq et al., 2022) may not fit this scenario since the
neural network we used only consists of one layer.

center
FedDA
FedGP
FedDA_Auto
FedGP_Auto
Source only
FedAvg
Ditto
FedRep
APFL
KNN-Per

0 (20%)
79.81
78.85
80.77
80.77
76.92
75.96
76.92
78.84
51.92
56.00

1 (20%)
78.65
80.45
80.90
79.78
75.96
76.40
73.03
65.17
57.30
56.00

2 (100%)
67.50
68.75
68.75
68.75
62.50
62.50
62.50
75.00
31.25
56.00

3 (20%)
62.22
65.33
71.11
69.78
55.56
55.56
55.56
60.00
42.22
56.00

Avg
72.05
73.35
75.38
74.77
67.74
67.61
67.00
69.75
45.67
56.00

Table 17: Target domain test accuracy (%) on Fed-Heart. FedGP and auto-weighted methods
generally outperform personalized FL methods with significant margins.

C.12 COMPARISON WITH THE SEMI-SUPERVISED DOMAIN ADAPTATION (SSDA) METHOD

In this sub-section, we show the performances of our methods compared with SSDA methods.
However, we note that the suggested SSDA methods cannot be directly adapted to the federated
setting without major modification. Kim & Kim (2020) uses feature alignments and requires access
to the source and target data at the same time, which is usually difficult to achieve in federated
learning. As for Saito et al. (2019), the overall adversarial learning objective functions consist of a

35

Published as a conference paper at ICLR 2024

loss objective on both source and target labeled data and the entropy coming from the unlabeled target
data, which also cannot be directly adapted to federated learning. On the contrary, auto-weighted
methods and FedGP have the flexibility to do the single source-target domain adaptation, which can
be compared with the SSDA method, though we notice that our setting is different from SSDA since
we do not have unlabeled data on the target domain and do not leverage the information coming
from this set of data. Here, we perform experiments on real-world datasets: our results suggest that
auto-weighted methods and FedGP outperform MME (Saito et al., 2019) when the shifts are large
even without using unlabeled data (overall our proposed methods have a comparable performance
with MME). Also, we observe in a single source-target domain adaptation setting, auto-weighted
FedGP usually has a better performance than auto-weighted FedDA and FedGP.

MME (Saito et al., 2019)
FedGP
FedDA_Auto
FedGP_Auto

0 ->1 / 1 ->0
79.55 / 89.28
79.34 / 88.34
79.31 / 88.38
79.34 / 71.59

0 ->2 / 2 ->0
25.98 / 12.99
90.23 / 64.47
87.00 / 56.07
90.23 / 80.95

1 ->2 / 2 ->1
14.66 / 24.43
90.23 / 78.70
89.13 / 66.59
90.23 / 79.34

Table 18: Colored-MNIST (0: +90%, 1: +80%, 2: -90%)

MME (Saito et al., 2019)
FedGP
FedDA_Auto
FedGP_Auto

0->1
68.68
67.92
67.92
68.45

1->2
72.86
75.30
75.73
74.21

2->3
81.90
77.45
76.97
77.69

Table 19: VLCS (0: C, 1: L, 2: V, 3: S)

MME (Saito et al., 2019)
FedGP
FedDA_Auto
FedGP_Auto

0->1
74.51
72.35
72.31
72.42

1->2
54.91
58.19
54.26
56.27

2->3
58.50
60.03
61.33
61.90

Table 20: TerraIncognita (0: L100, 1: L38, 2: L43, 3: L46)

36

