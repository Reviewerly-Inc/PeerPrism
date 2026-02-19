Published as a conference paper at ICLR 2024

LOCAL COMPOSITE SADDLE POINT OPTIMIZATION

Site Bai
Department of Computer Science
Purdue University
bai123@purdue.edu

Brian Bullins
Department of Computer Science
Purdue University
bbullins@purdue.edu

ABSTRACT

Distributed optimization (DO) approaches for saddle point problems (SPP) have
recently gained in popularity due to the critical role they play in machine learning
(ML). Existing works mostly target smooth unconstrained objectives in Euclidean
space, whereas ML problems often involve constraints or non-smooth regular-
ization, which results in a need for composite optimization. Moreover, although
non-smooth regularization often serves to induce structure (e.g., sparsity), standard
aggregation schemes in distributed optimization break this structure. Addressing
these issues, we propose Federated Dual Extrapolation (FeDualEx), an extra-step
primal-dual algorithm with local updates, which is the first of its kind to encompass
both saddle point optimization and composite objectives under the distributed
paradigm. Using a generalized notion of Bregman divergence, we analyze its con-
vergence and communication complexity in the homogeneous setting. Furthermore,
the empirical evaluation demonstrates the effectiveness of FeDualEx for inducing
structure in these challenging settings.

1

INTRODUCTION

A notable fraction of machine learning (ML) problems belong to saddle point problems (SPP),
including adversarial robustness (Madry et al., 2018; Chen & Hsieh, 2023), generative adversarial
networks (GAN) (Goodfellow et al., 2014), matrix games (Abernethy et al., 2018), multi-agent
reinforcement learning (Wai et al., 2018), among others. These applications call for effective
distributed saddle point optimization as their scale evolves beyond centralized learning. In typical
distributed optimization (DO) approaches, a central server coordinates collaborative learning among
clients through rounds of communication. In each round, clients learn a synchronized global model
locally without sharing their private data, then send the model to the server for aggregation, usually
through averaging (McMahan et al., 2017; Stich, 2019), to produce a new global model. The cost of
communication is known to dominate the optimization process (Koneˇcn`y et al., 2016).

Although preliminary progress has been made in distributed saddle point optimization (Beznosikov
et al., 2020; Hou et al., 2021), we would note that machine learning problems are commonly associated
with task-specific constraints or non-smooth regularization, which results in a need for composite
optimization (CO). Moreover, a common purpose for non-smooth regularization is to induce structure.
Typical ones include ℓ1 norm for sparsity and nuclear norm for low-rankness, which show up in
examples spanning from classical LASSO (Tibshirani, 1996), sparse regression (Hastie et al., 2015)
to deep learning such as adversarial example generation (Moosavi-Dezfooli et al., 2016), sparse GAN
(Zhou et al., 2020), convexified learning (Sahiner et al., 2022; Bai et al., 2024) and others.

Meanwhile, Yuan et al. (2021) identified the “curse of primal averaging” in standard aggregation
schemes of DO, where the specific regularization-imposed structure on the client models may no
longer hold after direct averaging on the server. For instance, each client may be able to obtain
a sparse solution, yet averaging the solutions across clients yields a dense solution. To address
this issue for convex optimization, they adopted the dual averaging technique (Nesterov, 2009),
but this approach is not specifically designed for SPP. Even in the sequential deterministic setting,
dual averaging or mirror descent (Nemirovskij & Yudin, 1983) achieve only a O(1/
T ) rate for
SPP (Bubeck et al., 2015), whereas extra-step methods achieve a O(1/T ) rate (Nemirovski, 2004;
Nesterov, 2007). At the same time, existing distributed methods for SPP fail to cover these composite
scenarios and address associated challenges, as summarized in Table 1.

√

1

Published as a conference paper at ICLR 2024

Task

Method

FedAvg
(Khaled et al., 2020)
FedDualAvg
(Yuan et al., 2021)
FeDualEx
(Ours)
Extra Step Local SGD
(Beznosikov et al., 2020)
SCCAFFOLD-S
(Hou et al., 2021)
FeDualEx
(Ours)

n
i
M

x
a
M
-
n
i
M

Composite &
Constrained &
Non-Euclidean
✗
✓
✓
✗
✗
✓

Convergence Rate

βB

RK + σB

M

βB

RK + σB

M
3
4

1
2 G
1
4 R

1
2 B
3
4

K

1
2

1
2 R

1
2
1
2 K
1
2
1
1
2 R
2 K
+ σB
1
2 R
M

1
2

+ β

+ β

1
2
1
2 K

2
3 B
2
3

1
3 σ
1
3 R
2
3 B
2
3

K
1
3 G

R
+ β

1
2

βB

RK + β

Convexity
Assumption

convex

convex

convex

2
3

2
3

1
3 G

R

2
3

2
3 B
2
3

B exp{− αKR

β2
α2 B exp{− αKR

α4K2R2

β } + σ2
β } + σ2
+ σB
1
2 R
M

M KR + β2σ2
M KR + β2σ2
+ β

α4KR2
1
2 B
1
2

1
2
1
2 K

1
2 G

R

1
2

3
4

βB

RK + β

1
2 G
1
4 R

1
2 B
3
4

K

α-strongly
convex-concave
α-strongly
convex-concave

convex-concave

3
4

Table 1: We list existing convergence rates on composite convex optimization and smooth saddle
point optimization in distributed settings similar to ours. Notations are R: communication rounds;
K: local steps; β: smoothness; B: diameter; G: gradient bound; M : clients; σ2: gradient variance.
FedAvg is also included as a reference. We further note that none of the work other than ours covers
composite SPP. They are included only for completeness.

We present the distributed paradigm for composite saddle point optimization defined in (1). In
particular, we propose Federated Dual Extrapolation (FeDualEx) (Algorithm 1), which builds on
Nesterov’s dual extrapolation (Nesterov, 2007), a classic extra-step algorithm suited for SPP. It
carries out a two-step evaluation of a proximal operator (Censor & Zenios, 1992) defined by the
Bregman Divergence (Bregman, 1967), which allows for SPP beyond the Euclidean space. To adapt
to composite regularization, FeDualEx also draws inspiration from recent progress in composite
convex optimization (Yuan et al., 2021) and adopts the notion of generalized Bregman divergence
(Flammarion & Bach, 2017) instead, which merges the regularization into its distance-generating
function. With some novel technical accommodations, we provide the convergence rate for FeDualEx
under the homogeneous setting, which is, to the best of our knowledge, the first convergence rate for
composite saddle point optimization under the DO paradigm. In support of the proposed method, we
conduct numerical evaluations to verify the effectiveness of FeDualEx on composite SPP.

To further demonstrate the quality of the induced structure, we include the primal twin of FeDualEx
based on mirror prox (Nemirovski, 2004), namely “Federated Mirror Prox (FedMiP)”, as a baseline
for comparison in Appendix H. This is in line with the dichotomy between Federated Mirror Descent
(FedMiD) and Federated Dual Averaging (FedDualAvg) (Yuan et al., 2021), from which Yuan et al.
(2021) identified the “curse of primal averaging” in DO, i.e., the specific regularization-imposed
structure on the client models may no longer hold after primal averaging on the server. It highlights
that FeDualEx naturally inherits the merit of dual aggregation from FedDualAvg. In addition, we
analyze FeDualEx for federated composite convex optimization and show that FeDualEx recovers
the same convergence rate as FedDualAvg under the convex setting.

Last but not least, by reducing the number of clients to one, we show for the sequential version of
FeDualEx that the analysis naturally yields a convergence rate for stochastic composite saddle point
optimization which, to our knowledge, is the first such algorithm for non-Euclidean settings and
matches the O( 1√
) rate in general stochastic saddle point optimization (Mishchenko et al., 2020;
T
Juditsky et al., 2011). Further removing the noise from gradient estimates, FeDualEx still generalizes
dual extrapolation to deterministic composite saddle point optimization with a O( 1
T ) convergence
rate that matches the smooth case and also the pioneering composite mirror prox (CoMP) (He et al.,
2015) as presented in Table 2.

Our Contributions:

• We propose FeDualEx for distributed learning of SPP with composite possibly non-smooth
regularization (Section 4.1). In support of the proposed algorithm, we provide a convergence
rate for FeDualEx under the homogeneous setting (Section 4.2). To the best of our knowledge,
FeDualEx is the first of its kind that encompasses composite possibly non-smooth regularization
for SPP under a distributed paradigm, as shown in Table 1.

• Additionally, we showcase the structure-preserving (e.g., sparsity) advantage of FeDualEx
achieved through dual-space averaging. In particular, we present its primal twin FedMiP as a
baseline to highlight this contrast (Appendix H).

2

Published as a conference paper at ICLR 2024

Noise

Deterministic

Stochastic

Rate

O (cid:0) 1

T

(cid:1)

O

(cid:17)

(cid:16) 1√

T

Composite SPP

CoMP (He et al., 2015)
Deterministic FeDualEx (Ours)

Extragradeint (Euclidean) (Mishchenko et al., 2020)
Sequential FeDualEx (Ours)

Smooth SPP
Mirror Prox (Nemirovski, 2004)
Dual Extrapolation (Nesterov, 2007)
Accelerated Proximal Gradient (Tseng, 2008)
Mirror Prox (Juditsky et al., 2011)
Sequential FeDualEx (Ours)

Table 2: Convergence rates for convex-concave SPP. The deterministic version of FeDualEx general-
izes dual extrapolation (DE) to composite SPP, and the sequential version of FeDualEx generalizes
DE to both smooth and composite stochastic saddle point optimization.

• FeDualEx produces several byproducts in the CO realm, as demonstrated in Table 2 : (1) The
sequential version of FeDualEx leads to the stochastic dual extrapolation for CO and yields, to
our knowledge, the first convergence rate for the stochastic optimization of composite SPP in
non-Euclidean settings . (2) Further removing the noise leads to its deterministic version, with
rates matching existing ones in smooth and composite saddle point optimization (Section 5).
• We demonstrate experimentally the effectiveness of FeDualEx on various composite saddle point
tasks, including bilinear problems on synthetic data with ℓ1 and nuclear norm regularization,
as well as the universal adversarial training of logistic regression with MNIST and CIFAR-10
(Section 6).

2 RELATED WORK

We provide a brief overview of some related work and defer extended discussions to Appendix B.

The distributed optimization paradigm we consider aligns with that in Local SGD (Stich, 2019),
which is also the homogeneous setting of Federated Averaging (FedAvg) (McMahan et al., 2017).
Stich (2019) provides the first convergence rate for FedAvg, and it has been improved with tighter
analysis and also analyzed under heterogeneity (e.g., (Khaled et al., 2020; Woodworth et al., 2020b)).
Recently, Yuan et al. (2021) extended FedAvg to composite convex optimization and proposed
FedDualAvg that aggregates learned parameters in the dual space and overcomes the “curse of primal
averaging” in federated composite optimization.

For SPP, Beznosikov et al. (2020) investigate the distributed extra-gradient method for strongly-
convex strongly-concave SPP in the Euclidean space. Hou et al. (2021) propose FedAvg-S and
SCAFFOLD-S based on FedAvg (McMahan et al., 2017) and SCAFFOLD (Karimireddy et al.,
2020) for SPP, which yields similar convergence rate to (Beznosikov et al., 2020). In addition,
Ramezani-Kebrya et al. (2023) study the problem from the information compression perspective
with the measure of communication bits. Yet, the aforementioned works are limited to smooth and
unconstrained SPP in the Euclidean space. The more general setting of composite SPP is only found
in sequential optimization literature, where the representative composite mirror prox (CoMP) (He
et al., 2015) generalizes the classic mirror prox (Nemirovski, 2004) yet keeps the O( 1
T ) convergence
rate. In the stochastic setting, Mishchenko et al. (2020) analyzed a variant of stochastic mirror prox
(Juditsky et al., 2011), which is then capable of handling composite terms in the Euclidean space. We
will later show that the sequential analysis of our proposed algorithm also yields the same rate for
dual extrapolation (Nesterov, 2007) in composite optimization, utilizing different proving techniques.
As a result, we focus on the distributed optimization of composite SPP and propose FeDualEx.

3 PRELIMINARIES AND DEFINITIONS

We provide some preliminaries and definitions necessary for introducing FeDualEx. More details are
included in Appendix C.1. To begin with, we lay out the notations.

Notations. We use [n] to represent the set {1, 2, ..., n}. We use ∥ · ∥ to denote an arbitrary norm,
∥ · ∥∗ to denote the dual norm, and ∥ · ∥2 to denote the Euclidean norm. We use ∇ for gradients, ∂
for subgradients, and ⟨·, ·⟩ for inner products. Related to the algorithm, we use English letters (e.g.,
z, x, y) to denote primal variables, Greek letters (e.g., ω, ς, µ, ν) to denote dual variables. We use
R for communication rounds, K for local updates, B for diameter bound, G for gradient bound, β
for smoothness constant, σ for standard deviation, ξ for random samples. We use h∗ to denote the
convex conjugate of a function h.

3

Published as a conference paper at ICLR 2024

Composite Saddle Point Optimization. We study composite saddle point optimization. Its
objective is formally given in the following definition.
Definition 1 (Composite SPP). The objective of composite saddle point optimization is defined as

min
x∈X

max
y∈Y

ϕ(x, y) = f (x, y) + ψ1(x) − ψ2(y)

(1)

where f (x, y) = 1
M

(cid:80)M

m=1 fm(x, y) and ψ1(x), ψ2(y) are possibly non-smooth.

It is typically evaluated by the duality gap: Gap(ˆx, ˆy) = maxy∈Y ϕ(ˆx, y) − minx∈X ϕ(x, ˆy).

x′(·) = arg minx{⟨·, x⟩ + V h

Mirror Prox and Dual Extrapolation. Mirror prox (Ne-
mirovski, 2004) and dual extrapolation (Nesterov, 2007) are
classic methods for convex-concave SPP. Both are proxi-
mal algorithms based on the proximal operator defined as
Prox h
x′(x) =
h(x) − h(x′) − ⟨∇h(x′), x − x′⟩ is the Bregman divergence
generated by some closed, strongly convex, and differentiable
function h. Both algorithms conduct two evaluations of the proximal operator, while dual extrapola-
tion carries out updates in the dual space. Figure 1 gives a brief illustration of dual extrapolation with
the proximal operator as in (Cohen et al., 2021), with details in Appendix C.1.

(ηg(xt))
µt+1 = µt + ηg(xt+1/2)

xt = Prox h
xt+1/2 = Prox h
xt

Figure 1: Dual Extrapolation.

x′(x)}, in which V h

¯x(µt)

Generalized Bregman Divergence. Recent advances in composite convex optimization (Yuan
et al., 2021) have utilized the Generalized Bregman Divergence (Flammarion & Bach, 2017) for
analyzing composite objectives. It incorporates the composite term into the distance-generating
function of the vanilla Bregman divergence, and measures the distance in terms of one variable and
the dual image of the other, with the key insight being the conjugate of a non-smooth generalized
distance-generating function is differentiable.
Definition 2 (Generalized Bregman Divergence (Flammarion & Bach, 2017)). Generalized Bregman
divergence is defined to be ˜V ht
t (µ′)⟩, where ht = h+tηψ
is a generalized distance-generating function that is closed and strongly convex, t is the current
number of iterations, η is the step size, h∗
t is the convex conjugate of ht, and µ′ is the dual image of
t (µ′).
x′, i.e., µ′ ∈ ∂ht(x′) and x′ = ∇h∗
Generalized Bregman divergence is suitable not only for non-smooth regularization but also for any
convex constraints C, taking ψ(x) = 0 if x ∈ C and +∞ otherwise.

µ′ (x) = ht(x)−ht(∇h∗

t (µ′))−⟨µ′, x−∇h∗

4 FEDERATED DUAL EXTRAPOLATION (FEDUALEX)

To tackle composite SPP in the DO paradigm, we acknowledge the challenges from several aspects.
Specifically, the generality afforded by composite and/or saddle point problems results in a need for
more sophisticated techniques that work with this additional structure. These concerns are further
complicated by the challenges that arise for DO, where communication and aggregation need to be
carefully handled under the distributed mechanism. In particular, Yuan et al. (2021) identified the “the
curse of primal averaging” in composite federated optimization and advocated for dual aggregation.
Dealing with these challenges altogether is rather non-trivial, as the techniques that are naturally
suited for one would fail for another. In this regard, we first present FeDualEx (Algorithm 1) and
several relevant novel definitions proposed for its adaptation to composite SPP. Then we analyze the
convergence rate in the homogeneous setting.

4.1 THE FEDUALEX ALGORITHM

FeDualEx builds its core on the classic dual extrapolation, an extra-step algorithm geared for saddle
point optimization. Its effectiveness has been widely verified in vanilla smooth convex-concave
SPP. Furthermore, its updating sequence lies in the dual space which would naturally inherit the
advantage of dual aggregation in composite federated optimization. The challenge remains for
composite optimization, as relevant work is limited, and the existing composite extension for the
extra-step method (He et al., 2015) is quite technically involved. Given that the smooth analysis
of dual extrapolation is already non-trivial (Nesterov, 2007), no attempts were previously made for
generalizing dual extrapolation to the composite optimization realm.

4

Published as a conference paper at ICLR 2024

Algorithm 1 FEDERATED-DUAL-EXTRAPOLATION (FeDualEx) for Composite SPP
Input: ϕ(z) = f (x, y)+ψ1(x)−ψ2(y) = 1
M

m=1 fm(x, y)+ψ1(x)−ψ2(y): objective function;
ℓ(z): distance-generating function; gm(z) = (∇xfm(x, y), −∇yfm(x, y)): gradient operator.
Hyperparameters: R: number of communication rounds; K: number of local update iterations; ηs:

(cid:80)M

server step size; ηc: client step size.

Dual Initialization: ς0 = 0: initial dual variable, ¯ς: fixed point in the dual space.
Output: Approximate solution z = (x, y) to minx∈X maxy∈Y ϕ(x, y)
1: for r = 0, 1, . . . , R − 1 do
2:
3:
4:
5:

Sample a subset of clients Cr ⊆ [M ]
for m ∈ Cr in parallel do

ς m
r,0 = ςr
for k = 0, 1, . . . , K − 1 do

▷ Two-step evaluation of the generalized proximal operator

6:

7:

ℓr,k
¯ς

r,k = ˜Prox
zm
r,k+1/2 = ˜Prox
zm
r,k+1 = ς m
ς m
end for

(ς m
r,k)
ℓr,k+1
¯ς−ςm
r,k

r,k + ηcgm(zm

(ηcgm(zm
r,k+1/2; ξm

r,k; ξm
r,k))
r,k+1/2)

end parallel for

8:
9:
10:
11: ∆r = 1
m∈Cr
|Cr|
ςr+1 = ςr + ηs∆r
12:
13: end for
14: Return:

(cid:80)R−1
r=0

1
RK

(cid:80)

(ς m

r,K − ς m
r,0)

(cid:80)K−1
k=0

(cid:92)zr,k+1/2 with (cid:92)zr,k+1/2 defined in (4).

▷ Dual variable update

▷ Server dual update

Inspired by recent advances in composite convex optimization, we recognize the Generalized Bregman
Divergence (Flammarion & Bach, 2017) as a powerful tool for analyzing proximal methods for
composite objectives. Adapting to the context of composite SPP, we make an extension to the
Generalized Bregman Divergence for saddle functions, and provide the definition below.
Definition 3 (Generalized Bregman Divergence for Saddle Functions). The generalized distance-
generating function for the optimization of (1) is ℓt(z) = ℓ(z)+tηψ(z), where ℓ(z) = h1(x)+h2(y),
h1 and h2 are distance-generating functions for x and y, ψ(z) = ψ1(x) + ψ2(y), η is the step size,
and t is the current number of iterations. It generates the following generalized Bregman divergence:
˜V ℓt
ς ′ (z) = ℓt(z) − ℓt(z′) − ⟨ς ′, z − z′⟩,

where ς ′ is the preimage of z′ with respect to the gradient of the conjugate of ℓt, i.e., z′ = ∇ℓ∗

t (ς ′).

Yet as we notice in previous works (Flammarion & Bach, 2017; Yuan et al., 2021), generalized
Bregman divergence is applied only for theoretical analysis. In terms of algorithm design, the
previous proximal operator for composite convex optimization is based on the vanilla Bregman
divergence plus the composite term, specifically, arg minx{⟨·, x⟩ + V h
x′(x) + ηψ(x)} in (Duchi et al.,
2010; He et al., 2015), and arg minx{⟨·, x⟩ + h(x) + ηtψ(x)} in (Xiao, 2010; Flammarion & Bach,
2017). However, we find this definition insufficient for dual extrapolation, as its dual update and the
composite term from the extra step break certain parts of the analysis. In this effort, we propose a
novel technical change to the proximal operator, directly replacing the Bregman divergence in the
proximal operator with the generalized Bregman divergence.
Definition 4 (Generalized Proximal Operator for Saddle Functions). A proximal operation in the
composite setting with generalized Bregman divergence for Saddle Functions is defined to be

˜Prox

ℓt
ς ′ (g) := arg min

{⟨g, z⟩ + ˜V ℓt

ς ′ (z)},

z
t (ς ′), and ς ′ ∈ ∂ℓt(z′) = ∇ℓ(z′) + ηt∂ψ(z′).

where ς ′ is the dual image of z′, i.e., z′ = ∇ℓ∗
Compared with the vanilla proximal operator in Section 3, this novel design for the composite
adaptation of dual extrapolation is quite natural. It is different from previous proximal operators,
which after expanding take the form arg minz{⟨· − ∇ℓ(z′), z⟩ + ℓt(z)} (Duchi et al., 2010) or
h
ς ′(·) = arg minz{⟨· − ς ′, z⟩ + ℓt(z)}.
arg minz{⟨·, z⟩ + ℓt(z)} (Xiao, 2010), whereas ours is
These adaptations are necessary for technical reasons, as our algorithm involves prox operators on
both the clients and the server to induce structure in the aggregated solution, which would otherwise

˜Prox

5

Published as a conference paper at ICLR 2024

break the conventional analysis. (Specifically, using these previous notions yields extra composite
terms in the analysis that do not cancel out but rather accumulate, thus hindering the convergence.)

With the novel definitions above, we are able to formally present FeDualEx in Algorithm 1. It
follows the general structure of DO. For each client, the two-step evaluation of the generalized
proximal operator and the final dual update are highlighted in green , which resembles the classic
dual extrapolation updates in Figure 1. To align with our generalized proximal operator, we also
move the primal initialization ¯x in the original dual extrapolation to the dual space as ¯ς. On the server,
the dual variables from clients are aggregated first in the dual space, then projected to the primal with
a mechanism later defined in (4).

4.2 CONVERGENCE ANALYSIS OF FEDUALEX

In this section, we provide the convergence analysis of FeDualEx for the homogeneous DO of
composite SPP. We further assume the full participation of clients in each round for simplicity, but
this condition can be trivially removed by lengthy analysis. We start by showing the equivalence
between primal-dual projection and the generalized proximal operator, and for the convenience of
analysis, reformulating the updating sequences with another pair of auxiliary dual variables.

r,k+1((¯ς − ς m

Projection Reformulation. Generalized proximal operators can be presented as projections, i.e., the
gradient of the conjugate of the generalized distance-generating function in Appendix C.2. Thus, line
6 to 8 in Algorithm 1 can be expanded by Definition 4, and rewrite as: (1) zm
r,k); (2)
r,k+1/2 = ∇ℓ∗
zm
Further define auxiliary dual variable ωm
in which ℓ∗
image of the intermediate variable zm
an equivalent updating sequence with the auxiliary dual variables.

r,k; ξm
r,k = ¯ς −ς m
r,k is the conjugate of ℓr,k = ℓ + (ηsrK + k)ηcψ. And define ωm
r,k+1(ωm

r,k + ηcgm(zm
r,k. It satisfies immediately that zm

r,k),
r,k+1/2 to be the dual
r,k+1/2). Then we get

r,k+1/2; ξm
r,k = ∇ℓ∗

r,k+1/2 such that zm

r,k+1/2).
r,k(ωm

r,k) − ηcgm(zm

r,k+1/2 = ∇ℓ∗

r,k)); (3) ς m

r,k+1 = ς m

r,k(¯ς − ς m

r,k = ∇ℓ∗

r,k+1/2 = ωm
ωm

r,k − ηgm(zm

r,k; ξm

r,k),

r,k+1 = ωm
ωm

r,k+1/2; ξm

r,k+1/2)

Define their average across clients, ωr,k = 1
M
we can analyze the following averaged dual shadow sequences:

m=1 ωm

r,k, gr,k = 1
M

(cid:80)M

m=1 gm(zm

r,k; ξm

r,k). Then

ωr,k+1/2 = ωr,k − ηcgr,k,

(2)

ωr,k+1 = ωr,k − ηcgr,k+1/2.

(3)

In the meantime, their shadow primal projections on the server are defined as

(cid:100)zr,k = ∇ℓ∗

r,k(ωr,k),

(cid:92)zr,k+1/2 = ∇ℓ∗

r,k+1(ωr,k+1/2).

(4)

Next, we list the key assumptions. Detailed presentation and additional remarks that ease the
understanding of proofs are also provided in Appendix C.3.
Assumptions For the composite saddle function ϕ(x, y) = 1
M
its gradient operator is given by g = (∇xf, −∇yf ) and g = 1
M

m=1 fm(x, y) + ψ1(x) − ψ2(y),
m=1 gm. We assume that

(cid:80)M
(cid:80)M

a.(Convexity of f ) ∀m ∈ [M ], fm(x, y) is convex in x and concave in y.
b.(Convexity of ψ) ψ1(x) is convex in x, and ψ2(y) is convex in y.

c. (Lipschitzness of g) gm(z) =
d.(Unbiased Estimate and Bounded Variance) ∀m ∈ [M ],
(cid:2)∥gm(zm; ξm) − gm(zm)∥2

Eξ[gm(zm; ξm)] = gm(zm), and Eξ

(cid:3) ≤ σ2

∗

(cid:104) ∇xfm(x,y)
−∇yfm(x,y)

(cid:105)

is β-Lipschitz: ∥gm(z) − gm(z′)∥∗ ≤ β∥z − z′∥

for random sample ξm,

e. (Bounded Gradient) ∀m ∈ [M ], ∥gm(zm; ξm)∥∗ ≤ G
f. The distance-generating function ℓ is a Legendre function that is 1-strongly convex, i.e., ∀z, z′,

g.The optimization domain Z is compact w.r.t. Bregman divergence, i.e., ∀z, z′ ∈ Z, V ℓ

z′(z) ≤ B.

ℓ(z′) − ℓ(z) − ⟨∇ℓ(z), z′ − z⟩ ≥ 1

2 ∥z′ − z∥2.

We would note that Assumption e (bounded gradient) is a standard assumption in classic distributed
composite optimization (Duchi et al., 2011), and is made in other DO analysis (Stich, 2019; Li et al.,
2020b; Yu et al., 2019; Yuan et al., 2021).

Main Theorem. Under the aforementioned assumptions, we present the following theorem that
provides the convergence rate of FeDualEx in terms of the duality gap.

6

r,k − ηgm(zm
(cid:80)M

Published as a conference paper at ICLR 2024

Theorem 1 (Main). Under assumptions, the duality gap evaluated with the ergodic sequence
generated by the intermediate steps of FeDualEx in Algorithm 1 is bounded by

(cid:104)
E

Gap

(cid:16) 1

R−1
(cid:88)

K−1
(cid:88)

RK

r=0

k=0

(cid:92)zr,k+1/2

(cid:17)(cid:105)

≤

B
ηcRK

+ 20β2(ηc)3K 2G2 +

5σ2ηc
M

+ 2

3

2 βηcKGB

1
2 .

Choosing step size ηc = min{ 1
1
2 β
5

,

1
4 β

20

B

1
2 G

1
4
1
2 K

3
4 R

1
4

, B
5

1
2 σR

1
2

1
2 M
1
2 K

,

1
2

3
4 β

2

B

1
2 G

1
4
1
2 KR

1
2

},

(cid:104)
E

Gap

(cid:16) 1

R−1
(cid:88)

K−1
(cid:88)

RK

r=0

k=0

(cid:92)zr,k+1/2

(cid:17)(cid:105)

≤

5 1
2 βB
RK

+

2 B 3

4

20 1

4 β 1
K 1

2 G 1
4 R 3

4

+

2

5 1
2 σB 1
2 R 1
M 1

2 K 1

2

+

2 3

4 β 1

2 B 3

4

2 G 1
R 1

2

.

RK ) and O(

To the best of our knowledge, this is the first convergence rate for federated composite saddle point
optimization. The O( 1
) terms roughly match previous DO algorithms, where
the noise term decays with the number of clients M . If M is large enough, then the O(1/R 1
2 ) term
takes domination in terms of communication complexity. The convergence analysis further validates
the effectiveness of FeDualEx, which then advances distributed optimization to a broad class of
composite saddle point problems. The complete proof of Theorem 1 can be found in Appendix E.

1√

M RK

On Composite Convex Optimization. We also analyze the convergence rate for FeDualEx under
the federated composite convex optimization setting. As the following theorem shows, FeDualEx
achieves the same O(1/R 2
Theorem 2. Under the convex counterparts of previous assumptions, choosing step size ηc =
min{ 1
}, the ergodic intermediate sequence gener-
1
2 β
5

3 ) as in (Yuan et al., 2021). The proof is provided in Appendix F.

, B
5
ated by FeDualEx for composite convex objectives satisfies

1
2 M
1
2 K

1
3
2
3 KR

1
4
1
2 K

1
2 σR

1
3 G

1
2 G

3
4 R

1
3 β

1
4 β

20

B

B

1
4

1
2

1
3

1
2

2

,

,

E(cid:2)ϕ(

1
RK

R−1
(cid:88)

K−1
(cid:88)

r=0

k=0

(cid:92)xr,k+1/2) − ϕ(x)(cid:3) ≤

5 1
2 βB
RK

+

2 B 3

4

20 1

4 β 1
K 1

2 G 1
4 R 3

4

+

2

5 1
2 σB 1
2 R 1
M 1

2 K 1

2

+

2 1

3 β 1

3 B 2

3

3 G 2
R 2

3

.

Even though this rate is not preserved in composite saddle point optimization, we note that the
optimization of SPP is much more general, and convexity itself is a stronger assumption. More
specifically, the complicated setting, including the non-smooth term, the primal-dual projection, the
extra-step saddle point optimization, etc., together limit the tools available for analysis.

Remark On Heterogeneity. Even for federated composite optimization (Yuan et al., 2021), the
heterogeneous setting presents significant hurdles. Specifically, the involvement of heterogeneity is
limited to quadratic functions, under which assumption the is gradient linear, and this simplifies the
analysis. It further relies on the norm generated by its Hessian. For saddle functions, “quadraticity”
(as well as a matrix-induced norm) is less well-defined, as the Jacobian of their gradient operator is
not (symmetric) positive semidefinite in general. Such further advancements go beyond the scope of
this paper. Thus, we regard Theorem 1 as a significant start for federated learning of composite SPP.

5 FEDUALEX IN SEQUENTIAL SETTINGS

Stochastic Composite Saddle Point Optimization FeDualEx can be naturally reduced to sequen-
tial stochastic optimization of composite SPP, which we term as Sequential FeDualEx or Stochastic
Dual Extrapolation. By reducing the number of clients to one, thus eliminating the need for communi-
cation, the convergence analysis follows through smoothly and yields O( 1√
) rate (denoting K as T )
T
expected for first-order stochastic algorithms. This is the first such rate in the non-Euclidean setting,
matching the previous Euclidean rate (Mishchenko et al., 2020) and non-composite rate (Juditsky
et al., 2011). Theorem 3 gives the result with proof in Appendix G.1.

Theorem 3. Under the sequential versions of previous assumptions, ∀z ∈ Z, choosing step size
, B
η = min{ 1
1
3
2 β
3
t=0 zt+1/2)(cid:3) ≤ 3
(cid:80)T −1

}, the ergodic intermediate sequence of stochastic dual extrapolation satisfies

E(cid:2) Gap( 1

T + 3

1
2 βB

1
2 σT

T

1
2

1
2

1
2

.

1
2 σB
1
2

T

7

Published as a conference paper at ICLR 2024

max
y∈Y

⟨Ax − b, y⟩ + λ∥x∥1 − λ∥y∥1

min
x∈X
A ∈ Rn×m, X = {Rm : ∥x∥∞ ≤ D},
Y = {Rn : ∥y∥∞ ≤ D}.
b ∈ Rn,

min
X∈X

max
Y∈Y
A ∈ Rn×m,
B ∈ Rn×p,

Tr(cid:0)(AX − B)⊤Y(cid:1) + λ∥X∥∗ − λ∥Y∥∗

X = {Rm×p : ∥X∥2 ≤ D},
Y = {Rn×p : ∥Y∥2 ≤ D}.

Figure 2: The composite SPP with ℓ1 regular-
ization for sparsity (Jiang & Mokhtari, 2022).

Figure 3: The composite SPP with nuclear norm low-
rank regularization.

Deterministic Composite Saddle Point Optimization Further removing the noise in gradient,
FeDualEx reduces to a deterministic algorithm for composite SPP. Even so, we are still generalizing
the classic dual extrapolation algorithm to CO, and thus term the algorithm Deterministic FeDualEx
or Composite Dual Extrapolation. Following a similar analysis, we are able to get the O( 1
T ) rate as
in previous work for CO (He et al., 2015) as well as the smooth dual extrapolation (Nesterov, 2007).
The proof for Theorem 4 is in Appendix G.2, which is a much simpler one based on the recently
proposed Relative Lipschitzness (Cohen et al., 2021).
Theorem 4. Under the basic convexity assumption and β-Lipschitzness of g, ∀z ∈ Z and η ≤ 1
β ,
(cid:80)T −1
composite dual extrapolation satisfies Gap( 1
T

t=0 zt+1/2) ≤ βB
T .

6 EXPERIMENTS

To complement our largely theoretical results, we verify in this section the effectiveness of FeDualEx
by numerical evaluation. Additional experiments and detailed settings are deferred to Appendix A.

Composite Bilinear SPP. We first test FeDualEx on composite bilinear problems with synthetic data.
The problems considered are demonstrated in Figure 2 and 3, in which m = 600, n = 300, p = 20,
λ = 0.1, D = 0.05. The corresponding composite terms are ℓ1 regularization with ℓ∞ ball constraint
and nuclear regularization with spectral constraint. The purpose of ℓ1 regularization is to encourage
sparsity and nuclear regularization to encourage a solution with low rank.

We compare FeDualEx against FedDualAvg, FedMiD (Yuan et al., 2021), and FedMiP proposed
in Algorithm 2 in Appendix H. We note that methods like Extra Step Local SGD (Beznosikov
et al., 2020) and SCAFFOLD-S (Karimireddy et al., 2020) are not suited to problems with non-
smooth terms, but we include one of them for completeness, given that their rates are similar. For
such a comparison, one can only compute the sub-gradient instead of the gradient (which does not
everywhere exist). Projection needs to be applied as well to account for the constraints.

(a) One Local Update (K = 1)

(b) Ten Local Updates (K = 10)

Figure 4: Duality gap and sparsity of the solution for ℓ1 regularized SPP with ℓ∞ constraint.

(a) K = 1

(b) K = 10

Figure 5: Duality gap and rank of the solution to the nuclear norm regularized SPP.

8

020004000Communication Rounds101100101Duality Gap020004000Communication Rounds0.60.81.0Sparsity0200400Communication Rounds101100101Duality Gap0200400Communication Rounds0.60.81.0Sparsity020406080100Communication Rounds102100Duality Gap020406080100Communication Rounds101520X Rank020406080100Communication Rounds101520Y Rank05101520Communication Rounds102101Duality Gap05101520Communication Rounds101520X Rank05101520Communication Rounds101520Y RankPublished as a conference paper at ICLR 2024

(a) MNIST

(b) CIFAR-10

Figure 6: Universal adversarial training loss and validation
accuracy of logistic regression on unattacked data.

(a) FeDualEx

(b) PGDA

Figure 7: Attack generated from the
universal-adversarially trained logistic
regression on MNIST and CIFAR-10.

We evaluate the convergence in terms of the duality gap and also demonstrate the structure of the
solution, i.e., sparsity or low-rankness. The duality gap of the problems of interest can be evaluated
in closed form, which is derived in Appendix A.1 and A.2. The sparsity is measured by the ratio of
non-zero entries to the parameter size, and we regard numbers less than 10−5 as zeros. For DO, we
simulate M = 100 clients. For the gradient query of each client in each local update, we inject a
Gaussian noise from N (0, σ2), where σ = 0.1. The evaluation is conducted for two different settings:
(a) K = 1 local update (b) K = 10 local updates. The results are demonstrated in Figure 4 and 5.

From the duality gap curves in Figure 4, we see that extra-step methods, i.e., FeDualEx and FedMiP
converge to the order of 10−1 whereas FedDualAvg and FedMiD stay above 100. Thus, it is evident
that methods for composite convex optimization are no longer suited for composite saddle point
optimization, and FeDualEx provides the first effective solution addressing the challenge. From the
sparsity of the solution, we see that the dual methods demonstrate better adherence to regularization.
Among the methods superior in saddle point optimization, FeDualEx reaches a sparsity of around
0.7 while FedMiP is around 0.95. This aligns with the previous analysis on the advantage of dual
aggregation and further validates the effectiveness of FeDualEx for solving composite SPP. In addition,
methods for smooth unconstrained optimization like ExtraStepLocalSGD do not converge for SPP
with composite non-smooth terms, nor does it impose any desired structure, e.g., sparsity. We observe
similar advantages of FeDualEx in convergence and inducing low-rankness from Figure 5 as well.

Universal Adversarial Training of Logistic Regression. We also consider the task of universal
adversarial training (Shafahi et al., 2020) of logistic regression, i.e. the adversarial training against
a universal adversarial perturbation (Moosavi-Dezfooli et al., 2017) targeted for all images in the
dataset. In order to encourage the sparsity of the attack, we also impose an l1 regularization on the
attack. The problem formulation is given in Appendix A.3. We compare FeDualEx against direct
aggregation of projected gradient descent ascent (PGDA) proposed in (Shafahi et al., 2020) Alg. 3.

We evaluate convergence with training loss, which is by no means an exact reflection of the duality
gap. Still, we observe in Figure 6 that FeDualEx converges faster and delivers a better-hardened
model with higher validation accuracy on unattacked data. Meanwhile, the vanilla aggregation of
PGDA solutions yields a dense attack whereas FeDualEx achieves much better sparsity, as visualized
in Figure 7. Furthermore, we observe that the attack generated by distributed PGDA is not only dense
but also smoothed out to small values close to zero, averaged by the number of clients.

7 CONCLUSION AND FUTURE WORK

We advance distributed optimization to the broad class of composite SPP by proposing FeDualEx and
providing, to our knowledge, the first convergence rate of its kind. We demonstrate the effectiveness
of FeDualEx for inducing structures with empirical evaluation. We also show that the sequential
version of FeDualEx provides a solution to composite stochastic saddle point optimization in the non-
Euclidean setting. We recognize further study of the heterogeneous federated setting of composite
saddle point optimization would be a challenging direction for future work.

9

51015Communication Rounds0.51.01.5Loss51015Communication Rounds7080AccuracyFeDualExProjected Gradient Descent Ascent02040Communication Rounds2.02.2Loss02040Communication Rounds102030AccuracyPublished as a conference paper at ICLR 2024

REFERENCES

Jacob Abernethy, Kevin A. Lai, Kfir Y. Levy, and Jun-Kun Wang. Faster rates for convex-concave
games. In Sébastien Bubeck, Vianney Perchet, and Philippe Rigollet (eds.), Proceedings of the
31st Conference On Learning Theory, volume 75 of Proceedings of Machine Learning Research,
pp. 1595–1625. PMLR, 06–09 Jul 2018. URL https://proceedings.mlr.press/v75/
abernethy18a.html.

Kimon Antonakopoulos, Veronica Belmega, and Panayotis Mertikopoulos. An adaptive mirror-prox
method for variational inequalities with singular operators. Advances in Neural Information
Processing Systems, 32, 2019.

K. J. Arrow, L. Hurwicz, and H. Uzawa. Studies in linear and non-linear programming. Stanford

University Press, 1958.

Jean-François Aujol and Antonin Chambolle. Dual norms and image decomposition models. Interna-

tional journal of computer vision, 63:85–104, 2005.

Necdet Serhat Aybat and Erfan Yazdandoost Hamedani. A primal-dual method for conic constrained
distributed optimization problems. Advances in neural information processing systems, 29, 2016.

Site Bai, Chuyang Ke, and Jean Honorio. On the dual problem of convexified convolutional
neural networks. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL
https://openreview.net/forum?id=0yMuNezwJ1.

Amir Beck and Marc Teboulle. Mirror descent and nonlinear projected subgradient methods for

convex optimization. Operations Research Letters, 31(3):167–175, 2003.

Aleksandr Beznosikov and Alexander Gasnikov. Similarity, compression and local steps: Three
arXiv preprint

pillars of efficient communications for distributed variational inequalities.
arXiv:2302.07615, 2023.

Aleksandr Beznosikov, Valentin Samokhin, and Alexander Gasnikov. Distributed saddle-point
problems: Lower bounds, optimal and robust algorithms. arXiv preprint arXiv:2010.13112, 2020.

Aleksandr Beznosikov, Gesualdo Scutari, Alexander Rogozin, and Alexander Gasnikov. Distributed
saddle-point problems under data similarity. Advances in Neural Information Processing Systems,
34:8172–8184, 2021.

Aleksandr Beznosikov, Pavel Dvurechenskii, Anastasiia Koloskova, Valentin Samokhin, Sebastian U
Stich, and Alexander Gasnikov. Decentralized local stochastic extra-gradient for variational
inequalities. Advances in Neural Information Processing Systems, 35:38116–38133, 2022.

Ekaterina Borodich, Vladislav Tominin, Yaroslav Tominin, Dmitry Kovalev, Alexander Gasnikov,
and Pavel Dvurechensky. Accelerated variance-reduced methods for saddle-point problems. EURO
Journal on Computational Optimization, 10:100048, 2022.

Ekaterina Borodich, Georgiy Kormakov, Dmitry Kovalev, Aleksandr Beznosikov, and Alexander
Gasnikov. Optimal algorithm with complexity separation for strongly convex-strongly concave
composite saddle point problems. arXiv preprint arXiv:2307.12946, 2023.

Stephen P Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.

Kristian Bredies, Dirk A Lorenz, and Stefan Reiterer. Minimization of non-smooth, non-convex
functionals by iterative thresholding. Journal of Optimization Theory and Applications, 165:
78–112, 2015.

L.M. Bregman. The relaxation method of finding the common point of convex sets and its ap-
plication to the solution of problems in convex programming. USSR Computational Mathe-
matics and Mathematical Physics, 7(3):200–217, 1967. ISSN 0041-5553. doi: https://doi.org/
10.1016/0041-5553(67)90040-7. URL https://www.sciencedirect.com/science/
article/pii/0041555367900407.

10

Published as a conference paper at ICLR 2024

Antoni Buades, Bartomeu Coll, and Jean-Michel Morel. A review of image denoising algorithms,

with a new one. Multiscale modeling & simulation, 4(2):490–530, 2005.

Sébastien Bubeck et al. Convex optimization: Algorithms and complexity. Foundations and Trends®

in Machine Learning, 8(3-4):231–357, 2015.

Brian Bullins and Kevin A Lai. Higher-order methods for convex-concave min-max optimization and

monotone variational inequalities. SIAM Journal on Optimization, 32(3):2208–2229, 2022.

Brian Bullins, Kshitij Patel, Ohad Shamir, Nathan Srebro, and Blake E Woodworth. A stochastic
newton algorithm for distributed convex optimization. Advances in Neural Information Processing
Systems, 34:26818–26830, 2021.

Jian-Feng Cai, Emmanuel J Candès, and Zuowei Shen. A singular value thresholding algorithm for

matrix completion. SIAM Journal on optimization, 20(4):1956–1982, 2010.

Xuanyu Cao, Tamer Ba¸sar, Suhas Diggavi, Yonina C Eldar, Khaled B Letaief, H Vincent Poor, and
Junshan Zhang. Communication-efficient distributed learning: An overview. IEEE journal on
selected areas in communications, 2023.

Y Censor and SA Zenios. Proximal minimization algorithm with d-functions. Journal of Optimization

Theory and Applications, 73(3):451–464, 1992.

A. Chambolle and Thomas Pock. A first-order primal-dual algorithm for convex problems with

applications to imaging. Journal of Mathematical Imaging and Vision, 40:120–145, 2011.

Antonin Chambolle and Thomas Pock. On the ergodic convergence rates of a first-order primal–dual

algorithm. Mathematical Programming, 159(1-2):253–287, 2016.

Cheng Chen, Luo Luo, Weinan Zhang, and Yong Yu. Efficient projection-free algorithms for saddle
point problems. Advances in Neural Information Processing Systems, 33:10799–10808, 2020.

Pin-Yu Chen and Cho-Jui Hsieh. Chapter 12 - adversarial training. In Pin-Yu Chen and Cho-Jui Hsieh
(eds.), Adversarial Robustness for Machine Learning, pp. 119–125. Academic Press, 2023. ISBN
978-0-12-824020-5. doi: https://doi.org/10.1016/B978-0-12-824020-5.00023-5. URL https:
//www.sciencedirect.com/science/article/pii/B9780128240205000235.

Michael B. Cohen, Aaron Sidford, and Kevin Tian. Relative lipschitzness in extragradient methods
In James R. Lee (ed.), 12th Innovations in Theoretical
and a direct recipe for acceleration.
Computer Science Conference, ITCS 2021, January 6-8, 2021, Virtual Conference, volume 185 of
LIPIcs, pp. 62:1–62:18. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2021. doi: 10.4230/
LIPIcs.ITCS.2021.62. URL https://doi.org/10.4230/LIPIcs.ITCS.2021.62.

Patrick L Combettes and Jean-Christophe Pesquet. Primal-dual splitting algorithm for solving
inclusions with mixtures of composite, lipschitzian, and parallel-sum type monotone operators.
Set-Valued and variational analysis, 20(2):307–330, 2012.

Alexandros G Dimakis, Anand D Sarwate, and Martin J Wainwright. Geographic gossip: Efficient
aggregation for sensor networks. In Proceedings of the 5th international conference on Information
processing in sensor networks, pp. 69–76, 2006.

John C Duchi, Shai Shalev-Shwartz, Yoram Singer, and Ambuj Tewari. Composite objective mirror

descent. In Conference on Learning Theory (COLT), volume 10, pp. 14–26. Citeseer, 2010.

John C Duchi, Alekh Agarwal, and Martin J Wainwright. Dual averaging for distributed optimization:
IEEE Transactions on Automatic control, 57(3):

Convergence analysis and network scaling.
592–606, 2011.

Nicolas Flammarion and Francis Bach. Stochastic composite least-squares regression with conver-

gence rate o(1/n). In Conference on Learning Theory (COLT), pp. 831–875. PMLR, 2017.

Margalit R Glasgow, Honglin Yuan, and Tengyu Ma. Sharp bounds for federated averaging (local sgd)
and continuous perspective. In International Conference on Artificial Intelligence and Statistics,
pp. 9050–9090. PMLR, 2022.

11

Published as a conference paper at ICLR 2024

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sher-
jil Ozair, Aaron Courville, and Yoshua Bengio.
In
Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K.Q. Weinberger (eds.), Ad-
vances in Neural Information Processing Systems, volume 27. Curran Associates, Inc.,
URL https://proceedings.neurips.cc/paper_files/paper/2014/
2014.
file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf.

Generative adversarial nets.

Vipul Gupta, Avishek Ghosh, Michał Derezi´nski, Rajiv Khanna, Kannan Ramchandran, and
Michael W. Mahoney. Localnewton: Reducing communication rounds for distributed learn-
ing. In Cassio de Campos and Marloes H. Maathuis (eds.), Proceedings of the Thirty-Seventh
Conference on Uncertainty in Artificial Intelligence, volume 161 of Proceedings of Machine
Learning Research, pp. 632–642. PMLR, 27–30 Jul 2021. URL https://proceedings.
mlr.press/v161/gupta21a.html.

Farzin Haddadpour, Mohammad Mahdi Kamani, Mehrdad Mahdavi, and Viveck Cadambe. Local
sgd with periodic averaging: Tighter analysis and adaptive synchronization. Advances in Neural
Information Processing Systems, 32, 2019.

Trevor Hastie, Robert Tibshirani, and Martin Wainwright. Statistical learning with sparsity: the lasso

and generalizations. CRC press, 2015.

Niao He, Anatoli Juditsky, and Arkadi Nemirovski. Mirror prox algorithm for multi-term composite
minimization and semi-separable problems. Computational Optimization and Applications, 61:
275–319, 2015.

Yunlong He and Renato DC Monteiro. Accelerating block-decomposition first-order methods for
solving composite saddle-point and two-player nash equilibrium problems. SIAM Journal on
Optimization, 25(4):2182–2211, 2015.

Yunlong He and Renato DC Monteiro. An accelerated hpe-type algorithm for a class of composite

convex-concave saddle-point problems. SIAM Journal on Optimization, 26(1):29–56, 2016.

Jean-Baptiste Hiriart-Urruty and Claude Lemaréchal. Fundamentals of convex analysis. Springer

Science & Business Media, 2004.

Charlie Hou, Kiran K Thekumparampil, Giulia Fanti, and Sewoong Oh. Efficient algorithms for

federated saddle point optimization. arXiv preprint arXiv:2102.06333, 2021.

Ruichen Jiang and Aryan Mokhtari. Generalized optimistic methods for convex-concave saddle point

problems. arXiv preprint arXiv:2202.09674, 2022.

Anatoli Juditsky, Arkadi Nemirovski, and Claire Tauvel. Solving variational inequalities with

stochastic mirror-prox algorithm. Stochastic Systems, 1(1):17–58, 2011.

Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin
Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Ad-
vances and open problems in federated learning. Foundations and Trends® in Machine Learning,
14(1–2):1–210, 2021.

Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and
Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning. In
International Conference on Machine Learning, pp. 5132–5143. PMLR, 2020.

Ahmed Khaled, Konstantin Mishchenko, and Peter Richtárik. Tighter theory for local sgd on identical
and heterogeneous data. In International Conference on Artificial Intelligence and Statistics, pp.
4519–4529. PMLR, 2020.

Jakub Koneˇcn`y, H Brendan McMahan, Felix X Yu, Peter Richtárik, Ananda Theertha Suresh, and
Dave Bacon. Federated learning: Strategies for improving communication efficiency. NeurIPS
Private Multi-Party Machine Learning Workshop, 2016.

G.M. Korpelevich. The extragradient method for finding saddle points and other problem. Ekonomika

i Matematicheskie Metody, 12:C747–C756, 1976.

12

Published as a conference paper at ICLR 2024

Georgios Kotsalis, Guanghui Lan, and Tianjiao Li. Simple and optimal methods for stochastic
variational inequalities, i: operator extrapolation. SIAM Journal on Optimization, 32(3):2041–
2073, 2022.

D. Kovalev, Elnur Gasanov, Peter Richtárik, and Alexander V. Gasnikov. Lower bounds and optimal
algorithms for smooth and strongly convex decentralized optimization over time-varying networks.
In Neural Information Processing Systems, 2021a.

Dmitry Kovalev, Egor Shulgin, Peter Richtárik, Alexander V Rogozin, and Alexander Gasnikov.
Adom: Accelerated decentralized optimization method for time-varying networks. In International
Conference on Machine Learning, pp. 5784–5793. PMLR, 2021b.

Dmitry Kovalev, Aleksandr Beznosikov, Ekaterina Borodich, Alexander Gasnikov, and Gesualdo
Scutari. Optimal gradient sliding and its application to optimal distributed optimization under
similarity. Advances in Neural Information Processing Systems, 35:33494–33507, 2022.

Sucheol Lee and Donghwan Kim. Fast extra gradient methods for smooth structured nonconvex-
nonconcave minimax problems. Advances in Neural Information Processing Systems, 34:22588–
22600, 2021.

Guoyin Li and Ting Kei Pong. Global convergence of splitting methods for nonconvex composite

optimization. SIAM Journal on Optimization, 25(4):2434–2460, 2015.

Li Li, Yuxi Fan, Mike Tse, and Kuo-Yi Lin. A review of applications in federated learning. Computers

& Industrial Engineering, 149:106854, 2020a.

Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, and Zhihua Zhang. On the convergence of
fedavg on non-iid data. In International Conference on Learning Representations, 2020b. URL
https://openreview.net/forum?id=HJxNAnVtDS.

Tianyi Lin, Chi Jin, and Michael I Jordan. Near-optimal algorithms for minimax optimization. In

Conference on Learning Theory, pp. 2738–2779. PMLR, 2020.

Changxin Liu, Zirui Zhou, Jian Pei, Yong Zhang, and Yang Shi. Decentralized composite optimization
in stochastic networks: A dual averaging approach with linear convergence. IEEE Transactions on
Automatic Control, 2022.

Weijie Liu, Aryan Mokhtari, Asuman Ozdaglar, Sarath Pattathil, Zebang Shen, and Nenggan Zheng.
A decentralized proximal point-type method for saddle point problems. OPT2020: 12th Annual
Workshop on Optimization for Machine Learning, 2020.

Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. To-
wards deep learning models resistant to adversarial attacks. In International Conference on Learn-
ing Representations, 2018. URL https://openreview.net/forum?id=rJzIBfZAb.

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas.
Communication-Efficient Learning of Deep Networks from Decentralized Data. In Aarti Singh
and Jerry Zhu (eds.), Proceedings of the 20th International Conference on Artificial Intelligence
and Statistics, volume 54 of Proceedings of Machine Learning Research, pp. 1273–1282. PMLR,
20–22 Apr 2017. URL https://proceedings.mlr.press/v54/mcmahan17a.html.

Panayotis Mertikopoulos, Bruno Lecouat, Houssam Zenati, Chuan-Sheng Foo, Vijay Chandrasekhar,
and Georgios Piliouras. Optimistic mirror descent in saddle-point problems: Going the extra(-
gradient) mile. In International Conference on Learning Representations, 2019. URL https:
//openreview.net/forum?id=Bkg8jjC9KQ.

Konstantin Mishchenko, Dmitry Kovalev, Egor Shulgin, Peter Richtárik, and Yura Malitsky. Revisit-
ing stochastic extragradient. In International Conference on Artificial Intelligence and Statistics,
pp. 4573–4582. PMLR, 2020.

Konstantin Mishchenko, Grigory Malinovsky, Sebastian Stich, and Peter Richtárik. Proxskip: Yes!
In International

local gradient steps provably lead to communication acceleration! finally!
Conference on Machine Learning, pp. 15750–15769. PMLR, 2022.

13

Published as a conference paper at ICLR 2024

Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, and Pascal Frossard. Deepfool: A simple and
accurate method to fool deep neural networks. In 2016 IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), pp. 2574–2582, 2016. doi: 10.1109/CVPR.2016.282.

Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, and Pascal Frossard. Universal
adversarial perturbations. In Proceedings of the IEEE conference on computer vision and pattern
recognition, pp. 1765–1773, 2017.

Angelia Nedich et al. Convergence rate of distributed averaging dynamics and optimization in

networks. Foundations and Trends® in Systems and Control, 2(1):1–100, 2015.

Arkadi Nemirovski. Prox-method with rate of convergence o (1/t) for variational inequalities with
lipschitz continuous monotone operators and smooth convex-concave saddle point problems. SIAM
Journal on Optimization, 15(1):229–251, 2004.

Arkadij Semenoviˇc Nemirovskij and David Borisovich Yudin. Problem complexity and method

efficiency in optimization. Wiley-Interscience, 1983.

Yu Nesterov. Smooth minimization of non-smooth functions. Mathematical programming, 103:

127–152, 2005.

Yurii Nesterov. Dual extrapolation and its applications to solving variational inequalities and related

problems. Mathematical Programming, 109(2-3):319–344, 2007.

Yurii Nesterov. Primal-dual subgradient methods for convex problems. Mathematical programming,

120(1):221–259, 2009.

Yuyuan Ouyang and Yangyang Xu. Lower complexity bounds of first-order methods for convex-

concave bilinear saddle-point problems. Mathematical Programming, 185(1-2):1–35, 2021.

Leonid Denisovich Popov. A modification of the arrow-hurwicz method for search of saddle points.

Mathematical notes of the Academy of Sciences of the USSR, 28:845–848, 1980.

Michael Rabbat. Multi-agent mirror descent for decentralized stochastic optimization. In 2015 IEEE
6th International Workshop on Computational Advances in Multi-Sensor Adaptive Processing
(CAMSAP), pp. 517–520. IEEE, 2015.

Ali Ramezani-Kebrya, Kimon Antonakopoulos, Igor Krawczuk, Justin Deschenaux, and Volkan
Cevher. Distributed extra-gradient with optimal complexity and communication guarantees.
In The Eleventh International Conference on Learning Representations, 2023. URL https:
//openreview.net/forum?id=b3itJyarLM0.

R. Tyrrell Rockafellar. Convex Analysis. Princeton Landmarks in Mathematics and Physics. Princeton

University Press, 1970. ISBN 978-1-4008-7317-3.

Alexander Rogozin, Aleksandr Beznosikov, Darina Dvinskikh, Dmitry Kovalev, Pavel Dvurechensky,
and Alexander Gasnikov. Decentralized distributed optimization for saddle point problems. arXiv
preprint arXiv:2102.07758, 2021.

Mher Safaryan, Rustem Islamov, Xun Qian, and Peter Richtarik. Fednl: Making newton-type
methods applicable to federated learning. In International Conference on Machine Learning, pp.
18959–19010. PMLR, 2022.

Arda Sahiner, Tolga Ergen, Batu Ozturkler, Burak Bartan, John M. Pauly, Morteza Mardani, and
Mert Pilanci. Hidden convexity of wasserstein GANs: Interpretable generative models with
closed-form solutions. In International Conference on Learning Representations, 2022. URL
https://openreview.net/forum?id=e2Lle5cij9D.

Ali Shafahi, Mahyar Najibi, Zheng Xu, John Dickerson, Larry S Davis, and Tom Goldstein. Universal
adversarial training. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34,
pp. 5636–5643, 2020.

Ohad Shamir, Nati Srebro, and Tong Zhang. Communication-efficient distributed optimization
using an approximate newton-type method. In International conference on machine learning, pp.
1000–1008. PMLR, 2014.

14

Published as a conference paper at ICLR 2024

Pranay Sharma, Rohan Panda, Gauri Joshi, and Pramod Varshney. Federated minimax optimization:
Improved convergence analyses and algorithms. In International Conference on Machine Learning,
pp. 19683–19730. PMLR, 2022.

Pranay Sharma, Rohan Panda, and Gauri Joshi. Federated minimax optimization with client hetero-

geneity. arXiv preprint arXiv:2302.04249, 2023.

Yan Shen, Jian Du, Han Zhao, Benyu Zhang, Zhanghexuan Ji, and Mingchen Gao. Fedmm: Sad-
dle point optimization for federated adversarial domain adaptation. In The 22nd International
Conference on Autonomous Agents and Multiagent Systems (AAMAS), 2023.

Zhan Shi, Xinhua Zhang, and Yaoliang Yu. Bregman divergence for stochastic variance reduction:
saddle-point and adversarial prediction. Advances in Neural Information Processing Systems, 30,
2017.

Mircea Sofonea and Andaluzia Matei. Variational inequalities with applications: a study of antiplane

frictional contact problems, volume 18. Springer Science & Business Media, 2009.

Mikhail V. Solodov and Benar Fux Svaiter. A hybrid approximate extragradient – proximal point
algorithm using the enlargement of a maximal monotone operator. Set-Valued Analysis, 7:323–345,
1999.

Chaobing Song, Zhengyuan Zhou, Yichao Zhou, Yong Jiang, and Yi Ma. Optimistic dual extrapolation
for coherent non-monotone variational inequalities. Advances in Neural Information Processing
Systems, 33:14303–14314, 2020.

Sebastian U. Stich. Local SGD converges fast and communicates little. In International Confer-
ence on Learning Representations, 2019. URL https://openreview.net/forum?id=
S1g2JnRcFX.

Gilbert Strang. Linear algebra and its applications. Belmont, CA: Thomson, Brooks/Cole, 2006.

Robert Tibshirani. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical
Society. Series B (Methodological), 58(1):267–288, 1996. ISSN 00359246. URL http://www.
jstor.org/stable/2346178.

Vladislav Tominin, Yaroslav Tominin, Ekaterina Borodich, Dmitry Kovalev, Alexander Gasnikov, and
Pavel Dvurechensky. On accelerated methods for saddle-point problems with composite structure.
arXiv preprint arXiv:2103.09344, 2021.

Qianqian Tong, Guannan Liang, Tan Zhu, and Jinbo Bi. Federated nonconvex sparse learning. arXiv

preprint arXiv:2101.00052, 2020.

Quoc Tran Dinh, Nhan H Pham, Dzung Phan, and Lam Nguyen. Feddr–randomized douglas-
rachford splitting algorithms for nonconvex federated composite optimization. Advances in Neural
Information Processing Systems, 34:30326–30338, 2021.

Paul Tseng. On accelerated proximal gradient methods for convex-concave optimization. submitted

to SIAM Journal on Optimization, 2(3), 2008.

Hoi-To Wai, Zhuoran Yang, Zhaoran Wang, and Mingyi Hong. Multi-agent reinforcement learning via
double averaging primal-dual optimization. Advances in Neural Information Processing Systems,
31, 2018.

Jianyu Wang, Zachary Charles, Zheng Xu, Gauri Joshi, H Brendan McMahan, Maruan Al-Shedivat,
Galen Andrew, Salman Avestimehr, Katharine Daly, Deepesh Data, et al. A field guide to federated
optimization. arXiv preprint arXiv:2107.06917, 2021.

Blake Woodworth, Kumar Kshitij Patel, Sebastian Stich, Zhen Dai, Brian Bullins, Brendan Mcmahan,
In International

Is local sgd better than minibatch sgd?

Ohad Shamir, and Nathan Srebro.
Conference on Machine Learning, pp. 10334–10343. PMLR, 2020a.

Blake E Woodworth, Kumar Kshitij Patel, and Nati Srebro. Minibatch vs local sgd for heterogeneous
distributed learning. Advances in Neural Information Processing Systems, 33:6281–6292, 2020b.

15

Published as a conference paper at ICLR 2024

Lin Xiao. Dual averaging methods for regularized stochastic learning and online optimization. The

Journal of Machine Learning Research, 11:2543–2596, 2010.

Tesi Xiao, Xuxing Chen, Krishnakumar Balasubramanian, and Saeed Ghadimi. A one-sample
decentralized proximal algorithm for non-convex stochastic composite optimization. In Robin J.
Evans and Ilya Shpitser (eds.), Proceedings of the Thirty-Ninth Conference on Uncertainty in
Artificial Intelligence, volume 216 of Proceedings of Machine Learning Research, pp. 2324–
2334. PMLR, 31 Jul–04 Aug 2023. URL https://proceedings.mlr.press/v216/
xiao23a.html.

Jinming Xu, Ye Tian, Ying Sun, and Gesualdo Scutari. Distributed algorithms for composite opti-
mization: Unified framework and convergence analysis. IEEE Transactions on Signal Processing,
69:3555–3570, 2021. doi: 10.1109/TSP.2021.3086579.

Yonggui Yan, Jie Chen, Pin-Yu Chen, Xiaodong Cui, Songtao Lu, and Yangyang Xu. Compressed
decentralized proximal stochastic gradient method for nonconvex composite problems with hetero-
geneous data. In International Conference on Machine Learning, 2023.

Hao Yu, Sen Yang, and Shenghuo Zhu. Parallel restarted sgd with faster convergence and less
communication: Demystifying why model averaging works for deep learning. In Proceedings of
the AAAI Conference on Artificial Intelligence, volume 33, pp. 5693–5700, 2019.

Honglin Yuan and Tengyu Ma. Federated accelerated stochastic gradient descent. Advances in Neural

Information Processing Systems, 33:5332–5344, 2020.

Honglin Yuan, Manzil Zaheer, and Sashank Reddi. Federated composite optimization. In International

Conference on Machine Learning, pp. 12253–12266. PMLR, 2021.

Fan Zhou and Guojing Cong. On the convergence properties of a k-step averaging stochastic
gradient descent algorithm for nonconvex optimization. In Proceedings of the Twenty-Seventh
International Joint Conference on Artificial Intelligence, IJCAI-18, pp. 3219–3227. International
Joint Conferences on Artificial Intelligence Organization, 7 2018. doi: 10.24963/ijcai.2018/447.
URL https://doi.org/10.24963/ijcai.2018/447.

Kang Zhou, Shenghua Gao, Jun Cheng, Zaiwang Gu, Huazhu Fu, Zhi Tu, Jianlong Yang, Yitian
Zhao, and Jiang Liu. Sparse-gan: Sparsity-constrained generative adversarial network for anomaly
detection in retinal oct image. In 2020 IEEE 17th International Symposium on Biomedical Imaging
(ISBI), pp. 1227–1231. IEEE, 2020.

Martin Zinkevich, Markus Weimer, Lihong Li, and Alex Smola. Parallelized stochastic gradient

descent. Advances in neural information processing systems, 23, 2010.

16

Published as a conference paper at ICLR 2024

Appendices
In Appendix A, we provide details on experiment settings and additional experiments on the universal
adversarial training of non-convex convolutional neural networks. In Appendix B, an extended
literature review on various related subfields is included. Appendix C and D provide additional
theoretical background, including relevant preliminaries, definitions, remarks, and technical lemmas.
Appendix E, F, and G provide the convergence rates and complete proofs for FeDualEx in federated
composite saddle point optimization, federated composite convex optimization, sequential stochastic
composite optimization, and sequential deterministic composite optimization respectively. Finally,
the algorithm of FedMiP is presented in Appendix H.

A Additional Experiments and Setup Details

A.1 Setup Details for Saddle Point Optimization with Sparsity Regularization . . . . .

A.2 Saddle Point Optimization with Low-Rank Regularization . . . . . . . . . . . . .

A.3 Universal Adversarial Training of Logistic Regression . . . . . . . . . . . . . . . .

A.4 Universal Adversarial Training of Neural Networks . . . . . . . . . . . . . . . . .

B Extended Literature Review

B.1 Distributed Optimization / Federated Learning . . . . . . . . . . . . . . . . . . . .

B.2 Saddle Point Optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

B.3 Composite Optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

B.4 Other Tangentially Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . .

C Additional Preliminaries, Definitions, and Remarks on Assumptions

C.1 Additional Preliminaries

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

C.1.1 Mirror Descent and Dual Averaging . . . . . . . . . . . . . . . . . . . . .

C.1.2 Mirror Prox and Dual Extrapolation . . . . . . . . . . . . . . . . . . . . .

C.2 Additional Definitions .

.

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

C.3 Formal Assumptions and Remarks . . . . . . . . . . . . . . . . . . . . . . . . . .

D Additional Technical Lemmas

E Complete Analysis of FeDualEx for Composite Saddle Point Problems

E.1 Main Theorem and Proof . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

E.2 Helping Lemmas

.

.

.

.

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

F Complete Analysis of FeDualEx for Composite Convex Optimization

G FeDualEx in Other Settings

G.1 Stochastic Dual Extrapolation for Composite Saddle Point Optimization . . . . . .

G.2 Deterministic Dual Extrapolation for Composite Saddle Point Optimization . . . .

H Federated Mirror Prox

18

18

19

20

21

21

21

21

22

23

23

23

24

25

26

26

27

29

29

33

36

39

39

41

42

17

Published as a conference paper at ICLR 2024

A ADDITIONAL EXPERIMENTS AND SETUP DETAILS

A.1 SETUP DETAILS FOR SADDLE POINT OPTIMIZATION WITH SPARSITY REGULARIZATION

We provide additional details for the SPP with the sparsity regularization demonstrated in the main
text. We start by restating its formulation below:

max
y∈Y

min
x∈X
A ∈ Rn×m,
b ∈ Rn,

⟨Ax − b, y⟩ + λ∥x∥1 − λ∥y∥1

X = {Rm : ∥x∥∞ ≤ D},
Y = {Rn : ∥y∥∞ ≤ D}.

Soft-Thresholding Operator for ℓ1 Norm Regularization. By choosing the distance-generating
function to be ℓ = 1
r,k(·) instantiates to the following element-wise
soft-thresholding operator (Hastie et al., 2015; Jiang & Mokhtari, 2022):

2, the projection ∇ℓ∗

2 ∥y∥2

2 ∥x∥2

2 + 1

Tλ′(ω) :=






0
(|ω| − λ′) · sgn(ω)
D · sgn(ω)

if |ω| ≤ λ′
if λ′ < |ω| ≤ λ′ + D
otherwise

,

in which λ′ = ληc(ηsrK + k).

Closed-Form Duality Gap. The closed-form duality gap is given by

Gap(x, y) = D∥(|Ax − b| − λ)+∥1 + λ∥x∥1 + D∥(|A⊤y| − λ)+∥1 + ⟨b, y⟩ + λ∥y∥1,

where | · | and ()+ = max{·, 0} are element-wise. We provide a brief derivation below. Since a
constraint is equivalent to an indicator regularization, we move the ℓ∞ constraint into the objective
if ∥ · ∥∞ ≤ D

(cid:26)0

. By the definitions of duality gap in

and denote g1(·) = ∥ · ∥1, g2(·) =

∞ otherwise

Definition 1 and convex conjugate in Definition 9, the duality gap equals to

Gap(x, y) = max

y

λ{⟨

1
λ

(Ax − b), y⟩ − g1(y) − g2(y) + ∥x∥1}

(A⊤y), x⟩ + g1(x) + g2(x) − ∥y∥1 −

1
λ

b⊤y}

− min
x

λ{⟨

1
λ
1
λ
1(u) + λg∗
2(

= inf
u

{λg∗

= λ(g1 + g2)∗(

(Ax − b)) + λ(g1 + g2)∗(

1
λ

(Ax − b) − u)} + inf
v

1
λ

(A⊤y)) + λ∥x∥1 + λ∥y∥1 + b⊤y
1
λ

1(v) + λg∗
2(

(A⊤y) − v)}

{λg∗

+ λ∥x∥1 + λ∥y∥1 + b⊤y,

in which the last equality holds by Theorem 2.3.2, namely infimal convolution, in Chapter E of
Hiriart-Urruty & Lemaréchal (2004). By definition of the convex conjugate, the convex conjugate

(cid:26)0

if ∥ · ∥q ≤ 1

∞ otherwise

of a norm g(·) = ∥ · ∥p is defined to be g∗(·) =

, in which ∥ · ∥q is the dual

norm of ∥ · ∥p. Given that ℓ1 and ℓ∞ are dual norms to each other, g∗

1(·) =

(cid:26)0

if ∥ · ∥∞ ≤ 1

∞ otherwise

g∗
2(·) = D∥ · ∥1. Therefore the infimum is achieved when ∀i ∈ [m], ∀j ∈ [n],

ui =

(cid:26) 1

λ (Ax − b)i
sgn( 1

λ (Ax − b)i) otherwise

if | 1

λ (Ax − b)i| ≤ 1

,

vj =

(cid:26) 1

λ (A⊤y)j
sgn( 1

λ (A⊤y)j) otherwise

if | 1

λ (A⊤y)j| ≤ 1

,

,

which yields the closed-form duality gap.

Additional Experiment Details. We generate a fixed pair of A and b with each entry independently
following the uniform distribution U[−1,1]. Each entry of the variables x and y is initialized indepen-
dently from the distribution U[−D,D]. As in (Jiang & Mokhtari, 2022), we take m = 600, n = 300,
λ = 0.1, D = 0.05. For DO, we simulate M = 100 clients. For the gradient query of each client

18

Published as a conference paper at ICLR 2024

in each local update, we inject a Gaussian noise from N (0, σ2). All M = 100 clients participate in
each round; noise on each client is i.i.d. with σ = 0.1.
We only tune the global step size ηs and the local step size ηc. For all experiments, the parameters
are searched from the combination of ηs ∈ {1, 3e − 1, 1e − 1, 3e − 2, 1e − 2} and ηc ∈ {1, 3e −
1, 1e − 1, 3e − 2, 1e − 2, 3e − 3, 1e − 3}. We run each setting for 10 different random seeds and
report the mean and standard deviation in Figure 4.

A.2 SADDLE POINT OPTIMIZATION WITH LOW-RANK REGULARIZATION

We test FeDualEx on the following SPP with nuclear norm regularization for low-rankness, in which
we overuse the notation ∥ · ∥∗ for the matrix nuclear norm and ∥ · ∥2 for the matrix spectral norm. We
use Tr(·) to denote the trace of a square matrix. And for the purpose of feasibility and convenience,
we impose spectral norm constraints on the variables as well.

max
Y∈Y

min
X∈X
A ∈ Rn×m,
B ∈ Rn×p,

Tr(cid:0)(AX − B)⊤Y(cid:1) + λ∥X∥∗ − λ∥Y∥∗
X = {Rm×p : ∥X∥2 ≤ D},
Y = {Rn×p : ∥Y∥2 ≤ D}.

Soft-Thresholding Operator for Nuclear Norm Regularization. By choosing the distance-
generating function to be ℓ = 1
F where ∥ · ∥F denotes the Frobenius norm, the
projection ∇ℓ∗
r,k(·) instantiates to the following element-wise singular value soft-thresholding opera-
tor (Cai et al., 2010):

2 ∥Y∥2

2 ∥X∥2

F + 1

Tλ′(W) := UTλ′(Σ)V⊤, Tλ′(Σ) = diag(sgn(σi(W)) · min{max{σi(W) − λ′, 0}, D}),

in which λ′ = ληc(ηsrK + k), W = UΣV⊤ is the singular value decomposition (SVD) of W, and
we overuse the notation σi(·) to represent the singular values.

Closed-Form Duality Gap. The closed-form duality gap is given by
Gap(X, Y) = D∥diag(cid:0)(|σi(AX − B)| − λ)+
+ D∥diag(cid:0)(|σj(A⊤Y)| − λ)+

(cid:1)∥∗ + λ∥X∥∗
(cid:1)∥∗ + Tr(cid:0)B⊤Y(cid:1) + λ∥Y∥∗,

We provide a brief derivation below. Since a constraint is equivalent to an indicator regular-
ization, we move the spectral norm constraint into the objective and denote g1(·) = ∥ · ∥∗,

. By the definitions of duality gap in Definition 1 and convex conjugate

g2(·) =

(cid:26)0

if ∥ · ∥2 ≤ D

∞ otherwise

in Definition 9, the duality gap equals to

Gap(X, Y) = max

Y

(AX − B)⊤Y(cid:1) − g1(Y) − g2(Y) + ∥X∥∗}

λ{Tr(cid:0) 1
λ
λ{{Tr(cid:0) 1
λ
1
λ

− min
X

(A⊤Y)⊤X(cid:1) + g1(X) + g2(X) − ∥Y∥∗ −

Tr(cid:0)B⊤Y(cid:1)}

1
λ

1
λ

(A⊤Y))

= λ(g1 + g2)∗(

(AX − B)) + λ(g1 + g2)∗(

= inf
P

+ λ∥X∥∗ + λ∥Y∥∗ + Tr(cid:0)B⊤Y(cid:1)
1
{λg∗
1(P) + λg∗
2(
λ
+ λ∥X∥∗ + λ∥Y∥∗ + Tr(cid:0)B⊤Y(cid:1),

(AX − B) − P)} + inf
Q

{λg∗

1(Q) + λg∗
2(

1
λ

(A⊤Y) − Q)}

in which the last equality holds by Theorem 2.3.2, namely infimal convolution, in Chapter E of
Hiriart-Urruty & Lemaréchal (2004). By definition of the dual norm, we know that the nuclear

norm and the spectral norm are dual norms to each other. Therefore, g∗

1(·) =

19

(cid:26)0

if ∥ · ∥2 ≤ 1

∞ otherwise

,

Published as a conference paper at ICLR 2024

g∗
2(·) = D∥ · ∥∗. And the infimum is achieved when

(cid:26)σi

σi(P) =

(cid:0) 1
λ (Ax − B)(cid:1)
sgn(cid:0)σi
λ (Ax − B)(cid:1)(cid:1)
(cid:0) 1
λ (A⊤y)(cid:1)
(cid:0) 1
(cid:26)σj
sgn(cid:0)σj
λ (A⊤y)(cid:1)(cid:1)
(cid:0) 1
which yields the closed-form duality gap.

σj(Q) =

λ (Ax − B)(cid:1)| ≤ 1
(cid:0) 1

if |σi
otherwise

λ (A⊤y)(cid:1)| ≤ 1
(cid:0) 1

if |σj
otherwise

,

,

Experiment Settings. We generate a fixed pair of A and B. Each entry of A and half of the columns
in B follows the uniform distribution U[−1,1] independently. Each entry of the variables X and Y
is initialized independently from the distribution U[−1,1]. We take m = 600, n = 300, p = 20,
λ = 0.1, D = 0.05. For DO, we simulate M = 100 clients. For the gradient query of each client
in each local update, we inject a Gaussian noise from N (0, σ2). All M = 100 clients participate
in each round; noise on each client is i.i.d. with σ = 0.1. We only tune the global step size ηs and
the local step size ηc. For all experiments, the parameters are searched from the combination of
ηs ∈ {1, 3e−1, 1e−1, 3e−2, 1e−2} and ηc ∈ {10, 3, 1, 3e−1, 1e−1, 3e−2, 1e−2, 3e−3, 1e−3}.
We run each setting for 10 different random seeds and plot the mean and the standard deviation.

We evaluate the convergence in terms of the duality gap and also demonstrate the rank of the solution,
for both X and Y. For the feasibility of low-rankness, we generate B to be of rank p
2 , i.e. half of
the columns of B is linearly dependent on the other half. With p = 20, the optimal rank for the
solution would most likely be 10. The evaluation is conducted for two different settings: (a) K = 1
local update for R = 100 rounds; (b) K = 10 local updates for R = 20 rounds. The results are
demonstrated in Figure 5 correspondingly.

Discussions. From Figure 5, we can see that in the setting for low-rankness regularization, dual
methods tend to perform better both in minimizing the duality gap and in encouraging a low-rank
solution. In particular, FeDualEx, as a method geared for saddle point optimization, demonstrates
better convergence in the duality gap than FedDualAvg. In the meantime, the solution given by
FeDualEx quickly reaches the optimal rank of 10. This further reveals the potential of FeDualEx in
coping with a variety of regularization and constraints.

A.3 UNIVERSAL ADVERSARIAL TRAINING OF LOGISTIC REGRESSION

We provide the problem formulation and detailed experiment setting for the universal adversarial
training of logistic regression demonstrated in the main text.

Problem Formulation. As introduced, we impose an l1 regularization on the attack to encourage
sparsity in addition to the ball constraint. The problem can be formulated as the following SPP:

min
w∈Rd

max
∥δ∥∞≤D

1
n

n
(cid:88)

i=1

ℓ(w⊤(xi + δ), yi) + λ∥δ∥1

in which ℓ is the cross-entropy loss for multiclass logistic regression; w ∈ Rd is the parameter;
xi ∈ Rd is the data and yi is the label; δ ∈ Rd is the attack.
Experiment Settings. The training data for MNIST is evenly distributed across M = 100 clients,
each possessing 600. The client makes K = 5 local updates and communicates for R = 20
rounds. For the CIFAR-10 experiments, each of the 100 clients holds 500 of the training data. The
client makes K = 5 local updates and communicates for R = 40 rounds. D = 0.05 for data
normalized between 0 and λ = 0.1. Validation is done on the whole validation dataset on the
server with unattacked data. As before, the hyper-parameters are searched from the combination of
ηs ∈ {1, 3e−1, 1e−1, 3e−2, 1e−2} and ηc ∈ {10, 3, 1, 3e−1, 1e−1, 3e−2, 1e−2, 3e−3, 1e−3}.
We run each setting for 10 different random seeds and plot the mean and the standard deviation in
Figure 6.

Attack Visualization. The attack for MNIST has only one channel and is directly visualized with
the color map from blue to red rescaled between the range of the attack, with blue being negative,
red being positive, and purple being zero. The attack for CIFAR-10 contains 3 channels and can be
directly visualized with RGB mode rescaled between 0 and 255. For the attack to be visible, we
divide the value by its maximum then times the result by 4.

20

Published as a conference paper at ICLR 2024

Figure 8: Training loss and validation accuracy of 3-layer CNN on unattacked data.

A.4 UNIVERSAL ADVERSARIAL TRAINING OF NEURAL NETWORKS

Even though the theoretical result is derived with respect to convex functions, we experimentally
demonstrate the convergence FeDualEx for non-convex functions with the adversarial training of
neural networks on CIFAR-10. The model tested is a 3-layer convolutional neural network (CNN)
with 16, 32, and 64 filters of size 3 × 3, each layer followed by a relu activation and a 2 × 2
max-pooling. The performance is demonstrated in Figure 8. The loss value is by no means an
exact reflection of the duality gap, nevertheless, FeDualEx also converges for non-convex functions,
yielding faster numerical convergence and better-hardened models in terms of validation accuracy on
unattacked data. In addition, the sparsity of the attack generated by FeDualEx is 50.38%, whereas
that by the vanilla distributed version of projected gradient descent ascent is 99.31%.

B EXTENDED LITERATURE REVIEW

B.1 DISTRIBUTED OPTIMIZATION / FEDERATED LEARNING

In recent years, distributed learning has received increasing attention in practice and theory. Earlier
works in the field were known as “parallel” (Zinkevich et al., 2010) or “local” (Zhou & Cong, 2018;
Stich, 2019), which are later recognized as the homogeneous case, where data across clients are
assumed to be balanced and i.i.d. (independent and identically distributed), of federated learning
(FL), specifically, Federated Averaging (FedAvg) (McMahan et al., 2017), DO or FL has been found
appealing in various applications (Li et al., 2020a). On the theoretical front, Stich (2019) provides the
first convergence rate for Local SGD, or, FedAvg under the homogeneous setting. The distributed
optimization paradigm we consider aligns with that in Local SGD (Stich, 2019). The rate for
LocalSGD has been improved with tighter analysis (Haddadpour et al., 2019; Khaled et al., 2020;
Woodworth et al., 2020a; Glasgow et al., 2022) and acceleration techniques (Yuan & Ma, 2020;
Mishchenko et al., 2022). Others also analyze FedAvg under heterogeneity (Haddadpour et al., 2019;
Khaled et al., 2020; Woodworth et al., 2020b) and non-i.i.d. data (Li et al., 2020b) or in light propose
improvements (Karimireddy et al., 2020). Recently, the idea of DO is further extended to higher-order
methods (Bullins et al., 2021; Gupta et al., 2021; Safaryan et al., 2022). Due to the page limit, we
refer the readers to (Cao et al., 2023; Wang et al., 2021; Kairouz et al., 2021) for more comprehensive
reviews of DO and FL. In the meantime, we point out that none of the work mentioned above covers
saddle point problems or non-smooth composite or constrained problems. For distributed saddle
point optimization and federated composite optimization, we defer to the following subsections.

B.2 SADDLE POINT OPTIMIZATION

The study of Saddle Point Optimization dates back to the very early gradient descent ascent (Arrow
et al., 1958). It was later improved by the important ideas of extra-gradient (Korpelevich, 1976) and
optimism (Popov, 1980). In light of these ideas, many algorithms were proposed for SPP (Solodov &
Svaiter, 1999; Nemirovski, 2004; Nesterov, 2007; Chambolle & Pock, 2011; Mertikopoulos et al.,
2019; Jiang & Mokhtari, 2022). Among them, in the convex-concave setting in particular, the most
relevant and prominent ones are Nemirovski’s mirror prox Nemirovski (2004) and Nesterov’s dual
extrapolation Nesterov (2007). They generalize respectively Mirror Descent (Nemirovskij & Yudin,
1983) and Dual Averaging (Nesterov, 2009) from convex optimization to monotone variational
inequalities (VIs) which include SPP as one realization. Along with Tseng’s Accelerated Proximal

21

0255075Communication Rounds1.4×1001.6×1001.8×1002×1002.2×100Loss0255075Communication Rounds2040AccuracyProjected Gradient Descent AscentFeDualExPublished as a conference paper at ICLR 2024

Gradient (Tseng, 2008), they are the three methods that converge to an ϵ-approximate solution in
terms of duality gap at O( 1
T ), the known best rate for a general convex-concave SPP (Ouyang &
Xu, 2021; Lin et al., 2020). Mirror prox inspired many papers (Antonakopoulos et al., 2019; Chen
et al., 2020) and is later extended to the stochastic setting (Juditsky et al., 2011; Mishchenko et al.,
2020), the higher-order setting (Bullins & Lai, 2022), and even the composite setting (He et al., 2015),
whose introduction we defer to the review of composite optimization. Dual extrapolation is later
extended to non-monotone VIs (Song et al., 2020), yet its stochastic and composite versions are, to
the best of our knowledge, not found. Kotsalis et al. (2022) recently studied optimal methods for
stochastic variational inequalities, yet their result is limited to smooth VIs, not composite ones.

From the perspective of distributed optimization, several works have made preliminary progress
for smooth and unconstrained SPP in the Euclidean space. Beznosikov et al. (2020) investigate the
distributed extra-gradient method under various conditions and provide upper and lower bounds
under strongly-convex strongly-concave and non-convex non-concave assumptions. Hou et al. (2021)
propose FedAvg-S and SCAFFOLD-S based on FedAvg (McMahan et al., 2017) and SCAFFOLD
(Karimireddy et al., 2020) for SPP, which achieves similar convergence rate to the distributed extra-
gradient algorithm (Beznosikov et al., 2020) under the strong-convexity-concavity assumption. In
addition, (Ramezani-Kebrya et al., 2023) studies the problem from the information compression
perspective with the measure of communication bits. The topic of distributed or federated saddle
point optimization is also found in recent applications of interest, e.g. adversarial domain adaptation
(Shen et al., 2023). Yet, none of the existing works includes the study for SPP with constraints or
composite possibly non-smooth regularization. Outside of our setting, Borodich et al. (2023) also
studies composite SPP, but assumes composite terms to be smooth as well.

B.3 COMPOSITE OPTIMIZATION

Composite optimization has been an important topic due to its reflection of real-world complexities.
Representative works include composite mirror descent (Duchi et al., 2010) and regularized dual
averaging (Xiao, 2010; Flammarion & Bach, 2017) that generalize mirror descent (Nemirovskij &
Yudin, 1983) and dual averaging (Nesterov, 2009) in the context of composite convex optimization.
Composite saddle point optimization, in comparison, appears dispersedly in early-day problems in
practice (Buades et al., 2005; Aujol & Chambolle, 2005), often as a primal-dual reformulation of
composite convex problems. Solving techniques such as smoothing (Nesterov, 2005) and primal-dual
splitting (Combettes & Pesquet, 2012) were proposed, and numerical speed-ups were studied (He &
Monteiro, 2015; 2016), while systematic convergence analysis on general composite SPP came later
in time (He et al., 2015; Chambolle & Pock, 2016; Jiang & Mokhtari, 2022). Recently, Tominin et al.
(2021); Borodich et al. (2022) also proposed acceleration techniques for composite SPP.

Most related among them, the pioneering composite mirror prox (CoMP) (He et al., 2015) constructs
auxiliary variables for the composite regularization terms as an upper bound and thus moves the
non-smooth term into the problem domain. Observing that the gradient operator for the auxiliary
variable is constant, CoMP operates “as if” there were no composite components at all (He et al.,
2015), and exhibits a O( 1
T ) convergence rate that matches its smooth version (Nemirovski, 2004). In
the stochastic setting, Mishchenko et al. (2020) analyzed a variant of stochastic mirror prox (Juditsky
et al., 2011), which is then capable of handling composite terms in the Euclidean space. In this paper,
we take a different approach that utilizes the generalized Bregman divergence and get the same rate
for composite dual extrapolation.

For distributed composite optimization with local updates, Yuan et al. (2021) study Federated Mirror
Descent, a natural extension of FedAvg that adapts to composite optimization under the convex setting.
Along the way, they identified the “curse of primal averaging” specific to composite optimization in
the DO paradigm, where the regularization-imposed structure on the client models may no longer
hold after server primal averaging. To resolve this issue, they further proposed Federated Dual
Averaging which brings the averaging step to the dual space. Tran Dinh et al. (2021) proposes a
federated Douglas-Rachford splitting algorithm for nonconvex composite optimization. On the less
related constrained optimization topic, Tong et al. (2020) proposed a federated learning algorithm for
nonconvex sparse learning under ℓ0 constraint. To the best of our knowledge, the field of distributed
optimization for composite SPP remains blank, which we regard as the main focus of this paper.

22

Published as a conference paper at ICLR 2024

B.4 OTHER TANGENTIALLY RELATED WORK

Decentralized Optimization. Parallel to FL or DO with local updates, there is another line of
work that studies decentralized optimization or consensus optimization over networks, in which
machines communicate directly with each other based on their topological connectivity (Nedich et al.,
2015). Classic algorithms mentioned previously are widely applied as well under this paradigm, for
example, decentralized mirror descent (Rabbat, 2015) and decentralized (composite) dual averaging
over networks (Duchi et al., 2011; Liu et al., 2022). Further in the context of composite optimization,
Yan et al. (2023); Xiao et al. (2023) focus on composite non-convex objectives under the decentralized
setting. Saddle point optimization has also been studied for decentralized optimization, including
for proximal point-type methods (Liu et al., 2020) and extra-gradient methods (Rogozin et al., 2021;
Beznosikov et al., 2021; 2022). In particular, Rogozin et al. (2021) studies decentralized “mirror prox”
in the Euclidean space. We would like to point out that mirror prox in the Euclidean space reduces to
vanilla extra-gradient methods. In addition, Aybat & Yazdandoost Hamedani (2016); Xu et al. (2021)
study the saddle point reformulation for composite convex objectives over decentralized networks,
which essentially focus on composite convex optimization. In the general context of distributed
learning of composite SPP, by the judgment of the authors, we came across no paper in decentralized
optimization similar to ours. More importantly, decentralized optimization focuses on topics like
time-varying network topology (Kovalev et al., 2021a;b) or gossip schema (Dimakis et al., 2006),
which are fundamentally different from our setting in terms of motivations, communication protocols,
and techniques (Kairouz et al., 2021).

Nonconvex-Nonconcave Saddle Point Problems. For nonconvex-nonconcave SPP, several dis-
tributed learning methods have recently been proposed, including extra-gradient methods (Lee &
Kim, 2021) and the Local Stochastic Gradient Descent Ascent (Local SGDA) (Sharma et al., 2022;
2023). Yet we emphasize that our object of analysis is composite SPP with possibly non-smooth
regularization, and as remarked by Yuan et al. (2021), non-convex optimization for composite possi-
bly non-smooth functions is in itself intricate even for sequential optimization, involving additional
assumptions and sophisticated algorithm design (Li & Pong, 2015; Bredies et al., 2015), let alone
distributed learning of SPP. Thus we focus on convex-concave analysis in this paper.

Finite-Sum Optimization with Function Similarity. Another line of work considers finite-sum
optimization with function similarity, following the setting similar to DANE (Shamir et al., 2014).
In this setting, each machine is assumed to maintain a fixed set of data so that the functions across
machines can be δ-similar with a high probability by large-sample concentration inequalities. In
the context of distributed saddle points optimization, examples include (Kovalev et al., 2022) and
(Beznosikov & Gasnikov, 2023). This setting is significantly different from ours because we do
not consider δ-similarity, and our optimization procedure is presented in an online scheme. In
(Beznosikov & Gasnikov, 2023) in particular, local steps are also considered, but we would note that
Beznosikov & Gasnikov (2023) require the server to take local steps instead of the clients, which
also requires the presence of data on the server. This is done by making the first client the server and
is in line with the setting in DANE (Shamir et al., 2014). In contrast, the server in our setting only
aggregates the model and does not access any data, which also aligns with the privacy-preserving
purpose in FL.

C ADDITIONAL PRELIMINARIES, DEFINITIONS, AND REMARKS ON

ASSUMPTIONS

In this section, we provide supplementary theoretical backgrounds for the algorithm and the con-
vergence analysis of FeDualEx. We start by providing a more detailed introduction to the related
algorithms, then list additional definitions necessary for the analysis. Before moving on to the main
proof for FeDualEx, we state formally the assumptions made and provide additional remarks on the
assumptions that better link them to their usage in the proof.

C.1 ADDITIONAL PRELIMINARIES

To make this paper as self-contained as possible, in this section, we provide a brief overview of mirror
descent, dual averaging, and their advancement in saddle point optimization, i.e., mirror prox and

23

Published as a conference paper at ICLR 2024

dual extrapolation. More comprehensive introductions can be found in the original papers and in
(Bubeck et al., 2015; Cohen et al., 2021). We slide into mirror descent from the simple and widely
known projected gradient descent, namely vanilla gradient descent with constraint, therefore plus
another projection of the updated sequence back to the feasible set.

C.1.1 MIRROR DESCENT AND DUAL AVERAGING

We start by introducing projected gradient descent. Projected gradient descent first takes the gradient
update, then projects the updated point back to the constraint by finding a feasible solution within the
constraint that minimizes its Euclidean distance to the current point. The updating sequence is given
below: ∀t ∈ [T ], xt ∈ X whereas not necessarily for x′
t,

x′
t+1 = xt − ηg(xt)

xt+1 = arg min

x∈X

1
2

∥x − x′

t+1∥2
2.

Mirror Descent (Nemirovskij & Yudin, 1983). Mirror descent generalizes projected gradient
descent to non-Euclidean space with the Bregman divergence (Bregman, 1967). We provide the
definition of the Bregman divergence below.
Definition 5 (Bregman Divergence (Bregman, 1967)). Let h : Rd → R ∪ {∞} be a prox function or
a distance-generating function that is closed, strictly convex, and differentiable in int dom h. The
Bregman divergence for x ∈ dom h and y ∈ int dom h is defined to be

V h
y (x) = h(x) − h(y) − ⟨∇h(y), x − y⟩.

Mirror descent regards ∇h as a mirror map to the dual space, and follows the procedure below:

∇h(x′

t+1) = ∇h(xt) − ηg(xt)
xt+1 = arg min
(x).

V h
x′

t+1

x∈X

By choosing h(·) = 1
to projected gradient descent.

2 ∥ · ∥2

2 in the Euclidean space whose dual space is itself, mirror descent reduces

Mirror descent can be presented from a proximal point of view, or in the online setting as in Beck &
Teboulle (2003):

xt+1 = arg min

x∈X

⟨ηg(xt), x⟩ + V h
xt

(x).

Such proximal operation with Bregman divergence is studied by others (Censor & Zenios, 1992), and
is recently represented by a neatly defined proximal operator (Cohen et al., 2021).
Definition 6 (Proximal Operator (Cohen et al., 2021)). The Bregman divergence defined proximal
operator is given by

Prox h

x′(·) := arg min

x∈X

{⟨·, x⟩ + V h

x′(x)}.

In this spirit, the mirror descent algorithm can be written with one proximal operation:

xt+1 = Prox h
xt

(ηg(xt)).

Composite Mirror Descent (Duchi et al., 2010). Mirror descent was later generalized to com-
posite convex functions, i.e., the ones with regularization. The key modification is to include the
regularization term in the proximal operator, yet not linearize the regularization term, since it could
be non-smooth and thus non-differentiable. The updating sequence is given by

xt+1 = arg min

x∈X

⟨ηg(xt), x⟩ + V h
xt

(x) + ηψ(x).

It can also be represented with a composite mirror map as in (Yuan et al., 2021):

xt+1 = ∇(h + ηψ)∗(∇h(xt) − ηg(xt)).

24

Published as a conference paper at ICLR 2024

Dual Averaging (Nesterov, 2009). Compared with mirror descent, dual averaging moves the
updating sequence to the dual space. The procedure of dual averaging is as follows (Bubeck et al.,
2015):

∇h(x′

t+1) = ∇h(x′
xt+1 = arg min

t) − ηg(xt)
(x),

V h
x′

t+1

x∈X

or equivalently as presented in (Nesterov, 2009) with the sequence of dual variables: ∀t ∈ [T ],
xt ∈ X , µt ∈ X ∗,

This can be further simplified to

µt+1 = µt − ηg(xt)
xt+1 = ∇h∗(µt+1).

xt+1 = arg min

x∈X

⟨η

t
(cid:88)

τ =0

g(xt), x⟩ + h(x).

Composite Dual Averaging (Xiao, 2010). Around the same time as composite mirror descent,
composite dual averaging, also known as regularized dual averaging, was proposed with a similar
idea of including the regularization term in the proximal operator. As presented in the original paper
(Xiao, 2010):

xt+1 = arg min

x∈X

⟨η

t
(cid:88)

τ =0

g(xτ ), x⟩ + ηβth(x) + tηψ(x),

in which {βt}t≥1 is a non-negative and non-decreasing input sequence. Flammarion & Bach (2017)
adopted the case with constant sequence βt = 1
η ,

xt+1 = arg min

x∈X

⟨η

t
(cid:88)

τ =0

g(xτ ), x⟩ + h(x) + tηψ(x),

and equivalently with composite mirror map:

µt+1 = µt − ηg(xt)
xt+1 = ∇(h + tηψ)∗(µt+1),

which is also presented in (Yuan et al., 2021).

C.1.2 MIRROR PROX AND DUAL EXTRAPOLATION

Mirror Prox (Nemirovski, 2004). Mirror prox generalizes the extra-gradient method to non-
Euclidean space as mirror descent compared with projected gradient descent. It was proposed for
variational inequalities (VIs), including SPP. We first present the corresponding Bregman divergence
in the saddle point setting, whose definition was not included in detail in (Nemirovski, 2004) but was
later more clearly stated in (Nesterov, 2007; Shi et al., 2017).
Definition 7 (Bregman Divergence for Saddle Functions (Nesterov, 2007)). Let ℓ : X ×Y → R∪{∞}
be a distance-generating function that is closed, strictly convex, and differentiable in int dom ℓ. For
z = (x, y) ∈ Z = X × Y, the function and its gradient are defined as

ℓ(z) = h1(x) + h2(y),

∇ℓ(z) =

(cid:21)

(cid:20)∇xh1(x)
∇yh2(y)

.

The Bregman divergence for z = (x, y) ∈ dom ℓ and z′ = (x′, y′) ∈ int dom ℓ is defined to be

z′(z) := ℓ(z) − ℓ(z′) − ⟨∇ℓ(z′), z − z′⟩.
V ℓ
Notice that our notion of ℓ is not a saddle function, slightly different from that in Shi et al. (2017), but
the Bregman divergence defined is the same as Eq. (6) in Shi et al. (2017) and Eq. (4.9) in Nesterov
(2007).

25

Published as a conference paper at ICLR 2024

Mirror prox can also be viewed as an extra-step mirror descent. Most intuitively, by introducing an
intermediate variable zt+1/2, its procedure is as follows:

∇h(z′

t+1/2) = ∇h(zt) − ηg(zt)
zt+1/2 = arg min

V h
z′
t+1/2

(z)

z∈Z

∇h(z′

t+1) = ∇h(zt) − ηg(zt+1/2)
zt+1 = arg min

(z).

V h
z′
t+1

z∈Z

And it can be represented with the proximal operator in Definition 6 as well. Following (Cohen et al.,
2021), ∀t ∈ [T ], zt, zt+1/2 ∈ Z,

zt+1/2 = Prox ℓ
zt
zt+1 = Prox ℓ
zt

(ηg(zt))

(ηg(zt+1/2)).

Dual Extrapolation (Nesterov, 2007). As in dual averaging, dual extrapolation moves the updating
sequence of mirror prox to the dual space. Slightly different from a two-step dual averaging, dual
extrapolation further initialize a fixed point in the primal space ¯z, and as presented in (Cohen et al.,
2021), its procedure is as follows: ∀t ∈ [T ], zt, zt+1/2 ∈ Z, ωt ∈ Z ∗,

¯z(ωt)

zt = Prox ℓ
zt+1/2 = Prox ℓ
zt

(ηg(zt))
ωt+1 = ωt + ηg(zt+1/2).

The updating sequence presented above is equivalent to that defined in the original paper (Nesterov,
2007), simply replacing the arg max with arg min, and the dual variables with its additive inverse in
the dual space.

C.2 ADDITIONAL DEFINITIONS

In this subsection, we list additional definitions involved in the theoretical analysis in subsequent
sections.
Definition 8 (Legendre function (Rockafellar, 1970)). A proper, convex, closed function h : Rd →
R ∪ {∞} is called a Legendre function or a function of Legendre-type if (a) h is strictly convex; (b)
h is essentially smooth, namely h is differentiable on int dom h, and ∥∇h(xt)∥ → ∞ for every
sequence {xt}∞
t=0 ⊂ int dom h converging to a boundary point of dom h as t → ∞.
Definition 9 (Convex Conjugate or Legendre–Fenchel Transformation (Boyd & Vandenberghe,
2004)). The convex conjugate of a function h is defined as

h(s) = sup

{⟨s, z⟩ − h(z)}.

z

Definition 10 (Differentiability of the conjugate of strictly convex function (Chapter E, Theorem
4.1.1 in Hiriart-Urruty & Lemaréchal (2004))). For a strictly convex function h, int dom h∗ ̸= ∅
and h∗ is continuously differentiable on int dom h∗, with gradient defined as:
∇h∗(s) = arg min

{⟨−s, z⟩ + h(z)}

(5)

C.3 FORMAL ASSUMPTIONS AND REMARKS

z

In this subsection, we state the assumptions formally and provide additional remarks that may help in
understanding the theoretical analysis.
Assumption 1 (Assumptions on the objective function). For the composite saddle function ϕ(z) =
f (x, y) + ψ1(x) − ψ2(y) = 1
M

m=1 fm(x, y) + ψ1(x) − ψ2(y), we assume that

(cid:80)M

a.(Local Convexity of f ) ∀m ∈ [M ], fm(x, y) is convex in x and concave in y.

b.(Convexity of ψ) ψ1(x) is convex in x, and ψ2(y) is convex in y.

26

Published as a conference paper at ICLR 2024

Assumption 2 (Assumptions on the gradient operator). For f in the objective function, its gradient
operator is given by g =
m=1 gm, and we
assume that

. By the linearity of gradient operators, g = 1
M

(cid:104) ∇xf
−∇yf

(cid:80)M

(cid:105)

a.(Local Lipschitzness of g) ∀m ∈ [M ], gm(z) =

(cid:104) ∇xfm(x,y)
−∇yfm(x,y)

(cid:105)

is β-Lipschitz:

∥gm(z) − gm(z′)∥∗ ≤ β∥z − z′∥

b.(Local Unbiased Estimate and Bounded Variance) For any client m ∈ [M ], the local gradient
queried by some local random sample ξm is unbiased and also bounded in variance, i.e.,
Eξ[gm(zm; ξm)] = gm(zm), and

Eξ

(cid:2)∥gm(zm; ξm) − gm(zm)∥2

∗

(cid:3) ≤ σ2

c. (Bounded Gradient) ∀m ∈ [M ],

∥gm(zm; ξm)∥∗ ≤ G

Assumption 3 (Assumption on the distance-generating function). The distance-generating function
h is a Legendre function that is 1-strongly convex, i.e., ∀x, y,

h(y) − h(x) − ⟨∇h(x), y − x⟩ ≥

1
2

∥y − x∥2.

Assumption 4. The domain of the optimization problem Z is compact in terms of Bregman Diver-
gence, i.e., ∀z, z′ ∈ Z, V ℓ
Remark 1. An immediate result of Assumption 1a is that, ∀z = (x, y), z′ = (x′, y′) ∈ Z

z′(z) ≤ B.

Summing them up,

f (x′, y′) − f (x, y′) ≤ ⟨∇xf (x′, y′), x′ − x⟩,
f (x′, y) − f (x′, y′) ≤ ⟨−∇yf (x′, y′), y′ − y⟩.

f (x′, y) − f (x, y′) ≤ ⟨g(z′), z′ − z⟩.

Remark 2. For any sequence of i.i.d. random variables ξm
let Fr,k denote the σ-field generated by the set {ξm
r, k ∈ {0, 1/2, ..., K − 1, K − 1/2}))}. Then any ξm
2b implies

0,0, ξm

0,1/2, ..., ξm

1,0, ξm
r,k+1/2,
j,t : ∀m ∈ [M ] and ((j = r, t ≤ k) or (j <
r,k is independent of Fr,k−1/2, and Assumption

1,1/2, ..., ξm

r,k, ξm

EFr,k

(cid:2)∥gm(zm

r,k; ξm

r,k) − gm(zm

r,k)∥2

∗ | Fr,k−1/2

(cid:3) ≤ σ2.

Remark 3 (Corollary 23.5.1. and Theorem 26.5. in Rockafellar (1970)). For a closed convex (not
necessarily differentiable) function h, ∂h is the inverse of ∂h∗ in the sense of multi-valued mappings,
i.e., z ∈ ∂h∗(ς) if and only if ς ∈ ∂h(z). Furthermore, if h is of Legendre-type, meaning it is
essentially strictly convex and essentially smooth, then ∂h yields a well-defined ∇h that acts as a
bijection, i.e., (∇h)−1 = ∇h∗.
Remark 4. Assumption 3 and Remark 3 also trivially hold for ℓ from Definition 7 in the saddle point
setting, and eventually, the generalized distance-generating function ℓt from Definition 3. Due to
the strong convexity of ℓt, ∇ℓ∗
t is well-defined as noted in Definition 10. Together with the potential
non-smoothness of ℓt, Remark 3 implies that z = ∇ℓ∗

t (ς) if and only if ς ∈ ∂ℓt(z).

D ADDITIONAL TECHNICAL LEMMAS

In this section, we list some technical lemmas that are referenced in the proofs of the main theorem
and its helping lemmas.
Lemma 4 (Jensen’s inequality). For a convex function φ(x), variables x1, ..., xn in its domain, and
positive weights a1, ..., an,

(cid:16) (cid:80)n
i=1 aixi
φ
(cid:80)n
i=1 ai

(cid:17)

≤

(cid:80)n

i=1 aiφ(xi)
(cid:80)n
i=1 ai

,

and the inequality is reversed if φ(x) is concave.

27

Published as a conference paper at ICLR 2024

Lemma 5 (Cauchy-Schwarz inequality (Strang, 2006)). For any x and y in an inner product space,

Lemma 6 (Young’s inequality (Lemma 1.45. in Sofonea & Matei (2009))). Let p, q ∈ R be two
conjugate exponents, that is 1 < p < ∞, and 1
p + 1

q = 1. Then ∀a, b ≥ 0,

⟨x, y⟩ ≤ ∥x∥∥y∥.

ab ≤

ap
p

+

bq
q

.

Lemma 7 (AM-QM inequality). For any set of positive integers x1, ..., xn,

n
(cid:88)

(cid:0)

i=1

(cid:1)2

xi

≤ n

n
(cid:88)

i=1

x2
i .

(6)

Lemma 8 (Lemma 2.3 in Jiang & Mokhtari (2022)). Suppose Assumption 1 and 2 hold, then
∀z = (x, y), z1, ..., zT ∈ Z and θ1, ..., θT ≥ 0 with (cid:80)T

t=1 θt = 1, we have

T
(cid:88)

ϕ(

t=1

θtxt, y) − ϕ(x,

T
(cid:88)

t=1

θtyt) ≤

T
(cid:88)

t=1

θt[⟨g(zt), zt − z⟩ + ψ(zt) − ψ(z)],

in which ψ(z) = ψ1(x) + ψ2(y).

Proof. For ψ(z) = ψ1(x) + ψ2(y),

ϕ(xt, y) − ϕ(x, yt) = f (xt, y) + ψ1(xt) − ψ2(y) − f (x, yt) − ψ1(x) + ψ2(yt)
= f (xt, y) − f (x, yt) + ψ(zt) − ψ(z)
≤ ⟨g(zt), zt − z⟩ + ψ(zt) − ψ(z),

where the inequality holds by convexity-concavity of f (x, y), i.e. Remark 1. Then sum the inequality
over t = 1, ..., T ,

T
(cid:88)

t=1

ϕ(θtxt, y) −

T
(cid:88)

t=1

ϕ(x, θtyt) ≤

T
(cid:88)

t=1

(cid:2)⟨g(zt), zt − z⟩ + ψ(zt) − ψ(z)(cid:3).

Finally, by Jensen’s inequality in Lemma 4,

T
(cid:88)

t=1

ϕ(θtxt, y) ≥ ϕ

(cid:16) T
(cid:88)

t=1

θtxt, y

(cid:17)

,

T
(cid:88)

t=1

ϕ(x, θtyt) ≤ ϕ

(cid:16)

x,

T
(cid:88)

t=1

(cid:17)

,

θtyt

which completes the proof.

Lemma 9 (Theorem 4.2.1 in Hiriart-Urruty & Lemaréchal (2004)). The conjugate of an α-strongly
convex function is 1

α -smooth. That is, for h that is strongly convex with modulus α > 0, ∀x, x′,

∥∇h∗(x) − ∇h∗(x′)∥ ≤

1
α

∥x − x′∥.

Lemma 10 (Lemma 2 in Flammarion & Bach (2017)). Generalized Bregman divergence upper-
bounds the Bregman divergence. That is, under Assumption 1 and 3, ∀x ∈ dom h, ∀µ′ ∈
int dom h∗

t where ht = h + tηψ,

in which x′ = ∇h∗

t (µ′).

˜V ht
µ′ (x) ≥ V h

x′(x),

28

Published as a conference paper at ICLR 2024

E COMPLETE ANALYSIS OF FEDUALEX FOR COMPOSITE SADDLE POINT

PROBLEMS

We begin by reformulating the updating sequences with another pair of auxiliary dual variables.
Expand the prox operator in Algorithm 1 line 6 to 8 by Definition 4, and rewrite by the gradient of
the conjugate function in Definition 10,

zm
r,k = arg min

z

zm
r,k+1/2 = arg min

z

{⟨ς m

r,k − ¯ς, z⟩ + ℓr,k(z)} = ∇ℓ∗

r,k(¯ς − ς m

r,k)

{⟨ηcgm(zm

r,k; ξm

r,k) − (¯ς − ς m

r,k), z⟩ + ℓr,k+1(z)} = ∇ℓ∗

r,k+1((¯ς − ς m

r,k) − ηcgm(zm

r,k; ξm

r,k))

r,k+1 = ς m
ς m

r,k + ηcgm(zm

r,k+1/2; ξm

r,k+1/2)

r,k = ¯ς − ς m
r,k is the conjugate of ℓr,k = ℓ + (ηsrK + k)ηcψ. And define ωm
r,k+1/2 such that zm

Define auxiliary dual variable ωm
which ℓ∗
of the intermediate variable zm
updating sequence, we get an equivalent updating sequence for the auxiliary dual variables.

r,k), in
r,k+1/2 to be the dual image
r,k+1/2). Then from the above

r,k. It satisfies immediately that zm

r,k+1/2 = ∇ℓ∗

r,k = ∇ℓ∗

r,k+1(ωm

r,k(ωm

r,k+1/2 = ωm
ωm
r,k+1 = ωm
ωm

r,k − ηgm(zm
r,k − ηgm(zm

r,k; ξm
r,k)
r,k+1/2; ξm

r,k+1/2)

Now we analyze the following shadow sequences. Define

ωr,k =

1
M

M
(cid:88)

m=1

ωm

r,k,

gr,k =

1
M

M
(cid:88)

m=1

then

In the meantime,

ωr,k+1/2 = ωr,k − ηcgr,k,

ωr,k+1 = ωr,k − ηcgr,k+1/2.

gm(zm

r,k; ξm

r,k),

(cid:100)zr,k = ∇ℓ∗

r,k(ωr,k),

(cid:92)zr,k+1/2 = ∇ℓ∗

r,k+1(ωr,k+1/2).

(2)

(3)

(4)

E.1 MAIN THEOREM AND PROOF

Theorem 1 (Main). Under assumptions, the duality gap evaluated with the ergodic sequence
generated by the intermediate steps of FeDualEx in Algorithm 1 is bounded by

(cid:104)
E

Gap

(cid:16) 1

R−1
(cid:88)

K−1
(cid:88)

RK

r=0

k=0

(cid:92)zr,k+1/2

(cid:17)(cid:105)

≤

B
ηcRK

+ 20β2(ηc)3K 2G2 +

5σ2ηc
M

+ 2

3

2 βηcKGB

1
2 .

Choosing step size ηc = min{ 1
1
2 β
5

,

1
4 β

20

B

1
2 G

1
4
1
2 K

3
4 R

1
4

, B
5

1
2 σR

1
2

1
2 M
1
2 K

,

1
2

3
4 β

2

B

1
2 G

1
4
1
2 KR

1
2

},

(cid:104)
E

Gap

(cid:16) 1

R−1
(cid:88)

K−1
(cid:88)

RK

r=0

k=0

(cid:92)zr,k+1/2

(cid:17)(cid:105)

≤

5 1
2 βB
RK

+

2 B 3

4

20 1

4 β 1
K 1

2 G 1
4 R 3

4

+

2

5 1
2 σB 1
2 R 1
M 1

2 K 1

2

+

2 3

4 β 1

2 B 3

4

2 G 1
R 1

2

.

Proof. The proof of the main theorem relies on Lemma 1, the bound for the non-smooth term, and
Lemma 2, the bound for the smooth term. These two lemmas are combined in Lemma 3 and then
yield the per-step progress for FeDualEx. The three lemmas are listed and proved right after this
theorem. Here, we finish proving the main theorem from the per-step progress.

29

Published as a conference paper at ICLR 2024

Starting from Lemma 3, we telescope for all local updates k ∈ {0, ..., K − 1} after the same
communication round r.

(cid:104) K−1
(cid:88)

ηcE

k=0

(cid:2)⟨g( (cid:92)zr,k+1/2), (cid:92)zr,k+1/2 − z⟩ + ψ( (cid:92)zr,k+1/2) − ψ(z)(cid:3)(cid:105)

≤ ˜V ℓr,0
ωr,0

(z) − ˜V

ℓr,K
ωr,K

(z) +

5σ2(ηc)2K
M

+ 20

≤ ˜V ℓr,0
ωr,0

(z) − ˜V

ℓr,K
ωr,K

(z) +

≤ ˜V ℓr,0
ωr,0

(z) − ˜V

ℓr,K
ωr,K

(z) +

5σ2(ηc)2K
M

5σ2(ηc)2K
M

+ 20

K−1
(cid:88)

k=0

K−1
(cid:88)

k=0

β2(ηc)4(k + 1)2G2 + 2

K−1
(cid:88)

3
2

k=0

β(ηc)2(k + 1)GB

1
2

β2(ηc)4K 2G2 + 2

K−1
(cid:88)

3
2

k=0

β(ηc)2KGB

1
2

+ 20β2(ηc)4K 3G2 + 2

3

2 β(ηc)2K 2GB

1
2 .

As we initialize the local dual updates on all clients after each communication with the dual average
of the previous round’s last update, ∀r ∈ {1, ..., R}, the first variable in this round ωr,0 is the same
as the last variable ωr−1,0 in the previous round. As a result, taking the server step size ηs = 1, we
can further telescope across all rounds and have

(cid:104) R−1
(cid:88)

ηcE

K−1
(cid:88)

r=0

k=0

(cid:2)⟨g( (cid:92)zr,k+1/2), (cid:92)zr,k+1/2 − z⟩ + ψ( (cid:92)zr,k+1/2) − ψ(z)(cid:3)(cid:105)

≤ ˜V ℓ0,0
ω0,0

(z) − ˜V ℓR,K
ωR,K

(z) +

5σ2(ηc)2KR
M

+ 20β2(ηc)4K 3RG2 + 2

3

2 β(ηc)2K 2RGB

1
2 .

Notice that the generalized Bregman divergence ˜V ℓ0,0
ω0,0
z0 = ∇ℓ∗(¯ς). Thus, by Assumption 4, ˜V ℓ0,0
ω0,0
we get

(z) = ˜V ℓ
(z), where
(z) ≤ B. Dividing ηcKR on both sides of the equation,

(z) = ˜V ℓ0,0
¯ς−ς0

¯ς (z) = V ℓ
z0

ηcE

(cid:104) 1

RK

R−1
(cid:88)

K−1
(cid:88)

(cid:2)⟨g( (cid:92)zr,k+1/2), (cid:92)zr,k+1/2 − z⟩ + ψ( (cid:92)zr,k+1/2) − ψ(z)(cid:3)(cid:105)

r=0
B
ηcRK

k=0

+

5σ2ηc
M

+ 20β2(ηc)3K 2G2 + 2

3

2 βηcKGB

1
2 .

≤

Finally, applying Lemma 8 completes the proof.

Lemma 1 (Bounding the Regularization Term). ∀z,
ηc(cid:2)ψ( (cid:92)zr,k+1/2) − ψ(z)(cid:3) = ˜V ℓr,k
ωr,k

(z) − ˜V ℓr,k+1
ωr,k+1

(z) − ˜V ℓr,k
ωr,k

( (cid:92)zr,k+1/2) − ˜V ℓr,k+1
ωr,k+1/2

((cid:92)zr,k+1)

+ ηc⟨gr,k+1/2 − gr,k, (cid:92)zr,k+1/2 − (cid:92)zr,k+1⟩ + ηc⟨gr,k+1/2, z − (cid:92)zr,k+1/2⟩

Proof. By the definition of generalized Bregman divergence and the updating sequence in Eq. (2),
∀z,

˜V ℓr,k+1
ωr,k+1/2

(z) = ℓr,k+1(z) − ℓr,k+1( (cid:92)zr,k+1/2) − ⟨ωr,k+1/2, z − (cid:92)zr,k+1/2⟩

= ℓr,k+1(z) − ℓr,k+1( (cid:92)zr,k+1/2) − ⟨ωr,k − ηcgr,k, z − (cid:92)zr,k+1/2⟩
= ℓr,k(z) − ℓr,k( (cid:92)zr,k+1/2) + ηc(cid:2)ψ(z) − ψ( (cid:92)zr,k+1/2)(cid:3)
− ⟨ωr,k, z − (cid:92)zr,k+1/2⟩ + ηc⟨gr,k, z − (cid:92)zr,k+1/2⟩.

Similarly, we can have for the updating sequence in Eq. (3) that ∀z,

˜V ℓr,k+1
ωr,k+1

(z) = ℓr,k(z) − ℓr,k((cid:92)zr,k+1) + ηc(cid:2)ψ(z) − ψ((cid:92)zr,k+1)(cid:3)

− ⟨ωr,k, z − (cid:92)zr,k+1⟩ + ηc⟨gr,k+1/2, z − (cid:92)zr,k+1⟩.

Plug z = (cid:92)zr,k+1 into Eq. (7),

˜V ℓr,k+1
ωr,k+1/2

((cid:92)zr,k+1) = ℓr,k((cid:92)zr,k+1) − ℓr,k( (cid:92)zr,k+1/2) + ηc(cid:2)ψ((cid:92)zr,k+1) − ψ( (cid:92)zr,k+1/2)(cid:3)

− ⟨ωr,k, (cid:92)zr,k+1 − (cid:92)zr,k+1/2⟩ + ηc⟨gr,k, (cid:92)zr,k+1 − (cid:92)zr,k+1/2⟩.

(7)

(8)

30

Published as a conference paper at ICLR 2024

Add this up with Eq. (8),

˜V ℓr,k+1
ωr,k+1/2

((cid:92)zr,k+1) + ˜V ℓr,k+1
ωr,k+1

(z) = ℓr,k(z) − ℓr,k( (cid:92)zr,k+1/2) − ⟨ωr,k, z − (cid:92)zr,k+1/2⟩
(cid:125)

(cid:124)

(cid:123)(cid:122)
A1

+ ηc(cid:2)ψ(z) − ψ( (cid:92)zr,k+1/2)(cid:3)
+ ηc⟨gr,k, (cid:92)zr,k+1 − (cid:92)zr,k+1/2⟩ + ηc⟨gr,k+1/2, z − (cid:92)zr,k+1⟩
(cid:123)(cid:122)
(cid:125)
A2

(cid:124)

.

For A1 we have

A1 = ℓr,k(z) − ℓr,k((cid:100)zr,k) − ⟨ωr,k, z − (cid:100)zr,k⟩ − ℓr,k( (cid:92)zr,k+1/2) + ℓr,k((cid:100)zr,k) + ⟨ωr,k, (cid:92)zr,k+1/2 − (cid:100)zr,k⟩

= ˜V

ℓr,k
ωr,k

(z) − ˜V

ℓr,k
ωr,k

( (cid:92)zr,k+1/2).

For A2 we have

A2 = ηc⟨gr,k, (cid:92)zr,k+1 − (cid:92)zr,k+1/2⟩ + ηc⟨gr,k+1/2, (cid:92)zr,k+1/2 − (cid:92)zr,k+1⟩ + ηc⟨gr,k+1/2, z − (cid:92)zr,k+1/2⟩

= ηc⟨gr,k+1/2, z − (cid:92)zr,k+1/2⟩ + ηc⟨gr,k+1/2 − gr,k, (cid:92)zr,k+1/2 − (cid:92)zr,k+1⟩

Plug A1 and A2 back in completes the proof.

For the purpose of clarity, we demonstrate how we generate the terms to be separately bounded for
the smooth part with the following Lemma 2, which holds trivially by the linearity of the gradient
operator g = 1
M
Lemma 2 (Bounding the Smooth Term). ∀z,

m=1 gm and then direct cancellation.

(cid:80)M

⟨g( (cid:92)zr,k+1/2), (cid:92)zr,k+1/2 − z⟩ = ⟨gr,k+1/2, (cid:92)zr,k+1/2 − z⟩ + ⟨

1
M

M
(cid:88)

m=1

gm(zm

r,k+1/2) − gr,k+1/2, (cid:92)zr,k+1/2 − z⟩

+ ⟨

1
M

M
(cid:88)

m=1

[gm( (cid:92)zr,k+1/2) − gm(zm

r,k+1/2)], (cid:92)zr,k+1/2 − z⟩

Based on the previous two lemmas, we arrive at the following lemma that bounds the per-step progress
of FeDualEx.
Lemma 3 (Per-step Progress for FeDualEx in Saddle Point Setting). For ηc ≤ 1
1
2 β
5

,

ηcE(cid:2)⟨g( (cid:92)zr,k+1/2), (cid:92)zr,k+1/2 − z⟩ + ψ( (cid:92)zr,k+1/2) − ψ(z)(cid:3)

≤ ˜V ℓr,k
ωr,k

(z) − ˜V ℓr,k+1
ωr,k+1

(z) +

5σ2(ηc)2
M

+ 20β2(ηc)4(k + 1)2G2 + 2

3

2 β(ηc)2(k + 1)GB

1

2 .

Proof. Based on the previous two lemmas, we can get the following simply by summing them up, in
which we denote the left-hand side as LHS for simplicity.

LHS := ηc(cid:2)⟨g( (cid:92)zr,k+1/2), (cid:92)zr,k+1/2 − z⟩ + ψ( (cid:92)zr,k+1/2) − ψ(z)(cid:3)
≤ ˜V ℓr,k
((cid:92)zr,k+1)
ωr,k
(cid:125)

( (cid:92)zr,k+1/2) − ˜V ℓr,k+1
(z) − ˜V ℓr,k
ωr,k+1/2
ωr,k
(cid:123)(cid:122)
(cid:124)
A3
+ ηc⟨gr,k+1/2 − gr,k, (cid:92)zr,k+1/2 − (cid:92)zr,k+1⟩

(z) − ˜V ℓr,k+1
ωr,k+1

+ ηc⟨

+ ηc⟨

1
M

1
M

M
(cid:88)

m=1

gm(zm

r,k+1/2) − gr,k+1/2, (cid:92)zr,k+1/2 − z⟩

M
(cid:88)

[gm( (cid:92)zr,k+1/2) − gm(zm

r,k+1/2)], (cid:92)zr,k+1/2 − z⟩

m=1

31

Published as a conference paper at ICLR 2024

For the two generalized Bregman divergence terms in A3, we bound them by Lemma 10 and the
strong convexity of ℓ in Remark 4,

A3 ≤ −V ℓ

( (cid:92)zr,k+1/2) − V ℓ

(cid:100)zr,k
1
∥(cid:100)zr,k − (cid:92)zr,k+1/2∥2 −
2

(cid:92)zr,k+1/2

((cid:92)zr,k+1)

1
2

∥ (cid:92)zr,k+1/2 − (cid:92)zr,k+1∥2

≤ −

As a result,

LHS ≤ ˜V ℓr,k
ωr,k
1
2

−

(z) − ˜V ℓr,k+1
ωr,k+1

(z) −

1
2

∥(cid:100)zr,k − (cid:92)zr,k+1/2∥2

∥ (cid:92)zr,k+1/2 − (cid:92)zr,k+1∥2 + ηc⟨gr,k+1/2 − gr,k, (cid:92)zr,k+1/2 − (cid:92)zr,k+1⟩
(cid:125)

(cid:124)

(cid:123)(cid:122)
A4

+ ηc⟨

+ ηc⟨

1
M

1
M

M
(cid:88)

m=1

gm(zm

r,k+1/2) − gr,k+1/2, (cid:92)zr,k+1/2 − z⟩

M
(cid:88)

[gm( (cid:92)zr,k+1/2) − gm(zm

r,k+1/2)], (cid:92)zr,k+1/2 − z⟩.

m=1

A4 can be bounded with Cauchy-Schwarz (Lemma 5) inequality and Young’s inequality (Lemma 6).

A4 ≤ −

∥ (cid:92)zr,k+1/2 − (cid:92)zr,k+1∥2 + ηc∥gr,k+1/2 − gr,k∥∗∥ (cid:92)zr,k+1/2 − (cid:92)zr,k+1∥

≤ −

∥ (cid:92)zr,k+1/2 − (cid:92)zr,k+1∥2 +

(ηc)2
2

∥gr,k+1/2 − gr,k∥2

∗ +

1
2

∥ (cid:92)zr,k+1/2 − (cid:92)zr,k+1∥2

1
2
1
2
(ηc)2
2

=

∥gr,k+1/2 − gr,k∥2
∗.

Then we have

ηc(cid:0)ϕ( (cid:92)zr,k+1/2) − ϕ(z)(cid:1) ≤ ˜V ℓr,k
ωr,k

(z) − ˜V ℓr,k+1
ωr,k+1

(z) −

1
2

∥(cid:100)zr,k − (cid:92)zr,k+1/2∥2 +

(ηc)2
2

∥gr,k+1/2 − gr,k∥2
∗

+ ηc⟨

+ ηc⟨

1
M

1
M

M
(cid:88)

m=1

gm(zm

r,k+1/2) − gr,k+1/2, (cid:92)zr,k+1/2 − z⟩

M
(cid:88)

[gm( (cid:92)zr,k+1/2) − gm(zm

r,k+1/2)], (cid:92)zr,k+1/2 − z⟩.

m=1

Taking expectations on both sides we get

ηcE(cid:2)ϕ( (cid:92)zr,k+1/2) − ϕ(z)(cid:3) ≤ ˜V ℓr,k
ωr,k
1
2

−

(z) − ˜V ℓr,k+1
ωr,k+1

(z)

E(cid:2)∥(cid:100)zr,k − (cid:92)zr,k+1/2∥2(cid:3)
(cid:123)(cid:122)
(cid:125)
B1

+

(ηc)2
2

(cid:124)

E(cid:2)∥gr,k+1/2 − gr,k∥2

∗

(cid:3)

(cid:123)(cid:122)
B2

(cid:125)

(cid:124)

1
M

M
(cid:88)

m=1

+ ηcE(cid:2)⟨

(cid:124)

gm(zm

r,k+1/2) − gr,k+1/2, (cid:92)zr,k+1/2 − z⟩(cid:3)

(cid:123)(cid:122)
B3

(cid:125)

+ ηcE(cid:2)⟨

1
M

(cid:124)

M
(cid:88)

[gm( (cid:92)zr,k+1/2) − gm(zm

r,k+1/2)], (cid:92)zr,k+1/2 − z⟩(cid:3)

.

m=1

32

(cid:123)(cid:122)
B4

(cid:125)

Published as a conference paper at ICLR 2024

B1 + B2 ≤

(ηc)2
2

B2 is bounded in Lemma 14. Therefore, we have
(cid:0) 10σ2
M
5(ηc)2β2
2
(cid:0) 10σ2
M

+ 40β2(ηc)2(k + 1)2G2(cid:1)
∥ (cid:92)zr,k+1/2 − (cid:100)zr,k∥2(cid:105)
1
2
+ 40β2(ηc)2(k + 1)2G2(cid:1) +

(ηc)2
2

(cid:104)
E

−

=

+

E(cid:2)∥(cid:100)zr,k − (cid:92)zr,k+1/2∥2(cid:3)
5(ηc)2β2 − 1
2

(cid:104)
E

∥ (cid:92)zr,k+1/2 − (cid:100)zr,k∥2(cid:105)

≤

5σ2(ηc)2
M

+ 20β2(ηc)4(k + 1)2G2,

for ηc ≤ 1
1
2 β
5

.

B3 is zero after taking the expectation as shown in Lemma 11. B4 is bounded in Lemma 13. Plugging
the bounds for B1 + B2, B3, and B4 back in completes the proof.

E.2 HELPING LEMMAS

In this section, we list the helping lemmas that were referenced in the proof of Lemma 1, 2, and 3.
Lemma 11 (Unbiased Gradient Estimate). Under Assumption 1 and 2,
M
(cid:88)

gm(zm

r,k+1/2) − gr,k+1/2, (cid:92)zr,k+1/2 − z⟩(cid:3) = 0

ηcEFr,k+1/2

(cid:2)⟨

1
M

m=1

Proof. By the unbiased gradient estimate in Assumption 2b and its following Remark 2,

ηcEFr,k+1/2

(cid:2)⟨

1
M

M
(cid:88)

m=1

gm(zm

r,k+1/2) − gr,k+1/2, (cid:92)zr,k+1/2 − z⟩(cid:3)

= ηcEFr,k

(cid:2)EFr,k+1/2

(cid:2)⟨

1
M

M
(cid:88)

m=1

= 0.

gm(zm

r,k+1/2) − gr,k+1/2, (cid:92)zr,k+1/2 − z⟩(cid:12)

(cid:12)Fr,k

(cid:3)(cid:3)

Lemma 12 (Bounded Client Drift under Assumption 2c). ∀m ∈ [M ], ∀k ∈ {0, ..., K − 1},

∥ (cid:92)zr,k+1/2 − zm

r,k+1/2∥ ≤ 2ηc(k + 1)G
r,k∥ ≤ 2ηckG

∥(cid:100)zr,k − zm

Proof. By the smoothness of the conjugate of a strongly convex function, i.e., Lemma 9,

∥ (cid:92)zr,k+1/2 − zm

r,k+1/2∥ = ∥∇ℓ∗

r,k(ωr,k+1/2) − ∇ℓ∗
r,k+1/2∥∗

≤ ∥ωr,k+1/2 − ωm

r,k(ωm

r,k+1/2)∥

After the same round of communication, by the updating sequence, we have ∀m ∈ [M ]:

r,k+1/2 = ωm
ωm

r,k − ηcgm(zm

r,k; ξm

r,k)

= −ηc

k−1
(cid:88)

ℓ=0

gm(zm

r,ℓ+1/2; ξm

r,ℓ+1/2) − ηcgm(zm

r,k; ξm

r,k)

Immediately after each round of communication, all machines are synchronized, i.e., ∀m1, m2 ∈ [M ],
r,0 = ωm2
ωm1

r,0 . Therefore, ∀k ∈ {0, ..., K − 1},

r,k+1/2 − ωm2
ωm1

r,k+1/2 = −ηc

k−1
(cid:88)

ℓ=0

gm1(zm1

r,ℓ+1/2; ξm1

r,ℓ+1/2) − ηcgm1(zm1

r,k ; ξm1
r,k )

+ ηc

k−1
(cid:88)

ℓ=0

gm2 (zm2

r,ℓ+1/2; ξm2

r,ℓ+1/2) + ηcgm2(zm2

r,k ; ξm2
r,k )

33

Published as a conference paper at ICLR 2024

Then ∀m1, m2 ∈ [M ], ∀k ∈ {0, ..., K − 1}, by triangle inequality, Jensen’s inequality, and the
bounded gradient Assumption 2c,

∥ωm1

r,k+1/2 − ωm2

r,k+1/2∥∗ ≤ ηc(cid:0)

k−1
(cid:88)

∥gm1(zm1

r,ℓ+1/2; ξm1

r,ℓ+1/2)∥∗ + ∥gm1 (zm1

r,k ; ξm1

r,k )∥∗

ℓ=0

k−1
(cid:88)

∥gm2(zm2

r,ℓ+1/2; ξm2

r,ℓ+1/2)∥∗ + ∥gm2 (zm2

r,k ; ξm2

r,k )∥∗

(cid:1)

+

ℓ=0
≤ 2ηc(k + 1)G.

As a result,

∥ (cid:92)zr,k+1/2 − zm

r,k+1/2∥ ≤ ∥ωr,k+1/2 − ωm
≤ sup
m1,m2

r,k+1/2∥∗
r,k+1/2 − ωm2

∥ωm1

r,k+1/2∥∗

Similarly, we can show that

≤ 2ηc(k + 1)G.

∥(cid:100)zr,k − zm

r,k∥ ≤ 2ηckG.

Lemma 13. Under Assumption 1-4,

ηcE(cid:2)⟨

1
M

M
(cid:88)

[gm( (cid:92)zr,k+1/2) − gm(zm

r,k+1/2)], (cid:92)zr,k+1/2 − z⟩(cid:3) ≤ 2

3

2 β(ηc)2(k + 1)GB

1
2 .

m=1

Proof. The proof of this lemma relies on the bounded client drift in Lemma 12. We start by splitting
the inner product using Cauchy-Schwarz inequality in Lemma 5, and state the reference for the
following derivation in the parenthesis.

ηcE(cid:2)⟨

1
M

M
(cid:88)

[gm( (cid:92)zr,k+1/2) − gm(zm

r,k+1/2)], (cid:92)zr,k+1/2 − z⟩(cid:3)

m=1

≤ ηcE(cid:2)∥

1
M

M
(cid:88)

[gm( (cid:92)zr,k+1/2) − gm(zm

r,k+1/2)]∥∗∥ (cid:92)zr,k+1/2 − z∥(cid:3)

m=1

≤ ηcE(cid:2) 1
M

≤ ηcE(cid:2) 1
M

≤ ηcE(cid:2) 1
M

M
(cid:88)

∥gm( (cid:92)zr,k+1/2) − gm(zm

r,k+1/2)∥∗∥ (cid:92)zr,k+1/2 − z∥(cid:3)

m=1

M
(cid:88)

m=1

M
(cid:88)

m=1

β∥ (cid:92)zr,k+1/2 − zm

r,k+1/2∥∗∥ (cid:92)zr,k+1/2 − z∥(cid:3)

2βηc(k + 1)G∥ (cid:92)zr,k+1/2 − z∥(cid:3)

≤ ηcE(cid:2)2βηc(k + 1)G ·

(cid:113)

2V ℓ

z ( (cid:92)zr,k+1/2)(cid:3)

≤ 2

3

2 β(ηc)2(k + 1)GB

1
2

(Jensen’s)

(Smoothness)

(Lemma 12)

(Strong-convexity of ℓ)

(Assumption 4)

Lemma 14 (Difference of Gradient and Extra-gradient). Under Assumption 1-4,

E(cid:2)∥gr,k+1/2 − gr,k∥2

∗

(cid:3) ≤

10σ2
M

(cid:104)
+ 40β2(ηc)2(k + 1)2G2 + 5β2E

∥ (cid:92)zr,k+1/2 − (cid:100)zr,k∥2(cid:105)

.

34

Published as a conference paper at ICLR 2024

Proof. By Lemma 7,

EFr,k+1/2

(cid:2)∥gr,k+1/2 − gr,k∥2

∗

(cid:3)

= E

(cid:104)(cid:13)
(cid:2)gr,k+1/2 −
(cid:13)

1
M

M
(cid:88)

m=1

gm(zm

r,k+1/2)(cid:3)

+ (cid:2) 1
M

M
(cid:88)

m=1

gm(zm

r,k) − gr,k

(cid:3) +

1
M

M
(cid:88)

m=1

(cid:2)gm(zm

r,k+1/2) − gm( (cid:92)zr,k+1/2)(cid:3)

+

1
M

M
(cid:88)

m=1

(cid:2)gm((cid:100)zr,k) − gm(zm

r,k)(cid:3) +

1
M

M
(cid:88)

m=1

(cid:2)gm( (cid:92)zr,k+1/2) − gm((cid:100)zr,k)(cid:3)(cid:13)
2
(cid:13)
∗

(cid:105)

(cid:105)

(cid:125)

(cid:104)
≤ 5 E

∥gr,k+1/2 −

(cid:124)

(cid:104)
+ 5 E

∥

(cid:124)

(cid:104)
+ 5 E

∥

(cid:124)

1
M

M
(cid:88)

m=1

1
M

M
(cid:88)

m=1

1
M

M
(cid:88)

m=1
(cid:123)(cid:122)
C1

gm(zm

r,k+1/2)∥2
∗

(cid:105)

(cid:104)
+5 E
∥

(cid:125)

(cid:124)

1
M

M
(cid:88)

m=1

gm(zm

r,k) − gr,k∥2
∗

(cid:123)(cid:122)
C2

(cid:2)gm(zm

r,k+1/2) − gm( (cid:92)zr,k+1/2)(cid:3)∥2

∗

(cid:123)(cid:122)
C3

(cid:105)

(cid:125)

(cid:2)gm((cid:100)zr,k) − gm(zm

r,k)(cid:3)∥2

∗

(cid:123)(cid:122)
C4

1
M

M
(cid:88)

m=1

(cid:105)

(cid:104)
+5 E

∥

(cid:125)

(cid:124)

(cid:2)gm( (cid:92)zr,k+1/2) − gm((cid:100)zr,k)(cid:3)∥2

∗

(cid:123)(cid:122)
C5

(cid:105)

(cid:125)

For C1, by Assumption 2b and its following Remark 2,
M
(cid:88)

(cid:104)

gm(zm

r,k+1/2; ξm

r,k+1/2) −

C1 = EFr,k+1/2

∥

1
M

1
M

M
(cid:88)

m=1

gm(zm

r,k+1/2)∥2
∗

(cid:105)

1
M 2

(cid:104)

EFr,k+1/2

1
M 2 VarFr,k+1/2

m=1

M
(cid:88)

∥
m=1
(cid:104) M
(cid:88)

m=1

(cid:2)gm(zm

r,k+1/2; ξm

r,k+1/2) − gm(zm

r,k+1/2)(cid:3)∥2

∗

(cid:105)

(cid:2)gm(zm

r,k+1/2; ξm

r,k+1/2) − gm(zm

r,k+1/2)(cid:3)(cid:105)

VarFr,k+1/2

(cid:104)(cid:2)gm(zm

r,k+1/2; ξm

r,k+1/2) − gm(zm

r,k+1/2)(cid:3)(cid:105)

EFr,k+1/2

(cid:104)
∥gm(zm

r,k+1/2; ξm

r,k+1/2) − gm(zm

r,k+1/2)∥2
∗

(cid:105)

EFr,k

(cid:104)
EFr,k+1/2

(cid:2)∥gm(zm

r,k+1/2; ξm

r,k+1/2) − gm(zm

r,k+1/2)∥2
∗

(i.i.d.)

(cid:3)(cid:105)

(cid:12)
(cid:12)Fr,k

M
(cid:88)

m=1

M
(cid:88)

m=1

M
(cid:88)

m=1

1
M 2

1
M 2

1
M 2

σ2
M

=

=

=

=

=

≤

Similarly, we have C2 ≤ σ2
M .

For C3, by Lemma 7, β-smoothness of fm, and finally Lemma 12, we have

C3 ≤ E

(cid:104) 1
M 2 · M

M
(cid:88)

m=1

∥gm(zm

r,k+1/2) − gm( (cid:92)zr,k+1/2)∥2

∗

(cid:105)

≤

β2
M

M
(cid:88)

m=1

(cid:104)
E

∥zm

r,k+1/2 − (cid:92)zr,k+1/2∥2(cid:105)

≤ 4β2(ηc)2(k + 1)2G2

35

Published as a conference paper at ICLR 2024

Similarly for C4, we have C4 ≤ 4β2(ηc)2k2G2.

For C5, by Lemma 7, β-smoothness of fm from Assumption 2a, and finally Lemma 12,

C5 = E

(cid:104) 1
M 2 ∥

M
(cid:88)

(cid:2)gm( (cid:92)zr,k+1/2) − gm((cid:100)zr,k)(cid:3)∥2

∗

(cid:105)

m=1

M
(cid:88)

≤ E

(cid:104) 1
M 2 · M
∥ (cid:92)zr,k+1/2 − (cid:100)zr,k∥2(cid:105)
(cid:104)
≤ β2E

m=1

.

∥gm( (cid:92)zr,k+1/2)) − gm((cid:100)zr,k)∥2

∗

(cid:105)

Plugging the bounds for C1, C2, C3, C4, and C5 back in completes the proof.

F COMPLETE ANALYSIS OF FEDUALEX FOR COMPOSITE CONVEX

OPTIMIZATION

In this section, we reduce the problem to composite convex optimization in the following form:

min
x∈X

ϕ(x) = f (x) + ψ(x)

(9)

(cid:80)M

where f (x) = 1
m=1 fm(x). The analysis builds upon the strong-convexity of the distance-
M
generating function h in Assumption 3 and the following set of assumptions in the convex optimization
setting:
Assumption 5. We make the following assumptions:

a.(Convexity of f ) ∀m ∈ [M ], fm is convex. That is, ∀x, x′ ∈ X ,

fm(x) − fm(x′) ≤ ⟨fm(x), x − x′⟩.

b.(Local Smoothness of f ) ∀m ∈ [M ], fm is β-smooth: ∀x, x′ ∈ X ,

fm(x) ≤ fm(x′) + ⟨fm(x′), x − x′⟩ +

β
2

∥x − x′∥.

c. (Convexity of ψ) ψ(x) is convex.

d.(Local Unbiased Estimate and Bounded Variance) For any client m ∈ [M ], the local gradient
queried by some local random sample ξm is unbiased and also bounded in variance, i.e.,
Eξ[gm(xm; ξm)] = gm(xm) and Eξ[∥gm(xm; ξm) − gm(xm)∥2

∗] ≤ σ2.

e. (Bounded Gradient) ∀m ∈ [M ], ∥gm(xm; ξm)∥∗ ≤ G.

Federated dual extrapolation for composite convex optimization is to replace the part of Algorithm 1
highlighted in green with the following updating sequence, where we overuse ς now as the notation
for dual variables in the convex setting as well.

ς m
r,0 = ςr
for k = 0, 1, . . . , K − 1 do
r,k = ˜Prox
xm
r,k+1/2 = ˜Prox
xm
r,k+1 = ς m
ς m

(ς m
r,k)
hr,k+1
¯ς−ςm
r,k

hr,k
¯ς

r,k + ηcgm(xm

(ηcgm(xm
r,k+1/2; ξm

r,k; ξm
r,k))
r,k+1/2)

end for

For the proximal operator defined by hr,k, reformulating from its Definition 4 to ∇h∗
10 yields
xm
r,k = arg min

r,k − ¯ς, x⟩ + hr,k(x)} = ∇h∗

r,k(¯ς − ς m

{⟨ς m

r,k)

r,k in Definition

x

xm
r,k+1/2 = arg min

{⟨ηcgm(xm

r,k; ξm

r,k) − (¯ς − ς m

r,k), x⟩ + hr,k+1(x)} = ∇h∗

r,k+1((¯ς − ς m

r,k) − ηcgm(xm

r,k; ξm

r,k))

x

r,k+1 = ς m
ς m

r,k + ηcgm(xm

r,k+1/2; ξm

r,k+1/2)

36

Published as a conference paper at ICLR 2024

Similarly, we define auxiliary dual variable µm
xm
r,k(µm
r,k+1/2. Then by definition, xm
dating sequence is equivalent to µm
r,k+1/2 = µm
ηgm(xm
gr,k = 1
M

r,k+1/2; ξm
(cid:80)M

m=1 gm(xm

r,k = ∇h∗

r,k; ξm

r,k),

r,k = ¯ς − ς m
r,k) and xm
r,k − ηgm(xm

r,k and µm
r,k+1/2 = ∇h∗
r,k; ξm

r,k) followed by µm

r,k+1/2 the dual image of
r,k+1(µm
r,k+1/2). The up-
r,k+1 = µm
r,k −
(cid:80)M
m=1 µm
r,k and

M

r,k+1/2). For the shadow sequence of averaged variables µr,k = 1

µr,k+1/2 = µr,k − ηcgr,k,

µr,k+1 = µr,k − ηcgr,k+1/2.

(10)

(11)
r,k(µr,k) and

r,k+1(µr,k+1/2)

Finally, the projections of the averaged dual back to the primal space are (cid:100)xr,k = ∇h∗
(cid:92)xr,k+1/2 = ∇h∗
Theorem 2. Under Assumption 5, the ergodic intermediate sequence generated by FeDualEx for
composite convex objectives satisfies
K−1
(cid:88)

R−1
(cid:88)

+ 20β2(ηc)3K 2G2 +

+ 2β(ηc)3K 2G2.

(cid:92)xr,k+1/2) − ϕ(x)(cid:3) ≤

5σ2ηc
M

B
ηcRK

E(cid:2)ϕ(

1
RK

r=0

k=0

Choosing step size

ηc = min{

1
5 1

2 β

,

4

B 1
2 G 1

20 1

4 β 1

2 K 3

4 R 1

4

,

2

2 M 1

B 1
2 σR 1

2 K 1

2

5 1

,

3

B 1
3 G 2

2 1

3 β 1

3 KR 1

3

further yields the following convergence rate:
5 1
2 βB
RK

(cid:92)xr,k+1/2) − ϕ(x)(cid:3) ≤

1
RK

E(cid:2)ϕ(

K−1
(cid:88)

R−1
(cid:88)

r=0

k=0

+

2 B 3

4

20 1

4 β 1
K 1

2 G 1
4 R 3

4

+

2

5 1
2 σB 1
2 R 1
M 1

2 K 1

2

}

+

2 1

3 β 1

3 B 2

3

3 G 2
R 2

3

.

Proof. As the proof for Theorem 1, the proof for this theorem depends on Lemma 15 and Lemma
16, which further yield Lemma 17. These lemmas are presented and proved right after this theorem.
Here, we start from Lemma 17. Telescoping over all k ∈ {0, ..., K − 1} and all r ∈ {0, ..., R − 1}
assuming ηs = 1 yields

R−1
(cid:88)

K−1
(cid:88)

ηcE(cid:2)

r=0

k=0

ϕ( (cid:92)xr,k+1/2) − RKϕ(x)(cid:3) ≤ ˜V h0,0

µ0,0

(x) − ˜V hR,K
µR,K

(x) +

5σ2(ηc)2KR
M

+ 20β2(ηc)4K 3RG2 + 2β(ηc)3K 3RG2.

By Assumption 4, ˜V h0,0
µ0,0
followed by applying Jensen’s inequality (Lemma 4) completes the proof.

(x) = V h
x0

(x) ≤ B, where x0 = ∇h∗(¯ς). Dividing both sides by ηcKR

Lemma 15 (Bounding the Regularization Term). ∀x,
ηc(cid:2)ψ( (cid:92)xr,k+1/2) − ψ(x)(cid:3) = ˜V hr,k
µr,k

(x) − ˜V hr,k+1
µr,k+1

(x) − ˜V hr,k
µr,k

( (cid:92)xr,k+1/2) − ˜V hr,k+1
µr,k+1/2

((cid:92)xr,k+1)

+ ηc⟨gr,k+1/2 − gr,k, (cid:92)xr,k+1/2 − (cid:92)xr,k+1⟩ + ηc⟨gr,k+1/2, x − (cid:92)xr,k+1/2⟩

Proof. The proof of this Lemma is almost identical to the proof of Lemma 1 with a mere change of
variables and distance-generating function from saddle point setting to convex setting.

The following Lemma highlights the primary difference in the analysis of convex optimization and
saddle point optimization. The smoothness of fm provides an alternative presentation to gradient
Lipschitzness that establishes the connection between (cid:92)xr,k+1/2, the primal projection of averaged
dual on the central server, and xm
Lemma 16 (Bounding the Smooth Term). ∀x,

r,k+1/2 on each client.

f ( (cid:92)xr,k+1/2) − f (x) ≤ ⟨gr,k+1/2, (cid:92)xr,k+1/2 − x⟩ + ⟨

1
M

M
(cid:88)

m=1

gm(xm

r,k+1/2) − gr,k+1/2, (cid:92)xr,k+1/2 − x⟩

+

β
2M

M
(cid:88)

m=1

∥ (cid:92)xr,k+1/2 − xm

r,k+1/2∥2.

37

Published as a conference paper at ICLR 2024

Proof. By the smoothness fm in the form of Assumption 5b and then the convexity of fm in the form
of Assumption 5a,

fm( (cid:92)xr,k+1/2) ≤ fm(xm

r,k+1/2) + ⟨gm(xm

r,k+1/2), (cid:92)xr,k+1/2 − xm

≤ fm(xm

r,k+1/2) + ⟨gm(xm

+ fm(x) − fm(xm

r,k+1/2), (cid:92)xr,k+1/2 − xm
r,k+1/2), xm

r,k+1/2) + ⟨gm(xm

r,k+1/2⟩ +

β
2
β
2
r,k+1/2 − x⟩

r,k+1/2⟩ +

∥ (cid:92)xr,k+1/2 − xm

r,k+1/2∥2

∥ (cid:92)xr,k+1/2 − xm

r,k+1/2∥2

≤ fm(x) + ⟨gm(xm

r,k+1/2), (cid:92)xr,k+1/2 − x⟩ +

β
2

∥ (cid:92)xr,k+1/2 − xm

r,k+1/2∥2

Then for function f = 1
M

(cid:80)M

m=1 fm,

f ( (cid:92)xr,k+1/2) − f (x) ≤

1
M

≤ ⟨

1
M

M
(cid:88)

(cid:2)fm( (cid:92)xr,k+1/2) − fm(x)(cid:3)

m=1

M
(cid:88)

m=1

gm(xm

r,k+1/2), (cid:92)xr,k+1/2 − x⟩ +

1
M

M
(cid:88)

m=1

β
2

∥ (cid:92)xr,k+1/2 − xm

r,k+1/2∥2

= ⟨gr,k+1/2, (cid:92)xr,k+1/2 − x⟩ + ⟨

1
M

M
(cid:88)

m=1

+

β
2M

M
(cid:88)

m=1

∥ (cid:92)xr,k+1/2 − xm

r,k+1/2∥2.

gm(xm

r,k+1/2) − gr,k+1/2, (cid:92)xr,k+1/2 − x⟩

Now we are ready to present the main lemma that combines Lemma 15 and Lemma 16. For the proof,
we utilize again Lemma 11, Lemma 12, and Lemma 14, all of which we claim to hold trivially in the
composite convex optimization setting.

Lemma 17 (Main Lemma for FeDualEx in Composite Convex Optimization). Under Assumption 5,

ηcE(cid:2)ϕ( (cid:92)xr,k+1/2) − ϕ(x)(cid:3) ≤ ˜V hr,k
µr,k

(x) − ˜V hr,k+1
µr,k+1

(x) +

5σ2ηc
M

+ 10β2(ηc)3(2k2 + 2k + 1)G2

+

(ηc)2σ2
2M (1 − ηc)

+ 2β(ηc)3(k + 1)2G2.

Proof. Summing the results in Lemma 15 and Lemma 16:

ηc(cid:0)ϕ( (cid:92)xr,k+1/2) − ϕ(x)(cid:1) ≤ ˜V

hr,k
µr,k

(x) − ˜V

hr,k+1
µr,k+1

(x) − ˜V

hr,k
µr,k

( (cid:92)xr,k+1/2) − ˜V

hr,k+1
µr,k+1/2

((cid:92)xr,k+1)

+ ηc⟨gr,k+1/2 − gr,k, (cid:92)xr,k+1/2 − (cid:92)xr,k+1⟩ +

ηcβ
2M

M
(cid:88)

∥ (cid:92)xr,k+1/2 − xm

r,k+1/2∥2

m=1

+ ηc⟨

1
M

M
(cid:88)

m=1

gm(xm

r,k+1/2) − gr,k+1/2, (cid:92)xr,k+1/2 − x⟩.

38

Published as a conference paper at ICLR 2024

( (cid:92)xr,k+1/2) − ˜V hr,k+1
For the latter two generalized Bregman divergence terms − ˜V hr,k
µr,k+1/2
µr,k
bound them by Lemma 10 and the strong convexity of h in Assumption 3. As a result,

((cid:92)xr,k+1), we

ηc(cid:0)ϕ( (cid:92)xr,k+1/2) − ϕ(x)(cid:1) ≤ ˜V hr,k
µr,k
1
2

−

(x) − ˜V hr,k+1
µr,k+1

(x) −

1
2

∥(cid:100)xr,k − (cid:92)xr,k+1/2∥2

∥ (cid:92)xr,k+1/2 − (cid:92)xr,k+1∥2 + ηc⟨gr,k+1/2 − gr,k, (cid:92)xr,k+1/2 − (cid:92)xr,k+1⟩
(cid:125)

(cid:124)

(cid:123)(cid:122)
A

+ ⟨

ηc
M

+

ηcβ
2M

M
(cid:88)

m=1

M
(cid:88)

gm(xm

r,k+1/2) − gr,k+1/2, (cid:92)xr,k+1/2 − x⟩

∥ (cid:92)xr,k+1/2 − xm

r,k+1/2∥2.

m=1

A can be bounded with Cauchy-Schwarz inequality (Lemma 5) and Young’s inequality (Lemma 6).

A ≤ −

∥ (cid:92)xr,k+1/2 − (cid:92)xr,k+1∥2 + ηc∥gr,k+1/2 − gr,k∥∗∥ (cid:92)xr,k+1/2 − (cid:92)xr,k+1∥

≤ −

∥ (cid:92)xr,k+1/2 − (cid:92)xr,k+1∥2 +

(ηc)2
2

∥gr,k+1/2 − gr,k∥2

∗ +

1
2

∥ (cid:92)xr,k+1/2 − (cid:92)xr,k+1∥2

1
2
1
2
(ηc)2
2

=

∥gr,k+1/2 − gr,k∥2
∗.

Taking expectations on both sides we get
ηcE(cid:2)ϕ( (cid:92)xr,k+1/2) − ϕ(x)(cid:3) ≤ ˜V hr,k
µr,k
1
2

−

(x) − ˜V hr,k+1
µr,k+1

(x)

E(cid:2)∥(cid:100)xr,k − (cid:92)xr,k+1/2∥2(cid:3)
(cid:123)(cid:122)
(cid:125)
B1

+

(ηc)2
2

(cid:124)

E(cid:2)∥gr,k+1/2 − gr,k∥2

∗

(cid:3)

(cid:123)(cid:122)
B2

(cid:125)

(cid:124)

ηc
M

M
(cid:88)

m=1

+ E(cid:2)⟨

(cid:124)

gm(xm

r,k+1/2) − gr,k+1/2, (cid:92)xr,k+1/2 − x⟩(cid:3)

(cid:123)(cid:122)
B3

(cid:125)

M
(cid:88)

m=1

+

ηcβ
2M
(cid:124)

E(cid:2)∥ (cid:92)xr,k+1/2 − xm

r,k+1/2∥2(cid:3)

.

(cid:123)(cid:122)
B4

(cid:125)

B2 is bounded in Lemma 14. Therefore, for ηc ≤ 1
1
2 β
5

,

B1 + B2 ≤

5σ2(ηc)2
M

+ 20β2(ηc)4(k + 1)2G2.

B3 is zero after taking the expectation by Lemma 11. B4 is bounded in Lemma 12. Plugging the
bounds for B1 + B2, B3, and B4 back in completes the proof.

G FEDUALEX IN OTHER SETTINGS

In this section, we provide the algorithm along with the convergence rate for sequential versions of
FeDualEx. The proofs in this section rely only on the Lipschitzness of the gradient operator. As
a result, the analysis applies to both composite saddle point optimization and composite convex
optimization.

G.1 STOCHASTIC DUAL EXTRAPOLATION FOR COMPOSITE SADDLE POINT OPTIMIZATION

The sequential version of FeDualEx immediately yields Algorithm 3, stochastic dual extrapolation
for Composite SPP. This algorithm generalizes dual extrapolation to both composite and smooth

39

Published as a conference paper at ICLR 2024

Algorithm 3 STOCHASTIC-DUAL-EXTRAPOLATION for Composite SPP
Input: ϕ(z) = f (x, y) + ψ1(x) − ψ2(y): objective function; ℓ(z): distance-generating function;

g(z) = (∇xf (x, y), −∇yf (x, y)): gradient operator.
Hyperparameters: T : number of iterations; η: step size.
Dual Initialization: ς0 = 0: initial dual variable, ¯ς ∈ S: fixed point in the dual space.
Output: Approximate solution z = (x, y) to minx∈X maxy∈Y ϕ(x, y)

for t = 0, 1, . . . , T − 1 do
ℓt
zt = ˜Prox
¯ς (ςt)
ℓt
zt+1/2 = ˜Prox
(ηcg(zt; ξt))
¯ς−ςt
ςt+1 = ςt + ηcg(zt+1/2; ξt+1/2)

end for
Return: 1
T

(cid:80)T −1

t=0 zt+1/2.

▷ Two-step evaluation of the generalized proximal operator

▷ Dual variable update

stochastic saddle point optimization with the latter taking ψ(z) = 0. Its convergence rate is analyzed
in the following theorem, which to the best of our knowledge, is the first one for stochastic composite
saddle point optimization.

Theorem 3. Under the sequential version of Assumption 1-4, namely with M = 1, ∀z ∈ Z, the
ergodic intermediate sequence generated by Algorithm 3 satisfies

E(cid:2) Gap(

1
T

T −1
(cid:88)

t=0

zt+1/2)(cid:3) ≤

B
ηT

+ 3σ2η.

Choosing step size

η = min{

1
3 1

2 β

,

further yields the following convergence rate:

2

B 1
2 σT 1

2

3 1

},

E(cid:2) Gap(

1
T

T −1
(cid:88)

t=0

zt+1/2)(cid:3) ≤

3 1
2 βB
T

+

3 1

2

2 σB 1
T 1

2

.

Proof. By proof similar to Lemma 1, we have

η(cid:2)ψ(zt+1/2) − ψ(z)(cid:3) = ˜V ℓt

ωt

(z) − ˜V ℓt+1
ωt+1

(z) − ˜V ℓt
ωt

(zt+1/2) − ˜V ℓt+1
ωt+1/2

(zt+1)

+ η⟨gt+1/2 − gt, zt+1/2 − zt+1⟩ + η⟨gt+1/2, z − zt+1/2⟩

≤ ˜V ℓt
ωt
1
2

−

(z) − ˜V ℓt+1
ωt+1

(z)

∥zt − zt+1/2∥2 −

1
2

∥zt+1/2 − zt+1∥2 + η⟨gt+1/2 − gt, zt+1/2 − zt+1⟩
(cid:125)

(cid:124)

(cid:123)(cid:122)
A
+ η⟨g(zt+1/2) − gt+1/2, zt+1/2 − z⟩
−η⟨g(zt+1/2), zt+1/2 − z⟩.
(cid:125)
(cid:123)(cid:122)
B

(cid:124)

where the inequality holds by Lemma 10 and the strong convexity of ℓ in Remark 4, and then simply
expanding the last term to build a connection between the stochastic gradient and true gradient. By

40

Published as a conference paper at ICLR 2024

Algorithm 4 COMPOSITE-DUAL-EXTRAPOLATION
Input: ϕ(z) = f (x, y) + ψ1(x) − ψ2(y): objective function; ℓ(z): distance-generating function;

g(z) = (∇xf (x, y), −∇yf (x, y)): gradient operator.
Hyperparameters: T : number of iterations; η: step size.
Dual Initialization: ς0 = 0: initial dual variable, ¯ς ∈ S: fixed point in the dual space.
Output: Approximate solution z = (x, y) to minx∈X maxy∈Y ϕ(x, y)

for t = 0, 1, . . . , T − 1 do
ℓt
zt = ˜Prox
¯ς (ςt)
ℓt
zt+1/2 = ˜Prox
¯ς−ςt
ςt+1 = ςt + ηcg(zt+1/2)

(ηcg(zt))

end for
Return: 1
T

(cid:80)T −1

t=0 zt+1/2.

▷ Two-step evaluation of the generalized proximal operator

▷ Dual variable update

Cauchy-Schwarz inequality (Lemma 5), Young’s inequality (Lemma 6), and Lemma 7,

A ≤ −

∥zt − zt+1/2∥2 −

∥zt+1/2 − zt+1∥2 +

η2
2

∥gt+1/2 − gt∥2

∗ +

1
2

∥zt+1/2 − zt+1∥2

= −

∥zt − zt+1/2∥2 +

∥[gt+1/2 − g(zt+1/2)] + [g(zt) − gt] + [g(zt+1/2) − g(zt)]∥2
∗

1
2
1
2
1
2
3η2
2
3η2β2 − 1
2

+

1
2
η2
2
3η2
2

3η2
2

∗ +
3η2
2

≤ −

∥zt − zt+1/2∥2 +

∥g(zt+1/2) − g(zt)∥2
∗

∥gt+1/2 − g(zt+1/2)∥2

∥g(zt) − gt∥2
∗

≤

∥zt − zt+1/2∥2 +

∥gt+1/2 − g(zt+1/2)∥2

∗ +

3η2
2

∥g(zt) − gt∥2
∗,

where the last inequality holds by the β-Lipschitzness of the gradient operator. After taking expec-
tations, the last two terms are bounded by the variance of the gradient σ2, and B becomes zero by
proof similar to Lemma 11. Therefore, for η ≤ 1√
3β

ηE(cid:2)⟨g(zt+1/2), zt+1/2 − z⟩ + ψ(zt+1/2) − ψ(z)(cid:3) ≤ ˜V ℓt

ωt

(z) − ˜V ℓt+1
ωt+1

(z) + 3η2σ2.

Telescoping over all t ∈ {0, ..., T − 1} and dividing both sides by ηT completes the proof.

G.2 DETERMINISTIC DUAL EXTRAPOLATION FOR COMPOSITE SADDLE POINT

OPTIMIZATION

Further removing the data-dependent noise in the gradient, we present the deterministic sequential
version of FeDualEx, which still generalizes Nesterov’s dual extrapolation (Nesterov, 2007) to
composite saddle point optimization. As a result, we term this algorithm composite dual extrapolation,
as presented in Algorithm 4.

We also provide a convergence analysis, which shows that composite dual extrapolation achieves the
O( 1
T ) convergence rate as its original non-composite smooth version (Nesterov, 2007), as well as
composite mirror prox (CoMP) (He et al., 2015). We do so with a very simple proof based on the
recently proposed notion of relative Lipschitzness (Cohen et al., 2021). We start by introducing the
definition of relative Lipschitzness and a relevant lemma.
Definition 11 (Relative Lipschitzness (Definition 1 in Cohen et al. (2021))). For convex distance-
generating function h : Z → R, we call operator g : Z → Z ∗ λ-relatively Lipschitz with respect to
h if ∀z, w, u ∈ Z,

⟨g(w) − g(z), w − u⟩ ≤ λ(V h

z (w) + V h

w (u)).

Lemma 18 (Lemma 1 in Cohen et al. (2021)). If g is β-Lipschitz and h is α-strongly convex, g is
β
α -relatively Lipschitz with respect to h.
Theorem 4. Under the basic convexity assumption and β-Lipschitzness of g, ∀z ∈ Z and η ≤ 1
β ,
(cid:80)T −1
composite dual extrapolation satisfies Gap( 1
T

t=0 zt+1/2) ≤ βB
T .

41

Published as a conference paper at ICLR 2024

Proof. By proof similar to Lemma 1, we have

η(cid:2)ψ(zt+1/2) − ψ(z)(cid:3) = ˜V ℓt

ωt

(z) − ˜V ℓt+1
ωt+1

(zt+1/2) − ˜V ℓt+1
ωt+1/2
+ η⟨g(zt+1/2) − g(zt), zt+1/2 − zt+1⟩ + η⟨g(zt+1/2), z − zt+1/2⟩.

(z) − ˜V ℓt
ωt

(zt+1)

By Lemma 18, we know that g is β-relatively Lipschitz with respect to ℓ under the β-Lipschitzness
assumption of g and 1-strong convexity assumption of ℓ. Then by Definition 11, we have
η(cid:2)ψ(zt+1/2) − ψ(z) + ⟨g(zt+1/2), zt+1/2 − z⟩(cid:3)
(zt+1/2) − ˜V ℓt+1
(z) − ˜V ℓt
ωt+1/2
ωt
(zt+1/2) − ˜V ℓt+1
(z) − ˜V ℓt
ωt+1/2
ωt

(zt+1) + ηc⟨g(zt+1/2) − g(zt), zt+1/2 − zt+1⟩
(zt+1) + ηcβ(cid:2)V ℓ
(zt+1/2) + V ℓ
zt

(zt+1)(cid:3)

zt+1/2

≤ ˜V ℓt
ωt
≤ ˜V ℓt
ωt
≤ ˜V ℓt
ωt

(z) − ˜V ℓt+1
ωt+1
(z) − ˜V ℓt+1
ωt+1
(z) − ˜V ℓt+1
ωt+1

(z).

where the last inequality holds for η ≤ 1
dividing both sides by ηT completes the proof.

β by Lemma 10. Telescoping over all t ∈ {0, ..., T − 1} and

H FEDERATED MIRROR PROX

We present Federated Mirror Prox (FedMiP) here in Algorithm 2 as a baseline. The part highlighted
in green resembles the mirror prox algorithm introduced in Section C.1.2. We use the composite
mirror map representation introduced in Section C.1.1 to avoid confusion, as the composite proximal
operator we proposed for FeDualEx is slightly different from that used in composite mirror descent
as discussed in Section 4.1.

Algorithm 2 FEDERATED-MIRROR-PROX (FedMiP) for Composite SPP
Input: ϕ(z) = f (x, y) + ψ1(x) − ψ2(y) = 1
M

(cid:80)M
m=1 fm(·) + ψ1(x) − ψ2(y): objective function;
ℓ(z): distance-generating function; gm(z) = (∇xfm(x, y), −∇yfm(x, y)): gradient operator.
Hyperparameters: R: number of rounds of communication; K: number of local update iterations;

zm
r,0 = zr
for k = 0, 1, . . . , K − 1 do

Sample a subset of clients Cr ⊆ [M ]
for m ∈ Cr in parallel do

ηs: server step size; ηc: client step size.
Primal Initialization: z0: initial primal variable.
Output: Approximate solution z = (x, y) to minx∈X maxy∈Y ϕ(x, y)
1: for r = 0, 1, . . . , R − 1 do
2:
3:
4:
5:
6:
7:
8:
9:
10: ∆r = 1
|Cr|
11:
12: end for
13: Return:

zm
r,k+1/2 = ∇(ℓ + ηcψ)∗(∇h(zm
r,k+1 = ∇(ℓ + ηcψ)∗(∇h(zm
zm
end for

(zm
zr+1 = ∇(ℓ + ηsηcKψ)∗(∇h(zr) + ηs∆r)

r,k; ξm
r,k))
r,k+1/2; ξm

r,k) − ηcg(zm

r,k) − ηcg(zm

end parallel for

r,K − zm

(cid:80)K−1

r,0)

m∈Cr

(cid:80)

k=0 zr,k+1/2.

(cid:80)R−1
r=0

1
RK

r,k+1/2))

42

