Under review as a conference paper at ICLR 2024

DISTRIBUTIONAL OFF-POLICY EVALUATION WITH
BELLMAN RESIDUAL MINIMIZATION

Anonymous authors
Paper under double-blind review

ABSTRACT

We consider the problem of distributional off-policy evaluation which serves as
the foundation of many distributional reinforcement learning (DRL) algorithms.
In contrast to most existing works (that rely on supremum-extended statistical dis-
tances), we study the expectation-extended statistical distance for quantifying the
distributional Bellman residuals and provide the corresponding theoretical sup-
ports. Extending the framework of Bellman residual minimization to DRL, we
propose a method called Energy Bellman Residual Minimizer (EBRM) to estimate
the return distribution. We establish a finite-sample error bound for the EBRM es-
timator under the realizability assumption. Additionally, we introduce a variant of
our method based on a multi-step bootstrapping procedure to enable multi-step ex-
tension. By selecting an appropriate step level, we obtain a better error bound for
this variant of EBRM compared to a single-step EBRM, under non-realizability
settings. Finally, we demonstrate the superior performance of our method through
simulation studies, comparing with other existing methods.

1

INTRODUCTION

In reinforcement learning (RL), the cumulative (discounted) reward, also known as the return, is a
crucial quantity for evaluating the performance of a policy. Most existing RL methods focus on only
the expectation of the return distribution. In Bellemare et al. (2017a), the focus has been extended
to the whole return distribution, and they introduce a distributional RL (DRL) algorithm (hereafter
called Categorical algorithm) that achieves a considerably better performance in Atari games than
expectation-oriented Deep-Q Networks (Mnih et al., 2015). This has sparked significant interests
among the RL community, and was later followed by a series of quantile-based methods including
QRDQN, QRTD (Dabney et al., 2018b), IQN (Dabney et al., 2018a), FQF (Yang et al., 2019),
EDRL (Rowland et al., 2019) and particle-based methods including MMDRL (Nguyen-Tang et al.,
2021), SinkhornDRL (Sun et al., 2022), MD3QN (Zhang et al., 2021). In this paper, we consider
the problem of off-policy evaluation in DRL, i.e., estimating the (conditional) return distribution of
a target policy based on offline data.

Despite their competitive performances, distributional RL methods are significantly underdeveloped
compared with the traditional expectation-based RL, especially in the theoretical development under
the offline setting. All aforementioned methods are motivated by supremum-extended distances due
to the contraction property (see (4) below), but their algorithms essentially minimize an expectation-
extended distance (see (6)), as summarized in the column “Distance Mismatch” of Table 1. This
leads to a theory-practice gap. Also, most of these work does not provide any statistical guarantee
such as the convergence rate. We note that Rowland et al. (2018) establishes the consistency of their
estimator, but no error bound analysis (and convergence rate) is provided. In terms of statistical
analysis, a very recent work FLE (Wu et al., 2023) only offers error bound analysis of their estimator
for the marginal distribution of return, which is hard to use for policy learning. In addition, their
analysis is based on a strong condition called completeness, which in general significantly restricts
model choices of return distributions and excludes the non-realizable scenario.

This paper proposes novel estimators, which we call Energy Bellman Residual Minimizer (EBRM),
based on the idea of Bellman residual minimization for the conditional distribution of the return.
In contrast to existing work, we provide solid theoretical ground for the application of expectation-
extended distance in measuring (distributional) Bellman residual. A multi-step extension of our

1

Under review as a conference paper at ICLR 2024

estimator is proposed for non-realizability settings. Our method comes with statistical error bound
analyses in both realizable and non-realizable settings. Table 1 provides some key comparisons
between our method and some existing works. More details is given in Table 3 in the Appendix
D.1. Finally, we summarize our contributions as follows. (1) We provide theoretical foundation
of the application of expectation-extended distance for Bellman residual minimization in DRL. See
Section 2.3. (2) We develop a novel distributional off-policy evaluation method (EBRM), together
with its finite-sample error bound. See Section 3. (3) We develop a multi-step extension of EBRM
for non-realizabile settings in Section 4. We also provide corresponding finite-sample error bound
under non-realizable settings. (4) Our numerical experiments in Section 5 demonstrate the strong
performance of EBRM compared with some baseline methods.

Table 1: Comparison among DRL methods in off-policy evaluation.

Method
Categorical (Bellemare et al., 2017a)
QRTD (Dabney et al., 2018b)
IQN (Dabney et al., 2018a)
FQF (Yang et al., 2019)
EDRL (Rowland et al., 2019)
MMDRL (Nguyen-Tang et al., 2021)
SinkhornDRL (Sun et al., 2022)
MD3QN (Zhang et al., 2021)
FLE (Wu et al., 2023)
EBRM (our method)

Distance
match
✗
✗
✗
✗
✗
✗
✗
✗
✓
✓

Statistical
error bound
✗
✗
✗
✗
✗
✗
✗
✗
✓
✓

Non-
realizable
NA
NA
NA
NA
NA
NA
NA
NA
NA
✓

Multi-
dimension
✓
✗
✗
✗
✗
✓
✓
✓
✓
✓

2 OFF-POLICY EVALUATION BASED ON BELLMAN EQUATION

2.1 BACKGROUND

We consider an off-policy evaluation (OPE) problem under the framework of infinite-horizon
Markov Decision Process (MDP), which is characterized by a state space S, a discrete action space
A, and a transition probability p : S × A → P(Rd × S) with P(X ) denoting the class of probability
measures over a generic space X . In other words, p defines a joint distribution of a d-dimensional
immediate reward and the next state conditioned on a state-action pair. At each time point, an action
is chosen by the agent based on a current state according to a (stochastic) policy, a mapping from S
to P(A). With the initial state-action pair (S(0), A(0)), a trajectory generated by such an MDP can
be written as {S(t), A(t), R(t+1)}t≥0. The return variable is defined as Z := (cid:80)∞
t=1 γt−1R(t) with
γ ∈ [0, 1) being a discount factor, based on which we can evaluate the performance of some target
policy π.

Traditional OPE methods are mainly focused on estimating the expectation of return Z under the
target policy π, whereas DRL aims to estimate the whole distribution of Z. Letting L(X) be the
probability measure of some random variable (or vector) X, our target is to estimate the collection
of return distributions conditioned on different initial state-action pairs (S(0), A(0)) = (s, a):

Υπ(s, a) := L

(cid:19)

γt−1R(t)

(cid:18) ∞
(cid:88)

t=1

, (R(t+1), S(t+1)) ∼ p(·|S(t), A(t)), A(t+1) ∼ π(·|S(t+1)),

(1)

collectively written as Υπ ∈ P(Rd)S×A. It is analogous to the Q-function in traditional RL, whose
evaluation at a state-action pair (s, a) is the expectation of the distribution Υπ(s, a). Our goal in this
paper is to use the offline data generated by the behavior policy b to estimate Υπ.

Similar to most existing DRL methods, our proposal is based on the distributional Bellman equa-
tion (Bellemare et al., 2017a). Define the distributional Bellman operator by T π : P(Rd)S×A →
P(Rd)S×A such that, for any Υ ∈ P(Rd)S×A,
(cid:90)

(gr,γ)#Υ(s′, a′)dπ(a′|s′)dp(r, s′|s, a),

(s, a) ∈ S × A,

(2)

(cid:0)T πΥ(cid:1)(s, a) :=

Rd×S×A

2

Under review as a conference paper at ICLR 2024

where (gr,γ)# : P(Rd) → P(Rd) maps the distribution of any random vector X to the distribution
of r + γX. One can show that Υπ is the unique solution to the distributional Bellman equation:

T πΥ = Υ.

(3)

Letting Zπ(s, a) be the random vector that follows the distribution Υπ(s, a), one can also express
the distributional Bellman equation (3) in a more intuitive way: for all (s, a) ∈ S × A,

Zπ(s, a) D= R + γZπ(S′, A′) where

(R, S′) ∼ p(·|s, a), A′ ∼ π(·|S′),

where D= refers to the equivalence in terms of the underlying distributions. Due to the distributional
Bellman equation (3), a sensible approach to find Υπ is based on minimizing the discrepancy be-
tween T πΥ and Υ with respect to Υ ∈ P(Rd)S×A, which will be called Bellman residual hereafter.
To proceed with this approach, two important issues need to be addressed. First, both T πΥ and Υ
are collections of distributions over Rd, based on which Bellman residual shall be quantified. Sec-
ond, T π may not be available and therefore needs to be estimated through data. We will focus on
the quantification of Bellman residual first, and defer the proposed estimator of T π and the formal
description of our estimator for Υπ to Section 3.

2.2 EXISTING MEASURES OF BELLMAN RESIDUALS

To quantify the discrepancy between the two sides of the distributional Bellman equation (3), one
can use a distance over P(Rd)S×A. Fixing a state-action pair, one can solely compare two dis-
tributions from P(Rd). Therefore, a common strategy is to start by selecting a statistical distance
η(·, ·) : P(Rd) × P(Rd) → [0, ∞], and then define an extended-distance over P(Rd)S×A through
combining the statistical distances over different state-action pairs. As shown in Table 3 in Appendix
D.1, most existing methods (e.g., Bellemare et al., 2017b;a; Nguyen et al., 2020) are based on some
supremum-extended distance η∞:

η∞(Υ1, Υ2) := sup
s,a

η

Υ1(s, a), Υ2(s, a)

.

(cid:26)

(cid:27)

(4)

Under various choices of η including Wasserstein-p metric with 1 ≤ p ≤ ∞ (Bellemare et al., 2017a;
Dabney et al., 2018b) and maximum mean discrepancy (Nguyen-Tang et al., 2021), it is shown that
T π is a contraction with respect to η∞. More specifically, η∞(T πΥ1, T πΥ2) ≤ γβ0 · η∞(Υ1, Υ2)
holds for any Υ1, Υ2 ∈ P(Rd)S×A, where the value of β0 > 0 depends on the choice of η. If η∞ is
a metric, then the contractive property implies, for any Υ ∈ P(Rd)S×A,

η∞(Υ, Υπ) ≤

∞
(cid:88)

k=1

(cid:8)(T π)k−1Υ, (T π)kΥ(cid:9) ≤

η∞

1
1 − γβ0

· η∞(Υ, T πΥ).

(5)

As such, minimizing Bellman residual measured by η∞ would be a sensible approach for finding
Υπ. However, as surveyed in Appendix D.1, most existing methods in practice essentially minimize
an empirical (and approximated) version of the expectation-extended distance defined by

¯η(Υ1, Υ2) := E(S,A)∼bµ η

(cid:26)

(cid:27)

Υ1(S, A), Υ2(S, A)

,

(6)

with (S, A) ∼ bµ. Here bµ = µ × b refer to data distribution over S × A induced by the behavior
policy b. With a slight abuse of notation, we will overload the notation bµ with its density (with
respect to some appropriate base measure of S × A, e.g., counting measure and Lebesgue measure).
We remark that (5) does not hold under ¯η because η∞ and ¯η are not necessarily equivalent for the
general state-action space, leading to a theory-practice gap in most methods (Column 1 of Table 1).

2.3 EXPECTATION-EXTENDED DISTANCE

Despite the implicit use of expectation-extended distances in some prior works, the corresponding
theoretical foundations are not well established. Regarding Bellman residual minimization, a very
natural and crucial question is:

In terms of an expectation-extended distance, does small Bellman residual of Υ
lead to closeness between Υ and Υπ?

3

Under review as a conference paper at ICLR 2024

To proceed, we focus on settings such that the state-action pairs of interest can be well covered by
bµ, as formally stated in the following assumption. Let qπ(s, a|˜s, ˜a) be the conditional probability
density of the next state-action pair at (s, a) conditional on the current state-action pair at (˜s, ˜a),
defined by the transition probability p and the target policy π.
Assumption 1. There exists pmin > 0 and pmax < ∞ such that bµ(s, a) ≥ pmin for all s, a ∈
S × A and qπ(s, a|˜s, ˜a) ≤ pmax for all (˜s, ˜a), (˜s, ˜a) ∈ S × A.

(s, a) be the probability density (or mass) of (S(t), A(t)) at (s, a), given (S(0), A(0)) ∼
that is

Let qπ:t
bµ
bµ and the target policy π. Assumption 1 implies uniformly bounded density ratio,
qπ:t
bµ

(s, a)/bµ(s, a) ≤ Csup(< ∞) for all t ∈ N, as proved in Appendix A.1.

In the following Theorem 1 (proved in Appendix A.2), we provide a solid ground for Bellman
residual minimization based on expectation-extended distances.
Theorem 1. Under Assumption 1, if the statistical distance η satisfies translation-invariance, scale-
sensitivity of order β0 > 0, convexity, and relaxed triangular inequality defined in Appendix A.2.1,
then we can bound the inaccuracy:

where B1(γ; β0) :=
and Csup is defined in (25).

1
2(1−γβ0 )

¯η(Υ, Υπ) ≤ 2CsupB1(γ; β0) · ¯η(Υ, T πΥ),

(7)
k=1 4kγ(2k−1−1)β0 < ∞ is an increasing function of γ ∈ (0, 1),

(cid:80)∞

Inequality (7) provides an analogy to Bound (5) for expectation-based distances, answering our prior
question positively for some expectation-extended distances. Note that Theorem 1 can be applied to
the settings with general state-action space, including continuous one.

In order to take advantage of Theorem 1, we should select a statistical distance that satisfies all the
properties stated in Theorem 1. One example is energy distance (Sz´ekely & Rizzo, 2013) as proved
in Appendix A.3, which is in fact a squared maximum mean discrepancy (Gretton et al., 2012) with
kernel k(x, y) = ∥x∥ + ∥y∥ − ∥x − y∥. The energy distance is defined as

E{L(X), L(Y)} := 2E∥X − Y∥ − E∥X − X′∥ − E∥Y − Y′∥,
(8)
where X′ and Y′ are independent copies of X and Y respectively, and X, X′, Y, Y′ are indepen-
dent. In below, we will use energy distance to construct our estimator.

3 ENERGY BELLMAN RESIDUAL MINIMIZER

3.1 ESTIMATED BELLMAN RESIDUAL

Despite applicability of Theorem 1 to general state-action space, we will focus on tabular case with
finite cardinality |S × A| < ∞ for simpler construction of estimation, which enables an in-depth
theoretical study under both realizable and non-realizable settings in Sections 3.2 and 4.3. But the
reward can be continuous. Our target objective of Bellman residual minimization is
bµ(s, a) · E(cid:8)Υ(s, a), T πΥ(s, a)(cid:9), where

¯E(Υ, T πΥ) =

(cid:88)

(9)

s,a

E(cid:8)Υ(s, a), T πΥ(s, a)(cid:9) = 2E∥Zα(s, a) − Z (1)
α (s, a) − Z (1)
α (s, a), Z (1)
where Zα(s, a), Zβ(s, a) ∼ Υ(s, a) and Z (1)
β (s, a) ∼ T πΥ(s, a) are all independent.
For the tabular case with offline data, we can estimate bµ and the transition p simply by empirical
distributions. That is, given observations D = {(si, ai, ri, s′

β (s, a)∥ − E∥Zα(s, a) − Zβ(s, a)∥
β (s, a)∥,

− E∥Z (1)

i=1, we consider

i)}N

ˆbµ(s, a) :=

N (s, a)
N

where N (s, a) :=

N
(cid:88)

i=1

1(cid:8)(si, ai) = (s, a)(cid:9),

and

(10)

ˆp(E|s, a) :=

(cid:40) 1

(cid:80)

N (s,a)
δ0,s(E)

i:(si,ai)=(s,a) δri,s′

i

(E)

if N (s, a) ≥ 1,
if N (s, a) = 0

for any measurable set E,

4

Under review as a conference paper at ICLR 2024

where δr,s′ is the Dirac measure at (r, s′). Based on this, we can estimate T π for any Υ ∈
P(Rd)S×A by the estimated transition ˆp and the target policy π, by replacing p of (2) with ˆp.
Denoting the conditional expectation by ˜E(· · · ) := E(· · · |D), we can compute

E(cid:8)Υ(s, a), ˆT πΥ(s, a)(cid:9) = 2˜E∥Zα(s, a) − ˆZ (1)
α (s, a) − ˆZ (1)
α (s, a), ˆZ (1)

β (s, a)∥ − ˜E∥Zα(s, a) − Zβ(s, a)∥
β (s, a)∥,
(11)
where Zα(s, a), Zβ(s, a) ∼ Υθ(s, a) and ˆZ (1)
β (s, a) ∼ ˆT πΥ(s, a) are all independent
conditioned on the observed data D that determines ˆT π. With the above construction, we can
estimate the objective function by

− ˜E∥ ˆZ (1)

ˆ¯E(Υ, ˆT πΥ) =

(cid:88)

s,a

ˆbµ(s, a) · E(cid:8)Υ(s, a), ˆT πΥ(s, a)(cid:9).

(12)

Now letting {Υθ : θ ∈ Θ} ⊆ P(Rd)S×A be the hypothesis class of Υπ, where each distribution
Υθ is indexed by an element of candidate space Θ, a special case of which is the parametric case
Θ ⊆ Rp. Then the proposed estimator of Υπ is Υˆθ where

ˆθ ∈ arg min
θ∈Θ

ˆ¯E(Υθ, ˆT πΥθ).

(13)

We call our method the Energy Bellman Residual Minimizer (EBRM) and summarize it in Algorithm
1. We will refer to the approach here as EBRM-single-step, as opposed to the multi-step extension
EBRM-multi-step in Section 4.2.

Algorithm 1 EBRM-single-step
Input: Θ, D = {(si, ai, ri, s′
Output: ˆθ
Estimate ˆbµ and ˆp.
Compute ˆθ = arg minθ∈Θ

i)}N

i=1

ˆ¯E(Υθ, ˆT πΥθ).

▷ Refer to Equation (10).
▷ Refer to Equations (11) and (12).

3.2 STATISTICAL ERROR BOUND

In this subsection, we will provide a statistical error bound for EBRM-single-step. As shown in
Table 1, most existing distributional OPE methods do not have a finite sample error bound for their
estimators. To the best of our knowledge, the only exception is the very recent work named FLE
(Wu et al., 2023), which is only able to analyze the marginal distribution of the return instead of
conditional distributions of the return on each state-action pair studied in this paper. In passing, we
also note that Rowland et al. (2018) also shows the consistency of their estimator, but no error bound
analysis (and so convergence rate) is provided. We will first focus on the realizability setting and
defer the analysis for the non-realizable case in Section 4.
Assumption 2. There exists a unique θ ∈ Θ such that Υπ(s, a) = Υθ(s, a) for all s, a ∈ S × A.

Note that realizability is a generally weaker assumption than the widely-assumed completeness as-
sumption (e.g., used in FLE (Wu et al., 2023)) which states that for all θ ∈ Θ, there exist θ′ ∈ Θ
such that T πΥθ = Υθ′, in that it implies realizability due to Υπ = limT →∞(T π)T Υθ under mild
conditions. In contrast with non-realizability settings (Section 4), the realizability assumption aligns
the minimizer of inaccuracy E(Υ, Υπ) (best approximation) and the minimizer of Bellman residual,
leading to stronger arguments and results.

Additionally, we make several mild assumptions regarding the transition probability p and the candi-
date space Θ, including the subgaussian rewards. A random variable (vector) X being subgaussian
implies its tail probability decaying as fast as gaussian distribution (e.g., gaussian mixture, bounded
random variable), quantified with finite subgaussian norm ∥X∥ψ2 < ∞, as explained in Appendix
A.4.
Assumption 3. For any θ ∈ Θ, the random element Z(s, a; θ), which follows Υθ(s, a), has finite
expectation with respect to their norms, and the reward distribution are subgaussian, i.e.,

sup
θ∈Θ

sup
s,a

E∥Z(s, a; θ)∥ < ∞ and

∥R(s, a)∥ψ2 < ∞.

sup
s,a

5

Under review as a conference paper at ICLR 2024

i)}N

i=1 are iid draws from bµ × p.

Assumption 4. The offline data D = {(si, ai, ri, s′
Assumption 5. There exists a metric ˜η over P(Rd)S×A such that diam(Θ; ˜η)
:=
supθ1,θ2∈Θ ˜η(θ1, θ2) < ∞, where ˜η(θ1, θ2) := ˜η(Υθ1, Υθ2 ). For arbitrary c ∈ Rd, γ1, γ2 ∈ [0, 1],
(s, a), (˜s, ˜a) ∈ S × A, letting Zi(s, a) ∼ Υi(s, a) be such that (Z1(s, a), Z3(s, a)) ∈ Rd × Rd and
(Z2(˜s, ˜a), Z4(˜s, ˜a)) ∈ Rd × Rd are mutually independent, ˜η should satisfy
(cid:12)
(cid:12)
E∥c + γ1Z1(s, a) − γ2Z2(˜s, ˜a)∥ − E∥c + γ1Z3(s, a) − γ2Z4(˜s, ˜a)∥
(cid:12)
(cid:12)

(14)

(cid:12)
(cid:12)
(cid:12)
(cid:12)

≤ γ1 · ˜η(Υ1, Υ3) + γ2 · ˜η(Υ2, Υ4).

Supremum-extended Wasserstein-1 metric W1,∞, which is shown to be a metric by Lemma 2 of
Bellemare et al. (2017b), is an example that satisfies (14), as proved in Appendix A.5. Then we can
obtain the convergence rate O((cid:112)log(N/δ)/N ) as follows, with the exact finite-sample error bound
demonstrated in Appendix A.6.7. Its proof can be found in Appendix A.6, and its special case for
Θ ⊆ Rp is covered in Corollary 3 of Appendix A.7.
Theorem 2. (Inaccuracy for realizable scenario) Under Assumptions 1–5, for any δ ∈ (0, 1),
given large enough sample size N ≥ N (δ), our estimator ˆθ ∈ Θ given by (13) satisfies the following
bound with probability at least 1 − δ,

(cid:114)

¯E(Υˆθ, Υπ) ≲

1
N

log(

(|S × A| + N )
δ

),

(15)

where N (δ) depends on the complexity of Θ and ≲ means bounded by the given bound (RHS)
multiplied by a positive number that does not depend on N , as defined in Appendix A.6.8.

4 NON-REALIZABLE SETTINGS

4.1 COMBATING NON-REALIZABILITY WITH MULTI-STEP EXTENSIONS

In the tabular case, most traditional OPE/RL methods do not suffer from model mis-specification and
thus realizability always holds. In contrast, in DRL, as our target is to estimate the conditional distri-
bution of return given any state-action pair, which is an infinite-dimensional object, non-realizability
could still happen. Hence understanding and analyzing DRL methods for the tabular case under the
non-realizable scenario is both important and challenging.

In the previous section under realizability, Theorem 1 played a fundamental role in our analysis.
Indeed, Theorem 1 is valid regardless of realizability (Assumption 2), and essentially implies

0 ≤ min
θ∈Θ

¯E(Υθ, Υπ) ≤ ¯E(Υθ∗ , Υπ) ≤ 2CsupB1(γ; β0) · ¯E(Υθ∗ , T πΥθ∗ ),

(16)

where θ∗ := arg minθ∈Θ ¯E(Υθ, T πΥθ). Violation of Assumption 2 (that is, non-realizability) im-
plies nonzero value of ¯E(Υθ∗ , T πΥθ∗ ) > 0, and so Theorem 1 no longer ensures that θ∗ has the
smallest inaccuracy among θ ∈ Θ. Thus non-realizability may lead to the following mismatch:
¯E(Υθ, Υπ) ̸= arg min
θ∈Θ

¯E(Υθ, T πΥθ) =: θ∗.

˜θ := arg min
θ∈Θ

(17)

Clearly, this mismatch is not due to sample variability, so it is unrealistic to hope that ˆθ defined by
(13) would necessarily converge in probability to ˜θ as N → ∞.

To solve this issue, we propose a new approach. Temporarily ignoring mathematical rigor, the
most important insight is that we can approximate (T π)mΥ ≈ Υπ with sufficiently large step level
m ∈ N. Thanks to the properties of energy distance, we have the following
| ¯E(Υθ, (T π)mΥθ) − ¯E(Υθ, Υπ)| ≤ Cγm,

for some constant C > 0.

(18)

sup
θ∈Θ

(See Appendix C.2.9 under assumptions of Theorem 3.) As m → ∞, the RHS of (18) shrinks to
zero, making m-step Bellman residual ¯E(Υθ, (T π)mΥθ) approximate the inaccuracy ¯E(Υθ, Υπ).
This leads the two minimizers to be close, as illustrated schematically in Figure 1:

θ(m)
∗

:= arg min
θ∈Θ

¯E(Υθ, (T π)mΥθ) ≈ arg min
θ∈Θ

¯E(Υθ, Υπ) =: ˜θ for large enough m.

(19)

6

Under review as a conference paper at ICLR 2024

One can intuitively guess that larger step level m is required when the extent of non-realizability
is large. Although multi-step idea has been widely employed for the purpose of improving sample
efficiency particularly in traditional RL (e.g., Chen et al., 2021), ours is the first approach to use it
in DRL for the purpose of overcoming non-realizability, to the best of our knowledge.

Figure 1: Larger m makes (T π)mΥθ ≈ Υπ in Energy Distance, and thereby leads to θ(m)

∗ ≈ ˜θ.

4.2 BOOTSTRAP OPERATOR

Generalizing from definition of ˆT π based on (10), we consider ˆZ (m)(s, a; θ) ∼ ( ˆT π)mΥθ(s, a)
as the distribution of an m-lengthed trajectories of tuples (s, a, r, s′) that is generated under the
estimated transition ˆp and the target policy π:

ˆZ (m)(s, a; θ) D=

m
(cid:88)

γt−1 ˆR(t) + γmZ( ˆS(m), ˆA(m); θ), where

(20)

t=1
( ˆR(t), ˆS(t)) ∼ ˆp(· · · | ˆS(t−1), ˆA(t−1))

and

ˆA(t) ∼ π(·| ˆS(t)) ∀t ≥ 1, ( ˆS(0), ˆA(0)) = (s, a).

Now we can define the estimated and the population Bellman residual, as well as the inaccuracy
function, along with their minimizers as:

ˆFm(θ) := ˆ¯E(cid:0)Υθ, ( ˆT π)mΥθ

ˆθ(m) := arg min
θ∈Θ

(cid:1), Fm(θ) := ¯E(cid:0)Υθ, (T π)mΥθ
Fm(θ),

θ(m)
∗

:= arg min
θ∈Θ

ˆFm(θ),

(cid:1), F (θ) := ¯E(cid:0)Υθ, Υπ
F (θ).

˜θ := arg min
θ∈Θ

(cid:1),

(21)

However, the estimation of m-step Bellman operator (20) generally requires computation of N m
trajectories (as discussed in Appendix B.1), which amounts to a heavy computational burden.
To alleviate such burden, we will instead bootstrap M ≪ N m many trajectories by first sampling
the initial state-action pairs (s(0)
) (1 ≤ i ≤ M ) from ˆbµ and then resampling the subsequent
i
) for m steps. Let ˆp(B)
r(t+1)
, a(t)
∼ ˆp(· · · |s(t)
m (· · · |s, a) be the
i
empirical probability measure of ((cid:80)m
, a(0)
) = (s, a). We
i
i
define the bootstrap operator as follows, with an abuse of notation BmZ(s, a; θ) ∼ BmΥθ(s, a),

, a(0)
i
i ) and a(t+1)

∼ π(·|s(t+1)
i
, s(m)
i

) conditioning on (s(0)

t=1 γt−1r(t)

, s(t+1)
i

i

i

i

BmZ(s, a; θ) :D=

m
(cid:88)

t=1

γt−1 ˆR(t)

b + γmZ( ˆS(m)

b

, ˆA(m)
b

; θ),

(22)

where

m
(cid:88)
(

t=1

γt−1 ˆR(t)

b , ˆS(m)

b

) ∼ ˆp(B)

m (· · · |s, a)

and

b ∼ π(·| ˆS(m)
ˆA(m)

b

).

Then we can compute our objective function and derive the bootstrap-based multi-step estimator.

m (θ) := ˆ¯E(cid:0)Υθ, BmΥθ
ˆF (B)

(cid:1)

and

ˆθ(B)
m := arg min
θ∈Θ

ˆF (B)

m (θ).

(23)

We will refer to this method as EBRM-multi-step, whose procedure is summarized in Algorithm 2.

4.3 STATISTICAL ERROR BOUND
In this section, we develop a theoretical guarantee for ¯E(cid:0)Υˆθ(B)
(cid:1), where Υ˜θ is the best one we can
achieve under the non-realizability. To proceed, we need to first deal with the parameter convergence
from ˆθ(B)
m to ˜θ, which relies on the following assumptions regarding the inaccuracy function F (·)
(21), distance ˜η (Assumpion 5), and candidate space Θ.

, Υ˜θ

m

7

Under review as a conference paper at ICLR 2024

Algorithm 2 EBRM-multi-step
Input: Θ, D = {(si, ai, ri, s′
Output: ˆθ(B)
m
Estimate ˆbµ and ˆp.
Randomly generate M tuples of ((cid:80)m
ˆθ(B)
m = arg minθ∈Θ

ˆ¯E(Υθ, BmΥθ).

i)}N

i=1, m, M

t=1 γt−1r(t)

i

▷ Refer to Equation (10).

, s(m)
i

) (1 ≤ i ≤ M ).

▷ Refer to Equations (22) and (23).

Assumption 6. The inaccuracy function (21) F (·) : Θ ⊂ Rp → R has a unique minimizer ˜θ,
and lower bounded by a polynomial of degree q ≥ 1. That is, for all θ ∈ Θ, we have F (θ) ≥
F (˜θ) + cq · ∥θ − ˜θ∥q for some constant cq > 0.
Assumption 7. The candidate space Θ is compact (i.e., diam(Θ; ∥ · ∥) < ∞). Furthermore, there
exists L > 0 such that

˜η(θ1, θ2) ≤ L∥θ1 − θ2∥

for ∀θ1, θ2 ∈ Θ.

Assumption 8. ˜η satisfies contractive property, i.e., ˜η(T πΥ1, T πΥ2) ≤ γ · ˜η(Υ1, Υ2), where T π
(2) may correspond to an arbitrary transition p(· · · |s, a).

Assumption 6 is used in quantifying the convergence rate. Compactness in Assumption 7 is for
ensuring the existence of a minimizer of the estimated objective function (23), which is proved to be
continuous in Appendix C.2.11. Compactness can be relaxed to “bounded” under mild conditions.
Assumption 8 makes (18) feasible, and thereby shrinks the disparity between Bellman minimizer
and the best approximation (19). This is satisfied by W1,∞ that also satisfies property (14), as
proved in Lemma 3 of Bellemare et al. (2017b).

Due to space constraints, we only present a simplified result below (proof in Appendix B.4), and a
more detailed version of the finite-sample error bound for a fixed m is given in Appendix B.4.3.
Theorem 3. Under Assumptions 1, 3–8, letting M = ⌊C1 ·N ⌋ and m = ⌊ 1
for arbitrary constants C1, C2 > 0, we have the optimal convergence rate of the upper bound

4 log(1/γ)(C2N/ log N )⌋

¯E(cid:0)Υˆθ(B)

, Υ˜θ

(cid:1) ≤ ˜Op

(cid:26)

(cid:20)

1
N 1/(4q)

·

log 1
γ

(cid:18) N

(cid:19)(cid:27)2/q(cid:21)
,

log N
where ˜Op indicates the rate of convergence up to logarithmic order.

m

The convergence rate of Theorem 3 is the result of the (asymptotically) optimal choice of M and
m. In our analysis, we notice a form of bias-variance trade-off in the selection of m, as explained
in Appendix B.4.4. Practically, we set M = N which works fine in the simulations of Section 5. A
practical rule of m will be discussed in Section 4.4.

Note that the finite-sample error bound in Appendix B.4.3 is applicable to the setting with m = 1 and
realizability assumption. For instance, assuming that the inaccuracy function F (θ) is lower-bounded
by a quadratic polynomial q = 2, it gives us the bound O[{log(N/δ1)/N }1/8] under the ideal case
where we can ignore the last two sources of inaccuracy specified in Appendix B.4.4, each associated
with bootstrap and non-realizability. We can see that it is much slower than the convergence rate
O((cid:112)log(N/δ)/N ) of Theorem 2, implying that it does not degenerate into Theorem 2. This is fun-
damentally due to a different proof structure that can be introduced via the application of Theorem
1 in the proof of Theorem 2, as intuitively explained in Appendix C.3.1. As explained earlier in
Section 4.1, Theorem 1 can be used effectively to construct convergence of ˆθ under realizability.

4.4 DATA-ADAPTIVE WAY OF SELECTING STEP LEVEL

We need to choose m in practice. We will apply Lepski’s rule (Lepskii, 1991). Since multi-step
construction includes bootstrapping from the observed samples D = {(si, ai, ri, s′
i=1 (Section
4.2), this enables us to form a confidence interval. Starting from large enough m, we can decrease
it until the intersection of the confidence intervals becomes a null set. To elaborate, given the data
m (say ˆθ(B)
D, we first generate multiple estimates of ˆθ(B)
m,j for 1 ≤ j ≤ J), and calculate the disparity

i)}N

8

Under review as a conference paper at ICLR 2024

from the single-step estimation which has no randomness once D is given, that is ˆ¯E(Υˆθ, Υˆθ(B)
)
(1 ≤ j ≤ J). Then we calculate their means and standard deviations, forming a (Mean ± SD)
interval for each m. Starting from a large enough value, we decrease m by one or more at a time,
and select the m that makes the intersection become a null set ∩k≥mI (k) = ∅. If we did not obtain
the null set until ∩k≥2I (k) ̸= ∅, then we use EBRM-single-step (13) without boostrap. Details are
explained in Algorithm 3 of Appendix D.3.1.

m,j

5 EXPERIMENTS

We assume a state space S = {1, 2, · · · , 30} and an action space A = {−1, 1}, each action rep-
resenting left or right. With the details of the environment in Appendix D.2.1, the initial state
distribution and behavior / target policies are

S ∼ Unif(cid:8)1, 2, · · · , 30(cid:9) and A ∼ b(·|S), where

b(a|s) = 1/2

for ∀s, a ∈ S × A,

π(−1|s) = 0

and π(1|s) = 1 for ∀s ∈ S.

(24)
We compare three methods: EBRM, FLE (Wu et al., 2023), and QRTD (Dabney et al., 2018b). Here,
we assume realizability where the correct model is known (details in Appendix D.2.2) under two
settings (with small and large variances), and the step level m for EBRM is chosen in a data-adaptive
way in Section 4.4. With other tuning parameter selections explained in Appendix D.3, we repeated
100 simulations with the given sample size for each case, whose mean and standard deviation (within
parenthesis) are recorded in Table 2. EBRM showed the lowest inaccuracy values measured by both
¯E(Υˆθ, Υπ) and W1(Υˆθ, Υπ), where W1 indicates expectation-extended (6) Wasserstein-1 metric.
We also performed simulations in non-realizable scenarios (Appendix D.2.3) with more variety of
sample sizes, and included Wasserstein-1 metric between marginal return distributions (Tables 8–13
of Appendix D.4). In most cases, EBRM showed outstanding performance.

Table 2: Mean ¯E-inaccuracy (top) and W1-inaccuracy (bottom) (standard deviation in parenthesis)
over 100 simulations under realizability (γ = 0.99). Smallest inaccuracy values are in boldface.

Small variance
5000
0.019
(0.022)
2.385
(2.883)
46.032
(30.909)

10000
0.008
(0.010)
1.220
(1.618)
49.402
(34.617)

2000
0.046
(0.060)
5.533
(6.448)
48.679
(34.323)

Small variance
5000
0.985
(0.388)
8.036
(5.091)
54.397
(22.259)

10000
0.782
(0.227)
5.694
(3.773)
57.145
(24.314)

2000
1.339
(0.651)
12.374
(7.843)
56.739
(23.716)

Large variance
5000
0.301
(0.354)
14.482
(16.101)
75.173
(21.515)

2000
0.728
(0.920)
24.603
(25.768)
105.274
(11.728)

10000
0.128
(0.167)
6.528
(7.814)
70.483
(33.965)

Large variance
5000
15.532
(6.117)
79.628
(46.772)
236.383
(22.376)

2000
21.221
(10.337)
101.232
(58.586)
274.405
(11.003)

10000
12.371
(3.595)
53.745
(33.948)
223.537
(38.935)

Sample size
EBRM (Ours)

FLE

QRTD

Sample size
EBRM (Ours)

FLE

QRTD

6 CONCLUSION

In this paper, we justify the use of expectation-extended distances for Bellman residual minimization
in DRL under general state-action space, based on which we propose a distributional OPE method
called EBRM. We establish finite sample error bounds of the proposed estimator for the tabular case
with or without realizability assumption. One interesting future direction is to extend EBRM to
non-tabular case via linear MDP (e.g., Lazic et al., 2020; Bradtke & Barto, 1996), as we will briefly
discuss in Appendix C.3.2.

9

Under review as a conference paper at ICLR 2024

REFERENCES

Marc G Bellemare, Will Dabney, and R´emi Munos. A distributional perspective on reinforcement

learning. In International conference on machine learning, pp. 449–458. PMLR, 2017a.

Marc G Bellemare, Ivo Danihelka, Will Dabney, Shakir Mohamed, Balaji Lakshminarayanan,
Stephan Hoyer, and R´emi Munos. The cramer distance as a solution to biased wasserstein gradi-
ents. arXiv preprint arXiv:1705.10743, 2017b.

Steven J Bradtke and Andrew G Barto. Linear least-squares algorithms for temporal difference

learning. Machine learning, 22:33–57, 1996.

Zaiwei Chen, Siva Theja Maguluri, Sanjay Shakkottai, and Karthikeyan Shanmugam. Finite-sample
analysis of off-policy td-learning via generalized bellman operators. Advances in Neural Infor-
mation Processing Systems, 34:21440–21452, 2021.

Will Dabney, Georg Ostrovski, David Silver, and R´emi Munos.

Implicit quantile networks for
distributional reinforcement learning. In International conference on machine learning, pp. 1096–
1105. PMLR, 2018a.

Will Dabney, Mark Rowland, Marc Bellemare, and R´emi Munos. Distributional reinforcement
learning with quantile regression. In Proceedings of the AAAI Conference on Artificial Intelli-
gence, volume 32, 2018b.

Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Sch¨olkopf, and Alexander Smola.
A kernel two-sample test. The Journal of Machine Learning Research, 13(1):723–773, 2012.

Jonas Moritz Kohler and Aurelien Lucchi. Sub-sampled cubic regularization for non-convex opti-

mization. In International Conference on Machine Learning, pp. 1895–1904. PMLR, 2017.

Nevena Lazic, Dong Yin, Mehrdad Farajtabar, Nir Levine, Dilan Gorur, Chris Harris, and Dale
Schuurmans. A maximum-entropy approach to off-policy evaluation in average-reward mdps.
Advances in Neural Information Processing Systems, 33:12461–12471, 2020.

OV Lepskii. On a problem of adaptive estimation in gaussian white noise. Theory of Probability &

Its Applications, 35(3):454–466, 1991.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Belle-
mare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level
control through deep reinforcement learning. nature, 518(7540):529–533, 2015.

Thanh Tang Nguyen, Sunil Gupta, and Svetha Venkatesh. Distributional reinforcement learning with
maximum mean discrepancy. Association for the Advancement of Artificial Intelligence (AAAI),
2020.

Thanh Nguyen-Tang, Sunil Gupta, and Svetha Venkatesh. Distributional reinforcement learning via
moment matching. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35,
pp. 9144–9152, 2021.

Mark Rowland, Marc Bellemare, Will Dabney, R´emi Munos, and Yee Whye Teh. An analysis
In International Conference on Artificial

of categorical distributional reinforcement learning.
Intelligence and Statistics, pp. 29–37. PMLR, 2018.

Mark Rowland, Robert Dadashi, Saurabh Kumar, R´emi Munos, Marc G Bellemare, and Will Dab-
ney. Statistics and samples in distributional reinforcement learning. In International Conference
on Machine Learning, pp. 5528–5536. PMLR, 2019.

Bodhisattva Sen. A gentle introduction to empirical process theory and applications. Lecture Notes,

Columbia University, 11:28–29, 2018.

Yi Su, Pavithra Srinath, and Akshay Krishnamurthy. Adaptive estimator selection for off-policy
evaluation. In International Conference on Machine Learning, pp. 9196–9205. PMLR, 2020.

Ke Sun, Yingnan Zhao, Yi Liu, Wulong Liu, Bei Jiang, and Linglong Kong. Distributional rein-

forcement learning via sinkhorn iterations. arXiv preprint arXiv:2202.00769, 2022.

10

Under review as a conference paper at ICLR 2024

G´abor J Sz´ekely and Maria L Rizzo. Energy statistics: A class of statistics based on distances.

Journal of statistical planning and inference, 143(8):1249–1272, 2013.

Roman Vershynin. High-dimensional probability: An introduction with applications in data science,

volume 47. Cambridge university press, 2018.

Jiayi Wang, Raymond KW Wong, and Xiaoke Zhang. Low-rank covariance function estimation
for multidimensional functional data. Journal of the American Statistical Association, 117(538):
809–822, 2022.

Runzhe Wu, Masatoshi Uehara, and Wen Sun. Distributional offline policy evaluation with predic-

tive error guarantees. arXiv preprint arXiv:2302.09456, 2023.

Derek Yang, Li Zhao, Zichuan Lin, Tao Qin, Jiang Bian, and Tie-Yan Liu. Fully parameterized quan-
tile function for distributional reinforcement learning. Advances in neural information processing
systems, 32, 2019.

Pushi Zhang, Xiaoyu Chen, Li Zhao, Wei Xiong, Tao Qin, and Tie-Yan Liu. Distributional re-
inforcement learning for multi-dimensional reward functions. Advances in Neural Information
Processing Systems, 34:1519–1529, 2021.

11

