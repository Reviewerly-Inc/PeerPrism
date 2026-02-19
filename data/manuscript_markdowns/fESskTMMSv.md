Under review as a conference paper at ICLR 2021

PRACTICAL MARGINALIZED IMPORTANCE SAMPLING
WITH THE SUCCESSOR REPRESENTATION

Anonymous authors
Paper under double-blind review

ABSTRACT

Marginalized importance sampling (MIS), which measures the density ratio be-
tween the state-action occupancy of a target policy and that of a sampling distri-
bution, is a promising approach for off-policy evaluation. However, current state-
of-the-art MIS methods rely on complex optimization tricks and succeed mostly
on simple toy problems. We bridge the gap between MIS and deep reinforcement
learning by observing that the density ratio can be computed from the successor
representation of the target policy. The successor representation can be trained
through deep reinforcement learning methodology and decouples the reward op-
timization from the dynamics of the environment, making the resulting algorithm
stable and applicable to high-dimensional domains. We evaluate the empirical
performance of our approach on a variety of challenging Atari and MuJoCo envi-
ronments.

1

INTRODUCTION

Off-policy evaluation (OPE) is a reinforcement learning (RL) task where the aim is to measure
the performance of a target policy from data collected by a separate behavior policy (Sutton &
Barto, 1998). As it can often be difﬁcult or costly to obtain new data, OPE offers an avenue for
re-using previously stored data, making it an important challenge for applying RL to real-world
domains (Zhao et al., 2009; Mandel et al., 2014; Swaminathan et al., 2017; Gauci et al., 2018).

Marginalized importance sampling (MIS) (Liu et al., 2018; Xie et al., 2019; Nachum et al., 2019a)
is a family of OPE methods which re-weight sampled rewards by directly learning the density ratio
between the state-action occupancy of the target policy and the sampling distribution. This approach
can have signiﬁcantly lower variance than traditional importance sampling methods (Precup et al.,
2001), which consider a product of ratios over trajectories, and is amenable to deterministic policies
and behavior agnostic settings where the sampling distribution is unknown. However, the body of
MIS work is largely theoretical, and as a result, empirical evaluations of MIS have mostly been
carried out on simple low-dimensional tasks, such as mountain car (state dim. of 2) or cartpole (state
dim. of 4). In comparison, deep RL algorithms have shown successful behaviors in high-dimensional
domains such as Humanoid locomotion (state dim. of 376) and Atari (image-based).

In this paper, we present a straightforward approach for MIS that can be computed from the suc-
cessor representation (SR) of the target policy. Our algorithm, the Successor Representation DIs-
tribution Correction Estimation (SR-DICE), is the ﬁrst method that allows MIS to scale to high-
dimensional systems, far outperforming previous approaches. In comparison to previous algorithms
which rely on minimax optimization or kernel methods (Liu et al., 2018; Nachum et al., 2019a;
Uehara & Jiang, 2019; Mousavi et al., 2020), SR-DICE requires only a simple convex loss ap-
plied to the linear function determining the reward, after computing the SR. Similar to the deep
RL methods which can learn in high-dimensional domains, the SR can be computed easily using
behavior-agnostic temporal-difference (TD) methods. This makes our algorithm highly amenable to
deep learning architectures and applicable to complex tasks.

Our derivation of SR-DICE also reveals an interesting connection between MIS methods and value
function learning. The key motivation for MIS methods is, unlike traditional importance sampling
methods, they can avoid variance with an exponential dependence on horizon, by re-weighting in-
dividual transitions rather than accumulating ratios along entire trajectories. We remark that while
the MIS ratios only consider individual transitions, the optimization procedure is still subject to the

1

Under review as a conference paper at ICLR 2021

dynamics of the underlying MDP. Subsequently, we use this insight to show a connection between
a well-known MIS method, DualDICE (Nachum et al., 2019a), and Bellman residual minimiza-
tion (Bellman, 1957; Baird, 1995), which can help explain some of the optimization properties and
performance of DualDICE, as well as other related MIS methods.

We benchmark the performance of SR-DICE on several high-dimensional domains in MuJoCo
(Todorov et al., 2012) and Atari (Bellemare et al., 2013), against several recent MIS methods. Our
results demonstrate two key ﬁndings regarding high-dimensional tasks.

SR-DICE signiﬁcantly outperforms the benchmark algorithms. We attribute this performance
gap to SR-DICE’s deep RL components, outperforming the MIS baselines in the same way that
deep RL outperforms traditional methods on high-dimensional domains. Unfortunately, part of this
performance gap is due to the fact that the baseline MIS methods scale poorly to challenging tasks.
In Atari we ﬁnd that the baseline MIS method exhibit unstable estimates, often reaching errors with
many orders of magnitude.

MIS underperforms deep RL. Although SR-DICE achieves a high performance, we ﬁnd its errors
are bounded by the quality of the SR. Consequently, we ﬁnd that SR-DICE and the standard SR
achieve a similar performance across all tasks. Worse so, we ﬁnd that using a deep TD method,
comparable to DQN (Mnih et al., 2015) for policy evaluation outperforms both methods. Although
the performance gap is minimal, for OPE there lacks a convincing argument for SR-DICE, or any
current MIS method, which introduce unnecessary complexity. However, this does not mean MIS
is useless. We remark that the density ratios themselves are an independent objective which have
been used for applications such as policy regularization (Nachum et al., 2019b; Touati et al., 2020),
imitation learning (Kostrikov et al., 2019), off-policy policy gradients (Imani et al., 2018; Liu et al.,
2019b; Zhang et al., 2019), and non-uniform sampling (Sinha et al., 2020). SR-DICE serves as a
stable, scalable approach for computing these ratios. We provide extensive experimental details in
the supplementary material and our code is made available.

2 BACKGROUND

Reinforcement Learning. RL is a framework for maximizing accumulated reward of an agent
interacting with its environment (Sutton & Barto, 1998). This problem is typically framed as a
Markov Decision Process (MDP) (S, A, R, p, d0, γ), with state space S, action space A, reward
function R, dynamics model p, initial state distribution d0 and discount factor γ. An agent selects
actions according to a policy π : S × A → [0, 1]. In this paper we address the problem of off-policy
evaluation (OPE) problem where the aim is to measure the normalized expected per-step reward of
the policy R(π) = (1 − γ)Eπ [(cid:80)∞
t=0 γtr(st, at)]. An important notion in OPE is the value function
Qπ(s, a) = Eπ[(cid:80)∞
t=0 γtr(st, at)|s0 = s, a0 = a], which measures the expected sum of discounted
rewards when following π, starting from (s, a).
We deﬁne dπ(s, a) as the discounted state-action occupancy, the probability of seeing (s, a) under
policy π with discount γ: dπ(s, a) = (1 − γ) (cid:80)∞
d0(s0)pπ(s0 → s, t)π(a|s)d(s0), where
pπ(s0 → s, t) is the probability of arriving at the state s after t time steps when starting from an
initial state s0. This distribution is important as R(π) equals the expected reward r(s, a) under dπ:
R(π) = E(s,a)∼dπ,r[r(s, a)].
(1)

t=0 γt (cid:82)

s0

Successor Representation. The successor representation (SR) (Dayan, 1993) of a policy is a
measure of occupancy of future states.
It can be viewed as a general value function that learns
a vector of the expected discounted visitation for each state. The successor representation Ψπ
of a given policy π is deﬁned as Ψπ(s(cid:48)|s) = Eπ[(cid:80)∞
t=0 γt1(st = s(cid:48))|s0 = s]. Importantly, the
value function can be recovered from the SR by summing over the expected reward of each state
V π(s) = (cid:80)
s(cid:48) Ψπ(s(cid:48)|s)Ea(cid:48)∼π[r(s(cid:48), a(cid:48))]. For inﬁnite state and action spaces, the SR can instead be
generalized to the expected occupancy over features, known as the deep SR (Kulkarni et al., 2016)
or successor features (Barreto et al., 2017). For a given encoding function φ : S ×A → Rn, the deep
SR ψπ : S × A → Rn is deﬁned as the expected discounted sum over features from the encoding
function φ when starting from a given state-action pair and following π:

ψπ(s, a) = Eπ

(cid:34) ∞
(cid:88)

t=0

(cid:12)
(cid:12)
γtφ(st, at)
(cid:12)
(cid:12)

(cid:35)

s0 = s, a0 = a

.

(2)

2

Under review as a conference paper at ICLR 2021

If the encoding φ(s, a) is learned such that the original reward function is a linear function of the
encoding r(s, a) = w(cid:62)φ(s, a), then similar to the original formulation of SR, the value function can
be recovered from a linear function of the SR: Qπ(s, a) = w(cid:62)ψπ(s, a). The deep SR network ψπ
is trained to minimize the MSE between ψπ(s, a) and φ(s, a) + γψ(cid:48)(s(cid:48), a(cid:48)) on transitions (s, a, s(cid:48))
sampled from the data set. A frozen target network ψ(cid:48) is used to provide stability (Mnih et al., 2015;
Kulkarni et al., 2016), and is updated to the current network ψ(cid:48) ← ψπ after a ﬁxed number of time
steps. The encoding function φ is typically trained by an encoder-decoder network (Kulkarni et al.,
2016; Machado et al., 2017; 2018a).

Marginalized Importance Sampling. Marginalized importance sampling (MIS) is a family of im-
portance sampling approaches for off-policy evaluation in which the performance R(π) is evaluated
by re-weighting rewards sampled from a data set D = {(s, a, r, s(cid:48))} ∼ p(s(cid:48)|s, a)dD(s, a), where
dD is an arbitrary distribution, typically but not necessarily, induced by some behavior policy. It
follows that R(π) can computed with importance sampling weights on the rewards dπ(s,a)
dD(s,a) :

R(π) = E(s,a)∼dD,r

(cid:20) dπ(s, a)
dD(s, a)

(cid:21)

r(s, a)

.

(3)

The goal of marginalized importance sampling methods is to learn the weights w(s, a) ≈ dπ(s,a)
dD(s,a) ,
using data contained in D. The main beneﬁt of MIS is that unlike traditional importance methods,
the ratios are applied to individual transitions rather than complete trajectories, which can reduce
the variance of long or inﬁnite horizon problems. In other cases, the ratios themselves can be used
for a variety of applications which require estimating the occupancy of state-action pairs.

DualDICE. Dual stationary DIstribution Correction Estimation (DualDICE) (Nachum et al., 2019a)
is a well-known MIS method which uses a minimax optimization to learn the density ratios. The
underlying objective which DualDICE aims to minimize is the following:

min
f

J(f ) :=

E(s,a)∼dD

(cid:104)

1
2

(f (s, a) − γEs(cid:48),π[f (s(cid:48), a(cid:48))])

2(cid:105)

− (1 − γ)Es0,a0∼π[f (s0, a0)].

(4)

It can be shown that Equation (4) is uniquely optimized by the MIS density ratio. However, since
f (s, a) − γEπ[f (s(cid:48), a(cid:48))] is dependent on transitions (s, a, s(cid:48)), there are two practical issues with
this underlying objective. First, the objective contains a square within an expectation, giving rise to
the double sampling problem (Baird, 1995), where the gradient will be biased when using only a
single sample of (s, a, s(cid:48)). Second, computing f (s, a) − γEs(cid:48),π[f (s(cid:48), a(cid:48))] for arbitrary state-action
pairs, particularly those not contained in the data set, is non-trivial, as it relies on an expectation over
succeeding states, which is generally inaccessible without a model of the environment. To address
both concerns, DualDICE uses Fenchel duality (Rockafellar, 1970) to create the following minimax
optimization problem:

min
f

max
w

J(f, w) := E(s,a)∼dD,a(cid:48)∼π,s(cid:48)

(cid:2)w(s, a)(f (s, a) − γf (s(cid:48), a(cid:48))) − 0.5w(s, a)2(cid:3)

− (1 − γ)Es0,a0[f (s0, a0)].

(5)

Similar to the original formulation, Equation (4), it can be shown that Equation (5) is minimized
when w(s, a) is the desired density ratio.

3 A REWARD FUNCTION PERSPECTIVE ON DISTRIBUTION CORRECTIONS

In this section, we present our behavior-agnostic approach to estimating MIS ratios, called the Suc-
cessor Representation DIstribution Correction Estimation (SR-DICE). Our main insight is that MIS
can be viewed as an optimization over the reward function, where the loss is uniquely optimized
when the reward is the desired density ratio. We then apply our reward function perspective on a
well-known MIS method, DualDICE (Nachum et al., 2019a), which enables us to observe difﬁcul-
ties in the optimization process and better understand related methods. All proofs for this section
are left to Appendix A.

3.1 THE SUCCESSOR REPRESENTATION DICE

We will now derive our MIS approach. Our derivation shows that by treating MIS as reward function
optimization, we can obtain the desired density ratios can be obtained in a straightforward manner

3

Under review as a conference paper at ICLR 2021

from the SR of the target policy. This pushes the challenging aspect of learning onto the computation
of the SR, rather than optimizing the density ratio estimate. Furthermore, when tackling high-
dimensional tasks, we can leverage deep RL approaches (Mnih et al., 2015; Kulkarni et al., 2016) to
make learning the SR stable, giving rise to a practical MIS method.
Our aim is to determine the MIS ratios dπ(s,a)
dD(s,a) , using only data sampled from the data set D and
the policy π. This presents a challenge as we have direct access to neither dπ nor dD. As a starting
point, we begin by following the derivation of DualDICE (Nachum et al., 2019a). We ﬁrst consider
the convex function 1
2 mx2 − nx, which is uniquely minimized by x∗ = n
m . Now by replacing x
with ˆr(s, a), m with dD(s, a), and n with dπ(s, a), we have reformulated the convex function as the
following objective:

min
ˆr(s,a)∀(s,a)

J(ˆr) :=

1
2

E(s,a)∼dD

(cid:2)ˆr(s, a)2(cid:3) − (1 − γ)E(s,a)∼dπ [ˆr(s, a)] .

(6)

While this objective is still impractical as it relies on expectations over both dD and dπ, from
Nachum et al. (2019a) we can state the following about Equation (6).

Observation 1 The objective J(ˆr) is minimized when ˆr(s, a) = dπ(s,a)

dD(s,a) , ∀(s, a).

Now we will diverge from the derivation of DualDICE. Note our choice of notation, ˆr(s, a), in
Equation (6). Describing the objective in terms of a ﬁctitious reward ˆr will allow us to draw on
familiar relationships between rewards and value functions and build stronger intuition. Consider
the equivalence between the value function over initial state-action pairs and the expectation of
rewards over the state-action visitation of the policy (1 − γ)Es0,a0[Qπ(s0, a0)] = Edπ [r(s, a)]. It
follows that the expectation over dπ in Equation (6) can be replaced with a value function ˆQπ over ˆr:

J(ˆr) :=

1
E(s,a)∼dD
min
2
ˆr(s,a)∀(s,a)
(cid:105)
(cid:104)
ˆQπ(s0, a0)

(cid:2)ˆr(s, a)2(cid:3) − (1 − γ)Es0,a0

(cid:104)

(cid:105)
ˆQπ(s0, a0)

.

(7)

Using (1 − γ)Es0,a0
= Edπ [ˆr(s, a)] provides a method for accessing the otherwise
intractable dπ. This form of the objective is convenient because we can estimate the expectation over
dD by sampling from the data set and Qπ can be computed using any policy evaluation method.

While we can estimate both terms in Equation (7) with relative ease, the optimization problem is not
directly differentiable and would require re-learning the value function ˆQπ with every adjustment
to the learned reward ˆr. Fortunately, there exists a straightforward paradigm which enables direct
reward function optimization known as successor representation (SR).
Consider the relationship between the SR Ψπ of the target policy π and its value function
Es0,a0[Qπ(s0, a0)] = Es0 [V π(s0)] = Es0 [(cid:80)
s Ψπ(s|s0)Eπ[r(s, a)]] in the tabular setting. It fol-
lows that we can create an optimization problem over the reward function ˆr from Equation (7):

min
ˆr(s,a)∀(s,a)

JΨ(ˆr) :=

1
2

E(s,a)∼dD

(cid:2)ˆr(s, a)2(cid:3) − (1 − γ)Es0

(cid:34)

(cid:88)

(cid:35)
Ψπ(s|s0)Ea∼π [ˆr(s, a)]

.

(8)

s

This objective can be generalized to continuous states by considering the deep SR ψπ over features
φ(s, a) and optimizing the weights of a linear function w. In this instance, the estimated density
ratio ˆr(s, a) is determined by w(cid:62)φ(s, a) and we can optimize w by minimizing the following:

min
w

J(w) :=

(cid:2)(w(cid:62)φ(s, a))2(cid:3) − (1 − γ)Es0,a0∼π
Since this optimization problem is convex, it has a closed form solution. Deﬁne D0 as the set of start
states contained in D. The unique optimizer of Equation (9) is as follows:

(cid:2)w(cid:62)ψπ(s0, a0)(cid:3) .

EdD

(9)

1
2

min
w

J(w) = (1 − γ)

|D|

(cid:80)

(s,a)∈D φ(s, a) (cid:80)

i φi(s, a)

1
|D0|

(cid:88)

s0∈D0

π(a0|s0)ψπ(s0, a0),

(10)

where φi is the ith entry of the vector φ. However, we may generally prefer iterative, gradient-
based solutions for scalability. We call the combination of learning the deep SR followed by op-
timizing Equation (9) the Successor Representation stationary DIstribution Correction Estimation

4

Under review as a conference paper at ICLR 2021

1

Algorithm 1 SR-DICE
1: At each time step sample mini-batch of N transitions (s, a, r, s(cid:48)) and start states s0 from D.
2: for t = 1 to T1 do
3: minφ,D
4: for t = 1 to T2 do
5: minψπ
6: for t = 1 to T3 do
7: minw
8: Output: R(π) estimate |D|−1 (cid:80)

2 (w(cid:62)φ(s, a))2 − (1 − γ)w(cid:62)ψπ(s0, a0).

2 (φ(s, a) + γψ(cid:48)(s(cid:48), a(cid:48)) − ψπ(s, a))2.

# Density ratio w loss (Equation (9))

2 (D(φ(s, a)) − (s, a))2.

# Deep successor representation ψπ loss

# Encoding φ loss

(s,a,r)∈D w(cid:62)φ(s, a)r(s, a).

1

1

(SR-DICE). SR-DICE is split into three learning phases: (1) learning the encoding φ, (2) learn-
ing the deep SR ψπ, and (3) optimizing Equation (9). For the ﬁrst two phases we follow standard
practices from prior work (Kulkarni et al., 2016; Machado et al., 2018a), training the encoding φ
via an encoder-decoder network to reconstruct the transition and training the deep SR ψπ using TD
learning-style methods. We summarize SR-DICE in Algorithm 1. Additional implementation-level
details can be found in Appendix D.
Although it is difﬁcult to make any guarantees on the accuracy of an approximate ψπ trained with
deep RL techniques, if we assume ψπ is exact, then we can show that SR-DICE learns the least
squares estimator to the desired density ratio.

Theorem 1 Assuming (1 − γ)Es0,a0 [ψπ(s0, a0)] = E(s,a)∼dπ [φ(s, a)], then the optimizer w∗ of
the objective J(w) is the least squares estimator of (cid:82)

w(cid:62)φ(s, a) − dπ(s,a)
dD(s,a)

d(s, a).

S×A

(cid:17)2

(cid:16)

Hence, the main sources of error in SR-DICE are learning the encoding φ and the deep SR ψπ.
Notably, both of these steps are independent of the main optimization problem of learning w, as
we have shifted the challenging aspects of density ratio estimation onto learning the deep SR. This
leaves deep RL to do the heavy lifting. The remaining optimization problem, Equation (9), only
involves directly updating the weights of a linear function, and unlike many other MIS methods,
requires no tricky minimax optimization.

SR-DICE can also be applied to any pre-existing SR, or included into standard deep RL algo-
rithms (Mnih et al., 2015; Lillicrap et al., 2015; Hessel et al., 2017; Fujimoto et al., 2018) by treating
the encoding φ as an auxiliary reward. This provides an alternate form of policy evaluation through
MIS, or a method to access density ratios between the target policy and the sampling distribution,
with possible applications to exploration, policy regularization, or unbiased off-policy gradients (Liu
et al., 2019b; Nachum et al., 2019b; Touati et al., 2020).

3.2 REWARD FUNCTIONS & MIS: A CASE STUDY ON DUALDICE

One of the main attractions for MIS methods is they use importance sampling ratios which re-weight
individual transitions rather than entire trajectories. While independent of the length of trajectories
collected by the behavior policy, we remark the optimization problem is not independent of the im-
plicit horizon deﬁned by the discount factor γ and MIS methods are still subject to the dynamics
of the underlying MDP. In SR-DICE we explicitly handle the dynamics of the MDP by learning
the SR with TD learning methods.
In this case study, we examine a well-known MIS method,
DualDICE (Nachum et al., 2019a), and discuss how it propagates updates through the MDP by
considering its relationship to residual algorithms which minimize the mean squared Bellman er-
ror (Baird, 1995). By viewing other MIS methods through the lens of reward function optimization,
we can understand their connection to value-based methods, shedding light on their optimization
properties and challenges.

Recall the underlying objective of DualDICE:

min
f

J(f ) :=

1
2

E(s,a)∼dD

(cid:104)
(f (s, a) − γEs(cid:48),π[f (s(cid:48), a(cid:48))])

2(cid:105)

− (1 − γ)Es0,a0∼π[f (s0, a0)].

(11)

By viewing the problem as reward function optimization, we can transform DualDICE into a more
familiar format that considers rewards and value functions. To begin, we state the following theorem.

5

Under review as a conference paper at ICLR 2021

Theorem 2 Given an MDP (S, A, ·, p, d0, γ), policy π, and function f : S × A → R, deﬁne the
reward function r : S × A → R where ˆr(s, a) = f (s, a) − γEs(cid:48),a(cid:48)∼π[f (s(cid:48), a(cid:48))]. Then it follows that
the value function ˆQπ deﬁned by the policy π, MDP, and reward function ˆr, is the function f .

The proof follows naturally from the Bellman equation (Bellman, 1957). Informally, Theorem 2
states that any function f can be treated as an exact value function ˆQπ, for a carefully chosen reward
function ˆr(s, a) = f (s, a) − γEs(cid:48),π[f (s(cid:48), a(cid:48))].
Theorem 2 provides two perspectives on DualDICE. By replacing terms in Equation (11) with re-
wards and value functions, it can be viewed as the same objective as Equation (7) from SR-DICE:

min
ˆr(s,a)∀(s,a)

J(ˆr) :=

1
2

E(s,a)∼dD

(cid:2)ˆr(s, a)2(cid:3) − (1 − γ)Es0,a0

(cid:104)

(cid:105)
ˆQπ(s0, a0)

.

(12)

The ﬁrst insight from this relationship is that like SR-DICE, DualDICE can be viewed as reward
function optimization and still requires some element of value learning. However, for DualDICE
the form of the reward and value functions are unique. From Theorem 2, we remark that f (s0, a0)
is always exactly ˆQπ(s0, a0) without additional computation. This occurs because f (s0, a0) is not a
function of the reward, rather, the rewards are deﬁned as a function of f . When the reward function
is adjusted, f (s0, a0) may remain unchanged and other rewards are adjusted to compensate.

To emphasize how DualDICE is subject to the properties of value learning, consider a second per-
spective on DualDICE taken from Theorem 2, where we replace f with ˆQπ:

J( ˆQπ) :=

min
ˆQπ

1
2
(cid:124)

(cid:20)(cid:16)

E(s,a)∼dD

ˆQπ(s, a) − γEs(cid:48),π

(cid:104)

ˆQπ(s(cid:48), a(cid:48))

(cid:105)(cid:17)2(cid:21)

−(1 − γ)Es0,a0

(cid:104)

ˆQπ(s0, a0)

(cid:105)

.

(cid:123)(cid:122)
Bellman residual minimization

(cid:125)

(13)

The ﬁrst term is equivalent to Bellman residual minimization (Bellman, 1957; Baird, 1995), where
the reward is 0 for all state-action pairs. The second term attempts to maximize only the initial value
function ˆQπ(s0, a0). From a practical perspective this relationship is concerning as the ﬁrst term
relies on successfully propagating updates throughout the MDP to balance out with changes to the
initial values, which may occur quickly. Consequently, in cases where DualDICE performs poorly,
we may see the initial values approach inﬁnity.

To understand how this objective performs em-
pirically, we measured the output of DualDICE
on a basic OPE task with an identical be-
havior and target policy.
In this case the
true MIS ratio is 1.0 for all state-action
pairs. Consequently, both the ﬁctiuous reward
EdD [f (s, a)−γEs(cid:48),π[f (s(cid:48), a(cid:48))]] and normalized
initial value function (1 − γ)Es0,a0[f (s0, a0)]
should approach 1.0.
In Figure 1, we graph
both EdD [w(s, a)], where w(s, a) ≈ f (s, a) −
γEs(cid:48),π[f (s(cid:48), a(cid:48))] is the ratio used by DualDICE
(Equation (5)), and (1 − γ)Es0,a0[f (s0, a0)]
output by DualDICE.

Figure 1: We plot the average values of E
dD [w(s, a)]
and (1 − γ)Es0,a0 [f (s0, a0)] output by DualDICE on
a task with an identical behavior and target policy, such
that the true value of both terms is 1. The 10 individual
trials are plotted lightly, with the mean in bold. The
estimates of DualDICE matches our hypothesis that
DualDICE overestimates f (s0, a0) as propagating up-
dates through the MDP occurs at a much slower rate.

While on the easier task, Pendulum, the per-
formance looks reasonable, on HalfCheetah we
can see that (1 − γ)Es0,a0 [f (s0, a0)] greatly
overestimates and EdD [w(s, a)] is highly unsta-
ble. This result is intuitive given the form of
Equation (13), where the ﬁrst term, which w(s, a) approximates, is pushed slowly towards 0 and the
second term is pushed towards ∞. On the lower dimensional problem, Pendulum, the objective is
optimized more easily and both terms approach 1.0. On the harder problem, HalfCheetah, we can
see how balancing residual learning, which is notoriously slow (Baird, 1995), with a maximization
term on initial states creates a difﬁcult optimzation procedure.

6

                   7 L P H  V W H S V    H                        0 , 6  5 D W L R 3 H Q G X O X P                   7 L P H  V W H S V    H        + D O I & K H H W D K0.00.20.40.60.81.00.00.20.40.60.81.0f(s0,a0)w(s,a)Under review as a conference paper at ICLR 2021

These results highlight the importance, and challenge, of propagating updates through the MDP.
MIS methods are not fundamentally different than value-based methods, and viewing them as such
may allow us to develop richer foundations for MIS.

4 RELATED WORK

Off-Policy Evaluation. Off-policy evaluation is a well-studied problem with several families of ap-
proaches. One family of approaches is based on importance sampling, which re-weights trajectories
by the ratio of likelihoods under the target and behavior policy (Precup et al., 2001). Importance
sampling methods are unbiased but suffer from variance which can grow exponentially with the
length of trajectories (Li et al., 2015; Jiang & Li, 2016). Consequently, research has focused on vari-
ance reduction (Thomas & Brunskill, 2016; Munos et al., 2016; Farajtabar et al., 2018) or contextual
bandits (Dud´ık et al., 2011; Wang et al., 2017). Marginalized importance sampling methods (Liu
et al., 2018) aim to avoid this exponential variance by considering the ratio in stationary distribu-
tions, giving an estimator with variance which is polynomial with respect to horizon (Xie et al.,
2019; Liu et al., 2019a). Follow-up work has introduced a variety of approaches and improvements,
allowing them to be behavior-agnostic (Nachum et al., 2019a; Uehara & Jiang, 2019; Mousavi et al.,
2020) and operate in the undiscounted setting (Zhang et al., 2020a;c). In a similar vein, some OPE
methods rely on emphasizing, or re-weighting, updates based on their stationary distribution (Sutton
et al., 2016; Mahmood et al., 2017; Hallak & Mannor, 2017; Gelada & Bellemare, 2019), or learning
the stationary distribution directly (Wang et al., 2007; 2008).

Successor Representation. Introduced originally by Dayan (1993) as an approach for improving
generalization in temporal-difference methods, successor representations (SR) were revived by re-
cent work on deep successor RL (Kulkarni et al., 2016) and successor features (Barreto et al., 2017)
which demonstrated that the SR could be generalized to a function approximation setting. The SR
has found applications for task transfer (Barreto et al., 2018; Grimm et al., 2019), navigation (Zhang
et al., 2017; Zhu et al., 2017), and exploration (Machado et al., 2018a; Janz et al., 2019). It has
also been used in a neuroscience context to model generalization and human reinforcement learn-
ing (Gershman et al., 2012; Momennejad et al., 2017; Gershman, 2018). The SR and our work also
relate to state representation learning (Lesort et al., 2018) and general value functions (Sutton &
Tanner, 2005; Sutton et al., 2011).

5 EXPERIMENTS

To evaluate our method, we perform several off-policy evaluation (OPE) experiments on a variety
of domains. The aim is to evaluate the normalized average discounted reward E(s,a)∼dπ,r[r(s, a)]
of a target policy π. We benchmark our algorithm against two MIS methods, DualDICE (Nachum
et al., 2019a) and GradientDICE (Zhang et al., 2020c), two deep RL approaches and the true return
of the behavior policy. The ﬁrst deep RL method is a DQN-style approach (Mnih et al., 2015) where
actions are selected by π (denoted Deep TD) and the second is the deep SR where the weight w is
trained to minimize the MSE between w(cid:62)φ(s, a) and r(s, a) (denoted Direct-SR) (Kulkarni et al.,
2016). Environment-speciﬁc experimental details are presented below and complete algorithmic and
hyper-parameter details are included in the supplementary material.

Continuous-Action Experiments. We evaluate the methods on a variety of MuJoCo environments
(Brockman et al., 2016; Todorov et al., 2012). We examine two experimental settings.
In both
settings the target policy π and behavior policy πb are stochastic versions of a deterministic policy
πd obtained from training the TD3 algorithm (Fujimoto et al., 2018). We evaluate a target policy
π = πd + N (0, σ2), where σ = 0.1.

• For the “easy” setting, we gather a data set of 500k transitions using a behavior policy πb =
b ), where σb = 0.133. This setting roughly matches the experimental setting deﬁned

πd + N (0, σ2
by Zhang et al. (2020a).

policy which acts randomly with p = 0.2 and uses πd + N (0, σ2

• For the “hard” setting, we gather a signiﬁcantly smaller data set of 50k transitions using a behavior
b ), where σb = 0.2, with p = 0.8.
Unless speciﬁed otherwise, we use a discount factor of γ = 0.99 and all hyper-parameters are kept
constant across environments. All experiments are performed over 10 seeds. We display the results
of the “easy” setting in Figure 2 and the “hard” setting in Figure 3.

7

Under review as a conference paper at ICLR 2021

Figure 2: Off-policy evaluation results on the continuous-action MuJoCo domain using the “easy” experimental
setting (500k time steps and σb = 0.133). The shaded area captures one standard deviation across 10 trials.
We remark that this setting can be considered easy as the behavior policy achieves a lower error, often outper-
forming all agents. SR-DICE signiﬁcantly outperforms the other MIS methods on all environments, except for
Humanoid, where GradientDICE achieves a comparable performance.

Figure 3: Off-policy evaluation results on the continuous-action MuJoCo domain using the “hard” experimental
setting (50k time steps, σb = 0.2, random actions with p = 0.2). The shaded area captures one standard
deviation across 10 trials. This setting uses signiﬁcantly fewer time steps than the “easy” setting and the
behavior policy is a poor estimate of the target policy. Again, we see SR-DICE outperforms the MIS methods,
demonstrating the beneﬁts of our proposed decomposition and simpler optimization. This setting also shows
the beneﬁts of deep RL methods over MIS methods for OPE in high dimensional domains, as deep TD performs
the strongest in every environment.

Atari Experiments. We also test each method on several Atari games (Bellemare et al., 2013),
which are challenging due to their high-dimensional image-based state space. Standard pre-
processing steps are applied (Castro et al., 2018) and sticky-actions are used (Machado et al., 2018b)
to increase difﬁculty and remove determinism. Each method is trained on a data set of one million
time steps. The target policy is the deterministic greedy policy trained by Double DQN (Van Hasselt
et al., 2016). The behavior policy is the (cid:15)-greedy policy with (cid:15) = 0.1. We use a discount factor of
γ = 0.99. Experiments are performed over 3 seeds. Results are displayed in Figure 4. Additional
experiments with different behavior policies can be found in the supplementary material.

Discussion. Across the board we ﬁnd SR-DICE signiﬁcantly outperforms the MIS methods. From
the MSE graphs, we can see SR-DICE achieves much lower error in every task. Looking at the
estimated values of R(π) in the continuous-action environments, Figure 3, we can see that SR-
DICE converges rapidly and maintains a stable estimate, while the MIS methods are particularly
unstable, especially in the case of DualDICE. These observations are consistent in the Atari domain
(Figure 4). Overall, we ﬁnd the general trend in performance is Deep TD > SR-DICE = Direct-
SR > MIS. Notably Direct-SR and SR-DICE perform similarly in every task, suggesting that the
limiting factor in SR-DICE is the quality of the deep successor representation.

Ablation. To study the robustness of SR-DICE relative to the competing methods, we perform an
ablation study and investigate the effects of data set size, discount factor, and two different behavior
policies. Unless speciﬁed otherwise, we use experimental settings matching the “hard” setting. We
report the results in Figure 5. In the data set size experiment (a), SR-DICE perform well with as
few as 5k transitions (5 trajectories). In some instances, the performance is unexpectedly improved
with less data, although incrementally. For small data sets, the SR methods outperform Deep TD.
One hypothesis is that the encoding acts as an auxiliary reward and helps stabilize learning in the
low data regime. In (b) we report the performance over changes in discount factor. The relative
ordering across methods is unchanged. In (c) we use a behavior policy of N (0, σ2
b ), with σb = 0.5,

8

                   7 L P H  V W H S V    H                             / R J  0 6 ( + D O I & K H H W D K                   7 L P H  V W H S V    H                           + R S S H U                   7 L P H  V W H S V    H            : D O N H U  G                   7 L P H  V W H S V    H             $ Q W                   7 L P H  V W H S V    H            + X P D Q R L G                                         6 5  ' , & ( ' X D O ' , & ( * U D G L H Q W ' , & ( ' L U H F W  6 5 ' H H S  7 ' % H K D Y L R U R(πb)                   7 L P H  V W H S V    H          / R J  0 6 ( + D O I & K H H W D K                   7 L P H  V W H S V    H        + R S S H U                   7 L P H  V W H S V    H         : D O N H U  G                   7 L P H  V W H S V    H          $ Q W                   7 L P H  V W H S V    H            + X P D Q R L G                                         6 5  ' , & ( ' X D O ' , & ( * U D G L H Q W ' , & ( ' L U H F W  6 5 ' H H S  7 ' % H K D Y L R U R(πb)Under review as a conference paper at ICLR 2021

Figure 4: We plot the log MSE for off-policy evaluation in the image-based Atari domain. The shaded area
captures one standard deviation across 3 trials. We can see the MIS baselines diverge on this challenging en-
vironment, while the remaining methods perform similarly. Perhaps surprisingly, on most games, the na¨ıve
baseline of using R(πb) from the behavior policy outperforms all methods by a fairly signiﬁcant margin. Al-
though the estimates from deep RL methods are stable, they are biased, resulting in a higher MSE.

(a) Data set size

(b) Discount factor γ

(c) Increased noise

(d) Deterministic policies

Figure 5: Ablation study results for the HalfCheetah task. We default to the “hard” setting wherever possible.
Error bars and the shaded area captures one standard deviation over 10 trials. (a) We vary the size of the data
set D. (b) We vary the discount factor γ. (c) We use a new behavior policy with N (0, σ2
b ) noise with σb = 0.5.
(d) We use the same deterministic behavior and target policy.

a much larger standard deviation than either setting for continuous control. The results are similar
to the original setting, with an increased bias on the deep RL methods. In (d) we use the underlying
deterministic policy as both the behavior and target policy. The baseline MIS methods perform
surprisingly poorly, once again demonstrating their weakness on harder domains.

6 CONCLUSION

In this paper, we introduce a method which can perform marginalized importance sampling (MIS)
using the successor representation (SR) of the target policy. This is achieved by deriving an MIS
formulation that can be viewed as reward function optimization. By using the SR, we effectively
disentangle the dynamics of the environment from learning the reward function. This allows us to
(a) use well-known deep RL methods to effectively learn the SR in challenging domains (Mnih et al.,
2015; Kulkarni et al., 2016) and (b) provide a straightforward loss function to learn the density ratios
without any optimization tricks necessary for previous methods (Liu et al., 2018; Uehara & Jiang,
2019; Nachum et al., 2019a; Zhang et al., 2020c). This reward function interpretation also provides
insight into prior MIS methods by showing how they are connected to value-based methods. Our
resulting algorithm, SR-DICE, outperforms prior MIS methods in terms of both performance and
stability and is the ﬁrst MIS method which demonstrably scales to high-dimensional problems.

As a secondary ﬁnding, our benchmarking shows that current MIS methods underperform more
traditional value-based methods at OPE on high-dimensional tasks, suggesting that for practical
applications, deep RL approaches should still be preferred. Regardless, outside of OPE there exists
a wealth of possible applications for MIS ratios, from imitation (Kostrikov et al., 2019) to policy
optimization (Imani et al., 2018; Liu et al., 2019b; Zhang et al., 2019) to mitigating distributional
shift in ofﬂine RL (Fujimoto et al., 2019b; Kumar et al., 2019). For ease of use, our code is provided,
and we hope our algorithm and insight will provide valuable contributions to the ﬁeld.

9

                   7 L P H  V W H S V    H              / R J  0 6 ( $ V W H U L [                   7 L P H  V W H S V    H            % H D P 5 L G H U                   7 L P H  V W H S V    H          % U H D N R X W                   7 L P H  V W H S V    H            . U X O O                   7 L P H  V W H S V    H                              3 R Q J                                         6 5  ' , & ( ' X D O ' , & ( * U D G L H Q W ' , & ( ' L U H F W  6 5 ' H H S  7 ' % H K D Y L R U R(πb)             7 U D Q V L W L R Q V    H         / R J  0 6 ( + D O I & K H H W D K                  ' L V F R X Q W  ) D F W R U γ        / R J  0 6 ( + D O I & K H H W D K                   7 L P H  V W H S V    H         / R J  0 6 ( + D O I & K H H W D K                   7 L P H  V W H S V    H              / R J  0 6 ( + D O I & K H H W D K                                         6 5  ' , & ( ' X D O ' , & ( * U D G L H Q W ' , & ( ' L U H F W  6 5 ' H H S  7 ' % H K D Y L R U R(πb)Under review as a conference paper at ICLR 2021

REFERENCES

Leemon Baird. Residual algorithms: Reinforcement learning with function approximation.

In

Machine Learning Proceedings 1995, pp. 30–37. Elsevier, 1995.

Andr´e Barreto, Will Dabney, R´emi Munos, Jonathan J Hunt, Tom Schaul, Hado P van Hasselt, and
David Silver. Successor features for transfer in reinforcement learning. In Advances in neural
information processing systems, pp. 4055–4065, 2017.

Andre Barreto, Diana Borsa, John Quan, Tom Schaul, David Silver, Matteo Hessel, Daniel
Mankowitz, Augustin Zidek, and Remi Munos. Transfer in deep reinforcement learning using
successor features and generalised policy improvement. In International Conference on Machine
Learning, pp. 501–510, 2018.

Marc G Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environ-
ment: An evaluation platform for general agents. Journal of Artiﬁcial Intelligence Research, 47:
253–279, 2013.

Richard Bellman. Dynamic Programming. Princeton University Press, 1957.

Dimitri P Bertsekas and John N. Tsitsiklis. Neuro-Dynamic Programming. Athena scientiﬁc Bel-

mont, MA, 1996.

Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and

Wojciech Zaremba. Openai gym, 2016.

Pablo Samuel Castro, Subhodeep Moitra, Carles Gelada, Saurabh Kumar, and Marc G Belle-
arXiv preprint

mare. Dopamine: A research framework for deep reinforcement learning.
arXiv:1812.06110, 2018.

Peter Dayan. Improving generalization for temporal difference learning: The successor representa-

tion. Neural Computation, 5(4):613–624, 1993.

Miroslav Dud´ık, John Langford, and Lihong Li. Doubly robust policy evaluation and learning.
In Proceedings of the 28th International Conference on International Conference on Machine
Learning, pp. 1097–1104, 2011.

Mehrdad Farajtabar, Yinlam Chow, and Mohammad Ghavamzadeh. More robust doubly robust
off-policy evaluation. In International Conference on Machine Learning, pp. 1447–1456, 2018.

Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in
actor-critic methods. In International Conference on Machine Learning, volume 80, pp. 1587–
1596. PMLR, 2018.

Scott Fujimoto, Edoardo Conti, Mohammad Ghavamzadeh, and Joelle Pineau. Benchmarking batch

deep reinforcement learning algorithms. arXiv preprint arXiv:1910.01708, 2019a.

Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without

exploration. In International Conference on Machine Learning, pp. 2052–2062, 2019b.

Jason Gauci, Edoardo Conti, Yitao Liang, Kittipat Virochsiri, Yuchen He, Zachary Kaden, Vivek
Narayanan, Xiaohui Ye, Zhengxing Chen, and Scott Fujimoto. Horizon: Facebook’s open source
applied reinforcement learning platform. arXiv preprint arXiv:1811.00260, 2018.

Carles Gelada and Marc G Bellemare. Off-policy deep reinforcement learning by bootstrapping the
covariate shift. In Proceedings of the AAAI Conference on Artiﬁcial Intelligence, volume 33, pp.
3647–3655, 2019.

Samuel J Gershman. The successor representation: its computational logic and neural substrates.

Journal of Neuroscience, 38(33):7193–7200, 2018.

Samuel J Gershman, Christopher D Moore, Michael T Todd, Kenneth A Norman, and Per B Seder-
berg. The successor representation and temporal context. Neural Computation, 24(6):1553–1568,
2012.

10

Under review as a conference paper at ICLR 2021

Christopher Grimm, Irina Higgins, Andre Barreto, Denis Teplyashin, Markus Wulfmeier, Tim Her-
tweck, Raia Hadsell, and Satinder Singh. Disentangled cumulants help successor representations
transfer to new tasks. arXiv preprint arXiv:1911.10866, 2019.

Assaf Hallak and Shie Mannor. Consistent on-line off-policy evaluation. In Proceedings of the 34th
International Conference on Machine Learning-Volume 70, pp. 1372–1383. JMLR. org, 2017.

Matteo Hessel, Joseph Modayil, Hado Van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan
Horgan, Bilal Piot, Mohammad Azar, and David Silver. Rainbow: Combining improvements in
deep reinforcement learning. arXiv preprint arXiv:1710.02298, 2017.

Ehsan Imani, Eric Graves, and Martha White. An off-policy policy gradient theorem using emphatic

weightings. In Advances in Neural Information Processing Systems, pp. 96–106, 2018.

David Janz, Jiri Hron, Przemysław Mazur, Katja Hofmann, Jos´e Miguel Hern´andez-Lobato, and Se-
bastian Tschiatschek. Successor uncertainties: exploration and uncertainty in temporal difference
learning. In Advances in Neural Information Processing Systems, pp. 4509–4518, 2019.

Nan Jiang and Lihong Li. Doubly robust off-policy value evaluation for reinforcement learning. In

International Conference on Machine Learning, pp. 652–661, 2016.

Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint

arXiv:1412.6980, 2014.

Ilya Kostrikov, Oﬁr Nachum, and Jonathan Tompson. Imitation learning via off-policy distribution

matching. arXiv preprint arXiv:1912.05032, 2019.

Tejas D Kulkarni, Ardavan Saeedi, Simanta Gautam, and Samuel J Gershman. Deep successor

reinforcement learning. arXiv preprint arXiv:1606.02396, 2016.

Aviral Kumar, Justin Fu, Matthew Soh, George Tucker, and Sergey Levine. Stabilizing off-policy
q-learning via bootstrapping error reduction. In Advances in Neural Information Processing Sys-
tems, pp. 11784–11794, 2019.

Timoth´ee Lesort, Natalia D´ıaz-Rodr´ıguez, Jean-Franois Goudou, and David Filliat. State represen-

tation learning for control: An overview. Neural Networks, 108:379–392, 2018.

Lihong Li, Remi Munos, and Csaba Szepesvari. Toward minimax off-policy value estimation. In

Artiﬁcial Intelligence and Statistics, pp. 608–616, 2015.

Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa,
David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv
preprint arXiv:1509.02971, 2015.

Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou. Breaking the curse of horizon: Inﬁnite-
horizon off-policy estimation. In Advances in Neural Information Processing Systems, pp. 5356–
5366, 2018.

Yao Liu, Pierre-Luc Bacon, and Emma Brunskill. Understanding the curse of horizon in off-policy

evaluation via conditional importance sampling. arXiv preprint arXiv:1910.06508, 2019a.

Yao Liu, Adith Swaminathan, Alekh Agarwal, and Emma Brunskill. Off-policy policy gradient with

state distribution correction. arXiv preprint arXiv:1904.08473, 2019b.

Marlos C Machado, Clemens Rosenbaum, Xiaoxiao Guo, Miao Liu, Gerald Tesauro, and Murray
Campbell. Eigenoption discovery through the deep successor representation. arXiv preprint
arXiv:1710.11089, 2017.

Marlos C Machado, Marc G Bellemare, and Michael Bowling. Count-based exploration with the

successor representation. arXiv preprint arXiv:1807.11622, 2018a.

Marlos C Machado, Marc G Bellemare, Erik Talvitie, Joel Veness, Matthew Hausknecht, and
Michael Bowling. Revisiting the arcade learning environment: Evaluation protocols and open
problems for general agents. Journal of Artiﬁcial Intelligence Research, 61:523–562, 2018b.

11

Under review as a conference paper at ICLR 2021

Ashique Rupam Mahmood, Huizhen Yu, and Richard S Sutton. Multi-step off-policy learning with-

out importance sampling ratios. arXiv preprint arXiv:1702.03006, 2017.

Travis Mandel, Yun-En Liu, Sergey Levine, Emma Brunskill, and Zoran Popovic. Ofﬂine policy
evaluation across representations with applications to educational games. In International Con-
ference on Autonomous Agents and Multiagent Systems, 2014.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Belle-
mare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level
control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.

Ida Momennejad, Evan M Russek, Jin H Cheong, Matthew M Botvinick, Nathaniel Douglass Daw,
and Samuel J Gershman. The successor representation in human reinforcement learning. Nature
Human Behaviour, 1(9):680–692, 2017.

Ali Mousavi, Lihong Li, Qiang Liu, and Denny Zhou. Black-box off-policy estimation for inﬁnite-

horizon reinforcement learning. arXiv preprint arXiv:2003.11126, 2020.

R´emi Munos, Tom Stepleton, Anna Harutyunyan, and Marc Bellemare. Safe and efﬁcient off-policy
reinforcement learning. In Advances in Neural Information Processing Systems, pp. 1054–1062,
2016.

Oﬁr Nachum, Yinlam Chow, Bo Dai, and Lihong Li. Dualdice: Behavior-agnostic estimation of
In Advances in Neural Information Processing

discounted stationary distribution corrections.
Systems, pp. 2315–2325, 2019a.

Oﬁr Nachum, Bo Dai, Ilya Kostrikov, Yinlam Chow, Lihong Li, and Dale Schuurmans. Algaedice:

Policy gradient from arbitrary experience. arXiv preprint arXiv:1912.02074, 2019b.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-
performance deep learning library. In Advances in Neural Information Processing Systems, pp.
8024–8035, 2019.

Doina Precup, Richard S Sutton, and Sanjoy Dasgupta. Off-policy temporal-difference learning with
function approximation. In International Conference on Machine Learning, pp. 417–424, 2001.

R Tyrrell Rockafellar. Convex analysis. Princeton university press, 1970.

Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay. In

International Conference on Learning Representations, Puerto Rico, 2016.

Samarth Sinha, Jiaming Song, Animesh Garg, and Stefano Ermon. Experience replay with

likelihood-free importance weights. arXiv preprint arXiv:2006.13169, 2020.

Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction, volume 1. MIT

press Cambridge, 1998.

Richard S Sutton and Brian Tanner. Temporal-difference networks. In Advances in neural informa-

tion processing systems, pp. 1377–1384, 2005.

Richard S Sutton, Hamid Reza Maei, Doina Precup, Shalabh Bhatnagar, David Silver, Csaba
Szepesv´ari, and Eric Wiewiora. Fast gradient-descent methods for temporal-difference learn-
ing with linear function approximation. In International Conference on Machine Learning, pp.
993–1000. ACM, 2009.

Richard S Sutton, Joseph Modayil, Michael Delp, Thomas Degris, Patrick M Pilarski, Adam White,
and Doina Precup. Horde: a scalable real-time architecture for learning knowledge from unsu-
pervised sensorimotor interaction. In The 10th International Conference on Autonomous Agents
and Multiagent Systems-Volume 2, pp. 761–768, 2011.

Richard S Sutton, A Rupam Mahmood, and Martha White. An emphatic approach to the problem
of off-policy temporal-difference learning. The Journal of Machine Learning Research, 17(1):
2603–2631, 2016.

12

Under review as a conference paper at ICLR 2021

Adith Swaminathan, Akshay Krishnamurthy, Alekh Agarwal, Miro Dudik, John Langford, Damien
Jose, and Imed Zitouni. Off-policy evaluation for slate recommendation. In Advances in Neural
Information Processing Systems, pp. 3632–3642, 2017.

Philip Thomas and Emma Brunskill. Data-efﬁcient off-policy policy evaluation for reinforcement

learning. In International Conference on Machine Learning, pp. 2139–2148, 2016.

Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control.
In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 5026–5033.
IEEE, 2012.

Ahmed Touati, Amy Zhang, Joelle Pineau, and Pascal Vincent. Stable policy optimization via off-

policy divergence regularization. arXiv preprint arXiv:2003.04108, 2020.

Masatoshi Uehara and Nan Jiang. Minimax weight and q-function learning for off-policy evaluation.

arXiv preprint arXiv:1910.12809, 2019.

Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double q-

learning. In AAAI, pp. 2094–2100, 2016.

Tao Wang, Michael Bowling, and Dale Schuurmans. Dual representations for dynamic programming
and reinforcement learning. In 2007 IEEE International Symposium on Approximate Dynamic
Programming and Reinforcement Learning, pp. 44–51. IEEE, 2007.

Tao Wang, Michael Bowling, Dale Schuurmans, and Daniel J Lizotte. Stable dual dynamic pro-

gramming. In Advances in neural information processing systems, pp. 1569–1576, 2008.

Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dud´ık. Optimal and adaptive off-policy evaluation
in contextual bandits. In International Conference on Machine Learning, pp. 3589–3597, 2017.

Eric Wiewiora, Garrison W Cottrell, and Charles Elkan. Principled methods for advising reinforce-
ment learning agents. In Proceedings of the 20th International Conference on Machine Learning
(ICML-03), pp. 792–799, 2003.

Tengyang Xie, Yifei Ma, and Yu-Xiang Wang. Towards optimal off-policy evaluation for rein-
forcement learning with marginalized importance sampling. In Advances in Neural Information
Processing Systems, pp. 9665–9675, 2019.

Jingwei Zhang, Jost Tobias Springenberg, Joschka Boedecker, and Wolfram Burgard. Deep rein-
forcement learning with successor features for navigation across similar environments. In 2017
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 2371–2378.
IEEE, 2017.

Ruiyi Zhang, Bo Dai, Lihong Li, and Dale Schuurmans. Gendice: Generalized ofﬂine estimation of

stationary values. arXiv preprint arXiv:2002.09072, 2020a.

Shangtong Zhang, Wendelin Boehmer, and Shimon Whiteson. Generalized off-policy actor-critic.

In Advances in Neural Information Processing Systems, pp. 1999–2009, 2019.

Shangtong Zhang, Wendelin Boehmer, and Shimon Whiteson. Deep residual reinforcement learn-
ing. In Proceedings of the 19th International Conference on Autonomous Agents and MultiAgent
Systems, pp. 1611–1619, 2020b.

Shangtong Zhang, Bo Liu, and Shimon Whiteson. Gradientdice: Rethinking generalized ofﬂine

estimation of stationary values. arXiv preprint arXiv:2001.11113, 2020c.

Yufan Zhao, Michael R Kosorok, and Donglin Zeng. Reinforcement learning design for cancer

clinical trials. Statistics in medicine, 28(26):3294, 2009.

Yuke Zhu, Daniel Gordon, Eric Kolve, Dieter Fox, Li Fei-Fei, Abhinav Gupta, Roozbeh Mottaghi,
and Ali Farhadi. Visual semantic planning using deep successor representations. In Proceedings
of the IEEE International Conference on Computer Vision, pp. 483–492, 2017.

13

Under review as a conference paper at ICLR 2021

A DETAILED PROOFS.

A.1 OBSERVATION 1

Observation 1 The objective J(ˆr) is minimized when ˆr(s, a) = dπ(s,a)

dD(s,a) , ∀(s, a).

min
ˆr(s,a)∀(s,a)

J(ˆr) :=

:=

1
2
1
2

E(s,a)∼dD

(cid:2)ˆr(s, a)2(cid:3) − (1 − γ)Es0,a0

(cid:104)

(cid:105)
ˆQπ(s0, a0)

E(s,a)∼dD

(cid:2)ˆr(s, a)2(cid:3) − E(s,a)∼dπ [ˆr(s, a)] .

(14)

(15)

Proof.

Take the partial derivative of J(ˆr) with respect to ˆr(s, a):

∂
∂ ˆr(s, a)

(cid:18) 1
2

E(s,a)∼dD

(cid:19)
(cid:2)ˆr(s, a)2(cid:3) − E(s,a)∼dπ [ˆr(s, a)]

= dD(s, a)ˆr(s, a) − dπ(s, a).

(16)

Then setting ∂J(ˆr)
pairs (s, a).

∂ ˆr(s,a) = 0, we have that J(ˆr) is minimized when ˆr(s, a) = dπ(s,a)

dD(s,a) for all state-action

A.2 THEOREM 1

(cid:4)

Theorem 1 Assuming (1 − γ)Es0,a0 [ψπ(s0, a0)] = E(s,a)∼dπ [φ(s, a)], then the optimizer w∗ of
the objective J(w) is the least squares estimator of (cid:82)

w(cid:62)φ(s, a) − dπ(s,a)
dD(s,a)

d(s, a).

S×A

(cid:17)2

(cid:16)

min
w

J(w) :=

1
2

EdD

(cid:2)(w(cid:62)φ(s, a))2(cid:3) − (1 − γ)Es0,a0∼π

(cid:2)w(cid:62)ψπ(s0, a0)(cid:3) .

(17)

Proof.

From our assumption we have:

min
w

J(w) :=

1
2

EdD

(cid:2)(w(cid:62)φ(s, a))2(cid:3) − E(s,a)∼dπ

(cid:2)w(cid:62)φ(s, a)(cid:3) .

(18)

Let M = |S| × |A| and N be the feature dimension. Let φ(s, a) be a N × 1 feature vector and Φ
the M × N matrix where each row corresponds to a φ(s, a)(cid:62) vector. Let w be a N × 1 vector of
parameters. Let dπ and dD be M × 1 vectors of the values of dπ(s, a) and dD(s, a) for all (s, a).
(cid:13)
(cid:16)
2
(cid:13)
(cid:13)

w(cid:62)φ(s, a) − dπ(s,a)
dD(s,a)

(cid:13)
(cid:13)Φw − dπ
(cid:13)

d(s, a) =

S×A

(cid:17)2

dD

is

First note the least squares estimator of (cid:82)
ˆw = (ΦT Φ)−1ΦT dπ
Now consider our optimization problem:

dD , where the division is element-wise.

J(w) = 0.5d(cid:62)
= 0.5d(cid:62)
= 0.5d(cid:62)

D(Φw)2 − d(cid:62)
π Φw
D(Φw)T Φw − d(cid:62)
DwT ΦT Φw − d(cid:62)

π Φw
π Φw.

14

(19)

Under review as a conference paper at ICLR 2021

Now take gradient of J(w) with respect to w and set it equal to 0:

DwT ΦT Φw − d(cid:62)

π Φw = 0

∇w0.5d(cid:62)
DwT ΦT Φ = d(cid:62)
d(cid:62)
(cid:18) dπ
dD
ΦT Φw = Φ(cid:62) dπ
dD

wT ΦT Φ =

π Φdπ
(cid:19)(cid:62)

Φ

w = (ΦT Φ)−1Φ(cid:62) dπ
dD

.

It follows that the optimizer to J(w) is the least squares estimator.

(20)

(cid:4)

A.3 THEOREM 2

Theorem 2 Given an MDP (S, A, ·, p, d0, γ), policy π, and function f : S × A → R, deﬁne the
reward function r : S × A → R as ˆr(s, a) = f (s, a) − γEs(cid:48),a(cid:48)∼π[f (s(cid:48), a(cid:48))]. Then it follows that the
value function ˆQπ deﬁned by the policy π, MDP, and reward function ˆr, is the function f .

Proof.
Deﬁne ˆr(s, a) = f (s, a) − γEπ[f (s(cid:48), a(cid:48))]. Then note for all state-action pairs (s, a) we have
f (s, a) = ˆr(s, a) + γEπ[f (s(cid:48), a(cid:48))] = f (s, a) − γEπ[f (s(cid:48), a(cid:48))] + γEπ[f (s(cid:48), a(cid:48))] = f (s, a), satisfying
the Bellman equation. It follows that f = ˆQπ by the uniqueness of the value function (Bertsekas &
Tsitsiklis, 1996).

This result can also be obtained by considering state-action reward shaping (Wiewiora et al., 2003),
and treating f as the potential function.

(cid:4)

B TABULAR SR-DICE

The experimental performance of SR-DICE, in particular its reliance on the SR can be partially
explained by examining its tabular counterpart. In fact, we can show that the value estimate derived
from SR-DICE is exactly equal to a value estimate derived directly from the SR.
Recall the form of SR-DICE’s objective in a tabular setting, as a function of the SR Ψπ:

min
ˆr(s,a)∀(s,a)

JΨ(ˆr) :=

1
2

E(s,a)∼dD

(cid:2)ˆr(s, a)2(cid:3) − (1 − γ)Es0

(cid:34)

(cid:88)

(cid:35)
Ψπ(s|s0)Ea∼π [ˆr(s, a)]

.

(21)

s

Which has the following gradient:

∇ˆr(s,a)JΨ(ˆr) := dD(s, a)ˆr(s, a) − (1 − γ)

p(s0)Ψπ(s|s0)π(a|s).

(22)

(cid:88)

s0

We can compute this gradient from samples. Deﬁne D0 as the set of start states s0 ∈ D. It follows:

∇ˆr(s,a)JΨ(ˆr) :=

1
|D|





(cid:88)



1(s(cid:48) = s, a(cid:48) = a)

 ˆr(s, a)

(s(cid:48),a(cid:48))∈D
1
|D0|

− (1 − γ)

(cid:88)

s0∈D0

Ψπ(s|s0)π(a|s).

(23)

15

Under review as a conference paper at ICLR 2021

Setting the above gradient to 0 and solving for ˆr(s, a) we have the optimizer of JΨ(ˆr).

ˆr(s, a) = (1 − γ)

1
|D0|

(cid:88)

s0∈D0

Ψπ(s|s0)π(a|s)

(cid:80)

(s(cid:48),a(cid:48))∈D

|D|
1(s(cid:48) = s, a(cid:48) = a)

.

(24)

Now consider the MIS equation for estimating the objective R(π) = (1 − γ)Es0 [V π(s0)], where ˆr
is an estimate of dπ(s,a)
dD(s,a) :

1
|D|

(cid:88)

(s,a)∈D

dπ(s, a)
dD(s, a)

r(s, a).

(25)

For convenience, assume every state-action pair (s, a) is contained at least once in D. Although
the result holds regardless, this assumption allows us to avoid some cumbersome details. Replace
dπ(s,a)
dD(s,a) with ˆr in Equation (25) and expand and simplify:

1
|D|

(cid:88)

(1 − γ)

(s,a)∈D

1
|D0|

(cid:88)

= (1 − γ)

= (1 − γ)

1
|D0|

1
|D0|

s0∈D0
(cid:88)

(s,a)∈D

(cid:88)

s0∈D0

(s,a)∈S×A

(cid:88)

Ψπ(s|s0)π(a|s)

s0∈D0
(cid:88)

Ψπ(s|s0)π(a|s)

(cid:80)

(s(cid:48),a(cid:48))∈D

|D|
1(s(cid:48) = s, a(cid:48) = a)

r(s, a)

(26)

(cid:80)

(s(cid:48),a(cid:48))∈D

1

1(s(cid:48) = s, a(cid:48) = a)

r(s, a)

(27)

Ψπ(s|s0)π(a|s)r(s, a).

(28)

Noting that (cid:80)
same solution as the SR solution for estimating R(π).

(s,a)∈S×A Ψπ(s|s0)π(a|s)r(s, a) = V π(s0) we can see that SR-DICE returns the

C ADDITIONAL EXPERIMENTS

In this section, we include additional experiments and visualizations, covering extra domains, addi-
tional ablation studies, run time experiments and additional behavior policies in the Atari domain.

C.1 EXTRA CONTINUOUS DOMAINS

Although our focus is on high-dimensional domains, the environments, Pendulum and Reacher, have
appeared in several related MIS papers (Nachum et al., 2019a; Zhang et al., 2020a). Therefore, we
have included results for these domains in Figure 6. All experimental settings match the experiments
in the main body, and are described fully in Appendix F.

(a) “Easy” setting

(b) “Hard” setting

Figure 6: Off-policy evaluation results for Pendulum and Reacher. The shaded area captures one standard
deviation across 10 trials. Even on these easier environment, we ﬁnd that SR-DICE outperforms the baseline
MIS methods.

16

                   7 L P H  V W H S V    H          / R J  0 6 ( 3 H Q G X O X P                   7 L P H  V W H S V    H             5 H D F K H U                   7 L P H  V W H S V    H        / R J  0 6 ( 3 H Q G X O X P                   7 L P H  V W H S V    H          5 H D F K H U                                         6 5  ' , & ( ' X D O ' , & ( * U D G L H Q W ' , & ( ' L U H F W  6 5 ' H H S  7 ' % H K D Y L R U R(πb)Under review as a conference paper at ICLR 2021

C.2 REPRESENTATION LEARNING & MIS

SR-DICE relies a disentangled representation learning phase where an encoding φ is learned, fol-
lowed by the deep successor representation ψπ which are used with a linear vector w to estimate
the density ratios. In this section we perform some experiments which attempt to evaluate the im-
portance of representation learning by comparing their inﬂuence on the baseline MIS methods.

Alternate representations. We examine both DualDICE (Nachum et al., 2019a) and
GradientDICE (Zhang et al., 2020c) under four settings where we pass the representations φ and
ψπ to their networks, where both φ and ψπ are learned in identical fashion to SR-DICE.

(1) Input encoding φ,
(2) Input SR ψπ,
(3) Input encoding φ, linear networks,
(4) Input SR ψπ, linear networks,

w(φ(s, a)).

f (φ(s, a)),
f (ψπ(s, a)), w(ψπ(s, a)).
f (cid:62)φ(s, a),
f (cid:62)ψπ(s, a), w(cid:62)ψπ(s, a).

w(cid:62)φ(s, a).

See Appendix E for speciﬁc details on the baselines. We report the results in Figure 7. For
GradientDICE, no beneﬁt is provided by varying the representations, although using the encoding φ
matches the performance of vanilla GradientDICE regardless of the choice of network, providing
some validation that φ is a reasonable encoding. Interestingly, for DualDICE, we see performance
gains from using the SR ψπ as a representation: slightly as input, but signiﬁcantly when used with
linear networks. On the other hand, as GradientDICE performs much worse with the SR, it is clear
that the SR cannot be used as a representation without some degree of forethought.

(a) DualDICE

(b) Linear DualDICE

(c) GradientDICE

(d) Linear GradientDICE

Figure 7: Off-policy evaluation results on HalfCheetah examining the value of differing representations added
to the baseline MIS methods. The experimental setting corresponds to the “hard” setting from the main body.
The shaded area captures one standard deviation across 10 trials. We see that using the SR ψπ as a represen-
tation improves the performance of DualDICE. On the other hand, GradientDICE performs much worse when
using the SR, suggesting it cannot be used naively to improve MIS methods.

Increased capacity. As SR-DICE uses a linear function on top of a representation trained with the
same capacity as the networks in DualDICE and GradientDICE, our next experiment examines if
this additional capacity provides beneﬁt to the baseline methods. To do, we expand each network in
both baselines by adding an additional hidden layer. The results are reported in Figure 8. We ﬁnd
there is a very slight decrease in performance when using the larger capacity networks. This suggests
the performance gap from SR-DICE over the baseline methods has little to do with model size.

Figure 8: Off-policy evaluation results on HalfCheetah evaluating the performance beneﬁts from larger network
capacity on the baseline MIS methods. “Big” refers to the models with an additional hidden layer. The exper-
imental setting corresponds to the “hard” setting from the main body. The shaded area captures one standard
deviation across 10 trials. We ﬁnd that there is no clear performance beneﬁt from increasing network capacity.

17

                   7 L P H  V W H S V    H        / R J  0 6 ( + D O I & K H H W D K                   7 L P H  V W H S V    H        / R J  0 6 ( + D O I & K H H W D K                   7 L P H  V W H S V    H                              / R J  0 6 ( + D O I & K H H W D K                   7 L P H  V W H S V    H              / R J  0 6 ( + D O I & K H H W D K                                         6 5  ' , & ( ' X D O ' , & ( * U D G L H Q W ' , & ( ( Q F R G L Q J  ; 6 5  ; % H K D Y L R U R(πb)                   7 L P H  V W H S V    H         / R J  0 6 ( + D O I & K H H W D K                                         ' X D O ' , & ( * U D G L H Q W ' , & ( % L J  ' X D O ' , & ( % L J  * U D G L H Q W ' , & ( % H K D Y L R U R(πb)Under review as a conference paper at ICLR 2021

C.3 TOY DOMAINS

We additional test the MIS algorithms on a toy random-walk experiment with varying feature repre-
sentations, based on a domain from (Sutton et al., 2009).

Domain. The domain is a simple 5-state MDP (x1, x2, x3, x4, x5) with two actions (a0, a1), where
action a0 induces the transition xi → xi−1 and action a1 induces the transition xi → xi+1, with the
state x1 looping to itself with action a0 and x5 looping to itself with action a5. Episodes begin in
the state x1.

Target. We evaluate policy π which selects actions uniformly, i.e. π(a0|xi) = π(a1|xi) = 0.5 for
all states xi. Our data set D contains all 10 possible state-action pairs and is sampled uniformly.
We use a discount factor of γ = 0.99. Methods are evaluated on the average MSE between their
estimate of dπ
dD on all state-action pairs and the ground-truth value, which is calculated analytically.
Hyper-parameters. Since we are mainly interested in a function approximation setting, each
method uses a small neural network with two hidden layers of 32, followed by tanh activation
functions. All networks used stochastic gradient descent with a learning rate α tuned for each
method out of {1, 0.5, 0.1, 0.05, 0.01, 0.001}. This resulted in α = 0.05 for DualDICE, α = 0.1
for GradientDICE, and α = 0.05 for SR-DICE. Although there are a small number of possible data
points, we use a batch size of 128 to resemble the regular training procedure. As recommended
by the authors we use λ = 1 for GradientDICE (Zhang et al., 2020c), which was not tuned. For
SR-DICE, we update the target network at every time step τ = 1, which was not tuned.

Since there are only 10 possible state-action pairs, we use the closed form solution for the vector
w (Equation (10)). Additionally, we skip the state representation phase of SR-DICE, instead learn-
ing the SR ψπ over the given representation of each state, such that the encoding φ = x. This allows
us to test SR-DICE to a variety of representations rather than using a learned encoding. Conse-
quently, with these choices, SR-DICE has no pre-training phase, and therefore, unlike every other
graph in this paper, we report the results as the SR is trained, rather than as the vector w is trained.

Features. To test the robustness of each method we examine three versions of the toy domain, each
using a different feature representation over the same 5-state MDP. These feature sets are again taken
from (Sutton et al., 2009).

• Tabular features: states are represented by a one-hot encoding, for example x2 = [0, 1, 0, 0, 0].
• Inverted features: states are represented by the inverse of a one-hot encoding, for example

x2 = (cid:2) 1

2 , 0, 1

2 , 1

2 , 1

2

(cid:3).

• Dependent features: states are represented by 3 features which is not sufﬁcient to cover all states
2 , 1√
exactly. In this case x1 = [1, 0, 0], x2 = [ 1√
2 ],
x5 = [0, 0, 1]. Since our experiments use neural networks rather than linear functions, this
representation is mainly meant to test SR-DICE, where we skip the state representation phase
for SR-DICE and use the encoding φ = x, limiting the representation of the SR.

3 ], x4 = [0, 1√

2 , 0], x3 = [ 1√

2 , 1√

3 , 1√

3 , 1√

(a) Tabular Features

(b) Inverted Features

(c) Dependent Features

Figure 9: Results measuring the log MSE between the estimated density ratio and the ground-truth on a simple
5-state MDP domain with three feature sets. The shaded area captures one standard deviation across 10 trials.
Results are evaluated every 100 time steps over 50k time steps total.

Results. We report the results in Figure 9. We remark on several observations. SR-DICE learns
signiﬁcantly faster than the baseline methods, likely due to its use of temporal difference methods
in the SR update, rather than using an update similar to residual learning, which is notoriously

18

       7 L P H  V W H S V    H         / R J  0 6 ( 7 D E X O D U  ) H D W X U H V       7 L P H  V W H S V    H         , Q Y H U W H G  ) H D W X U H V       7 L P H  V W H S V    H         ' H S H Q G H Q W  ) H D W X U H V                                         6 5  ' , & ( ' X D O ' , & ( * U D G L H Q W ' , & (Under review as a conference paper at ICLR 2021

slow (Baird, 1995; Zhang et al., 2020b). GradientDICE appears to still be improving, although we
limit training at 50k time steps, which we feel is sufﬁcient given the domain is deterministic and only
has 5 states. Notably, GradientDICE also uses a higher learning rate than SR-DICE and DualDICE.
We also ﬁnd the ﬁnal performance of SR-DICE is much better than DualDICE and GradientDICE in
the domains where the feature representation is not particularly destructive, highlighting the easier
optimization of SR-DICE. In the case of the dependent features, we ﬁnd DualDICE outperforms
SR-DICE after sufﬁcient updates. However, we remark that this concern could likely be resolved
by learning the features and that SR-DICE still outperforms GradientDICE. Overall, we believe
these results demonstrate that SR-DICE’s strong empirical performance is consistent across simpler
domains as well as the high dimensional domains we examine in the main body.

C.4 RUN TIME EXPERIMENTS

In this section, we evaluate the run time of each algorithm used in our experiments. Although SR-
DICE relies on pre-training the deep successor representation before learning the density ratios, we
ﬁnd each marginalized importance sampling (MIS) method uses a similar amount of compute, due
to the reduced cost of training w after the pre-training phase.

We evaluate the run time on the HalfCheetah environment in MuJoCo (Todorov et al., 2012) and
OpenAI gym (Brockman et al., 2016). As in the main set of experiments, each method is trained for
250k time steps. Additionally, SR-DICE and Direct-SR train the encoder-decoder for 30k time steps
and the deep successor representation for 100k time steps before training w. Run time is averaged
over 3 seeds. All time-based experiments are run on a single GeForce GTX 1080 GPU and a Intel
Core i7-6700K CPU. Results are reported in Figure 10.

Figure 10: The average run time of each off-policy evaluation approach in minutes. Each experiment is run for
250k time steps and is averaged over 3 seeds. SR-DICE and Direct-SR pre-train encoder-decoder for 30k time
steps and the deep successor representation 100k time steps.

We ﬁnd the MIS algorithms run in a comparable time, regardless of the pre-training step involved
in SR-DICE. This can be explained as training w in SR-DICE involves signiﬁcantly less compute
than DualDICE and GradientDICE which update multiple networks. On the other hand, the deep
reinforcement learning approaches run in about half the time of SR-DICE.

C.5 ATARI EXPERIMENTS

To better evaluate the algorithms in the Atari domain, we run two additional experiments where we
swap the behavior policy. We observe similar trends as the experiments in the main body of the
paper. In both experiments we keep all other settings ﬁxed. Notably, we continue to use the same
target policy, corresponding to the greedy policy trained by Double DQN (Van Hasselt et al., 2016),
the same discount factor γ = 0.99, and the same data set size of 1 million.

Increased noise. In our ﬁrst experiment, we attempt to increase the randomness of the behavior
policy. As this can cause destructive behavior in the performance of the agent, we adopt an episode-
dependent policy which selects between the noisy policy or the deterministic greedy policy at the
beginning of each episode. This is motivated by the ofﬂine deep reinforcement learning experiments
from (Fujimoto et al., 2019a). As a result, we use an (cid:15)-greedy policy with p = 0.8 and the deter-

19

 6 5  ' , & ( ' X D O ' , & ( * U D G L H Q W ' , & ( ' L U H F W  6 5 ' H H S  7 '          5 X Q  7 L P H   P L Q X W H V           Under review as a conference paper at ICLR 2021

ministic greedy policy (the target policy) with p = 0.2. (cid:15) is set to 0.2, rather than 0.1 as in the
experiments in the main body of the paper. Results are reported in Figure 11.

Figure 11: We plot the log MSE for off-policy evaluation in the image-based Atari domain, using an episode-
dependent noisy policy, where (cid:15) = 0.2 with p = 0.8 and (cid:15) = 0 with p = 0.2. This episode-dependent selection
ensures sufﬁcient state-coverage while using a stochastic policy. The shaded area captures one standard devia-
tion across 3 trials. Markers are not placed at every point for visual clarity.

We observe very similar trends to the original set of experiments. Again, we note DualDICE and
GradientDICE perform very poorly, while SR-DICE, Direct-SR, and Deep TD achieve a reasonable,
but biased, performance. In this setting, we still ﬁnd the behavior policy is the closest estimate of
the true value of R(π) .

Separate behavior policy. In this experiment, we use a behavior which is distinct from the target
policy, rather than simply adding noise. This behavior policy is derived from an agent trained with
prioritized experience replay and Double DQN (Schaul et al., 2016). Again, we use a (cid:15)-greedy
policy, with (cid:15) = 0.1. We report the results in Figure 12.

Figure 12: We plot the log MSE for off-policy evaluation in the image-based Atari domain, using a distinct
behavior policy, trained by a separate algorithm, from the target policy. This experiment tests the ability to
generalize to a more off-policy setting. The shaded area captures one standard deviation across 3 trials. Markers
are not placed at every point for visual clarity.

Again, we observe similar trends in performance. Notably, in the Asterix game, the performance of
Direct-SR surpasses the behavior policy, suggesting off-policy evaluation can outperform the na¨ıve
estimator in settings where the policy is sufﬁciently “off-policy” and distinct.

D SR-DICE PRACTICAL DETAILS

In this section, we cover some basic implementation-level details of SR-DICE. Note that code is
provided for additional clarity.

SR-DICE uses two parametric networks, an encoder-decoder network to learn the encoding φ and
a deep successor representation network ψπ. Additionally, SR-DICE uses the weights of a linear
function w. SR-DICE begins by pre-training the encoder-decoder network and the deep successor
representation before applying updates to w.

Encoder-Decoder. This encoder-decoder network encodes (s, a) to the feature vector φ(s, a), which
is then decoded by several decoder heads. For the Atari domain, we choose to condition the feature
vector only on states φ(s), as the reward is generally independent of the action selection. This

20

                   7 L P H  V W H S V    H            / R J  0 6 ( $ V W H U L [                   7 L P H  V W H S V    H                % H D P 5 L G H U                   7 L P H  V W H S V    H                         % U H D N R X W                   7 L P H  V W H S V    H            . U X O O                   7 L P H  V W H S V    H          3 R Q J                                         6 5  ' , & ( ' X D O ' , & ( * U D G L H Q W ' , & ( ' L U H F W  6 5 ' H H S  7 ' % H K D Y L R U R(πb)                   7 L P H  V W H S V    H            / R J  0 6 ( $ V W H U L [                   7 L P H  V W H S V    H              % H D P 5 L G H U                   7 L P H  V W H S V    H          % U H D N R X W                   7 L P H  V W H S V    H                . U X O O                   7 L P H  V W H S V    H            3 R Q J                                         6 5  ' , & ( ' X D O ' , & ( * U D G L H Q W ' , & ( ' L U H F W  6 5 ' H H S  7 ' % H K D Y L R U R(πb)Under review as a conference paper at ICLR 2021

Algorithm 2 SR-DICE
1: Input: Data set D, target policy π, number of iterations T1, T2, T3, mini-batch size N , tar-

get update rate.

2: for t = 1 to T1 do
3:
4: minφ,Ds(cid:48) ,Da,Dr λs(cid:48)(Ds(cid:48)(φ(s, a)) − s(cid:48))2

Sample mini-batch of N transitions (s, a, r, s(cid:48)) from D.

+λa(Da(φ(s, a)) − a)2 + λr(Dr(φ(s, a)) − r)2.

5: end for

Sample mini-batch of N transitions (s, a, r, s(cid:48)) from D.
Sample a(cid:48) ∼ π(s(cid:48)).

6: for t = 1 to T2 do
7:
8:
9: minψπ (φ(s, a) + γψ(cid:48)(s(cid:48), a(cid:48)) − ψπ(s, a))2.
10:
11: end for

If t mod target update rate = 0: ψ(cid:48) ← ψ.

12: for t = 1 to T3 do
13:
14:
15:
16: minw
17: end for

1

Sample mini-batch of N transitions (s, a, r, s(cid:48)) from D.
Sample mini-batch of N start states s0 from D.
Sample a0 ∼ π(s0).

2 (w(cid:62)φ(s, a))2 − (1 − γ)w(cid:62)ψπ(s0, a0).

Train encoder-
decoder

Train deep
successor
representation

Learn w

change applies to both SR-DICE and Direct-SR. Most design decisions are inspired by prior work
(Machado et al., 2017; 2018a).
For continuous control, given a mini-batch transition (s, a, r, s(cid:48)), the encoder-decoder network is
trained to map the state-action pair (s, a) to the next state s(cid:48), the action a and reward r. The resulting
loss function is as follows:

min
φ,Ds(cid:48) ,Da,Dr

L(φ, D) := λs(cid:48)(Ds(cid:48)(φ(s, a)) − s(cid:48))2 + λa(Da(φ(s, a)) − a)2 + λr(Dr(φ(s, a)) − r)2.

(29)

We use λs(cid:48) = 1, λa = 1 and λr = 0.1.
For the Atari games, given a mini-batch transition (s, a, r, s(cid:48)), the encoder-decoder network is
trained to map the state s to the next state s(cid:48) and reward r, while penalizing the size of φ(s). The
resulting loss function is as follows:

min
φ,Ds(cid:48) ,Dr

L(φ, D) := λs(cid:48)(Ds(cid:48)(φ(s)) − s(cid:48))2 + λr(Dr(φ(s)) − r)2 + λφφ(s)2.

(30)

We use λs(cid:48) = 1, λr = 0.1 and λφ = 0.1.
Deep Successor Representation. The deep successor representation ψπ is trained to estimate the
accumulation of φ. The training procedure resembles standard deep reinforcement learning algo-
rithms. Given a mini-batch of transitions (s, a, r, s(cid:48)) the network is trained to minimize the following
loss:

L(ψπ) := (φ(s, a) + γψ(cid:48)(s(cid:48), a(cid:48)) − ψπ(s, a))2,

(31)

min
ψπ

where ψ(cid:48) is the target network. A target network is a frozen network used to provide stability (Mnih
et al., 2015; Kulkarni et al., 2016) in the learning target. The target network is updated to the current
network ψ(cid:48) ← ψπ after a ﬁxed number of time steps, or updated with slowly at each time step
ψ(cid:48) ← τ ψπ + (1 − τ )ψπ (Lillicrap et al., 2015).

Marginalized Importance Sampling Weights. As described in the main body, we learn w by
optimizing the following objective:

min
w

J(w) :=

1
2

E(s,a)∼dD

(cid:2)(w(cid:62)φ(s, a))2(cid:3) − (1 − γ)Es0,a0∼π

(cid:2)w(cid:62)ψπ(s0, a0)(cid:3) .

(32)

21

Under review as a conference paper at ICLR 2021

This is achieved by sampling state-action pairs uniformly from the data set D, alongside a mini-batch
of start states s0, which are recorded at the beginning of each episode during data collection.

We summarize the learning procedure of SR-DICE in Algorithm 2.

E BASELINES

In this section, we cover some of the practical details of each of the baseline methods.

E.1 DUALDICE

Dual stationary DIstribution Correction Estimation (DualDICE) (Nachum et al., 2019a) uses two
networks f and w. The general optimization problem is deﬁned as follows:

min
f

max
w

J(f, w) := E(s,a)∼dD,a(cid:48)∼π,s(cid:48)

(cid:2)w(s, a)(f (s, a) − γf (s(cid:48), a(cid:48))) − 0.5w(s, a)2(cid:3)

− (1 − γ)Es0,a0 [f (s0, a0)].

(33)

In practice this corresponds to alternating single gradient updates to f and w. The authors suggest
possible alternative functions to the convex function 0.5w(s, a)2 such as 2
2 , however in
practice we found 0.5w(s, a)2 performed the best.

3 |w(s, a)| 3

E.2 GRADIENTDICE

Gradient stationary DIstribution Correction Estimation (GradientDICE) (Zhang et al., 2020c) uses
two networks f and w, and a scalar u. The general optimization problem is deﬁned as follows:

min
w

max
f,u

J(w, u, f ) := (1 − γ)Es0,a0[f (s0, a0)] + γE(s,a)∼dD,a(cid:48)∼π,s(cid:48)[w(s, a)f (s(cid:48), a(cid:48))]

− E(s,a)∼dD [w(s, a)f (s, a)] + λ (cid:0)E(s,a)∼dD [uw(s, a) − u] − 0.5u2(cid:1) .
(34)
Similarly to DualDICE, in practice this involves alternating single gradient updates to w, u and f .
As suggested by the authors we use λ = 1.

E.3 DIRECT-SR

Direct-SR is a policy evaluation version of deep successor representation (Kulkarni et al., 2016). The
encoder-decoder network and deep successor representation are trained in the exact same manner
as SR-DICE (see Section D). Then, rather than train w to learn the marginalized importance sam-
pling ratios, w is trained to recover the original reward function. Given a mini-batch of transitions
(s, a, r, s(cid:48)), the following loss is applied:

L(w) := (r − w(cid:62)φ(s, a))2.

min
w

(35)

E.4 DEEP TD

Deep TD, short for deep temporal-difference learning, takes the standard deep reinforcement learn-
ing methodology, akin to DQN (Mnih et al., 2015), and applies it to off-policy evaluation. Given a
mini-batch of transitions (s, a, r, s(cid:48)) the Q-network is updated by the following loss:

L(Qπ) := (r + γQ(cid:48)(s(cid:48), a(cid:48)) − Qπ(s, a))2,

min
Qπ

(36)

where a(cid:48) is sampled from the target policy π(·|s(cid:48)). Similarly, to training the deep successor repre-
sentation, Q(cid:48) is a frozen target network which is updated to the current network after a ﬁxed number
of time steps, or incrementally at every time step.

F EXPERIMENTAL DETAILS

All networks are trained with PyTorch (version 1.4.0) (Paszke et al., 2019). Any unspeciﬁed hyper-
parameter uses the PyTorch default setting.

22

Under review as a conference paper at ICLR 2021

Evaluation. The marginalized importance sampling methods are measured by the average weighted
reward from transitions sampled from a replay buffer 1
(s,a,r) w(s, a)r(s, a), with N = 10k,
N
while the deep RL methods use (1−γ)
s0 Q(s0, π(a0)), where M is the number of episodes. Each
OPE method is trained on data collected by some behavioral policy πb. We estimate the “true”
normalized average discounted reward of the target and behavior policies from 100 roll-outs in the
environment.

(cid:80)

(cid:80)

M

F.1 CONTINUOUS-ACTION ENVIRONMENTS

Our agents are evaluated via tasks interfaced through OpenAI gym (version 0.17.2) (Brockman et al.,
2016), which mainly rely on the MuJoCo simulator (mujoco-py version 1.50.1.68) (Todorov et al.,
2012). We provide a description of each environment in Table 1.

Table 1: Continuous-action environment descriptions.

Environment

State dim. Action dim. Episode Horizon

Task description

Pendulum-v0
Reacher-v2
HalfCheetah-v3
Hopper-v3
Walker2d-v3
Ant-v3
Humanoid-v3

3
11
17
11
17
111
376

1
2
6
3
6
8
17

200
50
1000
1000
1000
1000
1000

Balance a pendulum.
Move end effector to goal.
Locomotion.
Locomotion.
Locomotion.
Locomotion.
Locomotion.

Experiments. Our experiments are framed as off-policy evaluation tasks in which agents aim to
evaluate R(π) = E(s,a)∼dπ,r[r(s, a)] for some target policy π. In each of our experiments, π cor-
responds to a noisy version of a policy trained by a TD3 agent (Fujimoto et al., 2018), a commonly
used deep reinforcement learning algorithm. Denote πd, the deterministic policy trained by TD3
using the author’s GitHub https://github.com/sfujim/TD3. The target policy is deﬁned
as: π + N (0, σ2), where σ = 0.1. The off-policy evaluation algorithms are trained on a data set
generated by a single behavior policy πb. The experiments are done with two settings “easy” and
“hard” which vary the behavior policy and the size of the data set. All other settings are kept ﬁxed.
For the “easy” setting the behavior policy is deﬁned as:

πb = πd + N (0, σ2

b ), σb = 0.133,

(37)

and 500k time steps are collected (approximately 500 trajectories for most tasks). The “easy” setting
is roughly based on the experimental setting from Zhang et al. (2020a). For the “hard” setting the
behavior policy adds an increased noise and selects random actions with p = 0.2:
(cid:26)πd + N (0, σ2

πb =

Uniform random action

b ), σb = 0.2 p = 0.8,
p = 0.2,

(38)

and only 50k time steps are collected (approximately 50 trajectories for most tasks). For Pendulum-
v0 and Humanoid-v3, the range of actions is [−2, 2] and [−0.4, 0.4] respectively, rather than [−1, 1],
so we scale the size of the noise added to actions accordingly. We set the discount factor to γ = 0.99.
All continuous-action experiments are over 10 seeds.

Pre-training. Both SR-DICE and Direct-SR rely on pre-training the encoder-decoder and deep
successor representation ψ. These networks were trained for 30k and 100k time steps respectively.
As noted in Section C.4, even when including this pre-training step, both algorithm have a lower
running time than DualDICE and GradientDICE.

Architecture. For fair comparison, we use the same architecture for all algorithms except for
DualDICE. This a fully connected neural network with 2 hidden layers of 256 and ReLU activa-
tion functions. This architecture was based on the network deﬁned in the TD3 GitHub and was not
tuned. For DualDICE, we found tanh activation functions improved stability over ReLU.

For SR-DICE and SR-Direct we use a separate architecture for the encoder-decoder network. The
encoder is a network with a single hidden layer of 256, making each φ(s, a) a feature vector of 256.

23

Under review as a conference paper at ICLR 2021

There are three decoders for reward, action, and next state, respectively. For the action decoder
and next state decoder we use a network with one hidden layer of 256. The reward decoder is a
linear function of the encoding, without biases. All hidden layers are followed by ReLU activation
functions.

Network hyper-parameters. All networks are trained with the Adam optimizer (Kingma &
Ba, 2014). We use a learning rate of 3e−4, again based on TD3 for all networks except for
GradientDICE, which we found required careful tuning to achieve a reasonable performance.
For GradientDICE we found a learning rate of 1e−5 for f and w, and 1e−2 for u achieved
the highest performance. For DualDICE we chose the best performing learning rate out of
{1e − 2, 1e − 3, 3e − 4, 5e − 5, 1e − 5}. SR-DICE, Direct-SR and Deep TD were not tuned and use
default hyper-parameters from deep RL algorithms. For training ψπ and Qπ for the deep reinforce-
ment learning aspects of SR-DICE, Direct-SR and Deep TD we use a mini-batch size of 256 and
update the target networks using τ = 0.005, again based on TD3. For all MIS methods, we use a
mini-batch size of 2048 as described by (Nachum et al., 2019a). We found SR-DICE and DualDICE
succeeded with lower mini-batch sizes but did not test this in detail. All hyper-parameters are de-
scribed in Table 2.

Table 2: Continuous-action environment training hyper-parameters.

Hyper-parameter

SR-DICE DualDICE GradientDICE Direct-SR Deep TD

Optimizer
ψπ, Qπ Learning rate
w Learning rate
f Learning rate
w Learning rate
u Learning rate
ψπ, Qπ Mini-batch size
w, f , w, u, Mini-batch size
ψπ, Qπ Target update rate

Adam
3e − 4
3e − 4
-
-
-
256
2048
0.005

Adam
-
-
5e − 5
5e − 5
-
-
2048
-

Adam
-
-
1e − 5
1e − 5
1e − 2
-
2048
-

Adam
3e − 4
3e − 4
-
-
-
256
2048
0.005

Adam
3e − 4
-
-
-
-
256
-
0.005

Visualizations. We graph the log MSE between the estimate of R(π) and the true R(π), where
the log MSE is computed as log 0.5(X − R(π))2. We smooth the learning curves over a uniform
window of 10. Agents were evaluated every 1k time steps and performance is measured over 250k
time steps total. Markers are displayed every 25k time steps with offset for visual clarity.

F.2 ATARI

We interface with Atari by OpenAI gym (version 0.17.2) (Brockman et al., 2016), all agents use the
NoFrameskip-v0 environments that include sticky actions with p = 0.25 (Machado et al., 2018b).

Pre-processing. We use standard pre-processing steps based on Machado et al. (2018b) and Castro
et al. (2018). We base our description on (Fujimoto et al., 2019a), which our code is closely based
on. We deﬁne the following:

• Frame: output from the Arcade Learning Environment.
• State: conventional notion of a state in a MDP.
• Input: input to the network.

The standard pre-processing steps are as follows:

• Frame: gray-scaled and reduced to 84 × 84 pixels, tensor with shape (1, 84, 84).
• State: the maximum pixel value over the 2 most recent frames, tensor with shape (1, 84, 84).
• Input: concatenation over the previous 4 states, tensor with shape (4, 84, 84).

The notion of time steps is applied to states, rather than frames, and functionally, the concept of
frames can be abstracted away once pre-processing has been applied to the environment.

The agent receives a state every 4th frame and selects one action, which is repeated for the following
4 frames. If the environment terminates within these 4 frames, the state received will be the last 2
frames before termination. For the ﬁrst 3 time steps of an episode, the input, which considers the

24

Under review as a conference paper at ICLR 2021

previous 4 states, sets the non-existent states to all 0s. An episode terminates after the game itself
terminates, corresponding to multiple lives lost (which itself is game-dependent), or after 27k time
steps (108k frames or 30 minutes in real time). Rewards are clipped to be within a range of [−1, 1].

Sticky actions are applied to the environment (Machado et al., 2018b), where the action at taken
at time step t, is set to the previously taken action at−1 with p = 0.25, regardless of the action
selected by the agent. Note this replacement is abstracted away from the agent and data set. In other
words, if the agent selects action a at state s, the transition stored will contain (s, a), regardless if a
is replaced by the previously taken action.

Experiments. For the main experiments we use a behavior and target policy derived from a Double
DQN agent (Van Hasselt et al., 2016), a commonly used deep reinforcement learning algorithm. The
behavior policy is an (cid:15)-greedy policy with (cid:15) = 0.1 and the target policy is the greedy policy (i.e.
(cid:15) = 0). In Section C.5 we perform two additional experiments with a different behavior policy. Oth-
erwise, all hyper-parameters are ﬁxed across experiments. For each, the data set contains 1 million
transitions and uses a discount factor of γ = 0.99. Each experiment is evaluated over 3 seeds.

Pre-training. Both SR-DICE and Direct-SR rely on pre-training the encoder-decoder and deep
successor representation ψ. Similar to the continuous-action tasks, these networks were trained for
30k and 100k time steps respectively.

Architecture. We use the same architecture as most value-based deep reinforcement learning al-
gorithms for Atari, e.g. (Mnih et al., 2015; Van Hasselt et al., 2016; Schaul et al., 2016). This
architecture is used for all networks, other than the encoder-decoder network, for fair comparison
and was not tuned in any way.

The network has a 3-layer convolutional neural network (CNN) followed by a fully connected net-
work with a single hidden layer. As mentioned in pre-processing, the input to the network is a tensor
with shape (4, 84, 84). The ﬁrst layer of the CNN has a kernel depth of 32 of size 8 × 8 and a stride
of 4. The second layer has a kernel depth of 32 of size 4 × 4 and a stride of 2. The third layer has
a kernel depth of 64 of size 3 × 3 and a stride of 1. The output of the CNN is ﬂattened to a vector
of 3136 before being passed to the fully connected network. The fully connected network has a
single hidden layer of 512. Each layer, other than the output layer, is followed by a ReLU activation
function. The ﬁnal layer of the network outputs |A| values where |A| is the number of actions.

The encoder-decoder used by SR-DICE and SR-Direct has a slightly different architecture. The
encoder is identical to the aforementioned architecture, except the ﬁnal layer outputs the feature
vector φ(s) with 256 dimensions and is followed by a ReLU activation function. The next state
decoder uses a single fully connected layer which transforms the vector of 256 to 3136 and then is
passed through three transposed convolutional layers each mirroring the CNN. Hence, the ﬁrst layer
has a kernel depth of 64, kernel size of 3 × 3 and a stride of 1. The second layer has a kernel depth
of 32, kernel size of 4 × 4 and a stride of 2. The ﬁnal layer has a kernel depth of 32, kernel size of
8 × 8 and a stride of 4. This maps to a (1, 84, 84) tensor. All layers other than the ﬁnal layer are
followed by ReLU activation functions. Although the input uses a history of the four previous states,
as mentioned in the pre-processing section, we only reconstruct the succeeding state without history.
We do this because there is overlap in the history of the current input and the input corresponding to
the next time step. The reward decoder is a linear function without biases.

Network hyper-parameters. Our hyper-parameter choices are based on standard hyper-parameters
based largely on (Castro et al., 2018). All networks are trained with the Adam optimizer (Kingma
& Ba, 2014). We use a learning rate of 6.25e−5. Although not traditionally though of has a hyper-
parameter, in accordance to prior work, we modify (cid:15) used by Adam to be 1.5e−4. For w we use a
learning rate of 3e−4 with the default setting of (cid:15) = 1e−8. For u we use 1e−3. We use a mini-batch
size of 32 for all networks. SR-DICE, Direct-SR and Deep TD update the target network every 8k
time steps. All hyper-parameters are described in Table 3.

Visualizations. We use identical visualizations to the continuous-action environments. Graphs
display the log MSE between the estimate of R(π) and the true R(π) of the target policy, where
the log MSE is computed as log 0.5(X − R(π))2. We smooth the learning curves over a uniform
window of 10. Agents were evaluated every 1k time steps and performance is measured over 250k
time steps total. Markers are displayed every 25k time steps with offset for visual clarity.

25

Under review as a conference paper at ICLR 2021

Table 3: Training hyper-parameters for the Atari domain.

Hyper-parameter

SR-DICE DualDICE GradientDICE Direct-SR Deep TD

Optimizer
ψπ, Qπ Learning rate
ψπ, Qπ, f , w Adam (cid:15)
w, u Adam (cid:15)
w Learning rate
f Learning rate
w Learning rate
u Learning rate
ψπ, Qπ Mini-batch size
w, f , w, u, Mini-batch size
ψπ, Qπ Target update rate

Adam
6.25e − 5
1.5e−4
1e−8
3e − 4
-
-
-
32
32
8k

Adam
-
1.5e−4
-
-
6.25e − 5
6.25e − 5
-
-
32
-

Adam
-
1.5e−4
1e−8
-
6.25e − 5
6.25e − 5
1e − 3
-
32
-

Adam
6.25e − 5
1.5e−4
1e−8
3e − 4
-
-
-
32
32
8k

Adam
6.25e − 5
1.5e−4
-
-
-
-
-
32
-
8k

26

