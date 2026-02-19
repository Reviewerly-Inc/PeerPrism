Published as a conference paper at ICLR 2023

NEAR-OPTIMAL DEPLOYMENT EFFICIENCY
IN
REWARD-FREE REINFORCEMENT LEARNING WITH
LINEAR FUNCTION APPROXIMATION

Dan Qiao
Department of Computer Science, UCSB
danqiao@ucsb.edu

Yu-Xiang Wang
Department of Computer Science, UCSB
yuxiangw@cs.ucsb.edu

ABSTRACT

We study the problem of deployment efficient reinforcement learning (RL) with
linear function approximation under the reward-free exploration setting. This is
a well-motivated problem because deploying new policies is costly in real-life
RL applications. Under the linear MDP setting with feature dimension d and
planning horizon H, we propose a new algorithm that collects at most (cid:101)O( d2H 5
ϵ2 )
trajectories within H deployments to identify ϵ-optimal policy for any (possibly
data-dependent) choice of reward functions. To the best of our knowledge, our
approach is the first to achieve optimal deployment complexity and optimal d
dependence in sample complexity at the same time, even if the reward is known
ahead of time. Our novel techniques include an exploration-preserving policy
discretization and a generalized G-optimal experiment design, which could be of
independent interest. Lastly, we analyze the related problem of regret minimization
in low-adaptive RL and provide information-theoretic lower bounds for switching
cost and batch complexity.

INTRODUCTION

1
In many practical reinforcement learning (RL) based tasks, limited computing resources hinder
applications of fully adaptive algorithms that frequently deploy new exploration policy. Instead,
it is usually cheaper to collect data in large batches using the current policy deployment. Take
recommendation system (Afsar et al., 2021) as an instance, the system is able to gather plentiful new
data in very short time, while the deployment of a new policy often takes longer time, as it requires
extensive computing and human resources. Therefore, it is impractical to switch the policy based on
instantaneous data as a typical RL algorithm would demand. A feasible alternative is to run a large
batch of experiments in parallel and only decide whether to update the policy after the whole batch is
complete. The same constraint also appears in other RL applications such as healthcare (Yu et al.,
2021), robotics (Kober et al., 2013) and new material design (Zhou et al., 2019). In those scenarios,
the agent needs to minimize the number of policy deployment while learning a good policy using
(nearly) the same number of trajectories as its fully-adaptive counterparts. On the empirical side,
Matsushima et al. (2020) first proposed the notion deployment efficiency. Later, Huang et al. (2022)
formally defined deployment complexity. Briefly speaking, deployment complexity measures the
number of policy deployments while requiring each deployment to have similar size. We measure the
adaptivity of our algorithms via deployment complexity and leave its formal definition to Section 2.

Under the purpose of deployment efficiency, the recent work by Qiao et al. (2022) designed an
algorithm that could solve reward-free exploration in O(H) deployments. However, their sam-
ple complexity (cid:101)O(|S|2|A|H 5/ϵ2), although being near-optimal under the tabular setting, can be
unacceptably large under real-life applications where the state space is enormous or continuous.
For environments with large state space, function approximations are necessary for representing
the feature of each state. Among existing work that studies function approximation in RL, linear
function approximation is arguably the simplest yet most fundamental setting. In this paper, we study
deployment efficient RL with linear function approximation under the reward-free setting, and we
consider the following question:
Question 1.1. Is it possible to design deployment efficient and sample efficient reward-free RL
algorithms with linear function approximation?

1

Published as a conference paper at ICLR 2023

Algorithms for reward-free RL
Algorithm 1 & 2 in Wang et al. (2020)
FRANCIS (Zanette et al., 2020b)‡
RFLIN (Wagenmaker et al., 2022b)‡
Algorithm 2 & 4 in Huang et al. (2022)‡
LARFE (Qiao et al., 2022)†
Our Algorithm 1 & 2 (Theorem 5.1)‡
Our Algorithm 1 & 2 (Theorem 7.1)⋆
Lower bound (Wagenmaker et al., 2022b)
Lower bound (Huang et al., 2022)

Sample complexity
(cid:101)O( d3H 6
ϵ2 )
(cid:101)O( d3H 5
ϵ2 )
(cid:101)O( d2H 5
ϵ2 )
(cid:101)O( d3H 5
)∗
ϵ2ν2
(cid:101)O( S2AH 5
)
ϵ2
(cid:101)O( d2H 5
ϵ2 )
(cid:101)O( S2AH 5
)
ϵ2
Ω( d2H 2
ϵ2 )
If polynomial sample

min

Deployment complexity
(cid:101)O( d3H 6
ϵ2 )
(cid:101)O( d3H 5
ϵ2 )
(cid:101)O( d2H 5
ϵ2 )
H

2H
H
H
N.A.
(cid:101)Ω(H)

Table 1: Comparison of our results (in blue) to existing work regarding sample complexity and
deployment complexity. We highlight that our results match the best known results for both sample
complexity and deployment complexity at the same time. ‡: We ignore the lower order terms in
sample complexity for simplicity. ∗: νmin is the problem-dependent reachability coefficient which is
upper bounded by 1 and can be arbitrarily small. †: This work is done under tabular MDP and we
transfer the O(HSA) switching cost to 2H deployments. ⋆: When both our algorithms are applied
under tabular MDP, we can replace one d in sample complexity by S.

Our contributions. In this paper, we answer the above question affirmatively by constructing an
algorithm with near-optimal deployment and sample complexities. Our contributions are threefold.

• A new layer-by-layer type algorithm (Algorithm 1) for reward-free RL that achieves deploy-
ment complexity of H and sample complexity of (cid:101)O( d2H 5
ϵ2 ). Our deployment complexity
is optimal while sample complexity has optimal dependence in d and ϵ. In addition, when
applied to tabular MDP, our sample complexity (Theorem 7.1) recovers best known result
(cid:101)O( S2AH 5
ϵ2

).

• We generalize G-optimal design and select near-optimal policy via uniform policy evaluation
on a finite set of representative policies instead of using optimism and LSVI. Such technique
helps tighten our sample complexity and may be of independent interest.

• We show that “No optimal-regret online learners can be deployment efficient” and deploy-
ment efficiency is incompatible with the highly relevant regret minimization setting. For
regret minimization under linear MDP, we present lower bounds (Theorem 7.2 and 7.3) for
other measurements of adaptivity: switching cost and batch complexity.

1.1 CLOSELY RELATED WORKS

There is a large and growing body of literature on the statistical theory of reinforcement learning that
we will not attempt to thoroughly review. Detailed comparisons with existing work on reward-free RL
(Wang et al., 2020; Zanette et al., 2020b; Wagenmaker et al., 2022b; Huang et al., 2022; Qiao et al.,
2022) are given in Table 1. For more discussion of relevant literature, please refer to Appendix A and
the references therein. Notably, all existing algorithms under linear MDP either admit fully adaptive
structure (which leads to deployment inefficiency) or suffer from sub-optimal sample complexity. In
addition, when applied to tabular MDP, our algorithm has the same sample complexity and slightly
better deployment complexity compared to Qiao et al. (2022).

The deployment efficient setting is slightly different from other measurements of adaptivity. The low
switching setting (Bai et al., 2019) restricts the number of policy updates, while the agent can decide
whether to update the policy after collecting every single trajectory. This can be difficult to implement
in practical applications. A more relevant setting, the batched RL setting (Zhang et al., 2022) requires
decisions about policy changes to be made at only a few (often predefined) checkpoints. Compared
to batched RL, the requirement of deployment efficiency is stronger by requiring each deployment
to collect the same number of trajectories. Therefore, deployment efficient algorithms are easier to
deploy in parallel (see, e.g., Huang et al., 2022, for a more elaborate discussion). Lastly, we remark
that our algorithms also work under the batched RL setting by running in H batches.

2

Published as a conference paper at ICLR 2023

Technically, our method is inspired by optimal experiment design – a well-developed research area
from statistics. In particular, a major technical contribution of this paper is to solve a variant of
G-optimal experiment design while solving exploration in RL at the same time. Zanette et al. (2020b);
Wagenmaker et al. (2022b) choose policy through online experiment design, i.e., running no-regret
online learners to select policies adaptively for approximating the optimal design. Those online
approaches, however, cannot be applied under our problem due to the requirement of deployment
efficiency. To achieve deployment complexity of H, we can only deploy one policy for each layer, so
we need to decide the policy based on sufficient exploration for only previous layers. Therefore, our
approach requires offline experiment design and thus raises substantial technical challenge.

A remark on technical novelty. The general idea behind previous RL algorithms with low adaptivity
is optimism and doubling schedule for updating policies that originates from UCB2 (Auer et al., 2002).
The doubling schedule, however, can not provide optimal deployment complexity. Different from
those approaches, we apply layer-by-layer exploration to achieve the optimal deployment complexity,
and our approach is highly non-trivial. Since we can only deploy one policy for each layer, there are
two problems to be solved: the existence of a single policy that can explore all directions of a specific
layer and how to find such policy. We generalize G-optimal design to show the existence of such
explorative policy. Besides, we apply exploration-preserving policy discretization for approximating
our generalized G-optimal design. We leave detailed discussions about these techniques to Section 3.

√

2 PROBLEM SETUP
Notations. Throughout the paper, for n ∈ Z+, [n] = {1, 2, · · · , n}. We denote ∥x∥Λ =
x⊤Λx. For
matrix X ∈ Rd×d, ∥ · ∥2, ∥ · ∥F , λmin(·), λmax(·) denote the operator norm, Frobenius norm, smallest
eigenvalue and largest eigenvalue, respectively. For policy π, Eπ and Pπ denote the expectation and
probability measure induced by π under the MDP we consider. For any set U , ∆(U ) denotes the set
of all possible distributions over U . In addition, we use standard notations such as O and Ω to absorb
constants while (cid:101)O and (cid:101)Ω suppress logarithmic factors.
Markov Decision Processes. We consider finite-horizon episodic Markov Decision Processes (MDP)
with non-stationary transitions, denoted by a tuple M = (S, A, H, Ph, rh) (Sutton & Barto, 1998),
where S is the state space, A is the action space and H is the horizon. The non-stationary transition
kernel has the form Ph : S ×A×S (cid:55)→ [0, 1] with Ph(s′|s, a) representing the probability of transition
from state s, action a to next state s′ at time step h. In addition, rh(s, a) ∈ ∆([0, 1]) denotes the
corresponding distribution of reward.1 Without loss of generality, we assume there is a fixed initial
state s1.2 A policy can be seen as a series of mapping π = (π1, · · · , πH ), where each πh maps
each state s ∈ S to a probability distribution over actions, i.e. πh : S → ∆(A), ∀ h ∈ [H]. A
random trajectory (s1, a1, r1, · · · , sH , aH , rH , sH+1) is generated by the following rule: s1 is fixed,
ah ∼ πh(·|sh), rh ∼ rh(sh, ah), sh+1 ∼ Ph(·|sh, ah), ∀ h ∈ [H].

Q-values, Bellman (optimality) equations. Given a policy π and any h ∈ [H], the value function
V π
h (·) and Q-value function Qπ
h(s, a) =
Eπ[(cid:80)H
t=h rt|sh, ah = s, a], ∀ s, a ∈ S × A. Besides, the value function and Q-value function with
respect to the optimal policy π⋆ is denoted by V ⋆
h (·) and Q⋆
h(·, ·). Then Bellman (optimality) equation
follows ∀ h ∈ [H]:

h(·, ·) are defined as: V π

h (s) = Eπ[(cid:80)H

t=h rt|sh = s], Qπ

Qπ
Q⋆

h(s, a) = rh(s, a) + Ph(·|s, a)V π
h(s, a) = rh(s, a) + Ph(·|s, a)V ⋆

h+1, V π
h+1, V ⋆

h = Ea∼πh [Qπ
h],
h(·, a).
h = max

Q⋆

a

In this work, we consider the reward-free RL setting, where there may be different reward functions.
Therefore, we denote the value function of policy π with respect to reward r by V π(r). Similarly,
V ⋆(r) denotes the optimal value under reward function r. We say that a policy π is ϵ-optimal with
respect to r if V π(r) ≥ V ⋆(r) − ϵ.

Linear MDP (Jin et al., 2020b). An episodic MDP (S, A, H, P, r) is a linear MDP with known
feature map ϕ : S × A → Rd if there exist H unknown signed measures µh ∈ Rd over S and H
unknown reward vectors θh ∈ Rd such that
Ph (s′ | s, a) = ⟨ϕ(s, a), µh (s′)⟩ ,

∀ (h, s, a, s′) ∈ [H] × S × A × S.

rh (s, a) = ⟨ϕ(s, a), θh⟩ ,

1We abuse the notation r so that r also denotes the expected (immediate) reward function.
2The generalized case where the initial distribution is an arbitrary distribution can be recovered from this

setting by adding one layer to the MDP.

3

Published as a conference paper at ICLR 2023

Without loss of generality, we assume ∥ϕ(s, a)∥2 ≤ 1 for all s, a; and for all h ∈ [H], ∥µh(S)∥2 ≤
√

√

d, ∥θh∥2 ≤

d.

For policy π, we define Λπ,h := Eπ[ϕ(sh, ah)ϕ(sh, ah)⊤], the expected covariance matrix with
respect to policy π and time step h (here sh, ah follows the distribution induced by policy π). Let
λ⋆ = minh∈[H] supπ λmin(Λπ,h). We make the following assumption regarding explorability.
Assumption 2.1 (Explorability of all directions). The linear MDP we have satisfies λ⋆ > 0.

We remark that Assumption 2.1 only requires the existence of a (possibly non-Markovian) policy to
visit all directions for each layer and it is analogous to other explorability assumptions in papers about
RL under linear representation (Zanette et al., 2020b; Huang et al., 2022; Wagenmaker & Jamieson,
2022). In addition, the parameter λ⋆ only appears in lower order terms of sample complexity bound
and our algorithms do not take λ⋆ as an input.

Reward-Free RL. The reward-free RL setting contains two phases, the exploration phase and the
planning phase. Different from PAC RL3 setting, the learner does not observe the rewards during the
exploration phase. Besides, during the planning phase, the learner has to output a near-optimal policy
for any valid reward functions. More specifically, the procedure is:

1. Exploration phase: Given accuracy ϵ and failure probability δ, the learner explores an MDP

for K(ϵ, δ) episodes and collects the trajectories without rewards {sk

h, ak

h}(h,k)∈[H]×[K].

2. Planning phase: The learner outputs a function (cid:98)π(·) which takes reward function as input.
The function (cid:98)π(·) satisfies that for any valid reward function r, V (cid:98)π(r)(r) ≥ V ⋆(r) − ϵ.

The goal of reward-free RL is to design a procedure that satisfies the above guarantee with probability
at least 1 − δ while collecting as few episodes as possible. According to the definition, any procedure
satisfying the above guarantee is provably efficient for PAC RL setting.

Deployment Complexity. In this work, we measure the adaptivity of our algorithm through deploy-
ment complexity, which is defined as:
Definition 2.2 (Deployment complexity (Huang et al., 2022)). We say that an algorithm has deploy-
ment complexity of M , if the algorithm is guaranteed to finish running within M deployments. In
addition, the algorithm is only allowed to collect at most N trajectories during each deployment,
where N should be fixed a priori and cannot change adaptively.

We consider the deployment of non-Markovian policies (i.e. mixture of deterministic policies) (Huang
et al., 2022). The requirement of deployment efficiency is stronger than batched RL (Zhang et al.,
2022) or low switching RL (Bai et al., 2019), which makes deployment-efficient algorithms more
practical in real-life applications. For detailed comparison between these definitions, please refer to
Section 1.1 and Appendix A.

3 TECHNIQUE OVERVIEW

h, an

, where Λh = I + (cid:80)N

In order to achieve the optimal deployment complexity of H, we apply layer-by-layer exploration.
More specifically, we construct a single policy πh to explore layer h based on previous data. Follow-
ing the general methods in reward-free RL (Wang et al., 2020; Wagenmaker et al., 2022b), we do
exploration through minimizing uncertainty. As will be made clear in the analysis, given exploration
dataset D = {sn
h}h,n∈[H]×[N ], the uncertainty of layer h with respect to policy π can be char-
acterized by Eπ∥ϕ(sh, ah)∥Λ−1
h)⊤ is (regularized and
n=1 ϕ(sn
unnormalized) empirical covariance matrix. Note that although we can not directly optimize Λh, we
can maximize the expectation Nπh · Eπh[ϕhϕ⊤
h ] (Nπh is the number of trajectories we apply πh) by
optimizing the policy πh. Therefore, to minimize the uncertainty with respect to some policy set Π,
we search for an explorative policy π0 to minimize maxπ∈Π Eπϕ(sh, ah)(Eπ0ϕhϕ⊤
3.1 GENERALIZED G-OPTIMAL DESIGN
For the minimization problem above, traditional G-optimal design handles the case where each
deterministic policy π generates some ϕπ at layer h with probability 1 (i.e. we directly choose ϕ
instead of choosing π), as is the case under deterministic MDP. However, traditional G-optimal design

h )−1ϕ(sh, ah).

h)ϕ(sn

h, an

h, an

h

3Also known as reward-aware RL, which aims to identify near optimal policy given reward function.

4

Published as a conference paper at ICLR 2023

cannot tackle our problem since under general linear MDP, each π will generate a distribution over
the feature space instead of a single feature vector. We generalize G-optimal design and show that for
any policy set Π, the following Theorem 3.1 holds. More details are deferred to Appendix B.
Theorem 3.1 (Informal version of Theorem B.1). If there exists policy π0 ∈ ∆(Π) such that
λmin(Eπ0 ϕhϕ⊤

h ) > 0, then minπ0∈∆(Π) maxπ∈Π Eπϕ(sh, ah)(Eπ0ϕhϕ⊤

h )−1ϕ(sh, ah) ≤ d.

Generally speaking, Theorem 3.1 states that for any Π, there exists a single policy from ∆(Π)
(i.e., mixture of several policies in Π) that can efficiently reduce the uncertainty with respect to Π.
Therefore, assume we want to minimize the uncertainty with respect to Π and we are able to derive
the solution π0 of the minimization above, we can simply run π0 repeatedly for several episodes.

However, there are two gaps between Theorem 3.1 and our goal of reward free RL. First, under the
Reinforcement Learning setting, the association between policy π and the corresponding distribution
of ϕh is unknown, which means we need to approximate the above minimization. It can be done
by estimating the two expectations and we leave the discussion to Section 3.3. The second gap is
about choosing appropriate Π in Theorem 3.1, for which a natural idea is to use the set of all policies.
It is however infeasible to simultaneously estimate the expectations for all π accurately. The size
of {all policies} is infinity and ∆({all policies}) is even bigger. It seems intractable to control its
complexity using existing uniform convergence techniques (e.g., a covering number argument).

3.2 DISCRETIZATION OF POLICY SET

The key realization towards a solution to the above problem is that we do not need to consider the set
of all policies. It suffices to consider a smaller subset Π that is more amenable to an ϵ-net argument.
This set needs to satisfy a few conditions.

(1) Due to condition in Theorem 3.1, Π should contain explorative policies covering all directions.
(2) Π should contain a representative policy set Πeval such that it contains a near-optimal policy for
any reward function.

(3) Since we apply offline experimental design via approximating the expectations, Π must be “small”
enough for a uniform-convergence argument to work.

We show that we can construct a finite set Π with |Π| being small enough while satisfying Con-
dition (1) and (2). More specifically, given the feature map ϕ(·, ·) and the desired accuracy
ϵ, we can construct the explorative policy set Πexp
ϵ,h |) ≤ (cid:101)O(d2 log(1/ϵ)),
where Πexp
ϵ,h is the policy set for layer h. In addition, when ϵ is small compared to λ⋆, we have
supπ∈∆(Πexp
and
approximating the minimization problem, after the exploration phase we will be able to estimate the
value functions of all π ∈ Πexp

), which verifies Condition (1).4 Plugging in Πexp

such that log(|Πexp

h ) ≥ (cid:101)Ω( (λ⋆)2

) λmin(Eπϕhϕ⊤

accurately.

d

ϵ

ϵ

ϵ

ϵ

, we can further select a subset, and we call it policies to evaluate: Πeval
ϵ,h |) = (cid:101)O(d log(1/ϵ)) while for any possible linear MDP with feature map ϕ(·, ·), Πeval

It remains to check Condition (2) by formalizing the representative policy set discussed above. From
Πexp
. It satisfies that
ϵ
log(|Πeval
is
guaranteed to contain one ϵ-optimal policy. As a result, it suffices to estimate the value functions of
all policies in Πeval
and output the greedy one with the largest estimated value.5

ϵ

ϵ

ϵ

3.3 NEW APPROACH TO ESTIMATE VALUE FUNCTION

Now that we have a discrete policy set, we still need to estimate the two expectations in Theorem 3.1.
We design a new algorithm (Algorithm 4, details can be found in Appendix E) based on the technique
of LSVI (Jin et al., 2020b) to estimate Eπr(sh, ah) given policy π, reward r and exploration data.
Algorithm 4 can estimate the expectations accurately simultaneously for all π ∈ Πexp and r (that
appears in the minimization problem) given sufficient exploration of the first h − 1 layers. Therefore,
under our layer-by-layer exploration approach, after adequate exploration for the first h − 1 layers,
Algorithm 4 provides accurate estimations for Eπ0ϕhϕ⊤
h )−1ϕ(sh, ah)].
As a result, the (1) we solve serves as an accurate approximation of the minimization problem in
Theorem 3.1 and the solution πh of (1) is provably efficient in exploration.

h and Eπ[ϕ(sh, ah)⊤((cid:98)Eπ0ϕhϕ⊤

4For more details about explorative policies, please refer to Appendix C.3.
5For more details about policies to evaluate, please refer to Appendix C.2.

5

Published as a conference paper at ICLR 2023

Finally, after sufficient exploration of all H layers, the last step is to estimate the value functions of
all policies in Πeval. We design a slightly different algorithm (Algorithm 3, details in Appendix D)
for this purpose. Based on LSVI, Algorithm 3 takes π ∈ Πeval and reward function r as input, and
estimates V π(r) accurately given sufficient exploration for all H layers.

4 ALGORITHMS

In this section, we present our main algorithms. The algorithm for the exploration phase is Algorithm
1 which formalizes the ideas in Section 3, while the planning phase is presented in Algorithm 2.

Algorithm 1 Layer-by-layer Reward-Free Exploration via Experimental Design (Exploration)

1: Input: Accuracy ϵ. Failure probability δ.
2: Initialization: ι = log(dH/ϵδ). Error budget for each layer ¯ϵ = C1ϵ
√

Section 3.2. Number of episodes for each deployment N = C2dι

H 2

d·ι

¯ϵ2 = C2d2H 4ι3

C2

1 ϵ2

. Construct Πexp

ϵ/3 as in
. Dataset D = ∅.

3: for h = 1, 2, · · · , H do
4:
5:

Solve the following optimization problem.

πh =

argmin

π∈∆(Πexp

ϵ/3 ) s.t. λmin((cid:98)Σπ)≥C3d2H¯ϵι

(cid:104)

max
(cid:98)π∈Πexp
ϵ/3

(cid:98)E
(cid:98)π

ϕ(sh, ah)⊤(N · (cid:98)Σπ)−1ϕ(sh, ah)

(cid:105)

,

(1)

6:

7:

(cid:105)
(cid:104)
ϕ(sh, ah)⊤(N · (cid:98)Σπ)−1ϕ(sh, ah)
¯ϵ

where (cid:98)Σπ is (cid:98)Eπ[ϕ(sh, ah)ϕ(sh, ah)⊤] = EstimateER(π, ϕ(s, a)ϕ(s, a)⊤, A = 1, h, D, s1),
(cid:98)E
= EstimateER((cid:98)π, ϕ(s, a)⊤(N · (cid:98)Σπ)−1ϕ(s, a), A =
(cid:98)π
C2d3Hι2 , h, D, s1).
for n = 1, 2, · · · , N do

// Both expectations are estimated via Algorithm 4.

Run πh and add trajectory {sn

i , an

i }i∈[H] to D.

// Run Policy πh for N episodes.

8:
9:
10:
11: end for
12: Output: Dataset D.

end for

Exploration Phase. We apply layer-by-layer exploration and πh is the stochastic policy we deploy
to explore layer h. For solving πh, we approximate generalized G-optimal design via (1). For each
candidate π and (cid:98)π, we estimate the two expectations by calling EstimateER (Algorithm 4, details in
Appendix E). EstimateER is a generic subroutine for estimating the value function under a particular
reward design. We estimate the two expectations of interest by carefully choosing one specific reward
design for each coordinate separately, so that the resulting value function provides an estimate to the
desired quantity in that coordinate. 6 As mentioned above and will be made clear in the analysis,
given adequate exploration of the first h − 1 layers, all estimations will be accurate and the surrogate
policy πh is sufficiently explorative for all directions at layer h.

The restriction on λmin((cid:98)Σπ) is for technical reason only, and we will show that under the assumption
in Theorem 5.1, there exists valid solution of (1). Lastly, we remark that solving (1) is inefficient in
general. Detailed discussions about computation are deferred to Section 7.2.

Algorithm 2 Find Near-Optimal Policy Given Reward Function (Planning)
1: Input: Dataset D from Algorithm 1. Feasible linear reward function r = {rh}h∈[H].
2: Initialization: Construct Πeval
// The set of policies to evaluate.
3: for π ∈ Πeval
ϵ/3 do

ϵ/3 as in Section 3.2.

(cid:98)V π(r) = EstimateV(π, r, D, s1).

// Estimate value functions using Algorithm 3.

(cid:98)V π(r).

// Output the greedy policy w.r.t (cid:98)V π(r).

4:
5: end for
6: (cid:98)π = arg maxπ∈Πeval
7: Output: Policy (cid:98)π.

ϵ/3

6For (cid:98)Σπ, what we need to handle is matrix reward ϕhϕ⊤

h and stochastic policy π ∈ ∆(Πexp

ϵ/3 ), we apply

generalized version of Algorithm 4 to tackle this problem as discussed in Appendix F.1.

6

Published as a conference paper at ICLR 2023

Planning Phase. The output dataset D from the exploration phase contains sufficient information for
the planning phase. In the planning phase (Algorithm 2), we construct a set of policies to evaluate
and repeatedly apply Algorithm 3 (in Appendix D) to estimate the value function of each policy given
reward function. Finally, Algorithm 2 outputs the policy with the highest estimated value. Since D
has acquired sufficient information, all possible estimations in line 4 are accurate. Together with the
property that there exists near-optimal policy in Πeval

ϵ/3 , we have that the output (cid:98)π is near-optimal.

5 MAIN RESULTS
In this section, we state our main results, which formalize the techniques and algorithmic ideas we
discuss in previous sections.
Theorem 5.1. We run Algorithm 1 to collect data and let Planning(·) denote the output of Algorithm
2. There exist universal constants C1, C2, C3, C4 > 07 such that for any accuracy ϵ > 0 and failure
C4d7/2 log(1/λ⋆) , with probability 1 − δ, for any feasible linear
probability δ > 0, as well as ϵ <
reward function r, Planning(r) returns a policy that is ϵ-optimal with respect to r. In addition, the
deployment complexity of Algorithm 1 is H while the number of trajectories is (cid:101)O( d2H 5

H(λ⋆)2

ϵ2 ).

The proof of Theorem 5.1 is sketched in Section 6 with details in the Appendix. Below we discuss
some interesting aspects of our results.

Near optimal deployment efficiency. First, the deployment complexity of our Algorithm 1 is
optimal up to a log-factor among all reward-free algorithms with polynomial sample complexity,
according to a Ω(H/ logd(N H)) lower bound (Theorem B.3 of Huang et al. (2022)). In comparison,
the deployment complexity of RFLIN (Wagenmaker et al., 2022b) can be the same as their sample
complexity (also (cid:101)O(d2H 5/ϵ2)) in the worst case.

Near optimal sample complexity. Secondly, our sample complexity matches the best-known sam-
ple complexity (cid:101)O(d2H 5/ϵ2) (Wagenmaker et al., 2022b) of reward-free RL even when deployment
efficiency is not needed. It is also optimal in parameter d and ϵ up to lower-order terms, when
compared against the lower bound of Ω(d2H 2/ϵ2) (Theorem 2 of Wagenmaker et al. (2022b)).

Dependence on λ⋆. A striking difference of our result comparing to the closest existing work
(Huang et al., 2022) is that the sample complexity is independent to the explorability parameter λ⋆
in the small-ϵ regime. This is highly desirable because we only require a non-zero λ⋆ to exist, and
smaller λ⋆ does not affect the sample complexity asymptotically. In addition, our algorithm does
not take λ⋆ as an input (although we admit that the theoretical guarantee only holds when ϵ is small
compared to λ⋆). In contrast, the best existing result (Algorithm 2 of Huang et al. (2022)) requires
min) for any
the knowledge of explorability parameter νmin
ϵ > 0. We leave detailed comparisons with Huang et al. (2022) to Appendix G.

8 and a sample complexity of (cid:101)O(1/ϵ2ν2

H(λ⋆)2

Sample complexity in the large-ϵ regime. For the case when ϵ is larger than the threshold:
C4d7/2 log(1/λ⋆) , we can run the procedure with ϵ =
C4d7/2 log(1/λ⋆) , and the sample complexity will
be (cid:101)O( d9H 3
ϵ2 + d9H 3
(λ⋆)4 ).
This effectively says that the algorithm requires a “Burn-In” period before getting non-trivial results.
Similar limitations were observed for linear MDPs before (Huang et al., 2022; Wagenmaker &
Jamieson, 2022) so it is not a limitation of our analysis.

(λ⋆)4 ). So the overall sample complexity for any ϵ > 0 can be bounded by (cid:101)O( d2H 5

H(λ⋆)2

Comparison to Qiao et al. (2022). Algorithm 4 (LARFE) of Qiao et al. (2022) tackles reward-free
exploration under tabular MDP in O(H) deployments while collecting (cid:101)O( S2AH 5
) trajectories. We
generalize their result to reward-free RL under linear MDP with the same deployment complexity.
More importantly, although a naive instantiation of our main theorem to the tabular MDP only gives
), a small modification to an intermediate argument gives the same (cid:101)O( S2AH 5
(cid:101)O( S2A2H 5
), which
ϵ2
matches the best-known results for tabular MDP. More details will be discussed in Section 7.1.

ϵ2

ϵ2

7C1, C2, C3 are the universal constants in Algorithm 1.
8νmin in Huang et al. (2022) is defined as νmin = minh∈[H] min∥θ∥=1 maxπ

(cid:112)Eπ[(ϕ⊤

h θ)2], which is also

measurement of explorability. Note that νmin is always upper bounded by 1 and can be arbitrarily small.

7

Published as a conference paper at ICLR 2023

6 PROOF SKETCH
In this part, we sketch the proof of Theorem 5.1. Notations ι, ¯ϵ, Ci (i ∈ [4]), Πexp, Πeval, (cid:98)Σπ and
(cid:98)Eπ are defined in Algorithm 1. We start with the analysis of deployment complexity.
Deployment complexity. Since for each layer h ∈ [H], we only deploy one stochastic policy πh for
exploration, the deployment complexity is H. Next we focus on the sample complexity.
Sample complexity. Our proof of sample complexity bound results from induction. With the choice
of ¯ϵ and N from Algorithm 1, suppose that Λk
is empirical covariance matrix from data up to the
(cid:101)h
Eπ[(cid:80)h−1
(cid:101)h=1

k-th deployment9, we assume maxπ∈Πexp

(cid:101)h)] ≤ (h − 1)¯ϵ

)−1ϕ(s

(cid:101)h, a

ϕ(s

(cid:113)

ϵ/3

holds and prove that with high probability, maxπ∈Πexp

ϵ/3

Eπ[

h)−1ϕ(sh, ah)] ≤ ¯ϵ.

(cid:101)h, a
(cid:113)

(cid:101)h)⊤(Λh−1
(cid:101)h
ϕ(sh, ah)⊤(Λh

Note that the induction condition implies that the uncertainty for the first h − 1 layers is small, we
have the following key lemma that bounds the estimation error of (cid:98)Σπ from (1).
Lemma 6.1. With high probability, for all π ∈ ∆(Πexp

ϵ/3 ), ∥(cid:98)Σπ − Eπϕhϕ⊤

h ∥2 ≤ C3d2H¯ϵι

.

4

According to our assumption on ϵ,
λmin(E¯π⋆

ϕhϕ⊤

. Therefore, ¯π⋆

h ) ≥ 5C3d2H¯ϵι
(cid:104)
ϕ(sh, ah)⊤(N · (cid:98)Σπh )−1ϕ(sh, ah)

4

h

max
(cid:98)π∈Πexp
ϵ/3

(cid:98)E

(cid:98)π

the optimal policy for exploration ¯π⋆
h

10 satisfies that

h is a feasible solution of (1) and it holds that:

(cid:105)

≤ max
(cid:98)π∈Πexp
ϵ/3

(cid:104)

(cid:98)E

(cid:98)π

ϕ(sh, ah)⊤(N · (cid:98)Σ¯π⋆

h

(cid:105)
)−1ϕ(sh, ah)

.

h

)−1 and (N · (cid:98)Σπh)−1 ≽ (2Λh

Moreover, due to matrix concentration and the Lemma 6.1 we derive, we can prove that ( 4
((cid:98)Σ¯π⋆
following lemma bounds the estimation error of (cid:98)E
Lemma 6.2. With high probability, for all (cid:98)π ∈ Πexp

)−1 ≽
h)−1. 11 In addition, similar to the estimation error of (cid:98)Σπ, the

(cid:98)π[ϕ(sh, ah)⊤(N · (cid:98)Σπ)−1ϕ(sh, ah)] from (1).
ϵ/3 , π ∈ ∆(Πexp

ϵ/3 ) such that λmin((cid:98)Σπ) ≥ C3d2H¯ϵι,

5 Σ¯π⋆

h

(cid:12)
(cid:12)(cid:98)E
(cid:12)

(cid:98)π

(cid:104)

(cid:105)
ϕ(sh, ah)⊤(N · (cid:98)Σπ)−1ϕ(sh, ah)

− E

(cid:98)π

(cid:104)
ϕ(sh, ah)⊤(N · (cid:98)Σπ)−1ϕ(sh, ah)

(cid:105)(cid:12)
(cid:12)
(cid:12) ≤

¯ϵ2
2d2 ≤

¯ϵ2
8

.

With all the conclusions above, we have (Σπ is short for Eπ[ϕhϕ⊤

h ]):

3¯ϵ2
8

≥

5d
4N

+

¯ϵ2
8

≥ max
(cid:98)π∈Πexp
ϵ/3

E
(cid:98)π[ϕ(sh, ah)⊤(

4N
5

· Σ¯π⋆

h

)−1ϕ(sh, ah)] +

¯ϵ2
8

≥ max
(cid:98)π∈Πexp
ϵ/3

≥ max
(cid:98)π∈Πexp
ϵ/3

≥ max
(cid:98)π∈Πexp
ϵ/3

E
(cid:98)π[ϕ(sh, ah)⊤(N · (cid:98)Σ¯π⋆

h

)−1ϕ(sh, ah)] +

¯ϵ2
8

(cid:98)E
(cid:98)π[ϕ(sh, ah)⊤(N · (cid:98)Σ¯π⋆

h

(cid:98)E
(cid:98)π[ϕ(sh, ah)⊤(N · (cid:98)Σπh )−1ϕ(sh, ah)]

E
(cid:98)π[ϕ(sh, ah)⊤(N · (cid:98)Σπh )−1ϕ(sh, ah)] −

)−1ϕ(sh, ah)] ≥ max
(cid:98)π∈Πexp
¯ϵ2
8

ϵ/3

≥ max
(cid:98)π∈Πexp
ϵ/3

E
(cid:98)π[ϕ(sh, ah)⊤(2Λh

h)−1ϕ(sh, ah)] −

¯ϵ2
8

≥

1
2

max
(cid:98)π∈Πexp
ϵ/3

(cid:18)

E
(cid:98)π

(cid:113)

ϕ(sh, ah)⊤(Λh

h)−1ϕ(sh, ah)

(cid:19)2

−

¯ϵ2
8

.

As a result, the induction holds. Together with the fact that Πeval
maxπ∈Πeval
Lemma 6.3. With high probability, for all π ∈ Πeval

Eπ[(cid:80)H

h=1

ϵ/3

(cid:112)ϕ(sh, ah)⊤(Λh)−1ϕ(sh, ah)] ≤ H¯ϵ. We have the following lemma.

ϵ/3 and r, | (cid:98)V π(r) − V π(r)| ≤ (cid:101)O(H

ϵ/3 is subset of Πexp

ϵ/3 , we have

d) · H¯ϵ ≤ ϵ
3 .

√

Finally, since Πeval

ϵ/3 contains ϵ/3-optimal policy, the greedy policy with respect to (cid:98)V π(r) is ϵ-optimal.

9Detailed definition is deferred to Appendix F.4.
10Solution of the actual minimization problem, detailed definition in (39).
11Σ¯π⋆

= E¯π⋆

[ϕhϕ⊤

h ]. The proof is through direct calculation, details are deferred to Appendix F.6.

h

h

8

Published as a conference paper at ICLR 2023

7 SOME DISCUSSIONS
In this section, we discuss some interesting extensions of our main results.

h(s, a) > 0 where dπ

7.1 APPLICATION TO TABULAR MDP
Under the special case where the linear MDP is actually a tabular MDP and the feature map is
canonical basis (Jin et al., 2020b), our Algorithm 1 and 2 are still provably efficient. Suppose
the tabular MDP has discrete state-action space with cardinality |S| = S, |A| = A, let dm =
minh supπ mins,a dπ
h is occupancy measure, then the following theorem holds.
Theorem 7.1 (Informal version of Theorem H.2). With minor revision to Algorithm 1 and 2, when
ϵ is small compared to dm, our algorithms can solve reward-free exploration under tabular MDP
within H deployments and the sample complexity is bounded by (cid:101)O( S2AH 5
The detailed version and proof of Theorem 7.1 are deferred to Appendix H.1 due to space limit. We
highlight that we recover the best known result from Qiao et al. (2022) under mild assumption about
reachability to all (state,action) pairs. The replacement of one d by S is mainly because under tabular
MDP, there are AS different deterministic policies for layer h and the log-covering number of Πeval
can be improved from (cid:101)O(d) to (cid:101)O(S). In this way, we effectively save a factor of A.
7.2 COMPUTATIONAL EFFICIENCY
We admit that solving the optimization problem (1) is inefficient in general, while this can be solved
approximately in exponential time by enumerating π from a tight covering set of ∆(Πexp
ϵ/3 ). Note
that the issue of computational tractability arises in many previous works (Zanette et al., 2020a;
Wagenmaker & Jamieson, 2022) that focused on information-theoretic results under linear MDP, and
such issue is usually not considered as a fundamental barrier. For efficient surrogate of (1), we remark
that a possible method is to apply softmax (or other differentiable) representation of the policy space
and use gradient-based optimization techniques to find approximate solution of (1).

).

ϵ2

h

√

7.3 POSSIBLE EXTENSIONS TO REGRET MINIMIZATION WITH LOW ADAPTIVITY
In this paper, we tackle the problem of deployment efficient reward-free exploration while the optimal
adaptivity under regret minimization still remains open. We remark that deployment complexity is
not an ideal measurement of adaptivity for this problem since the definition requires all deployments
T ) if we want regret bound
to have similar sizes, which forces the deployment complexity to be (cid:101)Ω(
of order (cid:101)O(
T ). Therefore, the more reasonable task is to design algorithms with near optimal
switching cost or batch complexity. We present the following two lower bounds whose proof is
deferred to Appendix H.2. Here the number of episodes is K and the number of steps T := KH.
Theorem 7.2. For any algorithm with the optimal (cid:101)O((cid:112)poly(d, H)T ) regret bound, the switching
cost is at least Ω(dH log log T ).
Theorem 7.3. For any algorithm with the optimal (cid:101)O((cid:112)poly(d, H)T ) regret bound, the number of
batches is at least Ω( H

√

logd T + log log T ).

To generalize our Algorithm 1 to regret minimization, what remains is to remove Assumption 2.1.
Suppose we can do accurate uniform policy evaluation (as in Algorithm 2) with low adaptivity
without assumption on explorability of policy set, then we can apply iterative policy elimination
(i.e., eliminate the policies that are impossible to be optimal) and do exploration with the remaining
policies. Although Assumption 2.1 is common in relevant literature, it is not necessary intuitively
since under linear MDP, if some direction is hard to encounter, we do not necessarily need to gather
much information on this direction. Under tabular MDP, Qiao et al. (2022) applied absorbing MDP
to ignore those “hard to visit” states and we leave generalization of such idea as future work.

8 CONCLUSION
In this work, we studied the well-motivated deployment efficient reward-free RL with linear function
approximation. Under the linear MDP model, we designed a novel reward-free exploration algorithm
that collects (cid:101)O( d2H 5
ϵ2 ) trajectories in only H deployments. And both the sample and deployment
complexities are near optimal. An interesting future direction is to design algorithms to match our
lower bounds for regret minimization with low adaptivity. We believe the techniques we develop
(generalized G-optimal design and exploration-preserving policy discretization) could serve as basic
building blocks and we leave the generalization as future work.

9

Published as a conference paper at ICLR 2023

ACKNOWLEDGMENTS

The research is partially supported by NSF Awards #2007117. The authors would like to thank Jiawei
Huang and Nan Jiang for explaining the result of their paper.

REFERENCES

Yasin Abbasi-Yadkori, D´avid P´al, and Csaba Szepesv´ari. Improved algorithms for linear stochastic

bandits. In Advances in Neural Information Processing Systems, pp. 2312–2320, 2011.

M Mehdi Afsar, Trafford Crump, and Behrouz Far. Reinforcement learning based recommender

systems: A survey. arXiv preprint arXiv:2101.06286, 2021.

Alekh Agarwal, Sham Kakade, Akshay Krishnamurthy, and Wen Sun. Flambe: Structural complexity
and representation learning of low rank mdps. Advances in neural information processing systems,
33:20095–20107, 2020.

Shipra Agrawal and Randy Jia. Posterior sampling for reinforcement learning: worst-case regret

bounds. In Advances in Neural Information Processing Systems, pp. 1184–1194, 2017.

Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit

problem. Machine learning, 47(2):235–256, 2002.

Alex Ayoub, Zeyu Jia, Csaba Szepesvari, Mengdi Wang, and Lin Yang. Model-based reinforcement
learning with value-targeted regression. In International Conference on Machine Learning, pp.
463–474. PMLR, 2020.

Mohammad Gheshlaghi Azar, Ian Osband, and R´emi Munos. Minimax regret bounds for reinforce-
ment learning. In Proceedings of the 34th International Conference on Machine Learning-Volume
70, pp. 263–272. JMLR. org, 2017.

Yu Bai, Tengyang Xie, Nan Jiang, and Yu-Xiang Wang. Provably efficient q-learning with low

switching cost. Advances in Neural Information Processing Systems, 32, 2019.

Ronen I Brafman and Moshe Tennenholtz. R-max-a general polynomial time algorithm for near-
optimal reinforcement learning. Journal of Machine Learning Research, 3(Oct):213–231, 2002.

Nicolo Cesa-Bianchi, Ofer Dekel, and Ohad Shamir. Online learning with switching costs and other
adaptive adversaries. In Advances in Neural Information Processing Systems, pp. 1160–1168,
2013.

Xiaoyu Chen, Jiachen Hu, Lin F Yang, and Liwei Wang. Near-optimal reward-free exploration for

linear mixture mdps with plug-in solver. arXiv preprint arXiv:2110.03244, 2021.

Christoph Dann, Lihong Li, Wei Wei, and Emma Brunskill. Policy certificates: Towards accountable
reinforcement learning. In International Conference on Machine Learning, pp. 1507–1516. PMLR,
2019.

Hossein Esfandiari, Amin Karbasi, Abbas Mehrabian, and Vahab Mirrokni. Regret bounds for
batched bandits. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp.
7340–7348, 2021.

Minbo Gao, Tianle Xie, Simon S Du, and Lin F Yang. A provably efficient algorithm for linear

markov decision process with low switching cost. arXiv preprint arXiv:2101.00494, 2021.

Zijun Gao, Yanjun Han, Zhimei Ren, and Zhengqing Zhou. Batched multi-armed bandits problem.

Advances in Neural Information Processing Systems, 32, 2019.

Yanjun Han, Zhengqing Zhou, Zhengyuan Zhou, Jose Blanchet, Peter W Glynn, and Yinyu Ye. Se-
quential batch learning in finite-action linear contextual bandits. arXiv preprint arXiv:2004.06321,
2020.

10

Published as a conference paper at ICLR 2023

Pihe Hu, Yu Chen, and Longbo Huang. Nearly minimax optimal reinforcement learning with linear
function approximation. In International Conference on Machine Learning, pp. 8971–9019. PMLR,
2022.

Jiawei Huang, Jinglin Chen, Li Zhao, Tao Qin, Nan Jiang, and Tie-Yan Liu. Towards deployment-
efficient reinforcement learning: Lower bound and optimality. In International Conference on
Learning Representations, 2022.

Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement

learning. Journal of Machine Learning Research, 11(4), 2010.

Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I Jordan. Is q-learning provably efficient?

In Advances in Neural Information Processing Systems, pp. 4863–4873, 2018.

Chi Jin, Akshay Krishnamurthy, Max Simchowitz, and Tiancheng Yu. Reward-free exploration for
reinforcement learning. In International Conference on Machine Learning, pp. 4870–4879. PMLR,
2020a.

Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement
learning with linear function approximation. In Conference on Learning Theory, pp. 2137–2143.
PMLR, 2020b.

Chi Jin, Qinghua Liu, and Sobhan Miryoosefi. Bellman eluder dimension: New rich classes of rl
problems, and sample-efficient algorithms. Advances in neural information processing systems,
34:13406–13418, 2021.

Emilie Kaufmann, Pierre M´enard, Omar Darwiche Domingues, Anders Jonsson, Edouard Leurent,
and Michal Valko. Adaptive reward-free exploration. In Algorithmic Learning Theory, pp. 865–891.
PMLR, 2021.

Michael Kearns and Satinder Singh. Near-optimal reinforcement learning in polynomial time.

Machine learning, 49(2-3):209–232, 2002.

Jack Kiefer and Jacob Wolfowitz. The equivalence of two extremum problems. Canadian Journal of

Mathematics, 12:363–366, 1960.

Jens Kober, J Andrew Bagnell, and Jan Peters. Reinforcement learning in robotics: A survey. The

International Journal of Robotics Research, 32(11):1238–1274, 2013.

Tor Lattimore and Csaba Szepesv´ari. Bandit algorithms. Cambridge University Press, 2020.

Tatsuya Matsushima, Hiroki Furuta, Yutaka Matsuo, Ofir Nachum, and Shixiang Gu. Deployment-
efficient reinforcement learning via model-based offline optimization. In International Conference
on Learning Representations, 2020.

Pierre M´enard, Omar Darwiche Domingues, Anders Jonsson, Emilie Kaufmann, Edouard Leurent,
and Michal Valko. Fast active learning for pure exploration in reinforcement learning. In Interna-
tional Conference on Machine Learning, pp. 7599–7608. PMLR, 2021.

Yifei Min, Tianhao Wang, Dongruo Zhou, and Quanquan Gu. Variance-aware off-policy evaluation
with linear function approximation. Advances in neural information processing systems, 34:
7598–7610, 2021.

Ian Osband, Daniel Russo, and Benjamin Van Roy. (more) efficient reinforcement learning via

posterior sampling. Advances in Neural Information Processing Systems, 26, 2013.

Vianney Perchet, Philippe Rigollet, Sylvain Chassang, and Erik Snowberg. Batched bandit problems.

The Annals of Statistics, 44(2):660–681, 2016.

Dan Qiao, Ming Yin, Ming Min, and Yu-Xiang Wang. Sample-efficient reinforcement learning with
loglog(T) switching cost. In International Conference on Machine Learning, pp. 18031–18061.
PMLR, 2022.

11

Published as a conference paper at ICLR 2023

Yufei Ruan, Jiaqi Yang, and Yuan Zhou. Linear bandits with limited adaptivity and learning
distributional optimal design. In Proceedings of the 53rd Annual ACM SIGACT Symposium on
Theory of Computing, pp. 74–87, 2021.

David Simchi-Levi and Yunzong Xu. Phase transitions and cyclic phenomena in bandits with

switching constraints. Advances in Neural Information Processing Systems, 32, 2019.

Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction, volume 1. MIT

press Cambridge, 1998.

Andrew Wagenmaker and Kevin Jamieson. Instance-dependent near-optimal policy identification in

linear mdps via online experiment design. arXiv preprint arXiv:2207.02575, 2022.

Andrew J Wagenmaker, Yifang Chen, Max Simchowitz, Simon Du, and Kevin Jamieson. First-order
regret in reinforcement learning with linear function approximation: A robust estimation approach.
In International Conference on Machine Learning, pp. 22384–22429. PMLR, 2022a.

Andrew J Wagenmaker, Yifang Chen, Max Simchowitz, Simon Du, and Kevin Jamieson. Reward-
free rl is no harder than reward-aware rl in linear markov decision processes. In International
Conference on Machine Learning, pp. 22430–22456. PMLR, 2022b.

Ruosong Wang, Simon S Du, Lin Yang, and Russ R Salakhutdinov. On reward-free reinforcement
learning with linear function approximation. Advances in neural information processing systems,
33:17816–17826, 2020.

Tianhao Wang, Dongruo Zhou, and Quanquan Gu. Provably efficient reinforcement learning with
linear function approximation under adaptivity constraints. Advances in Neural Information
Processing Systems, 34, 2021.

Lin Yang and Mengdi Wang. Sample-optimal parametric q-learning using linearly additive features.

In International Conference on Machine Learning, pp. 6995–7004. PMLR, 2019.

Chao Yu, Jiming Liu, Shamim Nemati, and Guosheng Yin. Reinforcement learning in healthcare: A

survey. ACM Computing Surveys (CSUR), 55(1):1–36, 2021.

Andrea Zanette and Emma Brunskill. Tighter problem-dependent regret bounds in reinforcement
learning without domain knowledge using value function bounds. In International Conference on
Machine Learning, pp. 7304–7312. PMLR, 2019.

Andrea Zanette, Alessandro Lazaric, Mykel Kochenderfer, and Emma Brunskill. Learning near
optimal policies with low inherent bellman error. In International Conference on Machine Learning,
pp. 10978–10989. PMLR, 2020a.

Andrea Zanette, Alessandro Lazaric, Mykel J Kochenderfer, and Emma Brunskill. Provably efficient
reward-agnostic navigation with linear value iteration. Advances in Neural Information Processing
Systems, 33:11756–11766, 2020b.

Weitong Zhang, Dongruo Zhou, and Quanquan Gu. Reward-free model-based reinforcement learning
with linear function approximation. Advances in Neural Information Processing Systems, 34:
1582–1593, 2021a.

Xuezhou Zhang, Adish Singla, et al. Task-agnostic exploration in reinforcement learning. Advances

in Neural Information Processing Systems, 2020a.

Zihan Zhang, Simon S Du, and Xiangyang Ji. Nearly minimax optimal reward-free reinforcement

learning. arXiv preprint arXiv:2010.05901, 2020b.

Zihan Zhang, Yuan Zhou, and Xiangyang Ji. Almost optimal model-free reinforcement learning
via reference-advantage decomposition. Advances in Neural Information Processing Systems, 33:
15198–15207, 2020c.

Zihan Zhang, Jiaqi Yang, Xiangyang Ji, and Simon S Du. Variance-aware confidence set: Variance-
dependent bound for linear bandits and horizon-free bound for linear mixture mdp. arXiv preprint
arXiv:2101.12745, 2021b.

12

Published as a conference paper at ICLR 2023

Zihan Zhang, Yuhang Jiang, Yuan Zhou, and Xiangyang Ji. Near-optimal regret bounds for multi-

batch reinforcement learning. arXiv preprint arXiv:2210.08238, 2022.

Dongruo Zhou, Quanquan Gu, and Csaba Szepesvari. Nearly minimax optimal reinforcement learning
for linear mixture markov decision processes. In Conference on Learning Theory, pp. 4532–4576.
PMLR, 2021.

Zhenpeng Zhou, Steven Kearnes, Li Li, Richard N Zare, and Patrick Riley. Optimization of molecules

via deep reinforcement learning. Scientific reports, 9(1):1–10, 2019.

13

Published as a conference paper at ICLR 2023

A EXTENDED RELATED WORKS

√

√

Low regret reinforcement learning algorithms. Regret minimization under tabular MDP has
been extensively studied by a long line of works (Brafman & Tennenholtz, 2002; Kearns & Singh,
2002; Jaksch et al., 2010; Osband et al., 2013; Agrawal & Jia, 2017; Jin et al., 2018). Among those
optimal results, Azar et al. (2017) achieved the optimal regret bound (cid:101)O(
HSAT ) for stationary
MDP through model-based algorithm, while Zhang et al. (2020c) applied Q-learning type algorithm
H 2SAT ) regret under non-stationary MDP. Dann et al. (2019) provided
to achieve the optimal (cid:101)O(
policy certificates in addition to stating optimal regret bound. Different from these minimax optimal
algorithms, Zanette & Brunskill (2019) derived problem-dependent regret bound, which can imply
minimax regret bound. Another line of works studied regret minimization under linear MDP. Yang
& Wang (2019) developed the first efficient algorithm for linear MDP with simulator. Jin et al.
d3H 3T ). Later, Zanette et al. (2020a)
(2020b) applied LSVI-UCB to achieve the regret bound of (cid:101)O(
d2H 3T ) at the cost of computation. Recently, Hu et al. (2022) first
improved the regret bound to (cid:101)O(
d2H 2T ) via a computationally efficient algorithm. There
reached the minimax optimal regret (cid:101)O(
are some other works studying the linear mixture MDP setting (Ayoub et al., 2020; Zhou et al., 2021;
Zhang et al., 2021b) or more general settings like MDP with low Bellman Eluder dimension (Jin
et al., 2021).

√

√

√

Reward-free exploration. Jin et al. (2020a) first studied the problem of reward-free exploration, they
designed an algorithm while using EULER (Zanette & Brunskill, 2019) for exploration and arrived
at the sample complexity of (cid:101)O(S2AH 5/ϵ2). This sample complexity was improved by Kaufmann
et al. (2021) to (cid:101)O(S2AH 4/ϵ2) by building upper confidence bound for any reward function and
any policy. Finally, minimax optimal result (cid:101)O(S2AH 3/ϵ2) was derived in M´enard et al. (2021) by
constructing a novel exploration bonus. At the same time, a more general optimal result was achieved
by Zhang et al. (2020b) who considered MDP with stationary transition kernel and uniformly bounded
reward. Zhang et al. (2020a) studied a similar setting named task-agnostic exploration and designed
an algorithm that can find ϵ-optimal policies for N arbitrary tasks after at most (cid:101)O(SAH 5 log N/ϵ2)
episodes. For linear MDP setting, Wang et al. (2020) generalized LSVI-UCB and arrived at the
sample complexity of (cid:101)O(d3H 6/ϵ2). The sample complexity was improved by Zanette et al. (2020b)
to (cid:101)O(d3H 5/ϵ2) through approximating G-optimal design. Recently, Wagenmaker et al. (2022b) did
exploration through applying first-order regret algorithm (Wagenmaker et al., 2022a) and achieved
sample complexity bound of (cid:101)O(d2H 5/ϵ2), which matches their lower bound Ω(d2H 2/ϵ2) up to H
factors. There are other reward-free works under linear mixture MDP (Chen et al., 2021; Zhang
et al., 2021a). Meanwhile, there is a new setting that aims to do reward-free exploration under low
adaptivity and Huang et al. (2022); Qiao et al. (2022) designed provably efficient algorithms for linear
MDP and tabular MDP, respectively.

k (s) ̸= πh

switch = (cid:80)K−1

k=1 |{(h, s) ∈ [H] × S : πh

Low switching algorithms for bandits and RL. There are two kinds of switching costs. Global
switching cost simply measures the number of policy switches, while local switching cost is defined
(only under tabular MDP) as N local
k+1(s)}| where K
is the number of episodes. For multi-armed bandits with A arms and T episodes, Cesa-Bianchi et al.
√
AT ) regret with only O(A log log T ) policy switches. Simchi-
(2013) first achieved the optimal (cid:101)O(
Levi & Xu (2019) generalized the result by showing that to get optimal (cid:101)O(
T ) regret bound, both
the switching cost upper and lower bounds are of order A log log T . Under stochastic linear bandits,
Abbasi-Yadkori et al. (2011) applied doubling trick to achieve the optimal regret (cid:101)O(d
T ) with
O(d log T ) policy switches. Under slightly different setting, Ruan et al. (2021) improved the result
by improving the switching cost to O(log log T ) without worsening the regret bound. Under tabular
H 3SAT )
MDP, Bai et al. (2019) applied doubling trick to Q-learning and reached regret bound (cid:101)O(
with local switching cost O(H 3SA log T ). Zhang et al. (2020c) applied advantage decomposition
H 2SAT ) and O(H 2SA log T ),
to improve the regret bound and local switching cost bound to (cid:101)O(
respectively. Recently, Qiao et al. (2022) showed that to achieve the optimal (cid:101)O(
T ) regret, both the
global switching cost upper and lower bounds are of order HSA log log T . Under linear MDP, Gao
d3H 3T ) while
et al. (2021) applied doubling trick to LSVI-UCB and arrived at regret bound (cid:101)O(
global switching cost is O(dH log T ). This result is generalized by Wang et al. (2021) to work for

√

√

√

√

√

√

14

Published as a conference paper at ICLR 2023

arbitrary switching cost budget. Huang et al. (2022) managed to do pure exploration under linear
MDP within O(dH) switches.

Batched bandits and RL. In batched bandits problems, the agent decides a sequence of arms
and observes the reward of each arm after all arms in that sequence are pulled. More formally, at
the beginning of each batch, the agent decides a list of arms to be pulled. Afterwards, a list of
(arm,reward) pairs is given to the agent. Then the agent decides about the next batch (Esfandiari et al.,
2021). The batch sizes could be chosen non-adaptively or adaptively. In a non-adaptive algorithm,
the batch sizes should be decided before the algorithm starts, while in an adaptive algorithm, the
batch sizes may depend on the previous observations. Under multi-armed bandits with A arms and T
AT ) regret using O(log log T )
episodes, Cesa-Bianchi et al. (2013) designed an algorithm with (cid:101)O(
batches. Perchet et al. (2016) proved a regret lower bound of Ω(T
1−21−M ) for algorithms within
M batches under 2-armed bandits setting, which means Ω(log log T ) batches are necessary for a
T ). The result is generalized to K-armed bandits by Gao et al. (2019). Under
regret bound of (cid:101)O(
T )
stochastic linear bandits, Han et al. (2020) designed an algorithm that has regret bound (cid:101)O(
while running in O(log log T ) batches. Ruan et al. (2021) improved this result by using weaker
assumptions. For batched RL setting, Qiao et al. (2022) showed that their algorithm uses the optimal
O(H + log log T ) batches to achieve the optimal (cid:101)O(
T ) regret. Recently, the regret bound and
computational efficiency is improved by Zhang et al. (2022) through incorporating the idea of optimal
experimental design. The deployment efficient algorithms for pure exploration by Huang et al. (2022)
also satisfy the definition of batched RL.

√

√

√

√

1

B GENERALIZATION OF G-OPTIMAL DESIGN

Traditional G-optimal design. We first briefly introduce the problem setup of G-optimal design.
Assume there is some (possibly infinite) set A ⊆ Rd, let π : A → [0, 1] be a distribution on A so that
(cid:80)

a∈A π(a) = 1. V (π) ∈ Rd×d and g(π) ∈ R are given by

V (π) =

(cid:88)

a∈A

π(a)aa⊤,

g(π) = max
a∈A

∥a∥2

V (π)−1.

The problem of finding a design π that minimises g(π) is called the G-optimal design problem.
G-optimal design has wide application in regression problems and it can solve the linear bandit
problem (Lattimore & Szepesv´ari, 2020). However, traditional G-optimal design can not tackle our
problem under linear MDP where we can only choose π instead of choosing the feature vector ϕ
directly.

In this section, we generalize the well-known G-optimal design for our purpose under linear MDP.
Consider the following problem: Under some fixed linear MDP, given a fixed finite policy set Π, we
want to select a policy π0 from ∆(Π) (distribution over policy set Π) to minimize the following term:

max
π∈Π

Eπϕ(sh, ah)⊤(Eπ0ϕhϕ⊤

h )−1ϕ(sh, ah),

(2)

where the sh, ah follows the distribution according to π and the ϕh follows the distribution of policy
π0. We first consider its two special cases.

Special case 1. If the MDP is deterministic, then given any fixed deterministic policy π, the trajectory
generated from this π is deterministic. Therefore the feature ϕh at layer h is also deterministic. We
denote the feature at layer h from running policy π by ϕπ,h. In this case, the previous problem (2)
reduces to

min
π0∈∆(Π)

max
π∈Π

π,h(Eπ0ϕhϕ⊤
ϕ⊤

h )−1ϕπ,h,

(3)

which can be characterized by the traditional G-optimal design, for more details please refer to Kiefer
& Wolfowitz (1960) and chapter 21 of Lattimore & Szepesv´ari (2020). According to Theorem 21.1 of
Lattimore & Szepesv´ari (2020), the minimization of (3) can be bounded by d, which is the dimension
of the feature map ϕ.

Special case 2. When the linear MDP is actually a tabular MDP with finite state set |S| = S and finite
action set |A| = A, the feature map reduces to canonical basis in Rd = RSA with ϕ(s, a) = e(s,a)

15

Published as a conference paper at ICLR 2023

(Jin et al., 2020b). Let dπ
previous optimization problem (2) reduces to

h(s, a) = Pπ(sh = s, ah = a) denote the occupancy measure, then the

min
π0∈∆(Π)

max
π∈Π

(cid:88)

(s,a)∈S×A

dπ
h(s, a)
dπ0
h (s, a)

.

(4)

Such minimization problem corresponds to finding a policy π0 that can cover all policies from the
policy set Π. According to Lemma 1 in Zhang et al. (2022) (we only use the case where m = 1), the
minimization of (4) can be bounded by d = SA.

Different from these two special cases, under our problem setup (general linear MDP), the feature map
can be much more complex than canonical basis and running each π will lead to a distribution over the
feature map space rather than a fixed single feature. Next, we formalize the problem setup and present
the theorem. We are given finite policy set Π and finite action set Φ (we only consider finite action
set, the general case can be proven similarly by passing to the limit (Lattimore & Szepesv´ari, 2020)),
where each π ∈ Π is a distribution over Φ (with π(a) denoting the probability of choosing action a)
and each action a ∈ Φ is a vector in Rd. In addition, µ can be any distribution over Π. In the following
part, we characterize µ as a vector in R|Π| with µ(π) denoting the probability of choosing policy π.
Let Λ(π) = (cid:80)
a∈Φ π(a)aa⊤. The
function we want to minimize is g(µ) = maxπ∈Π
Theorem B.1. Define the set (cid:98)Φ = {a ∈ Φ : ∃ π ∈ Π, π(a) > 0}. If span((cid:98)Φ) = Rd, there exists a
distribution µ⋆ over Π such that g(µ⋆) ≤ d.

a∈Φ π(a)aa⊤ and V (µ) = (cid:80)

π∈Π µ(π)Λ(π) = (cid:80)

a∈Φ π(a)a⊤V (µ)−1a.

π∈Π µ(π) (cid:80)

(cid:80)

Proof of Theorem B.1. Define f (µ) = log det V (µ) and take µ⋆ to be

µ⋆ = arg max

µ

f (µ).

According to Exercise 21.2 of Lattimore & Szepesv´ari (2020), f is concave. Besides, according to
Exercise 21.1 of Lattimore & Szepesv´ari (2020), we have

d
dt

log det(A(t)) =

1
det(A(t))

T r(adj(A)

d
dt

A(t)) = T r(A−1 d
dt

A(t)).

Plugging f in, we directly have:

(▽f (µ))π = T r(V (µ)−1Λ(π)) =

π(a)a⊤V (µ)−1a.

(cid:88)

a∈Φ

In addition, by direct calculation, for any feasible µ,
(cid:88)

(cid:88)

(cid:88)

µ(π)(▽f (µ))π = T r(

µ(π)

π(a)aa⊤V (µ)−1) = T r(Id) = d.

π∈Π

π∈Π

a∈Φ

Since µ⋆ is the maximizer of f , by first order optimality criterion, for any feasible µ,

0 ≥⟨▽f (µ⋆), µ − µ⋆⟩

=

=

(cid:88)

π∈Π
(cid:88)

π∈Π

µ(π)

µ(π)

(cid:88)

a∈Φ
(cid:88)

a∈Φ

π(a)a⊤V (µ⋆)−1a −

µ⋆(π)

(cid:88)

π∈Π

(cid:88)

a∈Φ

π(a)a⊤V (µ⋆)−1a

π(a)a⊤V (µ⋆)−1a − d.

For any π ∈ Π, we can choose µ to be Dirac at π, which proves that for any π ∈ Π,
(cid:80)

a∈Φ π(a)a⊤V (µ⋆)−1a ≤ d. Due to the definition of g(µ⋆), we have g(µ⋆) ≤ d.

Remark B.2. By replacing the action set Φ with the set of all feasible features at layer h, Theorem
B.1 shows that for any linear MDP and fixed policy set Π,

min
π0∈∆(Π)

max
π∈Π

Eπϕ(sh, ah)⊤(Eπ0ϕhϕ⊤

h )−1ϕ(sh, ah) ≤ d.

(5)

This theorem serves as one of the critical theoretical bases for our analysis.

16

Published as a conference paper at ICLR 2023

Remark B.3. Although the proof is similar to Theorem 21.1 of Lattimore & Szepesv´ari (2020),
our Theorem B.1 is more general since it also holds under the case where each π will generate a
distribution over the action space. In contrast, G-optimal design is a special case of our setting where
each π will generate a fixed action from the action space.

Knowing the existence of such covering policy, the next lemma provides some properties of the
solution of (2) under some additional assumption.
Lemma B.4. Let π⋆ = arg minπ0∈∆(Π) maxπ∈Π Eπϕ(sh, ah)⊤(Eπ0 ϕhϕ⊤
that supπ∈∆(Π) λmin(Eπϕhϕ⊤

h )−1ϕ(sh, ah). Assume

h ) ≥ λ⋆, then it holds that

λmin(Eπ⋆ ϕhϕ⊤

h ) ≥

λ⋆
d

,

(6)

where d is the dimension of ϕ and λmin denotes the minimum eigenvalue.

Before we state the proof, we provide the description of the special case where the MDP is a tabular
MDP. The condition implies that there exists some policy (cid:101)π ∈ ∆(Π) such that for any s, a ∈ S × A,
h(s, a) ≥ λ⋆, where dπ
d(cid:101)π

h(·, ·) is occupancy measure. Due to Theorem B.1, π⋆ satisfies that

max
π∈Π

(cid:88)

(s,a)∈S×A

dπ
h(s, a)
dπ⋆
h (s, a)

≤ SA.

For any (s, a) ∈ S × A, choose πs,a = arg maxπ∈Π dπ
Therefore, it holds that dπ⋆

h(s, a) and dπs,a
SA for any s, a, which is equivalent to the conclusion of (6).

h (s, a) ≥ λ⋆

(s, a) ≥ d(cid:101)π

h(s, a) ≥ λ⋆.

h

h )−1) > d

Proof of Lemma B.4. If the conclusion (6) does not hold, we have λmin(Eπ⋆ ϕhϕ⊤
d , which
implies that λmax((Eπ⋆ ϕhϕ⊤
h )−1 by 0 < λ1 ≤
λ2 ≤ · · · ≤ λd. There exists a set of orthogonal and normalized vectors { ¯ϕi}i∈[d] such that ¯ϕi is a
corresponding eigenvector of λi.
According to the condition, there exists (cid:101)π ∈ ∆(Π) such that λmin(E
ϕ ∈ Rd with ∥ϕ∥2 = 1, ϕ⊤(E
h )ϕ = E
(cid:101)πϕhϕ⊤
where Eπ⋆ is short for Eπ⋆ ϕhϕ⊤
h . It holds that:

h ) ≥ λ⋆. Therefore, for any
h (Eπ⋆ )−1ϕh,
(cid:101)πϕ⊤

λ⋆ . Denote the eigenvalues of (Eπ⋆ ϕhϕ⊤

h ϕ)2 ≥ λ⋆. Now we consider E

(cid:101)πϕhϕ⊤

(cid:101)π(ϕ⊤

h ) < λ⋆

E
(cid:101)πϕ⊤

h (Eπ⋆ )−1ϕh =E

(cid:101)π[

d
(cid:88)

(ϕ⊤
h

i=1

¯ϕi) ¯ϕi]⊤(Eπ⋆ )−1[

d
(cid:88)

(ϕ⊤
h

¯ϕi) ¯ϕi]

i=1

=E
(cid:101)π

d
(cid:88)

(ϕ⊤
h

i=1

¯ϕi)2 ¯ϕ⊤

i (Eπ⋆ )−1 ¯ϕi

≥E

d (Eπ⋆ )−1 ¯ϕd

¯ϕd)2 ¯ϕ⊤
(cid:101)π(ϕ⊤
h
d
>λ⋆ ×
λ⋆ = d,
where the first equation is due to the fact that { ¯ϕi}i∈[d] forms a set of normalized basis. The second
equation results from the definition of eigenvectors. The last inequality is because our assumption
(λmax((Eπ⋆ ϕhϕ⊤
Finally, since this leads to contradiction with Theorem B.1, the proof is complete.

λ⋆ ) and condition (∀ ∥ϕ∥2 = 1, ϕ⊤(E

h ϕ)2 ≥ λ⋆).

h )−1) > d

h )ϕ = E

(cid:101)πϕhϕ⊤

(cid:101)π(ϕ⊤

C CONSTRUCTION OF POLICY SETS

In this section, we construct policy sets given the feature map ϕ(·, ·). We begin with several technical
lemmas.

17

Published as a conference paper at ICLR 2023

C.1 TECHNICAL LEMMAS

Lemma C.1 (Covering Number of Euclidean Ball (Jin et al., 2020b)). For any ϵ > 0, the ϵ-covering
number of the Euclidean ball in Rd with radius R > 0 is upper bounded by (1 + 2R
Lemma C.2 (Lemma B.1 of Jin et al. (2020b)). Let wπ
⟨ϕ(s, a), wπ
Lemma C.3 (Advantage Decomposition). For any MDP with fixed initial state s1, for any policy π,
it holds that

h denote the set of weights such that Qπ

h ⟩. Then ∥wπ

h ∥2 ≤ 2H

h(s, a) =

ϵ )d.

√

d.

1 (s1) − V π
V ⋆

1 (s1) = Eπ

H
(cid:88)

[V ⋆

h (sh) − Q⋆

h(sh, ah)],

here the expectation means that sh, ah follows the distribution generated by π.

h=1

Proof of Lemma C.3.

1 (s1) − V π
V ⋆

1 (s1) =Eπ[V ⋆
=Eπ[V ⋆

1 (s1) − Q⋆
1 (s1) − Q⋆

1(s1, a1)] + Eπ[Q⋆
1(s1, a1)] + Es1,a1∼π[

1(s1, a1) − Qπ
(cid:88)

1 (s1, a1)]
P1(s′|s1, a1)(V ⋆

2 (s′) − V π

2 (s′))]

1 (s1) − Q⋆

1(s1, a1)] + Es2∼π[V ⋆

s′∈S
2 (s2) − V π

2 (s2)]

=Eπ[V ⋆
= · · ·

H
(cid:88)

=Eπ

[V ⋆

h (sh) − Q⋆

h(sh, ah)],

h=1

where the second equation is because of Bellman Equation and the forth equation results from
applying the decomposition recursively from h = 1 to H.

Lemma C.4 (Elliptical Potential Lemma, Lemma 26 of Agarwal et al. (2020)). Consider a sequence
of d × d positive semi-definite matrices X1, · · · , XT with maxt T r(Xt) ≤ 1 and define M0 =
I, · · · , Mt = Mt−1 + Xt. Then

T
(cid:88)

t=1

T r(XtM −1

t−1) ≤ 2d log(1 +

T
d

).

C.2 CONSTRUCTION OF POLICIES TO EVALUATE

We construct the policy set Πeval given feature map ϕ(·, ·). The policy set Πeval satisfies that for any
feasible linear MDP with feature map ϕ, Πeval contains one near-optimal policy of this linear MDP.
We begin with the construction.

√

√

Construction of Πeval. Given ϵ > 0, let W be a ϵ
d) := {x ∈
d}. Next, we construct the Q-function set Q = { ¯Q(s, a) = ϕ(s, a)⊤w : w ∈ W}.
Rd : ∥x∥2 ≤ 2H
Then the policy set at layer h is defined as ∀ h ∈ [H], Πh = {π(s) = arg maxa∈A ¯Q(s, a)| ¯Q ∈ Q},
with ties broken arbitrarily. Finally, the policy set Πeval
Lemma C.5. The policy set Πeval

is Πeval
satisfies that for any h ∈ [H],

2H -cover of the Euclidean ball Bd(2H

ϵ = Π1 × Π2 × · · · × ΠH .

ϵ

ϵ

log |Πh| ≤ d log(1 +

√

d

8H 2
ϵ

) = (cid:101)O(d).

(7)

In addition, for any linear MDP with feature map ϕ(·, ·), there exists π = (π1, π2, · · · , πH ) such that
πh ∈ Πh for all h ∈ [H] and V π ≥ V ⋆ − ϵ.

Proof of Lemma C.5. Since W is a ϵ

2H -covering of Euclidean ball, by Lemma C.1 we have

log |W| ≤ d log(1 +

√

d

).

8H 2
ϵ

18

Published as a conference paper at ICLR 2023

In addition, for any w in W, there is at most one corresponding Q ∈ Q and one πh ∈ Πh. Therefore,
it holds that for any h ∈ [H],

log |Πh| ≤ log |Q| ≤ log |W| ≤ d log(1 +

√

d

).

8H 2
ϵ

For any linear MDP, according to Lemma C.2, the optimal Q-function can be written as:

√

Q⋆

h(s, a) = ⟨ϕ(s, a), w⋆

h⟩,

h∥2 ≤ 2H

with ∥w⋆
h∥2 ≤ ϵ
¯wh ∈ W such that ∥ ¯wh − w⋆
arg maxa∈A ¯Qh(s, a) from Πh. Note that for any h, s, a ∈ [H] × S × A,

2H -covering of the Euclidean ball, for any h ∈ [H] there exists
2H . Select ¯Qh(s, a) = ϕ(s, a)⊤ ¯wh from Q and πh(s) =

d. Since W is ϵ

|Q⋆

h(s, a) − ¯Qh(s, a)| ≤ ∥ϕ(s, a)∥2 · ∥w⋆

h − ¯wh∥2 ≤

ϵ
2H

.

(8)

Let π = (π1, π2, · · · , πH ), now we prove that this π is ϵ-optimal.
Denote the optimal policy under this linear MDP by π⋆, then we have for any s, h ∈ S × [H],

Q⋆
=[Q⋆
ϵ
2H

≤

h(s, πh(s))

h(s, π⋆
h(s, π⋆
+ 0 +

h(s)) − Q⋆
h(s)) − ¯Qh(s, π⋆
=

,

ϵ
2H

ϵ
H

h(s))] + [ ¯Qh(s, π⋆

h(s)) − ¯Qh(s, πh(s))] + [ ¯Qh(s, πh(s)) − Q⋆

h(s, πh(s))]

(9)

where the inequality results from the definition of πh and (8).

Now we apply the advantage decomposition (Lemma C.3), it holds that:

1 (s1) − V π
V ⋆

1 (s1) =Eπ

H
(cid:88)

[V ⋆

h (sh) − Q⋆

h(sh, ah)]

where the inequality comes from (9).

≤H ·

h=1
ϵ
H

= ϵ,

Remark C.6. Our concurrent work Wagenmaker & Jamieson (2022) also applies the idea of policy
discretization. However, to cover ϵ-optimal policies of all linear MDPs, the size of their policy set is
log |Πϵ| ≤ (cid:101)O(dH 2 · log 1
ϵ ) (stated in Corollary 1 of Wagenmaker & Jamieson (2022)). In comparison,
| ≤ H log |Π1| ≤ (cid:101)O(dH · log 1
our Πeval
ϵ ), which improves their results by a
factor of H. Such improvement is done by applying advantage decomposition. Finally, by plugging
in our Πeval
into Corollary 2 of Wagenmaker & Jamieson (2022), we can directly improve their
worst-case bound by a factor of H.

satisfies that log |Πeval

ϵ

ϵ

ϵ

C.3 CONSTRUCTION OF EXPLORATIVE POLICIES

Given the feature map ϕ(·, ·) and the condition that for any h ∈ [H], supπ λmin(Eπϕhϕ⊤
h ) ≥ λ⋆
where π can be any policy, we construct a finite policy set Πexp that covers explorative policies
under any feasible linear MDPs. Such exploratory is formalized as for any linear MDP and h ∈ [H],
there exists some policy π in ∆(Πexp) such that λmin(Eπϕhϕ⊤
h ) is large enough. We begin with the
construction.
Construction of Πexp. Given ϵ > 0, consider all reward functions that can be represented as

r(s, a) =

(cid:113)

ϕ(s, a)⊤(I + Σ)−1ϕ(s, a),

(10)

where Σ is positive semi-definite. According to Lemma D.6 of Jin et al. (2020b), we can construct
a ϵ
2H -cover Rϵ of all such reward functions while the size of Rϵ satisfies log |Rϵ| ≤ d2 log(1 +
32H 2
ϵ2

).

√

d

19

Published as a conference paper at ICLR 2023

For all h ∈ [H], denote Π1
Meanwhile, denote the policy set Πh (w.r.t ϵ) in the previous Section C.2 by Π2
Πh,ϵ = Π1
sets, Πexp

h,ϵ = {π(s) = arg maxa∈A r(s, a)|r ∈ Rϵ} with ties broken arbitrarily.
h,ϵ. Finally, let
h,ϵ be the policy set for layer h. The whole policy set is the product of these h policy

ϵ = Π1,ϵ × · · · × ΠH,ϵ.

h,ϵ ∪ Π2

Lemma C.7. For any ϵ > 0, we have Πeval
For any reward r that is the form of (10) and h ∈ [H], there exists a policy ¯π ∈ Πexp
E¯πr(sh, ah) ≥ sup
π

. In addition, log |Πh,ϵ| ≤ 2d2 log(1 + 32H 2
ϵ2
such that

Eπr(sh, ah) − ϵ.

ϵ ⊆ Πexp

ϵ

ϵ

√

d

).

Proof of Lemma C.7. The conclusion that Πeval
Π1

ϵ

h,ϵ ∪ Π2
h,ϵ.
In addition,

log |Πh,ϵ| ≤ log |Π1

h,ϵ| + log |Π2

h,ϵ| ≤ log |Rϵ| + d log(1 +

⊆ Πexp
ϵ

is because of our construction: Πh,ϵ =

√

d

8H 2
ϵ

) ≤ 2d2 log(1 +

√

d

).

32H 2
ϵ2

Consider the optimal Q-function under reward function r(sh, ah) (reward is always 0 at other layers).
We have Q⋆

h(s, a) = r(s, a) and for i ≤ h − 1,
(cid:88)
Q⋆

i (s, a) =0 +

⟨ϕ(s, a), µi(s′)⟩V ⋆

i+1(s′)

s′∈S

=⟨ϕ(s, a),

(cid:88)

µi(s′)V ⋆

i+1(s′)⟩

s′∈S
=⟨ϕ(s, a), w⋆
i ⟩,
√

i ∥2 ≤ 2

d. The first equation is because of Bellman Equation and our

i ∈ Rd with ∥w⋆

for some w⋆
design of reward function.
Since Q⋆
up to ϵ

h is covered by Rϵ up to ϵ

2H accuracy while Q⋆
2H accuracy, with identical proof to Lemma C.5, the last conclusion holds.

i (i ≤ h − 1) is covered by Q in section C.2

Lemma C.8. Assume supπ λmin(Eπϕhϕ⊤

h ) ≥ λ⋆, if ϵ ≤ λ⋆

sup
π∈∆(Πexp
ϵ

)

λmin(Eπϕhϕ⊤

h ) ≥

4 , we have
(λ⋆)2
64d log(1/λ⋆)

.

Proof of Lemma C.8. Fix t = 64d log(1/λ⋆)
π1 is arbitrary policy in Πexp
For any i ∈ [t], Σi = (cid:80)i
C.7, there exists policy πi+1 ∈ Πexp

.
Eπj ϕhϕ⊤
ϵ

(λ⋆)2

j=1

ϵ

, we construct the following policies:

h , ri(s, a) = (cid:112)ϕ(s, a)⊤(I + Σi)−1ϕ(s, a). Due to Lemma
such that Eπi+1ri(sh, ah) ≥ supπ

Eπri(sh, ah) − ϵ.

The following inequality holds:

t
(cid:88)

i=1
t
(cid:88)

≤

(cid:113)

Eπi

ϕ⊤
h (I + Σi−1)−1ϕh

(cid:113)

Eπiϕ⊤

h (I + Σi−1)−1ϕh

Eπiϕ⊤

h (I + Σi−1)−1ϕh

(11)

T r(Eπiϕhϕ⊤

h (I + Σi−1)−1)

i=1
(cid:118)
(cid:117)
(cid:117)
(cid:116)t ·

≤

(cid:118)
(cid:117)
(cid:117)
(cid:116)t ·

≤

(cid:114)

t
(cid:88)

i=1

t
(cid:88)

i=1

≤

2dt log(1 +

t
d

),

20

Published as a conference paper at ICLR 2023

where the second inequality holds because of Cauchy-Schwarz inequality and the last inequality
holds due to Lemma C.4.

(cid:113)

Eπ

(cid:113) 2d log(1+t/d)
ϕ⊤
h (I + Σt−1)−1ϕh ≤
2 because of our
t
4 and t = 64d log(1/λ⋆)
. According to Lemma E.1412 of Huang et al. (2022), we have

Therefore, we have that supπ
choice of ϵ ≤ λ⋆
that λmin(Σt−1) ≥ 1.
Finally, choose π = unif ({πi}i∈[t−1]), we have π ∈ ∆(Πexp

+ ϵ ≤ λ⋆

) and

(λ⋆)2

ϵ

λmin(Eπϕhϕ⊤

h ) ≥

(λ⋆)2
64d log(1/λ⋆)

.

C.4 A SUMMARY

Policy sets
The set of all policies
Explorative policies: Πexp

ϵ

Policies to evaluate: Πeval

ϵ

Cardinality
Infinity
ϵ,h | = (cid:101)O(d2)
ϵ,h | = (cid:101)O(d)

log |Πexp

log |Πeval

Description
The largest possible policy set
Sufficient for exploration
Uniform policy evaluation over Πeval
is sufficient for policy identification

ϵ

Relationship with each other
Contains the following two sets
Subset of all policies

Subset of Πexp

ϵ

Table 2: Comparison of different policy sets.

h

We compare the relationship between different policy sets in the Table 2 above. In summary, given
the feature map ϕ(·, ·) of linear MDP and any accuracy ϵ, we can construct policy set Πeval which
satisfies that log |Πeval
| = (cid:101)O(d). At the same time, for any linear MDP, the policy set Πeval is
guaranteed to contain one near-optimal policy. Therefore, it suffices to estimate the value functions
of all policies in Πeval accurately.
Similarly, given the feature map ϕ(·, ·) and some ϵ that is small enough compared to λ⋆, we can
construct policy set Πexp which satisfies that log |Πexp
h | = (cid:101)O(d2). At the same time, for any linear
MDP, the policy set Πexp is guaranteed to contain explorative policies for all layers, which means
that it suffices to do exploration using only policies from Πexp.

D ESTIMATION OF VALUE FUNCTIONS

According to the construction of Πeval in Section C.2 and Lemma C.5, it suffices to estimate the
value functions of policies in Πeval. In this section, we design an algorithm to estimate the value
functions of any policy in Πeval given any reward function. Recall that for accuracy ϵ0, we denote
and the policy set for layer h is denoted by Πeval
the policy set constructed in Section C.2 by Πeval
ϵ0,h .

ϵ0

12Our condition that supπ λmin(Eπϕhϕ⊤

h ) ≥ λ⋆ implies that for any u ∈ Rd with ∥u∥2 = 1,
h u)2 ≥ λ⋆. Therefore, the proof of Lemma E.14 of Huang et al. (2022) holds by plugging

maxπ Eπ(ϕ⊤
in c = 1.

21

Published as a conference paper at ICLR 2023

D.1 THE ALGORITHM

Algorithm 3 Estimation of V π(r) given exploration data (EstimateV)
1: Input: Policy to evaluate π ∈ Πeval

. Linear reward function r = {rh}h∈[H] bounded in [0, 1].

ϵ0

h}(h,n)∈[H]×[N ]. Initial state s1.

Exploration data {sn

Λh ← I + (cid:80)N
¯wh ← (Λh)−1 (cid:80)N

h, an
2: Initialization: QH+1(·, ·) ← 0, VH+1(·) ← 0.
3: for h = H, H − 1, . . . , 1 do
h, an
4:
n=1 ϕ(sn

h)⊤.
h, an
h)Vh+1(sn
5:
6: Qh(·, ·) ← (ϕ(·, ·)⊤ ¯wh + rh(·, ·))[0,H].
7:
8: end for
9: Output: V1(s1).

Vh(·) ← Qh(·, πh(·)).

h)ϕ(sn
h, an

n=1 ϕ(sn

h+1).

ϵ0

Algorithm 3 takes policy π from Πeval
and linear reward function r as input, and uses LSVI to
estimate the value function of this given policy and given reward function. From layer H to layer
1, we calculate Λh and ¯wh to estimate Qπ
h in line 6. In addition, according to our construction in
Section C.2, all policies in Πeval
are deterministic, which means we can use line 7 to approximate
V π
h . Algorithm 3 looks similar to Algorithm 2 in Wang et al. (2020). However, there are two key
differences. First, Algorithm 2 of Wang et al. (2020) aims to find near optimal policy for each reward
function while we do policy evaluation for each reward and policy. In addition, different from their
approach, we do not use optimism, which means we do not need to cover the bonus term. This is the
main reason why we can save a factor of

√

d.

ϵ0

D.2 TECHNICAL LEMMAS

Lemma D.1 (Lemma D.4 of Jin et al. (2020b)). Let {xτ }∞
S with corresponding filtration {Fτ }∞
ϕτ ∈ Fτ −1, and ∥ϕτ ∥ ≤ 1. Let Λk = I + (cid:80)k
least 1 − δ, for all k ≥ 0, and any V ∈ V so that supx |V (x)| ≤ H, we have:

τ =1 be a stochastic process on state space
τ =1 be an Rd-valued stochastic process where
τ . Then for any δ > 0, with probability at

τ =0. Let {ϕτ }∞

τ =1 ϕτ ϕ⊤

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

k
(cid:88)

τ =1

2
(cid:13)
(cid:13)
ϕτ {V (xτ ) − E[V (xτ )|Fτ −1]}
(cid:13)
(cid:13)
(cid:13)

Λ−1
k

≤ 4H 2

(cid:20) d
2

log(k + 1) + log(

(cid:21)
)

Nϵ
δ

+ 8k2ϵ2,

where Nϵ is the ϵ-covering number of V with respect to the distance dist(V, V ′) = supx |V (x) −
V ′(x)|.

Lemma D.2. The ¯wh in line 5 of Algorithm 3 is always bounded by ∥ ¯wh∥2 ≤ H

Proof of Lemma D.2. For any θ ∈ Rd with ∥θ∥2 = 1, we have

|θ⊤ ¯wh| =|θ⊤(Λh)−1

N
(cid:88)

n=1

ϕ(sn

h, an

h)Vh+1(sn

h+1)|

√

dN .

≤

N
(cid:88)

n=1

|θ⊤(Λh)−1ϕ(sn

h, an

h)| · H

(cid:118)
(cid:117)
(cid:117)
(cid:116)[

N
(cid:88)

≤H ·

√

≤H

n=1

dN .

θ⊤(Λh)−1θ] · [

N
(cid:88)

n=1

ϕ(sn

h, an

h)⊤(Λh)−1ϕ(sh, ah)]

The second inequality is because of Cauchy-Schwarz inequality. The last inequality holds according
to Lemma D.1 of Jin et al. (2020b).

22

Published as a conference paper at ICLR 2023

D.3 UPPER BOUND OF ESTIMATION ERROR

We first consider the covering number of Vh in Algorithm 3. All Vh can be written as:

[0,H] ,
where θh is the parameter with respect to rh (rh(s, a) = ⟨ϕ(s, a), θh⟩).

Vh(·) = (cid:0)ϕ(·, πh(·))⊤( ¯wh + θh)(cid:1)

Note that Πeval
the ϵ-covering number Nϵ of {Vh} is bounded by

ϵ0,h × Wϵ (where Wϵ is ϵ-cover of Bd(2H

√

log Nϵ ≤ log |Πeval

ϵ0,h | + log |Wϵ| ≤ d log(1 +

dN )) provides a ϵ-cover of {Vh}. Therefore,

√

8H 2
ϵ0

d

) + d log(1 +

4H

√

ϵ

dN

).

(13)

(12)

Now we have the following key lemma.
Lemma D.3. With probability 1 − δ, for any policy π ∈ Πeval
may appear in Algorithm 3, the {Vh}h∈[H] derived by Algorithm 3 satisfies that for any h ∈ [H],

and any linear reward function r that

ϵ0

N
(cid:88)

(cid:32)

ϕn
h

Vh+1(sn

h+1) −

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

n=1

(cid:88)

s′∈S

Ph(s′|sn

h, an

h)Vh+1(s′)

(cid:33)(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)Λ−1

h

√

≤ cH

(cid:114)

d ·

log(

Hd
ϵ0δ

) + log(

N
δ

),

for some universal constant c > 0.

Proof of Lemma D.3. The proof is by plugging ϵ = H

√
N in Lemma D.1 and using (13).

d

Remark D.4. Assume the final goal is to find ϵ-optimal policy for all reward functions, we can choose
that ϵ0 ≥ poly(ϵ) and N ≤ poly(d, H, 1
d),
which effectively saves a factor of

ϵ ). Then the R.H.S. of Lemma D.3 is of order (cid:101)O(H

d compared to Lemma A.1 of Wang et al. (2020).

√

√

Now we are ready to prove the following lemma.
Lemma D.5. With probability 1 − δ, for any policy π ∈ Πeval
and any linear reward function r that
may appear in Algorithm 3, the {Vh}h∈[H] and { ¯wh}h∈[H] derived by Algorithm 3 satisfies that for
all h, s, a ∈ [H] × S × A,

ϵ0

|ϕ(s, a)⊤ ¯wh −

(cid:88)

s′∈S

Ph(s′|s, a)Vh+1(s′)| ≤ c′H

for some universal constant c′ > 0.

√

(cid:114)

d ·

log(

Hd
ϵ0δ

) + log(

N
δ

) · ∥ϕ(s, a)∥Λ−1

h

,

This part of proof is similar to the proof of Lemma 3.1 in Wang et al. (2020). For completeness, we
state it here.

Proof of Lemma D.5. Since Ph(s′|s, a) = ϕ(s, a)⊤µh(s′), we have
Ph(s′|s, a)Vh+1(s′) = ϕ(s, a)⊤

(cid:88)

(cid:101)wh,

for some ∥ (cid:101)wh∥2 ≤ H

√

ϕ(s, a)⊤ ¯wh −

d. Therefore, we have
(cid:88)

Ph(s′|s, a)Vh+1(s′)

s′∈S

s′∈S

=ϕ(s, a)⊤(Λh)−1

N
(cid:88)

h · Vh+1(sn
ϕn

h+1) −

Ph(s′|s, a)Vh+1(s′)

(cid:88)

s′∈S

(cid:33)

=ϕ(s, a)⊤(Λh)−1

=ϕ(s, a)⊤(Λh)−1

=ϕ(s, a)⊤(Λh)−1

n=1
(cid:32) N
(cid:88)

n=1
(cid:32) N
(cid:88)

n=1
(cid:32) N
(cid:88)

n=1

h · Vh+1(sn
ϕn

h+1) − Λh (cid:101)wh

(14)

(cid:33)

h(ϕn
ϕn

h)⊤

(cid:101)wh

N
(cid:88)

n=1

(cid:33)

(cid:33)

Ph(s′|sn

h, an

h)Vh+1(s′)

− (cid:101)wh

.

hVh+1(sn
ϕn

h+1) − (cid:101)wh −

(cid:32)

ϕn
h

Vh+1(sn

h+1) −

(cid:88)

s′

23

Published as a conference paper at ICLR 2023

It holds that,

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

ϕ(s, a)⊤(Λh)−1

(cid:32)

ϕn
h

Vh+1(sn

h+1) −

(cid:32) N
(cid:88)

n=1

(cid:88)

s′

Ph(s′|sn

h, an

h)Vh+1(s′)

(cid:33)(cid:33)(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

≤∥ϕ(s, a)∥Λ−1

h

·

N
(cid:88)

(cid:32)

ϕn
h

Vh+1(sn

h+1) −

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

n=1

(cid:88)

s′∈S

Ph(s′|sn

h, an

h)Vh+1(s′)

(cid:33)(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)Λ−1

h

(15)

√

(cid:114)

≤cH

Hd
ϵ0δ
for some constant c due to Lemma D.3. In addition, we have

) · ∥ϕ(s, a)∥Λ−1

) + log(

N
δ

log(

d ·

,

h

|ϕ(s, a)⊤(Λh)−1

(cid:101)wh| ≤ ∥ϕ(s, a)∥Λ−1

h

· ∥ (cid:101)wh∥Λ−1

h

≤ H

Combining these two results, we have

√

d · ∥ϕ(s, a)∥Λ−1

h

.

|ϕ(s, a)⊤ ¯wh −

(cid:88)

s′∈S

Ph(s′|s, a)Vh+1(s′)| ≤ c′H

√

(cid:114)

d ·

log(

Hd
ϵ0δ

) + log(

N
δ

) · ∥ϕ(s, a)∥Λ−1

h

.

Finally, the error bound of our estimations are summarized in the following lemma.
Lemma D.6. For π ∈ Πeval
ϵ0
Then with probability 1 − δ, for any policy π ∈ Πeval

and linear reward function r, let the output of Algorithm 3 be (cid:98)V π(r).
and any linear reward function r, it holds that

ϵ0

| (cid:98)V π(r) − V π(r)| ≤ c′H

√

(cid:114)

d ·

log(

Hd
ϵ0δ

) + log(

N
δ

) · Eπ

H
(cid:88)

h=1

∥ϕ(sh, ah)∥Λ−1

h

,

(16)

for some universal constant c′ > 0.

Proof of Lemma D.6. For any policy π ∈ Πeval
functions and ¯wh in Algorithm 3, we have

ϵ0

and any linear reward function r, consider the Vh

|V1(s1) − V π

1 (s1)| ≤ Eπ

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

ϕ(s1, a1)⊤ ¯w1 + r1(s1, a1) −

P1(s′|s1, a1)V π

(cid:12)
(cid:12)
2 (s′) − r1(s1, a1)
(cid:12)
(cid:12)
(cid:12)

(cid:88)

s′∈S

≤Eπ

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

ϕ(s1, a1)⊤ ¯w1 −

(cid:12)
(cid:12)
P1(s′|s1, a1)V2(s′)
(cid:12)
(cid:12)
(cid:12)

(cid:88)

s′∈S

+ Eπ

(cid:88)

s′∈S

P1(s′|s1, a1) |V2(s′) − V π

2 (s′)|

≤Eπc′H

√

(cid:114)

d ·

log(

Hd
ϵ0δ

) + log(

N
δ

) · ∥ϕ(s1, a1)∥Λ−1

1

+ Eπ|V2(s2) − V π

2 (s2)|

≤ · · ·

√

≤c′H

(cid:114)

d ·

log(

Hd
ϵ0δ

) + log(

N
δ

) · Eπ

H
(cid:88)

h=1

∥ϕ(sh, ah)∥Λ−1

h

,

where the first inequality results from the fact that V π
Lemma D.5. The forth inequality is due to recursive application of decomposition.

1 (s1) ∈ [0, H]. The third inequality comes from

(17)

√

Remark D.7. Compared to the analysis in Wang et al. (2020) and Huang et al. (2022), our analysis
saves a factor of
d. This is achieved by discretization of the policy set and bypassing the need to
cover the quadratic bonus term. More specifically, the log-covering number of our Πeval
is (cid:101)O(d).
Combining with the covering set of Euclidean ball in Rd, the total log-covering number is still (cid:101)O(d).
In contrast, both previous works need to cover bonus like (cid:112)ϕ(·, ·)⊤(Λ)−1ϕ(·, ·), which requires the
log-covering number to be (cid:101)O(d2).

h

24

Published as a conference paper at ICLR 2023

E GENERALIZED ALGORITHMS FOR ESTIMATING VALUE FUNCTIONS

Since Πexp we construct in Section C.3 is guaranteed to cover explorative policies under any feasible
linear MDP, it suffices to do exploration using only policies from Πexp. In this section, we generalize
the algorithm we propose in Section D for our purpose during exploration phase. To be more specific,
we design an algorithm to estimate Eπr(sh, ah) for any policy π ∈ Πexp and any reward r. Recall
that given accuracy ϵ1, the policy set we construct in Section C.3 is Πexp
ϵ1 and the policy set for layer
h is Πexp
ϵ1,h.

E.1 THE ALGORITHM

Algorithm 4 Estimation of Eπr(sh, ah) given exploration data (EstimateER)
1: Input: Policy to evaluate π ∈ Πexp

Layer h. Exploration data {sn
(cid:101)h

ϵ1 . Reward function r(s, a) and its uniform upper bound A.
, an
}((cid:101)h,n)∈[H]×[N ]. Initial state s1.
(cid:101)h

2: Initialization: Qh(·, ·) ← r(·, ·), Vh(·) ← Qh(·, πh(·)).
3: for (cid:101)h = h − 1, h − 2, . . . , 1 do
4:

5:

n=1 ϕ(sn
, an
)ϕ(sn
(cid:101)h
(cid:101)h
(cid:101)h
n=1 ϕ(sn
, an
)V
(cid:101)h
(cid:101)h
(cid:101)h)[0,A].

(cid:101)h)−1 (cid:80)N

(cid:101)h ← I + (cid:80)N
Λ
(cid:101)h ← (Λ
¯w
(cid:101)h(·, ·) ← (ϕ(·, ·)⊤ ¯w
6: Q
(cid:101)h(·)).
(cid:101)h(·) ← Q
V
7:
8: end for
9: Output: V1(s1).

(cid:101)h(·, π

)⊤.
, an
(cid:101)h
(cid:101)h+1(sn

).

(cid:101)h+1

Algorithm 4 applies LSVI to estimate Eπr(sh, ah) for any π ∈ Πexp
(according to our construction,
ϵ1
all possible π’s are deterministic), any reward function r and any time step h. Note that the algorithm
takes the uniform upper bound A of all possible reward functions (i.e., for any reward function r that
may appear as the input, r ∈ [0, A]) as the input, and uses the value of A to truncate the Q-function in
line 6. Algorithm 4 looks similar to Algorithm 3 while there are two key differences. First, the reward
function is non-zero at only one layer in Algorithm 4 while the reward function in Algorithm 3 can
be any valid reward functions. In addition, Algorithm 4 takes the upper bound of reward function as
input and uses this value to bound the Q-functions while Algorithm 3 uses H as the upper bound.

E.2 TECHNICAL LEMMAS

Lemma E.1 (Generalization of Lemma D.4 of Jin et al. (2020b)). Let {xτ }∞
τ =1 be a stochastic
τ =1 be an Rd-valued
process on state space S with corresponding filtration {Fτ }∞
stochastic process where ϕτ ∈ Fτ −1, and ∥ϕτ ∥ ≤ 1. Let Λk = I + (cid:80)k
τ . Then for any
δ > 0, with probability at least 1 − δ, for all k ≥ 0, and any V ∈ V so that supx |V (x)| ≤ A, we
have:

τ =0. Let {ϕτ }∞

τ =1 ϕτ ϕ⊤

k
(cid:88)

ϕτ {V (xτ ) − E[V (xτ )|Fτ −1]}

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

τ =1

2
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

Λ−1
k

≤ 4A2

(cid:20) d
2

log(k + 1) + log(

(cid:21)

)

Nϵ
δ

+ 8k2ϵ2,

where Nϵ is the ϵ-covering number of V with respect to the distance dist(V, V ′) = supx |V (x) −
V ′(x)|.

Lemma E.2. If A ≤ 1, the ¯w

(cid:101)h in line 5 of Algorithm 4 is always bounded by ∥ ¯w

(cid:101)h∥2 ≤

√

dN .

Proof of Lemma E.2. The proof is almost identical to Lemma D.2, the only difference is that H is
replaced by 1.

E.3 UPPER BOUND OF ESTIMATION ERROR

We first consider the covering number of all possible Vh in Algorithm 4. In the remaining part
of this section, we assume that the set of all reward functions to be estimated is ¯R with uniform

25

Published as a conference paper at ICLR 2023

upper bound A ¯R ≤ 1. In addition, assume there exists ϵ-covering ¯Rϵ of ¯R with covering number
log(| ¯Rϵ|) = Bϵ.13
For fixed h ∈ [H], under the case where the layer to estimate is exactly h, Vh can be written as:

Vh(·) = r(·, πh(·)).

(18)

The set Πexp
ϵ1,h| · | ¯Rϵ|.
|Πexp

ϵ1,h × ¯Rϵ provides an ϵ-covering of Vh. Thus the covering number under this case is

In addition, if the layer to estimate is some h′ > h, then Vh can be written as:

where the set Πexp
number under this case is |Πexp

ϵ1,h ×Wϵ (Wϵ is ϵ-covering of Bd(

ϵ1,h| · |Wϵ|.

Vh(·) = (ϕ(·, πh(·))⊤ ¯wh)[0,A ¯R],
√

(19)

dN )) provides an ϵ-covering of Vh. The covering

Since all possible Vh is either the case in (18) (the layer to estimate is exactly h) or (19) (the layer to
estimate is larger than h), for any h ∈ [H] the ϵ-covering number Nϵ of all possible Vh satisfies that:

log Nϵ ≤ log(|Πexp
≤ log(|Πexp

ϵ1,h| · |Wϵ|)

ϵ1,h| · | ¯Rϵ| + |Πexp
ϵ1,h|) + log(| ¯Rϵ|) + log(|Wϵ|)
√
d
2

√

) + d log(1 +

32H 2
ϵ2
1

dN
ϵ

≤2d2 log(1 +

(20)

) + Bϵ.

Now we have the following key lemma. The proof is almost identical to Lemma D.3, so we omit it
here.
Lemma E.3. With probability 1 − δ, for any policy π ∈ Πexp
appear in Algorithm 4 (with the input A = A ¯R) and layer h, the {V
satisfies that for any (cid:101)h ∈ [h − 1],

, any reward function r ∈ ¯R that may
(cid:101)h∈[h] derived by Algorithm 4

(cid:101)h}

ϵ1

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

N
(cid:88)

n=1

(cid:32)

ϕn
(cid:101)h

(cid:101)h+1(sn
V

(cid:101)h+1

) −

(cid:88)

s′∈S

(cid:101)h(s′|sn
P
(cid:101)h

, an
(cid:101)h

)V

(cid:101)h+1(s′)

(cid:33)(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)Λ−1

(cid:101)h

(cid:114)

≤cA ¯R ·

d2 log(

Hd
ϵ1δ

) + d log(

N
δ

) + BA ¯R/N + log(

1
δ

),

for some universal constant c > 0.

(21)

Now we can provide the following Lemma E.4 whose proof is almost identical to Lemma D.5. The
only difference is that H is replaced by A ¯R.
Lemma E.4. With probability 1 − δ, for any policy π ∈ Πexp
appear in Algorithm 4 (with the input A = A ¯R) and layer h, the {V
by Algorithm 4 satisfies that for all (cid:101)h, s, a ∈ [h − 1] × S × A,
(cid:88)

, any reward function r ∈ ¯R that may
(cid:101)h∈[h−1] derived

(cid:101)h∈[h] and { ¯w

(cid:101)h}

(cid:101)h}

ϵ1

|ϕ(s, a)⊤ ¯w

(cid:101)h −

(cid:101)h(s′|s, a)V
P

(cid:101)h+1(s′)|

s′∈S

(cid:114)

Hd
ϵ1δ
for some universal constant c′ > 0.

≤c′A ¯R ·

d2 log(

) + d log(

N
δ

) + BA ¯R/N + log(

1
δ

) · ∥ϕ(s, a)∥Λ−1

(cid:101)h

,

(22)

Finally, the error bound of our estimations are summarized in the following lemma.
, any reward function r ∈ ¯R that may appear in Algorithm
Lemma E.5. For any policy π ∈ Πexp
4 (with the input A = A ¯R) and layer h, let the output of Algorithm 4 be (cid:98)Eπr(sh, ah). Then with

ϵ1

13We will show that all cases we consider in this paper satisfy these two assumptions.

26

Published as a conference paper at ICLR 2023

probability 1 − δ, for any policy π ∈ Πexp

ϵ1

, any reward function r ∈ ¯R and any layer h, it holds that

|(cid:98)Eπr(sh, ah) − Eπr(sh, ah)|

≤c′A ¯R ·

(cid:114)

d2 log(

Hd
ϵ1δ

) + d log(

N
δ

) + BA ¯R/N · Eπ

h−1
(cid:88)

(cid:101)h=1

for some universal constant c′ > 0.

∥ϕ(s

(cid:101)h, a

(cid:101)h)∥Λ−1

(cid:101)h

(23)

,

(cid:101)h}

ϵ0 , any reward function r ∈ ¯R and any layer h, consider
Proof of Lemma E.5. For any policy π ∈ Πexp
(cid:101)h∈[h−1] in Algorithm 4, we have (cid:98)Eπr(sh, ah) = V1(s1). Besides,
the {V
we abuse the notation and let r denote the reward function where rh′(s, a) = 1(h′ = h)r(s, a), let
the value function under this r be V π
(cid:101)h

1 (s1) = Eπr(sh, ah). It holds that

(cid:101)h∈[h] functions and { ¯w

(s), then V π

(cid:101)h}

|(cid:98)Eπr(sh, ah) − Eπr(sh, ah)|
1 (s1)|

=|V1(s1) − V π

≤Eπ

≤Eπ

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

ϕ(s1, a1)⊤ ¯w1 −

ϕ(s1, a1)⊤ ¯w1 −

(cid:88)

s′∈S

(cid:88)

s′∈S

P1(s′|s1, a1)V π

(cid:12)
(cid:12)
2 (s′)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
P1(s′|s1, a1)V2(s′)
(cid:12)
(cid:12)
(cid:12)

+ Eπ

(cid:88)

s′∈S

P1(s′|s1, a1) |V2(s′) − V π

2 (s′)|

≤Eπc′A ¯R ·

(cid:114)

d2 log(

Hd
ϵ1δ

) + d log(

N
δ

) + BA ¯R/N · ∥ϕ(s1, a1)∥Λ−1

1

+ Eπ|V2(s2) − V π

2 (s2)|

≤ · · ·

≤c′A ¯R ·

=c′A ¯R ·

(cid:114)

(cid:114)

d2 log(

d2 log(

Hd
ϵ1δ

Hd
ϵ1δ

) + d log(

) + d log(

N
δ

N
δ

) + BA ¯R/N · Eπ

) + BA ¯R/N · Eπ

h−1
(cid:88)

(cid:101)h=1
h−1
(cid:88)

(cid:101)h=1

∥ϕ(s

(cid:101)h, a

(cid:101)h)∥Λ−1

(cid:101)h

∥ϕ(s

(cid:101)h, a

(cid:101)h)∥Λ−1

(cid:101)h

+ Eπ|Vh(sh) − V π

h (sh)|

,

(24)
where the first inequality results from the fact that V π
1 (s1) ∈ [0, A ¯R]. The third inequality comes
from Lemma E.4. The fifth inequality is due to recursive application of decomposition. The last
equation holds since Vh(·) = V π

h (·) = r(·, πh(·)).

Remark E.6. From Lemma E.5, we can see that the estimation error at layer h can be bounded by
the summation of uncertainty from the previous layers, with additional factor of (cid:101)O(Ad). Therefore, if
the uncertainty of all previous layers are small with respect to Πexp, we can estimate Eπrh accurately
for any π ∈ Πexp and any reward r from a large set of reward functions.
Remark E.7. Note that we only need to estimate Eπr(sh, ah) accurately for π ∈ Πexp. For
π ∈ ∆(Πexp), if π takes policy πi ∈ Πexp with probability pi (for i ∈ [k]), then we define

(cid:98)Eπr(sh, ah) :=

(cid:88)

pi · (cid:98)Eπir(sh, ah),

(25)

i∈[k]
where (cid:98)Eπr(sh, ah) is the estimation we acquire w.r.t policy π and (cid:98)Eπir(sh, ah) is the output of
Algorithm 4 with input πi ∈ Πexp. Assume that for all π ∈ Πexp, |(cid:98)Eπr(sh, ah) − Eπr(sh, ah)| ≤ e,
we have for all π ∈ ∆(Πexp), |(cid:98)Eπr(sh, ah)−Eπr(sh, ah)| ≤ (cid:80)
i pi|(cid:98)Eπir(sh, ah)−Eπir(sh, ah)| ≤
e. Therefore, the conclusion of Lemma E.5 naturally holds for π ∈ ∆(Πexp).

F PROOF OF THEOREM 5.1

Recall that ι = log(dH/ϵδ), ¯ϵ = C1ϵ
√
policies to evaluate is Πeval

H 2

d·ι

ϵ
3

. The explorative policy set we construct is Πexp

. Number of episodes for each deployment is N = C2dι

ϵ
3

while the
¯ϵ2 = C2d2H 4ι3

C2

1 ϵ2

.

27

Published as a conference paper at ICLR 2023

In addition, Σπ is short for Eπ[ϕhϕ⊤
h ]. For clarity, we restrict
our choice that 0 < C1 < 1 and C2, C3 > 1. We begin with detailed explanation of (cid:98)Σπ and
(cid:98)E
(cid:98)π

(cid:105)
ϕ(sh, ah)⊤(N · (cid:98)Σπ)−1ϕ(sh, ah)

h ] while (cid:98)Σπ is short for (cid:98)Eπ[ϕhϕ⊤

from (1).

(cid:104)

F.1 DETAILED EXPLANATION

ϵ
3

ϕi(sh,ah)ϕj (sh,ah)+1
2

First of all, as have been pointed out in Algorithm 1, (cid:98)Σπ is short for (cid:98)Eπ[ϕ(sh, ah)ϕ(sh, ah)⊤].
Assume the feature map is ϕ(s, a) = (ϕ1(s, a), ϕ2(s, a), · · · , ϕd(s, a))⊤, where ϕi(s, a) ∈ R. Then
the estimation of covariance matrix is calculated pointwisely. For each coordinate i, j ∈ [d] × [d],
we use Algorithm 4 to estimate Eπr(sh, ah) = Eπ
14. More specifically, for
any π ∈ Πexp
, (cid:98)Σπ(ij) = 2 (cid:98)Eij − 1, where (cid:98)Eij is the output of Algorithm 4 with input π, r(s, a) =
ϕi(s,a)ϕj (s,a)+1
with A = 1, layer h and exploration dataset D. Therefore, the set of all possible
2
rewards is ¯R = { ϕi(s,a)ϕj (s,a)+1
, (i, j) ∈ [d] × [d]}. The set ¯R is a covering set of itself with
log-covering number Bϵ = log(| ¯R|) = 2 log d. In addition, note that the estimation (cid:98)Σπ(ij) = (cid:98)Σπ(ji)
for all i, j, which means the estimation (cid:98)Σπ is symmetric. The above discussion tackles the case where
, for the general case where π ∈ ∆(Πexp
π ∈ Πexp
), the estimation is derived by (25) in Remark E.7.
In the discussion below, we only need to bound ∥(cid:98)Eπϕhϕ⊤
same bound applies to all π ∈ ∆(Πexp

h ∥2 for all π ∈ Πexp

h − Eπϕhϕ⊤

and the

).

ϵ
3

ϵ
3

ϵ
3

2

ϵ
3

(cid:104)

C1ϵ

(cid:105)
The second estimator is (cid:98)E
ϕ(sh, ah)⊤(N · (cid:98)Σπ)−1ϕ(sh, ah)
, which is calculated via directly
(cid:98)π
applying Algorithm 4 with input (cid:98)π ∈ Πexp
, r(s, a) = ϕ(s, a)⊤(N · (cid:98)Σπ)−1ϕ(s, a) with A =
¯ϵ
C2d7/2H 3ι3 , layer h and exploration dataset D. Note that the validity of uniform upper
C2d3Hι2 =
bound A holds since we only consider the case where λmin((cid:98)Σπ) ≥ d2H¯ϵι, which means that
λmin(N · (cid:98)Σπ) ≥ d2H¯ϵι · C2dι
. Therefore the set of all possible rewards is subset
of ¯R = {r(s, a) = ϕ(s, a)⊤(Σ)−1ϕ(s, a)|λmin(Σ) ≥ C2d7/2H 3ι3
} and the ϵ-covering number is
characterized by Lemma F.3 below.

¯ϵ2 = C2d3Hι2

C1ϵ

ϵ
3

¯ϵ

F.2 TECHNICAL LEMMAS

In this part, we state some technical lemmas.
Lemma F.1 (Lemma H.4 of Min et al. (2021)). Let ϕ : S × A → Rd satisfies ∥ϕ(s, a)∥ ≤ C for
all s, a ∈ S × A. For any K > 0, λ > 0, define ¯GK = (cid:80)K
k=1 ϕ(sk, ak)ϕ(sk, ak)⊤ + λId where
(sk, ak)’s are i.i.d samples from some distribution ν. Then with probability 1 − δ,
(cid:20) ¯GK
K

¯GK
K

− Eν

√
√

2d
δ

(cid:19)1/2

log

(26)

≤

(cid:18)

4

.

2C 2
K

(cid:13)
(cid:13)
(cid:13)
(cid:13)

(cid:21)(cid:13)
(cid:13)
(cid:13)
(cid:13)2

Lemma F.2 (Corollary of Lemma D.6). There exists universal constant cD > 0, such that with our
choice of ϵ0 = ϵ

, the multiplicative factor of (16) satisfies that

3 and N = C2d2H 4ι3
√

C2
(cid:114)

1 ϵ2

c′H

d ·

log(

Hd
ϵ0δ

) + log(

N
δ

) ≤ cDH

√

d · log(

C2dH
C1ϵδ

).

(27)

Proof of Lemma F.2. The existence of universal constant cD holds since c′ in (16) is universal
constant and direct calculation.
Lemma F.3 (Covering number). Consider the set of possible rewards ¯R = {r(s, a) =
ϕ(s, a)⊤(Σ)−1ϕ(s, a)|λmin(Σ) ≥ C2d7/2H 3ι3
, we
have that the A ¯R

N -cover RA ¯R/N of ¯R satisfies that for some universal constant cF > 0,

C2d7/2H 3ι3 and N = C2d2H 4ι3

}. Let A ¯R =

1 ϵ2

C1ϵ

C1ϵ

C2

BA ¯R/N = log(| ¯RA ¯R/N |) ≤ cF d2 log(

C2dH
C1ϵ

).

(28)

14The transformation is to ensure that the reward is larger than 0.

28

Published as a conference paper at ICLR 2023

Proof of Lemma F.3. The conclusion holds due to Lemma D.6 of Jin et al. (2020b) and direct
calculation.

Lemma F.4 (Corollary of Lemma E.5). There exists universal constant c1
first case in Section F.1 with our choice of ϵ1 = ϵ
multiplicative factor of (23) satisfies that

3 , A = 1, B = 2 log(d) and N = C2d2H 4ι3

C2

E > 0 such that for the
, the

1 ϵ2

(cid:114)

d2 log(

c′A ¯R ·

Hd
ϵ1δ

) + d log(

N
δ

) + BA ¯R/N ≤ c1

E · d log(

C2dH
C1ϵδ

).

(29)

Proof of Lemma F.4. The existence of universal constant c1
stant and direct calculation.

E holds since c′ in (23) is universal con-

Lemma F.5 (Corollary of Lemma E.5). There exists universal constant c2
the second case in Section F.1 with our choice of ϵ1 = ϵ
¯ϵ
C2d3Hι2 =
cF d2 log( C2dH

, the multiplicative factor of (23) satisfies that

C1ϵ ) and N = C2d2H 4ι3

3 , A =

C2

1 ϵ2

E > 0 such that for
C1ϵ
C2d7/2H 3ι3 , B =

(cid:114)

d2 log(

c′A ¯R ·

Hd
ϵ1δ

) + d log(

N
δ

) + BA ¯R/N ≤ c2

E ·

¯ϵ
C2d2Hι

log(

C2dH
C1ϵδ

).

(30)

Proof of Lemma F.5. The existence of universal constant c2
stant and direct calculation.

E holds since c′ in (23) is universal con-

Now that we have the universal constants cD, cF , c1
max{c1

E}. Therefore, the conclusions of Lemma F.4 and F.5 hold if we replace ci

E, for notational simplicity, we let cE =

E with cE.

E, c2

E, c2

F.3 CHOICE OF UNIVERSAL CONSTANTS

In this section, we determine the choice of universal constants in Algorithm 1 and Theorem 5.1. First,
C1, C2 satisfies that C1 · C2 = 1, 0 < C1 < 1 and the following conditions:

√

cDH

d · log(

C2dH
C1ϵδ

) ≤

√

1
3C1

H

d log(

dH
ϵδ

).

cE ·

¯ϵ
C2d2Hι

log(

C2dH
C1ϵδ

) ≤

¯ϵ
2d2H

.

(31)

(32)

It is clear that when C2 is larger than some universal threshold and C1 = 1
C2
satisfy the previous four conditions.

, the constants C1, C2

Next, we choose C3 such that

C3
4

log(

dH
ϵδ

) ≥ cE log(

C2dH
C1ϵδ

),

(33)

and C4 = 80C1C3. Since cD, cE, cF are universal constants, our C1, C2, C3, C4 are also universal
constants that are independent with the parameters d, H, ϵ, δ.

F.4 RESTATE THEOREM 5.1 AND OUR INDUCTION

Theorem F.6 (Restate Theorem 5.1). We run Algorithm 1 to collect data and let Planning(·) denote
the output of Algorithm 2. For the universal constants C1, C2, C3, C4 we choose, for any ϵ > 0 and
C4d7/2 log(1/λ⋆) , with probability 1 − δ, for any feasible linear reward function
δ > 0, as well as ϵ <
r, Planning(r) returns a policy that is ϵ-optimal with respect to r.

H(λ⋆)2

Throughout the proof in this section, we assume that the condition ϵ <
we state our induction condition.

H(λ⋆)2

C4d7/2 log(1/λ⋆) holds. Then

29

Published as a conference paper at ICLR 2023

Condition F.7 (Induction Condition). Suppose after h − 1 deployments (i.e., after the explo-
ration of the first h − 1 layers), the dataset Dh−1 = {sn
=
(cid:101)h
I + (cid:80)(h−1)N

(cid:101)h,n∈[H]×[(h−1)N ] and Λh−1
}
)⊤ for all (cid:101)h ∈ [H]. The induction condition is:

(cid:101)h)

 ≤ (h − 1)¯ϵ.

(cid:101)h)⊤(Λh−1

)−1ϕ(s

(ϕn
(cid:101)h

, an
(cid:101)h

h−1
(cid:88)

(cid:101)h, a

(cid:101)h, a

ϕn
(cid:101)h

(34)

Eπ

ϕ(s

n=1

(cid:113)





(cid:101)h

(cid:101)h

max
π∈Πexp
ϵ
3

(cid:101)h=1

Suppose that after h deployments, the dataset Dh = {sn
(cid:101)h
(cid:80)hN

= I +
)⊤ for all (cid:101)h ∈ [H]. We will prove that given condition F.7 holds, with probability at

(cid:101)h,n∈[H]×[hN ] and Λh
}
(cid:101)h

, an
(cid:101)h

n=1 ϕn
(cid:101)h

(ϕn
(cid:101)h

least 1 − δ, the following induction holds:

(cid:20)(cid:113)

ϕ(sh, ah)⊤(Λh

(cid:21)
h)−1ϕ(sh, ah)

≤ ¯ϵ.

Eπ

max
π∈Πexp
ϵ
3

Note that the induction (35) naturally implies that

Eπ

max
π∈Πexp
ϵ
3





h
(cid:88)

(cid:113)

(cid:101)h=1

ϕ(s

(cid:101)h, a

(cid:101)h)⊤(Λh
(cid:101)h


(cid:101)h)

)−1ϕ(s

(cid:101)h, a

 ≤ h¯ϵ.

(35)

(36)

Suppose after the whole exploration process, the dataset D = {sn
I + (cid:80)HN
1 − Hδ,

h}h,n∈[H]×[HN ] and Λh =
h)⊤ for all h ∈ [H]. If the previous induction holds, we have with probability

n=1 ϕn

h(ϕn

h, an

Eπ

max
π∈Πexp
ϵ
3

(cid:34) H
(cid:88)

h=1

(cid:113)

ϕ(sh, ah)⊤(Λh)−1ϕ(sh, ah)

(cid:35)

≤ H¯ϵ.

(37)

Next we begin the proof of such induction. We assume the Condition F.7 holds and prove (35).

F.5 ERROR BOUND OF ESTIMATION

Recall that the policy we apply to explore the h-th layer is

πh =

argmin

π∈∆(Πexp

ϵ
3

) s.t. λmin((cid:98)Σπ)≥C3d2H¯ϵι

(cid:98)E
(cid:98)π

max
(cid:98)π∈Πexp
ϵ
3

(cid:104)

(cid:105)
ϕ(sh, ah)⊤(N · (cid:98)Σπ)−1ϕ(sh, ah)

,

(38)

where the detailed definition of (cid:98)Σπ and (cid:98)E
(cid:98)π
Section F.1. In addition, we define the optimal policy ¯π⋆

h for exploring layer h:

(cid:104)
ϕ(sh, ah)⊤(N · (cid:98)Σπ)−1ϕ(sh, ah)

(cid:105)

are explained in

¯π⋆
h = argmin
π∈∆(Πexp

ϵ
3

max
(cid:98)π∈Πexp
)
ϵ
3

E
(cid:98)π

(cid:2)ϕ(sh, ah)⊤(N · Σπ)−1ϕ(sh, ah)(cid:3) ,

(39)

where E

(cid:98)π means the actual expectation. Similarly, Σπ is short for Eπ[ϕ(sh, ah)ϕ(sh, ah)⊤].

According to Lemma C.8, since ϵ ≤

H(λ⋆)2

C4d7/2 log(1/λ⋆) ≤ λ⋆

4

15, we have

λmin(Eπϕhϕ⊤

h ) ≥

(λ⋆)2
64d log(1/λ⋆)

.

sup
π∈∆(Πexp
ϵ
3

)

Therefore, together with the conclusion of Lemma B.4 and our definition of ¯π⋆

h, it holds that:

λmin(E¯π⋆

h

ϕhϕ⊤

h ) ≥

(λ⋆)2
64d2 log(1/λ⋆)

.

(40)

(41)

15We ignore the extreme case where H is super large for simplicity. When H is very large, we can simply

construct Πexp

ϵ/H instead and the proof is identical.

30

Published as a conference paper at ICLR 2023

F.5.1 ERROR BOUND FOR THE FIRST ESTIMATOR

(cid:13)
(cid:13)
(cid:13)(cid:98)Eπ[ϕ(sh, ah)ϕ(sh, ah)⊤] − Eπ[ϕ(sh, ah)ϕ(sh, ah)⊤]
(cid:13)
(cid:13)
(cid:13)2

We first consider the upper bound of
. Re-
call that (as stated in first half of Section F.1), (cid:98)Eπ[ϕ(sh, ah)ϕ(sh, ah)⊤] is estimated through calling
Algorithm 4 for each coordinate i, j ∈ [d] × [d]. Therefore, we first bound the pointwise error.
Lemma F.8 (Pointwise error). With probability 1 − δ, for all π ∈ Πexp
[d] × [d], it holds that

and all coordinates (i, j) ∈

ϵ
3

(cid:12)
(cid:12)(cid:98)Eπ[ϕ(sh, ah)ϕ(sh, ah)⊤](ij) − Eπ[ϕ(sh, ah)ϕ(sh, ah)⊤](ij)
(cid:12)

(cid:12)
(cid:12)
(cid:12) ≤

C3dH¯ϵι
4

.

(42)

Proof of Lemma F.8. We have

(cid:114)

LHS ≤c′

d2 log(

3Hd
ϵδ

) + d log(

N
δ

) + 2 log(d) · Eπ

h−1
(cid:88)

(cid:101)h=1

C2dH
C1ϵδ

) · H¯ϵ

≤cE · d log(

≤

C3dH¯ϵι
4

.

∥ϕ(s

(cid:101)h, a

(cid:101)h)∥(Λh−1

(cid:101)h

)−1

(43)

The first inequality holds because Lemma E.5. The second inequality results from Lemma F.4 and
our induction condition F.7. The last inequality is due to our choice of C3 (33).

(cid:13)
(cid:13)
(cid:13)(cid:98)Eπ[ϕ(sh, ah)ϕ(sh, ah)⊤] − Eπ[ϕ(sh, ah)ϕ(sh, ah)⊤]
(cid:13)
(cid:13)
(cid:13)2

Now we can bound
lemma.
Lemma F.9 (ℓ2 norm bound). With probability 1 − δ, for all π ∈ Πexp

, it holds that

ϵ
3

by the following

(cid:13)
(cid:13)
(cid:13)(cid:98)Eπ[ϕ(sh, ah)ϕ(sh, ah)⊤] − Eπ[ϕ(sh, ah)ϕ(sh, ah)⊤]
(cid:13)
(cid:13)
(cid:13)2

≤

C3d2H¯ϵι
4

.

(44)

Proof of Lemma F.9. The inequality results from Lemma F.8 and the fact that for any X ∈ Rd×d,

∥X∥2 ≤ ∥X∥F .

Note that the conclusion also holds for all π ∈ ∆(Πexp

ϵ
3

) due to our discussion in Remark E.7.

According to our condition that ϵ <

H(λ⋆)2
C4d7/2 log(1/λ⋆) =

H(λ⋆)2

80C1C3d7/2 log(1/λ⋆) and (41), we have

λmin(E¯π⋆

h

ϕhϕ⊤

h ) ≥

(λ⋆)2
64d2 log(1/λ⋆)

≥

5C1C3d3/2ϵ
4H

=

5C3d2H¯ϵι
4

.

Therefore, under the high probability case in Lemma F.9, due to Weyl’s inequality,

λmin((cid:98)E¯π⋆

h

ϕhϕ⊤

h ) ≥ C3d2H¯ϵι.

(45)

(46)

(47)

We have (47) implies that ¯π⋆

(cid:104)
ϕ(sh, ah)⊤(N · (cid:98)Σπh )−1ϕ(sh, ah)

h is a feasible solution of the optimization problem (1) and therefore,
(cid:105)
(cid:104)
ϕ(sh, ah)⊤(N · (cid:98)Σ¯π⋆

)−1ϕ(sh, ah)

(cid:105)

,

h

≤ max
(cid:98)π∈Πexp

ϵ
3

(cid:98)E
(cid:98)π

(cid:98)E
(cid:98)π

max
(cid:98)π∈Πexp

ϵ
3

where πh is the policy we apply to explore layer h and λmin((cid:98)Σπh ) ≥ C3d2H¯ϵι.

(48)

31

Published as a conference paper at ICLR 2023

F.5.2 ERROR BOUND FOR THE SECOND ESTIMATOR

We consider the upper bound of

(cid:12)
(cid:12)(cid:98)E
(cid:12)
(cid:98)π

(cid:105)
(cid:104)
ϕ(sh, ah)⊤(N · (cid:98)Σπ)−1ϕ(sh, ah)

− E
(cid:98)π

(cid:104)
ϕ(sh, ah)⊤(N · (cid:98)Σπ)−1ϕ(sh, ah)

(cid:105)(cid:12)
(cid:12)
(cid:12) .

(cid:104)

(cid:105)

¯ϵ

ϕ(sh, ah)⊤(N · (cid:98)Σπ)−1ϕ(sh, ah)

Recall that (cid:98)E
(cid:98)π
C2d3Hι2 . Note that we only need to consider the case where (cid:98)π ∈ Πexp
λmin((cid:98)Σπ) ≥ C3d2H¯ϵι.
Lemma F.10. With probability 1 − δ, for all (cid:98)π ∈ Πexp
C3d2H¯ϵι, it holds that:

and all π ∈ ∆(Πexp

ϵ
3

ϵ
3

ϵ
3

, π ∈ ∆(Πexp

ϵ
3

) and

) such that λmin((cid:98)Σπ) ≥

is calculated by calling Algorithm 4 with A =

(cid:12)
(cid:12)(cid:98)E
(cid:12)
(cid:98)π

(cid:105)
(cid:104)
ϕ(sh, ah)⊤(N · (cid:98)Σπ)−1ϕ(sh, ah)

− E
(cid:98)π

(cid:104)
ϕ(sh, ah)⊤(N · (cid:98)Σπ)−1ϕ(sh, ah)

(cid:105)(cid:12)
(cid:12)
(cid:12) ≤

¯ϵ2
2d2 .

(49)

Proof of Lemma F.10. We have

LHS ≤cE ·

¯ϵ
C2d2Hι

log(

C2dH
C1ϵδ

) · E
(cid:98)π

h−1
(cid:88)

(cid:101)h=1

≤

¯ϵ
2d2H

· H¯ϵ =

¯ϵ2
2d2 .

∥ϕ(s

(cid:101)h, a

(cid:101)h)∥(Λh−1

(cid:101)h

)−1

(50)

The first inequality results from Lemma E.5 and Lemma F.5. The second inequality holds since our
choice of C2 (32) and induction condition F.7.

Remark F.11. We have with probability 1 − δ (under the high probability case in Lemma F.10), due
to the property of max{·}, for all π ∈ ∆(Πexp
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

) such that λmin((cid:98)Σπ) ≥ C3d2H¯ϵι, it holds that:

ϕ(sh, ah)⊤(N · (cid:98)Σπ)−1ϕ(sh, ah)

ϕ(sh, ah)⊤(N · (cid:98)Σπ)−1ϕ(sh, ah)

− max
(cid:98)π∈Πexp
ϵ
3

max
(cid:98)π∈Πexp
ϵ
3

(cid:98)E
(cid:98)π

E
(cid:98)π

(cid:104)

(cid:105)

(cid:104)

ϵ
3

≤

(cid:12)
(cid:12)
(cid:105)
(cid:12)
(cid:12)
(cid:12)
(51)

¯ϵ2
2d2 .

F.6 MAIN PROOF

With all preparations ready, we are ready to prove the main theorem. We assume the high probability
cases in Lemma F.8 (which implies Lemma F.9) and Lemma F.10 hold. First of all, we have:

max
(cid:98)π∈Πexp
ϵ
3

≤ max
(cid:98)π∈Πexp
ϵ
3

≤ max
(cid:98)π∈Πexp
ϵ
3

(cid:104)
ϕ(sh, ah)⊤(N · (cid:98)Σ¯π⋆

h

(cid:98)E
(cid:98)π

)−1ϕ(sh, ah)

(cid:104)
ϕ(sh, ah)⊤(N · (cid:98)Σ¯π⋆

h

E
(cid:98)π

)−1ϕ(sh, ah)

(cid:104)
ϕ(sh, ah)⊤(N · (cid:98)Σ¯π⋆

h

E
(cid:98)π

)−1ϕ(sh, ah)

(cid:105)

(cid:105)

(cid:105)

+

+

¯ϵ2
2d2

¯ϵ2
8

≤ max
(cid:98)π∈Πexp
ϵ
3

E
(cid:98)π

(cid:20)
ϕ(sh, ah)⊤(

4N
5

· Σ¯π⋆

h

)−1ϕ(sh, ah)

(cid:21)

+

¯ϵ2
8

≤

5d
4N

+

¯ϵ2
8

≤

3¯ϵ2
8

.

(52)

The first inequality holds because of Lemma F.10 (and Remark F.11). The second inequality is
because under meaningful case, d ≥ 2. The third inequality holds since under the high probability
)−1 ≼
− (cid:98)Σ¯π⋆
h and Theorem B.1. The last inequality

)−1.16 The forth inequality is due to the definition of ¯π⋆

case in Lemma F.9,
( 4
5 Σ¯π⋆
holds since our choice of N and C2.

, and thus ((cid:98)Σ¯π⋆

can imply (cid:98)Σ¯π⋆

Id ≽ Σ¯π⋆

≽ C3d2H¯ϵι

5 Σ¯π⋆

Σ¯π⋆
h
5

≽ 4

4

h

h

h

h

h

h

16Note that all matrices here are symmetric and positive definite.

32

Published as a conference paper at ICLR 2023

Combining (52) and (48), we have

3¯ϵ2
8

≥ max
(cid:98)π∈Πexp
ϵ
3

(cid:98)E
(cid:98)π

(cid:104)
ϕ(sh, ah)⊤(N · (cid:98)Σπh )−1ϕ(sh, ah)

(cid:105)

.

According to Lemma F.10, Remark F.11 and the fact that λmin((cid:98)Σπh ) ≥ C3d2H¯ϵι. It holds that

3¯ϵ2
8

E
(cid:98)π

(cid:104)
ϕ(sh, ah)⊤(N · (cid:98)Σπh )−1ϕ(sh, ah)

≥ max
(cid:98)π∈Πexp
ϵ
3
(cid:104)
ϕ(sh, ah)⊤(N · (cid:98)Σπh )−1ϕ(sh, ah)

(cid:105)
.

Or equivalently, ¯ϵ2

2 ≥ max
(cid:98)π∈Πexp

ϵ
3

E
(cid:98)π

(cid:105)

−

¯ϵ2
8

.

(53)

(54)

Suppose after applying policy πh for N episodes, the data we collect17 is {si
¯Λh = I + (cid:80)N
First, according to Lemma F.9, we have:

h)⊤, we now consider the relationship between ¯Λh and (cid:98)Σπh .

i=1 ϕ(si

h)ϕ(si

h, ai

h, ai

h, ai

h}i∈[N ]. Assume

N · (cid:98)Σπh − N · Σπh

≼ C3N d2H¯ϵι
4

· Id ≼ 1
4

N · (cid:98)Σπh .

(55)

Besides, due to Lemma F.1 (with C = 1), with probability 1 − δ,
N ι · Id ≼ C3N d2H¯ϵι

N · Σπh − ¯Λh ≼ 4

√
2

√

· Id ≼ 1
4

4

N · (cid:98)Σπh .

(56)

Combining (55) and (56), we have with probability 1 − δ,

or equivalently,

N · (cid:98)Σπh − ¯Λh ≼ 1
2

N · (cid:98)Σπh ,

(N · (cid:98)Σπh)−1 ≽ (2¯Λh)−1.

Plugging (58) into (54), we have with probability 1 − δ,

¯ϵ2
2

≥ max
(cid:98)π∈Πexp
ϵ
3
≥ max
(cid:98)π∈Πexp
ϵ
3
(cid:32)

≥

1
2

(cid:104)
ϕ(sh, ah)⊤(N · (cid:98)Σπh )−1ϕ(sh, ah)

(cid:105)

(cid:2)ϕ(sh, ah)⊤(2¯Λh)−1ϕ(sh, ah)(cid:3)

E
(cid:98)π

E
(cid:98)π

(cid:113)

ϕ(sh, ah)⊤(¯Λh)−1ϕ(sh, ah)

,

(cid:33)2

E
(cid:98)π

max
(cid:98)π∈Πexp
ϵ
3

(57)

(58)

(59)

where the last inequality follows Cauchy-Schwarz inequality.
Recall that after the exploration of layer h, Λh
which implies that Λh
h

≽ ¯Λh and (Λh

(cid:113)

¯ϵ ≥ max
(cid:98)π∈Πexp
ϵ
3

E
(cid:98)π

ϕ(sh, ah)⊤(¯Λh)−1ϕ(sh, ah) ≥ max
(cid:98)π∈Πexp

E
(cid:98)π

ϵ
3

h in (35) uses all previous data up to the h-th deployment,

h)−1 ≼ (¯Λh)−1. Therefore, with probability 1 − δ,
(cid:113)

ϕ(sh, ah)⊤(Λh

h)−1ϕ(sh, ah),

(60)

which implies that the induction process holds.

that after

Recall
{sn
h, an
have with probability 1 − Hδ,

h}h,n∈[H]×[HN ] and Λh = I + (cid:80)HN

the whole exploration process for all H layers,

n=1 ϕn

h(ϕn

the dataset D =
h)⊤ for all h ∈ [H]. Due to induction, we

Eπ

max
π∈Πexp

ϵ
3

(cid:34) H
(cid:88)

h=1

(cid:113)

ϕ(sh, ah)⊤(Λh)−1ϕ(sh, ah)

(cid:35)

≤ H¯ϵ.

(61)

17We only consider the data from layer h.

33

Published as a conference paper at ICLR 2023

In addition, according to Lemma C.7, Πeval

ϵ
3

⊆ Πexp

ϵ
3

, we have

Eπ

max
π∈Πeval
ϵ
3

(cid:34) H
(cid:88)

(cid:113)

h=1

ϕ(sh, ah)⊤(Λh)−1ϕ(sh, ah)

≤ H¯ϵ.

(62)

(cid:35)

Given (62), we are ready to prove the final result. Recall that the output of Algorithm 3 (with input π
and r) is (cid:98)V π(r). With probability 1 − δ, for all feasible linear reward function r, for all π ∈ Πeval
, it
holds that

ϵ
3

| (cid:98)V π(r) − V π(r)| ≤c′H

√

(cid:114)

d ·

log(

3Hd
ϵδ

) + log(

N
δ

) · Eπ

H
(cid:88)

h=1

∥ϕ(sh, ah)∥Λ−1

h

√

d · log(

≤cDH

√

≤

1
3C1

H

dι · H¯ϵ =

C2dH
C1ϵδ
ϵ
3

) · H¯ϵ

,

(63)

where the first inequality holds due to Lemma D.6. The second inequality is because of Lemma F.2
and (62). The third inequality holds since our choice of C1 (31). The last equation results from our
definition that ¯ϵ = C1ϵ
√
dι

H 2

.

Suppose (cid:101)π(r) = arg maxπ∈Πeval
respect to (cid:98)V π(r), we have

ϵ
3

V π(r). Since our output policy (cid:98)π(r) is the greedy policy with

V (cid:101)π(r)(r) − V (cid:98)π(r)(r) ≤V (cid:101)π(r)(r) − (cid:98)V (cid:101)π(r)(r) + (cid:98)V (cid:101)π(r)(r) − (cid:98)V (cid:98)π(r)(r) + (cid:98)V (cid:98)π(r)(r) − V (cid:98)π(r)(r)

≤

2ϵ
3

.

(64)

In addition, according to Lemma C.5, V ⋆(r) − V (cid:101)π(r)(r) ≤ ϵ
with probability 1 − δ, for all feasible linear reward function r,

3 . Combining these two results, we have

V ⋆(r) − V (cid:98)π(r)(r) ≤ ϵ.

(65)

Since the deployment complexity of Algorithm 1 is clearly bounded by H, the proof of Theorem 5.1
is completed.

G COMPARISONS ON RESULTS AND TECHNIQUES

In this section, we compare our results with the closest related work (Huang et al., 2022). We begin
with comparison of the conditions.

Comparison of conditions. In Assumption 2.1, we assume that the linear MDP satisfies

λ⋆ = min
h∈[H]

sup
π

λmin(Eπ[ϕ(sh, ah)ϕ(sh, ah)⊤]) > 0.

In comparison, Huang et al. (2022) assume that

νmin = min
h∈[H]

min
∥θ∥=1

max
π

(cid:113)

Eπ[(ϕ⊤

h θ)2] > 0.

Overall these two assumptions are analogous reachability assumptions, while our assumption is
slightly stronger since ν2

min is lower bounded by λ⋆.

Dependence on reachability coefficient. Our Algorithm 1 only takes ϵ as input and does not require
the knowledge of λ⋆, while the theoretical guarantee in Theorem 5.1 requires additional condition that
ϵ is small compared to λ⋆. For ϵ larger than a problem-dependent threshold, the theoretical guarantee
no longer holds. Such dependence is similar to the dependence on reachability coefficient νmin in
Zanette et al. (2020b) where their algorithm also takes ϵ as input and requires ϵ to be small compared

34

Published as a conference paper at ICLR 2023

to νmin. In comparison, Algorithm 2 in Huang et al. (2022) takes the reachability coefficient νmin as
input, which is a stronger requirement than requiring ϵ to be small compared to λ⋆.

ϵ2 ) with (cid:101)O( d3H 5

Comparison of sample complexity bounds. Our main improvement over Huang et al. (2022) is on
the sample complexity bound in the small-ϵ regime. Comparing our asymptotic sample complexity
bound (cid:101)O( d2H 5
, where
νmin is always upper bounded by 1 and can be arbitrarily small (please see the illustration below).
In the large-ϵ regime, the sample complexity bounds in both works look like poly(d, H, 1
λ⋆ ) (or
)), and such “Burn in” period is common in optimal experiment design based works
poly(d, H,
(Wagenmaker & Jamieson, 2022).

) in Huang et al. (2022), our bound is better by a factor of

1
νmin

d
ν2

ϵ2ν2

min

min

Illustration of νmin. In this part, we construct some examples to show what νmin will be like. First,
consider the following simple example where the linear MDP 1 is defined as:

1. The linear MDP is a tabular MDP with only one action and several states (A = 1, S > 1).

2. The features are canonical basis (Jin et al., 2020b) and thus d = S.

3. The transition from any (s, a) ∈ S × A at any time step h ∈ [H] is uniformly random.

Therefore, under linear MDP 1, both ν2
d and our improve-
ment on sample complexity is a factor of d2. Generally speaking, this example has a relatively large
νmin, and there are various examples with even smaller νmin. Next, we construct the linear MDP 2
that is similar to the linear MDP 1 but does not have uniform transition kernel:

min in Huang et al. (2022) and our λ⋆ are 1

1. The linear MDP is a tabular MDP with only one action and several states (A = 1, S > 1).

2. The features are canonical basis (Jin et al., 2020b) and thus d = S.

3. The transitions from any (s, a) ∈ S × A at any time step h ∈ [H] are the same and satisfies

mins′∈S Ph(s′|s, a) = pmin.

min in Huang et al. (2022) and our λ⋆ are pmin (pmin ≤ 1

Therefore, under linear MDP 2, both ν2
d ) and
our improvement on sample complexity is a factor of d/pmin which is always larger than d2 and can
be much larger. In the worst case, according to the condition (ϵ < ν8
min) for the asymptotic sample
complexity in Huang et al. (2022) to dominate, pmin = ν2
min can be as small as ϵ1/4, and the sample
complexity in Huang et al. (2022) is (cid:101)O( 1
ϵ2.25 ), which does not have optimal dependence on ϵ. In
conclusion, our improvement on sample complexity is at least a factor of d and can be much more
significant under various circumstances.

d
ν2

h

ϵ/3

Eπ∥ϕh∥Λ−1

Technique comparison. We discuss why we can get rid of the
dependence in Huang
et al. (2022). First, instead of minimizing maxπ Eπ∥ϕh∥Λ−1
, we only minimize the smaller
maxπ∈Πexp
, where the maximum is taken over our explorative policy set. Therefore, our
approximation of generalized G-optimal design helps save the factor of 1/ν2
min. In addition, note
that in Lemma 6.3, the dependence on d is only
d, this is because we estimate the value functions
(w.r.t π and r) instead of adding optimism and using LSVI. Compared to the log-covering number
(cid:101)O(d2) of the bonus term
ϵ/3,h, linear reward rh) has
log-covering number (cid:101)O(d).

h Λ−1ϕh, our covering of (policy πh ∈ Πeval
ϕ⊤

(cid:113)

√

min

h

H PROOF FOR SECTION 7

H.1 APPLICATION TO TABULAR MDP

Recall that the tabular MDP has discrete state-action space with |S| = S, |A| = A. We transfer our
Assumption 2.1 to its counterpart under tabular MDP, and assume it holds.
Assumption H.1. Define dπ
Let dm = minh supπ mins,a dπ

h(·, ·) to be the occupancy measure, i.e. dπ

h(s, a) = Pπ(sh = s, ah = a).

h(s, a), we assume that dm > 0.

35

Published as a conference paper at ICLR 2023

Theorem H.2. We select ¯ϵ = C1ϵ
√
¯ϵ2 = C2S2AH 4ι3
N = C2SAι

H 2

C2

Sι

1 ϵ2

, Πexp = Πeval = Π0 = {all deterministic policies} and

in Algorithm 1 and 2. The optimization problem is replaced by

πh =

argmin

π∈∆(Π0) s.t. ∀ (s,a), (cid:98)dπ

h(s,a)≥C3H

max
(cid:98)π∈Π0

√

S¯ϵι

(cid:88)

s,a

(cid:98)d(cid:98)π
h(s, a)
(cid:98)dπ
h(s, a)

,

(66)

h(s, a) is estimated through applying Algorithm 4. Suppose ϵ ≤ Hdm

where (cid:98)dπ
C4SA , with probability 1 − δ,
for any reward function r, Algorithm 2 returns a policy that is ϵ-optimal with respect to r. In addition,
the deployment complexity of Algorithm 1 is H while the number of trajectories is (cid:101)O( S2AH 5

).

ϵ2

Proof of Theorem H.2. Since the proof is quite similar to the proof of Theorem 5.1, we sketch the
proof and highlight the difference to the linear MDP setting while ignoring details.

Suppose after the h-th deployment, the visitation number of ((cid:101)h, s, a) is N h
(cid:101)h

condition becomes after the (h − 1)-th deployment, maxπ

We base on this condition and prove that with high probability, maxπ

(cid:34)

(cid:80)h−1
(cid:101)h=1

(cid:80)

s,a

(cid:113)

(s, a). Then our induction
(cid:35)

(s,a)

(s,a)

dπ
(cid:101)h
N h−1
(cid:101)h
dπ
h(s,a)
N h

√

h (s,a)

(cid:20)

(cid:80)

s,a

≤ (h − 1)¯ϵ.

(cid:21)

≤ ¯ϵ.

First, under tabular MDP, Algorithm 4 is equivalent to value iteration based on empirical transition
kernel. Therefore, due to standard methods like simulation lemma, we have with high probability, for
any π ∈ Π0 and reward r with upper bound A (the V
(cid:101)h function is the one we derive in Algorithm 4),

|(cid:98)Eπr(sh, ah) − Eπr(sh, ah)| ≤Eπ

≤Eπ

h−1
(cid:88)

(cid:101)h=1
h−1
(cid:88)

(cid:12)
(cid:16)
(cid:12)
(cid:12)

(cid:101)h − P
(cid:98)P
(cid:101)h

(cid:17)

· V

(cid:101)h+1(s

(cid:101)h, a

(cid:101)h)

A ·

(cid:13)
(cid:13)
(cid:13) (cid:98)P

(cid:101)h − P
(cid:101)h

(cid:13)
(cid:13)
(cid:13)1

(cid:12)
(cid:12)
(cid:12)





(67)

S · Eπ

(cid:115)

h−1
(cid:88)

(cid:101)h=1

N h−1
(cid:101)h

1
(s

(cid:101)h=1


√

≤ (cid:101)O

A



≤ (cid:101)O

A

√

S ·

h−1
(cid:88)

(cid:88)

(cid:113)

(cid:101)h=1

s,a

√

≤A

S · H¯ϵ.

(cid:101)h, a

(cid:101)h)


(s, a)

dπ
(cid:101)h
N h−1
(cid:101)h



(s, a)

Now we prove that our condition about ϵ is enough. Note that with high probability, for all policy
π ∈ Π0 and s, a, the estimation error of (cid:98)dπ
S · H¯ϵ. As a result, the estimation
error can be ignored compared to dπh
h (s, a). With identical proof to Section F.6, we have
the induction still holds.

h(s, a) is bounded by

h (s, a) or d¯π⋆

√

h

From the induction, suppose Nh(s, a) is the final visitation number of (h, s, a), we have

≤ H¯ϵ. Using identical proof to (67), we have with high proba-

(cid:20)

maxπ

(cid:80)H

h=1

(cid:80)

s,a

√

dπ
h(s,a)
Nh(s,a)

(cid:21)

bility, for all π ∈ Π0 and r,

| (cid:98)V π(r) − V π(r)| ≤ (cid:101)O(H

√

S · H¯ϵ) ≤

ϵ
2

.

(68)

Since Π0 contains the optimal policy, our output policy is ϵ-optimal.

H.2 PROOF OF LOWER BOUNDS

For regret minimization, we assume the number of episodes is K while the number of steps is
T := KH.

36

Published as a conference paper at ICLR 2023

Theorem H.3 (Restate Theorem 7.2). For any algorithm with the optimal (cid:101)O((cid:112)poly(d, H)T ) regret
bound, the switching cost is at least Ω(dH log log T ).

Proof of Theorem H.3. We first construct a linear MDP with two states, the initial state s1 and the
absorbing state s2.

For absorbing state s2, the choice of action is only a0, while for initial state s1, the choice of actions
is {a1, a2, · · · , ad−1}. Then we define the feature map:

ϕ(s2, a0) = (1, 0, 0, · · · , 0), ϕ(s1, ai) = (0, · · · , 0, 1, 0, · · · ),

where for s1, ai (i ∈ [d − 1]), the (i + 1)-th element is 1 while all other elements are 0. We now
define the measure µh and reward vector θh as:

µh(s1) = (0, 1, 0, 0, · · · , 0), µh(s2) = (1, 0, 1, 1, · · · , 1), ∀ h ∈ [H].

θh = (0, 0, rh,2, · · · , rh,d−1), where rh,i’s are unknown non-zero values.
Combining these definitions, we have: Ph(s2|s2, a0) = 1, rh(s2, a0) = 0, Ph(s1|s1, a1) = 1,
rh(s1, a1) = 0 for all h ∈ [H]. Besides, Ph(s2|s1, ai) = 1, rh(s1, ai) = rh,i for all h ∈ [H], i ≥ 2.

Therefore, for any deterministic policy, the only possible case is that the agent takes action a1 and
stays at s1 for the first h − 1 steps, then at step h the agent takes action ai (i ≥ 2) and transitions
to s2 with reward rh,i, later the agent always stays at s2 with no more reward. For this trajectory,
the total reward will be rh,i. Also, for any deterministic policy, the trajectory is fixed, like pulling
an “arm” in multi-armed bandits setting. Note that the total number of such “arms” with non-zero
unknown reward is at least (d − 2)H. Even if the transition kernel is known to the agent, this linear
MDP is still as difficult as a multi-armed bandits problem with Ω(dH) arms. Together will Lemma
H.4 below, the proof is complete.

Lemma H.4 (Theorem 2 in (Simchi-Levi & Xu, 2019)). Under the K-armed bandits problem, there
exists an absolute constant C > 0 such that for all K > 1, S ≥ 0, T ≥ 2K and for all policy π with
switching budget S, the regret satisfies

Rπ(K, T ) ≥

C
log T

1−

1
2−2−q(S,K)−1 T

1
2−2−q(S,K)−1 ,

· K

√

K−1 ⌋. This further implies that Ω(K log log T ) switches are necessary for

where q(S, K) = ⌊ S−1
achieving (cid:101)O(
Theorem H.5 (Restate Theorem 7.3). For any algorithm with the optimal (cid:101)O((cid:112)poly(d, H)T ) regret
bound, the number of batches is at least Ω( H

T ) regret bound.

logd T + log log T ).

√

Proof of Theorem H.5. Corollary 2 of Gao et al. (2019) proved that under multi-armed bandits
problem, for any algorithm with optimal (cid:101)O(
T ) regret bound, the number of batches is at least
Ω(log log T ). In the proof of Theorem H.3, we show that linear MDP can be at least as difficult as a
multi-armed bandits problem, which means the Ω(log log T ) lower bound on batches also applies to
linear MDP.
In addition, Theorem B.3 in Huang et al. (2022) stated an Ω( H
logd N H ) lower bound for deployment
complexity for any algorithm with PAC guarantee. Note that one deployment of arbitrary policy
is equivalent to one batch. Suppose we can design an algorithm to get (cid:101)O(
T ) regret within K
episodes and M batches, then we are able to identify near-optimal policy in M deployments while
each deployment is allowed to collect K trajectories. Therefore, we have M ≥ Ω( H

√

logd T ).

Combining these two results, the proof is complete.

37

