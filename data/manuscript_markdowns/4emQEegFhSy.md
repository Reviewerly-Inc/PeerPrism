Under review as a conference paper at ICLR 2021

ADAPTIVE MULTI-MODEL FUSION LEARNING FOR
SPARSE-REWARD REINFORCEMENT LEARNING

Anonymous authors
Paper under double-blind review

ABSTRACT

In this paper, we consider intrinsic reward generation for sparse-reward reinforce-
ment learning based on model prediction errors. In typical model-prediction-error-
based intrinsic reward generation, an agent has a learning model for the underlying
environment. Then, intrinsic reward is designed as the error between the model
prediction and the actual outcome of the environment, based on the fact that for
less-visited or non-visited states, the learned model yields larger prediction errors,
promoting exploration helpful for reinforcement learning. This paper generalizes
this model-prediction-error-based intrinsic reward generation method to multiple
prediction models. We propose a new adaptive fusion method relevant to the
multiple-model case, which learns optimal prediction-error fusion across the learn-
ing phase to enhance the overall learning performance. Numerical results show
that for representative locomotion tasks, the proposed intrinsic reward generation
method outperforms most of the previous methods, and the gain is signiﬁcant in
some tasks.

1

INTRODUCTION

Reinforcement learning (RL) with sparse reward is an active research area (Andrychowicz et al.,
2017; Tang et al., 2017; de Abril & Kanai, 2018; Oh et al., 2018; Kim et al., 2019). In sparse-reward
RL, the environment does not return a non-zero reward for every agent’s action but returns a non-zero
reward only when certain conditions are met. Such situations are encountered in many action control
problems (Houthooft et al., 2016; Andrychowicz et al., 2017; Oh et al., 2018). As in conventional
RL, exploration is essential at the early stage of learning in sparse-reward RL, whereas the balance
between exploration and exploitation is required later.

Intrinsically motivated RL has been studied to stimulate better exploration by generating intrinsic
reward for each action by the agent itself. Recently, many intrinsically-motivated RL algorithms have
been devised especially to deal with the sparsity of reward, e.g., based on the notion of curiosity
(Houthooft et al., 2016; Pathak et al., 2017), surprise (Achiam & Sastry, 2017). In essence, in these
intrinsic reward generation methods, the agent has a learning model for the next state or the transition
probability of the underlying environment, and intrinsic reward is designed as the error between the
model prediction and the actual outcome of the environment, based on the fact that for less-visited or
non-visited states, the learned model yields larger prediction errors, promoting exploration helpful
for reinforcement learning. These previous methods typically use a single prediction model for the
next state or the environment’s transition probability.

In this paper, we generalize this model-prediction-error-based approach to the case of multiple
prediction models and propose a new framework for intrinsic reward generation based on the optimal
adaptive fusion of multiple values from multiple models. The use of multiple models increases
diversity in modeling error values and the chance to design a better intrinsic reward from these
values. The critical task is to learn an optimal fusion rule to maximize the performance across the
entire learning phase. In order to devise such an optimal adaptive fusion algorithm, we adopt the
α-mean with the scale-free property from the ﬁeld of information geometry (Amari, 2016) and apply
the meta-gradient optimization to search for optimal fusion at each stage of learning. Numerical
results show that the proposed multi-model intrinsic reward generation combined with fusion learning
signiﬁcantly outperforms existing intrinsic reward generation methods.

1

Under review as a conference paper at ICLR 2021

2 RELATED WORK

Intrinsically-motivated RL and exploration methods can be classiﬁed mainly into two categories.
One is to explicitly generate intrinsic reward and train the agent with the sum of the extrinsic reward
and the adequately scaled intrinsic reward. The other is indirect methods that do not explicitly
generate intrinsic reward. Our work belongs to the ﬁrst category, and we conducted experiments
using baselines in the ﬁrst category. However, we also detailed the second category in Appendix H
for readers for further work in the intrinsically-motivated RL area.

Houthooft et al. (2016) used the information gain on the prediction model as an additional reward
based on the notion of curiosity. Tang et al. (2017) efﬁciently applied count-based exploration to
high-dimensional state space by mapping the states’ trained features into a hash table. The concept
of surprise was exploited to yield intrinsic rewards (Achiam & Sastry, 2017). Pathak et al. (2017)
deﬁned an intrinsic reward with the prediction error using a feature state space, and de Abril & Kanai
(2018) enhanced Pathak et al. (2017)’s work with the idea of homeostasis in biology.

Zheng et al. (2018) used a delayed reward environment to propose training the module to generate
intrinsic reward apart from training the policy. This delayed reward environment for sparse-reward
settings differs from the previous sparse-reward environment based on thresholding (Houthooft et al.,
2016). (The agent gets a non-zero reward when the agent achieves a speciﬁc physical quantity - such
as the distance from the origin - larger than the predeﬁned threshold.) Pathak et al. (2019) interpreted
the disagreement among the models as the variance of the predicted next states and used the variance
as the ﬁnal differentiable intrinsic reward. Our method is a generalized version of their work as we can
apply our proposed fusion method to the multiple squared error values between a predicted next state
and all the predicted next states’ average. Freirich et al. (2019) proposed generating intrinsic reward
by applying a generative model with the Wasserstein-1 distance. With the concept of state-action
embedding, Kim et al. (2019) adopted the Jensen-Shannon divergence (JSD) (Hjelm et al., 2019)
to construct a new variational lower bound of the corresponding mutual information, guaranteeing
numerical stability. Our work differs from these two works in that we use the adaptive fusion method
of multiple intrinsic reward at every timestep.

3 THE PROPOSED METHOD

3.1 SETUP

We consider a discrete-time continuous-state Markov Decision Process (MDP), denoted as
(S, A, P, r, ρ0, γ), where S and A are the sets of states and actions, respectively, P : S × A → Π(S)
is the transition probability function, where Π(S) is the space of probability distributions over S,
r : S × A × S → R is the extrinsic reward function, ρ0 is the probability distribution of the initial
state, and γ is the discounting factor. A (stochastic) policy is represented by π : S → Π(A), where
Π(A) is the space of probability distributions on A and π(a|s) represents the probability of choosing
action a ∈ A for given state s ∈ S. In sparse-reward RL, the environment does not return a non-zero
reward for every action but returns a non-zero reward only when certain conditions are met by the
current state, the action and the next state (Houthooft et al., 2016; Andrychowicz et al., 2017; Oh
et al., 2018). Our goal is to optimize the policy π to maximize the expected cumulative return η(π)
by properly generating intrinsic reward in such sparse-reward environments. We assume that the true
transition probability distribution P is unknown to the agent.

3.2

INTRINSIC REWARD DESIGN BASED ON MODEL PREDICTION ERRORS

Intrinsically-motivated RL adds a properly designed intrinsic reward at every timestep t to the actual
extrinsic reward to yield a non-zero total reward for training even when the extrinsic reward returned
by the environment is zero (Pathak et al., 2017; Tang et al., 2017; de Abril & Kanai, 2018). In the
model-prediction-error-based intrinsic reward design, the agent has a prediction model parametrized
by φ for the next state st+1 or the transition probability P (st+1|st, at), and the intrinsic reward
is designed as the error between the model prediction and the actual outcome of the environment
(Houthooft et al., 2016; Achiam & Sastry, 2017; Pathak et al., 2017; Burda et al., 2019; de Abril &
Kanai, 2018). Thus, the intrinsic-reward-incorporated problem under this approach is given in most

2

Under review as a conference paper at ICLR 2021

Figure 1: Adaptive fusion of K prediction errors from the multiple models

cases as

(cid:8)η(π) + c E(s,a)∼π[D(P ||Pφ)|(s, a)](cid:9)

max
π

(1)

for some constant c > 0 and some divergence function D(·||·), where η(π) is the cumulative reward
associated with policy π, and Pφ is the learning model parameterized by φ that the agent has regarding
the true unknown transition probability P of the environment. For the divergence, the mean squared
error (MSE) between the actual next state and the predicted next state can be used for the error
measure when the learning model predicts the next state itself, or alternatively the Kullback-Leibler
divergence (KLD) between the probability distribution for the next state st+1 and the predicted
probability distribution for st+1 can be used when the learning models learn the transition probability.
In the case of KLD, the intractable DKL(P ||Pφ)|(s, a) with unknown P can be approximated based
on the 1-step approximation (Achiam & Sastry, 2017).

3.3 THE PROPOSED ADAPTIVE FUSION LEARNING

We consider using multiple prediction models and the design of prediction-error-based intrinsic
reward from the multiple models. Suppose we have a collection of K(≥ 2) models parametrized by
φ1, · · · , φK to generate K prediction error (approximation) values at timestep t as intrinsic reward
rj
t,int(st, at, st+1), j = 1, · · · , K, respectively. The key problem of multi-model prediction-error-
based intrinsic reward design is how to learn φ1, · · · , φK and how to optimally fuse the K values
rj
t,int(st, at, st+1), j = 1, · · · , K, to generate a single intrinsic reward to be added to the scalar
cumulative return for policy update. The considered multi-model fusion structure is shown in Fig.
1. To fuse the K values for a single reward value, one can use one of the known methods such as
average, minimum, or maximum. However, there is no guarantee of optimality for such arbitrary
choices, and one ﬁxed fusion rule may not be optimal for the entire learning phase.

Let a fusion function be denoted as

rint = f (r1

int, r2

int, · · · , rK

(2)
int are the K input values and rint is the output value. To devise an optimal

where r1
adaptive fusion rule, we consider the following requirements for the fusion function f .
Condition 1. The fusion function f varies with some control parameter to adapt to the relative
importance of the K input values.

int),

int, r2

int, · · · , rK

We require Condition 1 so that the fusion of the K input values can adapt to the learning situation.
When the more aggressive fusion is required at some phase of learning, we want the function f to
be more like maximum. On the other hand, when the more conservative fusion is required at other
learning phases, we want the function f to be more like minimum. Furthermore, we want this optimal
adaptation is learned based on data to yield maximum cumulative return. In addition, we impose the
following relevant condition for any reasonable fusion function:
Condition 2. The fusion function f is scale-free, i.e.,
f (cr1

int, · · · , crK

int) = cf (r1

int, · · · , rK

int, cr2

int, r2

int).

(3)

3

𝑃𝑃𝜙𝜙1𝑟𝑟𝑖𝑖𝑖𝑖𝑖𝑖Fusion function 𝑓𝑓𝑃𝑃𝜙𝜙2𝑃𝑃𝜙𝜙𝐾𝐾𝐷𝐷𝑃𝑃∥𝑃𝑃𝜙𝜙1𝐷𝐷𝑃𝑃∥𝑃𝑃𝜙𝜙2𝐷𝐷𝑃𝑃∥𝑃𝑃𝜙𝜙𝐾𝐾Control𝑃𝑃𝑟𝑟𝑖𝑖𝑖𝑖𝑖𝑖2𝑟𝑟𝑖𝑖𝑖𝑖𝑖𝑖1𝑟𝑟𝑖𝑖𝑖𝑖𝑖𝑖𝐾𝐾Under review as a conference paper at ICLR 2021

Condition 2 implies that when we scale all the input values by the same factor c, the output is the
c-scaled version of the fusion output of the not-scaled inputs.

Condition 2 is a proper requirement for any reasonable averaging function. The necessity of Condition
2 is explained in detail in Appendix G. Such a fusion function can be found based on the α-mean of
positive measures in the ﬁeld of information geometry (Amari, 2016). For any K positive1 values
x1, · · · , xK > 0, the α-mean of x1, · · · , xK is deﬁned as

fα(x1, · · · , xK) = h−1

(cid:32)

1
K

K
(cid:88)

i=1

(cid:33)

h(xi)

where h(x) is given by the α-embedding transformation:

h(x) =

(cid:40)

1−α
2 ,
x
log x,

if α (cid:54)= 1
if α = 1

.

(4)

(5)

It is proven that the unique class of transformation h satisfying Condition 2 under the twice-
differentiability and the strict monotonicity of h is given by the α-embedding (5) (Amari, 2007;
(cid:17)
2016). Basically, Condition 2 is used to write fα(cx1, · · · , cxK) = h−1 (cid:16) 1
i=1 h(cxi)
=
cfα(x1, · · · , xK). Taking h(·) on both sides yields h(cfα(x1, · · · , xK)) = 1
i=1 h(cxi). Then,
K
taking partial derivative with respect to xi (1 ≤ i ≤ K) on both sides, we can show that the equation
(5) is the unique class of mapping functions (Amari, 2007; 2016).

K
(cid:80)K

(cid:80)K

Furthermore, by varying α, the α-mean includes all numeric fusions with the scale-free property
such as minimum, maximum, and conventional mean functions (Amari, 2016). When α = −∞,
fα(x1, · · · , xK) = maxi xi. On the other hand, when α = ∞, fα(x1, · · · , xK) = mini xi. As α
increases from −∞ to ∞, the α-mean output varies monotonically from maximum to minimum.
See Appendix B. Hence, we can perform aggressive fusion to conservative fusion by controlling the
parameter α.

3.3.1 LEARNING OF α WITH META-GRADIENT OPTIMIZATION

In the proposed adaptive fusion, we need to adaptively control α judiciously to maximize the expected
cumulative extrinsic return η(π). To learn optimal α maximizing η(π), we use the meta gradient
method (Xu et al., 2018; Zheng et al., 2018). Optimal α at each stage of learning is learned with the
proposed method, and it will be shown that optimal α varies according to the stage of learning. For
policy πθ with policy parameter θ, let us deﬁne the following quantities.
(cid:35)
: the expected cumulative sum of extrinsic rewards which we

γtr(st, at, st+1)

• η(πθ) = Eτ ∼πθ

(cid:34) ∞
(cid:88)

want to maximize. Here, τ is a sample trajectory.

t=0

• ηtotal(πθ) = Eτ ∼πθ

(cid:34) ∞
(cid:88)

γt(r(st, at, st+1) + cfα(st, at, st+1))

: the expected cumulative sum of both

(cid:35)

t=0

extrinsic and intrinsic rewards with which the policy πθ is updated. Here, the dependence of the
fusion output fα on (st, at, st+1) through rj
t,int(st, at, st+1) is shown with notation simpliﬁcation.

Then, for a given trajectory τ = (s0, a0, s1, a1, . . .) generated by πθ, we update θ towards the
direction of maximizing ηtotal(πθ):

˜θ = θ + δθ∇θηtotal(πθ)
where δθ is the learning rate for θ. Then, the fusion parameter α is updated to maximize the expected
cumulated sum of extrinsic rewards for the updated policy π˜θ:

(6)

˜α = α + δα∇αη(π˜θ)

(7)

1When an input value to the α-mean is negative due to divergence approximation in some cases, we can
use exponentiation at the input stage and its inverse logarithm at the output stage. We used the exponentiation
exp(−x) at the input stage with input x and the negative logarithm of the α-mean as its inverse at the output
stage for actual implementation. In this case, due to the monotone decreasing property of the input mapping:
x → exp(−x), the output is the maximum when α = ∞ and is the minimum when α = −∞.

4

Under review as a conference paper at ICLR 2021

where δα is the learning rate for α. Note that we update the policy parameter θ to maximize ηtotal(πθ)
so that the updated policy parameter ˜θ is a function of α. Therefore, ∇αη(π˜θ) is not zero and can be
computed by chain rule:

∇αη(π˜θ) = ∇˜θη(π˜θ) ∇α
(8)
To learn optimal α together with θ, we adopt an alternating optimization method widely used in
meta-parameter optimization. That is, we iterate the following two steps in an alternating manner:

˜θ

1) Update the policy parameter θ to maximize ηtotal(πθ).
2) Update the fusion parameter α to maximize η(π˜θ), where ˜θ is the updated policy parameter

from Step 1).

In this way, we can learn proper α adaptively over timesteps to maximize the performance.

3.4

IMPLEMENTATION

We consider the case of D(·||·) = DKL(·||·) for implementation example (See Appendix F for the
comparison of KLD and MSE). We use a collection of K prediction models Pφ1, · · · , PφK . Then,
from the j-th model Pφj , j = 1, · · · , K, we have the j-th prediction error, given by

DKL(P ||Pφj )|(st, at) = EP

(cid:20)

log

P (·|st, at)
P (cid:48)(·|st, at)

P (cid:48)(·|st, at)
Pφj (·|st, at)

(cid:21)

≥ EP

(cid:20)

log

(cid:21)

P (cid:48)(·|st, at)
Pφj (·|st, at)

.

(9)

Note that the j-th model prediction error DKL(P ||Pφj )|(st, at) is lower bounded as (9) for any
distribution P (cid:48). In order to obtain a tight lower bound, P (cid:48) should be learned to be close to the true
transition probability P . For increased degrees of freedom for better learning and estimation, we use
the mixture distribution of PK = (cid:80)K
i=1 qiPφi for P (cid:48) with the learnable mixing coefﬁcients qi ≥ 0
and (cid:80)K
i=1 qi = 1. The mixture model PK has increased model order for modeling the true P beyond
single-mode distribution. Then, the prediction error approximation as intrinsic reward for the j-th
model Pφj at timestep t is determined as rj
Pφj (st+1|st,at) , j = 1, · · · , K.
Note that each rj

t,int(st, at, st+1) = log PK (st+1|st,at)

t,int can be negative although the KLD is always nonnegative.

Although the proposed intrinsic reward generation method can be combined with general RL al-
gorithms, we consider the PPO algorithm (Schulman et al., 2017), a popular on-policy algorithm
generating a batch of experiences of length L with every current policy. Thus, the exposition below
is focused on application to PPO. For the K prediction models Pφ1 , · · · , PφK , we adopt the fully-
factorized Gaussian distribution (Houthooft et al., 2016; Achiam & Sastry, 2017). Then, PK becomes
the class of K-modal Gaussian mixture distributions.

We ﬁrst update the prediction models Pφ1, · · · , PφK and the corresponding mixing coefﬁcients
q1, . . . , qK. In the beginning, the parameters φ1, · · · , φK are independently initialized, and qi’s are
set to 1
K for all i = 1, · · · , K. At every batch period l of PPO, to jointly learn φi and qi, we apply
maximum-likelihood estimation (MLE) with an L2-norm regularizer with KL constraints (Williams
& Rasmussen, 2006; Achiam & Sastry, 2017):

maximize
φi qi, 1≤i≤K

E(s,a,s(cid:48)) log

(cid:124)

(cid:40) K
(cid:88)

(cid:41)

qiPφi (s(cid:48)|s, a)

−creg

i=1
(cid:123)(cid:122)
=:Llikelihood

(cid:125)

K
(cid:88)

(cid:107)φi(cid:107)2

i=1
(cid:124)

(cid:123)(cid:122)
=:Lreg

(cid:125)

subject to

(cid:104)

E(s,a)

DKL(Pφi ||Pφi

old

)(s, a)

(cid:105)

≤ κ,

K
(cid:88)

i=1

qi = 1

(10)

where φi
old is the parameter of the i-th model before the update caused by (10), creg is the regularization
coefﬁcient, and κ is a positive constant. To solve this optimization problem with respect to {φi}, we
apply the method based on second-order approximation (Schulman et al., 2015a). For the update of
{qi}, we apply the EM method proposed in Dempster et al. (1977) and set qi as

qi = E(s,a,s(cid:48))

i Pφi(s(cid:48)|s, a)
qold
j=1 qold

j Pφj (s(cid:48)|s, a)

(cid:80)K

(1 ≤ i ≤ K)

(11)

5

Under review as a conference paper at ICLR 2021

Figure 2: Performance comparison. All simulations were conducted over ten ﬁxed random seeds.
The y-axis in each ﬁgure with the title “Average Return” represents the mean value of the extrinsic
returns of the most recent 100 episodes averaged over the ten random seeds. Each colored band in
every ﬁgure represents the interval of ±σ around the mean curve, where σ is the standard deviation
of the ten instances of data from the ten random seeds. In order to give sufﬁcient time steps for each
environment, for the three environments in the top row, the experiments were performed for 3M
timesteps. For the environments in the bottom row, the experiments were conducted for 1M timesteps.
(For clarity, the ﬁrst author of each of the algorithms is shown in the Humanoid plot.)

i

where qold
is the mixing coefﬁcient of the i-th model before the update caused by (11). For numerical
stability, we use the “log-sum-exp” trick for computing (11) as well as Llikelihood deﬁned in (10) and
∇φiLlikelihood. In addition, we apply simultaneous update of all φi’s and qi’s, which was found to
perform better than one-by-one alternating update of the K models for the considered case.

The update of policy by using PPO is as follows. Let D be the batch of experiences for training the
policy, i.e., D = (st, at, rtotal
, st+1, · · · , rtotal
t+L−1), where at ∼ πθl (·|st),
st+1 ∼ P (·|st, at), and rtotal
is the total reward described below. Here, πθl is the parameterized
policy at the batch period l corresponding to timestep t, · · · , t + L − 1 (the batch period index l is
included in πθl for clarity). The total reward at timestep t for training the policy is given by

t+L−2, st+L−1, at+L−1, rtotal

t

t

rtotal
t

(st, at, st+1) = rt(st, at, st+1) + βrt,int(st, at, st+1)

(12)

where rt(st, at, st+1) is the actual sparse extrinsic reward at timestep t from the environment,
rt,int(st, at, st+1) is the intrinsic reward at timestep t, and β > 0 is the weighting factor. Here, for
actual computation of the intrinsic reward, we further applied two techniques: the 1-step technique
and the normalization technique used in Achiam & Sastry (2017) (which are described in Appendix
C). Then, the policy πθl is updated at every batch period l with D by following the standard PPO
procedure based on the total reward (12). Summarizing the above, we provide the pseudocode of our
algorithm, Algorithm 1, which assumes PPO as the base algorithm, in Appendix A.

4 RESULTS

4.1 PERFORMANCE COMPARISON

To evaluate the performance, we considered sparse-reward environments for continuous control. The
considered tasks were six environments of Mujoco (Todorov et al., 2012), OpenAI Gym (Brockman
et al., 2016): Walker2d, Hopper, InvertedPendulum, HalfCheetah, Ant, and Humanoid. To implement
a sparse-reward setting, we adopted the delay method (Oh et al., 2018). We ﬁrst accumulate extrinsic
rewards generated from the considered environments for every ∆ timesteps or until the episode ends.

6

0.00.51.01.52.02.53.0Timestep(M)050010001500200025003000Average ReturnWalker2dProposed MethodModuleSingle SurpriseInformation GainCuriosityDisagreementHashingPPO Only0.00.51.01.52.02.53.0Timestep(M)05001000150020002500Average ReturnHopperProposed MethodModuleSingle SurpriseInformation GainCuriosityDisagreementHashingPPO Only0.00.51.01.52.02.53.0Timestep(M)020040060080010001200Average ReturnInvertedPendulumProposed MethodModuleSingle SurpriseInformation GainCuriosityDisagreementHashingPPO Only0.00.20.40.60.81.0Timestep(M)500050010001500Average ReturnHalfCheetahProposed MethodModuleSingle SurpriseInformation GainCuriosityDisagreementHashingPPO Only0.00.20.40.60.81.0Timestep(M)5004003002001000100Average ReturnAntProposed MethodModuleSingle SurpriseInformation GainCuriosityDisagreementHashingPPO Only0.00.20.40.60.81.0Timestep(M)100200300400500Average ReturnHumanoidProposed MethodModule (Zheng)Single Surprise (Achiam)Information Gain (de Abril)Curiosity (Pathak 17)Disagreement (Pathak 19)Hashing (Tang)PPO OnlyUnder review as a conference paper at ICLR 2021

Then we provide the accumulated sum of rewards to the agent at the end of the ∆ timesteps or at the
end of the episode, and repeat this process. For our experiments, we set ∆ = 40 as used in (Zheng
et al., 2018). We compared the proposed method with existing intrinsic reward generation methods
by using PPO as the base algorithm. We considered the existing intrinsic reward generation methods:
single-model surprise (Achiam & Sastry, 2017), curiosity (Pathak et al., 2017), hashing (Tang et al.,
2017), and information gain approximation (de Abril & Kanai, 2018). We also considered the method
using intrinsic reward module (Zheng et al., 2018) among the most recent works introduced in
Section 2, which uses delayed sparse-reward setup and provides an implementation code. Finally, we
compared the proposed fusion with the disagreement method using the variance of multiple predicted
next states as the intrinsic reward (Pathak et al., 2019).

For fair comparison, we used PPO with the same neural network architecture and common hyperpa-
rameters. We also applied the same normalization technique in Appendix C for all the considered
intrinsic reward generation methods so that the performance difference results only from the intrinsic
reward generation method. In the case of the state-of-the-art algorithm by Zheng et al. (2018), we veri-
ﬁed reproducibility for the setup ∆ = 40 by obtaining the same result as the reference. (See Appendix
D for a detailed description of the overall hyperparameters for simulations and reproducibility.)

Fig. 2 shows the comparison results. It is observed that the proposed fusion-based intrinsic reward
generation method yields top-level performance. The gain is signiﬁcant in Hopper and Walker2d, and
the performance variance is much smaller than the state-of-the-art intrinsic reward module method in
most cases.

4.2 ABLATION STUDY

(a)

(b)

(c)

(d)

(a) Learning curve of α during the proposed fusion learning in HalfCheetah for 1M
Figure 3:
timesteps. (b) The performance comparison with static fusion methods. (c, d) Mean performance
for 1M timesteps as a function of K for (c) Walker2d and (d) HalfCheetah. K = 0 means PPO
without intrinsic reward, and K = 1 means the single-model surprise method. (K = 4 yielded
similar performance to that of K = 3, so we omitted the curve of K = 4 for simplicity.)

7

0.00.20.40.60.81.0Timestep(M)500050010001500Average ReturnHalfCheetahK=2, proposedK=2, minK=2, maxK=2, avg0.00.20.40.60.81.0Timestep(M)05001000150020002500Average ReturnWalker2dK=0K=1K=2K=30.00.20.40.60.81.0Timestep(M)500050010001500Average ReturnHalfCheetahK=0K=1K=2K=3Under review as a conference paper at ICLR 2021

4.2.1 LEARNING BEHAVIOR OF FUSION PARAMETER α

We investigated how the fusion parameter α changed adaptively during the training. Fig. 3(a) shows
the learning curve of the fusion parameter α in HalfCheetah. It is seen that starting from the initial
value α = 0, the fusion parameter α increases until it reaches approximately 5, maintains the level
until approximately 180 iterations (0.4 million timesteps), and then decreases monotonically. The
proposed fusion learning method takes relatively more aggressive fusion strategies with α being
around 5 (but this is not the too aggressive maximum corresponding to α = ∞) in the early stage of
learning. Then, the fusion learning takes more and more conservative fusion strategy by decreasing α
more and more to large negative values (i.e., towards minimum taking). This observation is consistent
with the general behavior of RL that aggressive exploration is essential in the early stage of learning
and conservative exploitation has a more considerable weight in the later stage of learning.

As seen in Fig. 3(b), in the ﬁxed fusion case, the method using the average has higher performance
than that with minimum or maximum in the early stage of training. However, the minimum selection
method yields better performance than average or maximum at the later stage. It is seen that the
proposed adaptive fusion yields the best performance because the proposed adaptive fusion takes
advantage of both fast performance improvement in the early stage and high ﬁnal performance at the
end by learning α optimally.

In order to see the difference between the proposed α-fusion learning and other fusion learning
method, we considered a fusion method directly using neural networks. In the considered method, we
designed a neural network fusion function fξ(x1, · · · , xK) of K inputs with (i) linear activation or
(ii) nonlinear (tanh) activation. In both cases, fξ has a single hidden layer of size 2K. It is observed
that our proposed method outperforms the fusion with learned neural networks using the same KLD
model error input. See Appendix E for the comparison result.

4.2.2 EFFECT OF THE NUMBER OF PREDICTION MODELS

We investigated the impact of the model order K. Since we adopt Gaussian distributions for the
prediction models Pφ1, · · · , PφK , the mixture PK is a Gaussian mixture for given state-action pair
(s, a). According to a recent result (Haarnoja et al., 2018), the model order of a Gaussian mixture
need not be too large to capture the true transition probability distribution effectively in practice. Thus,
we evaluated the performance for K = 1, 2, 3, 4. Fig. 3(c) and 3(d) show the mean performance as a
function of K in Walker2d and HalfCheetah. The performance improves as K increases. Once the
proper model order is reached, the performance does not improve further due to more difﬁcult model
estimation for higher model orders, as expected from our intuition. From this result, we found that
K = 2 or 3 seems proper for all the six environments considered in Section 4.1.

5 CONCLUSION

In this paper, we proposed a new adaptive fusion method with multiple prediction models for sparse-
reward RL. The mixture of multiple prediction models is used to better approximate the unknown
transition probability, and the intrinsic reward is generated by adaptive fusion learning with multiple
prediction error values. The ablation study shows that the general principle of RL is valid even in the
adaptive fusion that we need to take a more aggressive strategy in the early stage and less aggressive
strategy in the later stage. Numerical results show that the proposed method outperforms existing
intrinsic reward generation methods in the considered sparse environments. The proposed adaptive
fusion structure is useful not only to the speciﬁc problem considered here but also to other problems
involving numeric fusion with fusion learning.

8

Under review as a conference paper at ICLR 2021

REFERENCES

Joshua Achiam and Shankar Sastry. Surprise-based intrinsic motivation for deep reinforcement

learning. arXiv preprint arXiv:1703.01732, 2017.

Shun-ichi Amari. Integration of stochastic models by minimizing α-divergence. Neural computation,

19(10):2780–2796, 2007.

Shun-ichi Amari. Information geometry and its applications, volume 194. Springer, 2016.

Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob
McGrew, Josh Tobin, OpenAI Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay.
In Advances in Neural Information Processing Systems, pp. 5048–5058, 2017.

Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and

Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.

Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network
In International Conference on Learning Representations, 2019. URL https:

distillation.
//openreview.net/forum?id=H1lJJnR5Ym.

Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine. Deep reinforcement
In Advances in Neural

learning in a handful of trials using probabilistic dynamics models.
Information Processing Systems, pp. 4754–4765, 2018.

Cédric Colas, Olivier Sigaud, and Pierre-Yves Oudeyer. GEP-PG: Decoupling exploration and
exploitation in deep reinforcement learning algorithms. In Jennifer Dy and Andreas Krause (eds.),
Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings
of Machine Learning Research, pp. 1039–1048, Stockholmsmässan, Stockholm Sweden, 10–15
Jul 2018. PMLR. URL http://proceedings.mlr.press/v80/colas18a.html.

Ildefons Magrans de Abril and Ryota Kanai. Curiosity-driven reinforcement learning with homeostatic

regulation. arXiv preprint arXiv:1801.07440, 2018.

Arthur P. Dempster, Nan M. Laird, and Donald B. Rubin. Maximum likelihood from incomplete data
via the EM algorithm. Journal of the Royal Statistical Society: Series B (Methodological), 39(1):
1–22, 1977.

Prafulla Dhariwal, Christopher Hesse, Oleg Klimov, Alex Nichol, Matthias Plappert, Alec Radford,
John Schulman, Szymon Sidor, and Yuhuai Wu. Openai baselines. https://github.com/
openai/baselines, 2017.

Dror Freirich, Tzahi Shimkin, Ron Meir, and Aviv Tamar. Distributional multivariate policy evaluation
and exploration with the bellman gan. In International Conference on Machine Learning, pp.
1983–1992. PMLR, 2019.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy
maximum entropy deep reinforcement learning with a stochastic actor.
In Jennifer Dy and
Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning,
volume 80 of Proceedings of Machine Learning Research, pp. 1861–1870, Stockholmsmässan,
Stockholm Sweden, 10–15 Jul 2018. PMLR. URL http://proceedings.mlr.press/
v80/haarnoja18b.html.

Godfrey Harold Hardy, John Edensor Littlewood, György Pólya, DE Littlewood, G Polya, et al.

Inequalities. Cambridge university press, 1952.

R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam
Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation
and maximization. In International Conference on Learning Representations, 2019. URL https:
//openreview.net/forum?id=Bklr3j0cKX.

Zhang-Wei Hong, Tzu-Yun Shann, Shih-Yang Su, Yi-Hsiang Chang, Tsu-Jui Fu, and Chun-Yi Lee.
Diversity-driven exploration strategy for deep reinforcement learning. In Advances in Neural
Information Processing Systems, pp. 10489–10500, 2018.

9

Under review as a conference paper at ICLR 2021

Rein Houthooft, Xi Chen, Yan Duan, John Schulman, Filip De Turck, and Pieter Abbeel. VIME:
Variational information maximizing exploration. In Advances in Neural Information Processing
Systems, pp. 1109–1117, 2016.

Hyoungseok Kim, Jaekyeom Kim, Yeonwoo Jeong, Sergey Levine, and Hyun Oh Song. Emi:
Exploration with mutual information. In International Conference on Machine Learning, pp.
3360–3369, 2019.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint

arXiv:1412.6980, 2014.

Thanard Kurutach, Ignasi Clavera, Yan Duan, Aviv Tamar, and Pieter Abbeel. Model-ensemble
trust-region policy optimization. In International Conference on Learning Representations, 2018.
URL https://openreview.net/forum?id=SJJinbWRZ.

Anusha Nagabandi, Gregory Kahn, Ronald S. Fearing, and Sergey Levine. Neural network dy-
namics for model-based deep reinforcement learning with model-free ﬁne-tuning. arXiv preprint
arXiv:1708.02596, 2017.

Junhyuk Oh, Yijie Guo, Satinder Singh, and Honglak Lee. Self-imitation learning. In Jennifer Dy and
Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning,
volume 80 of Proceedings of Machine Learning Research, pp. 3878–3887, Stockholmsmässan,
Stockholm Sweden, 10–15 Jul 2018. PMLR. URL http://proceedings.mlr.press/
v80/oh18b.html.

Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, and Trevor Darrell. Curiosity-driven exploration by
self-supervised prediction. In International Conference on Machine Learning (ICML), volume
2017, 2017.

Deepak Pathak, Dhiraj Gandhi, and Abhinav Gupta. Self-supervised exploration via disagreement.

In International Conference on Machine Learning, pp. 5062–5071, 2019.

Alfréd Rényi et al. On measures of entropy and information. In Proceedings of the Fourth Berkeley
Symposium on Mathematical Statistics and Probability, Volume 1: Contributions to the Theory of
Statistics. The Regents of the University of California, 1961.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region
policy optimization. In International Conference on Machine Learning, pp. 1889–1897, 2015a.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional
continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438,
2015b.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy

optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Lior Shani, Yonathan Efroni, and Shie Mannor. Exploration conscious reinforcement learning
In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th
revisited.
International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning
Research, pp. 5680–5689, Long Beach, California, USA, 09–15 Jun 2019. PMLR. URL http:
//proceedings.mlr.press/v97/shani19a.html.

Pranav Shyam, Wojciech Ja´skowski, and Faustino Gomez. Model-based active exploration.
In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th Interna-
tional Conference on Machine Learning, volume 97 of Proceedings of Machine Learning
Research, pp. 5779–5788, Long Beach, California, USA, 09–15 Jun 2019. PMLR. URL
http://proceedings.mlr.press/v97/shyam19a.html.

Haoran Tang, Rein Houthooft, Davis Foote, Adam Stooke, OpenAI Xi Chen, Yan Duan, John
Schulman, Filip DeTurck, and Pieter Abbeel. # Exploration: A study of count-based exploration
for deep reinforcement learning. In Advances in Neural Information Processing Systems, pp.
2750–2759, 2017.

10

Under review as a conference paper at ICLR 2021

Arash Tavakoli, Fabio Pardo, and Petar Kormushev. Action branching architectures for deep rein-

forcement learning. In AAAI, 2018.

Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control.
In Intelligent Robots and Systems (IROS), 2012 IEEE/RSJ International Conference on, pp. 5026–
5033. IEEE, 2012.

Christopher K. I. Williams and Carl Edward Rasmussen. Gaussian processes for machine learning.

MIT Press Cambridge, MA, 2006.

Sirui Xie, Junning Huang, Lanxin Lei, Chunxiao Liu, Zheng Ma, Wei Zhang, and Liang Lin.
NADPEx: An on-policy temporally consistent exploration method for deep reinforcement learning.
In International Conference on Learning Representations, 2019. URL https://openreview.
net/forum?id=rkxciiC9tm.

Zhongwen Xu, Hado P van Hasselt, and David Silver. Meta-gradient reinforcement learning. In

Advances in Neural Information Processing Systems, pp. 2396–2407, 2018.

Zeyu Zheng, Junhyuk Oh, and Satinder Singh. On learning intrinsic rewards for policy gradient

methods. In Advances in Neural Information Processing Systems, pp. 4644–4654, 2018.

11

Under review as a conference paper at ICLR 2021

A ALGORITHM

Algorithm 1 Sparse RL with Fusion Learning with Multiple Prediction Models for Intrinsic Reward
Generation: PPO Case
1: L : batch size for policy training, L(cid:48) : batch size for model training.
2: Lmini : minibatch size for policy training, L(cid:48)
3: N : epoch size for policy training, N (cid:48) : epoch size for model training.
4: M AX : the maximum index of batch period l, K : the number of prediction models.
5: Initialize the policy πθ0, the K transition probability models Pφ1

mini : minibatch size for model training.

, the mixing coefﬁ-

, · · · , PφK

0

0

cients q1, · · · , qK, the fusion parameter α, and the extrinsic value function parameter ζ.

, · · · , PφK

6: Generate trajectories with πθ0 and store them to the initially empty replay buffer M .
7: for Batch period l = 0, · · · , M AX − 1 do
8:

Train Pφ1
performing iterations with (11). For each Pφi
randomly and uniformly from M , and perform the updates with minibatches of size L(cid:48)
drawn from D(cid:48)
for Timestep t = lL, lL + 1, · · · , lL + L − 1 do

by performing gradient updates for (10), and update q1, · · · , qK by
i of size L(cid:48)
mini

(1 ≤ i ≤ K), we draw a batch D(cid:48)

i for N (cid:48) epochs.

l

l

l

Collect st from the environment and at with the policy πθl .
Collect st+1 and the extrinsic reward rt from the environment and add (st, at, st+1) to M .

end for
Acquire the intrinsic reward rt,int of the current batch D of size L. (Detail is described in
Appendix C.)
Train πθl by using PPO with the total rewards (12) and minibatch size Lmini for N epochs.
Train α and ζ by using the parameter learning method described in Section 3.3 with the total
rewards (12) and minibatch size Lmini for N epochs.

9:
10:
11:
12:
13:

14:
15:

16: end for

12

Under review as a conference paper at ICLR 2021

B AN EXAMPLE OF α-MEAN WITH VARYING α

.

The α-mean includes all numeric fusions with the scale-free property such as minimum, maximum,
and conventional mean functions such as arithmetic, geometric, and harmonic mean, by varying α.
For example, the α-mean fα of two values x1 and x2 as a function of α is given as

fα(x1, x2) =






max{x1, x2} (maximum)
x1+x2
(arithmetic)
2
√
x1x2 (geometric)
1 +x−1
2

(cid:16) x−1

(harmonic)

(cid:17)−1

2

if α = −∞
if α = −1
if α = 1

.

if α = 3

(13)

min{x1, x2} (minimum)

if α = ∞

The ﬁgure below shows the α-mean of x1 = 1 and x2 = 5 with respect to α. The α-mean can
implement all the possible fusion of K values into a single value under the condition of scale-freeness
by varying α.

Figure 4: α-mean of x1 = 1 and x2 = 5

13

1510505101.01.52.02.53.03.54.04.55.0x1x2f(x1,x2)x1+x22x1x2(x11+x122)1Under review as a conference paper at ICLR 2021

C 1-STEP TECHNIQUE AND NORMALIZATION FOR rj

t,int

For actual computation of the intrinsic reward, we ﬁrst applied the 1-step technique (Achiam &
Sastry, 2017). Note that

rj
t,int(st, at, st+1) = log

PK(st+1|st, at)
Pφj (st+1|st, at)

(14)

where

K
(cid:88)

qiPφi.

PK =

i=1
With the 1-step technique, the batch index for the numerator in (14) is the current batch index l(t)
corresponding to time step t, whereas the batch index for the denominator in (14) is the previous
batch index l(t) − 1. Hence, with the 1-step technique, rj

t,int(st, at, st+1) is modiﬁed as

log

(cid:80)K

i=1 qiPφi
Pφj

l(t)−1

l(t)

(st+1|st, at)

(st+1|st, at)

.

(15)

Then, this modiﬁed j-th model’s prediction error value is applied to the α-fusion in Section 3.3.

To improve numerical stability, we further applied the normalization technique (Achiam & Sastry,
2017) to the output fα(st, at, st+1) of the α-fusion. Thus, the ﬁnal actual intrinsic reward given the
current batch D is expressed as

rt,int(st, at, st+1) =

max

where (16) describes the applied normalization.

fα(st, at, st+1)

(cid:110) | (cid:80)

(s,a,s(cid:48) )∈D fα(s,a,s(cid:48))|

|D|

(cid:111) ,

, 1

(16)

14

Under review as a conference paper at ICLR 2021

D NEURAL NETWORK ARCHITECTURE AND HYPERPARAMETERS

For actual implementation, the code implemented by Dhariwal et al. (2017) and Zheng et al. (2018)
are used. The policy, the prediction models, the value function of total reward, and the value function
of extrinsic reward Vζ were designed by fully-connected neural networks, all of which had two hidden
layers of size (64, 64) (Houthooft et al., 2016; Dhariwal et al., 2017; Tang et al., 2017). The tanh
activation function was used for all of the networks (Achiam & Sastry, 2017; Dhariwal et al., 2017).
The means of the fully factorized Gaussian prediction models were the outputs of our networks, and
the variances were trainable variables that were initialized to 1 (Dhariwal et al., 2017). Other than the
variances, all initialization is randomized so that each of the prediction models was set differently
(Kurutach et al., 2018; Tavakoli et al., 2018). For the implementation of the policy model and the
value function, our method and all the considered intrinsic reward generation method used the same
code for the module method (Zheng et al., 2018).

Although a recent work (Achiam & Sastry, 2017) used TRPO (Schulman et al., 2015a) as the baseline
learning engine, we used PPO (Schulman et al., 2017), one of the currently most popular algorithms
for continuous action control, as our baseline algorithm. While the same basic hyperparameters as
those in the previous work (Achiam & Sastry, 2017) were used, some hyperparameters were tuned
for PPO. λ for the GAE method (Schulman et al., 2015b) was ﬁxed to 0.95, while the discounting
factor was set to γ = 0.99. The batch size L for the training of the policy was ﬁxed to 2048. For the
policy update using PPO, the minibatch size Lmini was set to 64, the epoch number N 10, the value
function coefﬁcient 0.5, the clipping constant 0.2, and the entropy coefﬁcient 0.0. The initial values
of the learning rates of Adam optimizer (Kingma & Ba, 2014) for updating the policy parameter θ
and the extrinsic value function parameter ζ were ﬁxed to 0.0003 and 0.0001, respectively. The initial
value of the learning rate of α was 0.01 for Hopper, HalfCheetah, Humanoid, and InvertedPendulum,
and 0.001 for Ant and Walker2d. The three learning rates were linearly decayed as timestep passed
so that the values at the end of the training was 0. The initial value of α was set to 0.

Each of the single-model surprise method, the hashing method, and our proposed method requires
a replay buffer. The size of the used replay buffer for all these three methods is 1.1M. If the buffer
becomes full, the earlier samples are deleted ﬁrst. Before the beginning of the iterations, 2048 × B
samples from real trajectories generated by the initial policy were added to the replay buffer. We
set B = 40 for our experiments. For the methods not requiring a replay buffer, i.e., Curiosity,
Information Gain, Module, Disagreement, and PPO Only, we ran 2048 × B = 81920 timesteps
before measuring performance for fair comparison.
For the prediction model learning, we set the batch size L(cid:48) = 2048, L(cid:48)
mini = 64, and N (cid:48) = 4. The
optimization (10) was solved based on second-order approximation (Schulman et al., 2015a). When
K = 1, the optimization (10) reduces to the model learning problem in Achiam & Sastry (2017). In
Achiam & Sastry (2017), the constraint constant κ in the second-order optimization was well-tuned
as 0.001. Therefore, we used this value of κ not only to the case of K = 1 but also to the case of
K ≥ 2. We further tuned the value of creg in (10) for each environment, and we set creg = 0.01 .
For the information gain method, we need another hyperparameter h which is the weight to balance
the original intrinsic reward and the homeostatic regulation term (de Abril & Kanai, 2018). We
tuned this hyperparameter for each environment and the used value of h is shown in Table 1. For the
disagreement method, we used ﬁve deterministic models.

Curiosity
Hashing
Information Gain
Single Surprise
Disagreement

Ant
β = 0.01
β = 0.0001
β = 0.01, h = 4
β = 0.00001
β = 0.00003

Hopper
β = 0.01
β = 0.01
β = 0.1, h = 4
β = 0.02
β = 0.003

HalfCheetah
β = 0.0001
β = 0.00001
β = 0.0001, h = 4
β = 0.0003
β = 0.003

Humanoid
β = 0.1
β = 0.1
β = 0.01, h = 2
β = 0.1
β = 0.0003

InvertedPendulum Walker2d
β = 0.03
β = 0.003
β = 0.003
β = 0.0001
β = 0.03, h = 2
β = 0.0001, h = 4
β = 0.02
β = 0.0001
β = 0.1
β = 0.001

Table 1: Used hyperparameter values.

Table 1 summarizes the weighting factor β as well as the hyperparameter h in information gain
method. The weighting factor β in (12) between the extrinsic reward and the intrinsic reward should
be determined for all intrinsic reward generation methods. Since each of the considered methods
yields different scale of the intrinsic reward, we used the optimized weighting factor β in (12)
for each algorithm for each environment by testing β according to log scale (Zheng et al., 2018):

15

Under review as a conference paper at ICLR 2021

{1.0, 0.5, 0.3, 0.2, 0.1, · · · , 10−6}. In the single-model surprise method and the proposed method,
the proposed method employed the same hyperparameters as the single-model surprise method. We
conﬁrmed that the hyperparameters of the other ﬁve methods were well-tuned in the original papers
(Pathak et al., 2017; Tang et al., 2017; de Abril & Kanai, 2018; Zheng et al., 2018; Pathak et al.,
2019), and we used the hyperparameters provided by these methods.

For the intrinsic reward module method, we checked that the open-source code reproduced results in
Zheng et al. (2018), as shown in Fig. 5. ‘Module 0.01’ represents the module method with training
using the sum of intrinsic reward and the scaled extrinsic reward with scaling factor 0.01. ‘Module
0’ represents training using intrinsic reward only (no addition of extrinsic reward). Both methods
are introduced in Zheng et al. (2018), and we checked reproducibility when B = 0, i.e., we ran
2048 × B = 0 timesteps before measuring performance. We observed that our used code yielded the
same results as those in Zheng et al. (2018) (InvertedPendulum is not considered in this paper).

Thus, we used this code for the module method with only one change that we ran 2048 × B timesteps
with B = 40 before measuring performance for fair comparison. (Since the range of intrinsic reward
from the module method is [−1, 1], intrinsic reward normalization in (16) is not needed.) For the
module method in performance comparison 4.1, we selected a better method between ‘Module 0’ and
‘Module 0.01’, assuming B = 40. ‘Module 0’ performed better than ‘Module 0.01’ in Hopper and
Walker2d, and ‘Module 0.01’ performed better than ‘Module 0’ in the other three environments. We
also observed that both ‘Module 0’ and ‘Module 0.01’ performed poorly in InvertedPendulum, which
is not considered in Zheng et al. (2018). Therefore, we further ﬁne-tuned both of the two scaling
coefﬁcients of extrinsic and intrinsic reward and set the best values of 1.0 and 0.003, respectively.

Figure 5: Reproduced mean performance of the module method over 10 random seeds with ∆ = 40
when B = 0.

16

0.00.20.40.60.81.0Timestep(M)4003002001000Average ReturnAnt (B=0)PPO OnlyModule_0Module_0.010.00.20.40.60.81.0Timestep(M)05001000150020002500Average ReturnHopper (B=0)PPO OnlyModule_0Module_0.010.00.20.40.60.81.0Timestep(M)5000500100015002000Average ReturnHalfCheetah (B=0)PPO OnlyModule_0Module_0.010.00.20.40.60.81.0Timestep(M)100200300400500Average ReturnHumanoid (B=0)PPO OnlyModule_0Module_0.010.00.20.40.60.81.0Timestep(M)05001000150020002500Average ReturnWalker2d (B=0)PPO OnlyModule_0Module_0.01Under review as a conference paper at ICLR 2021

E COMPARISON TO DIRECT NEURAL NETWORK-BASED FUSION

Figure 6: Performance comparison between our proposed method (blue curve) and fusion with
neural network learning with nonlinear activation (green) when K = 2. (Fusion with neural network
learning with linear activation performed worse than the nonlinear activation, so we omitted the
result.) All simulations were conducted over ten ﬁxed random seeds.

In order to compare our fusion method to the fusion with neural network learning, we designed a
neural network fusion function fξ(x1, · · · , xK) of K = 2 inputs with (i) linear activation or (ii)
nonlinear (tanh) activation. In both cases, fξ has a single hidden layer of size 2K. Note that our fusion
2 y)
function with a learnable α is explicitly expressed as f (x, y) = − 2
]
for KLD error approximation inputs by the exponentiation at the input stage. Fig. 6 shows that our
method outperforms the fusion with neural network learning using the same KLD model error input.

1−α log[ exp (− 1−α

2 x)+exp (− 1−α

2

(a)

(b)

Figure 7: Visualizations of our fusion function and the fusion method based on neural network
learning (both were trained in the Hopper environment). The gray surface is the (x1, x2) input plane,
and and the y-axis is the fusion output: (a) the proposed fusion function (blue) and the linear neural
network fusion function (green). (b) The proposed fusion function (blue) and the nonlinear neural
network fusion function (green).

In Fig. 7, we plot both the proposed fusion function y = fα(x1, x2) and the neural-network-based
fusion function y = fξ(x1, x2) after training in the Hopper environment. Figs. 7(a) and 7(b) shows
the comparison in the linear and non-linear hyperbolic tangent activation cases, respectively. As
expected, in the linear activation case, (x1, x2, y) forms a hyperplane. Although the hyperplane is
ﬁt in a best way, still it cannot perform properly as a fusion function. For example, permutation
invariance is a property of a good fusion function. However, a single linear layer network cannot
achieve this property although the proposed α-fusion has the permutation invariance property. Now,

17

0.00.20.40.60.81.0Timestep(M)05001000150020002500Average ReturnWalker2dProposed MethodFusionnet0.00.20.40.60.81.0Timestep(M)05001000150020002500Average ReturnHopperProposed MethodFusionnet0.00.20.40.60.81.0Timestep(M)02004006008001000Average ReturnInvertedPendulumProposed MethodFusionnet0.00.20.40.60.81.0Timestep(M)500050010001500Average ReturnHalfCheetahProposed MethodFusionnet0.00.20.40.60.81.0Timestep(M)5004003002001000Average ReturnAntProposed MethodFusionnet0.00.20.40.60.81.0Timestep(M)100200300400500Average ReturnHumanoidProposed MethodFusionnet𝑥1𝑥2y𝑥2𝑥1yUnder review as a conference paper at ICLR 2021

consider the case of non-linear hyperbolic tangent activation. In this case, as seen in Fig. 7(b), the
learned fusion function seems a bit better than the linear activation case but is not still symmetric
around the line x1 = x2. Hence, it is not permutation invariant either. It seems that the neural-
network-based fusion function requires more complexity and more learning time. The proposed
adaptive α-fusion structure captures the fusion behavior only by using a single parameter α and the
corresponding learning is efﬁcient.

18

Under review as a conference paper at ICLR 2021

F FUSION WITH DIFFERENT PREDICTION ERRORS: THE MEAN SQUARED

ERROR (MSE) CASE

Figure 8: Performance comparison between the proposed method using KLD (blue) and the proposed
method using MSE (green). All simulations were conducted over ten ﬁxed random seeds.

We compared the proposed method using KLD with the proposed method using MSE. In the MSE
method, we trained ten deterministic models predicting the next state. These multiple models are
differently initialized and independently trained with different batch data as widely done in other
works (Nagabandi et al., 2017; Kurutach et al., 2018). Then, we calculated the MSE between the
actual next state and the predicted next state as the error measure.

Fig. 8 shows that the proposed method using KLD outperforms the proposed method using MSE.
Since the KLD uses joint training among multiple probabilistic models, the KLD method can reﬂect
the underlying dynamics more accurately. On the other hand, independently trained deterministic
models were not diverse enough to effectively capture the underlying dynamics as compared to the
joint training.

19

0.00.20.40.60.81.0Timestep(M)05001000150020002500Average ReturnWalker2dProposed MethodProposed Method_MSE0.00.20.40.60.81.0Timestep(M)05001000150020002500Average ReturnHopperProposed MethodProposed Method_MSE0.00.20.40.60.81.0Timestep(M)020040060080010001200Average ReturnInvertedPendulumProposed MethodProposed Method_MSE0.00.20.40.60.81.0Timestep(M)500050010001500Average ReturnHalfCheetahProposed MethodProposed Method_MSE0.00.20.40.60.81.0Timestep(M)4003002001000100Average ReturnAntProposed MethodProposed Method_MSE0.00.20.40.60.81.0Timestep(M)100200300400500Average ReturnHumanoidProposed MethodProposed Method_MSEUnder review as a conference paper at ICLR 2021

G NECESSITY OF SCALE-FREE PROPERTY (CONDITION 2)

Note that basically the fusion function is some kind of averaging function. Condition 2 is a proper
condition for any reasonable averaging function (Hardy et al., 1952). The scale-free property is also
called the homogeneous property.

Suppose that the homogeneous property for the function function f is not satisﬁed and assume the
nonhomogeneous relationship:

f (cx1, cx2) = c(cid:48)f (x1, x2),

where c(cid:48) ((cid:54)= c) is mapped for each scaling factor c for the completeness of the scaling operation.
Now suppose that there exist two constants c, c(cid:48) such that c > 1 and 0 < c(cid:48) < 1 and f (cx1, cx2) =
c(cid:48)f (x1, x2) for two input x1 and x2. Then, the monotonicity is broken. The two inputs x1 and x2 are
increased as cx1 and cx2 but the corresponding output is reduced as c(cid:48)f (x1, x2) as compared to the
original output f (x1, x2). This is not the situation that we want. Furthermore, by repeatedly applying
equation 17, we have

(17)

(18)

yielding

lim
n→∞

[f (cnx1, cnx2) = (c(cid:48))nf (x1, x2)]

f (∞, ∞) = 0 · f (x1, x2) = 0.
So, we have a contradiction. In the case of 0 < c < 1 and c(cid:48) > 1, we have a similar contradiction:

(19)

f (0, 0) = ∞ · f (x1, x2) = ∞.

(20)

Now assume that there exist c, c(cid:48) > 1 and c (cid:54)= c(cid:48). Set two inputs as x1 = x2 = x. Then, we have

(cid:52)
= f (x, x)

g(x)
g(cx) = f (cx, cx) = c(cid:48)f (x, x) = c(cid:48)g(x)

By repeating the iteration, we have

g(cnx) = (c(cid:48))ng(x).

(21)

(22)

(23)

In equation (23), the input to the function g(x) is exponentially increasing as cn and the output
increases exponentially as (c(cid:48))n. Such function g(x) is uniquely given by the form

where ∼ means the scaling equivalence. However, for a different pair ˜c and ˜c(cid:48), we also require

g(x) ∼ (c(cid:48))logc x,

g(x) ∼ (˜c(cid:48))log˜c x.

(24)

(25)

The two functions in (24) and (25) cannot be the same in general. Furthermore, g(x) should be the
same for all pairs (c, c(cid:48)). This cannot be satisﬁed in general. So, we have an indiscrepancy in the
nonhomogeneous case. A similar situation happens for 0 < c, c(cid:48) < 1. However, note that if we have
c = c(cid:48) for all scaling factor c, then the two functions in (24) and (25) are consistent. In this case, we
have

g(x) ∼ (c)logc x = x,
and this makes sense because if the input values are all the same, the output should be the same as the
input. Please note that the generalized mean or α-mean exactly satisﬁes the scaling behavior (26).

(26)

20

Under review as a conference paper at ICLR 2021

H EXPLORATION WITHOUT INTRINSIC REWARD GENERATION

Recent indirect methods can further be classiﬁed mainly into two groups: (i) revising the original ob-
jective function to stimulate exploration, exploiting intrinsic motivation implicitly, and (ii) perturbing
the parameter space of policy.

In the ﬁrst group, Andrychowicz et al. (2017) suggested additional sampling states from the replay
buffer and setting those data as new goals for sparse and binary extrinsic reward environments. In
their work, the policy was based on the input of both state and goal. In our work, on the other hand,
the goal concept is not necessary. Oh et al. (2018) proposed that exploration can be stimulated by
exploiting novel state-action pairs from the past and used sparse-reward environments by delaying
extrinsic rewards. Hong et al. (2018) revised the original objective function for training by considering
the maximization of the divergence between the current policy and recent policies, with an adaptive
scaling technique. The dropout concept was applied to the PPO algorithm to encourage the stochastic
behavior of the agent episode-wisely (Xie et al., 2019). Convex combination of the target policy
and any given policy is considered a new exploratory policy (Shani et al., 2019), which corresponds
to solving a surrogate Markov Decision Process, generalizing usual exploration methods such as
(cid:15)-greedy or Gaussian noise.

In the second group, Colas et al. (2018) proposed a goal-based exploration method for continuous
control, which alternates generating parameter-outcome pair and perturbing speciﬁc parameters based
on randomly drawn goal from the outcome space. Recently, inspired by Chua et al. (2018), Shyam
et al. (2019) considered pure exploration MDP without any extrinsic reward with the notion of utility,
where utility is based on JSD and the Jensen-Rényi divergence (Rényi et al., 1961). They considered
several models for the transition functions in this work, but they used this to compute utility based on
the multiple models’ average entropy.

21

