Published as a conference paper at ICLR 2021

HIERARCHICAL REINFORCEMENT LEARNING BY
DISCOVERING INTRINSIC OPTIONS

Jesse Zhang∗ † 1, Haonan Yu∗ 2, Wei Xu2
1University of Southern California, 2Horizon Robotics

ABSTRACT

We propose a hierarchical reinforcement learning method, HIDIO, that can learn
task-agnostic options in a self-supervised manner while jointly learning to utilize
them to solve sparse-reward tasks. Unlike current hierarchical RL approaches
that tend to formulate goal-reaching low-level tasks or pre-deﬁne ad hoc lower-
level policies, HIDIO encourages lower-level option learning that is independent
of the task at hand, requiring few assumptions or little knowledge about the task
structure. These options are learned through an intrinsic entropy minimization ob-
jective conditioned on the option sub-trajectories. The learned options are diverse
and task-agnostic. In experiments on sparse-reward robotic manipulation and nav-
igation tasks, HIDIO achieves higher success rates with greater sample efﬁciency
than regular RL baselines and two state-of-the-art hierarchical RL methods. Code
available at https://www.github.com/jesbu1/hidio.

1

INTRODUCTION

Imagine a wheeled robot learning to kick a soccer ball into a goal with sparse reward supervision.
In order to succeed, it must discover how to ﬁrst navigate in its environment, then touch the ball,
and ﬁnally kick it into the goal, only receiving a positive reward at the end for completing the task.
This is a naturally difﬁcult problem for traditional reinforcement learning (RL) to solve, unless the
task has been manually decomposed into temporally extended stages where each stage constitutes
a much easier subtask. In this paper we ask, how do we learn to decompose the task automatically
and utilize the decomposition to solve sparse reward problems?

Deep RL has made great strides solving a variety of tasks recently, with hierarchical RL (hRL)
demonstrating promise in solving such sparse reward tasks (Sharma et al., 2019b; Le et al., 2018;
Merel et al., 2019; Ranchod et al., 2015).
In hRL, the task is decomposed into a hierarchy of
subtasks, where policies at the top of the hierarchy call upon policies below to perform actions to
solve their respective subtasks. This abstracts away actions for the policies at the top levels of the
hierarchy. hRL makes exploration easier by potentially reducing the number of steps the agent needs
to take to explore its state space. Moreover, at higher levels of the hierarchy, temporal abstraction
results in more aggressive, multi-step value bootstrapping when temporal-difference (TD) learning
is employed. These beneﬁts are critical in sparse reward tasks as they allow an agent to more easily
discover reward signals and assign credit.

Many existing hRL methods make assumptions about the task structure (e.g., fetching an object
involves three stages: moving towards the object, picking it up, and combing back), and/or the skills
needed to solve the task (e.g., pre-programmed motor skills) (Florensa et al., 2016; Riedmiller et al.,
2018; Lee et al., 2019; Hausman et al., 2018; Lee et al., 2020; Sohn et al., 2018; Ghavamzadeh &
Mahadevan, 2003; Nachum et al., 2018). Thus these methods may require manually designing the
correct task decomposition, explicitly formulating the option space, or programming pre-deﬁned
options for higher level policies to compose. Instead, we seek to formulate a general method that
can learn these abstractions from scratch, for any task, with little manual design in the task domain.

The main contribution of this paper is HIDIO (HIerarchical RL by Discovering Intrinsic Options), a
hierarchical method that discovers task-agnostic intrinsic options in a self-supervised manner while

∗Denotes equal contribution. Email to jessez@usc.edu, {haonan.yu,wei.xu}@horizon.ai
†Work done as an intern at Horizon Robotics.

1

Published as a conference paper at ICLR 2021

learning to schedule them to accomplish environment tasks. The latent option representation is
uncovered as the option-conditioned policy is trained, both according to the same self-supervised
worker objective. The scheduling of options is simultaneously learned by maximizing environment
reward collected by the option-conditioned policy. HIDIO can be easily applied to new sparse-
reward tasks by simply re-discovering options. We propose and empirically evaluate various in-
stantiations of the option discovery process, comparing the resulting options with respect to their
ﬁnal task performance. We demonstrate that HIDIO is able to efﬁciently learn and discover di-
verse options to be utilized for higher task reward with superior sample efﬁciency compared to other
hierarchical methods.

2 PRELIMINARIES

We consider the reinforcement learning (RL) problem in a Markov Decision Process (MDP). Let
s ∈ RS be the agent state. We use the terms “state” and “observation” interchangeably to denote the
environment input to the agent. A state can be fully or partially observed. Without loss of generality,
we assume a continuous action space a ∈ RA for the agent. Let πθ(a|s) be the policy distribution
with learnable parameters θ, and P(st+1|st, at) the transition probability that measures how likely
the environment transitions to st+1 given that the agent samples an action by at ∼ πθ(·|st). After
the transition to st+1, the agent receives a deterministic scalar reward r(st, at, st+1).

The objective of RL is to maximize the sum of discounted rewards with respect to θ:
(cid:34) ∞
(cid:88)

(cid:35)
γtr(st, at, st+1)

E
πθ,P

t=0

(1)

where γ ∈ [0, 1] is a discount factor. We will omit P in the expectation for notational simplicity.

In the options framework (Sutton et al., 1999), the agent can switch between different options during
an episode, where an option is translated to a sequence of actions by an option-conditioned policy
with a termination condition. A set of options deﬁned over an MDP induces a hierarchy that models
temporal abstraction. For a typical two-level hierarchy, a higher-level policy produces options, and
the policy at the lower level outputs environment actions conditioned on the proposed options. The
expectation in Eq. 1 is taken over policies at both levels.

3 HIERARCHICAL RL BY DISCOVERING INTRINSIC OPTIONS

We now introduce our hierarchical method
for solving sparse reward tasks. We as-
sume little prior knowledge about the task
structure, except that it can be learned
through a hierarchy of two levels. The
higher-level policy (the scheduler πθ), is
trained to maximize environment reward,
while the lower-level policy (the worker
πφ) is trained in a self-supervised manner
to efﬁciently discover options that are uti-
lized by πθ to accomplish tasks.
Impor-
tantly, by self-supervision the worker gets
access to dense intrinsic rewards regard-
less of the sparsity of the extrinsic rewards.

Figure 1: The overall framework of HIDIO. The sched-
uler πθ samples an option uh every K (3 in this case)
time steps, which is used to guide the worker πφ to di-
rectly interact in the environment conditioned on uh
and the current sub-trajectory sh,k, ah,k−1. The sched-
uler receives accumulated environment rewards Rh,
while the worker receives intrinsic rewards rlo
h,k+1. Re-
fer to Eq. 2 for sampling and Eqs. 3 and 5 for training.

Without
loss of generality, we assume
that each episode has a length of T and
the scheduler outputs an option every K
steps. The scheduled option u ∈ [−1, 1]D
(where D is a pre-deﬁned dimensional-
ity), is a latent representation that will be
learned from scratch given the environment task. Modulated by u, the worker executes K steps
before the scheduler outputs the next option. Let the time horizon of the scheduler be H = (cid:100) T
K (cid:101).
Formally, we deﬁne

2

Scheduler 𝜋𝜃𝑢ℎ,0𝑠ℎ,0)Worker 𝜋𝜙𝑎ℎ,𝑘ҧ𝑠ℎ,𝑘,ത𝑎ℎ,𝑘−1,𝑢ℎ)𝐾TimeDiscriminator 𝑞𝜓(𝑢ℎ|ҧ𝑠ℎ,𝑘+1,ത𝑎ℎ,𝑘)Environment𝑟ℎ,𝑘+1𝑙𝑜𝑅ℎPublished as a conference paper at ICLR 2021

Scheduler policy:
Worker policy:
Environment dynamics:

uh ∼ πθ(·|sh,0),
ah,k ∼ πφ(·|sh,k, uh),
sh,k+1 ∼ P(·|sh,k, ah,k), 0 ≤ h < H, 0 ≤ k < K

0 ≤ h < H
0 ≤ k < K

(2)

where we denote sh,k and ah,k as the k-th state and action respectively, within the h-th option
window of length K. Note that given this sampling process, we have sh,K ≡ sh+1,0, namely, the
last state of the current option uh is the initial state of the next option uh+1. The overall framework
of our method is illustrated in Figure 1.

3.1 LEARNING THE SCHEDULER

Every time the scheduler issues an option uh, it receives an reward Rh computed by accumulating
environment rewards over the next K steps. Its objective is:

max
θ

Eπθ

(cid:34)H−1
(cid:88)

h=0

(cid:35)

βhRh

, where β = γK and Rh = Eπφ

γkr(sh,k, ah,k, sh,k+1)

(3)

(cid:35)

(cid:34)K−1
(cid:88)

k=0

This scheduler objective itself is not a new concept, as similar ones have been adopted by other hRL
methods (Vezhnevets et al., 2017; Nachum et al., 2018; Riedmiller et al., 2018). One signiﬁcant
difference between our option with that of prior work is that our option u is simply a latent variable;
there is no explicit constraint on what semantics u could represent. In contrast, existing methods
usually require their options to reside in a subspace of the state space, to be grounded to the envi-
ronment, or to have known structures, so that the scheduler can compute rewards and termination
conditions for the worker. Note that our latent options can be easily re-trained given a new task.

3.2 LEARNING THE WORKER

The main focus of this paper is to investigate how to effectively learn the worker policy in a self-
supervised manner. Our motivation is that it might be unnecessary to make an option dictate the
worker to reach some “(cid:15)-space” of goals (Vezhnevets et al., 2017; Nachum et al., 2018). As long as
the option can be translated to a short sequence of primitive actions, it does not need to be grounded
with concrete meanings such as goal reaching. Below we will treat the option as a latent variable
that modulates the worker, and propose to learn its latent representation in a hierarchical setting from
the environment task.

3.2.1 WORKER OBJECTIVE

We ﬁrst deﬁne a new meta MDP on top of the original task MDP so that for any h, k, and t:

1) sh,k := (sh,0, . . . , sh,k),
2) ah,k := (ah,0, . . . , ah,k),
3) r(sh,k, ah,k, sh,k+1) := r(sh,k, ah,k, sh,k+1), and
4) P(sh,k+1|sh,k, ah,k) := P(sh,k+1|sh,k, ah,k).

This new MDP equips the worker with historical state and action information since the time (h, 0)
when an option h was scheduled. Speciﬁcally, each state sh,k or action ah,k encodes the history from
the beginning (h, 0) up to (h, k) within the option. In the following, we will call pairs {ah,k, sh,k+1}
option sub-trajectories. The worker policy now takes option sub-trajectories as inputs: ah,k ∼
πφ(·|sh,k, ah,k−1, uh), 0 ≤ k < K, whereas the scheduler policy still operates in the original MDP.
Denote (cid:80)
to minimize the entropy of the option uh conditioned on the option sub-trajectory {ah,k, sh,k+1}:

k=0 for simplicity. The worker objective, deﬁned on this new MDP, is

h,k ≡ (cid:80)H−1

(cid:80)K−1

h=0

max
φ

E
πθ,πφ

(cid:88)

h,k

log p(uh|ah,k, sh,k+1)
(cid:125)
(cid:123)(cid:122)
(cid:124)
negative conditional option entropy

−β log πφ(ah,k|sh,k, ah,k−1, uh)
(cid:123)(cid:122)
(cid:125)
(cid:124)
worker policy entropy

(4)

where the expectation is over the current πθ and πφ but the maximization is only with respect to φ.
Intuitively, the ﬁrst term suggests that the worker is optimized to conﬁdently identify an option given

3

Published as a conference paper at ICLR 2021

a sub-trajectory. However, it alone will not guarantee the diversity of options because potentially
even very similar sub-trajectories can be classiﬁed into different options if the classiﬁcation model
has a high capacity, in which case we say that the resulting sub-trajectory space has a very high
“resolution”. As a result, the conditional entropy alone might not be able to generate useful options
to be exploited by the scheduler for task solving, because the coverage of the sub-trajectory space is
poor. To combat this degenerate solution, we add a second term which maximizes the entropy of the
worker policy. Intuitively, while the worker generates identiﬁable sub-trajectories corresponding to
a given option, it should act as randomly as possible to separate sub-trajectories of different options,
lowering the “resolution” of the sub-trajectory space to encourage its coverage.

Because directly estimating the posterior p(uh|ah,k, sh,k+1) is intractable, we approximate it with a
parameterized posterior log qψ(uh|ah,k, sh,k+1) to obtain a lower bound (Barber & Agakov, 2003),
where qψ is a discriminator to be learned. Then we can maximize this lower bound instead:

max
φ,ψ

E
πθ,πφ

(cid:88)

h,k

log qψ(uh|ah,k, sh,k+1) − β log πφ(ah,k|sh,k, ah,k−1, uh).

(5)

The discriminator qψ is trained by maximizing likelihoods of options given sampled sub-trajectories.
The worker πφ is trained via max-entropy RL (Soft Actor-Critic (SAC) (Haarnoja et al., 2018)) with
the intrinsic reward rlo

h,k+1 := log qψ(·) − β log πφ(·). β is ﬁxed to 0.01 in our experiments.

Note that there are at least four differences between Eq. 5 and the common option discovery objective
in either VIC (Gregor et al., 2016) or DIAYN (Eysenbach et al., 2019):

1. Both VIC and DIAYN assume that a sampled option will last through an entire episode,
and the option is always sampled at the beginning of an episode. Thus their option trajec-
tories “radiate” from the initial state set. In contrast, our worker policy learns options that
initialize every K steps within an episode, and they can have more diverse semantics de-
pending on the various states sh,0 visited by the agent. This is especially helpful for some
tasks where new options need to be discovered after the agent reaches unseen areas in later
stages of training.

2. Actions taken by the worker policy under the current option will have consequences on the
next option. This is because the ﬁnal state sh,K of the current option is deﬁned to be the
initial state sh+1,0 of the next option. So in general, the worker policy is trained not only to
discover diverse options across the current K steps, but also to make the discovery easier
in the future steps. In other words, the worker policy needs to solve the credit assignment
problem across options, under the expectation of the scheduler policy.

3. To enable the worker policy to learn from a discriminator that predicts based on option sub-
trajectories {ah,k, sh,k+1} instead of solely on individual states sh,k, we have constructed
a new meta MDP where each state sh,k encodes history from the beginning (h, 0) up to
(h, k) within an option h. This new meta MDP is critical, because otherwise one simply
cannot learn a worker policy from a reward function that is deﬁned by multiple time steps
(sub-trajectories) since the learning problem is no longer Markovian.

4. Lastly, thanks to the new MDP, we are able to explore various possible instantiations of
the discriminator (see Section 3.3). As observed in the experiments, individual states are
actually not the optimal features for identifying options.

These differences constitute the major novelty of our worker objective.

3.2.2 SHORTSIGHTED WORKER

It’s challenging for the worker to accurately predict values over a long horizon, since its rewards are
densely computed by a complex nonlinear function qψ. Also each option only lasts at most K steps.
Thus we set the discount η for the worker in two shortsighted ways:

1. Hard: setting η = 0 every K-th step and η = 1 otherwise. Basically this truncates the
temporal correlation (gradients) between adjacent options. Its beneﬁt might be faster and
easier value learning because the value is bootstrapped over at most K steps (K (cid:28) T ).

2. Soft: η = 1 − 1

K , which considers rewards of roughly K steps ahead. The worker policy
still needs to take into account the identiﬁcation of future option sub-trajectories, but their
importance quickly decays.

4

Published as a conference paper at ICLR 2021

We will evaluate both versions and compare their performance in Section 4.1.

3.3

INSTANTIATING THE DISCRIMINATOR

We explore various ways of instantiating the discriminator qψ in order to compute useful intrinsic
rewards for the worker. Previous work has utilized individual states (Eysenbach et al., 2019; Jabri
et al., 2019) or full observation trajectories (Warde-Farley et al., 2019; Sharma et al., 2019a; Achiam
et al., 2018) for option discrimination. Thanks to the newly deﬁned meta MDP, our discriminator is
able to take option sub-trajectories instead of current individual states for prediction. In this paper,
we investigate six sub-trajectory feature extractors fψ:

Feature extractor

fψ(ah,k, sh,k+1) =

Formulation
MLP(sh,k+1)
MLP([sh,0, ah,k])
MLP(sh,k+1 − sh,k) Difference between state pairs

Name
State
Action
StateDiff
StateAction MLP([ah,k, sh,k+1]) Action and next state
StateConcat MLP([sh,k+1])
ActionConcat MLP([sh,0, ah,k])

Explanation
Next state alone
Action in context

Concatenation of states
Concatenation of actions

where the operator [·] denotes concatenation and MLP denotes a multilayer perceptron1. Our State
feature extractor is most similar to DIAYN (Eysenbach et al., 2019), and StateConcat is similar
to (Warde-Farley et al., 2019; Sharma et al., 2019a; Achiam et al., 2018). However we note that
unlike these works, the distribution of our option sub-trajectories is also determined by the scheduler
in the context of hRL. The other four feature extractors have not been evaluated before. With the
extracted feature, the log-probability of predicting an option is simply computed as the negative
squared L2 norm: log qψ(uh|ah,k, sh,k+1) = −(cid:107)fψ(ah,k, sh,k+1) − uh(cid:107)2
2, by which we implicitly
assume the discriminator’s output distribution to be a N (0, ID) multivariate Gaussian.

3.4 OFF-POLICY TRAINING

The scheduler and worker objectives (Eq. 3 and Eq. 5) are trained jointly. In principle, on-policy
training such as A2C (Clemente et al., 2017) is needed due to the interplay between the scheduler
and worker. However, to reuse training data and improve sample efﬁciency, we employ off-policy
training (SAC (Haarnoja et al., 2018)) for both objectives with some modiﬁcations.

Modiﬁed worker objective In practice, the expectation over the scheduler πθ in Eq. 5 is replaced
with the expectation over its historical versions. Speciﬁcally, we sample options uh from a replay
buffer, together with sub-trajectories {ah,k, sh,k+1}. This type of data distribution modiﬁcation is
conventional in off-policy training (Lillicrap et al., 2016).

Intrinsic reward relabeling We always recompute the rewards in Eq. 5 using the up-to-date dis-
criminator for every update of φ, which can be trivially done without any additional interaction with
the environment.

Importance correction The data in the replay buffer was generated by historical worker policies.
Thus a sampled option sub-trajectory will be outdated under the same option, causing confusion
to the scheduler policy. To resolve this issue, when minimizing the temporal-difference (TD) error
between the values of sh,0 and sh+1,0 for the scheduler, an importance ratio can be multiplied:
(cid:81)K−1
πφ(ah,k|sh,k,ah,k−1,uh)
φ (ah,k|sh,k,ah,k−1,uh) . A similar correction can also be applied to the discriminator loss.
k=0
πold
However, in practice we ﬁnd that this ratio has a very high variance and hinders the training. Like
the similar observations made in Nachum et al. (2018); Fedus et al. (2020), even without importance
correction our method is able to perform well empirically2.

1In this paper we focus on non-image observations that can be processed with MLPs, although our method

doesn’t have any assumption about the observation space.

2One possible reason is that the deep RL process is “highly non-stationary anyway, due to changing policies,

state distributions and bootstrap targets” (Schaul et al., 2016).

5

Published as a conference paper at ICLR 2021

Figure 2: The four tasks we evaluate on. From left to right: 7-DOF PUSHER, 7-DOF REACHER,
GOALTASK, and KICKBALL. The ﬁrst two tasks simulate a one-armed PR2 robot environment
while the last two are in the SOCIALROBOT environment. The ﬁnal picture shows a closeup of the
PIONEER2DX robot used in SOCIALROBOT.

4 EXPERIMENTS

Environments We evaluate success rate and sample efﬁciency across two environment suites, as
shown in Figure 2. Important details are presented here with more information in appendix Section
B. The ﬁrst suite consists of two 7-DOF reaching and pushing environments evaluated in Chua et al.
(2018). They both emulate a one-armed PR2 robot. The tasks have sparse rewards: the agent gets
a reward of 0 at every timestep where the goal is not achieved, and 1 upon achieved. There is also
a small L2 action penalty applied. In 7-DOF REACHER, the goal is achieved when the gripper
reaches a 3D goal position. In 7-DOF PUSHER, the goal is to push an object to a 3D goal position.
Episodes have a ﬁxed length of 100; a success of an episode is deﬁned to be if the goal is achieved
at the ﬁnal step of the episode.

We also propose another suite of environments called SOCIALROBOT 3. We construct two sparse
reward robotic navigation and manipulation tasks, GOALTASK and KICKBALL. In GOALTASK, the
agent gets a reward of 1 when it successfully navigates to a goal, -1 if the goal becomes too far, -0.5
every time it is too close to a distractor object, and 0 otherwise. In KICKBALL, the agent receives
a reward of 1 for successfully pushing a ball into the goal, 0 otherwise, and has the same distractor
object penalty. At the beginning of each episode, both the agent and the ball are spawned randomly.
Both environments contain a small L2 action penalty, and terminate an episode upon a success.

Comparison methods One baseline algorithm for comparison is standard SAC (Haarnoja et al.,
2018), the building block of our hierarchical method. To verify if our worker policy can just be re-
placed with a na¨ıve action repetition strategy, we compare with SAC+ActRepeat with an action rep-
etition for the same length K as our option interval. We also compare against HIRO (Nachum et al.,
2018), a data efﬁcient hierarchical method with importance-based option relabeling, and HiPPO
(Li et al., 2020) which trains the lower level and higher level policies together with one uniﬁed
PPO-based objective. Both are state-of-the-art hierarchical methods proposed to solve sparse re-
ward tasks. Similar to our work, HiPPO makes no assumptions about options, however it utilizes a
discrete option space and its options are trained with environment reward.

We implement HIDIO based on an RL framework called ALF 4. A comprehensive hyperparameter
search is performed for every method, with a far greater search space over HiPPO and HIRO than
our method HIDIO to ensure maximum fairness in comparison; details are presented in Appendix D.

Evaluation For every evaluation point during training, we evaluate the agent with current determin-
istic policies (by taking arg max of action distributions) for a ﬁxed number of episodes and compute
the mean success rate. We plot the mean evaluation curve over 3 randomly seeded runs with standard
deviations shown as the shaded area around the curve.

4.1 WORKER DESIGN CHOICES

We ask and answer questions about the design choices in HIDIO speciﬁc to the worker policy πφ.

1. What sub-trajectory feature results in good option discovery? We evaluate all six features pro-
posed in Section 3.3 in all four environments. These features are selected to evaluate how different
types of subtrajectory information affect option discovery and ﬁnal performance. They encom-
pass varying types of both local and global subtrajectory information. We plot comparisons of

3https://github.com/HorizonRobotics/SocialRobot
4https://github.com/HorizonRobotics/alf

6

Published as a conference paper at ICLR 2021

Figure 3: Comparison of all discriminator features against each other across the four environments.
Solid lines indicate hard short-sighted workers (Hard), dotted lines indicated soft short-sighted
workers (Soft).

Figure 4: Comparisons of the mean success rates of three features of HIDIO (Action,
StateAction, StateDiff; solid lines) against other methods (dashed lines).

sample efﬁciency and ﬁnal performance in Figure 3 across all environments (solid lines), ﬁnd-
ing that Action, StateAction, and StateDiff are generally among the top performers.
StateAction includes the current action and next state, encouraging πφ to differentiate its op-
tions with different actions even at similar states. Similarly, Action includes the option initial state
and current action, encouraging option diversity by differentiating between actions conditioned on
initial states. Meanwhile StateDiff simply encodes the difference between the next and current
state, encouraging πφ to produce options with different state changes at each step.

2. How do soft shortsighted workers (Soft) compare against hard shortsighted workers (Hard)?
In Figure 3, we plot all features with Soft in dotted lines. We can see that in general there is not
much difference in performance between Hard and Soft except some extra instability of Soft in
REACHER regarding the StateConcat and State features. One reason of this similar general
performance could be that since our options are very short-term in Hard, the scheduler policy has
the opportunity of switching to a good option before the current one leads to bad consequences. In a
few cases, Hard seems better learned, perhaps due to an easier value bootstrapping for the worker.

4.2 COMPARISON RESULTS

We compare our three best sub-trajectory features of Hard, in Section 4.1, against the SAC baselines
and hierarchical RL methods across all four environments in Figure 4. Generally we see that HIDIO
(solid lines) achieves greater ﬁnal performance with superior sample efﬁciency than the compared
methods. Both SAC and SAC+ActRepeat perform poorly across all environments, and all baseline
methods perform signiﬁcantly worse than HIDIO on REACHER, GOALTASK, and KICKBALL.

In PUSHER, HiPPO displays competitive performance, rapidly improving from the start. How-
ever, all three HIDIO instantiations achieve nearly 100% success rates while HiPPO is unable to do
so. Furthermore, HIRO and SAC+ActRepeat take much longer to start performing well, but never
achieve similar success rates as HIDIO. HIDIO is able to solve REACHER while HiPPO achieves
only about a 60% success rate at best. Meanwhile, HIRO, SAC+ActRepeat, and SAC are unsta-
ble or non-competitive. REACHER is a difﬁcult exploration problem as the arm starts far from the
goal position, and we see that HIDIO’s automatically discovered options ease exploration for the
higher level policy to consistently reach the goal. HIDIO performs well on GOALTASK, achieving
60-80% success rates, while the task is too challenging for every other method. In KICKBALL, the
most challenging task, HIDIO achieves 30-40% success rates while every other learns poorly again,
highlighting the need for the intrinsic option discovery of HIDIO in these environments.

7

0123Num Timesteps1e60.00.20.40.60.81.0SuccessPusher Success Plot0123Num Timesteps1e60.00.20.40.60.81.0SuccessReacher Success Plot012345Num Timesteps1e60.00.20.40.60.81.0SuccessGoalTask Success Plot012345Num Timesteps1e60.00.10.20.30.40.5SuccessKickBall Success PlotActionStateActionActionConcatStateConcatStateDiffState0123Num Timesteps1e60.00.20.40.60.81.0SuccessPusher Success Plot0123Num Timesteps1e60.00.20.40.60.81.0SuccessReacher Success Plot012345Num Timesteps1e60.00.20.40.60.81.0SuccessGoalTask Success Plot012345Num Timesteps1e60.00.10.20.30.40.5SuccessKickBall Success PlotSACSAC+ActRepeatHiPPOHIROActionStateActionStateDiffPublished as a conference paper at ICLR 2021

Figure 5: Two example options from the StateAction instantiation on KICKBALL (top) and
PUSHER (bottom). The top option navigates directly to the goal by bypassing obstructions along the
way and the bottom option sweeps the puck towards one direction.

In summary, HIDIO demonstrates greater sample efﬁciency and ﬁnal reward gains over all other
baseline methods. Regular RL (SAC) fails on all four environments, and while HiPPO is a strong
baseline on PUSHER and REACHER, it is still outperformed in both by HIDIO. All other meth-
ods fail on GOALTASK and KICKBALL, while HIDIO is able to learn and perform better in both.
This demonstrates the importance of the intrinsic, short-term option discovery employed by HIDIO,
where the options are diverse enough to be useful for both exploration and task completion.

4.3

JOINT πφ AND πθ TRAINING

We ask the next question: is jointly training πθ and
πφ necessary? To answer this, we compare HIDIO
against a pre-training baseline where we ﬁrst pre-train
πφ, with uniformly sampled options u for a portion ρ
of total numbers of training time steps, and then ﬁx
πφ while training πθ for the remaining (1 − ρ) time
steps. This is essentially using pre-trained options
for downstream higher-level tasks as demonstrated in
DIAYN (Eysenbach et al., 2019). We conduct this
experiment with the StateAction feature on both
KICKBALL and PUSHER, with ρ = { 1
4 }. The
results are shown in Figure 6. We can see that in
PUSHER, fewer pre-training time steps are more sample efﬁcient, as the environment is simple and
options can be learned from a small amount of samples. The nature of PUSHER also only requires
options that can be learned independent of the scheduler policy evolution. Nevertheless, the pre-
training baselines seem less stable. In KICKBALL, the optimal pre-training baseline is on ρ = 1
8 of
the total time steps. However without the joint training scheme of HIDIO, the learned options are
unable to be used as efﬁciently for the difﬁcult obstacle avoidance, navigation, and ball manipulation
subtasks required for performing well.

Figure 6: Pretraining baseline comparison
at fractions { 1
8 , 1
4 } of the total number
of training time steps.

16 , 1

16 , 1

8 , 1

4.4 OPTION BEHAVIORS

Finally, since options discovered by HIDIO in our sparse reward environments help it achieve supe-
rior performance, we ask, what do useful options look like? To answer this question, after training,
we sample options from the scheduler πθ to visualize their behaviors in different environments in
Figure 5. For each sampled option u, we ﬁx it until the end of an episode and use the worker πφ
to output actions given u. We can see that the options learned by HIDIO are low-level navigation
and manipulation skills useful for the respective environments. We present more visualizations in
Figure 9 and more analysis in Section C.2 in the appendix. Furthermore, we present an analysis of
task performance for different option lengths in appendix Section C.1 and Figures 7 and 8.

5 RELATED WORK

Hierarchical RL Much of the previous work in hRL makes assumptions about the task structure
and/or the skills needed to solve the task. While obtaining promising results under speciﬁc settings,
they may have difﬁculties with different scenarios. For example, SAC-X (Riedmiller et al., 2018) re-
quires manually designing auxiliary subtasks as skills to solve a given downstream task. SNN4HRL
(Florensa et al., 2016) is geared towards tasks with pre-training and downstream components. Lee

8

0123Num Timesteps1e60.00.20.40.60.81.0SuccessPusher Success Plot012345Num Timesteps1e60.00.10.20.3SuccessKickBall Success PlotStateAction1418116Published as a conference paper at ICLR 2021

et al. (2019; 2020) learns to modulate or compose given primitive skills that are customized for their
particular robotics tasks. Ghavamzadeh & Mahadevan (2003) and Sohn et al. (2018) operate under
the assumption that tasks can be manually decomposed into subtasks.

The feudal reinforcement learning proposal (Dayan & Hinton, 1993) has inspired another line of
works (Vezhnevets et al., 2017; Nachum et al., 2018; Levy et al., 2019; Rafati & Noelle, 2019) which
make higher-level manager policies output goals for lower-level worker policies to achieve. Usually
the goal space is a subspace of the state space or deﬁned according to the task so that lower-level
rewards are easy to compute. This requirement of manually “grounding” goals in the environment
poses generalization challenges for tasks that cannot be decomposed into state or goal-reaching.

The MAXQ decomposition (Dietterich, 2000) deﬁnes an hRL task decomposition by breaking up
the target MDP into a hierarchy of smaller MDPs such that the value function in the target MDP is
represented as the sum of the value functions of the smaller ones. This has inspired works that use
such decompositions (Mehta et al., 2008; Winder et al., 2020; Li et al., 2017) to learn structured,
hierarchical world models or policies to complete target tasks or perform transfer learning. However,
building such hierarchies makes these methods limited to MDPs with discrete action spaces.

Our method HIDIO makes few assumptions about the speciﬁc task at hand. It follows from the op-
tions framework (Sutton et al., 1999), which has recently been applied to continuous domains (Bacon
et al., 2017), spawning a diverse set of recent hierarchical options methods (Bagaria & Konidaris,
2020; Klissarov et al., 2017; Riemer et al., 2018; Tiwari & Thomas, 2019; Jain et al., 2018). HIDIO
automatically learns intrinsic options that avoids having explicit initiation or termination policies de-
pendent on the task at hand. HiPPO (Li et al., 2020), like HIDIO, also makes no major assumptions
about the task, but does not employ self-supervised learning for training the lower-level policy.

Self-supervised option/skill discovery There are also plenty of prior works which attempt to learn
skills or options without task reward. DIAYN (Eysenbach et al., 2019) and VIC (Gregor et al., 2016)
learn skills by maximizing the mutual information between trajectory states and their corresponding
skills. VALOR (Achiam et al., 2018) learns options by maximizing the probability of options given
their resulting observation trajectory. DADS (Sharma et al., 2019a) learns skills that are predictable
by dynamics models. DISCERN (Warde-Farley et al., 2019) maximizes the mutual information
between goal and option termination states to learn a goal-conditioned reward function. Brunskill
& Li (2014) learns options in discrete MDPs that are guaranteed to improve a measure of sample
complexity. Portable Option Discovery (Topin et al., 2015) discovers options by merging options
from source policies to apply to some target domain. Eysenbach et al. (2019); Achiam et al. (2018);
Sharma et al. (2019a); Lynch et al. (2020) demonstrate pre-trained options to be useful for hRL.
These methods usually pre-train options in an initial stage separate from downstream task learning;
few works directly integrate option discovery into a hierarchical setting. For higher dimensional
input domains, Lynch et al. (2020) learns options from human-collected robot interaction data for
image-based, goal-conditioned tasks, and Chuck et al. (2020) learns a hierarchy of options by dis-
covering objects from environment images and forming options which can manipulate them. HIDIO
can also be applied to image-based environments by replacing fully-connected layers with convolu-
tional layers in the early stages of the policy and discriminator networks. However, we leave this to
future work to address possible practical challenges arising in this process.

6 CONCLUSION

Towards solving difﬁcult sparse reward tasks, we propose a new hierarchical reinforcement learn-
ing method, HIDIO, which can learn task-agnostic options in a self-supervised manner and simul-
taneously learn to utilize them to solve tasks. We evaluate several different instantiations of the
discriminator of HIDIO for providing intrinsic rewards for training the lower-level worker policy.
We demonstrate the effectiveness of HIDIO compared against other reinforcement learning methods
in achieving high rewards with better sample efﬁciency across a variety of robotic navigation and
manipulation tasks.

9

Published as a conference paper at ICLR 2021

REFERENCES

Joshua Achiam, Harrison Edwards, Dario Amodei, and Pieter Abbeel. Variational option discovery

algorithms. arXiv, 2018. 5, 9

Pierre-Luc Bacon, Jean Harb, and Doina Precup. The option-critic architecture. In AAAI, 2017. 9

Akhil Bagaria and George Konidaris. Option discovery using deep skill chaining. In ICLR, 2020. 9

D. Barber and F. Agakov. The im algorithm: A variational approach to information maximization.

In NeurIPS, 2003. 4

Emma Brunskill and Lihong Li. Pac-inspired option discovery in lifelong reinforcement learning.
volume 32 of Proceedings of Machine Learning Research, pp. 316–324, Bejing, China, 22–24 Jun
2014. PMLR. URL http://proceedings.mlr.press/v32/brunskill14.html. 9

Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine. Deep reinforcement

learning in a handful of trials using probabilistic dynamics models. In NeurIPS, 2018. 6

Caleb Chuck, Supawit Chockchowwat, and Scott Niekum. Hypothesis-driven skill discovery for

hierarchical deep reinforcement learning, 2020. 9

Alfredo V. Clemente, Humberto Nicol´as Castej´on Mart´ınez, and Arjun Chandra. Efﬁcient parallel

methods for deep reinforcement learning. CoRR, abs/1705.04862, 2017. 5

Peter Dayan and Geoffrey E Hinton. Feudal reinforcement learning. In NeurIPS, pp. 271–278, 1993.

9

Thomas G Dietterich. Hierarchical reinforcement learning with the maxq value function decompo-

sition. Journal of artiﬁcial intelligence research, 13:227–303, 2000. 9

Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, and Sergey Levine. Diversity is all you need:

Learning skills without a reward function. In ICLR, 2019. 4, 5, 8, 9

William Fedus, Prajit Ramachandran, Rishabh Agarwal, Yoshua Bengio, Hugo Larochelle, Mark
Rowland, and Will Dabney. Revisiting fundamentals of experience replay. In ICML, 2020. 5

Carlos Florensa, Yan Duan, and Pieter Abbeel. Stochastic neural networks for hierarchical rein-

forcement learning. In ICLR, 2016. 1, 8

Mohammad Ghavamzadeh and Sridhar Mahadevan. Hierarchical policy gradient algorithms. ICML,

2003. 1, 9

Karol Gregor, Danilo Jimenez Rezende, and Daan Wierstra. Variational intrinsic control. arXiv,

abs/1611.07507, 2016. 4, 9

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy
maximum entropy deep reinforcement learning with a stochastic actor. In ICML, 2018. 4, 5, 6

Karol Hausman, Jost Tobias Springenberg, Ziyu Wang, Nicolas Heess, and Martin Riedmiller.

Learning an embedding space for transferable robot skills. In ICLR, 2018. 1

Allan Jabri, Kyle Hsu, Ben Eysenbach, Abhishek Gupta, Sergey Levine, and Chelsea Finn. Unsu-

pervised curricula for visual meta-reinforcement learning. In NeurIPS, 2019. 5

Arushi Jain, Khimya Khetarpal, and Doina Precup. Safe option-critic: Learning safety in the option-

critic architecture. arXiv, 2018. 9

Martin Klissarov, Pierre-Luc Bacon, Jean Harb, and Doina Precup. Learnings options end-to-end

for continuous action tasks. arXiv, 2017. 9

Hoang M. Le, Nan Jiang, Alekh Agarwal, Miroslav Dud´ık, Yisong Yue, and Hal Daum´e III. Hier-

archical imitation and reinforcement learning. In ICML, 2018. 1

10

Published as a conference paper at ICLR 2021

Youngwoon Lee, Shao-Hua Sun, Sriram Somasundaram, Edward Hu, and Joseph J. Lim. Com-
posing complex skills by learning transition policies with proximity reward induction. In ICLR,
2019. 1, 8

Youngwoon Lee, Jingyun Yang, and Joseph J. Lim. Learning to coordinate manipulation skills via

skill behavior diversiﬁcation. In ICLR, 2020. 1, 9

Andrew Levy, Robert Platt Jr., and Kate Saenko. Learning multi-level hierarchies with hindsight. In

ICLR, 2019. 9

Alexander C. Li, Carlos Florensa, Ignasi Clavera, and Pieter Abbeel. Sub-policy adaptation for

hierarchical reinforcement learning. In ICLR, 2020. 6, 9

Zhuoru Li, Akshay Narayan, and Tze-Yun Leong. An efﬁcient approach to model-based hierarchical

reinforcement learning. In Thirty-First AAAI Conference on Artiﬁcial Intelligence, 2017. 9

Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa,
David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. In ICLR,
2016. 5

Corey Lynch, Mohi Khansari, Ted Xiao, Vikash Kumar, Jonathan Tompson, Sergey Levine, and
Pierre Sermanet. Learning latent plans from play. In Conference on Robot Learning, pp. 1113–
1132, 2020. 9

Neville Mehta, Soumya Ray, Prasad Tadepalli, and Thomas Dietterich. Automatic discovery and
transfer of maxq hierarchies. In Proceedings of the 25th international conference on Machine
learning, pp. 648–655, 2008. 9

Josh Merel, Arun Ahuja, Vu Pham, Saran Tunyasuvunakool, Siqi Liu, Dhruva Tirumala, Nicolas

Heess, and Greg Wayne. Hierarchical visuomotor control of humanoids. In ICLR, 2019. 1

Oﬁr Nachum, Shixiang Gu, Honglak Lee, and Sergey Levine. Data-efﬁcient hierarchical reinforce-

ment learning. In NeurIPS, 2018. 1, 3, 5, 6, 9

Jacob Rafati and David C Noelle. Learning representations in model-free hierarchical reinforcement
learning. In Proceedings of the AAAI Conference on Artiﬁcial Intelligence, volume 33, pp. 10009–
10010, 2019. 9

Pravesh Ranchod, Benjamin Rosman, and George Konidaris. Nonparametric bayesian reward seg-

mentation for skill discovery using inverse reinforcement learning. In IROS, 2015. 1

Martin Riedmiller, Roland Hafner, Thomas Lampe, Michael Neunert, Jonas Degrave, Tom van de
Wiele, Vlad Mnih, Nicolas Heess, and Jost Tobias Springenberg. Learning by playing solving
sparse reward tasks from scratch. In ICML, 2018. 1, 3, 8

Matthew Riemer, Miao Liu, and Gerald Tesauro. Learning abstract options. In NeurIPS, 2018. 9

Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay. In

ICLR, 2016. 5

Archit Sharma, Shixiang Gu, Sergey Levine, Vikash Kumar, and Karol Hausman. Dynamics-aware

unsupervised discovery of skills. arXiv, abs/1907.01657, 2019a. 5, 9

Arjun Sharma, Mohit Sharma, Nicholas Rhinehart, and Kris M. Kitani. Directed-info gail: Learning
In ICLR,

hierarchical policies from unsegmented demonstrations using directed information.
2019b. 1

Sungryull Sohn, Junhyuk Oh, and Honglak Lee. Hierarchical reinforcement learning for zero-shot

generalization with subtask dependencies. In NeurIPS, 2018. 1, 9

Richard S. Sutton, Doina Precup, and Satinder Singh. Between mdps and semi-mdps: A framework
for temporal abstraction in reinforcement learning. Artiﬁcial Intelligence, 112(1):181 – 211, 1999.
2, 9

Saket Tiwari and Philip S. Thomas. Natural option critic. In AAAI, 2019. 9

11

Published as a conference paper at ICLR 2021

Nicholay Topin, Nicholas Haltmeyer, S. Squire, J. Winder, M. desJardins, and J. MacGlashan.
Portable option discovery for automated learning transfer in object-oriented markov decision pro-
cesses. In IJCAI, 2015. 9

Alexander Sasha Vezhnevets, Simon Osindero, Tom Schaul, Nicolas Heess, Max Jaderberg, David
Silver, and Koray Kavukcuoglu. Feudal networks for hierarchical reinforcement learning.
In
ICML, 2017. 3, 9

David Warde-Farley, Tom Van de Wiele, Tejas Kulkarni, Catalin Ionescu, Steven Hansen, and
Volodymyr Mnih. Unsupervised control through non-parametric discriminative rewards. In ICLR,
2019. 5, 9

John Winder, Stephanie Milani, Matthew Landen, Erebus Oh, Shane Parr, Shawn Squire, Cynthia
Matuszek, et al. Planning with abstract learned models while learning transferable subtasks. In
Proceedings of the AAAI Conference on Artiﬁcial Intelligence, volume 34, pp. 9992–10000, 2020.
9

A PSEUDO CODE FOR HIDIO

Algorithm 1: Hierarchical RL with Intrinsic Options Discovery
Input:
T
B Batch size
K Option interval P(sh,k+1|ss,k, ah,k) Environment dynamics

Batches per iteration
Learning rate

Episode length M
α

πφ(ah,k|sh,k, uh)
qψ(uh|ah,k, sh,k+1) Discriminator
πθ(uh|sh,0)

Scheduler

Worker

Output: Learned parameters θ, φ, and ψ.
Initialize: Random model parameters θ, φ, and ψ; empty replay buffers Dscheduler and Dworker.
while termination not met do

/* Data collection
for scheduler step h = 0.. T

K − 1 do

Sample an option uh ∼ πθ(·|sh,0).
for worker step k = 0..K − 1 do

Sample an action ah,k ∼ πφ(·|sh,k, uh).
Step through the environment sh,k+1 ∼ P(·|sh,k, ah,k).
ah,k, sh,k+1 ← [ah,k−1, ah,k], [sh,k, sh,k+1]
Dworker ← Dworker ∪ (uh, ah,k, sh,k+1)

end
Rh ← (cid:80)K−1
k=0 r(sh,k, ah,k, sh,k+1)
Dscheduler ← Dscheduler ∪ (sh,0, uh, sh+1,0, Rh)

end
/* Model training
for batch m = 0..M − 1 do

b=1 ∼ Dscheduler.

/* Scheduler training
Uniformly sample transitions {(st, ut, st+1)}B
Compute gradient ∆θ according to Eq. 3.
Update models θ ← θ + α∆θ.
/* Worker training
Uniformly sample transitions {(uh, ah,k, sh,k+1)}B
Compute intrinsic rewards rlo
Compute gradient ∆ψ and ∆φ according to Eq. 5.
Update models φ ← φ + α∆φ and ψ ← ψ + α∆ψ.

h,k ← qψ(uh|ah,k, sh,k+1).

b=1 ∼ Dworker.

end

end

12

*/

*/

*/

*/

Published as a conference paper at ICLR 2021

Figure 7: Comparisons of the mean success rates of three features of HIDIO (Action,
StateAction, StateDiff at different option lengths K. Dotted lines indicate K = 1, solid
lines indicate K = 3, and dashed lines indicate K = 5. K = 3 was used across all environments
for the results in the main text.

B MORE ENVIRONMENT DETAILS

B.0.1 PUSHER AND REACHER

These environments both have a time horizon of 100 with no early termination: each episode always
runs for 100 steps regardless of goal achievement. For both, a success is when the agent achieves
the goal at the ﬁnal step of an episode. In REACHER, observations are 17-dimensional, including the
positions, angles, and velocities of the robot arm, and in PUSHER observations also include the 3D
object position. Both include the goal position in the observation space. Actions are 7-dimensional
vectors for joint velocity control. The action range is [−20, 20] in REACHER and [−2, 2] in PUSHER.

There is an action penalty in both environments: at every timestep the squared L2 norm of the agent
action is subtracted from the reward. In PUSHER, this penalty is multiplied by a coefﬁcient of 0.001.
In REACHER, it’s multiplied by 0.0001.

B.0.2 GOALTASK AND KICKBALL

For both SOCIALROBOT environments, an episode terminates early when either a success is reached
or the goal is out of range. For each episode, the positions of all objects (including the agent) are
randomly picked. Observations are 18-dimensional. In GOALTASK, these observations include ego-
centric positions, distances, and directions from the agent to different objects while in KICKBALL,
they are absolute positions and directions. In KICKBALL, the agent receives a reward of 1 for suc-
cessfully pushing a ball into the goal (episode termination) and 0 otherwise. At the beginning of
each episode, the ball is spawned randomly inside the neighborhood of the agent. Three distractor
objects are included on the ground to increase task difﬁculty. In GOALTASK, the number of dis-
tractor objects increases to 5. Both environments contain a small L2 action penalty: at every time
step the squared L2 norm of the agent action, multiplied by 0.01, is subtracted from the reward.
GOALTASK has a time horizon of 100 steps, while KICKBALL’s horizon is 200. Observations are
30-dimensional, including absolute poses and velocities of the goal, the ball, and the agent. Both
GOALTASK and KICKBALL use the same navigation robot PIONEER2DX which has 2-dimensional
actions that control the angular velocities (scaled to [−1, 1]) of the two wheels.

C OPTION DETAILS

C.1 OPTION LENGTH ABLATION

We ablate the option length K in all four environments on the three best HIDIO instantiations in
Figure 7. K = {1, 3, 5} timesteps per option are shown, with K = 3 and K = 5 performing
similarly across all environments, but K = 1 performing very poorly in comparison. K = 1
provides no temporal abstraction, resulting in worse sample efﬁciency in PUSHER and REACHER,
and failing to learn in GOALTASK and KICKBALL. Although K = 5 and K = 3 are generally
similar, we see in GOALTASK that K = 5 results in better performance than K = 3 across all three
instantiations, demonstrating the potential beneﬁt of longer temporal abstraction lengths.

13

0.00.51.01.5Num Timesteps1e60.00.20.40.60.81.0SuccessPusher Success Plot0.00.51.01.5Num Timesteps1e60.00.20.40.60.81.0SuccessReacher Success Plot012345Num Timesteps1e60.00.20.40.60.81.0SuccessGoalTask Success Plot012345Num Timesteps1e60.00.10.20.30.40.5SuccessKickBall Success PlotActionStateActionStateDiffPublished as a conference paper at ICLR 2021

Figure 8: Trajectory distributions compared for different option lengths K for the StateAction
HIDIO instantiation in both SOCIALROBOT environments. These are obtained by randomly sam-
pling an option uniformly in [−1, 1]D and keeping it ﬁxed for the entire trajectory. 100 trajectories
from each option are visualized and plotted in different colors.

We also plot the distribution of (x, y) velocities5 in GOALTASK and (x, y) coordinates in KICK-
BALL of randomly sampled options of different lengths in Figure 8. Despite the fact that these two
dimensions only represent a small subspace of the entire (30-dimensional) state space, they still
demonstrate a difference in option behavior at different option lengths. We can see that as the option
length K increases, the option behaviors become more consistent within a trajectory. Meanwhile
regarding coverage, K = 1’s (blue) trajectory distribution in both environments is less concentrated
near the center, while K = 5 (green) is the most concentrated at the center. K = 3 (orange) lies
somewhere in between. We believe that this difference in behavior signiﬁes a trade off between the
coverage of the state space and how consistent the learned options can be depending on the option
length. Given the same entropy coefﬁcient (β in Eq 5), with longer option lengths, it is likely that
the discriminator can more easily discriminate the sub-trajectories created by these options, so that
their coverage does not have to be as wide for the worker policy to obtain high intrinsic rewards.
Meanwhile, with shorter option lengths, the shorter sub-trajectories have to be more distinct for the
discriminator to be able to successfully differentiate between the options.

C.2 OPTION VISUALIZATIONS

We visualize more option behaviors in Figure 9, produced in the same way as in Figure 5 and as
detailed in Section 4.4. The top 4 picture reels are from KICKBALL. We see that KICKBALL
options lead to varied directional driving behaviors that can be utilized for efﬁcient navigation. For
example, the second, third, and fourth highlight options that produce right turning behavior, however
at different speeds and angles. The option in the third reel is a quick turn that results in the robot
tumbling over into an unrecoverable state, but the options in the second and fourth reels turn more
slowly and do not result in the robot ﬂipping. The ﬁrst option simply proceeds forward from the
robot starting position, kicking the ball into the goal.

The bottom 4 reels are from PUSHER. Each option results in different sweeping behaviors with
varied joint positioning and arm height. These sweeping and arm folding behaviors, when utilized
in short sub-trajectories, are useful for controlling where and how to move the arm to push the puck
into the goal.

D HYPERPARAMETERS

To ensure a fair comparison across all methods, we perform a hyperparameter search over the fol-
lowing values for each algorithm and suite of environments.

5Velocities are relative to the agent’s yaw rotation. Because GOALTASK has egocentric inputs, the agent is

not aware of the absolute (x, y) coordinates in this task.

14

42024X Velocity2.01.51.00.50.00.51.01.52.0Y VelocityGoalTask Option Length ComparisonK = 1K = 3K = 56420246810X Position10.07.55.02.50.02.55.07.510.0Y PositionKickBall Option Length ComparisonPublished as a conference paper at ICLR 2021

Figure 9: Eight example options from the StateAction instantiation on KICKBALL (top 4) and
PUSHER (bottom 4).

15

Published as a conference paper at ICLR 2021

D.1 PUSHER AND REACHER

Shared hyperparameters across all methods are listed below (where applicable, and except when
overridden by hyperparameters listed for each individual method). For all methods, we take the
hyperparameters that perform best across 3 random seeds in terms of the area under the evaluation
success curve (AUC) in the PUSHER environment.

• Number of parallel actors/environments per rollout: 20
• Steps per episode: 100
• Batch size: 2048
• Learning rate: 10−4 for all network modules
• Policy/Q network hidden layers: (256, 256, 256) with ReLU non-linearities
• Polyak averaging coefﬁcient for target Q: 0.999
• Target Q update interval (training iterations): 1
• Training batches per iteration: 100
• Episodes per evaluation: 50
• Initial environment steps for data collection before training: 10000

Rollouts and training iterations are performed alternatively, one after the other. The rollout length
searched below refers to how many time steps in each environment are taken per rollout/training
iteration, effectively controlling the ratio of gradient steps to environment steps. A smaller roll-
out length corresponds to a higher ratio. This ratio is also searched over for HIPPO and HIRO.
Other hyperparameters searched separately for each algorithm are listed below, and selected ones
are bolded.

D.1.1 SAC

• Target entropy min prob ∆6: {0.1, 0.2, 0.3}
• Replay buffer length per parallel actor: {50000, 200000}
• Rollout Length: {12, 25, 50, 100}

D.1.2 SAC W/ ACTION REPETITION

• Action repetition length7: 3
• Rollout Length: {4, 8, 16, 33}

Other hyperparameters are kept the same as the optimal SAC ones.

D.1.3 HIDIO

The hyperparameters of HIDIO were mostly heuristically chosen due to the hyperparameter search
space being too large.

• Latent option u vector dimension (D): {8, 12}
• Policy/Q network hidden layers for πφ : (128, 128, 128)
• Steps per option (K): 3
• πφ has a ﬁxed entropy coefﬁcient α of 0.01. Target entropy min prob ∆ for πθ is 0.2.
• Discriminator network hidden layers: (64, 64)
• Replay buffer length per parallel actor: {50000, 200000}
• Rollout Length: {25, 50, 100}

6The target entropy used for automatically adjusting α is calculated as: (cid:80)

i[ln(Mi − mi) + ln ∆] where
Mi/mi are the maximium/minimum value of action dim i. Intuitively, the target distribution concentrates on a
segment of length (Mi − mi)∆ with a constant probability.
7Chosen to match the option interval K of HIDIO.

16

Published as a conference paper at ICLR 2021

D.1.4 HIRO

• Steps per option: {3, 5, 8}

• Replay buffer size (total): {500000, 2000000}

• Meta action space (actions are relative, e.g., meta-action is current obs + action):

(-np.ones(obs space - 3D goal pos)*2, np.ones(obs space -
3D goal pos)*2)

• Policy stddev noise: {0.1, 0.3, 0.5}

• Number of gradient updates per training iteration: {100, 200, 400}

D.1.5 HIPPO

For most hyperparameters, the search ranges chosen were derived after discussion with the ﬁrst
author of HiPPO.

• Learning rate: 3 × 10−4

• Policy network hidden layers: (256, 256)

• Skill selection network hidden layers: {(32, 32), (128, 64)}

• Latent skill vector size: {5, 10, 15}

• PPO clipping parameter: {0.05, 0.1}

• Time commitment range: {(2, 5), (3, 7)}

• Policy training steps per epoch: {25, 50, 100}

D.2 SOCIALROBOT

For all methods, we select the hyperparameters with the best area under the evaluation success curve
(AUC) in the KICKBALL environment, and apply them to both KICKBALL and GOALTASK. The
shared hyperparameters are as follows (if applicable to the algorithm, and except when overridden
by the respective algorithm’s list of hyperparameters):

• Number of parallel actors/environments per rollout: 10

• Steps per episode: 100 (GOALTASK), 200 (KICKBALL)

• Batch size: 1024
• Learning rate: 5 × 10−4 for all network modules

• Policy/Q network hidden layers: (256, 256, 256) with ReLU non-linearities

• Polyak averaging coefﬁcient for target Q: 0.95

• Target Q update interval (training iterations): 1

• Training batches per iteration: 100

• Episodes per evaluation: 100

• Evaluation interval (training iterations): 100

• Initial environment steps for data collection before training: 100000

The training terminology here generally follows section D.1.

D.2.1 SAC

• Target entropy min prob ∆: {0.1, 0.2, 0.3}

• Replay buffer length per parallel actor: {20000, 100000}

• Rollout length: {12, 25, 50, 100}

17

Published as a conference paper at ICLR 2021

D.2.2 SAC W/ ACTION REPETITION

• Action repetition length8: 3
• Rollout Length: {4, 8, 16, 33}

Other hyperparameters are kept the same as the optimal SAC ones.

D.2.3 HIDIO

Due to the large hyperparameter search space, we only search over the option vector size and rollout
length, and select everything else heuristically.

• Latent option u vector dimension (D): {4, 6}
• Policy/Q network hidden layers for πφ (128, 128, 128)
• Steps per option (K): 3
• πφ has a ﬁxed entropy coefﬁcient α of 0.01. Target entropy min prob ∆ for πθ is 0.2.
• Discriminator network hidden layers: (32, 32)
• Replay buffer length per parallel actor: 20000
• Rollout Length: {50, 100}

D.2.4 HIRO

• Learning rate: 3 × 10−4
• Steps per option: {3, 5, 8}
• Replay buffer size (total): {500000, 2000000}
• Meta action space (actions are relative, e.g., meta-action is current obs + action):
(-np.ones(obs space) * 2, np.ones(obs space) *

– GOALTASK:

2)

– KICKBALL:

(-np.ones(obs space - goal space) * 2,
np.ones(obs space - goal space) * 2) (because the goal position
is given but will not change in the observation space)

• Policy stddev noise {0.1, 0.3, 0.5}
• Number of gradient updates per training iteration: {100, 200, 400}

D.2.5 HIPPO

• Learning rate: 3 × 10−4
• Policy network hidden layers: {(64, 64), (256, 256)}
• Skill selection network hidden layers: {(32, 32), (128, 64)}
• Latent skill vector size: {4, 8}
• PPO clipping parameter: {0.05, 0.1}
• Time commitment range: {(2, 5), (3, 7)}
• Policy training steps per epoch: {25, 50, 100}

8Chosen to match the option interval K of HIDIO.

18

