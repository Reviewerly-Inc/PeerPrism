Generalization of Model-Agnostic Meta-Learning
Algorithms: Recurring and Unseen Tasks

Alireza Fallah
EECS Department
Massachusetts Institute of Technology
afallah@mit.edu

Aryan Mokhtari
ECE Department
The University of Texas at Austin
mokhtari@austin.utexas.edu

Asuman Ozdaglar
EECS Department
Massachusetts Institute of Technology
asuman@mit.edu

Abstract

In this paper, we study the generalization properties of Model-Agnostic Meta-
Learning (MAML) algorithms for supervised learning problems. We focus on
the setting in which we train the MAML model over m tasks, each with n data
points, and characterize its generalization error from two points of view: First, we
assume the new task at test time is one of the training tasks, and we show that,
for strongly convex objective functions, the expected excess population loss is
bounded by O(1/mn). Second, we consider the MAML algorithm’s generalization
to an unseen task and show that the resulting generalization error depends on the
total variation distance between the underlying distributions of the new task and
the tasks observed during the training process. Our proof techniques rely on the
connections between algorithmic stability and generalization bounds of algorithms.
In particular, we propose a new deﬁnition of stability for meta-learning algorithms,
which allows us to capture the role of both the number of tasks m and number of
samples per task n on the generalization error of MAML.

1

Introduction

In several machine learning problems, it is of interest to design algorithms that can be adjusted
based on previous experiences and tasks to perform better on a new task.
In particular, meta-
learning algorithms achieve such a goal through various approaches, including ﬁnding a proper
meta-initialization for the new task [1–3], updating the model architecture [4–6], or learning the
parameters of optimization algorithms [7, 8].

A popular meta-learning framework that has shown promise in practice is Model-Agnostic Meta-
Learning (MAML), which was ﬁrst introduced in [1]. MAML algorithm uses available training data
on a number of tasks to come up with a meta-initialization that performs well after it is slightly
updated at test time with respect to the new task. In other words, unlike standard supervised learning,
in which we aim to ﬁnd a model that generalize well to a new task without any adaptation step, in
MAML our goal is to ﬁnd an initial model for learning a new task when we have access to limited
labeled data for that task to run one (or a few) step(s) of stochastic gradient descent (SGD).
As shown in Fig. 1, in MAML we are given m tasks with m corresponding datasets {Si}m
i=1 in the
training phase. Once the model is trained (w∗
train), a new task is revealed at test time for which we
have access to K labeled samples drawn from Dtest. We use these labeled samples of the new task to

35th Conference on Neural Information Processing Systems (NeurIPS 2021).

update the trained model by running a step of SGD leading to a new model for the test task (w∗
new).
We ﬁnally evaluate the performance of the updated model over the test task, denoted by Ltest(w∗
new).
MAML and its variants have been extensively studied over the past few years from both empiri-
cal and theoretical point of view [2, 9–16]. In particular, [13] provided convergence guarantees
for MAML algorithm under the assumption that access to
fresh samples at any round of the training stage is possible,
and [15] extended this results to the case that multiple
gradient steps can be performed at test time. However, one
shortcoming of such analysis is that, at training stage, we
often do not have access to fresh samples at every iteration.
Instead, we have access to a large set of realized samples
and we typically do multiple passes over the data points
during the training stage.

Training stage

Sm ∼ pm

S1 ∼ p1

. . .

. . .

train

w∗

Hence, it is essential to come up with a novel analysis that
addresses this issue by characterizing the training error and
generalization error of MAML separately. In this paper,
we accomplish this goal and showcase the role of different
problem parameters in the generalization error of MAML.
Speciﬁcally, we assume that we are given m supervised
learning tasks, with (possibly different) underlying distri-
butions p1, . . . pm, where for each task we have access to
n samples1. As we measure the performance of a model
by its loss after one step of SGD adaptation with K sam-
ples, the problem that one can solve in the training phase
is minimizing the average loss, over all given m tasks and
their n samples, after one step of SGD with K samples.
This empirical loss can be considered as a surrogate for
the desired expected loss (with respect to tasks data) over
all m tasks. Here, we focus on the case that MAML is
used to solve this empirical minimization problem, and
our goal is to quantify the test error of MAML output. To
tackle this problem, we ﬁrst brieﬂy revisit the results from
the optimization literature to bound the training error of MAML, assuming that the loss functions are
strongly convex. We next turn to the main focus of our paper which is the generalization properties
of MAML. More speciﬁcally, we address the following questions:

Figure 1: MAML framework

Test stage

Dtest ∼ ptest

train, Dtest)

Ltest(w∗

∇ ˆL(w∗

new)

w∗

new

• If one of the m given tasks recurs uniformly at random at test time, then how well (in expectation)
would the trained model perform after adaptation with SGD over the fresh samples of that task?
In other words, having training error minimized, what would be the generalization error and our
guarantee on test error? Here, we show that for strongly convex objective functions, we could
achieve a generalization error that decays at O(1/mn). Our analysis builds on the connections
between algorithmic stability and generalization of the output of algorithms. While this relation is
well-understood in classic statistical learning [17, 18], here we propose a novel stability deﬁnition for
meta-learning algorithms which allows us to restore such connection for our setting.

• Assuming that the task at test time is NOT one of the m tasks at training, how would the model
perform on that task after the adaptation step? We answer this question by focusing on the case that
the revealed task at the test time is a new unseen task with underlying data distribution pm+1, and
formally characterizing the generalization error of MAML in this case. We show that when the task
at test time is new, the generalization error also depends on the total variation distance between pm+1
and p1, . . . , pm.

Related work: Recently, there has been signiﬁcant progress in studying theoretical aspects of
meta-learning, in particular, MAML. Authors in [19] proposed iMAML which updates the model
using an approximation of one step of proximal point method and studied its convergence. In
[20], authors introduced the task-robust MAML by considering a minimax formulation rather than
minimization. Several papers have also studied MAML through more general frameworks such as
bilevel optimization [21], stochastic compositional optimization [22], and conditional stochastic

1More precisely, in our analysis we take 2n samples per each task to simplify derivations.

2

optimization [23]. Also, several works have studied the extension of meta-learning theory to online
learning [24, 3], federated learning [25], and reinforcement learning [26, 27].

√

The most relevant paper to our work is [28] that studies generalization of meta-learning algorithms
using stability techniques and shows a O(1/
m) bound for nonconvex loss functions. Here we
focus on strongly convex objective functions and present an analysis that differs from this work in
two fundamental aspects. First, we present a different notion of stability that allows us to capture the
number of data points per task in our bound. In particular, our stability notion measures sensitivity of
the algorithm to perturbations that involve changing K data points which is the data unit involved
in the adaptation step of the MAML algorithm. This enables us to obtain a much tighter bound
O(1/mn) (compared to O(1/m) achieved in [28] for strongly convex functions), highlighting the
dependence on the number of the data samples available for each task. Second, we also consider the
generalization of MAML for the case that the task at test time is not one of the available tasks during
the training stage.

The generalization of MAML has also been studied in [29] from an empirical point of view. In
particular, they show that the generalization of MAML to new tasks is correlated with the coherence
between their adaptation trajectories in parameter space. This is aligned with the connection of
generalization and closeness of underlying distributions that we observe in our results.

2 Problem formulation

In this paper, we consider the supervised learning setting, where each data point is denoted by
z = (x, y) ∈ Z with x ∈ X being the input (feature vector) and y ∈ Y being its corresponding label.
We use the loss function l : Rd × Z → R+ to evaluate the performance of a model parameterized
by w ∈ W, where W is a convex and closed subset of Rd.
In other words, for a data point
z = (x, y) ∈ Z, the loss (cid:96)(w, z) denotes the error of model w in predicting the label y given input x.

We consider access to m tasks denoted by T1, . . . , Tm, where the data corresponding to each task Ti
is generated from a distinct distribution pi. The population loss corresponding to task Ti for model w
is deﬁned as Li(w) := Ez∼pi[(cid:96)(w, z)].
We further use the notation ˆL(w; D) to denote the empirical loss corresponding to dataset D, which is
deﬁned as the average loss of w over the samples of dataset D, i.e., ˆL(w; D) := 1
z∈D (cid:96)(w, z),
|D|
where |D| is the size of dataset D. In general, and throughout the paper, we use the hat notation to
distinguish empirical losses from population losses.
Our goal is to ﬁnd w ∈ W that performs well on average2 over all tasks, after it is updated with
respect to the new task and by using one step of stochastic gradient descent (SGD) with a batch of
size K. To formally introduce this problem we ﬁrst deﬁne the function Fi(w) which captures the
performance of model w over task Ti once it is updated by a single step of SGD,

(cid:80)

Fi(w) := EDtest

i

(cid:16)

(cid:104)

Li

w − α∇ ˆL(w, Dtest
i )

(cid:17)(cid:105)

= EDtest

i

(cid:18)
(cid:20)
Ez∼pi
(cid:96)

w −

α
K

(cid:88)

∇(cid:96)(w, z(cid:48)), z

(cid:19)(cid:21)

(1)

z(cid:48)∈Dtest

i

i

where Dtest
the outer expectation is taken with respect to the choice of elements of Dtest
taken with respect to the data of task i.

is a batch with K different samples, drawn from the probability distribution pi. Note that
i while the inner one is

As our goal is to ﬁnd a model that performs well after one step of adaptation over all m tasks, we
minimize the average expected loss over all given tasks, which can be written as

min
w∈W

F (w) :=

1
m

m
(cid:88)

i=1

Fi(w).

(2)

As the underlying distribution of tasks are often unknown in most applications, we are often unable
to directly solve the problem in (2). On the other hand, for each task, we often have access to data
points that are drawn according to their data distributions. Therefore, instead of solving (2), we solve
its sample average surrogate problem in which each Fi is approximated by its empirical loss.

2Our analysis can be extended to the case that the distribution over tasks is not uniform.

3

To formally deﬁne the empirical loss for each task, suppose for each task Ti we have access to a
training set Si, where its elements are drawn independently according to the probability distribution
pi. We further divide the set Si into two disjoint sets of size n deﬁned as S in
, i.e., Si :=
i
{S in
i , S out
to estimate the inner gradient
∇ ˆL(w, Dtest
to estimate the outer function Li(.). Speciﬁcally, we
i and S out
deﬁne the sample average of Fi using data sets S in

| = n. Here, we use the elements of the S in
i

i ) and use the samples in the set S out

i } and |S in

i and S out

i | = |S out

as

i

i

i

ˆFi(w, Si) : =

1
(cid:0) n
K

(cid:1)

(cid:88)

(cid:16)

ˆL

Din

i ⊂S in

i |Din

i |=K

w − α∇ ˆL(w, Din

i ), S out

i

(cid:17)

(3)

=

1
(cid:0) n
K

(cid:1)

(cid:88)

Din

i ⊂S in

i |Din

i |=K



(cid:96)

w −

1
n

(cid:88)

z∈S out

i

α
K

(cid:88)

z(cid:48)∈Din
i

∇(cid:96)(w, z(cid:48)), z



 .

This expression shows that we use all n elements of S out
to approximate the expectation required for
the computation of Li, and we approximate the expectation with respect to the test set by averaging
over all subsets of S in
that have K elements. Given this expression, the sample average approximation
i
(empirical loss) of Problem (1) is given by

i

arg min
w∈W

ˆF (w, S) :=

1
m

m
(cid:88)

i=1

ˆFi(w, Si),

(4)

i=1 is deﬁned as the concatenation of all tasks data sets.

where S := {Si}m
Having the dataset S, a (possibly randomized) optimization algorithm A with output A(S) can be
used to ﬁnd an approximate solution to the problem in (4). The error of this solution with respect
to the MAML empirical loss, i.e., ˆF (A(S), S) − minW ˆF (., S), is called training error. In this
paper, we are mainly interested to bound the test error which is the error of A(S) with respect to the
population loss, i.e., F (A(S)) − minW F . The test error is also sometimes called excess (population)
loss. Note that the expected test error can be decomposed into three terms:

(cid:105)

(test error) =

(cid:104)

EA,S

EA,S
(cid:124)

F

F (A(S))−min
W
(cid:105)
F (A(S))− ˆF (A(S), S)

(cid:104)

(cid:123)(cid:122)
generalization error

(cid:125)

+ EA,S
(cid:124)

(cid:104) ˆF (A(S), S)−min
(cid:123)(cid:122)
training error

W

(cid:105)
ˆF (., S)

(cid:125)

(cid:104)

min
W

+ ES
(cid:124)

ˆF (., S)
(cid:123)(cid:122)
≤0

(cid:105)

−min
W

F

.

(cid:125)

It can be veriﬁed that the expectation of the third term (over A and S) is non-positive since
ES [minW ˆF (., S)] ≤ minW ES [ ˆF (., S)] and ES [ ˆF (., S)] = F. Hence, to bound the expected
test error, we should bound the expectation of training and generalization errors.

The Model-Agnostic Meta-Learning (MAML) method proposed in [1] is designed to solve the
empirical minimization problem deﬁned in (4). The steps of MAML are outlined in Algorithm 1.
MAML solves Problem (4) by using SGD update for the average loss function ˆF (w, S). To better
i=1 ∇ ˆFi(w, Si), where
highlight this point, note that the gradient of ∇ ˆF (w, S) can be written as 1
m
the i-th term corresponding to task Ti is given by

(cid:80)m

∇ ˆFi(w, Si) =

1
(cid:0) n
K

(cid:1)

(cid:88)

Din
|Din

i ⊂S in
i
i |=K

(cid:34)
(Id − α∇2 ˆL(w, Din

i )) × ∇ ˆL

(cid:16)

w − α∇ ˆL(w, Din

i ), S out

i

(cid:17)

(cid:35)
,

(5)

which involves the second-order information of the loss function. Therefore, to compute a mini-batch
i ⊂ S out
approximation for the above gradient, we consider the batches Din
with b elements. Replacing the above sums with their batch approximations leads to the following
stochastic gradient approximation

i with size K and Dout

i ⊂ S in

i

gi(w; Din

i , Dout

i ) := (Id − α∇2 ˆL(w, Din

i ))∇ ˆL

(cid:16)
w − α∇ ˆL(w, Din

i ), Dout
i

(cid:17)

,

(6)

which is indeed an unbiased estimator of the gradient ∇ ˆFi(w, Si) in (5). If for each task we perform
the update of SGD with gi and then compute their average it would be similar to running SGD for

4

Algorithm 1: MAML [1]

Input: The set of datasets S = {Si}m
summoned at each round r; # of iterations T .
Choose arbitrary initial point w0 ∈ W;
for t = 0 to T − 1 do

i=1 with Si = {S in

i , S out

i }; test time batch size K; # of tasks

Choose r tasks uniformly at random (out of m tasks) and store their indices in Bt;
for all Ti with i ∈ Bt do
Sample a batch Dt,in
Sample a batch Dt,out
wt+1
:= wt − βt
i
end for
wt+1 := rW

of K different elements from S in
of size b from S out
Id − α∇2 ˆL(wt, Dt,in

and with replacement;
(cid:17)
)

wt − α∇ ˆL(wt, Dt,in

i with replacement;

), Dt,out
i

∇ ˆL

(cid:80)

(cid:1);

i
(cid:16)

(cid:16)

(cid:17)

i

i

i

i

wt+1
i

(cid:0) 1
r

i∈Bt

;

end for
Return: wT and ¯wT := 1

(cid:80)T

t=0 wt

T +1

the average loss ∇ ˆF (w, S). This is exactly how MAML is implemented in practice as outlined in
Algorithm 1. In this paper, we consider a constrained problem, and as a result, we also need an extra
projection step in the last step to ensure the feasibility of iterates. Finally, the output of MAML could
be the last iterate wT or the time-average of all iterates ¯wT := 1

(cid:80)T

T +1

t=0 wt.

As stated earlier, the convergence properties of MAML-type methods from an optimization point of
view have been studied recently under different set of assumptions. In this paper, as we characterize
the sum of training error and generalization error, we brieﬂy discuss the optimization error of MAML
when it is used to solve the empirical problem in (4). However, the main focus of this paper is on
studying the generalization error of MAML with respect to new samples and new tasks. Speciﬁcally,
we aim to address the following questions: (i) How well does the solution of (4) generalize to the main
problem of interest in (2)? This could be seen as the generalization error of the MAML algorithm
over new samples for recurring tasks. (ii) How well does the solution of (4) generalize to samples
from new unseen tasks? To be more precise, how would the obtained model preform if the new task is
not one of the m tasks T1, . . . , Tm observed at training, and it is rather a new, unseen task Tm+1 with
an unknown underlying distribution pm+1? In the upcoming sections, we answer these questions
on the generalization properties of MAML in detail and characterize the role of number of tasks m,
number of samples per task n, and number of labeled samples revealed at test time K.

3 Theoretical results

In this section, we formally characterize the excess population loss (test error) of the MAML solution,
when we measure the performance of a model after one step of SGD adaptation. In particular, we
ﬁrst discuss the training error of MAML in detail. Then, we establish a generalization error bound
for the case that the solution of MAML is evaluated over new samples of a recurring task. Finally,
we state the generalization error of MAML once its solution is applied to a new unseen task. Before
stating our results, we mention our required assumptions.

Assumption 1. For any z ∈ Z, the function (cid:96)(., z) is twice continuously differentiable. Furthermore,
we assume it satisﬁes the following properties for any w, u ∈ Rd:

(i) For any z ∈ Z, the function (cid:96)(., z) is µ-strongly convex, i.e., (cid:107)∇(cid:96)(w, z) − ∇(cid:96)(u, z)(cid:107) ≥ µ(cid:107)w − u(cid:107);

(ii) The gradient norm is uniformly bounded by G over W, i.e., (cid:107)∇(cid:96)(w, z)(cid:107) ≤ G;
(iii) The loss is L-smooth over Rd, i.e., (cid:107)∇(cid:96)(w, z) − ∇(cid:96)(u, z)(cid:107) ≤ L(cid:107)w − u(cid:107);
(iv) Hessian is ρ-Lipschitz continuous over Rd, i.e., (cid:107)∇2(cid:96)(w, z) − ∇2(cid:96)(u, z)(cid:107) ≤ ρ(cid:107)w − u(cid:107).

We also require the following assumption on the tasks distribution. This assumption implies that,
with probability one, a set of ﬁnite samples generated from a distribution pi are all different.

5

Assumption 2. We assume Z is a Polish space (i.e., complete, separable, and metric) and FZ is
the Borel σ-algebra over Z. Moreover, for any i, pi is a non-atomic probability distribution over
(Z, FZ ), i.e., pi(z) = 0 for every z ∈ Z.

3.1 Training error

While the main focus of this paper is on studying the population error of MAML algorithm, we ﬁrst
study its training error which is required to provide characterization of the excess loss of MAML. To
do so, we ﬁrst state the following result from [24] and [13] on the strong convexity and smoothness
of (cid:96)(w − α∇ ˆL(w, D), z) for any batch D and any z ∈ Z.
Lemma 1 ([13] & [24]). If Assumption 1 holds, then for an arbitrary batch D and z ∈ Z, and
L , the function (cid:96)(w − α∇ ˆL(w, D), z) is 4L + 2αρG smooth over W. Furthermore,
with α ≤ 1
(cid:96)(w − α∇ ˆL(w, D), z) is µ

8 -strongly convex, if α ≤ min{ 1

2L , µ

8ρG }.

i , Dout

An immediate consequence of this Lemma is that the MAML empirical loss ˆF deﬁned in (4) is
also µ/8-strongly convex and 4L + 2αρG smooth over W. In addition, it can be shown that the
norm of gi(w; Din
i ) deﬁned in (6), which is the unbiased gradient estimate used in MAML, is
uniformly bounded above; for more details check Lemma 5 in Appendix A. Having these properties
of the MAML empirical loss established, we next state the following proposition on the training
error of MAML. This result is obtained by slightly modifying the well-known results on the conver-
gence of SGD in [30–32] in order to take into account the stepsize constraints that are imposed by
generalization analysis. For completeness, the proof of this result is provided in Appendix B.
Proposition 1. Consider ˆF (., S) deﬁned in (4) with α ≤ min{ 1
8
for MAML with βt = min(β,

µ(t+1) ) for β ≤ 8/µ, and for any set S, the last iterate wT satisﬁes

8ρG }. If Assumption 1 holds, then

2L , µ

E

(cid:104) ˆF (wT , S) − ˆF (w∗

S , S)

(cid:105)

≤ O(1)

G2(1 + 1
µ2

βµ )

(cid:18) L + ραG
T

(cid:19)

,

+

G
√
T

(7)

and the time-average of iterates ¯wT satisﬁes

E

(cid:104) ˆF ( ¯wT , S) − ˆF (w∗

S , S)

(cid:105)

≤ O(1)

G2(log(T ) + 1
µT

βµ )

,

where w∗

S := arg minw∈W

ˆF (., S) and the expectations are taken over the randomness of algorithm.

In the above expressions, the notation O(1) only hides absolute constants. It is worth noting that the
S , S) = 0.
term G/

S be a minimizer of the unconstrained problem, i.e., ∇ ˆF (w∗

T in (7) vanishes, if w∗

√

3.2 Generalization error

We derive our generalization bounds for MAML by establishing its algorithmic stability properties.
The stability approach has been used widely to characterize the generalization properties for optimiza-
tion algorithms such as stochastic gradient descent [18] or differentially private methods [33]. These
arguments are based on showing the uniform stability of algorithms [17] which we restate it here.
Deﬁnition 1 ([17]). Consider the problem of minimizing the empirical function ˆL(w, H) for some
dataset H. A randomized algorithm A with output wH given dataset H is called γ-uniformly stable
if the following condition holds: Take the dataset ˜H which is the same as H, except at one data
EA [|(cid:96)(wH, ˜z) − (cid:96)(w ˜H, ˜z)|] ≤ γ, where the expectation is taken over
points. Then, we have sup˜z∈Z
the randomness of A.

The above deﬁnition captures the stability of an algorithm. Speciﬁcally, it states that Algorithm A is
γ-stable, if the resulting loss of its outputs, when it is run using to two different datasets that only
differ in one data point, are at most γ away from each other. Note that the above deﬁnition holds if
the difference between the losses evaluated at any point ˜z is bounded by γ. The main importance
of this deﬁnition is its connection with generalization error. In particular, it can be shown that if an
algorithm is γ-uniformly stable and “symmetric", then its generalization error is bounded above by γ;
see, e.g., [17]. Next, we formally state the deﬁnition of a symmetric algorithm.

6

Deﬁnition 2. An algorithm A : Z n → Rd is called symmetric, if for any S ⊂ Z n, the distribution
of its output, i.e., A(S), does not depend on the ordering of elements of S, i.e., if we take S (cid:48) as a
permutation of S, the distribution of A(S) and A(S (cid:48)) would be similar.

Note that Deﬁnition 1 is useful for the case where we measure the performance of a model w by its
loss function over a sample, i.e., (cid:96)(w, ˜z). However, in this paper we measure the performance of a
model by looking at its loss after one step of SGD which involves K data points, as deﬁned in (6).
Therefore, we cannot directly use Deﬁnition 1 for characterizing the generalization error of MAML.
In fact, in what follows, we ﬁrst propose a modiﬁed version of the uniform stability deﬁnition, which
is compatible with our setting, and then show how such stability could lead to generalization bounds
for MAML-type algorithms.
Deﬁnition 3. Consider the problem in (4). A randomized algorithm A with output wS given dataset
S is called (γ, K)-uniformly stable if the following condition holds for any i ∈ {1, . . . m}: Take the
dataset ˜S which is the same as S, except that ˜S in
in at most K and
one data points, respectively. Then, for any ˜z ∈ Z and any K distinct points {z1, ..., zK} in Z,
(cid:17)
j=1), ˜z

wS − α∇ ˆL(wS , {zj}K

w ˜S − α∇ ˆL(w ˜S , {zj}K

i differ from S in

i and ˜S out

i and S out

j=1), ˜z

≤ γ,

EA

− (cid:96)

(cid:16)

(cid:16)

(cid:105)

i

(cid:104)(cid:12)
(cid:12)
(cid:12)(cid:96)

(cid:17)(cid:12)
(cid:12)
(cid:12)

where the expectation is taken over the randomness of A.

i

i , while we change only one point of the set S out

A few remarks about the above deﬁnition follow. First, one might wonder, why it is needed to change
K points of the set S in
. Note that, going from (1) to
(cid:1) possible batches Din
[.] is replaced by the sum over all (cid:0) n
(3), the expectation EDtest
i of size K from
K
i . In other words, for the empirical sum in (3), each batch Din
S in
i can be seen as a data unit. That said,
and similar to Deﬁnition 1, to characterize the stability, we need to change one data unit which is one
batch of size K. That is why we change K data points of S in
in the deﬁnition of (γ, K)-uniformly
i
stability. On the other hand, we replace Li(.) = Ez∼pi [(cid:96)(., z)] in (1) with a sum over n points of S out
in (3), and thus, for this one, each data unit is just a single data point. So, similar to Deﬁnition 1, we
just change one data point for the set S out

.

i

i

i

Second, it is worth comparing this deﬁnition with the other deﬁnition given for stability of meta-
learning algorithms in [28]. In that paper, the deﬁnition of stability is based on modifying the whole
dataset Si rather than what we do here which is changing just K + 1 points. While taking such a
deﬁnition makes the analysis relatively simpler, it prohibits us from characterizing the dependence
of generalization error on n, and hence the resulting upper bound for generalization error would be
larger. We will come back to this point later when we derive the stability of MAML with respect to
Deﬁnition 3 and compare it with the one obtained in [28].

As we discussed, the main reason that we are interested in the uniform stability of an algorithm is its
connection with generalization error. In the next theorem, we formalize this connection for MAML
formulation and show that if an Algorithm A is (γ, K)-uniformly stable and symmetric, then its
output generalization error is bounded above by γ. The proof of this result is available in Appendix C.
Theorem 1. Consider the population and empirical losses deﬁned in (2) and (4), respectively.
If Assumption 2 holds and A is a (possibly randomized) symmetric and (γ, K)-uniformly stable
algorithm with output wS ∈ W, then EA,S

(cid:104)
F (wS ) − ˆF (wS , S)

≤ γ.

(cid:105)

This result shows that if we prove a symmetric algorithm is (γ, K)-uniformly stable as deﬁned in
Deﬁnition 3, then we can bound its output model generalization error by γ. Hence, to characterize the
generalization error of the model trained by MAML algorithm, we only need to capture the uniform
stability parameter of MAML. Before stating this result, it is worth noting that while we limit our
focus to MAML in this paper, Deﬁnition 3 and Theorem 1 could provide a framework for studying
the generalization properties of a broader class of gradient-based meta-learning algorithms such as
Reptile [34], First-order MAML [1], and Hessian-Free MAML [13].
Theorem 2. If Assumption 1 holds, then MAML (Algorithm 1) with both last iterate and average
iterate outputs and with α ≤ min{ 1
4L+2αρG is (γ, K)-uniformly stable, where
γ := O(1) G2(1+αLK)

8ρG } and βt ≤

2L , µ

1

.

mnµ

According to the above discussion, the result of Theorem 2 guarantees that the generalization error of
MAML solution decays by a factor of O(K/mn), where m is the number of tasks in the training set

7

and n is the number of available samples per task. The classic lower bound for SGD over strongly
convex functions translates to a O(1/mn) lower bound in our setting. Hence, our bound is tight
in the small K regime, which is generally the case in few-shot learning problems. However, one
shortcoming of this result is that it is not tight in the large K regime. In Appendix E we show how
we could improve this result for the large K regime. However, throughout the paper, we keep our
discussion limited to the small K regime.
Remark 1. If instead of using our uniform stability deﬁnition (i.e., Deﬁnition 3), one uses the stability
deﬁnition given in [28], the resulted stability constant γ would be proportional to (1/m) rather than
(1/mn). In fact, our proposed uniform-stability deﬁnition empowers us to obtain a better bound and
indicates the role of number of samples per task n in the generalization error.
Remark 2. The algorithmic stability technique is mainly limited to the convex setting, since, in the
nonconvex case, we need to keep learning rate very small to obtain meaningful generalization results
which makes it impractical (Check Appendix G for further discussions on this matter). In fact, the
main reason that we assume (cid:96) is strongly convex and α ≤ Ω(µ) is to ensure that the meta-objective
is convex, as, in general, relaxing any of these two could lead to a nonconvex meta-objective function.
However, these two assumptions together make the objective function strongly convex, which is not
necessarily needed in our analysis. In fact, if we assume that (cid:96) and the meta-function are convex
(but not necessarily strongly convex), we could still use Deﬁnition 3 to derive similar generalization
bounds.

Putting Proposition 1 and Theorem 2 together, we obtain the following result on the excess population
loss of MAML algorithm. We only report the result for the averaged iterates here, but one can obtain
the result for the last iterate similarly by using Proposition 1.
Proposition 2. Consider the function F deﬁned in (2) with α ≤ min{ 1
If Assump-
tions 1 and 2 hold, then the average of iterates generated by MAML (Algorithm 1) with βt =
min(

µ(t+1) ) after T iterations satisﬁes

1
4L+2αρG ,

2L , µ

8ρG }.

8

(cid:104)

EA,S

F ( ¯wT ) − min
W

F

(cid:105)

≤ O(1)

G2
µ

(cid:18) log(T ) + L/µ
T

+

1 + αLK
mn

(cid:19)

,

where the expectation is taken over the sampling of S and the randomness of MAML algorithm.

As an immediate application, the following corollary characterizes MAML test error.
Corollary 1. Under the premise of Proposition 2, MAML algorithm after T = ˜O(mnL/µ) iterations
returns ¯wT such that EA,S

(cid:2)F ( ¯wT ) − minW F (cid:3) ≤ O (cid:0)G2(1 + αLK)/(mnµ)(cid:1) .

3.3 Generalization to an unseen task

As we discussed in Section 2, another generalization measure is how the model trained with respect to
the empirical problem in (4) performs on a new and unseen task Tm+1 with corresponding distribution
pm+1. To state our result for this case, we ﬁrst need to introduce the following distance notion between
probability distributions.
Deﬁnition 4. For two distributions P and Q, deﬁned over the sample space Ω and σ-ﬁeld F, the
total variation distance is deﬁned as (cid:107)P − Q(cid:107)T V := supA∈F |P (A) − Q(A)|.

It is well-known that the total variation distance admits the following characterization

(cid:107)P − Q(cid:107)T V = sup

f :0≤f ≤1

Ex∼P [f (x)] − Ex∼Q[f (x)].

(8)

Also, we require the following boundedness assumption for our result.
Assumption 3. For any z ∈ Z, the function (cid:96)(., z) is M -bounded over W.

Considering these assumptions, we are ready to state our result for the case when the task at test time
is a new task and is not observed during training.
Theorem 3. Consider the population losses deﬁned in (1) and (2). Suppose Assumptions 1, 2 and 3
hold. Then, for any w ∈ W, we have

|Fm+1(w) − F (w)| ≤ D(pm+1, {pi}m

i=1),

(9)

8

where

D(pm+1, {pi}m

i=1) :=

4αG2
m

m
(cid:88)

i=1

(cid:107)pm+1 − pi(cid:107)T V + (M + 2αG2)(cid:107)pm+1 −

1
m

m
(cid:88)

i=1

pi(cid:107)T V .

(10)

While the proof is provided in detail in Appendix F, here we discuss a sketch of it to highlight the
main technical contributions. To simplify the notation here, let us assume m = 1, meaning that p1 is
the distribution used for training and p2 is the distribution corresponding to the new task. Note that
we aim to bound |F2(w) − F1(w)|. Recalling the deﬁnition of population loss (2), we need to bound
the following expression (we drop the absolute value due to symmetry)

E

{z2

j ∼p2}K

j=1,˜z2∼p2

(cid:2)l(cid:0)w − α∇ ˆL(w, {z2

j }j), ˜z2(cid:1)(cid:3)− E

{z1

j ∼p1}K

j=1,˜z1∼p1

(cid:2)l(cid:0)w − α∇ ˆL(w, {z1

j }j), ˜z1(cid:1)(cid:3).
(11)

(cid:16)

Notice that this difference can be cast as E

({zj }K

j=1,˜z)∼pK+1

2

[X] − E

({zj }K

j=1,˜z)∼pK+1

1

[X], with

w − α∇ ˆL(w, {zj}K

X := l
. As a result, a naive approach would be using Lipschitz and
boundedness properties of l (Assumptions 1 and 3) along with (8) to obtain a bound depending on
(cid:107)pK+1
1

(cid:107)T V = O(K)(cid:107)p1 − p2(cid:107)T V . However, this bound is not tight as it grows with K.

− pK+1
2

(cid:17)
j=1), ˜z

j and z2

j and z2

j and z2

j ∼ p1, z2

j ∼ p2, and µ(z1

j . That said, for each j, we assume that z1

To address this issue, we exploit a coupling technique. Note that the expression in (11) does not
depend on the joint distribution of z1
j , and instead, it only depends on the marginal distribution
of z1
j are sampled from a distribution µ on
j (cid:54)= z2
Z × Z such that z1
j ) = (cid:107)p1 − p2(cid:107)T V . Such a coupling exists and
is called maximal coupling of p1 and p2 [35]. Using this idea, as we show in Appendix F, we can
eliminate the dependence on K, and as a result, the upper bound in (9) is independent of number of
available labeled samples at test time denoted by K.
Remark 3. Note that the terms 1
m
D(pm+1, {pi}m
the empirical problem (4).
arg minw∈W
D(pm+1, {pi}m

i=1 pi(cid:107)T V in
i=1) come from the fact that we consider uniform distribution over tasks in
if we instead consider the empirical problem
then

i=1 qi ˆFi(w, Si), for some non-negative weights qi with (cid:80)m

i=1 (cid:107)pm+1 − pi(cid:107)T V and (cid:107)pm+1 − 1
m

(cid:80)m
i=1) on the right hand side of (9) would change to
m
m
(cid:88)
(cid:88)

i=1 qi = 1,

In particular,

(cid:80)m

(cid:80)m

(M + 2αG2)(cid:107)pm+1 −

qipi(cid:107)T V + 12αG2

qi(cid:107)pm+1 − pi(cid:107)T V .

i=1

i=1

This result shows that by changing the training problem we can achieve a lower generalization error
for MAML, if we have some information about the distribution pm+1 at training time. For instance,
if we know pm+1 will be much closer to p1 compared to p2, making the weight of p1 larger than p2
would decrease the generalization error of MAML.
Corollary 2. Recall the population loss Fm+1 deﬁned in (2) and D(pm+1, {pi}m
i=1) deﬁned in
Theorem 3. Let A be an algorithm for solving the empirical problem (4) which achieves (cid:15) excess risk,
i.e., EA,S [F (A(S))] − minW F ≤ (cid:15). If Assumptions 1, 2 and 3 hold, then algorithm A ﬁnds a model
wS which achieves (cid:15) + D(pm+1, {pi}m

EA,S [Fm+1(wS )] − min
W

i=1) excess loss with respect to Fm+1,
Fm+1 ≤ (cid:15) + 2D(pm+1, {pi}m

i=1).

mn + D(pm+1, {pi}m

This corollary and Proposition 2 together imply that the MAML algorithm’s test error with respect
to the new task Tm+1 is O(1) (cid:0) 1
i=1)(cid:1). As a result, if the new task’s distribution
pm+1 is sufﬁciently close to the other tasks’ distributions, MAML will have a low test error on the
new unseen task. On the other hand, if pm+1 is far from p1, . . . , pm in TV distance, then test error of
the model trained {T m
i=1} over Tm+1 could be potentially large. In Appendix F.2 we show how this
result can be extended to the case that the task at test time is generated from a distribution over both
recurring tasks {Ti}m

i=1 and the unseen task Tm+1.

4 Conclusion and future work

In this work, we studied the generalization of MAML algorithm in two key cases: a) when the
test time task is a recurring task from the ones observed during the training stage, b) when it is a

9

new and unseen one. For the ﬁrst one, and under strong convexity assumption, we showed that the
generalization error improves as the number of tasks or the number of samples per task increases. For
the second case, we showed that when the distance between the unseen task’s distribution and the
distributions of training tasks is sufﬁciently small, the MAML output generalizes well to the new task
revealed at test time.

While we focused on the convex case in this paper, deriving generalization bounds when the meta-
function is nonconvex is a natural future direction to explore. However, this could be challenging
since the generalization of gradient methods is not well understood in the nonconvex setting even for
the classic supervised learning problem.

5 Acknowledgment

Alireza Fallah acknowledges support from the Apple Scholars in AI/ML PhD fellowship and the
MathWorks Engineering Fellowship. This research is sponsored by the United States Air Force
Research Laboratory and the United States Air Force Artiﬁcial Intelligence Accelerator and was
accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions
contained in this document are those of the authors and should not be interpreted as representing the
ofﬁcial policies, either expressed or implied, of the United States Air Force or the U.S. Government.
The U.S. Government is authorized to reproduce and distribute reprints for Government purposes
notwithstanding any copyright notation herein. This research of Aryan Mokhtari is supported in part
by NSF Grant 2007668, ARO Grant W911NF2110226, the Machine Learning Laboratory at UT
Austin, and the NSF AI Institute for Foundations of Machine Learning.

References

[1] C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning for fast adaptation of deep
networks,” in Proceedings of the 34th International Conference on Machine Learning, (Sydney,
Australia), 06–11 Aug 2017.

[2] A. Nichol, J. Achiam, and J. Schulman, “On ﬁrst-order meta-learning algorithms,” arXiv

preprint arXiv:1803.02999, 2018.

[3] M. Khodak, M.-F. F. Balcan, and A. S. Talwalkar, “Adaptive gradient-based meta-learning
methods,” in Advances in Neural Information Processing Systems, pp. 5915–5926, 2019.

[4] B. Baker, O. Gupta, N. Naik, and R. Raskar, “Designing neural network architectures using
reinforcement learning,” in International Conference on Learning Representations, 2017.

[5] B. Zoph and Q. V. Le, “Neural architecture search with reinforcement learning,” in International

Conference on Learning Representations, 2017.

[6] B. Zoph, V. Vasudevan, J. Shlens, and Q. V. Le, “Learning transferable architectures for scalable
image recognition,” in Proceedings of the IEEE conference on computer vision and pattern
recognition, pp. 8697–8710, 2018.

[7] S. Ravi and H. Larochelle, “Optimization as a model for few-shot learning,” in International

Conference on Learning Representations, 2017.

[8] M. Andrychowicz, M. Denil, S. Gómez, M. W. Hoffman, D. Pfau, T. Schaul, B. Shillingford,
and N. de Freitas, “Learning to learn by gradient descent by gradient descent,” in Advances in
Neural Information Processing Systems 29, pp. 3981–3989, Curran Associates, Inc., 2016.

[9] A. Antoniou, H. Edwards, and A. Storkey, “How to train your MAML,” in International

Conference on Learning Representations, 2019.

[10] Z. Li, F. Zhou, F. Chen, and H. Li, “Meta-SGD: Learning to learn quickly for few-shot learning,”

arXiv preprint arXiv:1707.09835, 2017.

[11] E. Grant, C. Finn, S. Levine, T. Darrell, and T. Grifﬁths, “Recasting gradient-based meta-
learning as hierarchical bayes,” in International Conference on Learning Representations,
2018.

10

[12] H. S. Behl, A. G. Baydin, and P. H. S. Torr, “Alpha MAML: adaptive model-agnostic meta-

learning,” 2019.

[13] A. Fallah, A. Mokhtari, and A. Ozdaglar, “On the convergence theory of gradient-based model-
agnostic meta-learning algorithms,” in International Conference on Artiﬁcial Intelligence and
Statistics, pp. 1082–1092, 2020.

[14] R. Xu, L. Chen, and A. Karbasi, “Meta learning in the continuous time limit,” arXiv preprint

arXiv:2006.10921, 2020.

[15] K. Ji, J. Yang, and Y. Liang, “Multi-step model-agnostic meta-learning: Convergence and

improved algorithms,” arXiv preprint arXiv:2002.07836, 2020.

[16] L. Wang, Q. Cai, Z. Yang, and Z. Wang, “On the global optimality of model-agnostic meta-

learning,” in International Conference on Machine Learning, pp. 9837–9846, PMLR, 2020.

[17] O. Bousquet and A. Elisseeff, “Stability and generalization,” Journal of machine learning

research, vol. 2, no. Mar, pp. 499–526, 2002.

[18] M. Hardt, B. Recht, and Y. Singer, “Train faster, generalize better: Stability of stochastic
gradient descent,” in International Conference on Machine Learning, pp. 1225–1234, PMLR,
2016.

[19] A. Rajeswaran, C. Finn, S. M. Kakade, and S. Levine, “Meta-learning with implicit gradients,” in
Advances in Neural Information Processing Systems (H. Wallach, H. Larochelle, A. Beygelzimer,
F. d'Alché-Buc, E. Fox, and R. Garnett, eds.), vol. 32, pp. 113–124, Curran Associates, Inc.,
2019.

[20] L. Collins, A. Mokhtari, and S. Shakkottai, “Task-robust model-agnostic meta-learning,” Ad-

vances in Neural Information Processing Systems, vol. 33, 2020.

[21] V. Likhosherstov, X. Song, K. Choromanski, J. Davis, and A. Weller, “Ufo-blo: Unbiased

ﬁrst-order bilevel optimization,” arXiv preprint arXiv:2006.03631, 2020.

[22] T. Chen, Y. Sun, and W. Yin, “Solving stochastic compositional optimization is nearly as easy

as solving stochastic optimization,” arXiv preprint arXiv:2008.10847, 2020.

[23] Y. Hu, S. Zhang, X. Chen, and N. He, “Biased stochastic gradient descent for conditional

stochastic optimization,” ArXiv, vol. abs/2002.10790, 2020.

[24] C. Finn, A. Rajeswaran, S. Kakade, and S. Levine, “Online meta-learning,” in Proceedings of
the 36th International Conference on Machine Learning, vol. 97 of Proceedings of Machine
Learning Research, (Long Beach, California, USA), pp. 1920–1930, PMLR, 09–15 Jun 2019.

[25] A. Fallah, A. Mokhtari, and A. Ozdaglar, “Personalized federated learning with theoretical guar-
antees: A model-agnostic meta-learning approach,” Advances in Neural Information Processing
Systems, vol. 33, 2020.

[26] H. Liu, R. Socher, and C. Xiong, “Taming maml: Efﬁcient unbiased meta-reinforcement

learning,” in International Conference on Machine Learning, pp. 4061–4071, PMLR, 2019.

[27] A. Fallah, K. Georgiev, A. Mokhtari, and A. Ozdaglar, “Provably convergent policy gradient
methods for model-agnostic meta-reinforcement learning,” arXiv preprint arXiv:2002.05135,
2020.

[28] J. Chen, X.-M. Wu, Y. Li, Q. Li, L.-M. Zhan, and F.-l. Chung, “A closer look at the training
strategy for modern meta-learning,” Advances in Neural Information Processing Systems,
vol. 33, 2020.

[29] S. Guiroy, V. Verma, and C. Pal, “Towards understanding generalization in gradient-based

meta-learning,” arXiv preprint arXiv:1907.07287, 2019.

[30] A. Rakhlin, O. Shamir, and K. Sridharan, “Making gradient descent optimal for strongly convex

stochastic optimization,” arXiv preprint arXiv:1109.5647, 2011.

11

[31] E. Hazan, A. Agarwal, and S. Kale, “Logarithmic regret algorithms for online convex optimiza-

tion,” Machine Learning, vol. 69, no. 2-3, pp. 169–192, 2007.

[32] A. Nemirovski, A. Juditsky, G. Lan, and A. Shapiro, “Robust stochastic approximation approach
to stochastic programming,” SIAM Journal on Optimization, vol. 19, no. 4, pp. 1574–1609,
2009.

[33] R. Bassily, V. Feldman, K. Talwar, and A. Guha Thakurta, “Private stochastic convex opti-
mization with optimal rates,” Advances in Neural Information Processing Systems, vol. 32,
pp. 11282–11291, 2019.

[34] A. Nichol, J. Achiam, and J. Schulman, “On ﬁrst-order meta-learning algorithms,” arXiv

preprint arXiv:1803.02999, 2018.

[35] F. Den Hollander, “Probability theory: The coupling method,” Lecture notes available online

(http://websites. math. leidenuniv. nl/probability/lecturenotes/CouplingLectures. pdf), 2012.

[36] Y. Nesterov, Introductory Lectures on Convex Optimization: A Basic Course, vol. 87. Springer,

2004.

12

