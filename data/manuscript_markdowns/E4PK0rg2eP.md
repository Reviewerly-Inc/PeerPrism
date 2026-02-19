Under review as a conference paper at ICLR 2021

PARAMETER-EFFICIENT TRANSFER LEARNING WITH
DIFF PRUNING

Anonymous authors
Paper under double-blind review

ABSTRACT

While task-speciﬁc ﬁnetuning of deep networks pretrained with self-supervision
has led to signiﬁcant empirical advances in NLP, their large size makes the stan-
dard ﬁnetuning approach difﬁcult to apply to multi-task, memory-constrained set-
tings, as storing the full model parameters for each task become prohibitively
expensive. We propose diff pruning as a simple approach to enable parameter-
efﬁcient transfer learning within the pretrain-ﬁnetune framework. This approach
views ﬁnetuning as learning a task-speciﬁc “diff” vector that is applied on top of
the pretrained parameter vector, which remains ﬁxed and is shared across different
tasks. The diff vector is adaptively pruned during training with a differentiable ap-
proximation to the L0-norm penalty to encourage sparsity. Diff pruning becomes
parameter-efﬁcient as the number of tasks increases, as it requires storing only the
nonzero positions and weights of the diff vector for each task, while the cost of
storing the shared pretrained model remains constant. We ﬁnd that models ﬁne-
tuned with diff pruning can match the performance of fully ﬁnetuned baselines
on the GLUE benchmark while only modifying 0.5% of the pretrained model’s
parameters per task.

1

INTRODUCTION

Task-speciﬁc ﬁnetuning of pretrained deep networks has become the dominant paradigm in contem-
porary NLP, achieving state-of-the-art results across a suite of natural language understanding tasks
(Devlin et al., 2019; Liu et al., 2019c; Yang et al., 2019; Lan et al., 2020). While straightforward and
empirically effective, this approach is difﬁcult to scale to multi-task, memory-constrained settings
(e.g. for on-device applications), as it requires shipping and storing a full set of model parameters for
each task. Inasmuch as these models are learning generalizable, task-agnostic language representa-
tions through self-supervised pretraining, ﬁnetuning the entire model for each task is an especially
inefﬁcient use of model parameters.

A popular approach to parameter-efﬁciency is to learn sparse models for each task where a subset
of the ﬁnal model parameters are exactly zero (Gordon et al., 2020; Sajjad et al., 2020; Zhao et al.,
2020; Sanh et al., 2020). Such approaches often face a steep sparsity/performance tradeoff, and a
substantial portion of nonzero parameters (e.g. 10%-30%) are still typically required to match the
performance of the dense counterparts. An alternative is to use multi-task learning or feature-based
transfer for more parameter-efﬁcient transfer learning with pretrained models (Liu et al., 2019b;
Clark et al., 2019; Stickland & Murray, 2019; Reimers & Gurevych, 2019; Feng et al., 2020). These
methods learn only a small number of additional parameters (e.g. a linear layer) on top of a shared
model. However, multi-task learning generally requires access to all tasks during training to prevent
catastrophic forgetting (French, 1999), while feature-based transfer learning (e.g. based on task-
agnostic sentence representations) is typically outperformed by full ﬁnetuning (Howard & Ruder,
2018).

Adapters (Rebufﬁ et al., 2018) have recently emerged as a promising approach to parameter-
efﬁcient transfer learning within the pretrain-ﬁnetune paradigm (Houlsby et al., 2019; Pfeiffer et al.,
2020a;b;c). Adapter layers are smaller, task-speciﬁc modules that are inserted between layers of a
pretrained model, which remains ﬁxed and is shared across tasks. These approaches do not require
access to all tasks during training making them attractive in settings where one hopes to obtain and
share performant models as new tasks arrive in stream. Houlsby et al. (2019) ﬁnd that adapter lay-
ers trained on BERT can match the performance of fully ﬁnetuned BERT on the GLUE benchmark
(Wang et al., 2019a) while only requiring 3.6% additional parameters (on average) per task.

1

Under review as a conference paper at ICLR 2021

In this work, we consider a similar setting as adapters but propose a new diff pruning approach with
the goal of even more parameter-efﬁcient transfer learning. Diff pruning views ﬁnetuning as learning
a task-speciﬁc difference vector that is applied on top of the pretrained parameter vector, which
remains ﬁxed and is shared across different tasks. In order to learn this vector, we reparameterize
the task-speciﬁc model parameters as θtask = θpretrained +δtask, where the pretrained parameter vector
θpretrained is ﬁxed and the task-speciﬁc diff vector δtask is ﬁnetuned. The diff vector is regularized
with a differentiable approximation to the L0-norm penalty (Louizos et al., 2018) to encourage
sparsity. This approach can become parameter-efﬁcient as the number of tasks increases as it only
requires storing the nonzero positions and weights of the diff vector for each task. The cost of
storing the shared pretrained model remains constant and is amortized across multiple tasks. On
the GLUE benchmark (Wang et al., 2019a), diff pruning can match the performance of the fully
ﬁnetuned BERT baselines while ﬁnetuning only 0.5% of the pretrained parameters per task, making
it a potential alternative to adapters for parameter-efﬁcient transfer learning.

2 BACKGROUND: TRANSFER LEARNING FOR NLP

The ﬁeld of NLP has recently seen remarkable progress through transfer learning with a pretrain-
and-ﬁnetune paradigm, which initializes a subset of the model parameters for all tasks from a pre-
trained model and then ﬁnetunes on a task speciﬁc objective. Pretraining objectives include context
prediction (Mikolov et al., 2013), autoencoding (Dai & Le, 2015), machine translation (McCann
et al., 2017), and more recently, variants of language modeling (Peters et al., 2018; Radford et al.,
2018; Devlin et al., 2019) objectives.

Here we consider applying transfer learning to multiple tasks. We consider a setting with a poten-
tially unknown set of tasks, where each τ ∈ T has an associated training set {x(n)
n=1.1 For
all tasks, the goal is to produce (possibly tied) model parameters θτ to minimize the empirical risk,

τ }N

, y(n)

τ

min
θτ

1
N

N
(cid:88)

n=1

(cid:16)

L

f (x(n)
τ

; θτ ), y(n)

τ

(cid:17)

+ λR(θτ )

where f (·; θ) is a parameterized function over the input (e.g. a neural network), L(·, ·) is a loss
function (e.g. cross-entropy), and R(·) is an optional regularizer with hyperparameter λ.

This multi-task setting can use the pretrain-then-ﬁnetune approach by simply learning indepen-
dent parameters for each task; however the large size of pretrained models makes this approach
exceedingly parameter inefﬁcient. For example, widely-adopted models such as BERTBASE and
BERTLARGE have 110M and 340M parameters respectively, while their contemporaries such as
T5 (Raffel et al., 2020), Megatron-LM (Shoeybi et al., 2019), and Turing-NLG (Rajbhandari
et al., 2019) have parameter counts in the billions. Storing the fully ﬁnetuned models becomes
difﬁcult even for a moderate number of tasks.2 A classic approach to tackling this parameter-
inefﬁciency (Caruana, 1997) is to train a single shared model (along with a task-speciﬁc output
layer) against multiple tasks through joint training. However, the usual formulation of multi-task
learning requires the set of tasks T to be known in advance in order to prevent catastrophic forget-
ting (French, 1999),3 making it unsuitable for applications in which the set of tasks is unknown (e.g.
when tasks arrive in stream).

3 DIFF PRUNING

Diff pruning formulates task-speciﬁc ﬁnetuning as learning a diff vector δτ that is added to the
pretrained model parameters θpretrained. We ﬁrst reparameterize the task-speciﬁc model parameters,

θτ = θpretrained + δτ ,

1Therefore our setup is different from the classic multitask setting which usually assumes that set of tasks

is known

2An intriguing line of work suggests that large-scale language models can be used without ﬁnetuning for a
variety of tasks if given the appropriate context (Radford et al., 2019; Brown et al., 2020). While interesting,
these models generally underperform task-speciﬁc models and require billions of parameters, though recent
work suggests that they can be made substantially smaller (Schick & Schutze, 2020).

3However, work on continual learning mitigates these issues to an extent (Shin et al., 2017; Lopez-Paz &

Ranzato, 2017; Lee et al., 2017; Kirkpatrick et al., 2017; Parisi et al., 2018).

2

Under review as a conference paper at ICLR 2021

which results in the following empirical risk minimization problem,

min
δτ

1
N

N
(cid:88)

n=1

(cid:16)

L

f (x(n)
τ

; θpretrained + δτ ), y(n)

τ

(cid:17)

+ λR(θpretrained + δτ ).

This trivial reparameterization is equivalent to the original formulation. Its beneﬁt comes in the
multi-task setting where the cost of storing the pretrained parameters θpretrained is amortized across
tasks, and the only marginal cost for new tasks is the diff vector. If we can regularize δτ to be sparse
such that (cid:107)δτ (cid:107)0 (cid:28) (cid:107)θpretrained(cid:107)0, then this approach can become more parameter-efﬁcient as the
number of tasks increases. We can specify this goal with an L0-norm penalty on the diff vector,

R(θpretrained + δτ ) = (cid:107)δτ (cid:107)0 =

d
(cid:88)

i=1

1{δτ,i (cid:54)= 0}.

3.1 DIFFERENTIABLE APPROXIMATION TO THE L0-NORM

This regularizer is difﬁcult to directly optimize as it is non-differentiable. In order to approximate
this L0 objective, we follow the standard approach for gradient-based learning with L0 sparsity
using a relaxed mask vector (Louizos et al., 2018). This approach involves relaxing a binary vector
into continuous space, and then multiplying it with a dense weight vector to determine how much
of the weight vector is applied during training. After training, the mask is deterministic and a large
portion of the diff vector is true zero.

To apply this method we ﬁrst decompose δτ into a binary mask vector multiplied with a dense
vector,

δτ = zτ (cid:12) wτ ,

zτ ∈ {0, 1}d, wτ ∈ Rd

We can now instead optimize an expectation with respect to zτ , whose distribution p(zτ ; ατ ) is
initially Bernoulli with parameters ατ ,

min
ατ ,wτ

Ezτ ∼p(zτ ;ατ )

(cid:34)

1
N

N
(cid:88)

n=1

(cid:16)

L

f (x(n)
τ

; θpretrained + zτ (cid:12) wτ , ), y(n)

τ

(cid:17)

(cid:35)

+ λ(cid:107)δτ (cid:107)0

.

This objective is still difﬁcult in practice due to zτ ’s being discrete (which requires the score func-
tion gradient estimator), but the expectation provides some guidance for empirically effective relax-
ations. We follow prior work (Louizos et al., 2018; Wang et al., 2019b) and relax zτ into continuous
space [0, 1]d with a stretched Hard-Concrete distribution (Jang et al., 2017; Maddison et al., 2017),
which allows for the use of pathwise gradient estimators. Speciﬁcally, zτ is now deﬁned to be a
deterministic and (sub)differentiable function of a sample u from a uniform distribution,

u ∼ U (0, 1),
¯sτ = sτ × (r − l) + l,

sτ = σ (log u − log(1 − u) + ατ ) ,
zτ = min(1, max(0, ¯sτ )).

Here l < 0 and r > 1 are two constants used to stretch sτ into the interval (l, r)d before it is clamped
to [0, 1]d with the min(1, max(0, ·)) operation. In this case we have a differentiable closed-form
expression for the expected L0-norm,

E [(cid:107)δτ (cid:107)0] =

d
(cid:88)

i=1

E [1{zτ,i > 0}] =

d
(cid:88)

i=1

(cid:18)

σ

ατ,i − log

(cid:19)

.

−l
r

Thus the ﬁnal optimization problem is given by,
N
(cid:88)

(cid:16)

(cid:34)

min
ατ ,wτ

Eu∼U [0,1]

1
N

n=1

L

f (x(n)
τ

; θpretrained + zτ (cid:12) wτ , ), y(n)

τ

(cid:35)

(cid:17)

+ λ

d
(cid:88)

i=1

(cid:18)

σ

ατ,i − log

(cid:19)

,

−l
r

and we can now utilize pathwise gradient estimators to optimize the ﬁrst term with respect to ατ
since the expectation no longer depends on it.4 After training we obtain the ﬁnal diff vector δτ by
sampling u once to obtain zτ (which is not necessarily a binary vector but has a signiﬁcant number
of dimensions equal to exactly zero due to the clamping function), then setting δτ = zτ (cid:12) wτ .5

4To reduce notation clutter we subsume the parameters of the task-speciﬁc output layer, which is not pre-

trained, into θpretrained. We do not apply the L0-norm penalty on these parameters during training.

5We found sampling once to work as well as more complicated alternatives (e.g. based on multiple samples).

3

Under review as a conference paper at ICLR 2021

3.2 L0-BALL PROJECTION WITH MAGNITUDE PRUNING FOR SPARSITY CONTROL

Differentiable L0 regularization provides a strong way to achieve high sparsity rate. However, it
would be ideal to have more ﬁne-grained control into the exact sparsity rate in the diff vector, espe-
cially considering applications which require speciﬁc parameter budgets. As λ is just the Lagrangian
multiplier for the constraint E [(cid:107)δτ (cid:107)0] < η for some η, this could be achieved in principle by search-
ing over different values of λ. However we found it more efﬁcient and empirically effective to
achieve an exact sparsity rate by simply projecting onto the L0-ball after training.

Speciﬁcally we use magnitude pruning on the diff vector δτ and target a sparsity rate t% by only
keeping the top t% × d values in δτ .6 Note that unlike standard magnitude pruning, this is based
on the magnitude of the diff vector values and not the model parameters. As is usual in magnitude
pruning, we found it important to further ﬁnetune δτ with the nonzero masks ﬁxed to maintain good
performance (Han et al., 2016). Since this type of parameter-efﬁciency through projection onto the
L0-ball can be applied without adaptive diff pruning,7 such an approach will serve as one of our
baselines in the empirical study.

3.3 STRUCTURED DIFF PRUNING

Diff pruning, as presented above, is architecture-agnostic and does not exploit the underlying model
structure—each dimension of zτ is independent from one another. While this makes the approach
potentially more ﬂexible, we might expect to achieve better sparsity/performance tradeoff through
a structured formulation which encourages active parameters to group together and other areas to
be fully sparse. Motivated by this intuition, we ﬁrst partition the parameter indices into G groups
{g(1), . . . , g(G)} where g(j) is a subset of parameter indices governed by group g(j).8 We then
introduce a scalar zj
τ ) for each group g(j), and decompose the
task-speciﬁc parameter for index i ∈ g(j) as δj
τ × wτ,i. The expected L0-norm is then
given by,

τ (with the associated parameter αj

τ,i = zτ,i × zj

E [(cid:107)δτ (cid:107)0] =

G
(cid:88)

(cid:88)

j=1

i∈g(j)

E [1{zτ,i · zg

τ > 0}] =

G
(cid:88)

(cid:88)

j=1

i∈g(j)

(cid:18)

σ

ατ,i − log

(cid:19)

−l
r

(cid:18)

× σ

αj

τ − log

(cid:19)

,

−l
r

and we can train with gradient-based optimization as before.

4 EXPERIMENTS

4.1 MODEL AND DATASETS

For evaluation we use the GLUE benchmark (Wang et al., 2019b), a popular ﬁnetuning dataset.
Following adapters (Houlsby et al., 2019), we test our approach on the following subset of the GLUE
tasks: Multi-Genre Natural Language Inference (MNLI), where the goal is two predict whether the
relationship between two sentences is entailment, contradiction, or neutral (we test on both MNLIm
and MNLImm which respectively tests on matched/mismatched domains); Quora Question Pairs
(QQP), a classiﬁcation task to predict whether two question are semantically equivalent; Question
Natural Language Inference (QNLI), which must predict whether a sentence is a correct answer
to the question; Stanford Sentiment Treebank (SST-2), a sentence classiﬁcation task to predict the
sentiment of movie reviews; Corpus of Linguistic Acceptability (CoLA), where the goal is predict
whether a sentence is linguistically acceptable or not; Semantic Textual Similarity Benchmark (STS-
B), which must predict a similarity rating between two sentences; Microsoft Research Paraphrase
Corpus (MRPC), where the goal is to predict whether two sentences are semantically equivalent;
Recognizing Textual Entailment (RTE), which must predict whether a second sentence is entailed
by the ﬁrst. For evaluation, the benchmark uses Matthew’s correlation for CoLA, Spearman for
STS-B, F1 score for MRPC/QQC, and accuracy for MNLI/QNLI/SST-2/RTE.

6Wang et al. (2019b) show that it also is possible to inject such a constraint softly into the training objec-
tive by regularizing the expected model size towards a certain rate. However, since the constraint is soft this
approach also makes it difﬁcult to target an exact sparsity rate.

7Concretely, one can obtain θτ through usual ﬁnetuning, set δτ = θτ − θpretrained, and then apply magnitude

pruning followed by additional ﬁnetuning on δτ .

8While groups can be deﬁned in various ways, we found that deﬁning groups based on each matrix/bias

vector of the pretrained model was simple and worked well enough.

4

Under review as a conference paper at ICLR 2021

Total New params
params

per task

QNLI∗ SST-2 MNLIm MNLImm CoLA MRPC STS-B RTE QQP

Full ﬁnetuning
Adapters (8-256)
Adapters (64)

9.00×
1.32×
1.19×

9.00×
Full ﬁnetuning
1.34×
Last layer
Non-adap. diff pruning 1.05×
1.05×
Diff pruning
1.05×
Diff pruning (struct.)

100%
3.6%
2.1%

100%
3.8%
0.5%
0.5%
0.5%

91.1
90.7
91.4

93.4
79.8
89.7
92.9
93.3

94.9
94.0
94.2

94.1
91.6
93.6
93.8
94.1

86.7
84.9
85.3

86.7
71.4
84.9
85.7
86.4

85.9
85.1
84.6

86.0
72.9
84.8
85.6
86.0

60.5
59.5
56.9

59.6
40.2
51.2
60.5
61.1

89.3
89.5
89.6

88.9
80.1
81.5
87.0
89.7

87.6
86.9
87.3

86.6
67.3
78.2
83.5
86.0

70.1 72.1
71.5 71.8
68.6 71.8

71.2 71.7
58.6 63.3
61.5 68.6
68.1 70.6
70.6 71.1

Avg

80.9
80.4
79.8

80.6
68.2
75.5
79.4
80.6

Table 1: GLUE benchmark test server results with BERTLARGE models. (Top) Results with adapter bottleneck
layers (brackets indicate the size of bottlenecks), taken from from Houlsby et al. (2019). (Bottom) Results
from this work. ∗QNLI results are not directly comparable across the two works as the GLUE benchmark has
updated the test set since then. To make our results comparable the average column is calculated without QNLI.

For all experiments, we use the BERTLARGE model from Devlin et al. (2019), which has 24 layers,
1024 hidden size, 16 attention heads, and 340M parameters. We use the Huggingface Transformer
library (Wolf et al., 2019) to conduct our experiments.

4.2 BASELINES

We compare both structured and non-structured variants of diff pruning against the following base-
lines: Full ﬁnetuning, which fully ﬁnetunes BERTLARGE as usual; Last layer ﬁnetuning, which
only ﬁnetunes the penultimate layer (along with the ﬁnal output layer)9; Adapters from Houlsby
et al. (2019), which train task-speciﬁc bottleneck layers between between each layer of a pretrained
model, where parameter-efﬁciency can be controlled by varying the size of the bottleneck layers;
and Non-adaptive diff pruning, which performs diff pruning just based on magnitude pruning (i.e.,
we obtain θτ through usual ﬁnetuning, set δτ = θτ − θpretrained, and then apply magnitude pruning
followed by additional ﬁnetuning on δτ ). For diff pruning we set our target sparsity rate to 0.5% and
investigate the effect of different target sparsity rates in section 5.1.

IMPLEMENTATION DETAILS AND HYPERPARAMETERS

4.3
Diff pruning introduces additional hyperparameters l, r (for stretching the Hard-Concrete distribu-
tion) and λ (for weighting the approximate L0-norm penalty). We found l = −1.5, r = 1.5, λ =
1.25 × 10−7 to work well across all tasks. We also initialize the weight vector wτ to 0, and ατ to a
positive vector (we use 5) to encourage zτ to be close to 1 at the start of training. While we mainly
experiment with BERTLARGE to compare against prior work with adapters (Houlsby et al., 2019),
in preliminary experiments we found these hyperparameters to work for ﬁnetuning RoBERTa (Liu
et al., 2019c) and XLNet (Yang et al., 2019) models as well.
For all tasks we use a learning rate of 1 × 10−5 and perform a hyperparameter search over batch size
∈ {4, 6, 8, 10} and the number of epochs ∈ {2, 3, 4, 5}.10 However we found the default settings
used for regular ﬁnetuning as suggested in the original BERT paper to work well for most tasks.
Finetuning with the ﬁxed mask after projecting onto the L0-ball with magnitude pruning is done
with a learning rate of 5 × 10−5 for 3 or 5 epochs (3 epochs for QNLI, SST-2, MNLI-m, MNLI-mm,
CoLA, QQP, 5 epochs for MRPC, STS-B, RTE). Grouping for the structured version of diff pruning
is based on the matrix/bias vectors (i.e. parameters that belong to the same matrix or bias vector are
assumed to be in the same group), which results in 393 groups.11

5 RESULTS AND ANALYSIS

Our main results on the GLUE benchmark are shown in Table 1. Structured diff pruning can match
the performance of a fully ﬁnetuned BERTLARGE model while only requiring 0.5% additional pa-

9Wu et al. (2020) observe that ﬁnetuning later layers generally performs better than ﬁnetuning earlier layers
10For the larger QNLI, SST-2, MNLI-m, MNLI-mm, CoLA, QQP datasets, we use batch size of 8 over 3

epochs. For the smaller MRPC, STS-B, RTE datasets, we use batch size of 8 over 3 epochs.

11This deﬁnition of groups is implementation-speciﬁc since it depends on how one concatenates the
input vector before each afﬁne layer. Our grouping is based on Huggingface’s BERT implementation at
commit 656e1386a296d696327a9db37de2ccccc79e2cc7 (available at https://github.com/
huggingface/transformers/blob/656e1386a296d696327a9db37de2ccccc79e2cc7/
src/transformers/modeling_bert.py). In preliminary experiments we found this simple deﬁnition
to work well compared to alternative group deﬁnitions (e.g. based on individual neurons).

5

Under review as a conference paper at ICLR 2021

Pruned Diff Groups

Non-structured

Structured

#

24
25
28

MRPC
STS-B
RTE

Avg

25.7

%

6.1
6.4
7.1

6.5

#

52
48
50

50.0

%

13.2
12.2
12.7

12.7

Figure 1: (Left) Average performance on the GLUE validation set across different target sparsity rates for
the different methods. (Right) Number of groups where all of the parameters in the group are fully zero for
structured vs. non-structured diff pruning at 0.5% target sparsity. We group based on each matrix/bias vector,
resulting in 393 groups in total.

Diff vector
target sparsity

0.10%
0.25%
0.50%
1.00%

100%

QNLI

SST-2 MNLIm MNLImm CoLA MRPC

STS-B RTE QQP

92.7
93.2
93.4
93.3

93.5

93.3
94.2
94.2
94.2

94.1

85.6
86.2
86.4
86.4

86.5

85.9
86.5
86.9
87.0

87.1

58.0
63.3
63.5
66.3

62.8

87.4
90.9
91.3
91.4

91.9

86.3
88.4
89.5
89.9

89.8

68.6
71.5
71.5
71.1

71.8

85.2
86.1
86.6
86.6

87.6

Avg

82.5
84.5
84.8
85.1

85.0

Table 2: Structured diff pruning results on the validation set with different target sparsity rates. Average
performance includes all 9 tasks.

rameters per task. Diff pruning without structured sparsity also performs well, though slightly worse
than the structured approach. Non-adaptive diff pruning, which magnitude prunes the diff vector
without learning the binary mask zτ , performs signiﬁcantly worse, indicating the importance of
learning the masking vector. Compared to adapters, diff pruning obtains similar performance while
requiring fewer parameters per task, making it a potential alternative for parameter-efﬁcient transfer
learning.12 We now perform a series of analysis experiments on the validation set.

5.1 VARYING THE TARGET SPARSITY

In Figure 1 (left), we plot results on the GLUE validation set averaged across all tasks at target
sparsity rates of 0.1%, 0.25%, 0.5%, 1.0% for the different baselines. Structured diff pruning con-
sistently outperforms non-structured and and non-adaptive variants across different sparsity rates.
The advantage of adaptive methods becomes more pronounced at extreme sparsity rates. In Table 2,
we report the breakdown of accuracy of structured diff pruning across different tasks and sparsity
rates, where we observe that different tasks have different sensitivity to target sparsity rates. This
suggests that we can obtain even greater parameter-efﬁciency through targeting task-speciﬁc sparsity
rates in the diff vector.

5.2 STRUCTURED VS. NON-STRUCTURED DIFF PRUNING

Structured diff pruning introduces an additional mask per group, which encourages pruning of entire
groups. This is less restrictive than traditional group sparsity techniques that have been used with
L0-norm relaxations which force all parameters in a group to share the same mask (Louizos et al.,
2018; Wang et al., 2019b). However we still expect entire groups to be pruned out more often in
the structured case, which might bias the learning process towards either eliminating completely or
clustering together nonzero diffs. In Figure 1 (right), we indeed ﬁnd that structured diff pruning
leads to ﬁnetuned models that are much more likely to leave entire groups unchanged from their
pretrained values (zero diffs).

5.3 TASK-SPECIFIC SPARSITY

Different layers of pretrained models have argued to encode different information (Liu et al., 2019a;
Tenney et al., 2019). Given that each task will likely recruit different kinds of language phenomena
embedded in the hidden layers, we hypothesize that diff pruning will modify different parts of the

12However diff pruning incurs additional storage cost due to storing the nonzero positions of the diff vector.

6

Under review as a conference paper at ICLR 2021

Figure 2: Percentage of modiﬁed parameters attributable to each layer for different tasks at 0.5% target sparsity.
The layers are ordered from earlier to later (i.e. the embedding layer is shown at the top). The x-axis for each
plot goes from 0% to 20%.

Sparsity
Performance

QNLI

SST-2 MNLIm MNLImm CoLA MRPC STS-B RTE QQP
3.3% 0.7% 0.6%
1.6%
86.5
71.8
89.7
63.1

0.8%
86.2

0.8%
86.8

2.4%
91.9

1.5% 0.6%
94.0
93.8

With 0.5% sparsity

93.4

94.2

86.4

86.9

63.5

91.3

89.5

71.5

86.6

Avg

1.4%
84.9

84.8

Table 3: (Top) Sparsity and performance before magnitude pruning on the validation set with structured diff
pruning. (Bottom) Performance with 0.5% target sparsity.

pretrained model through task-speciﬁc ﬁnetuning. Figure 2 shows the percentage of nonzero diff
parameters attributable to the different layers for each task. We ﬁnd that different tasks indeed
modify different parts of the network, although there are some qualitative similarities between some
tasks, for example between QNLI & QQP (both must encode questions), and MRPC & STS-B (both
must predict similarity between sentences). The embedding layer is very sparsely modiﬁed for all
tasks. While some of the variations in the sparsity distributions is due to simple randomness, we do
observe some level of consistency over multiple runs of the same task, as shown in Figure 3 of the
appendix.

The ability to modify different parts of the pretrained model for each task could explain the improved
parameter-efﬁciency of our approach compared to Houlsby et al. (2019)’s adapter layers, which can
only read/write to the pretrained model at certain points of the computational graph.13 This poten-
tially suggests that adapter layers with more ﬁne-grained access into model internals (e.g. adapters
for key/value/query transformations) might result in even greater parameter-efﬁciency. While left as
future work, we also note that diff pruning can be applied in conjunction with adapters, which might
further improve results.

5.4 EFFECT OF L0-BALL PROJECTION VIA MAGNITUDE PRUNING
Applying magnitude pruning to project onto the L0-ball was crucial in achieving exact sparsity
targets. As shown in Table 3, we observed little loss in performance through magnitude pruning. We
re-iterate that it was crucial to ﬁnetune with the ﬁxed mask in order to maintain good performance.14

5.5 SQUAD EXTRACTIVE QUESTION ANSWERING

To demonstrate the effectiveness of our approach beyond classiﬁcation, we additionally experiment
on the extractive question answering task SQuAD, which asks model to select the answer span to a
question given a Wikipedia paragraph. To make direct comparisons with Houlsby et al. (2019), we
run all experiments on SQuAD v1.1. For diff pruning, we use the same general hyper-parameters as
our full ﬁnetuning baseline.15 Results are shown in Table 4. Diff pruning is able achieve comparable
or better performance with only 1% additional parameters. Notably, we see that our method can
improve the F1 score of full ﬁnetuning baseline by a signiﬁcant margin (e.g. 90.8% ⇒ 93.2%)

13To simulate this restricted setting, we tried applying diff pruning only on the dense transformations just

before the output of each layer (i.e. after self-attention layers), and observed much worse performance.

14Even for the approach that does not apply magnitude pruning, we found it helpful to ﬁx the mask zτ after

an initial training phase and ﬁnetune just wτ .

15https://huggingface.co/transformers/v2.5.1/examples.html

7

Under review as a conference paper at ICLR 2021

Full ﬁnetuning
Adapters

Full ﬁnetuning
Diff pruning
Diff pruning (struct.)

Sparsity

F1

100%
2%

100%
1%
1%

90.7%
90.4%

90.8%
92.1%
93.2%

Table 4: SQuAD validation results with BERTLARGE model.

while modifying many fewer parameters (e.g., 100% ⇒ 1%), which potentially implies that diff
pruning can have a useful regularization effect.

6 DISCUSSION

6.1 MEMORY REQUIREMENTS

For training, our approach requires more memory than usual ﬁnetuning due to additionally opti-
mizing ατ and wτ . This did not present a signiﬁcant challenge for pretrained models that we
experimented with in this study, since majority of GPU memory was utilized by the minibatch’s
activation layers. However, this could present an issue as model sizes get larger and larger. While
training efﬁciency was not a primary concern of this work, diff pruning takes approxiamtely 1.5×
to 2× more time per batch, which results in slower training.

After training, storing the task-speciﬁc diff vector requires storing a compressed version with both
the nonzero positions and weights, which incurs additional storage requirements.

6.2

INFORMATION-EFFICIENT TRANSFER LEARNING

Efﬁciently representing pretrained models adapted to new tasks is becoming an increasingly impor-
tant problem in contemporary NLP. This paper focuses on a rather narrow deﬁnition of efﬁciency—
parameter-efﬁciency. An interesting direction might be to target generalizations of parameter-
efﬁciency, for example, information-efﬁciency, which aims to minimize the number of bits required
to represent the task-speciﬁc model when given the pretrained model for free. This view can suggest
other avenues for achieving information-efﬁcient transfer learning: for example, “what is the min-
imum number of (potentially synthetic) datapoints that we can ﬁnetune BERT on to obtain a good
task-speciﬁc model?”,16 or “what is the shortest preﬁx string that we can condition GPT3 on for it
to become a good task-speciﬁc model”?

7 RELATED WORK

Multi-task learning Multi-task learning (Caruana, 1997), broadly construed, aims to learn models
and representations that can be utilized across a diverse range of tasks, and offers a natural approach
to training parameter-efﬁcient deep models. Several works have shown that a single BERT model
can obtain good performance across multiple tasks when jointly trained (Liu et al., 2019b; Clark
et al., 2019; Stickland & Murray, 2019). Adapter layers, which are task-speciﬁc layers that read and
write to layers of a shared model (Rebufﬁ et al., 2018), offer an alternative approach to multi-task
learning that does not require access to all tasks during training, and have also been applied to obtain
parameter-efﬁcient BERT models (Houlsby et al., 2019; Pfeiffer et al., 2020a;b;c). A related line of
work targets extreme parameter-efﬁciency through task-agnostic sentence representations that can be
used without ﬁnetuning for downstream tasks (Le & Mikolov, 2014; Kiros et al., 2015; Wieting et al.,
2016; Hill et al., 2016; Arora et al., 2017; Conneau et al., 2017; Cer et al., 2018; Zhang et al., 2018;
Subramanian et al., 2018; Zhang et al., 2020). Reimers & Gurevych (2019), building on the earlier
work of Conneau et al. (2017), show that BERT ﬁnetuned on natural language inference obtains
sentence representations that perform well across multiple sentence-level tasks. These feature-based
transfer learning methods are however generally outperformed by fully ﬁnetuned models (Howard
& Ruder, 2018).

16Dataset distillation (Wang et al., 2018) tackles this question in the context of vision models.

8

Under review as a conference paper at ICLR 2021

Model compression There has been much recent work on compressing pretrained trained with
self-supervision (see Ganesh et al. (2020) for a recent survey). A particularly promising line of
work focuses on obtaining smaller pretrained models (for subsequent ﬁnetuning) through weight
pruning (Gordon et al., 2020; Sajjad et al., 2020; Chen et al., 2020) and/or knowledge distillation
(Sanh et al., 2019; Sun et al., 2019; Turc et al., 2019; Jiao et al., 2019; Sun et al., 2020). It would be
interesting to see whether our approach can be applied on top of these smaller pretrained models to
for even greater parameter-efﬁciency.

Learning to prune Our work is closely related to the line of work on learning to prune pretrained
models with differentiable relaxations of binary masks (Wang et al., 2019b; Zhao et al., 2020; Sanh
et al., 2020; Radiya-Dixit & Wang, 2020). While these works also enable parameter-efﬁcient transfer
learning, they generally apply the masks directly on the pretrained parameters instead of on the
difference vector as in the present work.

Regularization towards pretrained models Finally, diff pruning is also related to works which
regularize the learning process towards pretrained models for continual learning (Kirkpatrick et al.,
2017; Schwarz et al., 2018), domain adaptation (Wiese et al., 2017; Miceli Barone et al., 2017),
and stable ﬁnetuning (Lee et al., 2020). These works typically do not utilize sparse regularizers and
target a different goal than parameter-efﬁciency.

8 CONCLUSION

We propose diff pruning as a simple approach for parameter-efﬁcient transfer learning with pre-
trained models. Experiments on standard NLP benchmarks and models show that diff pruning can
match the performance of fully ﬁnetuned baselines while requiring only a few additional parameters
per task. We also propose a structured variant of diff pruning which provides further improvements.
Future work will consider (i) applying this approach to other architectures (e.g. ConvNets for vi-
sion applications), (ii) injecting parameter-efﬁciency objectives directly into the pretraining process
(to pretrain models that are better suited towards sparse transfer learning), and (iii) combining diff
pruning with other techniques (e.g. adapters) to achieve even greater parameter-efﬁciency.

REFERENCES

Sanjeev Arora, Yingyu Liang, and Tengyu Ma. A Simple but Tough-to-Beat Baseline for Sentence

Embeddings . In Proceedings of ICLR, 2017.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhari-
wal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal,
Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M.
Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin,
Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford,
Ilya Sutskever, and Dario Amodei. Language Models are Few-Shot Learners. 2020.

Rich Caruana. Multitask Learning. Machine Learning, 1997.

Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Con-
stant, Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, Brian Strope, and Ray Kurzweil. Uni-
versal sentence encoder for English. In Proceedings of EMNLP: System Demonstrations, 2018.

Tianlong Chen, Jonathan Frankle, Shiyu Chang, Sijia Liu, Yang Zhang, Zhangyang Wang,
The Lottery Ticket Hypothesis for Pre-trained BERT Networks.

and Michael Carbin.
arXiv:2007.12223, 2020.

Kevin Clark, Minh-Thang Luong, Urvashi Khandelwal, Christopher D. Manning, and Quoc V. Le.
BAM! Born-Again Multi-Task Networks for Natural Language Understanding. In Proceedings
of ACL, 2019.

Alexis Conneau, Douwe Kiela, Holger Schwenk, Loic Barrault, and Antoine Bordes. Supervised
Learning of Universal Sentence Representations from Natural Language Inference Data. In Pro-
ceedings of EMNLP, 2017.

9

Under review as a conference paper at ICLR 2021

Andrew Dai and Quoc V. Le. Semi-Supervised Sequence Learning. In Proceedings of NIPS, 2015.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep

Bidirectional Transformers for Language Understanding. In Proceedings of NAACL, 2019.

Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. Language-

agnostic BERT Sentence Embedding. arXiv:2007.01852, 2020.

Robert French. Catastrophic forgetting in connectionist networks. Trends in cognitive sciences, 3,

1999.

Prakhar Ganesh, Yao Chen, Xin Lou, Mohammad Ali Khan, Yin Yang, Deming Chen, Marianne
Winslett, Hassan Sajjad, and Preslav Nakov. Compressing Large-Scale Transformer-Based Mod-
els: A Case Study on BERT. arXiv:2002.11985, 2020.

Mitchell A. Gordon, Kevin Duh, and Nicholas Andrews. Compressing BERT: Studying the Effects
of Weight Pruning on Transfer Learning. In Proceedings of Rep4NLP 2020 Workshop at ACL
2020, 2020.

Song Han, Huizi Mao, and William J. Dally. Deep Compression: Compressing Deep Neural Net-
works with Pruning, Trained Quantization and Huffman Coding. In Proceedings of ICLR, 2016.

Felix Hill, Kyunghyun Cho, and Anna Korhonen. Learning distributed representations of sentences

from unlabelled data. In Proceedings of ACL, 2016.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, An-
drea Gesmundo, and Mona Attariyanand Sylvain Gelly. Parameter-efﬁcient transfer learning for
nlp. In Proceedings of ICML, 2019.

Jeremy Howard and Sebastian Ruder. Universal Language Model Fine-tuning for Text Classiﬁca-

tion. In Proceedings of ACL, 2018.

Eric Jang, Shixiang Gu, and Ben Poole. Categorical Reparameterization with Gumbel-Softmax. In

Proceedings of ICLR, 2017.

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu.
TinyBERT: Distilling BERT for Natural Language Understanding. arXiv:1909.10351, 2019.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A.
Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hass-
abis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting
in neural networks. Proceedings of the National Academy of Sciences, 14:3521–3526, 2017.

Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urta-

sun, and Sanja Fidler. Skip-Thought Vectors. In Proceedings of NeurIPS, 2015.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Sori-
cut. ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.
In
Proceedings of ICLR, 2020.

Quoc V. Le and Tomas Mikolov. Distributed Representations of Sentences and Documents.

In

Proceedings of ICML, 2014.

Cheolhyoung Lee, Kyunghyun Cho, and Wanmo Kang. Mixout: Effective Regularization to Fine-

tune Large-scale Pretrained Language Models. In Proceedings of ICLR, 2020.

Sang-Woo Lee, Jin-Hwa Kim, Jaehyun Jun, Jung-Woo Ha, and Byoung-Tak Zhang. Overcoming
In Advances in Neural Information

catastrophic forgetting by incremental moment matching.
Processing Systems. 2017.

Nelson F. Liu, Matt Gardner, Yonatan Belinkov, Matthew E. Peters, and Noah A. Smith. Linguistic
Knowledge and Transferability of Contextual Representations. In Proceedings of ACL, 2019a.

Xiaodong Liu, Pengcheng He, Weizhu Chen, and Jianfeng Gao. Multi-Task Deep Neural Networks

for Natural Language Understanding. In Proceedings of ACL, 2019b.

10

Under review as a conference paper at ICLR 2021

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike
Lewis, Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A Robustly Optimized BERT Pre-
training Approach. arXiv:1907.11692, 2019c.

David Lopez-Paz and Marc’Aurelio Ranzato. Gradient Episodic Memory for Continual Learning.

In Proceedings of NeurIPS, 2017.

Christos Louizos, Max Welling, Diederik P, and Kingma. Learning Sparse Neural Networks through

L0 Regularization. In Proceedings of ICLR, 2018.

Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The Concrete Distribution: A Continuous

Relaxation of Discrete Random Variables. In Proceedings of ICLR, 2017.

Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. Learned in translation:

Contextualized word vectors. In Proceedings of NeurIPS. 2017.

Antonio Valerio Miceli Barone, Barry Haddow, Ulrich Germann, and Rico Sennrich. Regularization

techniques for ﬁne-tuning in neural machine translation. In Proceedings of EMNLP, 2017.

Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efﬁcient Estimation of Word Repre-

sentations in Vector Space. arXiv:1301.3781, 2013.

German I. Parisi, Ronald Kemker, Jose L. Part, Christopher Kanan, and Stefan Wermter. Continual

Lifelong Learning with Neural Networks: A Review. arXiv:1802.07569, 2018.

Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and
Luke Zettlemoyer. Deep Contextualized Word Representations. In Proceedings of NAACL, 2018.

Jonas Pfeiffer, Aishwarya Kamath, Andreas Ruckle, and Kyunghyun Cho amd Iryna Gurevych.
AdapterFusion: Non-Destructive Task Composition for Transfer Learning. arXiv:2005.00247,
2020a.

Jonas Pfeiffer, Andreas Ruckle, Clifton Poth, Aishwarya Kamath, Ivan Vulic, Sebastian Ruder,
and Iryna Gurevych Kyunghyun Cho. AdapterHub: A Framework for Adapting Transformers.
arXiv:2007.07779, 2020b.

Jonas Pfeiffer, Ivan Vulic, Iryna Gurevych, and Sebastian Ruder. MAD-X: An Adapter-based Frame-

work for Multi-task Cross-lingual Transfer. arXiv:2005.00052, 2020c.

Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language under-

standing by generative pre-training. 2018.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language

Models are Unsupervised Multitask Learners. 2019.

Evani Radiya-Dixit and Xin Wang. How ﬁne can ﬁne-tuning be? Learning efﬁcient language mod-

els. In Proceedings of AISTATS, 2020.

Colin Raffel, Noam Shazeer, Katherine Lee Adam Roberts, Sharan Narang, Michael Matena, Yanqi
Zhou, Wei Li, and Peter J. Liu. Exploring the Limits of Transfer Learning with a Uniﬁed Text-to-
Text Transformer. Journal of Machine Learning Research, 21, 2020.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. ZeRO: Memory Optimiza-

tions Toward Training Trillion Parameter Models. arXiv:1910.02054, 2019.

S. Rebufﬁ, A. Vedaldi, and H. Bilen. Efﬁcient Parametrization of Multi-domain Deep Neural Net-

works. In Proceedings of CVPR, 2018.

Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-

Networks. In Proceedings of EMNLP, 2019.

Hassan Sajjad, Fahim Dalvi, Nadir Durrani, and Preslav Nakov. Poor Man’s BERT: Smaller and

Faster Transformer Models. arXiv:2004.03844, 2020.

11

Under review as a conference paper at ICLR 2021

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. DistilBERT, a distilled version
of BERT: smaller, faster, cheaper and lighter. In Proceedings of 5th Workshop on Energy Efﬁcient
Machine Learning and Cognitive Computing, 2019.

Victor Sanh, Thomas Wolf, and Alexander M. Rush. Movement Pruning: Adaptive Sparsity by

Fine-Tuning. arXiv:2005.07683, 2020.

Timo Schick and Hinrich Schutze. It’s Not Just Size That Matters: Small Language Models Are

Also Few-Shot Learners. arXiv:2009.07118, 2020.

Jonathan Schwarz, Jelena Luketina, Wojciech M. Czarnecki, Agnieszka Grabska-Barwinska,
Yee Whye Teh, Razvan Pascanu, and Raia Hadsell. Progress & Compress: A scalable frame-
work for continual learning. In Proceedings of ICML, 2018.

Hanul Shin, Jung Kwon Lee, Jaehong Kim, and Jiwon Kim. Continual Learning with Deep Gener-

ative Replay. In Proceedings of NeurIPS. 2017.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan
Catanzaro. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Par-
allelism. arXiv:1909.08053, 2019.

Asa Cooper Stickland and Iain Murray. BERT and PALs: Projected attention layers for efﬁcient

adaptation in multi-task learning. In Proceedings of ICML, 2019.

Sandeep Subramanian, Adam Trischler, Yoshua Bengio, and Christopher J. Pal. Learning General
Purpose Distributed Sentence Representations via Large Scale Multi-task Learning. In Proceed-
ings of ICLR, 2018.

Siqi Sun, Yu Cheng, Zhe Gan, and Jingjing Liu. Patient Knowledge Distillation for BERT Model

Compression. In Proceedings of EMNLP, 2019.

Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, and Denny Zhou. Mobile-
BERT: a compact task-agnostic BERT for resource-limited devices. In Proceedings of ACL, July
2020.

Ian Tenney, Dipanjan Das, and Ellie Pavlick. BERT Rediscovers the Classical NLP Pipeline. In

Proceedings of ACL, 2019.

Iulia Turc, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Well-Read Students Learn Better:

On the Importance of Pre-training Compact Models. arXiv:1908.08962, 2019.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. GLUE:
A multi-task benchmark and analysis platform for natural language understanding. In Proceedings
of ICLR, 2019a.

Tongzhou Wang, Jun-Yan Zhu, Antonio Torralba, and Alexei A. Efros. Dataset Distillation.

arXiv:1811.10959, 2018.

Ziheng Wang, Jeremy Wohlwend, and Tao Lei. Structured Pruning of Large Language Models.

arXiv:1910.04732, 2019b.

Georg Wiese, Dirk Weissenborn, and Mariana Neves. Neural domain adaptation for biomedical

question answering. In Proceedings of CoNLL, August 2017.

John Wieting, Mohit Bansal, Kevin Gimpel, and Karen Livescu. Towards Universal Paraphrastic

Sentence Embeddings. In Proceedings of ICLR, 2016.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi,
Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick
von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger,
Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Huggingface’s transformers: State-
of-the-art natural language processing. ArXiv, abs/1910.03771, 2019.

John M. Wu, Yonatan Belinkov, Hassan Sajjad, Nadir Durrani, Fahim Dalvi, and James Glass.
Similarity Analysis of Contextual Word Representation Models. In Proceedings of ACL, 2020.

12

Under review as a conference paper at ICLR 2021

Figure 3: Percentage of modiﬁed parameters attributable to each layer for 5 different runs of SST-2 at 0.5%
target sparsity. The layers are ordered from earlier to later (i.e. the embedding layer is shown at the top). The
x-axis for each plot goes from 0% to 20%.

Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le.
XLNet: Generalized Autoregressive Pretraining for Language Understanding. In Proceedings of
NeurIPS, 2019.

Minghua Zhang, Yunfang Wu, Weikang Li, and Wei Li. Learning universal sentence representations

with mean-max attention autoencoder. In Proceedings of EMNLP, 2018.

Yan Zhang, Ruidan He, Zuozhu Liu, Kwan Hui Lim, and Lidong Bing. An Unsupervised Sentence
Embedding Method byMutual Information Maximization. In Proceedings of EMNLP, 2020.

Mengjie Zhao, Tao Lin, Martin Jaggi, and Hinrich Schutze. Masking as an Efﬁcient Alternative to

Finetuning for Pretrained Language Models. arXiv:2004.12406, 2020.

A APPENDIX

A.1 CONSISTENCY OF NONZERO PARAMETERS

Figure 3 shows the percentage of modiﬁed parameters attributable to each layer across 5 runs of SST-
2. We ﬁnd that there is nonotrivial variation in sparsity across runs, but also a degree of consistency.
For example, the ﬁrst layer is modiﬁed considerably more than other layers across all runs.

13

