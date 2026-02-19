Published as a conference paper at ICLR 2023

WEIGHTED ENSEMBLE SELF-SUPERVISED LEARNING

Yangjun Ruan∗ †

Saurabh Singh

Warren Morningstar

Alexander A. Alemi

Sergey Ioffe

Ian Fischer†

Joshua V. Dillon†

Google Research

ABSTRACT

Ensembling has proven to be a powerful technique for boosting model perfor-
mance, uncertainty estimation, and robustness in supervised learning. Advances
in self-supervised learning (SSL) enable leveraging large unlabeled corpora for
state-of-the-art few-shot and supervised learning performance. In this paper, we
explore how ensemble methods can improve recent SSL techniques by developing
a framework that permits data-dependent weighted cross-entropy losses. We re-
frain from ensembling the representation backbone; this choice yields an efﬁcient
ensemble method that incurs a small training cost and requires no architectural
changes or computational overhead to downstream evaluation. The effectiveness of
our method is demonstrated with two state-of-the-art SSL methods, DINO (Caron
et al., 2021) and MSN (Assran et al., 2022). Our method outperforms both in
multiple evaluation metrics on ImageNet-1K, particularly in the few-shot setting.
We explore several weighting schemes and ﬁnd that those which increase the di-
versity of ensemble heads lead to better downstream evaluation results. Thorough
experiments yield improved prior art baselines which our method still surpasses;
e.g., our overall improvement with MSN ViT-B/16 is 3.9 p.p. for 1-shot learning.

↑ 0.9

↑ 1.2

1

INTRODUCTION

80

DINO ViT-B/8
DINO∗-ENT (Ours)

↑ 3.1

70

↑ 4.1

↑ 6.1

%
y
c
a
r
u
c
c
A

The promise of self-supervised learning (SSL)
is to extract information from unlabeled data
and leverage this information in downstream
tasks (He et al., 2020; Caron et al., 2021); e.g.,
semi-supervised learning (Chen et al., 2020a;b),
robust learning (Radford et al., 2021; Ruan et al.,
2022; Lee et al., 2021), few-shot learning (Ass-
ran et al., 2022), and supervised learning (Toma-
sev et al., 2022). These successes have en-
couraged increasingly advanced SSL techniques
(e.g., Grill et al., 2020; Zbontar et al., 2021; He et al., 2022). Perhaps surprisingly however, a simple
and otherwise common idea has received limited consideration: ensembling.

1% KNN Linear
Figure 1: Our improvements to DINO, including
baseline improvements and ensembling.

1-shot 2-shot 5-shot

↑ 7.5

50

60

Ensembling combines predictions from multiple trained models and has proven effective at improving
model accuracy (Hansen & Salamon, 1990; Perrone & Cooper, 1992) and capturing predictive
uncertainty in supervised learning (Lakshminarayanan et al., 2017; Ovadia et al., 2019). Ensembling
in the SSL regime is nuanced, however; since the goal is to learn useful representations from unlabeled
data, it is less obvious where and how to ensemble. We explore these questions in this work.

We develop an efﬁcient ensemble method tailored for SSL that replicates the non-representation parts
(e.g., projection heads) of the SSL model. In contrast with traditional “post-training” ensembling, our
ensembles are only used during training to facilitate the learning of a single representation encoder,
which yields no extra cost in downstream evaluation. We further present a family of weighted cross-
entropy losses to effectively train the ensembles. The key component of our losses is the introduction
of data-dependant importance weights for ensemble members. We empirically compare different
choices from our framework and ﬁnd that the choice of weighting schemes critically impacts ensemble
diversity, and that greater ensemble diversity correlates with improved downstream performance. Our
method is potentially applicable to many SSL methods; we focus on DINO (Caron et al., 2021) and
MSN (Assran et al., 2022) to demonstrate its effectiveness. Fig. 1 shows DINO improvements from
using our ensembling and weighted cross-entropy loss.

∗University of Toronto & Vector Institute. Work done as a student researcher at Google.
†Correspondence to yjruan@cs.toronto.edu, {iansf,jvdillon}@google.com.

1

Published as a conference paper at ICLR 2023

In summary, our core contributions are to:

• Develop a downstream-efﬁcient ensemble method suitable for many SSL techniques (Sec. 3.1).
• Characterize an ensemble loss family of weighted cross-entropy objectives (Sec. 3.2).
• Conduct extensive ablation studies that improve the prior art baselines by up to 6.3 p.p. (Sec. 5.1).
• Further improve those baselines with ensembling (e.g., up to 5.5 p.p. gain for 1-shot) (Table 2).

2 BACKGROUND

In this section, we frame SSL methods from the perspective of maximum likelihood estimation (MLE)
and use this as the notational basis to describe the state-of-the-art clustering-based SSL methods as
well as derive their ensembled variants in Sec. 3.

From Maximum Likelihood to SSL Denote unnormalized KL divergence (Dikmen et al., 2014)
between non-negative integrable functions p, q by K[p(X), q(X)] = H×[p(X), q(X)] − H[p(X)],
where H×[p(X), q(X)] = − (cid:82)
X q(x)dx − 1 is the unnormalized cross-entropy
(with 0 log 0 = 0) and H[p(X)] = H×[p(X), p(X)]. These quantities simplify to their usual
deﬁnitions when p, q are normalized, but critically they enable ﬂexible weighting of distributions for
the derivation of our weighted ensemble losses in Sec. 3.2.

X p(x) log q(x)dx + (cid:82)

Let ν(X, Y ) = ν(X)ν(Y |X) be nature’s distribution of input/target pairs over the space X × Y
and s(Y |θ, X) be a predictive model of target given the input parameterized by θ ∈ T . Supervised
maximum likelihood seeks the minimum expected conditional population risk with respect to θ,

Eν(X) K[ν(Y |X), s(Y |θ, X)] = Eν(X) H×[ν(Y |X), s(Y |θ, X)] − Eν(X) H[ν(Y |X)].

(1)

Henceforth omit Eν(X) H[ν(Y |X)] since it is constant in θ. Since ν(X, Y ) is unknown, a ﬁnite sample
approximation is often employed. Denote a size-n i.i.d. training set by Dn = {xi}i∈[n] ∼ ν⊗n and
x∈Dn,y∼ν(Y |x) δ(X − x)δ(Y − y) where δ : R → {0, 1}
empirical distribution by ˆν(X, Y ) = 1
n
is 1 when x = 0 and 0 otherwise. The sample risk is thus − 1
n

H×[ˆν(Y |x), s(Y |θ, x)].

(cid:80)

(cid:80)

x∈Dn

In SSL, we interpret ν(Y |x) as being the oracle teacher under a presumption of how the rep-
resentations will be evaluated on a downstream task. This assumption is similar to that made
in Arora et al. (2019); Nozawa et al. (2020). We also assume ˆν(Y |X) is inaccessible and/or un-
reliable. Under this view, some SSL techniques substitute ˆν(Y |x) for a weakly learned target or
“teacher”, t(Y |x). We don’t generally expect t(Y |x) to recover ν(Y |x); we only assume that an
optimal teacher exists and it is ν(Y |x). With the teacher providing the targets, the loss becomes
− 1
n

H×[t(Y |x), s(Y |θ, x)].

(cid:80)

x∈Dn

Teacher and student in clustering SSL methods Clustering SSL methods such as SWaV (Caron
et al., 2020), DINO (Caron et al., 2021), and MSN (Assran et al., 2022) employ a student model
characterized by proximity between learned codebook entries and a data-dependent code,

s(Y |θ, x) = softmax

(cid:18)(cid:26) 1
τ

(hψ ◦ rω)(x) · µy
(cid:107)(hψ ◦ rω)(x)(cid:107)2(cid:107)µy(cid:107)2

(cid:27)(cid:19)

: y ∈ [c]

θ = {ω, ψ, {µy}y∈[c]} ∈ T ,

(2)

(3)

where the encoder rω : X → Z produces the representations used for downstream tasks, and the
projection head hψ : Z → Rd and codebook entries {µy}y∈Y ∈ Rd characterize the SSL loss. Eq. (2)
can be viewed as “soft clustering”, where the input is assigned to those centroids that are closer to
the projection head’s output. The projection head and codebook are used during training but thrown
away for evaluation, which is empirically found vital for downstream tasks (Chen et al., 2020a;b).
Hyperparameters τ ∈ R>0, c ∈ Z>0 represent temperature and codebook size. The teacher is deﬁned
as t(Y |x) = s(Y | stopgrad(g(θ)), x) where g : T → T . Commonly g(θ) is an exponential moving
average of gradient descent iterates and the teacher uses a lower temperature than the student.

To capture desirable invariances and prevent degeneracy, data augmentation and regularization (e.g.,
Sinkhorn-Knopp normalization (Caron et al., 2020), mean entropy maximization (Assran et al., 2022))
are essential. As these are not directly relevant to our method, we omit them for brevity.

2

Published as a conference paper at ICLR 2023

3 METHOD

Ensembling is a technique that combines models to boost performance, and has been especially
successful in supervised learning. We are interested in ensembling methods that carry over this
success to SSL approaches. However, SSL has key differences, such as throw-away “projection
heads”, from supervised learning that result in a multitude of possibilities for how to ensemble. With
this in mind, we propose ﬁrst where to ensemble, and then how to ensemble. Those proposals result
in an efﬁcient “peri-training” ensembling technique speciﬁcally tailored for SSL and a family of
weighted ensemble objectives; we subsequently suggest different ways to select the weights.

3.1 WHERE TO ENSEMBLE?

the

ensembles

by
Denote
teacher/student
{ti(Y |x)}i∈[m] and {s(Y |θj, x)}j∈[m] and deﬁne
each as in Sec. 2; parameters θ = {θj}j∈[m] ∈ T m
are independently initialized, all students use one
temperature and all teachers another. We asymmet-
rically denote ti(Y |x) and s(Y |θj, x) to emphasize
that teachers’ gradients are zero and that the students
are distinct solely by way of θi
(cid:54)= θj. Studying
heterogeneous architectures and/or different teacher
parameterizations is left for future work.

Encoders

Heads

Preds.

Losses

x(cid:48)

Teacher rt

x

moving
average

Student rs

x(cid:48)(cid:48)

h1
...
hm

h1
...
hm

...

...

t1

tm

s1

sm

w11

H×
11

...

H×

mm

wmm

+

that θj parameterizes the encoder, pro-
Recall
jection head, and codebook parameters: θj =
(ωj, ψj, {µjy}y∈Y ). We further restrict T m such that
ωi = ωj, i.e., we limit our consideration to ensem-
bles of projection heads hψj and/or codebooks µj
but not encoders rωj . This choice makes our en-
semble method inherently different from traditional
supervised ensembling or encoder rω ensembling:
the ensembled parts are not used for evaluation but
for improving the learning of non-ensembled representation encoder during training, thus it requires
no change of downstream evaluation or computational cost. Ensembling of rω is left for future work.

Figure 2: Overview of (hψ, µ)-ensemble.
Two augmented inputs are encoded by the
teacher/student into representations, and then
processed by an ensemble of heads. The loss
for each head is weighted and summed into
the ﬁnal loss. Strike-through edges indicate
stop-gradients. See Appx. A for pseudocode.

3.2 HOW TO ENSEMBLE?

We would like to extend the loss to support an ensemble of teacher/student pairs while respecting the
MLE intuition of the loss as in Sec. 2. Additionally, we want to facilitate data-dependent importance
weights, thus enabling preferential treatment of some teacher/student pairs. We therefore propose a
weighted average (unnormalized) cross-entropy loss,

Ln(θ) =

1
n

(cid:88)

(cid:88)

H×[wijY (cid:12) ti(Y |x), s(Y |θj, x)]

where wijy = softmax

γ fijy(stopgrad(θ), x) : i, j ∈ [m]

x∈Dn

i,j∈[m]
(cid:16)(cid:110) 1

(4)

(5)

.

(cid:111)(cid:17)

The notation wijY (cid:12) ti(Y |x) denotes a Hadamard product; i.e., the product of event-speciﬁc weights
and probabilities for each y ∈ Y. The hyperparameter γ is the temperature. The function fijy is
deﬁned for brevity and discussed in the following section.

This objective admits generality and ﬂexibility for introducing various weighting schemes, as it
supports potential interactions between all teacher/student pairs and allows the weights to be both
model- and data-dependent. Up to a constant independent of θ, it is an importance weighted average
of (unnormalized) KL divergences between each teacher and each student; i.e., a mixture of MLE-like
objectives. We stop the gradient of wijy to θ in order to keep the overall gradient a weighted average
of students’ log-likelihood gradients, similar to Eq. (1). We also normalize the weights such that each
data point equally contributes to the loss.

3

Published as a conference paper at ICLR 2023

3.3 HOW TO WEIGHT?

In this section, we present several instantiations of our losses with different weighting schemes.
We empirically show in Sec. 5 that the particular choice of weighting scheme is critical for the
representation performance and the induced diversity of (hψ, µ)-ensembles. For simplicity we
assume γ = 1 in this section. We indicate with ⇐⇒ that a loss has the same arg min as Eq. (4).
For additional analysis and discussion, see Appx. D.

Uniform weighting (UNIF) The simplest strategy is to treat different teacher/student pairs inde-
pendently and average each with uniform weighting; i.e.,

fijy = log δ(i − j) ⇐⇒ LUNIF

n

(θ) =

1
n

(cid:88)

x∈Dn

1
m

(cid:88)

i∈[m]

H×[ti(Y |x), s(Y |θi, x)]

(6)

This strategy introduces uniform weights wi = 1
(here and elsewhere) is to sub-select corresponding teacher/student pairs rather than all m2 pairs.

m over ensemble elements. The role of log δ(i − j)

Probability weighting (PROB) An alternative to using the average cross-entropy loss (UNIF) is to
compute the cross-entropy loss of the average predictions whose gradient is weighted by wijy (see
Appx. D.1). At γ = 1, those gradient weights simplify into an average over the student probabilities:

fijy = log s(y|θj, x) ⇐⇒ LPROB

n

(θ) =

1
n

(cid:88)

H×

x∈Dn





1
m

(cid:88)

i∈[m]

ti(Y |x),

1
m

(cid:88)

j∈[m]



s(Y |θj, x)

 (7)

Averaging the predictive distributions introduces correspondence between codes from different heads;
thus different heads are no longer independent but instead cooperate to match the student to the
teachers. The loss favors student heads with more conﬁdent predictions (i.e., larger s(y|θj, x)).
Further motivation for averaging predictions comes from multi-sample losses studied in Morningstar
et al. (2022). Note that the joint convexity of (unnormalized) KL divergence implies that this loss is
upper bounded by the UNIF loss up to some constant in θ (see Appx. D).

Although the PROB strategy favors conﬁdent student predictions, the weights change as a function of
y ∈ Y. This may be in conﬂict with our intuition that SSL is like maximum likelihood (Sec. 2), since
under that view, the teacher is responsible for weighting outcomes.

Entropy weighting (ENT) Another way to favor heads with more conﬁdent predictions is to
directly weight by their predictive entropies; i.e.,

fijy = − H[ti(Y |x)] + log δ(i − j) ⇐⇒

LENT

n (θ) =

1
n

(cid:88)

(cid:88)

x∈Dn

i∈[m]

softmaxi({− 1

γ H[ti(cid:48)(Y |x)] : i(cid:48) ∈ [m]}) H× [ti(Y |x), s(Y |θi, x)]

(8)

(9)

where the weight wi = softmaxi({− 1
γ H[ti(cid:48)(Y |x)] : i(cid:48) ∈ [m]}) is inversely correlated with the
entropy of teacher predictions. In other words, the head whose teacher has a lower entropy (i.e., higher
conﬁdence about its prediction) is given a larger importance weight for learning the representation.
Like PROB, this strategy encourages “data specialists” by emphasizing strongly opinionated teacher
heads for different inputs. Like UNIF, different heads are treated more independent (than PROB),
since interaction between different heads is introduced only through the weight computation. By
preferring low-entropy teachers we also favor low variance teachers; this aligns with the intuition that
using a lower-variance teacher beneﬁts representation quality (Wang et al., 2022).

Countless other weighting schemes
following might also be interesting to study in detail but were omitted due to resource constraints.

It is impossible to fully explore the space of weightings; the

fijy = 0
fijy = log ti(y|x)
fijy = − H[s(Y |θj, x)]
fijy = K[ti(Y |x), s(Y |θj, x)]

(Favors all pairs of teachers/students equally)
(Favors opinionated teachers)
(Favors low-entropy students)
(Favors disagreeing teacher/student pairs)

(10)
(11)
(12)
(13)

4

Published as a conference paper at ICLR 2023

fijy = − 1

2 log(Varti(Y |x)[Y ] + (cid:15))

(Favors low variance teachers; e.g., (cid:15) = 1

12 )

(14)

Note that “aligned” versions of all schemes are possible by using fijy + log δ(i − j). We did early
experiments exploring Eqs. (11) and (12), but the results were inferior and are largely omitted below.

4 RELATED WORK

Self-supervised learning Recent work on self-supervised learning (SSL) focuses on discriminative
or generative approaches. Most discriminative approaches seek to learn augmentation-invariant
representations by enforcing the similarity between augmented pairs of the same image while
utilizing different techniques to avoid collapse. Contrastive methods (Chen et al., 2020a; He et al.,
2020; Wu et al., 2018; Hjelm et al., 2018; Bachman et al., 2019; Tian et al., 2020) use a large number
of negative samples with a noise-contrastive objective (Gutmann & Hyvärinen, 2010; Oord et al.,
2018). A large body of followup work eliminates the necessity of explicit negative samples with
various techniques, including clustering assignment constraints (Caron et al., 2018; 2020; 2021;
Asano et al., 2019), bootstrapping (Grill et al., 2020) or self-distillation (Caron et al., 2021) inspired
by mean teacher (Tarvainen & Valpola, 2017), asymmetric architecture design (Grill et al., 2020;
Chen & He, 2021), or redundancy reduction (Zbontar et al., 2021; Bardes et al., 2021). Recent
generative approaches that use masked image modeling as the pretraining task (Dosovitskiy et al.,
2020; Bao et al., 2021; He et al., 2022; Zhou et al., 2022; Xie et al., 2022) have achieved competitive
ﬁnetuning performance. Our method may be applicable to all of the above methods that have some
sort of “projection head”, such as most of the discriminative approaches.

Ensemble methods Ensembling has been extensively studied for improving model performance
(Hansen & Salamon, 1990; Perrone & Cooper, 1992; Dietterich, 2000) and uncertainty estimation
(Lakshminarayanan et al., 2017; Ovadia et al., 2019) in supervised learning and semi-supervised
learning (Laine & Aila, 2016). A major research direction is to train efﬁcient ensembles with partial
parameter sharing (Lee et al., 2015; Wen et al., 2020; Dusenberry et al., 2020; Havasi et al., 2020) or
intermediate checkpointing (Huang et al., 2017; Garipov et al., 2018). Our method also shares the
encoder parameters across ensembles, which is closely related to multi-headed networks (Lee et al.,
2015; Tran et al., 2020). Ensemble methods for SSL are less explored. Some recent work studies
ensembles of supervised models adapted from pretrained SSL models. Gontijo-Lopes et al. (2022)
conduct an empirical study of ensembles adapted from different SSL models and ﬁnd that higher
divergence in SSL methods leads to less correlated errors and better performance. Wortsman et al.
(2022) ensemble multiple ﬁnetuned models adapted from the same SSL model by averaging their
weights, which boosts the performance without any inference cost. Our method differs from them
in that it (1) applies to the SSL training stage to directly improve representation quality, rather than
aggregates multiple models in the post-training/ﬁnetuning stage; (2) introduces little training cost and
no evaluation cost; and (3) is complementary to these post-training/ﬁnetuning ensembling methods.

5 EXPERIMENTS

We carefully study the impact of (hψ, µ)-ensembles and our selected weighted ensemble losses
(UNIF, PROB, and ENT) on smaller DINO models in Sec. 5.1. Using what we learned in those
experiments, in Sec. 5.2 we present new state-of-the-art results on ImageNet-1K on various metrics
for multiple model sizes by ensembling both DINO- and MSN-based models. Finally, we explore
ensemble evaluations in the transfer learning setting in Sec. 5.3. Additional experimental details and
results are in Appx. B and Appx. C, respectively.

Experimental setup We assessed the effectiveness of our method with two SSL methods: DINO
(Caron et al., 2021) and MSN (Assran et al., 2022). In order to ensure that we are comparing against
strong baselines, we consider three different classes of baselines: (1) evaluation numbers reported in
the original works (Caron et al. (2021), Assran et al. (2022), and Zhou et al. (2022) for an additional
baseline iBOT); (2) evaluation of our implementation using the hyperparameters reported in the
original works (DINO only, for space reasons) to validate our implementation; and (3) evaluation of
our implementation using the best hyperparameters that we found by tuning the baselines (DINO and
MSN) for fair comparisons. In almost all models and evaluations, our retuned baselines give non-
trivial performance improvements on top of previously reported numbers. These type (3) baselines

5

Published as a conference paper at ICLR 2023

Table 1: Comparison of different ensemble strategies. ENT and PROB signiﬁcantly improve over
the non-ensembled baseline, while UNIF leads to no gains. Ensembling both the projection head and
the codebook works the best. All models are DINO∗ ViT-S/16 trained for 300 epochs. Averages and
standard deviations are over 3 initialization seeds. The linear evaluation results on ImageNet-1K with
different amounts of labeled data are reported here (see Table 11 in Appx. C.3 for all metrics).

How

Base

UNIF

PROB
PROB

ENT-ST

ENT
ENT
ENT

Where

# of Labels Per Class

Proj. hψ

Code. µ

1

5

∼13 (1%)

Full

(cid:88)

(cid:88)
(cid:88)

(cid:88)

(cid:88)
(cid:88)

40.6 ± 0.2

40.4 ± 0.4

57.9 ± 0.3

57.6 ± 0.3

63.4 ± 0.2

63.3 ± 0.3

74.4 ± 0.1

74.5 ± 0.2

39.8 ± 0.5 ↓ 0.9
41.9 ± 0.3 ↑ 1.3

57.4 ± 0.4 ↓ 0.5
59.6 ± 0.4 ↑ 1.7

63.0 ± 0.4 ↓ 0.4
65.1 ± 0.3 ↑ 1.7

74.8 ± 0.1 ↑ 0.4
75.4 ± 0.1 ↑ 1.0

40.0 ± 0.5 ↓ 0.6

57.3 ± 0.5 ↓ 0.6

62.7 ± 0.5 ↓ 0.7

74.0 ± 0.4 ↓ 0.4

40.8 ± 0.4
43.0 ± 0.6 ↑ 2.4
44.0 ± 0.2 ↑ 3.4

58.0 ± 0.4
59.7 ± 0.7 ↑ 1.8
60.5 ± 0.3 ↑ 2.6

63.5 ± 0.4
64.8 ± 0.5 ↑ 1.4
65.5 ± 0.1 ↑ 2.2

74.5 ± 0.3
75.1 ± 0.4 ↑ 0.7
75.3 ± 0.1 ↑ 0.9

(cid:88)

(cid:88)

(cid:88)

(cid:88)

(cid:88)

we label DINO∗ and MSN∗, and we use them as the base models for our experiments with (hψ, µ)-
ensembles and weighted ensemble losses. Appx. B.2.1 describes the details for getting such strong
performance for DINO∗ and MSN∗. In particular, we ﬁnd that the projection head has a crucial
impact on label efﬁciency of representations and using a smaller head (3-layer MLP with hidden size
1024) signiﬁcantly improves few-shot evaluation performance (see Appx. C.2).

Evaluation metrics We compared models trained with and without our (hψ, µ)-ensembles by mea-
suring various evaluation metrics on ImageNet-1K (Deng et al., 2009). The evaluation metrics reﬂect
the decodability and the label efﬁciency of learned representations. We measured the decodability
with respect to both the linear classiﬁer following the common linear evaluation protocol and the
k-NN classiﬁer following Caron et al. (2021). We measured the label efﬁciency by evaluating the
linear evaluation performance in few-shot settings, including 1% (∼13-shots) labeled data evaluation
(Chen et al., 2020a) and 1-/2-/5-shot evaluations (Assran et al., 2022). All evaluations used frozen
representations of the teacher encoder – we did not ﬁne tune the models. See Appx. B.3 for details.

5.1 EMPIRICAL STUDY OF (hψ, µ)-ENSEMBLES

Table 1 compares different strategies for where and how to ensemble. Fig. 4 compares the impact of
the weighted ensemble loss on (hψ, µ)-ensemble diversity. Fig. 3 shows the effect of increasing
the number of ensembles, adjusting the temperature γ, and increasing baseline projection head
parameters. In these experiments, we used DINO∗ with ViT-S/16 trained for 300 epochs as the base
model. We compared different ensemble methods applied to the base model with m = 16 heads
which we found to work the best. For the ENT strategy in Table 1, the entropy weighting temperature
γ is set to 0.05 × log(c) by default which is selected from {0.0125, 0.025, 0.05, 0.1, 0.2} × log(c),
where the scale log(c) gives the maximum entropy of the codebook size c. For PROB, we keep γ = 1.

Where to ensemble We study the where question by ensembling either the projection head hψ,
the codebook µ, or both with the ENT and the PROB ensemble strategies, as shown in Table 1. We
ﬁnd that ensembling both hψ and µ provides the largest gains for both losses, probably due to the
increased ﬂexibility for learning a diverse ensemble. Interestingly, only ensembling hψ also works
well for the ENT strategy.

How to ensemble We study the how question by considering four different loss variants: UNIF,
PROB, ENT, and the variant of ENT with student entropy weighting. We ﬁnd that when we ensemble
both the projection head hψ and the codebook µ, the ENT ensemble strategy leads to the most
signiﬁcant gains (e.g., 3.4 p.p. gains for 1-shot and 0.9 p.p. gains for full-data). The PROB strategy
also consistently improves the performance with a slightly larger gain (1 p.p.) in full-data evaluation.
In contrast, we see no gains for the UNIF strategy over the baseline. We also study a variant of ENT
that uses the student entropy (i.e., Eq. (12) with the log δ(i − j) term) for the importance weights
(denoted as ENT-ST). ENT-ST performs much worse than ENT and is even worse than the baseline.

6

Published as a conference paper at ICLR 2023

(a) Scaling of (hψ, µ)-ensembles.

(b) Effect of ENT temperature γ.

(c) Comparing different heads.

Figure 3: Empirical study of (hψ, µ)-ensembles.
(a) The gains of (hψ, µ)-ensembles start to
diminish above 16 heads. (b) The temperature for entropy weighting has a larger impact on few-shot
performance. 16 heads are used and γ is scaled by log(c). (c) Our (hψ, µ)-ensembles outperform all
non-ensembled baselines when controlling for number of parameters. A too powerful non-ensembled
projection head signiﬁcantly harms accuracy. 1%-data evaluation is shown. Also see Fig. 5.

We conjecture that this is because the student predictions typically have a larger variance than teacher
predictions (Wang et al., 2022) especially when multi-crop augmentation (Caron et al., 2020; 2021) is
applied to the student. Similar experiments on Eq. (11) and/or γ = 0 variants of PROB also resulted
in inferior performance (see Table 12).

Analysis of (hψ, µ)-ensemble diversity The previous exper-
iments showed that the choice of ensemble weighting strategy
has a large impact on performance. We hypothesize that this
choice substantially impacts the diversity of the codebook en-
sembles. Since the codes in different heads may not be aligned,
we align them by the similarity of their code assignment prob-
abilities across different input images, which measures how the
codes are effectively used to ‘cluster’ the data. See Appx. C.4
for detailed explanations and results. In Fig. 4, we visualize the
decay patterns of the similarity score between aligned codes
(1.0 means the most similar) in a random pair of heads for each
weighting strategy. ENT decays the fastest and UNIF decays the
slowest, indicating that ENT learns the most diverse codebooks
while UNIF is least diverse. This shows a positive correlation
between the diversity of (hψ, µ)-ensembles and the empirical
performance of the ensemble strategies from Table 1. Finally, for UNIF, we ﬁnd that different heads
tend to learn the same semantic mappings even when randomly initialized; i.e., the code assignments
in different heads become homogeneous up to permutation. See Fig. 8 for a visualization.

Figure 4: Visualization of code
similarity. ENT learns the most di-
verse (hψ, µ)-ensembles reﬂected
by the fastest decay of similarity
scores between aligned codes in dif-
ferent heads. UNIF has low diver-
sity between heads.

Number of (hψ, µ)-ensembles We study the effect of increasing the number of (hψ, µ)-ensembles
m for ENT in Fig. 3a. Having more (hψ, µ)-ensembles boosts the performance until m = 16.
Interestingly, using as few as m = 2 heads already signiﬁcantly improves over the baseline.

Effect of ENT temperature γ Fig. 3b studies the effect of entropy weighting temperature γ
for different evaluation metrics. We observe that γ has a relatively larger impact on few-shot
evaluation performance. γ should be neither too high nor too low: a high temperature leads to
under-specialization (i.e. less diversity) of heads similar to UNIF (γ → ∞) and a low temperature
may otherwise lead to over-specialization (i.e., only a single head is used for each input).

Comparison of different projection heads Our method linearly increases projection head param-
eters, thus a natural question is: Is the gain of (hψ, µ)-ensembles due to the increased power (or
number of parameters) in projection heads? We answer this question with an empirical study of
non-ensembled projection heads. Speciﬁcally, we studied non-ensembled hψ with (depth, width)
searched over {2, 3, 4} × {512, 1024, 2048, 4096} and measured the linear evaluation performance
with different amounts of labeled data. In Fig. 3c, we plot the 1%-data evaluation result with respect
to the number of parameters of the projection head both for ensembled and non-ensembled baselines.
See Appx. C.2 for detailed analysis and extra results for other metrics. Our key ﬁndings are:

7

 Ȱ ʌ Ȱ Ȱ ʌ Ȳ Ȱ ʌ ȴ Ȱ ʌ ȶ Ȱ ʌ ȸ ȱ ʌ Ȱ Ȱ ʌ Ȱ Ȱ ʌ Ȳ Ȱ ʌ ȴ Ȱ ʌ ȶ Ȱ ʌ ȸ ȱ ʌ Ȱ  Î Î ľ Į ´ Î Ř ȷ ȴ ȷ ȵ ȷ ȴ ʌ ȴ ȷ ȴ ʌ ȷ ȷ ȴ ʌ ȹ ȷ ȵ ʌ ȱ ȷ ȵ ʌ ȳ ȷ ȵ ʌ Ȳ ȷ ȵ ʌ ȳ ȶ ȳ ȶ ȴ ȶ ȵ ȶ ȳ ʌ ȴ ȶ ȴ ʌ ȴ ȶ ȴ ʌ ȷ ȶ ȵ ʌ ȳ ȶ ȵ ʌ ȵ ȶ ȵ ʌ ȳ ȶ ȵ ʌ ȴ ȱ Ȳ ȴ ȸ ȱ ȶ ȳ Ȳ ȶ ȴ Z ľ ċ ʌ  Ē ë  ( Č Ĳ Ø ċ Í Ć Ø Ĳ ȴ ȱ ȴ Ȳ ȴ ȳ ȴ ȴ ȴ Ȱ ʌ ȶ ȴ Ȳ ʌ ȱ ȴ ȳ ʌ ȱ ȴ ȳ ʌ ȵ ȴ ȴ ʌ Ȱ ȴ ȳ ʌ ȷ ȴ ȳ ʌ ȹ : ľ Ć Ć  Ô ´ Ĺ ´ ȱ ̑  Ô ´ Ĺ ´ ȱ ʲ Ĳ ñ Ē Ĺ0.00.20.40.60.81.00.00.20.40.60.81.0Accuracy747574.875.175.375.074.8646564.364.865.564.764.20.01250.0250.050.10.2Entropy Weight Temperature424441.442.344.042.842.5Full data1% data1-shotBaseline106107108Number of Head Paramters556065AccuracyDefaultOur baselineEnsembleNon-ensembleUnifProbEnt0.00.51.0Published as a conference paper at ICLR 2023

Table 2: Effectiveness of ensemble heads for DINO∗/MSN∗ with different ViT models. Our
ensemble heads consistently improve all downstream evaluation metrics on ImageNet-1K and achieve
a new state-of-the-art for few-shot evaluations. For ViT-S/16, we report linear evaluation results
probed from the last layer (left) and from the last 4 layers (right, following DINO). †We evaluated the
few-shot settings using DINO’s publicly-available pretrained weights in the cases those results were
not reported in Caron et al. (2021). ‡MSN ViT-B/16 and ViT-B/8 are both trained for 600 epochs in
Assran et al. (2022), whereas our models are trained for only 400, 300 epochs, respectively. For each
architecture, we highlight the best DINO baseline and weighted ensemble in blue . For MSN , the
corresponding highlights are yellow . The best results for each architecture and metric are bolded.

Method

ViT-S/16, 800 epochs

iBOT
DINO
DINO (Repro)
DINO∗ (Retuned)
MSN
MSN∗ (Retuned)
DINO∗-PROB (16)
DINO∗-ENT (4)
DINO∗-ENT (16)
MSN∗-ENT (2)
MSN∗-ENT (8)

ViT-B/16, 400 epochs

iBOT
DINO†
DINO∗ (Retuned)
MSN‡
MSN∗ (Retuned)
DINO∗-ENT (16)
MSN∗-ENT (8)

ViT-B/8, 300 epochs

DINO†
DINO∗ (Retuned)
MSN‡
MSN∗ (Retuned)
DINO∗-ENT (16)
MSN∗-ENT (8)

Few-shot

Full-data

1

2

5

∼13 (1%)

k-NN

Linear

40.4 ± 0.5
38.9 ± 0.4
39.1 ± 0.3
44.6 ± 0.2
47.1 ± 0.1
47.4 ± 0.1

45.2 ± 0.4
46.3 ± 0.1
47.6 ± 0.1 ↑ 3.0
48.8 ± 0.2
50.1 ± 0.1 ↑ 2.7

46.1 ± 0.3
43.0 ± 0.2
49.3 ± 0.1
49.8 ± 0.2
50.7 ± 0.1

52.8 ± 0.1 ↑ 3.5
53.7 ± 0.2 ↑ 3.0

50.8 ± 0.8
48.9 ± 0.3
49.1 ± 0.5
53.6 ± 0.3
55.8 ± 0.6
56.3 ± 0.4

54.9 ± 0.4
55.5 ± 0.6
56.8 ± 0.5
57.5 ± 0.5
58.9 ± 0.6

56.2 ± 0.7
52.7 ± 0.5
58.1 ± 0.5
58.9 ± 0.4
59.2 ± 0.4

61.5 ± 0.4
62.4 ± 0.6

59.9 ± 0.2
58.5 ± 0.1
58.6 ± 0.2
61.1 ± 0.2
62.8 ± 0.3
62.8 ± 0.2

62.5 ± 0.2
63.0 ± 0.3
64.0 ± 0.2
64.0 ± 0.2
65.1 ± 0.3

64.7 ± 0.3
61.8 ± 0.2
65.0 ± 0.3
65.5 ± 0.3
65.9 ± 0.2

67.6 ± 0.3
68.3 ± 0.2

65.9
64.5
64.7
66.2
67.2
67.1

67.3
67.5
68.3 ↑ 2.1
67.9
68.7 ↑ 1.6

69.7
67.4
69.1
-
69.7

71.1 ↑ 2.0
71.5 ↑ 1.8

47.5 ± 0.2
49.5 ± 0.5
55.1 ± 0.1
51.9 ± 0.3

55.0 ± 0.4 ↑ 5.5
55.6 ± 0.2 ↑ 3.7

57.3 ± 0.5
58.6 ± 0.6
64.9 ± 0.7
61.1 ± 0.4

63.4 ± 0.6
64.5 ± 0.5

65.4 ± 0.3
65.9 ± 0.3
71.6 ± 0.3
67.7 ± 0.3

69.5 ± 0.3
70.3 ± 0.2

70.3
70.7
-
71.7

73.4 ↑ 2.7
73.4 ↑ 1.7

75.2
74.5
74.3
74.1
-
73.3

75.1
74.8
75.3
74.6
75.2

77.1
76.1
76.0
-
74.7

77.1
77.2

77.4
77.1
-
75.7

78.6
78.9

-

/ 77.9
76.1 / 77.0
75.8 / 76.9
75.8 / 76.9
/ 76.9
75.6 / 76.6

-

76.5 / 77.6
76.2 / 77.2
76.8 / 77.7 ↑ 0.8
76.0 / 76.9
76.4 / 77.4 ↑ 0.8

79.5
78.2
78.5
-
78.1

79.1 ↑ 0.6
78.9 ↑ 0.8

80.1
80.2
-
80.3

81.0 ↑ 0.8
80.8 ↑ 0.5

• A too powerful non-ensembled hψ signiﬁcantly hurts the label efﬁciency of learned representa-
tions. This result is similar to Chen et al. (2020b), which found that probing from intermediate
layers of projection heads (which can be viewed as using a shallower head) could improve
semi-supervised learning (1%-/10% labeled data) results.

• The default head (3/2048, denoted as ‘Default’) used in recent SSL methods (SimCLRv2, DINO,
MSN, etc.) does not perform as well in few-shot evaluations, probably because it is selected
by looking at full-data evaluation metrics. In contrast, our baseline (3/1024, denoted as ‘Our
baseline’) signiﬁcantly improves few-shot evaluation performance.

• Our (hψ, µ)-ensembles outperform all non-ensembled baselines and lead to consistent improve-

ments in all evaluation metrics, despite the increase of parameters.

5.2

IMPROVING SOTA RESULTS WITH ENSEMBLEING

Next we apply (hψ, µ)-ensembles to DINO∗ and MSN∗ and compare with the state-of-the-art results.
We experimented with model architectures ViT-S/16, ViT-B/16, ViT-B/8 trained for 800, 400, 300
epochs respectively following Caron et al. (2021). We include both the published results and our
retuned versions to ensure strong baselines. For clarity, we denote our method as “{baseline}-
{ensemble strategy} (# of heads)”, e.g., DINO∗-ENT (4). We tuned both baselines and our methods
for all architectures. We report the best hyperparameters for all models in Appx. B.2.2.

8

Published as a conference paper at ICLR 2023

Table 2 compares the results of (hψ, µ)-ensembles and baselines. We ﬁnd that (hψ, µ)-ensembles
with ENT consistently improve all evaluation metrics (full-data, few-shot) across both SSL methods
(DINO∗, MSN∗) and all architectures (ViT-S/16, ViT-B/16, ViT-B/8) over their non-ensembled
counterparts. The gains in few-shot evaluation are particularly substantial, providing a new state-of-
the-art for ImageNet-1K evaluation from ImageNet pretraining.

5.3 MORE EVALUATIONS FOR (hψ, µ)-ENSEMBLES

Table 3: Comparison of transfer performance. ViT-S/16 is used. Our ensemble heads lead to
consistent improvements for MSN∗ and comparable results for DINO∗.

Food101 CIFAR10

CIFAR100

SUN397 Cars DTD Pets

Caltech-101

Flowers

Avg.

DINO∗
DINO∗-ENT (16)
MSN∗
MSN∗-ENT (8)

78.4
79.1

77.7
78.4

93.8
93.8

93.1
93.9

81.0
81.4

79.8
81.1

66.1
66.5

64.6
65.2

66.7
66.8

63.3
68.0

74.6
74.9

72.2
73.2

92.0
92.8

92.4
93.1

94.9
94.6

94.7
95.4

94.4
93.9

92.7
92.8

82.43
82.64

81.17
82.34

In Table 3, we compare the transfer learning performance of (hψ, µ)-ensembles
Transfer learning
and non-ensembled baselines. We used ViT-S-16 models trained on ImageNet-1K for 800 epochs and
evaluated on 9 natural downstream datasets from Chen et al. (2020a) with linear evaluation (details in
Appx. B.3). (hψ, µ)-ensembles lead to consistent improvements in transfer performance for MSN∗
and comparable results for DINO∗.

Training overhead In Table 4, we benchmark the compu-
tational overhead of (hψ, µ)-ensembles at training time. We
used a medium sized model, DINO∗ with ViT-B/16, trained
with the same setting used in all of our experiments. We bench-
marked the wall-clock time and peak memory on 128 TPUv3
cores. (hψ, µ)-ensembling is relatively cheap in terms of train-
ing cost because the ensembled parts typically account for a
small portion of total computation, especially when the back-
bone encoder is more computationally expensive (e.g., ViT-B/8).
Again, we emphasize that there is no evaluation overhead when (hψ, µ)-ensembles are removed.

Table 4: Training overhead. Wall-
clock time and peak memory per
core for training with different num-
bers of ensembles.

5.25G
5.40G
5.89G

5.81h
5.91h
6.34h

m Wall Time

Peak Memory

1
4
16

6 CONCLUSION & DISCUSSION

We introduced an efﬁcient ensemble method for SSL where multiple projection heads are ensembled
to effectively improve representation learning. We showed that with carefully designed ensemble
losses that induce diversity over ensemble heads, our method signiﬁcantly improves recent state-of-
the-art SSL methods in various evaluation metrics, particularly for few-shot evaluation. Although
ensembling is a well-known technique for improving evaluation performance of a single model, we
demonstrated that, for models with throw-away parts such as the projection heads in SSL, ensembling
these parts can improve the learning of the non-ensembled representation encoder and also achieve
signiﬁcant gains in downstream evaluation without introducing extra evaluation cost.

Our ensemble method is applicable to many SSL methods beyond the two we explored. For example,
one may consider generalization to BYOL (Grill et al., 2020) or SimSiam (Chen & He, 2021) that
ensembles projection and/or prediction heads, or MAE (He et al., 2022) that ensembles the decoders
(which introduces more training cost though). Our weighted ensemble losses can also be applied as
long as the original loss can be reformulated as MLE for some t, s, and Y , e.g., the MSE loss in
these methods is MLE under multivariate normal distributions. We hope our results and insights will
motivate more future work for extending our method or exploring more ensemble techniques for SSL.

In future work, we also hope to remove three limitations of our setting. First, considering ensembling
strategies that include the representation encoder, rω, may lead to further improvements in the
performance of weighted ensemble SSL, at the cost of increased computation requirements during
both training and evaluation. Second, considering heterogenous architectures in the ensemble may
further improve the learned representations (e.g., mixing Transformers with ResNets), whether the
heterogeneity is in rω, hψ, or both. Third, considering other possibilities for fijy may also reveal
performance gains and improve our understanding of the critical aspects that lead to good SSL
representations, similar to what we learned about the importance of ensemble diversity.

9

Published as a conference paper at ICLR 2023

ACKNOWLEDGMENTS

We would like to thank Mathilde Caron and Mahmoud Assran for their extensive help in reproducing
DINO and MSN baselines. We would also like to thank Ting Chen and Yann Dubois for their helpful
discussions and encouragements.

REPRODUCIBITLITY STATEMENT

We include detailed derivations for all our proposed losses in Appx. D. We report experimental
details in Appx. B, including the implementation details for reproducing the baselines (Appx. B.1),
training and evaluating our methods (Appx. B.2.1), and all hyper-parameters (Appx. B.2.2) used in
our experiments for reproducing our results in Table 2.

REFERENCES

TensorFlow Datasets, a collection of ready-to-use datasets. https://www.tensorflow.org/

datasets.

Sanjeev Arora, Hrishikesh Khandeparkar, Mikhail Khodak, Orestis Plevrakis, and Nikunj Saun-
shi. A theoretical analysis of contrastive unsupervised representation learning. arXiv preprint
arXiv:1902.09229, 2019.

Yuki Markus Asano, Christian Rupprecht, and Andrea Vedaldi. Self-labelling via simultaneous

clustering and representation learning. arXiv preprint arXiv:1911.05371, 2019.

Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Florian Bordes, Pascal Vincent,
Armand Joulin, Michael Rabbat, and Nicolas Ballas. Masked siamese networks for label-efﬁcient
learning. arXiv preprint arXiv:2204.07141, 2022.

Philip Bachman, R Devon Hjelm, and William Buchwalter. Learning representations by maximizing
mutual information across views. Advances in neural information processing systems, 32, 2019.

Hangbo Bao, Li Dong, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint

arXiv:2106.08254, 2021.

Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization

for self-supervised learning. arXiv preprint arXiv:2105.04906, 2021.

Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101–mining discriminative compo-
nents with random forests. In European conference on computer vision, pp. 446–461. Springer,
2014.

Yuri Burda, Roger B Grosse, and Ruslan Salakhutdinov. Importance weighted autoencoders. In

ICLR, 2016.

Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsuper-
vised learning of visual features. In Proceedings of the European conference on computer vision
(ECCV), pp. 132–149, 2018.

Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin.
Unsupervised learning of visual features by contrasting cluster assignments. Advances in Neural
Information Processing Systems, 33:9912–9924, 2020.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and
Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the
IEEE/CVF International Conference on Computer Vision, pp. 9650–9660, 2021.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for
contrastive learning of visual representations. In International conference on machine learning, pp.
1597–1607. PMLR, 2020a.

Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey E Hinton. Big
self-supervised models are strong semi-supervised learners. Advances in neural information
processing systems, 33:22243–22255, 2020b.

10

Published as a conference paper at ICLR 2023

Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15750–15758, 2021.

Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describ-
ing textures in the wild. In Proceedings of the IEEE conference on computer vision and pattern
recognition, pp. 3606–3613, 2014.

Thomas M Cover and Joy A Thomas. Elements of Information Theory. John Wiley & Sons, 1999.

Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural

information processing systems, 26, 2013.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale
hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition,
pp. 248–255. Ieee, 2009.

Thomas G Dietterich. Ensemble methods in machine learning. In International workshop on multiple

classiﬁer systems, pp. 1–15. Springer, 2000.

Onur Dikmen, Zhirong Yang, and Erkki Oja. Learning the information divergence. IEEE transactions

on pattern analysis and machine intelligence, 37(7):1442–1454, 2014.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image
is worth 16x16 words: Transformers for image recognition at scale. In International Conference
on Learning Representations, 2020.

Sever S Dragomir. A generalization of f -divergence measure to convex functions deﬁned on linear

spaces. Communications in Mathematical Analysis, 15(2):1–14, 2013.

Michael Dusenberry, Ghassen Jerfel, Yeming Wen, Yian Ma, Jasper Snoek, Katherine Heller, Balaji
Lakshminarayanan, and Dustin Tran. Efﬁcient and scalable bayesian neural nets with rank-1
factors. In International conference on machine learning, pp. 2782–2792. PMLR, 2020.

Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training
examples: An incremental bayesian approach tested on 101 object categories. In 2004 conference
on computer vision and pattern recognition workshop, pp. 178–178. IEEE, 2004.

Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry P Vetrov, and Andrew G Wilson.
Loss surfaces, mode connectivity, and fast ensembling of dnns. Advances in neural information
processing systems, 31, 2018.

Raphael Gontijo-Lopes, Yann Dauphin, and Ekin Dogus Cubuk. No one representation to rule
them all: Overlapping features of training methods. In International Conference on Learning
Representations, 2022. URL https://openreview.net/forum?id=BK-4qbGgIE3.

Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena
Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar,
et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural
information processing systems, 33:21271–21284, 2020.

Michael Gutmann and Aapo Hyvärinen. Noise-contrastive estimation: A new estimation principle
for unnormalized statistical models. In Proceedings of the thirteenth international conference on
artiﬁcial intelligence and statistics, pp. 297–304. JMLR Workshop and Conference Proceedings,
2010.

Abner Guzman-Rivera, Dhruv Batra, and Pushmeet Kohli. Multiple choice learning: Learning to
produce multiple structured outputs. Advances in neural information processing systems, 25, 2012.

Lars Kai Hansen and Peter Salamon. Neural network ensembles. IEEE transactions on pattern

analysis and machine intelligence, 12(10):993–1001, 1990.

Marton Havasi, Rodolphe Jenatton, Stanislav Fort, Jeremiah Zhe Liu, Jasper Snoek, Balaji Laksh-
minarayanan, Andrew M Dai, and Dustin Tran. Training independent subnetworks for robust
prediction. arXiv preprint arXiv:2010.06610, 2020.

11

Published as a conference paper at ICLR 2023

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for
unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pp. 9729–9738, 2020.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked
In Proceedings of the IEEE/CVF Conference on

autoencoders are scalable vision learners.
Computer Vision and Pattern Recognition, pp. 16000–16009, 2022.

R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam
Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation
and maximization. arXiv preprint arXiv:1808.06670, 2018.

Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with

stochastic depth. In European conference on computer vision, pp. 646–661. Springer, 2016.

Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu, John E Hopcroft, and Kilian Q Weinberger.

Snapshot ensembles: Train 1, get m for free. arXiv preprint arXiv:1704.00109, 2017.

Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for ﬁne-grained
categorization. In Proceedings of the IEEE international conference on computer vision workshops,
pp. 554–561, 2013.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics

quarterly, 2(1-2):83–97, 1955.

Samuli Laine and Timo Aila. Temporal ensembling for semi-supervised learning. arXiv preprint

arXiv:1610.02242, 2016.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive
uncertainty estimation using deep ensembles. Advances in neural information processing systems,
30, 2017.

Kuang-Huei Lee, Anurag Arnab, Sergio Guadarrama, John Canny, and Ian Fischer. Compressive
visual representations. Advances in Neural Information Processing Systems, 34:19538–19552,
2021.

Stefan Lee, Senthil Purushwalkam, Michael Cogswell, David Crandall, and Dhruv Batra. Why
m heads are better than one: Training a diverse ensemble of deep networks. arXiv preprint
arXiv:1511.06314, 2015.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Confer-

ence on Learning Representations, 2018.

Warren R Morningstar, Alex Alemi, and Joshua V Dillon. Pacm-bayes: Narrowing the empirical risk
gap in the misspeciﬁed bayesian regime. In International Conference on Artiﬁcial Intelligence and
Statistics, pp. 8270–8298. PMLR, 2022.

Wai Ho Mow. A tight upper bound on discrete entropy. IEEE Transactions on Information Theory,

44(2):775–778, 1998.

Yurii Nesterov. A method for solving the convex programming problem with convergence rate

o(1/k2). Proceedings of the USSR Academy of Sciences, 269:543–547, 1983.

Maria-Elena Nilsback and Andrew Zisserman. Automated ﬂower classiﬁcation over a large number
of classes. In 2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing,
pp. 722–729. IEEE, 2008.

Kento Nozawa, Pascal Germain, and Benjamin Guedj. Pac-bayesian contrastive unsupervised repre-
sentation learning. In Jonas Peters and David Sontag (eds.), Proceedings of the 36th Conference
on Uncertainty in Artiﬁcial Intelligence (UAI), volume 124 of Proceedings of Machine Learning
Research, pp. 21–30. PMLR, 03–06 Aug 2020. URL https://proceedings.mlr.press/
v124/nozawa20a.html.

12

Published as a conference paper at ICLR 2023

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive

coding. arXiv preprint arXiv:1807.03748, 2018.

Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin, Joshua
Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model’s uncertainty?
evaluating predictive uncertainty under dataset shift. Advances in neural information processing
systems, 32, 2019.

Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In 2012

IEEE conference on computer vision and pattern recognition, pp. 3498–3505. IEEE, 2012.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Pretten-
hofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and
E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research,
12:2825–2830, 2011.

Michael P Perrone and Leon N Cooper. When networks disagree: Ensemble methods for hybrid
neural networks. Technical report, Brown Univ Providence Ri Inst for Brain and Neural Systems,
1992.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In International Conference on Machine Learning, pp.
8748–8763. PMLR, 2021.

Yangjun Ruan, Yann Dubois, and Chris J. Maddison. Optimal representations for covariate shift. In
International Conference on Learning Representations, 2022. URL https://openreview.
net/forum?id=Rf58LPCwJj0.

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.
Dropout: a simple way to prevent neural networks from overﬁtting. The journal of machine
learning research, 15(1):1929–1958, 2014.

Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. On the importance of initialization
and momentum in deep learning. In International conference on machine learning, pp. 1139–1147.
PMLR, 2013.

Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency
targets improve semi-supervised deep learning results. Advances in neural information processing
systems, 30, 2017.

Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding.

In European

conference on computer vision, pp. 776–794. Springer, 2020.

Nenad Tomasev, Ioana Bica, Brian McWilliams, Lars Buesing, Razvan Pascanu, Charles Blundell,
and Jovana Mitrovic. Pushing the limits of self-supervised resnets: Can we outperform supervised
learning without labels on imagenet? arXiv preprint arXiv:2201.05119, 2022.

Linh Tran, Bastiaan S Veeling, Kevin Roth, Jakub Swiatkowski, Joshua V Dillon, Jasper Snoek,
Stephan Mandt, Tim Salimans, Sebastian Nowozin, and Rodolphe Jenatton. Hydra: Preserving
ensemble diversity for model distillation. arXiv preprint arXiv:2001.04694, 2020.

Xiao Wang, Haoqi Fan, Yuandong Tian, Daisuke Kihara, and Xinlei Chen. On the importance of
asymmetry for siamese representation learning. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pp. 16570–16579, 2022.

Yeming Wen, Dustin Tran, and Jimmy Ba. Batchensemble: an alternative approach to efﬁcient

ensemble and lifelong learning. arXiv preprint arXiv:2002.06715, 2020.

Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes,
Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. Model
soups: averaging weights of multiple ﬁne-tuned models improves accuracy without increasing
inference time. In International Conference on Machine Learning, pp. 23965–23998. PMLR,
2022.

13

Published as a conference paper at ICLR 2023

Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. Unsupervised feature learning via non-
parametric instance discrimination. In Proceedings of the IEEE conference on computer vision
and pattern recognition, pp. 3733–3742, 2018.

Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database:
Large-scale scene recognition from abbey to zoo. In 2010 IEEE computer society conference on
computer vision and pattern recognition, pp. 3485–3492. IEEE, 2010.

Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu.
Simmim: A simple framework for masked image modeling. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 9653–9663, 2022.

Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. Barlow twins: Self-supervised
learning via redundancy reduction. In International Conference on Machine Learning, pp. 12310–
12320. PMLR, 2021.

Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. Image
BERT pre-training with online tokenizer. In International Conference on Learning Representations,
2022. URL https://openreview.net/forum?id=ydopy-e6Dg.

14

Published as a conference paper at ICLR 2023

A PSEUDOCODE

Algorithm 1: Pseudocode for computing ensemble loss
# b, n, c,:
# log_ps, log_pt:
# strategy:
# tau_ent:

ensemble loss average strategy
temperature for entropy weighting

batch size, number of ensemble heads, codebook size

student, teacher log probabilities with n ensembles

def ensemble_loss(log_ps, log_pt, strategy, tau_ent):

b, n, c = log_pt.shape # axis 1 corresponds to ensemble
log_pt = stop_grad(log_pt) # stop gradient for teacher

if strategy == "Unif":

loss = - (exp(log_pt) * log_ps).sum(axis=-1)
loss = loss.mean(axis=1) # average over ensembles

elif strategy == "Prob":

log_mean_pt = logsumexp(log_pt, axis=1, b=1/n) # mean teacher
log_mean_ps = logsumexp(log_ps, axis=1, b=1/n) # mean student
loss = - (exp(log_mean_pt) * log_mean_ps).sum(axis=-1)

elif strategy == "Ent":

ent = - (exp(log_pt) * log_pt).sum(axis=-1) # teacher entropy
weight = softmax(-ent/tau_ent, axis=1) # entropy weights
loss = - (exp(log_pt) * log_ps).sum(axis=-1)
loss = (loss * weight).sum(axis=1) # entropy weighted average

return loss.mean() # average over samples

number of ensemble heads, codebook size, momentum update rate

student, teacher encoders

Algorithm 2: Pseudocode for ensemble heads with simpliﬁed DINO
# n, c, eta:
# fs, ft:
# hs_ens, ht_ens:
# mus_ens, mut_ens:
# taus, taut:
# strategy:
# tau_ent:

ensemble loss average strategy
temperature for entropy weighting

student, teacher temperatures

student, teacher projection heads with n ensembles, list with length n

student, teacher codebooks with n ensembles, list with length n

for x in dataloader:

# load a batch with b samples
xs, xt = augs(x), augt(x) # random augmentations
zs, zt = fs(xs), ft(xt) # representations, (b, l)

# all following computation can be parallelized with batch computation
log_ps, log_pt = [], []
for j in range(n):

hs_j, ht_j = hs_ens[j], ht_ens[j] # j-th projection head
mus_j, mut_j = mus_ens[j], mut_ens[j] # j-th codebook, (d, c)

es_j, et_j = hs_j(zs), ht_j(zt) # j-th embedding, (b, d)

rs_j = (es_j @ mus_j) / (es_j.norm(axis=1, keepdims=True) * mus_j.norm(axis=0,

keepdims=True)) / taus # student logits, (b, c)

rt_j = (et_j @ mut_j) / (et_j.norm(axis=1, keepdims=True) * mut_j.norm(axis=0,

keepdims=True)) / taut # teacher logits, (b, c)

log_ps_j = logsoftmax(rs_j, axis=-1) # (b, c)
log_pt_j = logsoftmax(rt_j, axis=-1) # (b, c)
log_pt_j = renorm(log_pt_j) # adjust teacher predictions with centering or sinkhorn,

omitted here for simplicity

log_ps.append(log_ps_j)
log_pt.append(log_pt_j)

log_ps = stack(log_ps_j, axis=1) # stacked student log probablities, (b, n, c)
log_pt = stack(log_pt_j, axis=1) # stacked teacher log probablities, (b, n, c)

loss = ensemble_loss(log_ps, log_pt, strategy=strategy) # compute ensemble loss

loss.backward() # back-propagate
sgd_update(fs, hs_ens, mus_ens) # apply gradient decent update for student
ema_update(ft, ht_ens, mut_ens, rate=eta) # apply momentum update for teacher

15

Published as a conference paper at ICLR 2023

B EXPERIMENTAL DETAILS

In this section, we provide details for our experiments. In Appx. B.1, we describe how we reproduced
and improved the baseline DINO/MSN models. We give the implementation details for SSL training
and evaluation in Appx. B.2 and Appx. B.3 respectively. All the hyper-parameters used in our
experiments are in Appx. B.2.2.

B.1 REPRODUCING & IMPROVING BASELINES

We carefully reproduced and further improved baseline methods (denoted as DINO∗ and MSN∗
respectively) with an extensive study and hyperparameter search (see Appx. B.1). In particular, we
systematically study the projection head design (which we found is crucial for few-shot evaluation
performance (Appx. C.2)) and different techniques for avoiding collapse used in both methods
(Appx. C.1). DINO∗ performs signiﬁcantly better than DINO on few-shot evaluation (e.g., 2∼6
percentage point (p.p.) gains for 1 shot) and maintains the full-data evaluation performance. The
main adjustments of DINO∗ are: (i) A 3-layer projection head with a hidden dimension of 1024
(instead of 2048); (ii) Sinkhorn–Knopp (SK) normalization (instead of centering) is applied to teacher
predictions, combined with a smaller teacher temperature τ = 0.025 and codebook size c =1024 or
4096. MSN∗ uses the same projection head as DINO∗ and applies ME-MAX regularization without
SK normalization (which is applied in MSN by default). Further details for DINO and MSN can be
found below.

B.1.1 DINO

Table 5: Reproducing & Improving DINO. Our reproduce results match the public numbers. We
further improve the DINO baseline (DINO∗) by studying projection heads and collapse-avoiding
techniques. The evaluation results of DINO/DINO∗ ViT-S/16 trained with 800 epochs are reported.

Few-shot

Full-data

1

2

5

∼13 (1%)

k-NN

Linear

DINO (Caron et al., 2021)
DINO (Ours reproduced)
DINO∗ (Retuned)

38.9 ± 0.4
39.1 ± 0.3
44.6 ± 0.2

48.9 ± 0.3
49.1 ± 0.5
53.6 ± 0.3

58.5 ± 0.1
58.6 ± 0.2
61.1 ± 0.2

64.5
64.7
66.2

74.5
74.3
74.1

76.1 / 77.0
75.8 / 76.9
75.8 / 76.9

Reproducing DINO We carefully reproduced DINO with JAX following the ofﬁcial DINO im-
plementation1. In Table 5, we report the evaluation results of DINO using ViT-S trained with 800
epochs following the exact training conﬁguration for ViT-S/16 in the ofﬁcial DINO code. The
ofﬁcial results of full-data evaluation and 1%-data evaluation are from Caron et al. (2021), the other
few-shot evaluation results are evaluated by Assran et al. (2022) and also validated by us. Note
that for consistency of full-data linear evaluation, we report the results with both the [CLS] token
representations of the last layer and the concatenation of the [CLS] token representations from the
last 4 layers following Caron et al. (2021). For 1-/2-/5-shots evaluation results, we report the mean
accuracy and standard deviation across 3 random splits of the data following Assran et al. (2022).
As shown in Table 5, our reproduced results are all comparable with the published numbers which
validates the implementation of our training and evaluation pipelines.

Improving DINO We improved the DINO baseline with a systematic empirical study of some
important components. We ﬁrst empirically compared different techniques for avoiding collapse
(see Appx. C.1) and ﬁnd that Sinkhorn-Knopp (SK) normalization is a more effective and also
simpler technique for encouraging codebook usage than the centering operation used in DINO. We
thus applied SK normalization, which enabled us to use a smaller teacher temperature τ = 0.025
(instead of τ = 0.07) and a much smaller codebook size c =1024 or 4096 (instead of 65536).
These modiﬁcations lead to similar performance as DINO with a much smaller codebook (up to
1M parameters, compared to 16M parameters for DINO). Next we empirically studied the effect
of projection heads for different evaluation metrics (see Appx. C.2), and found that the design of

1https://github.com/facebookresearch/dino

16

Published as a conference paper at ICLR 2023

projection heads is crucial for few-shot evaluation metrics and an too power powerful projection head
(e.g., the 3-layer MLP with a hidden dimension of 2048 used in DINO/MSN/etc.) could signiﬁcantly
hurt the few-shot performance. With an empirically study of projection head architectures, we found
that a simply reducing the hidden dimension to 1024 could signiﬁcantly improves the few-shot
evaluation performance while maintaining full-data evaluation performance. The improved results of
DINO∗ are shown in Table 5.

B.1.2 MSN

Table 6: Reproducing & improving MSN. We implement MSN∗ by adding ME-MAX regularization
and masking to DINO∗, which surpasses public MSN results. The evaluation results of MSN/MSN∗
ViT-S/16 trained with 800 epochs are reported.

Few-shot

Full-data

1

2

5

∼13 (1%)

k-NN

Linear

MSN (Assran et al., 2022)
MSN (Repro)
MSN∗ (Retuned)

47.1 ± 0.1
39.1 ± 0.3
47.4 ± 0.1

55.8 ± 0.6
49.2 ± 0.3
56.3 ± 0.4

62.8 ± 0.3
58.4 ± 0.1
62.8 ± 0.2

67.2
64.3
67.1

-
72.8
73.3

-

/ 76.9
74.7 / 75.5
75.6 / 76.6

We carefully implemented MSN by adding its main components, i.e., ME-MAX regularization and
masking, to the DINO implementation (denoted as MSN∗), which surpassed public results as shown
in Table 6. Note that the implementation of MSN∗ does not exactly match the public implementation
in the public MSN code2, where the main differences are:

• MSN applies ME-MAX with Sinkhorn-Knopp normalization by default (as in the released
training conﬁguration), which we empirically ﬁnd does not work very well (see Table 9).
MSN∗ does not apply SK normalization and tunes the regularization strength for ME-MAX.
• Some differences in implementation details, e.g., schedules for learning rate/weight decay,
batch normalization in projection heads, speciﬁc data augmentations, etc. MSN∗ uses the
exact same setup as DINO∗ which follows original DINO implementation.

We initially tried to exactly reproduce the original MSN following the public MSN code, but the
results are much below the public ones, as shown in Table 6. Incorporating the two differences above
bridges the gap and makes MSN∗ surpass the public results.

B.2 PRETRAINING DETAILS

In this subsection, we provide the general implementation details in Appx. B.2.1 and speciﬁc hyper-
parameters in Appx. B.2.2 in Appx. B.2.2 for reproducibility.

B.2.1

IMPLEMENTATION DETAILS

Common setup We experimented with DINO (Caron et al., 2021) and MSN (Assran et al., 2022)
models on ImageNet ILSVRC-2012 dataset (Deng et al., 2009). We mainly followed the training setup
in Caron et al. (2021). In particular, all models were trained with AdamW optimizer (Loshchilov
& Hutter, 2018) and a batch size of 1024. The learning rate was linearly warmuped to 0.002
(=0.001×batch size/512) and followed a cosine decay schedule. The weight decay followed a cosine
schedule from 0.04 to 0.4. The momentum rate for the teacher was increased from 0.996 to 1 with a
cosine schedule following BYOL (Grill et al., 2020). A stochastic depth (Huang et al., 2016) of 0.1
was applied without dropout (Srivastava et al., 2014). The student temperature τ is set to 0.1. As
with DINO, we used the data augmentations of BYOL and multi-crop augmentation of SWAV (Caron
et al., 2020). In particular, 2 global views with a 224×224 resolution and crop area range [0.25, 1.0]
were generated for the teacher and student, and another 10 local views with 96×96 resolution and
crop area range [0.08, 0.25] were used as extra augmented inputs for the student. For MSN, we
used the exact same setup and incorporated its major component: 1) mean entropy maximization
(ME-MAX) regularization; 2) masking as an extra augmentation applied to the student global view.

2https://github.com/facebookresearch/msn

17

Published as a conference paper at ICLR 2023

Main modiﬁcations We retuned the baselines (DINO∗ and MSN∗) as detailed in Appx. B.1, and
the main adjustments are as followed. We used a 3-layer projection head with a hidden dimension
of 1024. The output embedding (i.e., (hψ ◦ rω)(x)) and the codes (i.e., µ) both have a dimension
of 256 and are L2 normalized. For DINO∗, Sinkhorn–Knopp (SK) normalization was applied to
teacher predictions. For MSN∗, ME-MAX was used without SK normalization and the regularization
strength was tuned over {3, 4, 5}. For all models, we used teacher temperature τ = 0.025 which was
linearly decayed from 0.05 for the ﬁrst 30 epochs. The codebook size c was selected over {1024,
4096} for all models, and typically c =4096 was selected for baseline methods and c =1024 was
selected for ours. For our (hψ, µ)-ensembles with ENT, entropy weighting temperature γ is linearly
decayed from 0.5 to the speciﬁed value.

B.2.2 HYPER-PARAMETERS

We report the hyperparameters for training our models for reproducibility:

Table 7: Hyper-parameters for training the DINO∗ model.

Hyper-parameter

training epoch
batch size
learning rate
warmup epoch
min lr
weight decay
stochastic depth
gradient clip

momentum
# of multi-crops
masking ratio

proj. layer
proj. hidden dim
emb. dim d
rep. dim
codebook size c

student temp.
teacher temp.
te. temp. decay epoch
center
SK norm
ME-MAX weight

ent. weight temp. γ
γ init.
γ decay epoch

ViT-S/16
DINO∗-PROB (16)

DINO∗

DINO∗-ENT (4/16)

DINO∗

DINO∗-ENT (16)

DINO∗

ViT-B/16

ViT-B/8
DINO∗-ENT (16)

800
1024
2e-3
10
1e-5
0.04 → 0.4
0.1
3.0

0.996 → 1.0
10
-

4096

1024

3
1024
256
384

0.1
0.025
30
(cid:55)
(cid:51)
-

400
1024
2e-3
30
1e-5
0.04 → 0.4
0.1
1.0

0.996 → 1.0
10
-

3
1024
256
768

300
1024
2e-3
10
4e-5
0.04 → 0.4
0.1
3.0

0.996 → 1.0
10
-

3
1024
256
768

1024

4096

1024

4096

1024

0.1
0.025
30
(cid:55)
(cid:51)
-

0.1
0.025
30
(cid:55)
(cid:51)
-

-
-
-

-
-
-

0.05
0.5
30

-
-
-

0.05
0.5
30

-
-
-

0.06
0.5
30

B.3 EVALUATION PROTOCALS

Few-shot linear evaluation We followed the few-shot evaluation protocal in Assran et al. (2022).
Speciﬁcally, we used the 1-/2-/5-shot ImageNet dataset splits3 in Assran et al. (2022) and 1%
(∼13-shot) ImageNet dataset splits4. For given labelled images, we took a single central crop
of size 224 × 224 without additional data augmentations, and extracted the output [CLS] token
representations from the frozen pretrained model. Then we trained a linear classiﬁer with multi-class
logistic regression on top of the extracted representations. We used the scikit-learn package
(Pedregosa et al., 2011) for the logistric regression classiﬁer. For all few-shot evaluations, we searched
the L2 regularization strength over {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10}.

Full-data linear evaluation We followed the linear evaluation protocal in (Caron et al., 2021).
Speciﬁcally, we trained a linear classiﬁer on top of the representations extracted from the frozen
pretrained model. The linear classiﬁer is optimized by SGD with Nesterov momentum (Nesterov,
1983; Sutskever et al., 2013) of 0.9 and a batch size of 4096 for 100 epochs on the whole ImageNet
dataset, following a cosine learning rate decay schedule. We did not apply any weight decay.

3Publicly available at https://github.com/facebookresearch/msn
4Publicly available at https://github.com/google-research/simclr/tree/master/

imagenet_subsets

18

Published as a conference paper at ICLR 2023

Table 8: Hyper-parameters for training the MSN∗ model.

Hyper-parameter

training epoch
batch size
learning rate
warmup epoch
min lr
weight decay
stochastic depth
gradient clip

momentum
# of multi-crops
masking ratio

proj. layer
proj. hidden dim
emb. dim d
rep. dim
codebook size c

student temp.
teacher temp.
te. temp. decay epoch
center
SK norm
ME-MAX weight

ent. weight temp. γ
γ init.
γ decay epoch

ViT-S/16

ViT-B/16

ViT-B/8

DINO∗

MSN∗-ENT (2/8) MSN∗

MSN∗-ENT (8) MSN∗

MSN∗-ENT (8)

800
1024
2e-3
20
1e-5
0.04 → 0.4
0.1
1.0

0.996 → 1.0
10
0.2

3
1024
256
384

400
1024
2e-3
30
4e-5
0.04 → 0.4
0.1
1.0

0.996 → 1.0
10
0.2

3
1024
256
768

300
1024
2e-3
20
4e-5
0.04 → 0.4
0.1
1.0

0.996 → 1.0
10
0.15

3
1024
256
768

4096

1024

4096

1024

4096

1024

0.1
0.025
30
(cid:55)
(cid:55)
4.0

0.1
0.025
30
(cid:55)
(cid:55)
4.0

0.1
0.025
30
(cid:55)
(cid:55)
4.0

-
-
-

0.01
0.5
30

-
-
-

0.005
0.5
30

-
-
-

0.01
0.5
30

During training, we only applied basic data augmentations including random resized crops of size
224 × 224 and horizontal ﬂips. During testing, we took a single central crop of the same size. For
ViT-S/16, Caron et al. (2021) found that concatenating the [CLS] token representations from the
last l (speciﬁcally, l = 4) layers (c.f. Appendix F.2 in Caron et al. (2021)) improved the results by
about 1 p.p. We followed the same procedure, but reported linear evaluation results with both l = 1
and l = 4 in Table 2 for consistency. In our empirical study with ViT-S/16, we used the result with
l = 1. For larger models (e.g., ViT-B/16), we followed Caron et al. (2021); Zhou et al. (2022) to use
the concatenation of the [CLS] token representation and the average-pooled patch tokens from the
last l = 1 layer for linear evaluation. For all linear evaluations, we searched the base learning rate
over {4.8e-3, 1.6e-2, 4.8e-2, 1.6e-1, 4.8e-1, 1.6}.

Full-data k-NN evaluation We followed the k-NN evaluation protocal in Caron et al. (2021);
Wu et al. (2018). Speciﬁcally, for each image in the given dataset, we took a single central crop
of size 224 × 224 without additional data augmentations, and extracted the output [CLS] token
representations from the frozen pretrained model. The extracted representations are used for a
weighted k-Nearest-Neighbor classiﬁer. In particular, denote the stored training representations and
labels as D = {(zi, yi)}N
i=1. For a test image with extracted representation z, denote the set of its
top k-NN training samples as Dk[z] ⊆ D and |Dk[z]| = k. The k-NN set Dk[z] is used to make the
(cid:17)
prediction for the test image with a weighted vote, i.e., ˆy = arg maxy
,
where 1y=yj is the one-hot vector corresponding to label yj and αj is the weight induced by the
. We set τ (cid:48) = 0.07 without tuning as
cosine similarity between z and zj, i.e., αj = exp
in Caron et al. (2021); Wu et al. (2018). For all k-NN evaluations, we searched k over {5, 10, 20, 50,
100} and found that k = 10 or k = 20 was consistently the best.

(zj ,yj )∈Dk[z] αj1y=yj

zTzj
(cid:107)z(cid:107)(cid:107)zj (cid:107)

(cid:16) 1
τ (cid:48)

(cid:16)(cid:80)

(cid:17)

Transfer evaluation via linear probing We mainly followed the transfer evaluation protocal in
(Grill et al., 2020; Chen et al., 2020a). In particular, we used 9 of their 13 datasets that are available
in tensorflow-datasets (tfd), namely Food-101 (Bossard et al., 2014), CIFAR10 (Krizhevsky
et al., 2009), CIFAR100 (Krizhevsky et al., 2009), SUN397 scene dataset (Xiao et al., 2010), Stanford
Cars (Krause et al., 2013), Describable Textures Dataset (Cimpoi et al., 2014, DTD), Oxford-IIIT Pets
(Parkhi et al., 2012), Caltech-101 (Fei-Fei et al., 2004), Oxford 102 Flowers (Nilsback & Zisserman,

19

Published as a conference paper at ICLR 2023

Table 9: Empirical study of different techniques for avoiding collapse. Using Sinkhorn-Knopp
normalization instead of centering for DINO leads to improved performance, and matches the original
DINO even with a much smaller codebook. The ME-MAX regularization of MSN is very effective
and leads to signiﬁcant improvement for few-shot evaluations.

Center
(cid:88)

DINO

MSN

Technique

Few-shot

Full-data

Sinkhorn ME-MAX

1

2

5

∼13 (1%)

k-NN Linear

(cid:88)

(cid:88)

37.8 ± 0.4
39.1 ± 0.3

36.0 ± 0.4
43.9 ± 0.2

47.4 ± 0.3
49.4 ± 0.3

46.6 ± 0.6
53.0 ± 0.3

56.9 ± 0.4
58.7 ± 0.2

56.5 ± 0.2
61.1 ± 0.2

63.0
64.8

63.2
66.0

72.4
74.1

73.2
74.0

74.9
76.0

75.2
75.8

(cid:88)
(cid:88)

Table 10: ME-MAX regularization is sensitive to hyper-parameters.

Weight

Few-shot

Full-data

1

2

5

∼13 (1%) KNN Linear

1.0
3.0
5.0

37.6 ± 0.2
43.9 ± 0.2
43.6 ± 0.2

48.0 ± 0.4
53.0 ± 0.3
52.6 ± 0.4

57.7 ± 0.2
61.1 ± 0.2
60.4 ± 0.1

64.0
66.0
65.5

73.5
74.0
73.9

75.6
75.8
75.6

2008). Following their evaluation metrics, we reported mean per-class accuracy for Oxford-IIIT Pets,
Caltech-101, and Oxford 102 Flowers datasets and reported top-1 accuracy for other datasets. We
transferred the models pretrained on ImageNet (Deng et al., 2009) to these datasets by training a
linear classifer on top of frozen representations. In particular, we resized given images to 256 × 256
and took a single central crop of size 224 × 224 without additional data augmentations. We extracted
the output [CLS] token representations from the frozen pretrained model. Then we trained a linear
classiﬁer with multi-class logistic regression on top of the extracted representations. We used the
scikit-learn package (Pedregosa et al., 2011) for the logistric regression classiﬁer. For all
transfer evaluations, we searched the L2 regularization strength over {1e-6, 1e-5, 1e-4, 3e-4, 1e-3,
3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 1e1, 3e1, 1e2, 1e3, 1e4, 1e5}.

C ADDITIONAL RESULTS

C.1 EMPIRICAL STUDY OF TECHNIQUES FOR AVOIDING COLLAPSE

Most self-supervised learning methods utilize some techniques to avoid collapse of representations
with, e.g., contrastive loss (Chen et al., 2020a; He et al., 2020), batch normalization (Grill et al.,
2020), asymmetric architecture design with a predictor (Grill et al., 2020; Chen & He, 2021), etc. In
DINO and MSN, a learnable codebook is used for the learning objective and different techniques
are applied to encourage the effective codebook usage. There are two potential cases of collapse (as
discussed in Caron et al. (2021)):

• Dominating codes. This is the case of “winner-take-all”: only a small portion of codes are
being predicted while others are inactive. Typical solutions for avoiding this include applying
Sinkhorn–Knopp normalization (Cuturi, 2013) as in SWaV (Caron et al., 2020), centering
teacher logits as in DINO (Caron et al., 2021), and applying mean-entropy maximization
regularization (ME-MAX) as in MSN (Assran et al., 2022). Note that in MSN, ME-MAX is
combined with Sinkhorn–Knopp normalization by default.

• Uniform codes. This is the case where all codes are treated equally and the predictions
reduce to be uniform over codes. A simple and effective solution is to applying sharpening,
i.e., using a lower temperature for computing the teacher prediction.

We systematically study different techniques in a uniﬁed setup. In particular, we used DINO with the
ViT-S backbone, a 3-layer MLP projection head with hidden dimension 2048, and a codebook of
size 4096 and dimension 256. We applied different techniques to DINO and searched the teacher
temperature in {0.0125, 0.025, 0.05} for each. For ME-MAX, we searched regularization weight in

20

Published as a conference paper at ICLR 2023

(a) Merged

(b) 1-shot

(c) 1%-data

(d) Full-data

Figure 5: Effect of projection heads for different evaluation metrics. We compare non-ensemble
projection heads with different depths and widths as well as our (hψ, µ)-ensembles, and evaluate
linear evaluation performance with different amount of labeled data. (a) shows the comparison of
normalized metrics for non-ensembles. (b)-(d) compares non-ensemble and (hψ, µ)-ensembles by
unnormalized metrics. ‘Default’ denotes the default projection heads used in many SSL methods.
See analysis in Appx. C.2 for details.

{1.0, 3.0, 5.0}. For ME-MAX combined with Sinkhorn, we followed Assran et al. (2022) and used
default regularization weight of 1.0. The results are in Table 10. We observed that:

• DINO’s centering operation is not as strong as other techniques, and it favours a larger
teacher temperature (e.g., 0.05). It does not work well when the codebook size (4096)
is not as large as the one used in the original DINO model (65536). Switching to use
Sinkhorn–Knopp normalization leads to much more improved performance, and matches
the performance of original DINO (Table 5) with a much smaller codebook.

• MSN’s ME-MAX regularization is very effective, and leads to signiﬁcant improvements
over others. We also found it is sensitive to the regularization weight and teacher temperature
(c.f. Table 10). However, we observed that combining ME-MAX with Sinkhorn does not
work well without tuning the regularization weight (which is recommended by Assran et al.
(2022)).

C.2 EMPIRICAL STUDY OF PROJECTION HEADS

In this subsection, we systematically study the effect of projection heads for different evaluation
metrics. In particular, we used DINO∗ ViT-S/16 as the base model and used different projection
heads with (depth, width) searched over {2, 3, 4} × {512, 1024, 2048, 4096}. All models are trained
with 300 epochs using exact the same set of hyper-parameters. We measured the linear evaluation
performance with different amount of labeled data (i.e., full-data, 1%-data, 1-shot).

In Fig. 5a, we plot different evaluation metrics (normalized respectively by the best of each) versus
the number of projection head parameters. In Figs. 5b to 5d, we plot each unnormalized evaluation
metric respectively for different heads as well as our (hψ, µ)-ensembles. Our key ﬁndings are:

21

 ȱ Ȱ ȶ ȱ Ȱ ȷ Z ľ ċ ʌ  Ē ë  @ Ø ´ Ô  y ´ Į ´ ċ Ĺ Ø Į Ĳ Ȱ ʌ ȵ Ȱ ʌ ȶ Ȱ ʌ ȷ Ȱ ʌ ȸ Ȱ ʌ ȹ ȱ ʌ Ȱ Z Ē Į ċ ´ Ć ô Š Ø Ô  Y Ø Ĺ Į ô Î : ľ Ć Ć ʲ Ô ´ Ĺ ´ ȱ ̑  Ô ´ Ĺ ´ ȱ ʲ Ĳ ñ Ē Ĺ $ Ø ë ´ ľ Ć Ĺ ` ľ Į  Í ´ Ĳ Ø106107108Number of Head Paramters25303540AccuracyDefaultOur baselineEnsembleNon-ensemble106107108Number of Head Paramters556065AccuracyDefaultOur baselineEnsembleNon-ensemble106107108Number of Head Paramters72737475AccuracyDefaultOur baselineEnsembleNon-ensemblePublished as a conference paper at ICLR 2023

Figure 6: Effect of teacher temperature for non-ensemble DINO∗. DINO∗ with a lower temperature
can achieve better few-shot performance, but still under-performs our ensemble method (DINO∗-ENT
with 16 heads, orange lines). DINO∗ ViT-S/16 trained for 300 epochs is used and τ = 0.025 is used
for DINO∗-ENT.

• The projection head has a relatively larger impact on few-shot evaluation metrics, as reﬂected
by the relative magnitudes of different metrics in Fig. 5a. An too powerful non-ensemble
projection head signiﬁcantly hurts the label efﬁciency of learned representations, reﬂected
by a much larger drop in few-shot evaluation performance (up to 18 p.p. for 1-shot, 9 p.p.
for 1%-data). This result is also partially observed in Chen et al. (2020b), where they found
that probing from intermediate layers of projection heads (which can be viewed as using a
shallower head) could improve the semi-supervised learning (1%-/10%) results.

• The optimal projection head for different metrics can differ a lot. A weaker head improves
label efﬁciency (few-shot performance), while a stronger (but not too strong) head improves
linear decodability. As a result, the default projection head (3/2048) that is widely used
in SimCLR v2 (Chen et al., 2020b), DINO (Caron et al., 2021), iBOT (Zhou et al., 2022),
MSN (Assran et al., 2022), etc., does not perform well in few-shot evaluations (as shown by
the green cross denoted as ‘Default’), probably because it is selected by full-data evaluation
metrics.

• There exist some projection heads that performs decently well on all evaluation metrics, e.g.,
the baseline model (3/1024) used in our experiments (pink star denoted as ‘Our base’).

• Compared to naively tuning projection head architectures, our (hψ, µ)-ensembles (orange
curves in Figs. 5b to 5d) consistently improve all metrics with different amount of labeled
data, despite it also increases the number of parameters in projection heads. Our (hψ, µ)-
ensembles outperform all non-ensembles, which also include the counterparts of probing
from intermediate layers from the a deeper head (i.e., shallower heads).

C.3 EMPIRICAL STUDY OF (hψ, µ)-ENSEMBLES

Are the gains of ENT purely from sharper teacher predictions? Our ENT strategy assigns higher
weights to the heads that predict with lower entropies, thus effectively uses sharper teacher predictions
as the targets. One may be curious about how this effect accounts for the gains of the ENT strategy.
We empirically answer this question by studying the non-ensemble baseline that uses a sharper
teacher predictions in a data-independent manner (in contrast to ENT, which uses data-dependent
entropy weights). Speciﬁcally, we compare the non-ensemble DINO∗ that use different teacher
temperature τ ∈ {0.005, 0.01, 0.025, 0.05} and also our DINO∗-ENT (16) with τ = 0.025, as shown
in Fig. 6. We ﬁnd that the teacher temperature has a big impact on evaluation results especially for
few-shot evaluation. Compared to our default baseline that uses τ = 0.025, a lower temperature (e.g.,
τ = 0.01) can indeed improve the 1-shot performance (at the cost of worse full-data performance).
However, an too low temperature (τ = 0.005) will hurt the performance. Our DINO∗-ENT (16)
consistently outperform all the baselines, which implies the importance of selecting sharper teacher
predictions in a data-dependent manner.

22

0.00.20.40.60.81.00.00.20.40.60.81.0Accuracy737572.273.974.473.660626459.563.163.461.30.0050.010.0250.05Teacher Temperature3840424437.141.740.637.6Full data1% data1-shotEnsemblePublished as a conference paper at ICLR 2023

Table 11: Full table of Table 1 including all metrics for comparing different ensemble strategies.
ENT and PROB signiﬁcantly improves over the non-ensemble baseline, while UNIF leads to no gains.
Ensembling the whole projection head works the best. All models are DINO∗ ViT-S/16 trained for
300 epochs. The means and standard deviations over 3 initialization seeds for all evaluation results
are reported.

How

Base

UNIF

PROB
PROB

ENT
ENT
ENT

ENT-ST

Where

Few-shot

Full-data

Proj. Head Codebook

1

2

5

∼13 (1%)

k-NN

Linear

(cid:88)

(cid:88)
(cid:88)

(cid:88)
(cid:88)

(cid:88)

(cid:88)

(cid:88)

(cid:88)

(cid:88)

(cid:88)

40.6 ± 0.2

49.8 ± 0.2

57.9 ± 0.3

63.4 ± 0.2

72.3 ± 0.1

74.4 ± 0.1

40.4 ± 0.4

49.5 ± 0.4

57.6 ± 0.3

63.3 ± 0.3

72.2 ± 0.2

74.5 ± 0.2

39.7 ± 0.5
41.9 ± 0.3

40.6 ± 0.4
43.0 ± 0.6
44.0 ± 0.2

49.0 ± 0.5
51.5 ± 0.5

49.5 ± 0.6
52.2 ± 0.8
53.0 ± 0.5

57.4 ± 0.4
59.6 ± 0.4

58.0 ± 0.4
59.7 ± 0.7
60.5 ± 0.3

63.0 ± 0.4
65.1 ± 0.3

63.5 ± 0.4
64.8 ± 0.5
65.5 ± 0.1

72.8 ± 0.2
73.7 ± 0.3

72.1 ± 0.3
72.9 ± 0.6
73.2 ± 0.1

74.8 ± 0.1
75.4 ± 0.1

74.5 ± 0.3
75.1 ± 0.4
75.3 ± 0.1

40.0 ± 0.5

39.2 ± 0.6

57.3 ± 0.5

62.7 ± 0.5

71.9 ± 0.4

74.0 ± 0.4

Table 12: Comparison of different varaints of PROB. The PROB strategy used in our experiments
performs the best. ’-’ in the table denotes training divergence for PROB-MAX. The experimental
setup is the same as Table 11.

How

Base

Where

Few-shot

Full-data

Weight by

Temp. γ

1

2

5

∼13 (1%)

k-NN

Linear

40.6 ± 0.2

49.8 ± 0.2

57.9 ± 0.3

63.4 ± 0.2

72.3 ± 0.1

74.4 ± 0.1

PROB
PROB-TE
PROB-MAX
PROB-MAX-TE

student
teacher
student
teacher

1
1
0
0

41.9 ± 0.3
41.5 ± 0.2
-
41.4 ± 0.2

51.5 ± 0.5
50.4 ± 0.3
-
50.3 ± 0.3

59.6 ± 0.4
58.3 ± 0.3
-
58.1 ± 0.3

65.1 ± 0.3
63.7 ± 0.1
-
63.6 ± 0.2

73.7 ± 0.3
72.3 ± 0.2
-
72.3 ± 0.2

75.4 ± 0.1
74.6 ± 0.1
-
74.5 ± 0.2

Comparison of different ensemble strategies and variants We present the full table of Table 1
that includes all the metrics in Table 11. The same observation holds for all metrics.

For all previous studies, we considered a speciﬁc instantiation of PROB strategy, i.e., weight by
student predicted probabilities fijy = log s(y|θj, x) and γ = 1, which has a nice interpretation of
model average (see Sec. 3.3). We also studied different variants of the PROB strategy (see Appx. D.1),

• PROB-TE: weight by teacher fijy = log ti(y|x) and γ = 1;

• PROB-MAX: weight by student fijy = log sj(y|x) and γ → 0;

• PROB-MAX-TE: weight by teacher fijy = log ti(y|x) and γ → 0

Table 12 compares the downstream performance for all the variants. We ﬁnd that the our PROB (used
in our empirical studies) performs better than other variants. Interestingly, weighting by the teacher
(PROB-TE) performs worse than PROB. We conjecture that this is because the important weights turn
out to give a weighted average of teacher predictions as the surrogate target that is shared across
all students (like PROB) but does not give effective preferential treatment across students which are
directly optimized (unlike PROB-TE). Furthermore, PROB-MAX which sharpens the importance
weights leads to training divergence. This is probably because the student predictions have higher
variance based on which sharp weights lead to unstable training. In contrast, PROB-MAX-TE which
uses the (lower-variance) teacher gives reasonable results and comparable to PROB-TE.

Number of ensembles for MSN∗
In Fig. 7a, we study the effect of increasing the number of
(hψ, µ)-ensembles for MSN∗-ENT with ViT-S/16 trained for 800 epochs. The scaling trend is similar
to DINO∗-ENT (Fig. 3a) and the gains start to diminish when the number of heads increases above 8.

Effect of ENT temperature γ for MSN∗ Fig. 7b studies the effect of entropy weighting temper-
ature γ for MSN∗-ENT. We observed that MSN∗ is more robust to small temperatures, and the

23

Published as a conference paper at ICLR 2023

(a) Scaling of ensembles

(b) Effect of ENT temperature γ

Figure 7: Empirical study for MSN∗-ENT. (a) The gains by increasing the number of (hψ, µ)-
ensembles start to diminish when it is over 8 heads. (b) MSN∗ prefers smaller temperature for entropy
weighting than DINO∗.

best γ = 0.01 is smaller than that of DINO∗ (γ = 0.05). When the temperature is too high, the
performance drops as a result of under-specialization (i.e., less diversity) as with DINO∗.

C.4 ANALYZING (hψ, µ)-ENSEMBLE DIVERSITY

Visualizing (hψ, µ)-ensemble similarity We analyze the diversity between different heads by
visualizing the similarity matrix between their codes. Directly measuring the similarity between
codes in two heads could not work, because 1) they may live in different subspaces because of the
ensembled projection heads; 2) they may not align in the natural order but in a permuted order.

Therefore, we seek to align codes between different heads by how they are effectively used to ‘cluster’
the data. In particular, we use a set of randomly sampled inputs (cid:8)xi(cid:9)
i∈[b] of size b = 51200 to
obtain an empirical code assignment matrix Aj ∈ Rb×c for each (hψ, µ)-ensemble j ∈ [m], where
the i-th row of Aj corresponds to the teacher predictions tj(Y |xi). For the k-th code in the head
j, we extract the k-th column from Aj (i.e., its empirical assignment) as its embedding. For two
codes, we measure their similarity by the cosine similarity between their embeddings. For a pair of
heads j and j(cid:48), we align their codes using the Hungarian algorithm (Kuhn, 1955) to maximize the
sum of cosine similarity. After that, we plot the similarity matrix which is aligned and reordered
by the similarity value on the diagonal (in an descending order). Note that it is not necessary to
do the alignment procedure for the PROB strategy since it is naturally aligned because of the direct
distribution averaging over (hψ, µ)-ensembles, but we did for fair comparison with other strategies.
We applied the same procedure for different ensemble weighting strategies using DINO∗ with 4
(hψ, µ)-ensembles. We randomly picked a pair of heads and visualize the similarity matrix before
(top row) and after (bottom row) the alignment-reordering setup in Fig. 8. We found that before the
alignment procedure, the similarity matrix of the PROB strategy already mostly aligns because it
explicitly introduces code correspondence between different heads. Furthermore, by analyzing the
similarity decay pattern on the diagonal, it is clear that ENT learns the most diverse (hψ, µ)-ensembles
while UNIF learns the least ones, which may explain the difference of their empirical performance.
For completeness, we also include the visualization of aligned similarity matrices for all pairs of
heads in Figs. 9 to 11, the observations are the same.

24

 Ȱ ʌ Ȱ Ȱ ʌ Ȳ Ȱ ʌ ȴ Ȱ ʌ ȶ Ȱ ʌ ȸ ȱ ʌ Ȱ Ȱ ʌ Ȱ Ȱ ʌ Ȳ Ȱ ʌ ȴ Ȱ ʌ ȶ Ȱ ʌ ȸ ȱ ʌ Ȱ  Î Î ľ Į ´ Î Ř ȷ ȶ ȷ ȷ ȷ ȵ ʌ ȶ ȷ ȶ ʌ Ȱ ȷ ȶ ʌ ȳ ȷ ȶ ʌ ȴ ȷ ȶ ʌ ȶ ȶ ȷ ȶ ȸ ȶ ȷ ʌ ȱ ȶ ȷ ʌ ȹ ȶ ȸ ʌ ȳ ȶ ȸ ʌ ȷ ȶ ȸ ʌ ȶ ȱ Ȳ ȴ ȸ ȱ ȶ Z ľ ċ ʌ  Ē ë  ( Č Ĳ Ø ċ Í Ć Ø Ĳ ȴ ȷ ȴ ȸ ȴ ȹ ȵ Ȱ ȴ ȷ ʌ ȴ ȴ ȸ ʌ ȸ ȴ ȹ ʌ ȳ ȵ Ȱ ʌ ȱ ȴ ȹ ʌ ȹ : ľ Ć Ć  Ô ´ Ĺ ´ ȱ ̑  Ô ´ Ĺ ´ ȱ ʲ Ĳ ñ Ē Ĺ0.00.20.40.60.81.00.00.20.40.60.81.0Accuracy7676.276.376.276.276.076.063646564.664.964.865.064.464.10.0050.010.020.040.070.1Entropy Weight Temperature484948.849.349.249.048.548.3Full data5-shot1-shotBaselinePublished as a conference paper at ICLR 2023

(a) UNIF

(b) PROB

(c) ENT

Figure 8: Visualization of (hψ, µ)-ensemble diversity. ENT learns the most diverse (hψ, µ)-
ensembles while UNIF learns the least ones. We visualize the code similarity matrix between
a pair of randomly selected projection heads. Top row shows the original similarity matrix (i.e., in
natural order) and the bottom row shows the aligned similarity matrix which aligns codes by empirical
assignment probabilities. DINO∗ ViT-S/16 with 4 heads is used. Best viewed in color.

Figure 9: Visualization of (hψ, µ)-ensemble diversity between all pairs of heads for DINO∗-
UNIF. The UNIF strategy does not learn diverse (hψ, µ)-ensembles. DINO∗ with ViT-S/16 and 4
heads is used. Best viewed in color.

25

0.00.20.40.60.81.0Head 1 - Head 2Head 1 - Head 3Head 1 - Head 4Head 2 - Head 3Head 2 - Head 4Head 3 - Head 40.00.20.40.60.81.0Published as a conference paper at ICLR 2023

Figure 10: Visualization of (hψ, µ)-ensemble diversity between all pairs of heads for DINO∗-
PROB. The PROB strategy learns more diverse (hψ, µ)-ensembles than UNIF. DINO∗ with ViT-S/16
and 4 heads is used. Best viewed in color.

Figure 11: Visualization of (hψ, µ)-ensemble diversity between all pairs of heads for DINO∗-
ENT. The ENT strategy learns the most diverse (hψ, µ)-ensembles. DINO∗ with ViT-S/16 and 4
heads is used. Best viewed in color.

26

Head 1 - Head 2Head 1 - Head 3Head 1 - Head 4Head 2 - Head 3Head 2 - Head 4Head 3 - Head 40.00.20.40.60.81.0Head 1 - Head 2Head 1 - Head 3Head 1 - Head 4Head 2 - Head 3Head 2 - Head 4Head 3 - Head 40.00.20.40.60.81.0Published as a conference paper at ICLR 2023

D ANALYSIS

D.1 DERIVATIONS

In this subsection, we provide derivations for some non-trivial losses that we explore within our
framework.

Recall that our weighted cross-entropy loss is of the form,

Ln(θ) =

=

1
n

1
n

x∈Dn
(cid:88)

x∈Dn

(cid:88)

(cid:88)

H×[wijY (cid:12) ti(Y |x), s(Y |θj, x)]

i,j∈[m]

(cid:88)

(cid:88)

wijyti(y|x) log s(y|θj, x)

y∈Y

i,j∈[m]
(cid:16)(cid:110) 1

where wijy = softmax

γ fijy(stopgrad(θ), x) : i, j ∈ [m]

(15)

(16)

(17)

.

(cid:111)(cid:17)

Fuethermore, observe that,

(cid:88)

∇θ

i,j∈[m]

H×[wijY (cid:12) ti(Y |x), s(Y |θj, x)] =

(cid:90)

(cid:88)

i,j∈[m]

Y

wijyti(y|x)∇θ log s(y|θj, x)dy.

(18)

This indicates that the proposed weighted ensemble SSL loss is simply a reweighted log-likelihood
loss. We use this fact in our derivation of probability weighting (PROB) loss.

Uniform weighting (UNIF) Our UNIF strategy in Eq. (6) uses fijy = log δ(i − j) which gives
wijy = 1

m δ(i − j) (for any choice of γ), thus the loss,
1
n

LUNIF
n

(θ) =

1
m

(cid:88)

(cid:88)

(cid:88)

δ(i − j)ti(y|x) log s(y|θj, x)

=

1
n

y∈Y

x∈Dn
(cid:88)

x∈Dn

i,j∈[m]
1
m

(cid:88)

i∈[m]

H×[ti(Y |x), s(Y |θi, x)]

(19)

(20)

This loss assigns equal weights to m pairs of pairwised student/teacher.
An straightforward generalization is to assign equal weights to all possible pairs (m2) of stu-
dent/teacher with fijy = 0 and wijy = 1

m2 , which gives the UNIF-ALL loss,

LUNIF-ALL

n

(θ) =

1
n

(cid:88)

x∈Dn

1
m2

(cid:88)

i,j∈[m]

H× [ti(Y |x), s(Y |θj, x)] ,

(21)

Probability weighting (PROB) Recall our PROB loss in Eq. (7) has the form,

LPROB
n

(θ) =

1
n

(cid:88)

H×

x∈Dn





1
m

(cid:88)

i∈[m]

ti(Y |x),

1
m

(cid:88)

j∈[m]



s(Y |θj, x)

 .

(22)

We derive its equivalence with our general loss with fijy = log s(y|θj, x) and γ = 1 in terms of the
gradients,

∇θLPROB
n

(θ) =

=

=

1
m

1
m

1
m

(cid:90)

(cid:88)

Y

(cid:90)

i∈[m]

(cid:88)

i∈[m]

Y

(cid:90)

(cid:88)

i∈[m]

Y

ti(y|x) log

1
m

ti(y|x)∇θ log

(cid:88)

s(y|θj, x)dy

j∈[m]
1
m

(cid:88)

j∈[m]

s(y|θj, x)dy

ti(y|x)

(cid:80)

1
m

j∈[m] ∇θs(y|θj, x)
(cid:80)
j∈[m] s(y|θj, x)

dy

1
m

27

(23)

(24)

(25)

Published as a conference paper at ICLR 2023

=

=

1
m

1
m

= ∇θ

(cid:90)

(cid:88)

i∈[m]

(cid:88)

Y

(cid:90)

Y

i,j∈[m]
1
m

(cid:88)

i,j∈[m]

1
m

ti(y|x)

(cid:80)

j∈[m] s(y|θj, x)∇θ log s(y|θj, x)
j(cid:48)∈[m] s(y|θj(cid:48), x)

(cid:80)

1
m

dy

(26)

ti(y|x)

s(y|θj, x)
j(cid:48)∈[m] s(y|θj(cid:48), x)

(cid:80)

∇θ log s(y|θj, x)dy

(27)

H× [wijY (cid:12) ti(Y |x), s(Y |θj, x)]

(28)

s(y|θj ,x)

(cid:80)

where wijy =
j(cid:48) ∈[m] s(y|θj(cid:48) ,x) (or equivalently, fijy = log s(y|θj, x) and γ = 1). The last equality
is because wijy is stopped gradient with respect to θ. This is the same analysis as done in Burda et al.
(2016). The above formation establishes the equivalence of gradients between two losses, which
implies the same behavior (e.g., optimum) using gradient-based optimization, as the common practice
of deep learning.

We also generalize this loss to some variants which we explore in Table 12. A “dual” variant is to use
teacher predictions fijy = log ti(y|x) instead of student ones; this implies wijy =
and the PROB-TE loss,

ti(y|x)
i(cid:48)∈[m] ti(cid:48) (y|x)

(cid:80)

LPROB-TE

n

(θ) =

1
n

(cid:88)

(cid:88)

(cid:88)

x∈Dn

i,j∈[m]

y∈Y

ti(y|x)
i(cid:48)∈[m] ti(cid:48)(y|x)

(cid:80)

ti(y|x) log s(y|θj, x).

(29)

Note that this simply reduces to use a weighted teacher predictions
surrogate target that is shared across all students.

(cid:80)

ti(y|x)

i(cid:48) ∈[m] ti(cid:48) (y|x) ti(y|x) as the

Another generalization is to use “hard” weighting, i.e., γ → 0, which gives the PROB-MAX loss that
only assigns weight to the most conﬁdent student,

LPROB-MAX

n

(θ) =

(cid:88)

(cid:88)

(cid:88)

wijyti(y|x) log s(y|θj, x)

1
n

y∈Y
where wijy = δ(i − i∗)δ(j − j∗),

i,j∈[m]

x∈Dn

(i∗, j∗) = arg max

ij

fijy, ∀y ∈ Y.

(30)

(31)

This loss reduces to a generalization of multiple choice learning (Guzman-Rivera et al., 2012) used in
multi-headed networks (Lee et al., 2015) in our ensemble SSL setup. Similarly we can also derive
the dual variant of it that uses the teacher predictions, which is omitted here for brevity.

Entropy weighting (ENT) The derivation of ENT loss in Eq. (9) is similar to the UNIF loss but
applies an entropy weights. Recall that we use fijy = − H[ti(Y |x)] + log δ(i − j), which gives
wijy = softmaxi({− 1

γ H[ti(cid:48)(Y |x)] : i(cid:48) ∈ [m]}) and,

LENT

n (θ) =

1
n

(cid:88)

(cid:88)

x∈Dn

i∈[m]

softmaxi({− 1

γ H[ti(cid:48)(Y |x)] : i(cid:48) ∈ [m]}) H× [ti(Y |x), s(Y |θi, x)] .

(32)

One can also generalizes it to its dual variant which uses the student entopies, i.e.,fijy =
− H[s(Y |θj, x)] + log δ(i − j), which gives the ENT-ST loss,

LENT-ST
n

(θ) =

1
n

(cid:88)

(cid:88)

x∈Dn

i∈[m]

softmaxi({− 1

γ H[s(Y |θi(cid:48), x)] : i(cid:48) ∈ [m]}) H× [ti(Y |x), s(Y |θi, x)] .

(33)

D.2 RELATING SOME LOSSES

Here, we relate some losses derived above. Speciﬁcally, we relate the uniform weighting (UNIF,
UNIF-ALL) and probability weighting (PROB) in Appx. D.2.1, and relate entropy weighting (ENT)
and variance weighting in Appx. D.2.2.

28

Published as a conference paper at ICLR 2023

D.2.1 UNIFORM & PROBABILITY WEIGHTING

We ﬁrst establish the relation between UNIF and PROB using the joint convexity of unnormalized
KL divergence and the fact that our weighted cross-entropy loss is a weighted unnormalized KL
divergence up to some constant in θ. In particular, the joint convexity of unnormalized KL divergence
can be shown by combining the facts that Csiszàr f -divergences are jointly convex (Proposition 1 in
Dragomir (2013)) and unnormalized KL divergence corresponds to the convex generator, f (u) =
u log u − u + 1, as required by the proposition.

First, our weighted cross-entropy loss is unnormalized KL divergence up to some constant in θ:

1
m

i∈[m]


LUNIF
n

(θ) =

LPROB
n

(θ) =

1
n

1
n

(cid:88)

x∈Dn

(cid:88)

x∈Dn

(cid:88)

K[ti(Y |x), s(Y |θi, x)] + constant

(34)

K



1
m

(cid:88)

i∈[m]

ti(Y |x),



s(Y |θj, x)

 + constant

(35)

1
m

(cid:88)

j∈[m]

Therefore, the joint convexity of (unnormalized) KL divergence directly implies an ordering of the
loss up to some constant in θ, i.e.,

n ≤ LUNIF
Furthermore, we can also relate PROB and UNIF-ALL using the fact that the (unnormalized) cross-
entropy H×[p(X), q(X)] is linear in the ﬁrst argument p but convex in the second argument q, which
implies,

LPROB

(36)

n

LPROB

n ≤ LUNIF-ALL
n

D.2.2 ENTROPY & VARIANCE WEIGHTING

Suppose p(X) is a discrete distribution (normalized) on X = [c]. It can be shown that,

H[p(X)] ≤ 1

(cid:1) + 1
x∈[c] p(x)(x − µ)2 and µ = Ep[X] = (cid:80)

2 log (cid:0)Varp[X] + 1

12

2 log (2πe)

where Varp[X] = (cid:80)
x∈[c] p(x)x (Theorem 9.7.1, Cover &
Thomas (1999)). Note, a tighter bound (Mow, 1998) also exists but it places stronger restrictions on
p. This relationship suggests that choosing weights proportional to exp(− H[ti(Y |x)]) (as in ENT) is
potentially related to choosing weights proportional to weighting by variance (Varti(Y |x)[Y ]+(cid:15))−1/2
where ((cid:15) = 1

12 ).

(37)

(38)

29

