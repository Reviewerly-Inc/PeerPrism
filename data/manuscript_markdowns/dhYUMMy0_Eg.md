Published as a conference paper at ICLR 2023

EQUAL IMPROVABILITY: A NEW FAIRNESS NOTION
CONSIDERING THE LONG-TERM IMPACT

Ozgur Guldogan∗ ‡, Yuchen Zeng∗ §, Jy-yong Sohn† ¶, Ramtin Pedarsani‡, Kangwook Lee§
‡ University of California, Santa Barbara ¶ Yonsei University § University of Wisconsin-Madison

ABSTRACT

Devising a fair classifier that does not discriminate against different groups is an
important problem in machine learning. Recently, effort-based fairness notions are
getting attention, which considers the scenarios of each individual making effort
to improve its feature over time. Such scenarios happen in the real world, e.g.,
college admission and credit loaning, where each rejected sample makes effort to
change its features to get accepted afterward. In this paper, we propose a new effort-
based fairness notion called Equal Improvability (EI), which equalizes the potential
acceptance rate of the rejected samples across different groups assuming a bounded
level of effort will be spent by each rejected sample. We also propose and study
three different approaches for finding a classifier that satisfies the EI requirement.
Through experiments on both synthetic and real datasets, we demonstrate that the
proposed EI-regularized algorithms encourage us to find a fair classifier in terms
of EI. Additionally, we ran experiments on dynamic scenarios which highlight
the advantages of our EI metric in equalizing the distribution of features across
different groups, after the rejected samples make some effort to improve. Finally,
we provide mathematical analyses of several aspects of EI: the relationship between
EI and existing fairness notions, and the effect of EI in dynamic scenarios. Codes
are available in a GitHub repository 1.

1

INTRODUCTION

Over the past decade, machine learning has been used in a wide variety of applications. However,
these machine learning approaches are observed to be unfair to individuals having different ethnicity,
race, and gender. As the implicit bias in artificial intelligence tools raised concerns over potential
discrimination and equity issues, various researchers suggested defining fairness notions and develop-
ing classifiers that achieve fairness. One popular fairness notion is demographic parity (DP), which
requires the decision-making system to provide output such that the groups are equally likely to
be assigned to the desired prediction classes, e.g., acceptance in the admission procedure. DP and
related fairness notions are largely employed to mitigate the bias in many realistic problems such as
recruitment, credit lending, and university admissions (Zafar et al., 2017b; Hardt et al., 2016; Dwork
et al., 2012; Zafar et al., 2017a).

However, most of the existing fairness notions only focus on immediate fairness, without taking
potential follow-up inequity risk into consideration. In Fig. 1, we provide an example scenario when
using DP fairness has a long-term fairness issue, in a simple loan approval problem setting. Consider
two groups (group 0 and group 1) with different distributions, where each individual has one label
(approve the loan or not) and two features (credit score, income) that can be improved over time.
Suppose each group consists of two clusters (with three samples each), and the distance between the
clusters is different for two groups. Fig. 1 visualizes the distributions of two groups and the decision
boundary of a classifier f which achieves DP among the groups. We observe that the rejected samples
(left-hand-side of the decision boundary) in group 1 are located further away from the decision
boundary than the rejected samples in group 0. As a result, the rejected applicants in group 1 need to
make more effort to cross the decision boundary and get approval. This improvability gap between
the two groups can make the rejected applicants in group 1 less motivated to improve their features,
which may increase the gap between different groups in the future.

∗Equal Contribution. Emails: ozgurguldogan@ucsb.edu, yzeng58@wisc.edu.
†Work done at the University of Wisconsin-Madison.
1https://github.com/guldoganozgur/ei_fairness

1

Published as a conference paper at ICLR 2023

This motivated the advent of fairness notions
that consider dynamic scenarios when each
rejected sample makes effort to improve its
feature, and measure the group fairness after
such effort is made Gupta et al. (2019); Hei-
dari et al. (2019); Von Kügelgen et al. (2022).
However, as shown in Table 1, they have vari-
ous limitations e.g., vulnerable to imbalanced
group negative rates or outliers.

Figure 1: Toy example showing the insufficiency
of fairness notion that does not consider improv-
ability. We consider the binary classification (ac-
cept/reject) on 12 samples (dots), where x is the fea-
ture of the sample and the color of the dot represents
the group. The given classifier f is fair in terms of a
popular notion called demographic parity (DP), but
does not have equal improvability of rejected samples
(f (x) < 0.5) in two groups; the rejected samples in
group 1 needs more effort ∆x to be accepted, i.e.,
f (x + ∆x)
0.5, compared with the rejected sam-
≥
ples in group 0.

In this paper, we introduce another fair-
ness notion designed for dynamic scenarios,
dubbed as Equal Improvability (EI), which
does not suffer from these limitations. Let x
be the feature of a sample and f be a score-
based classifier, e.g., predicting a sample as
accepted if f (x)
0.5 holds and as rejected
otherwise. We assume each rejected individ-
ual wants to get accepted in the future, thus
improving its feature within a certain effort budget towards the direction that maximizes its score
f (x). Under this setting, we define EI fairness as the equity of the potential acceptance rate of the
different rejected groups, once each individual makes the best effort within the predefined budget.
This prevents the risk of exacerbating the gap between different groups in the long run.

≥

Our key contributions are as follows:

• We propose a new group fairness notion called Equal Improvability (EI), which aims to
equalize the probability of rejected samples being qualified after a certain amount of feature
improvement, for different groups. EI encourages rejected individuals in different groups to
have an equal amount of motivation to improve their feature to get accepted in the future. We
analyze the properties of EI and the connections of EI with other existing fairness notions.
• We provide three methods to find a classifier that is fair in terms of EI, each of which uses
a unique way of measuring the inequity in the improvability. Each method is solving a
min-max problem where the inner maximization problem is finding the best effort to measure
the EI unfairness, and the outer minimization problem is finding the classifier that has the
smallest fairness-regularized loss. Experiments on synthetic/real datasets demonstrate that
our algorithms find classifiers having low EI unfairness.

• We run experiments on dynamic scenarios where the data and the classifier evolve over
multiple rounds, and show that training a classifier with EI constraints is beneficial for
making the feature distributions of different groups identical in the long run.

2 EQUAL IMPROVABILITY

1

{

∈

∈

∈

=

∈ Y

∈ Z

∈ X ⊆

0, . . . , n

Rd and a label y

RdM, and immutable features xIM

Before defining our new fairness notion called Equal Improvability (EI), we first introduce necessary
notations. For an integer n, let [n] =
. We consider a binary classification setting
}
−
where each data sample has an input feature vector x
0, 1
. In
}
{
= [Z], where Z is the number of sensitive groups.
particular, we have a sensitive attribute z
Rd into three categories: improvable
As suggested by Chen et al. (2021), we sort d features x
RdIM , where
RdI, manipulable features xM
features xI
dI + dM + dIM = d holds. Here, improvable features xI refer to the features that can be improved and
can directly affect the outcome, e.g., salary in the credit lending problem, and GPA in the school’s
admission problem. In contrast, manipulable features xM can be altered, but are not directly related
to the outcome, e.g., marital status in the admission problem, and communication type in the credit
lending problem. Although individuals may manipulate these manipulable features to get the desired
outcome, we do not consider it as a way to make efforts as it does not affect the individual’s true
qualification status. Immutable features xIM are features that cannot be altered, such as race, age,
or date of birth. Note that if sensitive attribute z is included in the feature vector, then it belongs to
f :
immutable features. For ease of notation, we write x = (xI, xM, xIM). Let
[0, 1]
X →
{
be the set of classifiers, where each classifier is parameterized by w, i.e., f = fw. Given f
we consider the following deterministic prediction: ˆy
A
where 1
{
= 0 otherwise. We now introduce our new fairness notion.
A
condition A holds, and 1
{

}
,
∈ F
= 1 if

F
0.5
}

f (x)
{

x = 1

=

≥

∈

}

}

|

2

3Decision Boundary of Classifier  f{x:f(x)≥0.5}Hard to Cross  the BoundaryΔxGroup 0Group 1Published as a conference paper at ICLR 2023

Figure 2: Visualization of EI fairness. For binary classification on 12 samples (dots) in two groups
(red/blue), we visualize the fairness notion defined in this paper: (a) shows the original definition
in Def. 2.1, and (b) shows an equivalent definition in Prop. 2.2. Here we assume two-dimensional
features (both are improvable) and L∞ norm for µ(x) =
∞. The classifier fEI achieves equal
∥
improvability (EI) since the same portion (1 out of 3) of unqualified samples in each group can be
improved to qualified samples.
Definition 2.1 (Equal Improvability). Define a norm µ : RdI
a classifier f is said to achieve equal improvability with δ-effort if

). For a given constant δ > 0,

x
∥

[0,

∞

→

(cid:18)

P

max
µ(∆xI)≤δ

f (x + ∆x)

0.5

|

≥

f (x) < 0.5, z = z

(cid:19)

(cid:18)

= P

max
µ(∆xI)≤δ

f (x + ∆x)

0.5

|

≥

f (x) < 0.5

(cid:19)

∈ Z

, where ∆xI is the effort for improvable features and ∆x = (∆xI, 0, 0).

holds for all z
0.5
Note that the condition f (x) < 0.5 represents that an individual is unqualified, and f (x + ∆x)
implies that the effort ∆x allows the individual to become qualified. The above definition of fairness
in equal improvability requires that unqualified individuals from different groups z
[Z] are equally
∈
likely to become qualified if appropriate effort is made. Note that µ can be defined on a case-by-case
RdI×dI is a cost matrix that is
basis. For example, we can use
diagonal and positive definite. Here, the diagonal terms of C characterize how difficult to improve
each feature. For instance, consider the graduate school admission problem where xI contains features
“number of publications” and “GPA”. Since publishing more papers is harder than raising the GPA, the
corresponding diagonal term for the number of publications feature in C should be greater than that
for the GPA feature. The constant δ in Definition 2.1 can be selected depending on the classification
task and the features. Appendix B.1 contains the interpretation of each term in Def. 2.1. We introduce
an equivalent definition of EI fairness below.
Proposition 2.2. The EI fairness notion defined in Def. 2.1 has an equivalent format: a classifier f
achieves equal improvability with δ-effort if and only if

⊤CxI, where C

xI
∥

(cid:112)

xI

=

≥

∈

∥

P(x

∈ X

x

imp
− |
− =

∈ X

−, z = z) = P(x

imp
− |

x

∈ X

−)

∈ X

X

X

where

x : f (x) < 0.5
}
{

holds for all z
imp
and
− =
unqualified samples that can be improved to qualified samples by adding ∆x satisfying µ(∆xI )

is the set of features x for unqualified samples,
is the set of features x for
δ.

∈ Z
x : f (x) < 0.5, maxµ(∆xI )≤δ f (x + ∆x)
{

≤
imp
The proof of this proposition is trivial from the definition of
− |
x
−) in the above equation indicates the probability that unqualified samples can be improved to
qualified samples by changing the features within budget δ. This is how we define the “improvability”
of unqualified samples, and the EI fairness notion is equalizing this improvability for all groups.

imp
− . Note that P(x

0.5
}

− and

∈ X

∈ X

≥

X

X

≥

≥

∈ {

red, blue

Visualization of EI. Fig. 2 shows the geometric interpretation of EI fairness notion in Def. 2.1 and
Prop. 2.2, for a simple two-dimensional dataset having 12 samples in two groups z
.
}
Consider a linear classifier fEI shown in the figure, where the samples at the right-hand-side of the
decision boundary is classified as qualified samples (fEI(x)
0.5). In Fig. 2a, we have L∞ ball at
each unqualified sample, representing that these samples have a chance to improve their feature in a
way that the improved feature x + ∆x allows the sample to be classified as qualified, i.e., fEI(x +
0.5. One can confirm that P (cid:0)maxµ(∆xI)≤δ f (x + ∆x)
f (x) < 0.5, z = z(cid:1) = 1
∆x)
3
≥
holds for each group z
, thus satisfying equal improvability according to Def. 2.1. In
}
Fig. 2b, we check this in an alternative way by using the EI fairness definition in Prop. 2.2. Here,
instead of making a set of improved features at each sample, we partition the feature domain
into three parts: (i) the features for qualified samples
X
imp
− =
for unqualified samples that can be improved
∆x)
0.5, maxµ(∆xI)≤δ f (x + ∆x) < 0.5
is shown as the yellow region. From Prop. 2.2, EI fairness means that
identical at each group z

X
0.5
, (ii) the features
}
x : fEI(x) < 0.5, maxµ(∆xI)≤δ f (x +
x : fEI(x) <
{
. In the figure, (ii) is represented as the green region and (iii)
}
# samples in (ii)
# samples in (ii) + # samples in (iii) is

X
0.5
, and (iii) the features for unqualified samples that cannot be improved
}

x : fEI(x)
{

red, blue

red, blue

+ =

∈ {

0.5

≥

≥

{

|

, which is true for the example in Fig. 2b.
}

∈ {

3

7Classifier  achieving EIfEIδGroup 0Group 1(a) Original definition(b) Alternative definitionfEIfEI≥0.5{x:fEI(x)<0.5,maxμ(ΔxI)≤δf(x+Δx)≥0.5}{x:fEI(x)<0.5,maxμ(ΔxI)≤δf(x+Δx)<0.5}δfEI<0.5Published as a conference paper at ICLR 2023

Name of fairness

Table 1: Comparison of our EI fairness with existing fairness notions.
Consider
efforts?

Definition

Limitations

Equal Improvability (Ours)

Demographic Parity
Equal Opportunity (Hardt et al., 2016)
Equalized Odds (Hardt et al., 2016)
Bounded Effort
(Heidari et al., 2019)

f (x + ∆x)

0.5

f (x) < 0.5, z = z

P

(cid:18)

max
µ(∆xI)≤δ

0.5
0.5
0.5

P (f (x)
P (f (x)
P (f (x)
(cid:18)
P

≥
≥
≥
max
µ(∆xI)≤δ

|
|
|

|

≥
z = z) = P (f (x)
0.5)
y = 1, z = z) = P (f (x)
y = y, z = z) = P (f (x)

≥

≥
≥
0.5, f (x) < 0.5

f (x + ∆x)

≥

(cid:19)

(cid:18)

= P

max
µ(∆xI)≤δ

f (x + ∆x)

0.5

|

≥

f (x) < 0.5

(cid:19)

0.5
0.5

y = 1)
y = y)
(cid:19)
= P

|
|
z = z

|

(cid:18)

(cid:19)

max
µ(∆xI)≤δ

f (x + ∆x)

≥

0.5, f (x) < 0.5

Equal Recourse
(Gupta et al., 2019)

E

(cid:20)

min
f (x+∆x)≥0.5

µ(∆x)

|

f (x) < 0.5, z = z

(cid:21)

= E

(cid:20)

(cid:21)

min
f (x+∆x)≥0.5

µ(∆x)

|

f (x) < 0.5

Individual-Level Fair Causal Recourse
(Von Kügelgen et al., 2022)

min
f (x′)≥0.5

µ(x′

−

x(z, u)) = min

f (x′)≥0.5

µ(x′

−

x(z′, u)) for all latent variable u, all groups z, z′

Yes

No
No
No

Yes

Yes

Yes

-

-
-
-

handle
Cannot
imbalanced
group
negative rates (Ap-
pendix C.4.2)

to out-
Vulnerable
liers (Appendix B.3
and C.4.1)

Limitations of coun-
terfactual fairness

Comparison of EI with other fairness notions. The suggested fairness notion equal improvability
(EI) is in stark difference with existing popular fairness notions that do not consider dynamics,
i.e., demographic parity (DP), equal opportunity (EO) (Hardt et al., 2016), and equalized odds
(EOD) (Hardt et al., 2016), which can be “myopic” and focuses only on achieving classification
fairness in the current status. Our notion instead, aims at achieving classification fairness in the long
run when each sample improves its feature over time.

On the other hand, EI also has differences with existing fairness notions that capture the dynamics
of samples (Heidari et al., 2019; Huang et al., 2019; Gupta et al., 2019; Von Kügelgen et al., 2022).
Table 1 compares our fairness notion with the related existing notions. In particular, Bounded Effort
(BE) fairness proposed by Heidari et al. (2019) equalizes “the available reward after each individual
making a bounded effort” for different groups, which is very similar to EI when we set a proper reward
function (see Appendix B.2). To be more specific, the BE fairness can be represented as in Table 1.
Comparing this BE expression with EI in Definition 2.1, one can confirm the difference: the inequality
f (x) < 0.5 is located at the conditional part for EI, which is not true for BE. EI and BE are identical
if the negative prediction rates are equal across the groups, but in general, they are different. The
condition f (x) < 0.5 here is very important since only looking into the unqualified members makes
more sense when we consider improvability. More importantly, the BE definition is based on reward
functions and we are presenting BE in a form that is closest to our EI fairness expression. Besides,
Equal Recourse (ER) fairness proposed by Gupta et al. (2019) suggests equalizing the average effort
of different groups without limiting the amount of effort that each sample can make. Note that ER is
vulnerable to outliers. For example, when we have an unqualified outlier sample that is located far
way from the decision boundary, ER disparity will be dominated by this outlier and fail to reflect
the overall unfairness. Von Kügelgen et al. (2022) suggested Individual-Level Fair Causal Recourse
(ILFCR), a fairness notion that considers a more general setting that allows causal influence between
the features. This notion aims to equalize the cost of recourse required for a rejected individual
to obtain an improved outcome if the individual is from a different group. Individual-level equal
recourse shares a similar spirit with EI since both of them are taking care of equalizing the potential
to improve the decision outcome for the rejected samples. However, introducing individual-level
fairness with respect to different groups inherently requires counterfactual fairness, which has its own
limitation, as described in Wu et al. (2019), and it is also vulnerable to outliers. Huang et al. (2019)
proposed a causal-based fairness notion to equalize the minimum level of effort such that the expected
prediction score of the groups is equal to each other. Note that, their definition is specific to causal
settings and it considers the whole sensitive groups instead of the rejected samples of the sensitive
groups. In addition to fairness notions, we also discuss other related works such as fairness-aware
algorithms in Sec. 5.
Compatibility of EI with other fairness notions. Here we prove the compatibility of three fairness
notions (EI, DP, and BE), under two mild assumptions. Assumption 2.3 ensures that EI is well-defined,
while Assumption 2.4 implies that the norm µ and the effort budget δ are chosen such that we have
nonzero probability that unqualified individuals can become qualified after making efforts.
Assumption 2.3. For any classifier f , the probability of unqualified samples for each demographic
group is not equal to 0, i.e., P (f (x) < 0.5, z = z)
Assumption 2.4. For any classifier f , the probability of being qualified after the effort for unqualified
samples is not equal to 0, i.e., P (cid:0)maxµ(∆xI)≤δ f (x + ∆x)
Under these assumptions, the following theorem reveals the relationship between DP, EI and BE.
Theorem 2.5. If a classifier f achieves two of the following three fairness notions, DP, EI, and BE;
then it has to achieve the remaining fairness notion as well.

0.5, f (x) < 0.5(cid:1)

= 0 for all z

= 0.

∈ Z

≥

.

The proof of the Theorem 2.5 is provided in Appendix A.1. This theorem immediately implies the
following corollary, which provides a condition such that EI and BE conflict with each other.

4

̸
̸
Published as a conference paper at ICLR 2023

Corollary 2.6. The above theorem says that if a classifier f achieves EI and BE, it has to achieve DP.
Thus, by contraposition, if f does not achieve DP, then it cannot achieve EI and BE simultaneously.

Besides, we also investigate the connections between EI and ER in Appendix A.2.
3 ACHIEVING EQUAL IMPROVABILITY

In this section, we discuss methods for finding a classifier that achieves EI fairness. Following existing
in-processing techniques (Zafar et al., 2017c; Donini et al., 2018; Zafar et al., 2017a; Cho et al.,
2020), we focus on finding a fair classifier by solving a fairness-regularized optimization problem.
To be specific, we first derive a differentiable penalty term Uδ that approximates the unfairness with
respect to EI, and then solve a regularized empirical minimization problem having the unfairness as
the regularization term. This optimization problem can be represented as

(cid:40)

(1

min
f ∈F

λ)

N
(cid:88)

i=1

−
N

(cid:41)

ℓ(yi, f (xi)) + λUδ

,

(1)

(xi, yi)
}
{

N
i=1 is the given dataset, ℓ :
∈

0, 1
is the set
where
{
of classifiers we are searching over, and λ
[0, 1) is a hyperparameter that balances fairness and
prediction loss. Here we consider three different ways of defining the penalty term Uδ, which are (a)
covariance-based, (b) kernel density estimator (KDE)-based, and (c) loss-based methods. We first
introduce how we define Uδ in each method, and then discuss how we solve (1).

R is the loss function,

[0, 1]

} ×

→

F

Covariance-based EI penalty. Our first method is inspired by Zafar et al. (2017c), which measures
the unfairness of a score-based classifier f by the covariance of the sensitive attribute z and the score
f (x), when the demographic parity (DP) fairness condition (P(f (x) > 0.5
z = z) = P(f (x) > 0.5)
|
holds for all z) is considered. The intuition behind this idea of measuring the covariance is that
a perfect fair DP classifier should have zero correlation between z and f (x). By applying similar
approach to our fairness notion in Def. 2.1, the EI unfairness is measured by the covariance between
the sensitive attribute z and the maximally improved score of rejected samples within the effort budget.
f (x) < 0.5))2 represents the EI unfairness of
In other words, (Cov(z, max∥∆xI∥≤δ f (x + ∆x)
a classifier f where we took the square to penalize negative correlation case as well. Let I− =
i : f (xi) < 0.5
. Then, EI
{
|
}
unfairness can be approximated by the square of the empirical covariance, i.e.,

be the set of indices of unqualified samples, and ¯z = (cid:80)

zi/
|

i∈I−

I−

|

Uδ ≜





1
I−
|

(cid:88)

i∈I−

|



(zi

¯z)

 max

∥∆xI i∥≤δ

−

f (xi + ∆xi)

(cid:88)

−

j∈I−

∥

Since (cid:80)

i∈I−

we have Uδ =

(zi
−
(cid:16) 1
|I−|

(cid:16)(cid:80)

j∈I−

¯z)

(cid:80)

i∈I−

(zi

max
∥

∆xI j

≤δ f (xj + ∆xj)/
|
∥

I−

¯z) max∥∆xI i∥≤δ f (xi + ∆xi)

.

(cid:17)
|
(cid:17)2

−





2

I−




|

.

max
∆xI j

f (xj + ∆xj)/
|

≤δ
∥
= 0 from (cid:80)

i∈I−

(zi

−

¯z) = 0,

KDE-based EI penalty. The second method is inspired by Cho et al. (2020), which suggests to
first approximate the probability density function of the score f (x) via kernel density estimator
(KDE) and then put the estimated density formula into the probability term in the unfairness penalty.
Recall that given m samples y1, . . . , ym, the true density gy on y is estimated by KDE as ˆgy(ˆy) ≜

1
mh

(cid:80)m

i=1 gk

(cid:17)

(cid:16) ˆy−yi
h

, where gk is a kernel function and h is a smoothing parameter.

Here we apply this KDE-based method for estimating the EI penalty term in Def. 2.1. Let ymax
i =
max∥∆xI i∥≤δ f (xi + ∆xi) be the maximum score achievable by improving feature xi within budget
δ, and I−,z =
be the set of indices of unqualified samples of group z. Then,
the density of ymax

for the unqualified samples in group z can be approximated as2

i : f (xi) < 0.5, zi = z
{

}

i

ˆgymax|f (x)<0.5,z(ˆymax) ≜

1
I−,z
|

(cid:18) ˆymax

(cid:88)

gk

(cid:19)

ymax
i

.

−
h

h
|
Then, the estimate on the left-hand-side (LHS) probability term in Def. 2.1 is represented as
ˆP (cid:0)maxµ(∆xI)≤δ f (x + ∆x)
0.5 ˆgymax|f (x)<0.5,z(ˆymax)dˆymax =

f (x) < 0.5, z = z(cid:1) = (cid:82) ∞

i∈I−,z

0.5

≥

|

2This term is differentiable with respect to model parameters, since gk is differentiable w.r.t ymax

ymax
i = max∥∆xI i∥≤δ f (xi + ∆xi) is differentiable w.r.t. model parameters from (Danskin, 1967).

i

, and

5

Published as a conference paper at ICLR 2023

(cid:80)

1
τ gk(y)dy. Similarly, we can estimate the right-
|I−,z|h
hand-side (RHS) probability term in Def. 2.1, and the EI-penalty Uδ is computed as the summation
of the absolute difference of the two probability values (LHS and RHS) among all groups z.

where Gk(τ ) ≜ (cid:82) ∞

i∈I−,z

Gk

(cid:16) 0.5−ymax
h

i

(cid:17)

(cid:80)

Loss-based EI penalty. Another common way of approximating the fairness violation as a dif-
ferentiable term is to compute the absolute difference of group-specific losses (Roh et al., 2021;
Shen et al., 2022). Following the spirit of EI notion in Def. 2.1, we define EI loss of group z as
ℓ (cid:0)1, max∥∆xI i∥≤δ f (xi + ∆xi)(cid:1). Here, ˜Lz measures how far the rejected
˜Lz ≜ 1
samples in group z are away from being accepted after the feature improvement within budget δ.
Similarly, EI loss for all groups is written as ˜L ≜ (cid:80)
˜Lz. Finally, the EI penalty term is
defined as Uδ ≜ (cid:80)
˜Lz

I−,z
I−

i∈I−,z

|I−,z|

z∈Z

(cid:12)
(cid:12)
(cid:12)

z∈Z

(cid:12)
˜L
(cid:12)
(cid:12).

−

Solving (1). For each approach defined above (covariance-based, KDE-based and loss-based), the
penalty term Uδ is defined uniquely. Note that in all cases, we need to solve a maximization problem
max∥∆xI∥≤δ f (x + ∆x) in order to get Uδ. Since (1) is a minimization problem containing Uδ in
the cost function, it is essentially a minimax problem. We leverage adversarial training techniques
to solve (1). The inner maximization problem is solved using one of two methods: (i) derive the
closed-form solution for generalized linear models, (ii) use projected gradient descent for general
settings. The details can be found in Appendix B.4.

4 EXPERIMENTS

−

0.5

P(maxµ(∆xI)<δ f (x + ∆x)

P(maxµ(∆xI)<δ f (x + ∆x)

This section presents the empirical results of our EI fairness notion. To measure the fairness
f (x) < 0.5, z =
violation, we use EI disparity= maxz∈[Z] |
0.5
z)
. First, in Sec. 4.1, we show that our methods
|
suggested in Sec. 3 achieve EI fairness in various real/synthetic datasets. Second, in Sec. 4.2,
focusing on the dynamic scenario where each individual can make effort to improve its outcome, we
demonstrate that training an EI classifier at each time step promotes achieving the long-term fairness,
i.e., the feature distribution of two groups become identical in the long run. In Appendix C.4, we put
additional experimental results comparing the robustness of EI and related notions including ER and
BE. Specifically, we compare (i) the outlier-robustness of EI and ER, and (ii) the robustness of EI and
BE to imbalanced group negative rates.

f (x) < 0.5)

≥

≥

|

|

4.1 SUGGESTED METHODS ACHIEVE EI FAIRNESS

Recall that Sec. 3 provided three approaches for achieving EI fairness. Here we check whether such
methods successfully find a classifier with a small EI disparity, compared with ERM which does not
have fairness constraints. We train all algorithms using logistic regression (LR) in this experiment.
Due to the page limit, we defer the presentation of results for (1) a two-layer ReLU neural network
with four hidden neurons and (2) a five-layer ReLU network with 200 hidden neurons per layer to
Appendix C.3, and provide more details of hyperparameter selection in Appendix C.2.

Experiment setting. For all experiments, we use the Adam optimizer and cross-entropy loss. We
perform cross-validation on the training set to find the best hyperparameter. We provide statistics for
five trials having different random seeds. For the KDE-based approach, we use the Gaussian kernel.

Datasets. We perform the experiments on one synthetic dataset, and two real datasets: German
Statlog Credit (Dua & Graff, 2017), and ACSIncome-CA (Ding et al., 2021). The synthetic dataset
has two non-sensitive attributes x = (x1, x2), one binary sensitive attribute z, and a binary label y.
Both features x1 and x2 are assumed to be improvable. We generate 20,000 samples where (x, y, z)
pair of each sample is generated independently as below. We define z and (y
z = z) as Bernoulli
|
y = y, z = z) as multivariate Gaussian random
random variables for all z
0, 1
∈ {
}
variables for all y, z
. The numerical details are in Appendix C.1 The maximum effort δ for
}
this dataset is set to 0.5. The ratio of the training versus test data is 4:1.

, and define (x

∈ {

0, 1

|

German Statlog Credit dataset contains 1,000 samples and the ratio of the training versus test data is
4:1. The task is to predict the credit risk of an individual given its financial status. Following Jiang &
Nachum (2020), we divide the samples into two groups using the age of thirty as the boundary, i.e.,
z = 1 for samples with age above thirty. Four features x are considered as improvable: checking
account, saving account, housing and occupation, all of which are ordered categorical
features. For example, the occupation feature has four levels: (1) unemployed, (2) unskilled, (3)

6

Published as a conference paper at ICLR 2023

Table 2: Error rate and EI disparities of ERM and three proposed EI-regularized methods
on logistic regression (LR). For each dataset, the lowest EI disparity (disp.) value is in boldface.
Classifiers obtained by our three methods have much smaller EI disparity values than the ERM
solution, without having much additional error.

METHODS

DATASET

METRIC

ERM

COVARIANCE-BASED

KDE-BASED

LOSS-BASED

SYNTHETIC

GERMAN STAT.

ACSINCOME-CA

ERROR RATE(↓)
EI DISP.(↓)

.221 ± .001
.117 ± .007

ERROR RATE(↓)
EI DISP.(↓)

.220 ± .009
.041 ± .008

ERROR RATE(↓)
EI DISP.(↓)

.184 ± .000
.031 ± .001

.253 ± .003
.003 ± .001

.262 ± .009
.021 ± .019

.200 ± .000
.008 ± .001

.250 ± .001
.005 ± .003

.246 ± .001
.002 ± .001

.243 ± .024
.035 ± .026

.237 ± .008
.015 ± .009

.196 ± .000
.005 ± .001

.193 ± .000
.006 ± .001

skilled, and (4) highly qualified. We set the maximum effort δ = 1, meaning that an unskilled man
(with level 2) can become a skilled man, but cannot be a highly qualified man.

ACSIncome-CA dataset consists of data for 195,665 people and is split into training/test sets in the
ratio of 4:1. The task is predicting whether a person’s income would exceed 50K USD per year. We
use sex as the sensitive attribute; we have two sensitive groups, male and female. We select education
level (ordered categorical feature) as the improvable feature. We set the maximum effort δ = 3.

Figure 3: Tradeoff between EI disparity and error rate.
We run three EI-regularized methods suggested in Sec. 3
for different regularizer coefficient λ and plot the frontier
lines. For the synthetic dataset, the tradeoff curve for the
ideal classifier is located at the bottom left corner, which is
similar to the curves of proposed EI-regularized methods.
This shows that our methods successfully find classifiers
balancing EI disparity and error rate.

Results. Table 2 shows the test er-
ror rate and test EI disparity (disp.)
for ERM and our three EI-regularized
methods (covariance-based, KDE-
based, and loss-based) suggested in
Sec. 3. For all three datasets, our EI-
regularized methods successfully re-
duce the EI disparity without increas-
ing the error rate too much, compared
with ERM. Figure 3 shows the trade-
off between the error rate and EI dis-
parity of our EI-regularized methods.
We marked the dots after running each
method multiple times with different
penalty coefficients λ, and plotted the
frontier line. For the synthetic dataset with the Gaussian feature, we numerically obtained the per-
formance of the optimal EI classifier, which is added in the yellow line at the bottom left corner
of the first column plot. The details of finding the optimal EI classifier is in Appendix B.5. One
can confirm that our methods of regularizing EI are having similar tradeoff curves for the synthetic
dataset. Especially, for the synthetic dataset, the tradeoff curve of our methods nearly achieves that of
the optimal EI classifier. For German and ACSIncome-CA datasets, the loss-based method is having
a slightly better tradeoff curve than other methods.
4.2 EI PROMOTES LONG-TERM FAIRNESS IN DYNAMIC SCENARIOS
Here we provide simulation results on the dynamic setting, showing that EI classifier encourages the
long-term fairness, i.e., equalizes the feature distribution of different groups in the long run 3.
4.2.1 DYNAMIC SYSTEM DESCRIPTION
We consider a binary classification problem under the dynamic scenario with T rounds, where the
of each sample as well as the classifier ft evolve
improvable feature xt
0, 1
}
∈
0,
at each round t
0, 1
, and the estimated
}
· · ·
σ(z)
2). To mimic
label as ˆyt. We assume z
,
t }
{
(0, 1) of the population, i.e., the true label is
the admission problem, we only accept a fraction α
, where χ(t)
α) percentile of the feature distribution at
modeled as yt = 1
}
round t. We consider z-aware linear classifier outputting ˆyt = 1
, which is parameterized
{
by the thresholds (τ (0)
) for two groups. Note that this classification rule is equivalent to defining
t
score function ft(x, z) = 1/(exp(τ (z)
Updating data parameters (µ(z)
its feature from x to x + ϵ(x). Here we model ϵ(x) = ν(x; z) =

x) + 1) and ˆyt = 1
{
). At each round t, we allow each sample can improve
x < τ (z)
1
t }
{
3Appendix D provides analysis on the effect of fairness notions on the long-term fairness when a single
step of feature improvement is applied, for toy examples. Our results show that EI better equalizes the feature
distribution (compared with other fairness notions), which coincides with our empirical results in Sec. 4.2.

. We denote the sensitive attribute as z
}

R and the label yt
1
, T

−
Bern(0.5) and (xt

∈
α is the (1

1
t −x+β)2

t −
, σ(z)
t

∼
χ(t)
α

τ (z)
t }

∈ {
(µ(z)
t

f (xt, z)

(z)
t =

0.5
}

, τ (1)
t

z = z)

xt
{

∼ P

(τ (z)

∈ {

∈ {

xt

N

−

≥

≥

≥

|

.

t

7

0.000.050.10EIDisparity0.220.24ErrorRateSyntheticCovariance-basedKDE-basedLoss-BasedOptimal0.000.05EIDisparity0.200.250.30German0.000.02EIDisparity0.180.19ACSIncome-CAPublished as a conference paper at ICLR 2023

(0)
t

(1)
t

,

P

P

Figure 4: Long-term unfairness dTV(
) at each round t for various algorithms. Con-
sider the binary classification problem over two groups, under the dynamic scenario where the data
distribution and the classifier for each group evolve over multiple rounds. We plot how the long-term
unfairness (measured by the total variation distance between two groups) changes as round t increases.
Here, each column shows the result for different initial feature distributions, details of which are given
in Sec. 4.2.2. The long-term unfairness of EI classifier reduces faster than other existing fairness
notions, showing that EI proposed in this paper is helpful for achieving long-term fairness.
for a constant β > 0. In this model, the rejected samples with larger gap (τ (z)
x) with the
decision boundary are making less effort ∆x, which is inspired by the intuition that a rejected
sample is less motivated to improve its feature if it needs to take a large amount of effort to get
) selection is provided in Appendix C.5.
accepted in one scoop. More detailed justification of the ϵ(
·
After such effort is made, we compute the mean and standard deviation of each group: µ(z)
t+1 =
(cid:82) ∞
, σ(z)
−∞(x + ν(x; z))ϕ(x; µ(z)
)dx,
t
(µ, σ2). We assume that the feature xt+1 in the next round follows a
; µ, σ) is the pdf of
where ϕ(
·
t+1, σ(z)
Gaussian distribution parameterized by (µ(z)
Updating classifier parameters (τ (0)
, τ (1)
). At each round t, we update the classifier depending
t
, τ (1)
on the current feature distribution xt. The EI classifier considered in this paper updates (τ (0)
)
t
as below. Note that the maximized score max∥∆xI∥≤δ f (x + ∆x) in Def. 2.1 can be written as
0.5 is equivalent to xt + δt > τ (z)
ft(xt + δt, z), and the equation max∥∆xI∥≤δ f (x + ∆x)
.
Consequently, EI classifier obtains (τ (0)
) by solving

t+1) for ease of simulation.

t+1)2ϕ(x; µ(z)
µ(z)

−∞(x + ν(x; z)

)dx and σ(z)

(cid:113)(cid:82) ∞

, σ(z)
t

t+1 =

t −

N

−

≥

t

t

t

t

t

, τ (1)
t

t

min

τ (0)
t

,τ (1)
t

(cid:12)
P(xt +δt > τ (0)
(cid:12)
(cid:12)

t

|

z = 0,xt < τ (0)

t

)

P(xt +δt > τ (1)

t

−

z = 1,xt < τ (1)

t

|

(cid:12)
(cid:12) s.t. P(ˆyt ̸
(cid:12)
)

= yt)

c,

≤

∈
(cid:82) ∞
−∞ ν(x; z)ϕ(x; µ(z)

where c
[0, 1) is the maximum classification error rate we allow, and δt is the effort level at
iteration t. In our experiments, δt is chosen as the mean efforts the population makes, i.e., δt =
0.5 (cid:80)1
)dx. Meanwhile, we can similarly obtain the classifier for DP,
BE, ER, and ILFCR constraints, details of which are in Appendix C.6. In the experiments, we
numerically obtain the solution of this optimization problem.

, σ(z)
t

z=0

t

t

= σ(1)

0 ) or different variance (i.e., σ(0)

4.2.2 EXPERIMENTS ON LONG-TERM FAIRNESS
We first initialize the feature distribution in a way that both sensitive groups have either different mean
(i.e., µ(0)
= µ(1)
, T
0 ). At each round t
, we update
0
}
· · ·
, τ (1)
, σ(z)
and the classifier parameter (τ (0)
the data parameter (µ(z)
), follow-
0, 1
t
t
t
}
, we measure the long-term unfair-
ing the rule described in Sec. 4.2.1. At each round t
}
(1)) =
ness defined as the total variation distance between the two group distributions: dTV(
1
dx. We run experiments on four different initial feature dis-
2
0 , µ(1)
tributions: (i) (µ(0)
0 ) = (0, 0.5, 1, 1), (iii)
0 , σ(1)
0 , µ(1)
(µ(0)
0 ) = (0, 0.5, 1, 0.5), respectively.
We set α = 0.2, c = 0.1, β = 0.25.

, σ(0)
)
t
−
0 , µ(1)
0 , σ(0)
0 ) = (0, 2, 0, 1), and (iv) (µ(0)

0 ) = (0, 1, 1, 0.5), (ii) (µ(0)
0 , µ(1)

ϕ(x; µ(1)
0 , σ(1)

0 , σ(0)
0 , σ(1)

) for group z

(cid:82) ∞
−∞ |

0 , σ(0)

0 , σ(0)

0 , σ(1)

ϕ(x; µ(0)

, σ(1)
t

(0),

∈ {

∈ {

∈ {

)
|

· · ·

, T

1,

1,

P

P

0

t

t

Baselines. We compare our EI classifier with multiple baselines, including the empirical risk
minimization (ERM) and algorithms with fairness constraints: demographic parity (DP), bounded
effort (BE) (Heidari et al., 2019), equal recourse (ER) (Gupta et al., 2019), and individual-level
fair causal recourse (ILFCR) (Von Kügelgen et al., 2022). In particular, we assume a causal model
for solving the ILFCR classifier for this specific data distribution, which is described in detail in
Appendix C.6.

Results. Fig. 4 shows how the long-term unfairness dTV(
) changes as a function of
round t, for cases (i) – (iv) having different initial feature distribution. Note that except ILFCR, other
fairness notions (DP, BE, ER, and EI) yield lower long-term unfairness compared with ERM, for cases
(i), (ii), and (iv). More importantly, EI accelerates the process of mitigating long-term unfairness,

P

,

(0)
t
P

(1)
t

8

050.20.4(i)050.20.4(ii)050.00.20.4(iii)050.250.50(iv)ERMDPBEERILFCREI(Ours)RoundLong-termUnfairness̸
̸
Published as a conference paper at ICLR 2023

Figure 5: Evolution of the feature distribution, when we apply each algorithm for t = 3 rounds.
At each row, the leftmost column shows the initial distribution and the rest of the columns show the
evolved distribution for each algorithm, under the dynamic setting. Compared with existing fairness
notions (DP, BE, ER, and ILFCR), EI achieves a smaller feature distribution gap between groups.

compared to other fairness notions. This observation highlights the benefit of EI in promoting true
equity of groups in the long run. Fig. 5 visualizes the initial distribution (at the leftmost column) and
the evolved distribution at round t = 3 for multiple algorithms (at the rest of the columns). Each row
represents different initial feature distribution, for cases (i) – (iv). One can confirm that EI brings
the distribution of the two groups closer, compared with baselines. Moreover, in Appendix C.7, we
explore the long-term impact of fairness notions on a different dynamic model, where most existing
methods have an adverse effect on long-term fairness, while EI continues to enhance it.

5 RELATED WORKS
Fairness-aware algorithms. Most of the existing fair learning techniques fall into three categories:
i) pre-processing approaches (Kamiran & Calders, 2012; 2010; Gordaliza et al., 2019; Jiang &
Nachum, 2020), which primarily involves massaging the dataset to remove the bias; ii) in-processing
approaches (Fukuchi et al., 2013; Kamishima et al., 2012; Calders & Verwer, 2010; Zafar et al.,
2017c;a; Zhang et al., 2018; Cho et al., 2020; Roh et al., 2020; 2021; Shen et al., 2022), adjusting
the model training for fairness; iii) post-processing approaches (Calders & Verwer, 2010; Alghamdi
et al., 2020; Wei et al., 2020; Hardt et al., 2016) which achieve fairness by modifying a given unfair
classifier. Prior work (Woodworth et al., 2017) showed that the in-processing approach generally
outperforms other approaches due to its flexibility. Hence, we focus on the in-processing approach
and propose three methods to achieve EI. These methods achieve EI by solving fairness-regularized
optimization problems. In particular, our proposed fairness regularization terms are inspired by Zafar
et al. (2017c); Cho et al. (2020); Roh et al. (2021); Shen et al. (2022).

Fairness notions related with EI See Table 1 and the corresponding explanation given in Sec. 2.

Fairness dynamics. There are also a few attempts to study the long-term impact of different fairness
policies (Zhang et al., 2020; Heidari et al., 2019; Hashimoto et al., 2018). In particular, Hashimoto
et al. (2018) studies how ERM amplifies the unfairness of a classifier in the long run. The key idea
is that if the classifier of the previous iteration favors a certain candidate group, then the candidate
groups will be more unbalanced since fewer individuals from the unfavored group will apply for
this position. Thus, the negative feedback leads to a more unfair classifier. In contrast, Heidari et al.
(2019) and Zhang et al. (2020) focus more on long-term impact instead of classification fairness. To
be specific, Heidari et al. (2019) studies how fairness intervention affects the different groups in terms
of evenness, centralization, and clustering by simulating the population’s response through effort.
Zhang et al. (2020) investigates how different fairness policies affect the gap between the qualification
rates of different groups under a partially observed Markov decision process. Besides, there are a
few works which study how individuals may take strategic actions to improve their outcomes given a
classifier (Chen et al., 2021). However, Chen et al. (2021) aims to address this problem by designing
an optimization problem that is robust to strategic manipulation, which is orthogonal to our focus.
6 CONCLUSION
In this paper, we introduce Equal Improvability (EI), a group fairness notion that equalizes the
potential acceptance of rejected samples in different groups when appropriate effort is made by the
rejected samples. We analyze the properties of EI and provide three approaches to finding a classifier
that achieves EI. While our experiments demonstrate the effectiveness of the proposed approaches
in reducing EI disparity, the theoretical analysis of the approximated EI penalties remains open.
Moreover, we formulate a simplified dynamic model with one-dimensional features and a binary
sensitive attribute to showcase the benefits of EI in promoting equity in feature distribution across
different groups. We identify extending this model to settings with multiple sensitive attributes and
high-dimensional features as an interesting future direction.

9

0.00.5(i)dTV=0.55Initialdistribution0.00.1dTV=0.31ERMdTV=0.20DPdTV=0.20BEdTV=0.22ERdTV=0.29ILFCRdTV=0.20EIGroup0Group10.00.5(ii)dTV=0.550.00.1dTV=0.27dTV=0.15dTV=0.16dTV=0.13dTV=0.52dTV=0.090.00.5(iii)dTV=0.680.00.1dTV=0.38dTV=0.24dTV=0.28dTV=0.11dTV=0.30dTV=0.110200.000.25(iv)dTV=0.320200.00.1dTV=0.06020dTV=0.06020dTV=0.05020dTV=0.07020dTV=0.39020dTV=0.04xDensityPublished as a conference paper at ICLR 2023

ACKNOWLEDGEMENT

Yuchen Zeng was supported in part by NSF Award DMS-2023239. Ozgur Guldogan and Ramtin
Pedarsani were supported by NSF Award CNS-2003035, and NSF Award CCF-1909320. Kangwook
Lee was supported by NSF/Intel Partnership on Machine Learning for Wireless Networking Program
under Grant No. CNS-2003129 and by the Understanding and Reducing Inequalities Initiative of
the University of Wisconsin-Madison, Office of the Vice Chancellor for Research and Graduate
Education with funding from the Wisconsin Alumni Research Foundation. The authors would also
like to thank the anonymous reviewers and AC for their valuable suggestions.

REFERENCES

Wael Alghamdi, Shahab Asoodeh, Hao Wang, Flavio P. Calmon, Dennis Wei, and Karthikeyan Nate-
In
san Ramamurthy. Model projection: Theory and applications to fair machine learning.
2020 IEEE International Symposium on Information Theory (ISIT), pp. 2711–2716, 2020. doi:
10.1109/ISIT44484.2020.9173988.

Toon Calders and Sicco Verwer. Three naive bayes approaches for discrimination-free classification.
Data Mining and Knowledge Discovery, 21(2):277–292, 2010. doi: 10.1007/s10618-010-0190-x.
URL https://doi.org/10.1007/s10618-010-0190-x.

Yatong Chen, Jialu Wang, and Yang Liu. Linear classifiers that encourage constructive adaptation,

2021.

Valeriia Cherepanova, Vedant Nanda, Micah Goldblum, John P Dickerson, and Tom Goldstein.
Technical challenges for training fair neural networks. arXiv preprint arXiv:2102.06764, 2021.

Jaewoong Cho, Gyeongjo Hwang, and Changho Suh. A fair classifier using kernel density estimation.

Advances in Neural Information Processing Systems (NeurIPS), 33:15088–15099, 2020.

John Danskin. The theory of max-min and its application to weapons allocation problems. In

Springer, 1967.

Frances Ding, Moritz Hardt, John Miller, and Ludwig Schmidt. Retiring adult: New datasets for fair

machine learning. Advances in Neural Information Processing Systems, 34, 2021.

Michele Donini, Luca Oneto, Shai Ben-David, John Shawe-Taylor, and Massimiliano Pontil. Empiri-
cal risk minimization under fairness constraints. In Advances in Neural Information Processing
Systems (NeurIPS), volume 31, pp. 2796–2806, 2018.

Dheeru Dua and Casey Graff. UCI machine learning repository, 2017. URL http://archive.

ics.uci.edu/ml.

Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel. Fairness through
awareness. In Proceedings of the 3rd Innovations in Theoretical Computer Science Conference,
ITCS ’12, pp. 214–226, 2012. ISBN 9781450311151. doi: 10.1145/2090236.2090255. URL
https://doi.org/10.1145/2090236.2090255.

Kazuto Fukuchi, Jun Sakuma, and Toshihiro Kamishima. Prediction with model-based neutrality. In
Machine Learning and Knowledge Discovery in Databases, pp. 499–514, 2013. ISBN 978-3-642-
40991-2.

Paula Gordaliza, Eustasio Del Barrio, Gamboa Fabrice, and Jean-Michel Loubes. Obtaining fairness
using optimal transport theory. In Proceedings of the 36th International Conference on Machine
Learning (ICML), volume 97 of Proceedings of Machine Learning Research, pp. 2357–2365, 2019.
URL http://proceedings.mlr.press/v97/gordaliza19a.html.

Vivek Gupta, Pegah Nokhiz, Chitradeep Dutta Roy, and Suresh Venkatasubramanian. Equalizing

recourse across groups. arXiv preprint arXiv:1909.03166, 2019.

Moritz Hardt, Eric Price, and Nati Srebro. Equality of opportunity in supervised learning. In Advances

in Neural Information Processing Systems (NIPS), volume 29, pp. 3315–3323, 2016.

10

Published as a conference paper at ICLR 2023

Tatsunori Hashimoto, Megha Srivastava, Hongseok Namkoong, and Percy Liang. Fairness without
demographics in repeated loss minimization. In Proceedings of the 35th International Conference
on Machine Learning (ICML), volume 80 of Proceedings of Machine Learning Research, pp. 1929–
1938, 2018. URL http://proceedings.mlr.press/v80/hashimoto18a.html.

Hoda Heidari, Vedant Nanda, and Krishna P. Gummadi. On the long-term impact of algorithmic

decision policies: Effort unfairness and feature segregation through social learning, 2019.

Wen Huang, Yongkai Wu, Lu Zhang, and Xintao Wu. Fairness through equality of effort, 2019.

Heinrich Jiang and Ofir Nachum. Identifying and correcting label bias in machine learning. In
Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics,
volume 108 of Proceedings of Machine Learning Research, pp. 702–712, 2020. URL http:
//proceedings.mlr.press/v108/jiang20a.html.

F. Kamiran and T.G.K. Calders. Classification with no discrimination by preferential sampling.
In Informal proceedings of the 19th Annual Machine Learning Conference of Belgium and The
Netherlands, pp. 1–6, 2010.

Faisal Kamiran and Toon Calders. Data preprocessing techniques for classification without discrimi-

nation. Knowledge and Information Systems, 33:1–33, 2012.

Toshihiro Kamishima, Shotaro Akaho, Hideki Asoh, and Jun Sakuma. Fairness-aware classifier with
prejudice remover regularizer. In Machine Learning and Knowledge Discovery in Databases, pp.
35–50, 2012. ISBN 978-3-642-33486-3.

Dragoslav S Mitrinovic and Petar M Vasic. Analytic inequalities, volume 1. Springer, 1970.

Yuji Roh, Kangwook Lee, Steven Whang, and Changho Suh. FR-train: A mutual information-
based approach to fair and robust training. In Proceedings of the 37th International Conference
on Machine Learning (ICML), volume 119 of Proceedings of Machine Learning Research, pp.
8147–8157, 2020. URL http://proceedings.mlr.press/v119/roh20a.html.

Yuji Roh, Kangwook Lee, Steven Euijong Whang, and Changho Suh. Fairbatch: Batch selection
for model fairness. In International Conference on Learning Representations (ICLR), 2021. URL
https://openreview.net/forum?id=YNnpaAKeCfx.

Aili Shen, Xudong Han, Trevor Cohn, Timothy Baldwin, and Lea Frermann. Optimising equal

opportunity fairness in model training. arXiv preprint arXiv:2205.02393, 2022.

Julius Von Kügelgen, Amir-Hossein Karimi, Umang Bhatt, Isabel Valera, Adrian Weller, and Bernhard
Schölkopf. On the fairness of causal algorithmic recourse. In Proceedings of the AAAI Conference
on Artificial Intelligence, volume 36, pp. 9584–9594, 2022.

Dennis Wei, Karthikeyan Natesan Ramamurthy, and Flavio Calmon. Optimized score transformation
for fair classification. In International Conference on Artificial Intelligence and Statistics (AIS-
TATS), volume 108 of Proceedings of Machine Learning Research, pp. 1673–1683, 2020. URL
http://proceedings.mlr.press/v108/wei20a.html.

Blake Woodworth, Suriya Gunasekar, Mesrob I. Ohannessian, and Nathan Srebro. Learning
In Proceedings of the 2017 Conference on Learning The-
non-discriminatory predictors.
ory, volume 65 of Proceedings of Machine Learning Research, pp. 1920–1953, 2017. URL
http://proceedings.mlr.press/v65/woodworth17a.html.

Yongkai Wu, Lu Zhang, and Xintao Wu. Counterfactual fairness: Unidentification, bound and
In Proceedings of the Twenty-Eighth International Joint Conference on Artificial

algorithm.
Intelligence, 2019.

Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, and Krishna P. Gummadi. Fairness
beyond disparate treatment & disparate impact: Learning classification without disparate mistreat-
ment. In Proceedings of the 26th International Conference on World Wide Web, pp. 1171–1180,
2017a.

11

Published as a conference paper at ICLR 2023

Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, Krishna P. Gummadi, and Adrian
Weller. From parity to preference-based notions of fairness in classification. In Advances in Neural
Information Processing Systems (NIPS), NIPS’17, pp. 228–238, 2017b. ISBN 9781510860964.

Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rogriguez, and Krishna P. Gummadi. Fair-
ness Constraints: Mechanisms for Fair Classification. In International Conference on Artificial
Intelligence and Statistics (AISTATS), volume 54, pp. 962–970, 2017c.

Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell. Mitigating unwanted biases with adversarial
In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, pp.

learning.
335–340, 2018.

Xueru Zhang, Ruibo Tu, Yang Liu, Mingyan Liu, Hedvig Kjellström, Kun Zhang, and Cheng Zhang.

How do fair decisions fare in long-term qualification?, 2020.

12

Published as a conference paper at ICLR 2023

A THEORETICAL RESULTS

A.1 CONNECTIONS BETWEEN EI, DP, AND BE

In this section, we provide the proof of Theorem 2.5 and Corollary 2.6.

Proof of Theorem 2.5. All we need to prove are three statements:

1. Prove that EI and BE imply DP

2. Prove that DP and EI imply BE

3. Prove that BE and DP imply EI

Below we prove each statement.

1. EI, BE

(cid:18)

P

⇒

DP Suppose a classifier f achieves EI and BE. Recall that a classifier achieves EI if
(cid:19)
(cid:18)

(cid:19)

max
µ(∆xI)≤δ

f (x + ∆x)

0.5

|

≥

f (x) < 0.5, z = z

= P

and a classifier achieves BE if

P

(cid:18)

max
µ(∆xI)≤δ

f (x + ∆x)

≥

0.5, f (x) < 0.5

(cid:19)

z = z

|

(cid:18)

= P

By dividing both sides of 3 by the both sides of 2, we have

(cid:18)

P

(cid:18)

P

max
µ(∆xI)≤δ

f (x + ∆x)

≥

0.5, f (x) < 0.5

z = z

|

f (x + ∆x)

max
µ(∆xI)≤δ

≥
Then, it can be simplified as

f (x) < 0.5, z = z

0.5

|

max
µ(∆xI)≤δ

f (x + ∆x)

0.5

|

≥

f (x) < 0.5

max
µ(∆xI)≤δ

f (x + ∆x)

≥

0.5, f (x) < 0.5

(2)

(cid:19)

(cid:18)

(3)

(cid:19)

max
µ(∆xI)≤δ

f (x + ∆x)

≥

0.5, f (x) < 0.5

(cid:19)

max
µ(∆xI)≤δ

f (x + ∆x)

0.5

|

≥

f (x) < 0.5

(cid:19)

P

(cid:19) =

P

(cid:18)

which implies that the classifier achieves demographic parity,

P (f (x) < 0.5

z = z) = P (f (x) < 0.5) ,

|

P (f (x)

0.5

|

≥

z = z) = P (f (x)

0.5)

≥

BE Suppose a classifier f achieves DP and EI. Recall that a classifier achieves DP if

2. DP, EI

⇒

which implies

P (f (x)

0.5

≥

P (f (x) < 0.5

z = z) = P (f (x)

0.5) ,

≥

z = z) = P (f (x) < 0.5) .

|

|

Recall that a classifier achieves EI if

P

(cid:18)

(cid:19)

max
µ(∆xI)≤δ

f (x + ∆x)

0.5

|

≥

f (x) < 0.5, z = z

(cid:18)

= P

(4)

(cid:19)

max
µ(∆xI)≤δ

f (x + ∆x)

0.5

|

≥

f (x) < 0.5

(5)

By multiplying both sides of 4 and 5, we have

(cid:18)

P

max
µ(∆xI)≤δ

f (x + ∆x)

0.5

|

≥

f (x) < 0.5, z = z

(cid:19)

P (f (x) < 0.5

z = z)

|

(cid:19)

(cid:18)

= P

max
µ(∆xI)≤δ

f (x + ∆x)

0.5

|

≥

f (x) < 0.5

P (f (x) < 0.5)

Then, it can be simplified as

P

(cid:18)

max
µ(∆xI)≤δ

f (x + ∆x)

≥

0.5, f (x) < 0.5

(cid:19)

z = z

|

(cid:18)

= P

max
µ(∆xI)≤δ

f (x + ∆x)

≥

0.5, f (x) < 0.5

,

(cid:19)

which implies that the classifier f achieves BE.

13

Published as a conference paper at ICLR 2023

EI Suppose a classifier f achieves BE and DP. Recall that a classifier achieves DP if

3. BE, DP

⇒

which implies

(cid:18)

P

P (f (x)

0.5

≥

P (f (x) < 0.5

z = z) = P (f (x)

0.5) ,

≥

z = z) = P (f (x) < 0.5)

|

|

(6)

(cid:19)

(7)

(cid:19)

Recall that a classifier achieves BE if

max
µ(∆xI)≤δ

f (x + ∆x)

≥

0.5, f (x) < 0.5

max
µ(∆xI)≤δ

f (x + ∆x)

≥

0.5, f (x) < 0.5

.

(cid:19)

z = z

|

(cid:18)

= P

By dividing both sides of 7 by the both sides of 6, we have
(cid:18)

(cid:18)

(cid:19)

P

max
µ(∆xI)≤δ

f (x + ∆x)

≥

0.5, f (x) < 0.5

P (f (x) < 0.5
Then, it can be simplified as

|

z = z)

P

z = z

|

=

max
µ(∆xI)≤δ

f (x + ∆x)

≥

0.5, f (x) < 0.5

P (f (x) < 0.5)

P

(cid:18)

(cid:19)

max
µ(∆xI)≤δ

f (x + ∆x)

0.5

|

≥

f (x) < 0.5, z = z

(cid:18)

= P

max
µ(∆xI)≤δ

f (x + ∆x)

0.5

|

≥

f (x) < 0.5

,

(cid:19)

which implies that the classifier f achieves EI.

Proof of Corollary 2.6. The Corollary 2.6 can be proved directly from Theorem 2.5.

A.2 CONNECTIONS BETWEEN EI AND ER

In this part, we investigate the connections between EI and ER.
Lemma A.1. Consider x
characterized by two accepting thresholds (τ0, τ1), where τ0, τ1
it satisfies ER.

(µz, σ2) for z

z = z

∼ N

|

∈ {
∈

, µz, σ
0, 1
}

R, and classifiers
R. If a classifier satisfies EI, then

∈

Q and ϕ to denote the CDF and PDF of standard Gaussian distribution,

Proof. Here we use Φ = 1
respectively. We consider the cost function µ =

| · |
Recall the definition of EI disparity and ER disparity

−

.

EI Disparity =

(cid:12)
(cid:12)
(cid:12)

(cid:16)

P

max
µ(∆x)<δ
(cid:124)

f (x + ∆x) > 0.5

|

(cid:123)(cid:122)
x>τ0−δ

(cid:125)

(cid:17)

, z = 0
f (x) < 0.5
(cid:125)
(cid:123)(cid:122)
(cid:124)
x≤τ0

,

ER Disparity =

(cid:104)
E

(cid:12)
(cid:12)
(cid:12)

min
f (x+∆x)≥0.5
(cid:124)
(cid:123)(cid:122)
τ0−x

µ(∆x)

|

(cid:125)

f (x + ∆x) > 0.5

(cid:18)

P

−

max
µ(∆x)<δ
(cid:105)
, z = 0
f (x) < 0.5
(cid:125)
(cid:123)(cid:122)
(cid:124)
x≤τ0

(cid:104)
E

−

min
f (x+∆x)≥0.5

µ(∆x)

f (x) < 0.5, z = 1

(cid:19) (cid:12)
(cid:12)
(cid:12)

f (x) < 0.5, z = 1

(cid:105)(cid:12)
(cid:12)
(cid:12).

|

|

Consequently, the EI constraint and ER constraint can be written as
(cid:18) τ0

(cid:18) τ0

µ0

µ0

(cid:19)

(cid:19)

EI Disparity (τ0, τ1) =

Φ

(cid:12)
(cid:12)
(cid:12)
(cid:12)

−

−

δ
σ

/Φ

(cid:18) τ1

Φ

−

−
σ
δ
σ

−
(cid:19)

µ1

−

(cid:18) τ1

/Φ

µ1

(cid:19)(cid:12)
(cid:12)
(cid:12)
(cid:12)

−
σ

= 0,

(8)

ER Disparity (τ0, τ1) =

(cid:12)
(cid:12)
(cid:12)
(cid:12)

Φ((τ0

1

−

(cid:90) τ0

µ0)/σ)

−∞

(τ0

−

t)ϕ

(cid:18) t

(cid:19)

µ0

−
σ

dt

−

(cid:90) τ1

µ1)/σ)

−∞

(τ1

−

t)ϕ

(cid:18) t

µ1

−
σ

(cid:19)

(cid:12)
(cid:12)
(cid:12)
(cid:12)

dt

= 0 (9)

Φ((τ1

1

−

14

Published as a conference paper at ICLR 2023

In this proof, we will first show that achieving EI is equivalent to τ0
that the classifier with τ0

−
µ1 satisfies the ER constraint.

µ0 = τ1

−

−

µ0 = τ1

−

µ1, and then show

1. EI constraint.

Let φ(x) = Φ( x−δ

σ )/Φ( x

σ ). First, we show that φ is a strictly increasing function. Note that

φ′(x) =

1
σΦ (cid:0) x

σ

(cid:1)2

(cid:18) x

(cid:18)

ϕ

(cid:19)

δ

Φ

(cid:17)

(cid:16) x
σ

Φ

−

(cid:18) x

(cid:19)

δ

(cid:17)(cid:19)

.

ϕ

(cid:16) x
σ

−
σ

−
σ

Therefore, to show that φ is strictly increasing, it is sufficient to show that

(cid:19)

δ

(cid:18) x

ϕ

−
σ

Φ

(cid:17)

(cid:16) x
σ

(cid:18) x

> Φ

−
σ

(cid:19)

δ

ϕ

(cid:17)

.

(cid:16) x
σ

(10)

We show that (10) is equivalent as the following inequality by dividing both the left-hand
side and right-hand side by ϕ( x−δ

σ )ϕ( x

σ ):

Φ

(cid:17)

(cid:16) x
σ

/ϕ

(cid:17)

(cid:16) x
σ

(cid:18) x

> Φ

(cid:19)

δ

/ϕ

(cid:18) x

(cid:19)

δ

.

−
σ

−
σ

(11)

is known in literatures as Mills’ ratio (Mitrinovic & Vasic, 1970), which is
σ , (11)

ϕ(·) is strictly increasing on R. Since x

σ > x−δ

Note that 1−Φ(·)
ϕ(·)
strictly decreasing on R. Therefore, Φ(·)
holds, thereby (10) holds and φ is strictly increasing.
Given that φ(x) = Φ( x−δ

σ )/Φ( x
(8) =

σ ) is a strictly increasing function on R,
φ(τ1
φ(τ0
|

µ1)

µ0)

= 0

−

−

−

|

yields that

2. ER constraint.

We first note that

τ0

−

µ0 = τ1

µ1.

−

(cid:90) τ0

−∞

(τ0

−

t)ϕ

(cid:18) t

(cid:19)

µ0

−
σ

t′=t−µ0
=======

dt

(cid:90) τ0−µ0

−∞

(τ0

µ0

−

−

t′)ϕ

(cid:19)

(cid:18) t′
σ

dt′

Let ψ(x) = 1

Φ(x/σ)

(cid:82) x
−∞(x

σ )dt. It is clear that ER constraint is equivalent to

t)ϕ( t

−
(9) =

ψ(τ0
|

−

µ0)

−

ψ(τ1

.

µ1)
|

−

Therefore, the classifier with τ0

µ0 = τ1

−

−

µ1 clearly satisfies the ER constraint.

Combining all the discussion above completes the proof.

B SUPPLEMENTARY MATERIALS ON THE FAIRNESS NOTIONS

Recall that in Sec. 2 and Sec. 3, this paper proposes a new fairness notion called equal improvability
(EI), compares it with other existing effort-based fairness notions, and finds a classifier by solving a
EI-constrained optimization which is formulated as a minimax problem. In Sec. B.1, we first explain
what each term in the definition of EI means. Then, we provide more details of reward function
selection of bounded effort (BE) (Heidari et al., 2019) and discuss the vulnerability of equal recourse
(ER) (Gupta et al., 2019) in Sec. B.2 and B.3, respectively Then in Sec. B.4, we provide how we
solved the inner maximization problem in the EI-constrained optimization. Finally, in Sec. B.5, we
provide numerical methods for finding the optimal solution for the EI-constrained problem, under
simple synthetic dataset setting.

15

Published as a conference paper at ICLR 2023

B.1 MEANING OF EACH TERM IN THE DEFINITION OF EI

To help readers better understand our EI definition, here we explain what each term means in EI
definition means. Let the data sample (x, y, z) follows the distribution
P(x,y,z). The meaning of each
term of EI defined in Def. 2.1 is detailed below.
(cid:19)

(cid:18)

P

x, y, z
x,y,z
∼ P
(cid:124)
(cid:125)
(cid:123)(cid:122)
Randomness is over
the data distribution

max
µ(∆xI )≤δ
(cid:124)

f (x + ∆x)

0.5

≥

(cid:12)
(cid:12)
(cid:12)
(cid:12)

(cid:123)(cid:122)
Maximum score
after improvement

(cid:125)

(cid:124)

(cid:123)(cid:122)
Event in which the sample can
be accepted after improvement

(cid:125)

(cid:18)

f (x) < 0.5, z = z
(cid:124)
(cid:125)
(cid:123)(cid:122)
Event in which the sample
comes from group z
and gets rejected

= P
x, y, z
x,y,z
∼ P
(cid:124)
(cid:125)
(cid:123)(cid:122)
Randomness is over
the data distribution

f (x + ∆x)

max
µ(∆xI )≤δ
(cid:124)

(cid:123)(cid:122)
Maximum score
after improvement

(cid:12)
(cid:12)
(cid:12)
(cid:12)

0.5

≥

(cid:19)

f (x) < 0.5
(cid:124)
(cid:125)
(cid:123)(cid:122)
Event in which the
sample gets rejected

,

(cid:125)

(cid:124)

(cid:123)(cid:122)
Event in which the sample can
be accepted after improvement

(cid:125)

B.2 SPECIFYING THE REWARD FUNCTION FOR BE

In this part, we introduce how we set the reward function so that we can connect BE with EI. Recall
that the reward function in Heidari et al. (2019) is defined as the benefit gained by changing an
individual’s characteristics from w = (x, y) to w′ = (x′, y′):

(w, w′) = b(f (x′), y′)

R

b(f (x), y)

−

where b is the benefit function, and x′ = x + ∆x is the updated feature. If we set the benefit function
0.5
as b(f (x), y) = 1
, then the reward function becomes:
≥
}
(w, w′) = 1
{

f (x + ∆x)

f (x)
{

0.5
}

f (x)

1
{

} −

0.5

≥

≥

R
Then, the bounded effort fairness is defined as:

(cid:104)

E

max
∆x R

(w, w′) s.t. µ(∆x) < δ

(cid:105)

(cid:104)

= E

z = z
|

max
∆x R

(w, w′) s.t. µ(∆x) < δ

(cid:105)

for all z

Consequently,

(cid:16)

max
∆x R

(w, w′) s.t. µ(∆x) < δ

(cid:17)

=

(cid:26)1 if maxµ(∆x)<δ f (x + ∆x)

0 otherwise

0.5 and f (x) < 0.5

≥

Therefore, we can write expectation as probability:

(cid:18)

P

max
µ(∆x)≤δ

f (x + ∆x)

≥

0.5, f (x) < 0.5

z = z

= P

|

max
µ(∆x)≤δ

f (x + ∆x)

≥

0.5, f (x) < 0.5

,

(cid:19)

(cid:18)

(cid:19)

which is in Table 1.

B.3 VULNERABILITY OF ER TO OUTLIERS

As we mentioned in Table 1, ER is vulnerable to outliers. Here we provide a simple analysis. Suppose
an outlier, having feature-attribute pair (x, z), is added to the dataset with n samples. Let the outlier is
misqualified f (x) < 0.5, and requires effort µ(∆x) = M to achieve f (x + ∆x)
0.5. In such case,
the EI disparity increases at most 1
n since EI measures the portion of samples with improved outcomes
after making efforts. On the other hand, the ER disparity increases by M
n , since ER measures the
minimum required efforts averaged over all samples. Note that a single outlier with a large M can
significantly increase the ER disparity, which does not happen for EI. Thus, one can observe that EI
is much more robust to outliers, compared with ER. We conduct a numerical experiment in Sec. C.4.1
to further highlight the robustness of EI to outliers.

≥

16

Published as a conference paper at ICLR 2023

B.4 SOLVING THE INNER MAXIMIZATION PROBLEM FOR EI

As explained in Sec.3, finding a EI classifier can be formulated as a minimax problem (1), where
solving the inner maximization problem max∥∆xI∥≤δ f (x + ∆x) is required to compute Uδ in the
regularization term, and the outer problem is the regularized-loss minimization finding the optimal
model parameter w for the classifier f = fw. In this section, we provide two ways of solving
the inner maximization problem. In particular, in Sec. B.4.1 we give the explicit expression of the
optimizer ∆xI when generalized linear model is considered. In Sec. B.4.2, we solve the problem
under a more general setting via adversarial training.

B.4.1 CLOSED-FORM SOLUTION FOR GENERALIZED LINEAR MODEL

R is
Consider a Generalized Linear Model (GLM) written as f (x) = g−1(w⊤x), where g : [0, 1]
a strictly increasing link function, and w is the model parameter. Denote the weights corresponding
to xI as wI. Then, the inner maximization problem can be written as:

→

max
∥∆xI∥≤δ

f (x + ∆x) = max

∥∆xI∥≤δ

g−1(w⊤(x + ∆x))

= max

∥∆xI∥≤δ

g−1(w⊤x + w⊤

I ∆xI)

(∵ ∆x = (∆xI , 0, 0))

(cid:18)

= g−1

w⊤x + max

∥∆xI∥≤δ

(cid:19)

w⊤

I ∆xI

(∵ g is strictly increasing)

When
When

∥·∥∞, the maximum is achieved by letting ∆xI = δ sign(wI), w⊤∆xI = δ
=
∥2 , w⊤∆xI = δ
∥·∥2, the maximum can be achieved by letting ∆xI = δwI/
=

wI
∥

wI
∥
wI
∥

∥1.
∥2.

∥·∥
∥·∥

B.4.2 ADVERSARIAL TRAINING BASED APPROACH FOR GENERAL SETUP

Here we discuss how we solve the inner maximization problem under a more general setting.
Following popular adversarial training methods, we apply projected gradient descent (PGD) for
multiple times to update ∆xI, i.e., set
∆xI =

(12)

(∆xI + γ

∆xIf (x + ∆x)),
P
is the projection onto the constrained space

∇

P

δ. For
where γ > 0 is the step size, and
is equivalent to the clipping process when we use ℓ∞ norm. Denote the maximizer of the
instance,
inner maximization problem as ∆x⋆ = (∆xI
⋆, 0, 0). Then, from Danskin’s theorem Danskin (1967),
wfw(x + ∆x⋆). We can use this derivative to update
we have
w in the outer loss minimization problem. The pseudocode of this adversarial training based method
is shown in Algorithm 1.

w max∥∆xI∥≤δ fw(x + ∆x) =

∆xI

∥ ≤

∇

∇

P

∥

D

:Dataset

Algorithm 1 Pseudocode for achieving EI
Input
Output :Model parameter w for the classifier f .
Initialize w;
for each iteration do
for each (xi, yi)
∈ D
Initialize ∆x⋆
i ;
for each PGD iteration do

do

Update ∆x⋆

i according to (12);

Update w according to the regularized loss function defined in (1);

B.5 DERIVATION OF OPTIMAL EI CLASSIFIER FOR SYNTHETIC DATASET

This section shows how we obtain the optimal EI classifier (that minimizes the cost function in (1))
for a synthetic dataset having two features x = [x1, x2] sampled from a Gaussian distribution
0, 1

(µz,y, Σz,y) where the mean µz,y and the standard deviation Σz,y depends on the label y

N
0, 1
and the group attribute z
}
obtained in this section is provided in the yellow line in Fig. 3.

∈ {

}
. Note that the performance curve of the optimal EI classifier

∈ {

17

Published as a conference paper at ICLR 2023

The optimal EI classifier is obtained in the following steps: (i) define mathematical notations used
for analysis (Sec. B.5.1), (ii) compute the error probability (Sec. B.5.2), (iii) compute EI disparity
(Sec. B.5.3), and (iv) solve the EI-regularized optimization problem and find the optimal EI classifier
(Sec. B.5.4).

B.5.1 NOTATIONS

We consider finding a z-aware linear classifier which predicts the label y from two features x1, x2
and one sensitive attribute z. In other words, given x and z, the output of a model is represented as
f (x) = w1x1 + w2x2 + w3z + b 4 where [w1, w2, w3] is the weight vector, b is the bias.

For group z = 0,

For group z = 1,

ˆy =

(cid:26)1 if w1x1 + w2x2 >

0 else

b

−

ˆy =

(cid:26)1 if w1x1 + w2x2 >

0 else

w3

b

−

−

1 + w2

2 = 1, and parameterize them as w1 = sin θ, w2 = cos θ.

Without the loss of generality, let (cid:112)w2
Then, for group z = 0,

and for group z = 1,

ˆy =

ˆy =

(cid:26)1 if (sin θ)x1 + (cos θ)x2 > b0

0 else

(cid:26)1 if (sin θ)x1 + (cos θ)x2 > b1

0 else

where b0 =
−
−
univariate Gaussian, we have

b and b1 =

w3

−

b. Since the linear combination of multivariate Gaussian is a

w⊤

θ x

∼ N

(w⊤

θ µz,y, w⊤

θ Σz,ywθ)

(13)

where wθ = [sin θ, cos θ]. The decision rules can be written in terms of the wθ. For group z = 0,
(cid:26)1 if w⊤
0 else

θ x > b0

ˆy =

For group z = 1,

ˆy =

(cid:26)1 if w⊤
0 else

θ x > b1

Now, the question is, what is the optimal parameters θ, b0, b1 that solve the optimization problem
in (1). In order to answer this question, we need to understand how the equal improvability condition
is represented in terms of the model parameters. Suppose we use 0-1 loss function l, and use l∞
∞. From the result in Sec. B.4, given the effort budget δ, the maximum score
norm µ(x) =
∥
) where wI is the
improvement (maxµ(∆xI )≤δ f (x + ∆x)
f (x)) = δ
|
weights for improvable features x1, x2. Thus, if we denote the ˆymax as the estimated label after the
improvement, we have

1 = δ(
|

cos θ

x
∥

sin θ

wI

−

+

∥

∥

|

|

ˆymax =

(cid:26)1 if (sin θ)x1 + (cos θ)x2 > b0

0 else

for group z = 0 and

ˆymax =

(cid:26)1 if (sin θ)x1 + (cos θ)x2 > b1

0 else

δ(
|

−

sin θ

δ(
|

−

sin θ

+

+

|

|

|

|

cos θ

) = b′
0
|

cos θ

) = b′
1
|

for group z = 1.

4In this case, the decision boundary is {x : f (x) = 0} instead of {x : f (x) = 0.5} used in the main paper.

18

Published as a conference paper at ICLR 2023

B.5.2 COMPUTE ERROR PROBABILITY

The error probability can be written as,

Pr(ˆy

= y) =

1
(cid:88)

i=0

Pr(z = i) Pr(ˆy

= y

z = i)
|

We can derive the term Pr(ˆy

z = 0) as below:
= y
|

Pr(ˆy

= y
|

z = 0) = Pr(y = 0
|

z = 0) Pr(ˆy = 1
|

y = 0, z = 0)

z = 0) Pr(ˆy = 0
+ Pr(y = 1
|
|

y = 1, z = 0)

We can look each term Pr(ˆy = 1
y = 0, z = 0), Pr(ˆy = 0
|
|
terms of Q-functions because,

y = 1, z = 0) and write those terms in

Pr(ˆy = 1
|
Pr(ˆy = 0
|

y = 0, z = 0) = Pr(w⊤

θ x > b0

y = 1, z = 0) = Pr(w⊤

θ x < b0

y = 0, z = 0)

|

y = 1, z = 0)
|

From (13), we have

y = 0, z = 0) = Pr(w⊤
Pr(ˆy = 1
|

θ x > b0

y = 0, z = 0) = Q
|

y = 1, z = 0) = Pr(w⊤
Pr(ˆy = 0
|

θ x < b0

y = 1, z = 0) = Q
|

















b0
(cid:113)

w⊤

θ µ0,0
−
w⊤
θ Σ0,0wθ

w⊤
(cid:113)

b0

θ µ1,0
w⊤

−
θ Σ1,0wθ

One can derive the error rates for group z = 1 similarly. So, the total error rate can be written as



Pr(ˆy

= y) = Pr(z = 0)

z = 0)Q
Pr(y = 0
|





b0
(cid:113)





w⊤

θ µ0,0
−
w⊤
θ Σ0,0wθ



+ Pr(y = 1
|

z = 0)Q







w⊤
(cid:113)

b0

θ µ1,0
w⊤

−
θ Σ1,0wθ





+ Pr(z = 1)

z = 1)Q
Pr(y = 0
|

+ Pr(y = 1
|

z = 1)Q





w⊤
(cid:113)

b1

θ µ1,1
w⊤

−
θ Σ1,1wθ









b1
(cid:113)





w⊤

θ µ1,0
−
w⊤
θ Σ1,0wθ



We have three parameters to optimize the error rate θ, b0, b1. All the other terms are known.

B.5.3 COMPUTE EI DISPARITY

To compute EI disparity, we start with computing

Pr(ˆymax = 1
ˆy = 0, z = 0) =
|

The denominator of (14) can be expanded as

Pr(ˆymax = 1, ˆy = 0
|
Pr(ˆy = 0
|

z = 0)

z = 0)

(14)

z = 0) Pr(ˆy = 0
Pr(ˆy = 0
y = 0, z = 0)
z = 0) = Pr(y = 0
|
|
|

z = 0) Pr(ˆy = 0
y = 1, z = 0).
+ Pr(y = 1
|
|

We can look each term Pr(ˆy = 0
y = 0, z = 0), Pr(ˆy = 0
|
|
terms of Q-function because,

y = 1, z = 0) and write those terms in

Pr(ˆy = 0
|

y = 0, z = 0) = Pr(w⊤

θ x < b0

y = 0, z = 0)

|

19

̸
̸
̸
̸
̸
Published as a conference paper at ICLR 2023

Pr(ˆy = 0
|

y = 1, z = 0) = Pr(w⊤

θ x < b0

y = 1, z = 0)

|

From (13), we have

y = 0, z = 0) = Pr(w⊤
Pr(ˆy = 0
|

θ x < b0

y = 0, z = 0) = Q
|

y = 1, z = 0) = Pr(w⊤
Pr(ˆy = 0
|

θ x > b0

y = 1, z = 0) = Q
|

















w⊤
(cid:113)

b0

θ µ0,0
w⊤

−
θ Σ0,0wθ

w⊤
(cid:113)

b0

θ µ1,0
w⊤

−
θ Σ1,0wθ

Then,

Pr(ˆy = 0
z = 0)Q
z = 0) = Pr(y = 0
|
|









w⊤
(cid:113)

b0

θ µ0,0
w⊤

−
θ Σ0,0wθ

z = 0)Q
+ Pr(y = 1
|





The numerator of (14) can be expanded as





w⊤
(cid:113)

b0

θ µ1,0
w⊤

−
θ Σ1,0wθ

z = 0) Pr(ˆymax = 1, ˆy = 0
Pr(ˆymax = 1, ˆy = 0
y = 0, z = 0)
z = 0) = Pr(y = 0
|
|
|

+ Pr(y = 1
|

z = 0) Pr(ˆymax = 1, ˆy = 0
y = 1, z = 0).
|

We can look each term Pr(ˆymax = 1, ˆy = 0
write those terms in terms of Q-function because,

y = 0, z = 0), Pr(ˆymax = 1, ˆy = 0
|

y = 1, z = 0) and
|

Pr(ˆymax = 1, ˆy = 0
|
Pr(ˆymax = 1, ˆy = 0
|

From (13), we have

y = 0, z = 0) = Pr(b′

0 < w⊤

θ x < b0

y = 1, z = 0) = Pr(b′

0 < w⊤

θ x < b0

y = 0, z = 0)
|

y = 1, z = 0)
|

Pr(ˆymax = 1, ˆy = 0
y = 0, z = 0) = Pr(b′
|

0 < w⊤
θ x < b0


y = 0, z = 0) =

|
θ µ0,0
w⊤

w⊤
(cid:113)

−
θ Σ0,0wθ

b0







Q



−

w⊤
(cid:113)

b′
0

θ µ0,0
w⊤

−
θ Σ0,0wθ

Q



Pr(ˆymax = 1, ˆy = 0
y = 1, z = 0) = Pr(b′
|

0 < w⊤
θ x < b0


y = 1, z = 0) =

Then,

Pr(ˆymax = 1, ˆy = 0
z = 0)
z = 0) = Pr(y = 0
|
|

z = 0)
+ Pr(y = 1
|

Q







Q







Q



|
θ µ1,0
w⊤

w⊤
(cid:113)

−
θ Σ1,0wθ

b0

w⊤
(cid:113)

b0

θ µ0,0
w⊤

−
θ Σ0,0wθ

w⊤
(cid:113)

b0

θ µ1,0
w⊤

−
θ Σ1,0wθ







Q



−











Q



−



Q



−

w⊤
(cid:113)

b′
0

θ µ1,0
w⊤

−
θ Σ1,0wθ

w⊤
(cid:113)

b′
0

θ µ0,0
w⊤

−
θ Σ0,0wθ

w⊤
(cid:113)

b′
0

θ µ1,0
w⊤

−
θ Σ1,0wθ

It can be derived for group z = 1 similarly. So, we derived EI disparity in terms of θ, b0, b1.

20

























Published as a conference paper at ICLR 2023

B.5.4 SOLVE THE OPTIMIZATION PROBLEM

In the previous two sections, we derived the error rate and EI disparity in terms of Q-functions
containing parameters θ, b0, b1. Therefore, we can construct an EI-constrained optimization problem
(which is essentially same as (1)):

min
θ,b0,b1

Pr(ˆy

= y)

s.t. max

i∈{0,1} |

Pr(ˆymax = 1
|

ˆy = 0, z = i)

Pr(ˆymax = 1
|

ˆy = 0)
|

−

< c

where c is a hyperparameter we can choose to balance error rate and EI disparity.

After writing error rate and EI disparity in terms of Q-functions (using the derivations in Sec. B.5.2
and Sec. B.5.3), we numerically solve the constrained optimization problems above with a popular
python module scipy.optimize. To get the experimental results in Fig. 3, we numerically solved
the above problem for 20 different c values, where the maximum c is picked as the EI disparity of the
unconstrained optimization problem.

C SUPPLEMENTARY MATERIALS ON EXPERIMENTS

In this section, we provide the details of the experiment setup and additional experimental results.

C.1 SYNTHETIC DATASET GENERATION

We define y, z as z
∼
feature x follows the conditional distribution (x
|
of each cluster is

Bern(0.4), (y

z = 0)

∼

|

Bern(0.3), and (y
y = y, z = z)

∼ N

z = 1)

Bern(0.5). The
|
(µy,z, Σy,z) where the mean

∼

µ0,0 = [

0.1,

0.2], µ0,1 = [

0.2,

−
and the covariance matrix of each cluster is

−

−

0.3], µ1,0 = [0.1, 0.4], µ1,1 = [0.4, 0.3]

−

Σ0,0 =

(cid:21)

(cid:20)0.4 0.0
0.0 0.4

, Σ1,0 = Σ0,1 =

(cid:20)0.2
0.0

(cid:21)

0.0
0.2

, Σ1,1 =

(cid:20)0.1
0.0

(cid:21)

0.0
0.1

.

C.2 HYPERPARAMETER SELECTION

The selected hyperparameter for each experiment is provided in our anonymous Github. In all our
0.0001, 0.001, 0.01, 0.1
experiments, we perform cross-validation to select the learning rate from
.
}
{
In addition, for each penalty term we did two-step cross-validation to choose λ. In the first step, we
used a set λ
0, 0.2, 0.4, 0.6, 0.8, 0.9
. In the second step, we generate a second set around the best
}
λ⋆ found in the first step, i.e., the second set is
max
.
}}
{
For example, if λ⋆ = 0.4 is the best at the first step, then at the second step we use the set
λ

0.3, 0.35, 0.4, 0.45, 0.5

λ⋆ + ε, 0
{

0.05, 0, 0.05, 0.1

∈ {−

0.1,

∈ {

: ε

−

}

∈ {

.
}

C.3 ADDITIONAL EXPERIMENTAL RESULTS ON ALGORITHM EVALUATION

Table 3 shows the performance of ERM baseline and our three approaches (covariance-based, KDE-
based, loss-based) introduced in Sec. 3, for a multi-layer perceptron (MLP) network having one
hidden layer with four neurons. Similar to the result in Table 2 for the logistic regression model, our
methods can reduce the EI disparity without losing the classification accuracy (i.e., increasing the
error rate) much.

Meanwhile, according to a previous work (Cherepanova et al., 2021), large deep learning models are
observed to overfit to fairness constraints during training and therefore produce unfair predictions
on the test data. To confirm whether our method is also having such limitations, we investigate the
performance of our algorithms on over-parameterized models. Specifically, we conduct experiments
on a five-layer ReLU network with 200 hidden neurons per layer, which is over-parameterized
for German dataset. Table 4 reports the performance of EI-constrained classifiers on such over-
parameterized setting. One can confirm that our methods (covariance-based, KDE-based, loss-based)
perform well in both training and test dataset, and we do not observe the overfitting problem.

21

̸
Published as a conference paper at ICLR 2023

Table 3: Comparison of error rate and EI disparities of ERM baseline and proposed methods on
the synthetic, German Statlog Credit and ACSIncome-CA datasets on Multi-Layer Perceptron
(MLP). For each dataset, we boldfaced the lowest EI disparity value. Compared with ERM, all three
methods proposed in this paper enjoys much lower EI disparity without losing the accuracy much.
All reported numbers are evaluated on the test set.

DATASET

METRIC

ERM

METHODS
COVARIANCE-BASED KDE-BASED

LOSS-BASED

SYNTHETIC

ERROR RATE(↓)
EI DISP.(↓)

.205 ± .003
.141 ± .036

.242 ± .006
.004 ± .002

.227 ± .008
.011 ± .006

.229 ± .012
.018 ± .009

GERMAN
STAT.

ERROR RATE(↓)
EI DISP.(↓)

.221 ± .010
.059 ± .045

.299 ± .012
.013 ± .025

.232 ± .018
.041 ± .025

.238 ± .035
.013 ± .019

ACSINCOME-
CA

ERROR RATE(↓)
EI DISP.(↓)

.181 ± .002
.042 ± .002

.202 ± .002
.010 ± .006

.182 ± .002
.010 ± .002

.185 ± .001
.006 ± .003

Table 4: Error rate and EI disparities for ERM baseline and proposed methods, for an over-
parameterized neural network on German Statlog Credit dataset. Performances on train/test
dataset are presented. Note that we don’t observe the overfitting issue in the over-parameterized
setting.

DATASET

METRIC

ERM

COVARIANCE-BASED KDE-BASED LOSS-BASED

GERMAN
STAT.

TRAIN ERR.(↓)
TEST ERR.(↓)
TRAIN EI DISP.(↓)
TEST EI DISP.(↓)

.117 ± .004
.117 ± .010
.022 ± .017
.060 ± .032

.133 ± .003
.118 ± .010
.018 ± .011
.049 ± .024

.125 ± .008
.121 ± .010
.018 ± .009
.057 ± .028

.132 ± .011
.130 ± .009
.015 ± .013
.047 ± .025

METHODS

In addition, in Table 5, we include ER and BE as baselines for the synthetic dataset experiment
provided in Table 2. We leverage the algorithm suggested by Gupta et al. (2019) for mitigating ER
disparity. We extend the loss-based approach to reduce BE disparity, by redefining the BE loss of
group z as

˜LBE
z

≜

1
number of samples in group z

(cid:88)

i∈I−,z

ℓ(1, max

∥∆xIi∥≤δ

f (xi + ∆xi)),

where

˜LEI
z

≜

1
number of rejected samples in group z

(cid:88)

i∈I−,z

ℓ(1, max

∥∆xIi∥≤δ

f (xi + ∆xi)),

and I−,z is the set of rejected samples in group z for z

[Z].

∈

Table 5 shows that the minimum EI disparity is achieved by our methods. Hence, if the EI fairness
needs to be achieved, then it cannot be replaced with the existing other fairness notions for some
datasets.

Table 5: Comparison of error rate and EI disparities of ERM, ER, and BE baseline and
proposed methods on the synthetic dataset. We boldfaced the lowest EI disparity value. The three
EI-regularized approaches achieve the lowest EI disparity while maintaining low error rates. All
reported numbers are evaluated on the test set.

DATASET

METRIC

ERM

ER
(GUPTA ET AL. (2019))

BE
(LOSS-BASED)

COVARIANCE-BASED KDE-BASED

LOSS-BASED

SYNTHETIC

ERROR RATE(↓)
EI DISP.(↓)

.221 ± .001
.117 ± .007

.235 ± .009
.036 ± .018

.252 ± .006
.006 ± .004

.253 ± .003
.003 ± .001

.250 ± .001
.005 ± .003

.246 ± .001
.002 ± .001

METHODS

C.4 EVALUATING ROBUSTNESS OF EI

In this section, we highlight the advantages of EI over BE and ER in terms of robustness to outliers
and imbalanced negative prediction rates. In doing so, we consider certain data distributions and
follow the method discussed in Sec. B.5 for solving the classifiers.

22

Published as a conference paper at ICLR 2023

C.4.1 EI VERSUS ER: ROBUSTNESS TO OUTLIERS

As we discussed in Sec. B.3, ER is vulnerable to outliers. In this experiment, we systematically study
the robustness of EI and ER to outliers.

(Clean) Let sensitive attribute z

Bern(0.5) be
Data distributions
independent of sensitive attribute z. Given the sensitive attribute z and label y, feature x follows the
conditional distribution x
(µy,z, Σy,z), where the mean and covariance of the
four Gaussian clusters are

Bern(0.5) and label y

y = y, z = z

∼ N

∼

∼

|

and

µ0,0 = [1,

−

6], µ0,1 = [

−

1,

−

2], µ1,0 = [2, 1.5], µ1,1 = [1, 2.5],

Σ0,0 = Σ1,0 = Σ0,1 = Σ1,1 =

(cid:20)0.25
0.0

(cid:21)

0.0
0.25

.

(Contaminated) We contaminate the distribution by introducing additional 5% outliers to group
z = 0. The outliers follow Gaussian distribution with mean and covariance matrix

µoutlier,0 = [

1,

−

20], Σoutlier,0 =

−

(cid:20)0.05
0.0

(cid:21)

0.0
0.05

.

Figure 6: Visualizations of the EI and ER decision boundaries without and with the presence
of outliers. We observe that the decision boundary of ER changes a lot in the presence of outliers,
while the decision boundary of EI is not affected. This phenomenon implies the robustness of EI to
outliers.

Results The decision boundaries of EI and ER for both the clean dataset and contaminated dataset
are depicted in Fig. 6. These decision boundaries are the optimal linear decision boundaries based on
the distributional information, we followed the same procedure as we mentioned in B.5. The δ for
the EI classifier is picked as 1.5. We observe that the introduction of outliers makes the ER decision
boundary rotate a lot, leading to a drop in classification accuracy and ER disparity w.r.t.the clean
data distribution. Moreover, we note that the newly added outliers fail to destroy EI classifier, which
implies the robustness of EI to outliers. The Table 6 also presents the accuracy, EI, and ER disparity
of each classifier. These metrics are computed after excluding the outliers. It shows that the EI and
ER disparities have higher values under the dataset with the outliers for the ER classifier.

Table 6: Error rate and EI and ER disparities of linear EI and ER classifiers.

DATASET

METRIC

EI CLASSIFIER

ER CLASSIFIER

METHODS

DATASET WITHOUT OUTLIERS

DATASET WITH OUTLIERS

ERROR RATE(↓)
EI DISP.(↓)
ER DISP.(↓)

ERROR RATE(↓)
EI DISP.(↓)
ER DISP.(↓)

.001 ± .001
.001 ± .001
.004 ± .001

.001 ± .001
.017 ± .001
.020 ± .001

.001 ± .001
.005 ± .001
.001 ± .001

.020 ± .001
.348 ± .001
.291 ± .001

23

−202−20−15−10−50f(x)<0.5f(x)≥0.5CleanData−202−20−15−10−50f(x)<0.5f(x)≥0.5DatawithOutliersGroup0Group1EIDecisionBoundaryERDecisionBoundaryPublished as a conference paper at ICLR 2023

C.4.2 EI VERSUS BE: ROBUSTNESS TO IMBALANCED GROUP NEGATIVE RATE

In this experiment, we investigate the robustness of EI and BE to an imbalanced negative rate.

Figure 7: Visualizations of the EI and BE decision boundaries given the data distribution with
the same negative rates and different negative rates. The decision boundary of BE rotates a lot
when the negative rate of the dataset becomes different, implying the sensitivity of BE to imbalanced
negative rates. In contrast, the consistency of EI decision boundaries showcases the robustness of EI
w.r.t.imbalanced negative rates.

(Same Negative Rate) We consider the distribution with balanced subgroups.
Data Distributions
Bern(0.5) be independent of
In other words, let sensitive attribute z
sensitive attribute z. Given the sensitive attribute z and label y, feature, feature x follows the Gaussian
distribution (x

(µy,z, Σy,z) where the mean of each cluster is

Bern(0.5), and label y

y = y, z = z)

∼

∼

|

∼ N

µ0,0 = [

−

2,

−

1], µ0,1 = [

−

1,

−

2], µ1,0 = [1, 2], µ1,1 = [2, 1]

and the covariance matrix of each cluster is

Σ0,0 = Σ1,0 = Σ0,1 = Σ1,1 =

(cid:20)0.25
0.0

(cid:21)

0.0
0.25

.

(Different Negative Rate) We manipulate the distribution of label y for constructing data distribution
with different negative rates. To be more specific, we let y
Bern(0.3).

Bern(0.7), and y

z = 1

z = 0

∼

∼

|

|

Results Figure 7 shows the decision boundaries of EI and BE when (i) the dataset has the same
negative rate, and (ii) the dataset has a different negative rate. These decision boundaries are the
optimal linear decision boundaries based on the distributional information, we followed the same
procedure as we mentioned in B.5. The δ for the EI and BE classifiers is picked as 1.5. The huge
difference in BE decision boundaries under the two cases verifies our claim that BE cannot handle
imbalanced negative prediction rates. In contrast, the decision boundaries of EI learned with two
datasets with different group proportions are consistent.

C.5 DYNAMIC MODELING

Here we justify the dynamic model we used in our experiments in Sec. 4.2, for updating individual’s
features and for updating the classifier. Specifically, Sec. C.5.1 explains the choice of ϵ(x), the
amount of effort a sample (having feature x), while Sec. C.5.2 explains the choice of (τ (0)
, τ (1)
)
t
defining the classifier at round t.

t

C.5.1 UPDATING INDIVIDUAL’S FEATURES (CHOICE OF ϵ(x))

At first, we designed a model by assuming that each individual makes an effort, where the amount of
the effort he/she makes is (1) proportional to the reward (the improvement on the outcome) it will get
by making such effort, and (2) inversely proportional to the required amount of efforts to improve the

24

−202−202f(x)<0.5f(x)≥0.5SameNegativeRate−202−202f(x)<0.5f(x)≥0.5DiﬀerentNegativeRateGroup0Group1EIDecisionBoundaryBEDecisionBoundaryPublished as a conference paper at ICLR 2023

outcome, as shown in the below equation:

Realized effort ϵ(x) ≜

improvement on the outcome
required efforts for improving the outcome2

=

x < τ (z)
1
t }
{
(τ (z)
x)2
t −

,

t

where τ (z)
This equation can be interpreted as follows: an individual that is unqualified (x < τ (z)
make positive effort, but he/she is less motivated if the distance τ (z)
too large.

is the accepting threshold for the individuals from group z at round t, and x is the feature.
) is willing to
x to the decision boundary is

t −

t

In order to upper bound the amount of reward, we added a small constant β > 0 in the numerator,
thus having the final form:

ϵ(x) =

x < τ (z)
t }
x + β)2

1
{
(τ (z)
t −

.

C.5.2 UPDATING THE CLASSIFIER

Note that at each round t, each individual’s features are updated. Accordingly, we also update the
classifier (having parameters τ (0)
) in the same round. We update the EI classifier based on
the following constrained optimization problem:

and τ (1)

t

t

min (EI Disparity)

s.t.

error rate

threshold,

≤

which implies that we aim to find a classifier that guarantees good accuracy and small EI disparity.
This optimization can be rewritten as

min

τ (0)
t

,τ (1)
t

xt + δt > τ (0)
(cid:124)
(cid:125)
(cid:123)(cid:122)
the outcome can be improved

t

(cid:122)
(cid:12)
(cid:12)
(cid:12)
P(
(cid:12)
(cid:12)
(cid:12)
(cid:12)
error rate
(cid:125)(cid:124)
(cid:123)
= yt)

(cid:122)
P(ˆyt

s.t.

c.

≤

EI Disparity
(cid:125)(cid:124)

z = 0, xt < τ (0)
)
(cid:124)
(cid:125)

(cid:123)(cid:122)
rejected

t

|

P(xt + δt > τ (1)

t

z = 1, xt < τ (1)

t

|

−

(cid:123)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
)
(cid:12)
(cid:12)
(cid:12)

Given the data distribution at each round t, we numerically solve the constrained optimization
problems using a popular python module scipy.optimize.

C.6 OBTAINING BASELINE CLASSIFIERS FOR DYNAMIC SCENARIOS

Continued from Sec. 4.2, this section describes how we compute the classifiers that satisfy demo-
graphic parity (DP), bounded effort (BE) (Heidari et al., 2019) equal recourse (ER) (Gupta et al.,
2019) and individual-level fairness causal recourse (ILFCR) (Von Kügelgen et al., 2022), respectively.
Similar to EI classifier, we obtain the best DP classifier by considering the following constrained
optimization problem:

s.t. P(ˆyt ̸
The best BE classifier can be obtained by solving the following problem:

P(ˆyt = 1
|

z = 1)
|

P(ˆyt = 1

z = 0)

min

−

|

|

= yt)

c.

≤

min

(cid:12)
(cid:12)
(cid:12)

P(τ (0)

t −

δt < xt < τ (0)

t

z = 0)

|

P(τ (1)

t −

−

δt < xt < τ (1)

t

(cid:12)
(cid:12)
z = 1)
(cid:12)

|

s.t. P(ˆyt ̸

= yt)

c.

≤

Similarly, the optimization problem for obtaining the best ER classifier is written as:

min

(cid:104)

(cid:12)
(cid:12)
(cid:12)

E

τ (0)
t −

xt

|

xt < τ (0)

t

(cid:105)
, z = 0

E

(cid:104)
τ (1)
t −

xt

|

−

xt < τ (1)

t

, z = 1

(cid:105)(cid:12)
(cid:12)
(cid:12)

s.t. P(ˆyt ̸

= yt)

c.

≤

While the computation of ILFCR classifier is less straightforward, we first describe the setting
considered in Von Kügelgen et al. (2022). This paper assumes that each observed variable xi is

25

̸
Published as a conference paper at ICLR 2023

determined by (i) its direct causes (causal parents) which include the sensitive attribute z and other
observed variables xj, and (ii) an unobserved variable ui. Since our dynamics experiment considers
only one-dimensional variable x, it is determined by the sensitive attribute z and latent variable u.
Therefore, we write x as a function of z and u, i.e., x(z, u).

Before we describe how we compute ILFCR in our experiments, we introduce the definition of causal
recourse and ILFCR in more detail. Given a model and a sample with feature x, the causal recourse
C(x) of the sample is the minimum cost of changing the features, in order to alter the decision of
the model. Let τ (z)
be the
cost of improving the feature by ∆x. Let x′ denote the improved feature. In our one-dimensional
dynamic setting, the causal recourse is defined as

R be the acceptance threshold of group z

and µ(∆x) =

0, 1
}

∆x
|

∈ {

∈

|

C(x(z, u)) = min
x′≥τ (z)

µ(x′

−

x(z, u)) = max(τ (z)

x(z, u), 0),

−

for z = 0, 1. ILFCR requires different groups have the same causal recourse for all realizations of the
latent variable u:

max
u |

C(x(0, u))

C(x(1, u))

= 0.

|

−

0, 1

∈ {

, i.e., x
}

Now we describe how we compute ILFCR disparity. Recall that for dynamic experiments, we assumed
z ). This
the feature x follows Gaussian distribution for each group z
can be represented as a notation used in Von Kügelgen et al. (2022): the latent variable is u
(0, 1)
and the feature is represented as x(z, u) = µz + σzu. We measure the ILFCR disparity of a decision
boundary pair (τ0, τ1) by
ILFCR Disparity(τ (0), τ (1)) = max
u |
(cid:12)
(cid:12)
(cid:12)max
= max
(cid:12)
(cid:12)
(cid:12)max
(cid:124)

−
(cid:123)(cid:122)
=(τ (0)−τ (1)−µ0+µ1−(σ0−σ1)u)→∞ when u→−∞

C(x(1, u))
|

C(x(0, u))

τ (1)
(cid:16)

−
τ (1)

x(0, u), 0

x(1, u), 0

(µz, σ2

(cid:17)(cid:12)
(cid:12)
(cid:12)
(cid:125)

= max

σ1u, 0

σ0u, 0

(cid:17)(cid:12)
(cid:12)
(cid:12)

∼ N

∼ N

max

max

τ (0)

τ (0)

−
(cid:17)

µ1

µ0

−

−

−

−

−

−

(cid:16)

(cid:17)

(cid:16)

(cid:16)

z

u

u

|

.

Note that the ILFCR disparity is infinity if we take the maximum over all u. Therefore, we instead
3σz, µz + 3σz] for group
ignore the tail distribution and focus on the samples with x
z = 0, 1. In order to find a classifier with small ILFCR disparity, we numerically solve the following
constrained optimization problem

[µz

−

∈

min
τ (0),τ (1)

ILFCR Disparity(τ (0), τ (1))

s.t.

error rate

α/2.

≤

Given the data distribution at each round t, we numerically solve all the constrained optimization
problems above using a popular python module scipy.optimize.

C.7 PERFORMANCE COMPARISON BETWEEN FAIRNESS NOTIONS UNDER OTHER DYMANIC

MODELS

Here we provide one example in which EI improves long-term fairness, while some other methods
(EO, BE, and ER) harm long-term fairness.

In this example, we modified the effort
1
from ϵ(x) =
t −x+β)2
(cid:19)

x < τ (z)
t }
{

(τ (z)

(cid:18)

1

function
to ϵ(x) =

log

max(

1
t −x+β)2

(τ (z)

, 1)

1
{

x < τ (z)
t }

so that the re-

t

jected individuals make less improvement than the setting
in the original manuscript. Here τ (z)
is the acceptance
threshold for group z at iteration t, and β > 0 is a small
constant to avoid zero denominators. We set β = 0.2
and the true acceptance rate α = 0.5. Figure 8 reports
how long-term unfairness changes when the initial dis-
(0.0, 3.02) for group 0 and
tribution is x
z = 1
x

z = 0
|
∼ N
(1.0, 1.02) for group 1.
∼ N

|

26

Long-term unfairness
Figure 8:
(0)
dT V (
) at each round t for var-
,
t
P
ious algorithms in the setup described in
Sec. C.7.

(1)
t

P

0100.250.500.75ERMDPBEERILFCREI(Ours)0.00.51.0Round0.00.51.0Long-termUnFairnessPublished as a conference paper at ICLR 2023

D EXAMPLES: INVESTIGATING THE ONE-STEP IMPACT OF FAIRNESS

NOTIONS VIA MATHEMATICAL ANALYSIS

In this section, we conduct mathematical analysis on two examples to better understand how EI
improves long-term fairness compared to existing fairness notions. We first introduce a basic setup
for the mathematical analysis and then apply it to two examples to compare EI with BE, ERM, and
ER in terms of their one-step impact on data distributions.

D.1 BASIC SETUP AND PRELIMINARIES

D.1.1 CLASSIFIERS

We consider the z

aware classifier, having

−

(cid:26)1
{
1
{
which is parameterized by the threshold pair (τ0, τ1). We assume the effort budget is δ = m
m is defined in the dataset.

z = 0,
z = 1,

f (x) =

,
}
,
}

τ0
τ1

≥
≥

x
x

2 where

Recall the zero equal improvability (EI) disparity condition is:

(cid:18)

P

max
µ(∆x)≤δ

f (x + ∆x) ≥ 0.5 | f (x) < 0.5, z = 0

(cid:19)

(cid:18)

= P

max
µ(∆x)≤δ

f (x + ∆x) ≥ 0.5 | f (x) < 0.5, z = 1

(cid:19)

where µ(∆x) =

∆x

|

. This condition is equivalent to
|
(cid:82) τ1
τ1−δ p1(x)dx
(cid:82) τ1
−∞ p1(x)dx

(cid:82) τ0
τ0−δ p0(x)dx
(cid:82) τ0
−∞ p0(x)dx

=

We denote the improvability ratio of each group as
(cid:82) τ0
τ0−δ p0(x)dx
(cid:82) τ0
−∞ p0(x)dx
(cid:82) τ1
τ1−δ p1(x)dx
(cid:82) τ1
−∞ p1(x)dx

r0(τ0) =

r1(τ1) =

,

.

(15)

(16)

(17)

The classifier that satisfies this zero EI disparity condition and minimizes the error rate is denoted as
the optimal EI classifier.

Recall that the bounded effort (BE) fairness constraint is:

(cid:18)

P

max
µ(∆x)≤δ

f (x + ∆x) ≥ 0.5, f (x) < 0.5 | z = 0

(cid:19)

(cid:18)

= P

max
µ(∆x)≤δ

f (x + ∆x) ≥ 0.5, f (x) < 0.5 | z = 1

(cid:19)

where µ(∆x) =

∆x

|

. Meanwhile, the Equal Recourse (ER) constraint is defined as
|
(cid:20)

(cid:21)

(cid:20)

E

min
f (x+∆x)≥0.5

µ(∆x)

|

f (x) < 0.5, z = 0

= E

min
f (x+∆x)≥0.5

µ(∆x)

|

f (x) < 0.5, z = 1

(19)

(18)

(cid:21)

D.1.2 DYNAMIC SCENARIO

Suppose each rejected sample improves its feature by

ε(x) =

(cid:26)δ
δ

1
1

x
{
x
{

·
·

∈
∈

[τ0
[τ1

,
δ, τ0)
}
,
δ, τ1)
}

−
−

if z = 0,
if z = 1

Note that depending on the classifier we are using, the threshold pair (τ0, τ1) changes, thus the
formulation for ε(x) also changes. Here, we use εERM(x) to denote the improvement of features
given ERM classifier, and similarly define εEI(x), εBE(x) and εER(x).

27

Published as a conference paper at ICLR 2023

(x) = pz(x + εERM(x)) for z

Let pERM
0, 1
, which represent the data distribution after the
z
}
features are improved based on ERM classifier. Similarly, we define pEI
z (x), pBE
(x) for
EI/BE/ER classifiers, respectively. In the upcoming sections, we measure the total-variation (TV)
distance

z (x) and pER

∈ {

z

R|
between two groups after a single step of feature improvement, and provide how this measurement
differs for various classifiers.

−

dT V (p0, p1) =

p0(x)

dx
p1(x)
|

(cid:90)

1
2

D.2 EI VERSUS BE & ERM

D.2.1 FINDING THE OPTIMAL CLASSIFIER FOR EACH FAIRNESS CRITERION

Setup Let pz(x) be the data distribution of each group z
the case of each sample having one feature x, and the label is assigned as

0, 1
, shown in Fig. 9. We consider
}

∈ {

m/2
x
if z = 0
{
,
x
0
if z = 1
}
{
4 , P(y = 1
Note that we have P(y = 0
z = 0) = 1
z = 0) = 3
|
|
4 and P(z = 1) = 3
2 . We set P(z = 0) = 1
P(y = 1
z = 1) = 1
4 .
|

≥
≥

y =

,
}

(cid:26)1
1

4 , P(y = 0
z = 1) = 1
|

2 , and

Figure 9: The data distribution pz(x) for each group z
label y = 1, while samples with (

) sign have the true label y = 0.

∈ {

0, 1

. Samples with (+) sign have the true
}

−

ERM classifier We have

(τ ERM
0

, τ ERM
1

) =

(cid:17)

, 0

(cid:16) m
2

since this threshold pair has zero classification error.

BE classifier We have

(τ BE
0

, τ BE
1

) =

(cid:17)

, 0

(cid:16) m
2

since this threshold pair has zero classification error and satisfy the BE condition in (18).

EI classifier The optimal EI classifier for dataset in Fig. 9 is

(τ EI

0 , τ EI

1 ) = (0, 0).

(20)

(21)

1 ) = (0, 0) has error rate P(error) = P(z = 0)P(error
0 , τ EI
Proof. Note that (τ EI
|
1)P(error
4 + 3
1
z = 1) = 1
4 ·
4 ·
|
is having error rate less than 1
16 . Note that when
sufficient to consider cases when
| ≤

z = 0) + P(z =
16 . We prove that no other classifier satisfing EI condition in (15)
16 . Thus it is

m
6 . Combining this with the fact that

6 , the error rate is larger than 1

0 = 1

> m

τ1
|

τ1

|

|

28

9p0(x)xm2−m03m212m-+p1(x)xm−m0-+12m14mPublished as a conference paper at ICLR 2023

• r0(τ0) in (16) and r1(τ1) in (17) are monotonically decreasing,

• EI condition in (15) is satisfied when r0(τ0) = r1(τ1) holds,

• r0(τ0) = r1(τ1) holds for τ0 = τ1

m
2 ,

≤

we can see that the optimal EI classifier satisfies τ0 = τ1 and
classifiers is represented as P(error) = P(z = 0)P(error
|
1
4 ·
τ1
|

2m + 3
2 −
| ·
m
6 completes the proof.

m
τ1
6 . The error rate for these
|
z = 0) + P(z = 1)P(error
z = 1) =
|
1
2m . Plugging in τ0 = τ1 and optimizing the error probability over

4 · |

| ≤

( m

τ0)

| ≤

τ1

1

·

D.2.2 TOTAL-VARIATION DISTANCE BETWEEN TWO GROUPS

For the dataset given in Fig. 9, the total-variation distance between two groups for each classifier is:

) = 0.5,

dT V (pERM
0
dT V (pBE
dT V (pEI

, pERM
1
0 , pBE
0 , pEI

1 ) = 0.5,
1 ) = 0.125.

Proof. Since ERM solution is identical to BE solution, proving the above equation for BE and EI is
sufficient. Recall that the expression of BE/EI classifiers are in (20) and (21). Using this expression,
we can derive the distribution of each group:

pBE
0 (x) =

pBE
1 (x) =










3
4m ,
1
2m ,
1
4m ,
0,

1
m ,
1
2m ,
0,




pEI
0 (x) =

1
m ,
1
2m ,
1
4m ,
0,
pEI
1 (x) = pBE



1 (x)

∈
∈
∈
∈

∈
∈
∈

∈
∈
∈
∈

if x
if x
if x
if x

if x
if x
if x

if x
if x
if x
if x

x

∀

[
−

[ m
2 , m]
m, 0]
[
−
[m, 3m
2 ]
[0, m
2 ] or x /
∈
[0, m
2 ]
m
[ m
m,
2 ]
[
2 , m]
−
m
2 , 0] or x /
[
[
−
∈
−
[0, m
2 ]
[
m,
−
−
2 , 3m
[ m
2 ]
[
−

m
2 , 0] or x /
∈

[
−

m
2 ]

−

∪

m, 3m
2 ]

,

,

m, m]

,

m, 3m
2 ]

From this expression, we can derive the total-variation distance, which completes the proof.

D.3 EI VERSUS ER

D.3.1 FINDING THE OPTIMAL CLASSIFIER FOR EACH FAIRNESS CRITERION

Setup Let pz(x) be the data distribution of each group z
the case of each sample having one feature x, and the label is assigned as

0, 1
, show in Fig. 10. We consider
}

∈ {

y =

(cid:26)1
{
1
{

x
x

,
,

0
}
0
}

≥
≥

if z = 0
if z = 1

.

Note that we have P(y = 0
1) = 1

2 . We set P(z = 0) = P(z = 1) = 1
2 .

z = 0) = P(y = 0

|

z = 1) = P(y = 1

z = 0) = P(y = 1

|

z =

|

|

ER classifier The optimal ER classifier for the dataset in Fig. 10 is

(τ ER
0

, τ ER
1

) = (

9m, 0) .

−

29

(22)

Published as a conference paper at ICLR 2023

Figure 10: The data distribution pz(x) for each group z
true label y = 1, while samples with (

) sign have the true label y = 0.

0, 1
}

∈ {

. Samples with (+) sign have the

−

9m, 0) has error rate P(error) = P(error
Proof. Note that the accepting thresholds (
−
0) + P(error
2 = 1
1
1
2 + 0
·
constraint (19) is having error rate less than 1
6 .

z = 0)P(z =
6 . We prove that no other classifier satisfying ER

z = 1)P(z = 1) = 1

3 ·

|

|

The necessary condition for the classifier to have an error rate less than 1
that when τ0

2 ), we have the recourse of the group 0:

2 , m

m

6 is τ0, τ1

(
−

∈

m

2 , m

2 ). Note

(
−

∈

E

(cid:20)

(cid:21)

min
f (x+∆x)≥0.5

µ(∆x)

|

f (x) < 0.5, z = 0

= E [τ0
= τ0

−

x
−
E [x
(cid:124)

|
|

x < τ0, z = 0]
,
x < τ0, z = 0]
(cid:125)

(cid:123)(cid:122)
φ(τ0)

(23)

where

φ(τ0) = E [x
(cid:18)

|
x

= P

(cid:124)

∈

(cid:20)

−

10m,

x < τ0, z = 0]
19
2
(cid:123)(cid:122)
∆=λ
m
2

−

(cid:16)

x

(cid:104)

−

∈

+ P
(cid:124)

(cid:21)

|

(cid:19)

x < τ0, z = 0

(cid:20)

x

E

(cid:20)

x

|

∈

−

10m,

19
2

−

(cid:125)

(cid:21)

(24)

(cid:21)

m

, z = 0

m

(cid:105)

, τ0

|
(cid:123)(cid:122)
1−λ

x < τ0, z = 0

(cid:17)

(cid:125)

(cid:104)

x

E

(cid:104)

∈

−

m
2

x

|

(cid:105)

(cid:105)
, z = 0

,

, τ0

for all τ0
E (cid:2)x

x

|

∈

∈
(cid:2)
−

[
−
m
2 , τ0

m

2 ], where λ

(cid:2)
2 , m
[ 1
3 , 1].
−
(cid:3) , z = 0(cid:3), (24) is a decreasing function. Consequently,

Since E (cid:2)x

∈

∈

x

|

10m,

19

2 m(cid:3) , z = 0(cid:3) <

−

(cid:20)

x

E

1
3

x

|

39
12

−

m +

≤

=

(cid:20)

10m,

m
6

=

−

∈
τ0
3 −

19
2

−
τ0
3 −

41
12

m.

(cid:21)

(cid:21)

m

, z = 0

+

(cid:104)
x

E

2
3

x

|

∈

(cid:104)
−

m
2

(cid:105)

(cid:105)
, z = 0

, τ0

Thus, the recourse of group 0 is

(23)

2τ0
3

+

41
12

m

≥

37
12

m

≥

30

(25)

10p0(x)xm2023m-+p1(x)x−m0-+23mm2−m223m-−10m⏟m2Published as a conference paper at ICLR 2023

Meanwhile, when τ1

(

−

∈

m

2 , m
(cid:20)

2 ), we have the recourse of the group 1:
(cid:21)

min
f (x+∆x)≥0.5

µ(∆x)

|

f (x) < 0.5, z = 1

E

= E [τ1
= τ1

−

= τ1

−
τ1 + m
2

=

x
−
E [x
τ1

x < τ1, z = 0]
|
x < τ1, z = 0]
|
m
−
2

(cid:19)

(cid:18) m
4

,

3
4

∈

m

,

(26)

Therefore, combining (25) and (26) implies that when τ0, τ1
(
∈
−
cannot be satisfied. Thus, the way for achieving error rate < 1
6 is τ0 <
the recourse of group 0 which is written in (23) is a strictly increasing function when τ0 <
Therefore, there exists a unique classifier that achieves both EI while maintaining error rate < 1
can easily verify that (

2 ), the ER constraint (19)
m
2 and τ1 = 0. Note that
m
2 .
−
6 . One

9m, 0) is the optimal ER classifier.

−

m

2 , m

−

EI classifier We have

(τ EI
since this threshold pair has zero classification error and satisfies the EI constraint (15).

1 ) = (0, 0)

0 , τ EI

(27)

D.3.2 TOTAL-VARIATION DISTANCE BETWEEN TWO GROUPS

For the dataset given in Fig. 10, the total variation distance between two groups for EI and ER
classifier is

dT V (pER

0 , pER

1 ) =

dT V (pEI

0 , pEI

1 ) =

2
3
1
3

,

.

Proof. By (27) and (22),

pER
0 (x) =

pER
1 (x) =

pEI
0 (x) =

pEI
1 (x) =





















2
3m ,
2
3m ,
0,

2
3m ,
4
3m ,
0,

2
3m ,
4
3m ,
0,

2
3m ,
4
3m ,
0,

8m]

19m
2 ,
−
2 , m
m
2 ]

,

[
−
[
−

m,
[
−
[0, m
2 ]

−

m
2 ]

,

10m,
[
−
[0, m
2 ]

−

19m
2 ]

,

m,
[
−
[0, m
2 ]

−

m
2 ]

.

if x
if x
o.w.

if x
if x
o.w.

if x
if x
o.w.

if x
if x
o.w.

∈
∈

∈
∈

∈
∈

∈
∈

For this expression, we can derive the total-variation distance, which completes the proof.

31

