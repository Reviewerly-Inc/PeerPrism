2
2
0
2

b
e
F
7

]

G
L
.
s
c
[

1
v
1
7
0
3
0
.
2
0
2
2
:
v
i
X
r
a

Published as a conference paper at ICLR 2022

DISTRIBUTIONALLY ROBUST FAIR
PRINCIPAL COMPONENTS VIA GEODESIC DESCENTS

Hieu Vu, Toan Tran
VinAI Research, Vietnam

Man-Chung Yue
Hong Kong Polytechnic University

Viet Anh Nguyen
VinAI Research, Vietnam

ABSTRACT

Principal component analysis is a simple yet useful dimensionality reduction tech-
nique in modern machine learning pipelines. In consequential domains such as
college admission, healthcare and credit approval, it is imperative to take into
account emerging criteria such as the fairness and the robustness of the learned
projection. In this paper, we propose a distributionally robust optimization prob-
lem for principal component analysis which internalizes a fairness criterion in the
objective function. The learned projection thus balances the trade-off between
the total reconstruction error and the reconstruction error gap between subgroups,
taken in the min-max sense over all distributions in a moment-based ambiguity
set. The resulting optimization problem over the Stiefel manifold can be efﬁ-
ciently solved by a Riemannian subgradient descent algorithm with a sub-linear
convergence rate. Our experimental results on real-world datasets show the merits
of our proposed method over state-of-the-art baselines.

1

INTRODUCTION

Machine learning models are ubiquitous in our daily lives and supporting the decision-making pro-
cess in diverse domains. With their ﬂourishing applications, there also surface numerous concerns
regarding the fairness of the models’ outputs (Mehrabi et al., 2021). Indeed, these models are prone
to biases due to various reasons (Barocas et al., 2018). First, the collected training data is likely
to include some demographic disparities due to the bias in the data acquisition process (e.g., con-
ducting surveys on a speciﬁc region instead of uniformly distributed places), or the imbalance of
observed events at a speciﬁc period of time. Second, because machine learning methods only care
about data statistics and are objective driven, groups that are under-represented in the data can be
neglected in exchange for a better objective value. Finally, even human feedback to the predictive
models can also be biased, e.g., click counts are human feedback to recommendation systems but
they are highly correlated with the menu list suggested previously by a potentially biased system.
Real-world examples of machine learning models that amplify biases and hence potentially cause
unfairness are commonplace, ranging from recidivism prediction giving higher false positive rates
for African-American1 to facial recognition systems having large error rate for women2.

To tackle the issue, various fairness criteria for supervised learning have been proposed in the lit-
erature, which encourage the (conditional) independence of the model’s predictions on a particular
sensitive attribute (Dwork et al., 2012; Hardt et al., 2016b; Kusner et al., 2017; Chouldechova, 2017;
Verma & Rubin, 2018; Berk et al., 2021). Strategies to mitigate algorithmic bias are also investi-
gated for all stages of the machine learning pipelines (Berk et al., 2021). For the pre-processing
steps, (Kamiran & Calders, 2012) proposed reweighting or resampling techniques to achieve sta-
tistical parity between subgroups; in the training steps, fairness can be encouraged by adding con-
straints (Donini et al., 2018) or regularizing the original objective function (Kamishima et al., 2012;
Zemel et al., 2013); and in the post-processing steps, adjusting classiﬁcation threshold by examining
black-box models over a holdout dataset can be used (Hardt et al., 2016b; Wei et al., 2019).

Since biases may already exist in the raw data, it is reasonable to demand machine learning pipelines
to combat biases as early as possible. We focus in this paper on the Principal Component Analy-
sis (PCA), which is a fundamental dimensionality reduction technique in the early stage of the

1 https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing
2 https://news.mit.edu/2018/study-ﬁnds-gender-skin-type-bias-artiﬁcial-intelligence-systems-0212

1

Published as a conference paper at ICLR 2022

pipelines (Pearson, 1901; Hotelling, 1933). PCA ﬁnds a linear transformation that embeds the origi-
nal data into a lower-dimensional subspace that maximizes the variance of the projected data. Thus,
PCA may amplify biases if the data variability is different between the majority and the minority
subgroups, see an example in Figure 1. A naive approach to promote fairness is to train one inde-
pendent transformation for each subgroup. However, this requires knowing the sensitive attribute of
each sample, which would raise disparity concerns. On the contrary, using a single transformation
for all subgroups is “group-blinded” and faces no discrimination problem (Lipton et al., 2018).

Learning a fair PCA has attracted attention from many ﬁelds from machine learning, statistics to
signal processing. Samadi et al. (2018) and Zalcberg & Wiesel (2021) propose to ﬁnd the principal
components that minimize the maximum subgroup reconstruction error; the min-max formulations
can be relaxed and solved as semideﬁnite programs. Olfat & Aswani (2019) propose to learn a
transformation that minimizes the possibility of predicting the sensitive attribute from the projected
data. Apart from being a dimensionality reduction technique, PCA can also be thought of as a
representation learning toolkit. Viewed in this way, we can also consider a more general family of
fair representation learning methods that can be applied before any further analysis steps. There
are a number of works develop towards this idea (Kamiran & Calders, 2012; Zemel et al., 2013;
Calmon et al., 2017; Feldman et al., 2015; Beutel et al., 2017; Madras et al., 2018; Zhang et al.,
2018; Tantipongpipat et al., 2019), which apply a multitude of fairness criteria.

In addition, we also focus on the robustness criteria for the linear transformation. Recently, it has
been observed that machine learning models are susceptible to small perturbations of the data (Good-
fellow et al., 2014; Madry et al., 2017; Carlini & Wagner, 2017). These observations have fuelled
many defenses using adversarial training (Akhtar & Mian, 2018; Chakraborty et al., 2018) and dis-
tributionally robust optimization (Rahimian & Mehrotra, 2019; Kuhn et al., 2019; Blanchet et al.,
2021).

Contributions. This paper blends the ideas from the ﬁeld of fairness in artiﬁcal intelligence and
distributionally robust optimization. Our contributions can be described as follows.

• We propose the fair principal components which balance between the total reconstruction error
and the absolute gap of reconstruction error between subgroups. Moreover, we also add a layer
of robustness to the principal components by considering a min-max formulation that hedges
against all perturbations of the empirical distribution in a moment-based ambiguity set.

• We provide the reformulation of the distributionally robust fair PCA problem as a ﬁnite-
dimensional optimization problem over the Stiefel manifold. We provide a Riemannian gradient
descent algorithm and show that it has a sub-linear convergence rate.

Figure 1 illustrates the qualitative comparison
between (fair) PCA methods and our proposed
method on a 2-dimensional toy example. The
majority group (blue dots) spreads on the hor-
izontal axis, while the minority group (yellow
triangles) spreads on the slanted vertical axis.
The nominal PCA (red) captures the majority
direction to minimize the total error, while the
fair PCA of Samadi et al. (2018) returns the
diagonal direction to minimize the maximum
subgroup error. Our fair PCA can probe the
full spectrum in between these two extremes by
sweeping through our penalization parameters
appropriately.
If we do not penalize the error
gap between subgroups, we recover the PCA
method; if we penalize heavily, we recover the
fair PCA of Samadi et al. (2018). Extensive nu-
merical results on real datasets are provided in
Section 5. Proofs are relegated to the appendix.

Figure 1: Nominal PCA (red arrow), fair PCA by
Samadi et al. (2018) (green arrow), and our spec-
trum of fair PCA (shorter arrows). Arrows show
directions and are not normalized to unit length.

2

Published as a conference paper at ICLR 2022

2 FAIR PRINCIPAL COMPONENT ANALYSIS

2.1 PRINCIPAL COMPONENT ANALYSIS

We ﬁrst brieﬂy revisit the classical PCA. Suppose that we are given a collection of N i.i.d. samples
i=1 generated by some underlying distribution P. For simplicity, we assume that both the
{ˆxi}N
empirical and population mean are zero vectors. The goal of PCA is to ﬁnd a k-dimensional linear
subspace of Rd that explains as much variance contained in the data {ˆxi}N
i=1 as possible, where k <
d is a given integer. More precisely, we parametrize k-dimensional linear subspaces by orthonormal
matrices, i.e., matrices whose columns are orthogonal and have unit Euclidean norm. Given any
such matrix V , the associated k-dimensional subspace is the one spanned by the columns of V .
The projection matrix onto the subspace is V V (cid:62), and hence the variance of the projected data is
given by tr (cid:0)V V (cid:62)ΞΞ(cid:62)(cid:1), where Ξ = [ˆx1, · · · , ˆxN ] ∈ Rd×N is the data matrix. By a slight abuse of
terminology, sometimes we refer to V as the projection matrix. The problem of PCA then reads

max
V ∈Rd×k,V (cid:62)V =Ik

tr (cid:0)V V (cid:62)ΞΞ(cid:62)(cid:1) .

(1)

For any vector X ∈ Rd and orthonormal matrix V , denote by (cid:96)(V, X) the reconstruction error, i.e.,

The problem of PCA can alternatively be formulated as a stochastic optimization problem

(cid:96)(V, X) = (cid:107)X − V V (cid:62)X(cid:107)2

2 = X (cid:62)(Id − V V (cid:62))X.

min
V ∈Rd×k,V (cid:62)V =Ik

EˆP[(cid:96)(V, X)],

(2)

i=1 and X ∼ ˆP. It is well-
where ˆP is the empirical distribution associated with the samples {ˆxi}N
known that PCA admits an analytical solution. In particular, the optimal solution to problem (2) (and
also problem (1)) is given by any orthonormal matrix whose columns are the eigenvectors associated
with the k largest eigenvalues of the sample covariance matrix ΞΞ(cid:62).

2.2 FAIR PRINCIPAL COMPONENT ANALYSIS

In the fair PCA setting, we are also given a discrete sensitive attribute A ∈ A, where A may represent
features such as race, gender or education. We consider binary attribute A and let A = {0, 1}. A
straightforward idea to deﬁne fairness is to require the (strict) balance of a certain objective between
the two groups. For example, this is the strategy in Hardt et al. (2016a) for developing fair supervised
learning algorithms. A natural objective to balance in the PCA context is the reconstruction error.
Deﬁnition 2.1 (Fair projection). Let Q be an arbitrary distribution of (X, A). A projection matrix
V ∈ Rd×k is fair relative to Q if the conditional expected reconstruction error is equal between
subgroups, i.e., EQ[(cid:96)(V, X)|A = a] = EQ[(cid:96)(V, X)|A = a(cid:48)] for any (a, a(cid:48)) ∈ A × A.

Unfortunately, Deﬁnition 2.1 is too stringent: for a general probability distribution Q, it is possible
that there exists no fair projection matrix V .
Proposition 2.2 (Impossibility result). For any distribution Q on X ×A, there exists a fair projection
matrix V ∈ Rd×k relative to Q if and only if rank(EQ[XX (cid:62)|A = 0] − EQ[XX (cid:62)|A = 1]) ≤ k.

One way to circumvent the impossibility result is to relax the requirement of strict balance to ap-
proximate balance. In other words, an inequality constraint of the following form is imposed:

|EQ[(cid:96)(V, X)|A = a] − EQ[(cid:96)(V, X)|A = a(cid:48)]| ≤ (cid:15)

∀(a, a(cid:48)) ∈ A × A,

where (cid:15) > 0 is some prescribed fairness threshold. This approach has been adopted in other fair
machine learning settings, see Donini et al. (2018) and Agarwal et al. (2019) for example.

In this paper, instead of imposing the fairness requirement as a constraint, we penalize the unfairness
in the objective function. Speciﬁcally, for any projection matrix V , we deﬁne the unfairness as the
absolute difference between the conditional loss between two subgroups:

U(V, Q) (cid:44) |EQ[(cid:96)(V, X)|A = 0] − EQ[(cid:96)(V, X)|A = 1]|.

We thus consider the following fairness-aware PCA problem

min
V ∈Rd×k, V (cid:62)V =Ik

EˆP[(cid:96)(V, X)] + λU(V, ˆP),

(3)

where λ ≥ 0 is a penalty parameter to encourage fairness. Note that for fair PCA, the dataset is
{(ˆxi, ˆai)}N

i=1 and hence the empirical distribution ˆP is given by ˆP = 1

i=1 δ(ˆxi,ˆai).

(cid:80)N

N

3

Published as a conference paper at ICLR 2022

3 DISTRIBUTIONALLY ROBUST FAIR PCA

The weakness of empirical distribution-based stochastic optimization has been well-documented,
see (Smith & Winkler, 2006; Homem-de Mello & Bayraksan, 2014). In particular, due to overﬁt-
ting, the out-of-sample performance of the decision, prediction, or estimation obtained from such a
stochastic optimization model is unsatisfactory, especially in the low sample size regime. Ideally,
we could improve the performance by using the underlying distribution P instead of the empirical
distribution ˆP. But the underlying distribution P is unavailable in most practical situations, if not all.
Distributional robustiﬁcation is an emerging approach to handle this issue and has been shown to
deliver promising out-of-sample performance in many applications (Delage & Ye, 2010; Namkoong
& Duchi, 2017; Kuhn et al., 2019; Rahimian & Mehrotra, 2019). Motivated by the success of distri-
butional robustiﬁcation, especially in machine learning Nguyen et al. (2019); Taskesen et al. (2021),
we propose a robustiﬁed version of model (3), called the distributionally robust fairness-aware PCA:

min
V ∈Rd×k,V (cid:62)V =Ik

sup
Q∈B(ˆP)

EQ[(cid:96)(V, X)] + λU(V, Q),

(4)

where B(ˆP) is a set of probability distributions similar to the empirical distribution ˆP in a certain
sense, called the ambiguity set. The empirical distribution ˆP is also called the nominal distribu-
tion. Many different ambiguity sets have been developed and studied in the optimization literature,
see Rahimian & Mehrotra (2019) for an extensive overview.

3.1 THE WASSERSTEIN-TYPE AMBIGUITY SET

To present our ambiguity set and main results, we need to introduce some deﬁnitions and notations.
Deﬁnition 3.1 (Wasserstein-type divergence). The divergence W between two probability distribu-
tions Q1 ∼ (µ1, Σ1) ∈ Rd × Sd

+ and Q2 ∼ (µ2, Σ2) ∈ Rd × Sd

+ is deﬁned as

W(cid:0)Q1 (cid:107) Q2

(cid:1) (cid:44) (cid:107)µ1 − µ2(cid:107)2

2 + tr

(cid:16)

Σ1 + Σ2 − 2(cid:0)Σ

1
2

2 Σ1Σ

1
2
2

2 (cid:17)
(cid:1) 1

.

The divergence W coincides with the squared type-2 Wasserstein distance between two Gaus-
sian distributions N (µ1, Σ1) and N (µ2, Σ2) (Givens & Shortt, 1984). One can readily show that
W is non-negative, and it vanishes if and only if (µ1, Σ1) = (µ2, Σ2), which implies that Q1
and Q2 have the same ﬁrst- and second-moments. Recently, distributional robustiﬁcation with
Wasserstein-type ambiguity sets has been applied widely to various problems including domain
adaption (Taskesen et al., 2021), risk measurement (Nguyen et al., 2021b) and statistical estima-
tion (Nguyen et al., 2021a). The Wasserstein-type divergence in Deﬁnition 3.1 is also related to the
theory of optimal transport with its applications in robust decision making (Mohajerin Esfahani &
Kuhn, 2018; Blanchet & Murthy, 2019; Yue et al., 2021) and potential applications in fair machine
learning (Taskesen et al., 2020; Si et al., 2021; Wang et al., 2021).
Recall that the nominal distribution is ˆP = 1
N
bution given A = a is given by

i=1 δ(ˆxi,ˆai). For any a ∈ A, its conditional distri-

(cid:80)N

ˆPa =

1
|Ia|

(cid:88)

i∈Ia

δxi , where Ia (cid:44) {i ∈ {1, . . . , N } : ai = a}.

We also use (ˆµa, ˆΣa) to denote the empirical mean vector and covariance matrix of X given A = a:

ˆµa = EˆPa

[X] = EˆP[X|A = a]

and

ˆΣa + ˆµa ˆµ(cid:62)

a = EˆPa

[XX (cid:62)] = EˆP[XX (cid:62)|A = a].

For any a ∈ A, the empirical marginal distribution of A is denoted by ˆpa = |Ia|/N .

Finally, for any set S, we use P(S) to denote the set of all probability distributions supported on S.
For any integer k, the k-by-k identity matrix is denoted Ik. We then deﬁne our ambiguity set as

B(ˆP) (cid:44)






Q ∈ P(X × A) :

∃Qa ∈ P(X ) such that:
Q(X × {a}) = ˆpaQa(X) ∀X ⊆ Rd, a ∈ A
W(Qa, ˆPa) ≤ εa ∀a ∈ A






,

(5)

4

Published as a conference paper at ICLR 2022

where Qa is the conditional distribution of X|A = a. Intuitively, each Q ∈ B(ˆP) is a joint distribu-
tion of the random vector (X, A), formed by taking a mixture of conditional distributions Qa with
mixture weight ˆpa. Each conditional distribution Qa is constrained in an εa-neighborhood of the
nominal conditional distribution ˆPa with respect to the W divergence. Because the loss function (cid:96)
is a quadratic function of X, the (conditional) expected losses only involve the ﬁrst two moments of
X, and thus prescribing the ambiguity set using W would sufﬁce for the purpose of robustiﬁcation.

3.2 REFORMULATION

We now present the reformulation of problem (4) under the ambiguity set B(ˆP).
Theorem 3.2 (Reformulation). Suppose that for any a ∈ A, either of the following two conditions
holds:

(i) Marginal probability bounds: 0 ≤ λ ≤ ˆpa,

(ii) Eigenvalue bounds: the empirical second moment matrix ˆMa = 1
Na

i∈Ia
j=1 σj( ˆMa) ≥ εa, where σj( ˆMa) is the j-th smallest eigenvalues of ˆMa.

(cid:80)d−k

(cid:80)

ˆxi ˆx(cid:62)

i satisﬁes

Then problem (4) is equivalent to

min
V ∈Rd×k,V (cid:62)V =Ik

max{J0(V ), J1(V )},

where for each (a, a(cid:48)) ∈ {(0, 1), (1, 0)}, the function Ja is deﬁned as
(cid:113)(cid:10)Id − V V (cid:62), ˆMa(cid:48)

(cid:113)(cid:10)Id − V V (cid:62), ˆMa

Ja(V ) = κa + θa

(cid:11) + ϑa(cid:48)

(cid:11) + (cid:10)Id − V V (cid:62), Ca

(6a)

(cid:11),

(6b)

and the parameters κ ∈ R, θ ∈ R, ϑ ∈ R and C ∈ Sd

κa = (ˆpa + λ)εa + (ˆpa(cid:48) − λ)εa(cid:48),
Ca = (ˆpa + λ) ˆMa + (ˆpa(cid:48) − λ) ˆMa(cid:48).

+ are deﬁned as
√

θa = 2|ˆpa + λ|

εa, ϑa(cid:48) = 2|ˆpa(cid:48) − λ|

√

εa(cid:48),

(6c)

We now brieﬂy explain the steps that lead to the results in Theorem 3.2. Letting

J0(V ) = sup

Q∈B(ˆP)

J1(V ) = sup

Q∈B(ˆP)

(ˆp0 + λ)EQ[(cid:96)(V, X)|A = 0] + (ˆp1 − λ)EQ[(cid:96)(V, X)|A = 1],

(ˆp0 − λ)EQ[(cid:96)(V, X)|A = 0] + (ˆp1 + λ)EQ[(cid:96)(V, X)|A = 1],

then by expanding the term U(V, Q) using its deﬁnition, problem (4) becomes

min
V ∈Rd×k,V (cid:62)V =Ik

max{J0(V ), J1(V )}.

Leveraging the deﬁnition the ambiguity set B(ˆP), for any pair (a, a(cid:48)) ∈ {(0, 1), (1, 0)}, we can
decompose Ja into two separate supremum problems as follows

Ja(V ) =

sup
Qa:W(Qa,ˆPa)≤εa

(ˆpa + λ)EQa [(cid:96)(V, X)] +

sup
a(cid:48) ,ˆP

Q

a(cid:48) :W(Q

(ˆpa(cid:48) − λ)EQ

a(cid:48) [(cid:96)(V, X)].

a(cid:48) )≤εa(cid:48)

The next proposition asserts that each individual supremum in the above expression admits an ana-
lytical expression.
Proposition 3.3 (Reformulation). Fix a ∈ A. For any υ ∈ R, εa ∈ R+, it holds that

sup
Qa:W(Qa,ˆPa)≤εa

υEQa [(cid:96)(V, X)]

(cid:18)(cid:113)(cid:10)Id − V V (cid:62), ˆMa
(cid:18)(cid:113)(cid:10)Id − V V (cid:62), ˆMa

(cid:11) +

√

εa

(cid:11) −

√

εa

(cid:19)2

(cid:19)2






υ

υ

0

=

5

if υ ≥ 0,

if υ < 0 and (cid:10)Id − V V (cid:62), ˆMa
if υ < 0 and (cid:10)Id − V V (cid:62), ˆMa

(cid:11) ≥ εa,
(cid:11) < εa.

Published as a conference paper at ICLR 2022

The proof of Theorem 3.2 now follows by applying Proposition 3.3 to each term in Ja, and balance
the parameters to obtain (6c). A detailed proof is relegated to the appendix. In the next section, we
study an efﬁcient algorithm to solve problem (6a).
Remark 3.4 (Recovery of the nominal PCA). If λ = 0 and εa = 0 ∀a ∈ A, our formulation (4)
becomes the standard PCA problem (2). In this case, our robust fair principal components reduce
to the standard principal components. On the contrary, existing fair PCA methods such as Samadi
et al. (2018) and Olfat & Aswani (2019) cannot recover the standard principal components.

4 RIEMANNIAN GRADIENT DESCENT ALGORITHM

The distributionally robust fairness-aware PCA problem (4) is originally an inﬁnite-dimensional
min-max problem. Indeed, the inner maximization problem in (4) optimizes over the space of prob-
ability measures. Thanks to Theorem 3.2, it is reduced to the simpler ﬁnite-dimensional minimax
problem (6a), where the inner problem is only a maximization over two points. Problem (6a) is,
however, still challenging as it is a non-convex optimization problem over a non-convex feasible re-
gion deﬁned by the orthogonality constraint V (cid:62)V = Id. The purpose of this section is to devise an
efﬁcient algorithm for solving problem (6a) to local optimality based on Riemannian optimization.

4.1 REPARAMETRIZATION

As mentioned above, the non-convexity of problem (6a) comes from both the objective function and
the feasible region. It turns out that we can get rid of the non-convexity of the objective function
via a simple change of variables. To see that, we let U ∈ Rd×(d−k) be an orthonormal matrix
complement to V , that is, U and V satisfy U U (cid:62) + V V (cid:62) = Id. Thus, we can express the objective
function J via

J(V ) = F (U ) (cid:44) max{F0(U ), F1(U )},

where for (a, a(cid:48)) ∈ {(0, 1), (1, 0)}, the function Fa is deﬁned as

Fa(U ) (cid:44) κa + θa

(cid:113)(cid:10)U U (cid:62), ˆMa

(cid:11) + ϑa(cid:48)

(cid:113)(cid:10)U U (cid:62), ˆMa(cid:48)

(cid:11) + (cid:10)U U (cid:62), Ca

(cid:11).

Moreover, letting M (cid:44) {U ∈ Rd×(d−k) : U (cid:62)U = Id−k}, we can re-express problem (6a) as

min
U ∈M

F (U ).

(7)

The set M of problem (7) is a Riemannian manifold, called the Stiefel manifold (Absil et al., 2007,
Section 3.3.2). It is then natural to solve (7) using a Riemannian optimization algorithms (Absil
et al., 2007).
In fact, problem (6a) itself (before the change of variables) can also be cast as a
Riemannian optimization problem over another Stiefel manifold. The change of variables above
might seem unnecessary. Nonetheless, the upshot of problem (7) is that the objective function F is
convex (in the traditional sense). This faciliates the application of the theoretical and algorithmic
framework developed in Li et al. (2021) for (weakly) convex optimization over the Stiefel manifolds.

4.2 THE RIEMANNIAN SUBGRADIENT

Note that the objective function F is non-smooth since it is deﬁned as the maximum of two func-
tions F0 and F1. To apply the framework in Li et al. (2021), we need to compute the Riemannian
subgradient of the objective function F . Since the Stiefel manifold M is an embedded manifold in
Euclidean space, the Riemannian subgradient of F at any point U ∈ M is given by the orthogonal
projection of the usual Euclidean subgradient onto the tangent space of the manifold M at the point
U , see Absil et al. (2007, Section 3.6.1) for example.
Lemma 4.1. For any point U ∈ M, let3 aU ∈ arg maxa∈{0,1} Fa(U ) and a(cid:48)
Riemannian subgradient of the objective function F at the point U is given by

U = 1 − aU . Then, a

gradF (U ) = (Id − U U (cid:62))





θaU
(cid:113)(cid:10)U U (cid:62), ˆMaU

(cid:11)

ˆMaU U +

ϑa(cid:48)
(cid:113)(cid:10)U U (cid:62), ˆMa(cid:48)

U

U



ˆMa(cid:48)

U

(cid:11)

U + 2CaU U

 .

3 It is possible that the maximizer is not unique. In that case, choosing aU to be either 0 or 1 would work.

6

Published as a conference paper at ICLR 2022

4.3 RETRACTIONS

Another important instrument required by the framework in Li et al. (2021) is a retraction of the
Stiefel manifold M. At each iteration, the point U − γ∆ obtained by moving from the current
iterate U in the opposite direction of the Riemannian gradient ∆ may not lie on the manifold in
general, where γ > 0 is the stepsize.
In Riemannian optimization, this is circumvented by the
concept of retraction. Given a point U ∈ M on the manifold, the Riemannian gradient ∆ ∈
TU M (which must lie in the tangent space TU M) and a stepsize γ, the retraction map Rtr deﬁnes a
point RtrU (−γ∆) which is guaranteed to lie on the manifold M. Roughly speaking, the retraction
RtrU ( · ) approximates the geodesic curve through U along the input tangential direction. For a
formal deﬁnition of retractions, we refer the readers to (Absil et al., 2007, Section 4.1). In this
paper, we focus on the following two commonly used retractions for Stiefel manifolds. The ﬁrst one
is the QR decomposition-based retraction using the Q-factor qf( · ) in the QR decomposition:

Rtrqf

U (∆) = qf(U + ∆), U ∈ M, ∆ ∈ TU M.

The second one is the polar decomposition-based retraction
U (∆) = (U + ∆)(Id−k + ∆(cid:62)∆)− 1

Rtrpolar

2 , U ∈ M, ∆ ∈ TU M.

(8)

4.4 ALGORITHM AND CONVERGENCE GUARANTEES

Associated with any choice of retraction Rtr is a concrete instantiation of the Riemannian subgradi-
ent descent algorithm for our problem (7), which is presented in Algorithm 1 with speciﬁc choice of
the stepsizes γt motivated by the theoretical results of (Li et al., 2021).

Algorithm 1 Riemannian Subgradient Descent for (7)

1: Input: An initial point U0, a number of iterations τ and a retraction Rtr : (U, ∆) (cid:55)→ RtrU (∆).
2: for t = 0, 1, . . . , τ − 1, do
3:
4:

Find at (cid:44) arg maxa∈{0,1}{Fa(Ut)}.
Compute the Riemannian subgradient ∆t = gradF (Ut) using the formula

∆t = (I − UtUt

(cid:62))





θat
(cid:113)(cid:10)UtU (cid:62)
t , ˆMat

(cid:11)

ˆMatUt +

ϑa(cid:48)
(cid:113)(cid:10)UtU (cid:62)
t , ˆMa(cid:48)

t

t



ˆMa(cid:48)

t

(cid:11)

Ut + 2CatUt

 .

Set Ut+1 = RtrUt(−γt∆t), where the step-size γt ≡ 1√

τ +1 is constant.

5:
6: end for
7: Output: Uτ .

We now study the convergence guarantee of Algorithm 1. The following lemma shows that the
objective function F is Lipschitz continuous (with respect to the Riemannian metric on the Stiefel
manifold M) with an explicit Lipschitz constant L.
Lemma 4.2 (Lipschitz continuity). The function F is L-Lipschitz continuous on M, where L > 0
is given by

(cid:40)

L (cid:44) max

θ0

σmax( ˆM0)
(cid:113)
σmin( ˆM0)

, θ1

σmax( ˆM1)
(cid:113)
σmin( ˆM1)

, ϑ0

√
2

d − kσmax(C0), 2

√

d − kσmax(C1)

, ϑ1

σmax( ˆM1)
(cid:113)
σmin( ˆM1)

,

(9)

σmax( ˆM0)
(cid:113)
σmin( ˆM0)
(cid:41)
.

We now proceed to show that Algorithm 1 enjoys a sub-linear convergence rate. To state the result,
we deﬁne the Moreau envelope

Fµ(U ) (cid:44) min
U (cid:48)∈M

(cid:26)

F (U (cid:48)) +

(cid:107)U (cid:48) − U (cid:107)2
F

(cid:27)

,

1
2µ

7

Published as a conference paper at ICLR 2022

where (cid:107) · (cid:107)F denotes the Frobenius norm of a matrix. Also, to measure the progress of the algorithm,
we need to introduce the proximal mapping on the Stiefel manifold (Li et al., 2021):

proxµF (U ) ∈ arg min
U (cid:48)∈M

(cid:26)

F (U (cid:48)) +

(cid:107)U (cid:48) − U (cid:107)2
F

(cid:27)

.

1
2µ

From Li et al. (2021, Equation (22)), we have that

(cid:107)gradF (U )(cid:107)F ≤

(cid:13)proxµF (U ) − U (cid:13)
(cid:13)
(cid:13)F
µ

(cid:44) gapµ(U ).

Therefore, the number gapµ(U ) is a good candidate to quantify the progress of optimization algo-
rithms for solving problem (7).
Theorem 4.3 (Convergence guarantee). Let {Ut}t=1,...,τ be the sequence of iterates generated by
Algorithm 1. Suppose that µ = 1/4L, where L is the Lipschitz constant of F in (9). Then, we have

min
t=0,...,τ

gapµ(Ut) ≤

2(cid:112)Fµ(U0) − minU Fµ(U ) + 2L3(L + 1)
(τ + 1)1/4

.

5 NUMERICAL EXPERIMENTS

We compare our proposed method, denoted RFPCA, against two state-of-the-art methods for fair
PCA: 1) FairPCA Samadi et al. (2018)4, and 2) CFPCA Olfat & Aswani (2019)5 with both cases:
only mean constraint, and both mean and covariance constraints. We consider a wide variety of
datasets with ranging sample sizes and number of features. Further details about the datatasets can
be found in Appendix C. The code for all experiments is available in supplementary materials. We
include here some details about the hyper-parameters that we search in the cross-validation steps.

• RFPCA. We notice that the neighborhood size εa should be inversely proportional to the size of
subgroup a. Indeed, a subgroup with large sample size is likely to have more reliable estimate of
the moment information. Then we parameterize the neighborhood size εa by a common scalar
α, and we have εa = α/
Na, where Na is the number of samples in group a. We search
α ∈ {0.05, 0.1, 0.15} and λ ∈ {0., 0.5, 1., 1.5, 2.0, 2.5}. For better convergence quality, we set
the number of iteration for our subgradient descent algorithm to τ = 1000 and also repeat the
Riemannian descent for 20 randomly generated initial point U0.

√

• FairPCA. According to Samadi et al. (2018), we only need tens of iterations for the multiplica-
tive weight algorithm to provide good-quality solution; however, to ensure a fair comparison, we
set the number of iterations to 1000 for the convergence guarantee. We search the learning rate
η of the algorithm from set of 17 values evenly spaced in [0.25, 4.25] and {0.1}.

• CFPCA. Following Olfat & Aswani (2019), for the mean-constrained version of CFPCA, we
search δ from {0., 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9}, and for both the mean and covariance con-
strained version, we ﬁx δ = 0 while searching µ in {0.0001, 0.001, 0.01, 0.05, 0.5}.

Trade-offs. First, we examine the trade-off between the total reconstruction error and the gap be-
In this experiment, we only compare our model with FairPCA and
tween the subgroup error.
CFPCA mean-constraint version. We plot a pareto curve for each of them over the two criteria with
different hyper-parameters (hyper-parameters test range are mentioned above). The whole datasets
are used for training and evaluation. The results averaged over 5 runs are shown in Figure 2.

In testing methods with different principal components, we ﬁrst split each dataset into training set
and test set with equal size (50% each), the projection matrix of each method is learned from training
set and tested over both sets. In this case, we only compare our method with traditional PCA and
FairPCA method. We ﬁx one set hyper-parameters for each method. For FairPCA, we set η = 0.1
and for RFPCA we set α = 0.15, λ = 0.5, others hyper-parameters are kept as discussed before.
The results are averaged over 5 different splits. Figure 3 shows the consistence of our method
performing fair projections over different values of k. Our method (cross) exhibits smaller gap of
subgroup errors. More results and discussions on the effect of ε can be found in Appendix D.2.

4 https://github.com/samirasamadi/Fair-PCA 5 https://github.com/molfat66/FairML

8

Published as a conference paper at ICLR 2022

Figure 2: Pareto curves on Default Credit
dataset (all data) with 3 principal components

Figure 3: Subgroup average error with different k
on Biodeg dataset (Out-of-sample).

Cross-validations. Next, we report the performance of all methods based on three criteria: absolute
difference between average reconstruction error between groups (ABDiff.), average reconstruction
error of all data (ARE.), and the fairness criterion deﬁned by Olfat & Aswani (2019) with respect
to a linear SVM’s classiﬁer family ((cid:52)FLin).6 Due to the space constraint, we only include the
ﬁrst two criteria in the main text, see Appendix 4 for full results. To emphasize the generalization
capacity of each algorithm, we split each dataset into a training set and a test set with ratio of
30% − 70% respectively, and only extract top three principal components from the training set.
We ﬁnd the best hyper-parameters by 3-fold cross validation, and prioritize the one giving minimum
value of the summation (ABDiff.+ARE.). The results are averaged over 10 different training-testing
splits. We report the performance on both training set (In-sample data) and test set (Out-of-sample
data). The details results for Out-of-sample data is given in Table 1, more details about settings and
performance can be found at Appendix D.

Results. Our proposed RFPCA method outperforms on 11 out of 15 datasets in terms of the subgroup
error gap ABDiff, and 9 out of 15 with the totall error ARE. criterion. There are 5 datasets that
RFPCA gives the best results for both criteria, and for the remaining datasets, RFPCA has small
performance gaps compared with the best method.

Table 1: Out-of-sample errors on real datasets. Bold indicates the lowest error for each dataset.

Dataset
Default Credit
Biodeg
E. Coli
Energy
German Credit
Image
Letter
Magic
Parkinsons
SkillCraft
Statlog
Steel
Taiwan Credit
Wine Quality
LFW

RFPCA

FairPCA

ABDiff.
0.9483
23.0066
1.1500
0.0125
2.0588
0.7522
0.1712
1.8314
0.3273
0.7669
0.0838
1.1472
0.5523
0.6359
0.4463

ARE.
10.3995
33.8571
1.7210
0.2238
43.9032
6.0199
7.4176
3.9094
5.0597
8.2828
3.0998
12.5944
10.9845
4.2801
7.6229

ABDiff.
1.4401
27.5159
1.5280
0.0138
1.3670
1.6129
1.2489
2.9405
0.8678
0.7771
0.3356
1.2208
0.5710
0.3046
0.5340

ARE.
10.4439
34.6184
2.4799
0.2225
44.0064
10.2616
7.4470
3.3815
4.9044
8.2494
7.9734
12.3096
10.9415
6.0936
7.6361

CFPCA-Mean Con.
ARE.
ABDiff.
10.9451
0.9367
37.6052
29.1728
2.9466
1.1005
2.7318
0.1229
43.9648
1.7845
14.3725
1.1499
8.7445
0.4427
4.2105
5.5790
5.7260
3.3804
9.9484
1.0283
10.8263
0.4476
16.4015
4.8710
13.0437
0.5744
6.1118
1.5020

CFPCA - Both Con.
ARE.
ABDiff.
22.0310
3.3359
50.7090
37.9533
5.6674
5.1275
7.9511
0.1001
49.5014
1.4955
19.3356
4.7013
15.1779
0.5743
9.0064
8.7810
19.7001
18.3312
15.9751
1.2849
35.8268
13.8437
25.8953
3.8084
21.8963
0.9535
10.1001
3.0451

fail to converge

6 The code to estimate this quantity is provided at the author’s repository

9

Published as a conference paper at ICLR 2022

REFERENCES

Pierre-Antoine Absil, Robert Mahony, and Rodolphe Sepulchre. Optimization Algorithms on Matrix

Manifolds. Princeton University Press, 2007.

Alekh Agarwal, Miroslav Dud´ık, and Zhiwei Steven Wu. Fair regression: Quantitative deﬁnitions
and reduction-based algorithms. In International Conference on Machine Learning, pp. 120–129.
PMLR, 2019.

Naveed Akhtar and Ajmal Mian. Threat of adversarial attacks on deep learning in computer vision:

A survey. Ieee Access, 6:14410–14430, 2018.

Solon Barocas, Moritz Hardt, and Arvind Narayanan. Fairness and machine learning. fairmlbook.

org, 2019, 2018.

Richard Berk, Hoda Heidari, Shahin Jabbari, Michael Kearns, and Aaron Roth. Fairness in criminal
justice risk assessments: The state of the art. Sociological Methods & Research, 50(1):3–44,
2021.

Alex Beutel, Jilin Chen, Zhe Zhao, and Ed H Chi. Data decisions and theoretical implications when

adversarially learning fair representations. arXiv preprint arXiv:1707.00075, 2017.

Jose Blanchet and Karthyek Murthy. Quantifying distributional model risk via optimal transport.

Mathematics of Operations Research, 44(2):565–600, 2019.

Jose Blanchet, Karthyek Murthy, and Viet Anh Nguyen. Statistical analysis of Wasserstein distribu-

tionally robust estimators. INFORMS TutORials in Operations Research, 2021.

Flavio P Calmon, Dennis Wei, Bhanukiran Vinzamuri, Karthikeyan Natesan Ramamurthy, and
Kush R Varshney. Optimized pre-processing for discrimination prevention. In Proceedings of
the 31st International Conference on Neural Information Processing Systems, pp. 3995–4004,
2017.

Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017

IEEE Symposium on Security and Privacy (SP), pp. 39–57. IEEE, 2017.

Anirban Chakraborty, Manaar Alam, Vishal Dey, Anupam Chattopadhyay, and Debdeep Mukhopad-

hyay. Adversarial attacks and defences: A survey. arXiv preprint arXiv:1810.00069, 2018.

Alexandra Chouldechova. Fair prediction with disparate impact: A study of bias in recidivism

prediction instruments. Big Data, 5(2):153–163, 2017.

Erick Delage and Yinyu Ye. Distributionally robust optimization under moment uncertainty with

application to data-driven problems. Operations Research, 58(3):595–612, 2010.

Michele Donini, Luca Oneto, Shai Ben-David, John S Shawe-Taylor, and Massimiliano Pontil. Em-
pirical risk minimization under fairness constraints. In Advances in Neural Information Process-
ing Systems, pp. 2791–2801, 2018.

Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel. Fairness
through awareness. In Proceedings of the 3rd innovations in theoretical computer science confer-
ence, pp. 214–226, 2012.

Michael Feldman, Sorelle A Friedler, John Moeller, Carlos Scheidegger, and Suresh Venkatasubra-
manian. Certifying and removing disparate impact. In Proceedings of the 21th ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining, pp. 259–268, 2015.

C.R. Givens and R.M. Shortt. A class of Wasserstein metrics for probability distributions. The

Michigan Mathematical Journal, 31(2):231–240, 1984.

Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial

examples. arXiv preprint arXiv:1412.6572, 2014.

Moritz Hardt, Eric Price, Eric Price, and Nati Srebro. Equality of opportunity in supervised learning.

In Advances in Neural Information Processing Systems 29, pp. 3315–3323, 2016a.

10

Published as a conference paper at ICLR 2022

Moritz Hardt, Eric Price, and Nati Srebro. Equality of opportunity in supervised learning. Advances

in neural information processing systems, 29:3315–3323, 2016b.

Tito Homem-de Mello and G¨uzin Bayraksan. Monte Carlo sampling-based methods for stochastic
optimization. Surveys in Operations Research and Management Science, 19(1):56–85, 2014.

Harold Hotelling. Analysis of a complex of statistical variables into principal components. Journal

of educational psychology, 24(6):417, 1933.

Faisal Kamiran and Toon Calders. Data preprocessing techniques for classiﬁcation without discrim-

ination. Knowledge and Information Systems, 33(1):1–33, 2012.

Toshihiro Kamishima, Shotaro Akaho, Hideki Asoh, and Jun Sakuma. Fairness-aware classiﬁer
In Joint European Conference on Machine Learning and

with prejudice remover regularizer.
Knowledge Discovery in Databases, pp. 35–50. Springer, 2012.

Daniel Kuhn, Peyman Mohajerin Esfahani, Viet Anh Nguyen, and Soroosh Shaﬁeezadeh-Abadeh.
Wasserstein distributionally robust optimization: Theory and applications in machine learning. In
Operations Research & Management Science in the Age of Analytics, pp. 130–166. INFORMS,
2019.

Matt J Kusner, Joshua R Loftus, Chris Russell, and Ricardo Silva. Counterfactual fairness. arXiv

preprint arXiv:1703.06856, 2017.

Xiao Li, Shixiang Chen, Zengde Deng, Qing Qu, Zhihui Zhu, and Anthony Man Cho So. Weakly
convex optimization over Stiefel manifold using Riemannian subgradient-type methods. SIAM
Journal on Optimization, 33(3):1605–1634, 2021.

Zachary Lipton, Julian McAuley, and Alexandra Chouldechova. Does mitigating ML’s impact dis-
parity require treatment disparity? In Advances in Neural Information Processing Systems, pp.
8125–8135, 2018.

David Madras, Elliot Creager, Toniann Pitassi, and Richard Zemel. Learning adversarially fair and
transferable representations. In International Conference on Machine Learning, pp. 3384–3393.
PMLR, 2018.

Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu.
Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083,
2017.

Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. A survey
on bias and fairness in machine learning. ACM Computing Surveys (CSUR), 54(6):1–35, 2021.

Peyman Mohajerin Esfahani and Daniel Kuhn. Data-driven distributionally robust optimization us-
ing the Wasserstein metric: Performance guarantees and tractable reformulations. Mathematical
Programming, 171(1-2):115–166, 2018.

Hongseok Namkoong and John C Duchi. Variance-based regularization with convex objectives. In

Advances in Neural Information Processing Systems 30, pp. 2971–2980, 2017.

Viet Anh Nguyen. Adversarial Analytics. PhD thesis, Ecole Polytechnique F´ed´erale de Lausanne,

2019.

Viet Anh Nguyen, Soroosh Shaﬁeezadeh Abadeh, Man-Chung Yue, Daniel Kuhn, and Wolfram
Wiesemann. Calculating optimistic likelihoods using (geodesically) convex optimization.
In
Advances in Neural Information Processing Systems, pp. 13942–13953, 2019.

Viet Anh Nguyen, S. Shaﬁeezadeh-Abadeh, D. Kuhn, and P. Mohajerin Esfahani. Bridging Bayesian
and minimax mean square error estimation via Wasserstein distributionally robust optimization.
Mathematics of Operations Research, 2021a.

Viet Anh Nguyen, Soroosh Shaﬁeezadeh-Abadeh, Damir Filipovi´c, and Daniel Kuhn. Mean-

covariance robust risk measurement. arXiv preprint arXiv:2112.09959, 2021b.

11

Published as a conference paper at ICLR 2022

Matt Olfat and Anil Aswani. Convex formulations for fair principal component analysis. In Pro-

ceedings of the AAAI Conference on Artiﬁcial Intelligence, volume 33, pp. 663–670, 2019.

Karl Pearson. Liii. On lines and planes of closest ﬁt to systems of points in space. The London,
Edinburgh, and Dublin philosophical magazine and journal of science, 2(11):559–572, 1901.

Hamed Rahimian and Sanjay Mehrotra. Distributionally robust optimization: A review. arXiv

preprint arXiv:1908.05659, 2019.

Samira Samadi, Uthaipon Tantipongpipat, Jamie H Morgenstern, Mohit Singh, and Santosh Vem-
pala. The price of fair PCA: One extra dimension. In Advances in Neural Information Processing
Systems, pp. 10976–10987, 2018.

Nian Si, Karthyek Murthy, Jose Blanchet, and Viet Anh Nguyen. Testing group fairness via optimal
transport projections. In Proceedings of the 38th International Conference on Machine Learning,
2021.

James E Smith and Robert L Winkler. The optimizer’s curse: Skepticism and postdecision surprise

in decision analysis. Management Science, 52(3):311–322, 2006.

Uthaipon Tantipongpipat, Samira Samadi, Mohit Singh, Jamie Morgenstern, and Santosh Vem-
arXiv preprint

pala. Multi-criteria dimensionality reduction with applications to fairness.
arXiv:1902.11281, 2019.

Bahar Taskesen, Viet Anh Nguyen, Daniel Kuhn, and Jose Blanchet. A distributionally robust

approach to fair classiﬁcation. arXiv preprint arXiv:2007.09530, 2020.

Bahar Taskesen, Man-Chung Yue, Jose Blanchet, Daniel Kuhn, and Viet Anh Nguyen. Sequential
In Proceedings of the 38th

domain adaptation by synthesizing distributionally robust experts.
International Conference on Machine Learning, 2021.

Sahil Verma and Julia Rubin. Fairness deﬁnitions explained.

In 2018 IEEE/ACM International

Workshop on Software Fairness (fairware), pp. 1–7. IEEE, 2018.

Yijie Wang, Viet Anh Nguyen, and Grani Hanasusanto. Wasserstein robust support vector machines

with fairness constraints. arXiv preprint arXiv:2103.06828, 2021.

Dennis Wei, Karthikeyan Natesan Ramamurthy, and Flavio du Pin Calmon. Optimized score trans-

formation for fair classiﬁcation. arXiv preprint arXiv:1906.00066, 2019.

Man-Chung Yue, Daniel Kuhn, and Wolfram Wiesemann. On linear optimization over Wasserstein

balls. Mathematical Programming, 2021.

Gad Zalcberg and Ami Wiesel. Fair principal component analysis and ﬁlter design. IEEE Transac-

tions on Signal Processing, 69:4835–4842, 2021.

Rich Zemel, Yu Wu, Kevin Swersky, Toni Pitassi, and Cynthia Dwork. Learning fair representations.

In International Conference on Machine Learning, pp. 325–333. PMLR, 2013.

Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell. Mitigating unwanted biases with adver-
sarial learning. In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, pp.
335–340, 2018.

A PROOFS

A.1 PROOFS OF SECTION 2

Proof of Proposition 2.2. Let S = EQ[XX (cid:62)|A = 0] − EQ[XX (cid:62)|A = 1]. We ﬁrst prove the
“only if” direction. Suppose that there exists a fair projection matrix V ∈ Mk relative to Q. Let
U ∈ Md−k be a complement matrix of V . Then, Deﬁnition 2.1 can be rewritten as

(cid:104)U U (cid:62), S(cid:105) = 0,

12

Published as a conference paper at ICLR 2022

which implies that the null space of S has a dimension at least d − k. By the rank-nullity duality,
we have rank(S) ≤ k.

Next, we prove the “if” direction. Suppose that rank(S) ≤ k. Then, the matrix S has at least d − k
(repeated) zero eigenvalues. Let U ∈ Md−k be an orthonormal matrix whose columns are any d−k
eigenvectors corresponding to the zero eigenvalues of S and V ∈ Mk be a complement matrix of
U . Then,

(cid:104)Id − V V (cid:62), S(cid:105) = (cid:104)U U (cid:62), S(cid:105) = 0.

Therefore, V is a fair projection matrix relative to Q. This completes the proof.

A.2 PROOF OF SECTION 3

Proofs of Proposition 3.3. By exploiting the deﬁnition of the loss function (cid:96), we ﬁnd

υEQa [(cid:96)(V, X)]

tr (cid:0)υ(I − V V (cid:62))(Σa + µaµ(cid:62)

a )(cid:1)

sup
Qa:W(Qa,ˆPa)≤εa



sup
µa,Σa
s.t.

=

=







inf

s.t.

(cid:16)

Σa + ˆΣa − 2(cid:0) ˆΣ

1
2

a Σa ˆΣ

1
2
a

2 (cid:17)
(cid:1) 1

2 + tr
(cid:17)

(cid:107)µa − ˆµa(cid:107)2
(cid:16) ˆΣa
γ(εa − tr
(cid:20)γI − υ(I − V V (cid:62))
γ ˆµ(cid:62)
a

(cid:16)

) + γ2 tr

(γI − υ(I − V V (cid:62)))−1 ˆΣa
γ ˆµa
γ(cid:107)ˆµa(cid:107)2

(cid:23) 0,

(cid:21)

≤ εa
(cid:17)

+ τ

2 + τ
where the last equality follows from Nguyen (2019, Lemma 3.22). By the Woodbury matrix inver-
sion, we have

γI (cid:31) υ(I − V V (cid:62)),

γ ≥ 0,

(γI − υ(I − V V (cid:62)))−1 = γ−1I −

(I − V V (cid:62)).

υ
γ(υ − γ)

Moreover, using the Schur complement, the semideﬁnite constraint is equivalent to

γ(cid:107)ˆµa(cid:107)2

2 + τ ≥ γ2 ˆµ(cid:62)

a (γI − υ(I − V V (cid:62)))−1 ˆµa,

which implies that at optimality, we have

τ =

υγ
γ − υ

a (I − V V (cid:62))ˆµa.
ˆµ(cid:62)

At the same time, the constraint γI (cid:31) υ(I − V V (cid:62)) is equivalent to γ > υ. Combining all previous
equations, we have

sup
Qa:W(Qa,ˆPa)≤εa

υEQa [(cid:96)(V, X)] =

inf
γ>max{0,υ}

γεa +

γυ
γ − υ

(cid:10)Id − V V (cid:62), ˆMa

(cid:11).

The dual optimal solution γ(cid:63) is given by

γ(cid:63) =






(cid:32)

υ

1 +

(cid:32)

1 −

υ

0

(cid:114) (cid:10)Id−V V (cid:62), ˆMa

(cid:11)

(cid:33)

εa

(cid:114) (cid:10)Id−V V (cid:62), ˆMa

(cid:11)

(cid:33)

εa

if υ ≥ 0,

if υ < 0 and (cid:10)Id − V V (cid:62), ˆMa
if υ < 0 and (cid:10)Id − V V (cid:62), ˆMa

(cid:11) ≥ εa,
(cid:11) < εa.

Note that γ(cid:63) ≥ max{0, υ} in all the cases. Therefore, we have

sup
Qa:W(Qa,ˆPa)≤εa


υEQa [(cid:96)(V, X)]

=




υ

υ

0

(cid:18)√

(cid:18)√

(cid:113)(cid:10)Id − V V (cid:62), ˆMa
(cid:113)(cid:10)Id − V V (cid:62), ˆMa

(cid:11)

(cid:11)

(cid:19)2

(cid:19)2

εa +

εa −

if υ ≥ 0,

if υ < 0 and (cid:10)Id − V V (cid:62), ˆMa
if υ < 0 and (cid:10)Id − V V (cid:62), ˆMa

(cid:11) ≥ εa,
(cid:11) < εa.

This completes the proof.

13

Published as a conference paper at ICLR 2022

We are now ready to prove Theorem 3.2.

Proof of Theorem 3.2. By expanding the absolute value, problem (4) is equivalent to

min
V ∈Rd×k,V (cid:62)V =Ik

max{J0(V ), J1(V )},

where for each (a, a(cid:48)) ∈ {(0, 1), (1, 0)}, we can re-express Ja as

Ja(V ) =

sup
Qa:W(Qa,ˆPa)≤εa

(ˆpa + λ)EQa [(cid:96)(V, X)] +

sup
a(cid:48) ,ˆP

Q

a(cid:48) :W(Q

(ˆpa(cid:48) − λ)EQ

a(cid:48) [(cid:96)(V, X)]

a(cid:48) )≤εa(cid:48)

Using Proposition 3.3 to reformulate the two individual supremum problems, we have

Ja(V ) = (ˆpa + λ)εa + 2|ˆpa + λ|

(cid:113)

εa

+ (ˆpa(cid:48) − λ)εa(cid:48) + 2|ˆpa(cid:48) − λ|

εa(cid:48)

(cid:10)Id − V V (cid:62), ˆMa(cid:48)

(cid:11) + (ˆpa(cid:48) − λ)(cid:10)Id − V V (cid:62), ˆMa(cid:48)

(cid:11).

(cid:10)Id − V V (cid:62), ˆMa
(cid:113)

(cid:11) + (ˆpa + λ)(cid:10)Id − V V (cid:62), ˆMa

(cid:11)

By deﬁning the necessary parameters κ, θ, ϑ and C as in the statement of the theorem, we arrive at
the postulated result.

A.3 PROOFS OF SECTION 4

Proof of Lemma 4.1. Let aU ∈ arg maxa∈{0,1} Fa(U ) and a(cid:48)
subgradient of F is given by

U = 1 − aU . Then, an Euclidean

∇F (U ) =

θaU
(cid:113)(cid:10)U U (cid:62), ˆMaU

(cid:11)

ˆMaU U +

ϑa(cid:48)
(cid:113)(cid:10)U U (cid:62), ˆMa(cid:48)

U

U

ˆMa(cid:48)

U

(cid:11)

U + 2CaU U ∈ Rd×(d−k).

The tangent space of the Stiefel manifold M at U is given by

TU M = {∆ ∈ Rd×(d−k) : ∆(cid:62)U + U (cid:62)∆ = 0},

whose orthogonal projection (Absil et al., 2007, Example 3.6.2) can be computed explicitly via

ProjTU M(D) = (Id − U U (cid:62))D +

1
2

U (U (cid:62)D − D(cid:62)U ), D ∈ Rd×(d−k).

Therefore, a Riemannian subgradient of F at any point U ∈ M is given by

gradF (U ) = ProjTU M(∇F (U ))



= (Id − U U (cid:62))



θaU
(cid:113)(cid:10)U U (cid:62), ˆMaU

(cid:11)

ˆMaU U +

ϑa(cid:48)
(cid:113)(cid:10)U U (cid:62), ˆMa(cid:48)

U

U



ˆMa(cid:48)

U

(cid:11)

U + 2CaU U

 .

In the last line, we have used the fact that, if D = SU for some symmetric matrix S, then

U (cid:62)D − D(cid:62)U = U (cid:62)SU − U (cid:62)S(cid:62)U = 0.

This completes the proof.

The proof of Lemma 4.2 relies on the following preliminary result.
Lemma A.1. Let M ∈ R(d−k)×(d−k) be a positive deﬁnite matrix. Then,

(cid:10)U U (cid:62), M (cid:11) − (cid:10)U (cid:48)U (cid:48)(cid:62), M (cid:11)(cid:12)

(cid:12)
(cid:12)

(cid:12) ≤ 2

√

d − kσmax(M )(cid:107)U − U (cid:48)(cid:107)F

∀U, U (cid:48) ∈ M,

(10)

and

(cid:12)
(cid:12)
(cid:12)
(cid:12)

(cid:113)(cid:10)U U (cid:62), M (cid:11) −

(cid:113)(cid:10)U (cid:48)U (cid:48)(cid:62), M (cid:11)

(cid:12)
(cid:12)
(cid:12)
(cid:12)

≤

σmax(M )
(cid:112)σmin(M )

(cid:107)U − U (cid:48)(cid:107)F

∀U, U (cid:48) ∈ M,

(11)

where σmax(M ) and σmin(M ) denote the maximum and minimum eigenvalues of the matrix M .

14

Published as a conference paper at ICLR 2022

Proof of Lemma A.1. For inequality (10),

(cid:10)U U (cid:62), M (cid:11) − (cid:10)U (cid:48)U (cid:48)(cid:62), M (cid:11)(cid:12)

(cid:12)
(cid:12)

(cid:10)U U (cid:62), M (cid:11) − (cid:10)U U (cid:48)(cid:62), M (cid:11)(cid:12)
(cid:10)U, M (U − U (cid:48))(cid:11)(cid:12)

(cid:12) ≤ (cid:12)
(cid:12)
(cid:12) + (cid:12)
≤ (cid:12)
(cid:12)
(cid:12)
≤ (cid:107)U (cid:107)F (cid:107)M (U − U (cid:48))(cid:107)F + (cid:107)U (cid:48)(cid:107)F (cid:107)M (U − U (cid:48))(cid:107)F
= 2

(cid:12) + (cid:12)
(cid:12)
(cid:10)U (cid:48), M (U − U (cid:48))(cid:11)(cid:12)
(cid:12)

d − kσmax(cid:107)U − U (cid:48)(cid:107)F .

√

(cid:10)U U (cid:48)(cid:62), M (cid:11) − (cid:10)U (cid:48)U (cid:48)(cid:62), M (cid:11)(cid:12)
(cid:12)

For inequality (11), we ﬁrst note that the function x (cid:55)→
and that

(cid:10)U U (cid:62), M (cid:11) ≥ (d − k)σmin(M ) ∀U ∈ M.

√

√

x is 1/(2

xmin)-Lipschitz on [xmin, +∞)

Therefore,
(cid:12)
(cid:12)
(cid:12)
(cid:12)

(cid:113)(cid:10)U U (cid:62), M (cid:11) −

(cid:12)
(cid:113)(cid:10)U (cid:48)U (cid:48)(cid:62), M (cid:11)
(cid:12)
(cid:12)
(cid:12)

≤

≤

(cid:12)
(cid:12)

(cid:10)U U (cid:62), M (cid:11) − (cid:10)U (cid:48)U (cid:48)(cid:62), M (cid:11)(cid:12)
(cid:12)

1
2(cid:112)(d − k)σmin(M )
σmax(M )
(cid:112)σmin(M )

(cid:107)U − U (cid:48)(cid:107)F ,

where the last inequality follows from (10). This completes the proof.

We are now ready to prove Lemma 4.2.

Proof of Lemma 4.2. Let U , U (cid:48) ∈ M be two arbitrary points. We have

|F (U ) − F (U (cid:48))|
= |max {F0(U ), F1(U )} − max {F0(U (cid:48)), F1(U (cid:48))}|
≤ max
a∈{0,1}

|Fa(U ) − Fa(U (cid:48))|

≤ max
a∈{0,1}

max






θa

σmax( ˆMa)
(cid:113)
σmin( ˆMa)

, ϑ1−a

σmax( ˆM1−a)
(cid:113)
σmin( ˆM1−a)

√

, 2

d − kσmax(Ca)






(cid:107)U − U (cid:48)(cid:107)F ,

where the last inequality follows from the deﬁnition of Fa and Lemma A.1. This completes the
proof.

Proof of Theorem 4.3. The proof follows from the fact that F is convex on the Euclidean space
Rd×(d−k), Lemma 4.2 and Li et al. (2021, Theorem 2) (and the remarks following it).

B EXTENSION TO NON-BINARY SENSITIVE ATTRIBUTES

The main paper focuses on the case of a binary sensitive attribute with A = {0, 1}. In this appendix,
we extend our approach to the case when the sensitive attribute is non-binary. Concretely, we sup-
pose that the sensitive attribute A can take on any of the m possible values from 1 to m. In other
words, the attribute space now becomes A = {1, . . . , m}.
Deﬁnition B.1 (Generalized unfairness measure). The generalized unfairness measure is deﬁned as
the maximum pairwise unfairness measure, that is,

Umax(V, Q) (cid:44) max

(a,a(cid:48))∈A×A

|EQ[(cid:96)(V, X)|A = a] − EQ[(cid:96)(V, X)|A = a(cid:48)]|.

Notice that if A = {0, 1}, then Umax ≡ U recovers the unfairness measure for binary sensitive
attribute deﬁned in Section 2.2. We now consider the following generalized fairness-aware PCA
problem

EQ[(cid:96)(V, X)] + λUmax(V, Q).

(12)

min
V ∈Rd×k,V (cid:62)V =Ik

sup
Q∈B(ˆP)

Here we recall that the ambiguity set B(ˆP) is deﬁned in (5). The next theorem provides the refor-
mulation of (12).

15

Published as a conference paper at ICLR 2022

Theorem B.2 (Reformulation of non-binary fairness-aware PCA). Suppose that for any a ∈ A,
either of the following two conditions holds:

(i) Marginal probability bounds: 0 ≤ λ ≤ ˆpa,

(ii) Eigenvalue bounds: the empirical second moment matrix ˆMa = 1
Na

i∈Ia
j=1 σj( ˆMa) ≥ εa, where σj( ˆMa) is the j-th smallest eigenvalues of ˆMa.

(cid:80)d−k

(cid:80)

ˆxi ˆx(cid:62)

i satisﬁes

Then problem (12) is equivalent to

min
V ∈Rd×k,V (cid:62)V =Ik

max
a(cid:54)=a(cid:48)

(cid:40)

(cid:88)

b∈A

(cid:113)

2ca,a(cid:48),b

εb(cid:104)Id − V V (cid:62), ˆMb(cid:105) + λ(cid:104)Id − V V (cid:62), ˆMa − ˆMa(cid:48)(cid:105) + λ(εa − εa(cid:48))

,

(cid:41)

where the parameter ca,a(cid:48),b admits values

ca,a(cid:48),b =






ˆpa + λ
|ˆpa(cid:48) − λ|
ˆpb

if b = a,
if b = a(cid:48),
otherwise.

Proof of Theorem B.2. For simplicity, we let E(V, Q, b) = EQ[(cid:96)(V, X)|A = b]. Then, the objective
function of problem (12) can be re-written as

EQ[(cid:96)(V, X)] + λUmax(V, Q)

ˆpbE(V, Q, b) + λ max
a(cid:54)=a(cid:48)

{E(V, Q, a) − E(V, Q, a(cid:48))}

sup
Q∈B(ˆP)

= sup

Q∈B(ˆP)



= max
a(cid:54)=a(cid:48)


(cid:40)

= max
a(cid:54)=a(cid:48)

(cid:88)

b∈A

(cid:88)

b(cid:54)=a,a(cid:48)

(cid:18)(cid:113)

(cid:88)

ˆpb

b(cid:54)=a,a(cid:48)

(cid:104)Id − V V (cid:62), ˆMb(cid:105) +

(cid:19)2

√

εb

+ (ˆpa + λ)

(cid:18)(cid:113)

(cid:104)Id − V V (cid:62), ˆMa(cid:105) +

(cid:19)2

√

εa

+ (ˆpa(cid:48) − λ)

(cid:18)(cid:113)

(cid:104)Id − V V (cid:62), ˆMa(cid:48)(cid:105) + sgn(ˆpa(cid:48) − λ)

√

(cid:19)2 (cid:41)

εa(cid:48)

= max
a(cid:54)=a(cid:48)

(cid:40)

(cid:88)

(cid:16)

ˆpb

b∈A

(cid:104)Id − V V (cid:62), ˆMb(cid:105) + εb

(cid:17)

+

(cid:88)

2ca,a(cid:48),b

(cid:113)

εb(cid:104)Id − V V (cid:62), ˆMb(cid:105)

b∈A

(cid:16)

+ λ

(cid:104)Id − V V (cid:62), ˆMa − ˆMa(cid:48)(cid:105) + εa − εa(cid:48)

(cid:41)

(cid:17)

,

where the ﬁrst equality follows from the deﬁnition of Umax(V, Q) and E(V, Q, b), the second from
the deﬁnition (5) of the ambiguity set B(ˆP), the third from Proposition 3.3 and the fourth from the
deﬁnition of ca,a(cid:48),b. Noting that the ﬁrst sum in the above maximization is independent of a and a(cid:48),
the proof is completed.

Theorem 12 indicates that if the sensitive attribute admits ﬁnite values, then the distributionally
robust fairness-aware PCA problem using an Umax unfairness measure can be reformulated as an
optimization problem over the Stiefel manifold, where the objective function is a pointwise maxi-
mization of ﬁnite number of individual functions. It is also easy to see that each individual function
can be reparametrized using U , and the Riemannian gradient descent algorithm in Section 4 can be
adapted to solve for the optimal solution. The details on the algorithm are omitted.

16

sup
W(Qb,ˆPb)≤εb

ˆpbE(V, Qb, b) +

sup
W(Qa,ˆPa)≤εa

(ˆpa + λ)E(V, Qa, a) +

sup
a(cid:48) ,ˆP

a(cid:48) )≤εa(cid:48)

W(Q

(ˆpa(cid:48) − λ)E(V, Qa(cid:48), a(cid:48))






Published as a conference paper at ICLR 2022

C INFORMATION ON DATASETS

We summarize here the number of observations, dimensions, and the sensitive attribute of the data
sets. For further information about the data sets and pre-processing steps, please refer to Samadi
et al. (2018) for Default Credit and Labeled Faces in the Wild (LFW) data sets, and Olfat & Aswani
(2019) for others. For each data set, we further remove columns with too small standard deviation
(≤ 1e−5) as they do not signiﬁcantly affect the results, and ones with too large standard deviation
(≥ 1000) which we consider as unreliable features.

Table 2: Number of observations N , dimensions d, and sensitive attribute A of datasets used in this
paper. (y - yes, n - no)

Default Credit
30000
22
Education
(high/low)
Image
660
18
class (path/grass)
Statlog
3071
36
RedSoil
(vsgrey/dampgrey)

N
d

A

N
d
A

N
d

A

Biodeg
1055
40
Ready Biodegradable
(y/n)
Letter
20000
16
Vowel (y/n)
Steel
1941
24

E. Coli
333
7
isCytoplasm
(y/n)
Magic
19020
10
classIsGamma (y/n)
Taiwan Credit
29623
22

Energy
768
8
Orientation< 4
(y/n)
Parkinsons
5875
20
Sex (male/female)
Wine Quality
6497
11

German Credit
1000
48
A13 ≥ 200DM
(y/n)
SkillCraft
3337
17
Age> 20 (y/n)
LFW
4000
576

FaultOther (y/n)

Sex (male/female)

isWhite (y/n)

Sex (male/female)

D ADDITIONAL RESULTS

D.1 DETAIL PERFORMANCES

Table 3 shows the performances of four examined methods with two criteria ABDiff. and ARE. It is
clear that our method achieves the best results over all 14 datasets w.r.t. ABDiff., and 7 datasets on
ARE., which is equal to the number of datasets FairPCA out-perform others.

Table 4 complements Table 1 from the main text, from which we can see that two versions of CFPCA
out-perform others over all datasets w.r.t. (cid:52)FLin, which is the criteria they optimize for.

Table 3: In-sample performance over two criteria

Dataset
Default Credit
Biodeg
E. Coli
Energy
German Credit
Image
Letter
Magic
Parkinsons
SkillCraft
Statlog
Steel
Taiwan Credit
Wine Quality
LFW

RFPCA

FairPCA

ABDiff.
0.9457
9.4093
0.5678
0.0094
1.6265
0.1320
0.1121
1.7405
0.1238
0.4231
0.1972
0.6943
1.1516
0.1125
0.4147

ARE.
9.9072
23.1555
1.4804
0.2295
40.1512
5.0924
7.4088
3.8766
5.0471
8.1569
3.0588
11.0396
10.5136
4.1491
7.5137

ABDiff.
1.5821
14.2587
0.9191
0.0153
2.9824
0.7941
1.2560
2.8679
0.6702
0.5576
0.3315
1.8015
1.3362
0.1705
0.5300

ARE.
9.9049
23.8227
2.0840
0.2273
40.3393
9.0437
7.4375
3.3500
4.8760
8.1096
7.9980
10.7653
10.4478
5.8999
7.5127

CFPCA-Mean Con.
ABDiff.
ARE.
10.5164
0.9949
26.6540
15.5545
2.8360
0.9539
2.7893
0.2658
40.1860
2.6109
13.4491
0.6910
8.7764
0.4572
4.1938
5.5405
5.9379
3.9470
0.7156
9.7755
10.9358
0.3857
14.5680
2.8933
12.5867
1.3158
5.9117
1.1359

CFPCA - Both Con.
ABDiff.
3.2827
24.8706
4.5225
0.2136
2.8741
3.0118
0.5301
8.7963
17.8122
0.9334
13.0725
1.9322
2.2720
2.5852

ARE.
21.4523
39.8737
5.2155
7.8768
47.1006
18.0000
15.2234
8.9695
19.9788
15.8245
35.9214
23.9906
21.4365
9.8959

fail to converge

Adjustment for the LFW dataset. To demonstrate the efﬁcacy of our method on high-dimensional
data sets, we also do experiments on a subset of 2000 faces for each of male and female group (4000

17

Published as a conference paper at ICLR 2022

Table 4: Out-of-sample performance measured using the (cid:52)FLin criterion.

Default Credit
Biodeg
E. Coli
Energy
German Credit
Image
Letter
Magic
Parkinson’s
SkillCraft
Statlog
Steel
Taiwan Credit
Wine Quality

RFPCA FairPCA CFPCA-Mean Con. CFPCA - Both Con.
0.0574
0.1596
0.2014
0.4892
0.4455
0.8556
0.0502
0.0580
0.1408
0.1997
0.1874
0.9996
0.0556
0.0954
0.1561
0.2195
0.1805
0.1459
0.0721
0.1126
0.1359
0.9804
0.1418
0.2288
0.0391
0.0604
0.2192
0.9699

0.0413
0.1371
0.2532
0.0736
0.1093
0.2013
0.0455
0.0882
0.0480
0.0742
0.0669
0.0875
0.0370
0.0817

0.2236
0.4759
0.7444
0.0554
0.1737
0.9498
0.0942
0.2531
0.1061
0.1141
0.6309
0.2240
0.0535
0.4639

in total) from LFW dataset,7 all images are rescaled to resolution 24 × 24 (dimensions d = 576).
The experiment follows the same procedure in Section 5, with reducing the number of iterations
to 500 for both RFPCA and FairPCA and 2-fold cross validation, the results are averaged over
10 train-test simulations. Due to the high dimension of the input, the implementation of Olfat &
Aswani (2019) fails to return any result.

7 https://github.com/samirasamadi/Fair-PCA

18

Published as a conference paper at ICLR 2022

D.2 VISUALIZATION

D.2.1 EFFECTS OF THE AMBIGUITY SET RADIUS

We examine the change of the model’s performance with respect to the change of the radius of the
ambiguity sets. To generate the toy data (also used for Figure 1), we use two 2-dimensional Gaussian
distributions to represent two groups of a sensitive attribute, A = 0 and A = 1, or groups 0 and 1
for simplicity. The two distributions both have the mean at (0, 0) and covariance matrices for group
0 and 1 are

(cid:18)4.0
0

(cid:19)

0
0.2

and

(cid:18)0.2
0.4

(cid:19)

,

0.4
3.0

respectively. For the test set, the number of samples is 8000 for group 0 and 4000 for group 1, while
for the training set, we have 200 for group 0 and 100 for group 1. We average the results over 100
simulations, for each simulation, the test data is ﬁxed, the training data is randomly generated with
the number of samples mentioned above. The projections are learned on training data and measured
on test data by the summation of ARE. and ABDiff. We ﬁxed λ = 0.1, which is not too small for
achieving fair projections, and not too large to clearly observe the effects of ε, and we also ﬁxed
ε0 for better visualization. Note that we still compute ε1 = α/
N1 in which, α is tested with 100
values evenly spaced in [0, 10].

√

The experiment results are visualized in Figure 4. The result suggests that increasing the ambiguity
set radius can improve the overall model’s performance. This justiﬁes the beneﬁt of adding distribu-
tional robustness to the fairness-aware PCA model. After a saturation point, a too large radius can
lessen the role of empirical data, and the model prioritizes a more extreme distribution that is far
from the target distribution, which causes the reduction in the model’s performance on target data.

Figure 4: Performance changes w.r.t. the ambiguity set’s radius. The solid line is the average over
100 simulations, and the shade represent the 1-standard deviation range.

19

Published as a conference paper at ICLR 2022

D.2.2 PARETO CURVES

Figures 5 and 6 plot the Pareto frontier for two datasets (Biodeg and German Credit) with 3 principal
components. One can observe that RFPCA produces points that dominate other methods based on
the trade-off between ARE. and ABDiff.

Figure 5: Pareto curves on Biodeg
dataset (all data) with 3 principal components

Figure 6: Pareto curves on German Credit
dataset (all data) with 3 principal components

20

Published as a conference paper at ICLR 2022

D.2.3 PERFORMANCE WITH DIFFERENT PRINCIPAL COMPONENTS

We collect here the reconstruction errors for different numbers of principal components.

Figure 7: Subgroup average error
with different k on Default Credit dataset

Figure 8: Subgroup average error
with different k on Default Credit dataset

Figure 9: Subgroup average error
with different k on E. Coli dataset

Figure 10: Subgroup average error
with different k on E. Coli dataset

21

Published as a conference paper at ICLR 2022

Figure 11: Subgroup average error
with different k on Magic dataset

Figure 12: Subgroup average error
with different k on Magic dataset

Figure 13: Subgroup average error
with different k on Steel dataset

Figure 14: Subgroup average error
with different k on Steel dataset

22

Published as a conference paper at ICLR 2022

Figure 15: Subgroup average error
with different k on Wine Quality dataset

Figure 16: Subgroup average error
with different k on Wine Quality dataset

23

