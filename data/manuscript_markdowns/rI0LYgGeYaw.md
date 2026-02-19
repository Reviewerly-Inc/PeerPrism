Published as a conference paper at ICLR 2022

UNDERSTANDING APPROXIMATE AND UNROLLED
DICTIONARY LEARNING FOR PATTERN RECOVERY

Benoˆıt Mal´ezieux
Universit´e Paris-Saclay, Inria, CEA
L2S, Universit´e Paris-Saclay–CNRS–CentraleSupelec
benoit.malezieux@inria.fr

Thomas Moreau
Universit´e Paris-Saclay, Inria, CEA
Palaiseau, 91120, France
thomas.moreau@inria.fr

Matthieu Kowalski
L2S, Universit´e Paris-Saclay–CNRS–CentraleSupelec
Gif-sur-Yvette, 91190, France
matthieu.kowalski@universite-paris-saclay.fr

ABSTRACT

Dictionary learning consists of ﬁnding a sparse representation from noisy data and
is a common way to encode data-driven prior knowledge on signals. Alternating
minimization (AM) is standard for the underlying optimization, where gradient
descent steps alternate with sparse coding procedures. The major drawback of
this method is its prohibitive computational cost, making it unpractical on large
real-world data sets. This work studies an approximate formulation of dictionary
learning based on unrolling and compares it to alternating minimization to ﬁnd
the best trade-off between speed and precision. We analyze the asymptotic behav-
ior and convergence rate of gradients estimates in both methods. We show that
unrolling performs better on the support of the inner problem solution and during
the ﬁrst iterations. Finally, we apply unrolling on pattern learning in magnetoen-
cephalography (MEG) with the help of a stochastic algorithm and compare the
performance to a state-of-the-art method.

1

INTRODUCTION

Pattern learning provides insightful information on the data in various biomedical applications. Typ-
ical examples include the study of magnetoencephalography (MEG) recordings, where one aims to
analyze the electrical activity in the brain from measurements of the magnetic ﬁeld around the scalp
of the patient (Dupr´e la Tour et al., 2018). One may also mention neural oscillations study in the
local ﬁeld potential (Cole & Voytek, 2017) or QRS complex detection in electrocardiograms (Xiang
et al., 2018) among others.

Dictionary learning (Olshausen & Field, 1997; Aharon et al., 2006; Mairal et al., 2009) is particu-
larly efﬁcient on pattern learning tasks, such as blood cells detection (Yellin et al., 2017) and MEG
signals analysis (Dupr´e la Tour et al., 2018). This framework assumes that the signal can be de-
composed into a sparse representation in a redundant basis of patterns – also called atoms. In other
words, the goal is to recover a sparse code Z ∈ Rn×T and a dictionary D ∈ Rm×n from noisy mea-
surements Y ∈ Rm×T which are obtained as the linear transformation DZ, corrupted with noise
B ∈ Rm×T : Y = DZ + B. Theoretical elements on identiﬁability and local convergence have
been proven in several studies (Gribonval et al., 2015; Haeffele & Vidal, 2015; Agarwal et al., 2016;
Sun et al., 2016). Sparsity-based optimization problems related to dictionary learning generally rely
on the usage of the (cid:96)0 or (cid:96)1 regularizations. In this paper, we study Lasso-based (Tibshirani, 1996)
dictionary learning where the dictionary D is learned in a set of constraints C by solving

min
Z∈Rn×T ,D∈C

F (Z, D) (cid:44) 1
2

(cid:107)DZ − Y (cid:107)2

2 + λ (cid:107)Z(cid:107)1 .

(1)

1

Published as a conference paper at ICLR 2022

Dictionary learning can be written as a bi-level optimization problem to minimize the cost function
with respect to the dictionary only, as mentioned in Mairal et al. (2009),

min
D∈C

G(D) (cid:44) F (Z∗(D), D) with Z∗(D) = arg min
Z∈Rn×T

F (Z, D) .

(2)

Computing the data representation Z∗(D) is often referred to as the inner problem, while the global
minimization is the outer problem. Classical constraint sets include the unit norm, where each
atom is normalized to avoid scale-invariant issues, and normalized convolutional kernels to perform
Convolutional Dictionary Learning (Grosse et al., 2007).

Classical dictionary learning methods solve this bi-convex optimization problem through Alternat-
ing Minimization (AM) (Mairal et al., 2009). It consists in minimizing the cost function F over Z
with a ﬁxed dictionary D and then performing projected gradient descent to optimize the dictionary
with a ﬁxed Z. While AM provides a simple strategy to perform dictionary learning, it can be inefﬁ-
cient on large-scale data sets due to the need to resolve the inner problems precisely for all samples.
In recent years, many studies have focused on algorithm unrolling (Tolooshams et al., 2020; Scetbon
et al., 2021) to overcome this issue. The core idea consists of unrolling the algorithm, which solves
the inner problem, and then computing the gradient with respect to the dictionary with the help of
back-propagation through the iterates of this algorithm. Gregor & LeCun (2010) popularized this
method and ﬁrst proposed to unroll ISTA (Daubechies et al., 2004) – a proximal gradient descent
algorithm designed for the Lasso – to speed up the computation of Z∗(D). The N + 1-th layer
(W 1Y + W 2ZN ), with ST be-
of this network – called LISTA – is obtained as ZN +1 = ST λ
ing the soft-thresholding operator. This work has led to many contributions aiming at improving this
method and providing theoretical justiﬁcations in a supervised (Chen et al., 2018; Liu & Chen, 2019)
or unsupervised (Moreau & Bruna, 2017; Ablin et al., 2019) setting. For such unrolled algorithms,
the weights W 1 and W 2 can be re-parameterized as functions of D – as illustrated in Figure A in
appendix – such that the output ZN (D) matches the result of N iterations of ISTA, i.e.
(cid:18)

(cid:19)

L

W 1

D =

D(cid:62) and W 2

D =

I −

D(cid:62)D

, where L = (cid:107)D(cid:107)2 .

(3)

1
L

1
L

Then, the dictionary can be learned by minimizing the loss F (ZN (D), D) over D with back-
propagation. This approach is generally referred to as Deep Dictionary Learning (DDL). DDL and
variants with different kinds of regularization (Tolooshams et al., 2020; Lecouat et al., 2020; Scetbon
et al., 2021), image processing based on metric learning (Tang et al., 2020), and classiﬁcation tasks
with scattering (Zarka et al., 2019) have been proposed in the literature, among others. While these
techniques have achieved good performance levels on several signal processing tasks, the reasons
they speed up the learning process are still unclear.

In this work, we study unrolling in Lasso-based dictionary learning as an approximate bi-level opti-
mization problem. What makes this work different from Bertrand et al. (2020), Ablin et al. (2020)
and Tolooshams & Ba (2021) is that we study the instability of non-smooth bi-level optimization
and unrolled sparse coding out of the support, which is of major interest in practice with a small
number of layers. In Section 2, we analyze the convergence of the Jacobian computed with auto-
matic differentiation and ﬁnd out that its stability is guaranteed on the support of the sparse codes
only. De facto, numerical instabilities in its estimation make unrolling inefﬁcient after a few dozen
iterations. In Section 3, we empirically show that unrolling leads to better results than AM only with
a small number of iterations of sparse coding, making it possible to learn a good dictionary in this
setting. Then we adapt a stochastic approach to make this method usable on large data sets, and we
apply it to pattern learning in magnetoencephalography (MEG) in Section 4. We do so by adapting
unrolling to rank one convolutional dictionary learning on multivariate time series (Dupr´e la Tour
et al., 2018). We show that there is no need to unroll more than a few dozen iterations to obtain
satisfying results, leading to a signiﬁcant gain of time compared to a state-of-the-art algorithm.

2 BI-LEVEL OPTIMIZATION FOR APPROXIMATE DICTIONARY LEARNING

As Z∗(D) does not have a closed-form expression, G cannot be computed directly. A solution is to
replace the inner problem Z∗(D) by an approximation ZN (D) obtained through N iterations of a
numerical optimization algorithm or its unrolled version. This reduces the problem to minimizing
GN (D) (cid:44) F (ZN (D), D). The ﬁrst question is how sub-optimal global solutions of GN are

2

Published as a conference paper at ICLR 2022

compared to the ones of G. Proposition 2.1 shows that the global minima of GN converge as fast as
the numerical approximation ZN in function value.

Proposition 2.1 Let D∗ = arg minD∈C G(D) and D∗
N = arg minD∈C GN (D), where N is the
number of unrolled iterations. We denote by K(D∗) a constant depending on D∗, and by C(N ) the
convergence speed of the algorithm, which approximates the inner problem solution. We have

GN (D∗

N ) − G(D∗) ≤ K(D∗)C(N ) .

The proofs of all theoretical results are deferred to Appendix C. Proposition 2.1 implies that when
ZN is computed with FISTA (Beck & Teboulle, 2009), the function value for global minima of
GN converges with speed C(N ) = 1
N 2 towards the value of the global minima of F . Therefore,
solving the inner problem approximately leads to suitable solutions for equation 2, given that the
optimization procedure is efﬁcient enough to ﬁnd a proper minimum of GN . As the computational
cost of zN increases with N , the choice of N results in a trade-off between the precision of the
solution and the computational efﬁciency, which is critical for processing large data sets.

Moreover, learning the dictionary and computing the sparse codes are two different tasks. The loss
GN takes into account the dictionary and the corresponding approximation ZN (D) to evaluate the
quality of the solution. However, the dictionary evaluation should reﬂect its ability to generate
the same signals as the ground truth data and not consider an approximate sparse code that can be
recomputed afterward. Therefore, we should distinguish the ability of the algorithm to recover a
good dictionary from its ability to learn the dictionary and the sparse codes at the same time. In
this work, we use the metric proposed in Moreau & Gramfort (2020) for convolutions to evaluate
the quality of the dictionary. We compare the atoms using their correlation and denote as C the cost
matrix whose entry i, j compare the atom i of the ﬁrst dictionary and j of the second. We deﬁne a
i=1 |Cσ(i),i|, where Sn is the group
sign and permutation invariant metric S(C) = maxσ∈Sn
of permutations of [1, n]. This metric corresponds to the best linear sum assignment on the cost
matrix C, and it can be computed with the Hungarian algorithm. Note that doing so has several
limitations and that evaluating the dictionary is still an open problem. Without loss of generality, let
T = 1 and thus z ∈ Rn in the rest of this section.

(cid:80)n

1
n

Gradient estimation in dictionary learning. Approximate dictionary learning is a non-convex
problem, meaning that good or poor local minima of GN may be reached depending on the initial-
ization, the optimization path, and the structure of the problem. Therefore, a gradient descent on GN
has no guarantee to ﬁnd an adequate minimizer of G. While complete theoretical analysis of these
problems is arduous, we propose to study the correlation between the gradient obtained with GN
and the actual gradient of G, as a way to ensure that the optimization dynamics are similar. Once
z∗(D) is known, Danskin (1967, Thm 1) states that g∗(D) = ∇G(D) is equal to ∇2F (z∗(D), D),
where ∇2 indicates that the gradient is computed relatively to the second variable in F . Even though
the inner problem is non-smooth, this result holds as long as the solution z∗(D) is unique. In the
following, we will assume that D(cid:62)D is invertible on the support of z∗(D), which implies the
uniqueness of z∗(D). This occurs with probability one if D is sampled from a continuous distribu-
tion (Tibshirani, 2013). AM and DDL differ in how they estimate the gradient of G. AM relies on
the analytical formula of g∗ and uses an approximation zN of z∗, leading to the approximate gra-
N approximates g∗ in Proposition 2.2.
dient g1

N (D) = ∇2F (zN (D), D). We evaluate how well g1

Proposition 2.2 Let D ∈ Rm×n. Then, there exists a constant L1 > 0 such that for every number
of iterations N

(cid:13)
(cid:13)g1

N − g∗(cid:13)

(cid:13) ≤ L1 (cid:107)zN (D) − z∗(D)(cid:107) .

Proposition 2.2 shows that g1
N converges as fast as the iterates of ISTA converge. DDL computes
the gradient automatically through zN (D). As opposed to AM, this directly minimizes the loss
GN (D). Automatic differentiation yields a sub-gradient g2

N (D) such that

N (D) ∈ ∇2F (zN (D), D) + J+
g2

N

(cid:16)

∂1F (zN (D), D)

(cid:17)

,

(4)

where JN : Rm×n → Rn is the weak Jacobian of zN (D) with respect to D and J+
adjoint. The product between J+

N denotes its
N and ∂1F (zN (D), D) is computed via automatic differentiation.

3

Published as a conference paper at ICLR 2022

2 (cid:107)Dz − y(cid:107)2

2,1f (z∗, D) (cid:12) 1

1,1f (z∗, D) (cid:12) 1
(cid:101)S

Proposition 2.3 Let D ∈ Rm×n. Let S∗ be the support of z∗(D), SN be the support of zN and
(cid:101)SN = SN ∪ S∗. Let f (z, D) = 1
2 be the data-ﬁtting term in F . Let R(J, (cid:101)S) =
J+(cid:0)∇2
(cid:1) + ∇2
(cid:101)S. Then there exists a constant L2 > 0 and a sub-
sequence of (F)ISTA iterates zφ(N ) such that for all N ∈ N:
φ(N ) ∈ ∇2f (zφ(N ), D) + J+
(cid:13)
φ(N ) − g∗(cid:13)
(cid:13)
(cid:13)
(cid:13) ≤
(cid:13)R(Jφ(N ), (cid:101)Sφ(N ))

L2
2
This sub-sequence zφ(N ) corresponds to iterates on the support of z∗.

∇1f (zφ(N ), D) + λ∂(cid:107)·(cid:107)1
(cid:13)
(cid:13)zφ(N ) − z∗(cid:13)
(cid:13)
(cid:13)
(cid:13)

(cid:13)zφ(N ) − z∗(cid:13)
(cid:13)
2
(cid:13)

(zφ(N ))

(cid:13)
(cid:13)g2
(cid:13)

∃ g2

s.t. :

(cid:13) +

φ(N )

(cid:16)

(cid:17)

.

Proposition 2.3 shows that g2

N may converge faster than g1

N once the support is reached.

Ablin et al. (2020) and Tolooshams & Ba (2021) have studied the behavior of strongly convex
functions, as it is the case on the support, and found similar results. This allowed Tolooshams &
Ba (2021) to focus on support identiﬁcation and show that automatic differentiation leads to a better
gradient estimation in dictionary learning on the support under minor assumptions.

However, we are also interested in characterizing the behavior outside of the support, where the
In practice, automatic differenti-
gradient estimation is difﬁcult because of the sub-differential.
ation uses the sign operator as a sub-gradient of (cid:107)·(cid:107)1. The convergence behavior of g2
N is also
driven by R(JN , (cid:102)SN ) and thus by the weak Jacobian computed via back-propagation. We ﬁrst com-
pute a closed-form expression of the weak Jacobian of z∗(D) and zN (D). We then show that
R(JN , (cid:102)SN ) ≤ L (cid:107)JN − J∗(cid:107) and we analyze the convergence of JN towards J∗.

Study of the Jacobian. The computation of the Jacobian can be done by differentiating through
ISTA. In Theorem 2.4, we show that JN +1 depends on JN and the past iterate zN , and converges
towards a ﬁxed point. This formula can be used to compute the Jacobian during the forward pass,
avoiding the computational cost of back-propagation and saving memory.

Theorem 2.4 At iteration N + 1 of ISTA, the weak Jacobian of zN +1 relatively to Dl, where Dl is
the l-th row of D, is given by induction:
(cid:18) ∂(zN )
∂Dl

l zN − yl)In + D(cid:62)D

= 1|zN +1|>0 (cid:12)

N + (D(cid:62)

Dlz(cid:62)

1
L

(cid:19)(cid:19)

−

(cid:18)

∂(zN )
∂Dl
l of z∗ relatively to Dl,

.

∂(zN +1)
∂Dl
will be denoted by J N
l

∂(zN )
∂Dl

whose values are

. It converges towards the weak Jacobian J ∗

l S∗ = −(D(cid:62)
J ∗

:,S∗ D:,S∗ )−1(Dlz∗(cid:62) + (D(cid:62)

l z∗ − yl)In)S∗ ,

on the support S∗ of z∗, and 0 elsewhere. Moreover, R(J∗, S∗) = 0.

(cid:13)∇2

This result is similar to Bertrand et al. (2020) where the Jacobian of z is computed over λ to perform
hyper-parameter optimization in Lasso-type models. Using R(J∗, S∗) = 0, we can write

(cid:13)
(cid:13)
(cid:13) ≤

(cid:13)
(cid:13)R(JN , (cid:101)SN ) − R(J∗, S∗)
(cid:13)

(cid:13)
(cid:13)
(cid:13)R(JN , (cid:101)SN )
as (cid:13)
1,1f (z∗, D)(cid:13)
(cid:13)2 = L. If the back-propagation were to output an accurate estimate JN of the
(cid:13)
weak Jacobian J∗,
(cid:13)
N could be twice as
(cid:13)R(JN , (cid:102)SN )
N . To quantify this, we now analyze the convergence of JN towards J∗. In
fast as the one of g1
Proposition 2.5, we compute an upper bound of (cid:13)
(cid:13)
(cid:13)J N
(cid:13) with possible usage of truncated back-
propagation (Shaban et al., 2019). Truncated back-propagation of depth K corresponds to an initial
estimate of the Jacobian JN −K = 0 and iterating the induction in Theorem 2.4.

(cid:13)
(cid:13)
(cid:13) would be 0, and the convergence rate of g2

(cid:13)
(cid:13) ≤ L (cid:107)JN − J∗(cid:107) ,
(cid:13)

l − J ∗
l

(5)

Proposition 2.5 Let N be the number of iterations and K be the back-propagation depth. We as-
sume that ∀n ≥ N −K, S∗ ⊂ Sn. Let ¯EN = Sn \S∗, let L be the largest eigenvalue of D(cid:62)
:,S∗ D:,S∗ ,
(cid:13)
(cid:13)
(cid:13)
(cid:13)
and let µn be the smallest eigenvalue of D(cid:62)
(cid:13), where
:,S∗ PS∗
(cid:13)PEn
PS is the projection on RS and D† is the pseudo-inverse of D. We have
(cid:16) (cid:13)

:,Sn D:,Sn−1. Let Bn =

− D(cid:62)

K−1
(cid:88)

D†(cid:62)

k
(cid:89)

K
(cid:89)

:, ¯En

(cid:17)

(cid:16)

1 −

µN −k
L

(cid:107)J ∗

l (cid:107)+

(cid:107)Dl(cid:107)

2
L

(1−

µN −i
L

)

(cid:13)zN −k
l

− z∗
l

(cid:13)
(cid:13)+BN −k (cid:107)z∗
l (cid:107)

(cid:17)

.

(cid:13)
(cid:13)J N

l − J ∗
l

(cid:13)
(cid:13) ≤

k=1

k=0

i=1

4

Published as a conference paper at ICLR 2022

Figure 1: Average convergence of J N
towards J ∗
l for two samples from the same data set, gener-
l
ated with a random Gaussian matrix. (cid:13)
(cid:13)
l − J N
(cid:13) converges linearly on the support in both cases.
l
However, for sample 2, full back-propagation makes the convergence unstable, and truncated back-
propagation improves its behavior, as described in Proposition 2.5. The proportion of stable and
unstable samples in this particular example is displayed in Figure 2.

(cid:13)J ∗

Proposition 2.5 reveals multiple stages in the Jacobian estimation. First, one can see that if all
iterates used for the back-propagation lie on the support S∗, the Jacobian estimate has a quasi-linear
convergence, as shown in the following corollary.

Corollary 2.6 Let µ > 0 be the smallest eigenvalue of D(cid:62)
propagation depth and let ∆N = F (zN , D) − F (z∗, D) + L
[N − K, N ]; Sn ⊂ S∗. Then, we have
µ
L

l − J N
l

l (cid:107) + K

(cid:13)
(cid:13) ≤

(cid:13)
(cid:13)J ∗

(cid:107)J ∗

µ
L

1 −

1 −

(cid:17)K

(cid:16)

(cid:16)

(cid:17)K−1

(cid:107)Dl(cid:107)

4∆N −K
L2

.

:,S∗ D:,S∗ . Let K ≤ N be the back-
2 (cid:107)zN − z∗(cid:107). Suppose that ∀n ∈

N converges almost twice as fast as g1

Once the support is reached, ISTA also converges with the same linear rate (1 − µ
L ). Thus the
gradient estimate g2
N in the best case – with optimal sub-
gradient – as O(K(1 − µ
L )2K). This is similar to Ablin et al. (2020, Proposition.5) and Tolooshams
& Ba (2021). Second, Proposition 2.5 shows that (cid:13)
(cid:13)
(cid:13) may increase when the support is not
well-estimated, leading to a deterioration of the gradient estimate. This is due to an accumulation
of errors materialized by the sum in the right-hand side of the inequality, as the term BN (cid:107)z∗(cid:107) may
not vanish to 0 as long as SN (cid:54)⊂ S∗. Interestingly, once the support is reached at iteration S < N ,
the errors converge linearly towards 0, and we recover the fast estimation of g∗ with g2. Therefore,
Lasso-based DDL should either be used with a low number of steps or truncated back-propagation
to ensure stability. These results apply for all linear dictionaries, including convolutions.

l − J N
l

(cid:13)J ∗

Numerical illustrations. We now illustrate these theoretical re-
sults depending on the number N of unrolled iterations. The data
are generated from a random Gaussian dictionary D of size 30×50,
with Bernoulli-Gaussian sparse codes z (sparsity 0.3, σ2
z = 1), and
Gaussian noise (σ2

noise = 0.1) – more details in Appendix A.

Figure 1 conﬁrms the linear convergence of J N
l once the support
is reached. However, the convergence might be unstable when the
number of iteration grows, leading to exploding gradient, as illus-
trated in the second case. When this happens, using a small number
of iterations or truncated back-propagation becomes necessary to
prevent accumulating errors. It is also of interest to look at the pro-
portion of unstable Jacobians (see Figure 2). We recover behaviors
observed in the ﬁrst and second case in Figure 1. 40% samples suf-
fer from numerical instabilities in this example. This has a negative
impact on the gradient estimation outside of the support.

towards J ∗
l

Figure 2: Average conver-
gence of J N
for
l
50 samples. In this example,
40% of the Jacobians are un-
stable (red curves).

We display the convergence behavior of the gradients estimated by
AM and by DDL with different back-propagation depths (20, 50, full) for simulated data and images
in Figure 3. We unroll FISTA instead of ISTA to make the convergence faster. We observed similar

5

100102104Iterations N10-3100020100102104Iterations N010kJNl−J∗lk100102104Iterations N10-810-201020100102104Iterations N020Max BP depthfull2005020kJNl−J∗lkkSN−S∗k0101103Iterations N0102030kJNl−J∗lkPublished as a conference paper at ICLR 2022

Figure 3: Gradient convergence in angle for 1000 synthetic samples (left) and patches from a
noisy image (center). The image is normalized, decomposed into patches of dimension 10 × 10 and
with additive Gaussian noise (σ2 = 0.1). The dictionary for which the gradients are computed is
composed of 128 patches from the image. (right) Relative difference between angles from DDL and
AM. Convergence is faster with DDL in early iterations, and becomes unstable with too many steps.

behaviors for both algorithms in early iterations but using ISTA required too much memory to reach
full convergence. As we optimize using a line search algorithm, we are mainly interested in the abil-
ity of the estimate to provide an adequate descent direction. Therefore, we display the convergence
in angle deﬁned as the cosine similarity (cid:104)g, g∗(cid:105) = T r(gT g∗)
(cid:107)g(cid:107)(cid:107)g∗(cid:107) . The angle provides a good metric to
assert that the two gradients are correlated and thus will lead to similar optimization paths. We also
provide the convergence in norm in appendix. We compare g1
N with the relative difference
of their angles with g∗, deﬁned as (cid:104)g2
. When its value is positive, DDL provides the
best descent direction. Generally, when the back-propagation goes too deep, the performance of g2
N
decreases compared to g1
N , and we observe large numerical instabilities. This behavior is coherent
with the Jacobian convergence patterns studied in Proposition 2.5. Once on the support, g2
N reaches
back the performance of g1
N as anticipated. In the case of a real image, unrolling beats AM by up
to 20% in terms of gradient direction estimation when the number of iterations does not exceed 50,
especially with small back-propagation depth. This highlights that the principal interest of unrolled
algorithms is to use them with a small number of layers – i.e., a small number of iterations.

N ,g∗(cid:105)−(cid:104)g1
1−(cid:104)g1
N ,g∗(cid:105)

N and g2

N ,g∗(cid:105)

3 APPROXIMATE DICTIONARY LEARNING IN PRACTICE

This section introduces practical guidelines on Lasso-based approximate dictionary learning with
unit norm constraint, and we provide empirical justiﬁcations for its ability to recover the dictionary.
We also propose a strategy to scale DDL with a stochastic optimization method. We provide a full
description of all our experiments in Appendix A. We optimize with projected gradient descent com-
bined to a line search to compute high-quality steps sizes. The computations have been performed
on a GPU NVIDIA Tesla V100-DGXS 32GB using PyTorch (Paszke et al., 2019).1

Improvement of precision. As stated before, a low number of iterations allows for efﬁcient and
stable computations, but this makes the sparse code less precise. One can learn the steps sizes of
(F)ISTA to speed up convergence and compensate for imprecise representations, as proposed by
Ablin et al. (2019) for LISTA. To avoid poor results due to large degrees of freedom in unsuper-
vised learning, we propose a method in two steps to reﬁne the initialization of the dictionary before
relaxing the constraints on the steps sizes:

1. We learn the dictionary with ﬁxed steps sizes equal to 1

L where L = (cid:107)D(cid:107)2, given by convergence
conditions. Lipschitz constants or upper bounds are computed at each gradient step with norms,
or the FFT for convolutions, outside the scope of the network graph.

2. Then, once convergence is reached, we jointly learn the step sizes and the dictionary. Both are

still updated using gradient descent with line search to ensure stable optimization.

1Code is available at https://github.com/bmalezieux/unrolled_dl.

6

101103Iterations N10-810-41001−›g,g∗ﬁGaussian dictionary101103Iterations N10-31001−›g,g∗ﬁNoisy image101103Iterations N0.20.00.2Relative diff.Noisy imageBP depthAM2050fullPublished as a conference paper at ICLR 2022

Figure 4: (left) Number of gradient steps performed by the line search before convergence, (center)
distance to the optimal loss, and (right) distance to the optimal dictionary recovery score depending
on the number of unrolled iterations. The data are generated as in Figure 1. We display the mean and
the 10% and 90% quantiles over 50 random experiments. DDL needs less gradient steps to converge
in early iterations, and unrolling obtains high recovery scores with only a few dozens of iterations.

Figure 5: We consider a normalized image degraded by Gaussian noise. (left) PSNR depending on
the number of unrolled iterations for σ2
noise = 0.1, i.e. PSNR = 10 dB. DL-Oracle stands for full
AM dictionary learning (103 iterations of FISTA). There is no need to unroll too many iterations to
obtain satisfying results. (center) PSNR and average recovery score between dictionaries depending
on the SNR for 50 random initializations in CDL. (right) 10 loss landscapes in 1D for σ2
noise = 0.1.
DDL is robust to random initialization when there is not too much noise.

The use of LISTA-like algorithms with no ground truth generally aims at improving the speed of
sparse coding when high precision is not required. When it is the case, the ﬁnal sparse codes can be
computed separately with FISTA (Beck & Teboulle, 2009) or coordinate descent (Wu et al., 2008)
to improve the quality of the representation.

3.1 OPTIMIZATION DYNAMICS IN APPROXIMATE DICTIONARY LEARNING

In this part, we study empirical properties of approximate dictionary learning related to global opti-
mization dynamics to put our results on gradient estimation in a broader context.

Unrolling v. AM.
In Figure 4, we show the number of gradient steps before reaching convergence,
the behavior of the loss FN , and the recovery score deﬁned at the beginning of the section for syn-
i=1 |Cσ(i),i|
thetic data generated by a Gaussian dictionary. As a reminder, S(C) = maxσ∈Sn
where C is the correlation matrix between the columns of the true dictionary and the estimate. The
number of iterations corresponds to N in the estimate zN (D). First, DDL leads to fewer gradient
steps than AM in the ﬁrst iterations. This suggests that automatic differentiation better estimates
the directions of the gradients for small depths. However, computing the gradient requires back-
propagating through the algorithm, and DDL takes 1.5 times longer to perform one gradient step
than AM on average for the same number of iterations N . When looking at the loss and the recovery
score, we notice that the advantage of DDL for the minimization of FN is minor without learning
the steps sizes, but there is an increase of performance concerning the recovery score. DDL bet-
ter estimates the dictionary for small depths, inferior to 50. When unrolling more iterations, AM
performs as well as DDL on the approximate problem and is faster.

(cid:80)n

1
n

Approximate DL. Figure 4 shows that high-quality dictionaries are obtained before the conver-
gence of FN , either with AM or DDL. 40 iterations are sufﬁcient to reach a reasonable solution

7

101102103Iterations N100200NumberGradient steps101102103Iterations N10-1102FN−F∗Loss101102103Iterations N10-310-2SN−S∗Rec. scoreAMDDLDDL + steps100101102Iterations N2025PSNRDenoisingAMDDLDDL_stepsDL-Oracle1018SNR (dB)2125PSNR0.70.9Rec. scoreMin. distribution202Normalized distanceLossCDL minimaPublished as a conference paper at ICLR 2022

concerning the recovery score, even though the loss is still very far from the optimum. This suggests
that computing optimal sparse codes at each gradient step is unnecessary to recover the dictionary.
Figure 5 illustrates that by showing the PSNR of a noisy image reconstruction depending on the
number of iterations, compared to full AM dictionary learning with 103 iterations. As for synthetic
data, optimal performance is reached very fast. In this particular case, the model converges after
80 seconds with approximate DL unrolled for 20 iterations of FISTA compared to 600 seconds in
the case of standard DL. Note that the speed rate highly depends on the value of λ. Higher values
of λ tend to make FISTA converge faster, and unrolling becomes unnecessary in this case. On the
contrary, unrolling is more efﬁcient than AM for lower values of λ.

Loss landscape. The ability of gradient descent to ﬁnd adequate local minima strongly depends on
the structure of the problem. To quantify this, we evaluate the variation of PSNR depending on the
b) where σ2
Signal to Noise Ratio (SNR) (10 log10 (σ2/σ2
b is the variance of the noise) for 50 random
initializations in the context of convolutional dictionary learning on a task of image denoising, with
20 unrolled iterations. Figure 5 shows that approximate CDL is robust to random initialization when
the level of noise is not too high. In this case, all local minima are similar in terms of reconstruction
quality. We provide a visualization of the loss landscape with the help of ideas presented in Li
et al. (2018). The algorithm computes a minimum, and we chose two properly rescaled vectors to
create a plan from this minimum. The 3D landscape is displayed on this plan in Figure B using the
Python library K3D-Jupyter2. We also compare in Figure 5 (right) the shapes of local minima in 1D
by computing the values of the loss along a line between two local minima. These visualizations
conﬁrm that dictionary learning locally behaves like a convex function with similar local minima.

3.2 STOCHASTIC DDL

In order to apply DDL in realistic settings,
it
is tempting to adapt Stochastic Gradient Descent
(SGD), commonly used for neural networks. The
major advantage is that the sparse coding is not per-
formed on all data at each forward pass, leading to
signiﬁcant time and memory savings. The issue is
that the choice of gradient steps is critical to the op-
timization process in dictionary learning, and SGD
methods based on simple heuristics like rate decay
are difﬁcult to tune in this context. We propose to
leverage a new optimization scheme introduced in
Vaswani et al. (2019), which consists of performing
a stochastic line search. The algorithm computes a
good step size at each epoch, after which a heuristic
decreases the maximal step. Figure 6 displays the
recovery score function of the time for various mini-
batch sizes on a problem with 105 samples. The data were generated as in Figure 1 but with a
larger dictionary (50 × 100). The algorithm achieves good performance with small mini-batches and
thus limited memory usage. We also compare this method with Online dictionary learning (Mairal
et al., 2009) in Figure E. It shows that our method speeds up the dictionary recovery, especially
for lower values of λ. This strategy can be adapted very easily for convolutional models by taking
sub-windows of the full signal and performing a stochastic line search, as demonstrated in Section 4.
See Tolooshams et al. (2020) for another unrolled stochastic CDL algorithm applied to medical data.

time for 10
Figure 6: Recovery score vs.
random Gaussian matrices and 105 samples.
Initialization with random dictionaries.
In-
termediate batch sizes offer a good trade-off
between speed and memory usage.

4 APPLICATION TO PATTERN LEARNING IN MEG SIGNALS

In magnetoencephalography (MEG), the measurements over the scalp consist of hundreds of simul-
taneous recordings, which provide information on the neural activity during a large period. Convo-
lutional dictionary learning makes it possible to learn cognitive patterns corresponding to physiolog-
ical activities (Dupr´e la Tour et al., 2018). As the electromagnetic waves propagate through the brain
at the speed of light, every sensor measures the same waveform simultaneously but not at the same

2Package available at https://github.com/K3D-tools/K3D-jupyter.

8

01020304050Time (s)0.40.60.81.0Rec. scoreMinibatch size100500200010000Full batchComplete AMPublished as a conference paper at ICLR 2022

Figure 7:
Stochastic Deep
CDL on 6 minutes of MEG
data (204 channels, sampling
rate of 150Hz). The algo-
rithm uses 40 atoms, 30 un-
rolled iterations and 100 iter-
ations with batch size 20. We
recover heartbeat (0), blink-
ing (1) artifacts, and an au-
ditory evoked response (2)
among others.

Minibatch Time window Steps learning

5
5
5
20

20 s
20 s
10 s
10 s

True
False
True
True

Corr. u
0.85 ± 0.02
0.88 ± 0.02
0.83 ± 0.01
0.85 ± 0.01

Corr. v
0.84 ± 0.06
0.78 ± 0.06
0.82 ± 0.09
0.75 ± 0.09

Mean corr.
0.845
0.83
0.825
0.80

Time
110 s
57 s
56 s
163 s

Table 1: Stochastic Deep CDL on MEG data (as in Figure 7). We compare u and v to 12 important
atoms output by alphacsc (correlation averaged on 5 runs), depending on several hyperparame-
ters, with 30 layers, 10 epochs and 10 iterations per epochs. λrescaled = 0.3λmax, λmax = (cid:13)
(cid:13)∞.
The best setups achieve 80% – 90% average correlation with alphacsc in around 100 sec. com-
pared to around 1400 sec. Our method is also faster than convolutional K-SVD (Yellin et al., 2017).

(cid:13)DT y(cid:13)

intensity. The authors propose to rely on multivariate convolutional sparse coding (CSC) with rank-
1 constraint to leverage this physical property and learn prototypical patterns. In this case, space and
time patterns are disjoint in each atom: Dk = ukvT
k where u gathers the spatial activations on each
channel and v corresponds to the temporal pattern. This leads to the model

min
zk∈RT ,uk∈RS ,vk∈Rt

1
2

n
(cid:88)

(ukv(cid:62)

k ) ∗ zk − y

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

k=1

(cid:13)
2
(cid:13)
(cid:13)
(cid:13)
(cid:13)
2

+ λ

n
(cid:88)

k=1

(cid:107)zk(cid:107)1 ,

(6)

where n is the number of atoms, T is the total recording time, t is the kernel size, and S is the number
of sensors. We propose to learn u and v with Stochastic Deep CDL unrolled for a few iterations to
speed up the computations of the atoms. Figure 7 reproduces the multivariate CSC experiments of
alphacsc3 (Dupr´e la Tour et al., 2018) on the dataset sample of MNE (Gramfort et al., 2013) – 6
minutes of recordings with 204 channels sampled at 150Hz with visual and audio stimuli.

The algorithm recovers the main waveforms and spatial patterns with approximate sparse codes and
without performing the sparse coding on the whole data set at each gradient iteration, which leads
to a signiﬁcant gain of time. We are able to distinguish several meaningful patterns as heartbeat
and blinking artifacts or auditive evoked response. As this problem is unsupervised, it is difﬁcult
to provide robust quantitative quality measurements. Therefore, we compare our patterns to 12
important patterns recovered by alpahcsc in terms of correlation in Table 1. Good setups achieve
between 80% and 90% average correlation ten times faster.

5 CONCLUSION

Dictionary learning is an efﬁcient technique to learn patterns in a signal but is challenging to ap-
ply to large real-world problems. This work showed that approximate dictionary learning, which
consists in replacing the optimal solution of the Lasso with a time-efﬁcient approximation, offers
a valuable trade-off between computational cost and quality of the solution compared to complete
Alternating Minimization. This method, combined with a well-suited stochastic gradient descent
algorithm, scales up to large data sets, as demonstrated on a MEG pattern learning problem. This
work provided a theoretical study of the asymptotic behavior of unrolling in approximate dictionary
learning. In particular, we showed that numerical instabilities make DDL usage inefﬁcient when too
many iterations are unrolled. However, the super-efﬁciency of DDL in the ﬁrst iterations remains
unexplained, and our ﬁrst ﬁndings would beneﬁt from theoretical support.

3Package and experiments available at https://alphacsc.github.io

9

Spatial pattern 0Spatial pattern 1Spatial pattern 20.00.51.0Time (s)0.50.0Temporal pattern 00.00.51.0Time (s)0.20.0Temporal pattern 10.00.51.0Time (s)0.00.20.4Temporal pattern 2Published as a conference paper at ICLR 2022

ETHICS STATEMENT

The MEG data conform to ethic guidelines (no individual names, collected under individual’s con-
sent, . . . ).

REPRODUCIBILITY STATEMENT

Code is available at https://github.com/bmalezieux/unrolled_dl. We provide a
full description of all our experiments in Appendix A, and the proofs of our theoretical results in
Appendix C.

ACKNOWLEDGMENTS

This work was supported by grants from Digiteo France.

REFERENCES

Pierre Ablin, Thomas Moreau, Mathurin Massias, and Alexandre Gramfort. Learning step sizes
for unfolded sparse coding. In Advances in Neural Information Processing Systems, pp. 13100–
13110, 2019.

Pierre Ablin, Gabriel Peyr´e, and Thomas Moreau. Super-efﬁciency of automatic differentiation for
functions deﬁned as a minimum. In Proceedings of the 37th International Conference on Machine
Learning, pp. 32–41, 2020.

Alekh Agarwal, Animashree Anandkumar, Prateek Jain, and Praneeth Netrapalli. Learning sparsely
used overcomplete dictionaries via alternating minimization. SIAM Journal on Optimization, 26
(4):2775–2799, 2016.

Michal Aharon, Michael Elad, and Alfred Bruckstein. K-svd: An algorithm for designing overcom-
plete dictionaries for sparse representation. IEEE Transactions on Signal Processing, 54:4311 –
4322, 2006.

Amir Beck and Marc Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse

problems. SIAM J. Imaging Sciences, 2:183–202, 2009.

Quentin Bertrand, Quentin Klopfenstein, Mathieu Blondel, Samuel Vaiter, Alexandre Gramfort, and
Joseph Salmon. Implicit differentiation of lasso-type models for hyperparameter optimization. In
International Conference on Machine Learning, pp. 810–821. PMLR, 2020.

Xiaohan Chen, Jialin Liu, Zhangyang Wang, and Wotao Yin. Theoretical linear convergence of
unfolded ista and its practical weights and thresholds. Advances in Neural Information Processing
Systems, 2018.

Scott R Cole and Bradley Voytek. Brain oscillations and the importance of waveform shape. Trends

in cognitive sciences, 21(2):137–149, 2017.

John M. Danskin. Theory of Max-Min and Its Application to Weapons Allocation Problems. Springer

Berlin Heidelberg, Berlin/Heidelberg, 1967.

Ingrid Daubechies, Michel Defrise, and Christine Mol. An iterative thresholding algorithm for linear
inverse problems with a sparsity constrains. Communications on Pure and Applied Mathematics,
57, 2004.

Charles-Alban Deledalle, Samuel Vaiter, Jalal Fadili, and Gabriel Peyr´e. Stein unbiased gradient
estimator of the risk (sugar) for multiple parameter selection. SIAM Journal on Imaging Sciences,
7(4):2448–2487, 2014.

Tom Dupr´e la Tour, Thomas Moreau, Mainak Jas, and Alexandre Gramfort. Multivariate convolu-
tional sparse coding for electromagnetic brain signals. Advances in Neural Information Process-
ing Systems, 31:3292–3302, 2018.

10

Published as a conference paper at ICLR 2022

Alexandre Gramfort, Martin Luessi, Eric Larson, Denis A Engemann, Daniel Strohmeier, Christian
Brodbeck, Roman Goj, Mainak Jas, Teon Brooks, Lauri Parkkonen, et al. Meg and eeg data
analysis with mne-python. Frontiers in neuroscience, 7:267, 2013.

Karol Gregor and Yann LeCun. Learning fast approximations of sparse coding. International con-

ference on machine learning, pp. 399–406, 2010.

R´emi Gribonval, Rodolphe Jenatton, and Francis Bach. Sparse and spurious: dictionary learning
with noise and outliers. IEEE Transactions on Information Theory, 61(11):6298–6319, 2015.

Roger Grosse, Rajat Raina, Helen Kwong, and Andrew Y. Ng. Shift-Invariant Sparse Coding for

Audio Classiﬁcation. Cortex, 8:9, 2007.

Benjamin D Haeffele and Ren´e Vidal. Global optimality in tensor factorization, deep learning, and

beyond. arXiv preprint arXiv:1506.07540, 2015.

Bruno Lecouat, Jean Ponce, and Julien Mairal. A ﬂexible framework for designing trainable priors
In Advances in neural information processing

with adaptive smoothing and game encoding.
systems, 2020.

Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. Visualizing the loss land-
In Advances in neural information processing systems, pp. 6389–6399,

scape of neural nets.
2018.

Jialin Liu and Xiaohan Chen. Alista: Analytic weights are as good as learned weights in lista. In

International Conference on Learning Representations, 2019.

Julien Mairal, Francis Bach, J. Ponce, and Guillermo Sapiro. Online learning for matrix factorization

and sparse coding. Journal of Machine Learning Research, 11, 2009.

Thomas Moreau and Joan Bruna. Understanding neural sparse coding with matrix factorization. In

International Conference on Learning Representation, 2017.

Thomas Moreau and Alexandre Gramfort. Dicodile: Distributed convolutional dictionary learning.

IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020.

Bruno A. Olshausen and David J Field. Sparse coding with an incomplete basis set: A strategy

employed by \protect{V1}. Vision Research, 37(23):3311–3325, 1997.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-
performance deep learning library. In Advances in neural information processing systems, pp.
8026–8037, 2019.

Meyer Scetbon, Michael Elad, and Peyman Milanfar. Deep k-svd denoising. IEEE Transactions on

Image Processing, 30:5944–5955, 2021.

Amirreza Shaban, Ching-An Cheng, Nathan Hatch, and Byron Boots. Truncated back-propagation
for bilevel optimization. In International Conference on Artiﬁcial Intelligence and Statistics, pp.
1723–1732. PMLR, 2019.

Ju Sun, Qing Qu, and John Wright. Complete dictionary recovery over the sphere i: Overview and

the geometric picture. IEEE Transactions on Information Theory, 63(2):853–884, 2016.

Wen Tang, Emilie Chouzenoux, Jean-Christophe Pesquet, and Hamid Krim. Deep transform and
metric learning network: Wedding deep dictionary learning and neural networks. arXiv preprint
arXiv:2002.07898, 2020.

Robert Tibshirani. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical

Society Series B, 58:267–288, 1996.

Ryan J. Tibshirani. The lasso problem and uniqueness. Electronic Journal of Statistics, 7(1):1456–

1490, 2013.

11

Published as a conference paper at ICLR 2022

Bahareh Tolooshams and Demba Ba. Pudle: Implicit acceleration of dictionary learning by back-

propagation. arXiv preprint, 2021.

Bahareh Tolooshams, Sourav Dey, and Demba Ba. Deep residual autoencoders for expectation
maximization-inspired dictionary learning. IEEE Transactions on Neural Networks and Learning
Systems, PP:1–15, 2020.

Sharan Vaswani, Aaron Mishkin, Issam Laradji, Mark Schmidt, Gauthier Gidel, and Simon Lacoste-
Julien. Painless stochastic gradient: Interpolation, line-search, and convergence rates. Advances
in neural information processing systems, 32:3732–3745, 2019.

Tong Tong Wu, Kenneth Lange, et al. Coordinate descent algorithms for lasso penalized regression.

Annals of Applied Statistics, 2(1):224–244, 2008.

Yande Xiang, Zhitao Lin, and Jianyi Meng. Automatic qrs complex detection using two-level con-

volutional neural network. Biomedical engineering online, 17(1):1–17, 2018.

Florence Yellin, Benjamin D Haeffele, and Ren´e Vidal. Blood cell detection and counting in holo-
graphic lens-free imaging by convolutional sparse dictionary learning and coding. In International
Symposium on Biomedical Imaging, pp. 650–653. IEEE, 2017.

John Zarka, Louis Thiry, Tomas Angles, and Stephane Mallat. Deep network classiﬁcation by
scattering and homotopy dictionary learning. In International Conference on Learning Represen-
tations, 2019.

A FULL DESCRIPTION OF THE EXPERIMENTS

This section provides complementary information on the experiments presented in the paper.

A.1 CONVERGENCE OF THE JACOBIANS - FIGURE 1 AND FIGURE 2

We generate a normalized random Gaussian dictionary D of dimension 30 × 50, and sparse codes
z from a Bernoulli Gaussian distribution of sparsity 0.3 and σ2 = 1. The signal to process is y =
Dz + b where b is an additive Gaussian noise with σ2
noise = 0.1. The Jacobians are computed for a
random perturbation D +bD of D where bD is a Gaussian noise of scale 0.5σ2
l corresponds to
the approximate Jacobian with N iterations of ISTA with λ = 0.1. J ∗
l corresponds the true Jacobian
computed with sparse codes obtained after 104 iterations of ISTA with λ = 0.1.
In Figure 2, the norm (cid:13)

(cid:13)
(cid:13) is computed for 50 samples.

D. J N

(cid:13)J N

l − J ∗
l

A.2 CONVERGENCE OF THE GRADIENT ESTIMATES - FIGURE 3

Synthetic data. We generate a normalized random Gaussian dictionary D of dimension 30 × 50,
and 1000 sparse codes z from a Bernoulli Gaussian distribution of sparsity 0.3 and σ2 = 1. The
signal to process is y = Dz + b where b is an additive Gaussian noise with σ2
noise = 0.1. The
gradients are computed for a random perturbation D + bD of D where bD is a Gaussian noise of
scale 0.5σ2
D.

Noisy image. A 128 × 128 black-and-white image is degraded by a Gaussian noise with σ2
noise =
0.1 and normalized. We processed 1000 patches of dimension 10 × 10 from the image, and we
computed the gradients for a dictionary composed of 128 random patches.
gN corresponds to the gradient for N iterations of FISTA with λ = 0.1. g∗ corresponds to the true
gradient computed with a sparse code obtained after 104 iterations of FISTA.

A.3 OPTIMIZATION DYNAMICS ON SYNTHETIC DATA - FIGURE 4

We generate a normalized random Gaussian dictionary D of dimension 30 × 50, and sparse codes
z from a Bernoulli Gaussian distribution of sparsity 0.3 and σ2 = 1. The signal to process is
y = Dz + b where b is an additive Gaussian noise with σ2
noise = 0.1. The initial dictionary

12

Published as a conference paper at ICLR 2022

is taken as a random perturbation D + bD of D where bD is a Gaussian noise of scale 0.5σ2
D.
N corresponds to the number of unrolled iterations of FISTA. F ∗ is the value of the loss for 103
iterations minus 10−3. S∗ is the score obtained after 103 iterations plus 10−3. The optimization is
done with λ = 0.1. We compare the number of gradient steps (left), the loss values (center), and the
recovery scores (right) for 50 different dictionaries. DDL with steps sizes learning is evaluated on
100 iterations only due to memory and optimization time issues.

A.4 OPTIMIZATION DYNAMICS AND LOSS LANDSCAPES ON IMAGES - FIGURE 5

A 128 × 128 black-and-white image is degraded by a Gaussian noise and normalized.

In this experiment, σ2

noise = 0.1. We learn a dictionary composed of 128 atoms on 10 × 10
Left.
patches with FISTA and λ = 0.1 in all cases. The PSNR is obtained with sparse codes output by
the network. The results are compared to the truth with the Peak Signal to Noise Ratio. Dictionary
learning denoising with 1000 iterations of FISTA is taken as a baseline.

Center. We learn 50 dictionaries from 50 random initializations in convolutional dictionary learn-
ing with 50 kernels of size 8 × 8 with 20 unrolled iterations of FISTA and λ = 0.1. The PSNR is
obtained with sparse codes output by the network. We compare the average, minimal and maximal
PSNR, and recovery scores with all other dictionaries to study the robustness to random initialization
depending on the level of noise (SNR).

In this experiment, σ2

Right.
noise = 0.1. We learn 2 dictionaries from 2 random initializations in
convolutional dictionary learning with 50 kernels of size 8 × 8 with 20 unrolled iterations of FISTA
and λ = 0.1. We display the loss values on the line between these two dictionaries. The experiment
is repeated on 10 different random initializations.

A.5 STOCHASTIC DDL ON SYNTHETIC DATA - FIGURE 6

We generate a normalized random Gaussian dictionary D of dimension 50 × 100, and 105 sparse
codes z from a Bernoulli Gaussian distribution of sparsity 0.3 and σ2 = 1. The signal to process
is y = Dz + b where b is an additive Gaussian noise with σ2
noise = 0.1. The initial dictionary is
taken as a random gaussian dictionary. We compare stochastic and full-batch line search projected
gradient descent with 30 unrolled iterations of FISTA and λ = 0.1, without steps sizes learning.
Stochastic DDL is run for 10 epochs with a maximum of 100 iterations for each epoch.

A.6 PATTERN LEARNING IN MEG - FIGURE 7

Stochastic Deep CDL on 6 minutes of recordings of MEG data with 204 channels and a sampling
rate of 150Hz. We remove the powerline artifacts and high-pass ﬁlter the signal to remove the
drift which can impact the CSC technique. The signal is also resampled to 150 Hz to reduce the
computational burden. This preprocessing procedure is presented in alphacsc, and available in
the code in the supplementary materials. The algorithm learns 40 atoms of 1 second on mini batches
of 10 seconds, with 30 unrolled iterations of FISTA, λscaled = 0.3, and 10 epochs with 10 iterations
per epoch. The number of mini-batches per iteration is 20, with possible overlap.

B EXTRA FIGURES AND EXPERIMENTAL RESULTS

LISTA - Figure A.
D = 1
N = 3. W 1
by the network is an approximation of the solution of the LASSO.

Illustration of LISTA for Dictionary Learning with initialization Z0 = 0 for
L (D)(cid:62)D), where L = (cid:107)D(cid:107)2. The result ZN (D) output

L (D)(cid:62), W 2

D = (I − 1

Loss landscape in 2D - Figure B. We provide a visualization of the loss landscape with the help of
ideas presented in Li et al. (2018). The algorithm computes a minimum, and we chose two properly
rescaled vectors to create a plan from this minimum. The 3D landscape is displayed on this plan in
the appendix using the Python library K3D-Jupyter. This visualization and the visualization in 1D
conﬁrm that (approximate) dictionary learning locally behaves like a convex function with smooth
local minima.

13

Published as a conference paper at ICLR 2022

Figure A: LISTA

Figure B: Loss landscape in approximate CDL

Gradient convergence in norm - Figure C. Gradient estimates convergence in norm for synthetic
data (left) and patches from a noisy image (right). The setup is similar to Figure 3. Both gradient
estimates converge smoothly in early iterations. When the back-propagation goes too deep, the
performance of g2
N , and we observe large numerical instabilities. This
behavior is coherent with the Jacobian convergence patterns studied in Proposition 2.5. Once on the
support, g2
N reaches back the performance of g1
N .

N decreases compared to g1

Figure C: Gradient estimates convergence in norm for synthetic data (left) and patches from a noisy
image (right). Both gradient estimates converge smoothly in early iterations, after what DDL gradi-
ent becomes unstable. The behavior returns to normal once the algorithm reaches the support.

Computation time to reach 0.95 recovery score - Figure D. The setup is similar to Figure 6. A
random Gaussian dictionary of size 50 × 100 generates the data from 105 sparse codes with sparsity
0.3. The approximate sparse coding is solved with λ = 0.1 and 30 unrolled iterations of FISTA.
The algorithm achieves good performances with small mini-batches and thus limited memory usage.
Stochastic DDL can process large amounts of data and recovers good quality dictionaries faster than
full batch DDL.

Sto DDL vs. Online DL - Figure E. We compare the time Online DL from spams4 (Mairal
et al., 2009) and Stochastic DDL need to reach a recovery score of 0.95 with a batch size of 2000.
Online DL is run with 10 threads. We repeat the experiment 10 times for different values of λ from

4package available at http://thoth.inrialpes.fr/people/mairal/spams/

14

yW1DW2DW1DW2DW1DzN(D)100101102103Iterations N10-1101103||g∗−g||Gaussian dictionaryBP depth2050fullAM100101102103Iterations N101103||g∗−g||Noisy imagePublished as a conference paper at ICLR 2022

Figure D: Time to reach a recovery score of 0.95. Intermediate batch sizes offer a good trade-off
between speed and memory usage compared to full-batch DDL.

0.1 to 1.0. The setup is similar to Figure D, and we initialize both methods randomly. Stochastic
DDL is more efﬁcient for smaller values of λ, due to the fact that sparse coding is slower in this
case. For higher values of λ, both methods are equivalent. Another advantage of Stochastic DDL
is its modularity.
It works on various kinds of dictionary parameterization thanks to automatic
differentiation, as illustrated on 1-rank multivariate convolutional dictionary learning in Figure 7.

Figure E: Comparison between Online DL and Stochastic DDL. Stochastic DDL is more efﬁcient
for smaller values of λ, due to the fact that sparse coding is slower in this case.

C PROOFS OF THEORETICAL RESULTS

This section gives the proofs for the various theoretical results in the paper.

C.1 PROOF OF PROPOSITION 2.1.

Proposition 2.1 Let D∗ = arg minD∈C G(D) and D∗
N = arg minD∈C GN (D), where N is the
number of unrolled iterations. We denote by K(D∗) a constant depending on D∗, and by C(N ) the
convergence speed of the algorithm, which approximates the inner problem solution. We have

GN (D∗

N ) − G(D∗) ≤ K(D∗)C(N ) .

Let G(D) (cid:44) F (Z∗(D), D) and GN (D) (cid:44) F (ZN (D), D) where Z∗(D) =
arg minZ∈Rn×T F (Z, D) and ZN (D) = F IST A(D, N ). Let D∗ = arg minD∈C G(D) and
D∗

N = arg minD∈C GN (D). We have
GN (D∗

N ) − G(D∗) = GN (D∗

N ) − GN (D∗) + GN (D∗) − G(D∗)

= F (ZN (DN ), DN ) − F (ZN (D∗), D∗)
+ F (ZN (D∗), D∗) − F (Z(D∗), D∗)

15

(7)
(8)
(9)

100500200010000DDLOracle DLMinibatch size01020304050Time (s)TimeoutTime to reach a score of 0.950.20.40.60.81.0λ101102Time (s)Time to reach a score of 0.95Sto. DDLOnline DLPublished as a conference paper at ICLR 2022

By deﬁnition of D∗
N

F (ZN (D∗

N ), D∗

N ) − F (ZN (D∗), D∗) ≤ 0

The convergence rate of FISTA in function value for a ﬁxed dictionary D is

Therefore

Hence

F (ZN (D), D) − F (ZN (D), D) ≤

F (ZN (D∗), D∗) − F (Z(D∗), D∗) ≤

K(D)
N 2

K(D∗)
N 2

GN (D∗

N ) − G(D∗) ≤

K(D∗)
N 2

C.2 PROOF OF PROPOSITION 2.2

(10)

(11)

(12)

(13)

Proposition 2.2 Let D ∈ Rm×n. Then, there exists a constant L1 > 0 such that for every number
of iterations N

(cid:13)
(cid:13)g1

N − g∗(cid:13)

(cid:13) ≤ L1 (cid:107)zN (D) − z∗(D)(cid:107) .

We have

F (z, D) =

1
2

(cid:107)Dz − y(cid:107)2

2 + λ (cid:107)z(cid:107)1

∇2F (z, D) = (Dz − y)z(cid:62)

(14)

(15)

z0(D) = 0 and the iterates (zN (D))N ∈N converge towards z∗(D). Hence, they are contained in
a closed ball around z∗(D). As ∇2F (·, D) is continuously differentiable, it is locally Lipschitz on
this closed ball, and there exists a constant L1(D) depending on D such that

(cid:13)
(cid:13)g1

N − g∗(cid:13)

(cid:13) = (cid:107)∇2F (zN (D), D) − ∇2F (z∗(D), D)(cid:107)

≤ L1(D) (cid:107)zN (D) − z∗(D)(cid:107)

C.3 PROOF OF PROPOSITION 2.3.

(16)

(17)

Proposition 2.3 Let D ∈ Rm×n. Let S∗ be the support of z∗(D), SN be the support of zN and
(cid:101)SN = SN ∪ S∗. Let f (z, D) = 1
2 be the data-ﬁtting term in F . Let R(J, (cid:101)S) =
J+(cid:0)∇2
(cid:1) + ∇2
(cid:101)S. Then there exists a constant L2 > 0 and a sub-
sequence of (F)ISTA iterates zφ(N ) such that for all N ∈ N:

1,1f (z∗, D) (cid:12) 1
(cid:101)S

2,1f (z∗, D) (cid:12) 1

2 (cid:107)Dz − y(cid:107)2

∃ g2

φ(N ) ∈ ∇2f (zφ(N ), D) + J+
φ(N ) − g∗(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13) ≤
(cid:13)R(Jφ(N ), (cid:101)Sφ(N ))

φ(N )

(cid:13)
(cid:13)g2
(cid:13)

L2
2
This sub-sequence zφ(N ) corresponds to iterates on the support of z∗.

∇1f (zφ(N ), D) + λ∂(cid:107)·(cid:107)1
(cid:13)
(cid:13)
(cid:13)zφ(N ) − z∗(cid:13)
(cid:13)
(cid:13)

(cid:13) +

(cid:16)

(cid:17)

(zφ(N ))

s.t. :

(cid:13)
(cid:13)zφ(N ) − z∗(cid:13)
2
(cid:13)

We have

N (D) ∈ ∇2f (zN (D), D) + J+
g2

N

(cid:0)∇1f (zN (D), D) + λ∂(cid:107)·(cid:107)1

(zN )(cid:1)

We adapt equation (6) in Ablin et al. (2020)

N = g∗ + R(JN , (cid:102)SN )(zN − z∗) + RD,z
g2

N + J+

N Rz,z

N

where

R(J, (cid:101)S) = J+(cid:0)∇2

1,1f (z∗, D) (cid:12) 1
(cid:101)S
RD,z
N = ∇2f (zN , D) − ∇2f (z∗, D) − ∇2
Rz,z
(zN ) − ∇2
N ∈ ∇1f (zN , D) + λ∂(cid:107)·(cid:107)1

(cid:1) + ∇2

2,1f (z∗, D) (cid:12) 1
(cid:101)S

2,1f (z∗, D)(zN − z∗)
1,1f (z∗, D)(zN − z∗)

16

.

(18)

(19)

(20)

(21)

(22)

Published as a conference paper at ICLR 2022

As zN and z∗ are on (cid:102)SN

∇2
2,1f (z∗, D)(zN − z∗) =
∇2
1,1f (z∗, D)(zN − z∗)(cid:1) = J+(cid:16)

2,1f (z∗, D) (cid:12) 1

(cid:102)SN
1,1f (z∗, D) (cid:12) 1

∇2

J+(cid:0)∇2

(cid:16)

(cid:17)

(zN − z∗)

(cid:17)
(zN − z∗)

(cid:102)SN

(23)

(24)

As stated in Proposition 2.2, ∇2f (·, D) is locally Lipschitz, and RD,z
∇2f (·, D). Therefore, there exists a constant LD,z such that

N

is the Taylor rest of

(cid:13)
(cid:13)RD,z
(cid:13)
We know that 0 ∈ ∇1f (z∗, D) + λ∂(cid:107)·(cid:107)1
u∗ = 0. Therefore we have:

∀N ∈ N,

N

(cid:13)
(cid:13)
(cid:13) ≤

LD,z
2

(cid:107)zN (D) − z∗(D)(cid:107)2

(z∗). In other words, ∃u∗ ∈ λ∂(cid:107)·(cid:107)1

(25)

(z∗) s.t. ∇1f (z∗, D) +

Rz,z

N ∈ ∇1f (zN , D) − ∇1f (z∗, D) − ∇2

1,1f (z∗, x)(zN − z∗) + λ∂ (cid:107)zN (cid:107)1 − u∗

(26)

Let Lz,z be the Lipschitz constant of ∇1f (·, D).
(F)ISTA outputs a sequence such that there
exists a sub-sequence (zφ(N ))N ∈N which has the same support as z∗. For this sub-sequence,
u∗ ∈ λ∂(cid:107)·(cid:107)1

(zφ(N )). Therefore, there exists Rz,z

φ(N ) such that

1. Rz,z
φ(N ) ∈ ∇1f (zφ(N ), D) + λ∂(cid:107)·(cid:107)1
(cid:13)
(cid:13)zφ(N ) − z∗(cid:13)
(cid:13)
2
(cid:13)Rz,z
(cid:13)
(cid:13)

(cid:13)
(cid:13) ≤ Lz,z
(cid:13)

φ(N )

2.

2

(zφ(N )) − ∇2

1,1f (z∗, x)(zφ(N ) − z∗)

For this sub-sequence, we can adapt Proposition 2 from Ablin et al. (2020). Let L2 = LD,z + Lz,z,
we have

∃ g2
φ(N ) ∈ ∇2f (zφ(N ), D) + Jφ(N )
(cid:13)
(cid:13)
φ(N ) − g∗(cid:13)
(cid:13)
(cid:13)R(Jφ(N ), (cid:94)Sφ(N ))
(cid:13)g2
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13) ≤
(cid:13)

(cid:0)∇1f (zφ(N ), D) + λ∂ (cid:13)
L2
(cid:13)zφ(N ) − z∗(cid:13)
(cid:13)
2

(cid:13) +

(cid:13)
(cid:1), s.t. :
(cid:13)zφ(N )
(cid:13)1
(cid:13)zφ(N ) − z∗(cid:13)
(cid:13)
2
(cid:13)

(27)

(28)

C.4 PROOF OF THEOREM 2.4.

Theorem 2.4 At iteration N + 1 of ISTA, the weak Jacobian of zN +1 relatively to Dl, where Dl is
the l-th row of D, is given by induction:
(cid:18) ∂(zN )
∂Dl

l zN − yl)In + D(cid:62)D

∂(zN +1)
∂Dl

= 1|zN +1|>0 (cid:12)

∂(zN )
∂Dl

N + (D(cid:62)

Dlz(cid:62)

1
L

(cid:19)(cid:19)

−

(cid:18)

.

∂(zN )
∂Dl

whose values are

will be denoted by J N
l

. It converges towards the weak Jacobian J ∗

l of z∗ relatively to Dl,

l S∗ = −(D(cid:62)
J ∗

:,S∗ D:,S∗ )−1(Dlz∗(cid:62) + (D(cid:62)

l z∗ − yl)In)S∗ ,

on the support S∗ of z∗, and 0 elsewhere. Moreover, R(J∗, S∗) = 0.

We start by recalling a Lemma from Deledalle et al. (2014).

Lemma C.1 The soft-thresholding STµ deﬁned by STµ(z) = sgn(z) (cid:12) (|z| − µ)+ is weakly dif-
ferentiable with weak derivative dSTµ(z)

dz = 1|z|>µ.

Coordinate-wise, ISTA corresponds to the following equality:

zN +1 = STµ((I −

1
L

D(cid:62)D)zN +

1
L

D(cid:62)y)

(zN +1)i = STµ((zN )i −

DjiDjp)(zN )p +

1
L

m
(cid:88)

n
(cid:88)
(

p=1

j=1

17

1
L

n
(cid:88)

j=1

Djiyj)

(29)

(30)

Published as a conference paper at ICLR 2022

The Jacobian is computed coordinate wise with the chain rule:

∂(zN +1)i
∂Dlk

Last term:

= 1|(zN +1)i|>0 · (

∂(zN )i
∂Dlk

−

1
L

∂
∂Dlk

n
m
(cid:88)
(cid:88)
(

(

p=1

j=1

DjiDjp)(zN )p) +

1
L

∂
∂Dlk

n
(cid:88)

j=1

Djiyj))

∂
∂Dlk

n
(cid:88)

j=1

Djiyj = δikyl

(31)

(32)

Second term:

∂
∂Dlk

m
(cid:88)

n
(cid:88)

p=1

j=1

DjiDjp(zN )p =

m
(cid:88)

n
(cid:88)

p=1

j=1

DjiDjp

∂(zN )p
∂Dlk

+

m
(cid:88)

n
(cid:88)

p=1

j=1

∂DjiDjp
∂Dlk

(zN )p

(33)

∂DjiDjp
∂Dlk

=






2Dlk
Dlp
Dli
0

if j = l and i = p = k
if j = l and i = k and p (cid:54)= k
if j = l and i (cid:54)= k and p = k
else

Therefore:

m
(cid:88)

n
(cid:88)

p=1

j=1

∂DjiDjp
∂Dlk

(zN )p =

m
(cid:88)

(2Dlkδipδik + Dliδpk1i(cid:54)=k + Dlpδik1k(cid:54)=p)(zN )p

p=1

(34)

(35)

= 2Dlk(zN )kδik + Dli(zN )k1i(cid:54)=k +

= Dli(zN )k + δik

m
(cid:88)

p=1

Dlp (zN )p

Dlp(zN )pδik

(36)

m
(cid:88)

p=1
p(cid:54)=k

(37)

(38)

Hence:

∂(zN +1)i
∂Dlk

= 1|(zN +1)i|>0 ·

(cid:16) ∂(zN )i
∂Dlk
m
(cid:88)

δik(

p=1

−

1
L

(Dli(zN )k+

Dlp(zN )p) +

m
(cid:88)

n
(cid:88)

p=1

j=1

∂(zN )p
∂Dlk

DjiDjp − δikyl)

(cid:17)

This leads to the following vector formulation:

∂(zN +1)
∂Dl

= 1|zN +1|>0 (cid:12)

(cid:18) ∂(zN )
∂Dl

−

1
L

(cid:18)

Dlz(cid:62)

N + (D(cid:62)

l zN − yl)Im + D(cid:62)D

(cid:19)(cid:19)

∂(zN )
∂Dl

(39)

On the support of z∗, denoted by S∗, this quantity converges towards the ﬁxed point:

l = −(D(cid:62)
J ∗

:,S∗ D:,S∗ )−1(Dlz∗(cid:62) + (D(cid:62)

l z∗ − yl)Im)S∗

(40)

Elsewhere, J ∗
l
tion 39

is equal to 0. To prove that R(J∗, S∗) = 0, we use the expression given by equa-

J∗ = 1S∗ (cid:12)

(cid:18)

J∗ −

1
L

(cid:0)∇2

2,1f (z∗, Dl)(cid:62) + ∇2

1,1f (z∗, D)(cid:62)J∗(cid:1)

J∗ − 1S∗ (cid:12) J∗ =

2,1f (z∗, Dl)(cid:62) + 1S∗ (cid:12) ∇2

1,1f (z∗, D)(cid:62)J∗

1,1f (z∗, D) (cid:12) 1S∗

(cid:1) + ∇2

2,1f (z∗, D) (cid:12) 1S∗

1S∗ (cid:12) ∇2

1
L
0 = J∗+(cid:0)∇2
0 = R(J∗, S∗)

(cid:19)

(41)

(42)

(43)

(44)

18

Published as a conference paper at ICLR 2022

C.5 PROOF OF PROPOSITION 2.5 AND COROLLARY 2.6

Proposition 2.5 Let N be the number of iterations and K be the back-propagation depth. We as-
sume that ∀n ≥ N −K, S∗ ⊂ Sn. Let ¯EN = Sn \S∗, let L be the largest eigenvalue of D(cid:62)
:,S∗ D:,S∗ ,
(cid:13)
(cid:13)
(cid:13)
(cid:13)
and let µn be the smallest eigenvalue of D(cid:62)
(cid:13), where
(cid:13)PEn
:,S∗ PS∗
PS is the projection on RS and D† is the pseudo-inverse of D. We have

:,Sn D:,Sn−1. Let Bn =

− D(cid:62)

D†(cid:62)

:, ¯En

(cid:13)
(cid:13)J N

l − J ∗
l

(cid:13)
(cid:13) ≤

K
(cid:89)

(cid:16)

1 −

k=1

(cid:17)

µN −k
L

(cid:107)J ∗

l (cid:107)+

2
L

(cid:107)Dl(cid:107)

K−1
(cid:88)

k
(cid:89)

(1−

k=0

i=1

µN −i
L

)

(cid:16) (cid:13)

(cid:13)zN −k
l

− z∗
l

(cid:13)
(cid:13)+BN −k (cid:107)z∗
l (cid:107)

(cid:17)

.

We denote by G the matrix (I − 1
have with the induction in Theorem 2.4

L D(cid:62)D). For zN with support SN and z∗ with support S∗, we

l,SN = (cid:0)GJ N −1
J N
l,S∗ = (cid:0)GJ ∗
J ∗

l
l + u∗
l

(cid:1)

S∗

+ uN −1
l

(cid:1)

SN

(45)

(46)

l = − 1
L

where uN
We can thus decompose their difference as the sum of two terms, one on the support S∗ and one on
this complement EN = SN \ S∗

l zN − yl)I(cid:1) and the other terms on ¯SN and ¯S∗ are 0.

N + (D(cid:62)

(cid:0)Dlz(cid:62)

l − J N
J ∗

l = (J ∗

l − J N

l )S∗ + (J ∗

l − J N

l )EN

.

Recall that we assume S∗ ⊂ SN . Let’s study the terms separately on S∗ and EN = SN \ S∗. These
two terms can be decompose again to constitute a double recursion system,

(J N

l − J ∗

l )S∗ = GS∗ (J N −1

− J ∗

l
= GS∗,S∗ (J N −1

l

− u∗

l ) + (uN −1
− J ∗

l )S∗
l )S∗ + GS∗,EN −1

l

(J N −1
l

− J ∗)EN −1

+ (uN −1
l

− u∗

l )S∗ ,

(47)

(J N

l − J ∗

l )EN

= (J N
l )EN
= GEN
= GEN ,S∗ (J N −1
+ (uN −1
− u∗
l

− J ∗

l ) + GEN ,S∗ J ∗
(J N −1
l

(J N −1
l
− J ∗
l )S∗ + GEN ,EN −1
− D(cid:62)

l + (uN −1
)EN
l
− J ∗
l )EN −1
:,S∗ D:,S∗ )−1(u∗

D:,S∗ (D(cid:62)

(u∗

+

l )EN

l )EN

(cid:16)

l

:,EN

l )S∗

(48)

(49)

(50)

(cid:17)

.

We deﬁne as PSN ,EN
SN . As S∗ ∪ EN = SN , we get by combining these two expressions,

the operator which projects a vector from EN on (SN , EN ) with zeros on

(J N

l − J ∗

l )SN =GSN ,SN −1(J N −1
l
(cid:16)
(u∗

+ PSN ,EN

l )EN

− J ∗

l )SN −1 + (uN −1
− D(cid:62)

l
D:,S∗ (D(cid:62)

− u∗

l )SN
:,S∗ D:,S∗ )−1(u∗

:,EN

l )S∗

(51)

(cid:17)

Taking the norm yields to the following inequality,

(cid:13)
(cid:13)J N

l − J ∗
l

(cid:13)
(cid:13) ≤ (cid:13)

(cid:13)GSN ,SN −1
(cid:13)
(cid:13)(u∗
(cid:13)

l )EN

+

− u∗
l

(cid:13)
(cid:13)uN −1
(cid:13)
l
:,S∗ D:,S∗ )−1(u∗

(cid:13)
(cid:13)
(cid:13)J N −1
(cid:13)
l
− D(cid:62)

(cid:13)
(cid:13) + (cid:13)
− J ∗
l
D:,S∗ (D(cid:62)
:,SN D:,SN −1, then (cid:13)

:,EN

(cid:13)GSN ,SN −1

(52)

(cid:13)
(cid:13)
(cid:13) .

l )S∗

(cid:13)
(cid:13) = (1 − µN

L ) and we

Denoting by µN the smallest eigenvalue of D(cid:62)
get that

(cid:13)
(cid:13)J N

l − J ∗
l

(cid:13)
(cid:13) ≤

K
(cid:89)

(1 −

k=1

µN −k
L

K−1
(cid:88)

k
(cid:89)

+

(1 −

k=0

i=1

(cid:13)
(cid:13)J N −K
l

)

− J ∗
l

(cid:13)
(cid:13)

µN −i
L

(cid:16) (cid:13)
)

(cid:13)uN −k
l

− u∗
l

(cid:13)
(cid:13) +

(cid:13)
(cid:13)(u∗
(cid:13)

l )EN −k

− D(cid:62)

:,EN −k

19

(53)

(cid:17)

(cid:13)
(cid:13)
(cid:13)

.

D†(cid:62)

:,S∗ (u∗

l )S∗

Published as a conference paper at ICLR 2022

(cid:13)
(cid:13)
(cid:13)PEN −k
K
(cid:89)

k=1

− u∗
l

l )EN −k

(cid:13)zN −k
l

(cid:13)uN −k
l

(cid:13)
(cid:13)(u∗
(cid:13)

L (cid:107)Dl(cid:107) (cid:13)

= 0. Therefore (cid:13)
The back-propagation is initialized as J N −K
(cid:13)J N −K
l
l
(cid:13)
over (cid:13)
(cid:13)
(cid:13)
(cid:13) ≤ 2
(cid:13)
(cid:13)(u∗
− z∗
(cid:13). Finally,
l )EN −k
l
be rewritten with projection matrices PEN −k
and P ¯S∗ to obtain
(cid:13)
(cid:13)
(cid:13)PEN −k
(cid:13)
(cid:13)
(cid:13)PEN −k
(cid:13)
(cid:13)
(cid:13)PEN −k
(cid:13)
(cid:13)
(cid:13). We have

Let BN −k =

l − D(cid:62)

:,S∗ (u∗

(cid:13)
(cid:13)
(cid:13) ≤

:,S∗ PS∗

− D(cid:62)

− D(cid:62)

− D(cid:62)

− D(cid:62)

D†(cid:62)

D†(cid:62)

D†(cid:62)

D†(cid:62)

D†(cid:62)

:,EN −k

:,EN −k

:,EN −k

:,EN −k

:,EN −k

l )S∗

u∗

≤

≤

:,S∗ PS∗

:,S∗ PS∗

l

(cid:13)
:,S∗ PS∗ u∗
(cid:13)
(cid:13)
(cid:13)
(cid:13) (cid:107)u∗
(cid:13)
l (cid:107)
(cid:13)
2
(cid:13)
(cid:13)
L

− J ∗
l

− D(cid:62)

:,EN −k

(cid:13)
(cid:13) = (cid:107)J ∗
D†(cid:62)
:,S∗ (u∗

l (cid:107). More-
(cid:13)
(cid:13)
(cid:13) can
l )S∗

(54)

(55)

(56)

(cid:107)Dl(cid:107) (cid:107)z∗

l (cid:107) .

(cid:13)
(cid:13)J N

l − J ∗
l

(cid:13)
(cid:13) ≤

(1−

µN −k
L

) (cid:107)J ∗

l (cid:107)+

2
L

(cid:107)Dl(cid:107)

K−1
(cid:88)

k
(cid:89)

(1−

k=0

i=1

µN −i
L

)

(cid:16) (cid:13)

(cid:13)zN −k
l

− z∗
l

(cid:13)
(cid:13)+BN −k (cid:107)z∗
l (cid:107)

(cid:17)

.

(57)
We now suppose that the support is reached at iteration N − s, with s ≥ K. Therefore, ∀n ∈
[N − s, N ] Sn = S∗. Let ∆n = F (zn, D) − F (z∗, D) + L
2 (cid:107)zn − z∗(cid:107). On the support, F is a
µ-strongly convex function and the convergence rate of (zN ) is

(cid:107)z∗ − zN (cid:107) ≤ (cid:0)1 −

µ
L

(cid:1)s 2∆N −s
L

Thus, we obtain

(cid:13)
(cid:13)J N

l − J ∗
l

(cid:13)
(cid:13) ≤

K
(cid:89)

k=1

(1 −

µN −k
L

) (cid:107)J ∗
l (cid:107)

+

2
L

(cid:107)Dl(cid:107)

K−1
(cid:88)

k
(cid:89)

(1 −

k=0

i=1

µN −i
L

(cid:16) (cid:13)
)

(cid:13)zN −k
l

− z∗
l

(cid:13)
(cid:13) + BN −k (cid:107)u∗
l (cid:107)

(cid:17)

≤

K
(cid:89)

k=1

(1 −

µN −k
L

) (cid:107)J ∗
l (cid:107)

(58)

(59)

(60)

+

+

2
L

2
L

(cid:107)Dl(cid:107)

s−1
(cid:88)

k=0

(1 −

µ
L

)k(cid:16) (cid:13)

(cid:13)zN −k
l

− z∗
l

(cid:13)
(cid:13)

(cid:17)

(cid:107)Dl(cid:107) (1 −

µ
L

)s

K−1
(cid:88)

k
(cid:89)

(1 −

k=s−1

i=s−1

µN −i
L

)

(cid:16) (cid:13)

(cid:13)zN −k
l

− z∗
l

(cid:13)
(cid:13) + BN −k (cid:107)(u∗

l )(cid:107)

(cid:17)

K
(cid:89)

≤

(1 −

µN −k
L

) (cid:107)J ∗
l (cid:107)

(61)

k=1

+

+

2
L

2
L

(cid:107)Dl(cid:107)

s−1
(cid:88)

k=0

(1 −

)k(cid:0)1 −

µ
L

µ
L

(cid:1)s−1−k 2∆N −s

L

(cid:107)Dl(cid:107) (1 −

µ
L

)s

K−1
(cid:88)

k
(cid:89)

(1 −

k=s−1

i=s−1

µN −i
L

)

(cid:16) (cid:13)

(cid:13)zN −k
l

− z∗
l

(cid:13)
(cid:13) + BN −k (cid:107)(u∗

l )(cid:107)

(cid:17)

K
(cid:89)

≤

(1 −

k=1

µN −k
L

) (cid:107)J ∗
l (cid:107)

(62)

+ (cid:107)Dl(cid:107) (1 −

µ
L

)s−1s

+

2
L

(cid:107)Dl(cid:107) (1 −

µ
L

)s

4∆N −s
L2
K−1
(cid:88)

k
(cid:89)

(1 −

k=s−1

i=s−1

20

µN −i
L

)

(cid:16) (cid:13)

(cid:13)zN −k
l

− z∗
l

(cid:13)
(cid:13) + BN −k (cid:107)(u∗

l )(cid:107)

(cid:17)

(63)

Published as a conference paper at ICLR 2022

Corollary 2.6 Let µ > 0 be the smallest eigenvalue of D(cid:62)
propagation depth and let ∆N = F (zN , D) − F (z∗, D) + L
[N − K, N ]; Sn ⊂ S∗. Then, we have

:,S∗ D:,S∗ . Let K ≤ N be the back-
2 (cid:107)zN − z∗(cid:107). Suppose that ∀n ∈

(cid:13)
(cid:13)J ∗

l − J N
l

(cid:13)
(cid:13) ≤

(cid:16)

1 −

(cid:17)K

µ
L

(cid:107)J ∗

l (cid:107) + K

(cid:17)K−1

(cid:16)

1 −

µ
L

(cid:107)Dl(cid:107)

4∆N −K
L2

.

L (cid:107)Dl(cid:107) (1 − µ

The term 2
vanishes
when the algorithm is initialized on the support. Otherwise, it goes to 0 as s, K → N and N → ∞
because ∀n > N − s, µn = µ < 1.

i=s−1(1 − µN −i
L )

(cid:13)zN −k
l

L )s (cid:80)K−1

(cid:13)
(cid:13) + BN −k (cid:107)(u∗

− z∗
l

k=s−1

l )(cid:107)

(cid:81)k

(cid:16) (cid:13)

(cid:17)

D ITERATIVE ALGORITHMS FOR SPARSE CODING RESOLUTION.

ISTA. Algorithm to solve minz

Algorithm 1 ISTA

1

2 (cid:107)y − Dz(cid:107)2

2 + λ (cid:107)z(cid:107)1

y, D, λ, N
z0 = 0, n = 0
Compute the Lipschitz constant L of D(cid:62)D
while n < N do

un+1 ← zN − 1
zn+1 ← ST λ
n ← n + 1

L

(un+1)

L D(cid:62)(Dzn − y)

end while

FISTA. Algorithm to solve minz

Algorithm 2 FISTA

1

2 (cid:107)y − Dz(cid:107)2

2 + λ (cid:107)z(cid:107)1

y, D, λ, N
z0 = x0 = 0, n = 0, t0 = 1
Compute the Lipschitz constant L of D(cid:62)D
while n < N do

un+1 ← zn − 1
L D(cid:62)(Dzn − y)
xn+1 ← ST λ
(un+1)
√
L
1+4t2
n
2

tn+1 ←
zn+1 ← xn+1 + tn−1
tn+1
n ← n + 1

1+

end while

(xn+1 − xn)

21

