On Optimal Interpolation in Linear Regression

Eduard Oravkin
Department of Statistics
University of Oxford
eduard.oravkin@stats.ox.ac.uk

Patrick Rebeschini
Department of Statistics
University of Oxford
patrick.rebeschini@stats.ox.ac.uk

Abstract

Understanding when and why interpolating methods generalize well has recently
been a topic of interest in statistical learning theory. However, systematically
connecting interpolating methods to achievable notions of optimality has only
received partial attention. In this paper, we investigate the question of what is the
optimal way to interpolate in linear regression using functions that are linear in the
response variable (as the case for the Bayes optimal estimator in ridge regression)
and depend on the data, the population covariance of the data, the signal-to-noise
ratio and the covariance of the prior for the signal, but do not depend on the value
of the signal itself nor the noise vector in the training data. We provide a closed-
form expression for the interpolator that achieves this notion of optimality and
show that it can be derived as the limit of preconditioned gradient descent with a
speciﬁc initialization. We identify a regime where the minimum-norm interpolator
provably generalizes arbitrarily worse than the optimal response-linear achievable
interpolator that we introduce, and validate with numerical experiments that the
notion of optimality we consider can be achieved by interpolating methods that only
use the training data as input in the case of an isotropic prior. Finally, we extend the
notion of optimal response-linear interpolation to random features regression under
a linear data-generating model that has been previously studied in the literature.

1

Introduction

Establishing mathematical understanding for the good generalization properties of interpolating
methods, i.e. methods that ﬁt the training data perfectly, has attracted signiﬁcant interest in recent
years. Motivated by the quest to explain the generalization performance of neural networks which
have zero training error, for example even on randomly corrupted data (Zhang et al., 2017), this area
of research has established results for a variety of models. For instance, in kernel regression, Liang
and Rakhlin (2020) provide a data-dependent upper bound on the generalization performance of the
minimum-norm interpolator. By analyzing the upper bound, they show that small generalization
error of the minimum-norm interpolator occurs in a regime with favourable curvature of the kernel,
particular decay of the eigenvalues of the kernel and data population covariance matrices and,
importantly, in an overparametrized setting. In random features regression, Mei and Montanari
(2019) showed that for large signal-to-noise ratio and in the limit of large overparametrization, the
optimal regularization is zero, i.e. the optimal ridge regressor is an interpolator. Liang and Sur (2020)
characterized the precise high-dimensional asymptotic generalization of interpolating minimum-
(cid:96)1-norm classiﬁers and boosting algorithms which maximize the (cid:96)1 margin. Bartlett et al. (2020)
isolated a setting of benign overﬁtting in linear regression, dependent on notions of effective rank of
the population covariance matrix, in which the minimum-norm interpolator has small generalization
error. Similarly, this regime of benign overﬁtting occurs with large overparametrization.

Linear models, in particular, provide a fundamental playground to understand interpolators. On
the one hand, in overparametrized regimes, interpolators in linear models are seen to reproduce

35th Conference on Neural Information Processing Systems (NeurIPS 2021).

stylized phenomena observed in more general models. For example, the double descent phenomenon,
which was ﬁrst empirically observed in neural networks (Belkin et al., 2019), has also featured
in linear regression (Hastie et al., 2019). On the other hand, neural networks are known to be
well-approximated by linear models in some regimes. For example, with speciﬁc initialization and
sufﬁcient overparametrization, two-layer neural networks trained with gradient descent methods
are well-approximated by a ﬁrst-order Taylor expansion around their initialization (Chizat et al.,
2019). This linear approximation can be split into a random features component and a neural-tangent
component. The random features model, a two-layer neural network with randomly initialized
ﬁrst layer which is ﬁxed during training, shares similar generalization behavior with the full neural
network (Bartlett et al., 2021), and as such, the random features model provides a natural stepping
stone towards tackling a theoretical understanding of neural networks.

A major focus of the interpolation literature has so far been to theoretically study if and when
interpolating methods based on classical techniques such as ridge regression and gradient descent
can have optimal or near-optimal generalization (Bartlett et al., 2021). However the question of
understanding which interpolators are best, and designing data-dependent schemes to implement
them, seems to have received only partial attention. Work investigating which interpolators are
optimal in linear regression includes Muthukumar et al. (2019), where the authors constructed the
best-possible interpolator, i.e. a theoretical device which uses knowledge of the true parameter
and training noise vector to establish a fundamental limit on how well any interpolator in linear
regression can generalize. When the whitened features are sub-Gaussian, this fundamental limit is
lower bounded by a term proportional to n/d, up to an additive constant and with high probability,
which is small only in the regime of large overparametrization. Here, n and d are the size and the
dimension of the data. While this interpolator provides the best-possible generalization error, the
interpolator is not implementable in general, as one would need access to the realization of the true
data-generating parameter w(cid:63) and the realization of the noise in the training data. Rangamani et al.
(2020) studied generalization of interpolators in linear regression and showed that the minimum-norm
interpolator minimizes an upper bound on the generalization error related to stability. In (Mourtada,
2020), it was shown that the minimum-norm interpolator is minimax optimal over any choice of
the true parameter w(cid:63) ∈ Rd, distributions of the noise with mean 0 and bounded variance, and for
a ﬁxed nondegenerate distribution of the features. Amari et al. (2021) computed the asymptotic
risk of interpolating preconditioned gradient descent in linear regression and investigated the role
of its implicit bias on generalization. In particular, they identiﬁed the preconditioning which leads
to optimal asymptotic (as d/n → γ > 1 with n, d → ∞) bias and variance, separately, among
interpolators of the form w = P X T (XP X T )−1y for some matrix P , where X ∈ Rn×d is the
data matrix, y ∈ Rn is the response vector. They showed that, within this class of interpolators,
using the inverse of the population covariance matrix of the data as preconditioning achieves optimal
asymptotic variance. However, the interpolator with optimal risk is not given.

In this paper, we study the question of what is the optimal way to interpolate in overparametrized
linear regression by procedures that do not use the realization of the true parameter generating the data,
nor the realization of the training noise. The motivation for studying this question is twofold. First, in
designing new ways to interpolate that are directly related to notions of optimality in linear models, we
hope to provide a stepping stone to designing new ways to interpolate in more complex models, such
as neural networks. Second, our results illustrate that there can be arbitrarily large differences in the
generalization performance of interpolators, in particular considering the minimum-norm interpolator
as a benchmark. This is a phenomenon that does not seem to have received close attention in the
literature and may spark new interest in designing interpolators connected to optimality.

We consider the family of interpolators that can be achieved as an arbitrary function f of the data,
population covariance, signal-to-noise ratio and the prior covariance such that f is linear in the
response variable y (as the case for the Bayes optimal estimator in ridge regression). We call such
interpolators response-linear achievable (see Deﬁnition 3). We also introduce a natural notion of
optimality that assumes that the realization of true data-generating parameter and the realization of
the noise in the training data are unknown. Within this class of interpolators and under this notion
of optimality, we theoretically compute the optimal interpolator and show that this interpolator is
achieved as the implicit bias of a preconditioned gradient descent with proper initialization. We refer
to this interpolator as the optimal response-linear achievable interpolator.

Could it be that the commonly used minimum-norm interpolator is good enough so that the beneﬁt of
ﬁnding a better interpolator is negligible? We illustrate that the answer to this question is, in general,

2

no. In particular, we construct an example in linear regression where the minimum-norm interpolator
has arbitrarily worse generalization than the optimal response-linear achievable interpolator. Here,
the variance (hence also generalization error) of the minimum-norm interpolator diverges to inﬁnity
as a function of the eigenvalues of the population covariance matrix, while the generalization error of
the optimal response-linear achievable interpolator stays bounded, close to the optimal interpolator,
i.e. the theoretical device of Muthukumar et al. (2019) which uses the value of the signal and noise.

The optimal response-linear achievable interpolator uses knowledge of the population covariance
matrix of the data (similarly as in Amari et al. (2021)), the signal-to-noise ratio, and the covariance of
the true parameter (on which we place a prior distribution). Is it the case that the better performance
of our interpolator is simply a consequence of this population knowledge? We provide numerical
evidence that shows that the answer to this question is, in general, no. In particular, we construct
an algorithm to approximate the optimal response-linear achievable interpolator which does not
require any prior knowledge of the population covariance or the signal-to-noise ratio and uses only
the training data X and y, and we empirically observe that this new interpolator generalizes in a
nearly identical way to the optimal response-linear achievable interpolator.

Finally, we show that the concept of optimal response-linear achievable interpolation can be extended
to more complex models by providing analogous results for a random features model under the same
linear data-generating regime as also considered in (Mei and Montanari, 2019), for instance.

2 Problem setup

In this paper we investigate overparametrized linear regression. We assume there exists w(cid:63) ∈ Rd
(unknown) so that yi = (cid:104)w(cid:63), xi(cid:105) + ξi for i ∈ {1, . . . , n}, with i.i.d. noise ξi ∈ R (unknown) such that
i ) = σ2 and i.i.d. features xi ∈ Rd that follow a distribution Px with mean E(xi) = 0
E(ξi) = 0, E(ξ2
and covariance matrix E(xixT
i ) = Σ. We store the features in a random matrix X ∈ Rn×d with
rows xi ∈ Rd, the response variable in a random vector y ∈ Rn with entries yi ∈ R, and the noise
in a random vector ξ ∈ Rn with entries ξi ∈ R. Throughout the paper we assume that d ≥ n. We
consider the whitened data matrix Z = XΣ− 1
i ) = Id, where
Id ∈ Rd×d is the identity matrix. We place a prior on the true parameter in the form w(cid:63) ∼ Pw(cid:63)
such that E(w(cid:63)) = 0 and E(w(cid:63)w(cid:63)T ) = r2
d Φ. Here, Φ is a positive deﬁnite matrix and r2 is the
signal. We sometimes abuse terminology and refer to Φ as the covariance of the prior even though
r2
d Φ is the covariance matrix. Our results will be proved in general, but for the sake of exposition
it can be assumed that xi ∼ N (0, Σ), ξ ∼ N (0, σ2In) and w(cid:63)∼N (0, r2
d Φ). We also deﬁne the
signal-to-noise ratio δ = r2/σ2 and consider the squared error loss (cid:96) : (x, y) ∈ R2 (cid:55)→ (x − y)2.
Througout the paper, we assume the following two technical conditions hold.

2 ∈ Rn×d, whose rows satisfy E(zizT

Assumption 1. Px(xi ∈ V ) = 0 for any linear subspace V of Rd with dimension smaller than d.
Assumption 2. For all Lebesgue measurable sets A ⊆ Rd, ν(A) > 0 implies Pw(cid:63) (w(cid:63) ∈ A) > 0,
where ν is the standard Lebesgue measure on Rd.

Assumption 1 is needed only so that rank(X) = n with probability 1 (for a proof see A.4). A
sufﬁcient condition is that Px has a density on Rd. A sufﬁcient condition for Assumption 2 is that
Pw(cid:63) has a positive density on Rd. Now, our goal is to minimize the population risk
r(w) = E

(cid:0)((cid:104)w, (cid:101)x(cid:105) − (cid:101)y)2(cid:1),

(cid:101)x,(cid:101)ξ

or, equivalently, the excess risk r(w) − r(w(cid:63)). Here, ((cid:101)x, (cid:101)y, (cid:101)ξ) is a random variable which follows the
distribution of (x1, y1, ξ1), . . . , (xn, yn, ξn) and is independent from them. Throughout the paper we
write Ezg(z, ˜z) to denote the conditional expectation E(g(z, ˜z)|˜z), for two random variables z and ˜z
and for a function g. The population risk satisﬁes

r(w) = (w − w(cid:63))T Σ(w − w(cid:63)) + σ2 = (cid:107)w − w(cid:63)(cid:107)2

Σ + r(w(cid:63)),

where (cid:107)w(cid:107)2

Σ = wT Σw. We deﬁne the bias and variance of w ∈ Rd by the decomposition
Eξ,w(cid:63) r(w) = B(w) + V (w),

(1)

(2)

3

where

B(w) = Eξ,w(cid:63) (cid:107)E(w|w(cid:63), X) − w(cid:63)(cid:107)2
Σ

V (w) = Eξ,w(cid:63) (cid:107)w − E(w|w(cid:63), X)(cid:107)2
Σ.

(3)

One of the main paradigms to minimize the (unknown) population risk is based on minimizing
i=1((cid:104)w, xi(cid:105) − yi)2 = 1
the empirical risk R(w) = 1
i=1 (cid:96)(wT xi, yi) (Vapnik, 1995). In our
n
n
setting, minimizing the empirical risk is equivalent to ﬁnding w ∈ Rd such that Xw = y.

(cid:80)n

(cid:80)n

3

Interpolators

An interpolator is any minimizer of the empirical risk. Let G be the set of interpolators, which in
linear regression can be written as

G = {w ∈ Rd : Xw = y}.

As rank(X) = n with probability 1, we have G (cid:54)= ∅ with probability 1. In linear regression, the
implicit bias of gradient descent initialized at 0 is the minimum-norm interpolator (Gunasekar et al.,
2018). We deﬁne the minimum-norm interpolator by

w(cid:96)2 = arg min

w∈Rd : Xw=y

(cid:107)w(cid:107)2

2 = X †y,

where X † ∈ Rn×d is the Moore-Penrose pseudoinverse (Penrose, 1955). As rank(X) = n, we
can also write X † = X T (XX T )−1. The second interpolator of interest is a purely theoretical
device, previously used in (Muthukumar et al., 2019) to specify a fundamental limit to how well any
interpolator in linear regression can generalize.

Deﬁnition 1. The best possible interpolator is deﬁned as

Wb = arg min

w∈G

r(w).

We can write

Wb = arg min

w∈Rd : Xw=y

(cid:107)Σ

1

2 (w − w(cid:63))(cid:107)2
2,

and after a linear transformation and an application of a result on approximate solutions to linear
equations (Penrose, 1956), we obtain

Wb = w(cid:63) + Σ− 1

2 (XΣ− 1

2 )†ξ.

(4)

We notice that the best possible interpolator ﬁts the signal perfectly by having access to the true
parameter w(cid:63) and ﬁts the noise through the term Σ− 1
2 )†ξ by having access to the noise vector
ξ in the training data. In general, this interpolator cannot be implemented as it requires access to
the unknown quantities w(cid:63) and ξ. We are interested in interpolators which can be achieved by some
algorithm using the data X and y.

2 (XΣ− 1

Deﬁnition 2. We deﬁne an estimator w ∈ Rd to be achievable if there exists a function f such that
w = f (X, y, Σ, Φ, δ).

In our deﬁnition of achievability, we allow for knowledge of the population data covariance, the
signal-to-noise ratio, and the prior covariance to deﬁne a fundamental limit to what generalization
performance can be achieved also without access to these quantities, and we later empirically show
that we can successfully approach this limit using only the knowledge of the training data X and y,
in considered examples (see Section 5.1). Moreover, our theory is also useful in situations when one
has access to some prior information about the regression problem which they can incorporate into
an estimate of Σ, δ, Φ (for example, one may know the components of xi are independent and hence
Σ is diagonal) and hence it is relevant to consider a broader class than w = f (X, y).

Deﬁnition 3. We deﬁne the set of response-linear achievable estimators by

L = { w ∈ Rd : ∃f such that w = f (X, y, Σ, Φ, δ) where y ∈ Rn (cid:55)→ f (X, y, Σ, Φ, δ) is linear }.

4

Linearity of y ∈ Rn (cid:55)→ f (X, y, Σ, Φ, δ) is equivalent to f (X, y, Σ, Φ, δ) = g(X, Σ, Φ, δ)y, where g
is any function which has image in Rd×n. The notion of optimality that we introduce is that of the
optimal response-linear achievable interpolator, which is the interpolator that minimizes the expected
risk in the class L.

Deﬁnition 4. We deﬁne the optimal response-linear achievable interpolator by

wO = arg min
w∈G∩L

Eξ,w(cid:63) r(w) − r(w(cid:63)).

(5)

4 Main results

By deﬁnition, the interpolator wO has the smallest expected risk among all response-linear achievable
interpolators. Our ﬁrst contribution is the calculation of its exact form.

Proposition 1. The optimal response-linear achievable interpolator satisﬁes

wO =

(cid:18) δ
d

ΦX T +Σ− 1

2 (XΣ− 1

2 )†

(cid:19)(cid:18)

In +

δ
d

XΦX T

(cid:19)−1

y.

(6)

For an isotropic prior Φ = Id, wO depends only on the population covariance Σ and the signal-to-
noise ratio δ so that wO can be approximated using estimators of these quantities, which is what
we do in Sections 5 and A.8. Even if Φ (cid:54)= Id, one might have some information about the prior
covariance, which can be incorporated into an estimate (cid:98)Φ and used instead of Φ. However, even if no
such estimate is available, in Section A.9 we empirically show that, in our examples, using (cid:98)Φ = Id
when Φ (cid:54)= Id has a small effect on generalization.

Secondly, using results of Gunasekar et al. (2018) on the implicit bias of converging mirror descent,
we show that the optimal response-linear interpolator is the limit of gradient descent preconditioned
by the inverse of the population covariance, provided that it converges and is suitably initialized.

Proposition 2. The optimal response-linear achievable interpolator is the limit of preconditioned
gradient descent

provided that the algorithm converges, initialized at

wt+1 = wt − ηtΣ−1∇R(wt),

w0 =

(cid:18)

δ
d

ΦX T

In +

δ
d

XΦX T

(cid:19)−1

y.

(7)

(8)

The interpolator wO does not have the smallest bias or the smallest variance in the bias-variance
decomposition Eξ,w(cid:63) r(w) = B(w) + V (w), but rather achieves a balance. This is related to
the results of (Amari et al., 2021). Their setting looks at interpolators achieved as the limit of
preconditioned gradient descent in linear regression (preconditioned with some matrix P ) and
initialized at 0. Such interpolators can be written as w = P X T (XP X T )−1y. For these interpolators,
they compute the risk of w, separate the risk into a variance and a bias term and using random matrix
theory they ﬁnd what the variance and bias terms converge to when d → ∞, n → ∞ in a way
such that d/n → γ > 1. For these calculations to hold, they assume that the spectral distribution
of (Σd)d∈N converges weakly to a distribution supported on [c, C] for some c, C > 0. Then, after
obtaining the limiting variance and bias, they prove which matrices P minimize these limits separately
(not their sum, which is the overall asymptotic risk).

We approach the problem from the other direction. That is, we do not a priori consider interpolators
that can be achieved as limits of speciﬁc algorithms, but we directly look at which interpolator
minimizes the risk as a whole (not bias and variance separately). Only after computing the optimal
response-linear interpolator, we show in Proposition 2 that the interpolator is in fact the limit of
preconditioned gradient descent, however with a speciﬁc initialization. Our results hold for every
ﬁnite d ≥ n and we do not put assumptions on the eigenvalues or the spectral distribution of Σ.

In particular, we can recover the results of (Amari et al., 2021) as a special case of Proposition 1. If
we take the signal-to-noise ratio δ → 0 (by taking r2 → 0) in Proposition 1, we obtain the matrix P

5

which achieves optimal variance and if we take δ → ∞ (by taking σ2 → 0), we obtain the matrix P
which achieves optimal bias. Moreover, we provide a further extension in Proposition 3.
We show that the preconditioned gradient descent wt+1 = wt − ηtΣ−1∇R(wt) achieves optimal
variance among all interpolators when initialized at any deterministic w0 and for any ﬁnite d, n ∈ N.
Proposition 3. The limit of preconditioned gradient descent wt+1 = wt − ηtΣ−1∇R(wt) initialized
at a deterministic w0 ∈ Rd, provided that it converges, satisﬁes
V (w).

wt = arg min

(9)

lim
t→∞

w∈G

We note that the optimal variance is achieved among all interpolators, not only among response-linear
achievable interpolators.

A natural question to ask is whether the optimal response-linear achievable interpolator wO provides a
signiﬁcant beneﬁt compared to other interpolators. A second question is whether we can successfully
approximate the optimal response-linear achievable interpolator without knowledge of the population
covariance Σ and the signal-to-noise ratio δ. In the following section, we illustrate that both the
interpolator with optimal variance and the interpolator with optimal bias can generalize arbitrarily
badly in comparison to wO as a function of the eigenvalues of the population covariance. In the same
regimes where this happens, we present numerical evidence that we can successfully approximate wO
by an empirical interpolator wOe without any prior knowledge of Σ or δ by using the Graphical Lasso
estimator (Friedman et al., 2007) of the covariance matrix Σ and choosing the empirical estimate of δ
by crossvalidation on a subset of the data.

5 Comparison of interpolators

First, we present an example where the minimum-norm interpolator w(cid:96)2 generalizes arbitrarily worse
than the best response-linear achievable interpolator wO. Second, we give an example where an
interpolator with optimal variance generalizes arbitrarily worse than wO. This shows that arbitrarily
large differences in test error are possible within the class of estimators with zero training error.
In the examples, we consider a setting where xi ∼ N (0, Σ) and w(cid:63) ∼ N (0, r2
d Φ). Therefore,
throughout Section 5 we assume Px = N (0, Σ) and Pw(cid:63) = N (0, r2
d Φ). Before presenting these
examples, we discuss approximating wO by an interpolator, wOe, which uses only the data X and y.

5.1 Empirical approximation

The interpolator wO is the limit of the algorithm

wt+1 = wt − ηtΣ−1∇R(wt).
(10)
The population covariance Σ is required to run this algorithm. However, the matrix Σ is usually
unknown in practice. One may want to estimate Σ. However, if one replaces Σ by (cid:101)Σ = X T X/n+λId
(with λ ≥ 0), then the limit of (10) is the same as the limit of gradient descent (provided that both
algorithms converge). This is because, using the singular value decomposition of X, one can show

(cid:101)Σ− 1

2 (X (cid:101)Σ− 1

2 )†y = X †y.

The preconditioned gradient update wt+1 − wt = ηtP −1∇R(wt) has to not belong to Im(X T ) in
order to not converge to the minimum-norm interpolator. Hence, using P = (cid:101)Σ = X T X/n + λId (for
example, also the Ledoit-Wolf shrinkage approximation (Ledoit and Wolf, 2004)) in preconditioned
gradient descent removes the beneﬁt of preconditioning in terms of generalization of the limit.

We use the Graphical Lasso approximation (Friedman et al., 2007). We empirically observe that in
the examples considered in this paper (Figures 1, 2, 3, 4, 5, 6) using the Graphical Lasso covariance
Σe instead of Σ has nearly no effect on generalization. Under speciﬁc assumptions, Ravikumar et al.
(2011) provide some convergence guarantees of the Graphical Lasso.

In regards to approximating the signal-to-noise ratio δ, we choose δe that minimizes the crossvalidated
error on random subsets of the data. In this way, we arrive at the interpolator
(cid:19)−1

(cid:19)(cid:18)

wOe =

X T +Σe

− 1

2 (XΣe

− 1

2 )†

In +

XX T

y,

(11)

(cid:18) δe
d

δe
d

6

which approximates wO and is a function of only X and y. We note that the interpolator wOe uses Id
in place of the prior covariance matrix.

In the experiments (Figures 1, 2, 3, 4, 5, 6) we used the Graphical Lasso implementation of scikit-
learn (Pedregosa et al., 2011) with parameter α = 0.25 (α can also be crossvalidated for even
better performance) and in estimating δ, for each δe in {0.1, 0.2, . . . , 1, 2, . . . , 10}, we computed
the validation error on a random, unseen tenth of the data and averaged over 10 times. The δe with
smallest crossvalidated error was chosen.

5.2 Random matrix theory concepts

For presenting the discussed examples we need to review some concepts from random matrix theory.

Deﬁnition 5. For a symmetric matrix Σ ∈ Rd×d with eigenvalues λ1 ≥ λ2 ≥ · · · ≥ λd ≥ 0 we
deﬁne its spectral distribution by FΣ(x) = 1
d

1[λi,∞)(x).

(cid:80)d

i=1

The following assumptions will be occasionally considered for the covariance matrix Σ.

Assumption 3. There exists kmax > 0 such that λmax(Σ) ≤ kmax uniformly for d ∈ N.
Assumption 4. There exists kmin > 0 such that kmin ≤ λmin(Σ) uniformly for d ∈ N.
Assumption 5. The spectral distribution FΣ of the covariance matrix Σ converges weakly to a
distribution H supported on [0, ∞).

−→ (cid:101)Fγ,
Marˇcenko and Pastur (1967) showed that there exists a distribution (cid:101)Fγ such that F ZT Z
weakly, with probability 1 as n → ∞, d → ∞ with d/n → γ. In our discussion, xi = Σ 1
2 zi, where
zi∼N (0, Id) independently. Then, under Assumption 5, it can be shown that the spectral distribution
of (cid:98)Σ = X T X/n = Σ 1
2 /n converges weakly, with probability 1 to a distribution supported
on [0, ∞), which we denote by Fγ, see e.g. (Silverstein and Choi, 1995). Similar arguments also
show that the spectral distribution of XX T /n ∈ Rn×n converges weakly, with probability 1.

2 Z T ZΣ 1

n

Deﬁnition 6. For a distribution F supported on [0, ∞), we deﬁne the Stieltjes transform of F, for
any z ∈ C \ R+ by

mF (z) =

(cid:90) ∞

0

1
λ − z

dF(λ).

(cid:98)Σ(z) → m(z) and
The weak convergence of the spectral distribution of (cid:98)Σ to Fγ is equivalent to m
mXX T /n(z) → v(z) almost surely for all z ∈ C \ R+, where m and v are the Stieltjes transforms
of Fγ and the limiting spectral distribution of XX T /n, respectively (see e.g. Proposition 2.2 of
(Hachem et al., 2007)). We call v the companion Stieltjes transform of Fγ.

5.3 Diverging variance of interpolator with optimal bias

Using w0 = 0 in Proposition 3, we choose the interpolator with optimal variance to be (see A.3)

wV = Σ− 1

2 (XΣ− 1

2 )†y.

(12)

When Φ = Id, the interpolator with best bias among response-linear achievable interpolators is the
minimum-norm interpolator (see Section A.5). We identify an example where the minimum-norm
interpolator w(cid:96)2 generalizes arbitrarily worse than the best response-linear achievable interpolator
wO. For this, we exploit results of Hastie et al. (2019) on computing the asymptotic risk of the
minimum-norm interpolator. They show that if Φ = Id, under Assumptions 3, 4, 5 and if d
n → γ > 1
with n → ∞, d → ∞ then with probability 1,

Eξ,w(cid:63) r(w(cid:96)2) − r(w(cid:63)) −→

r2
γv(0)

+ σ2

(cid:18) v(cid:48)(0)

(cid:19)
v(0)2 − 1

,

(13)

where v is the companion Stieltjes transform introduced in Section 5.2. In comparison, similarly
as in (Amari et al., 2021), the asymptotic risk of the best variance estimator wV satisﬁes that under

7

Assumption 3 and 5, if n, d → ∞ with d

n → γ > 1 then with probability 1 we have that

lim
d→∞

Eξ,w∗ r(wV ) − r(w(cid:63)) = r2 γ − 1

γ

(cid:90) ∞

0

s dH(s) +

σ2
γ − 1

.

(14)

An alternative way is to write (cid:82) ∞
0 s dH(s) = limd→∞ Tr(Σ). This result follows by an application
of Theorem 1 of (Rubio and Mestre, 2011), which is in the supplementary material for completeness.

Now we ﬁnd a regime of covariance matrices Σ, for which the variance term of the minimum-
norm solution, V(cid:96)2 = σ2( v(cid:48)(0)
v(0)2 − 1), diverges to inﬁnity, while the risk of wO stays bounded and
close to optimal. For this, we consider a generalization of the spike model of covariance matrices
(Baik and Silverstein, 2006; Johnstone, 2001), which is a fundamental model in statistics. Here
Σ = diag(ρ1, . . . , ρ1, ρ2, . . . , ρ2) ∈ Rd×d, where the number of ρ1s is d · ψ1 with ψ1 ∈ [0, 1]. This
model was also considered in (Richards et al., 2021) where it is called the strong weak features model.
In this regime, it is possible to explicitly calculate the companion Stieltjes transform v(0) and v(cid:48)(0)
of (13). In the case that γ = 2, ψ1 = 1/2 we have

V(cid:96)2 = σ2

(cid:18) v(cid:48)(0)

(cid:19)
v(0)2 − 1

=

σ2
2

(cid:18)(cid:114) ρ1
ρ2

+

(cid:114) ρ2
ρ1

(cid:19)

+ 2

.

(15)

If we ﬁx ρ1 = 1 and take ρ2 → 0, then the variance term V(cid:96)2 diverges to inﬁnity. This also means that
the asymptotic risk of the minimum-norm interpolator diverges to inﬁnity. Moreover, the asymptotic
risk of wV in (14) evaluates to

lim
d→∞

Eξ,w∗ r(wV ) − r(w(cid:63)) =

(cid:18)

(cid:19)(cid:18)

ψ1ρ1 + (1 − ψ1)ρ2

1 −

(cid:19)

1
γ

+

σ2
γ − 1

.

(16)

In addition, by construction of wO, we know that Eξ,w(cid:63) r(wO) ≤ Eξ,w(cid:63) r(wV ) and therefore the
asymptotic limit of Eξ,w(cid:63) r(wO) − r(w(cid:63)), as d/n → γ > 1, stays bounded by (16) as ρ2 → 0. The
expected generalization error in the setting described above is illustrated in Figure 1.

Figure 1: Plot of Eξr(w) (points) for w ∈
{w(cid:96)2, wV , wO, wOe, wb} along with predictions
(crosses) from (14) and (13) in the strong weak
features model with r2 = 1, σ2 = 1, γ = 2, ψ1 =
1/2, n = 3000 and ρ1 = 1, ρ2 → 0.

Figure 2: Plot of Eξr(w) (points) for w ∈
{w(cid:96)2, wV , wO, wOe, wb} along with predictions
(crosses) from (14) and (13) in the strong weak
features model with r2 = 1, σ2 = 1, γ = 2, ψ1 =
1/2, n = 3000 and ρ2 = 1, ρ1 → ∞.

We note that the empirical estimator wOe (yellow points), which is a function of only the training
data X and y and does not use the population covariance Σ or the signal-to-noise ratio δ, performs
almost identically to the optimal response-linear achievable interpolator wO (cyan points).

In this example, we chose γ = 2 and ψ1 = 1/2 deliberately. One does not achieve diverging variance
for an arbitrary choice of γ and ψ1. However, for any γ > 1 such that γψ1 = 1, the phenomenon of
Figure 1 holds (see A.7 of the supplementary material).

8

204060801001/223456RiskComparison of interpolatorsw2wVwOwOewb204060801001/2510152025RiskComparison of interpolatorsw2wVwOwOewb5.4 Diverging bias of interpolator with optimal variance

Now, we illustrate a regime where the best variance interpolator wV generalizes arbitrarily worse
than wO. In the same strong and weak features covariance model described above in Section 5.3,
when γ = 2 and ψ1 = 1/2, if we instead have ρ1 → ∞ and ρ2 = 1, then the asymptotic risk (16)
diverges to inﬁnity linearly. However, the variance of the minimum-norm interpolator in (15) diverges
only like

√

ρ1. Moreover, the bias term satisﬁes
r2
γv(0)

B(cid:96)2 =

=

√

r2
γ

ρ1ρ2,

which also diverges like

ρ1. Now, because Eξ,w(cid:63) r(wO) ≤ Eξ,w(cid:63) r(w(cid:96)2), we have that

√

σ2
2
√
so that the asymptotic risk of wO diverges to inﬁnity as

Eξ,w(cid:63) r(wO) ≤

lim
d→∞

ρ1ρ2 +

r2
γ

√

(cid:18)(cid:114) ρ1
ρ2

+

(cid:114) ρ2
ρ1

(cid:19)

+ 3

,

ρ1. We illustrate this in Figure 2.

We notice that the empirical approximation wOe again performs in a nearly identical way to the
optimal response-linear achievable interpolator wO. Moreover, importantly, we note that wV and
wO are limits of the same algorithm, wt+1 = wt − ηtΣ−1∇R(wt), only with different initialization.
Hence, this shows that different initialization of the same optimization algorithm can have an
arbitrarily large inﬂuence on generalization through implicit bias.

6 Random features regression

The concept of optimal interpolation as a function which is linear in the response variable, is general
and can be extended beyond linear models. We present an extension of Proposition 1 to the setting of
random features regression. Random features models were introduced as a random approximation to
kernel methods (Rahimi and Recht, 2008) and can be viewed as a two-layer neural network with ﬁrst
layer randomly initialized and ﬁxed as far as training is concerned. They can be shown to approximate
neural networks in certain regimes of training and initialization and hence are often considered in
the literature as a ﬁrst step to address neural networks (e.g. (Jacot et al., 2018)). We consider data
generated in the same way as before, yi = (cid:104)xi, w(cid:63)(cid:105) + ξi, and the model to be a two-layer neural
d), where the ﬁrst layer Θ ∈ RN ×d is randomly initialized. This
network fa : Rd (cid:51) x (cid:55)→ aT σ(Θx/
setting, along with xi and rows of Θ belonging to the sphere Sd−1(
d in Rd, is often
considered in the literature on interpolation of random features models (Mei and Montanari, 2019;
Ghorbani et al., 2021). If we analogously deﬁne the optimal response-linear achievable interpolator
in random features regression by

d) with radius

√

√

√

aO = arg min
a∈G∩L
where here G = {a ∈ RN : Za = y} is the set of interpolators, Z = σ(XΘT /
same as in Deﬁnition 3, then the following analogue of Proposition 1 holds.

Eξ,w(cid:63) r(fa) − r(w(cid:63)),

√

(17)

d) and L is the

Proposition 4. The optimal response-linear achievable interpolator (17) in random features regression
satisﬁes

(cid:18)

aO = Σ−1

z

ΣzxΦX T +Z T (cid:0)ZΣ−1

z Z T (cid:1)−1(cid:0) d
δ

In +XΦX T −ZΣ−1

z ΣzxΦX T (cid:1)

(cid:19)(cid:18) d
δ

In +XΦX T

(cid:19)−1
y,

√

√

√

Here Σz = E˜x(σ(Θ˜x/
d)T ) and Σzx = E˜x(σ(Θ˜x/
d)˜xT ) are covariance and cross-
covariance matrices, respectively. This interpolator can be again obtained as the implicit bias of
preconditioned gradient descent using results of Gunasekar et al. (2018).

d)σ(Θ˜x/

Proposition 5. The optimal response-linear achievable interpolator (17) in random features regression
is the limit of preconditioned gradient descent on the last layer,

wt+1 = wt − ηtΣz

−1∇R(wt),

provided that the algorithm converges, initialized at
(cid:18) d
δ

z ΣzxΦX T

a0 = Σ−1

In + XΦX T

(cid:19)−1

y.

9

In Section A.12, we illustrate the test error of fa, with a = aO in comparison to the test error for the
minimum-norm interpolator a = a(cid:96)2 = Z †y on a standard example.

7 Conclusion

In this paper, we investigated how to design interpolators in linear regression which have optimal gen-
eralization performance. We designed an interpolator which has optimal risk among interpolators that
are a function of the training data, population covariance, signal-to-noise ratio and prior covariance,
but does not depend on the true parameter or the noise, where this function is linear in the response
variable. We showed that this interpolator is the implicit bias of a covariance-based preconditioned
gradient descent algorithm. We identiﬁed regimes where other interpolators of interest are arbitrarily
worse using computations of their asymptotic risk as d

n → γ > 1 with d, n → ∞.

In particular, we found a regime where the variance term of the minimum-norm interpolator is
arbitrarily large compared to our interpolator. This conﬁrms the phenomenon that implicit bias has an
important inﬂuence on generalization through the choice of optimization algorithm.

We identiﬁed a second regime where the interpolator that has best variance is arbitrarily worse than
our interpolator. In this second example, both interpolators are the implicit bias of the same algorithm,
but with different initialization. This contributes to illustrating that initialization has an important
inﬂuence on generalization.

We also considered an empirical approximation of the optimal response-linear achievable interpolator,
which uses only the training data X and y and does not assume knowledge of the population
covariance matrix, the signal-to-noise ratio or the prior covariance and empirically observe that it
generalizes in a nearly identical way to the optimal response-linear achievable interpolator in the
examples that we consider.

A limitation of this work includes a precise guarantee on the approximation error of the Graphical
Lasso for a general covariance matrix Σ. Some guarantees are in (Ravikumar et al., 2011), however
establishing guarantees for a general covariance matrix would be a contribution on its own.

A natural question for future research, which also motivated our work, is how to systematically design
new ways of interpolation, which are adapted to the distribution of the data and related to notions of
optimality, for more complex overparametrized machine learning models such as neural networks.

8 Acknowledgements

The authors would like to thank Dominic Richards, Edgar Dobriban and the anonymous reviewers
for valuable insights which contributed to the technical quality of the paper. Eduard Oravkin was
part-time employed at the Department of Statistics during a part of this project. Patrick Rebeschini
was supported in part by the Alan Turing Institute under the EPSRC grant EP/N510129/1.

References

S. Amari, J. Ba, R. B. Grosse, X. Li, A. Nitanda, T. Suzuki, D. Wu, and J. Xu. When does precon-
ditioning help or hurt generalization? In International Conference on Learning Representations,
2021.

Z. Bai and J. Silverstein. Spectral Analysis of Large Dimensional Random Matrices. Springer Series

in Statistics. Springer New York, 01 2010.

Z. D. Bai and Y. Q. Yin. Limit of the smallest eigenvalue of a large dimensional sample covariance

matrix. The Annals of Probability, 21(3):1275–1294, 07 1993.

J. Baik and J. W. Silverstein. Eigenvalues of large sample covariance matrices of spiked population

models. Journal of Multivariate Analysis, 97(6):1382 – 1408, 2006.

P. L. Bartlett, P. M. Long, G. Lugosi, and A. Tsigler. Benign overﬁtting in linear regression.

Proceedings of the National Academy of Sciences, 117(48):30063–30070, 2020.

10

P. L. Bartlett, A. Montanari, and A. Rakhlin. Deep learning: a statistical viewpoint, 2021. arXiv

preprint arXiv:2103.0917.

M. Belkin, D. Hsu, S. Ma, and S. Mandal. Reconciling modern machine-learning practice and the
classical bias–variance trade-off. Proceedings of the National Academy of Sciences, 116(32):
15849–15854, 2019.

L. Chizat, E. Oyallon, and F. Bach. On lazy training in differentiable programming. In Advances in

Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

E. Dobriban and S. Wager. High-dimensional asymptotics of prediction: Ridge regression and

classiﬁcation. The Annals of Statistics, 46(1):247–279, 07 2015.

J. Friedman, T. Hastie, and R. Tibshirani. Sparse inverse covariance estimation with the graphical

lasso. Biostatistics, 9(3):432–441, 12 2007.

B. Ghorbani, S. Mei, T. Misiakiewicz, and A. Montanari. Linearized two-layers neural networks in

high dimension. The Annals of Statistics, 49(2):1029 – 1054, 2021.

S. Gunasekar, J. Lee, D. Soudry, and N. Srebro. Characterizing implicit bias in terms of optimization
geometry. In Proceedings of the 35th International Conference on Machine Learning, volume 80,
pages 1832–1841, 2018.

W. Hachem, P. Loubaton, and J. Najim. Deterministic equivalents for certain functionals of large

random matrices. Ann. Appl. Probab., 17(3):875–930, 06 2007.

T. Hastie, A. Montanari, S. Rosset, and R. J. Tibshirani. Surprises in high-dimensional ridgeless least

squares interpolation, 2019. arXiv preprint arXiv:1903.08560.

A. Jacot, F. Gabriel, and C. Hongler. Neural tangent kernel: Convergence and generalization in neural
networks. In Advances in Neural Information Processing Systems, volume 31. Curran Associates,
Inc., 2018.

I. M. Johnstone. On the distribution of the largest eigenvalue in principal components analysis. Ann.

Statist., 29(2):295–327, 04 2001.

O. Ledoit and M. Wolf. A well-conditioned estimator for large-dimensional covariance matrices.

Journal of Multivariate Analysis, 88(2):365–411, 2004.

T. Liang and A. Rakhlin. Just interpolate: Kernel “ridgeless” regression can generalize. Ann. Statist.,

48(3):1329–1347, 06 2020.

T. Liang and P. Sur. A precise high-dimensional asymptotic theory for boosting and minimum-(cid:96)1-

norm interpolated classiﬁers, 2020. arXiv preprint arXiv:2002.01586.

V. Marˇcenko and L. Pastur. Distribution of eigenvalues for some sets of random matrices. Mathematics

of the USSR-Sbornik, 1:457–483, 01 1967.

S. Mei and A. Montanari. The generalization error of random features regression: Precise asymptotics

and double descent curve, 2019. arXiv preprint arXiv:1908.05355.

J. Mourtada. Exact minimax risk for linear least squares, and the lower tail of sample covariance

matrices, 2020. arXiv preprint arXiv:1912.10754.

V. Muthukumar, K. Vodrahalli, and A. Sahai. Harmless interpolation of noisy data in regression.

2019 IEEE International Symposium on Information Theory, pages 2299–2303, 2019.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Pretten-
hofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and
E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research,
12:2825–2830, 2011.

R. Penrose. A generalized inverse for matrices. Mathematical Proceedings of the Cambridge

Philosophical Society, 51(3):406–413, 1955.

11

R. Penrose. On best approximate solutions of linear matrix equations. Mathematical Proceedings of

the Cambridge Philosophical Society, 52(1):17–19, 1956.

A. Rahimi and B. Recht. Random features for large-scale kernel machines. In Advances in Neural

Information Processing Systems. Curran Associates, Inc., 2008.

A. Rangamani, L. Rosasco, and T. Poggio. For interpolating kernel machines, minimizing the norm

of the erm solution minimizes stability, 2020. arXiv preprint arXiv:2006.15522.

P. Ravikumar, M. J. Wainwright, G. Raskutti, and B. Yu. High-dimensional covariance estimation by
minimizing (cid:96)1-penalized log-determinant divergence. Electronic Journal of Statistics, 5:935 – 980,
2011.

D. Richards, J. Mourtada, and L. Rosasco. Asymptotics of ridge(less) regression under general source
condition. In Proceedings of The 24th International Conference on Artiﬁcial Intelligence and
Statistics, volume 130, pages 3889–3897, 13–15 Apr 2021.

F. Rubio and X. Mestre. Spectral convergence for a general class of random matrices. Statistics &

Probability Letters, 81(5):592 – 602, 2011.

J. Silverstein. Strong convergence of the empirical distribution of eigenvalues of large dimensional

random matrices. Journal of Multivariate Analysis, 55(2):331–339, 1995.

J. Silverstein and S. Choi. Analysis of the limiting spectral distribution of large dimensional random

matrices. Journal of Multivariate Analysis, 54(2):295 – 309, 1995.

V. N. Vapnik. The Nature of Statistical Learning Theory. Springer-Verlag Berlin, 1995.

C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals. Understanding deep learning requires
rethinking generalization. In 5th International Conference on Learning Representations, 2017.

12

Checklist

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reﬂect the paper’s

contributions and scope? [Yes]

(b) Did you describe the limitations of your work? [Yes] These are in the conclusion.
(c) Did you discuss any potential negative societal impacts of your work? [N/A]
(d) Have you read the ethics review guidelines and ensured that your paper conforms to

them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [Yes]
(b) Did you include complete proofs of all theoretical results? [Yes] They are in the

supplementary material.

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main exper-
imental results (either in the supplemental material or as a URL)? [Yes] The code is
in the supplementary material. The data generating process is described in Section 2.
The parameters used in Figures 1 and 2 are in the captions and the deﬁnitions of the
interpolators used in the ﬁgures are in the text. The method of obtaining the empirically
approximated interpolator is in 5.1.

(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they
were chosen)? [Yes] The parameters used in Figures 1 and 2 are in the captions. The
details of the datasplit and crossvalidation scheme of the method of obtaining the
empirically approximated interpolator is in the last paragraph of 5.1.

(c) Did you report error bars (e.g., with respect to the random seed after running experi-

ments multiple times)? [No]

(d) Did you include the total amount of compute and the type of resources used? [Yes] We
used the compute-optimized c2-standard-8 with 8 CPUs and 32GB RAM on Google
Cloud to obtain the ﬁgures.

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [N/A]
(b) Did you mention the license of the assets? [N/A]
(c) Did you include any new assets either in the supplemental material or as a URL? [N/A]

(d) Did you discuss whether and how consent was obtained from people whose data you’re

using/curating? [N/A]

(e) Did you discuss whether the data you are using/curating contains personally identiﬁable

information or offensive content? [N/A]

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if

applicable? [N/A]

(b) Did you describe any potential participant risks, with links to Institutional Review

Board (IRB) approvals, if applicable? [N/A]

(c) Did you include the estimated hourly wage paid to participants and the total amount

spent on participant compensation? [N/A]

13

