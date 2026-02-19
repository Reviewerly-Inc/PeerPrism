Embracing the chaos: analysis and diagnosis of
numerical instability in variational flows

Zuheng Xu

Trevor Campbell

Department of Statistics
University of British Columbia
[zuheng.xu | trevor]@stat.ubc.ca

Abstract

In this paper, we investigate the impact of numerical instability on the reliability
of sampling, density evaluation, and evidence lower bound (ELBO) estimation in
variational flows. We first empirically demonstrate that common flows can exhibit
a catastrophic accumulation of error: the numerical flow map deviates significantly
from the exact map—which affects sampling—and the numerical inverse flow map
does not accurately recover the initial input—which affects density and ELBO
computations. Surprisingly though, we find that results produced by flows are
often accurate enough for applications despite the presence of serious numerical
instability. In this work, we treat variational flows as dynamical systems, and
leverage shadowing theory to elucidate this behavior via theoretical guarantees
on the error of sampling, density evaluation, and ELBO estimation. Finally, we
develop and empirically test a diagnostic procedure that can be used to validate
results produced by numerically unstable flows in practice.

1

Introduction

Variational families of probability distributions play a prominent role in generative modelling and
probabilistic inference. A standard construction of a variational family involves passing a simple
reference distribution—such as a standard normal—through a sequence of parametrized invertible
transformations, i.e., a flow [1–7]. Modern flow-based variational families have the representational
capacity to adapt to highly complex target distributions while still providing computationally tractable
i.i.d. draws and density evaluations, which in turn enables convenient training by minimizing the
Kullback-Leibler (KL) divergence [8] via stochastic optimization methods [9–11].

In practice, creating a flexible variational flow often requires many flow layers, as the expressive power
of a variational flow generally increases with the number of transformations [5, 12, 13]. Composing
many transformations in this manner, however, can create numerical instability [5, 14–16], i.e., the
tendency for numerical errors from imprecise floating-point computations to be quickly magnified
along the flow. This accumulation of error can result in low-quality sample generation [14], numerical
non-invertibility [14, 15], and unstable training [16]. But in this work, we demonstrate that this is
not always the case with a counterintuitive empirical example: a flow that exhibits severe numerical
instability—with error that grows exponentially in the length of the flow—but which surprisingly still
returns accurate density values, i.i.d. draws, and evidence lower bound (ELBO) estimates.

Motivated by this example, the goal of this work is twofold: (1) to provide a theoretical understanding
of how numerical instability in flows relates to error in downstream results; and (2) to provide a
diagnostic procedure such that users of normalizing flows can check whether their particular results are
reliable in practice. We develop a theoretical framework that investigates the influence of numerical
instability using shadowing theory from dynamical systems [17–20]. Intuitively, shadowing theory
asserts that although a numerical flow trajectory may diverge quickly from its exact counterpart,

37th Conference on Neural Information Processing Systems (NeurIPS 2023).

there often exists another exact trajectory nearby that remains close to the numerical trajectory (the
shadowing property). We provide theoretical error bounds for three core tasks given a flow exhibiting
the shadowing property—sample generation, density evaluation, and ELBO estimation—that show
that the error grows much more slowly than the error of the trajectory itself. Our results pertain to
both normalizing flows [2, 3] and mixed variational flows [5]. We develop a diagnostic procedure
for practical verification of the shadowing property of a given flow. Finally, we validate our theory
and diagnostic procedure on MixFlow [5] on both synthetic and real data examples, and Hamiltonian
variational flow [6] on synthetic examples.

Related work. Chang et al. [21] connected the numerical invertibility of deep residual networks
(ResNets) and ODE stability analysis by interpreting ResNets as discretized differential equations,
but did not provide a quantitative analysis of how stability affects the error of sample generation
or density evaluation. This approach also does not apply to general flow transformations such as
coupling layers. Behrmann et al. [14] analyzed the numerical inversion error of generic normalizing
flows via bi-Lipschitz continuity for each flow layer, resulting in error bounds that grow exponentially
with flow length. These bounds do not reflect empirical results in which the error in downstream
statistical procedures (sampling, density evaluation, and ELBO estimation) tends to grow much more
slowly. Beyond the variational literature, Bahsoun et al. [22] investigated statistical properties of
numerical trajectories by modeling numerical round-off errors as small random perturbations, thus
viewing the numerical trajectory as a sample path of a Markov chain. However, their analysis was
constrained by the requirement that the exact trajectory follows a time-homogeneous, strongly mixing
measure-preserving dynamical system, a condition that does not generally hold for variational flows.
Tupper [23] established a relationship between shadowing and the weak distance between inexact
and exact orbits in dynamical systems. Unlike our work, their results are limited to quantifying the
quality of pushforward samples and do not extend to the evaluation of densities and ELBO estimation.
In contrast, our error analysis provides a more comprehensive understanding of the numerical error
of all three downstream statistical procedures.

2 Background: variational flows

A variational family is a parametrized set of probability distributions {qλ : λ ∈ Λ} on some space
X that each enable tractable i.i.d. sampling and density evaluation. The ability to obtain draws and
compute density values is crucial for fitting the family to data via maximum likelihood in generative
modelling, as well as maximizing the ELBO in variational inference,

ELBO (qλ||p) =

(cid:90)

qλ(x) log

p(x)
qλ(x)

dx,

where p(x) is an unnormalized density corresponding to a target probability distribution π. In this
work we focus on X ⊆ Rd endowed with its Borel σ-algebra and Euclidean norm ∥ · ∥, and parameter
set Λ ⊆ Rp. We assume all distributions have densities with respect to a common base measure on
X , and will use the same symbol to denote a distribution and its density. Finally, we will suppress the
parameter subscript λ, since we consider a fixed member of a variational family throughout.

Normalizing flows. One approach to building a variational distribution q is to push a reference
distribution, q0, through a measurable bijection on X . To ensure the function is flexible yet still
enables tractable computation, it is often constructed via the composition of simpler measurable,
invertible “layers” F1, . . . , FN , referred together as a (normalizing) flow [1, 2, 4]. To generate a
draw Y ∼ q, we draw X ∼ q0 and evaluate Y = FN ◦ FN −1 ◦ · · · ◦ F1(X). When each Fn is a
diffeomorphism,1 the density of q is

∀x ∈ X

q(x) =

q0(F −1
1
n=1 Jn(F −1

◦ · · · ◦ F −1
N (x))
n ◦ · · · ◦ F −1

N (x))

(cid:81)N

Jn(x) = |det ∇xFn(x)| .

(1)

An unbiased ELBO estimate can be obtained using a draw from q0 via

X ∼ q0, E(X) = log p(FN ◦ · · · ◦F1(X))−log q0(X)+

N
(cid:88)

n=1

log Jn(Fn ◦ · · · ◦F1(X)).

(2)

1A differentiable map with a differentiable inverse; we assume all flow maps in this work are diffeomorphisms.

2

(a) Banana: num. and exact orbits

(b) Banana: flow error

(c) Lin. Reg.: flow error

(d) Cross: num. and exact orbits

(e) Cross: flow error

(f) Log. Reg.: flow error

Figure 1: MixFlow forward (fwd) and backward (bwd) orbit errors on the Banana and Cross distribu-
tions, and on two real data examples. The first column visualizes the exact and numerical orbits
with the same starting point on the synthetic targets. For a better visualization, we display every 2nd
and 4th states of presented orbits (instead of the complete orbits) in Figs. 1a and 1d, respectively.
Figs. 1b, 1c, 1e and 1f show the median and upper/lower quartile forward error ∥F kx − ˆF kx∥ and
backward error ∥Bkx − ˆBkx∥ comparing k transformations of the forward exact/approximate maps
F , ˆF or backward exact/approximate maps B, ˆB. Statistics are plotted over 100 initialization draws
from the reference distribution q0. For the exact maps we use a 2048-bit BigFloat representation,
and for the numerical approximations we use 64-bit Float representation. The “exactness” of
BigFloat representation is justified in Fig. 15 in Appendix D.3.

Common choices of Fn are RealNVP [24], neural spline [25], Sylvester [26], Hamiltonian-based
[6, 7], and planar and radial [2] flows; see Papamakarios et al. [3] for a comprehensive overview.

MixFlows.
Instead of using only the final pushforward distribution as the variational distribution
q, Xu et al. [5] proposed averaging over all the pushforwards along a flow trajectory with identical
flow layers, i.e., Fn = F for some fixed F . When F is ergodic and measure-preserving for some
target distribution π, Xu et al. [5] show that averaged flows guarantee total variation convergence
to π in the limit of increasing flow length. To generate a draw Y ∼ q, we first draw X ∼ q0 and a
flow length K ∼ Unif{0, 1, . . . , N }, and then evaluate Y = F K(X) = F ◦ F ◦ · · · ◦ F (X) via K
iterations of the map F . The density formula is given by the mixture of intermediate flow densities,

∀x ∈ X ,

q(x) =

1
N + 1

N
(cid:88)

n=0

q0(F −nx)
j=1 J(F −jx)

(cid:81)n

,

J(x) = |det ∇xF (x)| .

(3)

An unbiased ELBO estimate can be obtained using a draw from q0 via

X ∼ q0, E(X) =

1
N + 1

N
(cid:88)

n=0

log

p(F nX)
q(F nX)

.

(4)

3

Instability in variational flows and layer-wise error analysis

In practice, normalizing flows often involve tens of flow layers, with each layer coupled with deep
neural network blocks [2, 4, 24] or discretized ODE simulations [6, 7]; past work on MixFlows
requires thousands of flow transformations [5]. These flows are typically chaotic and sensitive to small
perturbations, such as floating-point representation errors. As a result, the final output of a numerically
computed flow can significantly deviate from its exact counterpart, which may cause downstream
issues during training or evaluation of the flow [12, 14, 27]. To understand the accumulation of error,

3

(a) Banana sample scatter

(b) Cross sample scatter

(c) Banana log-densities

(d) Cross log-densities

Figure 2: Figs. 2a and 2b respectively show sample scatters produced by the naïve application of
MixFlow formulae targeting at the banana and cross distributions, without accounting for numerical
error. Figs. 2c and 2d display comparisons of log-densities on both synthetic examples. The true log-
target is on the left, the exact MixFlow evaluation (computed via 2048-bit BigFloat representation)
is in the middle, and the numerical MixFlow evaluation is on the right.

Behrmann et al. [14] assume that each flow layer Fn and its inverse are Lipschitz and have bounded
error, supx ∥Fn(x) − ˆFn(x)∥ ≤ δ, and analyze the error of each layer individually. This layer-wise
analysis tends to yield error bounds that grow exponentially in the flow length N (see Appendix A.1):

∥FN ◦ · · · ◦ F1(x) − ˆFN ◦ · · · ◦ ˆF1(x)∥ ≤ δ

N −1
(cid:88)

N
(cid:89)

n=0

j=n+2

Lip(Fj).

(5)

Given a constant flow map Fj = F with Lipschitz constant ℓ, the bound is O(δℓN ), which suggests
that the error accumulates exponentially in the length of the flow. In Fig. 1, we test this hypothesis
empirically using MixFlows on two synthetic targets (the banana and cross distribution) and two real
data examples (Bayesian linear regression and logistic problems) taken from past work on MixFlows
[5]. See Appendix D for the details of this test. Figs. 1b, 1c, 1e and 1f confirms that the exponential
growth in the error bound Eq. (5) is reasonable; the error does indeed grow exponentially quickly
in practice. And Fig. 16 (a)–(b) in Appendix D.3 further demonstrate that after fewer than 100
transformations both flows have error on the same order of magnitude as the scale of the target
distribution. Naïvely, this implies that sampling, density evaluation, and ELBO estimation may be
corrupted badly by numerical error.

But counterintuitively, in Fig. 2 we find that simply ignoring the issue of numerical error and using
the exact formulae yield reasonable-looking density evaluations and sample scatters. These results are
of course only qualitative in nature; we provide corresponding quantitative results later in Section 6.
But Fig. 2 appears to violate the principle of “garbage in, garbage out;” the buildup of significant
numerical error in the sample trajectories themselves does not seem to have a strong effect on the
quality of downstream sampling and density evaluation. The remainder of this paper is dedicated to
resolving this counterintuitive behavior using shadowing theory [28].

4 Global error analysis of variational flows via shadowing

4.1 The shadowing property

We analyze variational flows as finite discrete-time dynamical systems (see, e.g., [28]). In this work,
we consider a dynamical system to be a sequence of diffeomorphisms on (X , ∥ · ∥). We define the
forward dynamics (Fn)N
n=1 to
be comprised of the inverted flow layers Bn = F −1
N −(n−1) in reverse order. An orbit of a dynamical
system starting at x ∈ X is the sequence of states produced by the sequence of maps when initialized

n=1 to be the flow layer sequence, and the backward dynamics (Bn)N

4

Figure 3: A visualization of a pseudo-orbit and shadowing orbit. Solid arrows and filled dots indicate
exact orbits, while dashed arrows and open dots indicate pseudo-orbits (e.g., via numerical compu-
tations). Red indicates the exact orbit (x0, . . . , x4) that one intends to compute. Blue indicates the
numerically computed δ-pseudo-orbit (ˆx0, . . . , ˆx4). Grey indicates the corresponding ϵ-shadowing
orbit (s1, . . . , s4). At the top right of the figure, ∥ˆx4 − F4(ˆx3)∥ ≤ δ illustrates the δ numerical error
at each step of the pseudo-orbit. The grey dashed circles demonstrate the ϵ-shadowing window.

at x. Therefore, the forward and backward orbits initialized at x ∈ X are

Forward Orbit: x = x0

F1→ x1

F2→ x2 → . . . FN→ xN

x−N

BN =F −1

1← · · · ← x−2

B2=F −1

N −1← x−1

B1=F −1

N← x0 = x : Backward Orbit.

Given numerical implementations ˆFn ≈ Fn and ˆBn ≈ Bn with tolerance δ > 0, i.e.,
∥Fn(x) − ˆFn(x)∥ ≤ δ,

∥Bn(x) − ˆBn(x)∥ ≤ δ,

∀x ∈ X ,

(6)

we define the forward and backward pseudo-dynamics to be ( ˆFn)N
along with their forward and backward pseudo-orbits initialized at x ∈ X :

n=1 and ( ˆBn)N

n=1, respectively,

Forward Pseudo-Orbit: x = ˆx0

ˆF1≈F1→ ˆx1

ˆF2≈F2→ ˆx2 → . . .

ˆFN ≈FN→ ˆxN

ˆx−N

ˆBN ≈F −1

1← · · · ← ˆx−2

ˆB2≈F −1

N −1← ˆx−1

ˆB1≈F −1

N← ˆx0 = x : Backward Pseudo-Orbit.

For notational brevity, we use subscripts on elements of X throughout to denote forward/backward
orbit states. For example, given a random element Z ∈ X drawn from some distribution, Zk is
the kth state generated by the forward dynamics (Fn)N
n=1 initialized at Z, and Z−k is the kth state
generated by the backward dynamics (Bn)N
n=1 initialized at Z. Hat accents denote the same for the
pseudo-dynamics: for example, ˆZk is the kth state generated by ( ˆFn)N
The forward/backward orbits satisfy xk+1 = Fk(xk) and x−(k+1) = Bk(x−k) exactly at each step.
On the other hand, the forward/backward pseudo-orbits incur a small amount of error,

n=1 when initialized at Z.

∥ˆxk+1 − Fk(ˆxk)∥ ≤ δ

and ∥ˆx−(k+1) − Bk(ˆx−k)∥ ≤ δ,

which can be magnified quickly along the orbit. However, there often exists another exact orbit
starting at some other point s ∈ X that remains in a close neighbourhood of the numerical orbit. This
property, illustrated in Fig. 3, is referred to as the shadowing property [19, 28–31].
Definition 4.1 (Shadowing property [28]). The forward dynamics (Fn)N
property if for all x ∈ X and all ( ˆFn)N

n=1 satisfying Eq. (6), there exists an s ∈ X such that

n=1 has the (ϵ, δ)-shadowing

∀k = 0, 1, . . . , N,

∥sk − ˆxk∥ < ϵ.

An analogous definition holds for the backward dynamics (Bn)N
n=1—where there is a shadowing
orbit s−k that is nearby the pseudo-orbit ˆx−k—and for the joint forward and backward dynamics,
where there is a shadowing orbit nearby both the backward and forward pseudo-orbits simultaneously.
The key idea in this paper is that, intuitively, if the numerically computed pseudo-orbit is close to

5

some exact orbit (the shadowing orbit), statistical computations based on the pseudo-orbit—e.g.,
sampling, density evaluation, and ELBO estimation—should be close to those obtained via that exact
orbit. We will defer the examination of when (and to what extent) shadowing holds to Section 5; in
this section, we will use it as an assumption when analyzing statistical computations with numerically
implemented variational flows.

4.2 Error analysis of normalizing flows via shadowing

Sampling. Our first goal is to relate the marginal distributions of XN and ˆXN for X ∼ q0, i.e., to
quantify the error in sampling due to numerical approximation. Assume the normalizing flow has
the (ϵ, δ)-shadowing property, and let ξ0 be the marginal distribution of the shadowing orbit start
point. We suspect that ξ0 ≈ q0, in some sense; for example, we know that the bounded Lipschitz
distance DBL(ξ0, q0) is at most ϵ due to shadowing. And ξ0 is indeed an implicit function of q0; it
is a fixed point of a twice differentiable function involving the whole orbit starting at q0 [17, Page.
176]. But the distribution ξ0 is generally hard to describe more completely, and thus it is common
to impose additional assumptions. For example, past work shows that the Lévy-Prokhorov metric
DLP( ˆXN , XN ) is bounded by ϵ, under the assumption that ξ0 = q0 [23]. We provide a more general
result (Proposition 4.2) without distributional assumptions on ξ0. We control DBL(·, ·) rather than
DLP(·, ·), as its analysis is simpler and both metrize weak distance.
Proposition 4.2. Suppose the forward dynamics has the (ϵ, δ)-shadowing property, and X∼ q0. Then

sup
f :|f |≤U,Lip(f )≤ℓ

(cid:12)
(cid:12)
(cid:12)

Ef (XN ) − Ef ( ˆXN )

(cid:12)
(cid:12)
(cid:12) ≤ ℓϵ + 2U DTV (ξ0, q0) .

In particular, with ℓ = U = 1, we obtain that DBL(XN , ˆXN ) ≤ ϵ + 2DTV(ξ0, q0).

Recall the layerwise error bound from Eq. (5)—which suggests that the difference in orbit and
pseudo-orbit grows exponentially in N —and compare to Proposition 4.2, which asserts that the error
is controlled by the shadowing window size ϵ. This window size may depend on N , but we find in
Section 6 it is usually not much larger than δ in practice, which itself is typically near the precision of
the relevant numerical representation. We will show how to estimate ϵ later in Section 5.

Density evaluation. The exact density q(x) follows Eq. (1), while the approximation ˆq(x) is the
same except that we use the backward pseudo-dynamics. For x ∈ X , a differentiable function
g : X (cid:55)→ R+, define the local Lipschitz constant for the logarithm of g around x as:

Lg,ϵ(x) = sup

∥∇ log g(y)∥.

∥y−x∥≤ϵ

N
(cid:88)

Theorem 4.3 shows that, given the shadowing property, the numerical error is controlled by the
shadowing window size ϵ and the sum of the Lipschitz constants along the pseudo-orbit. The constant
Lq,ϵ occurs because we are essentially evaluating q(s) rather than q(x), where s ∈ X is the backward
shadowing orbit initialization. The remaining constants occur because of the approximation of the
shadowing orbit with the nearby, numerically-computed pseudo-orbit in the density formula.
Theorem 4.3. Suppose the backward dynamics has the (ϵ, δ)-shadowing property. Then
(cid:32)

(cid:33)

∀x ∈ X ,

| log ˆq(x) − log q(x)| ≤ ϵ ·

Lq,ϵ(x) + Lq0,ϵ(ˆx−N ) +

LJN −n+1,ϵ(ˆx−n)

.

n=1

ELBO estimation. The exact ELBO estimation function is given in Eq. (2); the numerical ELBO
estimation function ˆE(x) is the same except that we use the forward pseudo-dynamics. The quantity
E(X), X ∼ q0 is an unbiased estimate of the exact ELBO, while ˆE(X) is biased by numerical error;
Theorem 4.4 quantifies this error. Note that for simplicity we assume that the initial state distributions
q0, ξ0 described earlier are identical. It is possible to obtain a bound including a DTV(q0, ξ0) term,
but this would require the assumption that log p(x)/q(x) is uniformly bounded on X .
Theorem 4.4. Suppose the forward dynamics has the (ϵ, δ)-shadowing property, and ξ0 = q0. Then

(cid:12)
(cid:12)
(cid:12)ELBO (q||p) − E[ ˆE(X)]
(cid:12) ≤ ϵ · E
(cid:12)
(cid:12)

(cid:34)

Lp,ϵ( ˆXN ) + Lq0,ϵ(X) +

(cid:35)

LJn,ϵ( ˆXn)

for X ∼ q0.

N
(cid:88)

n=1

6

4.3 Error analysis of MixFlows via shadowing

The error analysis for MixFlows for finite length N ∈ N parallels that of normalizing flows, except
that Fn = F , Bn = B, and Jn = J for n = 1, . . . , N . However, when F and B are ergodic and
measure-preserving for the target π, we provide asymptotically simpler bounds in the large N limit
that do not depend on the difficult-to-analyze details of the Lipschitz constants along a pseudo-orbit.
These results show that the error of sampling tends to be constant in flow length, while the error of
density evaluation and ELBO estimation grows at most linearly. We say the forward dynamics has
the infinite (ϵ, δ)-shadowing property if (Fn)∞
n=1 has the (ϵ, δ)-shadowing property [32]. Analogous
definitions hold for both the backward and joint forward/backward dynamics.

Sampling. Similar to Proposition 4.2 for normalizing flows, Proposition 4.5 bounds the error in
exact draws Y and approximate draws ˆY from the MixFlow. The result demonstrates that error does
not directly depend on the flow length N , but rather on the shadowing window ϵ. In addition, in the
setting where the flow map F is π-ergodic and measure-preserving, we can (asymptotically) remove
the total variation term between the initial shadowing distribution ξ0 and the reference distribution q0.
Note that in the asymptotic bound, the distributions of Y and ˆY are functions of N .
Proposition 4.5. Suppose the forward dynamics has the (ϵ, δ)-shadowing property, and X ∼ q0. Let
Y = XK, ˆY = ˆXK for K ∼ Unif{0, 1, . . . , N }. Then
(cid:12)
(cid:12)
Ef (Y ) − Ef ( ˆY )
(cid:12)
(cid:12)
(cid:12) ≤ ℓϵ + 2U DTV (ξ0, q0) .
(cid:12)

sup
f :|f |≤U,Lip(f )≤ℓ

In particular, if ℓ = U = 1, we obtain that DBL(Y, ˆY ) ≤ ϵ + 2DTV(ξ0, q0). If additionally the
forward dynamics has the infinite (ϵ, δ)-shadowing property, F is π-ergodic and measure-preserving,
and ξ0, q0 ≪ π, then

lim sup
N→∞

DBL

(cid:16)

Y, ˆY

(cid:17)

≤ ϵ.

A direct corollary of the second result in Proposition 4.5 is that for W ∼ π, limN→∞ DBL(W, ˆY ) ≤ ϵ,
which results from the fact that DBL(W, Y ) → 0 [5, Theorem 4.1]. In other words, the bounded
Lipschitz distance of the approximated MixFlow and the target π is asymptotically controlled by
the shadowing window ϵ. This improves upon earlier theory governing approximate MixFlows, for
which the best guarantee available had total variation error growing as O(N ) [5, Theorem 4.3].

Density evaluation. The exact density follows Eq. (3); the numerical approximation is the same
except that we use the backward pseudo-dynamics. We obtain a similar finite-N result as in Theo-
rem 4.3. Further, we show that given infinite shadowing—where the shadowing window size ϵ is
independent of flow length N —the numerical approximation error asymptotically grows at most
linearly in N , in proportion to ϵ.
Theorem 4.6. Suppose the (ϵ, δ)-shadowing property holds for the backward dynamics. Then

∀x ∈ X ,

| log ˆq(x) − log q(x)| ≤ ϵ ·

Lq,ϵ(x) + max
0≤n≤N

Lq0,ϵ(ˆx−n) +

(cid:32)

(cid:33)

LJ,ϵ(ˆx−n)

.

N
(cid:88)

n=1

If additionally the backward dynamics has the infinite (ϵ, δ)-shadowing property, F is π-ergodic and
measure-preserving, Lq,ϵ(x) = o(N ) as N → ∞, and ξ0 ≪ π, then for q0-almost every x ∈ X ,

lim sup
N→∞

1
N

| log ˆq(x) − log q(x)| ≤ ϵ · E [Lq0,ϵ(X) + LJ,ϵ(X)] , X ∼ π.

ELBO estimation. The exact ELBO formula for the MixFlow is given in Eq. (4). Note that here we
do not simply substitute the forward/backward pseudo-orbits as needed; the naïve approximation of
the terms q(F nx) would involve n applications of ˆF followed by N applications of ˆB, which do not
exactly invert one another. Instead, we analyze the method proposed in [5], which involves simulating
a single forward pseudo-orbit ˆx1, . . . , ˆxN and backward pseudo-orbit ˆx−1, . . . , ˆx−N starting from
x ∈ X , and then caching these and using them as needed in the exact formula.

7

Theorem 4.7. Suppose the joint forward and backward dynamics has the (ϵ, δ)-shadowing property,
and ξ0 = q0. Then for X ∼ q0,

(cid:12)
(cid:12)ELBO (q||p) − E
(cid:12)

(cid:104) ˆE(X)

(cid:105)(cid:12)
(cid:12) ≤ ϵ · E
(cid:12)

(cid:34)

1
N + 1

N
(cid:88)

n=0

Lp,ϵ( ˆXn) + max

Lq0,ϵ( ˆXn) +

−N ≤n≤N

N
(cid:88)

(cid:35)
LJ,ϵ( ˆXn)

.

n=−N

If additionally the joint forward and backward dynamics has the infinite (ϵ, δ)-shadowing property,
F is π-ergodic and measure-preserving, and for some 1 ≤ m1 < ∞ and 1/m1 + 1/m2 = 1
Lp,ϵ, Lq0,ϵ, LJ,ϵ ∈ Lm1 (π) and dq0
1
N

dπ ∈ Lm2(π), then
(cid:105)(cid:12)
(cid:104) ˆE(X)
(cid:12) ≤ 2ϵ · E [Lq0,ϵ(X) + LJ,ϵ(X)] , X ∼ π.
(cid:12)

(cid:12)
(cid:12)ELBO (q||p) − E
(cid:12)

lim sup
N→∞

5 Computation of the shadowing window size

We have so far assumed the (ϵ, δ)-shadowing property throughout. To make use of our results in
practice, it is crucial to understand the shadowing window size ϵ. Theorem 5.1 presents sufficient
conditions for a finite dynamical system to have the shadowing property, and characterizes the size of
the shadowing window ϵ. Note that throughout this section we focus on the forward dynamics; the
backward and joint dynamics can be treated identically. Let ∥ · ∥ denote the spectral norm of a matrix.

Theorem 5.1 (Finite shadowing theorem). Suppose the dynamics (Fn)N
on X , and the pseudo-dynamics ( ˆFn)N
A : X N +1 (cid:55)→ X N by

n=1 are C 2 diffeomorphisms
n=1 satisfy Eq. (6). For a given x ∈ X , define the operator

(Au)k = uk+1 − ∇Fk+1(ˆxk)uk,

for u = (u0, u1, . . . , uN ) ∈ X N ,

k = 0, 1, . . . , N − 1.

Let

M := max

(cid:111)
(cid:110)
sup∥v∥≤2λδ ∥∇2Fn+1(ˆxn + v)∥ : n = 0, 1, . . . , N − 1

where λ = λmin(AAT )−1/2.

If 2M λ2δ ≤ 1, then the pseudo-orbit starting at x is shadowed with ϵ = 2λδ.

Proof. This result follows directly by following the proof of [18, Theorem 11.3] with nonconstant
flow maps Fn, and then using the right-inverse norm from [33, Corollary 4.2].

In order to apply Theorem 5.1 we need to (1) estimate δ, e.g., by examining numerical error of one
step of the map in practice; (2) compute λmin(AAT ); and (3) estimate M , e.g., by bounding the
third derivative of Fn. While estimating M and δ are problem-specific, one can employ standard
procedures for computing λmin(AAT ). The matrix AAT has a block-tridiagonal form,



D1DT

1 + I

−D2

AAT =








−DT
2
D2DT
2 + I
. . .

−DT
3
. . .

. . .
−DN −1 DN −1DT
N −1 + I
−DN

−DT
N
DN DT
N + I










,

where Dk = ∇Fk(ˆxk−1) ∈ Rd×d, ∀k ∈ [N ]. Notice that AAT is a symmetric positive definite
sparse matrix with bandwidth d and so has O(N d2) entries. The inherent structured sparsity can
be leveraged to design efficient eigenvalue computation methods, e.g., the inverse power method
[34], or tridiagonalization via Lanczos iterations [35] followed by divide-and-conquer algorithms
[36]. However, in our experiments, directly calling the eigmin function provided in Julia suffices; as
illustrated in Figs. 14 and 19, the shadowing window computation requires only a few seconds for low
dimensional synthetic examples with hundreds flow layers, and several minutes for real data examples.
Hence, we didn’t pursue specialized methods, leaving that for future work. It is also noteworthy that
practical computations of Dk can introduce floating-point errors, influencing the accuracy of λ. To
address this, one might consider adjusting the shadowing window size. We explain how to manage
this in Appendix B.1, and why such numerical discrepancies minimally impact results.

8

(a) Banana

(b) Cross

(c) Lin. Reg.

(d) Log. Reg.

Figure 4: MixFlow relative sample average computation error on three test
(cid:80)d
i=1[sin(x) + 1]i and (cid:80)d
ror regions indicate 25th to 75th percentile from 20 runs.

functions:
i=1[sigmoid(x)]i. The lines indicate the median, and er-

i=1[|x|]i, (cid:80)d

(a) Banana

(b) Cross

(c) Lin. Reg.

(d) Log. Reg.

Figure 5: MixFlow relative log-density evaluation error. The lines indicate the median, and error
regions indicate 25th to 75th percentile from 100 evaluations on different positions.

(a) Banana

(b) Cross

(c) Lin. Reg.

(d) Log. Reg.

Figure 6: MixFlow exact and numerical ELBO estimation over increasing flow length. The lines
indicate the averaged ELBO estimates over 200 independent forward orbits. The Monte Carlo error
for the given estimates is sufficiently small so we omit the error bar.

(a) Banana

(b) Cross

(c) Linear regression

(d) Logistic regression

Figure 7: MixFlow shadowing window size ϵ over increasing flow length. The lines indicate the
median, and error regions indicate 25th to 75th percentile from 10 runs.

Finally, since ϵ depends on λmin(AAT )−1/2, which itself depends on N , the shadowing window size
may potentially increase with N in practice. Understanding this dependence accurately for a general
dynamical system is challenging; there are examples where it remains constant, for instance (see
Examples B.1 and B.2 in Appendix B.2), but ϵ may grow with N . Our empirical results in next
section show that on representative inferential examples with over 500 flow layers, ϵ scales roughly
linearly with N and is of a similar order of magnitude as the floating point representation error. For
further discussion on this topic, we refer readers to Appendix B.2.

6 Experiments

In this section, we verify our error bounds and diagnostic procedure of MixFlow on the banana,
cross, and 2 real data targets—a Bayesian linear regression and logistic regression posterior; detailed

9

model and dataset descriptions can be found in Appendix D.1. We also provide a similar empirical
investigation for Hamiltonian flow [6] on the same synthetic targets in Appendix C.2.

We begin by assessing the error of trajectory-averaged estimates based on approximate draws. Fig. 4
displays the relative numerical estimation error compared to the exact estimation based on the same
initial draw. Although substantial numerical error was observed in orbit computation (see Fig. 16 in
Appendix D.3), the relative sample estimate error was around 1% for all four examples, suggesting
that the statistical properties of the forward orbits closely resemble those of the exact orbit.

We then focus on the density evaluation. For synthetic examples, we assessed densities at 100 evenly
distributed points within the target region (Figs. 2a and 2b). For real data, we evaluated densities at
the locations of 100 samples from MCMC method None-U-turn sampler (NUTS); detailed settings
for NUTS is described in Appendix D.2. It is evident from the relative error shown in Fig. 5 that the
numerical density closely matches the exact density evaluation, with the relative numerical error
ranging between 0.1% and 1%, which is quite small. The absolute error can be found in Fig. 17 in
Appendix D.3, showing that the density evaluation error does not grow as substantially as the orbit
computation error (Fig. 16 in Appendix D.3), which aligns with the bound in Theorem 4.6.

Fig. 6 further demonstrates the numerical error for the ELBO estimations (Eq. (4)). In each example,
both the averaged exact ELBO estimates and numerical ELBO estimates are plotted against an
increasing flow length. Each ELBO curve is averaged over 200 independent forward orbits. Across
all four examples, the numerical ELBO curve remain closely aligned with the exact ELBO curve,
indicating that the numerical error is small in comparison to the scale of the ELBO value. Moreover,
the error does not grow with an increasing flow length, and even presents a decreasing trend as N
increases in the two synthetic examples and the Logistic regression example. This aligns well with
the error bounds provided in Theorem 4.7.

Finally, Fig. 7 presents the size of the shadowing window ϵ as the flow length N increases. As noted
earlier, ϵ depends on the flow map approximation error δ and potentially N . We evaluated the size of
δ by calculating the approximation error of a single F or B for 1000 i.i.d. draws from the reference
distribution. Boxplots of δ for all examples can be found in Fig. 18 of Appendix D.3. These results
show that in the four examples, δ is close to the floating point representation error (approximately
10−14). Thus, we fixed δ at 10−14 when computing ϵ. As shown in Fig. 7, ϵ for both the forward and
backward orbits grows roughly linearly with the flow length. This growth is significantly less drastic
than the exponential growth of the orbit computation error. And crucially, the shadowing window
size is very small—smaller than 10−10 for MixFlow of length over 1000, which justifies the validity
of the downstream statistical procedures. In contrast, the orbit computation errors in the two synthetic
examples rapidly reach a magnitude similar to the scale of the target distributions, as shown in Fig. 1.

7 Conclusion

This work delves into the numerical stability of variational flows, drawing insights from shadowing
theory and introducing a diagnostic tool to assess the shadowing property. Experiments corroborate
our theory and demonstrate the effectiveness of the diagnostic procedure. However, the scope of our
current analysis is limited to downstream tasks post-training of discrete-time flows. Understanding
the impact of numerical instability during the training phase or probing into recent architectures like
continuous normalizing flows or neural ODEs [21, 37–40], remains an avenue for further exploration.
Additionally, while our theory is centered around error bounds that are proportional to the shadowing
window size ϵ, we have recognized that ϵ can grow with N . Further study on its theoretical growth
rate is needed. Finally, in our experiments, we employed a basic method to calculate the minimum
eigenvalue of AAT . Given the sparsity of this matrix, a deeper exploration into a more efficient
computational procedure is merited.

References

[1] Esteban Tabak and Cristina Turner. A family of non-parametric density estimation algorithms. Communi-

cations on Pure and Applied Mathematics, 66(2):145–164, 2013.

[2] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International

Conference on Machine Learning, 2015.

10

[3] George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshmi-
narayanan. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning
Research, 22:1–64, 2021.

[4] Ivan Kobyzev, Simon Prince, and Marcus Brubaker. Normalizing flows: an introduction and review of
current methods. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(11):3964–3979,
2021.

[5] Zuheng Xu, Naitong Chen, and Trevor Campbell. MixFlows: principled variational inference via mixed

flows. arXiv:2205.07475, 2022.

[6] Naitong Chen, Zuheng Xu, and Trevor Campbell. Bayesian inference via sparse Hamiltonian flows. In

Advances in Neural Information Processing Systems, 2022.

[7] Anthony Caterini, Arnaud Doucet, and Dino Sejdinovic. Hamiltonian variational auto-encoder. In Advances

in Neural Information Processing Systems, 2018.

[8] Solomon Kullback and Richard Leibler. On information and sufficiency. The Annals of Mathematical

Statistics, 22(1):79–86, 1951.

[9] Matthew Hoffmann, David Blei, Chong Wang, and John Paisley. Stochastic variational inference. Journal

of Machine Learning Research, 14:1303–1347, 2013.

[10] Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, and David Blei. Automatic Differentia-

tion Variational Inference. Journal of Machine Learning Research, 18(14), 2017.

[11] Rajesh Ranganath, Sean Gerrish, and David Blei. Black box variational inference. In International

Conference on Artificial Intelligence and Statistics, 2014.

[12] Jonas Köhler, Andreas Krämer, and Frank Noe. Smooth normalizing flows. In Advances in Neural

Information Processing Systems, 2021.

[13] Zhifeng Kong and Kamalika Chaudhuri. The expressive power of a class of normalizing flow models. In

International Conference on Artificial Intelligence and Statistics, 2020.

[14] Jens Behrmann, Paul Vicol, Kuan-Chieh Wang, Roger Grosse, and Jörn-Henrik Jacobsen. Understanding
and mitigating exploding inverses in invertible neural networks. In International Conference on Artificial
Intelligence and Statistics, 2021.

[15] Shifeng Zhang, Chen Zhang, Ning Kang, and Zhenguo Li. ivpf: Numerical invertible volume preserving
flow for efficient lossless compression. In The IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2021.

[16] Aidan N Gomez, Mengye Ren, Raquel Urtasun, and Roger B Grosse. The reversible residual network:
Backpropagation without storing activations. In Advances in Neural Information Processing Systems,
2017.

[17] Brian Coomes, Hüseyin Koçak, and Kenneth Palmer. Shadowing in discrete dynamical systems. In Six

lectures on dynamical systems, pages 163–211. World Scientific, 1996.

[18] Kenneth Palmer. Shadowing in dynamical systems: theory and applications, volume 501. Springer Science

& Business Media, 2000.

[19] Sergei Yu Pilyugin. Shadowing in dynamical systems. Springer, 2006.

[20] Lucas Backes and Davor Dragiˇcevi´c. Shadowing for nonautonomous dynamics. Advanced Nonlinear

Studies, 19(2):425–436, 2019.

[21] Bo Chang, Lili Meng, Eldad Haber, Lars Ruthotto, David Begert, and Elliot Holtham. Reversible
architectures for arbitrarily deep residual neural networks. In Proceedings of the AAAI Conference on
Artificial Intelligence, 2018.

[22] Wael Bahsoun, Huyi Hu, and Sandro Vaienti. Pseudo-orbits, stationary measures and metastability.

dynamical systems, 29(3):322–336, 2014.

[23] Paul Tupper. The relation between approximation in distribution and shadowing in molecular dynamics.

SIAM Journal on Applied Dynamical Systems, 8(2):734–755, 2009.

[24] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using Real NVP. In Interna-

tional Conference on Learning Representations, 2017.

11

[25] Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios. Neural spline flows. In Advances

in Neural Information Processing Systems, 2019.

[26] Rianne van den Berg, Leonard Hasenclever, Jakub Tomczak, and Max Welling. Sylvester normalizing

flows for variational inference. In Conference on Uncertainty in Artificial Intelligence, 2018.

[27] Sameera Ramasinghe, Kasun Fernando, Salman Khan, and Nick Barnes. Robust normalizing flows using

Bernstein-type polynomials. arXiv:2102.03509, 2021.

[28] Sergei Yu Pilyugin. Theory of pseudo-orbit shadowing in dynamical systems. Differential Equations, 47

(13):1929–1938, 2011.

[29] Abbas Fakhari and Helen Ghane. On shadowing: ordinary and ergodic. Journal of Mathematical Analysis

and Applications, 364(1):151–155, 2010.

[30] Robert Corless and Sergei Yu Pilyugin. Approximate and real trajectories for generic dynamical systems.

Journal of mathematical analysis and applications, 189(2):409–423, 1995.

[31] Jonathan Meddaugh.

Shadowing as a structural property of the space of dynamical systems.

arXiv:2106.01957, 2021.

[32] Sergei Yu Pilyugin and O.B. Plamenevskaya. Shadowing is generic. Topology and its Applications, 97(3):

253–266, 1999.

[33] Ivan Dokmani´c and Rémi Gribonval. Beyond moore-penrose part i: generalized inverses that minimize

matrix norms. arXiv:1706.08349, 2017.

[34] Michelle Schatzman. Numerical Analysis: a Mathematical Introduction. Oxford University Press, 2002.

[35] Cornelius Lanczos. An iteration method for the solution of the eigenvalue problem of linear differential

and integral operators. 1950.

[36] Ed S. Coakley and Vladimir Rokhlin. A fast divide-and-conquer algorithm for computing the spectra of

real symmetric tridiagonal matrices. Applied and Computational Harmonic Analysis, 2013.

[37] Juntang Zhuang, Nicha C Dvornek, sekhar tatikonda, and James s Duncan. MALI: A memory efficient and
reverse accurate integrator for neural ODEs. In International Conference on Learning Representations,
2021.

[38] Juntang Zhuang, Nicha Dvornek, Xiaoxiao Li, Sekhar Tatikonda, Xenophon Papademetris, and James
Duncan. Adaptive checkpoint adjoint method for gradient estimation in neural ODE. In International
Conference on Machine Learning, 2020.

[39] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential

equations. In Advances in Neural Information Processing Systems, 2018.

[40] Patrick Kidger. On neural differential equations. arXiv:2202.02435, 2022.

[41] Yu Qiao and Nobuaki Minematsu. A study on invariance of f -divergence and its application to speech

recognition. IEEE Transactions on Signal Processing, 58(7):3884–3890, 2010.

[42] George Birkhoff. Proof of the ergodic theorem. Proceedings of the National Academy of Sciences, 17(12):

656–660, 1931.

[43] Tanja Eisner, Bálint Farkas, Markus Haase, and Rainer Nagel. Operator Theoretic Aspects of Ergodic

Theory. Graduate Texts in Mathematics. Springer, 2015.

[44] Brian Coomes, Hüseyin Koçak, and Kenneth Palmer. Rigorous computational shadowing of orbits of

ordinary differential equations. Numerische Mathematik, 69:401–421, 1995.

[45] Joseph Frederick Elliott. The characteristic roots of certain real symmetric matrices. Master’s thesis,

University of Tennessee - Knoxville, 1953.

[46] Sergey Yu Pilyugin. Spaces of dynamical systems. de Gruyter, 2nd edition, 2019.

[47] Tomas Geffner and Justin Domke. MCMC variational inference via uncorrected Hamiltonian annealing.

In Advances in Neural Information Processing Systems, 2021.

[48] Guodong Zhang, Kyle Hsu, Jianing Li, Chelsea Finn, and Roger Grosse. Differentiable annealed importance
sampling and the perils of gradient noise. In Advances in Neural Information Processing Systems, 2021.

12

[49] Tomas Geffner and Justin Domke. Langevin diffusion variational inference. In International Conference

on Artificial Intelligence and Statistics, 2023.

[50] Achille Thin, Nikita Kotelevskii, Alain Durmus, Maxim Panov, Eric Moulines, and Arnaud Doucet. Monte

Carlo variational auto-encoders. In International Conference on Machine Learning, 2021.

[51] Heikki Haario, Eero Saksman, and Johanna Tamminen. An adaptive Metropolis algorithm. Bernoulli,

pages 223–242, 2001.

[52] Athanasios Tsanas, Max A. Little, Patrick E. McSharry, and Lorraine O. Ramig. Accurate telemonitoring of
parkinson’s disease progression by noninvasive speech tests. IEEE Transactions on Biomedical Engineering,
57:884–893, 2010. URL https://api.semanticscholar.org/CorpusID:7382779.

[53] Sérgio Moro, Paulo Cortez, and Paulo Rita. A data-driven approach to predict the success of bank

telemarketing. Decision Support Systems, 62:22–31, 2014.

[54] Radford Neal. MCMC using Hamiltonian dynamics. In Steve Brooks, Andrew Gelman, Galin Jones, and

Xiao-Li Meng, editors, Handbook of Markov chain Monte Carlo, chapter 5. CRC Press, 2011.

[55] Kai Xu, Hong Ge, Will Tebbutt, Mohamed Tarek, Martin Trapp, and Zoubin Ghahramani. Ad-
vancedHMC.jl: A robust, modular and efficient implementation of advanced HMC algorithms.
In
Symposium on Advances in Approximate Bayesian Inference, 2020.

13

A Proofs

A.1 Layer-wise error analysis

The layer-wise error analysis follows by recursing the one-step bound on the difference between the exact and
numerical flow:

∥FN ◦ · · · ◦ F1x − ˆFN ◦ · · · ◦ ˆF1x∥
≤ ∥FN ◦ · · · ◦ F1x − FN ◦ ˆFN −1 ◦ . . . ˆF1x∥ + ∥FN ◦ ˆFN −1 ◦ . . . ˆF1x − ˆFN ◦ · · · ◦ ˆF1x∥
≤ Lip(FN )∥FN −1 ◦ · · · ◦ F1x − ˆFN −1 ◦ · · · ◦ ˆF1x∥ + δ

≤ · · · ≤ δ

N −1
(cid:88)

N
(cid:89)

n=0

j=n+2

Lip(Fj).

A.2 Error bounds for normalizing flows

Proof of Proposition 4.2. Let S be the random initial state of the orbit that shadows the pseudo-orbit of X ∼ q0.
By triangle inequality,

(cid:12)
(cid:12)
(cid:12)

Ef (XN ) − Ef ( ˆXN )

(cid:12)
(cid:12) ≤ |Ef (XN ) − Ef (SN )| +
(cid:12)

(cid:12)
(cid:12)
(cid:12)

Ef (SN ) − Ef ( ˆXN )

(cid:12)
(cid:12)
(cid:12) .

By ϵ-shadowing and ℓ-Lipshitz continuity of f ,
(cid:12)
(cid:12)
(cid:12) ≤ ℓE∥SN − ˆXN ∥ ≤ ℓϵ.
(cid:12)f (SN ) − f ( ˆXN )
(cid:12)
(cid:12)

Ef (SN ) − Ef ( ˆXN )

(cid:12)
(cid:12) ≤ E
(cid:12)

(cid:12)
(cid:12)
(cid:12)

Next

sup
f :|f |≤U,Lip(f )≤ℓ

|Ef (XN ) − Ef (SN )|

≤ sup

f :|f |≤U

|Ef (XN ) − Ef (SN )| = 2U DTV (XN , SN ) = 2U DTV (X, S) .

The last equality is due to the fact that XN and SN are the map of X and S, respectively, under the bijection
FN ◦ · · · ◦ F1, and the fact that the total variation is invariant under bijections [41, Theorem 1].

Proof of Theorem 4.3. Let s ∈ X be the initial state of the backward shadowing orbit. By triangle inequality,

| log ˆq(x) − log q(x)| ≤ | log q(x) − log q(s)| + | log ˆq(x) − log q(s)|.

For the first term on the right-hand side, note that for some y on the segment from x to s,

|log q(x) − log q(s)| =

(cid:12)
(cid:12)∇ log q(y)T (x − s)
(cid:12)

(cid:12)
(cid:12)
(cid:12) ≤ ϵ

sup
∥y−x∥≤ϵ

∥∇ log q(y)∥.

(7)

For the second term,

log ˆq(x) − log q(s) = |log q0(ˆx−N ) − log q0(s−N )| +

N
(cid:88)

n=1

(cid:12)log JN −(n−1)(ˆx−n) − log JN −(n−1)(s−n)(cid:12)
(cid:12)
(cid:12) .

We apply the same technique to bound each term as in Eq. (7).

Proof of Theorem 4.4. By definition, ELBO (q||p) = E [E(X)], and by hypothesis the distribution q0 of the
initial flow state is equal to the distribution ξ0 of the shadowing orbit initial state, so E [E(X)] = E [E(S)].
Hence

(cid:104) ˆE(X)

. Finally,

(cid:105)

(cid:105)(cid:12)
(cid:12) ≤ E
(cid:12)

(cid:104)(cid:12)
(cid:12)E(S) − ˆE(X)
(cid:12)

(cid:12)
(cid:12)
(cid:12)

(cid:12)
(cid:12)ELBO (q||p) − E
(cid:12)

(cid:12)
(cid:12)E(s) − ˆE(x)
(cid:12)

(cid:12)
(cid:12)
(cid:12)

(cid:32)

log p(sN ) − log q0(s) +

=

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

N
(cid:88)

n=1

(cid:33)

(cid:32)

log Jn(sn)

−

log p(ˆxN ) − log q0(x) +

N
(cid:88)

n=1

log Jn(ˆxn)

(cid:33)(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

≤ |log p(sN ) − log p(ˆxN )| + |log q0(s) − log q0(x)| +

N
(cid:88)

n=1

|log Jn(sn) − log Jn(ˆxn)| .

The proof is completed by bounding each difference with the local Lipschitz constant around the pseudo-orbit
times the shadowing window size ϵ, and applying the expectation.

14

A.3 Error bounds for MixFlows

Proof of Proposition 4.5. Let S be the initial point of the random shadowing orbit. By the definition of
MixFlows,

(cid:12)
Ef (Y ) − Ef ( ˆY )
(cid:12)
(cid:12)

(cid:12)
(cid:12)
(cid:12) =

≤

≤

(cid:34)

(cid:34)

(cid:34)

(cid:12)
(cid:12)
E
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
E
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
E
(cid:12)
(cid:12)
(cid:12)

1
N + 1

1
N + 1

1
N + 1

N
(cid:88)

n=0

N
(cid:88)

n=0

N
(cid:88)

n=0

f (Xn) − f (Sn) + f (Sn) − f ( ˆXn)

(cid:35)(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

f (Xn) − f (Sn)

f (Xn) − f (Sn)

(cid:35)(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:35)(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

+ E

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

1
N + 1

N
(cid:88)

n=0

f (Sn) − f ( ˆXn)

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

+

1
N + 1

N
(cid:88)

n=0

E

(cid:12)
(cid:12)f (Sn) − f ( ˆXn)
(cid:12)

(cid:12)
(cid:12)
(cid:12) .

To bound the second term, we apply shadowing and the Lipschitz continuity of f to show that each
(cid:12)
(cid:12)
(cid:12)f (Sn) − f ( ˆXn)
(cid:12)
(cid:12)
(cid:12) ≤ ℓϵ, and hence their average has the same bound. For the first term, in the case where N is
fixed and finite,

sup
f :|f |≤U,Lip(f )≤ℓ

(cid:34)

(cid:12)
(cid:12)
E
(cid:12)
(cid:12)
(cid:12)

1
N + 1

N
(cid:88)

n=0

f (Xn) − f (Sn)

(cid:35)(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

≤

≤

=

=

1
N + 1

1
N + 1

1
N + 1

1
N + 1

N
(cid:88)

n=0

N
(cid:88)

n=0

N
(cid:88)

n=0

N
(cid:88)

n=0

sup
f :|f |≤U,Lip(f )≤ℓ

|E [f (Xn) − f (Sn)]|

2U

1
2

sup
f :|f |≤1

|E [f (Xn) − f (Sn)]|

2U DTV (Xn, Sn)

2U DTV (X, S)

= 2U DTV (ξ0, q0) .

We now consider the large-N limiting case when F is π-ergodic and measure-preserving, q0 ≪ π and ξ0 ≪ π,
and U = ℓ = 1. Let Z = SK , K ∼ Unif{0, 1, . . . , N }, and let W ∼ π. In this case, by the triangle inequality,

sup
f :|f |≤U,Lip(f )≤ℓ

(cid:34)

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

E

1
N + 1

N
(cid:88)

n=0

f (Xn) − f (Sn)

(cid:35)(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

≤ DBL (Y, W ) + DBL (Z, W ) .

By [5, Theorem 4.1], both terms on the right-hand side converge to 0 as N → ∞.

Lemma A.1. If ∀i ∈ [n], ai, bi > 0, then

(cid:12)
(cid:12)
log
(cid:12)
(cid:12)

a1 + · · · + an
b1 + · · · + bn

(cid:12)
(cid:12)
(cid:12)
(cid:12)

≤ max
1≤i≤n

(cid:26)(cid:12)
(cid:12)
(cid:12)
(cid:12)

log

(cid:27)

(cid:12)
(cid:12)
(cid:12)
(cid:12)

ai
bi

Proof. Define A := (cid:80)n

i=1 ai,

log

a1 + · · · + an
b1 + · · · + bn

= − log

b1 + · · · + bn
a1 + · · · + an

= − log

n
(cid:88)

i=1

ai
A

bi
ai

Jensen’s inequality yields that

log

a1 + · · · + an
b1 + · · · + bn

≤

n
(cid:88)

i=1

ai
A

log

ai
bi

≤ max
1≤i≤n

(cid:26)

log

(cid:27)

.

ai
bi

Applying the same technique to the other direction yields the result.

Proof of Theorem 4.6. First, shadowing and the triangle inequality yields

| log ˆq(x) − log q(x)| ≤ | log q(x) − log q(s)| + | log ˆq(x) − log q(s)|

≤ ϵLq,ϵ(x) + | log ˆq(x) − log q(s)|.

By Lemma A.1, we have

| log ˆq(x) − log q(s)| ≤ max
0≤n≤N

= max
0≤n≤N

q0(ˆx−n)/ (cid:81)n
q0(s−n)/ (cid:81)n

(cid:12)
(cid:12)
(cid:12)
log
(cid:12)
(cid:12)
|log ˆqn(x) − log qn(s)| ,

j=1 J(ˆx−j)
j=1 J(s−j)

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

15

n
(cid:88)

j=1

N
(cid:88)

n=1

N
(cid:88)

n=1

where qn is the density of the length-n normalizing flow with constant forward flow map F , and ˆqn is its
approximation via the pseudo-orbit. Using the same technique as in the proof of Theorem 4.3,
(cid:33)

(cid:32)

max
0≤n≤N

|log ˆqn(x) − log qn(s)| ≤ max
0≤n≤N

ϵ ·

Lq0,ϵ(ˆx−n) +

(cid:32)

≤ ϵ ·

max
0≤n≤N

Lq0,ϵ(ˆx−n) +

LJ,ϵ(ˆx−j)

(cid:33)

LJ,ϵ(ˆx−n)

.

Combining this with the earlier bound yields the first stated result. To obtain the second result in the infinite
shadowing setting, we repeat the process of bounding max0≤n≤N |log ˆqn(x) − log qn(s)|, but rather than
expressing each supremum around the pseudo-orbit ˆx−n, we express it around the shadowing orbit s−n to find
that

(cid:32)

(cid:33)

max
0≤n≤N

|log ˆqn(x) − log qn(s)| ≤ ϵ ·

max
0≤n≤N

Lq0,ϵ(s−n) +

LJ,ϵ(s−n)

.

We then bound max0≤n≤N with a sum and merge this with the first bound to find that

lim sup
N→∞

1
N

| log ˆq(x) − log q(x)| ≤ ϵ · lim sup
N→∞

(cid:32)

1
N

Lq,ϵ(x) +

1
N

N
(cid:88)

n=0

Lq0,ϵ(s−n) +

(cid:33)

LJ,ϵ(s−n)

.

1
N

N
(cid:88)

n=1

Since Lq,ϵ(x) = o(N ) as N → ∞, the first term decays to 0. The latter two terms are ergodic averages under
the backward dynamics initialized at s; if the pointwise ergodic theorem [42], [43, p. 212] holds at s, then
the result follows. However, the pointwise ergodic theorem holds only π-almost surely for each of Lq0,ϵ and
LJ,ϵ. Denote Z to be the set of π-measure zero for which the theorem does not apply. Suppose there is a set
A ⊆ X such that q0(A) > 0, and x ∈ A implies that s ∈ Z. But then ξ0(Z) > 0, which contradicts ξ0 ≪ π.
Therefore the pointwise ergodic theorem applies to s for q0-almost every x ∈ X .

Proof of Theorem 4.7. By definition, ELBO (q||p) = E [E(X)], and by hypothesis the distribution q0 of the
initial flow state is equal to the distribution ξ0 of the joint shadowing orbit initial state, so E [E(X)] = E [E(S)].
. So applying the triangle inequality, we find that
Hence

(cid:104) ˆE(X)

(cid:105)

(cid:12)
(cid:12)ELBO (q||p) − E
(cid:12)

(cid:105)(cid:12)
(cid:12) ≤ E
(cid:12)

(cid:12)
(cid:12)
(cid:12)E(s) − ˆE(x)
(cid:12)
(cid:12)
(cid:12) ≤

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

1
N + 1

N
(cid:88)

n=0

(cid:12)
(cid:12)
(cid:12)

(cid:104)(cid:12)
(cid:12)E(S) − ˆE(X)
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
(cid:12)
(cid:12)
(cid:12)

p(sn)
p(ˆxn)

log

+

1
N + 1

N
(cid:88)

n=0

log

1
N +1

(cid:80)N

j=0

1
N +1

(cid:80)N

j=0

q0(ˆxn−j )
i=1 J(ˆxn−i)
q0(sn−j )
i=1 J(sn−i)

(cid:81)j

(cid:81)j

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

For the first sum, each term is bounded using the local Lipschitz constant,

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

1
N + 1

N
(cid:88)

n=0

log

p(sn)
p(ˆxn)

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

≤

≤

1
N + 1

1
N + 1

N
(cid:88)

n=0

N
(cid:88)

n=0

|log p(sn) − log p(ˆxn)|

ϵLp,ϵ(ˆxn).

For the second, we apply Lemma A.1 and then use the local Lipschitz constants,
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

q0(ˆxn−j )
i=1 J(ˆxn−i)
q0(sn−j )
i=1 J(sn−i)

1
N + 1

1
N + 1

1
N +1

1
N +1

1
N +1

N
(cid:88)

N
(cid:88)

(cid:80)N

(cid:80)N

(cid:80)N

(cid:80)N

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
(cid:12)
(cid:12)
(cid:12)
(cid:12)

log

log

n=0

n=0

j=0

j=0

j=0

j=0

≤

(cid:81)j

(cid:81)j

(cid:81)j

(cid:81)j

q0(ˆxn−j )
i=1 J(ˆxn−i)
q0(sn−j )
i=1 J(sn−i)

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

1
N +1
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

max
0≤j≤N

log

q0(ˆxn−j )
i=1 J(ˆxn−i)
q0(sn−j )
i=1 J(sn−i)

(cid:81)j

(cid:81)j

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

≤

1
N + 1

≤

1
N + 1

N
(cid:88)

n=0

N
(cid:88)

n=0

≤ ϵ

(cid:32)

(cid:32)

1
N + 1

N
(cid:88)

n=0

max
0≤j≤N

|log q0(ˆxn−j) − log q0(sn−j)| +

max
0≤j≤N

Lq0,ϵ(ˆxn−j) +

j
(cid:88)

i=1

LJ,ϵ(ˆxn−i)

(cid:33)

LJ,ϵ(ˆxn−i)

≤ ϵ

max
−N ≤n≤N

Lq0,ϵ(ˆxn) +

1
N + 1

N
(cid:88)

N
(cid:88)

n=0

i=1

(cid:32)

≤ ϵ

max
−N ≤n≤N

Lq0,ϵ(ˆxn) +

(cid:33)

LJ,ϵ(ˆxn)

.

N
(cid:88)

n=−N

16

|log J(ˆxn−i) − log J(sn−i)|

j
(cid:88)

i=1
(cid:33)

Combining this with the previous bound and taking the expectation yields the first result,

(cid:12)
(cid:12)ELBO (q||p) − E
(cid:12)

(cid:104) ˆE(X)

(cid:105)(cid:12)
(cid:12) ≤ ϵ · E
(cid:12)

(cid:34)

1
N + 1

N
(cid:88)

n=0

Lp,ϵ( ˆXn) + max

Lq0,ϵ( ˆXn) +

−N ≤n≤N

N
(cid:88)

(cid:35)

LJ,ϵ( ˆXn)

.

n=−N

Once again to arrive at the second result, we redo the analysis but center the supremum Lipschitz constants
around the shadowing orbit points, and replace the maximum with a sum, yielding the bound

(cid:12)
(cid:12)ELBO (q||p) − E
(cid:12)

(cid:105)(cid:12)
(cid:104) ˆE(X)
(cid:12) ≤ ϵ · E
(cid:12)

≤ ϵ · E

(cid:34)

(cid:34)

1
N + 1

1
N + 1

N
(cid:88)

n=0

N
(cid:88)

n=0

Lp,ϵ(Sn) + max

Lq0,ϵ(Sn) +

−N ≤n≤N

Lp,ϵ(Sn) +

N
(cid:88)

Lq0,ϵ(Sn) +

n=−N

n=−N

N
(cid:88)

(cid:35)

LJ,ϵ(Sn)

n=−N

N
(cid:88)

(cid:35)

LJ,ϵ(Sn)

.

Now since ξ0 = q0 by assumption, Sn

d= Xn for each n ∈ N, so

(cid:12)
(cid:12)ELBO (q||p) − E
(cid:12)

(cid:104) ˆE(X)

(cid:105)(cid:12)
(cid:12) ≤ ϵ · E
(cid:12)

(cid:34)

1
N + 1

N
(cid:88)

n=0

Lp,ϵ(Xn) +

N
(cid:88)

Lq0,ϵ(Xn) +

N
(cid:88)

(cid:35)

LJ,ϵ(Xn)

.

n=−N

n=−N

We divide by N and take the limit supremum:
(cid:104) ˆE(X)

(cid:12)
(cid:12)ELBO (q||p) − E
(cid:12)

lim sup
N→∞

1
N

(cid:105)(cid:12)
(cid:12)
(cid:12)

≤ ϵ lim sup
N→∞

·E

(cid:34)

1
N (N + 1)

N
(cid:88)

n=0

Lp,ϵ(Xn) +

1
N

N
(cid:88)

Lq0,ϵ(Xn) +

n=−N

1
N

N
(cid:88)

n=−N

(cid:35)

LJ,ϵ(Xn)

.

Consider just the term
(cid:34)

E

1
N

N
(cid:88)

n=0

(cid:35)

LJ,ϵ(Xn)

=

(cid:90) 1
N

N
(cid:88)

n=0

LJ,ϵ(F nx)q0(dx) =

(cid:90) 1
N

N
(cid:88)

n=0

LJ,ϵ(F nx)

dq0
dπ

(x)π(dx).

[43, Theorem 8.10 (vi)] asserts that as long as LJ,ϵ ∈ Lm1 (π) and dq0
1 ≤ m1 < ∞, then

dπ ∈ Lm2 (π) for 1/m1 + 1/m2 = 1,

lim
N→∞

(cid:90) 1
N

N
(cid:88)

n=0

LJ,ϵ(F nx)

dq0
dπ

(cid:90)

(x)π(dx) =

LJ,ϵ(x)π(dx)

(cid:90) dq0
dπ

Applying this result to each term above yields the stated result.

π(dx) = E [LJ,ϵ(X)] , X ∼ π.

A.4 Norm of matrix right inverse

Notice that the operator A in Theorem 5.1 has a following matrix representation:



−D1

A =





I
−D2

I
. . .







. . .
−DN I

∈ RdN ×d(N +1), where Dk = ∇Fk(ˆxk−1), ∀k ∈ [N ].

(8)

Proposition A.2 ([33, Corollary 4.2]). Let A be as defined in Eq. (8). Suppose (Fn)N
Then

n=1 are all differentiable.

A† := AT (AAT )−1 ∈ arg min
X

∥X∥,

subject to AX = I,

(9)

and ∥A†∥ = σmin(A)−1, where σmin(A) denotes the smallest singular value of A.

Proof of Proposition A.2. Note that A has full row rank due to the identity blocks. In this case, AAT is invertible,
thus AT (AAT )−1 is a valid right inverse of A. Eq. (9) is then obtained by a direct application of [33, Corollary
4.2].
To obtain the norm of A†, since A has full row rank,

A = U ΣV T , where U ∈ RdN ×dN , V ∈ Rd(N +1)×d(N +1) are orthonormal matrices,

Σ ∈ RdN ×d(N +1) is a rectangular diagonal matrix with full row rank.

17

Then

∥A†∥ =

U ΣV T V ΣT U T (cid:17)−1(cid:13)

(cid:13)
(cid:13)
(cid:13)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

=

V ΣT U T (cid:16)
(cid:13)V ΣT U T U (ΣΣT )−1U T (cid:13)
= ∥Σ(ΣΣT )−1∥
= σmin(A)−1

(cid:13)
(cid:13)

Notice that since AAT is an invertible Hermitian matrix, σmin(A) is strictly positive.

B Discussion about the shadowing window ϵ

B.1 Numerical error when computing ϵ

As mentioned in Section 5, due to the floating-point error δ involved in evaluating ∇Fk, the evaluation of A is
perturbed by the numerical error. This consequntely leads to an inaccuracy of the estimation of λ = ∥A−1
r ∥.
Here A−1
r denotes the right inverse of A. In this section, we explain how to calibrate the calculation of λ to take
into account of this error.

This computational challenge, rooted in numerical inaccuracies, was previously acknowledged in Coomes et al.
[17, Section 3.4], with a provided remedy. Suppose we compute terms in ∇Fk(ˆxk−1) with a floating-point error
N δ) Frobenius norm error into the practically calculated A. Here we denote the digital
δ. This introduces O(
computed A by ˜A, ˜λ := ∥ ˜A−1
r ∥, and ˜δ := ∥A − ˜A∥. According to Coomes et al. [17, Eq. 45], we can employ
the following upper bound on λ:

√

λ ≤ (1 − ˜δ˜λ)−1˜λ =

1 − O(

(cid:16)

√

(cid:17)−1

N δ )˜λ

˜λ.

Substituting values of δ, N , and computed ˜λ from our experiments (refer to Fig. 7), this amounts to an
inconsequential relative error of around (1 − 10−9)−1 ≈ 1 on λ.

It is also worth discussing the the shadowing window computation methods proposed in Coomes et al. [17,
Section 3.4]. As opposed to our generic eigenvalue computation introduced in Section 5, the computation
procedure presented in Coomes et al. [17, Section 3.4] can potentially offer better scalability. However, the
methodology in Coomes et al. [17] operates under the heuristic that a dynamical system with shadowing is
intrinsically hyperbolic, or nearly so. Specifically, the “hyperbolic splitting” discussed in Coomes et al. [44,
P.413, Appendix B] (i.e., the choice of ℓ), and the “hyperbolic threshold” choice in Coomes et al. [44, P.415,
Appendix B] (i.e., the choice of p), both implicitly assume that the dynamics either exhibit hyperbolicity or
tend closely to it. Such presuppositions might be overly restrictive for generic variational flows, given that
the underlying dynamics are general time-inhomogeneous systems. When the underlying dynamics are not
(nearly) hyperbolic, applying strategies from Coomes et al. [17] could potentially lead to a poor estimation of
the shadowing window.

B.2 Dependence of ϵ on N

One may notice that ϵ depends on N implicitly through λmin(AAT )−1/2. Recall that A can be represented as a
matrix



−D1

A =





I
−D2

I
. . .







. . .
−DN I

∈ RdN ×d(N +1), where Dk = ∇Fk(ˆxk−1), ∀k ∈ [N ].

And the shadowing window is given by ϵ = 2δλmin(AAT )−1/2. If λmin(AAT )−1/2 scales badly with N , our
O(N ϵ) error bounds become vacuous. Therefore, in this section, we collect some insights about the scaling of ϵ
over increasing N . It is important to note that understanding the explicit quantitative relationship between ϵ and
N for general nonautonomous dynamical systems is quite challenging. Although current literature of dynamical
systems focuses on studying conditions so that ϵ does not scale with N , these conditions are typically difficult
to verify in practice or are only applicable in restrictive settings such as time-homogeneous linear dynamical
systems. For a more systematic study, see [17, 20, 28]. The examples provided below merely serve as synthetic
illustrations for intuition and should not be taken as definitive statements for all cases.
Example B.1 (Scaling map in R). Consider the time-homogeneous dynamics formed by T : R (cid:55)→ R such that
∀x ∈ R, T (x) = Cx for some C > 0. Then the corresponding shadowing window of a δ-pseudo-orbit is

(cid:20)

ϵ = 2δ

(C − 1)2 + 2C

(cid:18)

1 − cos

(cid:19)(cid:21)− 1
2

.

π
N + 1

18

Therefore, for all C ̸= 1, as N → ∞, ϵ = O(δ). For C = 1, ϵ = O(N δ). The formula for λmin(AAT ) above
arises because in this case, AAT has the following symmetric tridiagonal form,

C 2 + 1
−C










AAT =

−C

C 2 + 1 −C
. . .
−C C 2 + 1

. . .

. . .

whose eigenvalues have the following closed form [45, Page. 19–20]:

−C










,

−C
C 2 + 1

λk(AAT ) = C 2 + 1 − 2C cos

kπ
N + 1

,

k = 1, . . . , N.

◁

Example B.2 (Uniform hyperbolic linear map[46, Theorem 12.3 and p. 171]). Consider the time-homogeneous
linear dynamical system in Rd, T : x (cid:55)→ Ax, where A ∈ Rd×d is similar to a block-diagonal matrix
2 ∥ ≤ λ for some λ ∈ (0, 1). Then for all N ∈ N, any δ-pseudo-
diag(A1, A2) such that ∥A1∥ ≤ λ and ∥A−1
orbit is ϵ-shadowed, with ϵ = 1+λ
◁

1−λ δ.

As illustrated in Examples B.1 and B.2, it is possible for the ϵ-shadowing property to hold in a way that ϵ
does not depend on the flow length N and remains in a similar magnitude as δ. This phenomenon is referred
to as the infinite shadowing property [17, 19, 32] as mentioned in the beginning of Section 4.3. The infinite
shadowing property is shown to be generic in the space of all dynamical systems [32], meaning that dynamical
systems exhibiting this property form a dense subset of the collection of all dynamical systems. However, while
genericity offers valuable insights into the typical behavior of dynamical systems, it does not imply that all
systems possess the property. As previously explained, only a small class of systems has been shown to have the
infinite shadowing property, and verifying this property for a specific given system remains a challenging task.

C Normalizing flow via Hamiltonian dynamics

While past work on normalizing flows [2, 3, 24–26] typically builds a flexible flow family by composing
numerous parametric flow maps that are agnostic to the target distribution, recent target-informed variational
flows focus on designing architectures that take into account the structures of the target distribution. A common
approach for constructing such target-informed flows is to use Langevin and Hamiltonian dynamics as part of the
flow transformations [6, 7, 47–50]. In this section, we begin with a concise overview of two Hamiltonian-based
normalizing flows: Hamiltonian variational auto-encoder (HVAE) [7] and Hamiltonian flow (HamFlow) [6].
Subsequently, we present corollaries derived from Theorems 4.3 and 4.4, specifically applying these general
results to HVAE and HamFlow. Finally, we offer empirical evaluations of our theory for HamFlow on the same
synthetic examples as those examined for MixFlow.

Hamiltonian dynamics (Eq. (10)) describes the evolution of a particle’s position θt and momentum ρt within a
physical system. This movement is driven by a differentiable negative potential energy log p(θt) and kinetic
energy 1

2 ρT

t ρt:

dρt
dt

= ∇ log π (θt)

dθt
dt

= ρt.

(10)

For t ∈ R, we define the mappings Ht : R2d → R2d, transforming (θs, ρs) (cid:55)→ (θt+s, ρt+s) according to
the dynamics of Eq. (10). Intuitively, traversing the trajectory of Ht, t ∈ R allows particles to efficiently
explore the distribution of interest’s support, thereby motivating the use of Ht as a flow map. In practice, the
exact Hamiltonian flow Ht is replaced by its numerical counterpart—leapfrog integration—which involves K
iterations of the subsequent steps:

ˆρ ← ρ +

ϵ
2

∇ log π (θ)

θ ← θ + ϵˆρ

ρ ← ˆρ +

ϵ
2

∇ log π (θ) .

Here ϵ denotes the step size of the leapfrog integrator. We use TK,ϵ : R2d → R2d to define the map by
sequencing the above three steps K times. There are two key properties of TK,ϵ that make it suitable for
normalizing flow constructions. TK,ϵ is invertible (i.e., T −1
K,ϵ = TK,−ϵ) and has a simple Jacobian determinant
of the form: | det ∇TK,ϵ| = 1. This enables a straightforward density transformation formula: for any density
q0(θ, ρ) on R2d, pushforward of q under TK,ϵ has the density q0(TK,−ϵ(θ, ρ)).

Given the above properties, Caterini et al. [7] and Chen et al. [6] developed flexible normalizing flows—
HVAE and HamFlow, respectively—utilizing flow transformations through alternating compositions of multiple
TK,ϵ and linear transformations. Notably, the momentum variable ρ introduced in the Hamiltonian dynamics
Eq. (10) necessitates that both HVAE and HamFlow focus on approximating the augmented target distribution

19

¯π(θ, ρ) ∝ π(θ) · exp(− 1
momentum distribution N (0, Id).

2 ρT ρ) on R2d. This represents the joint distribution of the target π and an independent

Specifically, flow layers of HVAE alternate between the leapfrog transformation TK,ϵ and a momentum scaling
layer:

(θ, ρ) (cid:55)→ (θ, γρ),

γ ∈ R+.

And those of HamFlow alternate between TK,ϵ and a momentum standardization layer:

(θ, ρ) (cid:55)→ (θ, Λ(ρ − µ)), Λ ∈ Rd×d is lower triangular with positive diagonals , µ ∈ Rd.

C.1 Error analysis for Hamiltonian-based flows

We have provided error analysis for generic normalizing flows in Section 4.2. In this section, we specialize the
error bounds for density evaluation (Theorem 4.3) and ELBO estimation (Theorem 4.4) for the two Hamiltonian-
based flows. π denotes the target distribution of interest on Rd, and ¯p denotes the augmented target distribution
on R2d. p and ¯p respectively denote the unnormalized density of π and ¯π.
Corollary C.1. Let q be either HVAE or HamFlow of N layers with q0 = N (0, I2d). Suppose the backward
dynamics has the (ϵ, δ)-shadowing property. Then

∀x ∈ R2d,

| log ˆq(x) − log q(x)| ≤ ϵ · (Lq,ϵ(x) + ∥ˆx−N ∥) + ϵ2.

Proof of Corollary C.1. Since the Jacobian determinant of each HVAE/HamFlow layer is constant,

And since q0 is a standard normal distribution,

LJn,ϵ(x) = 0,

∀n ∈ [N ] and x ∈ R2d.

Lq0,ϵ = sup

∥y−x∥≤ϵ

∥y∥ ≤ ∥x∥ + ϵ.

(11)

(12)

The proof is then complete by directly applying Eqs. (11) and (12) to the error bound stated in Theorem 4.3.

Corollary C.2. Let q be either HVAE or HamFlow of N layers with q0 = N (0, I2d). Suppose the forward
dynamics has the (ϵ, δ)-shadowing property, and ξ0 = q0. Then
(cid:105)

√

(cid:16)

(cid:17)

(cid:12)
(cid:12)ELBO (q||¯p) − E[ ˆE(X)]
(cid:12)

(cid:12)
(cid:12)
(cid:12) ≤ ϵ ·

E

(cid:104)
L ¯p,ϵ( ˆXN )

+

2d

+ ϵ2

for X ∼ q0.

If additionally log p is L-smooth (i.e., ∀x, y, ∥∇ log p(x) − ∇ log p(y)∥ ≤ L∥x − y∥), then

(cid:12)
(cid:12)
(cid:12)ELBO (q||¯p) − E[ ˆE(X)]
(cid:12)
(cid:12)
(cid:12) ≤ ϵ ·

(cid:16)

E

(cid:104)

(cid:105)
∥∇ log ¯p( ˆXN )∥

+

√

(cid:17)

2d

+ (L ∨ 1 + 1)ϵ2

for X ∼ q0.

Proof of Corollary C.2. Similar to the proof of Fig. 2d, applying Eqs. (11) and (12) to the error bound stated in
Theorem 4.4 yields that

(cid:12)
(cid:12)
(cid:12)ELBO (q||¯p) − E[ ˆE(X)]
(cid:12) ≤ ϵ · E
(cid:12)
(cid:12)
The first result is then obtained by employing the following bound via Jensen’s inequality:

(cid:104)
L ¯p,ϵ( ˆXN ) + ∥X∥

+ ϵ2

(cid:105)

for X ∼ q0.

E[∥X∥] ≤ (cid:112)E[∥X∥2] =

√

2d

for X ∼ q0 = N (0, I2d).

To arrive at the second result, we apply the following bound on Lp,ϵ using the fact that ∇ log p is L-Lipschitz
continuous:

L ¯p,ϵ ≤ ∥∇ log ¯p(x)∥ + sup

∥∇ log ¯p(x) − ∇ log ¯p(y)∥ ≤ ∥∇ log ¯p(x)∥ + L ∨ 1 · ϵ.

∥y−x∥≤ϵ

C.2 Empirical results for Hamiltonian flow

In this section, we empirically validate our error bounds and the diagnostic procedure of HamFlow using two
synthetic targets: the banana and cross distributions. We employ a warm-started HamFlow, as detailed in
[6, Section 3.4]. This is achieved through a single pass of 10,000 samples from the reference distribution,
q0, without additional optimization. Specifically, the µ and Λ values for each flow standardization layer are
determined by the sample mean and the Cholesky factorization of the sample covariance, respectively, of the
batched momentum samples processed through the layer. For the banana distribution, we set the leapfrog step K
at 200 and a step size ϵ of 0.01. For the cross distribution, we set K to 100 with an ϵ value of 0.02. Figs. 8a

20

and 8b showcase a comparison between 1000 i.i.d. samples from the warm-started HamFlow with 400 layers
and the true targets. The results illustrate that HamFlow generates samples of decent quality.

We start by examining the numerical orbit computaiton error. As demonstrated in Fig. 9, the pointwise evaluation
error grows exponentially with increasing flow depth, and quickly reaches the same order of magnitude as the
scale of the target distributions. However, similar to what we have observed in the experiments of MixFlow
(described in Section 6), the substantial numerical error of orbit computation does not impact the accuracy
of Monte Carlo estimations of expectations and ELBO estimations significantly. Fig. 10 shows the relative
numerical error of Monte Carlo estimates of the expectations of three test functions. The relative samples
estimation error remains within 5% for the two bounded Lipschitz continuous test functions, aligning with our
error bound in Proposition 4.2. Moreover, even for the unbounded test function, the error stays within reasonable
limits: 10% for the banana and 5% for the cross examples. Fig. 11 then compares the numerical ELBO estimates
and exact ELBO estimates for HamFlow over increasing flow length. In both examples, the numerical ELBO
curve is almost identical to the exact one, notwithstanding the considerable sample computation error of the
forward orbit.

However, unlike the small log-density evaluation error we observed in the MixFlow experiments, HamFlow’s
log-density shows significant issues because of the numerical errors of backward orbits computation. In Fig. 12,
we measure the numerical error for the log-density evaluation of HamFlow at 100 i.i.d. points from the target
distribution. As we add more flow layers, the error grows quickly. For the banana example, the log-density
evaluation error can even go above 1000. This raises two main questions: (1) Why is the ELBO estimation error
different from the log-density evaluation error in HamFlow? (2) Why is the scaling of the log-density evaluation
error different between MixFlow and HamFlow? Our theory presented in Section 4 and Appendix C.1 offers
explanation to these questions.

To address the first question, we compare the bounds for HamFlow on log-density evaluation (Corollary C.1) and
ELBO estimation (Corollary C.2). The main difference between these bounds is in the local Lipschitz constants.
The error in log-density evaluation is based on Lq,ϵ while the error in ELBO estimation is based on E[L ¯p,ϵ].
These two quantities— Lq,ϵ and E[L ¯p,ϵ]—relate to the smoothness of the learned log-density of HamFlow and
log-density of the target. Figs. 8c to 8f display log p and log q side-by-side for both synthetic examples. Indeed,
in both cases, log q markedly deviates from log p and presents drastic fluctuations, indicating that Lq,ϵ can be
substantially larger than E[L ¯p,ϵ]. Fig. 13 further provides a quantitative comparison on how Lq,ϵ and E[L ¯p,ϵ]
change as we increase the flow length. Note that since both Lq,ϵ and L ¯p,ϵ are intractable in general, we instead
focus on the scaling of ∥∇ log q∥ (which is a lower bound of Lq,ϵ) and an upper bound of E[L ¯p,ϵ] (derived from
case-by-case analysis for both the banana and cross distributions; see Appendix C.3 for detailed derivation).
Results show that in both synthetic examples, Lq,ϵ can increase exponentially over increasing flow length and is
eventually larger than E[L ¯p,ϵ] by 30 orders of magnitude, while E[L ¯p,ϵ] does not grow and remains within a
reasonable scale.

Similarly, the differences in log-density errors between MixFlow and HamFlow come from how smooth the
learned log-density is in each method, according to the error bounds presented in Theorems 4.3 and 4.6. Fig. 2
visualizes the learned log-densities of MixFlow, which matches closely to the actual target distribution, and has
a smoother output than HamFlow. This matches what was noted in Xu et al. [5, Figure 3], where MixFlow was
shown to approximate the target density better than standard normalizing flows.

Finally, Fig. 14 presents the single step evaluation error δ of HamFlow and the size of the shadowing window ϵ
as the flow length N increases. Results show that for both synthetic examples, δ is approximately 10−7, hence
we fix δ to be this value when computing the shadowing window ϵ. Compared to the significant orbit evaluation
error when N is large, the size of ϵ remains small for both the forward and backward orbits (in the scale of 10−4
to 10−3 when N = 400). And more importantly, the size of ϵ grows less drastically than the orbit computation
error with the flow length—roughly quadratically in the banana example, and linearly in the cross example,
which is similar to what is observed in the MixFlow experiments.

C.3 Upper bounds of L ¯p,ϵ for synthetic examples

Recall that we denote ¯p(x) ∝ p([x]1:2) · exp(− 1
synthetic target we are interested.

2 [x]T

3:4[x]3:4) the augmented target and p denotes the 2d-

Notice that

L ¯p,ϵ(x) = sup

∥∇ log p([y]1:2) − [y]3:4∥ ≤ Lp,ϵ([x]1:2) + ∥[x]3:4∥ + ϵ,

∥y−x∥≤ϵ

where Lp,ϵ requires a case-by-case analysis. Therefore, in this section, we focus on bounding the local Lipschitz
constant for log p.
Proposition C.3. For the 2-dimensional banana distribution, i.e.,

(cid:18)

p(x) ∝ exp

−

1
2

x2
1
σ2 −

1
2

(cid:0)x2 + bx2

(cid:19)
1 − σ2b(cid:1)

,

21

we have that for all x =:

(cid:21)

(cid:20)x1
x2

∈ R2,

(cid:18)

Lp,ϵ(x) ≤ ∥∇ log p(x)∥ + ϵ ·

2b|x1| + max

(cid:26)(cid:12)
(cid:12)
(cid:12)
(cid:12)

6b2x2

1 + 2bx2 − 2σ2b2 +

(cid:27)

, 1

1
σ2

(cid:12)
(cid:12)
(cid:12)
(cid:12)

+ 2b max (cid:8)6b|x1| + 6b(ϵ + ϵ2), 1ϵ(cid:9)

(cid:19)

Proof of Proposition C.3. By the mean value theorem,

Lp,ϵ(x) ≤ ∥∇ log p(x)∥ + sup

∥∇ log p(y) − ∇ log p(x)∥

∥y−x∥≤ϵ

≤ ∥∇ log p(x)∥ + sup

∥∇2 log p(y)∥ · ϵ.

∥y−x∥≤ϵ

The proof will then complete by showing that
(cid:26)(cid:12)
(cid:12)
(cid:12)
(cid:12)

∥∇2 log p(y)∥ ≤ 2b|x1| + max

sup
∥y−x∥≤ϵ

6b2x2

1 + 2bx2 − 2σ2b2 +

(cid:27)

, 1

1
σ2

(cid:12)
(cid:12)
(cid:12)
(cid:12)

+ 2b max (cid:8)6b|x1| + 6b(ϵ + ϵ2), ϵ(cid:9) .

(13)

To verify Eq. (13), note that

∇2 log p(x) = −

(cid:20)6b2x2

1 + 2bx2 − 2σ2b2 + 1
σ2

2bx1

(cid:21)

2bx1
1

The Gershgorin circle theorem then yields that

∥∇2 log p(x)∥ ≤ 2b|x1| + max

(cid:26)(cid:12)
(cid:12)
(cid:12)
(cid:12)

6b2x2

1 + 2bx2 − 2σ2b2 +

(cid:27)

, 1

.

1
σ2

(cid:12)
(cid:12)
(cid:12)
(cid:12)

Additionally, we have

sup
∥y1−x1∥≤ϵ

|y1| ≤ |x1| + ϵ

and

sup
∥y−x∥≤ϵ

(cid:12)
(cid:12)
(cid:12)
(cid:12)

6b2y2

1 + 2by2 − 2σ2b2 +

1
σ2

(cid:12)
(cid:12)
(cid:12)
(cid:12)

≤

(cid:12)
(cid:12)
6b2x2
(cid:12)
(cid:12)

1 + 2bx2 − 2σ2b2 +

1
σ2

(cid:12)
(cid:12)
(cid:12)
(cid:12)

+ 2b max (cid:8)6b|x1| + 6b(ϵ + ϵ2), ϵ(cid:9) .

The proof is then completed.

Proposition C.4. For the cross distribution, a 4-component Gaussian mixture of the form
(cid:21)(cid:19)
(cid:20)0.152
0
(cid:20)1
0

(cid:21)
(cid:20)−2
0
(cid:20) 0
−2

0
0.152
(cid:20)0.152
0

0
0.152

(cid:20)1
0
(cid:21)

p(x) =

(cid:20)0
2

(cid:20)2
0

(cid:21)(cid:19)

(cid:21)(cid:19)

1
4

1
4

0
1

0
1

x|

x|

x|

x|

N

N

N

N

+

+

+

(cid:18)

(cid:18)

(cid:18)

(cid:18)

(cid:21)

(cid:21)

,

,

,

,

1
4

1
4

(cid:21)(cid:19)

,

we have that for all x ∈ R2,

Lp,ϵ(x) ≤

1

0.152 (2 + ∥x∥ + ϵ) .

Proof of Proposition C.4. For notational convenience, denote the density of 4 Gaussian component by
p1, p2, p3, p4, hence

Examining ∇ log p(x) directly yields that for all x ∈ R2,

log p(x) = log {p1(x) + p2(x) + p3(x) + p4(x)} − log 4.

∇ log p(x) =

1
p1(x) + p2(x) + p3(x) + p4(x)

4
(cid:88)

i=1

pi(x)Σ−1

i

(µi − x)

∈ conv (cid:8)Σ−1

i

(µi − x) : i = 1, 2, 3, 4(cid:9) .

Here µi and Σi denotes the Gaussian mean and covariance of pi. Then by Jensen’s inequality,
(cid:41)

(cid:40)

sup
∥y−x∥≤ϵ

∥∇ log p(y)∥ ≤ max

sup
∥y−x∥≤ϵ

∥Σ−1
i

(cid:40)

(cid:32)

(µi − y)∥ : i = 1, 2, 3, 4

(cid:33)(cid:41)

≤ max
i∈[4]

∥Σ−1

i ∥ ·

∥µi∥ + sup

∥y∥

∥y−x∥≤ϵ

≤

1

0.152 (2 + ∥x∥ + ϵ) .

22

(a) Banana: HamFlow sample scatters

(b) Cross: HamFlow sample scatter

(c) True log-pdf

(d) HamFlow log-pdf

(e) True log-pdf

(f) HamFlow log-pdf

Figure 8: Visualization of sample scatters and density of HamFlow. Figure (a)–(b) each shows
1000 i.i.d. draws from HamFlow distribution (red) and target distribution (green). (c) and (e) show
respectively the exact log-density of the Banana and the cross distribution, while (d) and (f) show
respectively the sliced HamFlow log-density evaluation at ρ = (0, 0) on those two targets. The
HamFlow log-density evaluations are computed via 2048-big BigFloat representation.

(a) Banana

(b) Banana (log-scale)

(c) Cross

(d) Cross (log-scale)

Figure 9: HamFlow forward (fwd) and backward (bwd) orbit errors on synthetic examples. (a)
and (c) show the median and upper/lower quartile forward error ∥F kx − ˆF kx∥ and backward error
∥Bky− ˆBky∥ comparing k transformations of the forward exact/approximate maps F , ˆF or backward
exact/approximate maps B, ˆB. And (b) and (d) respectively display (a) and (c) in a logarithmic scale,
intended to provide a clearer illustration of the exponential growth of the error. The lines indicate the
median, and error regions indicate 25th to 75th percentile from 100 independent runs, which take an
i.i.d. draw of x from q0 and draw y from the actual target distribution p.

(a) Banana

(b) Banana (log-scale)

(c) Cross

(d) Cross (log-scale)

i=1[|x|]i, (cid:80)d

Figure 10: HamFlow relative sample average computaion error on three test functions:
(cid:80)d
i=1[sigmoid(x)]i. Each line denotes the numerical error
computed from Monte Carlo average of the three test functions via 100 independent forward orbits
starting at i.i.d. draws from q0.

i=1[sin(x) + 1]i and (cid:80)d

23

(a) Banana

(b) Cross

Figure 11: HamFlow exact and numerical ELBO estimation over increasing flow length. The
lines indicate the averaged ELBO estimates over 100 independent forward orbits. The Monte Carlo
error for the given estimates is sufficiently small so we omit the error bar.

(a) Banana

(b) Banana (log-scale)

(c) Cross

(d) Cross (log-scale)

Figure 12: HamFlow log-density evaluation error on synthetic examples. Log-densities are assessed
at positions of 100 independent samples from the target distribution. (b) and (d) respectively display
(a) and (c) in a logarithmic scale for a better illustration of the growth of error over increasing flow
length. The lines indicate the median, and error regions indicate 25th to 75th percentile from the 100
evaluations.

(a) Banana

(c) Banana

(a) Cross

(c) Cross

Figure 13: Comparison of HamFlow’s Lq,ϵ and E[L ¯p,ϵ( ˆXN )] on the banana and cross distributions.
Figs. 13a and 13c plot ∥∇ log q(x)∥ against flow length N ; the lines indicate the median, and error
regions indicate 25th to 75th percentile from i.i.d. 100 independent draws of x from ¯p. Figs. 13b
and 13d plot estimated upper bounds of E[L ¯p,ϵ( ˆXN )] (averaging over 100 independent draws from q)
against flow length N . The choice and detailed derivation of the upper bounds of L ¯p,ϵ can be found
in Appendix C.3

D Additional experimental details

All experiments were conducted on a machine with an AMD Ryzen 9 3900X and 32G of RAM.

D.1 Model description

The two synthetic distributions tested in this experiment were

• the banana distribution [51]:

y =

(cid:21)

(cid:20)y1
y2

(cid:18)

∼ N

0,

(cid:20)100
0

(cid:21)(cid:19)

0
1

,

x =

(cid:20)

y1
y2 + by2
1 − 100b

(cid:21)

,

b = 0.1;

24

(a) Banana

(b) Banana

(c) Banana

(d) Cross

(e) Cross

(f) Cross

Figure 14: HamFlow forward (fwd) and backward (bwd) single transformation error δ on both
synthetic examples (Figure (a) and (d)), shadowing window size ϵ over increasing flow length (Figure
(b) and (e)), and shadowing window computation time (Figure (c) and (f)). Figure. (a) and (d)
(Fwd err.) and
show the distribution of forward error

(cid:110)
∥F nx0 − ˆF ◦ F n−1x0∥ : n = 1, . . . , N

(cid:111)

(cid:110)
∥F nx0 − ˆB ◦ F n+1x0∥ : n = 1, . . . , N

(Bwd err.). These errors are computed
backward error
along precise forward trajectories that originate at x0, which is sampled 50 times independently and
identically from the reference distribution q0. N here denotes the flow length for each example.
Figure (b) and (e) show ϵ over increasing flow length. Figure (c) and (f) show the wall time of
computing shadowing window over increasing flow length. The lines in Figure (b)–(c) and (e)–(f)
indicate the median, and error regions indicate 25th to 75th percentile from 50 runs.

(cid:111)

• a cross-shaped distribution: a Gaussian mixture of the form
(cid:21)
(cid:18)(cid:20)0
2
(cid:21)
(cid:18)(cid:20)2
0

(cid:20)0.152
0

0
1
(cid:21)(cid:19)

0
0.152

(cid:20)1
0

(cid:18)(cid:20)−2
0
(cid:18)(cid:20) 0
−2

x ∼

(cid:21)(cid:19)

1
4

1
4

N

N

N

N

+

+

+

,

,

(cid:21)

(cid:21)

1
4

1
4

(cid:20)1
0
(cid:20)0.152
0

,

,

(cid:21)(cid:19)

0
0.152

(cid:21)(cid:19)

;

0
1

The two real-data experiments are described below.

Linear regression. We consider a Bayesian linear regression problem where the model takes the form

Lin. Reg.: β i.i.d.∼ N (0, 1), log σ2 i.i.d.∼ N (0, 1),

yj | β, σ2 indep

∼ N

(cid:16)

j β, σ2(cid:17)
xT

,

where yj is the response and xj ∈ Rp is the feature vector for data point j. For this problem, we use the Oxford
Parkinson’s Disease Telemonitoring Dataset [52]. The original dataset is available at http://archive.ics.uci.
edu/dataset/189/parkinsons+telemonitoring, composed of 16 biomedical voice measurements from 42 patients
with early-stage Parkinson’s disease. The goal is to use these voice measurements, as well as subject age and
gender information, to predict the total UPDRS scores. We standardize all features and subsample the original
dataset down to 500 data points. The posterior dimension of the linear regression inference problems is 21.

Logistic regression. We then consider a Bayesian hierarchical logistic regression:
(cid:32)

Logis. Reg.: α ∼ Gam(1, 0.01), β | α ∼ N (0, α−1I),

yj | β

indep
∼ Bern

(cid:33)

,

1
1 + e−xT
j β

We use a bank marketing dataset [53] downsampled to 400 data points. The original dataset is available
at https://archive.ics.uci.edu/ml/datasets/bank+marketing. The goal is to use client information to predict
subscription to a term deposit. We include 8 features from dataset: client age, marital status, balance, housing
loan status, duration of last contact, number of contacts during campaign, number of days since last contact, and
number of contacts before the current campaign. For each of the binary variables (marital status and housing
loan status), all unknown entries are removed. All included features are standardized. The resulting posterior
dimension of the logistic regression problem is 9.

25

(a) Banana

(b) Cross

(c) Linear regression

(d) Logistic regression

Figure 15: MixFlow inversion error when using 2048-bit BigFloat representation. Verticle axis
shows the reconstruction error ∥BN ◦ F N (x0) − x0∥ (Fwd) and ∥F N ◦ BN (x0) − x0∥ (Bwd) for
the two synthetic examples (Figs. 15a and 15b) and the two real data examples (Figs. 15c and 15d).
F and B are implemented using 2048-bit BigFloat representation, and x0 is sampled from the
reference distribution q0. The lines indicate the median, and the upper and lower quantiles over 10
independent runs. It can be seen that in all four examples, the inversion error of MixFlow is ignorable
when using 2048-bit computation.

(a) Banana

(b) Cross

(c) Linear regression

(d) Logistic regression

Figure 16: MixFlow forward (fwd) and backward (bwd) orbit errors on both synthetic examples and
real data examples. (a)—(d) show the median and upper/lower quartile forward error ∥F kx − ˆF kx∥
and backward error ∥Bkx − ˆBkx∥ comparing k transformations of the forward exact/approximate
maps F , ˆF or backward exact/approximate maps B, ˆB. Note that Figs. 1b, 1e, 1c and 1f respectively
display the initial segments of (a)—(d) in a logarithmic scale, intended to provide a clearer illustration
of the exponential growth of the error.

D.2 MixFlow parameter settings

For all four examples, we use the uncorrected Hamiltonian MixFlow map F composed of a discretized Hamil-
tonian dynamic simulation (via leapfrog integrator) followed by a sinusoidal momentum pseudorefreshment
as used in Xu et al. [5] (see [5, Section 5, Appendices E.1 and E.2] for details). However, we use a Gaussian
momentum distribution, which is standard in Hamiltonian Monte Carlo methods [54] as opposed to the Laplace
momentum used previously in MixFlows [5]. In terms of the settings of leapfrog integrators for each target,
we used 200 leapfrog steps of size 0.02 and 60 leapfrog steps of size 0.005 for the banana and cross target
distributions respectively, and used 40 leapfrog steps of size 0.0006 and 50 leapfrog steps of size 0.002 for the
linear regression and logistic regression examples, respectively. The reference distribution q0 for each target is
chosen to be the mean-field Gaussian approximation as used in [5].

As for the NUTS used for assessing the log-density evaluation quality on two real data examples, we use the
Julia package AdvancedHMC.jl [55] with all default settings. NUTS is initialized with the learned mean of the
mean-field Gaussian approximation q0, and burns in the first 10,000 samples before collecting samples; the
burn-in samples are used for adapting the hyperparameters of NUTS with target acceptance rate set to be 0.7.

D.3 Additional MixFlow results

26

(a) Banana

(b) Cross

(c) Lin. Reg.

(d) Log. Reg.

Figure 17: MixFlow log-density evaluation error. For synthetic examples (Figs. 17a and 17b), we
assessed densities at 100 evenly distributed points within the target region (Figs. 2a and 2b). For real
data examples (Figs. 17c and 17d), we evaluated densities at the locations of 100 NUTS samples. The
lines indicate the median, and error regions indicate 25th to 75th percentile from 100 evaluations on
different positions.

(a) Banana

(b) Cross

(c) Linear regression

(d) Logistic regression

Figure 18: MixFlow forward (fwd) and backward (bwd) single transformation error δ
(a)—(d) show the distribution of
on both synthetic examples and real data examples.
(cid:111)
(cid:110)
∥F nx0 − ˆF ◦ F n−1x0∥ : n = 1, . . . , N
and backward error
(Fwd err.)
forward error
(cid:110)
∥F nx0 − ˆB ◦ F n+1x0∥ : n = 1, . . . , N
(Bwd err.). These errors are computed along precise
forward trajectories that originate at x0, which is sampled 50 times independently and identically
from the reference distribution q0. N here denotes the flow length for each example.

(cid:111)

(a) Banana

(b) Cross

(c) Linear regression

(d) Logistic regression

Figure 19: MixFlow shadowing window computation time (wall time in second) for both the forward
(Fwd) and backward (Bwd) orbits over increasing flow length. The lines indicate the median, and error
regions indicate 25th to 75th percentile from 10 runs. Note that here the Monte Carlo error between
different runs is so small that the error bar is too thin to be visualized.

27

