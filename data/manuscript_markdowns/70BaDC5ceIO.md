Under review as a conference paper at ICLR 2023

NEURAL NETWORK APPROXIMATIONS OF PDES BE-
YOND LINEARITY: REPRESENTATIONAL PERSPECTIVE

Anonymous authors
Paper under double-blind review

ABSTRACT

A burgeoning line of research has developed deep neural networks capable of
approximating the solutions to high dimensional PDEs, opening related lines of
theoretical inquiry focused on explaining how it is that these models appear to
evade the curse of dimensionality. However, most theoretical analyses thus far
have been limited to simple linear PDEs. In this work, we take a step towards
studying the representational power of neural networks for approximating solu-
tions to nonlinear PDEs. We focus on a class of PDEs known as nonlinear vari-
ational elliptic PDEs, whose solutions minimize an Euler-Lagrange energy func-
tional E(u) = (cid:82)
Ω L(∇u)dx. We show that if composing a function with Barron
norm b with L produces a function of Barron norm at most BLbp, the solution
to the PDE can be ϵ-approximated in the L2 sense by a function with Barron
norm O
. By a classical result due to Barron (1993), this cor-
respondingly bounds the size of a 2-layer neural network needed to approximate
the solution. Treating p, ϵ, BL as constants, this quantity is polynomial in dimen-
sion, thus showing that neural networks can evade the curse of dimensionality.
Our proof technique involves “neurally simulating” (preconditioned) gradient in
an appropriate Hilbert space, which converges exponentially fast to the solution
of the PDE, and such that we can bound the increase of the Barron norm at each
iterate. Our results subsume and substantially generalize analogous prior results
for linear elliptic PDEs.

(dBL)plog(1/ϵ) (cid:17)

(cid:16)

1

INTRODUCTION

Scientific applications have become one of the new frontiers for the application of deep learn-
ing (Jumper et al., 2021; Tunyasuvunakool et al., 2021; Sønderby et al., 2020). PDEs are one of the
fundamental modeling techniques in scientific domains, and designing neural network-aided solvers,
particularly in high-dimensions, is of widespread usage in many domains (Hsieh et al., 2019; Brand-
stetter et al., 2022). One of the most common approaches for applying neural networks to solve
PDEs is to parameterize the solution as a neural network and minimize a loss which characterizes
the solution (Sirignano & Spiliopoulos, 2018; E & Yu, 2017). The hope in doing so is to have a
method which computationally avoids the “curse of dimensionality”—i.e., that scales less than ex-
ponentially with the ambient dimension.

To date, neither theoretical analysis nor empirical applications have yielded a precise characteriza-
tion of the range of PDEs for which neural network-aided methods outperform classical methods.
Active research on the empirical side (Han et al., 2018; E et al., 2017; Li et al., 2020a;b) has ex-
plored several families of PDEs, e.g., Hamilton-Bellman-Jacobi and Black-Scholes, where neural
networks have been demonstrated to outperform classical grid-based methods. On the theory side,
a recent line of works (Marwah et al., 2021; Chen et al., 2021; 2022) has considered the following
fundamental question:

For what families of PDEs can the solution be represented by a small neural network?

The motivation for this question is computational: since the computational complexity of fitting a
neural network (by minimizing some objective) will grow with its size. Specifically, these works
focus on understanding when the approximating neural network can be sub-exponential in size, thus

1

Under review as a conference paper at ICLR 2023

avoiding the curse of dimensionality. Unfortunately, the techniques introduced in this line of work
have so far only been applicable to linear PDEs.

In this paper, we take the first step beyond linear PDEs, with a particular focus on nonlinear vari-
ational elliptic PDEs. These equations have the form −div(∇L(∇u)) = 0 and are instances of
nonlinear Euler-Lagrange equations. Equivalently, u is the minimizer of the energy functional
E(u) = (cid:82)
Ω L(∇u)dx. This paradigm is very generic: its origins are in Lagrangian formulations of
classical mechanics, and for different L, a variety of variational problems can be modeled or learned
(Schmidt & Lipson, 2009; Cranmer et al., 2020). These PDEs have a variety of applications in scien-
tific domains, e.g., (non-Newtonian) fluid dynamics (Koleva & Vulkov, 2018), meteorology(Weller
et al., 2016), and nonlinear diffusion equations (Burgers, 2013).

Our main result is to show that when the function L has “low complexity”, so does the solution.
The notion of complexity we work with is the Barron norm of the function, similar to Chen et al.
(2021); Lee et al. (2017). This is a frequently used notion of complexity, as a function with small
Barron norm can be represented by a small, two-layer neural network, due to a classical result
(Barron, 1993). Mathematically, our proof techniques are based on “neurally unfolding” an iterative
preconditioned gradient descent in an appropriate function space: namely, we show that each of the
iterates can be represented by a neural network with Barron norm not much worse than the Barron
norm of the previous iterate—along with showing a bound on the number of required steps.

Importantly, our results go beyond the typical non-parametric bounds on the size of an approximator
network that can be easily shown by classical regularity results of the solution to the nonlinear
variational PDEs (De Giorgi, 1957; Nash, 1957; 1958) along with universal approximation results
(Yarotsky, 2017).

2 OVERVIEW OF RESULTS

Let Ω ⊂ Rd be a bounded open set with 0 ∈ Ω and ∂Ω denote the boundary of Ω. Furthermore, we
assume that the domain Ω is such that the Poincare constant Cp is greater than 1 (see Theorem 2 for
the exact definition of the Poincare constant).

(cid:90)

We first define the energy functional whose minimizers are represented by a nonlinear variational
elliptic PDE—i.e., the Euler-Lagrange equation of the energy functional.
Definition 1 (Energy functional). For all u : Ω → R such that u|∂Ω = 0, we consider an energy
functional of the following form:

E(u) =

L(∇u)dx,

(1)

Ω
where L : Rd → R is a smooth and uniformly convex function , i.e., there exists constant 0 < λ ≤ Λ
such that for all x ∈ R we have λId ≤ D2L(x) ≤ ΛId. Further, without loss of generality1, we
assume that λ ≤ 1/Cp.

Note that due to the convexity of the function L, the minimizer u⋆ exists and is unique. The proof
of existence and uniqueness is standard (e.g., Theorem 3.3 in Fern´andez-Real & Ros-Oton (2020)).

Writing down the condition for stationarity, we can derive a (nonlinear) elliptic PDE for the mini-
mizer of the energy functional in Definition 1 .
Lemma 1. Let u⋆ : Ω → R be the unique minimizer for the energy functional in Definition 1. Then
for all φ ∈ H 1

0 (Ω) the minimizer u⋆ satisfies the following condition,

DE[u](φ) =

(cid:90)

Ω

∇L(∇u)∇φdx = 0,

(2)

where dE[u](φ) denotes the dirctional derivative of the energy functional calculated at u in the
direction of φ. Thus, the minimizer u⋆ of the energy functional satisfies the following PDE:

DE(u) := −div(∇L(∇u)) = 0

∀x ∈ Ω.

(3)

and u(x) = 0, ∀x ∈ ∂Ω. Here div denote the divergence operator.

1Since λ is a lower bound on the strong convexity constant. If we choose a weaker lower bound, we can

always ensure λ ≤ 1/Cp.

2

Under review as a conference paper at ICLR 2023

The proof for the Lemma can be found in Appendix Section A.1. Here, −div(∇L(∇·) is a functional
operator that acts on a function (in this case u). 2

Our goal is to determine if the solution to the PDE in Equation 3 can be expressed by a neural
network with a small number of parameters. In order do so, we utilize the concept of a Barron norm,
which measures the complexity of a function in terms of its Fourier representation. We show that if
composing with the function L is such that it increases it has a bounded increase in the Barron norm
of u, then the solution to the PDE in Equation 3 will have a bounded Barron norm. The motivation
for using this norm is a seminal paper (Barron, 1993), which established that any function with
Barron norm C can be ϵ-approximated in the L2 sense by a two-layer neural network with size
O(C 2/ϵ), thus evading the curse of dimensionality if C is substantially smaller than exponential in
d. Informally, we will show the following result:

Theorem 1 (Informal). Let L be convex and smooth, such that composing a function with Barron
norm b with L produces a function of Barron norm at most BLbp. Then, for all sufficiently small
ϵ > 0, the minimizer of the energy functional in Definition 1 can be ϵ-approximated in the L2 sense
by a function with Barron norm O

(dBL)plog(1/ϵ) (cid:17)

(cid:16)

.

As a consequence, when ϵ, p are thought of as constants, we can represent the solution to the Euler-
Lagrange PDE Equation 3 by a polynomially-sized network, as opposed to an exponentially sized
network, which is what we would get by standard universal approximation results and using regu-
larity results for the solutions of the PDE.

We establish this by “neurally simulating” a preconditioned gradient descent (for a strongly-convex
loss) in an appropriate Hilbert space, and show that the Barron norm of each iterate—which is a
function—is finite, and at most polynomially bigger than the Barron norm of the previous iterate.
We get the final bound by (i) bounding the growth of the Barron norm at every iteration; and (ii)
bounding the number of iterations required to reach an ϵ-approximation to the solution. The result
in formally stated in Section 5

3 RELATED WORK

Over the past few years, a growing line of work has focused on parameterizing the solutions to PDEs
with neural networks. Works such as E et al. (2017); E & Yu (2017); Sirignano & Spiliopoulos
(2018); Raissi et al. (2017) achieved impressive results on a variety of different applications and
demonstrated the empirical efficacy of neural networks in solving high-dimensional PDEs, even
outperforming previously dominant numerical approaches like finite differences and finite element
methods (LeVeque, 2007) that proceed by discretizing the input space, hence limiting their use to
problems on low-dimensional input spaces.

Several recent works look to theoretically analyze these neural network based approaches for solving
PDEs. Mishra & Molinaro (2020) look at the generalization properties of physics informed neural
networks. In Lu et al. (2021) show the generalization analysis for the Deep Ritz method for elliptic
equations like the Poisson equation and Lu & Lu (2021) extends their analysis to the Schr¨odinger
eigenvalue problem.

In addition to analyzing the generalization capabilities of the neural networks, theoretical analysis
into their representational capabilities has also gained a lot of attention. Khoo et al. (2021) show
the existence of a network by discretizing the input space into a mesh and then using convolutional
NNs, where the size of the layers is exponential in the input dimension. Sirignano & Spiliopoulos
(2018) provide a universal approximation result, showing that for sufficiently regularized PDEs,
there exists a multilayer network that approximates its solution. In Jentzen et al. (2018); Grohs
& Herrmann (2020); Hutzenthaler et al. (2020) provided a better-than-exponential dependence on
the input dimension for some special parabolic PDEs, based on a stochastic representation using
the Feynman-Kac Lemma, thus limiting the applicability of their approach to PDEs that have such
a probabilistic interpretation. Moreover, their results avoid the curse of dimensionality only over
domains with unit volume.

2For a vector valued function F : Rd → Rd, we will denote the divergence operator either by divF or by

∇ · F , where divF = ∇ · F = (cid:80)d

i=1

∂iF
∂xi

.

3

Under review as a conference paper at ICLR 2023

A recent line of work has focused on families of PDEs for which neural networks evade the curse
of dimensionality—i.e., the solution can be approximated by a neural network with a subexponen-
tial size. Marwah et al. (2021) show that for elliptic PDE’s whose coefficients are approximable by
neural networks with at most N parameters, a neural network exists that ϵ-approximates the solu-
tion and has size O(dlog(1/ϵ)N ). Chen et al. (2021) extends this analysis to elliptic PDEs with coef-
ficients with small Barron norm, and shows that if the coefficients have Barron norm bounded by B,
an ϵ-approximate solution exists with Barron norm at most O(dlog(1/ϵ)B). The work by Chen et al.
(2022) derives related results for the Schr¨odinger equation.

As mentioned, while most previous works show key regularity results for neural network approxi-
mations of solution to PDEs, most of their analysis is limited to simple linear PDEs. The focus of
this paper is towards extending these results to a family of PDEs referred to as nonlinear variational
PDEs. This particular family of PDEs consists of many famous PDEs, such as p−Laplacian (on a
bounded domain), which is used to model phenomena like non-Newtonian fluid dynamics and non-
linear diffusion processes. The regularity results for these family of PDEs was posed as Hilbert’s
XIXth problem. We note that there are classical results like De Giorgi (1957) and Nash (1957;
1958) that provide regularity estimates on the solutions of a nonlinear variational elliptic PDE of
the form in Equation 3. One can easily use these regularity estimates, along with standard univer-
sal approximation results (Yarotsky, 2017) to show that the solutions can be approximated arbitrar-
ily well. However, the size of the resulting networks will be exponentially large (i.e. suffer from the
curse of dimensionality)—so are of no use for our desired results.

4 NOTATION AND DEFINITION

In this section, we introduce some key concepts and notation that will be used throughout the paper.
For a vector x ∈ Rd, we use ∥x∥2 to denote its ℓ2 norm. Further, C∞(Ω) denotes the set of functions
f : Ω → R that are infinitely differentiable. We also define some important function spaces and
associated key results below.
Definition 2. For a vector valued function g : R → Rd, we define the Lp(Ω) norm for p ∈ [1, ∞) as

For p = ∞, we have

∥g∥Lp(Ω) =

(cid:32)(cid:90)

d
(cid:88)

Ω

i

(cid:33)1/p

|gi(x)|p dx

,

∥g∥L∞(Ω) = max
1≤i≤d

∥gi∥L∞(Ω),

where ∥gi∥L∞(Ω) = inf{c ≥ 0 : |g(x)| ≤ c for almost all x ∈ Ω}.
Definition 3. For a domain Ω, the space of functions H 1

0 (Ω) is defined as

H 1

0 (Ω) := {g : Ω → R : g ∈ L2(Ω), ∇g ∈ L2(Ω), g|∂Ω = 0}.

The corresponding norm for H 1

0 (Ω) is defined as ∥g∥H 1

0 (Ω) = ∥∇g∥L2(Ω).

We will make use of the Poincar´e inequality throughout several of our results.
Theorem 2 (Poincar´e inequality, Poincar´e (1890)). For Ω ⊂ Rd which is open and bounded, there
exists a constant Cp > 0 such that for all u ∈ H 1
0 (Ω)

∥u∥L2(Ω) ≤ Cp∥∇u∥L2(Ω).

This constant can be very benignly behaved with dimension for many natural domains—even di-
mension independent. Examples include convex domains (Payne & Weinberger, 1960).

4.1 BARRON NORMS

For a function f : Rd → R, the Fourier transform and the inverse Fourier transform are defined as

ˆf (ω) =

1
(2π)d

(cid:90)

Rd

f (x)e−ixT ωdx, and f (x) =

(cid:90)

Rd

ˆf (ω)eixT ωdω.

(4)

4

Under review as a conference paper at ICLR 2023

The Barron norm is an average of the norm of the frequency vector weighted by the Fourier magni-
tude | ˆf (ω)|. A slight technical issue is that the the Fourier transform is defined only for f : Rd → R.
Since we are interested in defining the Barron norm of functions defined over a bounded domain,
we allow for arbitrary extensions of a function outside of their domain. (This is the standard defini-
tion, e.g. in (Barron, 1993).)
Definition 4. We define F be the set of functions g ∈ L1(Ω) such that the Fourier inversion formula
g = (2π)d ˆˆf (x) holds over the domain Ω, i.e.,

(cid:26)

F =

g : Rd → R, ∀x ∈ Ω, g(x) = g(0) +

(eiωT x − 1)ˆg(ω)dω

(cid:27)

.

(cid:90)

Rd

Definition 5 (Spectral Barron Norm, (Barron, 1993)). Let Γ be a set of functions defined over Ω
such that their extension over Rd belong to F, that is,

Then we define the spectral Barron norm ∥ · ∥B(Ω) as

Γ = {f : Ω → R : ∃g, g|Ω = f, g ∈ F}

∥f ∥B(Ω) =

inf
g|Ω=f,g∈F

(cid:90)

Rd

(1 + ∥ω∥2)|ˆg(ω)|dω.

The Barron norm is an L1 relaxation of requiring sparsity in the Fourier basis—which is intuitively
why it confers representational benefits in terms of the size of a neural network required. We refer
to Barron (1993) for a more exhaustive list of the Barron norms of some common function classes.

The main theorem from Barron (1993) formalizes this intuition, by bounding the size of a 2-layer
network approximating a function with small Barron norm:
Theorem 3 (Theorem 1, Barron (1993)). Let f ∈ Γ such that ∥f ∥B(Ω) ≤ C and µ be a probability
measure defined over Ω. There exists ai ∈ Rd, bi ∈ R and ci ∈ R such that (cid:80)k
i=1 |ci| ≤ 2C, there
exists a function fk(x) = (cid:80)k

i=1 ciσ (cid:0)aT
(cid:90)

i x + b(cid:1), such that we have,
4C 2
k

(f (x) − fk(x))2 µ(dx) ≤

.

Ω

Here σ denotes a sigmoidal activation function, i.e., limx→∞ σ(x) = 1 and limx→−∞ σ(x) = 0.

Note that while Theorem 3 is stated for sigmoidal activations like sigmoid and tanh (after appropriate
rescaling), the results are also valid for ReLU activation functions, since ReLU(x) − ReLU(x − 1)
is in fact sigmoidal. We will also need to work with functions that do not have Fourier coefficients
beyond some size (i.e. are band limited), hence we introduce the following definition:
Definition 6. Let FW (Ω) be the set of functions whose Fourier coefficients vanish outside a bounded
ball, i.e.,

FW = {g : Rd → R : s.t. ∀w, ∥w∥∞ ≥ W, ˆg(w) = 0}.

Similarly, we denote

ΓW = {f : Ω → R : ∃g, g|Ω = f, g ∈ FW } .

Since we will work with vector valued function, we will also define the Barron norm of a vector-
valued function as the maximum of the Barron norms of its coordinates:
Definition 7. For a vector valued function g : Ω → Rd, we define ∥g∥B(Ω) = maxi ∥gi∥B(Ω).

5 MAIN RESULT

Before stating the main result we introduce the key assumption.
Assumption 1. The function L in Definition 1 can be approximated by a function ˜L : Rd → R such
that there exists a constant ϵL ∈ [0, λ) with supx∈Rd ∥∇L(x) − ∇ ˜L(x)∥2 ≤ ϵL∥x∥2.
Furthermore, we assume that ˜L is such that for any g ∈ H 1
0 (Ω), we have ˜L ◦ g ∈ H 1
and

0 (Ω), ˜L ◦ g ∈ F

∥ ˜L ◦ g∥B(Ω) ≤ B ˜L∥g∥p
for some constants B ˜L ≥ 0, and p ≥ 0. Furthermore, if g ∈ ΓW then ˜L ◦ g ∈ ΓkW for a k > 0.

B(Ω).

(5)

5

Under review as a conference paper at ICLR 2023

This assumption is fairly natural: it states that the function L can be approximated (up to ϵL, in
the sense of the gradients of the functions) by a function ˜L that has the property that when applied
to a function g with small Barron norm, the new Barron norm is not much bigger. The constant
p specifies the order of this growth. The functions for which our results are most interesting are
when the dependence of B ˜L on d is at most polynomial—so that the final size of the approximating
network does not exhibit curse of dimensionality. For instance, we can take L to be a multivariate
polynomial of degree up to P : we show in Lemma 7 the constant B ˜L is O(dP ) (intuitively, this
dependence comes from the total number of monomials of this degree), whereas p and k are both
O(P ).

With all the assumptions stated, we now state our main theorem:
Theorem 4 (Main Result). Consider the nonlinear variational elliptic PDE in Equation 3 which
satisfies Assumption 1 and let u⋆ ∈ H 1
0 (Ω) is a
function such that u0 ∈ ΓW0 , then for all sufficiently small ϵ > 0, and

0 (Ω) denote the solution to the PDE. If u0 ∈ H 1

(cid:38)

T :=

log

(cid:18) 2
ϵ

(E(u0) − E(u⋆))
λ

(cid:19)

/ log

(cid:32)

1

1 −

λ5
(1+Cp)Λ4

(cid:33)(cid:39)

,

there exists a function uT ∈ H 1

0 (Ω) such that uT ∈ ΓkT W0 and Barron norm bounded as

(cid:18)

∥uT ∥B(Ω) ≤

1 +

(cid:19)pT +1

∥u0∥pT

B(Ω).

0 B ˜L

λ3

(Cp + 1)Λ3 dk2W 2
0 (Ω) ≤ ϵ + ˜ϵ where

Furthermore, uT satisfies ∥uT − u⋆∥H 1

(cid:16)

ϵL

˜ϵ ≤

λ3
(Cp + 1)Λ3

∥u⋆∥H 1

0 (Ω) + 1
Λ + ϵL

λ E(u0)

(cid:17)

(cid:32)(cid:18)

1 +

λ3

(Cp + 1)Λ3 (Λ + ϵL)

(cid:19)T

(cid:33)

− 1

.

Remark 1: The function u0 can be seen as an initial estimate of the solution, that can be refined to an
estimate uT , which is progressively better at the expense of a larger Barron norm. A trivial choice
could be u0 = 0, which has Barron norm 1, and which by Lemma 2 would satisfy E(u0) − E(u∗) ≤
Λ∥u∗∥2

0 (Ω).
H 1

Remark 2: The final approximation error has two terms, T goes to ∞ as ϵ tends 0 and is a conse-
quence of the way uT is constructed—by simulating a functional (preconditioned) gradient descent
which converges to the solution to the PDE. The error term ˜ϵ stems from the approximation that we
make between ˜L and L, which grows as T increases—it is a consequence of the fact that the gradi-
ent descent updates with ˜L and L progressively drift apart as T tends to ∞.

Remark 3: As in the informal theorem, if we think of p, Λ, λ, Cp, k, ∥u0∥B(Ω) as constants, the
theorem implies u∗ can be ϵ-approximated in the L2 sense by a function with Barron norm
O
. Therefore, combining results from Theorem 4 and Theorem 3 the total param-

(dBL)plog(1/ϵ)(cid:17)

(cid:16)

eters required to ϵ-approximate u∗ by a 2-layer neural network is O

ϵ2 (dBL)2plog(1/ϵ)(cid:17)
(cid:16) 1

.

Remark 4: We further note that this result recovers (and generalizes) prior results which bound the
Barron norm of linear elliptic PDEs like Chen et al. (2021). In these results, the elliptic PDE takes
the form −div(A∇u) and A is assumed to have bounded Barron norm. Thus, ∥L ◦ u∥B(Ω) ≤
d2∥A∥B(Ω)∥u∥B(Ω), hence satisfying Equation 5 in Assumption 1 with p = 1.

6 PROOF OF MAIN RESULT

The proof will proceed by “neurally unfolding” a preconditioned gradient descent on the objective
E in the Hilbert space H 1
0 (Ω). This is inspired by previous works by Marwah et al. (2021); Chen
et al. (2021) where the authors show that for a linear elliptic PDE, an objective which is quadratic
can be designed. In our case, we show that E is “strongly convex” in some suitable sense—thus
again, bounding the amount of steps needed.

More precisely, the result will proceed in two parts:

6

Under review as a conference paper at ICLR 2023

1. First, we will show that the sequence of functions {ut}∞

t=0, where ut+1 ← ut − η(I −
∆)−1dE(ut) can be interpreted as performing preconditioned gradient descent, with the
(constant) preconditioner (I − ∆)−1. We show that in some appropriate sense (Lemma 2),
E is strongly convex in H 1

0 (Ω)—thus the updates converge at a rate of O(log(1/ϵ)).

2. We then show that the Barron norm of each iterate ut+1 can be bounded in terms of the
Barron norm of the prior iterate ut. We show this in Lemma 5, where we show that given
Assumptions1, the ∥ut+1∥B(Ω) is O(d∥ut∥p
B(Ω)). By unrolling this recursion we show that
the Barron norm of the ϵ-approximation of u⋆ is of the order O(dpT
B(Ω)), where T
are the total steps required for ϵ-approximation and ∥u0∥B(Ω) is the Barron norm of the first
function in the iterative updates.

∥u0∥p

We now proceed to delineate the main technical ingredients for both of these parts.

6.1 CONVERGENCE RATE OF SEQUENCE

The proof to show the convergence to the solution u⋆ is based on adapting the standard proof (in
finite dimension) for convergence of gradient descent when minimizing a strongly convex function
f . Recall, the basic idea is to Taylor expand f (x + δ) ≈ f (x) + ∇f (x)T δ + O(∥δ∥2). Taking
δ = η∇f (x), we lower bound the progress term η∥∇f (x)∥2 using the convexity of f , and upper
bound the second-order term η2∥∇f (x)∥2 using the smoothness of f .

We follow analogous steps, and prove that we can lower bound the progress term by using some
appropriate sense of convexity of E, and upper bound some appropriate sense of smoothness of E,
when considered as a function over H 1
0 (Ω). Precisely, we show:
Lemma 2 (Strong convexity of E in H 1
0 ). If E, L are as in Definition 1, we have
Ω −div(∇L(∇u))vdx = (cid:82)

0 (Ω) : ⟨DE(u), v⟩L2(Ω) = (cid:82)

Ω ∇L(∇u) · ∇vdx.

1. ∀u, v ∈ H 1

2. ∀u, v ∈ H 1

0 (Ω) : λ∥u − v∥H 1

0 (Ω) ≤ ⟨DE(u) − DE(v), u − v⟩L2(Ω) ≤ Λ∥u − v∥H 1

0 (Ω).

3. ∀u, v ∈ H 1
2 ∥∇v∥2

Λ

L2(Ω).

0 (Ω) : λ

2 ∥∇v∥2

L2(Ω)+⟨DE(u), v⟩L2(Ω) ≤ E(u+v)−E(u) ≤ ⟨DE(u), v⟩L2(Ω)+

4. ∀u ∈ H 1

0 (Ω) : λ

2 ∥u − u⋆∥2
H 1

0 (Ω) ≤ E(u) − E(u⋆) ≤ Λ

2 ∥u − u⋆∥2
0 (Ω).
H 1

Part 1 is a helpful way to rewrite an inner product of a “direction” v with DE(u)—it is essentially a
consequence of integration by parts and the Dirichlet boundary condition. Part 2 and 3 are common
proxies of convexity: they are ways of formalizing the notion that E is strongly convex, when viewed
as a function over H 1
0 (Ω). Finally, part 4 is a consequence of strong convexity, capturing the fact
that if the value of E(u) is suboptimal, u must be (quantitatively) far from u∗. The proof of the
Lemma can be found in the Appendix (Section B.1)

When analyzing gradient descent (in finite dimensions) to minimize a loss function E, the standard
condition for progress is that the inner product of the gradient with the direction towards the optimum
is lower bounded as ⟨DE(u), u∗ − u⟩L2(Ω) ≥ α∥u − u∗∥2
L2(Ω). From Parts 2 and 3, one can readily
see that the above condition is only satisfied “with the wrong norm”: i.e. we only have ⟨DE(u), u∗ −
u⟩L2(Ω) ≥ α∥u − u∗∥2
0 (Ω). We can fix this mismatch by instead doing preconditioned gradient,
H 1
using the fixed preconditioner (I − ∆)−1. Towards that, the main lemma about the preconditioning
we require is the following one:
Lemma 3 (Norms with preconditioning). For all v ∈ H 1

0 (Ω), we have

1. ∥(I − ∆)−1∇ · ∇u∥L2(Ω) = ∥(I − ∆)−1∆u∥L2(Ω) ≤ ∥u∥L2(Ω).

2. ⟨(I − ∆)−1v, v⟩L2(Ω) ≥ 1

1+Cp

⟨∆−1v, v⟩L2(Ω).

The first part of the lemma is a relatively simple consequence of the fact that the operators ∆ and
∇ “commute”, and therefore can be re-ordered. The latter lemma can be understood intuitively as

7

Under review as a conference paper at ICLR 2023

(I −∆)−1 and ∆−1 act as similar operators on eigenfunctions of ∆ with large eigenvalues (the extra
I does not do much)—and are only different for eigenfunctions for small eigenvalues. However,
since the smallest eigenvalues is lower bounded by 1/Cp, their gap can be bounded.

Next we utilize the results in Lemma 2 and Lemma 3 to show preconditioned gradient descent
exponentially converges to the solution to the nonlinear variational elliptic PDE in 3.
Lemma 4 (Preconditioned Gradient Descent Convergence). Let u⋆ denote the unique solution to
the PDE in Definition 3. For all t ∈ N, we define the sequence of functions

ut+1 ← ut −

λ3

(1 + Cp)Λ3 (I − ∆)−1DE(ut).

If u0 ∈ H 1

0 (Ω) after t iterations we have,
(cid:18)

E(ut+1) − E(u⋆) ≤

1 −

λ5
(1 + Cp)Λ4

(cid:19)t

(E(u0) − E(u⋆)) .

The complete proof for convergence can be found in Section B.3 of the Appendix.
Therefore, using the result from Lemma 4, i.e., ∥ut − u⋆∥2

λ (E(ut) − E(u⋆)), we have

∥ut − u⋆∥2

0 (Ω) ≤
H 1

(cid:18)

2
λ

1 −

λ5
2(1 + Cp)Λ4

(E(u0) − E(u⋆)) .

H 1

0 (Ω) ≤ 2
(cid:19)t

and ∥uT − u⋆∥2

0 (Ω) ≤ ϵ after T steps, where,
H 1

T ≥ log

(cid:18) E(u0) − E(u⋆)
λϵ/2

(cid:19)

/ log

(cid:32)

1

1 −

λ5
(1+Cp)Λ4

(cid:33)

.

(6)

(7)

6.2 BOUNDING THE BARRON NORM

Having obtained a sequence of functions that converge to the solution u⋆, we bound the Barron
norms of the iterates. We draw inspiration from Marwah et al. (2021); Lu et al. (2021) and show
that the Barron norm of each iterate in the sequence has a bounded increase on the Barron norm of
the previous iterate. Note that in general, the Fourier spectrum of a composition of functions can
not easily be expressed in terms of the Fourier spectrum of the functions being composed. However,
from Assumption 1, we know that the function L can be approximated by ˜L such that ˜L ◦ g has
a bounded increase the Barron norm of g. Thus, instead of tracking the iterates in Equation 6, we
track the Barron norm of the functions in the following sequence,

˜ut+1 = ˜ut − η (I − ∆)−1 D ˜E(˜ut).

(8)

We can derive the following result (the proof is deferred to Section D.1 of the Appendix):

Lemma 5. Consider the updates in Equation 8, if ˜ut ∈ ΓWt then for all η ∈ (0,
˜ut+1 ∈ ΓkWt and the Barron norm of ˜ut+1 can be bounded as
∥˜ut+1∥B(Ω) ≤ (cid:0)1 + ηd(kWt)2B ˜L

(cid:1) ∥˜ut∥p

B(Ω).

λ3

(Cp+1)Λ3 ] we have

The proof consists of using the result in Equation 5 about the Barron norm of composition of a
function with ˜L, as well as counting the increase in the Barron norm of a function by any basic
algebraic operation, as established in Lemma 6. Precisely we show:
Lemma 6 (Barron norm algebra). If h, h1, h2 ∈ Γ, then the following set of results hold,

• Addition: ∥h1 + h2∥B(Ω) ≤ ∥h1∥B(Ω) + ∥h2∥B(Ω) .

• Multiplication: ∥h1 · h2∥B(Ω) ≤ ∥h1∥B(Ω)∥h2∥B(Ω)

• Derivative: if h ∈ ΓW for i ∈ [d] we have ∥∂ih∥B(Ω) ∈ W ∥h∥B(Ω).

8

Under review as a conference paper at ICLR 2023

• Preconditioning: if h ∈ Γ, then ∥(I − ∆)−1h∥B(Ω) ≤ ∥h∥B(Ω).

The proof for the above Lemma can be found in Appendix D.3. It bears similarity to an analogous
result in Chen et al. (2021), with the difference being that our bounds are defined in the spectral
Barron space which is different from the definition of the Barron norm used in Chen et al. (2021).

Expanding on the recurrence in Lemma 6 we therefore have that after T we have, we have WT ≤
kT W0 and hence ut+1 ∈ ΓktW0, and the Barron norm of uT can be bounded as follows,

∥uT ∥B(Ω) ≤ (cid:0)1 + ηdk2W 2

0 B ˜L

(cid:1)pT +1

∥u0∥pT

B(Ω).

(9)

Finally, we exhibit a natural class of functions that satisfy the main Barron growth property in
Equations 5. Precisely, we show (multivariate) polynomials of bounded degree have an effective
bound on p and BL:
Lemma 7. Let f (x) = (cid:80)
where α is a multi-index and x ∈ Rd. If g :
i
Rd → Rd is such that g ∈ ΓW , then we have f ◦ g ∈ ΓP W and the Barron norm can be bounded
as, ∥f ◦ g∥B(Ω) ≤ dP/2 (cid:16)(cid:80)
∥g∥P

α,|α|≤P |Aα|2(cid:17)1/2

i=1 xαi

α,|α|≤P

B(Ω).

Aα

(cid:81)d

(cid:17)

(cid:16)

, and r = P .

α,|α|≤P |Aα|2(cid:17)1/2

Hence if ˜L is a polynomial of degree P the constants in Assumption 1 will take the following values
B ˜L = dP/2 (cid:16)(cid:80)
Finally, since we are using an approximation of the function L we will incur an error at each step of
the iteration. The following Lemma shows that the error between the iterates ut and the approximate
iterates ˜ut increases with t. The error is calculated by recursively tracking the error between ut and
˜ut for each t in terms of the error at t − 1. Note that this error can be controlled by using smaller
values of η.
Lemma 8. Let ˜L : Rd → R be the function satisfying the properties in Assumption 1 such
that supx∈Rd ∥∇L(x) − ∇ ˜L(x)∥2 ≤ ϵL∥x∥2 and we have E(u) = (cid:82)
Ω L(∇u)dx and ˜E(u) =
(cid:82)

λ3

˜L(∇u)dx. For η ∈ (0,

Ω

(Cp+1)Λ3 ] consider the sequences,

then for all t ∈ N and R ≤ ∥u⋆∥H 1

ut+1 = ut − η(I − ∆)−1DE(ut), and ˜ut+1 = ˜ut − η(I − ∆)−1D ˜E(ut)
λ E(u0) we have,

0 (Ω) + 1

∥ut − ˜ut∥H 1

0 (Ω) ≤

ϵLηR
Λ + ϵL

(cid:0)(1 + η(Λ + ϵL))t − 1(cid:1)

7 CONCLUSION AND FUTURE WORK

In this work, we take a representational complexity perspective on neural networks, as they are used
to approximate solutions of nonlinear variational elliptic PDEs of the form −div(∇L(∇u)) = 0.
We prove that if L is such that composing L with function of bounded Barron norm increases the
Barron norm in a bounded fashion, then we can bound the Barron norm of the solution u⋆ to the
PDE—potentially evading the curse of dimensionality depending on the rate of this increase. Our
results subsume and vastly generalize prior work on the linear case (Marwah et al., 2021; Chen et al.,
2021). Our proof consists of neurally simulating preconditioned gradient descent on the energy
function defining the PDE, which we prove is strongly convex in an appropriate sense.

There are many potential avenues for future work. Our techniques (and prior techniques) strongly
rely on the existence of a variational principle characterizing the solution of the PDE. In classical
PDE literature, these classes of PDEs are also considered better behaved: e.g., proving regularity
bounds is much easier for such PDEs (Fern´andez-Real & Ros-Oton, 2020). There are many non-
linear PDEs that come without a variational formulation—e.g. the Monge-Ampere equation—for
which regularity estimates are derived using non-constructive methods like comparison principles.
It is a wide-open question to construct representational bounds for any interesting family of PDEs of
this kind. It is also a very interesting question to explore other notions of complexity—e.g. number
of parameters in a (potentially deep) network like in Marwah et al. (2021), Rademacher complexity,
among others.

9

Under review as a conference paper at ICLR 2023

REFERENCES

Andrew R Barron. Universal approximation bounds for superpositions of a sigmoidal function.

IEEE Transactions on Information theory, 39(3):930–945, 1993.

Johannes Brandstetter, Daniel Worrall, and Max Welling. Message passing neural PDE solvers.

arXiv preprint arXiv:2202.03376, 2022.

Johannes Martinus Burgers. The nonlinear diffusion equation: asymptotic solutions and statistical

problems. Springer Science & Business Media, 2013.

Ziang Chen, Jianfeng Lu, and Yulong Lu. On the representation of solutions to elliptic PDEs in

Barron spaces. Advances in Neural Information Processing Systems, 34, 2021.

Ziang Chen, Jianfeng Lu, Yulong Lu, and Shengxuan Zhou. A regularity theory for static
Schr¨odinger equations on Rd in spectral Barron spaces. arXiv preprint arXiv:2201.10072, 2022.

Miles Cranmer, Sam Greydanus, Stephan Hoyer, Peter Battaglia, David Spergel, and Shirley Ho.

Lagrangian neural networks. arXiv preprint arXiv:2003.04630, 2020.

Memoria di Ennio De Giorgi. Sulla differenziabilitae l’analiticita delle estremali degli integrali

multipli regolari. Ennio De Giorgi, pp. 167, 1957.

Weinan E and Bing Yu. The deep Ritz method: a deep learning-based numerical algorithm for

solving variational problems. arXiv preprint arXiv:1710.00211, 2017.

Weinan E, Jiequn Han, and Arnulf Jentzen. Deep learning-based numerical methods for high-
dimensional parabolic partial differential equations and backward stochastic differential equa-
tions. Communications in Mathematics and Statistics, 5(4):349–380, 2017.

Lawrence C Evans. Partial differential equations, volume 19. American Mathematical Soc., 2010.

Xavier Fern´andez-Real and Xavier Ros-Oton. Regularity theory for elliptic PDE. Forthcoming

book, 2020.

Philipp Grohs and Lukas Herrmann. Deep neural network approximation for high-dimensional

elliptic PDEs with boundary conditions. arXiv preprint arXiv:2007.05384, 2020.

Jiequn Han, Arnulf Jentzen, and Weinan E. Solving high-dimensional partial differential equations
using deep learning. Proceedings of the National Academy of Sciences, 115(34):8505–8510,
2018.

Jun-Ting Hsieh, Shengjia Zhao, Stephan Eismann, Lucia Mirabella, and Stefano Ermon. Learning

neural PDE solvers with convergence guarantees. arXiv preprint arXiv:1906.01200, 2019.

Martin Hutzenthaler, Arnulf Jentzen, Thomas Kruse, and Tuan Anh Nguyen. A proof that rectified
deep neural networks overcome the curse of dimensionality in the numerical approximation of
semilinear heat equations. SN partial differential equations and applications, 1(2):1–34, 2020.

Arnulf Jentzen, Diyora Salimova, and Timo Welti. A proof that deep artificial neural networks
overcome the curse of dimensionality in the numerical approximation of Kolmogorov partial
differential equations with constant diffusion and nonlinear drift coefficients. arXiv preprint
arXiv:1809.07321, 2018.

John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger,
Kathryn Tunyasuvunakool, Russ Bates, Augustin ˇZ´ıdek, Anna Potapenko, et al. Highly accurate
protein structure prediction with AlphaFold. Nature, 596(7873):583–589, 2021.

Yuehaw Khoo, Jianfeng Lu, and Lexing Ying. Solving parametric PDE problems with artificial

neural networks. European Journal of Applied Mathematics, 32(3):421–435, 2021.

Miglena N Koleva and Lubin G Vulkov. Numerical solution of the Monge-Amp`ere equation with
an application to fluid dynamics. In AIP Conference Proceedings, volume 2048, pp. 030002. AIP
Publishing LLC, 2018.

10

Under review as a conference paper at ICLR 2023

Holden Lee, Rong Ge, Tengyu Ma, Andrej Risteski, and Sanjeev Arora. On the ability of neural
nets to express distributions. In Conference on Learning Theory, pp. 1271–1296. PMLR, 2017.

Randall J LeVeque. Finite difference methods for ordinary and partial differential equations: steady-

state and time-dependent problems. SIAM, 2007.

Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, An-
drew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential
equations. arXiv preprint arXiv:2010.08895, 2020a.

Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, An-
drew Stuart, and Anima Anandkumar. Neural operator: Graph kernel network for partial differ-
ential equations. arXiv preprint arXiv:2003.03485, 2020b.

Jianfeng Lu and Yulong Lu. A priori generalization error analysis of two-layer neural networks for
solving high dimensional Schr¨odinger eigenvalue problems. arXiv preprint arXiv:2105.01228,
2021.

Jianfeng Lu, Yulong Lu, and Min Wang. A priori generalization analysis of the deep Ritz method

for solving high dimensional elliptic equations. arXiv preprint arXiv:2101.01708, 2021.

Tanya Marwah, Zachary Lipton, and Andrej Risteski. Parametric complexity bounds for approx-
imating PDEs with neural networks. Advances in Neural Information Processing Systems, 34,
2021.

Siddhartha Mishra and Roberto Molinaro. Estimates on the generalization error of physics informed

neural networks (PINNs) for approximating PDEs. arXiv preprint arXiv:2006.16144, 2020.

John Nash. Parabolic equations. Proceedings of the National Academy of Sciences, 43(8):754–758,

1957.

John Nash. Continuity of solutions of parabolic and elliptic equations. American Journal of Mathe-

matics, 80(4):931–954, 1958.

Lawrence E Payne and Hans F Weinberger. An optimal Poincar´e inequality for convex domains.

Archive for Rational Mechanics and Analysis, 5(1):286–292, 1960.

Henri Poincar´e. Sur les ´equations aux d´eriv´ees partielles de la physique math´ematique. American

Journal of Mathematics, pp. 211–294, 1890.

Maziar Raissi, Paris Perdikaris, and George Em Karniadakis.

Physics informed deep learn-
ing (Part I): Data-driven solutions of nonlinear partial differential equations. arXiv preprint
arXiv:1711.10561, 2017.

Michael Schmidt and Hod Lipson. Distilling free-form natural laws from experimental data. Science,

324(5923):81–85, 2009.

Justin Sirignano and Konstantinos Spiliopoulos. DGM: A deep learning algorithm for solving partial

differential equations. Journal of computational physics, 375:1339–1364, 2018.

Casper Kaae Sønderby, Lasse Espeholt, Jonathan Heek, Mostafa Dehghani, Avital Oliver, Tim Sal-
imans, Shreya Agrawal, Jason Hickey, and Nal Kalchbrenner. Metnet: A neural weather model
for precipitation forecasting. arXiv preprint arXiv:2003.12140, 2020.

Kathryn Tunyasuvunakool, Jonas Adler, Zachary Wu, Tim Green, Michal Zielinski, Augustin ˇZ´ıdek,
Alex Bridgland, Andrew Cowie, Clemens Meyer, Agata Laydon, et al. Highly accurate protein
structure prediction for the human proteome. Nature, 596(7873):590–596, 2021.

Hilary Weller, Philip Browne, Chris Budd, and Mike Cullen. Mesh adaptation on the sphere using
optimal transport and the numerical solution of a Monge–Amp`ere type equation. Journal of
Computational Physics, 308:102–123, 2016.

Dmitry Yarotsky. Error bounds for approximations with deep ReLU networks. Neural Networks,

94:103–114, 2017.

11

Under review as a conference paper at ICLR 2023

A APPENDIX

A.1 PROOF FOR LEMMA 1

The proof follows form Fern´andez-Real & Ros-Oton (2020) Chapter 3. We are provided it here for
completeness.

Proof of Lemma 1. If the function u⋆ minimizes the energy functional in Definition 1 then we have
for all ϵ ∈ R

E(u) ≤ E(u + ϵφ)

where φ ∈ C∞

c (Ω). That is, we have a minima at ϵ = 0 and taking a derivative w.r.t ϵ we get,

dE[u](φ) = lim
ϵ→0

= lim
ϵ→0

= lim
ϵ→0

E(u + ϵφ) − E(u)
ϵ

= 0

Ω L (∇u + ϵ∇φ) dx − (cid:82)
(cid:82)

ϵ

Ω L(∇u)dx

= 0

(cid:82)
Ω ϵ∇L(∇u)∇φ + ϵ2
ϵ

2 r(x)

= 0

where for all x ∈ Ω we have |r(x)| ≤ |∇φ|2 supp∈Rd D2L(p).
Since ϵ → 0 the final derivative is of the form,

dE[u](φ) =

(cid:90)

Ω

∇L(∇u)∇φdx = 0

(10)

For functions r, s ∈ H 1

0 (Ω) note the following Green’s formula,
(cid:90)
(cid:90)

(cid:90)

(∇r)sdx = −

∇r · ∇sdx +

∂u
∂n

vdΓ

∂Ω

Ω

=⇒

(cid:90)

Ω

Ω

(∇r)sdx = −

(cid:90)

Ω

∇r · ∇sdx

(11)

Using the identity in Equation 11 in Equation 10 we get,

dE[u](φ) =

(cid:90)

Ω

−div (∇L(∇u)) φdx = 0

That is the minima for the energy functional is reached at a u which solves the following PDE,

dE(u) = −div (∇L(∇u)) = 0.

B PROOFS FROM SECTION 6.1

B.1 PROOF FOR LEMMA 2

Proof. In order to prove part 1, we will use the following integration by parts identity, for functions
r : Ω → R such that and s : Ω → R, and r, s ∈ H 1

0 (Ω),

(cid:90)

Ω

∂r
∂xi

sdx = −

(cid:90)

Ω

r

∂s
∂xi

(cid:90)

dx +

∂Ω

rsndΓ

(12)

where ni is a normal at the boundary and dΓ is an infinitesimal element of the boundary ∂Ω.

12

Under review as a conference paper at ICLR 2023

Using the formula in Equation 12 for functions u, v ∈ H 1

0 (Ω), we have

⟨DE(u), v⟩L2(Ω) = ⟨−∇ · ∇L(∇u), v⟩L2(Ω)
(cid:90)

= −

∇ · ∇L(∇u)vdx

Ω

(cid:90)

d
(cid:88)

Ω

i=1

= −

∂ (∇L(∇u))i
∂xi

vdx

(∇L(∇u))i

∂v
∂xi

dx +

(cid:90)

d
(cid:88)

Ω

i=1

(∇L(∇u))i vnidx

(cid:90)

d
(cid:88)

i=1

Ω

(cid:90)

=

=

∇L(∇u) · ∇vdx

where in the last equality we use the fact that the function v ∈ H 1
To show the second part, first note since L : Rd → R is strongly convex and smooth, we have

0 (Ω), thus v(x) = 0, ∀x ∈ ∂Ω.

(13)

(14)

∀x, y ∈ Rd, λ∥x − y∥2 ≤ ∥∇L(x) − ∇L(y)∥2 ≤ Λ∥x − y∥2.

This implies

∀x ∈ Ω, ∥∇L(∇u(x)) − ∇L(∇v(x))∥2 ≤ Λ∥∇u(x) − ∇v(x)∥2

Taking square on each side and itegrating over Ω we get

(cid:90)

Ω

∥∇L(∇u(x)) − ∇L(∇v(x))∥2

2dx ≤ Λ2

(cid:90)

Ω

∥∇u(x) − ∇v(x)∥2

2 dx

=⇒ ∥∇L(∇u) − ∇L(∇v)∥L2(Ω) ≤ Λ∥∇u − ∇v∥L2(Ω)

On the other hand, from part 1, we have

⟨DE(u) − DE(v), u − v⟩L2(Ω) = ⟨∇L(∇u) − ∇L(∇v), ∇u − ∇v⟩L2(Ω)

Hence, by Cauchy-Schwartz, we get

⟨DE(u) − DE(v), u − v⟩L2(Ω) = ⟨∇L(∇u) − ∇L(∇v), ∇u − ∇v⟩L2(Ω)

≤ ∥∇L(∇u) − ∇L(∇v)∥L2(Ω)∥∇u − ∇v∥L2(Ω)
≤ Λ∥∇u − ∇v∥2

L2(Ω)

which proves the right hand side of the inequality in part 2.
For the left size of the inequality, by convexity of L we have ∀x, y ∈ Rd, (∇L(x) − ∇L(y))T (x −
y) ≥ λ∥x − y∥2

2 is convex,

(∇L (∇u(x)) − ∇L (∇v(x)))T (∇u(x) − ∇v(x)) ≥ λ∥∇u(x) − ∇v(x)∥2
2

Integrating over Ω we get,

(cid:90)

Ω

(∇L (∇u(x)) − ∇L (∇v(x)))T (∇u(x) − ∇v(x))dx ≥ λ

(cid:90)

Ω

∥∇u(x) − ∇v(x)∥2

2dx

=⇒ ⟨∇L(∇u) − ∇L(∇v), ∇u − ∇v⟩L2(Ω) ≥ λ∥∇u − ∇v∥2

L2(Ω)

(15)

Using part 1 again, this implies Equation 15

⟨DE(u) − DE(v), u − v⟩L2(Ω) = ⟨∇L(∇u) − ∇L(∇v), ∇u − ∇v⟩L2(Ω)

≥ λ∥∇v − ∇v∥2

L2(Ω).

as we wanted.

13

Under review as a conference paper at ICLR 2023

To show part 3, we first Taylor expand L to rewrite the energy function as:

E(u + v) =

=

(cid:90)

Ω

(cid:90)

Ω

L(∇u(x) + ∇v(x))dx

(cid:18)

L(∇u(x)) + ∇L(∇u(x))∇v(x) +

D2L(˜x)∥∇v(x)∥2
2

(cid:19)

dx

1
2

(16)

where ˜x ∈ Rd (and is potentially different for every x ∈ Ω). Since the function L is strongly convex
we have

λId ≤ D2L(˜x) ≤ ΛId.

Plugging in these bounds in Equation 16, we have:

E(u + v) ≤

(cid:90)

(cid:18)

Ω

L(∇u(x)) + ∇L(∇u(x))∇v(x) +

∥∇v(x)∥2

(cid:19)

dx

Λ
2

=⇒ E(u + v) ≤ E(u) + ⟨DE(u), v⟩L2(Ω) +

Λ
2

⟨∇v, ∇v⟩L2(Ω).

as well as

E(u + v) ≥

(cid:90)

(cid:18)

Ω

L(∇u(x)) + ∇L(∇u(x))∇v(x) +

∥∇v(x)∥2

(cid:19)

dx

Λ
2

=⇒ E(u + v) ≥ E(u) + ⟨DE(u), v⟩L2(Ω) +

λ
2

⟨∇v, ∇v⟩L2(Ω).

(17)

(18)

Combining Equation 17 and Equation 18 we get,

λ
2

∥∇v∥2

L2(Ω) + ⟨DE(u), v⟩L2(Ω) ≤ E(u + v) − E(u) ≤ ⟨DE(u), v⟩L2(Ω) +

Λ
2

∥∇v∥2

L2(Ω)

Finally, part 4 follows by plugging in u = u⋆ and v = u − u⋆ in part 3 and using the fact that
DE(u⋆) = 0.

B.2 PROOF FOR LEMMA 3

Proof. Let {λi, ϕi}∞
λ1 ≤ λ2 ≤ · · · , which are real and countable. ( Evans (2010), Theorem 1, Section 6.5)

i=1 denote the (eigenvalue, eigenfunction) pairs of the operator −∆ where 0 <

Using the definition of eigenvalues and eigenfunctions, we have

⟨−∆v, v⟩L2(Ω)

∥v∥2

L2(Ω)

⟨∇v, ∇v⟩L2(Ω)

∥v∥2

L2(Ω)

λ1 = inf
v∈H 1

0 (Ω)

= inf
v∈H 1

0 (Ω)

=

1
Cp

.

where in the last equality we use Theorem 2.
Let us write the functions v, w in the eigenbasis as v = (cid:80)
−∆ is also an eigenfunction for (I − ∆)−1, with correspondinding eigenvalue

i µiϕi. Notice that an eigenfunction of

1
1+λi

.

14

Under review as a conference paper at ICLR 2023

Thus, to show part 1, we have,

∞
(cid:88)

(cid:13)(I − ∆)−1∆v(cid:13)
L2(Ω) = (cid:13)
(cid:13)(I − ∆)−1∇ · ∇v(cid:13)
(cid:13)
2
2
(cid:13)
(cid:13)
L2(Ω)
(cid:13)
(cid:13)
2
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
∞
(cid:88)

λi
1 + λi

(cid:13)
2
(cid:13)
(cid:13)
(cid:13)
(cid:13)

∞
(cid:88)

µiϕi

µiϕi

L2(Ω)

i=1

i=1

≤

=

=

i = ∥u∥2
µ2

L2(Ω)

L2(Ω)

i=1

where in the last equality we use the fact that ϕi are orthogonal.
Now, note that (I − ∆)−1v = (cid:80)∞
i=1
monotonically increasing, we have for all i ∈ N

1

(1+λi) µiϕi now, note that since λ1 ≤ λ2 ≤ · · · and x

1+x is

1
1 + λi

≥

1
(1 + Cp)λi

and note that 1
λi

are the eigenvalues for (−∆)−1 for all i ∈ N.

Now, bounding ⟨(I − ∆)−1v, v⟩L2(Ω)

⟨(I − ∆)−1v, v⟩L2(Ω) =

(cid:42) ∞
(cid:88)

i=1

µi
1 + λi

ϕi,

∞
(cid:88)

i=1

(cid:43)

µiϕi

L2(Ω)

=

∞
(cid:88)

i=1

µ2
i
1 + λi

∥ϕi∥2

L2(Ω)

(19)

(20)

where we the orthogonality of ϕ′
further lower bound ⟨(I − ∆)−1v, v⟩L2(Ω) as follows,

is to get Equation 20. Using the inequality in Equation 19 we can

⟨(I − ∆)−1v, v⟩L2(Ω) ≥

∞
(cid:88)

i=1

µ2
i
(1 + Cp)λi

∥ϕi∥2

L2(Ω) :=

1
1 + Cp

⟨(−∆)−1v, v⟩L2(Ω),

where we use the following set of equalities in the last step,
(cid:43)

⟨(−∆)−1v, v⟩L2(Ω) =

(cid:42) ∞
(cid:88)

∞
(cid:88)

ϕi,

µiϕi

µi
λi

i=1

i=1

L2(Ω)

=

∞
(cid:88)

i=1

µ2
i
λi

∥ϕi∥2

L2(Ω).

B.3 PROOF FOR CONVERGENCE: PROOF FOR LEMMA 4

Proof. For the analysis we consider η =

λ3
(1+Cp)Λ3

Taylor expanding as in Equation 17, we have

E(ut+1) ≤ E(ut) − η (cid:10)∇L(∇ut), ∇(I − ∆)−1DE(ut)(cid:11)

(cid:124)

(cid:123)(cid:122)
Term 1

+

η2Λ2
2

(cid:124)

(cid:13)
(cid:13)∇(I − ∆)−1DE(ut)(cid:13)
2
(cid:13)
L2(Ω)
(cid:125)
(cid:123)(cid:122)
Term 2

.

L2(Ω)
(cid:125)

(21)

where we have in Equation 17 plugged in ut+1 − ut = −η (I − ∆)−1 DE(ut).
First we lower bound Term 1. Since u⋆ is the solution to the PDE in Equation 3, we have DE(u⋆) =
0. Therefore we have
(cid:10)∇L(∇ut), ∇(I − ∆)−1DE(ut)(cid:11)

L2(Ω) = (cid:10)∇L(∇ut), ∇(I − ∆)−1 (DE(ut) − DE(u⋆))(cid:11)

L2(Ω)
(22)

15

Under review as a conference paper at ICLR 2023

Similarly, since u⋆ is the solution to the PDE in Equation 3 Equation 2 we have for all φ ∈ H 1
⟨∇L(∇u⋆), ∇φ⟩L2(Ω) = 0. Using this Equation 22 we get,

0 (Ω)

(cid:10)∇L(∇ut), ∇(I − ∆)−1DE(ut)(cid:11)

L2(Ω)

= (cid:10)∇L(∇ut), ∇(I − ∆)−1 (DE(ut) − DE(u⋆))(cid:11)
= (cid:10)∇L(∇ut), ∇(I − ∆)−1 (DE(ut) − DE(u⋆))(cid:11)
= (cid:10)∇L(∇ut) − ∇L(∇u⋆), ∇(I − ∆)−1 (DE(ut) − DE(u⋆))(cid:11)

L2(Ω)

L2(Ω) + (cid:10)∇L(∇u⋆), ∇(I − ∆)−1 (DE(ut) − DE(u⋆))(cid:11)

L2(Ω)

L2(Ω)

(23)

Using Equation 23, we can rewrite Term 1 as
(cid:10)∇L(∇ut), ∇(I − ∆)−1DE(ut)(cid:11)
= (cid:10)∇L(∇ut) − ∇L(∇u⋆), ∇(I − ∆)−1 (DE(ut) − DE(u⋆))(cid:11)

L2(Ω)

L2(Ω)

=

(i)
=

=

(cid:90)

Ω
(cid:90)

Ω

(cid:90)

Ω

(∇L(∇ut) − ∇L(∇u⋆)) · ∇(I − ∆)−1 (−∇ · (∇L(∇ut)) − ∇L(u⋆)) dx

(∇L(∇ut) − ∇L(∇u⋆)) · ∇(I − ∆)−1 (−∆ (L(∇ut) − L(u⋆))) dx

(∇L(∇ut) − ∇L(∇u⋆)) · (I − ∆)−1(−∆) (∇L(∇ut) − ∇L(u⋆)) dx

(24)

where in step (i) we use −∇ · ∇v = −∆v and the fact that ∇ commutes with (I − ∆)−1(−∆).

Plugging part 2 of Lemma 3 in Equation 24, we get
(cid:10)∇L(∇ut), ∇(I − ∆)−1DE(ut)(cid:11)

L2(Ω)

(∇L(∇ut) − ∇L(∇u⋆)) · (−∆)−1(−∆) (∇L(∇ut) − ∇L(∇u⋆)) dx

⟨∇L(∇ut) − ∇L(∇u⋆), ∇L(∇ut) − ∇L(∇u⋆)⟩L2(Ω)

≥

≥

(i)
≥

≥

(cid:90)

Ω

1
1 + Cp
1
1 + Cp
λ2
1 + Cp
2λ2
(1 + Cp)Λ

∥∇ut − ∇u⋆∥2

L2(Ω)

(E(ut) − E(u⋆))

(25)

where (i) follows by part 2 of Lemma 2 and we use part 4 of Lemma 2 for the last inequality.

We will proceed to upper bounding Term 2. Using part 1 of Lemma 3 we have

(cid:13)
L2(Ω) = (cid:13)
(cid:13)∇(I − ∆)−1DE(ut)(cid:13)
2
(cid:13)
= (cid:13)

(cid:13)∇(I − ∆)−1 (DE(ut) − DE(u⋆))(cid:13)
2
(cid:13)
L2(Ω)
(cid:13)∇(I − ∆)−1 (−∇ · (∇L(∇ut) − ∇L(∇u⋆)))(cid:13)
2
(cid:13)
L2(Ω)
(cid:13)∇(I − ∆)−1(−∆) (L(∇ut) − L(∇u⋆))(cid:13)
2
(cid:13)
L2(Ω)
(cid:13)(I − ∆)−1(−∆) (∇L(∇ut) − ∇L(∇u⋆))(cid:13)
2
(cid:13)
L2(Ω)

(i)

= (cid:13)

(ii)

= (cid:13)

(iii)
≤ ∥∇L(∇ut) − ∇L(∇u⋆)∥2
≤ Λ2 ∥∇ut − ∇u⋆∥2

L2(Ω)

L2(Ω)

≤

2Λ2
λ

(E(ut) − E(u⋆)) .

(26)

Here, we use the fact that −∇ · ∇ = −∆ in step (i) and the fact that ∇ commutes with (I −
∆)−1(−∆) in step (ii), and finally we use the result from part 2 of Lemma 3 to get the inequality
in (iii).

16

Under review as a conference paper at ICLR 2023

Combining Equation 25 and Equation 26 in Equation 21 we get

=⇒ E(ut+1) − E(u⋆) ≤ E(ut) − E(u⋆) −

(cid:18)

2λ2
(1 + Cp)Λ

− η

(cid:19)

Λ2
λ

η (E(ut) − E(u⋆))

Since η = λ3/((1 + Cp)Λ3) we have

E(ut+1) − E(u⋆) ≤ E(ut) − E(u⋆) −

=⇒ E(ut+1) − E(u⋆) ≤

(cid:18)

1 −

C ERROR ANALYSIS

λ5
(1 + Cp)Λ4

λ5

(1 + Cp)Λ4 η (E(ut) − E(u⋆))
(cid:19)t

(E(u0) − E(u⋆)) .

First, we will need the following simple technical lemma showing that the H 1
Lemma 9. The dual norm of ∥ · ∥H 1

0 (Ω) is ∥ · ∥H 1

0 (Ω).

0 (Ω) norm is self-dual:

Proof. If ∥u∥∗ denotes the dual norm of ∥u∥H 1

0 (Ω), by definition we have,
⟨u, v⟩H 1

0 (Ω)

⟨∇u, ∇v⟩L2(Ω)

∥∇u∥L2(Ω)∥∇v∥L2(Ω)

∥u∥∗ =

=

≤

sup
v∈H 1
∥v∥H1

0 (Ω)
0 (Ω)=1
sup
v∈H 1
∥v∥H1

0 (Ω)
0 (Ω)=1
sup
v∈H 1
∥v∥H1

0 (Ω)
0 (Ω)=1
= ∥∇u∥L2(Ω)

where the inequality follows by Cauchy- Schwarz. On the other hand, equality can be achieved by
taking v = u

. Thus, ∥u∥∗ = ∥∇u∥L2(Ω) = ∥u∥H 1

0 (Ω) as we wanted.

∥∇u∥2

With this we can prooceed to the proof of Lemma 8.

C.1 PROOF FOR LEMMA 8

Proof. We define for all t rt = ˜ut − ut, and will iteratively bound ∥rt∥L2(Ω).

Starting with u0 = 0 and ˜ut = 0, we define the iterative sequences as,
(cid:26)u0 = u0

ut+1 = ut − η(I − ∆)−1DE(ut)

(cid:26)˜ut = u0

˜ut+1 = ˜ut − η(I − ∆)−1D ˜E(˜ut)

where η ∈ (0,

λ3

(1+Cp)Λ3 ]. Subtracting the two we get,

˜ut+1 − ut+1 = ˜ut − ut − η(I − ∆)−1 (cid:16)
=⇒ rt+1 = rt − η(I − ∆)−1 (cid:16)

(cid:17)
D ˜E(˜ut) − DE(ut)
(cid:17)
D ˜E(ut + rt) − DE(ut)

Taking H 1

0 (Ω) norm on both sides we get,
(cid:13)
(cid:13)

0 (Ω) ≤ ∥rt∥H 1

0 (Ω) + η

∥rt+1∥H 1

(cid:13)(I − ∆)−1 (cid:16)

D ˜E(ut + rt) − DE(ut)

(cid:17)(cid:13)
(cid:13)
(cid:13)H 1

0 (Ω)

17

(27)

(28)

Under review as a conference paper at ICLR 2023

, from Lemma 9 we know that the

Towards bounding

dual norm of ∥w∥H 1

(cid:13)
(cid:13)
(cid:13)(I − ∆)−1D ˜E(ut + rt) − DE(ut)
(cid:13)
(cid:13)
(cid:13)H 1
0 (Ω) is ∥w∥H 1
0 (Ω), thus,
(cid:13)
(cid:13)
(cid:13)(I − ∆)−1D ˜E(ut + rt) − DE(ut)
(cid:13)
(cid:13)
(cid:13)H 1

0 (Ω)

∇(I − ∆)−1 (cid:16)
(cid:68)

sup
φ∈H 1
∥φ∥H1

0 (Ω)
0 (Ω)=1

0 (Ω)
(cid:17)
D ˜E(ut + rt) − DE(ut)

(cid:69)

, ∇φ

L2(Ω)

∇(I − ∆)−1 (cid:16)
(cid:68)

(cid:17)
D ˜E(ut + rt) − DE(ut + rt)

, ∇φ

(cid:69)

L2(Ω)

=

=

=

≤

0 (Ω)
0 (Ω)=1
(cid:68)

sup
φ∈H 1
∥φ∥H1

+

0 (Ω)
0 (Ω)=1
sup
φ∈H 1
∥φ∥H1

sup
φ∈H 1
∥φ∥H1

+

0 (Ω)
0 (Ω)=1
sup
φ∈H 1
∥φ∥H1

sup
φ∈H 1
∥φ∥H1

+

0 (Ω)
0 (Ω)=1
sup
φ∈H 1
∥φ∥H1

0 (Ω)
0 (Ω)=1

(cid:10)∇(I − ∆)−1 (DE(ut + rt) − DE(ut)) , ∇φ(cid:11)

L2(Ω)

∇(I − ∆)−1∇ ·

(cid:16)

(cid:17)
∇ ˜L(∇ut + ∇rt) − ∇L(∇ut + ∇rt)

(cid:69)

, ∇φ

L2(Ω)

(cid:10)∇(I − ∆)−1∇ · (∇L(∇ut + ∇rt)) − ∇L(∇ut), ∇φ(cid:11)

L2(Ω)

0 (Ω)
0 (Ω)=1
(cid:69)
(cid:68)
∇ ˜L(∇ut + ∇rt) − ∇L(∇ut + ∇rt), ∇φ

L2(Ω)

⟨∇L(∇ut + ∇rt) − ∇L(∇ut), ∇φ⟩L2(Ω)

≤ ϵL∥∇ut + ∇rt∥L2(Ω) + ∥∇L(∇ut + ∇rt) − ∇L(∇ut)∥L2(Ω)

(29)

Using the Lipschitzness of ∇L, we have

∥∇L(∇ut(x) + ∇rt(x)) − ∇L(∇ut)∥2 ≤ ∥∇L(∇ut(x)) − ∇L(∇ut(x))∥2 + sup
p∈Rd

D2F (p)∥∇rt(x)∥2

Squaring and integrating over Ω on both sides we get

≤ Λ∥∇rt(x)∥2

∥∇L(∇ut(x) + ∇rt(x)) − ∇L(∇ut)∥L2(Ω) ≤ Λ∥∇rt∥L2(Ω)

(30)

(31)

Pluggin in Equation 31 in Equation 29 we get,
(cid:13)
(cid:13)
(cid:13)(I − ∆)−1D ˜E(ut + rt) − DE(ut)
(cid:13)
(cid:13)
(cid:13)H 1

0 (Ω)

≤ (Λ + ϵL)∥∇rt∥L2(Ω) + ϵL∥∇ut∥L2(Ω)

(32)

Furthermore, from Lemma 4 we have for all t ∈ N,
(cid:18)

E(ut) − E(u⋆) ≤

1 −

(cid:19)t

λ5
CpΛ4

E(u0)

and

≤ E(u0)

∥ut − u⋆∥H 1

0 (Ω) ≤

≤

2
λ
2
λ

(E(ut) − E(u0))

E(u0)

Hence we have that for all t ∈ N,

∥ut∥H 1

0 (Ω) ≤ ∥u⋆∥H 1

0 (Ω) +

2
λ

E(u0) =: R.

18

Under review as a conference paper at ICLR 2023

Putting this all together, we have

(cid:13)
(cid:13)(I − ∆)−1D ˜E(ut + rt) − DE(ut)
(cid:13)

(cid:13)
(cid:13)
(cid:13)H 1

0 (Ω)

≤ (Λ + ϵL)∥∇rt∥L2(Ω) + ϵLR

(33)

Hence using the result from Equation 33 in Equation 28 to get,

∥rt+1∥H 1

0 (Ω) ≤ (1 + η(Λ + ϵL)) ∥rt∥H 1

0 (Ω) + ϵLηR

=⇒ ∥rt+1∥H 1

0 (Ω) ≤

ϵLηR
Λ + ϵL

(cid:0)(1 + η(Λ + ϵL))t − 1(cid:1)

(34)

where we use the fact that ∥rt∥H 1

0 (Ω) = 0.

Notice that ϵL << Λ. Further, we have η ∈ (0,
1.

λ3

(Cp+1)Λ3 ] that is η ≤ 1, it implies that η(Λ + ϵL) <

Hence we can further bound Equation 34 as follows,

∥rt+1∥H 1

0 (Ω) ≤

ϵLηR
Λ + ϵL

(cid:0)(1 + η(Λ + ϵL))t − 1(cid:1)

D PROOFS FOR BARRON NORM APPROXIMATION: SECTION 6.2

D.1 PROOF FOR LEMMA 5: BARRON NORM RECURSION

Proof. Note that the update equation looks like,

ut+1 = ut − η(I − ∆)−1DE(ut)

= ut − η(I − ∆)−1 (−∇ · ∇L(∇ut))

= ut − η(I − ∆)−1

−

(cid:32)

(cid:33)

∂2
i L(∇ut)

d
(cid:88)

i=1

∥∇ut∥B(Ω) = max
i∈[d]

∥∂iut∥B(Ω) ≤ Wt∥ut∥B(Ω)

(35)

(36)

From Lemma 6 we have

Note that since ut ∈ ΓWt we have ∇ut ∈ ΓWt and L(∇ut) ∈ ΓkWt (from Assumption 1).
Therefore, we can bound the Barron norm as,

(cid:13)
(cid:13)
(I − ∆)−1
(cid:13)
(cid:13)
(cid:13)

(cid:32)

−

d
(cid:88)

i=1

∂2
i L(∇ut)

(cid:33)(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)B(Ω)

(i)
≤

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

−

d
(cid:88)

i=1

(cid:13)
(cid:13)
∂2
(cid:13)
i L(∇ut)
(cid:13)
(cid:13)B(Ω)

(ii)

(cid:13)∂2

(cid:13)B(Ω)

≤ d (cid:13)
i L(∇ut)(cid:13)
≤ d(kWt)2∥L(∇ut)∥B(Ω)
≤ d(kWt)2B ˜L∥ut∥r

B(Ω)

where we use the fact that for a function h, we have ∥(I − ∆)−1h∥B(Ω) ≤ ∥h∥B(Ω) from Lemma 6
in (i) and the fact that L(∇ut) ∈ ΓkWt in (ii). Using the result of Addition from Lemma 6 we have

∥ut∥B(Ω) ≤ ∥ut∥B(Ω) + (cid:0)ηd(kWt)2B ˜L
(cid:1) ∥ut∥r
≤ (cid:0)1 + ηd(kWt)2B ˜L

(cid:1) ∥ut∥r
B(Ω).

B(Ω)

19

Under review as a conference paper at ICLR 2023

D.2 PROOF FOR BARRON NORM OF POLYNOMIAL

Lemma 10 ( Lemma 7 restated). Let

f (x) =

(cid:88)

α,|α|≤P

(cid:32)

Aα

(cid:33)

d
(cid:89)

i=1

xαi
i

where α is a multi-index and x ∈ Rd and Aα ∈ R is a scalar. If g : Rd → Rd is a function such
that g ∈ ΓW , then we have f ◦ g ∈ ΓP W and the Barron norm can be bounded as,

∥f ◦ g∥B(Ω) ≤ dP/2





(cid:88)

α,|α|≤P



1/2

|Aα|2



∥g∥P

B(Ω).

Proof. Recall from Definition 7 we know that for a vector valued function g : Rd → Rd, we have

∥g∥B(Ω) = max
i∈[d]

∥gi∥B(Ω).

Then, using Lemma 6, we have
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

∥f (g)∥B(Ω) =

P
(cid:88)

Aα

P
(cid:88)

α,|α|=0
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

α,|α|=0

Aα

P
(cid:88)

|Aα|

α,|α|=0

P
(cid:88)

α,|α|=0

P
(cid:88)

α,|α|=0

P
(cid:88)

|Aα|

|Aα|

|Aα|

≤

≤

≤

≤

≤

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)B(Ω)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)B(Ω)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)B(Ω)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)B(Ω)

gαi
i

gαi
i

d
(cid:89)

i=1

d
(cid:89)

gαi
i

gαi
i

i=1

d
(cid:89)

i=1
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
i=1
(cid:32) d
(cid:89)

d
(cid:89)

i=1
(cid:32) d
(cid:89)

i=1

(cid:33)

∥gαi

i ∥B(Ω)

(cid:33)

∥gi∥αi

B(Ω)

α,|α|=0


P
(cid:88)

≤





1/2 

|Aα|2





P
(cid:88)

(cid:32) d
(cid:89)

1/2

(cid:33)2


∥gi∥αi

B(Ω)

(37)

where we have repeatedly used Lemma 6 and Cauchy-Schwartz in the last line. Using the fact that
for a multivariate function g : Rd → Rd we have for all i ∈ [d]

α,|α|=0

α,|α|=0

i=1

Therefore, from Equation 37 we get,

∥g∥B(Ω) ≥ ∥gi∥B(Ω).

∥f (g)∥B(Ω) ≤





(cid:88)



1/2 

|Aα|2





(cid:88)

(cid:16)

α,|α|≤P

α,|α|≤P

∥g∥

(cid:80)d

i=1 αi

B(Ω)

(cid:17)2



1/2





≤



(cid:88)



1/2 

|Aα|2





(cid:88)

(cid:16)

(cid:17)2

∥g∥α

B(Ω)



1/2



α,|α|≤P


α,|α|≤P



1/2

≤ dP/2



(cid:88)

|Aα|2



∥g∥P

B(Ω)

α,|α|≤P

20

Under review as a conference paper at ICLR 2023

Since the maximum power of the polynomial can take is P from Corollary 1 we will have f ◦ g ∈
ΓP W .

D.3 PROOF FOR BARRON NORM ALGEBRA:LEMMA 6

The proof of Lemma 6 is fairly similar to the proof of Lemma 3.3 in Chen et al. (2021)—the change
stemming from the difference of the Barron norm being considered

Proof. We first show the result for Addition and bound ∥h1 + h2∥B(Ω),

∥h1 + h2∥B(Ω) =

inf
g1|Ω=h1,g1∈F
g2|Ω=h2,g2∈F

=

inf
g1|Ω=h1,g1∈F
g2|Ω=h2,g2∈F

≤

inf
g1|Ω=h1,g1∈F

(cid:90)

Rd

(cid:90)

Rd

(cid:90)

Rd

(1 + ∥ω∥2) | (cid:92)g1 + g2(ω)|dω

(1 + ∥ω∥2) |ˆg1(ω) + ˆg2(ω)|dω

(1 + ∥ω∥2) |ˆg1(ω)|dω +

inf
g2|Ω=h2,g2∈F

(cid:90)

Rd

(1 + ∥ω∥2) |ˆg2(ω)|dω

=⇒ ∥h1 + h2∥B(Ω) ≤ ∥h1∥B(Ω) + ∥h2∥B(Ω).

For Multiplication, first note that multiplication of functions is equal to convolution of the functions
in the frequency domain, i.e., for functions g1 : Rd → d and g2 : Rd → d, we have,

(cid:92)g1 · g2 = ˆg1 ∗ ˆg2

(38)

Now, to bound the Barron norm for the multiplication of two functions,

∥h1 · h2∥B(Ω) =

inf
g1|Ω=h1,g1∈F
g2|Ω=h2,g2∈F

=

inf
g1|Ω=h1,g1∈F
g2|Ω=h2,g2∈F

=

inf
g1|Ω=h1,g1∈F
g2|Ω=h2,g2∈F

≤

inf
g1|Ω=h1,g1∈F
g2|Ω=h2,g2∈F

(cid:90)

Rd

(cid:90)

Rd

(1 + ∥ω∥2)|(cid:92)g1 · g2(ω)|dω

(1 + ∥ω∥2)|ˆg1 ∗ ˆg2(ω)|dω

(cid:90)

(cid:90)

ω∈Rd

z∈Rd

(cid:90)

(cid:90)

ω∈Rd

z∈Rd

(1 + ∥ω∥2) |ˆg1(z)ˆg2(ω − z)| dωdz

(1 + ∥ω − z∥2 + ∥z∥2 + ∥z∥2∥ω − z∥2) |ˆg1(z)ˆg2(ω − z)| dωdz

Where we use ∥ω∥2 ≤ ∥ω − z∥2 + ∥z∥2 and the fact that
(cid:90)

(cid:90)

∥z∥2∥ω − z∥2|ˆg1(z)ˆg2(ω − z)|dωdz > 0.

ω

z

Collecting the relevant terms together we get,

∥h1 · h2∥B(Ω) ≤

=

inf
g1|Ω=h1,g1∈F
g2|Ω=h2,g2∈F
inf
g1|Ω=h1,g1∈F
g2|Ω=h2,g2∈F

(cid:90)

(cid:90)

ω∈Rd

z∈Rd

(1 + ∥ω − z∥2) · (1 + ∥z∥2) |ˆg1(z)| |ˆg2(ω − z)| dω

((1 + ∥ω∥2)ˆg1(ω)) ∗ ((1 + ∥ω∥2)ˆg2(ω))

Hence using Young’s convolution identity from Lemma 11 we have

∥h1 · h2∥B(Ω) ≤

inf
g1|Ω=h1,g1∈F
g2|Ω=h2,g2∈F

(cid:18)(cid:90)

ω∈Rd

(1 + ∥w∥2)ˆg1(ω)dω

(cid:19) (cid:18)(cid:90)

ω∈Rd

(1 + ∥w∥2)ˆg2(ω)dω

(cid:19)

=⇒ ∥h1 · h2∥B(Ω) ≤ ∥h1∥B(Ω)∥h2∥B(Ω).

21

In order to show the bound for Derivative, since h ∈ ΓW , there exists a function g : Rd → R such
that,

Under review as a conference paper at ICLR 2023

(cid:90)

g(x) =

eiωT xˆg(ω)dω

ieiωT xωj ˆg(ω)

Now taking derivative on both sides we get,
(cid:90)

∂jg(x) =

∥ω∥∞≤W

This implies that we can upper bound | (cid:99)∂ig(ω) as

∥ω∥∞≤W

(cid:100)∂jg(ω) = iωj ˆg(ω)

=⇒ |(cid:100)∂jg(ω)| ≤ W |ˆg(ω)|

Hence we can bound the Barron norm of ∂jh as follows:

(39)

(40)

∥∂jh∥B(Ω) =

inf
g|Ω=h,g∈FW

≤

inf
g|Ω=h,g∈FW

(cid:90)

∥ω∥∞≤W

(cid:90)

(1 + ∥ω∥∞) |(cid:100)∂jg(ω)|dω

(1 + ∥ω∥∞)|W ˆg(ω)|dω

∥ω∥∞≤W

(cid:90)

∥ω∥∞≤W

≤ W

inf
g|Ω=h,g∈FW

≤ W ∥h∥B(Ω)

(1 + ∥ω∥∞)|ˆg(ω)|dω

In order to show the preconditioning, note that for a function g : Rd → R, if f = (I − ∆)−1g then
we have then we have (I − ∆)f = g. Using the result form Lemma 12 we have

(1 + ∥ω∥2

2) ˆf (ω) = ˆg(ω) =⇒ ˆf (ω) =

ˆg(ω)
1 + ∥ω∥2
2

.

Bounding ∥(I − ∆)−1h∥B(Ω),

∥(I − ∆)−1h∥B(Ω) =

inf
g|Ω=h,g∈F

inf
g|Ω=h,g∈F
=⇒ ∥(I − ∆)−1h∥B(Ω) ≤ ∥h∥B(Ω).

≤

(cid:90)

ω∈Rd

(cid:90)

ω∈Rd

1 + ∥ω∥2
(1 + ∥ω∥2
2)

ˆg(ω)dω

(1 + ∥ω∥2)ˆg(ω)dω

Corollary 1. Let g : Rd → R then for any k ∈ N we have ∥gk∥B(Ω) ≤ ∥g∥k
the function g ∈ FW then the function gk ∈ ΓkW .

B(Ω). Furthermore, if

Proof. The result from ∥gk∥B(Ω) follows from the multiplication result in Lemma 6 and we can
show this by induction. For n = 2, we have from Lemma 6 we have,

∥g2∥B(Ω) ≤ ∥g∥2

B(Ω)

Assuming that we have for all n till k − 1 we have

∥gn∥B(Ω) ≤ ∥g∥n

B(Ω)

for n = k we get,

∥gk∥B(Ω) = ∥ggk−1∥B(Ω) ≤ ∥g∥B(Ω)∥gk−1∥B(Ω) ≤ ∥g∥k

B(Ω).

To show that for any k the function gk ∈ ΓkW , we write gk in the Fourier basis. We have:

22

(41)

(42)

(43)

Under review as a conference paper at ICLR 2023

(cid:32)(cid:90)

k
(cid:89)

gk(x) =

(cid:33)

ˆg(ωj)eiωT

j xdωj

j=1
(cid:90)

∥ωj ∥∞≤W
(cid:32)(cid:90)

∥ω∥∞≤kW

(cid:80)k

l=1 ωl=ω

=

Πk

j=1ˆg(ωj)dω1 . . . dωk

(cid:33)

eiωT kdω

In particular, the coefficients with ∥ω∥∞ > kW vanish, as we needed.
Lemma 11 (Young’s convolution identity). For functions g ∈ Lp(Rd) and h ∈ Lq(Rd) and

where 1 ≤ p, q, r ≤ ∞ we have

1
p

+

1
q

=

1
r

+ 1

∥f ∗ g∥r ≤ ∥g∥p∥h∥q.

Here ∗ denotes the convolution operator.
Lemma 12. For a differentiable function f : Rd → R, such that f ∈ L1(Rd) we have

(cid:99)∇f (ω) = iω ˆf (ω)

23

