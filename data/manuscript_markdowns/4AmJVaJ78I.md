Block-Coordinate Methods and Restarting for Solving
Extensive-Form Games∗

Darshan Chakrabarti
IEOR Department
Columbia University
New York, NY 10025
dc3595@columbia.edu

Jelena Diakonikolas
Department of Computer Sciences
University of Wisconsin-Madison
Madison, WI 53706
jelena@cs.wisc.edu

Christian Kroer
IEOR Department
Columbia University
New York, NY 10025
ck2945@columbia.edu

Abstract

Coordinate descent methods are popular in machine learning and optimization for
their simple sparse updates and excellent practical performance. In the context
of large-scale sequential game solving, these same properties would be attractive,
but until now no such methods were known, because the strategy spaces do not
satisfy the typical separable block structure exploited by such methods. We present
the first cyclic coordinate-descent-like method for the polytope of sequence-form
strategies, which form the strategy spaces for the players in an extensive-form
game (EFG). Our method exploits the recursive structure of the proximal update
induced by what are known as dilated regularizers, in order to allow for a pseudo
block-wise update. We show that our method enjoys a O(1/T ) convergence rate to
a two-player zero-sum Nash equilibrium, while avoiding the worst-case polynomial
scaling with the number of blocks common to cyclic methods. We empirically show
that our algorithm usually performs better than other state-of-the-art first-order
methods (i.e., mirror prox), and occasionally can even beat CFR+, a state-of-
the-art algorithm for numerical equilibrium computation in zero-sum EFGs. We
then introduce a restarting heuristic for EFG solving. We show empirically that
restarting can lead to speedups, sometimes huge, both for our cyclic method, as
well as for existing methods such as mirror prox and predictive CFR+.

1

Introduction

Extensive-form games (EFGs) are a broad class of game-theoretic models which are played on a
tree. They can compactly model both simultaneous and sequential moves, private and/or imperfect
information, and stochasticity. Equilibrium computation for a two-player zero-sum EFG can be
formulated as the following bilinear saddle-point problem (BSPP)

min
x∈X

max
y∈Y ⟨

Mx, y

.

⟩

(PD)

for the x and y players are convex polytopes known as sequence-form
Here, the set of strategies
polytopes [44]. The (PD) formulation lends itself to first-order methods (FOMs) [14, 25], linear

X

Y

,

∗Authors are ordered alphabetically.

37th Conference on Neural Information Processing Systems (NeurIPS 2023).

programming [44], and online learning-based approaches [6, 8, 15, 17, 42, 49], since the feasible sets
are convex and compact polytopes, and the objective is bilinear.

A common approach for solving BSPPs is by using first-order methods, where local gradient informa-
tion is used to iteratively improve the solution in order to converge to an equilibrium asymptotically.
In the game-solving context, such methods rely on two oracles: a first-order oracle that returns a
(sub)gradient at the current pair of strategies, and a pair of prox oracles for the strategy spaces
,
,
X
Y
which allow one to perform a generalized form of projected gradient descent steps on
. These
prox oracles are usually constructed through the choice of an appropriate regularizer. For EFGs, it
is standard to focus on regularizers for which the prox oracle can be computed in linear time with
respect to the size of the polytope, which is only known to be achievable through what is known
as dilated regularizers [23]. Most first-order methods for EFGs require full-tree traversals for the
first-order oracle, and full traversals of the decision sets for the prox computation, before making
a strategy update for each player. For large EFGs these full traversals, especially for the first-order
oracle, can be very expensive, and it may be desirable to make strategy updates before a full traversal
has been performed, in order to more rapidly incorporate partial first-order information.

X

Y

,

In other settings, one commonly used approach for solving large-scale problems is through coordinate
methods (CMs) [34, 46]. These methods involve computing the gradient for a restricted set of
coordinates at each iteration of the algorithm, and using these partial gradients to construct descent
directions. The convergence rate of these methods typically is able to match the rate of full gradient
methods. However, in some cases they may exhibit worse runtime due to constants introduced by the
method. In spite of this, they often serve practical benefits of being more time and space efficient,
and enabling distributed computation [2, 3, 11, 19, 22, 28, 30, 32, 34, 47, 48].

Generally, coordinate descent methods assume that the problem is separable, i.e., there exists a
partition of the coordinates into blocks so that the feasible set can be decomposed as a Cartesian
product of feasible sets, one for each block. This assumption is crucial, as it allows the methods to
perform block-wise updates without worrying about feasibility, and it simplifies the convergence
analysis. Extending CDMs to EFGs is non-trivial because the constraints of the sequence-form
polytope do not possess this separable structure; instead the strategy space is such that the decision
at a given decision point affects all variables that occur after that decision. We are only aware of a
couple examples in the literature where separability is not assumed [1, 10], but those methods require
strong assumptions which are not applicable in EFG settings.

Contributions. We propose the Extrapolated Cyclic Primal-Dual Algorithm (ECyclicPDA). Our
algorithm is the first cyclic coordinate method for the polytope of sequence-form strategies. It
achieves a O(1/T ) convergence rate to a two-player zero-sum Nash equilibrium, with no dependence
on the number of blocks; this, is in contrast with the worst-case polynomial dependence on the
number of blocks that commonly appears in convergence rate guarantees for cyclic methods. Our
method crucially leverages the recursive structure of the prox updates induced by dilated regularizers.
In contrast to true cyclic (block) coordinate descent methods, the intermediate iterates generated
during one iteration of ECyclicPDA are not feasible because of the non-separable nature of the
constraints of sequence-form polytopes. Due to this infeasibility we refer to our updates as being
pseudo-block updates. The only information that is fully determined after one pseudo-block update, is
the behavioral strategy for all sequences at decision points in the block that was just considered. The
behavioral strategy is converted back to sequence-form at the end of a full iteration of our algorithm.

At a very high level, our algorithm is inspired by the CODER algorithm due to Song and Diakonikolas
[39]. However, there are several important differences due to the specific structure of the bilinear
problem (PD) that we solve. First of all, the CODER algorithm is not directly applicable to our setting,
as the feasible set (treeplex) that appears in our problem formulation is not separable. Additionally,
CODER only considers Euclidean setups with quadratic regularizers, whereas our work considers
more general normed settings; in particular, the ℓ1 setup is of primary interest for our problem setup,
since it yields a much better dependence on the game size.

These two issues regarding the non-separability of the feasible set and the more general normed
spaces and regularizers are handled in our work by (i) considering dilated regularizers, which allow
for blockwise (up to scaling) updates in a bottom-up fashion, respecting the treeplex ordering;
and (ii) introducing different extrapolation steps (see Lines 10 and 13 in Algorithm 1) that are
unique to our work and specific to the bilinear EFG problem formulation. Additionally, our special
problem structure and the choice of the extrapolation sequences (cid:101)xk and (cid:101)yk allows us to remove

2

any nonstandard Lipschitz assumptions used in Song and Diakonikolas [39]. Notably, unlike Song
and Diakonikolas [39] and essentially all the work on cyclic methods we are aware of, which pay
polynomially for the number of blocks in the convergence bound, our convergence bound in the ℓ1
setting is never worse than the optimal bound of full vector-update methods such as mirror prox [33]
and dual extrapolation [35], which we consider a major contribution of our work.

Numerically, we demonstrate that our algorithm performs better than mirror prox (MP), and can be
competitive with CFR+ and its variants on certain domains. We also propose the use of adaptive
restarting as a general heuristic tool for EFG solving: whenever an EFG solver constructs a solution
with duality gap at most a constant fraction of its initial value since the last restart, we restart it and
initialize the new run with the output solution at restart. Restarting is theoretically supported by the
fact that BSPPs possess the sharpness property [5, 18, 21, 43], and restarting combined with certain
Euclidean-based FOMs leads to a linear convergence rate under sharpness [5, 21]. We show that with
restarting, it is possible for our ECyclicPDA methods to outperform CFR+ on some games; this is
the first time that a FOM has been observed to outperform CFR+ on non-trivial EFGs. Somewhat
surprisingly, we then show that for some games, restarting can drastically speed up CFR+ as well. In
particular, we find that on one game, CFR+ with restarting exhibits a linear convergence rate, and so
does a recent predictive variant of CFR+ [15], on the same game and on an additional one.
Related Work. CMs have been widely studied in the past decade and a half [1–3, 7, 10, 11, 19, 22,
28, 30, 32, 34, 37–39, 46, 48]. CMs can be grouped into three broad classes [38]: greedy methods,
which greedily select coordinates that will lead to the largest progress; randomized methods, which
select (blocks of) coordinates according to a probability distribution over the blocks; and cyclic
methods, which make updates in cyclic orders. Because greedy methods typically require full
gradient evaluation (to make the greedy selection), the focus in the literature has primarily been
on randomized (RCMs) and cyclic (CCMs) variants. RCMs require separability of the problem’s
constraints so we focus on CCMs. However, establishing convergence arguments for CCMs through
connections with convergence arguments for full gradient methods is difficult. Some guarantees
have been provided in the literature, either making restrictive assumptions [37] or by treating the
cyclical coordinate gradient as an approximation of a full gradient [7], and thus incurring a linear
dependence on the number of blocks in the convergence guarantee. Song and Diakonikolas [39]
were the first make an improvement on reducing the dependence on the number of blocks by using a
novel extrapolation strategy and introducing new block Lipschitz assumptions. That paper was the
main inspiration for our work, but inapplicable to our setting, thus necessitating new technical ideas,
as already discussed. While primal-dual coordinate methods for bilinear saddle-point problems have
been explored in Carmon et al. [9], their techniques are not clearly extendable to our problem. The
ℓ1
ℓ1 setup they consider is the one which is relevant to the two-player zero-sum game setting we
study, but their assumption in that case is that the feasible set for each of the players is a probability
simplex, which is a much simpler feasible set than the treeplex considered in our work. It is unclear
how to generalize their result to our setting, as their results depend on the simplex structure.

−

There has also been significant work on FOMs for two-player zero-sum EFG solving. Because this
is a BSPP, off-the-shelf FOMs for BSPPs can be applied, with the caveat that proximal oracles are
required. The most popular proximal oracles have been based on dilated regularizers [23], which
lead to a proximal update that can be performed with a single pass over the decision space, and
strong theoretical dependence on game constants [13, 14, 23, 25]. A second popular approach is
the counterfactual regret minimization (CFR) framework, which decomposes regret minimization
on the EFG decision sets into local simplex-based regret minimization [49]. In theory, CFR-based
results have mostly led to an inferior T −1/2 rate of convergence, but in practice the CFR framework
instantiated with regret matching+ (RM+) [41] or predictive RM+ (PRM+) [15] is the fastest
approach for essentially every EFG setting. The most competitive FOM-based approaches for
practical performance are based on dilated regularizers [14, 24], but these have not been able to
beat CFR+ on EFG settings; we show for the first time that it is possible to beat CFR+ through
a combination of block-coordinate updates and restarting, at least on some games. An extended
discussion of FOM and CFR approaches to EFG solving is given in Appendix A.

2 Notation and Preliminaries

In this section, we provide the necessary background and notation subsequently used to describe and
analyze our algorithm. As discussed in the introduction, our focus is on bilinear problems (PD).

3

2.1 Notation and Optimization Background

≥

≥

∥ · ∥

∥ · ∥

∥ · ∥

1, we have

z, x
⟨
∗ =

is denoted by

∗ = supx̸=0

induced matrix norm defined by

to denote an arbitrary ℓp norm for p

We use bold lowercase letters to denote vectors and bold uppercase letters to denote matrices. We
1 applied to a vector in either Rm or Rn, depending
use
∗ and defined in the standard way as
on the context. The norm dual to
⟨z,x⟩
∥x∥ , where
z
p,
∥
∥
p∗ = 1. We further use
where p
∗ to denote the
∥Mx∥∗
∥x∥ . In particular, for the Euclidean norm
M
2
∥
∥
∥
1, the dual norm is the ℓ∞-norm,
.

⟩
p∗ , where 1
∥ · ∥
M
∗ = supx̸=0
∥
∥
∗ =
∥ · ∥

∥ · ∥
denotes the standard inner product. In particular, for

M
∗ =
∞, while the matrix norm is
∥
∥
to denote the probability simplex in n dimensions.
= 1
}

x
≥
{
Primal-dual Gap. Given x
Similarly, the dual value of (PD) is defined by minu∈Y

1, x
⟨
⟩
Rd, the primal value of the problem (PD) is maxv∈X
Mu, y
⟨
, the primal-dual gap (or saddle-point gap) is defined by

Mx, v
. Given a primal-dual pair (x, y)

∥ · ∥
is the matrix operator norm. For the ℓ1 norm

2 is also the Euclidean norm, and

∗ =
∥ · ∥
∥ · ∥
We use ∆n =

∞→1 = supx̸=0

2, the dual norm

∥ · ∥
M
∥
∥

∗ =
M
∥

∥Mx∥∞
∥x∥1

= maxi,j

p + 1

Mij
|

Rn : x

.

⟩
∈

∥ · ∥

∥ · ∥

∥ · ∥

∥ · ∥

∥ · ∥

∥ · ∥

0,

=

=

=

∈

∈

⟨

⟩

|

X × Y

Gap(x, y) = max

Mx, v

v∈X ⟨
where we define Gapu,v(x, y) =
the relaxed gap Gapu,v(x, y) for some arbitrary but fixed u
∈ X
about a candidate solution by making concrete choices of u, v.

Mu, y

Mx, v

⟩ − ⟨

⟩ −

⟨

⟩

⟩

min
u∈Y ⟨

Mu, y

= max

Gapu,v(x, y),

(u,v)∈X ×Y

. For our analysis, it is useful to work with
, and then draw conclusions

, v

∈ Y

Definitions and Facts from Convex Analysis. In this paper, we primarily work with convex
functions f : Rn
that are differentiable on the interior of their domain. We say that f
→
is cf -strongly convex w.r.t. a norm

int domf ,

∪ {±∞}

Rn,

R

if

x

y

∥ · ∥
f (x) +

∀

∈

∀

f (x), y

⟨∇

−

∈
x

⟩

+

cf
2 ∥

y

x

2.
∥

−

f (y)

≥

R

We will also need convex conjugates and Bregman divergences. Given an extended real valued
function f : Rn
.
f (x)
}
Let f : Rn
be a function that is differentiable on the interior of its domain. Given
y
∈
f (x)

∪ {±∞}
int domf , the Bregman divergence Df (y, x) is defined by Df (y, x) = f (y)
2.

, its convex conjugate is defined by f ∗(z) = supz∈Rn

. If the function f is cf -strongly convex, then Df (y, x)
⟩

→
Rn and x

∪ {±∞}

f (x), y

cf
2 ∥

− ⟨∇

x
∥

→
R

z, x

⟩ −

{⟨

≥

−

−

−

∈

x

y

2.2 Extensive-Form Games: Background and Additional Notation

∈ {

1, . . . , n

c
} ∪ {

Extensive form games are represented by game trees. Each node v in the game tree belongs to exactly
one player i
whose turn it is to move. Player c is a special player called the
chance player; it is used to denote random events that happen in the game, such as drawing a card
from a deck or tossing a coin. At terminal nodes of the game, players are assigned payoffs. We
focus on two-player zero-sum games, where n = 2 and payoffs sum to zero. Private information is
modeled using information sets (infosets): a player cannot distinguish between nodes in the same
infoset, so the set of actions available to them must be the same at each node in the infoset.

}

|

J

Aj
|

Treeplexes. The decision problem for a player in a perfect recall EFG can be described as follows.
, and at each decision point j the player has a set of actions Aj
There exists a set of decision points
= nj actions in total. These decision points coincide with infosets in the EFG. Without loss
with
of generality, we let there be a single root decision point, representing the first decision the player
makes in the game. The choice to play an action a
is represented
using a sequence (j, a), and after playing this sequence, the set of possible next decision points is
j,a (which may be empty in case the game terminates). The set of decisions form a tree,
denoted by
C
unless j = j′ and a = a′; this is known as perfect recall. The last
meaning that
sequence (necessarily unique) encountered on the path from the root to decision point j is denoted by
pj. We define
j as the set consisting of all decision points that can be reached from j. An example
of the use of this notation for a player in Kuhn poker [26] can be found in Appendix B.

Aj for a decision point j

j′,a′ =

∈ J

∩ C

j,a

∈

C

∅

↓

The set of strategies for a player can be characterized using the sequence-form, where the value of
the decision variable assigned to playing the sequence (j, a) is the product of the decision variable
assigned to playing the parent sequence pj and the probability of playing action a when at j [44].

4

The set of all sequence-form strategies of a player form a polytope known as the sequence-form
polytope. Sequence-form polytopes fall into a class of polytopes known as treeplexes [23], which can
be characterized inductively using convex hull and Cartesian product operations:

Definition 2.1 (Treeplex). A treeplex
where r is the the root decision point for a player.

X

for a player can be characterized recursively as follows,

j,a =

X

(cid:89)

j′∈Cj,a

↓j′,

X

X

↓j =
=

X

(λ1, . . . , λ|Aj |, λ1x1, . . . , λ|Aj |x|Aj | : (λ1, . . . , λ|Aj |)
1
} × X

↓r.

{

{

∈

∆|Aj |, xa

j,a

∈ X

,
}

Mx, y
This formulation allows the expected loss of a player to be formulated as a bilinear function
⟩
⟨
of players’ strategies x, y. This gives rise to the BSPP in Equation (PD), and the set of saddle points
of that BSPP are exactly the set of Nash equilibria of the EFG. The payoff matrix M is a sparse
matrix, whose nonzeroes correspond to the set of leaf nodes of the game tree.

Indexing Notation. A sequence-form strategy of a player can be written as a vector v, with an entry
for each sequence (j, a). We use vj to denote the subset of size
of entries of v that correspond
Aj and let v↓j denote the subset of entries of v that
to sequences (j, a) formed by taking actions a
are indexed by sequences that occur in the subtreeplex rooted at j. Additionally, we use vpj to denote
the (scalar) value of the parent sequence of decision point j. By convention, for the root decision
point j, we let vpj = 1. Observe that for any j

, vj/vpj is in the probability simplex.

Aj
|

∈

|

∈ J

Y

Y

X

X

X

Z

Z

Z

Z

Z

Z

J

J

J

J

J

J

(1),

(1),

∈ J

≤ |J

Z
Z
|

(i), j′

(2), . . . ,

(2), . . . ,

J
(1), . . . ,

Given a treeplex

X and
J
(s), where s
Y

∈ J
Y . We assume that

(k) respects the treeplex ordering if for any two sets

we denote by
Z into k
sets
(i′) with i < i′ and any two infosets j

Z the set of infosets for this treeplex. We say that a partition of
(i),
(i′), j does not intersect the path from
X , while the set of
Y are partitioned into s nonempty
and the

J
J
j′ to the root decision point. The set of infosets for the player x is denoted by
infosets for player y is denoted by
J
(s) and
sets
≤
J
ordering of the sets in the two partitions respect the treeplex ordering of
X
Given a pair (t, t′), we use Mt,t′ to denote the full-dimensional (m
n) matrix obtained from the
(t′), and zeroing out the rest. When in
matrix M by keeping all entries indexed by
place of t or t′ we use “:”, it corresponds to keeping as non-zeros all rows (for the first index) or all
columns (for the second index). In particular, Mt,: is the matrix that keeps all rows of M indexed
(t) intact and zeros out the rest. Further, notation Mt′,t:s is used to indicate that we select
by
(s), while we zero
rows indexed by
out the rest; similarly for Mt:s,t′. Notation Mt′,1:t is used to indicate that we select rows indexed
(t), while we zero out the rest;
by
, x(t) denotes the entries of x indexed by the elements of
similarly for M1:t,t′. Given a vector x

(t′) and all columns of M indexed by

(t′) and all columns of M indexed by

,
{|J
|}
|
, respectively.

J
(2), . . . ,

(t+1), . . . ,

(t) and

min
,

(1),

(t),

|J

×

J

J

J

J

J

J

J

J

J

J

J

J

J

Y

X

X

X

X

X

Y

Y

Y

Y

Y

Y

Y

Y

, y(t) denotes the entries of y indexed by the elements of

(t).

Y

J
Additionally, we use M(t,t′) to denote the submatrix of M obtained by selecting rows indexed by
and

q)-dimensional, for p = (cid:80)

(t′). M(t,t′) is (p

∈ Y

Aj

J

(t)

Y

J
(t); similarly, for y

X

∈ X

×
. Notation “:” has the same meaning as in the previous paragraph.

J

j∈JX

|

|

X
J
q = (cid:80)

(t) and columns indexed by
Aj
|

j∈JY

(t′ )

|

R
Dilated Regularizers. We assume access to strongly convex functions ϕ :
with known strong convexity parameters cϕ > 0 and cψ > 0, and that are continuously differentiable
on the interiors of their respective domains. We further assume that these functions are nice as defined
by Farina et al. [14]: their gradients and the gradients of their convex conjugates can be computed in
time linear (or nearly linear) in the dimension of the treeplex.

R and ψ :

X →

Y →

A dilated regularizer is a framework for constructing nice regularizing functions for treeplexes. It
makes use of the inductive characterization of a treeplex via Cartesian product and convex hull
operations to generalize from the local simplex structure of the sequence-form polytope at a decision
point to the entire sequence-form polytope. In particular, given a local “nice” regularizer ϕj for each
decision point j, a dilated regularizer for the treeplex can be defined as ϕ(x) = (cid:80)
xpj ).

xpj ϕj( xj

j∈JX

5

⟩

The key property of these dilated regularizing functions is that the prox computations of the form
+ Dϕ(xk, xk−1)
h, x
xk = argminx∈X {⟨
decompose into bottom-up updates, where, up to a
}
(t) can be computed solely based on the coordinates
scaling factor, each set of coordinates from set
(t). Concretely,
(t−1) and coordinates of g from sets
(1), . . . ,
of xk from sets
the recursive structure of the prox update is as follows (this was originally shown by [23], here we
show a variation from Farina et al. [13]):
Proposition 2.2 (Farina et al. [13]). A prox update to compute xk, with gradient h and center xk−1
using a Bregman divergence constructed from a dilated DGF ϕ can be decomposed
on a treeplex
X as follows:
into local prox updates at each decision point j

(1), . . . ,

X

J

J

J

J

J

X

X

X

X

X

k = xpj
xj
k ·

argmin
bj ∈∆nj

(cid:26)

(cid:10)hj + ˆhj, bj(cid:11) + Dϕj

ˆh(j,a) =

(cid:20)

(cid:88)

ϕ↓j′∗ (cid:0)

j′∈Cj,a

h↓j +

−

∇

ϕ↓j′(cid:0)x↓j′

k−1

(cid:1)(cid:1)

−

3 Extrapolated Cyclic Algorithm

∈ J
(cid:18)

bj,

,

(cid:19)(cid:27)

xj
k−1
xpj
k−1
ϕj′(cid:18) xj′
k−1
x(j,a)
k−1

(cid:19)

(cid:28)

+

ϕj′(cid:18) xj′
k−1
x(j,a)
k−1

(cid:19)
,

xj′
k−1
x(j,a)
k−1

∇

(cid:29)(cid:21)
.

and

and

yj
k
pj
y
k

yj
k
pj
y
k

yj
k
pj
y
k

xj
k
pj
x
k

xj
k
pj
x
k

xj
k
pj
x
k

and hj

k (respectively,

Our extrapolated cyclic primal-dual algorithm is summarized in Algorithm 1. As discussed in
Section 2, under the block partition and ordering that respects the treeplex ordering, the updates for
k in Line 9 (respectively, y(t)
x(t)
in Line 12), up to scaling by the value of their respective parent
k
and gj
sequences, can be carried out using only the information about
k)
for infosets j that are “lower” on the treeplex. The specific choices of the extrapolation sequences (cid:101)xk
and (cid:101)yk that only utilize the information from prior cycles and the scaled values of
for
infosets j updated up to the block t updates for xk and yk are what crucially enables us to decompose
the updates for xk and yk into local block updates carried out in the bottom-up manner. At the end
of the cycle, once
has been updated for all infosets, we can carry out a top-to-bottom
update to fully determine vectors xk and yk, as summarized in the last two for loops in Algorithm 1.
We present an implementation-specific version of the algorithm in Appendix D, which explicitly
demonstrates that our algorithm’s runtime does not have a dependence on the number of blocks used.
In our analysis of the implementation-specific version of the algorithm, we argue that the per-iteration
complexity of our algorithm matches that of MP. To support this analysis, we compare the empirical
runtimes of our algorithm with MP and CFR+ variants in Section 4.
Our convergence argument is built on the decomposition of the relaxed gap Gapu,v(xk, vk) for
arbitrary but fixed (u, v)
into telescoping and non-positive terms, which is common in
first-order methods. The first idea that enables leveraging cyclic updates lies in replacing vectors
Mxk and M⊤yk by “extrapolated” vectors gk and hk that can be partially updated in a blockwise
fashion as a cycle of the algorithm progresses, as stated in Proposition 3.1. To our knowledge, this
basic idea originates in Song and Diakonikolas [39]. Unique to our work are the specific choices
of gk and hk, which leverage all the partial information known to the algorithm up to the current
iteration and block update. Crucially, we leverage the treeplex structure to show that our chosen
updates are sufficient to bound the error sequence
k and obtain the claimed convergence bound in
Theorem 3.2. Due to space constraints, the proof is deferred to Appendix C.

∈ X × Y

E

To simplify the exposition, we introduce the following notation:

s−1
(cid:88)

Mx :=

Mt,t+1:s, My := M

Mx =

−

s
(cid:88)

t=1

Mt:s,t;

(3.1)

t=1
Mx
∥
When the norm of the space is

µx :=

∗ +

∥

∥

My

∗, µy :=
∥

M⊤
y ∥
∥
1, both µx and µy are bounded above by 2 maxi,j

M⊤
x ∥

∗ +

∗.

∥

.
|
The next proposition decomposes the relaxed gap into an error term and telescoping terms. The
proposition is independent of the specific choices of extrapolated vectors gk, hk.

Mij
|

∥ · ∥

∥ · ∥

=

6

Algorithm 1 Extrapolated Cyclic Primal-Dual EFG Solver (ECyclicPDA)

, y0

∈ X

∈ Y

, η0 = H0 = 0, η =

√

cϕcψ
µx+µy

, ¯x0 = x0, ¯y0 = y0, g0 = 0, h0 = 0

xk−2), (cid:101)yk = yk−1 + ηk−1

ηk

(yk−1

−

yk−2)

−

1: Initialization: x0
2: for k = 1 : K do
3:
4:
5:
6:
7:

Choose ηk
≤
gk = gk−1, hk = hk−1
(cid:101)xk = xk−1 + ηk−1
for t = 1 : s do

ηk

(xk−1

η, Hk = Hk−1 + ηk

8:

9:

10:

11:

(cid:110)

(cid:101)yk
ηk
argminx∈X
xpj
k−1 + ηk−1
ηk

(cid:104) xj
k
pj
x
k

h(t)
k = (M(:,t))⊤
(cid:104)
x(t)
k =
(cid:101)x(t)
k =
g(t)
k = M(t,:)
(cid:104)
y(t)
k =
(cid:101)y(t)
k =

(cid:110)

(cid:101)xk
ηk
gk, v
argmaxv∈Y
⟨
(cid:16)
ypj
k−1 + ηk−1
yj
k−1 −
ηk

⟩ −

x, hk
⟨
(cid:16)

⟩

+ Dϕ(x, xk−1)
(cid:17)(cid:105)

xj
k−1 −

xj
k−1
pj
k−1

x

xpj
k−2

(cid:111)(cid:105)(t)

j∈JX

(t)

(cid:111)(cid:105)(t)

Dψ(v, yk−1)
yj
k−1
pj
k−1

ypj
k−2

(cid:17)(cid:105)

y

j∈JY

(t)

12:

13:

14:

(cid:104) yj
k
pj
y
k
for j
X do
∈ J
k = xpj
xj
k ·
for j
Y do
∈ J
k = ypj
yj
k ·
¯xk = Hk−ηk
17:
18: Return: ¯xK, ¯yK

16:

15:

Hk

(cid:1)

(cid:0) xj
k
pj
x
k

(cid:1)

(cid:0) yj
k
pj
y
k
¯xk−1 + ηk
Hk

xk, ¯yk = Hk−ηk

Hk

¯yk−1 + ηk
Hk

yk

Proposition 3.1. Let xk, yk be the iterates of Algorithm 1 for k
yk

, we have

1. Then, for all k

1, xk

,

∈ X

≥

≥

∈ Y

ηkGapu,v(xk, yk)
where the error sequence

k
≤ E

−

k := ηk
E

Mxk
⟨

−

gk, v

−

k is defined by
E
(cid:10)u

yk

ηk

⟩ −

Dϕ(u, xk) + Dϕ(u, xk−1)

Dψ(v, yk) + Dψ(v, yk−1),

−

xk, M⊤yk

(cid:11)

hk

−

−

Dψ(yk, yk−1)

−

Dϕ(xk, xk−1).

−

To obtain our main result, we leverage the blockwise structure of the problem, the bilinear structure
k. A
of the objective, and the treeplex structure of the feasible sets to control the error sequence
E
key property that enables this result is that normalized entries xj
k−1 from the same information
set belong to a probability simplex. This property is crucially used in controlling the error of the
extrapolation vectors. The main result is summarized in the following theorem.
Theorem 3.2. Consider the iterates xk, yk for k
¯xK, ¯yK. Then,
1,
k
∀
µxDϕ(x∗, xK) + µyDψ(y∗, yK)
µx + µy

Dϕ(x∗, x0) + Dψ(y∗, y0), and, further,

1 in Algorithm 1 and the output primal-dual pair

k/xpj

≤

≥

≥

Gap(¯xK, ¯yK) = sup

M¯xK, v

u∈X ,v∈Y{⟨

⟩ − ⟨
√

Mu, ¯yK

supu∈X ,v∈Y {

Dϕ(u, x0) + Dψ(v, y0)
}

HK

.

⟩} ≤

In the above bound, if

Gap(¯xK, ¯yK)

≤

k

cϕcψ
µx+µy
ϵ after at most (cid:6) (µx+µy)(supu∈X ,v∈Y {Dϕ(u,x0)+Dψ(v,y0)})

1, ηk = η =

≥

∀

√

cϕcψϵ

(cid:7) iterations.

, then HK = Kη. As a consequence, for any ϵ > 0,

4 Experimental Evaluation and Discussion

We evaluate the performance of ECyclicPDA instantiated with three different dilated regularizers:
dilated entropy [25], dilatable global entropy [14], and dilated ℓ2 [13]. In the case of the dilated ℓ2

7

regularizer, we use dual averaging of the “extrapolated” vectors gk and hk in our algorithm, since
otherwise we have no guarantee that the iterates would remain in the relative interior of the domain
of the dilated DGF, and the Bregman divergence may become undefined. We compare our method
to MP, which is state-of-the-art among first-order methods for EFG solving. We test ECyclicPDA
and MP with three different averaging schemes: uniform, linear, and quadratic averaging since
Gao et al. [20] suggest that these different averaging schemes can lead to faster convergence in
practice. We also compare against empirical state-of-the-art CFR+ variants: CFR+ [41], and the
predictive CFR+ variant (PCFR+) [15]. We emphasize that our method achieves the same O( 1
T )
average-iterate convergence rate as MP, and that all the CFR+ variants have the same suboptimal
O( 1√
) average-iterate convergence rate. We experiment on four standard benchmark games for
T
EFG solving: Goofspiel (4 ranks), Liar’s Dice, Leduc (13 ranks), and Battleship. In all experiments,
we run for 10, 000 full (or equivalent) gradient computations. This corresponds to 5,000 iterations
of ECyclicPDA, CFR+, and PCFR+, and 2,500 iterations of MP.2 A description of all games is
provided in Appendix E. Additional experimental details are provided in Appendix G.

∈

For each instantiation of ECyclicPDA considered on a given game (choice of regularizer, averaging,
and block construction strategy) the stepsize is tuned by taking power of 2 multiples of η (2l
η for
N), where η is the theoretical stepsize stated in Theorem 3.2, and then choosing the stepsize η∗
l
among these multiples of η that has the best performance. Within the algorithm, we use a constant
stepsize, letting ηk = η0 for all k. We apply the same tuning scheme for MP stepsizes (for a given
choice of regularizer and averaging). Note that this stepsize tuning is coarse, and so it is possible that
better results can be achieved for ECyclicPDA and MP using finer stepsize tuning.

·

We test our algorithm with four different block construction strategies. The single block construction
strategy puts every decision point in a single block, and thus it corresponds to the non-block-based
version of ECyclicPDA. The children construction strategy iterates through the decision points of the
treeplex bottom-up (by definition, this will respect the treeplex ordering), and placing each set of
decision points that have parent sequences starting from the same decision point in its own block. The
postorder construction strategy iterates through the decision points bottom-up (again, by definition,
this will respect the treeplex ordering). The order is given by a postorder traversal of the treeplex,
treating all decision points that have the same parent sequence as a single node (and when the node is
processed, all decision points are sequentially added to the block). It greedily makes blocks as large
as possible, while only creating a new block if it causes a parent decision point and child decision
point to end up in the same block. The infosets construction strategy places each decision point in its
own block. We provide further description of the block construction strategies in Appendix F.

We show the results of different block construction strategies in Figure 1. For each block construction
strategy, ECyclicPDA is instantiated with the choice of regularizer and averaging that yields the
fastest convergence among all choices of parameters. We can see that the different block construction
strategies do not make a significant difference in Goofspiel (4 ranks) or in Leduc (13 ranks). However,
we see benefits of using blocks in Liar’s Dice and Battleship. In Liar’s Dice, children and postorder
have a clear advantage, and children outperforms the other block construction strategies in Battleship.
We show the results of comparing our algorithm against MP, CFR+, and PCFR+ in Figure 2.
ECyclicPDA is instantiated with the choice of regularizer, averaging, and block construction strategy
that yields the fastest convergence among all choices for ECyclicPDA, and MP is instantiated with
the choice of regularizer and averaging that yields the fastest convergence among all choices for MP.
We see that ECyclicPDA performs better than MP in all games besides Goofspiel (4 ranks), where
they perform about the same. In Liar’s Dice and Battleship, the games where ECyclicPDA benefits
from having multiple blocks, we see competitiveness with CFR+ and PCFR+. In particular, in Liar’s
Dice, ECyclicPDA is overtaking CFR+ at 10, 000 gradient computations. On Battleship, we see that
both ECyclicPDA and MP outperform CFR+, and that ECyclicPDA is competitive with PCFR+.
Restarting. We now introduce restarting as a heuristic tool for speeding up EFG solving. While
restarting is only known to lead to a linear convergence rate in the case of using the ℓ2 regularizer
in certain FOMs [5, 21], we apply restarting as a heuristic across our methods based on dilated
regularizers and to CFR-based methods. To the best of our knowledge, restarting schemes have not
been empirically evaluated on EFG algorithms such as MP, CFR+, or (obviously), our new method.
We implement restarting by resetting the averaging process when the duality gap has halved since
the last time the averaging process was reset. After resetting, the initial iterate is set equal to the last

2Here we count one gradient evaluation for x and one for y as two gradient evaluations total.

8

Figure 1: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies.

Figure 2: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA, MP, CFR+, PCFR+.

iterate we saw before restarting. Since the restarting heuristic is one we introduce, we distinguish
restarted variants in the plots by prepending the name of the algorithm with “r”. For example, when
restarting is applied to CFR+, the label used is rCFR+.

We show the results of different block construction strategies when restarting is used on ECyclicPDA
in Figure 3. As before, we take the combination of regularizer and averaging scheme that works
best. Again, we can see that the different block construction strategies do not make a significant
difference in Goofspiel (4 ranks) or in Leduc (13 ranks), while making a difference for Liar’s Dice
and Battleship. However, with restarting, the benefit of the children and postorder for Liar’s Dice
and Battleship is even more pronounced relative to the other block construction strategies; the gap is
several orders of magnitude after 104 gradient computations. Note that while children performed
worse than single block for Battleship previously, with restarting, children performs much better.

Finally, we compare the performance of the restarted version of our algorithm, with restarted versions
of MP, CFR+, and PCFR+ in Figure 4. As before, we take the combination of regularizer, averaging
scheme, and block construction strategy that works best for ECyclicPDA, and the combination of
regularizer and averaging scheme that works best for MP. Firstly, we note that the scale of the y-axis
is different from Figure 2 for all games besides Leduc (13 ranks), because restarting tends to hit
much higher levels of precision. We see that restarting provides significant benefits for PCFR+
in Goofspiel (4 ranks) allowing it to converge to numerical precision, while the other algorithms
do not benefit much. In Liar’s Dice, restarted CFR+ and PCFR+ converge to numerical precision
within 200 gradient computations, and restarted ECyclicPDA converges to numerical precision at 104
gradient computations. Additionally, restarted MP achieves a much lower duality gap. For Battleship,
ECyclicPDA, MP, and PCFR+ all benefit from restarting, and restarted ECyclicPDA is competitive
with restarted PCFR+. Similar to the magnification in benefit of using blocks versus not using blocks
when restarting in Liar’s Dice and Battleship, we see that restarted ECyclicPDA achieves significantly
better duality gap than MP in these games.

Wall-clock time experiments. In Table 1, we show the average wall-clock time per iteration of
our algorithm instantiated with different block construction strategies as well as MP, CFR+, and
PCFR+. It is clear from this table that the runtimes are pretty similar to each other, and this is without
extensive optimization of our particular implementation. Clearly, our algorithm is at least as fast
as MP per “full” gradient computation. Since the computational bottleneck of gradient and prox
computations becomes apparent in bigger games, Battleship demonstrates the speed of our algorithm
relative to MP best.

9

102104Gradientcomputations10−2100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−410−1Liar’sDice102104Gradientcomputations10−310−1Leduc(13ranks)102104Gradientcomputations10−510−2BattleshipECyclicPDAsingleblockECyclicPDApostorderECyclicPDAchildrenECyclicPDAinfosets1102104Gradientcomputations10−410−1DualitygapGoofspiel(4ranks)102104Gradientcomputations10−610−2Liar’sDice102104Gradientcomputations10−2101Leduc(13ranks)102104Gradientcomputations10−510−1BattleshipECyclicPDAMPCFR+PCFR+1Figure 3: Duality gap as a function of the number of full (or equivalent) gradient computations for
when restarting is applied to ECyclicPDA with different block construction strategies. We take the
best duality gap seen so far so that the plot is monotonic.

Figure 4: Duality gap as a function of the number of full (or equivalent) gradient computations for
when restarting is applied to ECyclicPDA, MP, CFR+, PCFR+. We take the best duality gap seen so
far so that the plot is monotonic.

Table 1: The average wall clock time per iteration in milliseconds of ECyclicPDA instantiated with
different block construction strategies, MP, CFR+, and PCFR+. The duality gap is computed every
100 iterations.
Name
ECyclicPDA single block

Goofspiel (4 ranks) Liar’s Dice Leduc (13 ranks) Battleship

46.020

11.370

6.770

2.470

ECyclicPDA children

ECyclicPDA infosets

ECyclicPDA postorder

MP
CFR+
Predictive CFR+

4.270

8.850

4.340

3.720

1.330

1.750

15.020

16.240

14.350

19.530

1.360

1.920

7.450

7.370

7.430

10.360

0.370

0.430

47.230

53.490

49.020

83.450

7.980

9.410

Discussion. We develop the first cyclic block-coordinate-like method for two-player zero-sum
EFGs. Our algorithm relies on the recursive nature of the prox updates for dilated regularizers,
cycling through blocks that respect the partial order induced on decision points by the treeplex, and
extrapolation to conduct pseudo-block updates, produce feasible iterates, and achieve O( 1
T ) ergodic
convergence. Furthermore, the runtime of our algorithm has no dependence on the number of blocks.
We present empirical evidence that our algorithm generally outperforms MP, and is the first FOM to
compete with CFR+ and PCFR+ on non-trivial EFGs. Finally, we introduce a restarting heuristic for
EFG solving, and demonstrate often huge gains in convergence rate. An open question raised by our
work is understanding what makes restarting work for methods used with regularizers besides the
ℓ2 regularizer (the only setting for which there exist linear convergence guarantees). This may be
challenging because existing proofs require upper bounding the corresponding Bregman divergence
(for a given non-ℓ2 regularizer) between iterates by the distance to optimality. This is difficult for
entropy or any dilated regularizer because the initial iterate used by the algorithm after restarting
may have entries arbitrarily close to zero even if they are guaranteed to not exactly be zero (as is the
case for entropy). Relatedly, both our block-coordinate method and restarting have a much bigger
advantage in some numerical instances (Battleship, Liar’s Dice) than others (Leduc and Goofspiel); a
crucial question is to understand what type of game structure drives this behavior.

10

102104Gradientcomputations10−2100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−1010−4Liar’sDice102104Gradientcomputations10−2100Leduc(13ranks)102104Gradientcomputations10−1010−4BattleshiprECyclicPDAsingleblockrECyclicPDApostorderrECyclicPDAchildrenrECyclicPDAinfosets1102104Gradientcomputations10−1010−4DualitygapGoofspiel(4ranks)102104Gradientcomputations10−1010−4Liar’sDice102104Gradientcomputations10−2101Leduc(13ranks)102104Gradientcomputations10−1010−4BattleshiprECyclicPDArMPrCFR+rPCFR+1Acknowledgements

Darshan Chakrabarti was supported by National Science Foundation Graduate Research Fellowship
Program under award number DGE-2036197. Jelena Diakonikolas was supported by the Office
of Naval Research under award number N00014-22-1-2348. Christian Kroer was supported by
the Office of Naval Research awards N00014-22-1-2530 and N00014-23-1-2374, and the National
Science Foundation awards IIS-2147361 and IIS-2238960.

References

[1] Aviad Aberdam and Amir Beck. An accelerated coordinate gradient descent algorithm for
non-separable composite optimization. Journal of Optimization Theory and Applications, 193
(1-3):219–246, 2021.

[2] Ahmet Alacaoglu, Quoc Tran Dinh, Olivier Fercoq, and Volkan Cevher. Smooth primal-dual
coordinate descent algorithms for nonsmooth convex optimization. In Advances in Neural
Information Processing Systems, 2017.

[3] Zeyuan Allen-Zhu, Zheng Qu, Peter Richtárik, and Yang Yuan. Even faster accelerated
coordinate descent using non-uniform sampling. In Proceedings of International Conference on
Machine Learning, 2016.

[4] Ioannis Anagnostides, Gabriele Farina, and Tuomas Sandholm. Near-optimal ϕ-regret learning

in extensive-form games. arXiv preprint arXiv:2208.09747, 2022.

[5] David Applegate, Oliver Hinder, Haihao Lu, and Miles Lubin. Faster first-order primal-dual
methods for linear programming using restarts and sharpness. arXiv preprint arXiv:2105.12715,
2022.

[6] Yu Bai, Chi Jin, Song Mei, Ziang Song, and Tiancheng Yu. Efficient phi-regret minimization in
extensive-form games via online mirror descent. In Advances in Neural Information Processing
Systems, 2022.

[7] Amir Beck and Luba Tetruashvili. On the convergence of block coordinate descent type methods.

SIAM Journal on Optimization, 23(4):2037–2060, 2013.

[8] Noam Brown and Tuomas Sandholm. Solving imperfect-information games via discounted
regret minimization. In Proceedings of the AAAI Conference on Artificial Intelligence, 2019.

[9] Yair Carmon, Yujia Jin, Aaron Sidford, and Kevin Tian. Coordinate methods for matrix games.

arXiv preprint arXiv:2009.08447, 2020.

[10] Flavia Chorobura and Ion Necoara. Random coordinate descent methods for nonseparable

composite optimization. arxiv preprint arXiv:2203.14368, 2022.

[11] Jelena Diakonikolas and Lorenzo Orecchia. Alternating randomized block coordinate descent.

In Proceedings of International Conference on Machine Learning, 2018.

[12] Gabriele Farina, Christian Kroer, Noam Brown, and Tuomas Sandholm. Stable-predictive
optimistic counterfactual regret minimization. In Proceedings of the International Conference
on Machine Learning, 2019.

[13] Gabriele Farina, Christian Kroer, and Tuomas Sandholm. Optimistic regret minimization
for extensive-form games via dilated distance-generating functions. In Advances in Neural
Information Processing Systems, 2019.

[14] Gabriele Farina, Christian Kroer, and Tuomas Sandholm. Better regularization for sequential
decision spaces: Fast convergence rates for Nash, correlated, and team equilibria. In Proceedings
of the ACM Conference on Economics and Computation, 2021.

[15] Gabriele Farina, Christian Kroer, and Tuomas Sandholm. Faster game solving via predictive
blackwell approachability: Connecting regret matching and mirror descent. In Proceedings of
the AAAI Conference on Artificial Intelligence, 2021.

11

[16] Gabriele Farina, Ioannis Anagnostides, Haipeng Luo, Chung-Wei Lee, Christian Kroer, and
Tuomas Sandholm. Near-optimal no-regret learning dynamics for general convex games.
Advances in Neural Information Processing Systems, 2022.

[17] Gabriele Farina, Chung-Wei Lee, Haipeng Luo, and Christian Kroer. Kernelized multiplicative
weights for 0/1-polyhedral games: Bridging the gap between learning in extensive-form and
normal-form games. In Proceedings of the International Conference on Machine Learning,
2022.

[18] Olivier Fercoq. Quadratic error bound of the smoothed gap and the restarted averaged primal-

dual hybrid gradient. arXiv preprint arXiv:2206.03041, 2023.

[19] Jerome Friedman, Trevor Hastie, and Rob Tibshirani. Regularization paths for generalized

linear models via coordinate descent. Journal of statistical software, 33(1):1, 2010.

[20] Yuan Gao, Christian Kroer, and Donald Goldfarb. Increasing iterate averaging for solving
saddle-point problems. In Proceedings of the AAAI Conference on Artificial Intelligence, 2021.

[21] Andrew Gilpin, Javier Pena, and Tuomas Sandholm. First-order algorithm with

(ln(1/ϵ))
convergence for ϵ-equilibrium in two-person zero-sum games. Mathematical programming, 133
(1):279–298, 2012.

O

[22] Mert Gürbüzbalaban, Asuman Ozdaglar, Pablo A Parrilo, and N Denizcan Vanli. When
cyclic coordinate descent outperforms randomized coordinate descent. In Advances in Neural
Information Processing Systems, 2017.

[23] Samid Hoda, Andrew Gilpin, Javier Pena, and Tuomas Sandholm. Smoothing techniques for
computing Nash equilibria of sequential games. Mathematics of Operations Research, 35(2):
494–512, 2010.

[24] Christian Kroer, Gabriele Farina, and Tuomas Sandholm. Solving large sequential games with
the excessive gap technique. In Advances in Neural Information Processing Systems, 2018.

[25] Christian Kroer, Kevin Waugh, Fatma Kılınç-Karzan, and Tuomas Sandholm. Faster algo-
rithms for extensive-form game solving via improved smoothing functions. Mathematical
Programming, pages 1–33, 2020.

[26] Harold William Kuhn and Albert William Tucker, editors. 11. Extensive Games and the Problem

of Information, pages 193–216. Princeton University Press, 2016.

[27] Chung-Wei Lee, Christian Kroer, and Haipeng Luo. Last-iterate convergence in extensive-form

games. In Advances in Neural Information Processing Systems, 2021.

[28] Qihang Lin, Zhaosong Lu, and Lin Xiao. An accelerated randomized proximal coordinate
gradient method and its application to regularized empirical risk minimization. SIAM Journal
on Optimization, 25(4):2244–2273, 2015.

[29] Viliam Lisý, Marc Lanctot, and Michael Bowling. Online monte carlo counterfactual re-
gret minimization for search in imperfect information games. In Proceedings of the 2015
International Conference on Autonomous Agents and Multiagent Systems, AAMAS ’15, page
27–36. International Foundation for Autonomous Agents and Multiagent Systems, 2015. ISBN
9781450334136.

[30] Ji Liu, Stephen J. Wright, Christopher Ré, Victor Bittorf, and Srikrishna Sridhar. An asyn-
chronous parallel stochastic coordinate descent algorithm. arXiv preprint arxiv:1311.1873,
2014.

[31] Mingyang Liu, Asuman Ozdaglar, Tiancheng Yu, and Kaiqing Zhang. The power of regular-
ization in solving extensive-form games. In Proceedings of the International Conference on
Learning Representations, 2023.

[32] Rahul Mazumder, Jerome H Friedman, and Trevor Hastie. Sparsenet: Coordinate descent with
nonconvex penalties. Journal of the American Statistical Association, 106(495):1125–1138,
2011.

12

[33] Arkadi Nemirovski. Prox-method with rate of convergence o (1/t) for variational inequali-
ties with lipschitz continuous monotone operators and smooth convex-concave saddle point
problems. SIAM Journal on Optimization, 15(1):229–251, 2004.

[34] Yu. Nesterov. Efficiency of coordinate descent methods on huge-scale optimization problems.

SIAM Journal on Optimization, 22(2):341–362, 2012.

[35] Yurii Nesterov. Dual extrapolation and its applications to solving variational inequalities and

related problems. Mathematical Programming, 109(2-3):319–344, 2007.

[36] Sheldon M. Ross. Goofspiel — the game of pure strategy. Journal of Applied Probability, 8(3):

621–625, 1971.

[37] Ankan Saha and Ambuj Tewari. On the nonasymptotic convergence of cyclic coordinate descent

methods. SIAM Journal on Optimization, 23(1):576–601, 2013.

[38] Hao-Jun Michael Shi, Shenyinying Tu, Yangyang Xu, and Wotao Yin. A primer on coordinate

descent algorithms. arXiv preprint arXiv:1610.00040, 2016.

[39] Chaobing Song and Jelena Diakonikolas. Fast cyclic coordinate dual averaging with extrapola-

tion for generalized variational inequalities. arXiv preprint arXiv:2102.13244, 2021.

[40] Finnegan Southey, Michael P. Bowling, Bryce Larson, Carmelo Piccione, Neil Burch, Darse
Billings, and Chris Rayner. Bayes’ bluff: Opponent modelling in poker. arXiv preprint
arXiv:1207.1411, 2012.

[41] Oskari Tammelin. Solving large imperfect information games using CFR+. arXiv preprint

arXiv:1407.5042, 2014.

[42] Oskari Tammelin, Neil Burch, Michael Johanson, and Michael Bowling. Solving heads-up
limit Texas hold’em. In Twenty-Fourth International Joint Conference on Artificial Intelligence,
2015.

[43] Paul Tseng. On linear convergence of iterative methods for the variational inequality problem.

Journal of Computational and Applied Mathematics, 60(1-2):237–252, 1995.

[44] Bernhard von Stengel. Efficient computation of behavior strategies. Games and Economic

Behavior, 14(2):220–246, 1996.

[45] Chen-Yu Wei, Chung-Wei Lee, Mengxiao Zhang, and Haipeng Luo. Linear last-iterate conver-
gence in constrained saddle-point optimization. In Proceedings of International Conference on
Learning Representations, 2021.

[46] Stephen J. Wright. Coordinate descent algorithms. Mathematical Programming, 151(1):3–34,

2015.

[47] Tong Tong Wu and Kenneth Lange. Coordinate descent algorithms for lasso penalized regression.

The Annals of Applied Statistics, 2(1):224–244, 2008.

[48] Yuchen Zhang and Xiao Lin. Stochastic primal-dual coordinate method for regularized empirical

risk minimization. In Proceedings of International Conference on Machine Learning, 2015.

[49] Martin Zinkevich, Michael Johanson, Michael Bowling, and Carmelo Piccione. Regret mini-
mization in games with incomplete information. In Advances in Neural Information Processing
Systems, 2007.

13

A Additional Related Work

There has been significant work on FOMs for two-player zero-sum EFG solving. Because this
is a BSPP, off-the-shelf FOMs for BSPPs can be applied, with the caveat that proximal oracles
are required. The standard Euclidean distance has been used in some cases [21], but it requires
solving a projection problem that takes O(n log2 n) time, where n is the dimension of a player’s
decision space [16, 21]. While this is “nearly” linear time, such projections have not been used
much in practice. Proximal oracles have instead been based on dilated regularizers [23], which
lead to a proximal update that can be performed with a single pass over the decision space. With
the dilated entropy regularizer, this can be performed in linear time, and this regularizer leads to
the strongest bounds on game constants that impact the convergence rate of proximal-oracle-based
FOMs [14, 25]. More recently, it has been shown that a specialized kernelization can be used to
achieve linear-time proximal updates and stronger convergence rates specifically for the dilated
entropy with optimistic online mirror descent through a correspondence with optimistic multiplicative
weights on the exponentially-many vertices of the decision polytope [6, 17]. Yet this approach was
shown to have somewhat disappointing numerical performance in Farina et al. [17], and thus is less
important practically despite its theoretical significance.

A completely different approach for first-order-based updates is the CFR framework, which decom-
poses regret minimization on the EFG decision sets into local simplex-based regret minimization [49].
In theory, CFR-based results have mostly led to an inferior T −1/2 rate of convergence, but in practice
the CFR framework instantiated with regret matching+ (RM+) [41] or predictive RM+ (PRM+) [15]
is the fastest approach for essentially every EFG setting. RM+ is often fastest for “poker-like”
EFGs, while PRM+ is often fastest for other classes of games [15]. Improved rates on the order of
T −3/4 [12] and log T /T [4] have been achieved within the CFR framework, but only while using
regret minimizers that lead to significantly worse practical performance (in particular, numerically
these perform worse than the best 1/T FOMs such as mirror prox with appropriate stepsize tuning).

In the last few years there has been a growing literature on last-iterate convergence in EFGs. There,
the goal is to show that one can converge to an equilibrium without averaging the iterates generated
by a FOM or CFR-based method. It has long been known that with the Euclidean regularizer,
it is possible to converge at a linear rate in last iterate with e.g., the extragradient method (a.k.a.
mirror prox with the Euclidean regularizer) on BSPPs with polyhedral decision sets, as they are in
EFGs [21, 43, 45]. More recently, it has been shown that a linear rate can be achieved with certain
dilated regularizers [27], with the kernelization approach of Farina et al. [17], and in a regularized
CFR setup [31]. At this stage, however, these last-iterate results are of greater theoretical significance
than practical significance, because the linear rate often does not occur until after quite many iterations,
and typically the methods do not match the performance of ergodic methods at reasonable time scales.
For this reason, we do not compare to last-iterate algorithms in our experiments.

B Treeplex Example

X =

As an example, consider the treeplex of Kuhn poker [26] adapted from [13] shown in Figure 5. Kuhn
poker is a game played with a three card deck: jack, queen, and king. In this case, for example, we
, p1 = p2 = p3 = (0, start), p4 = (1, check), p5 = (2, check),
0, 1, 2, 3, 4, 5, 6
, p0 =
have
∅
}
{
J
fold, call
, A4 = A5 = A6 =
, A1 = A2 = A3 =
p6 = (3, check), A0 =
start
,
{
}
{
}
{
C(1,raise) =
C(0,start) =
C(6,fold) =
C(5,fold) =
C(4,fold) =
C(2,raise) =
C(1,raise) =
1, 2, 3
,
{
}
0 =
4
4 =
3, 6
3 =
2, 5
1 =
C(6,call) =
C(5,call) =
C(4,call) =
,
,
,
X ,
,
}
{
↓
}
{
↓
}
{
{
↓
J
↓
∅
6
6 =
5
5 =
represents the empty sequence.
. In this case,
,
}
{
↓
}
{
↓

check, raise
}
C(3,raise) =
2 =
1, 4
,
↓
}

∅

C Proofs

C.1 Proof of Proposition 3.1

Proof. The claim that xk
∈ Y
solutions to constrained optimization problems with these same constraints.

is immediate from the algorithm description, as both are

, yk

∈ X

14

Figure 5: The sequential decision problem for the first player in Kuhn poker. ⬣ represents the end of
the decision process and
represents an observation (which may lead to multiple decision points).
Adapted from [13].

⊗

For the remaining claim, observe first that

ηk

Mxk, v

= ηk

gk, v

⟨
Recall from Algorithm 1 that

⟨

⟩

Dψ(v, yk−1) + Dψ(v, yk−1) + ηk

Mxk

⟨

gk, v

.

⟩

−

⟩ −

yk = argmax

v∈Y

(cid:110)

ηk

gk, v
⟨

⟩ −

Dψ(v, yk−1)

(cid:111)
.

Define the function under the max defining yk by Ψk. Then as Ψk(
) and linear
ψ(
·
·
terms, we have DΨk (
, y), for any y. Further, as Ψk is maximized by yk, we have
Dψ(
, y) =
·
·
Dψ(v, yk). Thus, it follows that
Ψk(v)
Mxk, v
⟨

) is the sum of

Ψk(yk)

Mxk

gk, v

−
ηk

⟩ ≤

ηk

−

−

−

≤

⟨

⟩

Dψ(yk, yk−1) +
gk, yk
⟨
Dψ(v, yk) + Dψ(v, yk−1).

⟩ −

(C.1)

−

Using the same ideas for the primal side, we have

Mu, yk

ηk

⟨

⟩ ≥

xk, hk
ηk
+ Dϕ(u, xk)

+ Dϕ(xk, xk−1) + ηk
Dϕ(u, xk−1)

⟨

⟩

−

Combining (C.1) and (C.2),
ηkGapu,v(xk, yk)

≤

ηk

gk, v
Mxk
⟨
−
Dψ(yk, yk−1)
Dϕ(u, xk) + Dϕ(u, xk−1)
To complete the proof, it remains to combine the definition of
the last inequality.

−
−

ηk
yk
−
Dϕ(xk, xk−1)

⟩ −

−

−

(cid:10)u

(cid:10)u, M⊤yk

(cid:11)

hk

−

(C.2)

xk, M⊤yk

(cid:11)

hk

−

Dψ(v, yk) + Dψ(v, yk−1).
k from the proposition statement with

−
E

C.2 Proof of Theorem 3.2

xpj
k−1 for

k = xj
(xk−1

k
pj
k

x

For notational convenience, in this proof we define vectors ˆxk and ˆyk by ˆxj

k = yj

−

−

k
pj
k

∈ J

ypj
k−1 for j

X and ˆyj
ˆyk

ηk−1
ηk

ˆyk−1).

y
(yk−1

Y , so that (cid:101)xk = xk

j
∈ J
(cid:101)yk = yk
To prove the theorem, we first prove the following auxiliary lemma which bounds the inner product
terms appearing in the error terms
Lemma C.1. In all iterations k of Algorithm 1, for any (u, v)
ηk−1
ηk
yk

and any α, β > 0,
ˆxk−1), v

My(xk
ηk
⟨
+ ηk

My(xk−1
⟨
ηk−1

−
xk−2), v

ˆxk−1) and

Mxk
⟨

∈ X × Y

ˆxk), v

gk, v

k.
E

yk−1

ηk−1
ηk

⟩ ≤

ˆxk

yk

−

−

−

−

−

−

−
Mx(xk
Mx

⟨
+ ηk−1 ∥

−
∗ +
∥
2

yk
−
xk−1), v
My
∥

∥

∗

⟩ −
−
(cid:16)
α

−
Mx(xk−1
1
α ∥

2 +
∥

⟨
xk−2

−
yk

⟩ −
xk−1
∥

−

−
2(cid:17)
∥

.

yk−1

−

⟩
yk−1

⟩

15

and

(cid:10)u

ηk

−

−

xk, M⊤yk

(cid:11)

hk

−

ηk

ηk

≤ −

−

(cid:10)M⊤
(cid:10)M⊤

x (yk
y (yk
M⊤

−

ˆyk), u

−
yk−1), u

xk

−

−
x + M⊤
∗
y ∥
2

(cid:16)

+ ηk−1 ∥

(cid:10)M⊤

(cid:11) + ηk−1
xk

(cid:11) + ηk−1

x (yk−1
(cid:10)M⊤

−
y (yk−1
1
β ∥

yk−1

ˆyk−1), u

−

−
yk−2), u
2(cid:17)

yk−2

.

−

∥

(cid:11)

xk−1

(cid:11)

xk−1

−

β

xk−1
∥

−

xk

∥

2 +

Proof. Observe first that, by Algorithm 1,

M(t,:)xk

−

k = M(t,1:t)(cid:16)
g(t)

xk

−
+ M(t,t+1:s)(cid:16)

ˆxk

xk

−

−

ηk−1
ηk

(xk−1

(cid:17)(1:t)

ˆxk−1)

−
ηk−1
ηk

(xk−1

xk−1

−

(cid:17)(t+1:s)

xk−2)

.

−

(C.3)

Additionally, by definition (see Eq. (3.1)), (cid:80)s

t=1 Mt,1:t = M

Mx = My. Hence,

−

M(t,1:t)(cid:16)

xk

ˆxk

−

−

ηk−1
ηk

(xk−1

−

(cid:17)(1:t)

ˆxk−1)

, v(t)

y(t)
k

−

(cid:29)

Mt,1:t

(cid:16)

xk

ˆxk

−

−

ηk−1
ηk

(xk−1

−

(cid:17)

ˆxk−1)

(cid:29)

yk

, v

−
(cid:29)

ηk

= ηk

s
(cid:88)

(cid:28)

t=1
s
(cid:88)

(cid:28)

t=1
(cid:28)

= ηk

My

= ηk

yk

ˆxk−1), v

yk−1

⟩

−

−

−

(C.4)

(cid:16)

xk

ˆxk

−
−
ˆxk), v
My(xk−1

−

My(xk
⟨
ηk−1

(xk−1

ηk−1
ηk
−
ηk−1
yk
−
⟩ −
ˆxk−1), yk−1

ˆxk−1)

(cid:17)

, v

My(xk−1
.
yk

⟨
−
telescope,

⟩

−
first
two
My(xk−1
⟨

−

⟨
terms
in
ˆxk−1), yk−1

The

ηk−1

−

−

−

⟩

(C.4)
yk

we
. By definition of ˆxk, for all j

so

on

bounding

focus
X ,

∈ J

xj
k−1 −

ˆxj
k−1 =

xj
k−1
xpj
k−1

(cid:0)xpj

k−1 −

xpj
k−2

(cid:1).

By the definition of a treeplex, each vector

size. This further implies that

xj
k−1
pj
k−1

x

belongs to a probability simplex of the appropriate

xk−1
∥

−

ˆxk−1

=

∥

(cid:13)
(cid:20) xj
(cid:13)
k−1
(cid:13)
xpj
(cid:13)
k−1
[xpj
xk−1

k−1 −
−

(cid:0)xpj

xpj
k−2

k−1 −
xpj
k−2]j∈JX ∥
,
xk−2
∥

≤ ∥

≤ ∥

(cid:21)

(cid:1)

j∈JX

(cid:13)
(cid:13)
(cid:13)
(cid:13)

(C.5)

where the notation [aj]j∈JX is used to denote the vector with entries aj, for j
inequality in (C.5) holds for any ℓp norm (p
applying the definitions of the norms from the preliminaries,

1), by its definition and

(cid:68)
1, xj

≥

k−1

∈ J
(cid:69)

X . The first
j. Thus,

= 1,

∀

My(xk−1

− ⟨

−

ˆxk−1), yk−1

yk

−

ˆxk−1)
∗
∥
xk−2

∗

My(xk−1
−
My
xk−1
∥
∥
(cid:16)
My
∗
∥
2

α

−
xk−1
∥

⟩ ≤ ∥
≤ ∥
∥

≤

yk−1
∥
yk−1
∥∥
xk−2

2 +

yk
∥
yk
∥
yk

−
−
1
α ∥

∥

−

yk−1

−

2(cid:17)
∥

,

(C.6)

where the last line is by Young’s inequality and holds for any α > 0.

16

On the other hand, recalling that Mx = (cid:80)s−1

t=1 Mt,t+1:s, we also have

M(t,t+1:s)(cid:16)

xk

xk−1

−

ηk−1
ηk

−

(xk−1

−

(cid:17)(t+1:s)

xk−2)

, v(t)

y(t)
k

−

(cid:29)

Mt,t+1:s

(cid:16)

xk

xk−1

−

ηk−1
ηk

−

(xk−1

−

(cid:17)

xk−2)

, v

yk

−

(cid:29)

ηk

= ηk

s
(cid:88)

(cid:28)

t=1
s
(cid:88)

(cid:28)

t=1
(cid:28)

= ηk

Mx

xk−2)

(cid:17)

, v

(cid:29)

yk

−

(cid:16)

xk

xk−1

−
−
xk−1), v
xk−1), v

ηk−1
ηk
yk
⟩ −
−
yk
⟩ −
−
xk−2), yk

(xk−1

−
ηk−1
ηk−1

−
−

= ηk
= ηk

Mx(xk
⟨
Mx(xk
⟨
+ ηk−1

Mx(xk−1
⟨
Mx(xk−1
⟨
yk−1
(C.7)
Observe that in (C.7) the first two terms telescope and thus we only need to focus on bounding the last
term. Applying the definitions of dual and matrix norms from Section 2 and using Young’s inequality,
we have that for any α > 0,
xk−2), yk

xk−2), v
xk−2), v

yk
⟩
yk−1

Mx(xk−1

Mx(xk−1

xk−2)

yk−1

−
−

−
−

−

−

⟩

⟨

⟩

.

⟨

−

−

Mx(xk−1
⟩ ≤ ∥
Mx
∗
≤ ∥
∥
Mx
∥
∥
2

−
xk−1
∥
(cid:16)
∗

α

∗
∥
xk−2

yk
∥
yk
∥∥
xk−2

yk−1
∥
yk−1
1
α ∥

−
−
2 +
∥

yk

−

yk−1

−

2(cid:17)
∥

.

(C.8)

−
xk−1
∥
Hence, combining (C.3)–(C.8), we can conclude that
ηk

ˆxk), v

gk, v

yk

≤

Mxk
⟨

−

−

⟩ ≤

My(xk
ηk
⟨
+ ηk

⟨
+ ηk−1 ∥

−
Mx(xk
Mx

yk
−
xk−1), v
My
∥

∥

∗

⟩ −
−
(cid:16)
α

−
∗ +
∥
2

ηk−1
yk

My(xk−1
⟨
ηk−1

ˆxk−1), v

yk−1

−
xk−2), v

⟩
yk−1

−
Mx(xk−1
1
α ∥

2 +
∥

⟨
xk−2

−
yk

⟩ −
xk−1
∥

−

−
2(cid:17)
∥

,

yk−1

−

⟩

completing the proof of the first claim.

Similarly, we observe from Algorithm 1 that

−

yk

(cid:0)M(:,t)(cid:1)⊤

k = (cid:0)M(1:t−1,t)(cid:1)⊤(cid:16)
h(t)
+ (cid:0)M(t:s,t)(cid:1)⊤(cid:16)
Observing that (cid:80)s
t=1 M1:t−1,t = Mx and (cid:80)s
ments as for bounding (C.3), we can conclude that for any β > 0,

ηk−1
ηk

yk−1 +

ˆyk +

yk

yk

−

−

(cid:17)(1:t−1)

(yk−1

ˆyk−1)

−

ηk−1
ηk

(cid:17)(t:s)

yk−2)

.

(yk−1

−

t=1 Mt:s,t = My, using the same sequence of argu-

ηk

ηk

ηk

−

≤ −

−

xk, M⊤yk

(cid:11)

hk

(cid:10)u
−
(cid:10)M⊤
x (yk
(cid:10)M⊤
y (yk
M⊤

−
ˆyk), u

−
yk−1), u

−

−
x + M⊤
y ∥
2

(cid:16)

∗

+ ηk−1 ∥

(cid:10)M⊤

(cid:11) + ηk−1
xk

(cid:11) + ηk−1

xk

−

xk−1

β

∥

xk

2 +
∥

−

x (yk−1
(cid:10)M⊤

−
y (yk−1
1
β ∥

yk−1

ˆyk−1), u

−
yk−2), u
2(cid:17)
∥

yk−2

,

−

−

(cid:11)

xk−1

(cid:11)

xk−1

−

completing the proof.

Proof Theorem 3.2. Recalling the definition of
k, by Lemma C.1,
E
ηk−1
My(xk−1
⟨
ηk−1
yk

ηk
My(xk
⟨
+ ηk

ˆxk), v

k
E

≤

ˆxk−), v

yk−1

−
xk−2), v

−
Mx(xk
⟨
Mx
+ ηk−1 ∥

yk
−
xk−1), v
My

∗
∥

∥

−
∗ +
∥
2

−

(cid:10)M⊤
(cid:10)M⊤

ηk

ηk

x (yk
y (yk
M⊤
x ∥

+ ηk−1 ∥

−
∗ +
2
Dψ(yk, yk−1)

−

−

−

ˆyk), u

−
yk−1), u
M⊤
∗
y ∥
∥

−
(cid:16)

−

Dϕ(xk, xk−1).

17

⟩
−
2(cid:17)

yk−1

⟩

⟩ −
−
(cid:16)
α

⟩ −
xk−1
∥
−
(cid:11) + ηk−1
xk
xk

(cid:11) + ηk−1

−
Mx(xk−1
1
α ∥

⟨
2 +
xk−2
∥
(cid:10)M⊤
x (yk−1
(cid:10)M⊤

−
yk

yk−1

−
ˆyk−1), u

∥

−
yk−2), u

−

β

xk−1
∥

−

xk

∥

2 +

yk−1

yk−2

∥

−

−
y (yk−1
1
β ∥

(C.9)

(cid:11)

xk−1

(cid:11)

xk−1

−
2(cid:17)

Recalling that ψ is cψ-strongly convex, ϕ is cϕ-strongly convex, setting α = β =

∥Mx∥∗+∥My∥∗+∥M⊤

y ∥∗

=

√

cϕcψ
µx+µy

, (C.9) simplifies to

(cid:113) cϕ
cψ

, and using

ηk−1
My(xk−1
⟨
yk
ηk−1
⟩ −
(cid:11) + ηk−1
xk

−
Mx(xk−1
⟨
(cid:10)M⊤
x (yk−1
(cid:10)M⊤

(cid:11) + ηk−1

ˆxk−), v

yk−1

−
xk−2), v
−
ˆyk−1), u

⟩
−

yk−1
⟩
(cid:11)
xk−1

−
yk−2), u

−

(cid:11)

xk−1

−

(C.10)

that ηk−1

k
E

≤

√

cϕcψ

ˆxk), v

x ∥∗+∥M⊤
yk
−
xk−1), v
ˆyk), u

⟩ −
−
xk

−
yk−1), u

≤
ηk
My(xk
⟨
+ ηk
ηk

ηk

−
Mx(xk
⟨
(cid:10)M⊤
x (yk
(cid:10)M⊤
y (yk
cψµy
2(µx + µy)
cϕµx
2(µx + µy)

−

−
+

−

−

−
(cid:0)

(cid:0)

−
yk−2

yk−1
∥

−

2
∥

yk

−

− ∥

yk−1

−
y (yk−1
2(cid:1)
∥
2(cid:1).
∥

+

xk−1
∥
Telescoping (C.10) and recalling that η0 = 0, we now have

xk−2

− ∥

xk

−

−

xk−1

2
∥

K
(cid:88)

k=1

k
E

≤

ηK

Mx(xK
⟨

−

xK−1), v

yK

−

⟩ −

ηK

(cid:10)M⊤

y (yK

+ ηK

My(xK
⟨
cψµy
2(µx + µy) ∥
cϕµx
2(µx + µy) ∥
,

−

−

ˆxK), v

−
yK

−
yK−1

xK

xK−1

−

−

ηK

(cid:10)M⊤

x (yK

⟩ −

yK

2

∥

∥

2.

yK−1), u

(cid:11)

xK

−

ˆyK), u

(cid:11)

xK

−

−

−

(C.11)

) is linear in both its arguments. Hence, Gapu,v(¯xK, ¯yK) =
·
k=1 ηkGapu,v(xk, yk). Applying Proposition 3.1 and combining with (C.11), we now have

(cid:80)K

Observe that Gapu,v(
·
1
HK
HKGapu,v(¯xk, ¯yk)

≤

Dϕ(u, x0) + Dψ(v, y0)
Mx(xK
+ ηK

xK−1), v

−

−
yK

ˆxK), v

−
yK−1

−
yK

2

yK

⟩ −

y (yK

(cid:10)M⊤
x (yK

ηK
(cid:10)M⊤

⟩ −
ηK
cϕµx
2(µx + µy) ∥

xK

−

yK−1), u

−
ˆyK), u

−
(cid:11)
xK

xK−1

−
2
∥

(cid:11)

xK

−

∥
Dψ(v, yK).
To complete bounding the gap, it remains to argue that the right-hand side of (C.12) is bounded
µx
Dψ(v, yK). This is done using the same
by Dϕ(u, x0) + Dψ(v, y0)
µx+µy
sequence of arguments as in bounding

k and is omitted.

Dϕ(u, xK)

µy
µx+µy

(C.12)

−

−

−

−

−

−

−

⟨

+ ηK

My(xK
⟨
cψµy
2(µx + µy) ∥
Dϕ(u, xK)

Let (x∗, y∗)
∈ X × Y
can conclude that
µx
µx + µy

Further, using that Dϕ(

E

µy
µx + µy
)
·

be any primal-dual solution to (PD). Then Gap(x∗,y∗)(¯xK, ¯yK)

0 and we

≥

Dϕ(x∗, xK) +

Dψ(y∗, yK)

Dϕ(x∗, x0) + Dψ(y∗, y0).

)
·

0, Dψ(
,
,
·
·
Gapu,v(¯xK, ¯yK) = sup

≥

≥

sup
u∈X ,v∈Y

≤
0, we can also conclude that

u∈X ,v∈Y{⟨
supu∈X ,v∈Y {

≤

M¯xK, v

Mu, ¯yK

⟩}

⟩ − ⟨

Dϕ(u, x0) + Dψ(v, y0)
}

.

HK

√

cϕcψ
µx+µy

Finally, setting ηk =
as HK = (cid:80)K
and solving for K.

for all k

1 immediately leads to the conclusion that HK = K

≥
k=1 ηk, by definition. The last bound is by setting

supu∈X ,v∈Y {Dϕ(u,x0)+Dψ(v,y0)}
HK

√

,

cϕcψ
µx+µy
ϵ,

≤

D Algorithm Implementation Details

In Algorithm 2, we present an implementation-specific version of ECyclicPDA, in order to make
it clear that our algorithm can be implemented without any extra computation compared to the

18

computation needed for gradient and prox updates in MP. Note that MP performs two gradient
computations and two prox computations per player, due to how it achieves “extrapolation”; we want
to argue that we perform an equivalent number operations as needed for a single gradient computation
and prox computation per player. Note that the overall complexity of first-order methods when
applied to EFGs is dominated by the gradient and prox update computations; this is why we compare
our algorithm to MP on this basis. The key differences from Algorithm 1 are that we explicitly use
ˆxk and ˆyk to represent the behavioral strategy that is computed via the partial prox updates (which
are then scaled at the end of a full iteration of our method to xk and yk), and that we use ˆhj
k and ˆgj
k
to accumulate gradient contributions from decision points that occur underneath j, to make the partial
prox update explicit.

In Lines 8 and 13, we are only dealing with the columns and rows, respectively, of the payoff matrix
that correspond to the current block number t, which means that as t ranges from 1 to s, for the
computation of the gradient, we will only consider each column and row, respectively, once, as would
have to be done in a full gradient computation for MP.

The more difficult aspect of the implementation is ensuring that we do the same number of operations
for the prox computation in ECyclicPDA as an analogous single prox computation in MP. We achieve
this by applying the updates in Proposition 2.2 only for the decision points in the current block, in
Lines 9 to 12 for x and 14 to 17 for y.

We focus on the updates for x; the argument is analogous for y. When applying this local prox
(t), we have already correctly computed ˆhj, the contributions to
update for decision point j
the gradient for the local prox update that originate from the children of j, again because the blocks
represent the treeplex ordering; in particular, whenever we have encountered a child decision point
of j in the past, we accumulate its contribution to the gradient for its parent at ˆhpj . Since the prox
update decomposition from Proposition 2.2 has to be applied for every single decision point in
X in
a full prox update (as done in MP), we again do not incur any dependence on the number of blocks.

∈ J

J

X

E Description of EFG Benchmarks

We provide game descriptions of the games we run our experiments on below. Our game descriptions
are adapted from Farina et al. [15]. In Table 2, we provide the number of sequences for player x (n),
the number of sequences for player y (m), and the number of leaves in the game (NNZ(M)).

Table 2: Number of sequences for both players and number of leaves for each game. These correspond
to the dimensions n and m of M, and the number of nonzero entries of M, respectively.

Game

Num. of x sequences Num. of y sequences Num. of leaves

Goofspiel (4 ranks)

Liar’s Dice

Leduc (13 ranks)

Battleship

21329

24571

6007

73130

21329

24571

6007

253940

13824

147420

98956

552132

E.1 Goofspiel (4 ranks)

Goofspiel is a card-based game that is a standard benchmark in the EFG-solving community [36].
In the version that we test on, there are 4 unique cards (ranks), and there are 3 copies of each rank,
divided into 3 separate decks. Each player gets a deck, and the third deck is known as the prize deck.
Cards are randomly drawn from the prize deck, and each player submits a bid for the drawn card by
submitting a card from one of their respective decks, the value of which represents their bid. Whoever
submits the higher bid wins the card from the prize deck. Once all the cards from the prize deck have
been drawn, bid on, and won by one of the players, the game terminates, and the payoffs for players
are given by the sum of the prize cards they won.

19

, y0

Algorithm 2 Extrapolated Cyclic Primal-Dual EFG Solver (Implementation Version)
1: Input: M, m, n
2: Initialization: x0
3: for k = 1 : K do
4:
5:
6:
7:
8:

η, Hk = Hk−1 + ηk
Choose ηk
gk = 0, hk = 0, ˆgk = 0, ˆhk = 0
(cid:101)xk = xk−1 + ηk−1
ηk
for t = 1 : s do

xk−2), (cid:101)yk = yk−1 + ηk−1

, η0 = H0 = 0, η =

cϕcψ
µx+µy

yk−2)

(yk−1

(xk−1

∈ X

∈ Y

−

−

≤

ηk

√

, ¯x0 = x0, ¯y0 = y0, g0 = 0, h0 = 0

(cid:110)(cid:68)ˆhj

k + hj

k, bj(cid:69)

+ Dϕj

(cid:16)

bj, ˆxj

k−1

(cid:17)(cid:111)

h↓j

k +

ϕ↓j (cid:16)

x↓j
k−1

(cid:17)(cid:17)

∇

−

ϕj (cid:16)

ˆxj
k−1

(cid:68)

(cid:17)

+

∇

ϕj (cid:16)

ˆxj
k−1

(cid:17)

, ˆxj

k−1

(cid:69)(cid:105)

(cid:110)(cid:68)

k + gj
ˆgj

k, bj(cid:69)

+ Dψj

(cid:16)

bj, ˆyj

k−1

(cid:17)(cid:111)

g↓j
k +

−

ψ↓j (cid:16)

y↓j
k−1

(cid:17)(cid:17)

∇

−

ψj (cid:16)

ˆyj
k−1

(cid:68)

(cid:17)

+

∇

ψj (cid:16)

ˆyj
k−1

(cid:17)

, ˆyj

k−1

(cid:69)(cid:105)

(cid:104)

(cid:101)yk

h(t)
k = (M(:,t))⊤
(t) do
for j
X
∈ J
ˆxj
k = argminbj ∈∆nj
(j′, a) = pj
ϕ↓j∗ (cid:16)
ˆhpj
k +=
g(t)
k = M(t,:)
(cid:101)xk
(t) do
for j
∈ J
ˆyj
k = argminbj ∈∆nj
(j′, a) = pj
ˆhpj
k +=

ψ↓j∗ (cid:16)

−

(cid:104)

Y

9:

10:

11:

12:

13:

14:

15:

16:

17:

(t) do
X
(cid:104)
kxpj
ˆxj
(t) do
Y
(cid:104)
kypj
ˆyj

ˆxj
k

19:

18:

20:

21:

for j
∈ J
(cid:101)xj
k =
for j
∈ J
(cid:101)yj
k =
for j
X do
∈ J
k = xpj
xj
k ·
for j
Y do
∈ J
k = ypj
yj
k ·
¯xk = Hk−ηk
26:
27: Return: ¯xK, ¯yK

24:
25:

22:
23:

Hk

ˆyj
k
¯xk−1 + ηk
Hk

k−1 + ηk−1
ηk

k−1 + ηk−1
ηk

(cid:16)

xj
k−1 −

(cid:16)

yj
k−1 −

k−1xpj
ˆxj

k−2

(cid:17)(cid:105)

k−1ypj
ˆyj

k−2

(cid:17)(cid:105)

xk, ¯yk = Hk−ηk

Hk

¯yk−1 + ηk
Hk

yk

E.2 Liar’s Dice

Liar’s Dice is another standard benchmark in the EFG-solving community [29]. In the version that
we test on, each player rolls an unbiased six-sided die, and they take turns either calling higher bids
or challenging the other player. A bid consists of a combination of a value v between one and six,
and a number of dice between one and two, n, representing the number of dice between the two
players that has v pips showing. A higher bid involves either increasing n holding v fixed, increasing
v holding n fixed, or both. When a player is challenged (or the highest possible bid of “two dice
each showing six pips” is called), the dice are revealed, and whoever is correct wins 1 (either the
challenger if the bid is not true, or the player who last called a bid, if the bid is true), and the other
player receives a utility of -1.

E.3 Leduc (13 ranks)

Leduc is yet another standard benchmark in the EFG-solving community [40] and is a simplified
version of Texas Hold’Em. In the version we test on, there are 13 unique cards (ranks), and there are
2 copies of each rank (half the size of a standard 52 card deck). There are two rounds of betting that
take place, and before the first round each player places an ante of 1 into the pot, and is dealt a single
pocket (private) card. In addition, two cards are placed face down, and these are community cards
that will be used to form hands. The two hands that can be formed with the community cards are pair,
and highest card.

20

During the first round of betting, player 1 acts first. There is a max of two raises allowed in each round
of betting. Each player can either check, raise, or fold. If a player folds, the other player immediately
wins the pots and the game terminates. If a player checks, the other player has an opportunity to
raise if they have not already previously checked or raised, and if they previously checked, the game
moves on to the next round. If a player raises, the other player has an opportunity to raise if they
have not already previously raised. The game then moves on the second round, during which one
of the community cards is placed face up, and then similar betting dynamics as the first round take
place. After the second round terminates, there is a showdown, and whoever can form the better hand
(higher ranked pair, or highest card) with the community cards takes the pot.

E.4 Battleship

This is an instantiation of the classic board game, Battleship, in which players take turns shooting
at the opposing player’s ships. Before the game begins, the players place two ships of length 2 and
value 4, on a grid of size 2 by 3. The ships need to be placed in a way so that the ships take up exactly
four spaces within the grid (they do not overlap with each other, and are contained entirely in the
grid). Each player gets three shots, and players take turns firing at cells of their opponent’s grid. A
ship is sunk when the two cells it has been placed on have been shot at. At the end of the game, the
utility for a player is the difference between the cumulative value of the opponent’s sunk ships and
the cumulative value of the player’s sunk ships.

F Block Construction Strategies

As discussed in the main paper, the postorder block construction strategy can be viewed as traversing
the decision points of the treeplex in postorder, treating decision points with the same parent sequence
as the same node, and then greedily putting decision points in the same block until we reach a decision
point that has a child decision point in the current block (at which point we start a new block). We
make this postorder traversal and greedy block construction explicit in Algorithm 3.

In Algorithm 4 we provide pseudocode for constructing blocks using the children block construction
strategy. As discussed in the main paper, the children block construction strategy corresponds to
placing decision points with the same parent decision point (same decision point at which their parent
sequences start at) in the same block. In our implementation, instead of doing a bottom-up traversal,
we do a top down implementation, and at the end, reverse the order of the blocks (this allows us to
respect the treeplex ordering).

In both Algorithm 3 and Algorithm 4,

represents the empty sequence.

∅

∈ C
for a′
accumulator.insert(postorder(j′, a′))

Aj′ do

∈

∈ C

for j′

j,a do

accumulator.insert(j′)

accumulator = []
for j′
j,a do

Algorithm 3 Postorder Block Construction
1: procedure POSTORDERHELPER(j, a)
2:
3:
4:
5:
6:
7:
Return: accumulator
8:
9: procedure POSTORDERBLOCKS(
10:
11:
12:
13:
14:
15:
16:
17:
18:
19:

)
J
ordered = POSTORDERHELPER(
∅
blocks = []
current_block = []
for j
ordered do
∈
j′
if
∃
blocks.insert(current_block)
current_block = [j]

current_block.insert(j)

return: blocks

else

∈

)

current_block s.t. j′ is a child decision point of j then

21

)

J

blocks = []
explore =
for j

Algorithm 4 Children Block Construction
1: procedure CHILDRENBLOCKS(
2:
3:
4:
5:
6:
7:
8:
9:

C∅
explore do
∈
current_block = []
for a

current_block.insert(j’)
explore.insert(j’)

∈
for j′

Aj do

j,a do

∈ C

blocks.insert(current_block)

10:

return: blocks.reverse()

We can now illustrate each of the block construction strategies on the treeplex for player 1 in Kuhn that
X =
0, 1, 2, 3, 4, 5, 6
was presented in Appendix B. If we use single block, then we have
.
}
{
i
If we use infosets, then we have
(we have to subtract in
}
order to label the infosets in a manner that respects the treeplex ordering). If we use children, then
,
4
,
6
we have
. If we use
X
J
}
J
{
}
(2) =
postorder, then we have
J
{

7
−
{
(2) =
(3) =
5
,
X
{
J
}
(1) =
4, 5, 6
,
}
{

∈ {
(4) =
1, 2, 3
X
}
{
1, 2, 3
, and
X
J
}

(1) =
X
J
1, 2, 3, 4, 5, 6, 7

, and
J
(3) =
{

(5) =
X
0
.
}

0
}
{

(1) =

(i) =

for i

J
}

J

J

J

{

X

X

X

X

Note that in the implementation of our algorithm, it is not actually important that the number of
blocks for both players are the same; if one player has more blocks than the other, for iterations of
our algorithm that correspond to block numbers that do not exist for the other player, we just do not
do anything for the other player. Nevertheless, the output of the algorithm does not change if we
combine all the blocks for the player with more blocks after the minimum number of blocks between
the two players is exceeded, into one block. For example, if player 1 has s1 blocks, and player 2
has s2 blocks, with s1 < s2, we can actually combine blocks s1 + 1, . . . , s2 all into the same block
for player 2, and this would not change the execution of the algorithm. This is what we do in our
implementation.

Additionally, given a choice of a partition of decision points into blocks, there may exist many
permutations of decision points within the blocks which satisfy the treeplex ordering of the decision
points. Unless the game that is being tested upon possesses some structure which leads to a single
canonical ordering of the decision points (which respects the treeplex ordering), an arbitrary decision
needs to be made regarding what order is used.

G Experiments

G.1 Additional Experimental Details

Block Construction Strategy Comparison. In this section, we provide additional plots (Figures 6
to 14) comparing different block construction strategies for our algorithm, for specific choices of
regularizer and averaging scheme. Note that for the games for which there is a benefit to using
blocks (Liar’s Dice and Battleship), the benefit is generally apparent across different regularizers and
averaging schemes. Furthermore, when there is not a benefit for a particular regularizer and averaging
scheme, there is no significant cost either (using blocks does not lead to worse performance).

Block Construction Strategy Comparison with Restarts. We repeat a similar analysis as above
(comparing the block construction strategies holding a regularizer and averaging scheme fixed) but
this time with the adaptive restarting heuristic applied to our algorithm: the plots can be seen in
(Figures 15 to 23). As mentioned in Section 4, we prepend the algorithm name with “r” to denote the
restarted variant of the algorithm in the plots.

As discussed in the main body, the trend of the benefit of using blocks being more pronounced
with restarting (for games for which blocks are beneficial) holds generally even when holding the
regularizer and averaging scheme fixed. This can be seen by comparing each of the restarted block
construction strategy comparison plots with the corresponding non-restarted block construction
strategy comparison plot.

22

Figure 6: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies when using the dilated entropy regularizer
and uniform averaging.

Figure 7: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies when using the dilatable global entropy
regularizer and uniform averaging.

Figure 8: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies when using the dilated ℓ2 regularizer and
uniform averaging.

Figure 9: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies when using the dilated entropy regularizer
and linear averaging.

23

102104Gradientcomputations10−210−1100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−210−1100Liar’sDice102104Gradientcomputations10−210−1100Leduc(13ranks)102104Gradientcomputations10−1100BattleshipECyclicPDAsingleblockECyclicPDApostorderECyclicPDAchildrenECyclicPDAinfosets1102104Gradientcomputations10−210−1100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−210−1100Liar’sDice102104Gradientcomputations10−2100Leduc(13ranks)102104Gradientcomputations10−210−1100BattleshipECyclicPDAsingleblockECyclicPDApostorderECyclicPDAchildrenECyclicPDAinfosets1102104Gradientcomputations10−210−1100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−2100Liar’sDice102104Gradientcomputations10−2100Leduc(13ranks)102104Gradientcomputations10−210−1100BattleshipECyclicPDAsingleblockECyclicPDApostorderECyclicPDAchildrenECyclicPDAinfosets1102104Gradientcomputations10−210−1100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−3100Liar’sDice102104Gradientcomputations10−210−1100Leduc(13ranks)102104Gradientcomputations10−1100BattleshipECyclicPDAsingleblockECyclicPDApostorderECyclicPDAchildrenECyclicPDAinfosets1Figure 10: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies when using the dilatable global entropy
regularizer and linear averaging.

Figure 11: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies when using the dilated ℓ2 regularizer and
linear averaging.

Figure 12: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies when using the dilated entropy regularizer
and quadratic averaging.

Figure 13: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies when using the dilatable global entropy
regularizer and quadratic averaging.

24

102104Gradientcomputations10−2100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−2100Liar’sDice102104Gradientcomputations10−310−1Leduc(13ranks)102104Gradientcomputations10−3100BattleshipECyclicPDAsingleblockECyclicPDApostorderECyclicPDAchildrenECyclicPDAinfosets1102104Gradientcomputations10−2100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−2100Liar’sDice102104Gradientcomputations10−210−1100Leduc(13ranks)102104Gradientcomputations10−3100BattleshipECyclicPDAsingleblockECyclicPDApostorderECyclicPDAchildrenECyclicPDAinfosets1102104Gradientcomputations10−210−1100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−410−1Liar’sDice102104Gradientcomputations10−210−1100Leduc(13ranks)102104Gradientcomputations10−1100BattleshipECyclicPDAsingleblockECyclicPDApostorderECyclicPDAchildrenECyclicPDAinfosets1102104Gradientcomputations10−2100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−2100Liar’sDice102104Gradientcomputations10−310−1Leduc(13ranks)102104Gradientcomputations10−510−2BattleshipECyclicPDAsingleblockECyclicPDApostorderECyclicPDAchildrenECyclicPDAinfosets1Figure 14: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies when using the dilated ℓ2 regularizer and
quadratic averaging.

Figure 15: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies when using the dilated entropy regularizer
and uniform averaging as well as restarting. We take the best duality gap seen so far so that the plot
is monotonic.

Figure 16: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies when using the dilatable global entropy
regularizer and uniform averaging as well as restarting. We take the best duality gap seen so far so
that the plot is monotonic.

Figure 17: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies when using the dilated ℓ2 regularizer and
uniform averaging as well as restarting. We take the best duality gap seen so far so that the plot is
monotonic.

25

102104Gradientcomputations10−2100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−2100Liar’sDice102104Gradientcomputations10−210−1100Leduc(13ranks)102104Gradientcomputations10−410−1BattleshipECyclicPDAsingleblockECyclicPDApostorderECyclicPDAchildrenECyclicPDAinfosets1102104Gradientcomputations10−210−1100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−1010−4Liar’sDice102104Gradientcomputations10−1100Leduc(13ranks)102104Gradientcomputations10−1100BattleshiprECyclicPDAsingleblockrECyclicPDApostorderrECyclicPDAchildrenrECyclicPDAinfosets1102104Gradientcomputations10−2100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−2100Liar’sDice102104Gradientcomputations10−2100Leduc(13ranks)102104Gradientcomputations10−910−3BattleshiprECyclicPDAsingleblockrECyclicPDApostorderrECyclicPDAchildrenrECyclicPDAinfosets1102104Gradientcomputations10−210−1100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−2100Liar’sDice102104Gradientcomputations10−210−1100Leduc(13ranks)102104Gradientcomputations10−3100BattleshiprECyclicPDAsingleblockrECyclicPDApostorderrECyclicPDAchildrenrECyclicPDAinfosets1Figure 18: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies when using the dilated entropy regularizer
and linear averaging as well as restarting. We take the best duality gap seen so far so that the plot is
monotonic.

Figure 19: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies when using the dilatable global entropy
regularizer and linear averaging as well as restarting. We take the best duality gap seen so far so that
the plot is monotonic.

Figure 20: Duality gap as a function of the number of full (or equivalent) gradient computations
for ECyclicPDA with different block construction strategies when using the dilated ℓ2 regularizer
and linear averaging as well as restarting. We take the best duality gap seen so far so that the plot is
monotonic.

Figure 21: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies when using the dilated entropy regularizer
and quadratic averaging as well as restarting. We take the best duality gap seen so far so that the plot
is monotonic.

26

102104Gradientcomputations10−210−1100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−1010−4Liar’sDice102104Gradientcomputations10−1100Leduc(13ranks)102104Gradientcomputations10−1100BattleshiprECyclicPDAsingleblockrECyclicPDApostorderrECyclicPDAchildrenrECyclicPDAinfosets1102104Gradientcomputations10−2100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−2100Liar’sDice102104Gradientcomputations10−2100Leduc(13ranks)102104Gradientcomputations10−1010−4BattleshiprECyclicPDAsingleblockrECyclicPDApostorderrECyclicPDAchildrenrECyclicPDAinfosets1102104Gradientcomputations10−210−1100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−2100Liar’sDice102104Gradientcomputations10−210−1100Leduc(13ranks)102104Gradientcomputations10−3100BattleshiprECyclicPDAsingleblockrECyclicPDApostorderrECyclicPDAchildrenrECyclicPDAinfosets1102104Gradientcomputations10−210−1100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−1010−4Liar’sDice102104Gradientcomputations10−1100Leduc(13ranks)102104Gradientcomputations10−1100BattleshiprECyclicPDAsingleblockrECyclicPDApostorderrECyclicPDAchildrenrECyclicPDAinfosets1Figure 22: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies when using the dilatable global entropy
regularizer and quadratic averaging as well as restarting. We take the best duality gap seen so far so
that the plot is monotonic.

Figure 23: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA with different block construction strategies when using the dilated ℓ2 regularizer and
quadratic averaging as well as restarting. We take the best duality gap seen so far so that the plot is
monotonic.

Regularizer Comparison. In this section (Figures 24 to 26) we compare the performance of
ECyclicPDA and MP instantiated with different regularizers for each averaging scheme, against the
performance of CFR+ and PCFR+.

It is apparent from these plots, that our algorithm generally outperforms MP, holding the averaging
scheme and regularizer fixed. This can be seen by examining the corresponding figure for a choice of
averaging scheme, and noting that for any given regularizer, the corresponding MP line is generally
above the corresponding ECyclicPDA line.

Regularizer Comparisons with Restarts. We repeat a similar analysis in this section (Figures 27
to 29), instead now comparing the performance of ECyclicPDA and MP instantiated with different
regularizers for each averaging scheme, against the performance of CFR+ and PCFR+, when all
methods are restarted. The trend noted above of our method generally beating MP, even holding the
regularizer and averaging scheme fixed, still holds even when restarting.

Figure 24: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA, MP, CFR+, PCFR+, using a uniform averaging scheme for ECyclicPDA and MP.

27

102104Gradientcomputations10−2100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−310−1Liar’sDice102104Gradientcomputations10−2100Leduc(13ranks)102104Gradientcomputations10−1010−4BattleshiprECyclicPDAsingleblockrECyclicPDApostorderrECyclicPDAchildrenrECyclicPDAinfosets1102104Gradientcomputations10−210−1100DualitygapGoofspiel(4ranks)102104Gradientcomputations10−2100Liar’sDice102104Gradientcomputations10−210−1100Leduc(13ranks)102104Gradientcomputations10−3100BattleshiprECyclicPDAsingleblockrECyclicPDApostorderrECyclicPDAchildrenrECyclicPDAinfosets1102104Gradientcomputations10−410−1DualitygapGoofspiel(4ranks)102104Gradientcomputations10−610−2Liar’sDice102104Gradientcomputations10−2101Leduc(13ranks)102104Gradientcomputations10−510−1BattleshipECyclicPDAdilatedentropyECyclicPDAdilatableglobalentropyECyclicPDAdilated‘2MPdilatedentropyMPdilatableglobalentropyMPdilated‘2CFR+PCFR+1Figure 25: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA, MP, CFR+, PCFR+, using a linear averaging scheme for ECyclicPDA and MP.

Figure 26: Duality gap as a function of the number of full (or equivalent) gradient computations for
ECyclicPDA, MP, CFR+, PCFR+, using a quadratic averaging scheme for ECyclicPDA and MP.

Figure 27: Duality gap as a function of the number of full (or equivalent) gradient computations for
when restarting is applied to ECyclicPDA, MP, CFR+, PCFR+, using a uniform averaging scheme
for ECyclicPDA and MP. We take the best duality gap seen so far so that the plot is monotonic.

Figure 28: Duality gap as a function of the number of full (or equivalent) gradient computations for
when restarting is applied to ECyclicPDA, MP, CFR+, PCFR+, using a linear averaging scheme for
ECyclicPDA and MP. We take the best duality gap seen so far so that the plot is monotonic.

28

102104Gradientcomputations10−410−1DualitygapGoofspiel(4ranks)102104Gradientcomputations10−610−2Liar’sDice102104Gradientcomputations10−2101Leduc(13ranks)102104Gradientcomputations10−510−1BattleshipECyclicPDAdilatedentropyECyclicPDAdilatableglobalentropyECyclicPDAdilated‘2MPdilatedentropyMPdilatableglobalentropyMPdilated‘2CFR+PCFR+1102104Gradientcomputations10−410−1DualitygapGoofspiel(4ranks)102104Gradientcomputations10−610−2Liar’sDice102104Gradientcomputations10−2101Leduc(13ranks)102104Gradientcomputations10−510−1BattleshipECyclicPDAdilatedentropyECyclicPDAdilatableglobalentropyECyclicPDAdilated‘2MPdilatedentropyMPdilatableglobalentropyMPdilated‘2CFR+PCFR+1102104Gradientcomputations10−1010−4DualitygapGoofspiel(4ranks)102104Gradientcomputations10−1010−4Liar’sDice102104Gradientcomputations10−2101Leduc(13ranks)102104Gradientcomputations10−1010−4BattleshiprECyclicPDAdilatedentropyrECyclicPDAdilatableglobalentropyrECyclicPDAdilated‘2rMPdilatedentropyrMPdilatableglobalentropyrMPdilated‘2rCFR+rPCFR+1102104Gradientcomputations10−1010−4DualitygapGoofspiel(4ranks)102104Gradientcomputations10−1010−4Liar’sDice102104Gradientcomputations10−2101Leduc(13ranks)102104Gradientcomputations10−1010−4BattleshiprECyclicPDAdilatedentropyrECyclicPDAdilatableglobalentropyrECyclicPDAdilated‘2rMPdilatedentropyrMPdilatableglobalentropyrMPdilated‘2rCFR+rPCFR+1Figure 29: Duality gap as a function of the number of full (or equivalent) gradient computations for
when restarting is applied to ECyclicPDA, MP, CFR+, PCFR+, using a quadratic averaging scheme
for ECyclicPDA and MP. We take the best duality gap seen so far so that the plot is monotonic.

Additional wall-clock time experiments In Table 3, we show the wall-clock time required to reach
a duality gap of 10−4. It is clear that our algorithm is competitive with MP: MP and its restarted
variant time out on Leduc, and MP times out on Battleship (while our algorithm does not even take
close to 30 seconds). Furthermore, we are outperforming CFR+ and its restarted variant in Battleship.

Table 3: The wall clock time in seconds required for ECyclicPDA, MP, CFR+, and PCFR+, and their
restarted variants, denoted using an “r” at the front of the algorithm name, to reach a duality gap of
10−4. We let each algorithm run for at most 30 seconds; a value of 30.000 means the algorithm could
not reach the target gap in 30 seconds. The duality gap is computed every 100 iterations.

Name
ECyclicPDA

rECyclicPDA

MP

rMP
CFR+
rCFR+
PCFR+
rPCFR+

Goofspiel (4 ranks) Liar’s Dice Leduc (13 ranks) Battleship

8.797

6.688

3.183

2.691

5.869

5.866

0.244

0.208

6.148

3.519

3.930

2.360

0.111

0.057

0.067

0.077

9.343

10.724

30.000

30.000

0.236

0.161

0.225

0.197

16.818

4.914

30.000

8.374

6.001

6.042

0.736

0.736

29

102104Gradientcomputations10−1010−4DualitygapGoofspiel(4ranks)102104Gradientcomputations10−1010−4Liar’sDice102104Gradientcomputations10−2101Leduc(13ranks)102104Gradientcomputations10−1010−4BattleshiprECyclicPDAdilatedentropyrECyclicPDAdilatableglobalentropyrECyclicPDAdilated‘2rMPdilatedentropyrMPdilatableglobalentropyrMPdilated‘2rCFR+rPCFR+1