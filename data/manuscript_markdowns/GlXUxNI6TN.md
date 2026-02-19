Abductive Reasoning in Logical Credal Networks

Radu Marinescu
IBM Research, Ireland
radu.marinescu@ie.ibm.com

Junkyu Lee
IBM Research, USA
junkyu.lee@ibm.com

Debarun Bhattacharjya
IBM Research, USA
debarunb@us.ibm.com

Fabio Cozman
Universidade de São Paulo, Brazil
fgcozman@usp.br

Alexander Gray
Centaur AI Institute, USA
alexander.gray@centaurinstitute.org

Abstract

Logical Credal Networks or LCNs were recently introduced as a powerful proba-
bilistic logic framework for representing and reasoning with imprecise knowledge.
Unlike many existing formalisms, LCNs have the ability to represent cycles and
allow specifying marginal and conditional probability bounds on logic formulae
which may be important in many realistic scenarios. Previous work on LCNs has
focused exclusively on marginal inference, namely computing posterior lower and
upper probability bounds on a query formula. In this paper, we explore abductive
reasoning tasks such as solving MAP and Marginal MAP queries in LCNs given
some evidence. We first formally define the MAP and Marginal MAP tasks for
LCNs and subsequently show how to solve these tasks exactly using search-based
approaches. We then propose several approximate schemes that allow us to scale
MAP and Marginal MAP inference to larger problem instances. An extensive em-
pirical evaluation demonstrates the effectiveness of our algorithms on both random
LCN instances as well as LCNs derived from more realistic use-cases.

1

Introduction

Probabilistic logic which combines probability and logic in a principled manner has emerged over
the past decades as a unified representational and reasoning framework capable of dealing effectively
with complex real-world applications that require efficient handling of uncertainty and compact
representations of domain expert knowledge [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]. Logical Credal Networks or
LCNs [11] were introduced recently as a probabilistic logic designed for representing and reasoning
with imprecise knowledge. Unlike many existing probabilistic logics, LCNs have the ability to
represent cycles (e.g., feedback loops) as well as allow specifying marginal and conditional probability
bounds on logic formulae which may be important in many realistic usecases.

Up until now, the work on LCNs has focused exclusively on marginal inference, i.e. efficiently
computing posterior lower and upper probability bounds on a query formula. However, abductive
reasoning tasks such as explaining the evidence observed in an LCN are equally important in many
real-world applications. In probabilistic graphical models, these tasks are commonly known as MAP
and Marginal MAP (MMAP) inference and have received extensive attention over the past decades
[12, 13]. They are typically tackled efficiently with dynamic programming (e.g., variable elimination)
or heuristic search (e.g., depth-first branch and bound) based algorithms [13, 14, 15, 16].

38th Conference on Neural Information Processing Systems (NeurIPS 2024).

Contribution.
In this paper, we consider solving MAP and Marginal MAP inference queries
in LCNs. Unlike in graphical models, an LCN encodes a set of probability distributions over its
interpretations. Therefore, a complete or a partial explanation of the evidence which represents a
complete or a partial truth assignment to the LCN’s propositions may correspond to more than one
distribution. Our work builds on very recent work on Marginal MAP inference for credal networks, a
class of probabilistic graphical models that allow reasoning with imprecise probabilities [17]. We
formally introduce the MAP and Marginal MAP tasks for LCNs as finding a complete or a partial
truth assignment to the LCN’s propositions with maximum lower (respectively, upper) probability,
given some evidence in the LCN. We show how to evaluate such MAP assignments using exact
marginal inference for LCNs and, subsequently, propose several search schemes based on depth-first
search, limited discrepancy search and simulated annealing to solve these tasks in practice. We then
extend a recent message-passing scheme for approximate marginal inference in LCNs [18] to handle
effectively the MAP and MMAP inference tasks in LCNs as well as adapt the limited discrepancy
search and simulated annealing methods to use an approximate evaluation of the MAP assignments
during search. We experiment and evaluate our proposed exact and approximate algorithms on several
classes of LCNs including random as well as more realistic LCN instances. Our results show that
the search methods based on exact evaluation of the MAP assignments are limited to solving small
size problems in practice, while the approximate message-passing scheme and, to some extent, the
approximate search-based methods can scale to much larger problem instances. This is important
because it allows us to tackle practical problems involving hundreds and possibly many thousands of
propositions. The supplementary material includes additional details and experiments.

2 Background

We provide next a brief overview of basic concepts about LCNs and marginal inference in these
models. Throughout the paper we will use the following notations. Logical propositions are denoted
by uppercase letters (e.g., A, B, C, ...) while for sets of propositions we use boldfaced uppercase
letters (e.g., A, B, C, ...). Truth assignments to propositions (i.e., literals) are denoted by either
lowercase or uppercase letters, namely we use a or A to indicate that proposition A holds true, and
¬a or ¬A if A is false. Sets of literals are denoted by boldfaced lowercase letters (e.g., a, b, c, ...).

2.1 Logical Credal Networks

A Logical Credal Network (LCN) [11] is defined by a tuple L = ⟨A, C⟩, where A = {A1, . . . , An}
is a set of propositions (or atoms), and C is a set of probability labeled sentences (or constraints)
having the following two forms:

α ≤ P (ϕ) ≤ β
α ≤ P (ϕ|φ) ≤ β

(1)
(2)

Here, ϕ and φ are arbitrary propositional logic formulae1 involving propositions in A and logical
connectives such as negation, disjunction and conjunction, and 0 ≤ α ≤ β ≤ 1 are lower and upper
probability bounds, respectively.

An LCN is associated with primal graph which is a directed graph G containing formula nodes and
proposition nodes, as well as directed edges from each proposition node in a formula ϕ to the formula
node representing ϕ (for type 1 sentences), and directed edges from each of the proposition nodes
in φ to φ, a directed edge from φ to ϕ, and bi-directed edges from ϕ to the proposition nodes in ϕ,
respectively (for type 2 sentences) [11]. A parent of a proposition A in G is a proposition B such that
there is a directed path in G from B to A in which all intermediate nodes are formulae. A descendant
of a proposition A in G is a proposition B such that there is a directed path in G from A to B in
which no intermediate node is a parent of A [11].

An LCN is endowed with a Local Markov Condition (LMC) where a proposition node A is indepen-
dent, given its parents, of all proposition nodes that are not A itself nor descendants of A nor parents
of A [11]. Therefore, an LCN represents a set of probability distributions over all interpretations of
its formulae that satisfy the constraints represented by the type (1) and (2) sentences as well as the
constraints induced by the independence relations given by the local Markov condition [11].

1The original definition of LCNs allows for relational structures and first-order logic formulae, but their

semantics is obtained by grounding on finite domains thus yieling a propositional LCN [11].

2

(a) LCN sentences

(b) Primal graph

Figure 1: A simple LCN and its primal graph.

Example 1. Figure 1 describes a simple LCN whose sentences shown in Figure 1a state that:
Bronchitis (B) is more likely than Smoking (S); Smoking may cause Cancer (C) or Bronchitis;
Dyspnea (D) or shortness of breadth is a common symptom for Cancer and Bronchitis; in case
of Cancer we have either a positive X-Ray result (X) and Dyspnea, or a negative X-Ray and no
Dyspnea. Figure 1b shows the primal graph where the formula and proposition nodes are displayed
as rectangles and shaded circles, respectively.

2.2 Marginal Inference in Logical Credal Networks

(3)

(6)

(5)

(7)

(4)

m
(cid:88)

pi = 1

i=1
pi ≥ 0, ∀i = 1, . . . , m
α ≤ ⃗Iϕ ⊙ ⃗p ≤ β
α · ⃗Iφ ⊙ ⃗p ≤ ⃗Iϕ∧φ ⊙ ⃗p ≤ β · ⃗Iφ ⊙ ⃗p
(⃗Ia ⊙ ⃗p) · (⃗Ib ⊙ ⃗p) − (⃗Ic ⊙ ⃗p) · (⃗Id · ⃗p) = 0
minimize/maximize ⃗Iψ ⊙ ⃗p

Given an LCN L with n propositions, the
marginal inference task is to compute lower
and upper bounds on the posterior probabil-
ity P (ψ) of a query formula ψ, which we de-
note by P (ψ) and P (ψ), respectively. This
is achieved by solving a non-linear program
given by Equations (3)–(8) and defined by
a set of non-negative real-valued variables
representing the probabilities of L’s inter-
pretations, a set of linear constraints derived
from L’s sentences, a set of non-linear con-
straints corresponding to the independence
assumptions given by the local Markov condition, and a linear objective function encoding the
query P (ψ) which is minimized and maximized to yield the desired bounds. More specifically, let
⃗p = (p1, . . . , pm) be the vector of real-valued variables representing the probabilities of L’s inter-
pretations, where m = 2n, and let ⃗Iϕ = (aϕ
m) be a binary vector, called an indicator vector,
such that aϕ
i is 1 if formula ϕ is true in the i-th interpretation and 0 otherwise. Since the probability
of a formula ϕ is the sum of the probabilities of the interpretations in which ϕ is true, we can write
P (ϕ) as ⃗Iϕ ⊙ ⃗p where ⊙ is the dot-product of two vectors. Therefore, Equations (3) and (4) ensure
that ⃗p is a valid probability distribution, Equations (5) and (6) encode the type (1) and (2) sentences in
L while Equation 7 encodes the conditional independencies of the form P (Xj|Sj, Tj) = P (Xj|Sj),
where Xj is a proposition, Sj = {Sj1, . . . , Sjk} and Tj = {Tj1, . . . , Tjl} are Xj’s parents and
non-descendants in the primal graph of L, ⃗Iϕ and ⃗Iϕ∧φ are the indicator vectors for formulae ϕ and
ϕ ∧ φ involved in L’s sentences, and ⃗Ia, ⃗Ib, ⃗Ic and ⃗Id are the indicator vectors corresponding to the
formulae a = (xj ∧ sj1 ∧ · · · ∧ sjk ∧ tj1 ∧ · · · ∧ tjl), b = (sj1 ∧ · · · ∧ sjk), c = (xj ∧ sj1 ∧ · · · ∧ sjk),
and d = (sj1 ∧ · · · ∧ sjk ∧ tj1 ∧ · · · ∧ tjl), respectively (see also [11] for more details).

1 , . . . , aϕ

(8)

3 MAP and Marginal MAP Inference in LCNs

Maximum A Posteriori (MAP) and Marginal MAP (MMAP) inference are well known abductive
reasoning tasks in probabilistic graphical models such as Bayesian networks and Markov networks
[12, 13, 14, 15, 16]. Specifically, the MAP task calls for finding a complete assignment to all variables
having maximum probability, given the evidence. Marginal MAP generalizes MAP and looks for
a partial variable assignment that has maximum marginal probability, given the evidence. MAP

3

and MMAP inference tasks appear in many real-world applications such as diagnosis, abduction
and explanation and are typically tackled with dynamic programming (e.g., variable elimination) or
heuristic search (e.g., depth-first branch and bound) based algorithms [13, 14, 15, 16].

In this section, we present our novel approach for solving the MAP and Marginal MAP inference
tasks in Logical Credal Networks. Unlike in graphical models, a (partial) variable assignment (or
interpretation) in an LCN may correspond to more than one distribution. Therefore, we begin by
formally defining two MAP and MMAP inference tasks for LCNs, called maximin MAP (resp.
maximin MMAP) and maximax MAP (resp. maximax MMAP). Subsequently, we develop several
exact and approximation schemes for solving these tasks efficiently in practice.

3.1 The MAP and Marginal MAP Tasks in LCNs

Let L = ⟨A, C⟩ be an LCN with n propositions and let E = {E1, . . . , Ek} ⊆ A be a subset
of k propositions, called evidence, for which the truth values e = {e1, . . . , ek} are known. Let
Y = {Y1, . . . , Ym} ⊆ A \ E be a subset of m propositions called MAP propositions. A truth
assignment to Y is is called a MAP assignment and is denoted by y = {y1, . . . , ym}, respectively.
Clearly, if Y = A \ E (i.e., m = n − k) then we have a MAP task, otherwise we have a MMAP task
(i.e., m < n − k). The maximin and maximax MAP/MMAP tasks are defined as follows:

Definition 1 (maximin). Given an LCN L with n propositions, evidence e, and MAP propositions Y,
the maximin MAP (or maximin MMAP if m < n − k) task is finding a truth assignment y∗ to Y
having maximum lower probability, given evidence e, namely:

y∗ = argmax
y∈Ω(Y)

P (ψy∧e)

(9)

where Ω(Y) is the set of all truth assignments to the MAP propositions, and ψy∧e = y1 ∧ · · · ∧ ym ∧
e1 ∧ · · · ∧ ek is the conjunction of the literals in y and e, respectively.
Definition 2 (maximax). Given an LCN L with n propositions, evidence e, and MAP propositions
Y, the maximax MAP (or maximax MMAP if m < n − k) task is finding a truth assignment y∗ to
Y having maximum upper probability, given evidence e, namely:

y∗ = argmax
y∈Ω(Y)

P (ψy∧e)

(10)

where Ω(Y) is the set of all truth assignments to the MAP propositions, and ψy∧e = y1 ∧ · · · ∧ ym ∧
e1 ∧ · · · ∧ ek is the conjunction of the literals in y and e, respectively.

3.2 Search Algorithms Using Exact MAP Assignment Evaluations

We present next three search-based schemes for solving the MAP and MMAP tasks in LCNs. These
methods employ different search strategies for exploring the search space defined by the MAP
propositions while evaluating exactly each complete or partial MAP assignment.

Exact Evaluation of a MAP Assignment. Clearly, computing the lower and upper probabilities
P (ψy∧e) and P (ψy∧e) of a MAP assignment y given evidence e can be done easily by minimizing
and, respectively maximizing the non-linear program defined by Equations (3)–(8), where the query
formula is the conjunction of positive or negative literals in y and e, namely ψy∧e = y1 ∧ · · · ∧ ym ∧
e1 ∧ · · · ∧ ek. Therefore, evaluating a MAP assignment in case of both MAP and Marginal MAP
inference in LCNs is quite difficult as it involves solving a maginal inference problem for LCNs
which is know to be NP-hard [11]. This is in contrast with graphical models where, at least for MAP
inference, the evaluation of a MAP assignment is linear in the number of variables [13].

Example 2. For illustration, consider the LCN example from Figure 1 and assume that we have
evidence e = {x, ¬s}, namely a patient has a positive X-Ray result (X = x) and is not smoking
(S = ¬s). The MAP propositions in this case are Y = {B, C, D} and the MAP assignment
y = (b, ¬c, ¬d) corresponds to the query formula ψy∧e = b ∧ ¬c ∧ ¬d ∧ x ∧ ¬s. The lower and
upper probabilities P (ψy∧e) and P (ψy∧e) of the MAP assignment are 9.9e-09 and 0.1, respectively.

4

Algorithm 1 Depth-First Search for MAP and Marginal MAP Inference in LCNs

initialize y∗ ← ∅, best ← −∞

1: procedure DFS(L = ⟨A, C⟩, E = e, Y)
2:
3: SEARCH(∅, Y)
return y∗
4:
5: procedure SEARCH(y, Y)
6:
7:
8:
9:

if size(y) == size(Y) then
if maximin then
score(y) ← P (ψy∧e)
else

10:

11:
12:
13:
14:
15:
16:
17:

score(y) ← P (ψy∧e)
if score(y) > best then
y∗ ← y, best ← score(y)

else
select unassigned proposition Yi ∈ Y
for all values y ∈ {yi, ¬yi} do
y ← y ∪ {Yi = y}
SEARCH(y, Y)

Algorithm 2 Limited Discrepancy Search for MAP and Marginal MAP Inference in LCNs

1: procedure LDS(L = ⟨A, C⟩, E = e, Y, δ)
initialize y0 randomly and let y∗ ← y0
2:
best ← score(y∗)
3:
for all θ = 1 . . . δ do
4:
SEARCH(y∗, Y, θ, 1)
5:
return y∗, best
6:
7: procedure SEARCH(y, Y, θ, i)
if θ == 0 or i > |Y| then
8:
if maximin then
9:
score(y) ← P (ψy∧e)
10:
else
11:

12:

13:
14:
15:
16:
17:
18:
19:
20:
21:
22:

score(y) ← P (ψy∧e)
if score(y) > best then
y∗ ← y, best ← score(y)

else
for all values y ∈ {yi, ¬yi} do
if y[i] == y then
z ← SEARCH(y, Y, i + 1, θ)
else
y′ ← y; y′[i] ← y
z ← SEARCH(y′, Y, i + 1, θ − 1)
return z

Depth-First Search. Our first approach for solving the MAP and MMAP tasks, called DFS, is
described by Algorithm 1. It takes as input an LCN L = ⟨A, C⟩, evidence E = e and a set of
MAP propositions Y ⊆ A \ E and outputs the optimal MAP assignment y∗. The method conducts
a depth-first search over the space of partial assignments to the MAP propositions, and, for each
complete MAP assignment y computes its score as the exact lower probability P (ψy∧e) for maximin
tasks, and respectively, the upper probability P (ψy∧e) for maximax tasks, given the evidence e. This
way, the optimal solution y∗ corresponds to the MAP assignment with the highest score.

Theorem 1 (complexity). Given an LCN L = ⟨A, C⟩ with n propositions, evidence E = e and MAP
propositions Y ⊆ A \ E, algorithm DFS is sound and complete. The time and space complexity of
the algorithm is O(2m+2n
) and O(2n), respectively, where m is the number of MAP propositions.

Example 3. Consider again the LCN from Figure 1 with evidence e = {x, ¬s}. In this case, the
exact maximin MAP assignment found by algorithm DFS is y∗ = {¬b, c, d} with value 9.99e-09,
while the exact maximax MAP assignment is y∗ = {¬b, ¬c, d} with value 0.7, respectively.

Limited Discrepancy Search. Our second approach for MAP and MMAP inference in LCNs
uses Limited Discrepancy Search (LDS) [19, 20] to explore the search space and is described by
Algorithm 2. Specifically, LDS is a depth-first search strategy that searches for new solutions by
iteratively increasing the number of discrepancy values, where a discrepancy value indicates the
maximum number of allowed variable-value assignment changes to an initial solution [19]. Function
SEARCH (lines 7–22) performs the actual exploration of the search space limited by discrepancy θ. If
the selected truth value y ∈ {yi, ¬yi} is different from the one corresponding to proposition Yi ∈ Y
at position i in the assignment y, θ is decremented to reduce the number of changes allowed to the
remaining MAP propositions. Otherwise, the truth value for proposition Yi remains unchanged and
the θ value is preserved. As before, complete MAP assignments are evaluated exactly (lines 9–12)
and the best solution found so far is maintained (lines 13-14).

5

Algorithm 3 Simulated Annealing for MAP and Marginal MAP Inference in LCNs

1: procedure SA(L = ⟨A, C⟩, E = e, Y)
initialize y0 randomly and let y∗ ← y0
2:
best ← score(y∗)
3:
for all iterations i = 1 . . . N do
4:
set y ← y∗, T ← Tinit
5:
for all flips j = 1 . . . M do
6:
let N be y’s neighbors
7:
select random neighbor y′ ∈ N
8:
∆ ← log score(y′) − log score(y)
9:

10:
11:
12:
13:
14:
15:
16:
17:

if ∆ > 0 then y ← y′
else
sample randomly p ∈ (0, 1)
if p < e ∆
T then y ← y′
if score(y) > best then
y∗ ← y, best ← score(y)
T ← T ∗ σ

return y∗

Algorithm 4 Approximate MAP and Marginal MAP Inference in LCNs

1: procedure AMAP(L = ⟨A, C⟩, E = e, Y)
2: Create factor graph F of L
3: Apply the ARIEL scheme from [18] on F
for all MAP propositions Y ∈ Y do
4:
if maximin then
5:
P (y) = maxf ∈N (Y ) lf→Y
6:
P (¬y) = 1 − P (y)
7:
if P (y) > P (¬y) then y∗ ← y∗ ∪ {y}
8:

9:
10:
11:
12:
13:
14:
15:

else y∗ ← y∗ ∪ {¬y}
else
P (y) = minf ∈N (Y ) uf→Y
P (¬y) = 1 − P (y)
if P (y) > P (¬y) then y∗ ← y∗ ∪ {y}
else y∗ ← y∗ ∪ {¬y}

return y∗

Theorem 2 (complexity). Given an LCN L = ⟨A, C⟩ with n propositions, evidence E = e and MAP
propositions Y ⊆ A \ E, algorithm LDS is sound and complete. The time and space complexity of
the algorithm is O(2m+2n
) and O(2n), respectively, where m is the number of MAP propositions.

Simulated Annealing. The third approach for solving MAP and MMAP tasks in LCNs is described
by Algorithm 3 and employs a form of stochastic local search known as Simulated Annealing (SA)
[21] to explore the search space defined by the MAP propositions. The algorithm starts from an
initial guess y as a truth assignment to the MAP propositions Y, and iteratively tries to improve it
by moving to a better neighbor y′ that has a higher score. A neighbor y′ of y is defined as a new
assignment y′ which results from changing the truth value of a single proposition Y in Y. At each
step, the transition from the current state y to a neighboring state y′ is decided probabilistically using
an acceptance probability function P (y′, y, T ) that depends on the scores of the two states as well as
a global time-varying parameter T called temperature which is decreased using a cooling schedule
σ < 1 [21]. We chose P (y′, y, T ) = e ∆
Theorem 3 (complexity). Given an LCN L = ⟨A, C⟩ with n propositions, evidence E = e and MAP
propositions Y ⊆ A \ E, the time and space complexity of algorithm SA is O(N · M · 22n
) and
O(2n), respectively, where N is the number of iterations and M is the number of flips per iterations.

T , where ∆ = log score(y′) − log score(y).

3.3 Approximate MAP and Marginal MAP Inference

The main bottleneck in the proposed search algorithms is the exact evaluation of the MAP assignments
which is computationally very expensive [11]. This limits the applicability of these methods to
relatively small LCNs. Therefore, in order to be able to tackle larger LCNs, we extend a recent
message-passing approximation scheme for marginal inference in LCNs [18] to solve the MAP and
MMAP tasks in LCNs. Subsequently, we also adapt the limited discrepancy search and simulated
annealing methods to use an approximate evaluation of the MAP assignments during search.

Algorithm 4 describes our message-passing based approximation scheme for MAP and MMAP
inference in LCNs which we denote hereafter by AMAP. We build upon a recent scheme for
approximate marginal inference in LCNs, called ARIEL [18], which propagates messages along the
edges of a factor graph associated with the input LCN until convergence. The factor graph F of an

6

LCN L is a bi-partite graph the connects proposition nodes labeled by the propositions in L with
factor nodes associated with sentences that involve the same set of propositions [18]. The messages
propagated between the nodes of F are intervals representing lower and upper bounds on the marginal
probabilities of L’s propositions and are computed as follows: the message sent from a proposition to
a factor node tightens these bounds based on the incoming messages from the factor nodes connected
to it; the message sent from a factor to a proposition node computes new bounds by solving a local
non-linear program defined by the factor’s sentences and the constraints encoding the assumption
that the factor’s propositions are independent of each other and the marginal probabilities of the
factor’s propositions are within the bounds given by the incoming proposition-to-factor messages (see
also [18] for more details). Upon convergence, the maximin MAP assignment y∗ can be obtained
as follows: for each MAP proposition Y ∈ Y we compute the tightest lower probability bound
P (y) by maximizing the lower bound of all incoming factor-to-proposition messages to Y , and,
subsequently, select y as the most likely value assignment to Y if P (y) > P (¬y) and ¬y otherwise
(for the maximax tasks we use the upper probability bounds P (y) and P (¬y), respectively).

Theorem 4 (complexity). Given an LCN L = ⟨A, C⟩ with n propositions, evidence E = e and MAP
propositions Y ⊆ A \ E, the time and space complexity of algorithm AMAP is O(N · M · 22r
) and
O(2r), where N is the number of iterations, M bounds the number of factor-to-node messages per
iteration and r bounds the number of propositions in the factor nodes, respectively.

3.4 Search Algorithms Based on Approximate MAP Evaluations

The main assumption behind algorithm AMAP is that all MAP propositions are independent of each
other and therefore the solution y∗ returned by AMAP is likely to correspond to a local maxima. One
way to escape such a local optima and obtain a potentially better solution is to employ a search scheme
based on either limited discrepancy search or simulated annealing that continues the exploration of
the search space starting from y∗. However, in order to scale to larger LCNs, we would like the
search schemes to rely on an approximate rather than an exact evaluation of the MAP assignments.

Approximate Evaluation of a MAP Assignment. Estimating the lower and upper probabilities
of a MAP assignment y can be done by approximate marginal inference on an augmented LCN
as follows. Let L = ⟨A, C⟩ be the input LCN and let y = (y1, . . . , ym) be a MAP assignment to
propositions Y = {Y1, . . . , Ym} (for simplicity, we include the evidence e in y). The augmented
LCN L′ = ⟨A′, C′⟩ is constructed by adding a set of auxiliary propositions W = {W1, . . . , Wm},
one for each MAP proposition, and additional constraints of the following two forms: P (W1|Y1) and
P (Wj|Wj−1∧Yj), for all 2 ≤ j ≤ m, such that P (w1|y1) = 1, P (w1|¬y1) = 0, P (wj|wj−1∧yj) =
1, P (wj|wj−1 ∧ ¬yj) = 0, P (wj|¬wj−1 ∧ yj) = 0 and P (wj|¬wj−1 ∧ ¬yj) = 0, respectively.
Then, we can estimate P (ψy) and P (ψy), where ψy = y1 ∧ · · · ∧ ym, by computing the posterior
marginals P (wm) and P (wm) in the augmented LCN L′ using the method from [18].

Limited Discrepancy Search and Simulated Annealing. Our approximate LDS and SA based
algorithms denoted by ALDS and ASA can be obtained from Algorithms 2 and 3 by replacing the
score(y) function with the approximate MAP evaluation scheme described above. These algorithms
can start the search either from a random MAP assignment or from the solution found by algorithm
AMAP. Finally, the time complexity of algorithms ALDS and ASA can be bounded by O(2m+2r
)
and O(N · M · 22r
), respectively, where m is the number of MAP propositions, N is the number of
iterations used by ASA, M is the maximum number of flips per iteration, and r bounds the number
of propositions in the factor nodes of the factor graph associated with the input LCN [18].

4 Experiments

In this section, we empirically evaluate the proposed exact and approximate schemes for MAP and
MMAP inference in LCNs. All competing algorithms were implemented2 in Python 3.10 and used
the ipopt 3.14 solver [22] with default settings to handle the non-linear constraint programs. We
ran all experiments on a 3.0GHz Intel Core processor with 128GB of RAM.

2The open-source implementation of LCNs is available at: https://github.com/IBM/LCN

7

Table 1: Results for MAP tasks obtained on small/large scale polytree, dag, and random LCNs.
Average CPU time in seconds and number of problem instances solved. Time limit is 2 hours.

size
n

exact MAP eval

DFS

LDS(3)

approx MAP eval

AMAP

ALDS(3)

ASA

SA
polytree

5
8
10
30
50
70

5
8
10
30
50
70

5
8
10
30
50
70

15.30 (10)
3246.28 (4)
-
-
-
-

21.09 (10)
1633.38 (8)
-
-
-
-

19.51 (10)
3152.57 (1)
-
-
-
-

26.07 (10)
3072.18 (4)
-
-
-
-

15.66 (10)
1958.16 (9)
-
-
-
-

17.56 (10)
3209.54 (5)
-
-
-
-

20.18 (10)
1199.51 (10)
-
-
-
-
dag
24.04 (10)
633.77 (10)
-
-
-
-
random

20.37 (10)
1226.88 (10)
-
-
-
-

2.87 (10)
8.05 (10)
11.81 (10)
31.55 (10)
52.30 (10)
79.28 (10)

5.54 (10)
13.05 (10)
15.55 (10)
49.94 (10)
89.13 (10)
132.34 (10)

5.26 (10)
10.29 (10)
12.21 (10)
40.54 (10)
76.83 (10)
105.70 (10)

174.17 (10)
1054.53 (10)
2273.16 (10)
-
-
-

163.02 (10)
1339.71 (10)
2903.05 (10)
-
-
-

152.99 (10)
954.46 (10)
2150.27 (10)
-
-
-

188.27 (10)
518.18 (10)
813.30 (10)
3091.74 (10)
5324.71 (10)
7279.56 (10)

156.34 (10)
571.55 (10)
944.17 (10)
3593.71 (10)
5639.90 (10)
6093.28 (10)

143.60 (10)
444.17 (10)
717.75 (10)
3335.14 (10)
5276.93 (10)
6059.57 (7)

Random LCNs. We generated three classes of random LCNs with n propositions {X1, . . . Xn}
and sentences of the following types: (a) l ≤ P (xi) ≤ u, (b) l ≤ P (xi|xj) ≤ u, i ̸= j and (c)
l ≤ P (xi|Xj ∧ Xk) ≤ u, i ̸= j ̸= k, such that the corresponding primal graph is a polytree, a
dag or a random graph. The type (c) sentences were generated for all truth values of propositions
Xj and Xk, namely P (xi|xj), P (xi|¬xj), P (xi|xj ∧ xk), P (xi|xj ∧ ¬xk), P (xi|¬xj ∧ xk) and
P (xi|¬xj ∧ ¬xk), respectively. The probability bounds l and u were selected uniformly at random
between 0 and 1 such that u − l ≤ 0.6, and we ensured that all instances with n ≤ 10 were consistent.

Table 1 summarizes the results obtained for maximax MAP queries on the random LCNs.
For each problem class we consider both
smaller (5 ≤ n ≤ 10) and larger (30 ≤ n ≤
70) scale instances, respectively. We report
the average CPU time in seconds and num-
ber of problem instance solved (out of 10)
for each problem size. A ’-’ indicates that
the respective algorithm exceeded the 2 hour
time limit. The maximum discrepancy value
use by algorithms LDS and ALDS was set to
δ = 3, while algorithms SA and ASA used
up to 30 flips over a single iteration. We
can see that the algorithms using exact MAP
assignment evaluations (i.e., DFS, LDS and
SA) are limited to small scale problem in-
stances with up to 8 propositions and they
run out of time on the larger instances. This is caused by the prohibitively large computational
overhead associated with the exact evaluation of the MAP assignments during search. In contrast, the
approximate search algorithms ALDS and specially ASA can scale to much larger problem instances
due to the less expensive approximate MAP assignment evaluations. AMAP is the best performing
algorithm in terms of running time and number of problems solved for all reported problem sizes.
However, since the solution found by AMAP is only a local maxima, in Figure 2 we report on
the solution quality found by algorithms AMAP, ALDS and ASA on LCN instances of size 10.
Specifically, we show the number of wins as the number of times (out of 10) each algorithm found
the best solution. In this case, algorithms ALDS and ASA were initialized with the MAP assignment

Figure 2: Wins for LCNs with n = 10.

8

Figure 3: Average CPU time in seconds and standard deviation vs discrepancy δ for ALDS(δ).

Table 2: Results for MMAP tasks on realistic LCNs. CPU time in seconds. Time limit is 2 hours.

LCN

Toy
Earth
Cancer
Asia
Credit
Engine
Suicide
Tank
Alarm
Hepatitis

exact MAP eval
LDS(3)
3.18
7.67
14.09
800.18
6719.30
4502.34
-
-
-
-

SA AMAP ALDS(3)
134.83
1.85
150.99
2.75
157.92
8.52
187.44
312.10
204.77
2976.55
212.61
2033.77
220.31
-
263.65
-
216.19
-
260.38
-

approx MAP eval
ASA
141.17
162.35
159.66
201.76
222.52
235.70
203.68
281.73
186.67
250.45

0.85
1.28
2.64
4.07
5.09
6.57
5.99
8.04
4.28
8.22

DFS
2.20
9.19
16.34
811.82
-
4786.12
-
-
-
-

found by AMAP. We can see that almost always the search-based approaches ALDS and ASA are
able to find better solutions than AMAP. This is important in practice, particularly on larger scale
problems where we can use AMAP to find a MAP solution quickly, and subsequently refine that
solution using a search-based algorithm like ALDS or ASA if the time budget allows it. Finally, in
Figure 3 we show the impact of the maximum discrepancy value δ on the running time of algorithm
ALDS(δ). It is easy to see that as the discrepancy value δ increases, the search space explored by
ALDS(δ) becomes larger, and therefore its corresponding running time increases as well.

Realistic LCNs. We experimented with a set of more realistic LCNs which were first introduced
in [18]. These LCNs were derived from real-world Bayesian networks [23] and contain up to 10
propositions as well as up to 24 sentences of the form l ≤ P (xi) ≤ u and l ≤ P (xi|πi) ≤ u,
respectively, where xi is the positive literal of proposition Xi and πi = yi1 ∧ · · · ∧ yik is the
conjunction of the positive or negative literals corresponding to a particular configuration of the
parents {Yi1, . . . Yik} of Xi in the Bayesian network. The specification of these LCNs is included
in the supplementary material. Table 2 reports the results obtained on 10 LCN instances for the
maximax MMAP task with 4 MAP propositions selected randomly. As before, algorithms DFS,
LDS(3) and SA which rely on exact evaluations of the MAP assignments during search can only solve
the smallest problem instances within the 2 hour time limit. In contrast, algorithms ALDS(3) and
ASA solve all problem instances due to a much reduced overhead associated with the approximate
MAP assignment evaluations. In this case, the search spaces explored by ALDS(3) and ASA
are approximately the same in size and therefore the corresponding running times are comparable.
AMAP is the fastest algorithm in this case as well.

Application to Factuality in Large Language Models. We consider an application of MMAP
inference in LCNs to assess the factuality of the output A generated by a large language model
(LLM) in response to a user query Q with respect to an external source of knowledge C that
may contain contradicting factual information (e.g., Wikipedia) [24]. The goal is to compute a
factuality score for response A, denoted by fC(A), in the context of the information from C. In
the following, we assume that A can be decomposed into a set of n atomic facts (or just atoms)
A = {A1, . . . , An} (e.g., one way to do that is to split A into sentences) and, for each atom Ai, up
to k relevant passages {Ci1, . . . , Cik} called contexts can be retrieved from C. A natural language
inference (NLI) classifier such as SBERT [25] can be used to infer the entailment, contradiction
and neutrality relationships between the texts corresponding to the atoms and contexts together

9

Table 3: Results for factuality LCNs. Average CPU time in seconds and number of problem instances
solved. Time limit is 2 hours.

size
n, k = 2
2
4
6
10
20
50
100

exact MAP eval

DFS
56.95 (10)
-
-
-
-
-
-

LDS(2)
57.37 (10)
-
-
-
-
-
-

SA
60.09 (10)
-
-
-
-
-
-

AMAP
0.31 (10)
0.98 (10)
1.97 (10)
7.33 (10)
28.42 (10)
379.18 (10)
1807.10 (10)

approx MAP eval

ALDS(2)
5.25 (10)
80.07 (10)
453.88 (10)
2713.90 (10)
-
-
-

ASA
4.13 (10)
54.15(10)
219.57 (10)
928.28 (10)
3809.23 (10)
-
-

with their corresponding probabilities (or scores). Specifically, we consider relationships between
an atom and a context r(Ai, Cij), and between two contexts r(Cij, Cpq), respectively, where r ∈
{entailment, contradiction}. We define an LCN L containing n + n × k propositions for each of the
atoms and contexts, and two types of sentences corresponding to the entailment and contradiction
relationships as follows: l ≤ P (Y |X) ≤ u if X entails Y , and l ≤ P (¬Y |X) ≤ u if X contradicts
Y , where X and Y are the propositions corresponding to a context and an atom, or to two different
contexts, respectively. The lower and upper bounds l and u can be calculated easily from the
probabilities obtained by running multiple NLI classifiers. Finally, the factuality score fC(A) is the
proportion of true atoms in the MAP assignment obtained by solving a maximax MMAP task over
L where the MAP propositions are those corresponding to A’s atoms.

Table 3 displays the results obtained on randomly generated factuality LCNs. More specifically, for
each reported problem size n ∈ {2, 4, 6, 10, 20, 50, 100}, we generated 10 random instances with
n atoms and k = 2 contexts per atom such that 10% of all possible pairwise relationships between
atoms and contexts were selected to be either entailment or contradiction with probability 0.5 while
the remaining relationships were labeled as neutral and thus ignored. The lower and upper probability
bounds l and u in the corresponding LCN sentences were also generated randomly between 0 and
1 such that u − l ≤ 0.6. In this case, the maximum discrepancy value was set to 2 and simulated
annealing was allowed a single iteration and 30 flips. We observe again that algorithms DFS, LDS(2)
and SA can only solve the smallest instances due to large computational overhead associated with
exact evaluation of the MAP assignments. In contrast, algorithms ALDS(2) and ASA which rely on
less expensive approximate evaluations of the MAP assignments can scale to larger problems with up
to 20 atoms. Algorithm AMAP outperforms its competitors and solves all problem instances.

In summary, our empirical evaluation showed that the exact search-based MAP/MMAP algorithms
are limited to solving relatively small problem instances. In contrast, the approximate MAP/MMAP
schemes based on either message-passing or search can scale to much larger LCN instances.

5 Conclusions

In this paper, we address abductive reasoning tasks such as generating MAP and Marginal MAP
(MMAP) explanations in Logical Credal Networks (LCNs), a recently introduced probabilistic logic
framework for reasoning with imprecise knowledge. Since an LCN encodes a set of distributions
over its interpretations, a complete or partial explanation of the evidence (i.e., a MAP assignment)
may correspond to more than one distribution. Therefore, we define the maximin/maximax MAP
and MMAP tasks for LCNs as finding complete or partial MAP assignments that have maximum
lower/upper probability given the evidence. We propose several search algorithms that combine depth-
first search, limited-discrepancy search or simulated annealing with exact evaluations of the MAP
assignments using marginal inference for LCNs. We also develop an approximate message-passing
scheme as well as extend limited discrepancy search and simulated annealing to use an approximate
evaluation of the MAP assignments during search. Our experiments with random LCNs and LCNs
derived from realistic use-cases demonstrate conclusively that the search methods based on exact
evaluations of the MAP assignments are limited to small size problems, while the approximation
schemes can scale to much larger problems. For future work we plan to investigate more advanced
depth-first branch-and-bound and best-first search techniques. However, these kinds of methods
require developing novel heuristic bounding schemes to guide the search more effectively [16].

10

Acknowledgements

Fabio Cozman thanks CNPq (grant 305753/2022-3) and the Center for AI at Universidade de São
Paulo, funded by FAPESP (grant 2019/07665-4) and IBM.

References

[1] Nils Nilsson. Probabilistic logic. Artificial Intelligence, 28(1):71–87, 1986.

[2] Ronald Fagin, Joseph Halpern, and Nimrod Megiddo. A logic for reasoning about probabilities.

Information and Computation, 87(1-2):78–128, 1990.

[3] Jochen Heinsohn. Probabilistic description logics. In Proceedings of the International Confer-

ence on Uncertainty in Artificial Intelligence, pages 311–318, 1994.

[4] Manfred Jaeger. Probabilistic reasoning in terminological logics. In Principles of Knowledge

Representation and Reasoning, pages 305–316. Elsevier, 1994.

[5] Kent Andersen and John Hooker. Bayesian logic. Decision Support Systems, 11(2):191–210,

1994.

[6] Vijay Chandru and John Hooker. Optimization Methods for Logical Inference. John Wiley &

Sons, 1999.

[7] Michael Dürig and Thomas Studer. Probabilistic abox reasoning: Preliminary results. In

Description Logics, pages 104–111, 2005.

[8] Matthew Richardson and Pedro Domingos. Markov logic networks. Machine Learning, 62(1-

2):107–136, 2006.

[9] Lise Getoor and Ben Taskar. Introduction to Statistical Relational Learning (Adaptive Compu-

tation and Machine Learning). MIT Press, 2007.

[10] Luc De Raedt, Paolo Frasconi, Kristian Kersting, and Stephen Muggleton. Probabilistic

Inductive Logic Programming - Theory and Applications. Springer, 2008.

[11] Radu Marinescu, Haifeng Qian, Alexander Gray, Debarun Bhattacharjya, Francisco Barahona,
Tian Gao, Ryan Riegel, and Pravinda Sahu. Logical credal networks. In 36th Conference on
Neural Information Processing Systems (NeurIPS), 2022.

[12] Judea Pearl. Probabilistic Reasoning in Intelligent Systems. Morgan Kaufmann, 1988.

[13] Daphne Koller and Nir Friedman. Probabilistic Graphical Models: Principles and Techniques.

MIT Press, 2009.

[14] Radu Marinescu and Rina Dechter. AND/OR branch-and-bound search for combinatorial
optimization in graphical models. Artificial Intelligence, 173(16-17):1457–1491, 2009.

[15] Radu Marinescu and Rina Dechter. Memory intensive AND/OR search for combinatorial
optimization in graphical models. Artificial Intelligence, 173(16-17):1492–1524, 2009.

[16] Radu Marinescu, Junkyu Lee, Rina Dechter, and Alexander Ihler. AND/OR search for marginal

MAP. Journal or Artificial Intelligence Research (JAIR), 63(1):875 – 921, 2018.

[17] Radu Marinescu, Debarun Bhattacharjya, Junkyu Lee, Alexander Gray, and Fabio Cozman.
Credal marginal map. In 37th Conference on Neural Information Processing Systems (NeurIPS),
2023.

[18] Radu Marinescu, Haifeng Qian, Alexander Gray, Debarun Bhattacharjya, Francisco Barahona,
In 32nd

Tian Gao, and Ryan Riegel. Approximate inference in logical credal networks.
International Joint Conference on Artificial Intelligence (IJCAI), 2023.

[19] William Harvey and Matthew Ginsberg. Limited discrepancy search. In International Joint

Conference on Artificial Intelligence (IJCAI), pages 607–613, 1995.

11

[20] Richard Korf. Improved limited discrepancy search. In AAAI Conference on Artificial Intelli-

gence (AAAI), pages 286–291, 1996.

[21] Scott Kirkpatrick, Daniel Gelatt, and Mario Vecchi. Optimization by simulated annealing.

Science, 220:671–680, 1983.

[22] Andreas Wächter and Lorenz Biegler. On the implementation of an interior-point filter
line-search algorithm for large-scale nonlinear programming. Mathematical Programming,
106(1):25–57, 2006.

[23] Anthony Constantinou, Yang Liu, Kiattikun Chobtham, Zhigao Guo, and Neville Kitson. The
bayesys data and bayesian network repository. Technical report, Bayesian Artificial Intelligence
research lab, Queen Mary University of London, London, UK, 2020.

[24] Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer,
Luke Zettlemoyer, and Hannaneh Hajishirzi. FActScore: Fine-grained atomic evaluation of
factual precision in long form text generation. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing, pages 12076–12100, 2023.

[25] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-
networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing, pages 3973–3983, 2019.

12

NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research,
addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove
the checklist: The papers not including the checklist will be desk rejected. The checklist should
follow the references and precede the (optional) supplemental material. The checklist does NOT
count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For
each question in the checklist:

• You should answer [Yes] , [No] , or [NA] .

• [NA] means either that the question is Not Applicable for that particular paper or the

relevant information is Not Available.

• Please provide a short (1–2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the
reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it
(after eventual revisions) with the final version of your paper, and its final version will be published
with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation.
While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a
proper justification is given (e.g., "error bars are not reported because it would be too computationally
expensive" or "we were unable to find the license for the dataset we used"). In general, answering
"[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we
acknowledge that the true answer is often more nuanced, so please just use your best judgment and
write a justification to elaborate. All supporting evidence can appear either in the main paper or the
supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification
please point to the section(s) where related material for the question can be found.

IMPORTANT, please:

• Delete this instruction block, but keep the section heading “NeurIPS paper checklist",

• Keep the checklist subsection headings, questions/answers and guidelines below.

• Do not modify the questions and only use the provided macros for your answers.

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: Sections 3 and 4

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims

made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A No or
NA answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how

much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals

are not attained by the paper.

2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

13

Justification: Section 3, 4 and 5. Essentially the exact inference methods proposed in
this paper are limited to small size problems with up to 8 propositions/variables while the
proposed approximate inference methods can scale to much larger problems with tens and
even hundreds of variables.
Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that

the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The authors
should reflect on how these assumptions might be violated in practice and what the
implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms

and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to

address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]
Justification: Section 3
Guidelines:

• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-

referenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a short
proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented

by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: Section 4 and supplementary material

14

Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken

to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general. releasing code and data is often
one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case
of a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submis-
sions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how

to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe

the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct
the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.

5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?

Answer: [Yes]

Justification: Section 4 and supplementary material

Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/

guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).

• The instructions should contain the exact command and environment needed to run
to reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.

15

• At submission time, to preserve anonymity, the authors should release anonymized

versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the

paper) is recommended, but including URLs to data and code is permitted.

6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?

Answer: [Yes]

Justification: Section 4 and the supplementary material

Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail

that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental

material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Section 4 and the supplementary material

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confi-
dence intervals, or statistical significance tests, at least for the experiments that support
the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula,

call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error

of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g. negative
error rates).

• If error bars are reported in tables or plots, The authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.

8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?

Answer: [Yes]

Justification: Section 4

Guidelines:

• The answer NA means that the paper does not include experiments.

16

• The paper should indicate the type of compute workers CPU or GPU, internal cluster,

or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual

experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).

9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification:
Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a

deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-

eration due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [Yes]
Justification:
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal

impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
generate deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).

11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pretrained language models,
image generators, or scraped datasets)?
Answer: [NA]
Justification:

17

Guidelines:

• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors

should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.

12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?

Answer: [Yes]

Justification: Section 4

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a

URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of

service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package
should be provided. For popular datasets, paperswithcode.com/datasets has curated
licenses for some datasets. Their licensing guide can help determine the license of a
dataset.

• For existing datasets that are re-packaged, both the original license and the license of

the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to

the asset’s creators.

13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?

Answer: [Yes]

Justification: Section 4

Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose

asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either

create an anonymized URL or include an anonymized zip file.

14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?

18

Answer: [NA]
Justification:
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with

human subjects.

• Including this information in the supplemental material is fine, but if the main contribu-
tion of the paper involves human subjects, then as much detail as possible should be
included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
or other labor should be paid at least the minimum wage in the country of the data
collector.

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human

Subjects

Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [NA]
Justification:
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with

human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if

applicable), such as the institution conducting the review.

19

