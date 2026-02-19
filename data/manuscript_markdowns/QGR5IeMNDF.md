Under review as a conference paper at ICLR 2024

PURE MESSAGE PASSING CAN ESTIMATE COMMON
NEIGHBOR FOR LINK PREDICTION

Anonymous authors
Paper under double-blind review

ABSTRACT

Message Passing Neural Networks (MPNNs) have emerged as the de facto stan-
dard in graph representation learning. However, when it comes to link predic-
tion, they are not always superior to simple heuristics such as Common Neighbor
(CN). This discrepancy stems from a fundamental limitation: while MPNNs ex-
cel in node-level representation, they stumble with encoding the joint structural
features essential to link prediction, like CN. To bridge this gap, we posit that, by
harnessing the orthogonality of input vectors, pure message-passing can indeed
capture joint structural features. Specifically, we study the proficiency of MPNNs
in approximating CN heuristics. Based on our findings, we introduce the Message
Passing Link Predictor (MPLP), a novel link prediction model. MPLP taps into
quasi-orthogonal vectors to estimate link-level structural features, all while pre-
serving the node-level complexities. Moreover, our approach demonstrates that
leveraging message-passing to capture structural features could offset MPNNs’
expressiveness limitations at the expense of estimation variance. We conduct ex-
periments on benchmark datasets from various domains, where our method con-
sistently outperforms the baseline methods.

1

INTRODUCTION

Link prediction is a cornerstone task in the field of graph machine learning, with broad-ranging
implications across numerous industrial applications. From identifying potential new acquaintances
on social networks (Liben-Nowell & Kleinberg, 2003) to predicting protein interactions (Szklarczyk
et al., 2019), from enhancing recommendation systems (Koren et al., 2009) to completing knowledge
graphs (Zhu et al., 2021), the impact of link prediction is felt across diverse domains. Recently,
with the advent of Graph Neural Networks (GNNs) (Kipf & Welling, 2017) and more specifically,
Message-Passing Neural Networks (MPNNs) (Gilmer et al., 2017), these models have become the
primary tools for tackling link prediction tasks. Despite the resounding success of MPNNs in the
realm of node and graph classification tasks (Kipf & Welling, 2017; Hamilton et al., 2018; Veliˇckovi´c
et al., 2018; Xu et al., 2018), it is intriguing to note that their performance in link prediction does
not always surpass that of simpler heuristic methods (Hu et al., 2021).

Zhang et al. (2021) highlights the limitations of GNNs/MPNNs for link prediction tasks arising from
its intrinsic property of permutation invariance. Owing to this property, isomorphic nodes invariably
receive identical representations. This poses a challenge when attempting to distinguish links whose
endpoints are isomorphic nodes. As illustrated in Figure 1a, nodes v1 and v3 share a Common
Neighbor v2, while nodes v1 and v5 do not. Ideally, due to their disparate local structures, these two
links (v1, v3) and (v1, v5) should receive distinct predictions. However, the permutation invariance
of MPNNs results in identical representations for nodes v3 and v5, leading to identical predictions
for the two links. As Zhang et al. (2021) asserts, such node-level representation, even with the most
expressive MPNNs, cannot capture structural link representation such as Common Neighbors (CN),
a critical aspect of link prediction.

In this work, we posit that the pure Message Passing paradigm (Gilmer et al., 2017) can indeed
capture structural link representation by exploiting orthogonality within the vector space. We begin
by presenting a motivating example, considering a non-attributed graph as depicted in Figure 1a.
In order to fulfill the Message Passing’s requirement for node vectors as input, we assign a one-hot
vector to each node vi, such that the i-th dimension has a value of one, with the rest set to zero.

1

Under review as a conference paper at ICLR 2024

(a)

(b)

Figure 1: (a) Isomorphic nodes result in identical MPNN node representation, making it impossible
to distinguish links such as (v1, v3) and (v1, v5) based on these representations. (b) MPNN counts
Common Neighbor through the inner product of neighboring nodes’ one-hot representation.

These vectors, viewed as signatures rather than mere permutation-invariant node representations,
can illuminate pairwise relationships. Subsequently, we execute a single iteration of message passing
as shown in Figure 1b, updating each node’s vector by summing the vector of its neighbors. This
process enables us to compute CN for any node pair by taking the inner product of the vectors of the
two target nodes.

At its core, this naive method employs an orthonormal basis as the node signatures, thereby en-
suring that the inner product of distinct nodes’ signatures is consistently zero. While this approach
effectively computes CN, its scalability poses a significant challenge, given that its space complexity
is quadratically proportional to the size of the graph. To overcome this, we draw inspiration from
DotHash (Nunes et al., 2023) and capitalize on the premise that the family of vectors almost orthog-
onal to each other swells exponentially, even with just linearly scaled dimensions Kainen & K˚urkov´a
(1993). Instead of relying on the orthogonal basis, we can propagate these quasi-orthogonal (QO)
vectors and utilize the inner product to estimate the joint structural information of any node pair.
Furthermore, by strategically selecting which pair of node signatures to compute the inner product,
we can boost the expressiveness of MPNNs to estimate substructures—a feat previously deemed
impossible in the literature (Chen et al., 2020).

In sum, our paper presents several pioneering advances in the realm of GNNs for link prediction:

• We are the first, both empirically and theoretically, to delve into the proficiency of GNNs in
approximating heuristic predictors like CN for link prediction. This uncovers a previously
uncharted territory in GNN research.

• Drawing upon the insights gleaned from GNNs’ capabilities in counting CN, we introduce
MPLP, a novel link prediction model. Uniquely, MPLP discerns joint structures of links
and their associated substructures within a graph, setting a new paradigm in the field.

• Our empirical investigations provide compelling evidence of MPLP’s dominance. Bench-
mark tests reveal that MPLP not only holds its own but outstrips state-of-the-art models in
link prediction performance.

2 PRELIMINARIES AND RELATED WORK

Notations. Consider an undirected graph G = (V, E, X), where V represents the set of nodes
with cardinality n, indexed as {1, . . . , n}, E ⊆ V × V denotes the observed set of edges, and
Xi,: ∈ RFx encapsulates the attributes associated with node i. Additionally, let Nv signify the
neighborhood of a node v, that is Nv = {u|SPD(u, v) = 1} where the function SPD(·, ·) measures
the shortest path distance between two nodes. Furthermore, the node degree of v is given by dv =
|Nv|. To generalize, we introduce the shortest path neighborhood N s
v , representing the set of nodes
that are s hops away from node v, defined as N s

v = {u|SPD(u, v) = s}.

Link predictions. Alongside the observed set of edges E, there exists an unobserved set of edges,
which we denote as Ec ⊆ V × V \ E. This unobserved set encompasses edges that are either
absent from the original observation or are anticipated to materialize in the future within the graph
G. Consequently, we can formulate the link prediction task as discerning the unobserved set of
edges Ec. Heuristics link predictors include Common Neighbor (CN) (Liben-Nowell & Kleinberg,
2003), Adamic-Adar index (AA) (Adamic & Adar, 2003), and Resource Allocation (RA) (Zhou

2

Under review as a conference paper at ICLR 2024

Figure 2: GNNs estimate CN, AA and RA via MSE regression, using the mean value as a Baseline.
Lower values are better.

et al., 2009). CN is simply counting the cardinality of the common neighbors, while AA and RA
count them weighted to reflect their relative importance as a common neighbor.

CN(u, v) =

(cid:88)

k∈Nu

(cid:84) Nv

1 ; AA(u, v) =

(cid:88)

k∈Nu

(cid:84) Nv

1
log dk

; RA(u, v) =

(cid:88)

k∈Nu

(cid:84) Nv

1
dk

.

(1)

Though heuristic link predictors are effective across various graph domains, their growing computa-
tional demands clash with the need for low latency. To mitigate this, approaches like ELPH (Cham-
berlain et al., 2022) and DotHash (Nunes et al., 2023) propose using estimations rather than exact
calculations for these predictors. Our study, inspired by these works, seeks to further refine tech-
niques for efficient link predictions. A detailed comparison with related works and our method is
available in Appendix A.

GNNs for link prediction. The advent of graphs incorporating node attributes has caused a sig-
nificant shift in research focus toward methods grounded in GNNs. Most practical GNNs follow the
paradigm of the Message Passing (Gilmer et al., 2017). It can be formulated as:

h(l+1)
v

= UPDATE

(cid:16)

{h(l)

v , AGGREGATE

(cid:16)

{h(l)

u , h(l)

v , ∀u ∈ Nv}

(cid:17)

(cid:17)
}

,

(2)

represents the vector of node v at layer l and h(0)

where h(l)
v = Xv,:. For simplicity, we use hv
v
to represent the node vector at the last layer. The specific choice of the neighborhood aggregation
function, AGGREGATE(·), and the updating function, UPDATE(·), dictates the instantiation of the
GNN model, with different choices leading to variations of model architectures. In the context of
link prediction tasks, the GAE model (Kipf & Welling, 2016) derives link representation, h(i, j),
as a Hadamard product of the target node pair representations, h(i,j) = hi ⊙ hj. Despite its sem-
inal approach, the SEAL model (Zhang & Chen, 2018), which labels nodes based on proximity to
target links and then performs message-passing for each target link, is hindered by computational
expense, limiting its scalability. Efficient alternatives like ELPH (Chamberlain et al., 2022) estimate
node labels, while NCNC (Wang et al., 2023) directly learns edgewise features by aggregating node
representations of common neighbors.

3 CAN MESSAGE PASSING COUNT COMMON NEIGHBOR?

In this section, we delve deep into the potential of MPNNs for heuristic link predictor estimation. We
commence with an empirical evaluation to recognize the proficiency of MPNNs in approximating
link predictors. Following this, we unravel the intrinsic characteristics of 1-layer MPNNs, shedding
light on their propensity to act as biased estimators for heuristic link predictors and proposing an
unbiased alternative. Ultimately, we cast light on how successive rounds of message passing can
estimate the number of walks connecting a target node pair with other nodes in the graph. All proofs
related to the theorem are provided in Appendix E.

3.1 ESTIMATION VIA MEAN SQUARED ERROR REGRESSION

To explore the capacity of MPNNs in capturing the overlap information inherent in heuristic link
predictors, such as CN, AA and RA, we conduct an empirical investigation, adopting the GAE

3

Under review as a conference paper at ICLR 2024

framework (Kipf & Welling, 2016) with GCN (Kipf & Welling, 2017) and SAGE (Hamilton et al.,
2018) as representative encoders. SEAL (Zhang & Chen, 2018), known for its proven proficiency in
capturing heuristic link predictors, serves as a benchmark in our comparison. Additionally, we select
a non-informative baseline estimation, simply using the mean of the heuristic link predictors on the
training sets. The datasets comprise eight non-attributed graphs (more details in Section 5). Given
that GNN encoders require node features for initial representation, we have to generate such features
for our non-attributed graphs. We achieved this by sampling from a high-dimensional Gaussian
distribution with a mean of 0 and standard deviation of 1. Although one-hot encoding is frequently
employed for feature initialization on non-attributed graphs, we choose to forgo this approach due
to the associated time and space complexity.

To evaluate the ability of GNNs to estimate CN information, we adopt a training procedure analo-
gous to a conventional link prediction task. However, we reframe the task as a regression problem
aimed at predicting heuristic link predictors, rather than a binary classification problem predict-
ing link existence. This shift requires changing the objective function from cross-entropy to Mean
Squared Error (MSE). Such an approach allows us to directly observe GNNs’ capacity to approxi-
mate heuristic link predictors.

Our experimental findings, depicted in Figure 2, reveal that GCN and SAGE both display an ability
to estimate heuristic link predictors, albeit to varying degrees, in contrast to the non-informative
baseline estimation. More specifically, GCN demonstrates a pronounced aptitude for estimating RA
and nearly matches the performance of SEAL on datasets such as C.ele, Yeast, and PB. Nonethe-
less, both GCN and SAGE substantially lag behind SEAL in approximating CN and AA. In the
subsequent section, we delve deeper into the elements within the GNN models that facilitate this
approximation of link predictors while also identifying factors that impede their accuracy.

3.2 ESTIMATION CAPABILITIES OF GNNS FOR LINK PREDICTORS

GNNs exhibit the capability of estimating link predictors. In this section, we aim to uncover the
mechanisms behind these estimations, hoping to offer insights that could guide the development of
more precise and efficient methods for link prediction. We commence with the following theorem:
Theorem 1. Let G = (V, E) be a non-attributed graph and consider a 1-layer GCN/SAGE. Define
the input vectors X ∈ RN ×F initialized randomly from a zero-mean distribution with standard
deviation σnode. Additionally, let the weight matrix W ∈ RF ′×F be initialized from a zero-mean
distribution with standard deviation σweight. After performing message passing, for any pair of
nodes {(u, v)|(u, v) ∈ V × V \ E}, the expected value of their inner product is given by:

GCN: E(hu · hv) =

C
(cid:112) ˆdu

ˆdv

(cid:88)

k∈Nu

(cid:84) Nv

1
ˆdk

; SAGE: E(hu · hv) =

√

C
dudv

(cid:88)

1,

k∈Nu

(cid:84) Nv

where ˆdv = dv + 1 and the constant C is defined as C = σ2
The theorem suggests that given proper initialization of input vectors and weight matrices, MPNN-
based models, such as GCN and SAGE, can adeptly approximate heuristic link predictors. This
makes them apt for encapsulating joint structural features of any node pair. Interestingly, SAGE
predominantly functions as a CN estimator, whereas the aggregation function in GCN grants it the
ability to weigh the count of common neighbors in a way similar to RA. This particular trait of GCN
is evidenced by its enhanced approximation of RA, as depicted in Figure 2.

weightF F ′.

nodeσ2

Quasi-orthogonal vectors. The GNN’s capability to approximate heuristic link predictors is pri-
marily grounded in the properties of their input vectors in a linear space. When vectors are sampled
from a high-dimensional linear space, they tend to be quasi-orthogonal, implying that their inner
product is nearly 0 w.h.p. With message-passing, these QO vectors propagate through the graph,
yielding in a linear combination of QO vectors at each node. The inner product between pairs of
QO vector sets essentially echoes the norms of shared vectors while nullifying the rest. Such a trait
enables GNNs to estimate CN through message-passing. A key advantage of QO vectors, especially
when compared with orthonormal basis, is their computational efficiency. For a modest linear incre-
ment in space dimensions, the number of QO vectors can grow exponentially, given an acceptable
margin of error (Kainen & K˚urkov´a, 1993). An intriguing observation is that the orthogonality of
QO vectors remains intact even after GNNs undergo linear transformations post message-passing,

4

Under review as a conference paper at ICLR 2024

attributed to the randomized weight matrix initialization. This mirrors the dimension reduction ob-
served in random projection (Johnson & Lindenstrauss, 1984).

Limitations. While GNNs manifest a marked ability in estimating heuristic link predictors, they
are not unbiased estimators and can be influenced by factors such as node pair degrees, thereby
compromising their accuracy. Another challenge when employing such MPNNs is their limited
generalization to unseen nodes. The neural networks, exposed to randomly generated vectors, may
struggle to transform newly added nodes in the graph with novel random vectors. This practice
also violates the permutation-invariance principle of GNNs when utilizing random vectors as node
representation. It could strengthen generalizability if we regard these randomly generated vectors as
signatures of the nodes, instead of their node features, and circumvent the use of MLPs for them.

Unbiased estimator. Addressing the biased element in Theorem 1, we propose the subsequent
instantiation for the message-passing functions:

h(l+1)
v

=

(cid:88)

u∈Nv

h(l)
u .

(3)

Such an implementation aligns with the SAGE model that employs sum aggregation devoid of self-
node propagation. This methodology also finds mention in DotHash (Nunes et al., 2023), serving as
a cornerstone for our research. With this kind of message-passing design, the inner product of any
node pair signatures can estimate CN impartially:
Theorem 2. Let G = (V, E) be a graph, and let the vector dimension be given by F ∈ N+. Define
the input vectors X = (Xi,j), which are initialized from a random variable x having a mean of 0
and a standard deviation of
. Using the 1-layer message-passing in Equation 3, for any pair of
nodes {(u, v)|(u, v) ∈ V × V }, the expected value and variance of their inner product are:

1√
F

E(hu · hv) = CN(u, v),

Var(hu · hv) =

1
F

(cid:0)dudv + CN(u, v)2 − 2CN(u, v)(cid:1) + F Var(cid:0)x2(cid:1)CN(u, v).

Though this estimator provides an unbiased estimate for CN, its accuracy can be affected by its
variance. Specifically, DotHash recommends selecting a distribution for input vector sampling from
vertices of a hypercube with unit length, which curtails variance given that Var(cid:0)x2(cid:1) = 0. However,
the variance influenced by the graph structure isn’t adequately addressed, and this issue will be
delved into in Section 4.

Orthogonal node attributes. Both Theorem 1 and Theorem 2 underscore the significance of quasi
orthogonality in input vectors, enabling message-passing to efficiently count CN. Intriguingly, in
most attributed graphs, node attributes, often represented as bag-of-words (Purchase et al., 2022),
exhibit inherent orthogonality. This brings forth a critical question: In the context of link prediction,
do GNNs primarily approximate neighborhood overlap, sidelining the intrinsic value of node at-
tributes? We earmark this pivotal question for in-depth empirical exploration in Appendix C, where
we find that random vectors as input to GNNs can catch up with or even outperform node attributes.

3.3 MULTI-LAYER MESSAGE PASSING

Theorem 2 elucidates the estimation of CN based on a single iteration of message passing. This
section explores the implications of multiple message-passing iterations and the properties inherent
to the iteratively updated node signatures. We begin with a theorem delineating the expected value
of the inner product for two nodes’ signatures derived from any iteration of message passing:
Theorem 3. Under the conditions defined in Theorem 2, let h(l)
the l-th message-passing iteration. We have:

u denote the vector for node u after

(cid:16)

E

u · h(q)
h(p)
v

(cid:17)

=

(cid:88)

k∈V

|walks(p)(k, u)||walks(q)(k, v)|,

where |walks(l)(u, v)| counts the number of length-l walks between nodes u and v.
This theorem posits that the message-passing procedure computes the number of walks between the
target node pair and all other nodes. In essence, each message-passing trajectory mirrors the path

5

Under review as a conference paper at ICLR 2024

of the corresponding walk. As such, h(l)
u aggregates the initial QO vectors originating from nodes
reachable by length-l walks from node u. In instances where multiple length-l walks connect node
k to u, the associated QO vector Xk,: is incorporated into the sum |walks(l)(k, u)| times.
One might surmise a paradox, given that message-passing calculates the number of walks, not nodes.
However, in a simple graph devoid of self-loops, where at most one edge can connect any two nodes,
it is guaranteed that |walks(1)(u, v)| = 1 iff SPD(u, v) = 1. Consequently, the quantity of length-1
walks to a target node pair equates to CN, a first-order heuristic. It’s essential to recognize, however,
that |walks(l)(u, v)| ≥ 1 only implies SPD(u, v) ≤ l. This understanding becomes vital when
employing message-passing for estimating the local structure of a target node pair in Section 4.

4 METHOD

In this section, we introduce our novel link prediction model, denoted as MPLP. Distinctively de-
signed, MPLP leverages the pure essence of the message-passing mechanism to adeptly learn struc-
tural information. Not only does MPLP encapsulate the local structure of the target node pair by
assessing node counts based on varying shortest-path distances, but it also pioneers in estimating the
count of triangles linked to any of the target node pair— an ability traditionally deemed unattainable
for GNNs (Chen et al., 2020).

Node representation. While MPLP is specifically
it
designed for its exceptional structural capture,
also embraces the inherent attribute associations of
graphs that speak volumes about individual node
characteristics. To fuse the attributes (if they ex-
ist in the graph) and structures, MPLP begins with
a GNN, utilized to encode node u’s representation:
GNN(u) ∈ RFx. This node representation will
be integrated into the structural features when con-
structing the QO vectors. Importantly, this encoding
remains flexible, permitting the choice of any node-
level GNN.

4.1 QO VECTORS CONSTRUCTION

Figure 3: Representation of the target link
(u, v) within our model (MPLP), with nodes
color-coded based on their distance from the
target link.

Probabilistic hypercube sampling. Though de-
terministic avenues for QO vector construction are
documented (Kainen, 1992; Kainen & Kurkova,
2020), our preference leans toward probabilistic techniques for their inherent simplicity. We inherit
the sampling paradigm from DotHash (Nunes et al., 2023), where each node k is assigned with a
node signature h(0)
k , acquired via random sampling from the vertices of an F -dimensional hypercube
with unit vector norms. Consequently, the sampling space for h(0)

F }F .

F , 1/

√

√

k becomes {−1/

Harnessing One-hot hubs for variance reduction. The stochastic nature of our estimator brings
along an inevitable accompaniment: variance. Theorem 2 elucidates that a graph’s topology can
augment estimator variance, irrespective of the chosen QO vector distribution. At the heart of this
issue is the imperfectness of quasi-orthogonality. While a pair of vectors might approach orthogo-
nality, the same cannot be confidently said for the subspaces spanned by larger sets of QO vectors.

Capitalizing on the empirical observation that real-world graphs predominantly obey the power-
law distribution (Barab´asi & Albert, 1999), we discerned a strategy to control variance. Leverag-
ing the prevalence of high-degree nodes—or hubs—we designate unique one-hot vectors for the
foremost hubs. Consider the graph’s top-b hubs; while other nodes draw their QO vectors from
F − b}F −b×{0}b, these hubs are assigned one-hot vectors from
a hypercube {−1/
{0}F −b×{0, 1}b, reserving a distinct subspace of the linear space to safeguard orthogonality. Note
that when new nodes are added to the graph, their QO vectors are sampled the same way as the
non-hub nodes, which can ensure a tractable computation complexity.

F − b, 1/

√

√

6

Under review as a conference paper at ICLR 2024

Norm rescaling to facilitate weighted counts. Theorem 1 alludes to an intriguing proposition:
the estimator’s potential to encapsulate not just CN, but also RA. Essentially, RA and AA are nu-
anced heuristics translating to weighted enumerations of shared neighbors, based on their node de-
grees. In Theorem 2, such counts are anchored by vector norms during dot products. MPLP en-
hances this count methodology by rescaling node vector norms, drawing inspiration from previous
works (Nunes et al., 2023; Yun et al., 2021). This rescaling is determined by the node’s representa-
tion, GNN(u), and its degree du. The rescaled vector is formally expressed as:

˜h(0)
k = f (GNN(k)||[dk]) · h(0)
k ,
(4)
where f : RFx+1 → R is an MLP mapping the node representation and degree to a scalar, enabling
the flexible weighted count paradigm.

4.2 STRUCTURAL FEATURE ESTIMATIONS

Node label estimation. The estimator in Theorem 2 can effectively quantify CN. Nonetheless,
solely relying on CN fails to encompass diverse topological structures embedded within the local
neighborhood. To offer a richer representation, we turn to Distance Encoding (DE) (Li et al., 2020).
DE acts as an adept labeling tool (Zhang et al., 2021), demarcating nodes based on their shortest-
path distances relative to a target node pair. For a given pair (u, v), a node k belongs to DE(p, q)
iff SPD(u, k) = p and SPD(v, k) = q. Unlike its usage as node labels, we opt to enumerate these
labels, producing a link feature defined by #(p, q) = |DE(p, q)|. Our model adopts a philosophy
akin to ELPH (Chamberlain et al., 2022), albeit with a distinct node-estimation mechanism.

Returning to Theorem 3, we recall that message-passing as in Equation 3 essentially corresponds
to walks. Our ambition to enumerate nodes necessitates a single-layer message-passing alteration,
reformulating Equation 3 to:
(cid:88)

ηs

v =

˜h(0)
k .

Here, N s
v pinpoints v’s shortest-path neighborhoods distanced by the shortest-path s. This method
sidesteps the duplication dilemma highlighted in Theorem 3, ensuring that ηs
v aggregates at most
one QO vector per node. Similar strategies are explored in (Abboud et al., 2022; Feng et al., 2022).

k∈N s
v

For a tractable computation, we limit the largest shortest-path distance as r ≥ max(p, q). Conse-
quently, to capture the varied proximities of nodes to the target pair (u, v), we can deduce:
u · ηq
v | −

r ≥ p, q ≥ 1

v),
(cid:88)

#(s, q),

E(ηp

p = 0

|N q



(5)

(6)

#(p, q) =




|N p

u | −

1≤s≤r
(cid:88)

1≤s≤r

#(p, s),

q = 0

Concatenating the resulting estimates yields the expressive structural features of MPLP.

Shortcut removal. The intricately designed structural features improve the expressiveness
of MPLP. However, this augmented expressiveness introduces susceptibility to distribution shifts
during link prediction tasks (Dong et al., 2022). Consider a scenario wherein the neighborhood of a
target node pair contains a node k. Node k resides a single hop away from one of the target nodes but
requires multiple steps to connect with the other. When such a target node pair embodies a positive
instance in the training data (indicative of an existing link), node k can exploit both the closer target
node and the link between the target nodes as a shortcut to the farther one. This dynamic ensures
that for training-set positive instances, the maximum shortest-path distance from any neighboring
node to the target pair is constrained to the smaller distance increased by one. This can engender a
discrepancy in distributions between training and testing phases, potentially diminishing the model’s
generalization capability.

To circumvent this pitfall, we adopt an approach similar to preceding works (Zhang & Chen, 2018;
Yin et al., 2022; Wang et al., 2023; Jin et al., 2022). Specifically, we exclude target links from
the original graph during each training batch, as shown by the dash line in Figure 3. This maneu-
ver ensures these links are not utilized as shortcuts, thereby preserving the fidelity of link feature
construction.

7

Under review as a conference paper at ICLR 2024

Table 1: Link prediction results on non-attributed benchmarks evaluated by Hits@50. The format is
average score ± standard deviation. The top three models are colored by First, Second, Third.

USAir

NS

PB

Yeast

C.ele

Power

CN
AA
RA

GCN
SAGE

80.52±4.07
85.51±2.25
85.95±1.83

73.29±4.70
83.81±3.09

74.00±1.98
74.00±1.98
74.00±1.98

78.32±2.57
56.62±9.41

90.47±3.00
SEAL
Neo-GNN 86.07±1.96
87.60±1.49
ELPH
86.16±1.77
NCNC

86.59±3.03
83.54±3.92
88.49±2.14
83.18±3.17

37.22±3.52
39.48±3.53
38.94±3.54

37.32±4.69
47.26±2.53

44.47±2.86
44.04±1.89
46.91±2.21
46.85±3.18

72.60±3.85
73.62±1.01
73.62±1.01

73.15±2.41
71.06±5.12

83.92±1.17
83.14±0.73
82.74±1.19
82.00±0.97

47.67±10.87
58.34±2.88
61.47±4.59

40.68±5.45
58.97±4.77

64.80±4.23
63.22±4.32
64.45±3.91
60.49±5.09

11.57±0.55
11.57±0.55
11.57±0.55

15.40±2.90
6.89±0.95

Router

9.38±1.05
9.38±1.05
9.38±1.05

24.42±4.59
42.25±4.32

31.46±3.25
21.98±4.62
26.61±1.73
23.28±1.55

61.00±10.10
42.81±4.13
61.07±3.06
52.45±8.77

E.coli

51.74±2.70
68.13±1.61
74.45±0.55

61.02±11.91
75.60±2.40

83.42±1.01
73.76±1.94
75.25±1.44
83.94±1.57

MPLP

92.05±1.20

89.47±1.98

52.55±2.90

85.36±0.68

74.29±2.78

32.25±1.43

60.83±1.97

87.11±0.83

4.3 TRIANGLE ESTIMATIONS

Constructing the structural feature with DE can provably enhance the expressiveness of the link pre-
diction model (Li et al., 2020; Zhang et al., 2021). However, there are still prominent cases where
labelling trick also fails to capture. Since labelling trick only considers the relationship between the
neighbors and the target node pair, it can sometimes miss the subtleties of intra-neighbor relation-
ships. For example, the nodes of DE(1, 1) in Figure 3 exhibit different local structures. Nevertheless,
labelling trick like DE tends to treat them equally, which makes the model overlook the triangle sub-
structure shown in the neighborhood. Chen et al. (2020) discusses the challenge of counting such a
substructure with a pure message-passing framework. We next give an implementation of message-
passing to approximate triangle counts linked to a target node pair—equivalent in complexity to
conventional MPNNs.

For a triangle to form, two nodes must connect with each other and the target node. Key to our
methodology is recognizing the obligatory presence of length-1 and length-2 walks to the target
node. Thus, according to Theorem 3, our estimation can formalize as:

) =

#(△
u

1
2

(cid:16)˜h(1)
E

u · ˜h(2)

u

(cid:17)

.

(7)

Augmenting the node label counts with triangle estimates gives rise to a more expressive structural
feature set of MPLP.

Feature integration for link prediction. Having procured the structural features, we proceed to
formulate the encompassing link representation for a target node pair (u, v) as:

h(u,v) = (GNN(u) ⊙ GNN(v))||[#(1, 1), . . . , #(r, r), #(△
u

), #(△
v

)],

(8)

which can be fed into a classifier for a link prediction between nodes (u, v).

5 EXPERIMENTS

Datasets, baselines and experimental setup We evaluate our approach on a diverse set of 8 non-
attributed and 5 attributed graph benchmarks. In the absence of predefined train/test splits, links are
partitioned into train, validation, and test splits following a 70-10-20 percentage distribution. Our
comparison spans three categories of link prediction models: (1) heuristic-based methods encom-
passing CN, AA, and RA; (2) node-level models like GCN and SAGE; and (3) link-level models,
including SEAL, Neo-GNN (Yun et al., 2021), ELPH (Chamberlain et al., 2022), and NCNC (Wang
et al., 2023). Each experiment is conducted 10 times, with the average score and standard deviations
reported using the Hits@50 metric, a well-accepted standard for the link prediction task (Hu et al.,
2021). We limit the number of hops r = 2, which results in a good balance of performance and
efficiency. A comprehensive description of the experimental setup is available in Appendix B.

Results Performance metrics are presented in Table 1 and Table 2. MPLP outperforms other mod-
els on 12 of the 13 benchmarks. In the context of non-attributed graphs, MPLP takes the lead on 7
out of the 8 datasets, followed by SEAL and ELPH. For attributed graphs, MPLP reigns supreme
on all 5 datasets. Notably, MPLP consistently demonstrates superior results across a wide range of
graph domains, with a performance advantage ranging from 2% to 10% in Hits@50 over the closest
competitors. More ablation study can be found in Appendix D.

8

Under review as a conference paper at ICLR 2024

Table 2: Link prediction results on attributed benchmarks
evaluated by Hits@50. The format is average score ± stan-
dard deviation. The top three models are colored by First,
Second, Third.

CS

Physics

Computers

Photo

Collab

CN
AA
RA

GCN
SAGE

51.04±15.56
68.26±1.28
68.25±1.29

66.00±2.90
57.79±18.23

61.46±6.12
70.98±1.96
72.29±1.69

73.71±2.28
74.10±2.51

68.50±0.76
SEAL
Neo-GNN 71.13±1.69
72.26±2.58
ELPH
74.65±1.23
NCNC

74.27±2.58
72.28±2.33
65.80±2.26
75.96±1.73

21.95±2.00
26.96±2.08
28.05±1.59

22.95±10.58
33.79±3.11

30.43±2.07
22.76±3.07
29.01±2.66
36.48±4.16

29.33±2.74
37.35±2.65
40.77±3.41

28.14±7.81
46.01±1.83

61.37±0.00
64.35±0.00
64.00±0.00

35.53±2.39
36.82±7.41

46.08±3.27
44.83±3.23
43.51±2.37
47.98±2.36

64.74±0.43
57.52±0.37
65.94±0.58
66.61±0.71

MPLP

76.40±1.44

76.06±2.31

40.51±2.91

56.50±2.82

67.05±0.51

Figure 4: Evaluation of model size and
inference time on Collab. The infer-
ence time encompasses the entire cycle
within a single epoch.

Figure 5: MSE of estimation for #(1, 1), #(1, 2) and #(1, 0) on Collab. Lower values are better.

Model size and inference time A separate assessment focuses on the trade-off between model
size and inference time using the Collab dataset, with findings presented in Figure 4. Observing the
prominent role of graph structure in link prediction performance on Collab, we introduce a stream-
lined version of our model, termed MPLP(no feat). This variant solely capitalizes on structural
features, resulting in a compact model with merely 260 parameters. Nevertheless, its efficacy rivals
that of models which are orders of magnitude larger. Furthermore, MPLP’s inference time for a sin-
gle epoch ranks among the quickest in state-of-the-art approaches, underscoring its efficiency both
in terms of time and memory footprint. More details can be found in Appendix B.3.

Estimation accuracy We investigate the precision of MPLP in estimating #(p, q), which denotes
the count of node labels, using the Collab dataset. The outcomes of this examination are illustrated
in Figure 5. Although ELPH possesses the capability to approximate these counts utilizing tech-
niques like MinHash and Hyperloglog, our method exhibits superior accuracy. Moreover, ELPH
runs out of memory when the dimension is larger than 3000. Remarkably, deploying a one-hot en-
coding strategy for the hubs further bolsters the accuracy of MPLP, concurrently diminishing the
variance introduced by inherent graph structures. An exhaustive analysis, including time efficiency
considerations, is provided in Appendix D.1.

6 CONCLUSION

In this work, we delved into the potential of message-passing GNNs to encapsulate joint struc-
tural features of graphs. Stemming from this investigation, we introduced a novel link predic-
tion paradigm that consistently outperforms state-of-the-art baselines across a varied suite of graph
benchmarks. The inherent capability to adeptly capture structures enhances the expressivity of
GNNs, all while maintaining their computational efficiency. Our findings hint at a promising av-
enue for elevating the expressiveness of GNNs through probabilistic approaches.

9

Under review as a conference paper at ICLR 2024

REFERENCES
Ralph Abboud, ˙Ismail ˙Ilkan Ceylan, Martin Grohe, and Thomas Lukasiewicz. The Surprising Power

of Graph Neural Networks with Random Node Initialization, 2021. eprint: 2010.01179.

Ralph Abboud, Radoslav Dimitrov, and Ismail Ilkan Ceylan. Shortest Path Networks for Graph
Property Prediction. November 2022. URL https://openreview.net/forum?id=
mWzWvMxuFg1.

Robert Ackland and others. Mapping the US political blogosphere: Are conservative bloggers more

prominent? In BlogTalk Downunder 2005 Conference, Sydney, 2005.

Lada A. Adamic and Eytan Adar.

Social Net-
ISSN 0378-8733. doi: https://doi.org/10.1016/S0378-8733(03)
URL https://www.sciencedirect.com/science/article/pii/

Friends and neighbors on the Web.

works, 25(3):211–230, 2003.
00009-1.
S0378873303000091.

Albert-L´aszl´o Barab´asi and R´eka Albert.
286(5439):509–512,

Emergence of Scaling in Random Net-
works.
10.1126/science.286.5439.509.
1999.
URL https://www.science.org/doi/abs/10.1126/science.286.5439.509.
eprint: https://www.science.org/doi/pdf/10.1126/science.286.5439.509.

Science,

doi:

Vladimir Batagelj and Andrej Mrvar. Pajek datasets website, 2006. URL http://vlado.fmf.

uni-lj.si/pub/networks/data/.

Sergey Brin and Lawrence Page. The Anatomy of a Large-Scale Hypertextual Web Search En-
gine. Computer Networks, 30:107–117, 1998. URL http://www-db.stanford.edu/
˜backrub/google.html.

Benjamin Paul Chamberlain, Sergey Shirobokov, Emanuele Rossi, Fabrizio Frasca, Thomas
Markovich, Nils Yannick Hammerla, Michael M. Bronstein, and Max Hansmire. Graph Neu-
ral Networks for Link Prediction with Subgraph Sketching. September 2022. URL https:
//openreview.net/forum?id=m1oqEOAozQU.

Zhengdao Chen, Lei Chen, Soledad Villar, and Joan Bruna. Can Graph Neural Networks Count
Substructures? arXiv:2002.04025 [cs, stat], October 2020. URL http://arxiv.org/abs/
2002.04025. arXiv: 2002.04025.

Kaiwen Dong, Yijun Tian, Zhichun Guo, Yang Yang, and Nitesh Chawla. FakeEdge: Alleviate
In The First Learning on Graphs Conference (LOG), 2022.

Dataset Shift in Link Prediction.
URL https://openreview.net/forum?id=QDN0jSXuvtX.

Jiarui Feng, Yixin Chen, Fuhai Li, Anindya Sarkar, and Muhan Zhang. How Powerful are K-hop
Message Passing Graph Neural Networks. May 2022. URL https://openreview.net/
forum?id=nN3aVRQsxGd.

Matthias Fey and Jan E. Lenssen. Fast Graph Representation Learning with PyTorch Geometric. In

ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.

Fabrizio Frasca, Beatrice Bevilacqua, Michael M. Bronstein, and Haggai Maron. Understanding
and Extending Subgraph GNNs by Rethinking Their Symmetries, June 2022. URL http://
arxiv.org/abs/2206.11140. arXiv:2206.11140 [cs].

Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neu-
ral Message Passing for Quantum Chemistry. CoRR, abs/1704.01212, 2017. URL http:
//arxiv.org/abs/1704.01212. arXiv: 1704.01212.

William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive Representation Learning on Large
Graphs. arXiv:1706.02216 [cs, stat], September 2018. URL http://arxiv.org/abs/
1706.02216. arXiv: 1706.02216.

Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele
Catasta, and Jure Leskovec. Open Graph Benchmark: Datasets for Machine Learning on
Graphs. arXiv:2005.00687 [cs, stat], February 2021. URL http://arxiv.org/abs/
2005.00687. arXiv: 2005.00687.

10

Under review as a conference paper at ICLR 2024

Jiarui Jin, Yangkun Wang, Weinan Zhang, Quan Gan, Xiang Song, Yong Yu, Zheng Zhang,
and David Wipf. Refined Edge Usage of Graph Neural Networks for Edge Prediction. De-
cember 2022. doi: 10.48550/arXiv.2212.12970. URL https://arxiv.org/abs/2212.
12970v1.

William Johnson and Joram Lindenstrauss. Extensions of Lipschitz maps into a Hilbert space.
Contemporary Mathematics, 26:189–206, January 1984. ISSN 9780821850305. doi: 10.1090/
conm/026/737400.

Paul C Kainen. Orthogonal dimension and tolerance. Unpublished report, Washington DC: Indus-

trial Math, 1992.

Paul C Kainen and Vˇera Kurkova. Quasiorthogonal dimension. In Beyond traditional probabilistic
data processing techniques: Interval, fuzzy etc. Methods and their applications, pp. 615–629.
Springer, 2020.

Paul C. Kainen and V˘era K˚urkov´a. Quasiorthogonal dimension of euclidean spaces. Applied
ISSN 0893-9659. doi: 10.1016/0893-9659(93)
URL https://www.sciencedirect.com/science/article/pii/

Mathematics Letters, 6(3):7–10, May 1993.
90023-G.
089396599390023G.

Leo Katz. A new status index derived from sociometric analysis. Psychometrika, 18(1):39–43,
March 1953. ISSN 1860-0980. doi: 10.1007/BF02289026. URL https://doi.org/10.
1007/BF02289026.

Thomas N. Kipf and Max Welling. Variational Graph Auto-Encoders, 2016. eprint: 1611.07308.

Thomas N. Kipf and Max Welling. Semi-Supervised Classification with Graph Convolutional Net-
works. arXiv:1609.02907 [cs, stat], February 2017. URL http://arxiv.org/abs/1609.
02907. arXiv: 1609.02907.

Yehuda Koren, Robert Bell, and Chris Volinsky. Matrix factorization techniques for recommender

systems. Computer, 42(8):30–37, 2009. Publisher: IEEE.

Pan Li, Yanbang Wang, Hongwei Wang, and Jure Leskovec.

Distance Encoding: De-
sign Provably More Powerful Neural Networks for Graph Representation Learning.
In
H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin (eds.), Advances
in Neural
Information Processing Systems, volume 33, pp. 4465–4478. Curran Asso-
ciates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/
2f73168bf3656f697507752ec592c437-Paper.pdf.

David Liben-Nowell and Jon Kleinberg. The link prediction problem for social networks.

In
Proceedings of the twelfth international conference on Information and knowledge manage-
ment, CIKM ’03, pp. 556–559, New York, NY, USA, November 2003. Association for Com-
ISBN 978-1-58113-723-1. doi: 10.1145/956863.956972. URL http:
puting Machinery.
//doi.org/10.1145/956863.956972.

Haggai Maron, Heli Ben-Hamu, Hadar Serviansky, and Yaron Lipman. Provably Powerful Graph
Networks. arXiv:1905.11136 [cs, stat], June 2020. URL http://arxiv.org/abs/1905.
11136. arXiv: 1905.11136.

Christopher Morris, Martin Ritzert, Matthias Fey, William L. Hamilton, Jan Eric Lenssen, Gaurav
Rattan, and Martin Grohe. Weisfeiler and Leman Go Neural: Higher-order Graph Neural Net-
works, November 2021. URL http://arxiv.org/abs/1810.02244. arXiv:1810.02244
[cs, stat].

Mark EJ Newman. Finding community structure in networks using the eigenvectors of matrices.

Physical review E, 74(3):036104, 2006. Publisher: APS.

Igor Nunes, Mike Heddes, Pere Verg´es, Danny Abraham, Alexander Veidenbaum, Alexandru
Nicolau, and Tony Givargis. DotHash: Estimating Set Similarity Metrics for Link Prediction
and Document Deduplication, May 2023. URL http://arxiv.org/abs/2305.17310.
arXiv:2305.17310 [cs].

11

Under review as a conference paper at ICLR 2024

P´al Andr´as Papp, Karolis Martinkus, Lukas Faber, and Roger Wattenhofer. DropGNN: Random
Dropouts Increase the Expressiveness of Graph Neural Networks, November 2021. URL http:
//arxiv.org/abs/2111.06283. arXiv:2111.06283 [cs].

Skye Purchase, Yiren Zhao, and Robert D. Mullins. Revisiting Embeddings for Graph Neural Net-
works. November 2022. URL https://openreview.net/forum?id=Ri2dzVt_a1h.

Gerard Salton and Michael J. McGill. Introduction to Modern Information Retrieval. McGraw-Hill,

Inc., USA, 1986. ISBN 0-07-054484-0.

Ryoma Sato, Makoto Yamada, and Hisashi Kashima. Random Features Strengthen Graph Neural

Networks, 2021. eprint: 2002.03155.

Oleksandr Shchur, Maximilian Mumme, Aleksandar Bojchevski, and Stephan G¨unnemann. Pitfalls
of Graph Neural Network Evaluation, June 2019. URL http://arxiv.org/abs/1811.
05868. arXiv:1811.05868 [cs, stat].

Neil Spring, Ratul Mahajan, and David Wetherall. Measuring ISP topologies with Rocketfuel.
ACM SIGCOMM Computer Communication Review, 32(4):133–145, 2002. Publisher: ACM
New York, NY, USA.

Damian Szklarczyk, Annika L. Gable, David Lyon, Alexander Junge, Stefan Wyder, Jaime Huerta-
Cepas, Milan Simonovic, Nadezhda T. Doncheva, John H. Morris, Peer Bork, Lars J. Jensen,
and Christian von Mering. STRING v11: protein-protein association networks with increased
coverage, supporting functional discovery in genome-wide experimental datasets. Nucleic Acids
Research, 47(D1):D607–D613, January 2019. ISSN 1362-4962. doi: 10.1093/nar/gky1131.

Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li`o, and Yoshua
Bengio. Graph Attention Networks. arXiv:1710.10903 [cs, stat], February 2018. URL http:
//arxiv.org/abs/1710.10903. arXiv: 1710.10903.

Christian Von Mering, Roland Krause, Berend Snel, Michael Cornell, Stephen G Oliver, Stanley
Fields, and Peer Bork. Comparative assessment of large-scale data sets of protein–protein inter-
actions. Nature, 417(6887):399–403, 2002. Publisher: Nature Publishing Group.

Xiyuan Wang, Haotong Yang, and Muhan Zhang. Neural Common Neighbor with Comple-
tion for Link Prediction, February 2023. URL http://arxiv.org/abs/2302.00890.
arXiv:2302.00890 [cs].

Duncan J. Watts and Steven H. Strogatz. Collective dynamics of ‘small-world’ networks. Na-
ture, 393:440–442, 1998. URL https://api.semanticscholar.org/CorpusID:
3034643.

Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How Powerful are Graph Neural
Networks? CoRR, abs/1810.00826, 2018. URL http://arxiv.org/abs/1810.00826.
arXiv: 1810.00826.

Haoteng Yin, Muhan Zhang, Yanbang Wang, Jianguo Wang, and Pan Li. Algorithm and System Co-
design for Efficient Subgraph-based Graph Representation Learning. Proceedings of the VLDB
Endowment, 15(11):2788–2796, July 2022. ISSN 2150-8097. doi: 10.14778/3551793.3551831.
URL http://arxiv.org/abs/2202.13538. arXiv:2202.13538 [cs].

Seongjun Yun, Seoyoon Kim, Junhyun Lee, Jaewoo Kang, and Hyunwoo J. Kim. Neo-GNNs:
Neighborhood Overlap-aware Graph Neural Networks for Link Prediction. November 2021. URL
https://openreview.net/forum?id=Ic9vRN3VpZ.

Muhan Zhang and Yixin Chen.

In
S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett
(eds.), Advances in Neural
Information Processing Systems, volume 31. Curran Asso-
ciates, Inc., 2018. URL https://proceedings.neurips.cc/paper/2018/file/
53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf.

Link Prediction Based on Graph Neural Networks.

Muhan Zhang and Pan Li. Nested Graph Neural Networks, 2021. URL https://arxiv.org/

abs/2110.13197.

12

Under review as a conference paper at ICLR 2024

Muhan Zhang, Zhicheng Cui, Shali Jiang, and Yixin Chen. Beyond link prediction: Predicting
hyperlinks in adjacency space. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.

Muhan Zhang, Pan Li, Yinglong Xia, Kai Wang, and Long Jin. Labeling Trick: A The-
In M. Ran-
ory of Using Graph Neural Networks for Multi-Node Representation Learning.
zato, A. Beygelzimer, Y. Dauphin, P. S. Liang, and J. Wortman Vaughan (eds.), Ad-
vances in Neural Information Processing Systems, volume 34, pp. 9061–9073. Curran Asso-
ciates, Inc., 2021. URL https://proceedings.neurips.cc/paper/2021/file/
4be49c79f233b4f4070794825c323733-Paper.pdf.

Tao Zhou, Linyuan L¨u, and Yi-Cheng Zhang. Predicting missing links via local information. The

European Physical Journal B, 71(4):623–630, 2009. Publisher: Springer.

Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal Xhonneux, and Jian Tang. Neural bellman-ford net-
works: A general graph neural network framework for link prediction. Advances in Neural Infor-
mation Processing Systems, 34, 2021.

13

Under review as a conference paper at ICLR 2024

A RELATED WORK

Link prediction Link prediction, inherent to graph data analysis, has witnessed a paradigm shift
from its conventional heuristic-based methods to the contemporary, more sophisticated GNNs ap-
proaches. Initial explorations in this domain primarily revolve around heuristic methods such as
CN, AA, RA, alongside seminal heuristics like the Katz Index (Katz, 1953), Jaccard Index (Salton
& McGill, 1986), Page Rank (Brin & Page, 1998), and Preferential Attachment (Barab´asi & Al-
bert, 1999). However, the emergence of graphs associated with node attributes has shifted the
research landscape towards GNN-based methods. Specifically, these GNN-centric techniques bi-
furcate into node-level and link-level paradigms. Pioneers like Kipf & Welling introduce the Graph
Auto-Encoder (GAE) to ascertain node pair similarity through GNN-generated node representation.
On the other hand, link-level models, represented by SEAL (Zhang & Chen, 2018), opt for subgraph
extractions centered on node pairs, even though this can present scalability challenges.

Amplifying GNN Expressiveness with Randomness The expressiveness of GNNs, particularly
those of the MPNNs, has been the subject of rigorous exploration (Xu et al., 2018). A known limita-
tion of MPNNs, their equivalence to the 1-Weisfeiler-Lehman test, often results in indistinguishable
representation for non-isomorphic graphs. A suite of contributions has surfaced to boost GNN ex-
pressiveness, of which (Morris et al., 2021; Maron et al., 2020; Zhang & Li, 2021; Frasca et al.,
2022) stand out. An elegant, yet effective paradigm involves symmetry-breaking through stochas-
ticity injection (Sato et al., 2021; Abboud et al., 2021; Papp et al., 2021). Although enhancing
expressiveness, such random perturbations can occasionally undermine generalizability. Diverging
from these approaches, our methodology exploits probabilistic orthogonality within random vectors,
culminating in a robust structural feature estimator that introduces minimal estimator variance.

Link-Level Link Prediction While node-level models like GAE offer enviable efficiency, they oc-
casionally fall short in performance when compared with rudimentary heuristics (Chamberlain et al.,
2022). Efforts to build scalable link-level alternatives have culminated in innovative methods such as
Neo-GNN (Yun et al., 2021), which distills structural features from adjacency matrices for link pre-
diction. Elsewhere, ELPH (Chamberlain et al., 2022) harnesses hashing mechanisms for structural
feature representation, while NCNC (Wang et al., 2023) adeptly aggregates common neighbors’
node representation. Notably, DotHash (Nunes et al., 2023), which profoundly influenced our ap-
proach, employs quasi-orthogonal random vectors for set similarity computations, applying these in
link prediction tasks.

Distinctively, our proposition builds upon, yet diversifies from, the frameworks of ELPH and
DotHash. While resonating with ELPH’s architectural spirit, we utilize a streamlined, efficacious
hashing technique over MinHash for set similarity computations. Moreover, we resolve ELPH’s
limitations through strategic implementations like shortcut removal and norm rescaling. When
paralleled with DotHash, our approach magnifies its potential, integrating it with GNNs for link
predictions and extrapolating its applicability to multi-hop scenarios. It also judiciously optimizes
variance induced by the structural feature estimator in sync with graph data. We further explore the
potential of achieving higher expressiveness with linear computational complexity by estimating the
substructure counting (Chen et al., 2020).

B EXPERIMENTAL DETAILS

B.1 BENCHMARK DATASETS

The statistics of each benchmark dataset are shown in Table 3. The benchmarks without attributes
are:

• USAir (Batagelj & Mrvar, 2006): a graph of US airlines;

• NS (Newman, 2006): a collaboration network of network science researchers;

• PB (Ackland & others, 2005): a graph of links between web pages on US political topics;

• Yeast (Von Mering et al., 2002): a protein-protein interaction network in yeast;

• C.ele (Watts & Strogatz, 1998): the neural network of Caenorhabditis elegans;

14

Under review as a conference paper at ICLR 2024

Table 3: Statistics of benchmark datasets.

Dataset

#Nodes

#Edges Avg. node deg.

Std. node deg. Max. node deg. Density Attr. Dimension

C.ele

Yeast

Power

Router

USAir

E.coli

NS

PB

CS

Physics

Computers

Photo

Collab

297

2375

4941

5022

332

1805

1589

1222

18333

34493

13752

7650

4296

23386

13188

12516

4252

29320

5484

33428

163788

495924

491722

238162

235868

2358104

14.46

9.85

2.67

2.49

12.81

16.24

3.45

27.36

8.93

14.38

35.76

31.13

10.00

12.97

15.50

1.79

5.29

20.13

48.38

3.47

38.42

9.11

15.57

70.31

47.28

18.98

134

118

19

106

139

1030

34

351

136

382

2992

1434

671

9.7734%

0.8295%

0.1081%

0.0993%

7.7385%

1.8009%

0.4347%

4.4808%

0.0975%

0.0834%

0.5201%

0.8140%

0.0085%

-

-

-

-

-

-

-

-

6805

8415

767

745

128

• Power (Watts & Strogatz, 1998): the network of the western US’s electric grid;

• Router (Spring et al., 2002): the Internet connection at the router-level;

• E.coli (Zhang et al., 2018): the reaction network of metabolites in Escherichia coli.

4 out of 5 benchmarks with node attributes come from (Shchur et al., 2019), while Collab is from
Open Graph Benchmark (Hu et al., 2021):

• CS: co-authorship graphs in the field of computer science, where nodes represent authors,
edges represent that two authors collaborated on a paper, and node features indicate the
keywords for each author’s papers;

• Physics: co-authorship graphs in the field of physics with the same node/edge/feature def-

inition as of CS;

• Computers: a segment of the Amazon co-purchase graph for computer-related equipment,
where nodes represent goods, edges represent that two goods are frequently purchased
together together, and node features represent the product reviews;

• Physics: a segment of the Amazon co-purchase graph for photo-related equipment with the

same node/edge/feature definition as of Computers;

• Collab: a large-scale collaboration network, showcasing a wide array of interdisciplinary

partnerships.

Since Collab has a fixed split, no train test split is needed for it. For the other benchmarks, we
randomly split the edges into 70-10-20 as train, validation, and test sets. The validation and test sets
are not observed in the graph during the entire cycle of training and testing. They are only used for
evaluation purposes. For Collab, it is allowed to use the validation set in the graph when evaluating
on the test set.

We run the experiments 10 times on each dataset with different splits. For each run, we cache the
split edges and evaluate every model on the same split to ensure a fair comparison. The average
score and standard deviation are reported for Hits@50.

B.2 MORE DETAILS IN BASELINE METHODS

In our experiments, we explore advanced variants of the baseline models ELPH and NCNC. Specif-
ically, for ELPH, Chamberlain et al. (2022) propose BUDDY, an enhanced link prediction method
that preprocesses node representations for efficiency. NCNC (Wang et al., 2023) builds upon its
predecessor, NCN, by first estimating the complete graph structure and then performing inference.

15

Under review as a conference paper at ICLR 2024

We incorporate these latest and most accurate versions of both models to establish robust baselines
in our study.

B.3 EVALUATION DETAILS: INFERENCE TIME

In Figure 4, we assess the inference time across different models on the Collab dataset for a single
epoch of test links. Specifically, we clock the wall time taken by models to score the complete test
set. This encompasses preprocessing, message-passing, and the actual prediction. For the SEAL
model, we employ a dynamic subgraph generator during the preprocessing phase, which dynam-
ically computes the subgraph. Meanwhile, for both ELPH and our proposed method, MPLP, we
initially propagate the node features and signatures just once at the onset of inference. These are
then cached for subsequent scoring sessions.

B.4 SOFTWARE AND HARDWARE DETAILS

We implement MPLP in Pytorch Geometric framework (Fey & Lenssen, 2019). We run our experi-
ments on a Linux system equipped with an NVIDIA V100 GPU with 32GB of memory.

B.5 TIME COMPLEXITY

The efficiency of MPLP stands out when it comes to link prediction inference. Let’s denote t as the
number of target links, d as the maximum node degree, r as the number of hops to compute, and F
as the dimension count of node signatures.

For preprocessing node signatures, MPLP involves two primary steps:

1. Initially, the algorithm computes all-pairs unweighted shortest paths across the input graph
v for each node. This can be achieved using a

to acquire the shortest-path neighborhood N s
BFS approach for each node, with a time complexity of O(|V ||E|).

2. Following this, MPLP propagates the QO vectors through the shortest-path neighborhood,

which has a complexity of O(tdrF ), and then caches these vectors in memory.

During online scoring, MPLP performs the inner product operation with a complexity of O(tF ),
enabling the extraction of structural feature estimations.

However, during training, the graph’s structure might vary depending on the batch of target links
due to the shortcut removal operation. As such, MPLP proceeds in three primary steps:

1. Firstly, the algorithm extracts the r-hop induced subgraph corresponding to these t target
links. In essence, we deploy a BFS starting at each node of the target links to determine their
receptive fields. This process, conceptually similar to message-passing but in a reversed
message flow, has a time complexity of O(tdr). Note that, different from SEAL, we extract
one r-hop subgraph induced from a batch of target links.

2. To identify the shortest-path neighborhood N s

v , we simply apply sparse-sparse matrix
multiplications of the adjacency matrix to get the s-power adjacency matrix, where s =
1, 2, . . . , r. Due to the sparsity, this takes O(|V |dr).

3. Finally, the algorithm engages in message-passing to propagate the QO vectors along the
shortest-path neighborhoods, with a complexity of O(tdrF ), followed by performing the
inner product at O(tF ).

Summing up, the overall time complexity for the training phase stands at O(tdr + |V |dr + tdrF ).

B.6 HYPERPARAMETERS

We determine the optimal hyperparameters for our model through systematic exploration. The set-
ting with the best performance on the validation set is selected. The chosen hyperparameters are as
follows:

• Number of Hops (r): We set the maximum number of hops to r = 2. Empirical evaluation
suggests this provides an optimal trade-off between accuracy and computational efficiency.

16

Under review as a conference paper at ICLR 2024

Figure 6: Heatmap illustrating the inner product of node attributes across CS, Photo, and Collab
datasets.

Figure 7: Heatmap illustrating the inner product of node attributes, arranged by node labels, across
CS and Photo. The rightmost showcases the inner product of QO vectors.

• Node Signature Dimension (F ): The dimension of node signatures, F , is fixed at 1024,
except for Collab with 2048. This configuration ensures that MPLP is both efficient and
accurate across all benchmark datasets.

• The minimum degree of nodes to be considered as hubs (b): This parameter indicates the
minimum degree of the nodes which are considered as hubs to one-hot encode in the node
signatures. We experiment with values in the set [50, 100, 150].

• Batch Size (B): We vary the batch size depending on the graph type: For the 8 non-
attributed graphs, we explore batch sizes within [512, 1024]. For the 5 attributed graphs,
we extend our search to [2048, 4096].

More ablation study can be found in Appendix D.4.

C EXPLORING BAG-OF-WORDS NODE ATTRIBUTES

In Section 3, we delved into the capability of GNNs to discern joint structural features, particularly
when presented with Quasi-Orthogonal (QO) vectors. Notably, many graph benchmarks utilize text
data to construct node attributes, representing them as Bag-Of-Words (BOW). BOW is a method
that counts word occurrences, assigning these counts as dimensional values. With a large dictio-
nary, these BOW node attribute vectors often lean towards QO due to the sparse nature of word
representations. Consequently, many node attributes in graph benchmarks inherently possess the
QO trait. Acknowledging GNNs’ proficiency with QO vector input, we propose the question: Is it
the QO property or the information embedded within these attributes that significantly impacts link
prediction in benchmarks? This section is an empirical exploration of this inquiry.

17

Under review as a conference paper at ICLR 2024

Table 4: Performance comparison of GNNs using node attributes versus random vectors (Hits@50).
For simplicity, all GNNs are configured with two layers.

CS

Physics

Computers

Photo

Collab

GCN
GCN(random feat)

66.00±2.90
51.67±2.70

73.71±2.28
69.55±2.45

22.95±10.58
35.86±3.17

28.14±7.81
46.84±2.53

35.53±2.39
17.25±1.15

SAGE
SAGE(random feat)

57.79±18.23
11.78±1.62

74.10±2.51
64.71±3.65

1.86±2.53
29.23±3.92

5.70±10.15
39.94±3.41

36.82±7.41
28.87±2.36

GCN(F = 1000)
GCN(F = 2000)
GCN(F = 3000)
GCN(F = 4000)
GCN(F = 5000)
GCN(F = 6000)
GCN(F = 7000)
GCN(F = 8000)
GCN(F = 9000)
GCN(F = 10000)

3.73±1.44
24.97±2.67
39.51±6.47
43.23±3.37
48.25±3.28
51.44±1.50
52.00±1.74
54.21±3.47
53.16±2.80
55.91±2.63

Random feat
49.28±2.74
49.13±4.64
53.76±3.85
61.86±4.10
63.19±4.31
65.10±4.11
66.76±3.32
69.27±2.94
70.79±2.83
71.88±3.29

36.92±3.36
40.24±3.04
42.33±3.82
42.85±3.60
44.52±2.78
44.90±2.74
45.11±3.69
44.47±4.11
45.03±3.13
45.26±1.94

48.72±3.84
53.49±3.50
56.27±3.47
56.87±3.59
58.13±3.79
58.10±3.35
57.41±2.62
58.67±3.90
57.15±3.87
58.12±2.54

31.93±2.10
40.16±1.70
47.22±1.60
50.40±1.28
52.13±1.02
53.78±0.84
55.04±1.06
55.36±1.15
OOM
OOM

C.1 NODE ATTRIBUTE ORTHOGONALITY

Our inquiry begins with the assessment of node attribute orthogonality across three attributed graphs:
CS, Photo, and Collab. CS possesses extensive BOW vocabulary, resulting in node attributes span-
ning over 8000 dimensions. Contrarily, Photo has a comparatively minimal dictionary, encompass-
ing just 745 dimensions. Collab, deriving node attributes from word embeddings, limits to 128
dimensions.

For our analysis, we sample 10000 nodes (7650 for Photo) and compute the inner product of their
attributes. The results are visualized in Figure 6. Our findings confirm that with a larger BOW di-
mension, CS node attributes closely follow QO. However, this orthogonality isn’t as pronounced in
Photo and Collab—especially Collab, where word embeddings replace BOW. Given that increased
node signature dimensions can mitigate estimation variance (as elaborated in Theorem 2), one could
posit GNNs might offer enhanced performance on CS, due to its extensive BOW dimensions. Em-
pirical evidence from Table 2 supports this claim.

Further, in Figure 7, we showcase the inner product of node attributes in CS and Photo, but this
time, nodes are sequenced by class labels. This order reveals that nodes sharing labels tend to have
diminished orthogonality compared to random pairs—a potential variance amplifier in structural
feature estimation using node attributes.

C.2 ROLE OF NODE ATTRIBUTE INFORMATION

To discern the role of embedded information within node attributes, we replace the original attributes
in CS, Photo, and Collab with random vectors—denoted as random feat. These vectors maintain the
original attribute dimensions, though each dimension gets randomly assigned values from {−1, 1}.
The subsequent findings are summarized in Table 4. Intriguingly, even with this “noise” as input,
performance remains largely unaltered. CS attributes appear to convey valuable insights for link
predictions, but the same isn’t evident for the other datasets. In fact, introducing random vectors
to Computers and Photo resulted in enhanced outcomes, perhaps due to their original attribute’s in-
sufficient orthogonality hampering effective structural feature capture. Collab shows a performance
drop with random vectors, implying that the original word embedding can contribute more to the
link prediction than structural feature estimation with merely 128 QO vectors.

18

Under review as a conference paper at ICLR 2024

Figure 8: MSE of estimation for #(2, 2), #(2, 0) and estimation time on Collab. Lower values are
better.

C.3 EXPANDING QO VECTOR DIMENSIONS

Lastly, we substitute node attributes with QO vectors of varied dimensions, utilizing GCN as the en-
coder. The outcomes of this experiment are cataloged in Table 4. What’s striking is that GCNs, when
furnished with lengthier random vectors, often amplify link prediction results across datasets, with
the exception of CS. On Computers and Photo, a GCN even rivals our proposed model (Figure 2),
potentially attributed to the enlarged vector dimensions. This suggests that when computational re-
sources permit, expanding our main experiment’s node signature dimensions (currently set at 1024)
could elevate our model’s performance. On Collab. the performance increases significantly com-
pared to the experiments which are input with 128-dimensional vectors, indicating that the structural
features are more critical for Collab than the word embedding.

D ADDITIONAL EXPERIMENTS

D.1 NODE LABEL ESTIMATION ACCURACY AND TIME

In Figure 5, we assess the accuracy of node label count estimation. For ELPH, the node signature
dimension corresponds to the number of MinHash permutations. We employ a default hyperpa-
rameter setting for Hyperloglog, with p = 8, a configuration that has demonstrated its adequacy
in (Chamberlain et al., 2022). For time efficiency evaluation, we initially propagate and cache node
signatures, followed by performing the estimation.

Furthermore, we evaluate the node label count estimation for #(2, 2) and #(2, 0). The outcomes are
detailed in Figure 8. While MPLP consistently surpasses ELPH in estimation accuracy, the gains
achieved via one-hot hubs diminish for #(2, 2) and #(2, 0) relative to node counts at a shortest-
path distance of 1. This diminishing performance gain can be attributed to our selection criteria for
one-hot encoding, which prioritizes nodes that function as hubs within a one-hop radius. However,
one-hop hubs don’t necessarily serve as two-hop hubs. While we haven’t identified a performance
drop for these two-hop node label counts, an intriguing avenue for future research would be to refine
variance reduction strategies for both one-hop and two-hop estimations simultaneously.

Regarding the efficiency of estimation, MPLP consistently demonstrates superior computational
efficiency in contrast to ELPH. When we increase the node signature dimension to minimize estima-
tion variance, ELPH’s time complexity grows exponentially and becomes impractical. In contrast,
MPLP displays a sublinear surge in estimation duration.

It’s also worth noting that ELPH exhausts available memory when the node signature dimension
surpasses 3000. This constraint arises as ELPH, while estimating structural features, has to cache
node signatures for both MinHash and Hyperloglog. Conversely, MPLP maintains efficiency by
caching only one type of node signatures.

19

Under review as a conference paper at ICLR 2024

Table 5: Ablation study on non-attributed benchmarks evaluated by Hits@50. The format is average
score ± standard deviation. The top three models are colored by First, Second, Third.

USAir

NS

PB

Yeast

C.ele

Power

Router

E.coli

w/o Shortcut removal
w/o One-hot hubs
w/o Norm rescaling
MPLP

80.94±3.49
84.04±4.53
85.04±2.64
85.19±4.59

85.47±2.60
89.45±2.60
89.34±2.79
89.58±2.60

49.51±3.57
51.49±2.63
52.50±2.90
52.84±3.39

82.62±0.99
85.11±0.62
83.01±1.03
85.11±0.62

57.51±2.09
66.85±3.04
66.81±4.11
67.97±2.96

19.99±2.54
29.54±1.79
29.00±2.30
29.54±1.79

36.67±10.03
50.81±3.74
50.43±3.59
51.04±4.03

76.94±1.54
79.07±2.47
79.36±2.18
79.35±2.35

Table 6: Ablation study on attributed benchmarks evaluated by Hits@50. The format is average
score ± standard deviation. The top three models are colored by First, Second, Third.

CS

Physics

Computers

Photo

Collab

w/o Shortcut removal
w/o One-hot hubs
w/o Norm rescaling
MPLP

41.63±7.27
65.49±4.28
65.20±2.92
65.70±3.86

62.58±2.40
71.58±2.28
67.73±2.54
71.03±3.55

32.74±3.03
36.09±4.08
35.83±3.24
37.56±3.57

52.09±2.52
55.63±2.48
52.59±3.57
55.63±2.48

60.45±1.44
65.07±0.47
63.99±0.59
66.07±0.47

D.2 MODEL ENHANCEMENT ABLATION

We investigate the individual performance contributions of three primary components in MPLP:
Shortcut removal, One-hot hubs, and Norm rescaling. To ensure a fair comparison, we maintain
consistent hyperparameters across benchmark datasets, modifying only the specific component un-
der evaluation. Moreover, node attributes are excluded from the model’s input for this analysis. The
outcomes of this investigation are detailed in Table 5 and Table 6.

Among the three components, Shortcut removal emerges as the most pivotal for MPLP. This high-
lights the essential role of ensuring the structural distribution of positive links is aligned between the
training and testing datasets (Dong et al., 2022).

Regarding One-hot hubs, while they exhibited strong results in the estimation accuracy evaluations
presented in Figure 5 and Figure 8, their impact on the overall performance is relatively subdued.
We hypothesize that, in the context of these sparse benchmark graphs, the estimation variance may
not be sufficiently influential on the model’s outcomes.

Finally, Norm rescaling stands out as a significant enhancement in MPLP. This is particularly evident
in its positive impact on datasets like Yeast, Physics, Photo, and Collab.

D.3 STRUCTURAL FEATURES ABLATION

We further examine the contribution of various structural features to the link prediction task. These
features include: #(1, 1), #(1, 2), #(1, 0), #(2, 2), #(2, 0), and #(△). To ensure fair comparison,
we utilize only the structural features for link representation, excluding the node representations
derived from GNN(·). Given the combinatorial nature of these features, they are grouped into four
categories:

• #(1, 1);
• #(1, 2), #(1, 0);
• #(2, 2), #(2, 0);
• #(△).

The configuration of these structural features and their corresponding results are detailed in Table 7
and Table 8.

Our analysis reveals that distinct benchmark datasets have varied preferences for structural features,
reflecting their unique underlying distributions. For example, datasets PB and Power exhibit superior
performance with 2-hop structural features, whereas others predominantly favor 1-hop features.
Although #(1, 1), which counts Common Neighbors, is often considered pivotal for link prediction,
the two other 1-hop structural features, #(1, 2) and #(1, 0), demonstrate a more pronounced impact

20

Under review as a conference paper at ICLR 2024

Table 7: The mapping between the configuration number and the used structural features in MPLP.

Configurations

#(1, 1)

#(1, 2)

#(1, 0)

#(2, 2)

#(2, 0)

#(△)

(1)
(2)
(3)
(4)
(5)
(6)
(7)
(8)
(9)
(10)
(11)
(12)
(13)
(14)
(15)

(cid:33)

-
-
-
(cid:33)
(cid:33)
(cid:33)

-
-
-
(cid:33)
(cid:33)
(cid:33)

-
(cid:33)

-
(cid:33)

-
-
(cid:33)

-
-
(cid:33)
(cid:33)

-
(cid:33)
(cid:33)

-
(cid:33)
(cid:33)

-
(cid:33)

-
-
(cid:33)

-
-
(cid:33)
(cid:33)

-
(cid:33)
(cid:33)

-
(cid:33)
(cid:33)

-
-
(cid:33)

-
-
(cid:33)

-
(cid:33)

-
(cid:33)
(cid:33)

-
(cid:33)
(cid:33)
(cid:33)

-
-
(cid:33)

-
-
(cid:33)

-
(cid:33)

-
(cid:33)
(cid:33)

-
(cid:33)
(cid:33)
(cid:33)

-
-
-
(cid:33)

-
-
(cid:33)

-
(cid:33)
(cid:33)

-
(cid:33)
(cid:33)
(cid:33)
(cid:33)

Table 8: Ablation analysis highlighting the impact of various structural features on link prediction.
Refer to Table 7 for detailed configurations of the structural features used.

Configurations

USAir

NS

PB

Yeast

C.ele

Power

Router

E.coli

(1)
(2)
(3)
(4)
(5)
(6)
(7)
(8)
(9)
(10)
(11)
(12)
(13)
(14)
(15)

76.64±26.74
82.54±4.61
67.76±23.65
37.18±37.57
86.24±2.70
77.41±5.27
71.11±25.51
80.16±4.82
75.13±26.51
76.82±4.28
82.82±5.52
87.29±1.08
78.21±2.74
80.75±5.02
81.06±6.62

75.26±2.79
84.76±3.63
70.05±2.35
25.13±1.99
84.91±2.80
80.00±2.39
76.72±2.37
88.67±2.72
87.28±3.33
77.04±3.70
88.91±2.90
88.08±2.59
88.08±3.27
89.14±2.38
89.73±2.12

37.48±13.30
41.84±15.51
44.81±2.63
12.35±10.75
48.35±3.76
46.05±2.76
43.57±3.70
52.16±2.25
48.10±3.43
45.42±2.77
52.57±3.05
48.86±3.42
46.00±2.31
51.63±2.67
53.49±2.66

58.70±30.50
80.56±0.65
67.02±2.53
7.42±10.80
84.42±0.56
74.70±1.45
73.08±1.23
82.52±0.85
80.84±0.97
67.34±3.20
84.61±0.67
84.59±0.69
74.88±2.49
82.68±0.67
85.06±0.69

46.22±24.84
56.22±20.39
36.53±19.68
30.75±18.69
66.69±3.60
46.88±5.79
54.99±20.14
63.82±4.02
60.63±4.54
41.66±13.47
67.11±2.52
66.06±3.74
54.64±4.99
63.01±3.21
66.41±3.02

14.40±1.40
21.38±1.46
25.24±4.07
5.47±1.13
22.25±1.39
27.74±3.23
14.50±1.64
28.41±2.00
23.85±1.37
26.95±1.47
28.98±1.73
23.79±1.87
28.82±1.29
29.41±1.44
28.86±2.40

17.29±3.96
48.97±3.34
21.32±2.66
30.47±3.10
49.68±3.79
22.37±2.06
31.26±2.87
50.97±3.57
49.78±3.56
28.31±2.76
50.63±3.72
50.06±3.66
26.24±2.18
51.08±4.12
50.63±3.79

60.10±30.80
67.78±23.83
56.59±1.78
34.90±36.63
80.94±1.62
71.41±2.47
80.22±2.09
77.26±1.31
76.13±1.81
70.14±0.77
80.16±2.20
79.57±2.46
74.67±3.96
76.88±1.86
78.91±2.58

on link prediction outcomes. Meanwhile, while the count of triangles, #(△), possesses theoretical
significance for model expressiveness, it seems less influential for link prediction when assessed in
isolation. However, its presence can bolster link prediction performance when combined with other
key structural features.

D.4 PARAMETER SENSITIVITY

We perform an ablation study to assess the hyperparameter sensitivity of MPLP, focusing specifi-
cally on two parameters: Batch Size (B) and Node Signature Dimension (F ).

Our heightened attention to B stems from its role during training. Within each batch, MPLP ex-
ecutes the shortcut removal. Ideally, if B = 1, only one target link would be removed, thereby

21

Under review as a conference paper at ICLR 2024

Table 9: Ablation study of Batch Size (B) on non-attributed benchmarks evaluated by Hits@50. The
format is average score ± standard deviation. The top three models are colored by First, Second,
Third.

USAir

NS

PB

Yeast

C.ele

Power

Router

E.coli

MPLP(B = 256)
MPLP(B = 512)
MPLP(B = 1024)
MPLP(B = 2048)
MPLP(B = 4096)
MPLP(B = 8192)

90.31±1.32
90.40±2.47
90.49±2.22
81.20±2.80
81.20±2.80
81.20±2.80

88.98±2.48
89.40±2.12
88.49±2.34
61.79±18.55
61.79±18.55
56.20±21.34

51.14±2.44
49.63±2.08
50.60±3.40
50.34±3.05
52.59±2.36
51.91±2.08

84.07±0.69
84.17±0.60
83.67±0.57
76.79±6.79
58.26±7.20
24.47±21.12

71.59±2.83
71.72±3.35
70.61±4.13
31.79±19.88
31.54±18.53
31.79±19.88

28.92±1.67
28.60±1.66
28.63±1.60
28.45±1.88
27.25±3.30
17.22±3.17

56.15±3.80
53.25±6.57
49.75±5.14
49.37±3.89
50.26±3.89
38.67±7.78

85.12±1.00
84.72±1.04
84.52±1.03
84.43±1.28
85.15±1.15
85.67±0.90

Table 10: Ablation study of Batch Size (B) on attributed benchmarks evaluated by Hits@50. The
format is average score ± standard deviation. The top three models are colored by First, Second,
Third.

CS

Physics

Computers

Photo

MPLP(B = 256)
MPLP(B = 512)
MPLP(B = 1024)
MPLP(B = 2048)
MPLP(B = 4096)
MPLP(B = 8192)

74.96±1.87
75.61±2.25
74.89±2.00
75.02±2.68
75.46±1.78
75.26±1.91

76.06±1.47
75.38±1.79
74.89±1.97
75.47±1.68
74.88±2.57
74.14±2.17

43.38±2.83
42.95±2.56
42.69±2.41
41.39±2.87
40.65±2.85
40.00±3.40

57.58±2.92
57.19±2.51
56.97±3.20
55.89±3.03
55.89±2.88
55.90±2.52

preserving the local structures of other links. However, this approach is computationally ineffi-
cient. Although shortcut removal can markedly enhance performance and address the distribution
shift issue (as elaborated in Appendix D.2), it can also inadvertently modify the graph structure.
Thus, striking a balance between computational efficiency and minimal graph structure alteration is
essential.

Our findings are delineated in Table 9, Table 10, Table 11, and Table 12. Concerning the batch
size, our results indicate that opting for a smaller batch size typically benefits performance. However,
if this size is increased past a certain benchmark threshold, there can be a noticeable performance
drop. This underscores the importance of pinpointing an optimal batch size for MPLP. Regarding the
node signature dimension, our data suggests that utilizing longer QO vectors consistently improves
accuracy by reducing variance. This implies that, where resources allow, selecting a more substantial
node signature dimension is consistently advantageous.

D.5 EXPERIMENTAL RESULTS UNDER DIFFERENT METRICS

We extend our model evaluation to include additional metrics such as Hits@20 and Hits@100, with
the results detailed in Table 13, Table 14, Table 15, and Table 16. In the Hits@20 metric, MPLP
maintains its lead, ranking as the top model in 6 out of 8 non-attributed datasets and excelling in
3 attributed datasets. For Hits@100, MPLP consistently ranks among the top two models across
all non-attributed datasets and achieves the best performance in 3 attributed benchmarks, securing
second place in the Physics and Collab dataset.

Table 11: Ablation study of Node Signature Dimension (F ) on non-attributed benchmarks evaluated
by Hits@50. The format is average score ± standard deviation. The top three models are colored by
First, Second, Third.

USAir

NS

PB

Yeast

C.ele

Power

Router

E.coli

MPLP(F = 256)
MPLP(F = 512)
MPLP(F = 1024)
MPLP(F = 2048)
MPLP(F = 4096)

90.64±2.50
90.49±1.95
90.16±1.61
90.14±2.24
89.95±1.48

88.52±3.07
89.18±2.35
89.40±2.12
89.36±1.92
89.54±2.22

50.42±3.86
51.48±2.63
50.60±3.40
51.26±1.67
51.07±2.87

80.63±0.84
82.41±1.10
83.87±1.06
84.20±1.02
84.89±0.64

70.89±4.70
70.91±4.68
70.61±4.13
72.24±3.31
71.91±3.52

25.74±1.59
27.58±1.80
28.88±2.24
29.27±1.92
29.26±1.51

51.84±2.90
51.98±4.38
53.92±2.88
54.50±4.52
54.71±5.07

84.60±0.92
84.70±1.33
84.81±0.85
84.58±1.42
84.67±0.61

22

Under review as a conference paper at ICLR 2024

Table 12: Ablation study of Node Signature Dimension (F ) on attributed benchmarks evaluated by
Hits@50. The format is average score ± standard deviation. The top three models are colored by
First, Second, Third.

CS

Physics

Computers

Photo

MPLP(F = 256)
MPLP(F = 512)
MPLP(F = 1024)
MPLP(F = 2048)
MPLP(F = 4096)

74.90±1.88
74.67±2.63
75.02±2.68
75.30±2.14
76.04±1.57

73.91±1.41
74.49±2.05
75.27±2.95
75.82±2.15
76.17±2.04

40.65±3.24
39.36±2.28
42.27±3.96
41.98±3.21
43.33±2.93

55.13±2.98
55.93±3.31
55.89±3.03
57.11±2.56
58.55±2.47

Table 13: Link prediction results on non-attributed benchmarks evaluated by Hits@20. The format
is average score ± standard deviation. The top three models are colored by First, Second, Third.

USAir

NS

PB

Yeast

C.ele

Power

CN
AA
RA

GCN
SAGE

65.55±4.10
74.66±4.64
77.81±3.34

74.00±1.98
74.00±1.98
74.00±1.98

23.66±3.11
24.65±3.20
22.66±4.69

61.95±4.94
73.04±4.03

75.57±4.25
48.92±10.31

23.60±5.63
31.14±2.27

60.44±3.32
66.70±2.73
67.97±2.48

66.09±2.96
62.42±6.51

28.23±8.47
38.74±5.00
39.74±3.79

23.89±3.75
36.25±3.43

11.57±0.55
11.57±0.55
11.57±0.55

12.37±3.20
3.36±0.80

Router

9.38±1.05
9.38±1.05
9.38±1.05

19.35±3.40
25.01±6.42

E.coli

47.46±1.60
58.03±2.93
67.60±1.99

55.39±9.43
66.81±4.82

76.68±6.32
SEAL
Neo-GNN 74.14±4.49
74.21±4.94
ELPH
74.40±3.70
NCNC

81.33±3.69
80.97±2.73
85.42±2.19
80.95±2.46

27.97±1.63
28.96±2.51
26.88±4.18
28.47±3.76

76.50±2.30
75.35±3.38
68.13±3.76
72.48±2.47

41.33±5.26
38.18±3.88
41.61±4.68
36.15±5.22

26.56±1.62
19.53±5.53
22.00±1.78
19.86±1.05

46.80±12.24
31.94±4.39
48.97±3.60
37.23±10.52

74.21±3.24
65.42±4.07
66.48±2.30
74.85±4.58

MPLP

83.67±2.75

83.67±3.66

33.69±2.64

81.95±1.20

50.07±3.27

28.70±1.32

44.04±5.08

80.37±1.89

Table 14: Link prediction results on attributed benchmarks evaluated by Hits@20. The format is
average score ± standard deviation. The top three models are colored by First, Second, Third.

CS

Physics

Computers

Photo

Collab

CN
AA
RA

GCN
SAGE

38.86±0.28
57.94±3.66
57.97±2.72

44.17±0.13
58.27±3.35
56.12±3.65

12.92±1.96
14.13±2.45
14.02±1.57

18.97±3.02
23.18±3.70
24.21±5.37

50.13±5.89
44.50±19.37

56.58±3.48
61.32±3.95

14.63±7.09
21.92±2.88

16.69±4.88
31.71±3.62

49.98±0.00
55.79±0.00
55.01±0.00

24.39±1.37
18.74±4.64

54.46±2.04
SEAL
Neo-GNN 58.17±4.06
55.67±2.87
ELPH
59.99±3.61
NCNC

57.66±3.13
58.64±3.30
48.05±3.17
60.80±5.71

16.81±0.90
14.84±1.19
17.33±2.73
22.40±3.82

27.13±2.23
28.00±2.86
27.18±2.54
29.08±6.52

54.66±0.91
49.73±0.82
59.92±0.19
56.89±4.40

MPLP

62.41±3.07

60.43±3.92

25.58±3.76

37.53±3.18

56.69±1.15

Table 15: Link prediction results on non-attributed benchmarks evaluated by Hits@100. The format
is average score ± standard deviation. The top three models are colored by First, Second, Third.

USAir

NS

PB

Yeast

C.ele

Power

CN
AA
RA

GCN
SAGE

84.31±4.21
90.80±1.67
90.80±1.67

80.75±3.86
90.92±1.69

74.00±1.98
74.00±1.98
74.00±1.98

80.29±2.64
65.15±7.99

95.74±1.18
SEAL
Neo-GNN 91.53±1.63
94.52±0.94
ELPH
91.13±1.80
NCNC

91.02±3.01
85.29±3.57
92.01±1.28
84.87±3.76

49.15±3.87
53.07±3.30
53.91±3.67

49.19±4.35
61.00±2.43

59.87±1.93
58.38±2.67
61.11±2.81
59.34±2.87

73.76±0.86
73.76±0.86
73.76±0.86

77.13±1.89
77.34±2.79

56.69±1.55
75.66±2.24
75.76±2.12

59.04±4.34
78.04±4.32

11.57±0.55
11.57±0.55
11.57±0.55

20.52±3.02
11.51±1.30

87.82±0.66
84.98±0.80
85.92±0.57
85.92±0.78

82.35±3.11
77.76±3.05
80.70±2.48
76.64±3.02

38.85±2.65
26.46±3.94
33.49±1.42
27.31±2.39

Router

9.38±1.05
9.38±1.05
9.38±1.05

28.55±5.88
55.96±4.52

71.32±4.97
49.95±4.54
69.26±1.88
63.93±6.35

E.coli

58.00±1.48
74.83±1.48
78.70±0.65

64.78±12.96
80.45±1.65

86.95±0.67
78.75±1.70
80.04±1.41
87.82±0.65

MPLP

95.41±1.36

92.01±1.56

65.98±2.40

86.78±0.69

88.30±2.09

37.03±1.12

69.79±1.22

89.66±0.68

23

Under review as a conference paper at ICLR 2024

Table 16: Link prediction results on attributed benchmarks evaluated by Hits@100. The format is
average score ± standard deviation. The top three models are colored by First, Second, Third.

CS

Physics

Computers

Photo

Collab

CN
AA
RA

GCN
SAGE

69.05±0.31
69.05±0.31
69.05±0.31

63.39±0.14
80.85±0.80
80.87±0.91

29.58±2.27
38.02±1.16
41.74±1.52

41.74±1.87
50.46±2.33
55.09±2.98

73.94±1.69
68.71±14.24

83.01±0.60
82.59±1.04

31.36±13.66
45.79±4.06

38.15±10.88
59.44±2.11

65.60±0.00
65.60±0.00
65.60±0.00

47.40±2.08
48.86±5.19

76.81±0.88
SEAL
Neo-GNN 76.43±1.32
79.03±1.67
ELPH
81.32±0.70
NCNC

81.77±1.89
81.02±0.80
75.90±1.97
84.41±0.90

44.92±1.08
34.38±1.16
41.40±2.65
49.38±3.53

62.93±3.32
58.13±2.92
57.80±2.52
61.52±2.48

70.24±0.25
62.34±0.20
70.43±1.28
71.87±0.18

MPLP

82.25±1.01

83.02±1.02

53.92±1.43

70.20±2.56

71.55±0.40

E THEORETICAL ANALYSIS

E.1 PROOF FOR THEOREM 1

We begin by restating Theorem 1 and then proceed with its proof:

Let G = (V, E) be a non-attributed graph and consider a 1-layer GCN/SAGE. Define the input
vectors X ∈ RN ×F initialized randomly from a zero-mean distribution with standard deviation
σnode. Additionally, let the weight matrix W ∈ RF ′×F be initialized from a zero-mean distri-
bution with standard deviation σweight. After performing message passing, for any pair of nodes
{(u, v)|(u, v) ∈ V × V \ E}, the expected value of their inner product is given by:

For GCN:

For SAGE:

E(hu · hv) =

C
(cid:112) ˆdu

ˆdv

(cid:88)

k∈Nu

(cid:84) Nv

1
ˆdk

,

E(hu · hv) =

√

C
dudv

(cid:88)

1,

k∈Nu

(cid:84) Nv

where ˆdv = dv + 1 and the constant C is defined as C = σ2

nodeσ2

weightF F ′.

Proof. Define X as (cid:0)X ⊤
Using GCN as the MPNN, the node representation is updated by:

and W as (W1, W2, . . . , WF ).

1 , . . . , X ⊤
N

(cid:1)⊤

hu = W

(cid:88)

k∈N (u)∪{u}

1
(cid:112) ˆdk

ˆdu

Xk,

where ˆdv = dv + 1.

24

Under review as a conference paper at ICLR 2024

For any two nodes (u, v) from {(u, v)|(u, v) ∈ V × V \ E}, we compute:

hu · hv = h⊤


u hv

=

W

(cid:88)

1
(cid:112) ˆda

ˆdu



⊤ 

Xa



W

(cid:88)

b∈N (v)∪{v}



Xb



1
(cid:112) ˆdb

ˆdv

(cid:88)

a∈N (u)∪{u}
1
(cid:112) ˆda

ˆdu

a∈N (u)∪{u}

X ⊤

a W ⊤W

(cid:88)

a∈N (u)∪{u}

1
(cid:112) ˆda

ˆdu

X ⊤
a






W ⊤
1 W1
...
W ⊤
F W1

b∈N (v)∪{v}

Xb

(cid:88)

ˆdv

1
(cid:112) ˆdb
· · · W ⊤
1 WF
...
...
· · · W ⊤
F WF






(cid:88)

b∈N (v)∪{v}

1
(cid:112) ˆdb

ˆdv

Xb.

=

=

Given that

1. E(cid:0)W ⊤
2. E(cid:0)W ⊤

i Wj

i Wj

(cid:1) = σ2

weightF ′ when i = j,

(cid:1) = 0 when i ̸= j,

we obtain:

E(hu · hv) = σ2

weightF ′ (cid:88)

a∈N (u)∪{u}

1
(cid:112) ˆda

ˆdu

X ⊤
a

(cid:88)

b∈N (v)∪{v}

1
(cid:112) ˆdb

ˆdv

Xb.

Also the orthogonal of the random vectors guarantee that E(cid:0)X ⊤
have:

a Xb

(cid:1) = 0 when a ̸= b. Then, we

E(hu · hv) =

C
(cid:112) ˆdu

ˆdv

(cid:88)

k∈Nu

(cid:84) Nv

1
ˆdk

where C = σ2

nodeσ2

weightF F ′.

This completes the proof for the GCN variant. A similar approach, utilizing the probabilistic or-
thogonality of the input vectors and weight matrix, can be employed to derive the expected value for
SAGE as the MPNN.

E.2 PROOF FOR THEOREM 2

We begin by restating Theorem 2 and then proceed with its proof:
Let G = (V, E) be a graph, and let the vector dimension be given by F ∈ N+. Define the input
vectors X = (Xi,j), which are initialized from a random variable x having a mean of 0 and a
. Using the message-passing as described by Equation 3, for any pair of
standard deviation of
nodes {(u, v)|(u, v) ∈ V × V }, the expected value and variance of their inner product are:

1√
F

E(hu · hv) = CN(u, v),

Var(hu · hv) =

1
F

(cid:0)dudv + CN(u, v)2 − 2CN(u, v)(cid:1) + F Var(cid:0)x2(cid:1)CN(u, v).

Proof. We follow the proof of the theorem in Nunes et al. (2023). Based on the message-passing
defined in Equation 3:

(cid:32)(cid:32)

E(hu · hv) = E

(cid:88)

(cid:33)

(cid:32)

Xku,:

·

(cid:88)

Xkv,:

(cid:33)(cid:33)

ku∈Nu

(cid:88)

(cid:88)

(cid:32)

= E

ku∈Nu

kv∈Nv

kv∈Nv
(cid:33)

Xku,:Xkv,:

(cid:88)

(cid:88)

=

ku∈Nu

kv∈Nv

E(Xku,:Xkv,:).

25

Under review as a conference paper at ICLR 2024

Since the sampling of each dimension is independent of each other, we get:

E(hu · hv) =

(cid:88)

(cid:88)

F
(cid:88)

ku∈Nu

kv∈Nv

i=1

E(Xku,iXkv,i).

E(Xku,iXkv,i) = E(cid:0)x2(cid:1) =

1
F

.

E(Xku,iXkv,i) = E(Xku,i)E(Xkv,i) = 0.

When ku = kv,

When ku ̸= kv,

Thus:

E(hu · hv) =

(cid:88)

(cid:88)

F
(cid:88)

1 (ku = kv)

1
F

kv∈Nv

i=1

ku∈Nu
(cid:88)

=

1 = CN(u, v).

For the variance, we separate the equal from the non-equal pairs of ku and kv. Note that there is no
covariance between the equal pairs and the non-equal pairs due to the independence:

k∈Nu∩Nv

Var(hu · hv) = Var

(cid:32)

(cid:88)

(cid:88)

F
(cid:88)

Xku,iXkv,i

(cid:33)

(cid:33)

kv∈Nv

i=1

ku∈Nu
(cid:32)

(cid:88)

(cid:88)

kv∈Nv

Xku,iXkv,i

=

=

F
(cid:88)

i=1

F
(cid:88)

i=1

Var



Var

ku∈Nu
(cid:32)

(cid:88)

(cid:33)

x2



+ Var



(cid:88)

(cid:88)





Xku,iXkv,i



 .

k∈Nu∩Nv

ku∈Nu

kv∈Nv\{ku}

For the first term, we can obtain:

(cid:32)

Var

(cid:88)

(cid:33)

x2

k∈Nu∩Nv

= Var(cid:0)x2(cid:1)CN(u, v).

For the second term, we further split the variance of linear combinations to the linear combinations
of variances and covariances:



Var



(cid:88)

(cid:88)

Xku,iXkv,i

 =

(cid:88)

(cid:88)

Var(Xku,iXkv,i)+



ku∈Nu

kv∈Nv\{ku}

ku∈Nu
(cid:88)

kv∈Nv\{ku}

(cid:88)

a∈Nu\{ku}

b∈Nv\{kv,a}

Cov(Xku,iXkv,i, Xa,iXb,i).

Note that the Cov(Xku,iXkv,i, Xa,iXb,i) is Var(Xku,iXkv,i) = 1
otherwise 0.

F 2 when (ku, kv) = (b, a), and

Thus, we have:




(cid:88)

(cid:88)

Var



Xku,iXkv,i

 =

ku∈Nu

kv∈Nv\{ku}

1
F 2

(cid:0)dudv + CN(u, v)2 − 2CN(u, v)(cid:1) ,

and the variance is:

Var(hu · hv) =

1
F

(cid:0)dudv + CN(u, v)2 − 2CN(u, v)(cid:1) + F Var(cid:0)x2(cid:1)CN(u, v).

26

Under review as a conference paper at ICLR 2024

E.3 PROOF FOR THEOREM 3

We begin by restating Theorem 3 and then proceed with its proof:

Under the conditions defined in Theorem 2, let h(l)
message-passing iteration. We have:
(cid:17)
u · h(q)
h(p)
v

(cid:88)

=

E

(cid:16)

|walks(p)(k, u)||walks(q)(k, v)|,

u denote the vector for node u after the l-th

k∈V
where |walks(l)(u, v)| counts the number of length-l walks between nodes u and v.

Proof. Reinterpreting the message-passing described in Equation 3, we can equivalently express it
as:

ms(l+1)
v

=

ms(l)

u , h(l+1)
v

=

(cid:91)

(cid:88)

h(0)
u ,

(9)

u∈Nv

u∈ms(l+1)
v

where ms(l)
v
The node vector h(l)

refers to a multiset, a union of multisets from its neighbors. Initially, ms(0)

v = {{v}}.

v is derived by summing the initial QO vectors of the multiset’s elements.

We proceed by induction: Base Case (l = 1):
(cid:91)

(cid:91)

ms(1)

v =

ms(0)

u =

{{u}} = {{k|ω ∈ walks(1)(k, v)}}

u∈Nv

u∈Nv

Inductive Step (l ≥ 1): Let’s assume that ms(l)
arbitrary l. Utilizing Equation 9 and the inductive hypothesis, we deduce:

v = {{k|ω ∈ walks(l)(k, v)}} holds true for an

ms(l+1)
v

=

(cid:91)

u∈Nv

{{k|ω ∈ walks(l)(k, u)}}.

If k initiates the l-length walks terminating at v and if v is adjacent to u, then k must similarly
initiate the l-length walks terminating at u. This consolidates our inductive premise.

With the induction established:

(cid:16)

E

u · h(q)
h(p)
v

(cid:17)



= E



(cid:88)

h(0)
ku

·

(cid:88)



h(0)
kv



kv∈ms(q)
The inherent independence among node vectors concludes the proof.

ku∈ms(p)

u

v

F LIMITATIONS

Despite the promising capabilities of MPLP, there are distinct limitations that warrant attention:

1. Training cost vs. inference cost: The computational cost during training significantly out-
weighs that of inference. This arises from the necessity to remove shortcut edges for
positive links in the training phase, causing the graph structure to change across different
batches. This, in turn, mandates a repeated computation of the shortest-path neighborhood.
A potential remedy is to consider only a subset of links in the graph as positive instances
and mask them, enabling a single round of preprocessing. Exploring this approach will be
the focus of future work.

2. Estimation variance influenced by graph structure: The structure of the graph itself can
magnify the variance of our estimations. Specifically, in dense graphs or those with a
high concentration of hubs, the variance can become substantial, thereby compromising
the accuracy of structural feature estimation.

3. Optimality of estimating structural features: Our research demonstrates the feasibility of
using message-passing to derive structural features. However, its optimality remains un-
determined. Message-passing, by nature, involves matrix multiplication operations, which
can pose challenges in terms of computational time and space, particularly for exceedingly
large graphs.

27

