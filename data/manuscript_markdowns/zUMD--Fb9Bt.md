Under review as a conference paper at ICLR 2021

A UNIFIED FRAMEWORK FOR CONVOLUTION-BASED
GRAPH NEURAL NETWORKS

Anonymous authors
Paper under double-blind review

ABSTRACT

Graph Convolutional Networks (GCNs) have attracted a lot of research interest in
the machine learning community in recent years. Although many variants have
been proposed, we still lack a systematic view of different GCN models and deep
understanding of the relations among them. In this paper, we take a step forward
to establish a uniﬁed framework for convolution-based graph neural networks, by
formulating the basic graph convolution operation as an optimization problem in
the graph Fourier space. Under this framework, a variety of popular GCN models,
including the vanilla-GCNs, attention-based GCNs and topology-based GCNs,
can be interpreted as a same optimization problem but with different carefully de-
signed regularizers. This novel perspective enables a better understanding of the
similarities and differences among many widely used GCNs, and may inspire new
approaches for designing better models. As a showcase, we also present a novel
regularization technique under the proposed framework to tackle the oversmooth-
ing problem in graph convolution. The effectiveness of the newly designed model
is validated empirically.

1

INTRODUCTION

Recent years have witnessed a fast development in graph processing by generalizing convolution
operation to graph-structured data, which is known as Graph Convolutional Networks (GCNs) (Kipf
& Welling, 2017). Due to the great success, numerous variants of GCNs have been developed and
extensively adopted in the ﬁeld of social network analysis (Hamilton et al., 2017; Wu et al., 2019a;
Veliˇckovi´c et al., 2018), biology (Zitnik et al., 2018), transportation forecasting (Li et al., 2017) and
natural language processing (Wu et al., 2019b; Yao et al., 2019).

Inspired by GCN, a wide variety of convolution-based graph learning approaches are proposed to
enhance the generalization performance of graph neural networks. Several research aim to achieve
higher expressiveness by exploring higher-order information or introducing additional learning
mechanisms like attention modules. Although proposed from different perspectives, their exist some
connections between these approaches. For example, attention-based GCNs like GAT (Veliˇckovi´c
et al., 2018) and AGNN (Thekumparampil et al., 2018) share the similar intention by adjusting the
adjacency matrix with a function of edge and node features. Similarly, TAGCN (Du et al., 2017)
and MixHop (Kapoor et al., 2019) can be viewed as particular instances of PPNP (Klicpera et al.,
2018) under certain approximation. However, the relations among these graph learning models are
rarely studied and the comparisons are still limited in analyzing generalization performances on
public datasets. As a consequence, we still lack a systematic view of different GCN models and
deep understanding of the relations among them.

In this paper, we resort to the techniques in graph signal processing and attempt to understand
GCN-based approaches from a general perspective. Speciﬁcally, we present a uniﬁed graph convo-
lution framework by establishing graph convolution operations with optimization problems in the
graph Fourier domain. We consider a Laplacian regularized least squares optimization problem and
show that most of the convolution-based approaches can be interpreted in this framework by adding
carefully designed regularizers. Besides vanilla GCNs, we also extend our framework to formulat-
ing non-convolutional operations (Xu et al., 2018a; Hamilton et al., 2017), attention-based GCNs
(Veliˇckovi´c et al., 2018; Thekumparampil et al., 2018) and topology-based GCNs (Klicpera et al.,
2018; Kapoor et al., 2019), which cover a large fraction of the state-of-the-art graph learning ap-

1

Under review as a conference paper at ICLR 2021

proaches. This novel perspective provides a re-interpretation of graph convolution operations and
enables a better understanding of the similarities and differences among many widely used GCNs,
and may inspire new approaches for designing better models.

As a conclusion, we summarize our contributions as follow:

1. We introduce a uniﬁed framework for convolution-based graph neural networks and interpret
various convolution ﬁlters as carefully designed regularizers in the graph Fourier domain, which
provides a general methodology for evaluating and relating different graph learning modules.

2. Based on the proposed framework, we provide new insights on understanding the limitations of
GCNs and show new directions to tackle common problems and improve the generalization per-
formance of current graph neural networks in the graph Fourier domain. Additionally, the uniﬁed
framework can serve as a once-for-all platform for expert-designed modules on convolution-based
approaches, where newly designed modules can be easily implemented on other networks as a plug-
in module with trivial adaptations. We believe that our framework can provide convenience for
designing new graph learning modules and searching for better combinations.

3. As a showcase, we present a novel regularization technique under the proposed framework to
alleviate the oversmoothing problem in graph representation learning. As shown in Section 4, the
newly designed regularizer can be implemented on several convolution-based networks and effec-
tively improve the generalization performance of graph learning models.

2 PRELIMINARY

We start with an overview of the basic concepts of graph signal processing. Let G = (V, A)
denote a graph with node feature vectors where V represents the vertex set consisting of nodes
{v1, v2, . . . , vN } and A = (aij) ∈ RN ×N is the adjacency matrix implying the connectivity be-
tween nodes in the graph. Let D = diag(d(1), . . . , d(N )) ∈ RN ×N be the degree matrix of A
where d(i) = (cid:80)
j∈V aij is the degree of vertex i. Then, L = D − A is the combinatorial Laplacian
and ˜L = I − D(−1/2)AD(−1/2) is the normalized Laplacian of G. Additionally, we let ˜A = A + I
and ˜D = D + I denote the augmented adjacency and degree matrices with added self-loops. Then
˜Lsym = I − ˜D−1/2 ˜A ˜D−1/2 ( ˜Asym = ˜D−1/2 ˜A ˜D−1/2) and ˜Lrw = I − ˜D−1 ˜A ( ˜Arw = ˜D−1 ˜A)
are the augmented symmetric normalized and random walk normalized Laplacian (augmented adja-
cency matrices) of G, respectively.
Let x ∈ RN be a signal on the vertices of the graph. The spectral convolution is deﬁned as a function
of a ﬁlter gθ parameterized in the Fourier domain (Kipf & Welling, 2017):

gθ (cid:63) x = U gθ(Λ)U T x,
(1)
where U and Λ are the eigenvectors and eigenvalues of the normalized Laplacian ˜L. Also, we
follow Hoang & Maehara (2019) and deﬁne the variation ∆ and ˜D-inner product as:

∆(x) =

(cid:88)

i,j∈V

aij(x(i) − x(j))2 = xT Lx,

(x, y) ˜D =

(d(i) + 1)x(i)y(i) = xT ˜Dy,

(2)

(cid:88)

i∈V

which speciﬁes the smoothness and importance of the signal respectively.

3 UNIFIED GRAPH CONVOLUTION FRAMEWORK

With the success of GCNs, a wide variety of convolution-based approaches are proposed which pro-
gressively enhance the expressive power and generalization performance of graph neural networks.
Despite the effectiveness of GCN and its derivatives on speciﬁc tasks, there still lack a comprehen-
sive understanding on the relations and differences among various graph learning modules.

Graph signal processing is a powerful technique which has been adopted in several graph learning
researches (Kipf & Welling, 2017; Hoang & Maehara, 2019; Zhao & Akoglu, 2019). However,
existing researches mainly focus on analyzing the properties of GCNs while ignore the connec-
tions between different graph learning modules. Innovatively, in this work, we consider interpreting
convolution-based approaches from a general perspective with graph signal processing techniques.

2

Under review as a conference paper at ICLR 2021

In speciﬁc, we establish the connections between graph convolution operations and optimization
problems in graph Fourier space, showing the effect of each module explicitly with speciﬁc reg-
ularizers. This novel perspective provides a systematic view of different GCN models and deep
understanding of the relations among them.

3.1 UNIFIED GRAPH CONVOLUTION FRAMEWORK

Several researches have proved that, in the ﬁeld of graph signal processing, the representative fea-
tures are mostly preserved in the low-frequency signals while noises are mostly contained in the
high-frequency signals (Hoang & Maehara, 2019). Based on this observation, numerous graph rep-
resentation learning methods are designed to decrease the high-frequency components, which can
be viewed as low-pass ﬁlters in the graph Fourier space. With similar inspiration, we consider a
Laplacian regularized least squares optimization problem with graph signal regularizers and attempt
to build connections with these ﬁlters.

Deﬁnition 1 Uniﬁed Graph Convolution Framework. Graph convolution ﬁlters can be achieved
by solving the following Laplacian regularized least squares optimization:

min
¯X

(cid:88)

i∈V

(cid:107) ¯x(i) − x(i)(cid:107)2

˜D + λLreg,

(3)

where (cid:107)x(cid:107) ˜D = (cid:112)(x, x) ˜D denotes the norm induced by ˜D.
In the following sections, we will show that a wide range of convolution-based graph neural net-
works can be derived from Deﬁnition 1 with different carefully designed regularizers, and provide
new insights on understanding different graph learning modules from the graph signal perspective.

3.1.1 GRAPH CONVOLUTIONAL NETWORKS

Graph convolutional networks (GCNs) (Kipf & Welling, 2017) are the foundation of numerous graph
learning models and have received widespread concerns. Several researches have demonstrated that
the vanilla GCN is essentially a type of Laplacian smoothing over the whole graph, which makes
the features of the connected nodes similar. Therefore, to reformulate GCNs in the graph Fourier
space, we consider utilizing the variation ∆(x) as the regularizer.

Deﬁnition 2 Vanilla GCNs. Let ¯x(i)i∈V be the estimation of the input observation x(i)i∈V. A
low-pass ﬁlter:

¯X = ˜ArwX,

is the ﬁrst-order approximation of the optimal solution of the following optimization:

min
¯X

(cid:88)

i∈V

(cid:107) ¯x(i) − x(i)(cid:107)2

˜D +

(cid:88)

i,j∈V

aij(cid:107) ¯x(i) − ¯x(j)(cid:107)2
2.

(4)

(5)

Derivations of the deﬁnitions are presented in Appendix A.
As the eigenvalues of the approximated ﬁlter ˜Arw are bounded by 1, it resembles a low-pass ﬁl-
ter that removes the high-frequency signals. By exchanging ˜Arw with ˜Asym (which has the same
eigenvalues as ˜Arw), we obtain the same formulation adopted in GCNs.
It has been stated that the second term ∆(x) in Eq.(5) measures the variation of the estimation ¯x
over the graph structure. By adding this regularizer to the objective function, the obtained ﬁlter em-
phasizes the low-frequency signals through minimizing the variation over the local graph structure,
while keeping the estimation close to the input in the graph Fourier space.

3.1.2 NON-CONVOLUTIONAL OPERATIONS

Residual Connection. Residual connection is ﬁrst proposed by He et al. (2016) and has been
widely adopted in graph representation learning approaches. In the vanilla GCNs, norms of the
eigenvalues of the ﬁlter ˜Arw (or ˜Asym) are bounded by 1 which ensures numerical stability in the
training procedure. However, on the other hand, signals in all frequency band will shrink as the
convolution layer stacks, leading to a consistent information loss. Therefore, adding the residual
connection is deemed to preserve the strength of the input signal.

3

Under review as a conference paper at ICLR 2021

Deﬁnition 3 Residual Connection. A graph convolution ﬁlter with residual connection:

¯X = ˜ArwX + (cid:15)X,
where (cid:15) > 0 controls the strength of residual connection, is the ﬁrst-order approximation of the
optimal solution of the following optimization:
(cid:88)

(cid:88)

(6)

((cid:107) ¯x(i) − x(i)(cid:107)2

˜D − (cid:15)(cid:107) ¯x(i)(cid:107)2

˜D) +

aij(cid:107) ¯x(i) − ¯x(j)(cid:107)2
2.

(7)

min
¯X

i∈V

i,j∈V

By adding the negative regularizer to penalize the estimations with small norms, we can induce the
same formulation as the vanilla graph convolution with residual connection.

Concatenation. Concatenation is practically a residual connection with different learning weights.
Deﬁnition 3’ Concatenation. A graph convolution ﬁlter concatenating with the input signal:

is the ﬁrst-order approximation of the optimal solution of the following optimization:

¯X = ˜ArwX + (cid:15)XΘΘT ,

min
¯X

(cid:88)

i∈V

((cid:107) ¯x(i) − x(i)(cid:107)2

˜D − (cid:15)(cid:107) ¯x(i)Θ(cid:107)2

˜D) +

(cid:88)

i,j∈V

aij(cid:107) ¯x(i) − ¯x(j)(cid:107)2
2,

(8)

(9)

where (cid:15) > 0 controls the strength of concatenation and Θ is the learning coefﬁcient.

Although the learning weights ΘΘT has a constrained expressive capability, it can be compensated
by the following feature learning modules.

3.1.3 ATTENTION-BASED CONVOLUTIONAL NETWORKS

Since the convolution ﬁlters in GCNs are dependent only on the graph structure, GCNs are proved
to have restricted expressive power and may cause the oversmoothing problem. Several researches
try to introduce the attention mechanism to the convolution ﬁlter, learning to assign different
edge weights at each layer based on nodes and edges. GAT (Veliˇckovi´c et al., 2018) and AGNN
(Thekumparampil et al., 2018) compute the attention coefﬁcients as a function of the features of con-
nected nodes, while ECC (Simonovsky & Komodakis, 2017) and GatedGCN (Bresson & Laurent,
2017) consider the activations for each connected edge. Although these approaches have different
insights, they can be all formulated as (See details in Appendix A):

pij = aijfθ(x(i), x(j), eij), i, j ∈ V,

(10)

where eij denotes the edge representation if applicable. Therefore, we replace aij in Deﬁnition 2
with learned coefﬁcients to enforce different regularization strength on the connected edges.

Deﬁnition 4 Attention-based GCNs. An attention-based graph convolution ﬁlter:

¯X = P X,

(11)

is the ﬁrst-order approximation of the optimal solution of the following optimization:

min
¯X

(cid:88)

i∈V

(cid:107) ¯x(i) − x(i)(cid:107)2

˜D +

(cid:88)

i,j∈V

pij(cid:107) ¯x(i) − ¯x(j)(cid:107)2
2,

s.t.

(cid:88)

j∈V

pij = ˜Dii, ∀i ∈ V.

(12)

Notice that we use a normalization trick to constrain the degree of attention matrix to be the same
as the original degree matrix ˜D as we want to preserve the strength of the regularization for each
node. The formulated ﬁlter P corresponds to the matrix ˜D−1p with row sum equals to 1, which is
also consistent with most of the attention-based approaches after normalization. Through adjusting
the regularization strength for edges, nodes with higher attention coefﬁcients tend to have similar
features while the distance for nodes with low attention coefﬁcients will be further.

3.1.4 TOPOLOGY-BASED CONVOLUTIONAL NETWORKS

Attention-based approaches are mostly designed based on the local structure. Besides focusing
on the ﬁrst-order adjacency matrix, several approaches (Klicpera et al., 2018; 2019; Kapoor et al.,
2019; Du et al., 2017) propose to adopt the structural information in the multi-hop neighborhood,

4

Under review as a conference paper at ICLR 2021

which are referred to as topology-based convolutional networks. We start with an analysis of PPNP
(Klicpera et al., 2018) and then derive a general formulation for topology-based approaches.

PPNP. PPNP provides insights towards the propagation scheme by combining message-passing
function with personalized PageRank. As proved in (Xu et al., 2018b), the inﬂuence of node i
on node j is proportional to a k-step random walk, which converges to the limit distribution with
multiple stacked convolution layers. By involving the restart probability, PPNP is able to preserve
the starting node i’s information. Similarly, in Deﬁnition 2, the ﬁrst term can also be viewed as a
regularization of preserving the original signal information. Therefore, we may achieve the same
purpose by adjusting the regularization strength.

Deﬁnition 5 PPNP. A graph convolution ﬁlter with personalized propagation (PPNP):

¯X = α(In − (1 − α) ˜Arw)−1X,
is equivalent to the optimal solution of the following optimization:

min
¯X

α

(cid:88)

i∈V

(cid:107) ¯x(i) − x(i)(cid:107)2

˜D + (1 − α)

(cid:88)

i,j∈V

aij(cid:107) ¯x(i) − ¯x(j)(cid:107)2
2,

(13)

(14)

where α ∈ (0, 1] is the restart probability.

Higher α means a higher possibility to teleport back to the starting node, which is consistent with
the higher regularization on the original signal in (14).

Multi-hop PPNP. One of the possible weakness of the original PPNP is that personalized PageRank
only utilizes the regularizer over the local structure. Therefore, we may improve the expressive
capability by involving multi-hop information, which is equivalent to adding regularizers for higher-
order variations.

Deﬁnition 6 Multi-hop PPNP. Let t be the highest order adopted in the algorithm. A graph convo-
lution ﬁlter with multi-hop personalized propagation (Multi-hop PPNP):

¯X = α0(In −

t
(cid:88)

k=1

αk

˜Ak

rw)−1X,

(15)

where (cid:80)t
the following optimization:

k=0 αk = 1, α0 > 0 and αk ≥ 0, k = 1, 2, . . . , t, is equivalent to the optimal solution of

min
¯X

α0

(cid:88)

i∈V

(cid:107) ¯x(i) − x(i)(cid:107)2

˜D +

t
(cid:88)

(cid:88)

αk

k=1

i,j∈V

a(k)
ij (cid:107) ¯x(i) − ¯x(j)(cid:107)2
2,

(16)

where a(k)
ij
normalization trick in Section 3.1.3 is adopted on {a(k)

is proportional to the transition probability of the k-step random walk and the same
ij }.

Solving Eq.(15) directly is computationally expensive. Therefore, we derive a ﬁrst-order approxi-
mation by Taylor expansion and result in the form of:

T
(cid:88)

¯X = (

αi

i=0

˜Ai

rw)X + O( ˜AT

rwX).

(17)

As the norm of the eigenvalues of ˜Arw are bounded by 1, we can keep the ﬁrst term in Eq.(17) as a
close approximation.

By comparing the approximated solution with topology-based graph convolutional networks, we
ﬁnd that most of the approaches can be reformulated as particular instances of Deﬁnition 6. For
example, the formulation for Mixhop (Kapoor et al., 2019) can be derived as an approximation of
Eq.(17) if we let t = 2 and α0 = α1 = α2 = 1/3. Different learning weights can be applied to each
hop as Section 3.1.2 to concatenate multi-hop signals. See more examples in Appendix B.

3.2 REMARKS

In this section, we build a bridge between graph convolution operations and optimization problems
in the graph Fourier space and provide insights into interpreting graph convolution operations with
regularizers. For conclusion, we rewrite the general form of the uniﬁed framework as follow.

5

Under review as a conference paper at ICLR 2021

Deﬁnition 1’ Uniﬁed Graph Convolution Framework. Convolution-based graph neural networks
can be reformulated (after approximation) as particular instances of the optimal solution of the
following optimization problem:

min
¯X

α0

(cid:88)

i∈V

((cid:107) ¯x(i) − x(i)(cid:107)2

˜D − (cid:15)(cid:107) ¯x(i)Θ(cid:107)2
) +
˜D
(cid:125)

(cid:124)
(cid:123)(cid:122)
Non−Conv

t
(cid:88)

(cid:88)

αk

k=1

i,j∈V

p(k)
ij (cid:107) ¯x(i)Θ(k) − ¯x(j)Θ(k)(cid:107)2
2
(cid:125)
(cid:123)(cid:122)
(cid:124)
Attention−based

+λLreg,

(cid:124)

(cid:123)(cid:122)
Topology−based

(cid:125)

where (cid:80)t

k=0 αk = 1, αk ≥ 0 and (cid:80)

j∈V pij = ˜Dii, ∀i ∈ V.
If we let d be the feature dimension of X, then Θ, Θ(k) ∈ Rd×d are the corresponding learning
weights. Lreg corresponds to the personalized regularizer based on the framework, which can be
effective if carefully designed as we will show in Section 4.

(18)

By establishing the uniﬁed framework, we interpret various convolution ﬁlters as carefully designed
regularizers in the graph Fourier domain, which provides new insights on understanding graph learn-
ing modules from the graph signal perspective. Several graph learning modules are reformulated as
smoothing regularizers over the graph structure with different intentions. While vanilla GCNs fo-
cus on minimizing the variation over the local graph structure, attention-based and topology-based
GCNs take a step forward and concentrate on the differences between connected edges and graph
structure with larger receptive ﬁeld. This novel perspective enables a better understanding of the
similarities and differences among many widely used GCNs, and may inspire new approaches for
designing better models.

4 TACKLING OVERSMOOTHING UNDER THE UNIFIED FRAMEWORK

Based on the proposed framework, we provide new insights on understanding the limitations of
GCNs and inspire a new line of work towards designing better graph learning models. As a show-
case, we present a novel regularization technique under the framework to tackle the oversmooth-
ing problem. It is shown that the newly designed regularizer can be easily implemented on other
convolution-based networks with trivial adaptations and effectively improve the generalization per-
formances of graph learning approaches.

4.1 REGULARIZATION ON FEATURE VARIANCE

Here, we adopt the deﬁnition of feature-wise oversmoothing in (Zhao & Akoglu, 2019). Due to
multiple layers of Laplacian smoothing, all features fall into the same subspace spanned by the
dominated eigenvectors of the normalized adjacency matrix, which also corresponds to the similar
situation described in (Klicpera et al., 2018). To tackle this problem, we propose to penalize the
features when they are close to each other. Speciﬁcally, we consider the pairwise distance between
normalized features, which is summarized as:

δ(X) =

(cid:88)

1
d2

i,j∈d

(cid:107)x·i/(cid:107)x·i(cid:107) − x·j/(cid:107)x·j(cid:107)(cid:107)2
2,

(19)

where d is the feature dimension and x·i ∈ Rn represents the i-th dimension for all nodes. There-
fore, Eq.(19) can be interpreted as a feature variance regularizer, representing the distance between
features after normalization. By adding this regularizer to the uniﬁed framework, the proposed ﬁlter
should have the property to drive different features away.

Deﬁnition 7 Regularized Feature Variance. Let ⊗ be the Kronecker product operator, vec(X) ∈
Rnd be the vectorized signal X. Let DX be a diagonal matrix whose value is deﬁned by DX (i, i) =
(cid:107)x·i(cid:107)2. A graph convolution ﬁlter with regularized feature variance:

vec( ¯X) = (In ⊗ [(α1 + α2)I − α2

˜Arw] − α3[D−1

x (I −

1
d

11T )D−1

x ] ⊗ ˜D−1)−1vec(X)

(20)

is equivalent to the optimal solution of the following optimization:

min
¯X

α1

(cid:88)

i∈V

(cid:107) ¯x(i)−x(i)(cid:107)2

˜D + α2

(cid:88)

i,j∈V

aij(cid:107) ¯x(i)−¯x(j)(cid:107)2

2−α3

1
d

(cid:88)

i,j∈d

(cid:107) ¯x·i/(cid:107)x·i(cid:107)−¯x·j/(cid:107)x·j(cid:107)(cid:107)2

2, (21)

6

Under review as a conference paper at ICLR 2021

Table 1: Test accuracy (%) on transductive learning datasets. We report mean values and standard
deviations in 30 independent experiments. The best results are highlighted with boldface.

Method

Vanilla

Attention

Topology

Dataset
FastGCN (Chen et al., 2018)
DGI (Veliˇckovi´c et al., 2019)
GIN (Xu et al., 2018a)
SGC (Wu et al., 2019a)
GCN (Kipf & Welling, 2017)
GCN+reg (ours)
AGNN (Thekumparampil et al., 2018)
GatedGCN (Bresson & Laurent, 2017)
MoNet (Monti et al., 2017)
GAT (Veliˇckovi´c et al., 2018)
GAT+reg (ours)
TAGCN (Du et al., 2017)
MixHop (Kapoor et al., 2019)
APPNP (Klicpera et al., 2018)
APPNP+reg (ours)

Citeseer
68.8±0.6
71.8±0.7
66.1±0.9
71.9±0.1
70.3±0.4
72.2±0.4
71.6±0.5
72.0±0.4
-
72.5±0.7
73.3±0.4
70.9
71.4±0.8
70.5±0.9
71.9±0.4

Cora
79.8±0.3
82.3±0.6
77.6±1.1
81.0±0.0
81.5±0.5
83.6±0.3
82.7±0.4
82.4±0.6
81.7±0.5
83.0±0.6
83.9±0.6
82.5
81.9±0.4
82.7±0.8
84.0±0.6

Pubmed
76.8±0.6
76.8±0.6
77.0±1.2
78.9±0.0
79.0±0.4
79.8±0.2
78.9±0.4
78.9±0.3
78.8±0.4
78.5±0.3
80.3±0.3
81.1
80.8±0.6
79.4±0.6
80.2±0.3

where α1 > 0, α2, α3 ≥ 0. For computation efﬁciency, we approximate (cid:107) ¯x·i(cid:107) with (cid:107)x·i(cid:107) as we
assume that a single convolution ﬁlter provides little effect to the norm of features.

Calculating the Kronecker product and inverse operators are computationally expensive. Neverthe-
less, we can approximate Eq.(20) via Taylor expansion with an iterative algorithm. If we let:

A = (α1 + α2)I − α2

˜Arw,

C = −α3 ˜D−1,

D = D−1

x (1 −

11T )D−1
x .

B = In,
1
d

Then, a t-order approximated formulation is summarized as:

¯X (0) = X,
¯X (k+1) = X + ¯X (k) − A ¯X (k)B − C ¯X (k)D,

k = 0, 1, . . . , t − 1.

(22)

(23)

(24)

(25)

Through approximation, computation overhead is greatly reduced. See details in the Appendix A.

As far as we are concerned, the advantages of utilizing feature variance regularization are three-
fold. First, the regularizer measures the difference between features, therefore explicitly preventing
all features from falling into the same subspace. Second, the modiﬁed convolution ﬁlter does not
require additional training parameters, avoiding the risk of overﬁtting. Third, the regularizer is
designed based on the proposed uniﬁed framework, which means it can be easily implemented on
other convolution-based networks as a plug-in module.

4.2 DISCUSSION

Several researches have also shared insights on understanding and tackling oversmoothing. It is
shown in (Li et al., 2018) that the graph convolution of GCN is a special form of Laplacian smooth-
ing and the authors try to compensate the long-range dependencies by co-training GCN with a
random walk model. JKNet (Xu et al., 2018b) proved that the inﬂuence score between nodes con-
verges to a ﬁxed distribution when layer stacks, therefore losing local information. As a remedy,
they proposed to concatenate layer-wise representations to perform mixed structural information.
More recently, Oono & Suzuki (2020) theoretically demonstrated that graph neural networks lose
expressive power exponentially due to oversmoothing. Comparing to the aforementioned researches,
our proposed method acts explicitly on the graph signals and can be easily implemented on other
convolution-based networks as a plug-in module with trivial adaptations.

7

Under review as a conference paper at ICLR 2021

Figure 1: Accuracy and mean feature variance on Cora. Use GCN and GAT for comparison.

(a) Accuracy

(b) Mean Feature Variance

4.3 EXPERIMENT

To testify the effectiveness of the regularizer, we empirically validate the proposed method on sev-
eral widely used semi-supervised node classiﬁcation benchmarks, including transductive and in-
ductive settings. As we have stated in Section 4.1, our regularizer can be implemented on vari-
ous convolution-based approaches under the uniﬁed graph convolution framework. Therefore, we
consider three different versions by implementing the regularizer on vanilla-GCNs, attention-based
GCNs and topology-based GCNs. We achieve state-of-the-art results on almost all of the settings
and show the effectiveness of tackling oversmoothing on graph-structured data.

Dataset and Experimental Setup. We conduct experiments on four real-world graph datasets. For
transductive learning, we evaluate our method on the Cora, Citeseer, Pubmed datasets, following
the experimental setup in (Sen et al., 2008). PPI (Zitnik & Leskovec, 2017) is adopted for induc-
tive learning. Dataset statistics and more experimental setups are presented in Appendix C. For
comparison, we categorize state-of-the-art convolution-based graph neural networks into three spe-
ciﬁc classes, corresponding to the three versions of our proposed method. The ﬁrst category is
based on the vanilla-GCN proposed by Kipf & Welling (2017), including GCN, FastGCN (Chen
et al., 2018), SGC (Wu et al., 2019a), GIN (Xu et al., 2018a), and DGI (Veliˇckovi´c et al., 2019).
Since GIN is not initially evaluated on citation networks, we implement GIN following the setting
in (Xu et al., 2018a). The second category corresponds to the attention-based approaches, includ-
ing GAT (Veliˇckovi´c et al., 2018), AGNN (Thekumparampil et al., 2018), MoNet (Monti et al.,
2017) and GatedGCN (Bresson & Laurent, 2017). The last category of approaches is topology-
based GCNs which utilizes the structural information in the multi-hop neighborhood. We consider
APPNP (Klicpera et al., 2018), TAGCN (Du et al., 2017) and MixHop (Kapoor et al., 2019) as the
baselines.

Table 2: Test Micro-F1 Score on inductive learn-
ing dataset. We report mean values and standard
deviations in 5 independent experiments.

Transductive Learning. Table 1 presents the
performance of our method and several state-
of-the-art graph neural networks on transduc-
tive learning datasets. For three classes of
convolution-based approaches, we implement
our regularizer with GCN, GAT and APPNP as
comparisons with other baselines, respectively.
For a fair comparison, we adopt the same net-
work structure, hyperparameters and training
conﬁgurations as baseline models. It is shown
that the proposed model achieves state-of-the-
art results on all three settings. On all of the
datasets, we can observe a 0.5∼1.0% higher
performance after adopting the proposed reg-
ularizer. Notably, the proposed model achieves the highest improvement on the vanilla GCNs as
this simplest version suffers most from the oversmoothing problem. Meanwhile, when combining
with GAT, the model achieves the highest results comparing with almost all the baselines. Con-
sidering that attention mechanism and the regularization on oversmoothing focus on the local and
global properties respectively, this can be an ideal combination for graph representation learning.
We also conduct experiments on three citation networks with random splits and present the result in
Appendix D.

Dataset
GCN (Kipf & Welling, 2017)
GAT (Veliˇckovi´c et al., 2018)
SGC (Wu et al., 2019a)
JKNet (Xu et al., 2018b)
GraphSAGE (Hamilton et al., 2017)
DGI (Veliˇckovi´c et al., 2019)
GCN+reg (ours)
GAT+reg (ours)

PPI
92.4
97.3
66.4
97.6
61.2
63.8
97.69±0.32
98.23±0.08

Inductive Learning. For the inductive learning task, we implement our method on the vanilla
GCN and GAT, and adopt the same experimental setup. Table 2 presents the comparison results on

8

2345678Layer0.70.750.80.85Accuracy(%)GCNGCN+regGATGAT+reg2345678Layer0.250.30.350.40.450.5Mean Feature VarianceGCNGCN+regGATGAT+regUnder review as a conference paper at ICLR 2021

Table 3: Comparison results on transductive learning datasets. We report mean values and standard
deviations in 30 independent experiments. The best results are highlighted with boldface. The
number in the brackets represent the number of GCN layers when achieving the best performance.

Method
GCN + DropEdgeRong et al. (2019)
GCN + PairNormZhao & Akoglu (2019)
GCN + regs (ours)

Citeseer
73.2±0.1 (4)
71.0±0.5 (3)
73.6±0.4 (4)

Cora
84.2±0.6 (6)
82.2±0.6 (2)
84.9±0.2 (5)

Pubmed
80.1±0.3 (6)
79.6±0.6 (4)
81.0±0.4 (5)

inductive learning dataset. It can be seen that our model compares favorably with all the competitive
baselines. On the PPI dataset, out model achieves 0.5∼1% higher on test Micro-F1 score, showing
the effectiveness of applying our method under inductive settings.

Comparison with Other Related Works. To validate the effectiveness of our model, we compare
the proposed regularizer with two state-of-the-art approaches on tackling oversmoothing, DropE-
dge(Rong et al., 2019) and PairNorm(Zhao & Akoglu, 2019). For fair comparison, all approaches
are adopted on vanilla-GCN with 2∼8 layers and show the best performance on three transductive
datasets respectively. As shown in Table 3, our regularizer achieves best performance on all three
settings. As PairNorm is more suitable when a subset of the nodes lack feature vectors, it is less
competitive in the general settings.

Analysis. As we have stated above, the regularizer can be interpreted as the mean feature variance,
which prevents different features from falling into the same subspace. To testify the effect of our
method, we compute the mean pairwise distance (Eq.(19)) of the last hidden layer of GCN and GAT,
with and without regularizer on the Cora dataset. We show the result of models with 2-8 layers in
Figure 1. As we can observe, the feature variances and the accuracies of models with regularization
are comparably higher than vanilla models with obvious gaps. Therefore, after applying the regular-
izer, features are more separated from each other, and the oversmoothing problem is alleviated.

5 CONCLUSION

In this paper, we develop a uniﬁed graph convolution framework by establishing graph convolution
ﬁlters with optimization problems in the graph Fourier space. We show that most convolution-
based graph learning models are equivalent to adding carefully designed regularizers. Besides
vanilla GCN, our framework is extended to formulating non-convolutional operations, attention-
based GCNs and topology-based GCNs, which cover a large fraction of state-of-the-art graph learn-
ing models. On this basis, we propose a novel regularization on tackling the oversmoothing prob-
lem as a showcase, proving the effectiveness of designing new modules based on the framework.
Through the uniﬁed framework, we provide a general methodology for understanding and relating
different graph learning modules, with new insights on tackling common problems and improving
the generalization performance of current graph neural networks in the graph Fourier domain. Mean-
while, the uniﬁed framework can also serve as a once-for-all platform for expert-designed modules
on convolution-based approaches. We hope our work can promote the understandings towards graph
convolutional networks and inspire more insights in this ﬁeld.

REFERENCES

S. Bai, F. Zhang, and P. Torr. Hypergraph convolution and hypergraph attention.

ArXiv,

abs/1901.08150, 2019.

Xavier Bresson and Thomas Laurent.

Residual gated graph convnets.

arXiv preprint

arXiv:1711.07553, 2017.

Jie Chen, Tengfei Ma, and Cao Xiao. Fastgcn: fast learning with graph convolutional networks via

importance sampling. International Conference on Learning Representations, 2018.

Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, S. Bengio, and Cho-Jui Hsieh. Cluster-gcn: An

efﬁcient algorithm for training deep and large graph convolutional networks. 2019.

9

Under review as a conference paper at ICLR 2021

Jian Du, Shanghang Zhang, Guanhang Wu, Jos´e MF Moura, and Soummya Kar. Topology adaptive

graph convolutional networks. CoRR, abs/1710.10370, 2017.

M. Fey, J. E. Lenssen, F. Weichert, and H. M¨uller. Splinecnn: Fast geometric deep learning with
continuous b-spline kernels. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recog-
nition, pp. 869–877, 2018.

Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric.

CoRR, abs/1903.02428, 2019.

Xavier Glorot and Yoshua Bengio. Understanding the difﬁculty of training deep feedforward neural
networks. In Proceedings of the Thirteenth International Conference on Artiﬁcial Intelligence
and Statistics, pp. 249–256, 2010.

Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs.

In Advances in Neural Information Processing Systems, pp. 1024–1034, 2017.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recog-
nition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp.
770–778, 2016.

NT Hoang and Takanori Maehara. Revisiting graph neural networks: All we have is low-pass ﬁlters.

arXiv preprint arXiv:1905.09550, 2019.

Amol Kapoor, Aram Galstyan, Bryan Perozzi, Greg Ver Steeg, Hrayr Harutyunyan, Kristina Ler-
man, Nazanin Alipourfard, and Sami Abu-El-Haija. Mixhop: Higher-order graph convolutional
architectures via sparsiﬁed neighborhood mixing. In International Conference on Machine Learn-
ing, 2019.

Thomas N Kipf and Max Welling. Semi-supervised classiﬁcation with graph convolutional net-

works. In International Conference on Learning Representations, 2017.

Johannes Klicpera, Aleksandar Bojchevski, and Stephan G¨unnemann. Predict then propagate:
Graph neural networks meet personalized pagerank. International Conference on Learning Rep-
resentations, 2018.

Johannes Klicpera, Stefan Weißenberger, and Stephan G¨unnemann. Diffusion improves graph learn-

ing, 2019.

Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional networks for
semi-supervised learning. In Proceedings of the AAAI Conference on Artiﬁcial Intelligence, 2018.

Yaguang Li, Rose Yu, Cyrus Shahabi, and Yan Liu. Diffusion convolutional recurrent neural net-

work: Data-driven trafﬁc forecasting. arXiv preprint arXiv:1707.01926, 2017.

Federico Monti, Davide Boscaini, Jonathan Masci, Emanuele Rodola, Jan Svoboda, and Michael M
Bronstein. Geometric deep learning on graphs and manifolds using mixture model cnns.
In
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5115–
5124, 2017.

Kenta Oono and Taiji Suzuki. Graph neural networks exponentially lose expressive power for node

classiﬁcation. In International Conference on Learning Representations, 2020.

Y. Rong, W. Huang, Tingyang Xu, and Junzhou Huang. Dropedge: Towards the very deep graph

convolutional networks for node classiﬁcation. 2019.

M. Schlichtkrull, Thomas Kipf, P. Bloem, R. V. Berg, Ivan Titov, and M. Welling. Modeling rela-

tional data with graph convolutional networks. In ESWC, 2018.

Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad.

Collective classiﬁcation in network data. AI magazine, 29(3):93–93, 2008.

Oleksandr Shchur, Maximilian Mumme, Aleksandar Bojchevski, and Stephan G¨unnemann. Pitfalls

of graph neural network evaluation. CoRR, abs/1811.05868, 2018.

10

Under review as a conference paper at ICLR 2021

Martin Simonovsky and Nikos Komodakis. Dynamic edge-conditioned ﬁlters in convolutional neu-
ral networks on graphs. In Proceedings of the IEEE conference on computer vision and pattern
recognition, pp. 3693–3702, 2017.

Kiran K Thekumparampil, Chong Wang, Sewoong Oh, and Li-Jia Li. Attention-based graph neural

network for semi-supervised learning. CoRR, abs/1803.03735, 2018.

Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua
Bengio. Graph attention networks. In International Conference on Learning Representations,
2018.

Petar Veliˇckovi´c, William Fedus, William L Hamilton, Pietro Li`o, Yoshua Bengio, and R Devon

Hjelm. Deep graph infomax. International Conference on Learning Representations, 2019.

Felix Wu, Tianyi Zhang, Amauri Holanda de Souza Jr, Christopher Fifty, Tao Yu, and Kilian Q
Weinberger. Simplifying graph convolutional networks. International Conference on Machine
Learning, 2019a.

Shu Wu, Yuyuan Tang, Yanqiao Zhu, Liang Wang, Xing Xie, and Tieniu Tan. Session-based rec-
ommendation with graph neural networks. In Proceedings of the AAAI Conference on Artiﬁcial
Intelligence, volume 33, pp. 346–353, 2019b.

Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural

networks? CoRR, abs/1810.00826, 2018a.

Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, and Stefanie
Jegelka. Representation learning on graphs with jumping knowledge networks. arXiv preprint
arXiv:1806.03536, 2018b.

Liang Yao, Chengsheng Mao, and Yuan Luo. Graph convolutional networks for text classiﬁcation.
In Proceedings of the AAAI Conference on Artiﬁcial Intelligence, volume 33, pp. 7370–7377,
2019.

Lingxiao Zhao and Leman Akoglu. Pairnorm: Tackling oversmoothing in gnns. arXiv preprint

arXiv:1909.12223, 2019.

Marinka Zitnik and Jure Leskovec. Predicting multicellular function through multi-layer tissue

networks. Bioinformatics, 33(14):i190–i198, 2017.

Marinka Zitnik, Monica Agrawal, and Jure Leskovec. Modeling polypharmacy side effects with

graph convolutional networks. Bioinformatics, 34(13):i457–i466, 2018.

11

Under review as a conference paper at ICLR 2021

APPENDIX

A. PROOFS OF THE DEFINITIONS

Deﬁnition 2 Vanilla GCNs. Let ¯x(i)i∈V be the estimation of the input observation x(i)i∈V. A
low-pass ﬁlter:

¯X = ˜ArwX,

is the ﬁrst-order approximation of the optimal solution of the following optimization:

min
¯X

(cid:88)

i∈V

(cid:107) ¯x(i) − x(i)(cid:107)2

˜D +

(cid:88)

i,j∈V

aij(cid:107) ¯x(i) − ¯x(j)(cid:107)2
2.

(26)

(27)

Proof. Let l denote the objective function. We have

l = tr[( ¯X − X)T ˜D( ¯X − X)] + tr( ¯X T L ¯X).

Then,

If we let ∂l

∂ ¯X = 0:

∂l
∂ ¯X

= 2 ˜D( ¯X − X) + 2L ¯X.

( ˜D + L) ¯X = ˜DX
(I + ˜Lrw) ¯X = X.
As the norm of eigenvalues of ˜Arw = I − ˜Lrw is bounded by 1, I + ˜Lrw has eigenvalues in range
[1, 3], which proves that I + ˜Lrw is a positive deﬁnite matrix. Therefore,

¯X = (I + ˜Lrw)−1X.
(28)
Unfortunately, solving the closed-form solution of Eq.(28) is computationally expensive. Neverthe-
less, we can derive a simpler form, ¯X ≈ (I− ˜Lrw)X = ˜ArwX, via ﬁrst-order Taylor approximation
which establishes the Deﬁnition.

Deﬁnition 3 Residual Connection. A graph convolution ﬁlter with residual connection:

¯X = ˜ArwX + (cid:15)X,
(29)
where (cid:15) > 0 controls the strength of residual connection, is the ﬁrst-order approximation of the
optimal solution of the following optimization:
(cid:88)

(cid:88)

((cid:107) ¯x(i) − x(i)(cid:107)2

˜D − (cid:15)(cid:107) ¯x(i)(cid:107)2

˜D) +

aij(cid:107) ¯x(i) − ¯x(j)(cid:107)2
2.

(30)

min
¯X

i∈V

i,j∈V

Proof. Let l denote the objective function. We have

l = tr[( ¯X − X)T ˜D( ¯X − X)] − (cid:15)tr( ¯X T ˜D ¯X) + tr( ¯X T L ¯X).

Then,

If we let ∂l

∂ ¯X = 0:

∂l
∂ ¯X

= 2 ˜D( ¯X − X) + 2(L − (cid:15) ˜D) ¯X.

[(1 − (cid:15)) ˜D + L] ¯X = ˜DX

¯X = [(1 − (cid:15))I + ˜Lrw]−1X
¯X = [I + ( ˜Lrw − (cid:15)I)]−1X.

Therefore, the ﬁrst-order approximation of the optimal solution is

¯X ≈ [I − ( ˜Lrw − (cid:15)I)]X
= ˜ArwX + (cid:15)X.

12

Under review as a conference paper at ICLR 2021

Deﬁnition 3’ Concatenation. A graph convolution ﬁlter concatenating with the input signal:

is the ﬁrst-order approximation of the optimal solution of the following optimization:

¯X = ˜ArwX + (cid:15)XΘΘT ,

min
¯X

(cid:88)

i∈V

((cid:107) ¯x(i) − x(i)(cid:107)2

˜D − (cid:15)(cid:107) ¯x(i)Θ(cid:107)2

˜D) +

(cid:88)

i,j∈V

aij(cid:107) ¯x(i) − ¯x(j)(cid:107)2
2,

(31)

(32)

where (cid:15) > 0 controls the strength of concatenation and Θ is the learning coefﬁcients for the con-
catenated signal.

Proof. Let l denote the objective function. We have

l = tr[( ¯X − X)T ˜D( ¯X − X)] − (cid:15)tr(( ¯XΘ)T ˜D( ¯XΘ)) + tr( ¯X T L ¯X).

Then,

If we let ∂l

∂ ¯X = 0:

∂l
∂ ¯X

= 2 ˜D( ¯X − X) + 2L ¯X − 2(cid:15) ˜D ¯XΘΘT .

( ˜D + L) ¯X − (cid:15) ˜D ¯XΘΘT = ˜DX
(I + ˜Lrw) ¯X − (cid:15) ¯XΘΘT = X.

With the help of the Kronecker product operator ⊗ and ﬁrst-order Taylor expansion, we have
vec( ¯X) = [(I ⊗ (I + ˜Lrw)) − (cid:15)((ΘΘT ) ⊗ I)]−1vec(X)

≈ [2I − (I ⊗ (I + ˜Lrw)) + (cid:15)((ΘΘT ) ⊗ I)]vec(X)
= vec(2X − (I + ˜Lrw)X + (cid:15) ¯XΘΘT )
= vec( ˜ArwX + (cid:15)XΘΘT ).

Deﬁnition 4 Attention-based GCNs. An attention-based graph convolution ﬁlter:

¯X = P X,

(33)

is the ﬁrst-order approximation of the optimal solution of the following optimization:

min
¯X

(cid:88)

i∈V

(cid:107) ¯x(i) − x(i)(cid:107)2

˜D +

(cid:88)

i,j∈V

pij(cid:107) ¯x(i) − ¯x(j)(cid:107)2
2,

s.t.

(cid:88)

j∈V

pij = ˜Dii, ∀i ∈ V.

(34)

Proof. Let l denote the objective function. We have

Then,

If we let ∂l

∂ ¯X = 0:

l = tr[( ¯X − X)T ˜D( ¯X − X)] + tr( ¯X T L ¯X).

∂l
∂ ¯X

= 2 ˜D( ¯X − X) + 2( ˜D − ˜DP ) ¯X.

(2 ˜D − ˜DP ) ¯X = ˜DX
(2I − P ) ¯X = X.

Similarly, we can prove that (2I − P ) is a positive deﬁnite matrix, with eigenvalues in range [1, 3].
Therefore,

¯X = (2I − P )−1X

≈ P X.

13

t
(cid:88)

Under review as a conference paper at ICLR 2021

Deﬁnition 5 & 6 Topology-based GCNs Due to the fact that most of the topology-based models
adopt non-convolutional operations like concatenation, we derive a more general objective function
by combining with the non-convolutional operations:

min
¯X

α0

(cid:88)

i∈V

(cid:107) ¯x(i) − x(i)(cid:107)2

˜D +

t
(cid:88)

(cid:88)

αk

k=1

i,j∈V

a(k)
ij (cid:107) ¯x(i)Θ(k) − ¯x(j)Θ(k)(cid:107)2
2,

(35)

where (cid:80)t
k=0 αk = 1, α0 > 0 and αk ≥ 0, k = 1, 2, . . . , t. If we let d be the feature dimension of
X, Θ(k) ∈ Rd×d correspond to the learning weights for the kth hop neighborhood. Let l denote the
objective function, we have:

∂l
∂ ¯X

= α0 ˜D( ¯X − X) +

t
(cid:88)

k=1

αk( ˜D − ˜D ˜Ak

rw) ¯XΘ(k)(Θ(k))T .

By letting ∂l

∂ ¯X = 0, we have:

α0 ¯X +

t
(cid:88)

k=1

(In − ˜Ak

rw) ¯XΘ(k)(Θ(k))T = α0X.

Therefore, with the help of the Kronecker product operator ⊗ and ﬁrst-order Taylor expansion, we
have

[α0In +

(αkΘ(k)(Θ(k))T ) ⊗ (In − ˜Ak

rw)]vec( ¯X) = α0vec(X).

(36)

k=1

k=1(αkΘ(k)(Θ(k))T ) and (In − ˜Ak

We can observe that (cid:80)t
rw) have non-negative eigenvalues. Due to
the property of the Kronecker product that the eigenvalues of the Kronecker product (A ⊗ B) equal
to the product of eigenvalues of A and B, the ﬁlter (α0In + (cid:80)t
k=1(αkΘ(k)(Θ(k))T ) is proved to
be a positive deﬁnite matrix. Therefore,

vec( ¯X) = α0[α0In +

t
(cid:88)

(αkΘ(k)(Θ(k))T ) ⊗ (In − ˜Ak

rw)]−1vec(X)

k=1

≈ α0[(2 − α0)In −

t
(cid:88)

(αkΘ(k)(Θ(k))T ) ⊗ (In − ˜Ak

rw)]vec(X)

k=1

= α0vec[(2 − α0)X −

t
(cid:88)

k=1

αk(In − ˜Ak

rw)XΘ(k)(Θ(k))T ].

If we let

W (0) =

2 − α0
α0

In −

t
(cid:88)

k=1

αk
α0

Θ(k)(Θ(k))T ;

W (k) = Θ(k)(Θ(k))T ), k = 1, 2, . . . , t;

we can denote the convolution ﬁlter as:

¯X =

t
(cid:88)

k=0

αk

˜Ak

rwXW (k).

(37)

(38)

(39)

As we have stated in the Section 2.2.2, although the learning weights has a constrained expressive
capability, it can be compensated by the following feature learning module. We omit the proofs of
Deﬁnition 5 and 6, as they can be viewed as particular instances of (35).

Deﬁnition 7 Regularized Feature Variance. Let ⊗ be the Kronecker product operator, vec(X) ∈
Rnd be the vectorized signal X. Let DX be a diagonal matrix whose value is deﬁned by DX (i, i) =
(cid:107)x·i(cid:107)2. A graph convolution ﬁlter with regularized feature variance:

vec( ¯X) = (In ⊗ [(α1 + α2)I − α2

˜Arw] − α3[D−1

x (I −

1
d

11T )D−1

x ] ⊗ ˜D−1)−1vec(X)

(40)

14

Under review as a conference paper at ICLR 2021

is equivalent to the optimal solution of the following optimization:

min
¯X

α1

(cid:88)

i∈V

(cid:107) ¯x(i) − x(i)(cid:107)2

˜D + α2

(cid:88)

i,j∈V

aij(cid:107) ¯x(i) − ¯x(j)(cid:107)2

2 − α3

1
d

(cid:88)

i,j∈d

(cid:107) ¯x·i/(cid:107)x·i(cid:107) − ¯x·j/(cid:107)x·j(cid:107)(cid:107)2
2,

(41)
where α1 > 0, α2, α3 ≥ 0. For computation efﬁciency, we approximate D ¯X with DX as we assume
that a single convolution ﬁlter provides little effect to the norm of features.

Proof. Let l denote the objective function. We have

l = α1tr[( ¯X − X)T ˜D( ¯X − X)] + α2( ¯X T L ¯X) − α3tr[ ¯XD−1

x (I −

1
d

11T )D−1

x

¯X T ].

Then,

∂l
∂ ¯X
∂ ¯X = 0:

If we let ∂l

= 2α1 ˜D( ¯X − X) + 2α2L ¯X − 2α3 ¯XD−1

x (I −

1
d

11T )D−1
x .

[(α1 + α2)I − α2 ˜D−1 ˜Arw] ¯X − α3 ˜D−1 ¯XD−1

x (I −

1
d

11T )D−1

x = α1X.

With the help of the Kronecker product operator ⊗, we have

(In ⊗ [(α1 + α2)I − α2

˜Arw] − α3[D−1

x (I −

1
d

11T )D−1

x ] ⊗ ˜D−1)vec( ¯X) = vec(X).

(42)

By setting α3 with a small positive value, the ﬁlter in Eq.(42) is still a positive deﬁnite matrix.
Therefore we complete the proof.

Similarly, we can derive a simpler form via Taylor approximation. If we let:

A = (α1 + α2)I − α2

B = In,
1
d
Then, the ﬁrst-order approximation of Eq.(40) is summarized as:

C = −α3 ˜D−1,

D = D−1

x (1 −

˜Arw,

11T )D−1
x .

(43)

(44)

vec( ¯X) = (BT ⊗ A + DT ⊗ C)−1vec(X)

≈ (2I − BT ⊗ A − DT ⊗ C)vec(X)
= vec(2X − AXB − CXD).

Additionally, we can also derive a t-order approximated formulation:

vec( ¯X (t)) = (I +

t
(cid:88)

[I − (BT ⊗ A + DT ⊗ C)]i)vec(X).

i=1

However, it is computationally expensive to calculate the Kronecker product. Therefore, we consider
utilizing a iterative algorithm. For any 0 ≤ k < t

vec( ¯X (k+1)) = (I +

k+1
(cid:88)

[I − (BT ⊗ A + DT ⊗ C)]i)vec(X)

i=1

= [I − (BT ⊗ A + DT ⊗ C)](I +

k
(cid:88)

[I − (BT ⊗ A + DT ⊗ C)]i)vec(X) + vec(X)

= [I − (BT ⊗ A + DT ⊗ C)]vec( ¯X (k)) + vec(X)
= vec(X + ¯X (k) − A ¯X (k)B − C ¯X (k)D).

i=1

(45)

B. REFORMULATION EXAMPLES

The reformulation examples of GCN derivatives are presented in Table 4.

15

Under review as a conference paper at ICLR 2021

Table 4: Reformulation of convolution-based graph neural networks. D and di in the attention-based
modules are normalization coefﬁcients.

Models

Non-Conv Module

Attention-based Module

Topology-based Module

GIN (Xu et al., 2018a)

Residual Connection

GraphSAGE (Hamilton et al., 2017)

Concatenation

RGCN

(Schlichtkrull

et al.,
2018)

Concatenation

W = (cid:80)

r∈R

1
cr

Wr

-

-

-

SplineCNN (Fey et al., 2018)

-

ECC (Simonovsky & Komodakis, 2017)

Concatenation

pij = hθ(eij)

pij = hθ(eij)

AGNN (Thekumparampil et al., 2018)

-

pij = di

(cid:80)

exp(βcos(xi,xj ))
k∈N(i)∪i exp(βcos(xi,xk))

MoNet (Monti et al., 2017)

Concatenation

p(k)
ij = diexp(− 1

2 (eij − µk)T Σ(−1)

k

(eij − µk))

GAT Veliˇckovi´c et al. (2018)

Concatenation

p(k)
ij = di

(cid:80)

exp(σ(aT
k∈N(i)∪i exp(σ(aT

(k)[θxi||θxj ]))

(k)[θxi||θxk]))

Cluster GCN (Chiang et al., 2019)

Concatenation

P = D( ˜Arw + λdiag( ˜Arw))

SGC (Wu et al., 2019a)

Hyper-Atten (Bai et al., 2019)

APPNP (Klicpera et al., 2018)

GDC (Klicpera et al., 2019)

TAGCN (Du et al., 2017)

-

-

-

-

-

MixHop (Kapoor et al., 2019)

Concatenation

P = D ˜Ak

sym

P = HW B−1H T

-

-

-

-

-

-

-

-

-

-

-

-

-

-

α0 = γ, α1 = 1 − γ

αi = θi

α0 = · · · = αk = 1/(k + 1)

α0 = α1 = α2 = 1/3

Table 5: Dataset Statistics

Dataset
Nodes
Edges
Features
Classes
Training Nodes
Validation Nodes
Test Nodes

Cora
2,708
5,429
1,433
7
140
500
1,000

Citeseer
3,327
4,732
3,703
6
120
500
1,000

Pubmed
19,717
44,338
500
3
60
500
1,000

PPI
56,944(24 graphs)
818,716
50
121(multilabel)
44,906(20 graphs)
6,514(2 graphs)
5,524(2 graphs)

C. DATA STATISTICS AND EXPERIMENTAL SETUPS

We conduct experiments on four real-world graph datasets, whose statistics are listed in Table 5. For
transductive learning, we evaluate our method on the Cora, Citeseer, Pubmed datasets, following
the experimental setup in (Sen et al., 2008). There are 20 nodes per class with labels to be used
for training and all the nodes’ features are available. 500 nodes are used for validation and the
generalization performance is tested on 1000 nodes with unseen labels. PPI (Zitnik & Leskovec,
2017) is adopted for inductive learning, which is a protein-protein interaction dataset containing 20
graphs for training, 2 for validation and 2 for testing while testing graphs remain unobserved during
training.

To ensure a fair comparison with other methods, we implement our module without interfering the
original network structure. In all three settings, we use two convolution layers with hidden dimen-

16

Under review as a conference paper at ICLR 2021

Table 6: Test accuracy (%) on transductive learning datasets with random slits. We report mean
values and standard deviations of the test accuracies over 100 random train/validation/test splits.

Dataset
GCN(Kipf & Welling, 2017)
GAT(Veliˇckovi´c et al., 2018)
MoNet (Monti et al., 2017)
GraphSAGE (Hamilton et al., 2017)
GCN+reg (ours)

Citeseer
71.9±1.9
71.4±1.9
71.2±2.0
71.6±1.9
72.9±1.4

Cora
81.5±1.3
81.8±1.3
81.3±1.3
79.2±7.7
83.6±1.2

Pubmed
77.8±2.9
78.7±2.3
78.6±2.3
77.4±2.2
79.9±1.6

Table 7: Training and test time on Cora. We report mean values in 5 independent experiments. The
best results are highlighted with boldface.
Method
GCN (Kipf & Welling, 2017)
GAT (Veliˇckovi´c et al., 2018)
AGNN (Thekumparampil et al., 2018)
APPNP (Klicpera et al., 2018)
GCN + regs (ours)

Training Time (s) Training Time (ms / epoch) Test Time (ms)
3.8
8.5
7.9
14.2
8.0

1.9
3.3
3.2
13.6
3.2

1.8
5.4
5.3
9.8
5.7

sion h = 64. We set α1 = 0.2, α2 = 0.8 and α3 = 0.05 for all four datasets. We apply L2
regularization with λ = 0.0005 and use dropout on both layers. For training strategy, we initialize
weights using the initialization described in (Glorot & Bengio, 2010) and follow the method pro-
posed in GCN, adopting an early stop if validation loss does not decrease for certain consecutive
epochs. The implementations of baseline models are based on the PyTorch-Geometric library (Fey
& Lenssen, 2019) in all experiments.

D. RANDOM SPLITS

As illustrated in (Shchur et al., 2018), using the same train/validation/test splits of the same datasets
precludes a fair comparison of different architectures. Therefore, we follow the setup in (Shchur
et al., 2018) and evaluate the performance of our model on three citation networks with random
splits. Empirically, for each dataset, we use 20 labeled nodes per class as the training set, 30 nodes
per class as the validation set, and the rest as the test set. For every model, we choose the hy-
perparameters that achieve the best average accuracy on Cora and CiteSeer datasets and applied to
Pubmed dataset.

Table 6 shows the results on three citation networks under the random split setting. As we can
observe, our model consistently achieves higher performances on all the datasets. On Citeseer, our
model achieves higher accuracy than on the original split. On Cora and Pubmed, the test accuracies
of our model are comparable to the original split, while most of the baselines suffer from a serious
decline.

E. TIME CONSUMPTION

As we have shown in Eq.(20), the computation of graph ﬁlter with the regularizer is greatly increased
with Kronecker product and inverse matrix operations. Nevertheless, we approximate the ﬁlter with
an iterative algorithm as stated in Eq.(25) and realize an efﬁcient implement. To empirically testify
the computation efﬁciency, we conduct experiments on Cora and report the training and test time
of several GCN models on a single RTX 2080 Ti GPU. Due to the early stopping rule (see details
in Appendix C), the training epoch for each module is different. The results are shown in Table 7.
As we can observe, when combining with vanilla GCNs, the training and test time of our model is
similar to GAT and AGNN and faster than APPNP.

17

Under review as a conference paper at ICLR 2021

Table 8: Ablation study on the regularization strength. We report mean values and standard devia-
tions in 30 independent experiments. The best results are highlighted with boldface.

Regularization Strength Citeseer

0
0.01
0.05
0.1
0.2

Pubmed
Cora
70.3±0.4 81.5±0.5 79.0±0.4
71.7±0.6 83.0±0.5 79.2±0.6
72.2±0.4 83.6±0.3 79.8±0.2
72.4±0.5 83.5±0.2 79.4±0.7
68.4±1.0 76.8±1.3 70.2±2.2

F. ABLATION STUDY

To analyze the effects of the regularization strength, we conduct experiments on three transductive
datasets and present the results in Table 8. As we can observe, with reasonable choice of the regu-
larization strength, our approach can achieve consistent improvement under all settings. However,
when the regularization strength is too large, the training procedure becomes unstable and the model
performance suffers from a severe decrease.

18

