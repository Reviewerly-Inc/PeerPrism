Published as a conference paper at ICLR 2022

WHY PROPAGATE ALONE?
PARALLEL USE OF LABELS AND FEATURES ON GRAPHS

Yangkun Wang1†, Jiarui Jin1†, Weinan Zhang1, Yongyi Yang2†, Jiuhai Chen3†, Quan Gan4,
Yong Yu1, Zheng Zhang4, Zengfeng Huang2, David Wipf4
1Shanghai Jiao Tong University, 2Fudan University, 3University of Maryland, 4Amazon
daviwipf@amazon.com

ABSTRACT

Graph neural networks (GNNs) and label propagation represent two interrelated
modeling strategies designed to exploit graph structure in tasks such as node prop-
erty prediction. The former is typically based on stacked message-passing layers
that share neighborhood information to transform node features into predictive em-
beddings. In contrast, the latter involves spreading label information to unlabeled
nodes via a parameter-free diffusion process, but operates independently of the
node features. Given then that the material difference is merely whether features
or labels are smoothed across the graph, it is natural to consider combinations of
the two for improving performance. In this regard, we have recently proposed to
use a randomly-selected portion of the training labels as GNN inputs, concatenated
with the original node features for making predictions on the remaining labels.
This so-called label trick accommodates the parallel use of features and labels,
and is foundational to many of the top-ranking submissions on the Open Graph
Benchmark (OGB) leaderboard. And yet despite its wide-spread adoption, thus far
there has been little attempt to carefully unpack exactly what statistical properties
the label trick introduces into the training pipeline, intended or otherwise. To this
end, we prove that under certain simplifying assumptions, the stochastic label trick
can be reduced to an interpretable, deterministic training objective composed of
two factors. The ﬁrst is a data-ﬁtting term that naturally resolves potential label
leakage issues, while the second serves as a regularization factor conditioned on
graph structure that adapts to graph size and connectivity. Later, we leverage
this perspective to motivate a broader range of label trick use cases, and provide
experiments to verify the efﬁcacy of these extensions.

1

INTRODUCTION

Node property prediction is a ubiquitous task involving graph data with node features and/or labels,
with a wide range of instantiations across real-world scenarios such as node classiﬁcation (Velickovic
et al., 2018) and link prediction (Zhang & Chen, 2018), while also empowering graph classiﬁcation
(Gilmer et al., 2017), etc. Different from conventional machine learning problems where there is
typically no explicit non-iid structure among samples, nodes are connected by pre-speciﬁed edges,
and a natural assumption is that labels and features vary smoothly over the graph. This smoothing
assumption has inspired two interrelated lines of research: First, graph neural networks (GNNs) (Kipf
& Welling, 2017; Hamilton et al., 2017; Li et al., 2018; Xu et al., 2018; Liao et al., 2019; Xu et al.,
2019) leverage a parameterized message passing strategy to convert the original node features into
predictive embeddings that reﬂect the features of neighboring nodes. However, this approach does
not directly utilize existing label information beyond their inﬂuence on model parameters through
training. Second, label propagation algorithms (LPA) (Zhu, 2005; Zhou et al., 2003; Zhang & Lee,
2006; Wang & Zhang, 2006; Karasuyama & Mamitsuka, 2013; Gong et al., 2017; Liu et al., 2019)
spread label information via graph diffusion to make predictions, but cannot exploit node features.

As GNNs follow a similar propagation mechanism as the label propagation algorithm, the principal
difference being whether features or labels are smoothed across the graph, it is natural to consider
combinations of the two for improving performance. Examples motivated by this intuition, at least to
varying degrees, include APPNP (Klicpera et al., 2019), Correct and Smooth (C&S) (Huang et al.,

†Work done during internship at Amazon Web Services Shanghai AI Lab.

1

Published as a conference paper at ICLR 2022

2021), GCN-LPA (Wang & Leskovec, 2020), and TPN (Liu et al., 2019). While often effective, these
methods are not all end-to-end trainable and easily paired with arbitrary GNN architectures. And in a
related fashion Jia & Benson (2021) introduce an elegant generative framework that uniﬁes LPA and
GNNs, although the parallel propagation of both features and labels is not explicitly considered.

Among these various combination strategies, our previously proposed label trick (Wang et al., 2021)
has enjoyed widespread success in facilitating the parallel use of node features and labels via a
stochastic label splitting technique. In brief, the basic idea is to use a randomly-selected portion of the
training labels as GNN inputs, concatenated with the original node features for making predictions
on the remaining labels during each mini-batch. The ubiquity of this simple label trick is evidenced
by its adoption across numerous GNN architectures and graph benchmarks (Sun & Wu, 2020; Kong
et al., 2020; Shi et al., 2021; Li et al., 2021). And with respect to practical performance, this technique
is foundational to many of the top-ranking submissions on the Open Graph Benchmark (OGB)
leaderboard (Hu et al., 2020). For example, at the time of this submission, the top 10 results spanning
multiple research teams all rely on the label trick, as do the top 3 results from the recent KDDCUP
2021 Large-Scale Challenge MAG240M-LSC (Hu et al., 2021).

And yet despite its far-reaching adoption, thus far the label trick has been motivated primarily as a
training heuristic without a strong theoretical foundation. Moreover, many aspects of its underlying
operational behavior have not been explored, with non-trivial open questions remaining. For example,
while originally motivated from a stochastic perspective, is the label trick reducible to a more
transparent deterministic form that is amenable to interpretation and analysis? Similarly, are there any
indirect regularization effects with desirable (or possibly undesirable) downstream consequences?
And how do the implicit predictions applied by the model to test nodes during the stochastic training
process compare with the actual deterministic predictions used during inference? If there is a
discrepancy, then the generalization ability of the model could be compromised. And ﬁnally, are
there natural use cases for the label trick that have so far ﬂown under the radar and been missed? We
take a step towards answering these questions via the following two primary contributions:

• We prove that in certain restricted settings, the original stochastic label trick can be reduced to
an interpretable, deterministic training objective composed of two terms: (1) a data-ﬁtting term
that naturally resolves potential label leakage issues and maintains consistent predictions during
training and inference, and (2) a regularization factor conditioned on graph structure that adapts to
graph size and connectivity. Furthermore, complementary experiments applying the label trick
to a broader class of graph neural network models corroborate that similar effects exists in more
practical real-world settings, consistent with our theoretical ﬁndings.

• Although in prior work the label trick has already been integrated within a wide variety of GNN
models, we introduce novel use-cases motivated by our analysis. This includes exploiting the
label trick to: (i) train simple end-to-end variants of label propagation and C&S, and (ii) replace
stochastic use cases of the label trick with more stable, deterministic analogues that are applicable
to GNN models with linear propagation layers such as SGC (Wu et al., 2019), TWIRLS (Yang
et al., 2021) and SIGN (Frasca et al., 2020). Empirical results on node classiﬁcation benchmarks
verify the efﬁcacy of these simple enhancements.

Collectively, these efforts establish a more sturdy foundation for the label trick, and in doing so, help
to ensure that it is not underutilized.

2 BACKGROUND

Consider an undirected graph G = (V, E) with n = |V | nodes, the node feature matrix is denoted by
X ∈ Rn×d and the label matrix of the nodes is denoted by Y ∈ Rn×c, with d and c being the number
of channels of features and labels, respectively. Let A be the (unweighted) adjacency matrix, D the
degree matrix and S = D− 1
2 the symmetric normalized adjacency matrix. The symmetric
normalized Laplacian L can then be formulated as L = I n − S. We also deﬁne a training mask

2 AD− 1

matrix as M tr =

, where w.l.o.g. we are assuming that the ﬁrst m nodes, denoted

(cid:19)

(cid:18)I m 0
0
0

n×n

Dtr, form the training dataset. We use P to denote a propagation matrix, where the speciﬁc P will
be described in each context.

2

Published as a conference paper at ICLR 2022

2.1 LABEL PROPAGATION ALGORITHM
Label propagation is a semi-supervised algorithm that predicts unlabeled nodes by propagating the
observed labels across the edges of the graph, with the underlying smoothness assumption that two
nodes connected by an edge are likely to share the same label. Following Zhou et al. (2003); Yang
et al. (2021), the implicit energy function of label propagation is given by
2 + λ tr[F (cid:62)LF ],

E(F ) = (1 − λ)(cid:107)F − Y tr(cid:107)2

(1)

where F is the smoothed labels, Y tr = M trY is the label matrix of training nodes, and λ ∈ (0, 1)
is a regularization coefﬁcient that determines the trade-off between the two terms. The ﬁrst term is a
ﬁtting constraint, with the intuition that the predictions of a good classiﬁer should remain close to the
initial label assignments, while the second term introduces a smoothness constraint, which favors
similar predictions between neighboring nodes in the graph.

It is not hard to derive that the closed-formed optimal solution of this energy function is given by
F ∗ = P Y , where P = (1 − λ)(I n − λS)−1. However, since the stated inverse is impractical to
compute for large graphs, P Y is often approximated in practice via P ≈ (1 − λ) (cid:80)k
i=0 λiSiY .
From this expression, it follows that F can be estimated by the more efﬁcient iterations F (k+1) =
λSF (k) + (1 − λ)F (0), where F (0) = Y tr and for each k, S smooths the training labels across the
edges of the graph.

2.2 GRAPH NEURAL NETWORKS FOR PROPAGATING NODE FEATURES
In contrast to the propagation of labels across the graph, GNN models transform and propagate
node features using a series of feed-forward neural network layers. Popular examples include GCN
(Kipf & Welling, 2017), GraphSAGE (Hamilton et al., 2017), GAT (Velickovic et al., 2018), and
GIN (Xu et al., 2019). For instance, the layer-wise propagation rule of GCN can be formulated as
X (k+1) = σ(SX (k)W (k)) where σ(·) is an activation function such as ReLU, X (k) is the k-th
layer node representations with X (0) = X, and W (k) is a trainable weight matrix of the k-th layer.
Compared with label propagation, GNNs can sometimes exhibit a more powerful generalization
capability via the interaction between discriminative node features and trainable weights.

2.3 COMBINING LABEL AND FEATURE PROPAGATION
While performing satisfactorily in many circumstances, GNNs only indirectly incorporate ground-
truth training labels via their inﬂuence on the learned model weights. But these labels are not actually
used during inference, which can potentially degrade performance relative to label propagation,
especially when the node features are noisy or unreliable. Therefore, it is natural to consider the
combination of label and feature propagation to synergistically exploit the beneﬁts of both as has
been proposed in Klicpera et al. (2019); Liu et al. (2019); Wang & Leskovec (2020); Wang et al.
(2021); Shi et al. (2021); Huang et al. (2021).

One of the most successful among these hybrid methods is our previously proposed label trick (Wang
et al., 2021), which can be conveniently retroﬁtted within most standard GNN architectures while
facilitating the parallel propagation of labels and features in an end-to-end trainable fashion. As
mentioned previously, a number of top-performing GNN pipelines have already adopted this trick,
which serves to establish its widespread relevance (Sun & Wu, 2020; Kong et al., 2020; Li et al., 2021;
Shi et al., 2021) and motivates our investigation of its properties herein. To this end, we formally
deﬁne the label trick as follows:

Deﬁnition 1 (label trick) The label trick is based on creating random partitions of the training data
as in Dtr = Din ∪ Dout and Din ∩ Dout = ∅, where node labels from Din are concatenated with the
original features X and provided as GNN inputs (for nodes not in Din zero-padding is used), while
the labels from Dout serve in the traditional role as supervision. The resulting training objective then
becomes

E
splits

(cid:104) (cid:88)

i∈Dout

(cid:96)(cid:0) yi, f [X, Y in; W]i

(cid:1)(cid:105)

(2)

where Y in ∈ Rn×c, is deﬁned row-wise as yin,i =
for all i, the function
f (X, Y in; W) represents a message-passing neural network with parameters W and the concate-
nation of X and Y in as inputs, and (cid:96)(·, ·) denotes a point-wise loss function over one sample/node.
At inference time, we then use the deterministic predictor f [X, Y tr; W]i for all test nodes i /∈ Dtr.

(cid:26)yi
0

if i ∈ Din
otherwise

3

Published as a conference paper at ICLR 2022

3 RELIABLE RANDOMNESS THOUGH THE LABEL TRICK

Despite its widespread adoption, the label trick has thus far been motivated as merely a training
heuristic without formal justiﬁcation. To address this issue, we will now attempt to quantify the
induced regularization effect that naturally emerges when using the label trick. However, since the
formal analysis of deep networks is challenging, we initially adopt the simplifying assumption that
the function f from (2) is linear, analogous to the popular SGC model from Wu et al. (2019). For
simplicity of exposition, in Sections 3.1 and 3.2 we will consider the case where no node features are
present to isolate label-trick-speciﬁc phenomena. Later in Sections 3.3 and 3.4 we will reintroduce
node features to present our general results, as well as considering nonlinear extensions.

3.1 LABEL TRICK WITHOUT NODE FEATURES
Assuming no node features X, we begin by considering the deterministic node label loss
(cid:88)

L(W ) =

(cid:96)(cid:0)yi, [P Y trW ]i

(cid:1),

(3)

i∈Dtr

where [ · ]i indicates the i-th row of a matrix P Y trW is a linear predictor akin to SGC, but with only
the zero-padded training label matrix Y tr as an input. Additionally, P here can in principle be any
reasonable propagation matrix, not necessarily the one associated with the original label propagation
algorithm. However, directly employing (3) for training suffers from potential label leakage issues
given that a simple identity mapping sufﬁces to achieve the minimal loss at the expense of accurate
generalization to test nodes. Furthermore, there exists an inherent asymmetry between the predictions
computed for training nodes, where the corresponding labels are also used as inputs to the model,
and the predictions for testing nodes where no labels are available.

At a conceptual level, these issues can be resolved by the label trick, in which case we introduce
random splits of Dtr and modify (3) to

L(W ) = E

splits

(cid:104) (cid:88)

(cid:96)(cid:0)yi, [P Y inW ]i

(cid:1)(cid:105)

i∈Dout

.

(4)

For each random split, the resulting predictor only includes the label information of Din (through
Y in), and thus there is no unresolved label leakage issue when predicting the labels of Dout. In
practice, we typically sample the splits by assigning a given node to Din with some probability α ∈
(0, 1); otherwise the node is set to Dout. It then follows that E[|Din|] = α|Dtr| and E[Y in] = αY tr.

3.2 SELF-EXCLUDED SIMPLIFICATION OF THE LABEL TRICK
Because there exists an exponential number of different possible random splits, for analysis purposes
(with later practical beneﬁts as well) we ﬁrst consider a simpliﬁed one-versus-all case whereby we
enforce that |Dout| = 1 across all random splits, with each node landing with equal probability in
Dout. In this situation, the objective function from (4) can be re-expressed more transparently without
any expectation as
L(W ) = E

(cid:96)(cid:0)yi, [P (Y tr − Y i)W ]i

(cid:96)(cid:0)yi, [P Y inW ]i

(cid:104) (cid:88)

(cid:104) (cid:88)

= E

(cid:1)(cid:105)

(cid:1)(cid:105)

i∈Dout

(5)

splits

=

1
|Dtr|

i∈Dout
(cid:88)

(cid:96)(cid:0)yi, [(P − C)Y trW ]i

i∈Dtr

(cid:1),

splits

where Y i represents a matrix that shares the i-th row of Y and pads the rest with zeros, and
C = diag(P ). This then motivates the revised predictor given by
f(Y tr; W ) = (P − C)Y trW .

(6)

Remark 1 From this expression we observe the intuitive role that C plays in blocking the direct
pathway between each training node input label to the output predicted label for that same node.
In this way the predictor propagates the labels of each training node excluding itself, and for both
training and testing nodes alike, the predicted label of a node is only a function of other node labels.
This resolves the asymmetry mentioned previously with respect to the predictions from (3).

Remark 2 It is generally desirable that a candidate model produces the same predictions on test
nodes during training and inference to better ensure proper generalization. Fortunately, this is in fact
the case when applying (6), which on test nodes makes the same predictions as label propagation. To
see this, note that M te(P − C)Y tr = M teP Y tr, where M te = I n − M tr is the diagonal mask
matrix of test nodes and P Y tr is the original label propagation predictor.

4

Published as a conference paper at ICLR 2022

Although ultimately (6) will serve as a useful analysis tool below, it is also possible to adopt this
predictor in certain practical settings. In this regard, C can be easily computed with the same
computational complexity as is needed to approximate P as discussed in Section 2.1 (and for
alternative propagation operators that are available explicitly, e.g., the normalized adjacency matrix,
C is directly available).
3.3 FULL EXECUTION OF THE LABEL TRICK AS A REGULARIZER
We are now positioned to extend the self-excluded simpliﬁcation of the label trick to full execution
with arbitrary random sampling, as well as later, the reintroduction of node features. For this purpose,
we ﬁrst deﬁne Y out = Y tr − Y in, and also rescale by a factor of 1/α to produce (cid:101)Y in = Y in/α.
The latter allows us to maintain a consistent mean and variance of the predictor across different
sampling probabilities.
Assuming a mean square error (MSE) loss as computed for each node via (cid:96)(y, (cid:98)y) = ||y − (cid:98)y||2
we consider categorical cross-entropy), our overall objective is to minimize

2 (later

L(W ) = E

splits

(cid:104) (cid:88)

i∈Dout

(cid:96)(yi, [P (cid:101)Y inW ]i)

(cid:105)

= E

splits

(cid:104)
(cid:107)Y out − M outP ˜Y inW (cid:107)2
F

(cid:105)

,

(7)

where M out is a diagonal mask matrix deﬁned such that Y out = M outY = Y tr − Y in and the
random splits follow a node-wise Bernoulli distribution with parameter α as discussed previously.
We then have the following:
Theorem 1 Deﬁne Γ = (cid:0) diag(P T P ) − CT C(cid:1) 1

2 Y tr. Then the label trick loss from (7) satisﬁes

(cid:105)

1
1 − α

(cid:104)

E
splits

(cid:107)Y out − M outP ˜Y inW (cid:107)2
F

= (cid:107)Y tr − M tr(P − C)Y trW (cid:107)2

(cid:107)ΓW (cid:107)2
F .
(8)
Note that (diag(P T P ) − CT C) is a positive semi-deﬁnite diagonal matrix, and hence its real
square root will always exist. Furthermore, we can extend this analysis to include node features
by incorporating the SGC-like linear predictor P XW x such that Theorem 1 can then naturally be
generalized as follows:

F +

1 − α
α

Corollary 1 Under the same conditions as Theorem 1, if we add the node feature term P XW x to
the label-based predictor from (7), we have that

1
1 − α

E
splits

(cid:104)
(cid:107)Y out − M outP XW x − M outP ˜Y inW y(cid:107)2
F

(cid:105)

= (cid:107)Y tr − M trP XW x − M tr(P − C)Y trW y(cid:107)2

F +

1 − α
α

(cid:107)ΓW y(cid:107)2
F .

(9)

The details of the proofs of Theorem 1 and Corollary 1 are provided in Appendix B.1. This then
effectively leads to the more general, feature and label aware predictor

f(X, Y tr; W) = P XW x + (P − C)Y trW y,
where W = {W x, W y}. These theoretical results reveal a number of interesting properties
regarding how the label trick behaves, which we summarize as follows:

(10)

Remark 3 Although the original loss involves an expectation over random data splits that is some-
what difﬁcult to interpret, based on (9), the label trick can be interpreted as inducing a deterministic
objective composed of two terms: (i) The error accrued when combining the original node features
with the self-excluded label propagation predictor from (6) to mitigate label leakage, and (ii) An
additional graph-dependent regularization factor on the model weights associated with the labels
that depends on α (more on this below). We can also easily verify from (10) that the model applies
the same prediction to test nodes during both training and inference, consistent with Remark 2.

Remark 4 If the graph has no edges, then there is no chance for overﬁtting to labels and
(cid:0) diag(P T P ) − CT C(cid:1) 1
2 = 0 shuts off the superﬂuous regularization. In contrast, for a fully
connected graph, the value of Γ can be signiﬁcantly larger, which can potentially provide a beneﬁcial
regularization effect. Additionally, given that Γ also grows larger with graph size (assuming edges
grow as well), (cid:107)ΓW y(cid:107)2
F scales proportionately with the data ﬁtting term, which is generally expected
to increase linearly with the number of nodes. Hence (9) is naturally balanced to problem size.

5

Published as a conference paper at ICLR 2022

Remark 5 The splitting probability α in (9) controls the regularization strength. Speciﬁcally, when
α tends to zero, fewer labels are used as input to predict a large number of output labels, which may
be less reliable, and corresponds with adding a larger regularization effect. Additionally, it means
placing more emphasis on the original node features and downplaying the importance of the labels as
input in (9), which explains the addition of a penalty on W y. Conversely, when α tends to one, more
labels are used as input to predict the output and the model approaches the deterministic self-excluded
label trick. Speciﬁcally, for random splits where |Dout| = 1, the loss mimics one random term from
the self-excluded label trick summation, while for the splits when |Dout| = 0, the contribution to the
expectation is zero and therefore does not inﬂuence the loss. Splits with |Dout| > 1 will have very
low probability, so this situation naturally corresponds with canceling out the regularization term.
Later in Section 3.4 we will extend these observations to general nonlinear models.

We now turn to the categorical cross-entropy loss, which is more commonly applied to node classi-
ﬁcation problems. While we can no longer compute closed-form simpliﬁcations as we could with
MSE, it is nonetheless possible to show that the resulting objective when using the original label trick
is an upper bound on the analogous objective from the self-excluded label trick. More speciﬁcally,
we have the following (see Appendix B.3 for the proof):

Theorem 2 Under the same conditions as in Theorem 1 and Corollary 1, if we replace the MSE loss
with categorical cross-entropy we obtain the bound

1
1 − α

E
splits

(cid:104)

(cid:105)
CEDout(Y out, P XW x + P ˜Y inW y)

≥ CEDtr (Y tr, P XW x+(P −C)Y trW y),
(11)

where CES(·, ·) denotes the sum of row-wise cross-entropy of S.

3.4 NONLINEAR EXTENSIONS
When we move towards more complex GNN models with arbitrary nonlinear interactions, it is no
longer feasible to establish explicit, deterministic functional equivalents of the label trick for general
α. However, we can still at least elucidate the situation at the two extremes where α → 0 or α → 1
alluded to in Remark 5. Regarding the former, clearly with probability approaching one, Y in will
always equal zero and hence the model will default to a regular GNN, effectively involving no label
information as an input. In contrast, for the latter we provide the following:

Theorem 3 Let fGN N (X, Y ; W) denote an arbitrary GNN model with concatenated inputs X
(cid:1) is bounded for
and Y , and (cid:96)(y, ˆy) a training loss such that (cid:80)
all Dout. It then follows that

(cid:96)(cid:0)yi, fGN N [X, Y in; W]i

i∈Dout

(cid:40)

lim
α→1

1
1 − α

E
splits

(cid:104) (cid:88)

i∈Dout

(cid:96)(cid:0)yi, fGN N [X, Y in; W]i

(cid:41)

(cid:1)(cid:105)

=

m
(cid:88)

i=1

(cid:96)(cid:0)yi, fGN N [X, Y tr−Y i; W]i

(cid:1).

(12)

The proof is given in Appendix B.4. This result can be viewed as a natural generalization of (5),
with one minor caveat: we can no longer guarantee that the predictor implicitly applied to test nodes
during training will exactly match the explicit function fGN N [X, Y tr; W] applied at inference time.
Indeed, each fGN N [X, Y tr − Y i; W]i will generally produce slightly different predictions for all
test nodes depending on i unless fGN N is linear. But in practice this is unlikely to be consequential.

4 BROADER USE CASES OF THE LABEL TRICK

Although the label trick has already been integrated within a wide variety of GNN pipelines, in this
section we introduce three novel use-cases motivated by our analysis.

4.1 TRAINABLE LABEL PROPAGATION
In Sections 3.1 and 3.2 we excluded the use of node features to simplify the exposition of the label
trick; however, analytical points aside, the presented methodology can also be useful in and of itself
for facilitating a simple, trainable label propagation baseline when we choose P as in Section 2.1.

The original label propagation algorithm from Zhou et al. (2003) is motivated as a parameter-free,
deterministic mapping from a training label set to predictions across the entire graph. However,
clearly the randomized label trick from Section 3.1, or its deterministic simpliﬁcation from Section

6

Published as a conference paper at ICLR 2022

3.2 can be adopted to learn a label propagation weight matrix W . The latter represents a reasonable
enhancement that can potentially help to compensate for interrelated class labels that may arise,
especially in multi-label settings. In contrast, the original label propagation algorithm implicitly
assumes that different classes are independent. Beyond this, other entry points for adding trainable
weights are also feasible such as node-dependent or edge-dependent weights (Wang & Leskovec,
2020), nonlinear weighted mappings (Kipf & Welling, 2017), step-wise weights for heterophily
graphs (Yamaguchi et al., 2016), or weights for different node types for heterogeneous graphs
(Schlichtkrull et al., 2018).

4.2 DETERMINISTIC APPLICATION TO GNNS WITH LINEAR PROPAGATION LAYERS
Many prevailing GNN models follow the architecture of message passing neural networks (MPNNs).
Among these are efﬁcient variants that share node embeddings only through linear propagation layers.
Representative examples include SGC (Wu et al., 2019), SIGN (Frasca et al., 2020) and TWIRLS
(Yang et al., 2021). We now show how to apply the deterministic label trick algorithm as introduced
in Section 3.2 with the aforementioned GNN methods.

We begin with a linear SGC model. In this case, we can compute (P − C)Y tr beforehand as
the self-excluded label information and then train the resulting features individually without graph
information, while avoiding label leakage problems. And if desired, we can also concatenate with the
original node features. In this way, we have an algorithm that minimizes an energy function involving
both labels and input features.

(cid:16) (cid:88)

Additionally, for more complex situations where the propagation layers are not at the beginning of
the model, the predictor can be more complicated such as

f (X, Y ; W) =

(cid:2)Ph0([X, Y − Y i])(cid:3)

i

h1

(13)
where [·, ·] denotes the concatenation operation, P = [P 0, P 1, . . . , P k−1]T is the integrated prop-
agation matrix, C = [diag(P 0), diag(P 1), . . . , diag(P k−1)]T , h0 and h1 can be arbitrary node-
independent functions, typically multi-layer perceptrons (MLPs).

= h1(Ph0([X, Y ]) − Ch0([X, Y ]) + Ch0([X, 0])),

i

(cid:17)

4.3 TRAINABLE CORRECT AND SMOOTH

Correct and Smooth (C&S) (Huang et al., 2021) is a simple yet powerful method which consists of
multiple stages. A prediction matrix ˜Y is ﬁrst obtained whose rows correspond with a prediction
from a shallow node-wise model. ˜Y is subsequently modiﬁed via two post-processing steps, correct
and smooth, using two propagation matrices {P c, P s}, where typically P i = (1 − λi)(I n −
λiS)−1, i ∈ {c, s}. For the former, we compute the difference between the ground truth and
predictions on the training set as E = Y tr − ˜Y tr and then form ˜E = γ(P cE) as the correction
matrix, where γ(·) is some row-independent scaling function. The ﬁnal smoothed prediction is
formed as fC&S( ˜Y ) = P s(Y tr + M te( ˜Y + ˜E)). This formulation is not directly amenable to
end-to-end training because of label leakage issues introduced through Y tr.

In contrast, with the label trick, we can equip C&S with trainable weights to further boost performance.
To this end, we ﬁrst split the training dataset into Din and Dout as before. Then we multiply
˜Ein = γ(P c(Y in − ˜Y in)), the correction matrix with respect to Y in, with a weight matrix W c.
We also multiply the result after smoothing with another weight matrix W s. Thus the predictor under
this split is

fT C&S(Y in, ˜Y ; W) = P s(Y in + (M te + M out)( ˜Y + ˜EinW c))W s

= P s((Y in + (M te + M out) ˜Y )W s + P s(M te + M out) ˜Ein ˆW c,
where W = { ˆW c, W s} and ˆW c = W cW s. The objective function for optimizing W is

L(W ) = E

splits

(cid:104) (cid:88)

(cid:105)
(cid:96)(yi, fT C&S[Y in, ˜Y ; W]i)
.

i∈Dout

(14)

(15)

Since this objective resolves the label leakage issue, it allows end-to-end training of C&S with
gradients passing through both the neural network layers for computing ˜Y and the C&S steps. At
times, however, this approach may have disadvantages, including potential overﬁtting problems
or inefﬁciencies due to computationally expensive backpropagation. Consequently, an alternative
option is to preserve the two-stage training. In this situation, the base prediction in the ﬁrst stage
is the same as the original algorithm; however, we can nonetheless still train the C&S module as a
post-processing step, with parameters as in (14).

7

Published as a conference paper at ICLR 2022

Table 1. Accuracy results (%) of label propagation and
trainable label propagation.

Method

Label Propagation Trainable Label Propagation

Cora-full
Pubmed
ArXiv
Products

66.44 ± 0.93
83.45 ± 0.63
67.11 ± 0.00
74.24 ± 0.00

67.40 ± 0.96
83.52 ± 0.59
68.42 ± 0.01
75.61 ± 0.21

Table 2. R2 on node regression tasks
with GBDT base model and regular/end-
to-end C&S.

Method

C&S

Trainable C&S

House
County
VK
Avazu

0.797 ± 0.005
0.625 ± 0.048
0.163 ± 0.011
0.405 ± 0.036

0.854 ± 0.005
0.788 ± 0.015
0.172 ± 0.011
0.413 ± 0.033

5 EXPERIMENTS

As mentioned previously, the effectiveness of the label trick in improving GNN performance has
already been demonstrated in prior work, and therefore, our goal here is not to repeat these efforts.
Instead, in this section we will focus on conducting experiments that complement our analysis from
Section 3 and showcase the broader application scenarios from Section 4.

The label trick can actually be implemented in two ways. The ﬁrst is the one that randomly splits
the training nodes, and the second is the simpler version introduced herein with the deterministic
one-versus-all splitting strategy, which does not require any random sampling. To differentiate the
two versions, we denote the stochastic label trick with random splits by label trick (S), and the
deterministic one by label trick (D). The latter is an efﬁcient way to approximate the former with a
higher splitting probability α, which is sometimes advantageous in cases where the training process
is slowed down by high α. Accordingly, we conduct experiments with both, thus supporting our
analysis with comparisons involving the two versions.

We use four relatively large datasets for evaluation, namely Cora-full, Pubmed (Sen et al., 2008),
ogbn-arxiv and ogbn-products (Hu et al., 2020). For Cora-full and Pubmed, we randomly split the
nodes into training, validation, and test datasets with the ratio of 6:2:2 using different random seeds.
For ogbn-arxiv and ogbn-products, we adopt the standard split from OGB (Hu et al., 2020). We
report the average classiﬁcation accuracy and standard deviation after 10 runs with different random
seeds, and these are the results on the test dataset when not otherwise speciﬁed. See Appendix C for
further implementation details.

5.1 TRAINABLE LABEL PROPAGATION
We ﬁrst investigate the performance of applying the label trick to label propagation as introduced in
(2) in the absence of features, and compare it with the original label propagation algorithm. Table 1
shows that the trainable weights applied to the label propagation algorithm can boost the performance
consistently. Given the notable simplicity of label propagation, this represents a useful enhancement.

5.2 DETERMINISTIC LABEL TRICK APPLIED TO GNNS WITH LINEAR LAYERS
We also test the deterministic label trick by applying it to different GNN architectures involving
linear propagation layers along with (13). Due to the considerable computational effort required
to produce P and C with large propagation steps for larger graphs, we only conduct tests on the
Cora-full, Pubmed and ogbn-arxiv datasets, where the results are presented in Table 3.

Table 3. Accuracy results (%) with/without label trick (D).

Method
label trick (D)

SGC

(cid:55)

(cid:51)

(cid:55)

SIGN

(cid:51)

(cid:55)

TWIRLS

(cid:51)

Cora-full
Pubmed
ArXiv

65.87 ± 0.61
85.02 ± 0.43
69.07 ± 0.01

65.81 ± 0.69
85.23 ± 0.57
70.22 ± 0.03

68.54 ± 0.76
87.94 ± 0.52
69.97 ± 0.16

68.44 ± 0.88
88.09 ± 0.59
70.98 ± 0.21

70.36 ± 0.51
89.81 ± 0.56
72.93 ± 0.19

70.40 ± 0.71
90.08 ± 0.52
73.22 ± 0.10

From these results we observe that on Pubmed and ogbn-arxiv, the deterministic label trick boosts the
performance consistently on different models, while on Cora-full, it performs comparably. This is
reasonable given that the training accuracy on Cora-full (not shown) is close to 100%, in which case
the model does not beneﬁt signiﬁcantly from the ground-truth training labels, as the label information
is already adequately embedded in the model.

5.3 EFFECT OF SPLITTING PROBABILITY
In terms of the effect of the splitting probability α, we compare the accuracy results of a linear model
and a three-layer GCN on ogbn-arxiv as shown in Figure 1. For the linear model, α serves as a

8

Published as a conference paper at ICLR 2022

Figure 1. Validation accuracy varying α. Left: linear propagation of features & labels, Right: GCN.

regularization coefﬁcient. More speciﬁcally, when α tends to zero, the model converges to the one
without the label trick, while when α tends to one, it converges to the case with the self-excluded
label trick. For the the nonlinear GCN, α has a similar effect as predicted by theory for the linear
models. As α decreases, the model converges to that without the label trick. Moreover, a larger α is
preferable for linear models that may not require strong regularization like more complex GNNs.

5.4 TRAINABLE CORRECT AND SMOOTH

We also verify the effectiveness of our approach when applied to Correct and Smooth (C&S), as
described in Section 4.3. Due to the signiﬁcant impact of end-to-end training on the model, it
is not suitable for direct comparison with vanilla C&S for a natural ablation. Therefore, in this
experiment, we train the C&S as post-processing steps. In Table 4, we report the test and validation
accuracy, showing that our method outperforms the vanilla C&S on Cora-full and Pubmed. And for
ogbn-arxiv and ogbn-products, the trainable C&S performs better in terms of validation accuracy
while is comparable to vanilla C&S in terms of test accuracy.

In principle, C&S can be applied to any base predictor model. To accommodate tabular node features,
we choose to use gradient boosted decision trees (GBDT), which can be trained end-to-end using
methods such as Chen et al. (2022); Ivanov & Prokhorenkova (2021) combined with the label
trick to avoid data leakage issues as we have discussed. For evaluation, we adopt the four tabular
node regression data sets from Ivanov & Prokhorenkova (2021) and train using the approach from
Chen et al. (2022). Results are shown in Table 2, where the label trick can signiﬁcantly improve
performance.

Table 4. Test and validation accuracy (%) of C&S and trainable C&S with MLP as base predictor.
Validation accuracy reported in parentheses.

Method

Cora-full
Pubmed
ArXiv
Products

MLP

MLP+C&S

MLP+Trainable C&S

60.12 ± 0.29 (61.09 ± 0.39)
88.72 ± 0.34 (89.25 ± 0.26)
71.48 ± 0.15 (72.95 ± 0.05)
67.60 ± 0.15 (87.07 ± 0.05)

66.95 ± 1.46 (68.26 ± 1.24)
89.12 ± 0.27 (89.45 ± 0.17)
73.05 ± 0.35 (74.01 ± 0.17)
83.16 ± 0.13 (91.70 ± 0.06)

67.89 ± 1.37 (69.09 ± 1.25)
89.76 ± 0.17 (89.62 ± 0.18)
73.03 ± 0.18 (74.44 ± 0.08)
83.10 ± 0.15 (91.99 ± 0.07)

6 CONCLUSION

In this work we closely examine from a theoretical prospective our recently proposed label trick,
which enables the parallel propagation of labels and features and beneﬁts various SOTA GNN
architectures, and yet has thus far not be subject to rigorous analysis. In ﬁlling up this gap, we ﬁrst
introduce a deterministic self-excluded simpliﬁcation of the label trick, and then prove that the full
stochastic version can be regarded as introducing a regularization effect on the self-excluded label
weights. Beyond this, we also discuss broader applications of the label trick with respect to: (i)
Facilitating the introduction of trainable weights within graph-based methods that were previously
either parameter-free (e.g., label propagation) or not end-to-end (e.g., C&S), and (ii) Eliminating the
effects of randomness by incorporating self-excluded propagation within GNNs composed of linear
propagation layers. We verify these applications and evaluate the performance gains against existing
approaches over multiple benchmark datasets.

9

0.00.20.40.60.81.0α71.071.572.072.573.0Accuracy(%)labeltrick(S)labeltrick(D)w/olabeltrick0.00.20.40.60.81.0α73.273.473.673.874.0Accuracy(%)labeltrick(S)w/olabeltrickPublished as a conference paper at ICLR 2022

Acknowledgments. The Shanghai Jiao Tong University Team is supported by Shanghai Municipal
Science and Technology Major Project (2021SHZDZX0102) and National Natural Science Founda-
tion of China (62076161, 62177033). We would also like to thank Wu Wen Jun Honorary Doctoral
Scholarship from AI Institute, Shanghai Jiao Tong University.

REFERENCES

Jiuhai Chen, Jonas Mueller, Vassilis N. Ioannidis, Soji Adeshina, Yangkun Wang, Tom Goldstein, and
David Wipf. Convergent boosted smoothing for modeling graphdata with tabular node features.
ICLR, 2022. 9

Fabrizio Frasca, Emanuele Rossi, Davide Eynard, Ben Chamberlain, Michael Bronstein, and Federico
Monti. Sign: Scalable inception graph neural networks. arXiv preprint arXiv:2004.11198, 2020.
2, 7

Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural
message passing for quantum chemistry. In Proceedings of the 34th International Conference on
Machine Learning, 2017. 1

Chen Gong, Dacheng Tao, Wei Liu, Liu Liu, and Jie Yang. Label propagation via teaching-to-learn
and learning-to-teach. IEEE Trans. Neural Networks Learn. Syst., 28(6):1452–1465, 2017. 1

William L. Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large

graphs. In Advances in Neural Information Processing Systems 30, 2017. 1, 3

Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta,
and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. In Advances
in Neural Information Processing Systems 33, 2020. 2, 8

Weihua Hu, Matthias Fey, Hongyu Ren, Maho Nakata, Yuxiao Dong, and Jure Leskovec. Ogb-lsc: A
large-scale challenge for machine learning on graphs. arXiv preprint arXiv:2103.09430, 2021. 2

Qian Huang, Horace He, Abhay Singh, Ser-Nam Lim, and Austin R. Benson. Combining label prop-
agation and simple models out-performs graph neural networks. In 9th International Conference
on Learning Representations, 2021. 1, 3, 7, 20

Sergei Ivanov and Liudmila Prokhorenkova. Boost then convolve: Gradient boosting meets graph

neural networks. In 9th International Conference on Learning Representations, 2021. 9

Junteng Jia and Austin R Benson. A unifying generative model for graph learning algorithms: Label
propagation, graph convolutions, and combinations. arXiv preprint arXiv:2101.07730, 2021. 2

Masayuki Karasuyama and Hiroshi Mamitsuka. Manifold-based similarity adaptation for label

propagation. In Advances in Neural Information Processing Systems 26, 2013. 1

Thomas N. Kipf and Max Welling. Semi-supervised classiﬁcation with graph convolutional networks.

In 5th International Conference on Learning Representations, 2017. 1, 3, 7, 19

Johannes Klicpera, Aleksandar Bojchevski, and Stephan G¨unnemann. Predict then propagate:
Graph neural networks meet personalized pagerank. In 7th International Conference on Learning
Representations, 2019. 1, 3

Kezhi Kong, Guohao Li, Mucong Ding, Zuxuan Wu, Chen Zhu, Bernard Ghanem, Gavin Taylor, and
Tom Goldstein. Flag: Adversarial data augmentation for graph neural networks. arXiv preprint
arXiv:2010.09891, 2020. 2, 3

Guohao Li, Matthias M¨uller, Bernard Ghanem, and Vladlen Koltun. Training graph neural networks
with 1000 layers. In Proceedings of the 38th International Conference on Machine Learning, 2021.
2, 3

Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional networks
for semi-supervised learning. In Proceedings of the Thirty-Second AAAI Conference on Artiﬁcial
Intelligence, 2018. 1

10

Published as a conference paper at ICLR 2022

Renjie Liao, Zhizhen Zhao, Raquel Urtasun, and Richard S. Zemel. Lanczosnet: Multi-scale deep
graph convolutional networks. In 7th International Conference on Learning Representations, 2019.
1

Yanbin Liu, Juho Lee, Minseop Park, Saehoon Kim, Eunho Yang, Sung Ju Hwang, and Yi Yang.
Learning to propagate labels: Transductive propagation network for few-shot learning. In 7th
International Conference on Learning Representations, 2019. 1, 2, 3

Michael Sejr Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max
Welling. Modeling relational data with graph convolutional networks. In The Semantic Web - 15th
International Conference, 2018. 7

Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Gallagher, and Tina Eliassi-Rad.

Collective classiﬁcation in network data. AI Mag., 29(3):93–106, 2008. 8

Yunsheng Shi, Zhengjie Huang, Shikun Feng, Hui Zhong, Wenjing Wang, and Yu Sun. Masked label
prediction: Uniﬁed message passing model for semi-supervised classiﬁcation. In Proceedings of
the Thirtieth International Joint Conference on Artiﬁcial Intelligence, 2021. 2, 3

Chuxiong Sun and Guoshi Wu. Adaptive graph diffusion networks with hop-wise attention. arXiv

preprint arXiv:2012.15024, 2020. 2, 3

Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li`o, and Yoshua
Bengio. Graph attention networks. In 6th International Conference on Learning Representations,
2018. 1, 3

Fei Wang and Changshui Zhang. Label propagation through linear neighborhoods. In Proceedings of

the Twenty-Third International Conference on Machine Learning, 2006. 1

Hongwei Wang and Jure Leskovec. Unifying graph convolutional neural networks and label propaga-

tion. arXiv preprint arXiv:2002.06755, 2020. 2, 3, 7

Yangkun Wang, Jiarui Jin, Weinan Zhang, Yong Yu, Zheng Zhang, and David Wipf. Bag of tricks for
node classiﬁcation with graph neural networks. arXiv preprint arXiv:2103.13355, 2021. 2, 3

Felix Wu, Amauri H. Souza Jr., Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian Q. Weinberger.
Simplifying graph convolutional networks. In Proceedings of the 36th International Conference
on Machine Learning, 2019. 2, 4, 7

Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, and Stefanie
Jegelka. Representation learning on graphs with jumping knowledge networks. In Proceedings of
the 35th International Conference on Machine Learning, 2018. 1

Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural

networks? In 7th International Conference on Learning Representations, 2019. 1, 3

Yuto Yamaguchi, Christos Faloutsos, and Hiroyuki Kitagawa. CAMLP: conﬁdence-aware modulated
label propagation. In Proceedings of the 2016 SIAM International Conference on Data Mining, pp.
513–521, 2016. 7

Yongyi Yang, Tang Liu, Yangkun Wang, Jinjing Zhou, Quan Gan, Zhewei Wei, Zheng Zhang,
Zengfeng Huang, and David Wipf. Graph neural networks inspired by classical iterative algorithms.
In Proceedings of the 38th International Conference on Machine Learning, 2021. 2, 3, 7, 19

Muhan Zhang and Yixin Chen. Link prediction based on graph neural networks. In Advances in

Neural Information Processing Systems 31, 2018. 1

Xinhua Zhang and Wee Sun Lee. Hyperparameter learning for graph based semi-supervised learning

algorithms. In Advances in Neural Information Processing Systems 19, 2006. 1

Dengyong Zhou, Olivier Bousquet, Thomas Navin Lal, Jason Weston, and Bernhard Sch¨olkopf.
Learning with local and global consistency. In Advances in Neural Information Processing Systems
16, 2003. 1, 3, 6

Xiaojin Jerry Zhu. Semi-supervised learning literature survey. In University of Wisconsin, Madison,

Technical Report, 2005. 1

11

Published as a conference paper at ICLR 2022

A ADDITIONAL EXPERIMENTS

A.1 EFFECT OF TRAINING ACCURACY

Intuitively, the label trick introduces more information about the ground-truth labels in the training
set to the model. In this experiment, we investigate the role of training accuracy in training with the
label trick. We produce two new datasets by projecting the input features on Cora-full and Pubmed
to 32 dimensions (i.e., Cora-full-32d and Pubmed-32d, respectively) via PCA. This is a much more
challenging task because less feature information is available for prediction. We train SGC, SIGN
and TWIRLS under the same settings as in Table 3. The following tables show the test and training
accuracy on these datasets respectively, as well as results using original features as in Table 3. For
each comparison, we use the same set of model hyperparameters with/without label trick, based on
the best performance without label trick.

Table 5. Test accuracy (%) with/without label trick (D).

Method
label trick (D)

Cora-full-32d
Pubmed-32d
Cora-full
Pubmed
ArXiv

SGC

(cid:55)

(cid:51)

(cid:55)

SIGN

(cid:51)

(cid:55)

TWIRLS

(cid:51)

55.63 ± 0.53
83.15 ± 0.41
65.87 ± 0.61
85.02 ± 0.43
69.07 ± 0.01

62.37 ± 0.87
84.33 ± 0.68
65.81 ± 0.69
85.23 ± 0.57
70.22 ± 0.03

55.10 ± 0.55
84.75 ± 0.37
68.54 ± 0.76
87.94 ± 0.52
69.97 ± 0.16

62.05 ± 0.47
85.72 ± 0.84
68.44 ± 0.88
88.09 ± 0.59
70.98 ± 0.21

58.57 ± 0.48
84.94 ± 0.37
70.36 ± 0.51
89.81 ± 0.56
72.93 ± 0.19

61.74 ± 0.46
87.60 ± 0.44
70.40 ± 0.71
90.08 ± 0.52
73.22 ± 0.10

Table 6. Training accuracy (%) with/without label trick (D). Each comparison shares the model
hyperparameters.

Method
label trick (D)

Cora-full-32d
Pubmed-32d
Cora-full
Pubmed
ArXiv

SGC

(cid:55)

(cid:51)

(cid:55)

SIGN

(cid:51)

(cid:55)

TWIRLS

(cid:51)

57.69 ± 0.21
83.43 ± 0.35
95.31 ± 0.16
85.63 ± 0.17
71.66 ± 0.01

65.75 ± 0.19
84.74 ± 0.17
97.59 ± 0.12
85.69 ± 0.23
74.94 ± 0.01

57.20 ± 0.21
84.96 ± 0.13
99.78 ± 0.03
91.26 ± 0.12
71.26 ± 0.08

65.88 ± 0.27
86.05 ± 0.19
99.99 ± 0.01
91.43 ± 0.17
74.94 ± 0.08

63.44 ± 0.41
85.97 ± 0.22
99.95 ± 0.01
97.32 ± 0.30
80.92 ± 0.17

67.40 ± 0.36
88.43 ± 0.26
99.76 ± 0.05
97.48 ± 0.28
85.08 ± 0.12

From Tables 5 and 6, we can see that the degree to which the label trick can improve accuracy is
directly related to the training accuracy. When the training accuracy is near 100%, the label trick
minimally beneﬁts the model as the label information has already been adequately embedded in the
model. However, when the training accuracy is lower, the label trick can increase both the training
and test accuracy signiﬁcantly (once the variance from the dataset divisions is taken into account,
please see Appendix A.2).

Moreover, for large-scale graphs it may be less likely that high accuracy or overﬁtting is experienced
during training due to the generous number of available samples. So in such cases, the label trick is
more likely to be effective.

A.2 EFFECT OF RANDOM DATASET DIVISION

Due to the random division of the dataset into training, validation and test dataset for Cora-full,
Pubmed, and the regression datasets, the standard deviations across trials are considerably higher
than when using ﬁxed splits. Consequently, for many of the results, the performance gap remains
consistent across trials such that the improvement is actually still signiﬁcant.

As an illustration, Table 7 presents the itemized, trial-to-trial results that were averaged to produce the
Cora-full results presented earlier in Table 1. This reveals that the label trick consistently improves
performance over different dataset splits even after careful tuning of the hyperparameters for the
baseline model. This phenomena is similarly present in many of other experimental results.

12

Published as a conference paper at ICLR 2022

Table 7. Trail-to-trail accuracy results (%) on Cora-full.

Trail

Label Propagation Trainable Label Propagation

1
2
3
4
5
6
7
8
9
10

65.52
65.77
64.84
66.00
68.10
67.39
67.29
66.86
66.48
66.13

67.57
66.03
66.28
66.23
68.58
67.87
68.07
68.83
67.85
66.73

Overall

66.44 ± 0.93

67.40 ± 0.96

A.3 EFFECT OF REGULARIZATION FACTOR WITH SELF-EXCLUDED LINEAR PREDICTORS

In order to demonstrate the effect of the theoretically-derived regularization factor in Theorem 1
and Corollary 1, we apply the regularizer to classiﬁcation experiments involving the deterministic
self-excluded linear predictors from Section 4.2. Figure 2 provides such an example using the Cora
dataset and the TWIRLS linear predictor plus regularization as is varied. Note that if, there is inﬁnite
penalty on the label weights which is equivalent to no labels; in contrast if the regularization factor is
zero.

Figure 2. Accuracy results varying regularization coefﬁcient with corresponding α. When α = 0,
there is no actual label in use.

From Figure 2 we observe that including labels plus the theoretically-motivated regularization (with
the right value for α) can indeed improve performance. This suggests that even with a classiﬁcation
loss (which only loosely aligns with the stated theorem/corollary assumption of an MSE loss), the
theory has some practical/predictive value in producing a useful regularization factor.

B PROOFS

B.1 PROOF OF THEOREM 1

We ﬁrst deﬁne the mask matrix for input labels:

M in =

(cid:18)diag(ri) 0
0
0

(cid:19)

, ri ∼ Bernoulli(α),

(16)

where α ∈ (0, 1) is the label rate, and the mask matrix for output is deﬁned as M out = M tr − M in.
Then the objective is

13

0.00.20.40.60.81.0α82.583.083.584.084.5Accuracy(%)αPublished as a conference paper at ICLR 2022

L(W ) = E

splits

[(cid:107)Y out − M outP ˜Y inW (cid:107)2
F ]
(cid:34)(cid:13)
(cid:13)
(cid:13)
(cid:13)

M outP Y inW

Y out −

1
α

(cid:35)

(cid:13)
2
(cid:13)
(cid:13)
(cid:13)

F

= E

splits

= E

splits

[(cid:107)Y out(cid:107)2

F ] −

2
α

[tr[Y T

outM outP Y inW ]] +

1
α2

E
splits

[(cid:107)M outP Y inW (cid:107)2
F ]

E
splits
2
α

= E

[(cid:107)M outY tr(cid:107)2

F ] −

E
splits

[tr[Y T

trM outP M inY trW ]]

splits
1
α2

+

E
splits

[(cid:107)M outP M inY trW (cid:107)2

F ].

Next we compute each term separately. For notational convenience, we denote the i-th row and j-th
column entry of M in by Mij.

(17)

1. For Esplits[(cid:107)M outY tr(cid:107)2
E
splits

F ], we have
[(cid:107)M outY tr(cid:107)2

F ] = E
splits

[tr[Y T

trM T

outM outY tr]]

= E

[tr[Y T

trM outY tr]]

splits

= E

splits
= tr[ E

splits

[tr[M outY trY T

tr]]

[M out]Y trY T
tr]

= (1 − α) tr[M trY trY T
tr]
= (1 − α) tr[Y trY T
tr]
= (1 − α)(cid:107)Y tr(cid:107)2
F .

2. For Esplits[tr[Y T

trM outP M inY trW ]], since M out = M tr − M in, we then have

E
splits

[tr[Y T

trM outP M inY trW ]]

= E

splits
= tr[ E

splits

[tr[M outP M inY trW Y T

tr]]

[M outP M in]Y trW Y T
tr]

= tr[ E

[(M tr − M in)P M in]Y trW Y T
tr]

splits
= tr[(M trP E

splits

= tr[(αM trP M tr − E

splits

[M in] − E

[M inP M in])Y trW Y T
tr]

splits
[M inP M in])Y trW Y T
tr]

= α tr[M trP M trY trW Y T

tr] − tr

(cid:104)

E
splits

[M inP M in]Y trW Y T
tr

(cid:105)

= α tr[Y T

trP Y trW ] − tr

(cid:104)

E
splits

[M inP M in]Y trW Y T
tr

(cid:105)

Let us calculate Esplits[M inP M in]. Since M in is a diagonal matrix, we have

E
splits

[M inP M in] = ( E

[MiiMjjPij]) = α2P tr + (α − α2)Ctr,

splits

(18)

(19)

(20)

where P tr = M trP M tr and Ctr = diag(P tr). Then we have
tr[ E

[M inP M in]Y trW Y T

tr] = α2 tr[Y T

trP trY trW ] + (α − α2) tr[Y T

trCtrY trW ].

splits

14

(21)

Published as a conference paper at ICLR 2022

Therefore,

E
splits

[tr[Y T

trM outP M inY trW ]]

= (α − α2) tr[Y T
= (α − α2) tr[Y T
3. For Esplits[(cid:107)M outP M inY trW (cid:107)2

trP trY trW ] − (α − α2) tr[Y T
tr(P tr − Ctr)Y trW ].
F ], we have

trCtrY trW ]

E
splits
= E

splits

= E

splits
= tr[ E

splits

[(cid:107)M outP M inY trW (cid:107)2
F ]

[tr[W T Y T

trM T

inP T M T

outM outP M inY trW ]]

[tr[W T Y T

trM T

inP T M outP M inY trW ]]

[W T Y T

trM T

inP T (M tr − M in)P M in]Y trW ]

= tr[ E

[W T Y T

trM T

inP T M trP M in]Y trW ]

splits
− tr[ E

splits
= tr[W T Y T
tr

[W T Y T

trM T

inP T M inP M in]Y trW ]

[M T

inP T M trP M in]Y trW ]

− tr[W T Y T
tr

[M T

inP T M inP M in]Y trW ].

E
splits
E
splits

Let us consider each term separately.
(a) To compute tr[W T Y T
tr
tr) 1

2 , and then

diag(P 2

(cid:104)

E
splits

M T

inP T M trP M in

Esplits[M T

inP T M trP M in]Y trW ], we ﬁrst deﬁne Qtr =

(cid:105)

= E

splits

ij

(cid:34) m
(cid:88)

k=1

(cid:35)

MiiMjjPkiPkj

= E

splits

[MiiMjj]

m
(cid:88)

k=1

PkiPkj

(cid:104)
α2P T

(cid:104)
α2P T

=

=

trP tr + (α − α2) diag(P 2

trP tr + (α − α2)(Q2

(cid:105)
tr)

.

ij

(cid:105)
tr)

ij

Therefore,

tr[W T Y T
tr

[M T

E
splits
= α2(cid:107)P trY trW (cid:107)2
Esplits[M T

inP T M trP M in]Y trW ]

F + (α − α2)(cid:107)QtrY trW (cid:107)2
F .

(b) To compute tr[W T Y T
tr

inP T M inP M in]Y trW ], we have that

(cid:104)
M inP T M inP M in

(cid:105)

E
splits

ij

= E

splits

[MiiMjj

n
(cid:88)

k=1

MkkPkiPkj]

=

n
(cid:88)

k=1

E
splits

[MiiMjjMkk]PkiPkj.

• For the diagonal entries, where i = j, we have

n
(cid:88)

k=1

E
splits

[MiiMjjMkk]PkiPkj =

n
(cid:88)

k=1

α2P 2

tr,ki + (α − α2)P 2

tr,ii,

(cid:17)
[M inP T M inP M in]

= α2 diag(P 2

tr) + (α − α2)C2
tr

and therefore,

diag

(cid:16)

E
splits
= α2Q2

tr + (α − α2)C2
tr.

15

(22)

(23)

(24)

(25)

(26)

(27)

(28)

Published as a conference paper at ICLR 2022

• For the off-diagonal entries, where i (cid:54)= j, we have

[MiiMjjMkk]PkiPkj = α3 (cid:88)

Ptr,kiPtr,kj +α2(Ptr,iiPtr,ij +Ptr,jiPtr,jj),

E
splits

(cid:88)

k

and

k(cid:54)=i∧k(cid:54)=j

E
splits

[M inP T M inP M in]ij
tr + (α2 − α3)(CtrP tr + P trCtr)(cid:3)

= (cid:2)α3P 2

ij .

From (28) and (30), we know that

[M inP T M inP M in]

E
splits
=α3P 2

tr + (α2 − α3)Q2

tr + (α − 3α2 + 2α3)C2

tr + (α2 − α3)(CtrP tr + P trCtr).

(29)

(30)

(31)

Unfortunately because CtrP tr + P trCtr is not necessarily positive semi-deﬁnite, we cannot
simplify the proof using the Cholesky decomposition. So instead we proceed as follows:

trE[M T

tr[W T Y T
= tr[W T Y T

tr(α3P 2

inP trM inP trM in]Y trW ]
tr + (α2 − α3) diag(P 2
+ (α2 − α3) diag((CtrP tr + P trCtr)) + (α − α2)C2
tr diag(P 2

trY trW + (α2 − α3)W T Y T

= tr[α3W T Y T

trP 2

tr)Y trW ]
tr)Y trW

tr) + (α2 − α3)(CtrP tr + P trCtr)

+ (α2 − α3)W T Y T
+ (α − 3α2 + 2α3)W T Y T

tr(CtrP tr + P trCtr)Y trW
trY trW ]

trC2
F + (α2 − α3)(cid:107)QtrY trW (cid:107)2

= α3(cid:107)P trY trW (cid:107)2

F + (α − 3α2 + 2α3)(cid:107)CtrY trW (cid:107)2
F

+ (α2 − α3) tr[W T Y T

tr(CtrP tr + P trCtr)Y trW ].

(32)

Combining the three terms, we get

[(cid:107)M outP M inY trW (cid:107)2
F ]

E
splits
= (α2 − α3)(cid:107)P trY trW (cid:107)2

F + (α − 2α2 + α3)(cid:107)QtrY trW (cid:107)2
F

− (α2 − α3) tr[W T Y tr(P trCtr + CtrP tr)Y trW ] − (α − 3α2 + 2α3)(cid:107)CtrY trW (cid:107)2
F .
(33)

The overall result L(W ) is

E
splits

[(cid:107)Y out − M outP ˜Y inW (cid:107)2
F ]

= E

[(cid:107)M outY tr(cid:107)2

F ] −

2
α

E
splits

[tr[Y T

trM outP M inY trW ]]

E
splits

[(cid:107)M outP M inY trW (cid:107)2
F ]

(34)

F − 2(1 − α) tr[Y T

tr(P tr − Ctr)Y trW ] + (1 − α)(cid:107)P trY trW (cid:107)2
F
F − (1 − α) tr[W T Y tr(P trCtr + CtrP tr)Y trW ]

− 2 + α)(cid:107)QtrY trW (cid:107)2

splits
1
α2

+

+ (

= (1 − α)(cid:107)Y tr(cid:107)2
1
α
1
α

− (

− 3 + 2α)(cid:107)CtrY trW (cid:107)2
F ,

16

Published as a conference paper at ICLR 2022

where Ctr = diag(P tr) and Qtr = diag(P 2

tr) 1
2 .

Next we compute (cid:107)Y tr − M tr(P − C)Y trW (cid:107)2

F . Since

(cid:107)M tr(P − C)Y trW (cid:107)2
= tr[W T Y T
= (cid:107)P trY trW (cid:107)2

F = (cid:107)(P tr − Ctr)Y trW (cid:107)2
F

tr(P tr − Ctr)T (P tr − Ctr)Y trW ]

F + (cid:107)CtrY trW (cid:107)2

F − tr[W T Y tr(P trCtr + CtrP tr)Y trW ],

we have that

(cid:107)Y tr − M tr(P − C)Y trW (cid:107)2
F
= (cid:107)Y tr(cid:107)2

F + (cid:107)P trY trW (cid:107)2

F + (cid:107)CtrY trW (cid:107)2
F

− tr[W T Y tr(P trCtr + CtrP tr)Y trW ] − 2 tr[Y T

tr(P tr − Ctr)Y trW ].

Therefore,

(35)

(36)

[(cid:107)Y out − M outP ˜Y inW (cid:107)2
F ]

F − 2 tr[Y T

tr(P tr − Ctr)Y trW ] + (cid:107)P trY trW (cid:107)2
F

1
E
1 − α
splits
= (cid:107)Y tr(cid:107)2
1 − α
α

+

= (cid:107)Y tr − M tr(P − C)Y trW (cid:107)2

F +

1 − α
α

(cid:107)QtrY trW (cid:107)2

F −

1 − α
α

Let U = (diag(P T P ) − CT C) 1

2 , and Γ = U Y tr. Then we have

2α − 1
α
(cid:107)CtrY trW (cid:107)2
F .

(cid:107)QtrY trW (cid:107)2

F − tr[W T Y tr(P trCtr + CtrP tr)Y trW ] +

(cid:107)CtrY trW (cid:107)2
F

1
1 − α

E
splits

[(cid:107)Y out − M outP ˜Y inW (cid:107)2
F ]

= (cid:107)Y tr − M tr(P − C)Y trW (cid:107)2

F +

= (cid:107)Y tr − M tr(P − C)Y trW (cid:107)2

F +

= (cid:107)Y tr − M tr(P − C)Y trW (cid:107)2

F +

1 − α
α
1 − α
α
1 − α
α

(cid:107)M trU Y trW (cid:107)2
F

(cid:107)U Y trW (cid:107)2
F

(cid:107)ΓW (cid:107)2
F .

B.2 PROOF OF COROLLARY 1

We have that

(37)

(38)

1
E
1 − α
splits
1
1 − α

=

E
splits

(cid:104)
(cid:107)Y out − M outP XW x − M outP ˜Y inW y(cid:107)2
F

(cid:105)

(cid:104)

(cid:107)Y out − M outP ˜Y inW y(cid:107)2
F

+ (cid:107)M outP XW x(cid:107)2

F − 2 tr[Y T

outM outP XW x] +

tr[W T

y Y T

(cid:105)
inP T M outP XW x]

2
α

= (cid:107)Y tr − (P tr − Ctr)Y trW y(cid:107)2

+ (cid:107)M trP XW x(cid:107)2

F − 2 tr[Y T

1 − α
α

(cid:107)ΓW y(cid:107)2
F

F +
trP XW x] + 2 tr[W T

tr(P tr − Ctr)P XW x]

= (cid:107)Y tr − M trP XW x − (P tr − Ctr)Y trW y(cid:107)2

F +

(cid:107)ΓW y(cid:107)2
F

= (cid:107)Y tr − M trP XW x − M tr(P − C)Y trW y(cid:107)2

F +

(cid:107)ΓW y(cid:107)2
F ,

(39)

y Y T
1 − α
α
1 − α
α

which proves the conclusion.

17

Published as a conference paper at ICLR 2022

B.3 PROOF OF THEOREM 2

We begin by noting that

E
splits

[M outP ˜Y inW y] =

=

1
α
1
α

E
splits

[M trP Y inW y] −

1
α

[M inP Y inW y]

M trP E

splits

[M in]Y trW y −

[M inP M in]Y trW y

E
splits
1
α

E
splits

= M trP Y trW y − (αP tr − αCtr + Ctr)Y trW y
= (1 − α)M trP Y trW y − (1 − α)CtrY trW y
= (1 − α)M tr(P − C)Y trW y.

And then by Jensen’s inequality we have

[CrossEntropyDout

E
splits
≥ CrossEntropyDout

(Y out, M outP XW x + M outP ˜Y inW y)]

(Y out, E

[M outP XW x + M outP ˜Y inW y])

splits

= CrossEntropyDtr

(Y tr, (1 − α)(P XW x + (P − C)Y trW y))).

Notice based on the nature of cross-entropy, if 0 < α < 1, we have

CrossEntropy(Z, (1 − α) ˜Z) ≥ (1 − α)CrossEntropy(Z, ˜Z).

Therefore,

1
E
1 − α
splits
≥ CrossEntropyDtr

[CrossEntropyDout

(Y out, P XW x + P ˜Y inW y)]

(Y tr, XW x + (P − C)Y trW y).

(40)

(41)

(42)

(43)

B.4 PROOF OF THEOREM 3

Although the label trick splits are drawn randomly using an iid Bernoulli distribution to sample the
elements to be added to either Din or Dout, we can equivalently decompose this process into two
parts. First, we draw |Dout| from a binomial distribution P [|Dout|] = Binom[m, 1 − α], where m
is the number of trials and 1 − α is the success probability. And then we choose the elements to be
assigned to Dout uniformly over the possible combinations (cid:0) m
|Dout|

(cid:1).

Given that when |Dout| = 0 there is no contribution to the loss, we may therefore reexpress the
lefthand side of (12), excluding the limit, as

1
1 − α

E
splits

(cid:104) (cid:88)

(cid:96)(cid:0)yi, fGN N [X, Y in; W]i

(cid:1)(cid:105)

=

i∈Dout

P [|Dout| = 1]
1 − α

E
P (cid:2)Dout||Dout|=1(cid:3)

(cid:104) (cid:88)

(cid:96)(cid:0)yi, fGN N [X, Y in; W]i

(cid:1)(cid:105)

i∈Dout

(44)

+

P [|Dout| > 1]
1 − α

E
P (cid:2)Dout||Dout|>1(cid:3)

(cid:104) (cid:88)

(cid:96)(cid:0)yi, fGN N [X, Y in; W]i

(cid:1)(cid:105)

,

i∈Dout

where it naturally follows that

E
P (cid:2)Dout||Dout|=1(cid:3)

(cid:104) (cid:88)

i∈Dout

(cid:96)(cid:0)yi, fGN N [X, Y in; W]i

(cid:1)(cid:105)

= 1
m

m
(cid:88)

i=1

(cid:96)(cid:0)yi, fGN N [X, Y tr−Y i; W]i

(cid:1).

We also have that

P [|Dout| > 1]
1 − α

=

(cid:80)m

i=2

(cid:0)m
i

(cid:1)(1 − α)iαm−i
1 − α

=

m
(cid:88)

i=2

(cid:19)

(cid:18)m
i

(1 − α)i−1αm−i

(45)

(46)

18

(47)

(48)

Published as a conference paper at ICLR 2022

and

P [|Dout| = 1]
1 − α

=

m(1 − α)αm−1
1 − α

= mαm−1.

From these expressions, we then have that

lim
α→1

P [|Dout| > 1]
1 − α
(cid:12)(cid:80)

i∈Dout

= 0 and lim
α→1

P [|Dout| = 1]
1 − α
(cid:96)(cid:0)yi, fGN N [X, Y in; W]i

= m.

(cid:1)(cid:12)
(cid:12) < B < ∞ for some B, we

And given the assumption that (cid:12)
also have

E
P (cid:2)Dout||Dout|>1(cid:3)

(cid:104) (cid:88)

(cid:96)(cid:0)yi, fGN N [X, Y in; W]i

(cid:1)(cid:105)

<

i∈Dout

E
P (cid:2)Dout||Dout|>1(cid:3)

(cid:105)

(cid:104)

B

= B,

(49)

where the bound is independent of α. Consequently,

lim
α→1

P [|Dout| > 1]
1 − α

E
P (cid:2)Dout||Dout|>1(cid:3)

(cid:104) (cid:88)

(cid:96)(cid:0)yi, fGN N [X, Y in; W]i

(cid:1)(cid:105)

= 0

(50)

i∈Dout

such that when combined with the other expressions above, the conclusion

(cid:40)

lim
α→1

1
1 − α

E
splits

(cid:104) (cid:88)

i∈Dout

obviously follows.

(cid:96)(cid:0)yi, fGN N [X, Y in; W]i

(cid:41)

(cid:1)(cid:105)

=

m
(cid:88)

i=1

(cid:96)(cid:0)yi, fGN N [X, Y tr−Y i; W]i

(cid:1)

(51)

C EXPERIMENTAL DETAILS

C.1 TRAINABLE LABEL PROPAGATION

In Table 1, we employ trainable label propagation with the same settings as the original label
propagation algorithm, including the trade-off term λ in (1) and the number of propagation steps.
In the implementation, we set λ as 0.6 and use 50 propagation steps. We search the best splitting
probability α in the range of {0.05, 0.1, . . . , 0.95}.

C.2 DETERMINISTIC LABEL TRICK APPLIED TO GNNS WITH LINEAR LAYERS

We use three models with linear propagation layers and simply choose one speciﬁc set of hyperparam-
eters and run the model on each dataset with or without label trick in Table 3. The hyperparameters
for each model is described as follows. For TWIRLS, we use 2 propagation steps for the linear
propagation layers and λ is set to 1 (Yang et al., 2021). For SIGN, we sum up all immediate results
instead of concatenating them to simplify the implementation and speed up training. Our SIGN model
also use 5 propagation steps, and we tune the number of MLP layers from 1 or 2 on each dataset.
And for SGC, the number of propagation steps is set to 3, and there is one extra linear transformation
after the propagation steps.

C.3 EFFECT OF SPLITTING PROBABILITY

To further present the effect of splitting probability α, in Figure 1, we use α in the range of
{0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.9875, 0.99375}
to compare a linear model (similar to a trainable label propagation algorithm with features and
labels as input) with a three-layer GCN (Kipf & Welling, 2017). Each model uses the same set of
hyperparameters, except for the number of epochs, since when α is close to zero or one, the model
requires more epochs to converge. For the linear model, λ is set to 0.9 and the number of propagation
steps is 9. The GCN has 256 hidden channels and an activation function of ReLU.

19

Published as a conference paper at ICLR 2022

C.4 TRAINABLE CORRECT AND SMOOTH

We begin by restating the predictor formulation of trainable Correct and Smooth in (14):

fT C&S( ˜Y ; W) = P s(Y in + (M te + M out)( ˜Y + ˜EinW c))W s

= P s((Y in + (M te + M out) ˜Y )W s + P s(M te + M out) ˜EinW cW s
= P s((Y in + (M te + M out) ˜Y )
(cid:123)(cid:122)
(cid:125)
ˆY s

W s + P s(M te + M out) ˜Ein
(cid:125)

ˆW c,

(cid:123)(cid:122)
ˆY c

(cid:124)

(cid:124)

(52)

Note that the computation of ˜Y s and ˜Y c does not involve any trainable parameters. Therefore, we
can compute them beforehand for each training split.

We now describe our experimental setup used to produce Table 4. We ﬁrst trained an MLP with
exactly the same hyperparameters as in Huang et al. (2021). For each splitting probability α ∈
{0.1, 0.2, · · · , 0.9}, we generate 10 splits and precompute ˜Y s and ˜Y c. Then for each training epoch,
we cycle over the set of ˜Y s and ˜Y c.

20

