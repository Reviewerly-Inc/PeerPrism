Hierarchical Prototype Networks for Continual
Graph Representation Learning

Anonymous Author(s)
Afﬁliation
Address
email

Abstract
Despite signiﬁcant advances in graph representation learning, little attention has
been paid to graph data in which new categories of nodes (e.g., new research
areas in citation networks or new types of products in co-purchasing networks)
and their associated edges are continuously emerging. The key challenge is to
incorporate the feature and topological information of new nodes in a continuous
and effective manner such that performance over existing nodes is uninterrupted. To
this end, we present Hierarchical Prototype Networks (HPNs) which can adaptively
extract different levels of abstract knowledge in the form of prototypes to represent
continually expanded graphs. Speciﬁcally, we ﬁrst leverage a set of Atomic Feature
Extractors (AFEs) to generate basic features which can encode both the elemental
attribute information and the topological structure of the target node. Next, we
develop HPNs by adaptively selecting relevant AFEs and represent each node
with three-levels of prototypes, i.e., atomic-level, node-level, and class-level. In
this way, whenever a new category of nodes is given, only the relevant AFEs
and prototypes at each level will be activated and reﬁned, while others remain
uninterrupted. Finally, we provide the theoretical analysis on memory consumption
bound and the continual learning capability of HPNs. Extensive empirical studies
on eight different public datasets justify that HPNs are memory efﬁcient and can
achieve state-of-the-art performance on different continual graph representation
learning tasks.

Introduction

1
Graph representation learning aims to pursue a meaningful vector representation of each node so as
to facilitate downstream applications such as node classiﬁcation, link prediction, etc. Traditional
methods are developed based on graph statistics [23] or hand-crafted features [3, 16]. Recently,
a great amount of attention has been paid to graph neural networks (GNNs), such as graph con-
volutional network (GCNs) [12], GraphSAGE [10], Graph Attention Networks (GATs) [31], and
their extensions [34, 6, 41, 14, 7, 24, 38]. This is because they can jointly consider the feature and
topological information of each node. Most of these approaches, however, focus on static graphs and
cannot generalize to the case when new categories of nodes are emerging.
In many real world applications, different categories of nodes and their associated edges (in the form
of subgraphs) are often continuously emerging in existing graphs. For instance, in a citation network
[27, 32, 20], papers describing new research areas will gradually appear in the citation graph; in a
co-purchasing network such as Amazon [4], new types of products will continuously be updated to
the graph. Given these facts, how to incorporate the feature and topological information of new nodes
in a continuous and effective manner such that performance over existing nodes is uninterrupted is a
critical problem to investigate.
To address this issue, various types of continual learning approaches can be considered. Existing
continual learning techniques fall into three main categories, i.e., regularization-based methods that
penalize (or reward) their model objectives so as to maintain satisfactory performance on previous
tasks [11, 9, 26], e.g., Learning without Forgetting (LwF) [15] and Elastic Weight Consolidation
(EWC) [13]; memory-replay based methods that constantly feed a model with representative data

Submitted to 35th Conference on Neural Information Processing Systems (NeurIPS 2021). Do not distribute.

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

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

34

35

36

37

38

39

40

41

42

43

44

45

46

47

48

49

50

51

52

53

54

55

56

57

58

59

60

61

62

63

64

65

66

67

68

69

70

71

72

73

74

75

76

77

78

79

80

81

82

83

84

85

86

87

88

89

90

91

92

93

94

95

96

97

or exemplars of previous tasks to prevent them from being forgotten [18, 28, 2, 5, 8], e.g., Gradient
Episodic Memory (GEM) [18]; and parametric isolation based methods that adaptively introduce
new parameters for new tasks and avoid the existing parameters of previous tasks being drastically
changed [25, 36, 35, 33]. Although these approaches exhibited promising performance in mitigating
the problem of catastrophic forgetting in different applications, e.g., image classiﬁcation, action
recognition, and reinforcement learning, they are not suitable for continual graph representation
learning since both the feature information and topological structure of the target node need to be
considered appropriately.
More recently, Zhou et al. [39] proposed to store a set of representative experience nodes in a buffer
and replay them along with new tasks (categories) to prevent forgetting existing tasks (categories).
The buffer, however, only stores node features and ignores the topological information of graphs.
Liu et al. [17] developed topology-aware weight preserving (TWP) that can preserve the topological
information of existing graphs. However, its design hinders the capability of learning topology on
new tasks (categories). Note that continual graph representation learning is essentially different from
dynamic graph works which mainly concern time dependent graphs in which nodes and (or) edges
change over time [37, 21, 40, 19]. Therefore, the methods developed for dynamic graphs cannot be
directly applied to this task.
A desired learning system for continual graph representation learning is to continuously grasp
knowledge from new categories of emerging nodes and capture their topological structures without
interfering with the learned knowledge over existing graphs. To this end, we present a completely
novel framework, i.e., Hierarchical Prototype Networks (HPNs), to continuously extract different
levels of abstract knowledge (in the form of prototypes) from graph data such that new knowledge
will be accommodated while earlier experience can still be well retained. Within this framework,
representation learning is simultaneously conducted to avoid catastrophic forgetting, instead of
considering these two objectives separately. Speciﬁcally, based on the assumption that each node
can be decomposed into basic atomic characteristics belonging to a set of attributes (e.g., gender,
nationality, hobby, etc.) and the relationship between a pair of nodes can be categorized into different
types (e.g., trust or distrust in a social network), we develop the Atomic Feature Extractors (AFEs)
to decompose each node into two sets of atomic embeddings, i.e., atomic node embeddings which
encode the node feature information and atomic structure embeddings which encode its relations to
neighboring nodes within multi-hop. Next, we present Hierarchical Prototype Networks to adaptively
select, compose, and store representative embeddings with three levels of prototypes, i.e., atomic-
level, node-level, and class-level. Given a new node, only the relevant AFEs and prototypes in each
level will be activated and reﬁned, while others are uninterrupted. Eventually, each node can be
represented with a tri-level prototypes which encode its feature as well as structure information from
different abstract levels and can be used for downstream tasks such as node classiﬁcation. Finally, we
provide the theoretical analysis for the memory consumption upper bound of HPNs and its continual
learning capability. To summarize, the main contributions of our work include:

• We present a novel framework, i.e., Hierarchical Prototype Networks (HPNs), to contin-
uously extract different levels of abstract knowledge (in the form of prototypes) from the
graph data such that new knowledge will be accommodated while earlier experience can be
well retained.

• We provide the theoretical analysis for the memory consumption upper bound of HPNs and

its continual learning capability.

• Our experiment results on eight different public datasets demonstrate that the proposed
HPNs not only achieve state-of-the-art performance, exhibiting good continual learning
capability, but also use less parameters (more efﬁcient). For instance, on OGB-Products
dataset that contains more than 2 million nodes and 47 categories of nodes, HPNs achieves
around 80% accuracy with only thousands of parameters.

2 Hierarchical Prototype Networks

In this section, we ﬁrst state the problem we aim to study and the notations. Then we present
Hierarchical Prototype Networks (HPNs) that consist of two core modules, i.e., Atomic Feature
Extractor (AFEs) and Hierarchical Prototype Networks (HPNs), as shown in Figure 1. AFEs serve to
extract a set of atomic features from the given graph, and the HPNs aim to select, compose, and store
the representative features in the form of different levels of prototypes. During the training stage,
each node will only reﬁne the relevant AFEs and prototypes of the model without interfering with the

2

Figure 1: The framework of HPNs. On the left, subgraphs from different tasks come in sequentially. Given a node
v. uj
k denotes the j-th sampled node from k-hop neighbors. In the middle, node v and the sampled neighbors
are fed into the selected AFEs to get atomic embeddings, which are either matched to existing A-prototypes or
used as new A-prototypes. The selected A-prototypes are further matched to a N- and a C-prototype for the
hierarchical representation, which is ﬁnally fed into the classiﬁer to perform node classiﬁcation.

irrelevant parts (i.e., to avoid catastrophic forgetting). In the test stage, the model will activate the
relevant AFEs and prototypes to perform the inference.

2.1 Problem Statement and Notations
We study continual learning on graphs that have new categories of nodes and associated edges (in the
form of subgraphs) emerging in a continuous manner. In the context of continual learning, assuming
we have a sequence of p tasks {T i|i = 1, ..., p}, in which each task T i aims to learn a satisﬁed
representation for a new subgraph Gi consisting of nodes belonging to some new categories. A
desired model should maintain its performance on all previous tasks after being successively trained
on the sequence of p tasks from T 1 to T p.
For simplicity, we omit the subscripts in this section. Full notations will be used in the theoretical
analysis. Each graph G consists of a node set V = {vi|i = 1, ..., N } with N nodes and an edge set
E = {(vi, vj)} denoting the connections of nodes in V. Each node vi can be represented as a feature
vector x(vi) ∈ Rdv that encodes node attributes, e.g., gender, nationality, hobby, etc. The set of l-hop
neighboring nodes of vi is deﬁned as N l(vi), with N 0(vi) = {vi}.
2.2 Atomic Feature Extractors
Based on the assumption that different nodes can be decomposed into basic atomic characteristics
belonging to a set of attributes (e.g., gender, nationality, hobby, etc.) and the relations between a
pair of nodes can also be categorized into different types (e.g., trust or distrust in a social network),
we develop Atomic Feature Extractors (AFEs) to consider two different sets of atomic embeddings,
i.e., atomic node embeddings which encode the node features and atomic structure embeddings
that encode its relations to neighbors within multi-hop. Speciﬁcally, to ensure that each node can
be represented as different combinations of a subset of atomic features, AFEs are designed as
learnable linear transformations AFEnode = {Ai ∈ Rdv×da |i ∈ {1, ..., la}} and AFEstruct = {Rj ∈
Rdv×dr |j ∈ {1, ..., lr}}where Ai and Rj are real matrices to encode atomic node and structure
information, respectively. la and lr denotes the cardinality of AFEnode and AFEstruct, respectively.
Given a node v, a set of atomic node embeddings is obtained by applying AFEnode to the feature
vector x(v):

Enode
A (v) = {xT (v)Ai|Ai ∈ AFEnode}.
(1)
To obtain atomic structure embeddings, the multi-hop neighboring nodes of v have to be considered.
We ﬁrst uniformly sample a ﬁxed number of vertices from 1-hop up to h-hop neighborhood, i.e.,
N l(v). Then these selected nodes are embedded via projection matrices in
Nsub(v) ⊆

(cid:83)

l∈{1,...,h}

AFEstruct to encode different types of interactions with the target node v:

Estruct
A (v) = {xT (u)Ri|Ri ∈ AFEstruct, u ∈ Nsub}.

(2)

3

98

99

100

101

102

103

104

105

106

107

108

109

110

111

112

113

114

115

116

117

118

119

120

121

122

123

124

125

126

127

128

129

Finally, the complete atomic feature set of target node v is:

EA(v) = Enode

A (v) ∪ Estruct

A (v).

(3)

130

131

132

Note that Ai and Ri are designed to generate different types of atomic features. To ensure that, we
impose a divergence loss on AFEs to ensure they are be uncorrelated with each other and thus can
map features to different subspaces:

Ldiv =

(cid:88)

i(cid:54)=j

AT

i Aj +

RT

i Rj.

(cid:88)

i(cid:54)=j

(4)

133

134

135

136

137

138

139

140

141

142

143

144

145

146

147

148

149

150

151

152

153

154

155

156

157

158

159

160

161

2.3 Hierarchical Prototype Networks
With the atomic features extracted based on AFEs, hierarchical prototype networks (HPNs) will select,
compose, and store representative features in the form of different levels of prototypes as shown in
Figure 1. This is mainly achieved by reﬁning existing prototypes and creating new prototypes only
when necessary. Speciﬁcally, HPNs will produce three different levels of prototypes, i.e., atomic-
level prototypes (A-prototypes), node-level prototypes (N-prototypes), and class-level prototypes
(C-prototypes). From atomic-level to class-level, the prototypes denote abstract knowledge of the
graph at different scales which is analog to the feature maps of convolutional neural networks at
different layers.
We ﬁrst introduce how HPNs can reﬁne existing prototypes. For each task that contains certain
categories of nodes, instead of using all atomic embeddings generated by existing AFEs, HPNs only
select a small and ﬁxed number of AFEs from both AFEnode and AFEstruct which are more relevant
to the given task. In this way, only the relevant AFEs are reﬁned while others remain uninterrupted.
Speciﬁcally, as shown in Figure 1, given a node from an incoming subgraph, each AFE is used to
generate an embedding. Those AFEs with embeddings that are closer to existing A-prototypes are
deemed as more conﬁdent ones and chosen. Formally, we ﬁrst obtain Enode
A (v) via Eq.
(1) and Eq. (2), respectively. Then, we calculate the maximum cosine similarity between atomic
embeddings of each AFE (ei) and the A-prototypes as:

A (v) and Estruct

p

(5)

(
i = max

), ei ∈ Eid

SimMAXid

A(v), p ∈ PA,

node = {Ai(cid:48) ∈ Rdv×da |i(cid:48) ∈ {1, ..., la}} and AFEsort
a and top l(cid:48)

eT
i p
(cid:107)ei(cid:107)2(cid:107)p(cid:107)2
where id ∈ {node, struct}, i ranges from 1 to la (or lr), and PA is the atomic prototype set containing
all A-prototypes. After that, we sort the AFEs in a descending order according to SimMAXid
i as
struct = {Rj(cid:48) ∈ Rdv×dr |j(cid:48) ∈ {1, ..., lr}}.
AFEsort
node and AFEselect
Finally, we select the top l(cid:48)
struct,
respectively. l(cid:48)
r are ﬁxed hyperparameters with l(cid:48)
r ≤ lr. The atomic embeddings
generated by these selected AFEs are denoted as Eselect
A (v).
Based on Eselect
A (v), HPNs then starts to distill representative features, which is conducted by reﬁning
existing prototypes and creating new prototypes simultaneously. A matching process is ﬁrst conducted
between the Eselect
A (v) and PA to recognize the atomic features that are compatible with exiting A-
prototypes and those ones to be accommodated with new A-prototypes. Formally, we measure the
cosine similarity between elements in Eselect

r ranked AFEs from these two sets as AFEselect
a ≤ la and l(cid:48)

A (v) and elements in PA as

a and l(cid:48)

SimE→A(v) = {

eT
i p
(cid:107)ei(cid:107)2(cid:107)p(cid:107)2

|ei ∈ Eselect

A (v), p ∈ PA}.

(6)

162

163

The atomic embeddings that are compatible with existing A-prototypes are these ones with cosine
similarity not less than a certain threshold tA to have at least one existing A-prototype, i.e.,

Eold(v) = {ei| ∃p ∈ PA s.t.

eT
i p
(cid:107)ei(cid:107)2(cid:107)p(cid:107)2

(cid:62) tA}.

(7)

Eold(v) collects a set of atomic embeddings satisfying the previous condition and can be used to
reﬁne PA. To this end, a distance loss Ldis is computed to enhance the cosine similarity between
each ei ∈ Eold(v) and its corresponding A-prototype pi ∈ PA, i.e.,

Ldis = −

(cid:88)

eT
i pi
(cid:107)ei(cid:107)2(cid:107)pi(cid:107)2

(8)

ei∈Eold(v)
By minimizing Ldis, not only the existing A-prototypes in PA will get reﬁned, the atomic embeddings
will also be closer to ‘standard’ A-prototypes.

164

165

166

167

168

4

Algorithm 1: Learning Procedure for HPNs.
Input

:Task sequence: {T1, ..., Tp}, HPNs

1 for T ← 1 to p do
2

node and AFEselect
struct.

Get the data of the current task: V, E, X(V) = {x(v)|v ∈ V}.
Select AFEselect
Compute L = HPNs(V,X(V),E).
L = HPNs(V,X(V),E).
Optimize L.

3

4

5

6

169

170

171

172

173

174

175

176

177

178

179

180

181

182

183

184

185

186

187

188

189

190

191

192

193

194

195

196

197

198
199

200

201

202

203

204

205

Output :updated HPNs

A (v)\Eold(v) or Enew(v) = {ei| ∀p ∈ PA,

Next, we discuss how to deal with the atomic embeddings that are not close to any existing prototypes,
i.e., Enew(v) = Eselect
Contrary to Eold(v), atomic embeddings in Enew(v) are regarded as new atomic features of the
corresponding AFEs. In this case, new prototypes should be generated to accommodate them.
Considering that very similar embeddings may exist in Enew(v) and cause HPNs to create redundant
prototypes, we ﬁrst ﬁlter Enew(v) into E(cid:48)

new(v) to keep only the representative ones such that

eT
i p
(cid:107)ei(cid:107)2(cid:107)p(cid:107)2

< tA}.

∀ei, ej ∈ E(cid:48)

new(v),

eT
i ej
(cid:107)ei(cid:107)2(cid:107)ej(cid:107)2

< tA.

(9)

new(v).

Then, E(cid:48)

PA = PA ∪ E(cid:48)

new(v) is included into PA as new A-prototypes, which will be further reﬁned in the future.
(10)
After generating new prototypes, the matching will be conducted to get a new SimE→A(v) in which
each element is not less than tA. Then each element in Eselect
A (v) is assigned a closest A-prototype
according to SimE→A(v), and each node is associated with a set of atomic prototypes A(v).
To map A(v) to high level prototypes so as to obtain hierarchical prototype representations. A(v) is
ﬁrstly mapped to a N-prototype denoting the overall features of v. We assume that N-prototypes lie
in a dn dimensional space and a fully connected layer is applied to transform A(v) into the new space
EN (v) = FCA→N (a1 ⊕ · · · ⊕ al(cid:48)
), ∀ai ∈ A(v), where ⊕ denotes the concatenation operator.
With EN (v), we then ﬁnd a matching N-prototype or establish a new one, which is similar to the
process at atomic level except that the threshold is set as tN , instead of tA. Learning class-level
prototypes from node-level prototypes is same except that we set the matching threshold as tC.
Finally, the hierarchical prototype representations of the target node is contained in the following set
PH (v) = A(v) ∪ N(v) ∪ C(v).
(11)
Note that A(v) contains multiple A-prototypes denoting atomic features of v from different aspects.
N(v) and C(v) only contain one N-prototype and one C-prototype, representing the overall character-
istics of v and the common characteristics shared by the community containing v, respectively.

a+l(cid:48)
r

2.4 Learning Objective
The obtained hierarchical prototypes for each node are ﬁrst concatenated into a uniﬁed vector and
then pass through a fully connected layer FC to obtain a c (the number of classes) dimensional
r+2), ∀hi ∈ PH (v). In this paper, we aim to perform node
feature vector, i.e., FC(h1 ⊕ · · · ⊕ hl(cid:48)
classiﬁcation. Therefore, based on the c dimensional feature vector and the softmax function σ(·),
r+2))i where i is the index of class. To
we can estimate the label with ˆyi = σ(FC(h1 ⊕ · · · ⊕ hl(cid:48)
perform node classiﬁcation, with the output predictions ˆyi and the target label yi ∈ {1, 2, ..., c}, the
corresponding classiﬁcation loss is given by

a+l(cid:48)

a+l(cid:48)

c
(cid:88)

Lcls =

−yi log( ˆyi),

(12)

i=1

which is essentially the cross entropy loss function. Note that besides node classiﬁcation, PH (v) may
also be used for other tasks based on different objective functions. In this paper, we focus on node
classiﬁcation and the overall loss of HPNs is:

L = Ldis + Ldiv + Lcls.
During the training stage, subgraphs with different tasks (containing different categories of nodes) are
continuously fed to HPNs. Note that unlike topology-aware weight preserving (TWP) method [17],
HPNs do not require task indicator for training and test, and therefore is more practical for real-world
continual graph representation learning applications.

(13)

5

206

207

208

209

210

211

212

213

214

2.5 Theoretical Analysis
In this subsection, we provide the theoretical upper bound for the memory consumption and analyze
how the model conﬁguration would affect HPNs’ capacity in dealing with different tasks. Both
theoretical results are justiﬁed and analyzed in the experiments. Only the main results are provided
here, while the detailed proof and analysis are given in Appendix.
We ﬁrst show that the numbers of different prototypes are upper bounded by the number of atomic
feature extractors and the dimension of the prototypes. Speciﬁcally, we have:
Theorem 1 (Upper bounds for numbers of prototypes). Given the notations deﬁned in HPNs, the
upper bound for the number of A-prototypes na can be given by

nA (cid:54) (la + lr) max

N

S(da, N, 1 − tA),

215

and the upper bounds for the number of N-prototypes and the C-prototypes are:

nN (cid:54) max

N

S(dn, N, 1 − tN )

and

nC (cid:54) max

N

S(dc, N, 1 − tC)

(14)

(15)

216

217

218

219

220

221

222

223

224

225

226

227

228

229

230

231

232

233

234

235

236

237

238

239

240

241

242

243

244

245

246

247

248

249

250

251

252

253

254

255

where S(n, N, t) is the spherical code deﬁned on a n dimensional hypersphere (details in Appendix).
Theorem 1 provides an upper bound for the memory consumption of HPNs. In our experiments, we
show that the number of parameters for most baseline methods are even higher than this upper bound.
Besides memory consumption, the more important problem for a continual learning model is the
capability to maintain memory on previously learned tasks. Based on our model design, we formulate
this as: whether learning new tasks affect the representations the model generates for old task data.
We give explicit deﬁnitions on tasks and task distances based on set theory (in Appendix), then
construct a bound to indicate what conﬁguration would the model have to ensure this capability.
Theorem 2 (Task distance preserving). For HPNs trained on consecutive tasks T p and T p+1.
If lada + lrdr (cid:62) (lr + 1)dv and W is column full rank, then as long as tA < λmin(lr +
1)dist(Vp, Vp+1), learning on T p+1 will not modify representations HPNs generate for data from
T p , i.e. catastrophic forgetting is avoided.
In Theorem 2, λi is eigenvalues of the WT W, where W is a matrix constructed via AFEs (details in
Appendix). dv, da and dr are dimensions of data and two kinds of atomic embeddings. The bound in
this theorem is not tight, as the tight bound would be dependant on the speciﬁc dataset properties.
But this informs us that either the number of AFEs or the dimension of the prototypes has to be large
enough to ensure that data from two tasks can be well separated in the representation space.
According to Theorem 1, the upper bound of the memory consumption is dependent on S(da, N, tA),
S(dn, N, tN ), and S(dc, N, tC). As S(n, N, t) grows fast with n, we prefer larger number of AFEs
with smaller prototype dimensions. We also empirically demonstrate this in Section 3.6. Besides, the
upper bound proposed in Theorem 1 is explicitly computed and compared to experimental results.For
both theorems, proofs and detailed explanations are included in Appendix.
3 Experiments
In the experiments, we answer the following six questions: (1) Whether HPNs can outperform
state-of-the-art approaches? (2) How does each component of HPNs contribute to its performance?
(3) Whether HPNs can memorize previous tasks after learning each new task? (4) Are HPNs sensitive
to the hyperparameters? (5) Whether the theoretical results can be empirically veriﬁed? (6) Whether
the learned prototypes can be interpreted via visualization?
3.1 Datasets
To assess the effectiveness of the proposed HPNs, we consider 8 datasets which include 3 citation
networks (Cora [27], Citeseer[27], OGB-Arxiv [32, 20]), 3 web page networks (Wisconsin, Cornell,
Texas) [22], 1 actor co-occurence network (Actor) [22], and 1 product co-purchasing networks
(OGB-Products [4]). Detailed statistics about these datasets are provided in the Appendix.
Among these datasets, the results of 4 datasets, i.e., Cora, Citeseer, OGB-Arxiv (169,343 nodes,
1,166,243 edges), and OGB-Products (2,449,029 nodes, 61,859,140 edges), are reported in the paper
and the results of other 4 datasets are available in the Appendix.
3.2 Experimental Setup and Evaluation Metrics
To perform continual graph representation learning with new categories of nodes continuously
emerging, we adopt a class-incremental scheme for all datasets. Each new task brings a subgraph with
new categories of nodes and associated edges, e.g., task 1 contains classes 1 and 2, task 2 contains

6

Table 1: Performance comparisons between HPNs and baselines on 4 different datasets.

C.L.T.

Base

Cora

Citeseer

OGB-Arxiv

OGB-Products

AM/%

FM/%

AM/%

FM /%

AM/%

FM /%

AM/%

FM /%

None

EWC
[13]

LwF
[15]

GEM
[18]

MAS
[1]

ERGN.
[39]

TWP
[17]

Join.

GCN
GAT
GIN

GCN
GAT
GIN

GCN
GAT
GIN

GCN
GAT
GIN

GCN
GAT
GIN

GCN
GAT
GIN

GCN
GAT
GIN

GCN
GAT
GIN

63.5±1.9
71.9±3.8
68.3±2.3

63.1±1.2
72.2±1.5
69.6±2.6

76.1±1.4
70.8±2.8
74.1±2.7

75.7±3.0
69.8±3.0
80.2±3.3

65.5±1.9
84.7±0.7
76.7±2.6

63.5±2.4
71.1±2.5
68.3±0.4

68.9±0.9
81.3±3.2
73.7±3.2

93.7 ± 0.5
93.9 ± 0.9
93.2 ± 1.2

-42.3±0.4
-33.1±2.3
-35.4±3.4

-42.7±1.6
-32.2±1.6
-28.5±2.8

-21.3±2.4
-34.6±4.1
-23.3±0.8

-6.5±4.4
-26.1±2.6
-2.0±4.2

-21.4±3.7
-5.6±2.0
-4.0±3.6

-42.3±0.7
-34.3±1.0
-35.4±0.4

-5.7±1.5
-14.4±1.5
-3.9 ±2.6

0.0±0.0
0.0±0.0
0.0±0.0

64.5±3.9
66.8±0.9
57.7±2.3

54.4±4.2
65.7±2.5
57.9±3.4

67.0±0.2
66.1±4.1
63.1±1.9

41.8±2.6
71.3±2.2
49.7±0.5

59.5±3.1
69.1±1.1
65.2±3.9

54.2±3.9
65.5±0.3
57.7±3.1

60.5±3.8
69.8±1.5
68.9±0.7

78.9 ± 0.4
79.3 ± 0.8
78.7 ± 0.9

-7.7±1.6
-19.6±0.3
-36.4±0.3

-30.3±0.9
-19.7±2.3
-36.3±2.4

-8.3±2.7
-18.9±1.5
-16.5±2.2

-31.9±1.4
+9.0±1.5
-24.5±0.9

-0.1±2.4
-4.8±3.3
+0.0±2.0

-30.3±1.9
-20.4±3.9
-36.4±1.3

-0.3±4.4
-8.9±2.6
-2.4±1.9

0.0±0.0
0.0±0.0
0.0±0.0

56.8±4.3
54.3±3.5
53.2± 6.5

72.1±2.4
73.2 ±1.1
74.1 ±1.7

69.9 ± 3.9
68.9±4.4
71.4 ±4.8

75.4±1.7
76.6 ±0.7
77.3 ±2.1

69.8 ±0.4
70.6 ±1.3
65.3 ±2.9

63.3±1.7
63.5±2.4
69.2± 1.8

75.6±0.3
75.8±0.5
76.6±1.8

77.2±0.8
81.8±0.3
82.3±1.9

-19.8±3.2
-21.76± 4.6
-23.59 ±8.1

-9.1±1.9
-10.8 ±2.1
-8.3 ±2.0

-12.1±2.8
-13.6±3.3
-15.9±5.6

-13.6±0.5
-11.3±0.4
-11.2±1.6

-18.8±0.9
-16.7 ±1.6
-17.0±2.3

-18.1±0.9
-19.5±1.9
-11.8±1.4

-10.4±0.5
-5.9±0.3
-11.3±1.1

0.0±0.0
0.0±0.0
0.0±0.0

45.2±5.6
44.9±6.9
43.1±7.4

66.7±0.5
67.9±1.0
67.3±2.3

66.3±2.5
65.1±4.1
65.9±4.0

71.3±1.7
70.4±0.8
76.5±3.3

62.0±1.1
64.4±2.3
61.4±3.8

60.7±2.8
61.3±1.7
61.8±4.7

69.9±0.4
69.3±2.3
69.9±1.4

72.9±1.2
73.7±2.4
77.9±2.1

-27.8±7.1
-30.3±5.2
-31.4±8.8

-8.4±0.4
-9.65±1.3
-13.6±1.5

-11.8±3.4
-13.2±2.9
-10.7±3.1

-10.5±0.9
-10.9±1.6
-7.2±2.5

-17.9±1.9
-14.5±3.2
-20.9±2.9

-26.6±3.3
-25.1±0.8
-23.4±7.9

-9.0±1.1
-8.9±1.5
-10.3±2.7

0.0±0.0
0.0±0.0
0.0±0.0

HPNs

93.7±1.5

+0.6±1.0

79.0±0.9

-0.6±0.7

85.8± 0.7

+0.6±0.9

80.1±0.8

+2.9±1.0

Figure 2: (a) and (b) are AM and FM of HPNs with different number of AFEs and prototype dimensions on
OGB-Arxiv. (c) and (d) are AM and FM change with when tA varies on Cora.
new classes 3 and 4, etc. Each model is trained on a sequence of tasks, and the performance will be
evaluated on all previous tasks. Speciﬁcally, we adopt accuracy mean (AM) and forgetting mean (FM)
as metrics for evaluation. After learning on all tasks, the AM and FM are computed as the average
accuracy and the average accuracy decrease on all previous tasks. Negative FM indicates the existence
of forgetting , zero FM denotes no forgetting and positive FM denotes positive knowledge transfer
between tasks. For HPNs, we set da = dn = dc = 16, la = lr = 22, and h = 2. The threshold tA,
tN , and tC are selected by cross validation on {0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4}. The
experiments on the important hyperparameters are provided in Section 3.6. All experiments are run
on an Nvidia Titan Xp GPU. Full implementation details are in Appendix, and the code is available
in supplementary materials.
3.3 Comparisons with Baseline Methods
We compare HPNs with various baseline methods. Experience Replay based GNN (ERGNN) [39]
and Topology-aware Weight Preserving (TWP) [17] are developed for continual graph representation
learning. The others approaches, including Elastic Weight Consolidation (EWC) [13], Learning with-
out Forgetting (LwF) [15], Gradient Episodic Memory (GEM) [18], and Memory Aware Synapses
(MAS) [1]) are popular continual learning methods for Euclidean data. All the baselines are imple-

256

257

258

259

260

261

262

263

264

265

266

267

268

269

270

271

Table 2: Ablation study on prototypes of different
levels of prototypes over Cora.

Table 3: Ablation study on different loss terms over
Cora.

Conf.

A-p.

N-p.

C-p.

AM%

FM%

Conf.

Lcls

Ldiv

Ldis

AM%

FM%

1

2

3

(cid:88)

(cid:88)

(cid:88)

(cid:88)

(cid:88)

89.2±1.3

-0.1±0.5

91.7±1.1

-0.2±0.8

(cid:88)

93.7±1.5

+0.6±1.0

(cid:88)

(cid:88)

(cid:88)

(cid:88)

(cid:88)

(cid:88)

(cid:88)

(cid:88)

92.4±1.3

+0.8±0.7

92.9±1.1

+0.3±1.0

92.8±0.9

+0.0±1.2

93.7±1.5

+0.6±1.0

1

2

3

4

7

Figure 3: Left: dynamics of ARS for continual learning tasks on OGB-Arxiv. Middle: impact of tA on the
number of prototypes in HPNs over Cora. Right: dynamics of memory consumption of HPNs on OGB-Products.

Table 4: Final parameter amount for models trained on OGB-Products

None

EWC

LwF

GEM

MAS

ERGNN

TWP

Joint

HPNs

GCN
GAT
GIN

2,336
20,032
2,352

46,720
400,640
47,040

4,672
40,064
4,704

2,202,336
2,220,032
2,202,352

2,336
20,032
2,352

6,738
24,432
6,752

9,344
80,128
9,408

2,336
20,032
2,352

4,908

272

273

274

275

276

277

278

279

280

281

282

283

284

285

286

287

288

289

290

291

292

293

294

295

296

297

298

299

300

301

302

303

304

305

306

307

308

309

310

311

312

mented based on three popular backbone models, i.e., Graph Convolutional Networks (GCNs) [12],
Graph Attentional Networks (GATs) [31], and Graph Isomorphism Network (GIN) [34].

Note that Joint training (Join.) in Table 1 does not represent continual learning. It allows a model to
access data of all tasks at any time and thus is often used as an upper bound for continual learning.[29].
In Table 1, we observe that regularization based approaches, e.g., EWC and TWP, generally obtain
lower forgetting, but the accuracy (AM) is limited by the constraints. However, the forgetting
problem of regularization based methods will become increasingly severe when the number of
tasks is relatively large, as shown in Section 3.5. Memory replay based methods such as GEM
achieve better performance without using any constraint. However, the memory consumption is
higher (Section 3.7). HPNs signiﬁcantly outperform all baselines without inheriting their limitations.
Compared to regularization based methods, HPNs do not impose constraints to limit the model’s
expressiveness, therefore the performance is much better. Compared to memory replay based methods,
HPNs do not only perform better but also are memory efﬁcient as shown in Section 3.7. Joint training
(Join.) achieves comparable performance to HPNs on small datasets but is signiﬁcantly worse on large
OGB datasets. This is because joint training (Join.) is a multi-task setting, inter-task interference
may cause negative transfer, which is not obvious on small datasets with only a few tasks but
becomes prominent on large datasets with tens of tasks. In HPNs, different tasks can choose different
combinations of the parameters and thus task interference is dramatically alleviated.

3.4 Ablation Study
We conduct ablation studies on different levels of prototypes and different combinations of three loss
terms. In Table 2, we show the performance of HPNs when A-, N-, and C-Prototypes are gradually
added (Cora dataset). We notice both AM and FM of HPNs increase when higher level prototypes
are considered. This suggests that high level prototypes can enhance the model’s performance and
robustness against forgetting.The effect of different combinations of loss terms are shown in Table 3.
The ﬁrst three rows show that adding Ldiv or Ldis with Lcls may slightly improve the performance.
By jointly considering these three terms, the performance (AM) can be further improved. This is
because Ldiv pushes different AFEs away from each other and Ldis makes the prototypes of each
AFE be more close to its output. Jointly considering Ldiv and Ldis with Lcls can make the prototype
space better separated as shown in Section 3.8.

3.5 Learning Dynamics
For continual learning, it is important to memorize previous tasks after learning each new task. To
measure this, instead of directly measuring the average accuracy on previous tasks which may mix up
the accuracy change caused by forgetting and task differences, we develop a new metric, i.e., average
retaining score (ARS), to address this problem. Speciﬁcally, after learning on a task T i, the ratio
between the model’s accuracy on a previous task T i−m and its accuracy on T i−m after it had been
just learned on T i−m is deﬁned as the retaining ratio. Then the ARS is the average retaining ratio of
all previous tasks after learning a new task.
Figure 3(left) shows the ARS change of HPNs and two baselines. GAT represents the models without
continual learning techniques. TWP+GAT is the best baseline in terms of forgetting. GAT forgets
quickly, while TWP signiﬁcantly alleviates the forgetting problem for GAT. But as more tasks come
in, the forgetting of TWP+GAT increases. As different tasks require different parameters, TWP+GAT

8

Figure 4: Visualization of hierarchical prototype representations of nodes in the test set of Cora.
(regularization based) is seeking a trade off between old and new tasks. With more new tasks,
TWP+GAT tends to gradually adapt to new tasks and forget old ones. On contrary, HPNs maintain
the ARS very well. This is because HPNs learn prototypes to denote the common basic features and
learning new tasks does not hurt the parameters for old tasks. New tasks can be handled with new
combinations of the existing basic prototypes. If necessary, new prototypes can be established for
more expressiveness.

3.6 Parameter Sensitivity
As discussed in Section 2.5, the number of AFEs and the prototype dimensions are key factors
in determining the continual learning capability and memory consumption. Here, we conduct
experiments with different number of AFEs and prototype dimensions to justify the theoretical
results. We keep the dimensions of different prototypes equal and the number of two types of AFEs
equal for simplicity.
As shown in Figure 2(a) and (b), larger dimensions and the number of AFEs yield better AM and
FM, which is consistent with Theorem 2. Besides, AM is mostly determined by the number of AFEs
since HPNs compose prototypes with different AFEs to represent each target node. The number of
possible combinations determines its expressiveness. Considering the above results and the bound
(Theorem 1) for the number of prototypes, using large number of AFEs and small dimension can
ensure both high performance and low memory usage, as veriﬁed in Section 3.7.
We also evaluate the effectiveness of HPNs when prototype thresholds vary from 0.01 to 0.4. Here,
we set tA = tN = tC for simplicity. In Figure 2(c) and (d), we observe that the performance
(AM and FM) of HPNs are generally stable when tA varies and slightly better when tA is between
0.2 and 0.3. This is because when tA is too small or too large, we will have too many or too less
prototypes (consistent with Theorem 1) as shown in Figure 3(middle), which may cause the problem
of overﬁtting or underﬁting.

3.7 Memory Consumption
We compare memory consumption of different methods, as well as a explicitly theoretical memory
upper bound, with the baselines on OGB-Products (the largest dataset). We also show the actual
memory consumption of HPNs in the process of continual learning.
In Table 4, even on the dataset with millions of nodes and 23 tasks, HPNs can accommodate all tasks
with a small amount of parameters. Besides, the dynamic change of parameter amount is shown in
Figure 3(right). The red dashed line denotes the theoretical upper bound (6,163), and the computation
details are included in Appendix. In Figure 3(right), we notice the actual memory usage of HPNs is
much lower than the upper bound. Moreover, even the upper bound is among the lowest for memory
consumption compared to baselines. The model we use here is the same as the one in Section 3.3

3.8 Visualization
To show that HPNs can generate interpretable prototype representations, we apply t-SNE [30] to
visualize the node representations of the Cora dataset (test set) after learning each task. As shown in
Figure 4, each task contains two classes corresponding to (red, blue), (green, salmon), and (purple,
orange), as new tasks come in gradually, the representations are consistently well separated, which
will be beneﬁcial for downstream tasks.

4 Conclusion
In this paper, we proposed Hierarchical Prototype Networks (HPNs), to continuously extract different
levels of abstract knowledge (in the form of prototypes) from streams of tasks on graph representation
learning. The performance of HPNs is both theoretically and experimentally justiﬁed. In the future,
we will apply HPNs to more application scenarios like link prediction, multi-label classiﬁcation,
anomaly detection, etc.

313

314

315

316

317

318

319

320

321

322

323

324

325

326

327

328

329

330

331

332

333

334

335

336

337

338

339

340

341

342

343

344

345

346

347

348

349

350

351

352

353

354

355

356

357

358

9

359

360

361

362

363

364

365

366

367

368

369

370

371

372

373

374

375

376

377

378

379

380

381

382

383

384

385

386

387

388

389

390

391

392

393

394

395

396

397

398

399

400

401

402

403

404

References

[1] Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuyte-
laars. Memory aware synapses: Learning what (not) to forget. In Proceedings of the European
Conference on Computer Vision (ECCV), pages 139–154, 2018.

[2] Rahaf Aljundi, Min Lin, Baptiste Goujaud, and Yoshua Bengio. Gradient based sample selection
for online continual learning. In Advances in Neural Information Processing Systems, pages
11816–11825, 2019.

[3] Smriti Bhagat, Graham Cormode, and S Muthukrishnan. Node classiﬁcation in social networks.

In Social network data analytics, pages 115–148. Springer, 2011.

[4] K. Bhatia, K. Dahiya, H. Jain, P. Kar, A. Mittal, Y. Prabhu, and M. Varma. The extreme

classiﬁcation repository: Multi-label datasets and code, 2016.

[5] Lucas Caccia, Eugene Belilovsky, Massimo Caccia, and Joelle Pineau. Online learned continual
compression with adaptive quantization modules. In International Conference on Machine
Learning, pages 1240–1250. PMLR, 2020.

[6] Jie Chen, Tengfei Ma, and Cao Xiao. Fastgcn: fast learning with graph convolutional networks

via importance sampling. arXiv preprint arXiv:1801.10247, 2018.

[7] Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, and Yaliang Li. Simple and deep graph
convolutional networks. In International Conference on Machine Learning, pages 1725–1735.
PMLR, 2020.

[8] Aristotelis Chrysakis and Marie-Francine Moens. Online continual learning from imbalanced
data. In International Conference on Machine Learning, pages 1952–1961. PMLR, 2020.

[9] Mehrdad Farajtabar, Navid Azizan, Alex Mott, and Ang Li. Orthogonal gradient descent for
continual learning. In International Conference on Artiﬁcial Intelligence and Statistics, pages
3762–3773. PMLR, 2020.

[10] Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large
graphs. In Advances in neural information processing systems, pages 1024–1034, 2017.

[11] Heechul Jung, Jeongwoo Ju, Minju Jung, and Junmo Kim. Less-forgetting learning in deep

neural networks. arXiv preprint arXiv:1607.00122, 2016.

[12] Thomas N Kipf and Max Welling. Semi-supervised classiﬁcation with graph convolutional

networks. arXiv preprint arXiv:1609.02907, 2016.

[13] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins,
Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al.
Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of
sciences, 114(13):3521–3526, 2017.

[14] Guohao Li, Matthias Muller, Ali Thabet, and Bernard Ghanem. Deepgcns: Can gcns go as
deep as cnns? In Proceedings of the IEEE International Conference on Computer Vision, pages
9267–9276, 2019.

[15] Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE transactions on pattern

analysis and machine intelligence, 40(12):2935–2947, 2017.

[16] David Liben-Nowell and Jon Kleinberg. The link-prediction problem for social networks.
Journal of the American society for information science and technology, 58(7):1019–1031,
2007.

[17] Huihui Liu, Yiding Yang, and Xinchao Wang. Overcoming catastrophic forgetting in graph

neural networks. arXiv preprint arXiv:2012.06002, 2020.

[18] David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning.

In Advances in neural information processing systems, pages 6467–6476, 2017.

10

405

406

407

408

409

410

411

412

413

414

415

416

417

418

419

420

421

422

423

424

425

426

427

428

429

430

431

432

433

434

435

436

437

438

439

440

441

442

443

444

445

446

447

448

449

450

[19] Yao Ma, Ziyi Guo, Zhaocun Ren, Jiliang Tang, and Dawei Yin. Streaming graph neural
networks. In Proceedings of the 43rd International ACM SIGIR Conference on Research and
Development in Information Retrieval, pages 719–728, 2020.

[20] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed
representations of words and phrases and their compositionality. arXiv preprint arXiv:1310.4546,
2013.

[21] Giang Hoang Nguyen, John Boaz Lee, Ryan A Rossi, Nesreen K Ahmed, Eunyee Koh, and
Sungchul Kim. Continuous-time dynamic network embeddings. In Companion Proceedings of
the The Web Conference 2018, pages 969–976, 2018.

[22] Hongbin Pei, Bingzhe Wei, Kevin Chen-Chuan Chang, Yu Lei, and Bo Yang. Geom-gcn:

Geometric graph convolutional networks. arXiv preprint arXiv:2002.05287, 2020.

[23] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. Deepwalk: Online learning of social repre-
sentations. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge
discovery and data mining, pages 701–710, 2014.

[24] Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou Huang. Dropedge: Towards deep
graph convolutional networks on node classiﬁcation. In International Conference on Learning
Representations, 2019.

[25] Andrei A Rusu, Neil C Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick,
Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. arXiv
preprint arXiv:1606.04671, 2016.

[26] Gobinda Saha and Kaushik Roy. Gradient projection memory for continual learning.

In

International Conference on Learning Representation, 2021.

[27] Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-

Rad. Collective classiﬁcation in network data. AI magazine, 29(3):93–93, 2008.

[28] Hanul Shin, Jung Kwon Lee, Jaehong Kim, and Jiwon Kim. Continual learning with deep
generative replay. In Advances in neural information processing systems, pages 2990–2999,
2017.

[29] Gido M Van de Ven and Andreas S Tolias. Three scenarios for continual learning. arXiv preprint

arXiv:1904.07734, 2019.

[30] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine

learning research, 9(11), 2008.

[31] Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua

Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017.

[32] Kuansan Wang, Zhihong Shen, Chiyuan Huang, Chieh-Han Wu, Yuxiao Dong, and Anshul
Kanakia. Microsoft academic graph: When experts are not enough. Quantitative Science
Studies, 1(1):396–413, 2020.

[33] Mitchell Wortsman, Vivek Ramanujan, Rosanne Liu, Aniruddha Kembhavi, Mohammad
Rastegari, Jason Yosinski, and Ali Farhadi. Supermasks in superposition. arXiv preprint
arXiv:2006.14769, 2020.

[34] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural

networks? arXiv preprint arXiv:1810.00826, 2018.

[35] Jaehong Yoon, Saehoon Kim, Eunho Yang, and Sung Ju Hwang. Scalable and order-robust
continual learning with additive parameter decomposition. In International Conference on
Learning Representation, 2020.

[36] Jaehong Yoon, Eunho Yang, Jeongtae Lee, and Sung Ju Hwang. Lifelong learning with

dynamically expandable networks. arXiv preprint arXiv:1708.01547, 2017.

11

451

452

453

454

455

456

457

458

459

460

461

462

463

464

465

466

467

468

469

470

471

472

473

474

475

476

477

478

479

480

481

482

483

484

485

486

487

488

489

490

491

492

493

494

495

496

497

498

[37] Wenchao Yu, Wei Cheng, Charu C Aggarwal, Kai Zhang, Haifeng Chen, and Wei Wang.
Netwalk: A ﬂexible deep embedding approach for anomaly detection in dynamic networks. In
Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery &
Data Mining, pages 2672–2681, 2018.

[38] Xikun Zhang, Chang Xu, and Dacheng Tao. On dropping clusters to regularize graph convolu-

tional neural networks. 2020.

[39] Fan Zhou, Chengtai Cao, Ting Zhong, Kunpeng Zhang, Goce Trajcevski, and Ji Geng. Continual

graph learning. arXiv preprint arXiv:2003.09908, 2020.

[40] Lekui Zhou, Yang Yang, Xiang Ren, Fei Wu, and Yueting Zhuang. Dynamic network embedding
by modeling triadic closure process. In Proceedings of the AAAI Conference on Artiﬁcial
Intelligence, volume 32, 2018.

[41] Difan Zou, Ziniu Hu, Yewen Wang, Song Jiang, Yizhou Sun, and Quanquan Gu. Layer-
dependent importance sampling for training deep and large graph convolutional networks. In
Advances in Neural Information Processing Systems, pages 11247–11256, 2019.

Checklist

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reﬂect the paper’s

contributions and scope? [Yes]

(b) Did you describe the limitations of your work? [Yes] In Conclusion, and in the

theoretical part of Appendix.

(c) Did you discuss any potential negative societal impacts of your work? [No] Our work
solves the continual graph representation learning problem. As far as we know, there is
no potential negative societal impacts of our work.

(d) Have you read the ethics review guidelines and ensured that your paper conforms to

them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [Yes] Details are

included in Appendix.

(b) Did you include complete proofs of all theoretical results? [Yes] Proofs are in Appendix

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main ex-
perimental results (either in the supplemental material or as a URL)? [Yes] Code is
included in the supplementary materials.

(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they

were chosen)? [Yes] Details are included in Appendix.

(c) Did you report error bars (e.g., with respect to the random seed after running experi-

ments multiple times)? [Yes] In all tables and in Figure 2

(d) Did you include the total amount of compute and the type of resources used (e.g., type
of GPUs, internal cluster, or cloud provider)? [Yes] Relevant details are included in
Appendix

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [Yes]
(b) Did you mention the license of the assets? [Yes] We mentioned this in the dataset

detail part in Appendix.

(c) Did you include any new assets either in the supplemental material or as a URL? [Yes]

The code of our model is included in the supplementary materials.

(d) Did you discuss whether and how consent was obtained from people whose data you’re
using/curating? [Yes] We mentioned this in the dataset detail part in Appendix.

12

499

500

501

502

503

504

505

506

507

508

(e) Did you discuss whether the data you are using/curating contains personally identiﬁable
information or offensive content? [Yes] We mentioned this in the dataset detail part in
Appendix.

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if

applicable? [N/A]

(b) Did you describe any potential participant risks, with links to Institutional Review

Board (IRB) approvals, if applicable? [N/A]

(c) Did you include the estimated hourly wage paid to participants and the total amount

spent on participant compensation? [N/A]

13

