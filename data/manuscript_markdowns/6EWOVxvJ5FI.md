Stateless actor-critic for instance segmentation with
high-level priors

Anonymous Author(s)
Afﬁliation
Address
email

Abstract

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

Instance segmentation is an important computer vision problem which remains
challenging despite impressive recent advances due to deep learning-based meth-
ods. Given sufﬁcient training data, fully supervised methods can yield excellent
performance, but annotation of ground-truth data remains a major bottleneck, espe-
cially for biomedical applications where it has to be performed by domain experts.
The amount of labels required can be drastically reduced by using rules derived
from prior knowledge to guide the segmentation. However, these rules are in
general not differentiable and thus cannot be used with existing methods. Here,
we relax this requirement by using stateless actor critic reinforcement learning,
which enables non-differentiable rewards. We formulate the instance segmentation
problem as graph partitioning and the actor critic predicts the edge weights driven
by the rewards, which are based on the conformity of segmented instances to
high-level priors on object shape, position or size. The experiments on toy and real
datasets demonstrate that we can achieve excellent performance without any direct
supervision based only on a rich set of priors.

1

Introduction

Instance segmentation is the task of segmenting all objects in an image and assigning each of them
a different label. It forms the necessary ﬁrst step to the analysis of individual objects in a scene
and is thus of paramount importance in many practical applications of computer vision. Over the
recent years, fully supervised instance segmentation methods have made tremendous progress both
in natural image applications and in scientiﬁc imaging, achieving excellent segmentations for very
difﬁcult tasks [1, 2].

A large corpus of training images is hard to avoid when the segmentation method needs to take
into account the full variability of the natural world. However, in many practical segmentation
tasks the appearance of the objects can be expected to conform to certain rules which are known a
priori. Examples include surveillance, industrial quality control and especially medical and biological
imaging applications where full exploitation of such prior knowledge is particularly important as the
training data is sparse and difﬁcult to acquire: pixelwise annotation of the necessary instance-level
groundtruth for a microscopy experiment can take weeks or even months of expert time. The use of
shape priors has a strong history in this domain [3, 4], but the most powerful learned shape models
still require groundtruth [5] and generic shapes are hard to combine with the CNN losses and other,
non-shape, priors. For many high-level priors it has already been demonstrated that integration of
the prior directly into the CNN loss can lead to superior segmentations while signiﬁcantly reducing
the necessary amounts of training data [6]. However, the requirement of formulating the prior as
a differentiable function poses a severe limitation on the kinds of high-level knowledge that can
be exploited with such an approach. The aim of our contribution is to address this limitation and

Submitted to 35th Conference on Neural Information Processing Systems (NeurIPS 2021). Do not distribute.

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

establish a framework in which a rich set of non-differentiable rules and expectations can be used to
steer the network training.

To circumvent the requirement of a differentiable loss function, we turn to the reinforcement learning
paradigm, where the rewards can be computed from a non-differentiable cost function. We base
our framework on a stateless actor-critic setup [7], providing one of the ﬁrst practical applications
of this important theoretical construct. In more detail, we solve the instance segmentation problem
as agglomeration of image superpixels, with the agent predicting the weights of the edges in the
superpixel region adjacency graph. Based on the predicted weights, the segmentation is obtained
through (non-differentiable) graph partitioning and the segmented objects are then evaluated by the
critic, which learns to approximate the rewards based on the object- and image-level reasoning (see
Fig. 1).

The main contributions of this work can be summarized as follows: (i) we formulate instance segmen-
tation as a RL problem based on a stateless actor-critic setup, encapsulating the non-differentiable step
of instance extraction into the environment and thus achieving end-to-end learning; (ii) we exploit
prior knowledge on instance appearance and morphology by tying the rewards to the conformity of
the predicted objects to pre-deﬁned rules and learning to approximate the (non-differentiable) reward
function with the critic; (iii) we introduce a strategy for spatial decomposition of rewards based on
ﬁxed-sized subgraphs to enable localized supervision from combinations of object- and image-level
rules. (iv) we demonstrate the feasibility of our approach on synthetic and real images and show
an application to an important segmentation task in developmental biology, where our framework
delivers an excellent segmentation with no supervision other than high-level rules.

2 Related work
Reinforcement learning has so far not found signiﬁcant adoption in the segmentation domain. The
closest to our work are two methods in which RL has been introduced to learn a sequence of
segmentation decision steps as a Markov Decision Process. In the actor critic framework of [8], the
actor recurrently predicts one instance mask at a time based on the gradient provided by the critic.
The training needs fully segmented images as supervision and the overall system, including an LSTM
sub-network between the encoder and the decoder, is fairly complex. In [9], the individual decision
steps correspond to merges of clusters while their sequence deﬁnes a hierarchical agglomeration
process on a superpixel graph. The reward function is based on Rand index and thus not differentiable,
but the overall framework requires full (super)pixelwise supervision for training.

Reward decomposition was introduced for multi agent RL by [10] where a global reward is decom-
posed into a per agent reward. [11] proves that a stateless RL setup with decomposed rewards requires
far less training samples than a RL setup with a global reward. In [12] reward decomposition is
applied both temporally and spatially for zero-shot inference on unseen environments by training on
locally selected samples to learn the underlying physics of the environment.

The restriction to differentiable losses is present in all application domains of deep learning. Common
ways to address it are usually based on a soft relaxation of the loss that can be differentiated. The
relaxation can be designed speciﬁcally for the loss, such as, for example, Area-under-Curve [13] for
classiﬁcation or Jaccard Index [14] for semantic segmentation. These approaches are not directly
applicable to our use case as we aim to enable the use of a variety of object- and image-level priors
which can easily be combined without handcrafting an approximate loss for each case. More generally,
but still for a concrete task loss, Direct Loss Minimization has been proposed for CNN training in
[15]. For semi-supervised learning of a classiﬁcation or ranking task, Discriminative Adversarial
Networks have been proposed as a means to learn an approximation to the loss [16]. Most generally,
Grabocka et al. in [17] propose to train a surrogate neural network which will serve as a smooth
approximation of the true loss. In our setup, the critic can informally be viewed as a surrogate network
as it learns to approximate the priors through the rewards by Q-learning.

Incorporation of rules and priors is particularly important in biomedical imaging applications, where
such knowledge can be exploited to augment or even substitute scarce groundtruth annotations.
For example, the shape prior is explicitly encoded in the popular nuclear [18] and cellular [19]
segmentation algorithms based on spatial embedding learning. Learned non-linear representations
of the shape are used in [5], while in [20] the loss for object boundary prediction is made topology-
aware. Domain-speciﬁc priors can also be exploited in post-processing by graph partitioning [21].
Interestingly, the energy minimization procedure underlying the graph partitioning can also be
incorporated into the learning step [22, 23].

2

Figure 1: Interaction of the agent with the environment: (a) shows the state which is composed of
the raw image and the superpixel over-segmentation; (b) depicts the agent and the superpixel graph,
which accumulates the features for nodes of the GNN from pixels which belong to the corresponding
superpixels; (c) given the state, the agent performs the actions by predicting edge weights on the
superpixel graph; (d) the environment, which includes the graph partitioning built from the weights
predicted through agent actions; (e) rewards are obtained by evaluating the segmentation arising from
the graph partitioning, based on pre-deﬁned and data dependent rules. The rewards are given back to
the agent where they are used for training.

93

94

95

96

97

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

3 Methods

The task of instance segmentation can be formalized as transforming an image x into a labeling y,
where y maps each pixel to a label value. An instance corresponds to the maximal set of pixels with
the same label value. Typically, the instance segmentation problem is solved via supervised learning,
i.e. using a training set with ground-truth labels ˆy. Note that y is invariant under the permutation
of label values. In general, it is difﬁcult to formulate instance segmentation in a fully differentiable
manner. Most approaches ﬁrst predict a "soft" representation with a CNN, e.g. afﬁnities [1, 24, 25],
boundaries [26, 27] or embeddings [28, 29] and apply non-differentiable post-processing, such as
agglomeration [27, 30], clustering [31, 32] or partitioning [33], to obtain the instance segmentation.
Alternatively, proposal-based methods predict a bounding-box per instance and then predict the
instance mask for each bounding-box [34]. Furthermore, the common evaluation metrics for instance
segmentation [35, 36] are also not differentiable.

Our main motivation to explore RL for the instance segmentation task is to circumvent the restriction
to differentiable losses and - regardless of the loss - to make the whole pipeline differentiable end-to-
end even in presence of non-differentiable steps which transform pixelwise CNN predictions into
individual instances.

We formulate the instance segmentation problem using a region adjacency graph G = (V, E),
where the nodes V correspond to superpixels (homogeneous clusters of pixels) and the edges E
connect nodes which belong to spatially adjacent superpixels. Given edge weights W , an instance
segmentation can be obtained by partitioning the graph, here using an approximate multicut solver
[37]. Together, the image data, superpixels, graph and the graph partitioning make up the environment
E of our RL setup. Based on the state s of E, the agent A predicts actions a, which are used to
compute the partitioning. The reward r is then computed based on this partitioning. Our agent A is a
stateless actor-critic [38], represented by two graph neural networks (GNN) [39]. The actor predicts
the actions a based on the graph and its node features F . The node(superpixel) features are computed
by pooling together the corrresponding pixel features based on the raw image data.

3

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

130

131

132

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

Here, we make use of two different setups: Method 1, where the per-pixel features are computed
based on the image data with the feature extractor being part of the agent A and Method 2 where the
feature extractor is part of the environment E. The feature extractor is trained end-to-end in Method
1, whereas it is ﬁxed and thus needs to be pre-trained in Method 2. We use a U-Net [40] as feature
extractor and can use hand-crafted features in addition to the learned features. More details about
the pre- training can be found in the Appendix. The agent - environment interaction for Method 1 is
depicted in Figure 1. For Method 2 we refer to the Appendix.

Importantly, this setup enables us to use both a non-differentiable instance segmentation step and
reward function, by encapsulation of the “pixels to instances” step in the environment and learning a
policy based on the rewards with a stateless actor critic.

3.1 Stateless Reinforcement Learning Setup
Unlike most RL settings [41], our approach does not require an explicitly time dependent state: the
actions returned by the agent correspond to the real-valued edge weights in [0, 1], which are used to
compute the graph partitioning. Any state can be reached by a single step from the initial state and
there exists no time dependency in the state transition. Unlike [9], we predict all edge values at once
which allows us to avoid the iterative strategy of [8] and deliver and evaluate a complete segmentation
in every step. We implement a stateless actor critic formulation with episodes of length 1.

To the best of our knowledge, stateless RL was introduced in [7] to study the connection between
generative adversarial networks and actor critics and our method is one of the ﬁrst practical applica-
tions of this concept. Here, the agent consists of an actor, which predicts the actions a and a critic,
which predicts the action value Q (expected future discounted reward) given the actions. The stateless
approach simpliﬁes the action value function: the action value has to estimate the reward for a single
step instead of estimating the expected sum of discounted future rewards for many steps. We have
explored a multi-step setup as well, but found that it yields inferior results for our application; details
can be found in the Appendix. As described in detail in 3.2, we compute localized sub-graph rewards
instead of relying on a single global reward.

The actor corresponds to a single GNN, which predicts the mean and variance of a normal distribution
for each edge. The actions a are determined by sampling from this distribution and applying a
sigmoid to the result to obtain continuous edge weights in the value range [0, 1]. The GNN takes the
state s = (G, F ) as input arguments and its graph convolution for the ith node is deﬁned as in [39]:



fi = γπ

fi,



φπ (fi, fj)



1
|N (i)|

(cid:88)

j∈N (i)

(1)

149

150

where γπ as well as φπ are MLPs, (·, ·) is the concatenation of vectors and N (i) is the set of neighbors
of node i. The gradient of the loss for the actor is given by:

∇θLactor = ∇θ

1
|SG|

(cid:88)



α

(cid:88)

sg∈G

ˆa∈sg

log(πθ(ˆa|s)) − Qsg(s, a)



(2)



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

This loss gradient is derived following [38]. We adapt it to the sub-graph reward structure by
calculating the joint action probability of the policy πθ over each sub-graph sg in the set of all
sub-graphs SG. Using this loss to optimize the policy parameters θ minimizes the Kullback-Leibler
divergence between the Gibbs distribution of action values for each sub-graph Qsg(s, a) and the
policy with respect to the parameters θ of the policy. α is a trainable temperature parameter which is
optimized following the method introduced by [38].

The critic predicts the action value Qsg for each sub-graph sg ∈ SG. It consists of a GNN Qsg(s, a)
that takes the state s = (G, F ) as well as the actions a predicted by the actor as input and predicts a
feature vector for each edge. The graph convolution from Equation 2 is slightly modiﬁed:



fi = γQ

fi,

1
|N (i)|

(cid:88)

φQ

(cid:0)fi, fj, a(i,j)

(cid:1)

j∈N (i)





(3)

161

162

again γQ and φQ are MLPs. Based on these edge features Qsg is predicted for each sub-graph via an
MLP. Here, we use a set of subgraph sizes (typically, 6, 12, 32, 128) to generate a supervison signal

4

Figure 2: The graph is subdivided into sub-
graphs, each sub-graph is highlighted by a
different color. All sub-graphs have the same
number of edges (here 3). Overall, we use a
variety of sizes covering different notions of
locality.

Figure 3: An example reward landscape Cir-
cle Hough Transform (CHT) rewards. High
rewards are given if the overall number of
predicted objects is not too high and if the
respective object has a large CHT value. We
found an exponential gradient of the reward
landscape to work best.

163

164

for different neighborhood scales. A given MLP is only valid for a ﬁxed graph size, so we employ a
different MLP for each size. The loss for the critic is given by:

Lcritic =

1
|SG|

(cid:88)

sg∈G

1
2

(Qδ

sg(s, a) − r)2

(4)

165

166

167

168

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

Minimizing this loss with respect to the action value function’s parameters δ minimizes the difference
between the expected reward and action values Qδ

sg(s, a).

3.2 Localized Supervision Signals
The RL paradigm is to provide a global reward for a given state transition [41]. However, we ﬁnd
that for our application it is possible and desirable to instead provide several more localized rewards
per state transition: Given a large action space with a policy represented by a complex multivariate
probability distribution, it is beneﬁcial to learn from rewards for the speciﬁc actions rather than from
a scalar global reward for the union of all actions. Of course then requirement arises that the union of
local rewards must resemble to the global reward. E.g. the optimal policy is the same for local as for
the global reward.

Our actor critic setup (Section 3.1) expects rewards per sub-graph. A good set of sub-graphs should
fulﬁll the following requirements: each sub-graph should be connected so that the information
presented to the MLP computing the action value for this sub-graph is correlated. The size of
the sub-graphs, given by the number of edges, should be a parameter and all sub-graphs should
be extracted with exactly that size to serve as valid input for one of the MLPs. The union of all
sub-graphs should cover the complete graph so that each edge contributes to at least one action
value Qsg. The sub-graphs should overlap to provide a smooth sum of action values. We introduce
Algorithm 1 to extract such a set of sub-graphs (see Appendix). Figure 2 shows the sub-graphs for a
small example graph.

While some of the rewards used in our experiments can be directly deﬁned for the sub-graphs, most
are instead deﬁned per object (see Appendix for details on reward design). We use the following
general procedure to map object-level rewards to sub-graphs: ﬁrst assign to each superpixel the
reward of its corresponding object, then determine the reward per edge as the maximum value of its
two incident superpixels’ rewards and average the edge rewards to obtain the reward per sub-graph.
Here, we use the maximum because high object scores indicate that all actions contributing to the
respective object should get a high reward. However, for low object scores it is not possible to localize
the speciﬁc action responsible for the low score. Hence, by taking the maximum we assign the
higher score to edges whose incident superpixels belong to different objects, because they probably
correspond to a correct split. Note that the uncertainty in the assignment of low rewards can lead to
a noisy reward signal, but the averaging of the edge rewards over the sub-graphs and the overlaps

5

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

206

207

208

209

210

211

212

213

214

215

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

between the sub-graphs smooth and partially denoise the rewards. We have also explored a different
actor critic setup that can use object level rewards directly, eliminating the need for the sub-graph
extraction and mapping. However, this approach yields inferior results, see the Appendix for details.

4 Experiments
The agent of our setup acts on the superpixel graph and thus depends on the features assigned to the
nodes of the graph. We introduced two variants of our algorithm: in the base variant (Method 1)
we start from random features and make them part of the agent, allowing them to change through
back-propagation (Fig. 1). In contrast, Method 2 acts on predeﬁned features which are provided
as part of the environment and are computed before training, e.g. through unsupervised clustering.
A very accurate clustering in the features produces an easy problem for the agent to solve where
even a global reward for all actions might be sufﬁcient. However, in a real-world setting with no
supervision, the noisier the features become the more local the reward has to be. We evaluate Method
2 on synthetic data where self-supervised pretraining can deliver noisy, but meaningful node features.
Our full setup with Method 1 is evaluated on a dataset from a light microscopy experiment, where
highly regular object shapes are to be expected, but no good feature pre-training is possible.

To transform the edge weight predictions of the agent into an instance segmentation we use the
Multicut [42] algorithm. Here, other options are also possible such as hierarchical clustering used in
[9], but we choose the Multicut for its global optimality property. Hyperparameters of the pipeline
were found by cross-validation (see Appendix).

4.1 Synthetic dataset: circles on structured ground
To evaluate the feasibility of our approach, we create a synthetic dataset with prominent structured
background. Our aim is to segment irregular disks on such background using only rule-based
supervision. We generate the superpixels by the mutex watershed algorithm [25] which we run on
the Gaussian gradient image. The node features of the superpixel graph were computed through
self-supervised pretraining with contrastive loss as described in Appendix and ﬁxed as part of the
environment.

As we aim to segment disks, we compute the circularity of the segmented objects for the rewards
using the Circle Hough Transform [43]. This object-level reward is combined with the global rough
estimate of the number of objects in the image to create the reward surface depicted in Fig. 3. The
reward for the number of objects provides useful gradient during early training stages: for example,
when too few potential objects are found in the prediction, a low reward can be given to what is
thought to be the background object. On the other hand, if too many potential objects are found, a
low reward can be given to all the foreground objects with a low CHT value.

In more detail, the object rewards rf g are composed as follows. We deﬁne a threshold γ on the CHT
value (γ = 0.8 in the reward surface shown in Fig. 3). Let c ∈ [0, 1] be the CHT value corresponding
to the object and let k be the total number of objects that we expect and n be the number of predicted
objects. Then

rlocal =

rglobal =

(cid:40)

σ

(cid:16)

(cid:17)
( c−γ
1−γ − 0.5)6

0.4,

if c ≥ γ

0,
(cid:26)rexp
0.6,

(cid:0) k
n

(cid:1) ,

if n ≥ k
otw

otw

rf g = rlocal + rglobal

(5)

(6)

(7)

232

233

234

235

236

Here σ(·) is the sigmoid function. The input to the sigmoid function is normalized to the interval
[−3, 3] which was empirically found to be a good range. The rewards are always in [0, 1] here this is
split up into [0, 0.5] for the local reward as well as for the global reward.
For the largest predicted object we strongly suspect the background object. For this object background
rewards rbg are calculated by

rbg =

(cid:0) n
k

(cid:1) ,

(cid:26)rexp
1,

if n ≤ k
otw

(8)

237

238

Note that this rewards have a large globally calculated part which makes this setup not ﬁt for Method
1. It needs some feature representation that already gives a good idea for the clustering. The only

6

(a) Reinforcement learning output.

(b) Mutex watershed baseline.

Figure 4: The “Circles” dataset. Top left to right: ground truth segmentation, raw data, superpixel over-
segmentation and a visualizataion for the actions on every edge, where a merge action is displayed in
green and a split action in red. Bottom left to right: the pre-trained pixel embeddings projected to
their ﬁrst 3 PCA components shown as RGB, an edge image of the superpixels, the segmentation
resulting from the graph agglomeration on the predicted edge weights and a visualization of the
rewards based on the CHT, where light green shows high rewards and dark red low rewards.

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

useful local information in the reward is the CHT value. Therefore, if the features have a fairly
distinct structure for circles, the agent should be able to ﬁnd and to correctly cluster them.
Fig. 4 shows the output of all algorithm components on a sample image. For comparison, we also
computed mutex watershed [25] predictions. Texture within objects and structured background are
inherently difﬁcult for region-growing algorithms, but our approach can exploit higher-level reasoning
along with low-level information and achieve a good segmentation.

4.2 Real dataset: light microscopy imaging

Biomedical applications often require segmentation of objects of known morphology which are
positioned in regular patterns, while extensive prior knowledge is available on variability of both
under normal experimental conditions [44]. Such data presents the best use case for our algorithm as
the reward function can leverage the known characteristics of individual object shape and texture and
the overall similarity of the objects.

The dataset used for this experiment contains 317 2D images extracted from a video of a developing
fruitﬂy embryo acquired with a light-sheet microscope [45] (Fig. 5). The image shows boundaries
(plasma membranes) of the embryo cells. Across the dataset, 10 images were fully segmented by an
expert, we use those for validation.

Fruitﬂy embryo is a well-studied system for which we can exploit the prior knowledge on the expected
cell shape and the radial pattern of cells. Furthermore, as the analysis of cell shape dynamics is
a paramount part of many biological experiments, multiple pre-trained networks are available for
the cell segmentation task [18, 19, 46, 47]. Due to the differences in sample preparation and image
acquisition settings, none of these would work out-of-the-box for our data. However, the CNNs in
[47] which are trained to predict boundaries in confocal microscope images of plant tissue, can serve
as a strong edge detector to create superpixels in our images. The superpixels are obtained using the
seeded watershed algorithm on seeds at the local minima of the predicted edge map.

The rewards for this experiment are designed as follows: we set a high reward for merging the
superpixels which are certain to lie in the background (close to the image boundary or the image
center). For the background edges near the foreground area we modulate the reward by the circularity
of the overall foreground contour. Finally, for the edges which are likely to be in the foreground
we compute object-level rewards by ﬁtting a rotated bounding box to each object and comparing its
side lengths as well as its orientation to predeﬁned template values. We do not perform semantic
segmentation to deﬁne precise foreground/background boundaries, but instead use a soft weighting
scheme with Gaussian weights to combine object and background rewards based on on the prior

7

Figure 5: Microscopy dataset experiment. Top left to right: ground truth segmentation; raw data;
edge map; superpixel over-segmentation; visualization for the actions on every edge, where a merge
action is displayed in green and a split action in red. Bottom left to right: a) handcrafted features;
b) learned features accumulated on superpixels; c) learned features projected to their ﬁrst 3 PCA
components shown as RGB; the segmentation resulting from the Multicut on the predicted edge
weights; visualization of the rewards, where light green shows high rewards and dark red low rewards.

knowledge of the embryo width. An image of the weights for different locations in the image can be
found in the appendix.

More formally the edge rewards redge are calculated as follows. For each edge, we deﬁne the distance
h between the edge and the center of the image as the average distance of the incident objects’ center
of mass and the center c of the image. j is the approximate radius of the circle that lies within the
foreground and m is the maximal distance between c and the image boarder. Let further K(·) be the
Gaussian kernel function. Then redge yields

rbg =




K

(cid:17)

(cid:16) ||h−c||
γ
(cid:16) ||m−h||
η



K
(cid:18) ||h − j||
δ
redge = rf g + rbg

rf g = K

if h ≤ j

(1 − a),
(cid:17)

(1 − a), otw

(cid:19)

max(ro1, ro2)

(9)

(10)

(11)

Here γ, η, δ are normalization constants. Equation 9 ﬁrst determines the background probability for
an edge by the kernel values. 1 − a constitutes a reward that directly favors merges which is scaled
by the background probability. For each edge, ro1 and ro2 are the rewards corresponding to the two
objects connected to that edge. The object rewards are given by ﬁtting a rotated bounding box to the
object and then compare rotation and dimensions to template values.

Note that in this experiment no self-supervised pretraining is used for the node features in the
agent’s GNNs. Unlike the “Circles” dataset, all objects in these images have very similar intensity
distributions and can only be separated through the detection of boundaries between them. Instead
of the pretraining, we experiment with using a few hand-crafted features like the polar coordinate
of the node’s respective superpixel’s center of mass with respect to the coordinate system sitting
at the center of the image as well as the superpixel’s mass, and with learning other features by
back-propagation from the agent. The handcrafted features are normalized, concatenated to the
learned features and used as input to the GNN. The projection of the ﬁrst 3 PCA components of these
features into RGB space is shown in Fig. 5 respectively for learned feature maps, their projection
to node features through the accumulation procedure and ﬁnally the concatenation of those and the

8

271

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

Table 1: Quantitative evaluation on the microscopy dataset. Note that the projection of superpixels to
the ground truth (sp gt) sets an upper (lower for VI) bound for our method. We use Symmetric Best
Dice as well as the Variation of Information metric to compare all results on the validation set.

Method

sp gt

SBD

VI merge

VI split

0.656 ± 0.019

0.672 ± 0.061

0.594 ± 0.028

ours + augmentation noise
ours
ours without edges
ours only handcrafted
edge + mc [47]
contrastive [28]
contrastive + edge [28]

0.508 ± 0.031
0.482 ± 0.020
0.446 ± 0.041
0.408 ± 0.087
0.283 ± 0.023
0.215 ± 0.009
0.248 ± 0.014

1.233 ± 0.156
0.839 ± 0.118
0.953 ± 0.212
0.987 ± 0.101
3.019 ± 0.040
1.155 ± 0.037
1.229 ± 0.048

1.060 ± 0.258
1.374 ± 0.357
0.994 ± 0.200
1.536 ± 0.410
0.342 ± 0.045
3.285 ± 0.084
3.336 ± 0.073

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

handcrafted features. Note that the learned features converge to a representation which resembles a
semantic segmentation of boundaries in the image.

We train the complete setup for Method 1 end-to-end on a Nvidia GeForce RTX 3090 GPU for 4
days. For comparison we keep the model which achieved the highest reward on the test set. This
makes training as well as the validation independent from ground truth annotations. The evolution
of the rewards on the validation set for different random seeds is shown in the Appendix. All of the
conducted trainings show a stride for high rewards regardless of different random seeding.

For the validation scores we use the variation of information (VI) for both input combinations (merge
and split) and the Symmetric Best Dice score. To show the inﬂuence of the imperfect superpixels on
the ﬁnal clustering, we project the superpixels to their respective ground truth clustering ("sp gt" in
Table 1) which sets an upper (lower in case of VI) bound for our method. In this study we use several
versions of our approach. In Table 1 (ours) refers to method 1 as described in section 4.2, (ours +
augmentation noise) is the same method but add some noise to the input data during training, (ours
without edges) is our method but without the additional edge prediction as an input and (ours only
handcrafted) is our method where we only use the handcrafted features as described in section 4.2.
We ﬁnd that learned features signiﬁcantly contribute to the performance of our method.
We compare to the following baseline approaches: edge + mc, which solves the Multicut graph
partitioning based on edge weights derived from boundary predictions used for superpixel creation,
contrastive, which predicts a pixel-wise embedding space that is clustered into instances using
k-means and for which the embeddings are trained using the discriminative loss function of [28] on
the ovules dataset from [47] and contrastive + edge, which is similar to contrastive, but receives the
[47] boundary predictions as additional input channel.

5 Discussion

We introduced an end-to-end instance segmentation algorithm which can exploit non-differentiable
loss functions and high-level prior information. Our RL approach is based on stateless actor-critic
and predicts the full segmentation at every step, allowing us to assign rewards to all objects and
reach stable convergence. The segmentation problem is formulated as graph partitioning; we design
a reward decomposition algorithm which maps object- and image-level rewards to sub-graphs for
localized supervision.

We performed proof-of-concept experiments to demonstrate the feasibility of our approach on
synthetic and real data and showed in particular that our setup can segment microscopy images
with no direct supervision other than high-level reasoning. In the future, we plan to explore other
problems and reward functions as well as a semi-supervised setup (brieﬂy introduced in Appendix)
where we think our approach can be very beneﬁcial. Furthermore, even in case of full supervision
with ample groundtruth, our RL-based formulation enables end-to-end instance segmentation with
direct object-level reasoning, which will allow for post-processing-aware training of the CNN which
predicts object boundaries or embeddings.

9

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

References

[1] Kisuk Lee, Jonathan Zung, Peter Li, Viren Jain, and H Sebastian Seung. Superhuman accuracy

on the snemi3d connectomics challenge. arXiv preprint arXiv:1706.00120, 2017.

[2] Liang-Chieh Chen, Huiyu Wang, and Siyuan Qiao. Scaling wide residual networks for panoptic

segmentation, 2021.

[3] S. Osher and N. Paragios. Geometric Level Set Methods in Imaging, Vision, and Graphics.
Springer New York, 2007. ISBN 9780387218106. URL https://books.google.de/books?
id=ZWzrBwAAQBAJ.

[4] Ricard Delgado-Gonzalo, Virginie Uhlmann, Daniel Schmitter, and Michael Unser. Snakes on
a plane: A perfect snap for bioimage analysis. IEEE Signal Processing Magazine, 32(1):41–48,
2014.

[5] Ozan Oktay, Enzo Ferrante, Konstantinos Kamnitsas, Mattias Heinrich, Wenjia Bai, Jose
Caballero, Stuart A. Cook, Antonio de Marvao, Timothy Dawes, Declan P. O‘Regan, Bernhard
Kainz, Ben Glocker, and Daniel Rueckert. Anatomically constrained neural networks (acnns):
Application to cardiac image enhancement and segmentation. IEEE Transactions on Medical
Imaging, 37(2):384–395, 2018. doi: 10.1109/TMI.2017.2743464.

[6] Hoel Kervadec, Jose Dolz, Meng Tang, Eric Granger, Yuri Boykov, and Ismail Ben Ayed.
Constrained-cnn losses for weakly supervised segmentation. Medical Image Analysis, 54:88–99,
2019. ISSN 1361-8415. doi: https://doi.org/10.1016/j.media.2019.02.009.

[7] David Pfau and Oriol Vinyals. Connecting generative adversarial networks and actor-critic
methods. CoRR, abs/1610.01945, 2016. URL http://arxiv.org/abs/1610.01945.

[8] Nikita Araslanov, Constantin Rothkopf, and Stefan Roth. Actor-critic instance segmentation.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
2019.

[9] Viren Jain, Srinivas Turaga, Kevin Briggman, Moritz Helmstaedter, Winfried Denk, and Hyun-
june Seung. Learning to agglomerate superpixel hierarchies. Advances in Neural Information
Processing Systems, 24, 01 2011.

[10] Peter Sunehag, Guy Lever, Audrunas Gruslys, Wojciech Marian Czarnecki, Vinicius Zambaldi,
Max Jaderberg, Marc Lanctot, Nicolas Sonnerat, Joel Z. Leibo, Karl Tuyls, and Thore Graepel.
Value-decomposition networks for cooperative multi-agent learning, 2017.

[11] Drew Bagnell and Andrew Ng. On local rewards and scaling distributed reinforcement learning.
In Y. Weiss, B. Schölkopf, and J. Platt, editors, Advances in Neural Information Processing
Systems, volume 18. MIT Press, 2006. URL https://proceedings.neurips.cc/paper/
2005/file/02180771a9b609a26dcea07f272e141f-Paper.pdf.

[12] Huazhe Xu, Boyuan Chen, Yang Gao, and Trevor Darrell. Scoring-aggregating-planning:
Learning task-agnostic priors from interactions and sparse rewards for zero-shot generalization.
CoRR, abs/1910.08143, 2019. URL http://arxiv.org/abs/1910.08143.

[13] Elad Eban, Mariano Schain, Alan Mackey, Ariel Gordon, Ryan Rifkin, and Gal Elidan. Scalable
learning of non-decomposable objectives. In Artiﬁcial intelligence and statistics, pages 832–840.
PMLR, 2017.

[14] Maxim Berman, Amal Rannen Triki, and Matthew B Blaschko. The lovász-softmax loss:
A tractable surrogate for the optimization of the intersection-over-union measure in neural
networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
pages 4413–4421, 2018.

[15] Yang Song, Alexander G. Schwing, Richard S. Zemel, and Raquel Urtasun. Training deep
neural networks via direct loss minimization. International Conference on Machine Learning,
2016.

10

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

[16] Cicero Nogueira dos Santos, Kahini Wadhawan, and Bowen Zhou. Learning loss functions for

semi-supervised learning via discriminative adversarial networks, 2017.

[17] Josif Grabocka, Randolf Scholz, and Lars Schmidt-Thieme. Learning surrogate losses. arXiv

preprint arXiv:1905.10108, 2019.

[18] Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers. Cell detection with
star-convex polygons. In International Conference on Medical Image Computing and Computer-
Assisted Intervention, pages 265–273. Springer, 2018.

[19] Carsen Stringer, Tim Wang, Michalis Michaelos, and Marius Pachitariu. Cellpose: a generalist

algorithm for cellular segmentation. Nature Methods, 18(1):100–106, 2021.

[20] Xiaoling Hu, Fuxin Li, Dimitris Samaras, and Chao Chen.

deep image segmentation.
volume 32, 2019.
2d95666e2649fcfc6e3af75e09f5adb9-Paper.pdf.

Topology-preserving
In Advances in Neural Information Processing Systems,
URL https://proceedings.neurips.cc/paper/2019/file/

[21] Constantin Pape, Alex Matskevych, Adrian Wolny, Julian Hennies, Giulia Mizzon, Marion
Louveaux, Jacob Musser, Alexis Maizel, Detlev Arendt, and Anna Kreshuk. Leveraging domain
knowledge to improve microscopy image segmentation with lifted multicuts. Frontiers in
Computer Science, 1:6, 2019.

[22] Jeremy B Maitin-Shepard, Viren Jain, Michal Januszewski, Peter Li, and Pieter Abbeel. Combi-
natorial energy learning for image segmentation. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon,
and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 29. Cur-
ran Associates, Inc., 2016. URL https://proceedings.neurips.cc/paper/2016/file/
31857b449c407203749ae32dd0e7d64a-Paper.pdf.

[23] Jie Song, Bjoern Andres, Michael J Black, Otmar Hilliges, and Siyu Tang. End-to-end learning
In Proceedings of the IEEE/CVF International Conference on

for graph decomposition.
Computer Vision, pages 10093–10102, 2019.

[24] Naiyu Gao, Yanhu Shan, Yupei Wang, Xin Zhao, Yinan Yu, Ming Yang, and Kaiqi Huang.
Ssap: Single-shot instance segmentation with afﬁnity pyramid. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pages 642–651, 2019.

[25] Steffen Wolf, Alberto Bailoni, Constantin Pape, Nasim Rahaman, Anna Kreshuk, Ullrich Köthe,
and Fred A Hamprecht. The mutex watershed and its objective: Efﬁcient, parameter-free graph
partitioning. IEEE transactions on pattern analysis and machine intelligence, 2020.

[26] Thorsten Beier, Constantin Pape, Nasim Rahaman, Timo Prange, Stuart Berg, Davi D Bock,
Albert Cardona, Graham W Knott, Stephen M Plaza, Louis K Scheffer, et al. Multicut brings
automated neurite segmentation closer to human performance. Nature methods, 14(2):101–102,
2017.

[27] Jan Funke, Fabian Tschopp, William Grisaitis, Arlo Sheridan, Chandan Singh, Stephan Saalfeld,
and Srinivas C Turaga. Large scale image segmentation with structured loss based deep learning
for connectome reconstruction. IEEE transactions on pattern analysis and machine intelligence,
41(7):1669–1680, 2018.

[28] Bert De Brabandere, Davy Neven, and Luc Van Gool. Semantic instance segmentation with a

discriminative loss function. arXiv preprint arXiv:1708.02551, 2017.

[29] Davy Neven, Bert De Brabandere, Marc Proesmans, and Luc Van Gool. Instance segmentation
by jointly optimizing spatial embeddings and clustering bandwidth. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8837–8845, 2019.

[30] Alberto Bailoni, Constantin Pape, Steffen Wolf, Thorsten Beier, Anna Kreshuk, and Fred A
Hamprecht. A generalized framework for agglomerative clustering of signed graphs applied to
instance segmentation. arXiv preprint arXiv:1906.11713, 2019.

11

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

[31] Leland McInnes and John Healy. Accelerated hierarchical density based clustering. In 2017
IEEE International Conference on Data Mining Workshops (ICDMW), pages 33–42. IEEE,
2017.

[32] Dorin Comaniciu and Peter Meer. Mean shift: A robust approach toward feature space analysis.
IEEE Transactions on pattern analysis and machine intelligence, 24(5):603–619, 2002.

[33] Bjoern Andres, Thorben Kroeger, Kevin L Briggman, Winfried Denk, Natalya Korogod, Graham
Knott, Ullrich Koethe, and Fred A Hamprecht. Globally optimal closed-surface segmentation
for connectomics. In European Conference on Computer Vision, pages 778–791. Springer,
2012.

[34] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In Proceedings of

the IEEE international conference on computer vision, pages 2961–2969, 2017.

[35] Marina Meil˘a. Comparing clusterings by the variation of information. In Learning theory and

kernel machines, pages 173–187. Springer, 2003.

[36] William M Rand. Objective criteria for the evaluation of clustering methods. Journal of the

American Statistical association, 66(336):846–850, 1971.

[37] Brian W Kernighan and Shen Lin. An efﬁcient heuristic procedure for partitioning graphs. The

Bell system technical journal, 49(2):291–307, 1970.

[38] Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan,
Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, and Sergey Levine. Soft actor-critic
algorithms and applications. CoRR, abs/1812.05905, 2018. URL http://arxiv.org/abs/
1812.05905.

[39] Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl.
Neural message passing for quantum chemistry. CoRR, abs/1704.01212, 2017. URL http:
//arxiv.org/abs/1704.01212.

[40] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for
biomedical image segmentation. CoRR, abs/1505.04597, 2015. URL http://arxiv.org/
abs/1505.04597.

[41] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT
Press, second edition, 2018. URL http://incompleteideas.net/book/the-book-2nd.
html.

[42] Jörg Hendrik Kappes, Markus Speth, Björn Andres, Gerhard Reinelt, and Christoph Schn.
Globally optimal image partitioning by multicuts.
In Yuri Boykov, Fredrik Kahl, Victor
Lempitsky, and Frank R. Schmidt, editors, Energy Minimization Methods in Computer Vision
and Pattern Recognition, pages 31–44, Berlin, Heidelberg, 2011. Springer Berlin Heidelberg.
ISBN 978-3-642-23094-3.

[43] Allam Shehata Hassanein, Sherien Mohammad, Mohamed Sameer, and Mohammad Ehab Ragab.
A survey on hough transform, theory, techniques and applications. CoRR, abs/1502.02160,
2015. URL http://arxiv.org/abs/1502.02160.

[44] D’Arcy Wentworth Thompson. On Growth and Form. Canto. Cambridge University Press,

1992. doi: 10.1017/CBO9781107325852.

[45] Sourabh Bhide, Ralf Mikut, Maria Leptin, and Johannes Stegmaier. Semi-automatic generation
of tight binary masks and non-convex isosurfaces for quantitative analysis of 3d biological
samples, 2020.

[46] Lucas von Chamier, Romain F Laine, Johanna Jukkala, Christoph Spahn, Daniel Krentzel, Elias
Nehme, Martina Lerche, Sara Hernández-Pérez, Pieta K Mattila, Eleni Karinou, Séamus Holden,
Ahmet Can Solak, Alexander Krull, Tim-Oliver Buchholz, Martin L Jones, Loïc A Royer,
Christophe Leterrier, Yoav Shechtman, Florian Jug, Mike Heilemann, Guillaume Jacquemet,
and Ricardo Henriques. Democratising deep learning for microscopy with ZeroCostDL4Mic.
Nature Communications, 4 2021.

12

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

509

510

511

512

513

514

515

516

517

518

519

520

521

522

[47] Adrian Wolny, Lorenzo Cerrone, Athul Vijayan, Rachele Tofanelli, Amaya Vilches Barro,
Marion Louveaux, Christian Wenzl, Sören Strauss, David Wilson-Sánchez, Rena Lymbouridou,
et al. Accurate and versatile 3d segmentation of plant tissues at cellular resolution. Elife, 9:
e57613, 2020.

[48] Bert De Brabandere, Davy Neven, and Luc Van Gool. Semantic instance segmentation with a

discriminative loss function, 2017.

[49] Reuben R. Shamir, Yuval Duchin, Jinyoung Kim, Guillermo Sapiro, and Noam Harel.
Continuous dice coefﬁcient: a method for evaluating probabilistic segmentations. CoRR,
abs/1906.11031, 2019. URL http://arxiv.org/abs/1906.11031.

Checklist

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reﬂect the paper’s

contributions and scope? [Yes]

(b) Did you describe the limitations of your work? [Yes]
(c) Did you discuss any potential negative societal impacts of your work? [No] It does not

have any negative societal impacts.

(d) Have you read the ethics review guidelines and ensured that your paper conforms to

them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [N/A] We do not

claim any theoretical assumptions.

(b) Did you include complete proofs of all theoretical results? [N/A] See above.

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main experi-
mental results (either in the supplemental material or as a URL)? [Yes] Yes, see the
Supplementary.

(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they

were chosen)? [Yes] See the Supplementary

(c) Did you report error bars (e.g., with respect to the random seed after running experi-
ments multiple times)? [Yes] The bars are reported w.r.t. to the samples within the best
seed, see Supplementary section for further details

(d) Did you include the total amount of compute and the type of resources used (e.g., type

of GPUs, internal cluster, or cloud provider)? [Yes]

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [Yes]
(b) Did you mention the license of the assets? [Yes]
(c) Did you include any new assets either in the supplemental material or as a URL? [Yes]

The link to the biological dataset is given in the Supplementary

(d) Did you discuss whether and how consent was obtained from people whose data
you’re using/curating? The dataset was provided from the other laboratory as a part of
collaboration.

(e) Did you discuss whether the data you are using/curating contains personally identiﬁable

information or offensive content? [No] No human data is used

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if

applicable? [N/A]

(b) Did you describe any potential participant risks, with links to Institutional Review

Board (IRB) approvals, if applicable? [N/A]

(c) Did you include the estimated hourly wage paid to participants and the total amount

spent on participant compensation? [N/A]

13

