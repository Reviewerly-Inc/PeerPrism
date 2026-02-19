CobBO: Coordinate Backoff Bayesian Optimization
with Two-Stage Kernels

Anonymous Author(s)
Afﬁliation
Address
email

Abstract

Bayesian optimization is a popular method for optimizing expensive black-box
functions. Yet it oftentimes struggles in high dimensions where the computation
could be prohibitively expensive and a sufﬁcient estimation of the global landscape
requires more observations. We introduce Coordinate backoff Bayesian optimiza-
tion (CobBO) with two-stage kernels to alleviate this problem. In each iteration, a
promising subset of coordinates is selected in the ﬁrst stage, as past observed points
in the full space are projected to the selected subspace adopting a simple kernel
that sacriﬁces the approximation accuracy for computational efﬁciency. Then in
the second stage of the same iteration a more sophisticated kernel is applied for
estimating the landscape in the selected low dimensional subspace where the com-
putational cost becomes affordable. Effectively, this second stage kernel reﬁnes the
approximation of the global landscape estimated by the ﬁrst stage kernel through
a sequence of observations in the local subspace. This reﬁnement lasts until a
stopping rule is met determining when to back off from a certain subspace and
switch to another coordinate subset. This decoupling signiﬁcantly reduces the com-
putational burden in high dimensions, while the two-stage kernels of the Gaussian
process regressions fully leverage the observations in the whole space rather than
only relying on observations in each coordinate subspace. Extensive evaluations
show that CobBO ﬁnds solutions comparable to or better than other state-of-the-art
methods for dimensions ranging from tens to hundreds, while reducing the trial
complexity and computational costs.

1

Introduction

Bayesian optimization (BO) has emerged as an effective zero-order paradigm for optimizing expen-
sive black-box functions. The entire sequence of iterations rely only on the function values of the
already queried points without information on their derivatives. Though highly competitive in low
dimensions (e.g., the dimension D ≤ 20 [15]), Bayesian optimization based on Gaussian Process
(GP) regression has obstacles that impede its effectiveness, especially in high dimensions.

27
28 Approximation accuracy: GP regression assumes a class of random functions in a probability space
29
as surrogates that iteratively yield posterior distributions by conditioning on the queried points. When
30
suggesting new query points, for complex functions with numerous local optima and saddle points
due to local ﬂuctuations, always exactly using the values on the queried points as the conditional
events may mismatch the function’s local landscape by overemphasizing the approximation accuracy
of the global landscape.

34
35 Curse of dimensionality: As a sample efﬁcient method, Bayesian optimization often suffers from
36
high dimensions. Fitting the GP model (estimating the parameters, e.g., length_scales [14]), comput-
37
ing the Gaussian process posterior and optimizing the acquisition function in high dimensions all

33

32

31

38

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

Figure 1: Minimize the ﬂuctuated Rastrigin function on [−5, 10]50 with 20 initial samples. [Left]
Computation times for training the GP regression model and maximizing the acquisition function
at each iteration. CobBO signiﬁcantly reduces the execution time compared with a vanilla BO, e.g.
×13 faster in this case. [Right] The average error between the GP predictions before making queries
and the true function values at the queried points (solid curves, the higher the better) and the best
observed function value (dashed curves, the lower the better) at iteration t. CobBO captures the
global landscape less accurately using the RBF kernel, and then explores selected subspaces Ωt more
accurately using the Matern kernel. This eventually better exploits the promising subspaces.

incur large computational costs. It also results in statistical insufﬁciency of exploration [11, 65]. As
the GP regression’s error grows with dimensions [8], more samples are required to balance that in
high dimensions, which could cubically increase the computational costs in the worst case [45].

To alleviate these issues, we design coordinate backoff Bayesian optimization (CobBO) with two-
stage kernels, by challenging a seemingly natural intuition stating that it is always better for Bayesian
optimization to have a more accurate approximation of the objective function at all times. We
demonstrate that this is not necessarily true, by showing that smoothing out local ﬂuctuations and
using the estimated function values instead of the true observations to serve as the conditional events
in selected subspaces can not only signiﬁcantly reduce the computation time due to the curse of
dimensionality but also help in capturing the large-scale properties of the objective function f (x).

Speciﬁcally, CobBO introduces the two-stage kernels with a stopping rule. The ﬁrst stage of each
iteration adopts a simple kernel that sacriﬁces the approximation accuracy of f (x) for computational
efﬁciency. For example, by using a universal radial basis function (RBF) approximation without
learnable parameters [50], CobBO can eliminate the model ﬁtting time in the full space. It captures
a smooth approximation ˆf (x) of the global landscape by interpolating the values of queried points
projected to selected promising subspaces. These projected points serve as the conditional events
for GP regression. In a selected coordinate subspace, the second stage of the same iteration applies
a sophisticated kernel that can tolerate high computational cost in low dimensions. For example,
CobBO uses the Automatic Relevance Determination (ARD) Matérn 5/2 kernel [40]. It reﬁnes the
approximation of the local landscape by a sequence of observations determined by a stopping rule that
backs off from a certain subspace and switches to another coordinate subset. In addition, computing
the Gaussian process posterior and optimizing the acquisition function are both efﬁciently conducted
in the low dimensional subspaces, bypassing the curse of dimensionality.

For iteration t,
instead of directly computing the Gaussian process posterior distribution
(cid:12)
(cid:110) ˆf (x)
(cid:12)Ht = {(xi, yi)}t
(cid:12)
by conditioning on the observations yi = f (xi) at queried
points xi in the full space Ω ⊂ RD for i = 1, . . . , t, we change the conditional events, and consider

i=1 , x ∈ Ω

(cid:111)

(cid:110) ˆf (x)

(cid:12)
(cid:12)
(cid:12)R (PΩt(x1, . . . , xt), Ht) , x ∈ Ωt, Ωt ⊂ Ω

(cid:111)

for a projection function PΩt(·) to a random subspace Ωt and an interpolation function R(·, ·), e.g.,
using a RBF approximation without learnable parameters [50] as the simple kernel for the ﬁrst
stage. The projection PΩt(·) maps the queried points to virtual points on a subspace Ωt of a lower
dimension [51]. The interpolation function R(·, ·) estimates the objective values at the virtual points
using the queried points and their values as speciﬁed by Ht. The second stage within the subspace Ωt
uses the more sophisticated kernel, e.g., Matérn 5/2 kernel [40], which has a number of parameters
that otherwise would be expensive to be learned in high dimensions.

2

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

This method can be viewed as a variant of block coordinate ascent tailored to Bayesian optimization by
applying backoff stopping rules for switching coordinate blocks. While similar work exists [43, 48],
CobBO differs by introducing the two-stage kernels and addressing the following three issues:

1. Selecting a block of coordinates for ascending requires determining the block size as well as
the coordinates therein. CobBO selects the coordinate subsets by a multiplicative weights
update method [2] to the preference probability associated with each coordinate. Thus, it
samples more promising subspaces with higher probabilities.

2. A coordinate subspace requires a sufﬁcient number of query points acting as the conditional
events for the GP regression. CobBO leverages all observations in the whole space by
interpolating the values of queried points projected to selected promising subspaces, rather
than simply starting from scratch in each subspace.

3. Querying a certain subspace, under some trial budget, comes at the expense of exploring
other coordinate blocks. Yet prematurely shifting to different subspaces does not fully
exploit the full potential of a given subspace. Hence determining the number of consecutive
function queries within a subspace makes a trade-off between exploration and exploitation.
CobBO uses a stopping rule in each subspace to switch the selected coordinates. When
consecutively querying data points in the same subspace, CobBO does not need to conduct
the ﬁrst-stage function approximation in the full space, which is far more efﬁcient.

Through comprehensive evaluations, CobBO demonstrates appealing performance for dimensions
ranging from tens to hundreds. It obtains comparable or better solutions with fewer queries, in
comparison with the state-of-the-art methods, for most of the problems tested in Section 4.2.

2 Related work

Certain assumptions are often imposed on the latent structure in high dimensions. Typical assumptions
include low dimensional structures and additive structures. Their advantages manifest on problems
with a low dimension or a low effective dimension. However, these assumptions do not necessarily
hold for non-separable functions with no redundant dimensions.

Low dimensional structure: The black-box function f is assumed to have a low effective dimen-
sion [30, 58], e.g., f (x) = g(Φx) with some function g(·) and a matrix Φ of d × D, d << D. A
number of different methods have been developed, including random embedding [66, 11, 63, 36,
44, 70, 5, 32], low-rank matrix recovery [11, 58], and learning subspaces by derivative informa-
tion [11, 13]. In contrast to existing work on subspace selections, e.g., Hashing-enhanced Subspace
BO (HeSBO) [44], Mahalanobis kernel for linear embeddings [33], DROPOUT [35] and LineBO [29]
(which receives a special treatment in Appendix F), CobBO efﬁciently leverages all the observations
in the whole space using the two-stage kernels and the stopping rule in each subspace for consecutive
observations, rather than only relying on limited observations in each coordinate subspace. It exploits
subspace structure from a perspective of block coordinate ascent, independent of the dimensions,
different from some algorithms that are more suitable for low dimensions, e.g., BADS [1].
Additive structure: A decomposition assumption is often made by f (x) = (cid:80)k
i=1 f (i) (xi), with xi
deﬁned over low-dimensional components. In this case, the effective dimensionality of the model is
the largest dimension among all additive groups [45], which is usually small. The Gaussian process
is structured as an additive model [17, 28], e.g., projected-additive functions [36], ensemble Bayesian
optimization (EBO) [61], latent additive structural kernel learning (HDBBO) [65] and group additive
models [28, 36]. However, learning the unknown structure incurs a considerable computational
cost [44], and is not applicable for non-separable functions, for which CobBO can still be applied.

Trust regions and space partitions: Trust region BO has been proven effective for high-dimensional
problems. A typical pattern is to alternate between global and local search regions. In the local
trust regions, many efﬁcient methods have been applied, e.g., local Gaussian models (TurBO [14]),
adaptive search on a mesh grid (BADS [1]) or quasi-Newton local optimization (BLOSSOM [41]).
TurBO [14] uses Thompson sampling to allocate samples across multiple regions. A related method is
to use space partitions, e.g., LA-MCTS[60] on a Monte Carlo tree search algorithm to learn efﬁcient
partitions. CobBO differs by selecting low dimensional subspaces. It can also incorporate trust
regions in the ﬁrst-stage global approximation, as shown in the Appendix.

3

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

3 Algorithm

Without loss of generality, suppose that the goal is to solve a maximization problem x∗ =
argmaxx∈Ωf (x) for a black-box function f : Ω → R. The domain is normalized Ω = [0, 1]D
with the coordinates indexed by I = {1, 2, · · · , D}.

For a sequence of points Xt = {x1, x2, · · · , xt} with t indexing the most recent iteration, we observe
Ht = {(xi, yi = f (xi))}t
i=1. A random subset Ct ⊆ I of the coordinates is selected, forming a
subspace Ωt ⊆ Ω at iteration t. As a variant of coordinate ascent, the subspace Ωt contains a pivot
(cid:1).
f (x) with Mt = f (cid:0)xM
point Vt, which presumably is the maximum point xM
CobBO may set Vt different from xM
to escape local optima. Then, BO is conducted within Ωt while
t
ﬁxing all the other coordinates C c

t = I \ Ct, i.e., the complement of Ct.

t = argmaxx∈Xt

t

Algorithm 1: CobBO(f, τ , T)

1 Hτ ← sample τ initial points and evaluate their values
2 Vτ , Mτ ← Find the tuple with the maximal objective value in Hτ
3 qτ ← 0 Initialize the number of consecutive failed queries
4 πτ ← Initialize a uniform preference distribution on the coordinates
5 for t ← τ to T do
6

if switch Ωt−1 by the backoff stopping rule (Section 3.2) then

Ct ← Sample a promising coordinate block according to πt (Section 3.1)
Ωt ← Take the subspace of Ωt over the coordinate block Ct, such that Vt ∈ Ωt

else

(cid:105)
(cid:104)
Project Xt onto Ωt to obtain a set of virtual points (Eq. 1)
(cid:17) (cid:104)

Smooth function values on ˆXt by interpolation using Ht

(cid:105)

Ωt ← Ωt−1
ˆXt ← PΩt(Xt)
(cid:16) ˆXt, Ht
ˆHt ← R
(cid:104) ˆfΩt(x)| ˆHt
(cid:105)
p
conditional on ˆHt
Q ˆf ∼p( ˆf | ˆHt)(x| ˆHt)
xt+1 ← argmaxx∈Ωt
yt+1 ← Evaluate the black-box function yt+1 = f (xt+1)
if yt+1 > Mt then

(cid:104)

← Compute the posterior distribution of the Gaussian process in Ωt

Suggest the next query in Ωt (Section 3)

(cid:105)

Vt+1 ← xt+1, Mt+1 ← yt+1, qt+1 ← 0

else

Vt+1 ← Vt, Mt+1 ← Mt, qt+1 ← qt + 1

πt+1 ← Update πt by a multiplicative weights update method (Eq. 2)
Ht+1 ← Ht

(cid:83){(xt+1, yt+1)}, Xt+1 ← Xt

(cid:83){xt+1}

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
22 end

For BO in Ωt, we use Gaussian processes as the random surrogates ˆf = ˆfΩt(x) to describe the
Bayesian statistics of f (x) for x ∈ Ωt. At each iteration, the next query point is generated by solving

xt+1 = argmaxx∈Ωt,Vt∈Ωt

Q ˆfΩt (x)∼p( ˆf |Ht)(x|Ht),

where the acquisition function Q(x|Ht) incorporates the posterior distribution of the Gaussian
processes p( ˆf |Ht). Typical acquisition functions include the expected improvement (EI) [42, 27],
the upper conﬁdence bound (UCB) [3, 54, 55], the entropy search [24, 25, 64], and the knowledge
gradient [16, 53, 69].
Instead of directly computing the posterior distribution p( ˆf |Ht), we replace the conditional events
Ht by

i=1
with an interpolation function R(·, ·) and a projection function PΩt(·),

ˆHt := R (PΩt (Xt) , Ht) = {(ˆxi, ˆyi)}t

PΩt(x)(j) =

(cid:40)

x(j)
V (j)
t

if j ∈ Ct
if j /∈ Ct

4

(1)

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

162

163

164

165

at coordinate j. It simply keeps the values of x whose corresponding coordinates are in Ct and
replaces the rest by the corresponding values of Vt, as illustrated in Fig. 2.
Applying PΩt(·) on Xt and discarding duplicates generate a new set of distinct virtual points ˆXt =
{ˆx1, ˆx2, ˆx3, · · · , ˆxˆt}, ˆxi ∈ Ωt ∀ 1 ≤ i ≤ ˆt ≤ t. The function values at ˆxi ∈ ˆXt are interpolated as
ˆyi = R(ˆxi, Ht) using the standard radial basis function [6, 7] and the observed points in Ht. It not
only signiﬁcantly reduces the GP regression time due to the efﬁciency of RBF [6] and the acquisition
function optimization in low dimensions [11], but also eventually improves the model accuracy using
the more sophisticated kernel applied on Ωt.
Note that only a fraction of the points in ˆXt ∩ Xt
directly observe the exact function values. The
function values on the rest ones in ˆXt\Xt are
estimated by interpolation, which captures the
landscape of f (x) by smoothing out the local
ﬂuctuations. To control the trade-off between
the inaccurate estimations and the exact obser-
vations in Ωt, we design a stopping rule that
optimizes the number of consistent queries in
Ωt. The more consistent queries conducted in a
given subspace, the more accurate observations
could be obtained, albeit at the expense of a smaller remaining budget for exploring other regions.

Figure 2: Two-stage kernels: subspace projection
and function value interpolation

The key features of CobBO are listed in Algorithm 1, with more details in the following sections.
Several auxiliary components are utilized and presented in Appendix C to deal with a larger variety
of problems and corner cases.

166

3.1 Block coordinate ascent and subspace selection

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

For Bayesian optimization, consider an infeasible assumption that each iteration can exactly maximize
the function f (x) in Ωt. This is not possible for one iteration but only if one can consistently query
in Ωt, since the points converge to the maximum, e.g., under the expected improvement acquisition
function with ﬁxed priors [59] and the convergence rate can be characterized for smooth functions
in the reproducing kernel Hilbert space [8]. However, even with this infeasible assumption, it is
known that coordinate ascent with ﬁxed blocks can cause stagnation at a non-critical point, e.g., for
non-differentiable [67] or non-convex functions [49]. This motivates us to select a subspace with a
variable-size coordinate block Ct for each query. A good coordinate block can help the iterations
to escape the trapped non-critical points. For example, one condition can be based on the result
in [21] that assumes f (x) to be differentiable and strictly quasi-convex over a collection of blocks. In
practice, we do not restrict ourselves to these assumptions.

We induce a preference distribution πt over the coordinate set I, and sample a variable-size coordinate
block Ct accordingly. This distribution is updated at iteration t through a multiplicative weights
update method [2]. Speciﬁcally, the values of πt at coordinates in Ct starts off uniform and increase
in face of an improvement or decrease otherwise according to different multiplicative ratios α > 1
and β > 1, respectively,

wt,j = wt−1,j ·






if j ∈ Ct and yt > Mt−1
α
1/β if j ∈ Ct and yt ≤ Mt−1
1

if j /∈ Ct

; w0,j =

1
D

;

πt,j =

wt,j
j=1 wt,j

(cid:80)D

(2)

This update characterizes how likely a coordinate block can generate a promising search subspace.
The multiplicative ratio α is chosen to be relatively large, e.g., α = 2.0, and β relatively small, e.g.,
β = 1.1, since the queries that improve the best observations yt > Mt−1 happen more rarely than
the opposite yt ≤ Mt−1.

How to dynamically select the size |Ct|? It is known that Bayesian optimization works well for low
dimensions [15]. Thus, we specify an upper bound for the dimension of the subspace (e.g. |Ct| ≤ 30).
In principle, |Ct| can be any random number in a ﬁnite set of possible block sizes C. This is different
from the method that partitions the coordinates into ﬁxed blocks and selects one according to, e.g.,
cyclic order [68], random sampling or Gauss-Southwell [46].

5

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

206

207

3.2 Backoff stopping rule for consistent queries

Applying BO on Ωt requires a strategy to determine the number of consecutive queries for making a
sufﬁcient progress. This strategy is based on previous observations, thus forming a stopping rule. In
principle, there are two different scenarios, exemplifying exploration and exploitation, respectively.
Persistently querying a given subspace refrains from opportunistically exploring other coordinate
combinations. Abruptly shifting to different subspaces does not fully exploit the potential of a given
subspace.

CobBO designs a heuristic stopping rule in compromise. It takes the above two scenarios into
joint consideration, by considering not only the number of consecutive queries that fail to improve
the objective function but also other factors including the improved difference Mt − Mt−1, the
point distance ||xt − xt−1||, the query budget T and the problem dimension D. On the one hand,
switching to another subspace Ωt+1 ((cid:54)= Ωt) prematurely without fully exploiting Ωt incurs an
additional approximation error associated with the interpolation of observations in Ωt projected to
Ωt+1. On the other hand, it is also possible to over-exploit a subspace, spending high query budget
on marginal improvements around local optima. In order to mitigate this, even when a query leads to
an improvement, other factors are considered for sampling a new subspace.

208

3.3 Theoretical Analysis

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

One can view our block coordinate selection approach in section 3.1 as a combinatorial mixture of
experts problem [10], where each coordinate is a single expert and the forecaster aims at choosing
the best combination of experts in each step. Under this view, we bound the regret of our selection
method with respect to the policy of selecting the best (unknown) block of coordinates at each step.
Assume that there is a ﬁxed optimal choice I ∗ for the block of coordinates to pick at all steps. This
block is characterized by improving the objective function for the largest number of times among
all the possible coordinate blocks when performing Bayesian optimization over the corresponding
subspaces. The following particular design of losses expresses this cause:

(cid:96)t,i =






− log(˜α)
log( ˜β)
0

if i ∈ Ct and yt > Mt−1
if i ∈ Ct and yt ≤ Mt−1
if i /∈ Ct

;

˜α, ˜β > 1

(3)

as all the coordinates participating in the selected block incur the same loss that effectively rewards
these coordinates for improving the objective and penalizes these for failing to improve the objective.
All other coordinates that are not selected receive a zero loss and remain untouched.
Note that ˜α and ˜β express the extent of reward and penalty, e.g. for ˜α = ˜β = e we have losses of
(cid:96)t,i ∈ {−1, 1, 0}. Yet, ˜α is better chosen to be larger than ˜β, since the frequency of improving the
objective is expected to be smaller.

The loss received by the forecaster is to reﬂect the same motivation. This is done by averaging
the losses of the individual coordinates in the selected block, so that the size of the block does not
matter explicitly, i.e. a bigger block should not incur more loss just due to its size but only due to its
performance. Such that for each coordinate block It ⊂ I = {1, · · · , D} selected at time step t, the
loss incurred by the forecaster is Lt,It = 1
(cid:96)t,i. This is also the common loss incurred by
i∈It
|It|
all the coordinates participating in that block.

(cid:80)

In each step we have the following multiplicative update rule of the weights associated with each
coordinate (setting α = ˜αη and β = ˜βη yields the update rule in Eq. 2):

wt,i = wt−1,i · e−η(cid:96)t,i = wt−1,i ·




˜αη
1/ ˜βη

1

if i ∈ Ct and yt > Mt−1
if i ∈ Ct and yt ≤ Mt−1
if i /∈ Ct

(4)

The probability ˜πt,It of selecting a certain coordinate block It is induced by πt as speciﬁed next.
Thus the expected cumulative loss of the foreceaster is:

LT =

T
(cid:88)

(cid:88)

(cid:88)

t=1

c∈C

It∈Sc

˜πt,It ·

1
|It|

(cid:88)

i∈It

(cid:96)t,i

6

Assume the best coordinate block is I ∗, then the corresponding cumulative loss is:

L∗

T =

T
(cid:88)

t=1

Lt,I∗ =

T
(cid:88)

t=1

1
|I ∗|

(cid:88)

i∈I∗

(cid:96)t,i

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

256

257

We hence aim at bounding the regret RegretT = LT − L∗
T .
Theorem 1. Sample from the combinatorial space of all possible coordinate blocks It ∈ (cid:83)
with probability ˜πt,It = (cid:81)
α = ˜αη, β = ˜βη and η = log(˜α ˜β)−1(cid:112)T −1|C|D log(D) yields:

c∈C Sc
j∈ ˆI ˜wt, ˆI. Then the update rule in Eq. 2 with

˜wt,It/(cid:80)

ˆI∈Sc

i∈It

c∈C

(cid:80)

(cid:81)

Regrett ≤ O

(cid:16)

(cid:17)
(log(˜α ˜β) · (cid:112)T |C|D log(D))

(5)

w

i∈It

1/|It|
t,i

c∈C Sc ≤ D|C|D in our combinatorial setup, as typically |C| (cid:28) D.

where ˜wt,It = (cid:81)
is the geometric mean of weights in block It. The upper bound in Eq. 5
is tight, as the lower bound can be shown to be of Ω((cid:112)T log(N )) [23] where the number of experts
is N = (cid:80)
In practice, the direct sampling policy introduced in Theorem 1 involves high computational costs due
to the exponential growth of combinations in D. Thus CobBO suggests an alternative computationally
efﬁcient sampling policy with a linear growth in D.
Theorem 2. Sample a block size c ∈ C with probability pc and c coordinates without replacement
according to πt. Assume C ⊃ {1}, then the update rule in Eq. 2, with α = ˜αη, β = ˜βη and
η =

≥ 1 yields:

(cid:113)

log(D)
T (log( ˜α ˜β)2−log(p1))

(cid:18)(cid:113)

Regrett ≤ O

(log(˜α ˜β)2 − log(p1)) · (cid:112)T log(D))

(cid:19)

(6)

where pc > 0 for all c ∈ C and (cid:80)
c∈C pc = 1, e.g., uniformly set pc ≡ |C|−1.The proof and detailed
sampling policy are in Appendix A. The regret upper bound in Eq. 6 is tight, as the lower bound for
an easier setup can be shown to be of Ω((cid:112)T log(D)) [23]. The implication on η is valid only for
settings of a very high dimensionality and low query budget. In particular, CobBO is designed for
this kind of problems.

Remark: Similar analysis and results follow when incorporating consistent queries from Section 3.2
and sampling a new coordinate block once every several steps. This is done by effectively performing
less steps of aggregated temporal losses, as shown in Appendix A.

4 Numerical Experiments

This section presents detailed ablation studies of the key components presented in Section 3 and
comparisons with other algorithms.

4.1 Empirical analysis and ablation study

Ablation studies are designed to study the contributions of the key components in Algorithm 1 by
experimenting with the Rastrigin function on [−5, 10]50 with 20 initial points. The best performing
run out of 5 experiments for each conﬁguration is presented in Figure 3.

258

Figure 3: Ablation study using Rastrigin on [−5, 10]50 with 20 initial random samples

7

(a)(b)(c)259

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

RBF interpolation: RBF calculation is time efﬁcient. Speciﬁcally, this is much beneﬁcial in high
dimensions. Figure 1 (left) shows the computation time of plain Bayesian optimization compared to
CobBO’s. While the former applies GP regression using the Matérn kernel in the high dimensional
space directly, the later applies RBF interpolation in the high dimensional space and GP regression
with the Matérn kernel in the low dimensional subspace. This two-step composite kernel leads to a
signiﬁcant speed-up. Other time efﬁcient alternatives are, e.g., the inverse distance weighting [26]
and the simple approach of assigning the value of the observed nearest neighbour. Figure 3 (a) shows
that RBF is the most favorable.

Backoff stopping rule: CobBO applies a stopping rule to query a variable number of points in
subspace Ωt (Section 3.2). To validate its effectiveness, we compare it with schemes that use a ﬁxed
budget of queries for Ωt. Figure 3 (b) shows that the stopping rule yields superior results.
Coordinate blocks of a varying size: CobBO selects a block of coordinates of a varying size Ct
(Section 3.1). Figure 3 (c) shows that a varying size is better than ﬁxed.

Preference probability over coordinates: For
demonstrating the effectiveness of coordinate
selection (Section 3.1), we artiﬁcially let the
function value only depend on the ﬁrst 25 coor-
dinates of its input and ignore the rest. It forms
two separate sets of active and inactive coordi-
nates, respectively. We expect CobBO to refrain
from selecting inactive coordinates. Figure 4
shows the entropy of this preference probability
πt over coordinates and the overall probability
for picking active and inactive coordinate at each
iteration. We see that the entropy decreases, as
the preference distribution concentrates on the
signiﬁcant active coordinates.

Figure 4: The preference probability focuses on
active coordinates as the entropy decreases

Figure 5: Performance over low (left) medium (middle) and high (right) dimensional problems

4.2 Comparisons with other methods

The default conﬁguration for CobBO is speciﬁed in the supplementary materials. CobBO performs
on par or outperforms a collection of state-of-the-art methods across the following experiments. Most
of the experiments are conducted using the same settings as in TurBO [14], where it is compared
with a comprehensive list of baselines, including BFGS, BOCK [47], BOHAMIANN, CMA-ES [22],
BOBYQA, EBO [61], GP-TS, HeSBO [44], Nelder-Mead and random search. To avoid repetitions,
we only show TuRBO and CMA-ES that achieve the best performance among this list, and additionally
compare CobBO with BADS [1], REMBO [63], Differetial Evolution (Diff-Evo) [56], Tree Parzen
Estimator (TPE) [4] and Adaptive TPE (ATPE) [12].

8

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

4.2.1 Low dimensional tests

To evaluate CobBO on low dimensional problems, we use the lunar landing [38, 14] and robot
pushing [62], by following the setup in [14]. Conﬁdence intervals (95%) over 30 independent
experiments for each problem are shown in Fig. 5.

Lunar landing (maximization): This controller learning problem (12 dimensions) is provided by
the OpenAI gym [38] and evaluated in [14]. Each algorithm has 50 initial points and a budget of
1, 500 trials. TuRBO is conﬁgured with 5 trust regions and a batch size of 50 as in [14]. Fig. 5 (upper
left) shows that, among the 30 independent tests, CobBO quickly exceeds 300 along some good
sample paths.

Robot pushing (maximization): This control problem (14 dimensions) is introduced in [62] and
extensively tested in [14]. We follow the setting in [14], where TuRBO is conﬁgured with a batch
size of 50 and 15 trust regions, each of which has 30 initial points. Each experiment has a budget of
10, 000 evaluations. On average CobBO exceeds 10 within 5500 trials, as shown in Fig. 5 (lower
left).

4.2.2 High dimensional tests
Since the duration of each experiment in this section is long, conﬁdence intervals (95%) over repeated
10 independent experiments for each problem are presented.

Additive latent structure (minimization): As mentioned in Section 2, additive latent structures
have been exploited in high dimensions. We construct an additive function of 56 dimensions, deﬁned
as f56(x) = Ackley(x1) + Levy(x2) + Rastrigin(x3) + Hartmann(x4) + Rosenbrock(x5) +
Schwefel(x6), where the ﬁrst three terms express the exact functions and domains described in
Section 4.2.1, the Hartmann function on [0, 1]6 and the Rosenbrock and Schwefel functions on
[−5, 10]10 and [−500, 500]10, respectively.

We compare CobBO with TPE, BADS, CMA-ES and TuRBO, each with 100 initial points. Speciﬁ-
cally, TuRBO is conﬁgured with 15 trust regions and a batch size 100. ATPE is excluded as it takes
more than 24 hours per run to ﬁnish. The results are shown in Fig. 5 (upper middle), where CobBO
quickly ﬁnds the best solution among the algorithms tested.

Rover trajectory planning (maximization): This problem (60 dimensions) is introduced in [62].
The objective is to ﬁnd a collision-avoiding trajectory of a sequence consisting of 30 positions in a
2-D plane. We compare CobBO with TuRBO, TPE and CMA-ES, each with a budget of 20, 000
evaluations and 200 initial points. TuRBO is conﬁgured with 15 trust regions and a batch size of
100, as in [14]. ATPE, BADS and REMBO are excluded for this problem and the following ones,
as they all take more than 24 hours per run. Fig. 5 (lower middle) shows that CobBO has a good
performance.

The 200-dimensional Levy and Ackley functions (minimization): We minimize the Levy and
Ackley functions over [−5, 10]200 with 500 initial points. TuRBO is conﬁgured with 15 trust regions
and a batch size of 100. These two problems are challenging and have no redundant dimensions. For
Levy, in Fig. 5 (upper right), CobBO reaches 100.0 within 2, 000 trials, while CMA-ES and TuRBO
obtain 200.0 after 8, 000 trials. TPE cannot ﬁnd a comparable solution within 10, 000 trials in this
case. For Ackley, in Fig. 5 (lower right), CobBO reaches the best solution among all of the algorithms
tested.

Regarding running times, for Ackley, CobBO runs for 12.8 CPU hours and TuRBO-1 run for more
than 80 CPU hours or 9.6 GPU hours. Most other methods either cannot make any progress or ﬁnd
far worse solutions.

5 Conclusion

CobBO is a variant of coordinate ascent tailored for Bayesian optimization with a stopping rule to
switch coordinate subspaces. The sampling policy of subspaces is proven to have tight regret bounds
with respect to the best subspace in hindsight. Combining this projection on random subspaces with a
two-stage kernels for function value interpolation and GP regression, we provide a practical Bayesian
optimization method of affordable computational costs in high dimensions. Empirically, CobBO
consistently ﬁnds comparable or better solutions with reduced trial complexity in comparison with
the state-of-the-art methods across a variety of benchmarks.

9

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

References

[1] Luigi Acerbi and Wei Ji Ma. Practical bayesian optimization for model ﬁtting with bayesian
adaptive direct search. In Proceedings of the 31st International Conference on Neural Infor-
mation Processing Systems, NIPS’17, page 1834–1844, Red Hook, NY, USA, 2017. Curran
Associates Inc.

[2] Sanjeev Arora, Elad Hazan, and Satyen Kale. The multiplicative weights update method: a

meta-algorithm and applications. Theory of Computing, 8(6):121–164, 2012.

[3] Peter Auer. Using conﬁdence bounds for exploitation-exploration trade-offs. J. Mach. Learn.

Res., 3(null):397–422, Mar. 2003.

[4] James S. Bergstra, Rémi Bardenet, Yoshua Bengio, and Balázs Kégl. Algorithms for hyper-
parameter optimization. In J. Shawe-Taylor, R. S. Zemel, P. L. Bartlett, F. Pereira, and K. Q.
Weinberger, editors, Advances in Neural Information Processing Systems 24, pages 2546–2554.
Curran Associates, Inc., 2011.

[5] Mickaël Binois, David Ginsbourger, and Olivier Roustant. On the choice of the low-dimensional
domain for global optimization via random embeddings. Journal of Global Optimization,
76(1):69–90, January 2020.

[6] Martin D. Buhmann. Radial Basis Functions: Theory and Implementations. Cambridge
Monographs on Applied and Computational Mathematics. Cambridge University Press, 2003.
[7] Martin D. Buhmann and M. D. Buhmann. Radial Basis Functions. Cambridge University Press,

USA, 2003.

[8] Adam D. Bull. Convergence rates of efﬁcient global optimization algorithms. Journal of

Machine Learning Research, 12(88):2879–2904, 2011.

[9] Roberto Calandra, André Seyfarth, Jan Peters, and Marc Peter Deisenroth. Bayesian optimiza-
tion for learning gaits under uncertainty. Annals of Mathematics and Artiﬁcial Intelligence,
76(1):5–23, 2016.

[10] Nicolo Cesa-Bianchi and Gábor Lugosi. Prediction, learning, and games. Cambridge university

press, 2006.

[11] Josip Djolonga, Andreas Krause, and Volkan Cevher. High-dimensional gaussian process
bandits. In C. J. C. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Q. Weinberger,
editors, Advances in Neural Information Processing Systems 26, pages 1025–1033. Curran
Associates, Inc., 2013.

[12] ElectricBrain. Blog: Learning to optimize, 2018.
[13] David Eriksson, Kun Dong, Eric Lee, David Bindel, and Andrew G Wilson. Scaling gaussian
process regression with derivatives. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N.
Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems 31,
pages 6867–6877. Curran Associates, Inc., 2018.

[14] David Eriksson, Michael Pearce, Jacob Gardner, Ryan D Turner, and Matthias Poloczek.
Scalable global optimization via local bayesian optimization. In Advances in Neural Information
Processing Systems 32, pages 5496–5507. Curran Associates, Inc., 2019.

[15] Peter I. Frazier. A tutorial on bayesian optimization, 2018.
[16] Peter I. Frazier, Warren B. Powell, and Savas Dayanik. A knowledge-gradient policy for
sequential information collection. SIAM J. Control Optim., 47(5):2410–2439, Sept. 2008.
[17] Elad Gilboa, Yunus Saatçi, and John P. Cunningham. Scaling multidimensional Gaussian
processes using projected additive approximations. In Proceedings of the 30th International
Conference on International Conference on Machine Learning - Volume 28, ICML’13, page
I–454–I–461. JMLR.org, 2013.

[18] Daniel Golovin, Benjamin Solnik, Subhodeep Moitra, Greg Kochanski, John Karro, and D
Sculley. Google vizier: A service for black-box optimization. In Proceedings of the 23rd ACM
SIGKDD international conference on knowledge discovery and data mining, pages 1487–1495,
2017.

[19] Rafael Gómez-Bombarelli, Jennifer N Wei, David Duvenaud, José Miguel Hernández-Lobato,
Benjamín Sánchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D Hirzel,
Ryan P Adams, and Alán Aspuru-Guzik. Automatic chemical design using a data-driven
continuous representation of molecules. ACS central science, 4(2):268–276, 2018.

[20] Javier Gonzalez, Zhenwen Dai, Philipp Hennig, and Neil Lawrence. Batch bayesian optimization
via local penalization. In Arthur Gretton and Christian C. Robert, editors, Proceedings of the

10

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

19th International Conference on Artiﬁcial Intelligence and Statistics, volume 51 of Proceedings
of Machine Learning Research, pages 648–657, Cadiz, Spain, 09–11 May 2016. PMLR.
[21] Luigi Grippo and Marco Sciandrone. On the convergence of the block nonlinear gauss-seidel
method under convex constraints. Operations Research Letters, 26(3):127–136, 2000.

[22] Nikolaus Hansen and Andreas Ostermeier. Completely derandomized self-adaptation in evolu-

tion strategies. Evolutionary Computation, 9(2):159–195, June 2001.

[23] David Haussler, Jyrki Kivinen, and Manfred K Warmuth. Tight worst-case loss bounds for
predicting with expert advice. In European Conference on Computational Learning Theory,
pages 69–83. Springer, 1995.

[24] Philipp Hennig and Christian J. Schuler. Entropy search for information-efﬁcient global

optimization. J. Mach. Learn. Res., 13(1):1809–1837, June 2012.

[25] José Miguel Henrández-Lobato, Matthew W. Hoffman, and Zoubin Ghahramani. Predictive
entropy search for efﬁcient global optimization of black-box functions. In Proceedings of the
27th International Conference on Neural Information Processing Systems - Volume 1, NIPS’14,
page 918–926, Cambridge, MA, USA, 2014. MIT Press.

[26] IDW. https://en.wikipedia.org/wiki/Inverse_distance_weighting.
[27] Donald R. Jones, Matthias Schonlau, and William J. Welch. Efﬁcient global optimization of

expensive black-box functions. Journal of Global optimization, 13(4):455–492, 1998.

[28] Kirthevasan Kandasamy, Jeff Schneider, and Barnabás Póczos. High dimensional bayesian
optimization and bandits via additive models. In Proceedings of the 32nd International Confer-
ence on International Conference on Machine Learning - Volume 37, ICML’15, page 295–304.
JMLR.org, 2015.

[29] Johannes Kirschner, Mojmir Mutny, Nicole Hiller, Rasmus Ischebeck, and Andreas Krause.
Adaptive and safe Bayesian optimization in high dimensions via one-dimensional subspaces. In
Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International
Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research,
pages 3429–3438, Long Beach, California, USA, 09–15 Jun 2019. PMLR.

[30] H. J. Kushner. A new method of locating the maximum point of an arbitrary multipeak curve in

the presence of noise. Journal of Basic Engineering, 86(1):97–106, mar 1964.

[31] Rémi Lam, Matthias Poloczek, Peter Frazier, and Karen E Willcox. Advances in bayesian
optimization with applications in aerospace engineering. In 2018 AIAA Non-Deterministic
Approaches Conference, page 1656, 2018.

[32] Ben Letham, Roberto Calandra, Akshara Rai, and Eytan Bakshy. Re-examining linear embed-
dings for high-dimensional bayesian optimization. Advances in Neural Information Processing
Systems, 33, 2020.

[33] Ben Letham, Roberto Calandra, Akshara Rai, and Eytan Bakshy. Re-examining linear embed-
dings for high-dimensional bayesian optimization. In H. Larochelle, M. Ranzato, R. Hadsell,
M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, vol-
ume 33, pages 1546–1558. Curran Associates, Inc., 2020.

[34] Benjamin Letham, Brian Karrer, Guilherme Ottoni, Eytan Bakshy, et al. Constrained bayesian

optimization with noisy experiments. Bayesian Analysis, 14(2):495–519, 2019.

[35] Cheng Li, Sunil Gupta, Santu Rana, Vu Nguyen, Svetha Venkatesh, and Alistair Shilton.
High dimensional bayesian optimization using dropout. In Proceedings of the Twenty-Sixth
International Joint Conference on Artiﬁcial Intelligence, IJCAI-17, pages 2096–2102, 2017.

[36] Chun-Liang Li, Kirthevasan Kandasamy, Barnabas Poczos, and Jeff Schneider. High dimen-
sional bayesian optimization via restricted projection pursuit models. In Arthur Gretton and
Christian C. Robert, editors, Proceedings of the 19th International Conference on Artiﬁcial
Intelligence and Statistics, volume 51 of Proceedings of Machine Learning Research, pages
884–892, Cadiz, Spain, 09–11 May 2016. PMLR.

[37] Daniel J Lizotte, Tao Wang, Michael H Bowling, and Dale Schuurmans. Automatic gait
optimization with gaussian process regression. In IJCAI, volume 7, pages 944–949, 2007.

[38] LunarLander v2. https://gym.openai.com/envs/LunarLander-v2/.
[39] Horia Mania, Aurelia Guy, and Benjamin Recht. Simple random search of static linear policies
is competitive for reinforcement learning. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman,
N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems
31, pages 1800–1809. Curran Associates, Inc., 2018.

[40] Matern kernel.

https://scikit-learn.org/stable/modules/generated/sklearn.

gaussian_process.kernels.Matern.html.

11

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

[41] Mark McLeod, Michael A. Osborne, and Stephen J. Roberts. Optimization, fast and slow:

Optimally switching between local and bayesian optimization. In ICML, 2018.

[42] J. Moˇckus. On bayesian methods for seeking the extremum. In G. I. Marchuk, editor, Optimiza-
tion Techniques IFIP Technical Conference Novosibirsk, July 1–7, 1974, pages 400–404, Berlin,
Heidelberg, 1975. Springer Berlin Heidelberg.

[43] Riccardo Moriconi, K. S. Sesh Kumar, and Marc Peter Deisenroth. High-dimensional bayesian
optimization with projections using quantile gaussian processes. Optimization Letters, 14:51–64,
2020.

[44] Alexander Munteanu, Amin Nayebi, and Matthias Poloczek. A framework for Bayesian
In Kamalika Chaudhuri and Ruslan Salakhutdinov,
optimization in embedded subspaces.
editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of
Proceedings of Machine Learning Research, pages 4752–4761, Long Beach, California, USA,
09–15 Jun 2019. PMLR.

[45] Mojmir Mutny and Andreas Krause. Efﬁcient high dimensional bayesian optimization with
additivity and quadrature fourier features. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman,
N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems
31, pages 9005–9016. Curran Associates, Inc., 2018.

[46] Julie Nutini, Mark Schmidt, Issam H. Laradji, Michael Friedlander, and Hoyt Koepke. Coordi-
nate descent converges faster with the gauss-southwell rule than random selection. ICML’15:
Proceedings of the 32nd International Conference on International Conference on Machine
Learning, 37, July 2015.

[47] ChangYong Oh, Efstratios Gavves, and Max Welling. BOCK : Bayesian optimization with
In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th
cylindrical kernels.
International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning
Research, pages 3868–3877, Stockholm, Sweden, 10–15 Jul 2018. PMLR.

[48] Rafael Oliveira, Fernando Rocha, Lionel Ott, Vitor Guizilini, Fabio Ramos, and Valdir Jr.
Learning to race through coordinate descent bayesian optimisation. In IEEE International
Conference on Robotics and Automation (ICRA), February 2018.

[49] M.J.D. Powell. On Search Directions for Minimization Algorithms. AERE-TP. AERE, Theoret-

ical Physics Division, 1972.

[50] Radial basis function. https://docs.scipy.org/doc/scipy/reference/generated/

scipy.interpolate.Rbf.html.

[51] Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. In J. C. Platt,
D. Koller, Y. Singer, and S. T. Roweis, editors, Advances in Neural Information Processing
Systems 20, pages 1177–1184. Curran Associates, Inc., 2008.

[52] Akshara Rai, Rika Antonova, Seungmoon Song, William Martin, Hartmut Geyer, and Christo-
pher Atkeson. Bayesian optimization using domain knowledge on the atrias biped. In 2018
IEEE International Conference on Robotics and Automation (ICRA), pages 1771–1778. IEEE,
2018.

[53] Warren Scott, Peter Frazier, and Warren Powell. The correlated knowledge gradient for
simulation optimization of continuous parameters using gaussian process regression. SIAM
Journal on Optimization, 21(3):996–1026, 2011.

[54] Niranjan Srinivas, Andreas Krause, Sham Kakade, and Matthias Seeger. Gaussian process
optimization in the bandit setting: No regret and experimental design. In Proceedings of the
27th International Conference on International Conference on Machine Learning, ICML’10,
page 1015–1022, Madison, WI, USA, 2010.

[55] N. Srinivas, A. Krause, S. M. Kakade, and M. W. Seeger. Information-theoretic regret bounds
for gaussian process optimization in the bandit setting. IEEE Transactions on Information
Theory, 58(5):3250–3265, 2012.

[56] Rainer Storn and Kenneth Price. Differential evolution–a simple and efﬁcient heuristic for
global optimization over continuous spaces. Journal of global optimization, 11(4):341–359,
1997.

[57] Sonja Surjanovic and Derek Bingham. Optimization test problems, 2013.
[58] Hemant Tyagi and Volkan Cevher. Learning non-parametric basis independent models from
point queries via low-rank methods. Applied and Computational Harmonic Analysis, 37(3):389
– 412, 2014.

12

517

518

519

520

521

522

523

524

525

526

527

528

529

530

531

532

533

534

535

536

537

538

539

540

541

542

543

544

545

546

547

548

549

550

551

552

553

554

555

556

557

558

559

560

561

562

563

564

565

566

567

568

[59] Emmanuel Vazquez and Julien Bect. Convergence properties of the expected improvement
algorithm with ﬁxed mean and covariance functions. Journal of Statistical Planning and
Inference, 140(11):3088 – 3095, 2010.

[60] Linnan Wang, Rodrigo Fonseca, and Yuandong Tian. Learning search space partition for

black-box optimization using monte carlo tree search. ArXiv, abs/2007.00708, 2020.

[61] Zi Wang, Clement Gehring, Pushmeet Kohli, and Stefanie Jegelka. Batched large-scale bayesian
optimization in high-dimensional spaces. In International Conference on Artiﬁcial Intelligence
and Statistics (AISTATS), 2018.

[62] Zi Wang, Clement Gehring, Pushmeet Kohli, and Stefanie Jegelka. Batched large-scale bayesian
optimization in high-dimensional spaces. In International Conference on Artiﬁcial Intelligence
and Statistics, pages 745–754, 2018.

[63] Ziyu Wang, Frank Hutter, Masrour Zoghi, David Matheson, and Nando De Freitas. Bayesian
optimization in a billion dimensions via random embeddings. J. Artif. Int. Res., 55(1):361–387,
Jan. 2016.

[64] Zi Wang and Stefanie Jegelka. Max-value entropy search for efﬁcient Bayesian optimization. In
Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on
Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 3627–3635,
International Convention Centre, Sydney, Australia, 06–11 Aug 2017. PMLR.

[65] Zi Wang, Chengtao Li, Stefanie Jegelka, and Pushmeet Kohli. Batched high-dimensional
bayesian optimization via structural kernel learning. In Proceedings of the 34th International
Conference on Machine Learning - Volume 70, ICML’17, page 3656–3664. JMLR.org, 2017.
[66] Ziyu Wang, Masrour Zoghi, Frank Hutter, David Matheson, and Nando De Freitas. Bayesian
optimization in high dimensions via random embeddings. In Proceedings of the Twenty-Third
International Joint Conference on Artiﬁcial Intelligence, IJCAI ’13, page 1778–1784. AAAI
Press, 2013.

[67] J. Warga. Minimizing certain convex functions. Journal of the Society for Industrial and

Applied Mathematics, 11(3):588–593, 1963.

[68] Stephen J. Wright. Coordinate descent algorithms. Mathematical Programming: Series A and

B, June 2015.

[69] Jian Wu and Peter I. Frazier. The parallel knowledge gradient method for batch bayesian
optimization. In Proceedings of the 30th International Conference on Neural Information
Processing Systems, NIPS’16, page 3134–3142, Red Hook, NY, USA, 2016. Curran Associates
Inc.

[70] Miao Zhang, Huiqi Li, and Steven Su. High dimensional bayesian optimization via supervised
dimension reduction. In Proceedings of the Twenty-Eighth International Joint Conference on
Artiﬁcial Intelligence, IJCAI-19, pages 4292–4298. International Joint Conferences on Artiﬁcial
Intelligence Organization, 7 2019.

[71] Yichi Zhang, Daniel W Apley, and Wei Chen. Bayesian optimization for materials design with

mixed quantitative and qualitative variables. Scientiﬁc reports, 10(1):1–13, 2020.

Broader Impact

As stated in [32], Bayesian optimization is a powerful optimization technique used in a wide range of
industries and applications, such as robotics [37, 9, 52], internet tech companies [18, 34], designing
novel molecules for pharmaceutics [19], material design for increasing efﬁciency of solar cells [71],
and aerospace engineering [31]. All of these settings have high-dimensional optimization problems,
and advances in BO will reﬂect on improved capabilities on these ﬁelds as well. We have fully open-
sourced our code for CobBO using the MIT license to be available for researchers and practitioners
in these ﬁelds, and many others. The ability to optimize a larger number of parameters than has
previously been possible will bring further improvements to and further accelerate work in these
areas.

Checklist

The checklist follows the references. Please read the checklist guidelines carefully for information on
how to answer these questions. For each question, change the default [TODO] to [Yes] , [No] , or

13

569

570

571

572

573

574

575

576

577

578

579

580

581

582

583

584

585

586

587

588

589

590

591

592

593

594

595

596

597

598

599

600

601

602

603

604

605

606

607

608

609

610

611

612

613

614

615

[N/A] . You are strongly encouraged to include a justiﬁcation to your answer, either by referencing
the appropriate section of your paper or providing a brief inline description. For example:

• Did you include the license to the code and datasets? [Yes] See Section ??.
• Did you include the license to the code and datasets? [No] The code and the data are

proprietary.

• Did you include the license to the code and datasets? [N/A]

Please do not modify the questions and only use the provided macros for your answers. Note that the
Checklist section does not count towards the page limit. In your paper, please delete this instructions
block and only keep the Checklist section heading above along with the questions/answers below.

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reﬂect the paper’s

contributions and scope? [Yes] See Section 3.
(b) Did you describe the limitations of your work? [No]
(c) Did you discuss any potential negative societal impacts of your work? [Yes] See

Broader Impact.

(d) Have you read the ethics review guidelines and ensured that your paper conforms to

them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [Yes] See Section 3.3.
(b) Did you include complete proofs of all theoretical results? [Yes] See the Appendix A.

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main ex-
perimental results (either in the supplemental material or as a URL)? [Yes] In the
supplemental material.

(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they
were chosen)? [Yes] See Table 2 in the appendix, which contains the default hyperpa-
rameters.

(c) Did you report error bars (e.g., with respect to the random seed after running experi-

ments multiple times)? [Yes] See Fig.5 and Fig.8-10.

(d) Did you include the total amount of compute and the type of resources used (e.g., type

of GPUs, internal cluster, or cloud provider)? [Yes] See page 9, line 324.

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [N/A]
(b) Did you mention the license of the assets? [Yes] MIT license; see the appendix.
(c) Did you include any new assets either in the supplemental material or as a URL? [No]
(d) Did you discuss whether and how consent was obtained from people whose data you’re

using/curating? [Yes] See Section 4.2.

(e) Did you discuss whether the data you are using/curating contains personally identiﬁable
information or offensive content? [N/A] It does not contain personal identiﬁable
information or offensive content.

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if

applicable? [N/A]

(b) Did you describe any potential participant risks, with links to Institutional Review

Board (IRB) approvals, if applicable? [N/A]

(c) Did you include the estimated hourly wage paid to participants and the total amount

spent on participant compensation? [N/A]

14

