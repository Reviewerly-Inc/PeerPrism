Polynomial Width is Sufficient for Set Representation
with High-dimensional Features

Anonymous Author(s)
Affiliation
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

Set representation has become ubiquitous in deep learning for modeling the induc-
tive bias of neural networks that are insensitive to the input order. DeepSets is the
most widely used neural network architecture for set representation. It involves
embedding each set element into a latent space with dimension L, followed by a
sum pooling to obtain a whole-set embedding, and finally mapping the whole-set
embedding to the output. In this work, we investigate the impact of the dimension
L on the expressive power of DeepSets. Previous analyses either oversimplified
high-dimensional features to be one-dimensional features or were limited to ana-
lytic activations, thereby diverging from practical use or resulting in L that grows
exponentially with the set size N and feature dimension D. To investigate the
minimal value of L that achieves sufficient expressive power, we present two set-
element embedding layers: (a) linear + power activation (LP) and (b) logarithm +
linear + exponential activations (LLE). We demonstrate that L being poly(N, D)
is sufficient for set representation using both embedding layers. We also provide a
lower bound of L for the LP embedding layer. Furthermore, we extend our results
to permutation-equivariant set functions and the complex field.

1

Introduction

Enforcing invariance into neural network architectures has become a widely-used principle to design
deep learning models [1–7]. In particular, when a task is to learn a function with a set as the input, the
architecture enforces permutation invariance that asks the output to be invariant to the permutation
of the input set elements [8, 9]. Neural networks to learn a set function have found a variety of
applications in particle physics [10, 11], computer vision [12, 13] and population statistics [14–16],
and have recently become a fundamental module (the aggregation operation of neighbors’ features in
a graph [17–19]) in graph neural networks (GNNs) [20, 21] that show even broader applications.

Previous works have studied the expressive power of neural network architectures to represent set
functions [8,9,22–26]. Formally, a set with N elements can be represented as S = {x(1), · · · , x(N )}
where x(i) is in a feature space X , typically X = RD. To represent a set function that takes S and
outputs a real value, the most widely used architecture DeepSets [9] follows Eq. (1).

f (S) = ρ

(cid:32) N
(cid:88)

i=1

(cid:33)

ϕ(x(i))

, where ϕ : X → RL and ρ : RL → R are continuous functions.

(1)

DeepSets encodes each set element individually via ϕ, and then maps the encoded vectors after sum
pooling to the output via ρ. The continuity of ϕ and ρ ensure that they can be well approximated
by fully-connected neural networks [27, 28], which has practical implications. DeepSets enforces
permutation invariance because of the sum pooling, as shuffling the order of x(i) does not change

Submitted to 37th Conference on Neural Information Processing Systems (NeurIPS 2023). Do not distribute.

Table 1: A comprehensive comparison among all prior works on expressiveness analysis with L. Our
results achieve the tightest bound on L while being able to analyze high-dimensional set features and
extend to the equivariance case.

Prior Arts
DeepSets [9]
Wagstaff et al. [23]
Segol et al. [25]
Zweig & Bruna [26]
Our results

L
D + 1
D
(cid:1) − 1
(cid:0)N +D
√
N

exp(min{

N , D})

poly(N, D)

✗
✗
✓
✓
✓

✓
✓
✗
✗
✓

✓
✓
✓
✗
✓

D > 1 Exact Rep. Equivariance

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

the output. However, the sum pooling compresses the whole set into an L-dimension vector, which
places an information bottleneck in the middle of the architecture. Therefore, a core question on
using DeepSets for set function representation is that given the input feature dimension D and the
set size N , what the minimal L is needed so that the architecture Eq. (1) can represent/universally
approximate any continuous set functions. The question has attracted attention in many previous
works [9, 23–26] and is the focus of the present work.

An extensive understanding has been achieved for the case with one-dimensional features (D = 1).
Zaheer et al. [9] proved that this architecture with bottleneck dimension L = N suffices to accurately
represent any continuous set functions when D = 1. Later, Wagstaff et al. proved that accurate
representations cannot be achieved when L < N [23] and further strengthened the statement to a
failure in approximation to arbitrary precision in the infinity norm when L < N [24].

However, for the case with high-dimensional features (D > 1), the characterization of the minimal
possible L is still missing. Most of previous works [9, 25, 29] proposed to generate multi-symmetric
polynomials to approximate permutation invariant functions [30]. As the algebraic basis of multi-
symmetric polynomials is of size L∗ = (cid:0)N +D
(cid:1) − 1 [31] (exponential in min{D, N }), these works by
default claim that if L ≥ L∗, f in Eq. 1 can approximate any continuous set functions, while they do
not check the possibility of using a smaller L. Zweig and Bruna [26] constructed a set function that f
requires bottleneck dimension L > N −2 exp(O(min{D,
N })
to approximate while it relies on the condition that ϕ, ρ only adopt analytic activations. This condition
is overly strict, as most of the practical neural networks allow using non-analytic activations, such as
ReLU. Zweig and Bruna thus left an open question whether the exponential dependence on N or D
of L is still necessary if ϕ, ρ allow using non-analytic activations.

N })) (still exponential in min{D,

√

√

N

Present work The main contribution of this work is to confirm a negative response to the above
question. Specfically, we present the first theoretical justification that L being polynomial in N and
D is sufficient for DeepSets (Eq. (1)) like architecture to represent any continuous set functions
with high-dimensional features (D > 1). To mitigate the gap to the practical use, we consider two
architectures to implement feature embedding ϕ (in Eq. 1) and specify the bounds on L accordingly:

• ϕ adopts a linear layer with power mapping: The minimal L holds a lower bound and an upper

bound, which is N (D + 1) ≤ L < N 5D2.

• Constrained on the entry-wise positive input space RN ×D

, ϕ adopts two layers with logarithmic
and exponential activations respectively: The minimal L holds a tighter upper bound L ≤ 2N 2D2.

>0

We prove that if the function ρ could be any continuous function, the above two architectures
reproduce the precise construction of any set functions for high-dimensional features D > 1, akin
to the result in [9] for D = 1. This result contrasts with [25, 26] which only present approximating
representations. If ρ adopts a fully-connected neural network that allows approximation of any
continuous functions on a bounded input space [27, 28], then the DeepSets architecture f (·) can
approximate any set functions universally on that bounded input space. Moreover, our theory can be
easily extended to permutation-equivariant functions and complex set functions, where the minimal
L shares the same bounds up to some multiplicative constants.

Another comment on our contributions is that Zweig and Bruna [26] use difference in the needed
dimension L to illustrate the gap between DeepSets [9] and Relational Network [32] in their expressive
powers, where the latter encodes set elements in a pairwise manner rather than in a separate manner.
The gap well explains the empirical observation that Relational Network achieves better expressive
power with smaller L [23,33]. Our theory does not violate such an observation while it shows that the

2

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

gap can be reduced from an exponential order in N and D to a polynomial order. Moreover, many real-
world applications have computation constraints where only DeepSets instead of Relational Network
can be used, e.g., the neighbor aggregation operation in GNN being applied to large networks [21],
and the hypergraph neural diffusion operation in hypergraph neural networks [7]. Our theory points
out that in this case, it is sufficient to use polynomial L dimension to embed each element, while one
needs to adopt a function ρ with non-analytic activitions.

2 Preliminaries

2.1 Notations and Problem Setup

We are interested in the approximation and representation of functions defined over sets 1.
In
convention, an N -sized set S = {x(1), · · · , x(N )}, where x(i) ∈ RD, ∀i ∈ [N ](≜ {1, 2, ..., N }),
can be denoted by a data matrix X = (cid:2)x(1)
∈ RN ×D. Note that we use the
superscript (i) to denote the i-th set element and the subscript i to denote the i-th column/feature

· · · x(N )(cid:3)⊤

(cid:104)

x(1)
i

(cid:105)⊤

x(N )
i

· · ·

channel of X, i.e., xi =
matrices. To characterize the unorderedness of a set, we define an equivalence class over RN ×D:
Definition 2.1 (Equivalence Class). If matrices X, X ′ ∈ RN ×D represent the same set X , then they
are called equivalent up a row permutation, denoted as X ∼ X ′. Or equivalently, X ∼ X ′ if and
only if there exists a matrix P ∈ Π(N ) such that X = P X ′.

. Let Π(N ) denote the set of all N -by-N permutation

Set functions can be in general considered as permutation-invariant or permutation-equivariant
functions, which process the input matrices regardless of the order by which rows are organized. The
formal definitions of permutation-invariant/equivariant functions are presented as below:
Definition 2.2. (Permutation Invariance) A function f : RN ×D → RD′
invariant if f (P X) = f (X) for any P ∈ Π(N ).
Definition 2.3. (Permutation Equivariance) A function f : RN ×D → RN ×D′
equivariant if f (P X) = P f (X) for any P ∈ Π(N ).

is called permutation-

is called permutation-

In this paper, we investigate the approach to design a neural network architecture with permutation in-
variance/equivariance. Below we will first focus on permutation-invariant functions f : RN ×D → R.
Then, in Sec. 5, we show that we can easily extend the established results to permutation-equivariant
functions through the results provided in [7, 34] and to the complex field. The obtained results for
fD′]⊤ and
D′ = 1 can also be easily extended to D′ > 1 as otherwise f can be written as [f1
each fi has single output feature channel.

· · ·

2.2 DeepSets and The Difficulty in the High-Dimensional Case D > 1

The seminal work [9] establishes the following result which induces a neural network architecture for
permutation-invariant functions.
Theorem 2.4 (DeepSets [9], D = 1). A continuous function f : RN → R is permutation-invariant
(i.e., a set function) if and only if there exists continuous functions ϕ : R → RL and ρ : RL → R
such that f (X) = ρ

, where L can be as small as N . Note that, here x(i) ∈ R.

(cid:16)(cid:80)N

(cid:17)

i=1 ϕ(x(i))

Remark 2.5. The original result presented in [9] states the latent dimension should be as large as
N + 1. [23] tighten this dimension to exactly N .

Theorem 2.4 implies that as long as the latent space dimension L ≥ N , any permutation-invariant
functions can be implemented by a unified manner as DeepSets (Eq.(1)). Furthermore, DeepSets
suggests a useful architecture for ϕ at the analysis convenience and empirical utility, which is formally
defined below (ϕ = ψL):
Definition 2.6 (Power mapping). A power mapping of degree K is a function ψK : R → RK which
transforms a scalar to a power series: ψK(z) = (cid:2)z

zK(cid:3)⊤

· · ·

z2

.

1In fact, we allow repeating elements in S, therefore, S should be more precisely called multiset. With a

slight abuse of terminology, we interchangeably use terms multiset and set throughout the whole paper.

3

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

Figure 1: Illustration of the proposed linear + power mapping embedding layer (LP) and logarithm
activation + linear + exponential activation embedding layer (LLE).

However, DeepSets [9] focuses on the case that the feature dimension of each set element is one
(i.e., D = 1). To demonstrate the difficulty extending Theorem 2.4 to high-dimensional features,
we reproduce the proof next, which simultaneously reveals its significance and limitation. Some
intermediate results and mathematical tools will be recalled along the way later in our proof.
We begin by defining sum-of-power mapping (of degree K) ΨK(X) = (cid:80)N
i=1 ψK(xi), where ψK
is the power mapping following Definition 2.6. Afterwards, we reveal that sum-of-power mapping
ΨK(X) has a continuous inverse. Before stating the formal argument, we formally define the
injectivity of permutation-invariant mappings:
Definition 2.7 (Injectivity). A set function h : RN ×D → RL is injective if there exists a function
g : RL → RN ×D such that for any X ∈ RN ×D, we have g ◦ f (X) ∼ X. Then g is an inverse of f .

And we summarize the existence of continuous inverse of ΨK(x) into the following lemma shown
by [9] and improved by [23]. This result comes from homeomorphism between roots and coefficients
of monic polynomials [35].
Lemma 2.8 (Existence of Continuous Inverse of Sum-of-Power [9,23]). ΨN : RN → RN is injective,
thus the inverse Ψ−1

N : RN → RN exists. Moreover, Ψ−1

N is continuous.

Now we are ready to prove necessity in Theorem 2.4 as sufficiency is easy to check. By choosing
ϕ = ψN : R → RN to be the power mapping (cf. Definition 2.6), and ρ = f ◦ Ψ−1
N . For any scalar-
(cid:16)(cid:80)N
valued set X = (cid:2)x(1)
N ◦ ΨN (x) = f (P X) = f (X)
for some P ∈ Π(N ). The existence and continuity of Ψ−1

(cid:17)
i=1 ϕ(x(i))

= f ◦ Ψ−1

x(N )(cid:3)⊤

N are due to Lemma 2.8.

· · ·

, ρ

Theorem 2.4 gives the exact decomposable form [23] for permutation-invariant functions, which
is stricter than approximation error based expressiveness analysis. In summary, the key idea is to
establish a mapping ϕ whose element-wise sum-pooling has a continuous inverse.

Curse of High-dimensional Features. We argue that the proof of Theorem 2.4 is not applicable
to high-dimensional set features (D ≥ 2). The main reason is that power mapping defined in
Definition 2.6 only receives scalar input. It remains elusive how to extend it to a multivariate version
that admits injectivity and a continuous inverse. A plausible idea seems to be applying power mapping
for each channel xi independently, and due to the injectivity of sum-of-power mapping ΨN , each
channel can be uniquely recovered individually via the inverse Ψ−1
N . However, we point out that
each recovered feature channel x′
D] ∼ X, where
the alignment of features across channels gets lost. Hence, channel-wise power encoding no more
composes an injective mapping. Zaheer et al. [9] proposed to adopt multivariate polynomials as ϕ for
high-dimensional case, which leverages the fact that multivariate symmetric polynomials are dense in
the space of permutation invariant functions (akin to Stone-Wasserstein theorem) [30]. This idea later
· · · (cid:81)
where α ∈ ND traverses
got formalized in [25] by setting ϕ(x(i)) =
all (cid:80)
j∈[D] αj ≤ n and extended to permutation equivariant functions. Nevertheless, the dimension
L = (cid:0)N +D
(cid:1), i.e., exponential in min{N, D} in this case, and unlike DeepSets [9] which exactly
recovers f for D = 1, the architecture in [9, 25] can only approximate the desired function.

i ∼ xi, ∀i ∈ [D], does not imply [x′

j∈[D](x(i)

· · · x′

j )αj

· · ·

(cid:105)

(cid:104)

D

1

3 Main Results

In this section, we present our main result which extends Theorem 2.4 to high-dimensional features.
Our conclusion is that to universally represent a set function on sets of length N and feature dimension

4

...LPLLE...............SumSum...161

162

163

164

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

D with the DeepSets architecture [9] (Eq. (1)), a dimension L at most polynomial in N and D is
needed for expressing the intermediate embedding space.

Formally, we summarize our main result in the following theorem.
Theorem 3.1 (The main result). Suppose D ≥ 2. For any continuous permutation-invariant function
f : KN ×D → R, K ⊆ R, there exists two continuous mappings ϕ : RD → RL and ρ : RL → R
such that for every X ∈ KN ×D, f (X) = ρ

(cid:16)(cid:80)N

where

(cid:17)
i=1 ϕ(x(i))

• For some L ∈ [N (D + 1), N 5D2] when ϕ admits linear layer + power mapping (LP) architecture:
ϕ(x) = (cid:2)ψN (w1x)⊤ · · · ψN (wKx)⊤(cid:3)

(2)

for some w1, · · · , wK ∈ RD, and K = L/N .

• For some L ∈ [N D, 2N 2D2] when ϕ admits logarithm activations + linear layer + exponential

activations (LLE) architecture:

ϕ(x) = [exp(w1 log(x))

· · ·

exp(wL log(x))]

(3)

for some w1, · · · , wL ∈ RD and K ⊆ R>0.

The bounds of L depend on the choice of the architecture of ϕ, which are illustrated in Fig. 1. In
the LP setting, we adopt a linear layer that maps each set element into K dimension. Then we apply
a channel-wise power mapping that separately transforms each value in the feature vector into an
N -order power series, and concatenates all the activations together, resulting in a KN dimension
feature. The LP architecture is closer to DeepSets [9] as they share the power mapping as the main
component. Theorem 3.1 guarantees the existence of ρ and ϕ (in the form of Eq. (2)) which satisfy
Eq. (1) without the need to set K larger than N 4D2 while K ≥ D + 1 is necessary. Therefore, the
total embedding size L = KN is bounded by N 5D2 above and N (D + 1) below. Note that this
lower bound is not trivial as N D is the degree of freedom of the input X. No matter how w1, ..., wK
are adopted, one cannot achieve an injective mapping by just using N D dimension.

In the LLE architecture, we investigate the utilization of logarithmic and exponential activations in set
representation, which are also valid activations to build deep neural networks [36, 37]. Each set entry
will be squashed by a element-wise logarithm first, then linearly embedded into an L-dimensional
space via a group of weights, and finally transformed by an element-wise exponential activation.
Essentially, each exp(wi log(x)), i ∈ [L] gives a monomial of x. The LLE architecture requires the
feature space constrained on the positive orthant to ensure logarithmic operations are feasible. But
the advantage is that the upper bound of L is improved to be 2N 2D2. The lower bound N D for
the LLE architecture is a trivial bound due to the degree of freedom of the input X. Note that the
constraint on the positive orthant R>0 is not essential. If we are able to use monomial activations to
process a vector x as used in [25, 26], then, the constraint on the positive orthant can be removed.
Remark 3.2. The bounds in Theorem 3.1 are non-asymptotic. This implies the latent dimensions
specified by the corresponding architectures are precisely sufficient for expressing the input.
Remark 3.3. Unlike ϕ, the form of ρ cannot be explicitly specified, as it depends on the desired
function f . The complexity of ρ remains unexplored in this paper, which may be high in practice.

Importance of Continuity. We argue that the requirements of continuity on ρ and ϕ are essential
for our discussion. First, practical neural networks can only provably approximate continuous
functions [27, 28]. Moreover, set representation without such requirements can be straightforward
(but likely meaningless in practice). This is due to the following lemma.
Lemma 3.4 ( [38]). There exists a discontinuous bijective mapping between RD and R if D ≥ 2.

By Lemma 3.4, we can define a bijective mapping r : RD → R which maps the high-dimensional
features to scalars, and its inverse exists. Then, the same proof of Theorem 2.4 goes through by
letting ϕ = ψN ◦ r and ρ = f ◦ r−1 ◦ Ψ−1

N . However, we note both ρ and ϕ lose continuity.

Comparison with Prior Arts. Below we highlight the significance of Theorem 3.1 in contrast
to the existing literature. A quick overview is listed in Tab. 1 for illustration. The lower bound
in Theorem 3.1 corrects a natural misconception that the degree of freedom (i.e., L = N D for

5

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

multi-channel cases) is not enough for representing the embedding space. Fortunately, the upper
bound in Theorem 3.1 shows the complexity of representing vector-valued sets is still manageable as
it merely scales polynomially in N and D. Compared with Zweig and Bruna’s finding [26], our result
significantly improves this bound on L from exponential to polynomial by allowing non-analytic
functions to amortize the expressiveness. Besides, Zweig and Bruna’s work [26] is hard to be applied
to the real domain, while ours are extensible to complex numbers and equivariant functions.

4 Proof Sketch

In this section, we introduce the proof techniques of Theorem 3.1, while deferring a full version and
all missing proofs to the supplementary materials.

The proof of Theorem 3.1 mainly consists of two steps below, which is completely constructive:

1. For the LP architecture, we construct a group of K linear weights w1 · · · , wK with K ≤ N 4D2
such that the summation over the associated LP embedding (Eq. (2)): Ψ(X) = (cid:80)N
i=1 ϕ(x(i)) is
injective and has a continuous inverse. Moreover, if K ≤ D, such weights do not exist, which
induces the lower bound.

2. Similarly, for the LLE architecture, we construct a group of L linear weights w1 · · · , wL with
L ≤ 2N 2D2 such that the summation over the associated LLE embedding (Eq. (3)) is injective
and has a continuous inverse. Trivially, if L < N D, such weights do not exist, which induces the
lower bound.

3. Then the proof of upper bounds can be concluded for both settings by letting ρ = f ◦ Ψ−1 since

(cid:16)(cid:80)N

ρ

i=1 ϕ(x(i))

(cid:17)

= f ◦ Ψ−1 ◦ Ψ(X) = f (P X) = f (X) for some P ∈ Π(N ).

Next, we elaborate on the construction idea which yields injectivity for both embedding layers in Sec.
4.1 and 4.2, respectively. To show injectivity, it is equivalent to establish the following statement for
both Eq. (2) and Eq. (3), respectively:

∀X, X ′ ∈ RN ×D,

N
(cid:88)

i=1

ϕ(x(i)) =

N
(cid:88)

i=1

ϕ(x′(i)) ⇒ X ∼ X ′

(4)

230

In Sec. 4.3, we prove the continuity of the inverse map for LP and LLE via arguments similar to [35].

231

4.1

Injectivity of LP

In this section, we consider ϕ follows the definition in Eq. (2), which amounts to first linearly
transforming each set element and then applying channel-wise power mapping. This is, we seek
a group of linear transformations w1, · · · , wK such that X ∼ X ′ can be induced from Xwi ∼
X ′wi, ∀i ∈ [K] for some K larger than N while being polynomial in N and D. The intuition is that
linear mixing among each channel can encode relative positional information. Only if X ∼ X ′, the
mixing information can be reproduced.

Formally, the first step accords to the property of power mapping (cf. Lemma 2.8), and we can obtain:

N
(cid:88)

i=1

ϕ(x(i)) =

N
(cid:88)

i=1

ϕ(x′(i)) ⇒ Xwi ∼ X ′wi, ∀i ∈ [K].

(5)

To induce X ∼ X ′ from Xwi ∼ X ′wi, ∀i ∈ [K], our construction divides the weights {wi, i ∈
[K]} into three groups: {w(1)
i,j,k : i ∈ [D], j ∈ [K1], k ∈
[K2]}. Each block is outlined as below:

: j ∈ [K1]}, and {w(3)

: i ∈ [D]}, {w(2)

j

i

1. Let the first group of weights w(1)
ei is the i-th canonical basis.

1 = e1, · · · , w(1)

D = eD to buffer the original features, where

2. Design the second group of linear weights, w(2)

1 , · · · , w(2)
K1

1)/2 + 1, which, by Lemma 4.4 latter, guarantees at least one of Xw(2)
anchor defined below:

j

for K1 as large as N (N − 1)(D −
, j ∈ [K1] forms an

6

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

287

288

289

290

Definition 4.1 (Anchor). Consider the data matrix X ∈ RN ×D, then a ∈ RN is called an anchor
of X if ai ̸= aj for any i, j ∈ [N ] such that x(i) ̸= x(j).
And suppose a = Xw(2)
show the following statement is true by Lemma 4.3 latter:

j∗ is an anchor of X for some j∗ ∈ [K1] and a′ = X ′w(2)

j∗ , then we

[a xi] ∼ [a′ x′

i] , ∀i ∈ [D] ⇒ X ∼ X ′.

(6)

3. Design a group of weights w(3)

mixes each original channel xi with each Xw(2)
show in Lemma 4.5 that:

i,j,k for i ∈ [D], j ∈ [K1], k ∈ [K2] with K2 = N (N − 1) + 1 that
. Then we

, j ∈ [K1] by w(3)

i,j,k = ei − γkw(2)

j

j

Xwi ∼ X ′wi, ∀i ∈ [K] ⇒

(cid:104)
Xw(2)
j

(cid:105)

(cid:104)

∼

X ′w(2)

j

(cid:105)

x′

i

, ∀i ∈ [D], j ∈ [K1]

(7)

xi

With such configuration, injectivity can be concluded by the entailment along Eq. (5), (7), (6): Eq. (5)
guarantees the RHS of Eq. (7); The existence of the anchor in Lemma 4.4 paired with Eq. (6)
guarantees X ∼ X ′. The total required number of weights K = D + K1 + DK1K2 ≤ N 4D2.
Below we provides a series of lemmas that demonstrate the desirable properties of anchors and
elaborate on the construction complexity. Detailed proofs are left in Appendix. In plain language, by
Definition 4.1, two entries in the anchor must be distinctive if the set elements at the corresponding
indices are not equal. As a consequence, we derive the following property of anchors:
Lemma 4.2. Consider the data matrix X ∈ RN ×D and a ∈ RN an anchor of X. Then if there
exists P ∈ Π(N ) such that P a = a then P xi = xi for every i ∈ [D].

With the above property, anchors defined in Definition 4.1 indeed have the entailment in Eq. (6):
Lemma 4.3 (Union Alignment based on Anchor Alignment). Consider the data matrix X, X ′ ∈
RN ×D, a ∈ RN is an anchor of X and a′ ∈ RN is an arbitrary vector. If [a xi] ∼ [a′ x′
i] for
every i ∈ [D], then X ∼ X ′.

However, the anchor a is required to be generated from X via a point-wise linear transformation.
The strategy to generate an anchor is to enumerate as many linear weights as needs, so that for any X,
at least one j such that Xw(2)
j becomes an anchor. We show that at most N (N − 1)(D − 1)/2 + 1
linear weights are enough to guarantee the existence of an anchor for any X:
Lemma 4.4 (Anchor Construction). There exists a set of weights w1, · · · , wK where K = N (N −
1)(D − 1)/2 + 1 such that for every data matrix X ∈ RN ×D, there exists j ∈ [K], Xwj is an
anchor of X.

We wrap off the proof by presenting the following lemma which is applied to prove Eq. (7) by fixing
any i ∈ [D], j ∈ [K1] in Eq. (7) while checking the condition for all k ∈ [K2]:
Lemma 4.5 (Anchor Matching). There exists a group of coefficients γ1, · · · , γK2 where K2 =
N (N − 1) + 1 such that the following statement holds: Given any x, x′, y, y′ ∈ RN such that
x ∼ x′ and y ∼ y′, if (x − γky) ∼ (x′ − γky′) for every k ∈ [K2], then [x y] ∼ [x′ y′].

For completeness, we add the following lemma which implies LP-induced sum-pooling cannot be
injective if K ≤ N D, when D ≥ 2.
Theorem 4.6 (Lower Bound). Consider data matrices X ∈ RN ×D where D ≥ 2. If K ≤ D, then
for every w1, · · · , wK, there exists X ′ ∈ RN ×D such that X ̸∼ X ′ but Xwi ∼ X ′wi for every
i ∈ [K].
Remark 4.7. Theorem 4.6 is significant in that with high-dimensional features, the injectivity is
provably not satisfied when the embedding space has dimension equal to the degree of freedom.

286

4.2

Injectivity of LLE

In this section, we consider ϕ follows the definition in Eq. (3). First of all, we note that each term in
the RHS of Eq. (3) can be rewritten as a monomial as shown in Eq. (8). Suppose we are able to use
monomial activations to process a vector x(i). Then, the constraint on the positive orthant R>0 in our
main result Theorem 3.1 can be even removed.

(cid:104)
· · · (cid:81)D

j=1 xwi,j

j

(cid:105)

· · ·

(8)

ϕ(x) = [· · ·

exp(wi log(x))

· · ·] =

7

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

Then, the assignment of w1, · · · , wL amounts to specifying the exponents for D power functions
within the product. Next, we prepare our construction with the following two lemmas:
Lemma 4.8. For any pair of vectors x1, x2 ∈ RN , y1, y2 ∈ RN , if (cid:80)
(cid:80)

2,i for every l, k ∈ [N ] such that 0 ≤ k ≤ l, then [x1 x2] ∼ [y1 y2].

i∈[N ] xl−k

i∈[N ] yl−k

1,i xk

1,i yk

2,i =

The above lemma is to show that we may use summations of monic bivariate monomials to align every
two feature columns. The next lemma shows that such pairwise alignment yields union alignment.
Lemma 4.9 (Union Alignment based on Pairwise Alignment). Consider data matrices X, X ′ ∈
RN ×D. If [xi xj] ∼ [x′
j] for every i, j ∈ [D], then X ∼ X ′.

i x′

Then the construction idea of w1, · · · , wL can be drawn from Lemma 4.8 and 4.9:

1. Lemma 4.8 indicates if the weights in Eq. (8) enumerate all the monic bivariate monomials in
j for all i, j ∈ [D] and p + q ≤ N ,

each pair of channels with degrees less or equal to N , i.e., xp
then we can yield:

i xq

N
(cid:88)

i=1

ϕ(x(i)) =

N
(cid:88)

i=1

ϕ(x′(i)) ⇒ [xi xj] ∼ [x′

i x′

j] , ∀i, j ∈ [D].

(9)

2. The next step is to invoke Lemma 4.9 which implies if every pair of feature channels is aligned,

then we can conclude all the channels are aligned with each other as well.
j] , ∀i, j ∈ [D] ⇒ X ∼ X ′.

[xi xj] ∼ [x′

i x′

(10)

Based on these motivations, we assign the weights that induce all bivariate monic monomials with
the degree no more than N . First of all, we reindex {wi, i ∈ [L]} as {wi,j,p,q, i ∈ [D], j ∈ [D], p ∈
[N ], q ∈ [p + 1]}. Then weights can be explicitly specified as wi,j,p,q = (q − 1)ei + (p − q + 1)ej,
where ei is the i-th canonical basis. With such weights, injectivity can be concluded by entailment
along Eq. (9) and (10). Moreover, the total number of linear weights is L = D2(N + 3)N/2 ≤
2N 2D2, as desired.

311

4.3 Continuous Lemma

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

In this section, we show that the LP and LLE induced sum-pooling are both homeomorphic. We
note that it is intractable to obtain the closed form of their inverse maps. Notably, the following
remarkable result can get rid of inversing a functions explicitly by merely examining the topological
relationship between the domain and image space.
Lemma 4.10. (Theorem 1.2 [35]) Let (X , dX ) and (Y, dY ) be two metric spaces and f : X → Y is
a bijection such that (a) each bounded and closed subset of X is compact, (b) f is continuous, (c)
f −1 maps each bounded set in Y into a bounded set in X . Then f −1 is continuous.

Subsequently, we show the continuity in an informal but more intuitive way while deferring a rigorous
version to the supplementary materials. Denote Ψ(X) = (cid:80)
i∈[N ] ϕ(x(i)). To begin with, we set
X = RN ×D/ ∼ with metric dX (X, X ′) = minP ∈Π(N ) ∥X − P X ′∥1 and Y = {Ψ(X)|X ∈
X } ⊆ RL with metric dY (y, y′) = ∥y − y′∥∞. It is easy to show that X satisfies the conditions
(a) and Ψ(X) satisfies (b) for both LP and LLE embedding layers. Then it remains to conclude the
proof by verifying the condition (c) for the mapping Y → X , i.e., the inverse of Ψ(X). We visualize
this mapping following our arguments on injectivity:

(LP )

Ψ(X)

(LLE) Ψ(X)
(cid:124) (cid:123)(cid:122) (cid:125)
Y

[· · · P iXwi

Eq. (5)
−−−→
−−−→ (cid:2)· · · Qi,jxi Qi,jxj
Eq. (9)
(cid:123)(cid:122)
Z

(cid:124)

· · ·] , i ∈ [K]

· · ·(cid:3) , i, j ∈ [D]
(cid:125)

Eqs. (6) + (7)
−−−−−−−→ P X

Eq. (10)
−−−−→ QX
(cid:124)(cid:123)(cid:122)(cid:125)
X

,

326

327

328

329

330

for some X dependent P , Q. Here, P i, i ∈ [K] and Qi,j, i, j ∈ [D] ∈ Π(N ). According to
homeomorphism between polynomial coefficients and roots (Theorem 3.4 in [35]), any bounded set
in Y will induce a bound set in Z. Moreover, since elements in Z contains all the columns of X (up
to some changes of the entry orders), a bounded set in Z also corresponds to a bounded set in X .
Through this line of arguments, we conclude the proof.

8

331

5 Extensions

332

In this section, we discuss two extensions to Theorem 3.1, which strengthen our main result.

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

Permutation Equivariance. Permutation-equivariant functions (cf. Definition 2.3) are considered
as a more general family of set functions. Our main result does not lose generality to this class of
functions. By Lemma 2 of [7], Theorem 3.1 can be directly extended to permutation-equivariant
functions with the same lower and upper bounds, stated as follows:
Theorem 5.1 (Extension to Equivariance). For any permutation-equivariant function f : KN ×D →
RN , K ⊆ R, there exists continuous functions ϕ : RD → RL and ρ : RD × RL → R such that
for every j ∈ [N ], where L ∈ [N (D + 1), N 5D2] when ϕ
f (X)j = ρ
admits LP architecture, and L ∈ [N D, 2N 2D2] when ϕ admits LLE architecture (K ∈ R>0).

i∈[N ] ϕ(x(i))

x(j), (cid:80)

(cid:16)

(cid:17)

Complex Domain. The upper bounds in Theorem 3.1 is also true to complex features up to a
constant scale (i.e., K ⊆ C). When features are defined over CN ×D, our primary idea is to divide
each channel into two real feature vectors, and recall Theorem 3.1 to conclude the arguments on an
RN ×2D input. All of our proof strategies are still applied. This result directly contrasts to Zweig
and Bruna’s work [26] whose main arguments were established on complex numbers. We show
that even moving to the complex domain, polynomial length of L is still sufficient for the DeepSets
architecture [9]. We state a formal version of the theorem in the supplementary material.

348

6 Related Work

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

Works on neural networks to represent set functions have been discussed extensively in the Sec. 1.
Here, we review other related works on the expressive power analysis of neural networks.

Early works studied the expressive power of feed-forward neural networks with different activa-
tions [27, 28]. Recent works focused on characterizing the benefits of the expressive power of deep
architectures to explain their empirical success [39–43]. Modern neural networks often enforce some
invariance properties into their architectures such as CNNs that capture spatial translation invariance.
The expressive power of invariant neural networks has been analyzed recently [22, 44, 45].

The architectures studied in the above works allow universal approximation of continuous func-
tions defined on their inputs. However, the family of practically useful architectures that enforce
permutation invariance often fail in achieving universal approximation. Graph Neural Networks
(GNNs) enforce permutation invariance and can be viewed as an extension of set neural networks
to encode a set of pair-wise relations instead of a set of individual elements [20, 21, 46, 47]. GNNs
suffer from limited expressive power [5, 17, 18] unless they adopt exponential-order tensors [48].
Hence, previous studies often characterized GNNs’ expressive power based on their capability of
distinguishing non-isomorphic graphs. Only a few works have ever discussed the function approxima-
tion property of GNNs [49–51] while these works still miss characterizing such dependence on the
depth and width of the architectures [52]. As practical GNNs commonly adopt the architectures that
combine feed-forward neural networks with set operations (neighborhood aggregation), we believe
the characterization of the needed size for set function approximation studied in [26] and this work
may provide useful tools to study finer-grained characterizations of the expressive power of GNNs.

7 Conclusion

This work investigates how many neurons are needed to model the embedding space for set repre-
sentation learning with the DeepSets architecture [9]. Our paper provides an affirmative answer that
polynomial many neurons in the set size and feature dimension are sufficient. Compared with prior
arts, our theory takes high-dimensional features into consideration while significantly advancing the
state-of-the-art results from exponential to polynomial.

Limitations. The tightness of our bounds is not examined in this paper, and the complexity of ρ is
uninvestigated and left for future exploration. Besides, deriving an embedding layer agnostic lower
bound for the embedding space remains another widely open question.

9

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

424

425

426

427

428

References

[1] Yann LeCun, Yoshua Bengio, et al. Convolutional networks for images, speech, and time series.

The handbook of brain theory and neural networks, 3361(10):1995, 1995.

[2] Taco Cohen and Max Welling. Group equivariant convolutional networks. In International

conference on machine learning, pages 2990–2999. PMLR, 2016.

[3] Michael M Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, and Pierre Vandergheynst.
Geometric deep learning: going beyond euclidean data. IEEE Signal Processing Magazine,
34(4):18–42, 2017.

[4] Risi Kondor and Shubhendu Trivedi. On the generalization of equivariance and convolution
in neural networks to the action of compact groups. In International Conference on Machine
Learning, pages 2747–2755, 2018.

[5] Haggai Maron, Heli Ben-Hamu, Nadav Shamir, and Yaron Lipman. Invariant and equivariant
graph networks. In International Conference on Learning Representations (ICLR), 2018.

[6] Alexander Bogatskiy, Brandon Anderson, Jan Offermann, Marwah Roussi, David Miller, and
Risi Kondor. Lorentz group equivariant neural network for particle physics. In International
Conference on Machine Learning, pages 992–1002. PMLR, 2020.

[7] Peihao Wang, Shenghao Yang, Yunyu Liu, Zhangyang Wang, and Pan Li. Equivariant hyper-
graph diffusion neural operators. In International Conference on Learning Representations
(ICLR), 2023.

[8] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point
sets for 3d classification and segmentation. In Proceedings of the IEEE conference on computer
vision and pattern recognition, pages 652–660, 2017.

[9] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Russ R Salakhutdinov,
and Alexander J Smola. Deep sets. In Advances in Neural Information Processing Systems
(NeurIPS), 2017.

[10] Vinicius Mikuni and Florencia Canelli. Point cloud transformers applied to collider physics.

Machine Learning: Science and Technology, 2(3):035027, 2021.

[11] Huilin Qu and Loukas Gouskos.

Jet tagging via particle clouds. Physical Review D,

101(5):056019, 2020.

[12] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and Vladlen Koltun. Point transformer. In
Proceedings of the IEEE/CVF international conference on computer vision, pages 16259–16268,
2021.

[13] Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh.
Set transformer: A framework for attention-based permutation-invariant neural networks. In
International conference on machine learning, pages 3744–3753. PMLR, 2019.

[14] Yan Zhang, Jonathon Hare, and Adam Prugel-Bennett. Deep set prediction networks. Advances

in Neural Information Processing Systems, 32, 2019.

[15] Yan Zhang, Jonathon Hare, and Adam Prügel-Bennett. Fspool: Learning set representations
with featurewise sort pooling. In International Conference on Learning Representations, 2020.

[16] Aditya Grover, Eric Wang, Aaron Zweig, and Stefano Ermon. Stochastic optimization of sorting
networks via continuous relaxations. In International Conference on Learning Representations,
2020.

[17] Christopher Morris, Martin Ritzert, Matthias Fey, William L Hamilton, Jan Eric Lenssen,
Gaurav Rattan, and Martin Grohe. Weisfeiler and leman go neural: Higher-order graph neural
networks. In the AAAI Conference on Artificial Intelligence, volume 33, pages 4602–4609,
2019.

[18] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural

networks? In International Conference on Learning Representations, 2019.

[19] Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Liò, and Petar Veliˇckovi´c. Principal
neighbourhood aggregation for graph nets. Advances in Neural Information Processing Systems,
33:13260–13271, 2020.

10

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

473

474

475

476

477

478

479

[20] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini.
The graph neural network model. IEEE Transactions on Neural Networks, 20(1):61–80, 2008.

[21] Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large

graphs. In Advances in Neural Information Processing Systems, 2017.

[22] Haggai Maron, Ethan Fetaya, Nimrod Segol, and Yaron Lipman. On the universality of invariant
networks. In International conference on machine learning, pages 4363–4371. PMLR, 2019.

[23] Edward Wagstaff, Fabian Fuchs, Martin Engelcke, Ingmar Posner, and Michael A Osborne.
On the limitations of representing functions on sets. In International Conference on Machine
Learning, pages 6487–6494. PMLR, 2019.

[24] Edward Wagstaff, Fabian B Fuchs, Martin Engelcke, Michael A Osborne, and Ingmar Posner.
Universal approximation of functions on sets. Journal of Machine Learning Research, 23(151):1–
56, 2022.

[25] Nimrod Segol and Yaron Lipman. On universal equivariant set networks. In International

Conference on Learning Representations (ICLR), 2020.

[26] Aaron Zweig and Joan Bruna. Exponential separations in symmetric neural networks. arXiv

preprint arXiv:2206.01266, 2022.

[27] George Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of

control, signals and systems, 2(4):303–314, 1989.

[28] Kurt Hornik, Maxwell Stinchcombe, Halbert White, et al. Multilayer feedforward networks are

universal approximators. Neural Networks, 2(5):359–366, 1989.

[29] Shupeng Gui, Xiangliang Zhang, Pan Zhong, Shuang Qiu, Mingrui Wu, Jieping Ye, Zhengdao
Wang, and Ji Liu. Pine: Universal deep embedding for graph nodes via partial permutation
IEEE Transactions on Pattern Analysis and Machine Intelligence,
invariant set functions.
44(2):770–782, 2021.

[30] Nicolas Bourbaki. Éléments d’histoire des mathématiques, volume 4. Springer Science &

Business Media, 2007.

[31] David Rydh. A minimal set of generators for the ring of multisymmetric functions. In Annales

de l’institut Fourier, volume 57, pages 1741–1769, 2007.

[32] Adam Santoro, David Raposo, David G Barrett, Mateusz Malinowski, Razvan Pascanu, Peter
Battaglia, and Timothy Lillicrap. A simple neural network module for relational reasoning.
Advances in neural information processing systems, 30, 2017.

[33] R. L. Murphy, B. Srinivasan, V. Rao, and B. Ribeiro.

Janossy pooling: Learning deep
permutation-invariant functions for variable-size inputs. In International Conference on Learn-
ing Representations (ICLR), 2018.

[34] Akiyoshi Sannai, Yuuki Takai, and Matthieu Cordonnier. Universal approximations of permuta-
tion invariant/equivariant functions by deep neural networks. arXiv preprint arXiv:1903.01939,
2019.

[35] Branko ´Curgus and Vania Mascioni. Roots and polynomials as homeomorphic spaces. Exposi-

tiones Mathematicae, 24(1):81–95, 2006.

[36] Andrew R Barron. Universal approximation bounds for superpositions of a sigmoidal function.

IEEE Transactions on Information theory, 39(3):930–945, 1993.

[37] Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter. Fast and accurate deep network

learning by exponential linear units (elus). arXiv preprint arXiv:1511.07289, 2015.

[38] Fernando Q Gouvêa. Was cantor surprised? The American Mathematical Monthly, 118(3):198–

209, 2011.

[39] Dmitry Yarotsky. Error bounds for approximations with deep relu networks. Neural Networks,

94:103–114, 2017.

[40] Shiyu Liang and R Srikant. Why deep neural networks for function approximation?

In

International Conference on Learning Representations, 2017.

[41] Joe Kileel, Matthew Trager, and Joan Bruna. On the expressive power of deep polynomial

neural networks. Advances in neural information processing systems, 32, 2019.

11

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

[42] Nadav Cohen, Or Sharir, and Amnon Shashua. On the expressive power of deep learning: A

tensor analysis. In Conference on learning theory, pages 698–728. PMLR, 2016.

[43] Maithra Raghu, Ben Poole, Jon Kleinberg, Surya Ganguli, and Jascha Sohl-Dickstein. On the
expressive power of deep neural networks. In international conference on machine learning,
pages 2847–2854. PMLR, 2017.

[44] Dmitry Yarotsky. Universal approximations of invariant maps by neural networks. Constructive

Approximation, 55(1):407–474, 2022.

[45] Ding-Xuan Zhou. Universality of deep convolutional neural networks. Applied and computa-

tional harmonic analysis, 48(2):787–794, 2020.

[46] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural
message passing for quantum chemistry. In International Conference on Machine Learning
(ICML), 2017.

[47] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional

networks. In International Conference on Learning Representations (ICLR), 2017.

[48] Nicolas Keriven and Gabriel Peyré. Universal invariant and equivariant graph neural networks.

In Advances in Neural Information Processing Systems, pages 7090–7099, 2019.

[49] Zhengdao Chen, Soledad Villar, Lei Chen, and Joan Bruna. On the equivalence between graph
isomorphism testing and function approximation with gnns. In Advances in Neural Information
Processing Systems, pages 15868–15876, 2019.

[50] Zhengdao Chen, Lei Chen, Soledad Villar, and Joan Bruna. Can graph neural networks count

substructures? volume 33, 2020.

[51] Waïss Azizian and Marc Lelarge. Expressive power of invariant and equivariant graph neural
networks. In ICLR 2021-International Conference on Learning Representations, 2021.
[52] Andreas Loukas. What graph neural networks cannot learn: depth vs width. In International

Conference on Learning Representations, 2020.

12

