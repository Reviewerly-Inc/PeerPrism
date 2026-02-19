Stability and Sharper Risk Bounds with
Convergence Rate O(1/n2)

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

The sharpest known high probability excess risk bounds are up to O(1/n) for
empirical risk minimization and projected gradient descent via algorithmic stability
[Klochkov and Zhivotovskiy, 2021]. In this paper, we show that high probability
excess risk bounds of order up to O(1/n2) are possible. We discuss how high prob-
ability excess risk bounds reach O(1/n2) under strongly convexity, smoothness
and Lipschitz continuity assumptions for empirical risk minimization, projected
gradient descent and stochastic gradient descent. Besides, to the best of our knowl-
edge, our high probability results on the generalization gap measured by gradients
for nonconvex problems are also the sharpest.

1

Introduction

Algorithmic stability is a fundamental concept in learning theory [Bousquet and Elisseeff, 2002],
which can be traced back to the foundational works of Vapnik and Chervonenkis [1974] and has
a deep connection with learnability [Rakhlin et al., 2005, Shalev-Shwartz et al., 2010, Shalev-
Shwartz and Ben-David, 2014]. It is not difficult for only providing in-expectation error bounds via
stability arguments. However, high probability bounds are beneficial to understand the robustness of
optimization algorithms [Bousquet et al., 2020, Klochkov and Zhivotovskiy, 2021] and are much
more challenging [Feldman and Vondrak, 2019, Bousquet et al., 2020, Lv et al., 2021]. In this paper,
our goal is to improve the high probability risk bounds via algorithmic stability.

Let us start with some standard notations. We have a set of independent and identically distributed
observations S = {z1, . . . , zn} sampled from a probability measure ρ defined on a sample space
Z := X × Y. Based on the training set S, our goal is to build a model h : X (cid:55)→ Y for prediction,
where the model is determined by parameter w from parameter space W ⊂ Rd. The performance of
a model w on an example z can be quantified by a loss function f (w; z), where f : W × Z (cid:55)→ R+.
Then the population risk and the empirical risk of w ∈ W, respectively as

F (w) := Ez [f (w; z)] , FS(w) :=

1
n

n
(cid:88)

i=1

f (w; zi),

where Ez denotes the expectation w.r.t. z.
Let w∗ ∈ arg minw∈W F (w) be the model with the minimal population risk in W and w∗(S) ∈
arg minw∈W FS(w) be the model with the minimal empirical risk w.r.t. dataset S. Let A(S) be the
output of a (possibly randomized) algorithm A on the dataset S. Let ∥ · ∥2 denote the Euclidean norm
and ∇g(w) denote a subgradient of g at w.

Traditional generalization analysis aims to bound the generalization error F (A(S)) − FS(A(S)) w.r.t
the algorithm A and the dataset S. Based on the technique developed by Feldman and Vondrak [2018,

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.

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

n), where the
2019], Bousquet et al. [2020] provide the sharpest high probability bounds of O (L/
loss function f (·, ·) is bounded by M . No matter how stable the algorithm is, the high probability
generalization bound will not be smaller than O(L/
n). This is sampling error term scaling as
O (1/

n) that controls the generalization error [Klochkov and Zhivotovskiy, 2021].

√

√

√

A frequently used alternative to generalization bounds, that can avoid the sampling error, are the
excess risk bounds. The excess risk of algorithm A w.r.t. the dataset S is F (A(S)) − F (w∗), which
is more essential because it considers both generalization error and optimization error. Recently,
Klochkov and Zhivotovskiy [2021] provided the best high probability excess risk bounds of order
up to O(log n/n) for empirical risk minimization (ERM) and projected gradient descent (PGD)
algorithms via algorithmic stability.

On the other hand, Zhang et al. [2017], Li and Liu [2021], Xu and Zeevi [2024] derived high
probability excess risk bounds with O (cid:0)1/n2(cid:1) for ERM and stochastic gradient descent (SGD) via
uniform convergence when the sample number satisfies n = Ω(d), which implied that the rate
O (cid:0)1/n2(cid:1) is possible. However, the results obtained by the uniform convergence technique are related
to the dimension d, which is unacceptable in high-dimensional learning problems. Since stability
analysis can yield dimension-free bounds, we naturally have the following question:

Can algorithmic stability provide high probability excess risk bounds with the rate beyond O(1/n)?

The main results of this paper answers this question positively. We provides the first high probability
bounds that are dimension-free with the rate O(1/n2) for ERM, PGD and SGD. Our framework can
also be used to solve other stable algorithms.

To this end, we develop the generalization gap measured by gradients. Our bounds under nonconvex
setting are tighter than existing works based on both algorithmic stability [Fan and Lei, 2024] and
uniform convergence [Xu and Zeevi, 2024]. This is why we can achieve dimension-free excess risk
bounds of order O(1/n2). In fact, in nonconvex problems, optimization algorithms can only find
a local minimizer and we can only obtain optimization error bounds for ∥∇FS(A(S))∥2 [Ghadimi
and Lan, 2013]. Therefore, it is important to study the generalization behavior of A(S) measured by
gradients. Under Polyak-Lojasiewicz condition, we also obtain sharper results for both generalization
bounds of gradients and excess risk bounds. Our route to excess risk bounds can also be applied
to various stable algorithms and complex learning scenarios. In this paper, we take ERM, PGD,
and SGD as examples to explore the stability of stochastic convex optimization algorithms with
strongly convex losses. We provide tighter high probability dimension-free excess risk bounds of
up to O(1/n2) comapring with existing works based on both algorithmic stability [Klochkov and
Zhivotovskiy, 2021, Fan and Lei, 2024] and uniform convergence [Zhang et al., 2017, Li and Liu,
2021, Xu and Zeevi, 2024].

Besides, to obtain tighter bounds, we obtain a tighter p-moment bound for sums of vector-valued
functions by introducing the optimal Marcinkiewicz-Zygmund’s inequality for random variables
taking values in a Hilbert space in the proof, which has more potential applications in vector-valued
functional data.

This paper is organized as follows. The related work is reviewed in Section 2. In Section 3, we
present our main results for stability and generalization. We give applications to ERM, PGD and
SGD in Section 4. The conclusion is given in Section 5. All the proofs and additional lemmata are
deferred to the Appendix.

2 Related Work

Algorithmic stability. Algorithmic stability is a classical approach in generalization analysis, which
can be traced back to the foundational works of [Vapnik and Chervonenkis, 1974]. It gave the
generalization bound by analyzing the sensitivity of a particular learning algorithm when changing
one data point in the dataset. Modern framework of stability analysis was established by Bousquet
and Elisseeff [2002], where they presented an important concept called uniform stability. Since
then, a lot of works based on uniform stability have emerged. On one hand, generalization bounds
with algorithmic stability have been significantly improved by Feldman and Vondrak [2018, 2019],
Bousquet et al. [2020], Klochkov and Zhivotovskiy [2021]. On the other hand, different algorithmic
stability measures such as uniform argument stability [Liu et al., 2017, Bassily et al., 2020], on
average stability [Shalev-Shwartz et al., 2010, Kuzborskij and Lampert, 2018], hypothesis stability

2

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

[Bousquet and Elisseeff, 2002, Charles and Papailiopoulos, 2018], hypothesis set stability [Foster
et al., 2019], pointwise uniform stability [Fan and Lei, 2024], PAC-Bayesian stability [Li et al., 2020],
locally elastic stability [Deng et al., 2021], collective stability [London et al., 2016] and uniform
stability in gradients [Lei, 2023, Fan and Lei, 2024] have been developed. Most of them provided the
connection on stability and generalization in expectation. Bousquet and Elisseeff [2002], Elisseeff
et al. [2005], Feldman and Vondrak [2018, 2019], Bousquet et al. [2020], Klochkov and Zhivotovskiy
[2021], Fan and Lei [2024] considered high probability bounds. However, only Fan and Lei [2024]
developed vector-valued bounds (eg: generalization bounds of gradients), which can be the order at
most O (M/

n) and remain improvement.

√

(cid:17)
(cid:16)(cid:112)d/n

Uniform convergence. Uniform convergence is another popular approach in statistical learning
theory to study generalization bounds [Fisher, 1922, Vapnik, 1999, Van der Vaart, 2000]. The main
idea is to bound the generalization gap by its supremum over the whole (or a subset) of the hypothesis
space via some space complexity measures, such as VC dimension, covering number and Rademacher
complexity. For finite-dimensional problem, Kleywegt et al. [2002] provided that the generalization
error is O
depended on the sample size n and the dimension of parameters d in high
probability. In nonconvex settings, Mei et al. [2018] showed that the empirical of generalization error
is O((cid:112)d/n). Xu and Zeevi [2024] developed a novel “uniform localized convergence” framework
using generic chaining for the minimization problem and provided the localized generalization bounds
n + d
n
consider the order of n. However, uniform convergence results are related to the dimension d, which
is unacceptable in high-dimensional learning problems.

, which is the optimal result when we only

max (cid:8)∥w − w∗∥2, 1

in gradients O

(cid:18)(cid:113) d

(cid:19)(cid:19)

(cid:18)

(cid:9)

n

106

3 Stability and Generalization

107

108

109

110

111

To derive sharper generalization bounds of gradients, we need to develop a novel concentration
inequality which provide p-moment bound for sums of vector-valued functions. For a real-valued
random variable Y , the Lp-norm of Y is defined by ∥Y ∥p := (E[|Y |p])
p . Similarly, let ∥ · ∥ denote
the norm in a Hilbert space H. Then for a random variable X taking values in a Hilbert space, the
Lp-norm of X is defined by ∥∥X∥∥p := (E [∥X∥p])

1
p .

1

112

3.1 A Moment Bound for Sums of Vector-valued Functions

113

114

115

116

117

118

119

120

121

Here we present our sharper moment bound for sums of vector-valued functions of n independent
variables.

Theorem 1. Let Z = (Z1, . . . , Zn) be a vector of independent random variables each taking values
in Z, and let g1, . . . , gn be some functions: gi : Z n (cid:55)→ H such that the following holds for any
i ∈ [n]:

• ∥E[gi(Z)|Zi]∥ ≤ M a.s.,
• E (cid:2)gi(Z)|Z[n]\{i}

(cid:3) = 0 a.s.,

• gi satisfies the bounded difference property with β, namely, for any i = 1, . . . , n, the following

inequality holds

sup
z1,...,zn,z′
j

∥gi(z1, . . . , zj−1, zj, zj+1, . . . , zn) − gi(z1, . . . , zj−1, z′

j, zj+1, . . . , zn)∥ ≤ β.

(1)

122

Then, for any p ≥ 2, we have

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

n
(cid:88)

i=1

gi

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)p
(cid:13)

≤ 2((cid:112)2p + 1)

√

nM + 4 × 2

1
2p

(cid:19)

(cid:18)(cid:114) p
e

((cid:112)2p + 1)nβ ⌈log2 n⌉ .

3

123

124

Remark 1. The proof is motivated by Bousquet et al. [2020]. Under the same assumptions, Fan and
Lei [2024] also established the following inequality1

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

n
(cid:88)

i=1

gi

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)p
(cid:13)

√

≤ 2(

√

2 + 1)

npM + 4(

√

2 + 1)npβ ⌈log2 n⌉ .

(2)

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

It is easy to verify that our result is tighter than result provided by Fan and Lei [2024] for both the
first and second term. Comparing Theorem 1 with (2), the larger p is, the tighter our result is relative
to (2). In the worst case, when p = 2, the constant of our first term is 0.879 times tighter than (2),
and the constant of our second term is 0.634 times tighter than (2). This is because we derive the
optimal Marcinkiewicz-Zygmund’s inequality for random variables taking values in a Hilbert space
in the proof.

The improvement of this concentration inequality is meaningful. On one hand, we derive the optimal
Marcinkiewicz-Zygmund’s inequality for random variables taking values in a Hilbert space. On the
other hand, in Section 3.2, we will carefully construct vector-valued functions which satisfies all
the assumptions in Theorem 1 and ensures M = 0 at the same time. Under this condition, we can
eliminate the first term. When we use Theorem 1 instead of (2) in the whole proofs, at least 0.634
times tighter bound can be obtained strictly.

137

3.2 Sharper Generalization Bounds in Gradients

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

1, . . . , z′

n} be its independent copy. For any i ∈ [n], define S(i) = {zi, . . . , zi−1, z′

Let S = {z1, . . . , zn} be a set of independent random variables each taking values in Z and S′ =
{z′
i, zi+1, . . . , zn}
be a dataset replacing the i-th sample in S with another i.i.d. sample z′
i. We introduce some basic
definitions here and we want to emphasize that our main Theorem 2 and Theorem 3 do not need
smoothness assumption and PL condition.
Definition 1. Let g : W (cid:55)→ R. Let γ, µ < 0.

• We say g is γ-smooth if

∥∇g(w) − ∇g(w′)∥2 ≤ γ∥w − w′∥2,

∀w, w′ ∈ W.

• Let g∗ = minw∈W g(w). We say g satisfies the Polyak-Lojasiewicz (PL) condition with

parameter µ > 0 on W if

g(w) − g∗ ≤

1
2µ

∥∇g(w)∥2
2,

∀w ∈ W.

Then we define uniform stability in gradients.

Definition 2 (Uniform Stability in Gradients). Let A be a randomized algorithm. We say A is
β-uniformly-stable in gradients if for all neighboring datasets S, S(i), we have

(cid:13)
(cid:13)
(cid:13)∇f (A(S); z) − ∇f (A(S(i)); z)
(cid:13)
(cid:13)
(cid:13)2

sup
z

≤ β.

(3)

Remark 2. Gradient-based stability was firstly introduced by Lei [2023], Fan and Lei [2024] to
describe the generalization performance for nonconvex problems. In nonconvex problems, we can
only find a local minimizer by optimization algorithms which may be far away from the global
minimizer. Thus the convergence does not make much sense in function values.
Instead, the
convergence of ∥∇FS(A(S))∥2 was often studied in the optimization community [Ghadimi and Lan,
2013]. Since the population risk of gradients ∥∇F (A(S))∥2 can be decomposed as the convergence
of ∥∇FS(A(S))∥2 and the generalization gap ∥∇F (A(S)) − ∇FS(A(S))∥2, the generalization
analysis of ∥∇F (A(S)) − ∇FS(A(S))∥2 is important, which can be achieved by uniform stability
in gradients.

Theorem 2 (Generalization via Stability in Gradients). Assume for any S and any z,
If A is β-uniformly-stable in gradients, then for any δ ∈ (0, 1), the
∥∇f (A(S); z)∥2 ≤ M .

1They assume n = 2k, k ∈ N. Here we give the version of their result with general n.

4

161

following inequality holds with probability at least 1 − δ

∥∇F (A(S)) − ∇FS(A(S))∥2
(cid:17)
1 + e(cid:112)2 log (e/δ)

4M

(cid:16)

≤2β +

√

n

+ 8 × 2

√

1
4 (

2 + 1)

√

eβ ⌈log2 n⌉ log (e/δ).

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

i

(cid:2)EZ

Remark 3. Theorem 2 is a direct application via Theorem 1 where we denote gi(S) =
(cid:2)∇f (A(S(i)), Z)(cid:3) − ∇f (A(S(i)), zi)(cid:3) and find that gi(S) satisfies all the assumptions
Ez′
in Theorem 1. As a comparison, Fan and Lei [2024] also developed high probability bounds under
same assumptions, but our bounds are sharper since our moment inequality for sums of vector-valued
functions are tighter as we have discussed in Remark 1. Next, we derive sharper generalization bound
of gradients under same assumptions.

Theorem 3 (Sharper Generalization via Stability in Gradients). Assume for any S and any z,
∥∇f (A(S); z)∥2 ≤ M . If A is β-uniformly-stable in gradients, then for any δ ∈ (0, 1), the following
inequality holds with probability at least 1 − δ

∥∇F (A(S)) − ∇FS(A(S))∥2
(cid:115)

4EZ [∥∇f (A(S); Z)∥2

2] log 6
δ

≤

(cid:115) (cid:0) 1

+

n

+ 16 × 2

√

3
4

eβ ⌈log2 n⌉ log (3e/δ) + 32

√

eβ ⌈log2 n⌉ (cid:112)log 3e/δ.

2 β2 + 32nβ2 log(3/δ)(cid:1) log 6
n

δ

+

M log 6
δ
n

(cid:16)(cid:112)EZ [∥∇f (A(S); Z)∥2

Remark 4. Note that the factor in Theorem 2 before 1/
n is
depends on the bound of ∥∇f (·, ·)∥2. However, the factor in Theorem 3 before 1/
O
, not involving the possibly large term M . As is
known, optimization algorithms often provide parameters approaching the optimal solution, which
make the term EZ[∥∇f (A(S); Z)∥2
2] much more smaller than M . We will give further reasonable
results under more assumptions such as smoothness in Lemma 1 and Lemma 2.

(cid:17)
2] log 1/δ + β log(1/δ)

, which

n is O

√

√

(cid:16)

M (cid:112)log (e/δ)

(cid:17)

On the other hand, best high probability bounds based on uniform convergence [Xu and Zeevi, 2024]
is

∥∇F (A(S)) − ∇FS(A(S))∥2
(cid:114) EZ [∇∥f (w∗; Z)∥2

2] log(1/δ)

n

≲

+

log(1/δ)
n

(cid:26)

+ max

∥w − w∗∥2,

(cid:27) (cid:32)(cid:114)

d
n

+

(cid:33)

,

d
n

1
n

(4)

which is the optimal result when we only consider the order of n. However, uniform convergence
results are related to the dimension d, which is unacceptable in high-dimensional learning problems.

Note that (4) requires an additional smoothness-type assumption. As a comparison, when f is
γ-smoothness, our result in Theorem 3 can be easily derived as

∥∇F (A(S)) − ∇FS(A(S))∥2

≲β log n log(1/δ) +

log(1/δ)
n

+

(cid:114) EZ [∇∥f (w∗; Z)∥2

2] log(1/δ)

n

(cid:114)

+ ∥A(S) − w∗∥

log(1/δ)
n

.

This result implies that when the uniformly stable in gradients parameter β is smaller than 1/
n, our
bound is tighter than (4) and is dimension independent. Note that Theorem 3 holds in nonconvex
problems, to the best of our knowledge, this is the sharpest upper bound in both uniform convergence
and algorithmic stability analysis.

√

Here we give the proof sketch of Theorem 3, which is motivated by the analysis in Klochkov and
Zhivotovskiy [2021]. The key idea is to build vector functions qi(S) = hi(S) − ES{zi}[hi(S)]
(cid:2)∇f (A(S(i)), Z)(cid:3) − ∇f (A(S(i)), zi)(cid:3). These functions satisfy
where we define hi(S) = Ez′
all the assumptions in Theorem 1 and ensure the factor M in Theorem 1 to 0. Then the term O(1/
n)
can be eliminated.

(cid:2)EZ

√

i

5

192

193

194

Lemma 1. Let assumptions in Theorem 3 hold. Suppose the function f is γ-smooth and the population
risk F satisfies the PL condition with parameter µ. Then for any δ ∈ (0, 1), when n ≥ 16γ2 log 6
,
with probability at least 1 − δ, we have

µ2

δ

∥∇F (A(S)) − ∇FS(A(S))∥2

≤∥∇FS(A(S))∥2 + 4

(cid:115)

EZ [∥∇f (w∗; Z)∥2

2] log 6
δ

n

+ 2

(cid:115) (cid:0) 1

+

2M log 6
δ
n

+ 32 × 2

√

3
4

eβ ⌈log2 n⌉ log (3e/δ) + 64

2 β2 + 32nβ2 log(3/δ)(cid:1) log 6
n
eβ ⌈log2 n⌉ (cid:112)log 3e/δ.

δ

√

195

196

Remark 5. The following inequality can be easily derived using triangle inequality and Cauchy-
Bunyakovsky-Schwarz inequality:

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

F (A(S)) − F (w∗) ≲ ∥∇FS(A(S))∥2 +

F (w∗) log (1/δ)
n

+

log2(1/δ)
n2

+ β2 log2 n log2(1/δ).

(5)

Above inequality implies that excess risk can be bound by the optimization gradient error
∥∇FS(A(S))∥2 and uniform stability in gradients β. Note that the assumption F (w∗) = O(1/n)
is common and can be found in Srebro et al. [2010], Lei and Ying [2020], Liu et al. [2018], Zhang
et al. [2017], Zhang and Zhou [2019]. This is natural since F (w∗) is the minimal population risk.
On the other hand, we can derive that under µ-strongly convex and γ-smooth assumptions for the
objective function f , uniform stability in gradients can be bounded of order O(1/n) for ERM and
PGD. Thus high probability excess risk can be bounded of order up to O (cid:0)1/n2(cid:1) under these common
assumptions via algorithmic stability. Comparing with current best related work [Klochkov and
Zhivotovskiy, 2021], they are insensitive to the stability parameter being smaller than O(1/n) and
their best rates can only up to O(1/n). Although we involve extra smoothness and PL condition
assumptions, these assumptions are also common in optimization community and our work can fully
utilize these assumptions.

Besides, we discuss uniform stability in gradients for common algorithms such as ERM, PGD, and
SGD in Section 4. Our results can be easily extended to other stable algorithms. Due to smoothness’s
property to link the uniform stability in gradients with uniform argument stability, many works
[Bassily et al., 2020, Feldman and Vondrak, 2019, Hardt et al., 2016] exploring uniform argument
stability can also use our framework.

Finally, the population risk of gradients ∥∇F (A(S))∥2 can be gracefully bounded by the empirical
risk of gradients ∥∇FS(A(S))∥2 under strong growth condition (SGC), that connects the rates at
which the stochastic gradients shrink relative to the full gradient Vaswani et al. [2019].
Definition 3 (Strong Growth Condition). We say SGC holds if
EZ

2
Remark 6. There has been some related work that takes SGC into assumption Solodov [1998],
Vaswani et al. [2019], Lei [2023]. Vaswani et al. [2019] has proved that the squared-hinge loss with
linearly separable data and finite support features satisfies the SGC. Note that we only suppose this
condition holds in Lemma 2.
Lemma 2 (SGC case). Let assumptions in Theorem 3 hold and suppose SGC holds. Then for any
δ > 0, with probability at least 1 − δ, we have

(cid:3) ≤ ρ∥∇F (w)∥2
2.

(cid:2)∥∇f (w; Z)∥2

∥∇F (A(S))∥ ≲ (1 + η)∥∇FS(A(S))∥ +

1 + η
η

(cid:18) M
n

log

6
δ

+ β log n log

(cid:19)

.

1
δ

Remark 7. Lemma 2 build a connection between the population gradient error and the empirical
gradient error under Lipschitz, nonconvex, nonsmooth and SGC case and elucidate that the population
gradient error can be bounded of up to O(1/n) under nonconvex problems.

4 Application

In this section, we analysis stochastic convex optimization with strongly convex losses. The most
common setting is where at each round, the learner gets information on f through a stochastic
gradient oracle [Rakhlin et al., 2012]. To derive uniform stability in gradients for algorithms, we
firstly introduce the strongly convex assumption.

6

232

Definition 4. We say g is µ-strongly convex if

g(w) ≥ g(w′) + ⟨w − w′, ∇g(w′)⟩ +

µ
2

∥w − w′∥2
2,

∀w, w′ ∈ W.

233

4.1 Empirical Risk Minimizer

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

Empirical risk minimizer is one of the classical approaches for solving stochastic optimization (also
referred to as sample average approximation (SAA)) in machine learning community. The following
lemma shows the uniform stability in gradient for ERM under µ-strongly convexity and γ-smoothness
assumptions.

Lemma 3 (Stability of ERM). Suppose the objective function f is µ-strongly-convex and γ-smooth.
For any w ∈ W and any z, suppose that ∥∇f (w; z) ≤ M ∥. Let ˆw∗(S(i)) be the ERM of FS(i)(w)
i, ..., zn} and ˆw∗(S) be the ERM of
that denotes the empirical risk on the samples S(i) = {z1, ..., z′
FS(w) on the samples S = {z1, ..., zi, ..., zn}. For any S(i) and S, there holds the following uniform
stability bound of ERM:

∀z ∈ Z,

(cid:13)
(cid:13)
(cid:13)∇f ( ˆw∗(S(i)); z) − ∇f ( ˆw∗(S); z)
(cid:13)
(cid:13)
(cid:13)2

≤

4M γ
nµ

.

Then, we present the application of our main sharper Theorem 3. In the strongly convex and smooth
case, we provide a up to O (cid:0)1/n2(cid:1) high probability excess risk guarantee valid for any algorithms
depending on the optimal population error F (w∗).
Theorem 4. Let assumptions in Theorem 3 and Lemma 3 hold. Suppose the function f is nonnegative.
Then for any δ ∈ (0, 1), when n ≥ 16γ2 log 6

, with probability at least 1 − δ, we have

δ

µ2

F ( ˆw) − F (w∗) ≲ F (w∗) log (1/δ)

n

+

log2 n log2(1/δ)
n2

.

248

Furthermore, assume F (w∗) = O( 1

n ), we have

F ( ˆw) − F (w∗) ≲ log2 n log2(1/δ)

n2

.

249

250

251

252

253

254

255

256

Remark 8. Theorem 4 shows that when the objective function f is µ-strongly convex, γ-smooth
and nonnegative, high probability risk bounds can even up to O (cid:0)1/n2(cid:1) for ERM. The most related
work to ours is Zhang et al. [2017]. They also obtain the O (cid:0)1/n2(cid:1)-type bounds for ERM by uniform
convergence of gradients approach. However, they need the sample number n = Ω(γd/µ), which
is related to the dimension d. Our risk bounds are dimension independent and only require the
sample number n = Ω(γ2/µ2). Comparing with Klochkov and Zhivotovskiy [2021], we add two
assumptions, smoothness and F (w∗) = O(1/n), but our bounds also tighter, from O(1/n) to
O (cid:0)1/n2(cid:1).

257

4.2 Projected Gradient Descent

258

259

260

261

262

263

264

265

266

Note that when the objective function f is strongly convex and smooth, the optimization error can be
ignored. However, the generalization analysis framework proposed by Klochkov and Zhivotovskiy
[2021] does not use smoothness assumption, which only derive high probability excess risk bound
of order O(1/n) after T = O(log n) steps under strongly convex and smooth assumptions. In this
subsection, we provide sharper risk bound under the same iteration steps, which is because our
generalization analysis also fully utilized the smooth assumptions. Here we give the definition of
PGD.
Definition 5 (Projected Gradient Descent). Let w1 = o ∈ Rd be an initial point and {ηt}t be a
sequence of positive step sizes. PGD updates parameters by

267

where ∇FS(wt) denotes a subgradient of F w.r.t. wt and ΠW is the projection operator onto W.

wt+1 = ΠW (wt − ηt∇FS (wt)) ,

7

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

Lemma 4 (Stability of Gradient Descent). Suppose the objective function f is µ-strongly-convex
and γ-smooth. For any w ∈ W and any z, suppose that ∥∇f (w; z)∥2 ≤ M . Let wi
t be the output of
FS(i)(w) on t-th iteration on the samples S(i) = {z1, ..., z′
i, ..., zn} in running PGD, and wt be the
output of FS(w) on t-th iteration on the samples S = {z1, ..., zi, ..., zn} in running PGD. Let the
constant step size ηt = 1/γ. For any S(i) and S, there holds the following uniform stability bound of
PGD:

∀z ∈ Z,

(cid:13)
(cid:13)
(cid:13)∇f ( ˆw∗(S(i)); z) − ∇f ( ˆw∗(S); z)
(cid:13)
(cid:13)
(cid:13)2

≤

4M γ
nµ

.

Remark 9. The derivations of Feldman and Vondrak [2019] in Section 4.1.2 (See also Hardt et al.
[2016] in Section 3.4) imply that if the objective function f is γ-smooth in addition to µ-strongly
(cid:17)
convexity and M -Lipschitz property, then PGD with the constant step size η = 1/γ is
-

(cid:16) 2M
nµ

uniformly argument stable for any number of steps, which means that PGD is
stable in gradients regardless of iteration steps.
Theorem 5. Let assumptions in Theorem 3 and Lemma 3 hold. Suppose the function f is nonnegative.
Let {wt}t be the sequence produced by PGD with ηt = 1/γ. Then for any δ ∈ (0, 1), when
n ≥ 16γ2 log 6

, with probability at least 1 − δ, we have

-uniformly-

δ

µ2

(cid:17)

(cid:16) 2M γ
nµ

F (w) − F (w∗) ≲

(cid:19)2T

(cid:18)

1 −

µ
γ

+

F (w∗) log (1/δ)
n

+

log2 n log2(1/δ)
n2

.

Furthermore, assume F (w∗) = O( 1

n ) and let T ≍ log n, we have
F ( ˆw) − F (w∗) ≲ log2 n log2(1/δ)

.

n2

(cid:16) F (w∗) log(1/δ)
n

Remark 10. Theorem 5 shows that under the same assumptions as Klochkov and Zhivotovskiy [2021],
(cid:17)
(cid:17)
our bound is O
,
we are sharper because F (w∗) is the minimal population risk, which is a common assumption
towards sharper risk bounds Srebro et al. [2010], Lei and Ying [2020], Liu et al. [2018], Zhang et al.
[2017], Zhang and Zhou [2019].

. Comparing with their bound O

+ log2 n log2(1/δ)
n2

(cid:16) log n log(1/δ)
n

288

4.3 Stochastic Gradient Descent

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

Stochastic gradient descent optimization algorithm has been widely used in machine learning due to
its simplicity in implementation, low memory requirement and low computational complexity per
iteration, as well as good practical behavior. Here we give the definition of standard SGD.
Definition 6 (Stochastic Gradient Descent). Let w1 = o ∈ Rd be an initial point and {ηt}t be a
sequence of positive step sizes. SGD updates parameters by

wt+1 = ΠW (wt − ηt∇f (wt; zit)) ,

where ∇f (wt; zit) denotes a subgradient of f w.r.t. wt and it is independently drawn from the
uniform distribution over [n] := {1, 2, . . . , n}.
Lemma 5 (Stability of SGD). Suppose the objective function f is µ-strongly-convex and γ-smooth.
For any w ∈ W and any z, suppose that ∥∇f (w; z)∥2 ≤ M . Let wi
t be the output of FS(i) (w) on
t-th iteration on the samples S(i) = {z1, ..., z′
i, ..., zn} in running PGD and and wt be the output of
FS(w) on t-th iteration on the samples S = {z1, ..., zi, ..., zn} in running SGD. For any S(i) and S,
there holds the following uniform stability bound of SGD:

(cid:13)
(cid:13)∇f (wt; z) − ∇f (wi

t; z)(cid:13)

(cid:13)2 ≤ 2γ

(cid:115)

2ϵopt(wt)
µ

+

4M γ
nµ

,

∀z ∈ Z,

301

where ϵopt(wt) = FS(wt) − FS( ˆw∗(S)) and ˆw∗(S) is the ERM of FS(w).

302

Next, we introduce a necessary assumption in stochastic optimization theory.

8

303

Assumption 1. Assume the existence of σ > 0 satisfying

Eit[∥∇f (wt; zit) − ∇FS(wt)∥2

2] ≤ σ2,

∀t ∈ N,

(6)

where Eit denotes the expectation w.r.t. it.
Remark 11. Assumption 1 is a standard assumption from the stochastic optimization theory [Ne-
mirovski et al., 2009, Ghadimi and Lan, 2013, Ghadimi et al., 2016, Kuzborskij and Lampert, 2018,
Zhou et al., 2018, Bottou et al., 2018, Lei and Tang, 2021], which essentially bounds the variance of
the stochastic gradients for dataset S.

Theorem 6. Let assumptions in Theorem 3 and Lemma 5 hold. Suppose Assumption 1 holds and the
function f is nonnegative. Let {wt}t be the sequence produced by SGD with ηt = η1t−θ, θ ∈ (0, 1)
and η1 < 1
, with probability at least 1 − δ, we have

2γ . Then for any δ ∈ (0, 1), when n ≥ 16γ2 log 6

µ2

δ

(cid:32) T

(cid:88)

ηt

(cid:33)−1 T

(cid:88)

ηt∥∇F (wt)∥2
2

t=1

t=1





(cid:16) log2 n log3(1/δ)
T −θ
(cid:16) log2 n log3(1/δ)
T − 1
(cid:16) log2 n log3(1/δ)
T θ−1

2

(cid:17)

(cid:17)

(cid:17)

O

O

O

+ O

+ O

+ O

(cid:16) log2 n log2(1/δ)
n2
(cid:16) log2 n log2(1/δ)
n2
(cid:16) log2 n log2(1/δ)
n2

+ F (w∗) log2(1/δ)
n
+ F (w∗) log2(1/δ)
n
+ F (w∗) log2(1/δ)
n

(cid:17)

(cid:17)

(cid:17)

,

,

,

if θ < 1/2

if θ = 1/2

if θ > 1/2.

=

+ F (w∗) log2(1/δ)
n

(cid:16) log2 n log3(1/δ)
n2

Remark 12. When θ < 1/2, we take T ≍ n2/θ. When θ = 1/2, we take T ≍ n4 and when θ > 1/2,
we set T ≍ n2/(1−θ). Then according to Theorem 6, the population risk of gradient is bounded by
O
. To the best of our knowledge, this is the first high probability
population gradient bound ∥∇F(wt)∥2 for SGD via algorithmic stability.
Theorem 7. Let Assumptions in Theorem 3 and Lemma 5 hold. Suppose Assumption 1 holds and the
function f is nonnegative. Let {wt}t be the sequence produced by SGD with ηt =
µ(t+t0) such that
(cid:111)
. Then for any δ > 0, when n ≥ 16γ2 log 6
and T ≍ n2, with probability at least

(cid:110) 4γ

(cid:17)

2

δ

µ2

t0 ≥ max
1 − δ, we have

µ , 1

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

F (wT +1) − F (w∗) = O

(cid:18) log4 n log5(1/δ)
n2

+

F (w∗) log(1/δ))
n

(cid:19)

.

320

Furthermore, assume F (w∗) = O( 1

n ), we have

F (wT +1) − F (w∗) = O

(cid:18) log4 n log5(1/δ)
n2

(cid:19)

.

321

322

323

324

325

326

327

Remark 13. Theorem 7 implies that high probability risk bounds for SGD optimization algorithm
can be up to O(1/n2) and the rate is dimension-free in high-dimensional learning problems. We
compare Theorem 7 with most related work. For algorithmic stability, high probability risk bounds in
Fan and Lei [2024] is up to O(1/n) when choosing optimal iterate number T for SGD optimization
algorithm. To the best of knowledge, we are faster than all the existing bounds. The best high
probability risk bounds of order O(1/n2) are given by Li and Liu [2021] via uniform convergence,
which require the sample number n = Ω(γd/µ) depending on dimension d.

328

5 Conclusion

329

330

331

332

333

334

In this paper, we improve a p-moment concentration inequality for sums of vector-valued functions.
By carefully constructing functions, we apply this moment concentration to derive sharper gener-
alization bounds in gradients in nonconvex problems, which can further be used to obtain sharper
high probability excess risk bounds for stable optimization algorithms. In application, we study three
common algorithms: ERM, PGD, SGD. To the best of our knowledge, we provide the sharpest high
probability dimension independent O(1/n2)-type for these algorithms.

9

335

References

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

R. Bassily, V. Feldman, C. Guzmán, and K. Talwar. Stability of stochastic gradient descent on nonsmooth convex
losses. In Proceedings of the 34th International Conference on Neural Information Processing Systems
(NeurIPS), volume 33, pages 4381–4391, 2020.

L. Bottou, F. E. Curtis, and J. Nocedal. Optimization methods for large-scale machine learning. SIAM review,

60(2):223–311, 2018.

O. Bousquet and A. Elisseeff. Stability and generalization. The Journal of Machine Learning Research, 2:

499–526, 2002.

O. Bousquet, Y. Klochkov, and N. Zhivotovskiy. Sharper bounds for uniformly stable algorithms. In Conference

on Learning Theory, pages 610–626. PMLR, 2020.

Z. Charles and D. Papailiopoulos. Stability and generalization of learning algorithms that converge to global

optima. In International conference on machine learning, pages 745–754. PMLR, 2018.

P. J. Davis. Gamma function and related functions. Handbook of mathematical functions, 256, 1972.
V. De la Pena and E. Giné. Decoupling: from dependence to independence. Springer Science & Business Media,

2012.

Z. Deng, H. He, and W. Su. Toward better generalization bounds with locally elastic stability. In International

Conference on Machine Learning, pages 2590–2600. PMLR, 2021.

A. Elisseeff, T. Evgeniou, M. Pontil, and L. P. Kaelbing. Stability of randomized learning algorithms. Journal of

Machine Learning Research, 6(1), 2005.

J. Fan and Y. Lei. High-probability generalization bounds for pointwise uniformly stable algorithms. Applied

and Computational Harmonic Analysis, 70:101632, 2024.

V. Feldman and J. Vondrak. Generalization bounds for uniformly stable algorithms. Advances in Neural

Information Processing Systems, 31, 2018.

V. Feldman and J. Vondrak. High probability generalization bounds for uniformly stable algorithms with nearly

optimal rate. In Conference on Learning Theory, pages 1270–1279. PMLR, 2019.

R. A. Fisher. On the mathematical foundations of theoretical statistics. Philosophical transactions of the Royal
Society of London. Series A, containing papers of a mathematical or physical character, 222(594-604):
309–368, 1922.

D. J. Foster, S. Greenberg, S. Kale, H. Luo, M. Mohri, and K. Sridharan. Hypothesis set stability and

generalization. Advances in Neural Information Processing Systems, 32, 2019.

S. Ghadimi and G. Lan. Stochastic first-and zeroth-order methods for nonconvex stochastic programming. SIAM

journal on optimization, 23(4):2341–2368, 2013.

S. Ghadimi, G. Lan, and H. Zhang. Mini-batch stochastic approximation methods for nonconvex stochastic

composite optimization. Mathematical Programming, 155(1):267–305, 2016.

M. Hardt, B. Recht, and Y. Singer. Train faster, generalize better: Stability of stochastic gradient descent. In

International conference on machine learning, pages 1225–1234. PMLR, 2016.

H. Karimi, J. Nutini, and M. Schmidt. Linear convergence of gradient and proximal-gradient methods under the

polyak-łojasiewicz condition. In ECML, pages 795–811. Springer, 2016.

A. J. Kleywegt, A. Shapiro, and T. Homem-de Mello. The sample average approximation method for stochastic

discrete optimization. SIAM Journal on optimization, 12(2):479–502, 2002.

Y. Klochkov and N. Zhivotovskiy. Stability and deviation optimal risk bounds with convergence rate o(1/n).

Advances in Neural Information Processing Systems, 34:5065–5076, 2021.

I. Kuzborskij and C. Lampert. Data-dependent stability of stochastic gradient descent. In Proceedings of the

35th International Conference on Machine Learning (ICML), pages 2815–2824. PMLR, 2018.

R. Latała and K. Oleszkiewicz. On the best constant in the khinchin-kahane inequality. Studia Mathematica,

109(1):101–104, 1994.

Y. Lei. Stability and generalization of stochastic optimization with nonconvex and nonsmooth problems. In The

Thirty Sixth Annual Conference on Learning Theory, pages 191–227. PMLR, 2023.

Y. Lei and K. Tang. Learning rates for stochastic gradient descent with nonconvex objectives. IEEE Transactions

on Pattern Analysis and Machine Intelligence, 43(12):4505–4511, 2021.

Y. Lei and Y. Ying. Fine-grained analysis of stability and generalization for stochastic gradient descent. In

International Conference on Machine Learning, pages 5809–5819. PMLR, 2020.

J. Li, X. Luo, and M. Qiao. On generalization error bounds of noisy gradient methods for non-convex learning.

In International Conference on Learning Representations, 2020.

S. Li and Y. Liu. Improved learning rates for stochastic optimization: Two theoretical viewpoints. arXiv preprint

arXiv:2107.08686, 2021.

M. Liu, X. Zhang, L. Zhang, R. Jin, and T. Yang. Fast rates of erm and stochastic approximation: Adaptive to

error bound conditions. Advances in Neural Information Processing Systems, 31, 2018.

10

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

T. Liu, G. Lugosi, G. Neu, and D. Tao. Algorithmic stability and hypothesis complexity. In International

Conference on Machine Learning, pages 2159–2167. PMLR, 2017.

B. London, B. Huang, and L. Getoor. Stability and generalization in structured prediction. The Journal of

Machine Learning Research, 17(1):7808–7859, 2016.

X. Luo and D. Zhang. Khintchine inequality on normed spaces and the application to banach-mazur distance.

arXiv preprint arXiv:2005.03728, 2020.

s. Lv, J. Wang, J. Liu, and Y. Liu. Improved learning rates of a functional lasso-type svm with sparse multi-kernel

representation. Advances in Neural Information Processing Systems, 34:21467–21479, 2021.

S. Mei, Y. Bai, and A. Montanari. The landscape of empirical risk for nonconvex losses. The Annals of Statistics,

46(6A):2747–2774, 2018.

A. Nemirovski, A. Juditsky, G. Lan, and A. Shapiro. Robust stochastic approximation approach to stochastic

programming. SIAM Journal on optimization, 19(4):1574–1609, 2009.

I. Pinelis. Optimum bounds for the distributions of martingales in banach spaces. The Annals of Probability,

pages 1679–1706, 1994.

A. Rakhlin, S. Mukherjee, and T. Poggio. Stability results in learning theory. Analysis and Applications, 3(04):

397–417, 2005.

A. Rakhlin, O. Shamir, and K. Sridharan. Making gradient descent optimal for strongly convex stochastic
optimization. In Proceedings of the 29th International Coference on International Conference on Machine
Learning, pages 1571–1578, 2012.

O. Rivasplata, E. Parrado-Hernández, J. S. Shawe-Taylor, S. Sun, and C. Szepesvári. Pac-bayes bounds for stable
algorithms with instance-dependent priors. Advances in Neural Information Processing Systems, 31, 2018.
S. Shalev-Shwartz and S. Ben-David. Understanding machine learning: From theory to algorithms. Cambridge

university press, 2014.

S. Shalev-Shwartz, O. Shamir, N. Srebro, and K. Sridharan. Learnability, stability and uniform convergence.

The Journal of Machine Learning Research, 11:2635–2670, 2010.

S. Smale and D.-X. Zhou. Learning theory estimates via integral operators and their approximations. Constructive

approximation, 26(2):153–172, 2007.

M. V. Solodov. Incremental gradient algorithms with stepsizes bounded away from zero. Computational

Optimization and Applications, 11:23–35, 1998.

N. Srebro, K. Sridharan, and A. Tewari. Optimistic rates for learning with a smooth loss. arXiv preprint

arXiv:1009.3896, 2010.

A. W. Van der Vaart. Asymptotic statistics, volume 3. Cambridge university press, 2000.
V. Vapnik and A. Chervonenkis. Theory of Pattern Recognition. 1974.
V. N. Vapnik. An overview of statistical learning theory. IEEE transactions on neural networks, 10(5):988–999,

1999.

S. Vaswani, F. Bach, and M. Schmidt. Fast and faster convergence of sgd for over-parameterized models and an
accelerated perceptron. In The 22nd international conference on artificial intelligence and statistics, pages
1195–1204. PMLR, 2019.

R. Vershynin. High-dimensional probability: An introduction with applications in data science, volume 47.

Cambridge university press, 2018.

Y. Xu and A. Zeevi. Towards optimal problem dependent generalization error bounds in statistical learning

theory. Mathematics of Operations Research, 2024.

L. Zhang and Z.-H. Zhou. Stochastic approximation of smooth and strongly convex functions: Beyond the o(1/t)

convergence rate. In Conference on Learning Theory, pages 3160–3179. PMLR, 2019.

L. Zhang, T. Yang, and R. Jin. Empirical risk minimization for stochastic convex optimization: O(1/n)-and

o(1/n**2)-type of risk bounds. In Conference on Learning Theory, pages 1954–1979. PMLR, 2017.

Y. Zhou, Y. Liang, and H. Zhang. Generalization error bounds with probabilistic guarantee for sgd in nonconvex

optimization. arXiv preprint arXiv:1802.06903, 2018.

11

441

442

443

A Additional definitions and lemmata

Lemma 6 (Equivalence of tails and moments for random vectors [Bassily et al., 2020]). Let X be a
random variable with

∥X∥p ≤

√

pa + pb

444

for some a, b ≥ 0 and for any p ≥ 2. Then for any δ ∈ (0, 1) we have, with probability at least 1 − δ,

(cid:18)

(cid:114)

|X| ≤ e

a

log

(cid:17)

(cid:16) e
δ

+ b log

(cid:19)

.

e
δ

445

446

447

448

Lemma 7 (Vector Bernstein’s inequality [Pinelis, 1994, Smale and Zhou, 2007]). Let {Xi}n
i=1 be
a sequence of i.i.d. random variables taking values in a real separable Hilbert space. Assume
that E[Xi] = µ, E[∥Xi − µ∥2] = σ2, and ∥Xi∥ ≤ M , ∀1 ≤ i ≤ n, then for all δ ∈ (0, 1), with
probability at least 1 − δ we have

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

1
n

n
(cid:88)

i=1

Xi − µ

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

(cid:115)

≤

2σ2 log( 2
δ )
n

+

M log 2
δ
n

.

449

450

451

Definition 7 (Weakly self-Bounded Function). Assume that a, b > 0. A function f : Z n (cid:55)→ [0, +∞)
is said to be (a, b)-weakly self-bounded if there exist functions fi : Z n−1 (cid:55)→ [0, +∞) that satisfies
for all Z n ∈ Z n,

n
(cid:88)

(fi(Z n) − f (Z n))2 ≤ af (Z n) + b.

i=1

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

Lemma 8 ([Klochkov and Zhivotovskiy, 2021]). Suppose that z1, . . . , zn are independent random
variables and the function f : Z n (cid:55)→ [0, +∞) is (a, b)-weakly self-bounded and the corresponding
function fi satisfy fi(Z n) ≥ f (Z n) for ∀i ∈ [n] and any Z n ∈ Z n. Then, for any t > 0,
(cid:19)

(cid:18)

P r(Ef (z1, . . . , zn) ≥ f (z1, . . . , zn) + t) ≤ exp

−

t2
2aEf (z1, . . . , zn) + 2b

.

Definition 8. A Rademacher random variable is a Bernoulli variable that takes values ±1 with
probability 1

2 each.

B Proofs of Section 3.1

The proof of Theorem 1 is motivated by Bousquet et al. [2020], which need the Marcinkiewicz-
Zygmund’s inequality for random variables taking values in a Hilbert space and the McDiarmid’s
inequality for vector-valued functions.

Firstly, we derive the optimal constants in the Marcinkiewicz-Zygmund’s inequality for random
variables taking values in a Hilbert space.

Lemma 9 (Marcinkiewicz-Zygmund’s Inequality for Random Variables Taking Values in a Hilbert
Space). Let X1, . . . , Xn be random variables taking values in a Hilbert space with E[Xi] = 0 for
all i ∈ [n]. Then for p ≥ 2 we have

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

n
(cid:88)

i=1

Xi

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)p

≤ 2 · 2

1
2p

(cid:32)

(cid:114) np
e

1
n

n
(cid:88)

i=1

(cid:13)
(cid:13)
(cid:13) ∥Xi∥

(cid:13)
p
(cid:13)
(cid:13)
p

(cid:33) 1

p

.

Remark 14. Comparing with Marcinkiewicz-Zygmund’s inequality given by Fan and Lei [2024],
we provide best constants. Next, we give the proof of Lemma 9.

The Marcinkiewicz-Zygmund’s inequality can be proved by using its connection to Khintchine-
Kahane’s inequality. Thus, we introduce the best constants in Khintchine-Kahane’s inequality for
random variables taking values from a Hilbert space here.

12

471

472

473

474

Lemma 10 (Best constants in Khintchine-Kahane’s inequality in Hilbert space [Latała and
Oleszkiewicz, 1994, Luo and Zhang, 2020]). For all p ∈ [2, ∞) and for all choices of Hilbert
space H, finite sets of vectors Xi, . . . , Xn ∈ X ∈ H, and independent Rademacher variables
r1, . . . , rn,

(cid:34)

E

475

where Cp = 2 1

2

(cid:26) Γ( p+1
2 )
π

√

(cid:27) 1

p

.

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

n
(cid:88)

i=1

riXi

p

p(cid:35) 1
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

≤ Cp ·

(cid:35) 1

2

∥Xi∥2

,

(cid:34) n
(cid:88)

i=1

476

477

478

479

Proof of Lemma 9. The symmetrization argument goes as follows: Let (r1, . . . , rn) be i.i.d. with
P(ri = 1) = P(ri = −1) = 1/2 and besides such that r1, . . . , rn and (X1, . . . , Xn) are independent.
Then by independence and symmetry, according to Lemma 1.2.6 of De la Pena and Giné [2012],
conditioning on (X1, . . . , Xn) yields

E

(cid:34)(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

n
(cid:88)

i=1

Xi

p(cid:35)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

= 2pE

(cid:34)(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

n
(cid:88)

i=1

riXi

p(cid:35)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

(cid:34)

≤2pE

E

(cid:34)(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

n
(cid:88)

i=1

riXi

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

p (cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

(cid:35)(cid:35)

X1, . . . , Xn

.

(7)

480

As for the conditional expectation in (7), notice that by independence
p (cid:12)
(cid:34)(cid:13)
(cid:12)
(cid:13)
(cid:12)
(cid:13)
(cid:12)
(cid:13)
(cid:12)
(cid:13)

X1 = x1, . . . , Xn = xn

(cid:34)(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

n
(cid:88)

riXi

= E

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

E

(cid:35)

i=1

n
(cid:88)

i=1

riXi

p(cid:35)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

(8)

481

482

According to Lemma 10, for vn-almost every x1, . . . , xn ∈ Rn, where vn := P ◦ (X1, . . . , Xn)−1
denotes the distribution of (X1, . . . , Xn), we have

(cid:34)

E

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

n
(cid:88)

i=1

rixi

p(cid:35)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

≤ C ·

(cid:35) p

2

∥xi∥2

,

(cid:34) n
(cid:88)

i=1

483

where C = 2

p
2

Γ( p+1
2 )
√
π

and C is optimal. This means that for any constant C ′ such that

(cid:34)

E

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

n
(cid:88)

i=1

rixi

p(cid:35)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

≤ C ′ ·

(cid:35) p

2

∥xi∥2

,

(cid:34) n
(cid:88)

i=1

484

485

for all n ∈ N and for each collection of vectors x1, . . . , xn, it follows that C ′ ≥ C.
From (8) and (9), we can infer that

E

(cid:34)(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

n
(cid:88)

i=1

riXi

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

p (cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

X1 = x1, . . . , Xn = xn

≤ C ·

(cid:35)

(cid:35) p

2

∥Xi∥2

.

(cid:34) n
(cid:88)

i=1

486

Taking expectations in the above inequalities and (7) yield that

E

(cid:34)(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

n
(cid:88)

i=1

Xi

p(cid:35)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

≤ C · E

(cid:35) p

2

∥Xi∥2

.

(cid:34) n
(cid:88)

i=1

(9)

(10)

(11)

487

488

To see optimality let the above statement hold for some constants C ′ in place of C. Then if we choose
Xi := xiri, 1 ≤ i ≤ n with arbitrary reals vectors x1, . . . , xn, it follows that

E

(cid:34)(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

n
(cid:88)

i=1

rixi

p(cid:35)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

≤ C ′ · E

(cid:35) p

2

∥xi∥2

,

(cid:34) n
(cid:88)

i=1

489

whence we can conclude from (10) that C ′ ≥ C. Thus we obtain that C ′ = C.

13

490

Notice that by Holder’s inequality

(cid:35) p

2

∥Xi∥2

≤ np/2−1

(cid:34) n
(cid:88)

i=1

n
(cid:88)

i=1

∥Xi∥p.

(12)

491

Plugging (12) into (11), we have

E

(cid:34)(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

n
(cid:88)

i=1

Xi

p(cid:35)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

≤ C · 2pnp/2−1 · E

(cid:35)

∥Xi∥p

,

(cid:34) n
(cid:88)

i=1

where C = 2

p
2

Γ( p+1
2 )
√
π

is a constant.

Next, we use the following form of Stirling’s formula for the Gamma-function, which follows from
(6.1.5), (6.1.15) and (6.1.38) in Davis [1972] to bound the constant C. For every x > 0, there exists a
µ(x) ∈ (0, 1/(12x)) such that

492

493

494

495

√

Γ(x) =

2πxx−1/2e−xeµ(x).

C = 2

p
2

(cid:1)

Γ (cid:0) p+1
2
√
π

√

= g(p)

2e−p/2pp/2,

ev(p)−1/2, where 0 < v(p) < 1/(6(p + 1)). By Taylor’s formula we have

496

Thus

497

498

with g(p) =
that

(cid:17)p/2

(cid:16)

1 + 1
p

log(1 + x) =

∞
(cid:88)

m=1

1
m

(−1)m−1xm,

∀x ∈ (−1, 1],

499

and that for every k ∈ N0

2k
(cid:88)

m=1

1
m

(−1)m−1xm ≤ log(1 + x) ≤

2k+1
(cid:88)

m=1

1
m

(−1)m−1xm, ∀x ≥ 0.

500

501

Therefor we obtain with k = 1 that
1
p

1
6p2 +
where the last equality follows from elementary calculus. Similarly,

) + v(p) −

log g(p) =

log(1 +

1
4p

≤ −

p
2

1
2

+

1
6(p + 1)

≤ −

1
18p

,

log g(p) =

p
2

log(1 +

1
p

) + v(p) −

1
2

≥ −

1
4p

+ v(p) ≥ −

1
4p

,

502

Thus, we have

√

e− 1

4p

2e−p/2pp/2 < C < e− 1
√

18p

√

2e−p/2pp/2,

503

504

505

which implies that C is strictly smaller than

2e−p/2pp/2 for all p ≥ 2.

√

Since C = 1
g(p)
√
2e−p/2pp/2 is equal to

2e−p/2pp/2 and g(p) ≥ e− 1

4p , we can obtain that the relative error between C and

1
g(p)

− 1 ≤ e− 1

4p − 1 ≤

1
4p

1
4p

e

using Mean Value Theorem. This implies that the corresponding relative errors between C and

506 √
507

2e−p/2pp/2 converge to zero as p tends to infinity.

The proof is complete.

508

509

14

510

511

512

513

514

515

Then we introduce the McDiarmid’s inequality for vector-valued functions. We firstly consider
real-valued functions, which follows from the standard tail-bound of McDiarmid’s inequality and
Proposition 2.5.2 in Vershynin [2018].
Lemma 11 (McDiarmid’s Inequality for real-valued functions). Let Zi, . . . , Zn be indepen-
dent random variables, and f : Z n (cid:55)→ R such that the following inequality holds for any
zi, . . . , zi−1, zi+1, . . . , zn
sup
zi,z′
i

|f (z1, . . . , zi−1, zi, zi+1, . . . , zn) − f (z1, . . . , zi−1, z′

i, zi+1, . . . , zn)| ≤ β,

516

Then for any p > 1 we have

∥f (Z1, . . . , Zn) − Ef (Z1, . . . , Zn)∥p ≤ (cid:112)2pnβ.

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

To derive the McDiarmid’s inequality for vector-valued functions, we need the expected distance
between f (Z1, . . . , Zn) and its expectation.
Lemma 12 ([Rivasplata et al., 2018]). Let Zi, . . . , Zn be independent random variables, and
f : Z n (cid:55)→ H is a function into a Hilbert space H such that the following inequality holds for
any zi, . . . , zi−1, zi+1, . . . , zn

sup
zi,z′
i
Then we have

∥f (z1, . . . , zi−1, zi, zi+1, . . . , zn) − f (z1, . . . , zi−1, z′

i, zi+1, . . . , zn)∥ ≤ β,

E [∥f (Z1, . . . , Zn) − Ef (Z1, . . . , Zn)∥] ≤

√

nβ.

Now, we can easily derive the p-norm McDiarmid’s inequality for vector-valued functions which
refines from Fan and Lei [2024] with better constants.
Lemma 13 (McDiarmid’s inequality for vector-valued functions). Let Zi, . . . , Zn be independent
random variables, and f : Z n (cid:55)→ H is a function into a Hilbert space H such that the following
inequality holds for any zi, . . . , zi−1, zi+1, . . . , zn

∥f (z1, . . . , zi−1, zi, zi+1, . . . , zn) − f (z1, . . . , zi−1, z′

i, zi+1, . . . , zn)∥ ≤ β,

(13)

sup
zi,z′
i

528

Then for any p > 1 we have

∥∥f (Z1, . . . , Zn) − Ef (Z1, . . . , Zn)∥∥p ≤ ((cid:112)2p + 1)

√

nβ.

Proof of Lemma 13. Define a real-valued function h : Z n (cid:55)→ R as

h(z1, . . . , zn) = ∥f (z1, . . . , zn) − E[f (Z1, . . . , Zn)]∥.

529

530

531

532

533

534

535

|h(z1, . . . , zi−1, zi, zi+1, . . . , zn) − h(z1, . . . , zi−1, z′

We notice that this function satisfies the increment condition. For any i and z1, . . . , zi−1, zi+1, . . . , zn,
we have
sup
zi,z′
i
= sup
zi,z′
i
≤ sup
zi,z′
i

|∥f (z1, . . . , zn) − E[f (Z1, . . . , Zn)]∥ − ∥f (z1, . . . , zi−1, z′

i, zi+1, . . . , zn) − E[f (Z1, . . . , Zn)]∥|

|∥f (z1, . . . , zn) − f (z1, . . . , zi−1, z′

i, zi+1, . . . , zn)∥ ≤ β.

i, zi+1, . . . , zn)|

Therefore, we can apply Lemma 11 to the real-valued function h and derive the following inequality
∥h(Z1, . . . , Zn) − E[h(Z1, . . . , Zn)]∥p ≤ (cid:112)2pnβ.

According to Lemma 12, we know the following inequality E[h(Z1, . . . , Zn)] ≤
above two inequalities together and we can derive the following inequality
∥∥f (Z1, . . . , Zn) − Ef (Z1, . . . , Zn)∥∥p

√

nβ. Combing the

≤∥h(Z1, . . . , Zn) − E[h(Z1, . . . , Zn)]∥p + ∥E[h(Z1, . . . , Zn)]∥p
≤((cid:112)2p + 1)

nβ.

√

The proof is complete.

15

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

1
Proof of Theorem 1. For g(Z1, . . . , Zn) and A ⊂ [n], we write ∥∥g∥∥p(ZA) = (E [∥f ∥p ZA])
p .
Without loss of generality, we suppose that n = 2k. Otherwise, we can add extra functions equal to
zero, increasing the number of therms by at most two times.

Consider a sequence of partitions P0, . . . , Pk with P0 = {{i} : i ∈ [n]}, Pk with Pn = {[n]}, and
to get Pl from Pl+1 we split each subset in Pl+1 into two equal parts. We have

P0 = {{1}, . . . , {2k}}, P1 = {{1, 2}, {3, 4}, . . . , {2k − 1, 2k}}, Pk = {{1, . . . , 2k}}.

We have |Pl| = 2k−l and |P | = 2l for each P ∈ Pl. For each i ∈ [n] and l = 0, . . . , k, denote by
P l(i) ∈ Pl the only set from Pl that contains i. In particular, P 0(i) = {i} and P K(i) = [n].
For each i ∈ [n] and every l = 0, . . . , k consider the random variables

i = gl
gl

i(Zi, Z[n]\P l(i)) = E[gi|Zi, Z[n]\P l(i)],

i.e. conditioned on zi and all the variables that are not in the same set as Zi in the partition Pl. In
particular, g0

i = E[gi|Zi]. We can write a telescopic sum for each i ∈ [n],

i = gi and gk

546

Then, by the triangle inequality

gi − E[gi|Zi] =

k−1
(cid:88)

l=1

i − gl+1
gl
i

.

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

n
(cid:88)

i=1

gi

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)p
(cid:13)

≤

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

n
(cid:88)

i=1

(cid:13)
(cid:13)
(cid:13)
(cid:13)
E[gi|Zi]
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)p
(cid:13)

+

k−1
(cid:88)

l=0

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

n
(cid:88)

i=1

i − gl+1
gl
i

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)p
(cid:13)

.

(14)

547

548

549

To bound the first term, since ∥E[gi|Zi]∥ ≤ M , we can check that the vector-valued function
f (Z1, . . . , Zn) = (cid:80)n
E[gi|Zi] satisfies (13) with β = 2M , and E[E[gi|Zi]] = 0, applying Lemma
13 with β = 2M , we have

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

n
(cid:88)

i=1

E[gi|Zi]

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)p
(cid:13)

≤ 2((cid:112)2p + 1)

√

nM.

(15)

550

551

552

553

554

555

556

557

558

Then we start to bound the second term of the right hand side of (14). Observe that
(cid:3) ,

(Zi, Z[n]\P l+1(i)) = E (cid:2)gl

i(Zi, Z[n]\P l(i))(cid:12)

(cid:12)Zi, Z[n]\P l+1(i)

gl+1
i

i by β. Therefore, we apply Lemma 13 with f = gl

where the expectation is taken with respect to the variables Zj, j ∈ P l+1(i)\P l(i). Changing any
i where there are 2l random
Zj would change gl
variables and obtain a uniform bound
(cid:13)
(cid:13)
(cid:13)gl
(cid:13)

i − gl+1
i
(cid:13)
Taking integration over (Zi, Z[n]\P l+1(i)), we have (cid:13)
(cid:13)gl
(cid:13)
Next, we turn to the sum (cid:80)
for i ∈ P l depends
only on Zi, Z[n]\P l , the terms are independent and zero mean conditioned on Z[n]\P l . Applying
Lemma 9, we have for any p ≥ 2,

(cid:13)p (Zi, Z[n]\P l+1(i)) ≤ ((cid:112)2p + 1)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
i − gl+1
(cid:13)p ≤ (
(cid:13)
i
for any P l ∈ Pl. Since gl

2lβ as well.

i − gl+1
i

i − gl+1
i

i∈P l gl

∀p ≥ 2,

2p + 1)

2lβ,

√

√

√

(cid:88)

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

i − gl+1
gl
i

(cid:13)
(cid:13)
p
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
p
(cid:13)
Integrating with respect to (Z[n]\P l ) and using (cid:13)
(cid:13)gl
(cid:13)

(Z[n]\P l ) ≤

2 · 2

i∈P l

(cid:114)

(cid:32)

1
2p

2lp
e

(cid:33)p

1
2l

(cid:88)

i∈P l

(cid:13)
(cid:13)
(cid:13)gl
(cid:13)

i − gl+1
i

(cid:13)
(cid:13)

(cid:13)
p
p (Z[n]\P l ).
(cid:13)

√

(cid:13)
(cid:13)
(cid:13)p ≤ (
(cid:13)

2p + 1)

√

2lβ, we have

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

(cid:88)

i − gl+1
gl
i

i∈P l

(cid:32)

≤

2 · 2

1
2p

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
(cid:13)p
(cid:13)

i − gl+1
i
(cid:33)

(cid:114)

2lp
e

1

2l × 2l((cid:112)2p + 1)

√

2lβ

=21+ 1

2p

(cid:19)

(cid:18)(cid:114) p
e

((cid:112)2p + 1)2lβ.

16

559

Then using triangle inequality over all sets P l ∈ Pl, we have

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

(cid:88)

i − gl+1
gl
i

i∈[n]

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
(cid:13)p
(cid:13)

(cid:88)

≤

P l∈Pl

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

(cid:88)

i − gl+1
gl
i

i∈P l

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
(cid:13)p
(cid:13)

≤2k−l × 21+ 1

2p

(cid:19)

(cid:18)(cid:114) p
e

((cid:112)2p + 1)2lβ

≤21+ 1

2p

(cid:19)

(cid:18)(cid:114) p
e

((cid:112)2p + 1)2kβ.

560

Recall that 2k ≤ n due to the possible extension of the sample. Then we have
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)p
(cid:13)

(cid:18)(cid:114) p
e

i − gi+1
gl
i

≤ 22+ 1

((cid:112)2p + 1)nβ ⌈log2 n⌉ .

k−1
(cid:88)

n
(cid:88)

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

i=0

i=1

(cid:19)

2p

561

We can plug the above bound together with (15) into (14), to derive the following inequality

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

n
(cid:88)

i=1

gi

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)p
(cid:13)

≤ 2((cid:112)2p + 1)

√

nM + 22+ 1

2p

(cid:19)

(cid:18)(cid:114) p
e

((cid:112)2p + 1)nβ ⌈log2 n⌉ .

562

563

564

565

566

567

568

The proof is completed.

C Proofs of Section 3

Proof of Theorem 2. Let S = {z1, . . . , zn} be a set of independent random variables each taking
values in Z and S′ = {z′
n} be its independent copy. For any i ∈ [n], define S(i) =
{zi, . . . , zi−1, z′
i, zi+1, . . . , zn} be a dataset replacing the i-th sample in S with another i.i.d. sample
z′
i. Then we can firstly write the following decomposition

1, . . . , z′

n∇F (A(S)) − n∇FS(A(S))

=

n
(cid:88)

i=1

EZ

(cid:104)
∇f (A(S); Z)] − Ez′

i

(cid:104)

∇f (A(S(i)), Z)

(cid:105)(cid:105)

+

n
(cid:88)

i=1

(cid:104)

(cid:104)
EZ

Ez′

i

∇f (A(S(i)), Z)

(cid:105)

− ∇f (A(S(i)), zi)

(cid:105)

+

n
(cid:88)

i=1

Ez′

i

(cid:104)

(cid:105)
∇f (A(S(i)), zi)

−

n
(cid:88)

i=1

∇f (A(S), zi).

569

We denote that gi(S) = Ez′

i

(cid:2)EZ

(cid:2)∇f (A(S(i)), Z)(cid:3) − ∇f (A(S(i)), zi)(cid:3), thus we have

∥n∇F (A(S)) − n∇FS(A(S))∥2
(cid:13)
(cid:13)
(cid:104)
∇f (A(S(i)), Z)
(cid:13)
(cid:13)
(cid:13)

∇f (A(S); Z)] − Ez′

n
(cid:88)

EZ

(cid:104)

i

(cid:105)(cid:105)

=

i=1
n
(cid:88)

+

(cid:104)

EZ

(cid:105)
(cid:104)
∇f (A(S(i)), Z)

(cid:105)
− ∇f (A(S(i)), zi)

Ez′

i

n
(cid:88)

i=1

∇f (A(S), zi)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)2

i=1

+

n
(cid:88)

i=1

(cid:104)

(cid:105)
∇f (A(S(i)), zi)

−

Ez′

i

≤ 2nβ +

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)

n
(cid:88)

i=1

gi(S)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)2

,

17

(16)

570

571

where the inequality holds from the definition of uniform stability in gradients.

According to our assumptions, we get ∥gi(S)∥2 ≤ 2M and

Ezi[gi(S)] = Ezi

= Ez′

i

Ez′
(cid:104)

i

EZ

(cid:104)
EZ
(cid:104)

(cid:104)

∇f (A(S(i)); Z)

(cid:105)

− ∇f (A(S(i)); zi)

(cid:105)

(cid:105)
∇f (A(S(i)); Z)

(cid:104)

− Ezi

∇f (A(S(i)); zi)

(cid:105)(cid:105)

= 0,

572

573

where this equality holds from the fact that zi and Z follow from the same distribution. For any
i ∈ [n], any j ̸= i and any z′′

j , we have

(cid:104)

≤

j , zj+1, . . . , zn)(cid:13)
(cid:13)
(cid:13)gi(z1, . . . , zj−1, zj, zj+1, . . . , zn) − gi(z1, . . . , zj−1, z′′
(cid:13)2
(cid:13)
(cid:105)
(cid:104)
∇f (A(S(i)
∇f (A(S(i)); Z)
Ez′
EZ
(cid:13)
(cid:13)
(cid:13)
(cid:104)
EZ
(cid:13)
(cid:13)
≤2β,

∇f (A(S(i)); Z) − ∇f (A(S(i)

− Ez′
i
(cid:13)
(cid:13)
(cid:13)

− ∇f (A(S(i)); zi)

(cid:104)
EZ
(cid:104)
EZ

(cid:105)(cid:105)(cid:13)
(cid:13)
(cid:13)2

j ); Z)

Ez′

Ez′

+

≤

(cid:104)

(cid:105)

(cid:104)

(cid:104)

i

i

i

(cid:105)
j ); Z)

− ∇f (A(S(i)

j ); zi)

(cid:105)(cid:13)
(cid:13)
(cid:13)2

(cid:105)
∇f (A(S(i)); Z)

− ∇f (A(S(i)

j ); zi)

(cid:105)(cid:13)
(cid:13)
(cid:13)2

574

575

where S(i) = {zi, . . . , zi−1, z′
Theorem 1 are satisfied for gi(S). We have the following result for any p > 2

i, zi+1, . . . , zn}. Thus, we have verified that three conditions in

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

n
(cid:88)

i=1

gi(S)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)p
(cid:13)

≤ 4((cid:112)2p + 1)

√

nM + 8 × 2

(cid:19)

1
4

(cid:18)(cid:114) p
e

((cid:112)2p + 1)nβ ⌈log2 n⌉ .

576

We can combine the above inequality and (16) to derive the following inequality

n ∥∥∇F (A(S)) − n∇FS(A(S))∥∥p

≤2nβ + 4((cid:112)2p + 1)

√

nM + 8 × 2

(cid:19)

1
4

(cid:18)(cid:114) p
e

((cid:112)2p + 1)nβ ⌈log2 n⌉ .

577

According to Lemma 6 for any δ ∈ (0, 1), with probability at least 1 − δ, we have

n∥∇F (A(S)) − ∇FS(A(S))∥2

≤2nβ + 4

√

nM + 8 × 2

3
4

578

This implies that

√

enβ ⌈log2 n⌉ log (e/δ) + (4e

2nM + 8 × 2

√

1
4

√

enβ ⌈log2 n⌉)(cid:112)log e/δ.

∥∇F (A(S)) − ∇FS(A(S))∥2
(cid:17)
1 + e(cid:112)2 log (e/δ)

4M

(cid:16)

≤2β +

√

n

579

The proof is completed.

+ 8 × 2

√

1
4 (

2 + 1)

√

eβ ⌈log2 n⌉ log (e/δ).

580

Proof of Theorem 3. We can firstly write the following decomposition

n∇F (A(S)) − n∇FS(A(S))

=

n
(cid:88)

i=1

EZ

(cid:104)
∇f (A(S); Z)] − Ez′

i

(cid:104)

∇f (A(S(i)), Z)

(cid:105)(cid:105)

+

n
(cid:88)

i=1

(cid:104)

(cid:104)
EZ

Ez′

i

∇f (A(S(i)), Z)

(cid:105)

− ∇f (A(S(i)), zi)

(cid:105)

+

n
(cid:88)

i=1

Ez′

i

(cid:104)

(cid:105)
∇f (A(S(i)), zi)

−

n
(cid:88)

i=1

∇f (A(S), zi).

18

581

We denote that hi(S) = Ez′

i

(cid:2)EZ

(cid:2)∇f (A(S(i)), Z)(cid:3) − ∇f (A(S(i)), zi)(cid:3), we have

n∇F (A(S)) − n∇FS(A(S)) −

n
(cid:88)

i=1

hi(S)

=

n
(cid:88)

i=1

(cid:104)

EZ

∇f (A(S); Z)] − Ez′

i

(cid:104)

∇f (A(S(i)), Z)

(cid:105)(cid:105)

+

n
(cid:88)

i=1

Ez′

i

(cid:104)

(cid:105)
∇f (A(S(i)), zi)

−

n
(cid:88)

i=1

∇f (A(S), zi),

582

which implies that

=

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

n∇F (A(S)) − n∇FS(A(S)) −

n
(cid:88)

i=1

hi(S)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)2

n
(cid:88)

(cid:104)

EZ

∇f (A(S); Z)] − Ez′

i

(cid:104)

∇f (A(S(i)), Z)

(cid:105)(cid:105)

i=1

n
(cid:88)

+

Ez′

i

(cid:104)

(cid:105)
∇f (A(S(i)), zi)

−

n
(cid:88)

i=1

∇f (A(S), zi)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)2

i=1
≤ 2nβ,

(17)

583

584

585

586

where the inequality holds from the definition of uniform stability in gradients.
Then, for any i = 1, . . . , n, we define qi(S) = hi(S) − ES{zi}[hi(S)]. It is easy to verify that
ES\{zi}[qi(S)] = 0 and Ezi[hi(S)] = Ezi[qi(S)] − Ezi
ES\{zi}[qi(S)] = 0 − 0 = 0. Also, for any
j ∈ [n] with j ̸= i and z′′

j ∈ Z, we have the following inequality

∥qi(S) − qi(z1, . . . , zj−1, z′′
≤∥hi(S) − hi(z1, . . . , zj−1, z′′

j , zj+1, . . . , zn)∥2
j , zj+1, . . . , zn)∥2

+ ∥ES\{zi}[hi(S)] − ES\{zi}[hi(1, . . . , zj−1, z′′

j , zj+1, . . . , zn)]∥2.

587

588

589

590

591

592

593

For the first term ∥hi(S) − hi(z1, . . . , zj−1, z′′
j , zj+1, . . . , zn)∥2, it can be bounded by 2β according
to the definition of uniform stability. Similar result holds for the second term ∥ES\{zi}[hi(S)] −
ES\{zi}[hi(1, . . . , zj−1, z′′
j , zj+1, . . . , zn)]∥2 according to the uniform stability. By a combina-
tion of the above analysis, we get ∥qi(S) − qi(1, . . . , zj−1, z′′
j , zj+1, . . . , zn)∥2 ≤ ∥hi(S) −
hi(1, . . . , zj−1, z′′

j , zj+1, . . . , zn)∥2 ≤ 4β.

Thus, we have verified that three conditions in Theorem 1 are satisfied for qi(S). We have the
following result for any p ≥ 2

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

n
(cid:88)

i=1

qi(S)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)p

≤ 24+ 1

4

(cid:19)

(cid:18)(cid:114) p
e

((cid:112)2p + 1)nβ ⌈log2 n⌉ .

(18)

594

Furthermore, we can derive that

n∇F (A(S)) − n∇FS(A(S)) −

=n∇F (A(S)) − n∇FS(A(S)) −

n
(cid:88)

i=1
n
(cid:88)

hi(S) +

n
(cid:88)

i=1

qi(S)

ES\{zi}[hi(S)]

i=1
=n∇F (A(S)) − n∇FS(A(S)) − nES′[∇F (A(S′))] + nES[∇F (A(S))].

19

595

596

Due to the i.i.d. property between S and S′, we know that ES′[∇F (A(S′))] = ES[∇F (A(S))].
Thus, combined above equality, (17) and (18), we have

n
(cid:88)

hi(S)

n∇F (A(S)) − n∇FS(A(S)) −

∥∥n∇F (A(S)) − n∇FS(A(S)) − nES[∇F (A(S))] + nES′[∇FS(A(S′))]∥∥p
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
(cid:13)
(cid:13)
(cid:13)
(cid:13)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
hi(S) − nES[∇F (A(S))] + nES′FS[A(S′)]
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)p
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

n∇F (A(S)) − n∇FS(A(S)) −

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)p
(cid:13)

hi(S)

qi(S)

n
(cid:88)

n
(cid:88)

n
(cid:88)

i=1

i=1

+

i=1

i=1

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)p
(cid:13)

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:13)p
(cid:13)

≤

+

=

≤2nβ + 24+ 1

4

(cid:19)

(cid:18)(cid:114) p
e

≤16 × 2

3
4

(cid:33)

(cid:32)(cid:114) 1
e

((cid:112)2p + 1)nβ ⌈log2 n⌉
(cid:33)
(cid:32)(cid:114) 1
e

pnβ ⌈log2 n⌉ + 32

√

pnβ ⌈log2 n⌉ .

597

According to Lemma 6 for any δ ∈ (0, 1), with probability at least 1 − δ/3, we have

∥∇F (A(S)) − ∇FS(A(S))∥2

≤∥ES′[∇FS(A(S′))] − ES[∇F (A(S))]∥2

+ 16 × 2

3
4

√

eβ ⌈log2 n⌉ log (3e/δ) + 32

√

eβ ⌈log2 n⌉ (cid:112)log 3e/δ.

(19)

598

599

600

Next, we need to bound the term ∥ES′[∇FS(A(S′))] − ES[∇F (A(S))]∥2. There holds that
∥ESES′[∇FS(A(S′))]∥2 = ∥ES[∇F (A(S))]∥2. Then, by the Bernstein inequality in Lemma 7, we
obtain the following inequality with probability at least 1 − δ/3,

(cid:13)ES′ [∇FS(A(S′))] − ES[∇F (A(S))](cid:13)
(cid:13)
(cid:13)2

≤

(cid:115)

2Ezi [∥ES′ ∇f (A(S′); zi)∥2

2] log 6
δ

n

+

M log 6
δ
n

.

(20)

601

Then using Jensen’s inequality, we have

Ezi[∥ES′∇f (A(S′); zi)∥2

2] ≤ Ezi

ES′∥∇f (A(S′); zi)∥2
2
= EZES′∥∇f (A(S′); Z)∥2
2
= EZES∥∇f (A(S); Z)∥2
2.

602

Combing (19), (20) with (21), we finally obtain that with probability at least 1 − 2δ/3,

∥∇F (A(S)) − ∇FS(A(S))∥2
(cid:115)

2EZES∥∇f (A(S); Z)∥2

2 log 6
δ

≤

n

+ 16 × 2

√

3
4

+

M log 6
δ
n

eβ ⌈log2 n⌉ log (3e/δ) + 32

√

eβ ⌈log2 n⌉ (cid:112)log 3e/δ.

(21)

(22)

603

604

Next, since S = {zi, . . . , zn}, we define p = p(z1, . . . , zn) = EZ[∥∇f (A(S); Z)∥2
2] and pi =
pi(z1, . . . , zn) = supzi∈Z p(zi, . . . , zn). So there holds pi ≥ p for any i = 1, . . . , n and any

20

605

{z1, . . . , zn} ∈ Z n. Also, there holds that

n
(cid:88)

i=1
n
(cid:88)

i=1
n
(cid:88)

i=1
n
(cid:88)

i=1
n
(cid:88)

i=1

=

≤

=

≤

(pi − p)2

(cid:18)

(cid:18)

sup
zi∈Z
(cid:20)

EZ

(cid:18)

EZ

(cid:18)

EZ

β2

sup
zi∈Z

(cid:20)(cid:18)

EZ [∥∇f (A(S′); Z)∥2

2] − EZ [∥∇f (A(S); Z)∥2
2]

(cid:19)2

∥∇f (A(S′); Z)∥2

2 − ∥∇f (A(S); Z)∥2
2

(cid:21)(cid:19)2

∥∇f (A(S′); Z)∥2 − ∥∇f (A(S); Z)∥2

∥∇f (A(S′); Z)∥2 + ∥∇f (A(S); Z)∥2

(cid:19)(cid:21)(cid:19)2

(cid:19) (cid:18)

sup
zi∈Z

(cid:21)(cid:19)2

sup
zi∈Z
(cid:20)

∥∇f (A(S); Z)∥2 + sup
zi∈Z

∥∇f (A(S); Z)∥2

≤nβ2 (2EZ [∥∇f (A(S); Z)∥2 + β])2
≤8nβ2p + 2nβ4,

606

607

608

609

610

where the first inequality follows from the Jensen’s inequality. The second and third inequalities
follow from the definition of uniform stability in gradients. The last inequality holds from that
(a + b)2 ≤ 2a2 + 2b2.
From (23), we know that p is (8nβ2, 2nβ4) weakly self-bounded. Thus, by Lemma 8, we obtain that
with probability at least 1 − δ/3,

(23)

EZES[∥∇f (A(S); Z)∥2
(cid:113)

2] − EZ[∥∇f (A(S); Z)∥2
2]

≤

(16nβ2ESEZ[∥∇f (A(S); Z)∥2

2] + 4nβ4) log(3/δ)

(cid:114)

=

≤

1
2

(ESEZ[∥∇f (A(S); Z)∥2

2] +

β2)16nβ2 log(3/δ)

(ESEZ[∥∇f (A(S); Z)∥2

2] +

β2) + 8nβ2 log(3/δ),

1
4
1
4

√

ab ≤ a+b
2

611

where the last inequality follows from that

for all a, b > 0. Thus, we have

EZES[∥∇f (A(S); Z)∥2

2] ≤ 2EZ[∥∇f (A(S); Z)∥2

2] +

1
4

β2 + 16nβ2 log(3/δ).

(24)

612

Substituting (24) into (22), we finally obtain that with probability at least 1 − δ

∥∇F (A(S)) − ∇FS(A(S))∥2
(cid:115)
2 (cid:0)2EZ[∥∇f (A(S); Z)∥2

≤

4 β2 + 16nβ2 log(3/δ)(cid:1) log 6
2] + 1
n

M log 6
δ
n
eβ ⌈log2 n⌉ (cid:112)log 3e/δ.

√

+

δ

eβ ⌈log2 n⌉ log (3e/δ) + 32

+ 16 × 2

√

3
4

(25)

613

According to inequality

√

a + b =

√

√

a +

b for any a, b > 0, with probability at least 1 − δ, we have

∥∇F (A(S)) − ∇FS(A(S))∥2
(cid:115)

4EZ[∥∇f (A(S); Z)∥2

2] log 6
δ

≤

(cid:115) (cid:0) 1

+

n

+ 16 × 2

√

3
4

The proof is complete.

614

615

eβ ⌈log2 n⌉ log (3e/δ) + 32

21

2 β2 + 32nβ2 log(3/δ)(cid:1) log 6
n

δ

+

M log 6
δ
n

√

eβ ⌈log2 n⌉ (cid:112)log 3e/δ.

616

617

Proof of Remark 4. According to the proof in Theorem 3, we have the following inequality that with
probability at least 1 − δ

∥∇F (A(S)) − ∇FS(A(S))∥2
(cid:115)

4EZ[∥∇f (A(S); Z)∥2

2] log 6
δ

≤

(cid:115) (cid:0) 1

+

n

+ 16 × 2

√

3
4

eβ ⌈log2 n⌉ log (3e/δ) + 32

√

eβ ⌈log2 n⌉ (cid:112)log 3e/δ.

2 β2 + 32nβ2 log(3/δ)(cid:1) log 6
n

δ

+

M log 6
δ
n

(26)

618

Since f (w) is γ-smooth, we have

EZ[∥∇f (A(S); Z)∥2
2]

≤EZ[∥∇f (A(S); Z) − ∇f (w∗; Z)∥2
≤γ2∥A(S) − w∗∥2

2 + EZ[∥∇f (w∗; Z)∥2
2]

2 + ∥∇f (w∗; Z)∥2
2]

619

Plugging (27) into (26), we have

∥∇F (A(S)) − ∇FS(A(S))∥2
(cid:115)

4(γ2∥A(S) − w∗∥2

2 + EZ[∥∇f (w∗; Z)∥2

2]) log 6
δ

≤

(cid:115) (cid:0) 1

2 β2 + 32nβ2 log(3/δ)(cid:1) log 6
n

δ

+

+

M log 6
δ
n

+ 16 × 2

3
4

eβ ⌈log2 n⌉ log (3e/δ) + 32

√

eβ ⌈log2 n⌉ (cid:112)log 3e/δ

n
√

(27)

(28)

4EZ[∥∇f (w∗; Z)∥2

2] log 6
δ

≤2γ∥A(S) − w∗∥2

(cid:115)

(cid:115)

+

log 6
δ
n
2 β2 + 32nβ2 log(3/δ)(cid:1) log 6
n

δ

(cid:115) (cid:0) 1

+

n

+

M log 6
δ
n

√

+ 16 × 2

3
4

√

eβ ⌈log2 n⌉ log (3e/δ) + 32

√

eβ ⌈log2 n⌉ (cid:112)log 3e/δ,
√
a +

√

620

where the second inequality holds because

a + b +

b for any a, b > 0, which means that

∥∇F (A(S)) − ∇FS(A(S))∥2

≲β log n log(1/δ) +

log(1/δ)
n

+

(cid:114) EZ [∇∥f (w∗; Z)∥2

2] log(1/δ)

n

(cid:114)

+ ∥A(S) − w∗∥

log(1/δ)
n

.

The proof is complete.

621

622

623

Proof of Lemma 1. Inequality (28) implies that

n
√

∥∇F (A(S))∥2 − ∥∇FS(A(S))∥2
(cid:115)

4(γ2∥A(S) − w∗∥2

2 + EZ[∥∇f (w∗; Z)∥2

2]) log 6
δ

≤

(cid:115) (cid:0) 1

2 β2 + 32nβ2 log(3/δ)(cid:1) log 6
n

δ

+

+

M log 6
δ
n

+ 16 × 2

3
4

eβ ⌈log2 n⌉ log (3e/δ) + 32

√

eβ ⌈log2 n⌉ (cid:112)log 3e/δ

≤2γ∥A(S) − w∗∥2

(cid:115)

(cid:115)

log 6
δ
n

+

4EZ[∥∇f (w∗; Z)∥2

2] log 6
δ

n

+

M log 6
δ
n

+ 16 × 2

√

3
4

eβ ⌈log2 n⌉ log (3e/δ) + 32

√

(cid:115) (cid:0) 1

+

2 β2 + 32nβ2 log(3/δ)(cid:1) log 6
n
eβ ⌈log2 n⌉ (cid:112)log 3e/δ,

δ

624

625

When F (w) satisfies the PL condition, there holds the following error bound property (refer to
Theorem 2 in Karimi et al. [2016])

∥∇F (A(S))∥2 ≥ µ∥A(S) − w∗∥2.

22

626

Thus, we have

µ∥A(S) − w∗∥2 ≤ ∥∇F (A(S))∥2

≤∥∇FS(A(S))∥2 + 2γ∥A(S) − w∗∥2

(cid:115)

(cid:115)

log 6
δ
n

+

4EZ[∥∇f (w∗; Z)∥2

2] log 6
δ

n

(cid:115) (cid:0) 1

+

2 β2 + 32nβ2 log(3/δ)(cid:1) log 6
n

δ

+

M log 6
δ
n

√

+ 16 × 2

3
4

√

eβ ⌈log2 n⌉ log (3e/δ) + 32

eβ ⌈log2 n⌉ (cid:112)log 3e/δ.

627

When n ≥ 16γ2 log 6

δ

µ2

, we have 2γ

(cid:113) log 6

δ

n ≤ µ

2 , then we can derive that

µ∥A(S) − w∗∥2 ≤ ∥∇F (A(S))∥2

≤∥∇FS(A(S))∥2 +

µ
2

∥A(S) − w∗∥2 +

(cid:115) (cid:0) 1

+

2 β2 + 32nβ2 log(3/δ)(cid:1) log 6
n

δ

(cid:115)

+

4EZ[∥∇f (w∗; Z)∥2

2] log 6
δ

n

M log 6
δ
n

√

√

eβ ⌈log2 n⌉ log (3e/δ) + 32

eβ ⌈log2 n⌉ (cid:112)log 3e/δ.

+ 16 × 2

3
4

628

This implies that

∥A(S) − w∗∥2

(cid:115)

4EZ[∥∇f (w∗; Z)∥2

2] log 6
δ

≤

(cid:16)

2
µ

∥∇FS(A(S))∥2 +

(cid:115) (cid:0) 1

+

n
2 β2 + 32nβ2 log(3/δ)(cid:1) log 6
n

δ

+

M log 6
δ
n

(29)

+ 16 × 2

√

3
4

eβ ⌈log2 n⌉ log (3e/δ) + 32

√

eβ ⌈log2 n⌉ (cid:112)log 3e/δ

(cid:17)

.

629

Then, substituting (29) into (28), when n ≥ 16γ2 log 6

δ

µ2

, with probability at least 1 − δ

∥∇F (A(S)) − ∇FS(A(S))∥

≤∥∇FS(A(S))∥ + 4

(cid:115)

EZ[∥∇f (w∗; Z)∥2] log 6
δ
n

+ 2

(cid:115) (cid:0) 1

+

2M log 6
δ
n

+ 32 × 2

√

3
4

630

The proof is complete.

eβ ⌈log2 n⌉ log (3e/δ) + 64

2 β2 + 32nβ2 log(3/δ)(cid:1) log 6
n
eβ ⌈log2 n⌉ (cid:112)log 3e/δ.

√

δ

631

632

Proof of Remark 5. Here we briefly prove the results given in Remark 5. Since F satisfies the PL
condition with µ, we have

F (A(S)) − F (w∗) ≤

∥∇F (A(S))∥2
2µ

,

∀w ∈ W.

(30)

633

So to bound F (A(S)) − F (A(S)), we need to bound the term ∥∇F (A(S))∥2. And there holds

∥∇F (A(S))∥2

2 = 2 ∥∇F (A(S)) − ∇FS(A(S))∥2 + 2∥∇FS(A(S))∥2
2.

(31)

23

From Lemma 1, if f is M -Lipschitz and γ-smooth and F satisfies PL condition with µ, for any δ > 0,
when n ≥ 16γ2 log 6

, with probability at least 1 − δ, there holds

δ

µ2

∥∇F (A(S)) − ∇FS(A(S))∥2



(cid:115)

≤ ∥∇FS(A(S))∥2 + C



2EZ[∥∇f (w∗; Z)∥2

2] log 6
δ

n

+

M log 6
δ
n



+ eβ ⌈log2 n⌉ log (3e/δ)





(cid:115)

≤ ∥∇FS(A(S))∥2 + C



8γF (w∗) log 6
δ
n

+

M log 6
δ
n



+ eβ ⌈log2 n⌉ log (3e/δ)

 ,

where C is a positive constant and the last inequality follows from Lemma 4.1 of Srebro et al. [2010]
when f is nonegative and γ-smooth (see (44)).

Combing above inequality with (30), (31), we can derive that

F (A(S)) − F (w∗) ≲ ∥∇FS(A(S))∥2 +

F (w∗) log (1/δ)
n

+

M log2(1/δ)
n2

+ β2 log2 n log2(1/δ).

The proof is complete.

634

635

636

637

638

639

640

641

642

Proof of Lemma 2. According to the proof in Theorem 3, we have the following inequality with
probability at least 1 − δ

∥∇F (A(S)) − ∇FS(A(S))∥2
(cid:115)
2 (cid:0)2EZ[∥∇f (A(S); Z)∥2

4 β2 + 16nβ2 log(3/δ)(cid:1) log 6
2] + 1
n

δ

≤

(32)

+

M log 6
δ
n

+ 16 × 2

√

3
4

eβ ⌈log2 n⌉ log (3e/δ) + 32

√

eβ ⌈log2 n⌉ (cid:112)log 3e/δ.

643

644

645

√

Since SGC implies that EZ[∥∇f (w; Z)∥2
a +
and
least 1 − δ

a + b ≤

√

√

ab ≤ ηa+ 1
η b
b for any a, b, η > 0, we have the following inequality with probability at

2, according to inequalities

2] ≤ ρ∥∇F (w)∥2

√

∥∇F (A(S)) − ∇FS(A(S))∥2
(cid:115)

2 (cid:0)2ρ∥∇F (A(S))∥2

2 + 1

4 β2 + 16nβ2 log(3/δ)(cid:1) log 6

δ

≤

≤

+
(cid:115) (cid:0) 1

n

M log 6
δ
n

+ 16 × 2

√

3
4

eβ ⌈log2 n⌉ log (3e/δ) + 32

√

eβ ⌈log2 n⌉ (cid:112)log 3e/δ

2 β2 + 32nβ2 log(3/δ)(cid:1) log 6
n

δ

+

η
1 + η

∥∇F (A(S))∥ +

1 + η
η

4ρM log 6
δ
n

+

M log 6
δ
n

+ 16 × 2

√

3
4

eβ ⌈log2 n⌉ log (3e/δ) + 32

√

eβ ⌈log2 n⌉ (cid:112)log 3e/δ.

646

which implies that

∥∇F (A(S))∥2 ≤ (1 + η)∥∇FS(A(S))∥2 + C

1 + η
η

(cid:18) M
n

log

6
δ

+ β log n log

(cid:19)

.

1
δ

647

The proof is complete.

24

648

D Proofs of ERM

649

Proof of Lemma 3. Since FS(i)(w) = 1
n

(cid:16)

f (w; z′

i) + (cid:80)

j̸=i f (w, zj)

(cid:17)

, we have

FS( ˆw∗(S(i))) − FS( ˆw∗(S))
f ( ˆw∗(S(i)); zi) − f ( ˆw∗(S); zi)
n
f ( ˆw∗(S(i)); zi) − f ( ˆw∗(S); zi)
n

+

+

(cid:16)

+

FS(i)( ˆw∗(S(i))) − FS(i)( ˆw∗(S))

(cid:80)

j̸=i(f ( ˆw∗(S(i)); zj) − f ( ˆw∗(S); zj))
n
i) − f ( ˆw∗(S(i)); z′
i)

f ( ˆw∗(S); z′

(cid:17)

n

f ( ˆw∗(S(i)); zi) − f ( ˆw∗(S); zi)
n
∥ ˆw∗(S(i)) − ˆw∗(S)∥2,

2M
n

+

f ( ˆw∗(S); z′

i) − f ( ˆw∗(S(i)); z′
i)

n

=

=

≤

≤

650

651

652

653

where the first inequality follows from the fact that ˆw∗(S(i)) is the ERM of FS(i) and the second
inequality follows from the Lipschitz property. Furthermore, for ˆw∗(S(i)), the convexity of f and
the strongly-convex property of FS imply that its closest optima point of FS is ˆw∗(S) (the global
minimizer of FS is unique). Then, there holds that

FS( ˆw∗(S(i))) − FS( ˆw∗(S)) ≥

µ
2

∥ ˆw∗(S(i)) − ˆw∗(S)∥2
2.

654

Then we get

µ
2

∥ ˆw∗(S(i)) − ˆw∗(S)∥2

2 ≤ FS( ˆw∗(S(i))) − FS( ˆw∗(S)) ≤

2M
n

∥ ˆw∗(S(i)) − ˆw∗(S)∥2,

655

656

which implies that ∥ ˆw∗(S(i)) − ˆw∗(S)∥2 ≤ 4M
obtain that for any S(i) and S

nµ . Combined with the smoothness property of f we

∀z ∈ Z,

(cid:13)
(cid:13)
(cid:13)∇f ( ˆw∗(S(i)); z) − ∇f ( ˆw∗(S); z)
(cid:13)
(cid:13)
(cid:13)2

≤

4M γ
nµ

.

657

The proof is complete.

658

Proof of Theorem 4. Since F is µ-strongly convex, we have

F (w) − F (w∗) ≤

∥∇F (w)∥2
2
2µ

,

∀w ∈ W.

659

660

661

So to bound F ( ˆw∗) − F (w∗), we need to bound the term ∥∇F ( ˆw∗)∥2

2. And there holds

∥∇F ( ˆw∗)∥2

2 = 2 ∥∇F ( ˆw∗) − ∇FS( ˆw∗)∥2

2 + 2∥∇FS( ˆw∗)∥2
2.

From Lemma 1, if f is M -Lipschitz and γ-smooth and FS is µ-strongly convex, for any δ > 0, when
n ≥ 16γ2 log 6

, with probability at least 1 − δ, there holds

δ

µ2

∥∇F ( ˆw∗) − ∇FS( ˆw∗)∥2



(cid:115)

≤ ∥∇FS( ˆw∗)∥2 + C



2EZ[∥∇f (w∗; Z)∥2

2] log 6
δ

n

+

M log 6
δ
n

+ e ˆβ ⌈log2 n⌉ log (3e/δ)







(cid:115)

≤ ∥∇FS( ˆw∗)∥2 + C



8γF (w∗) log 6
δ
n

+

M log 6
δ
n

+ e ˆβ ⌈log2 n⌉ log (3e/δ)

 ,



25

(35)

(33)

(34)

where the last inequality follows from Lemma 4.1 of Srebro et al. [2010] when f is nonegative (see
(44)) and γ-smooth and ˆβ = ∥∇f ( ˆw∗(S); z) − ∇f ( ˆw∗(S′); z)∥2. C is a positive constant.
From Lemma 3, we have ∥∇f ( ˆw∗(S); z) − ∇f ( ˆw∗(S′); z)∥2 ≤ 4M γ
have ∥∇FS( ˆw∗)∥2 = 0, then we can derive that

nµ . Since ∇FS( ˆw∗) = 0, we

F (w) − F (w∗) ≲ F (w∗) log (1/δ)

n

+

log2 n log2(1/δ)
n2

.

662

663

664

665

666

667

E Proofs of PGD

668

Proof of Theorem 5. According to smoothness assumption and η = 1/γ, we can derive that

FS(wt+1) − FS(wt)

≤⟨wt+1 − wt, ∇FS(wt)⟩ +

γ
2

∥wt+1 − wt∥2
2

= − ηt∥∇FS(wt)∥2

2 +

t ∥∇FS(wt)∥2
η2
2

γ
2

=

η2
t − ηt

∥∇FS(wt)∥2
2

(cid:17)

≤ −

ηt∥∇FS(wt)∥2
2.

(cid:16) γ
2
1
2

669

According to above inequality and the assumptions that FS is µ-strongly convex, we can prove that

FS(wt+1) − FS(wt) ≤ −

1
2

ηt∥∇FS(wt)∥2

2 ≤ −µηt(FS(wt) − FS( ˆw∗)),

670

which implies that

FS(wt+1) − FS( ˆw∗) ≤ (1 − µηt)(FS(wt) − FS( ˆw∗)).

671

672

According to the property for γ-smooth for FS and the property for µ-strongly convex for FS, we
have

1
2γ

∥∇FS(w)∥2

2 ≤ FS(w) − FS( ˆw∗) ≤

1
2µ

∥∇FS(w)∥2
2,

673

674

which means that µ

γ ≤ 1.

Then If ηt = 1/γ, 0 ≤ 1 − µηt < 1, taking over T iterations, we get

FS(wt+1) − FS( ˆw∗) ≤ (1 − µηt)T (FS(wt) − FS( ˆw∗)).

(36)

675

Combined (36), the smoothness of FS and the nonnegative property of f , it can be derive that

∥∇FS(wT +1))∥2

2 = O

(cid:18)

(1 −

(cid:19)

.

µ
γ

)T

676

Furthermore, since F is µ-strongly convex, we have

F (w) − F (w∗) ≤

∥∇F (w)∥2
2
2µ

,

∀w ∈ W.

677

So to bound F (wT +1) − F (w∗), we need to bound the term ∥∇F (wT +1)∥2

2. And there holds

∥∇F (wT +1)∥2

2 = 2 ∥∇F (wT +1) − ∇FS(wT +1)∥2

2 + 2∥∇FS(wT +1)∥2
2.

(37)

(38)

26

678

679

680

681

682

683

From Lemma 1, if f is M -Lipschitz and γ-smooth and FS is µ-strongly convex, for any δ > 0, when
n ≥ 16γ2 log 6

, with probability at least 1 − δ, there holds

δ

µ2

∥∇F (wT +1) − ∇FS(wT +1)∥2



(cid:115)

≤ ∥∇FS(wT +1)∥2 + C



2EZ[∥∇f (w∗; Z)∥2

2] log 6
δ

n

+

M log 6
δ
n



+ eβ ⌈log2 n⌉ log (3e/δ)





(cid:115)

≤ ∥∇FS(wT +1)∥2 + C



8γF (w∗) log 6
δ
n

+

M log 6
δ
n



+ eβ ⌈log2 n⌉ log (3e/δ)

 ,

(39)
where the last inequality follows from Lemma 4.1 of Srebro et al. [2010] when f is nonegative and
γ-smooth (see (44)) and β = ∥∇f (wT +1(S); z) − ∇f (wT +1(S′); z)∥2. C is a positive constant.
From Lemma 4, we have β = ∥∇f (wT +1(S); z) − ∇f (wT +1(S′); z)∥2 ≤ 2M γ
nµ . Since
∥∇FS(wT +1)∥2 = O

, then we can derive that

(1 − µ

(cid:16)

F (w) − F (w∗) ≲

1 −

(cid:19)2T

µ
γ

+

F (w∗) log (1/δ)
n

+

log2 n log2(1/δ)
n2

.

γ )T (cid:17)
(cid:18)

684

Let T ≍ log n, we have

F (w) − F (w∗) ≲ F (w∗) log (1/δ)

n

+

log2 n log2(1/δ)
n2

.

685

The proof is complete.

686

687

688

689

690

691

692

693

694

695

696

697

698

699

700

701

F Proofs of SGD

We first introduce some necessary lemmata on the empirical risk.
Lemma 14 ([Lei and Tang, 2021]). Let {wt}t be the sequence produced by SGD with ηt ≤ 1
2γ for
all t ∈ N. Suppose Assumption 1 hold. Assume for all z, the function w (cid:55)→ f (w; z) is M -Lipschitz
and γ-smooth. Then, for any δ ∈ (0, 1), with probability at least 1 − δ, there holds that

ηk∥∇FS(wk)∥2

2 = O

log

(cid:32)

t
(cid:88)

k=1

(cid:33)

η2
k

.

1
δ

+

t
(cid:88)

k=1

2
Lemma 15 ([Lei and Tang, 2021]). Let {wt}t be the sequence produced by SGD with ηt =
µ(t+t0)
such that t0 ≥ max{ 4γ
µ , 1} for all t ∈ N. Suppose Assumption 1 hold. Assume for all z, the function
w (cid:55)→ f (w; z) is M -Lipschitz and γ-smooth and assume FS satisfies PL condition with parameter µ.
Then, for any δ ∈ (0, 1), with probability at least 1 − δ, there holds that
(cid:18) log(T ) log3(1/δ)
T

FS(wT +1) − FS( ˆw∗) = O

(cid:19)

.

Lemma 16 ([Lei and Tang, 2021]). Let e be the base of the natural logarithm. There holds the
following elementary inequalities.
• If θ ∈ (0, 1), then (cid:80)t
• If θ = 1, then (cid:80)t
• If θ > 1, then (cid:80)t

k=1 k−θ ≤ log(et);
k=1 k−θ ≤ θ

k=1 k−θ ≤ t1−θ/(1 − θ);

θ−1 .

Proof of Lemma 5. We have known that FS(i) (w) = 1
n
ˆw∗(S(i)) be the ERM of FS(i) (w) and ˆw∗

i) + (cid:80)
S be the ERM of FS(w). From Lemma 3, we know that

j̸=i f (w; zj)

f (w; z′

. We denote

(cid:16)

(cid:17)

∀z ∈ Z,

(cid:13)
(cid:13)
(cid:13)∇f ( ˆw∗(S(i)); z) − f ( ˆw∗(S); z)
(cid:13)
(cid:13)
(cid:13)2

≤

4M γ
nµ

.

27

704

705

706

707

708

709

711

712

713

714

715

702

703

Also, for wt, the convexity of f and the strongly-convex property implies that its closest optima point
of FS is ˆw∗(S) (the global minimizer of FS is unique). Then, there holds that

µ
2

∥wt − ˆw∗(S)∥2

2 ≤ FS(wt) − FS( ˆw∗(S)) = ϵopt(wt).

Thus we have ∥wt − ˆw∗(S)∥2 ≤
Combined with the Lipschitz property of f we obtain that for ∀z ∈ Z, there holds that

. A similar relation holds between ˆw∗(S(i)) and wi
t.

(cid:113) 2ϵopt(wt)
µ

(cid:13)
(cid:13)∇f (wt; z) − ∇f (wi

t; z)(cid:13)
(cid:13)2

≤ ∥∇f (wt; z) − ∇f ( ˆw∗(S); z)∥2 +

+

(cid:13)
(cid:13)∇f ( ˆw∗(S(i)); z) − ∇f (wi
(cid:13)

(cid:13)
(cid:13)
(cid:13)∇f ( ˆw∗(S); z) − ∇f ( ˆw∗(S(i)); z)
(cid:13)
(cid:13)
(cid:13)2
(cid:13)
(cid:13)
t; z)
(cid:13)2

≤ γ∥wt − ˆw∗(S)∥2 +

4M γ
nµ

+ γ∥ ˆw∗(S(i)) − wi

t∥2

(cid:115)

≤ γ

2ϵopt(wt)
µ

+

4M γ
nµ

+ γ

(cid:115)

2ϵopt(wi
t)
µ

.

According to Lemma 15, for any dataset S, the optimization error ϵopt(wt) is uniformly bounded by
the same upper bound. Therefore, we write (cid:13)
+ 4M γ
nµ
here.

(cid:13)∇f (wt; z) − ∇f (wi

(cid:113) 2ϵopt(wt)
µ

(cid:13)2 ≤ 2γ

t; z)(cid:13)

The proof is complete.

710

Now We begin to prove Lemma 6.

Proof of Lemma 6. If f is L-Lipschitz and γ-smooth and FS is µ-strongly convex. According to
Lemma 1, we know that for all w ∈ W and any δ ∈ (0, 1), with probability at least 1 − δ/2, when
n > 16γ2 log 6

, we have

δ

µ2

(cid:32) T

(cid:88)

ηt

(cid:33)−1 T

(cid:88)

ηt∥∇F (wt)∥2
2

t=1
(cid:32) T

(cid:88)

t=1
(cid:32) T

(cid:88)

t=1
(cid:33)−1 T

(cid:88)

ηt

t=1
(cid:33)−1 T

(cid:88)

ηt

t=1

t=1

≤16

+

ηt∥∇FS(wt)∥2

2 +

4C 2L2 log2 6
δ
n2

+

8C 2EZ[∥∇f (w∗; Z)∥2

2] log2 6
δ

n

(40)

ηtC 2e2β2

t ⌈log2 n⌉2 log2 (3e/δ),

where βt = (cid:13)

(cid:13)∇f (wt; z) − ∇f (wi

t; z)(cid:13)

(cid:13)2 and C is a positive constant.

From Lemma 5, we have (cid:13)

(cid:13)∇f (wt; z) − ∇f (wi

t; z)(cid:13)

(cid:113) 2ϵopt(wt)
µ

+ 4M γ

nµ , thus

(cid:13)2 ≤ 2γ
t; z)(cid:13)
2
(cid:13)
2
(cid:33)2

t = (cid:13)
β2
(cid:13)∇f (wt; z) − ∇f (wi
(cid:32)

(cid:115)

≤

2γ

2ϵopt(wt)
µ

+

4M γ
nµ

≤

≤

16γ2(FS(wt) − FS( ˆw∗(S)))
µ

+

32M 2γ2
n2µ2

8γ2∥∇FS(wt)∥2
2
µ2

+

32M 2γ2
n2µ2

,

(41)

716

717

where the second inequality holds from Cauchy-Bunyakovsky-Schwarz inequality and the second
inequality satisfies because FS is µ-strongly convex.

28

718

Plugging (41) into (40), with probability at least 1 − δ/2, when n > 16γ2 log 6

δ

µ2

, we have

(cid:32) T

(cid:88)

ηt

(cid:33)−1 T

(cid:88)

t=1

t=1

ηt∥∇F (wt)∥2
2

(cid:32)

≤

16 +

8γ2C 2e2 ⌈log2 n⌉2 log2 (6e/δ)
µ2

(cid:33) (cid:32) T

(cid:88)

(cid:33)−1 T

(cid:88)

ηt

ηt∥∇FS(wt)∥2
2

+

4C 2L2 log2 12
δ
n2

+

8C 2EZ[∥∇f (w∗; Z)∥2

t=1
2] log2 12
δ

n

t=1

+

32L2γ2C 2e2 ⌈log2 n⌉2 log2 (6e/δ)
n2µ2

,

(42)

719

720

When ηt = η1t−θ, θ ∈ (0, 1), with η1 ≤ 1
16, we obtain the following inequality with probability at least 1 − δ/2,

2β and Assumption 1, according to Lemma 14 and Lemma

(cid:32) T

(cid:88)

ηt

(cid:33)−1 T

(cid:88)

t=1

t=1

ηt∥∇FS(wt)∥2 =





O

O

O

(cid:16) log(1/δ)
T −θ
(cid:16) log(1/δ)
T − 1
(cid:16) log(1/δ)
T θ−1

2

(cid:17)

(cid:17)

(cid:17)

,

,

,

if θ < 1/2

if θ = 1/2

if θ > 1/2.

(43)

721

722

On the other hand, when f is nonegative and γ-smooth, from Lemma 4.1 of Srebro et al. [2010], we
have

∥∇f (w∗; z)∥2

2 ≤ 4γf (w∗; z),

723

which implies that

2] ≤ 4γEZf (w∗; Z) = 4γF (w∗).
Plugging (44), (43) into (42), with probability at least 1 − δ, we derive that

EZ[∥∇f (w∗; Z)∥2

724

(44)

(cid:32) T

(cid:88)

ηt

(cid:33)−1 T

(cid:88)

ηt∥∇F (wt)∥2
2

t=1

t=1





(cid:16) log2 n log3(1/δ)
T −θ
(cid:16) log2 n log3(1/δ)
T − 1
(cid:16) log2 n log3(1/δ)
T θ−1

2

(cid:17)

(cid:17)

(cid:17)

O

O

O

+ O

+ O

+ O

(cid:16) log2 n log2(1/δ)
n2
(cid:16) log2 n log2(1/δ)
n2
(cid:16) log2 n log2(1/δ)
n2

+ F (w∗) log2(1/δ)
n
+ F (w∗) log2(1/δ)
n
+ F (w∗) log2(1/δ)
n

(cid:17)

(cid:17)

(cid:17)

,

,

,

if θ < 1/2

if θ = 1/2

if θ > 1/2.

=

725

726

727

728

When θ < 1/2, we set T ≍ n 2
with probability at least 1 − δ

θ and assume F (w∗) = O( 1

n ), then we obtain the following result

(cid:32) T

(cid:88)

ηt

(cid:33)−1 T

(cid:88)

t=1

t=1

ηt∥∇F (wt)∥2

2 = O

(cid:18) log2 n log3(1/δ)
n2

(cid:19)

.

When θ = 1/2, we set T ≍ n4 and assume F (w∗) = O( 1
probability at least 1 − δ
(cid:32) T

(cid:33)−1 T

(cid:88)

ηt

(cid:88)

ηt∥∇F (wt)∥2

2 = O

t=1

t=1

(cid:18) log2 n log3(1/δ)
n2

(cid:19)

.

n ), then we obtain the following result with

729

730

When θ > 1/2, we set T ≍ n
with probability at least 1 − δ

2

1−θ and assume F (w∗) = O( 1

n ), then we obtain the following result

(cid:32) T

(cid:88)

ηt

(cid:33)−1 T

(cid:88)

t=1

t=1

ηt∥∇F (wt)∥2

2 = O

(cid:18) log2 n log3(1/δ)
n2

(cid:19)

.

The proof is complete.

731

732

29

733

Proof of Theorem 7. Since F is µ-strongly convex, we have

F (w) − F (w∗) ≤

∥∇F (w)∥2
2
2µ

,

∀w ∈ W.

(45)

So to bound F (wT +1) − F (w∗), we need to bound the term ∥∇F (wT +1)∥2

2. And there holds

∥∇F (wT +1)∥2

2 = 2 ∥∇F (wT +1) − ∇FS(wT +1)∥2 + 2∥∇FS(wT +1)∥2
2.

(46)

From Lemma 1, if f is L-Lipschitz and γ-smooth and FS is µ-strongly convex, for all w ∈ W and
any δ > 0, when n ≥ 16γ2 log 6

, with probability at least 1 − δ/2, there holds

δ

µ2

734

735

736

∥∇F (wT +1) − ∇FS(wT +1)∥2



(cid:115)

≤ ∥∇FS(wT +1)∥2 + C



2EZ[∥∇f (w∗; Z)∥2

2] log 12
δ

n

+

M log 12
δ
n



+ eβ ⌈log2 n⌉ log (6e/δ)





(cid:115)

≤ ∥∇FS(wT +1)∥2 + C



8γF (w∗) log 12
δ
n

+

M log 12
δ
n



+ eβ ⌈log2 n⌉ log (6e/δ)

 ,

(47)
where the last inequality follows from Lemma 4.1 of Srebro et al. [2010] when f is nonegative and
γ-smooth (see (44)) and C is a positive constant. Then we can derive that

737

738

∥∇F (wT +1) − ∇FS(wT +1)∥2
2
32C 2γF (w∗) log 12
δ
n

≤4∥∇FS(wT +1)∥2

2 +

+

4M 2C 2 log2 12
δ
n2

+ 4e2β2

T +1 ⌈log2 n⌉2 log2 (6e/δ).
(48)

From Lemma 5, we have (cid:13)

(cid:13)∇f (wt; z) − ∇f (wi

t; z)(cid:13)

(cid:113) 2ϵopt(wt)
µ

+ 4M γ

nµ , thus

(cid:13)2 ≤ 2γ
t; z)(cid:13)
2
(cid:13)
2
(cid:33)2

t = (cid:13)
(cid:13)∇f (wt; z) − ∇f (wi
β2
(cid:32)

(cid:115)

≤

2γ

2ϵopt(wt)
µ

+

4M γ
nµ

≤

≤

16γ2(FS(wt) − FS( ˆw∗(S)))
µ

+

32M 2γ2
n2µ2

8γ2∥∇FS(wt)∥2
2
µ2

+

32M 2γ2
n2µ2

,

(49)

where the second inequality holds from Cauchy-Bunyakovsky-Schwarz inequality and the second
inequality satisfies because FS is µ-strongly convex.

Plugging (49) into (48), with probability at least 1 − δ/2, when , we have

739

740

741

742

∥∇F (wT +1) − ∇FS(wT +1)∥2
2
(cid:16)

(cid:17)
4 + 32e2 ⌈log2 n⌉2 log2 (6e/δ)

≤

+

4L2C 2 log2 12
δ
n2

+

∥∇FS(wT +1)∥2
128M 2γ2e2 ⌈log2 n⌉2 log2 (6e/δ)
n2µ2

2 +

.

32C 2γF (w∗) log 6
δ
n

(50)

743

744

According to the smoothness property of FS and Lemma 15, it can be derived that with propability at
least 1 − δ/2

∥∇FS(wT +1)∥2

2 = O

(cid:18) log T log3(1/δ)
T

(cid:19)

.

(51)

30

745

Substituting (51), (50) into (46), we derive that

∥∇F (wT +1)∥2
2

(cid:32)

=O

⌈log2 n⌉2 log T log5(1/δ)
T

(cid:33)

(cid:32)

+ O

⌈log2 n⌉2 log2(1/δ)
n2

+

F (w∗) log(1/δ)
n

(cid:33)

.

(52)

746

747

Further substituting (52) into (45) and choosing T ≍ n2, we finally obtain that when n, with
probability at least 1 − δ

F (wT +1) − F (w∗) = O

(cid:18) log4 n log5(1/δ)
n2

+

F (w∗) log(1/δ)
n

(cid:19)

.

748

31

749

750

751

752

753

754

755

756

757

758

759

760

761

762

763

764

765

766

767

768

769

770

771

772

773

774

775

776

777

778

779

780

781

782

783

784

785

786

787

788

789

790

791

792

793

794

795

796

797

798

799

800

NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [Yes]

Justification: We have checked that the abstract and introduction accurately reflect our
contributions and scope.

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

Justification: We have clearly stated the required assumptions for each theorem and lemma,
and the conditions for the assumptions to hold are also stated in the main text.

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

32

801

802

803

804

805

806

807

808

809

810

811

812

813

814

815

816

817

818

819

820

821

822

823

824

825

826

827

828

829

830

831

832

833

834

835

836

837

838

839

840

841

842

843

844

845

846

847

848

849

850

851

852

853

Answer: [Yes]

Justification: We have clearly stated the required assumptions for each theorem and lemma,
and all proofs are provided in the appendix.

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

Answer: [NA]

Justification: This paper focuses on learning theory.

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

33

854

855

856

857

858

859

860

861

862

863

864

865

866

867

868

869

870

871

872

873

874

875

876

877

878

879

880

881

882

883

884

885

886

887

888

889

890

891

892

893

894

895

896

897

898

899

900

901

902

903

904

905

Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [NA]
Justification: This paper focuses on learning theory and does not include experiments.
Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/

public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be
possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).

• The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized

versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the

paper) is recommended, but including URLs to data and code is permitted.

6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyper-
parameters, how they were chosen, type of optimizer, etc.) necessary to understand the
results?
Answer: [NA]
Justification: This paper focuses on learning theory and does not include experiments.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail

that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental

material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [NA]
Justification: This paper focuses on learning theory and does not include experiments.
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

34

906

907

908

909

910

911

912

913

914

915

916

917

918

919

920

921

922

923

924

925

926

927

928

929

930

931

932

933

934

935

936

937

938

939

940

941

942

943

944

945

946

947

948

949

950

951

952

953

954

955

956

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
Answer: [NA]
Justification: This paper focuses on learning theory and does not include experiments.
Guidelines:

• The answer NA means that the paper does not include experiments.
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
Justification: We have reviewed the NeurIPS Code of Ethics.
Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a

deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-

eration due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [NA]
Justification: This paper focuses on learning theory and there is no societal impact of the
work performed.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal

impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

35

957

958

959

960

961

962

963

964

965

966

967

968

969

970

971

972

973

974

975

976

977

978

979

980

981

982

983

984

985

986

987

988

989

990

991

992

993

994

995

996

997

998

999

1000

1001

1002

1003

1004

1005

1006

1007

1008

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

Justification: This paper focuses on learning theory and poses no such risks.

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

Answer: [NA]

Justification: This paper focuses on learning theory and does not use existing assets.

Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a

URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of

service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of

the derived asset (if it has changed) should be provided.

36

1009

1010

1011

1012

1013

1014

1015

1016

1017

1018

1019

1020

1021

1022

1023

1024

1025

1026

1027

1028

1029

1030

1031

1032

1033

1034

1035

1036

1037

1038

1039

1040

1041

1042

1043

1044

1045

1046

1047

1048

1049

1050

1051

1052

1053

1054

1055

1056

1057

1058

• If this information is not available online, the authors are encouraged to reach out to

the asset’s creators.

13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: The paper does not release new assets.
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
Answer: [NA]
Justification: This paper does not involve crowdsourcing nor research with human subjects.
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
Justification: This paper does not involve crowdsourcing nor research with human subjects.
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

37

