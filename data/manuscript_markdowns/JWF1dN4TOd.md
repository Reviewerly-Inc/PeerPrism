Large-Scale Contextual Market Equilibrium
Computation through Deep Learning

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

33

34

35

36

Market equilibrium is one of the most fundamental solution concepts in economics
and social optimization analysis. Existing works on market equilibrium computa-
tion primarily focus on settings with a relatively small number of buyers. Motivated
by this, our paper investigates the computation of market equilibrium in scenarios
with a large-scale buyer population, where buyers and goods are represented by
their contexts. Building on this realistic and generalized contextual market model,
we introduce MarketFCNet, a deep learning-based method for approximating mar-
ket equilibrium. We start by parameterizing the allocation of each good to each
buyer using a neural network, which depends solely on the context of the buyer
and the good. Next, we propose an efficient method to estimate the loss function of
the training algorithm unbiasedly, enabling us to optimize the network parameters
through gradient descent. To evaluate the approximated solution, we introduce
a metric called Nash Gap, which quantifies the deviation of the given allocation
and price pair from the market equilibrium. Experimental results indicate that
MarketFCNet delivers competitive performance and significantly lower running
times compared to existing methods as the market scale expands, demonstrating
the potential of deep learning-based methods to accelerate the approximation of
large-scale contextual market equilibrium.

1

Introduction

Market equilibrium is a solution concept in microeconomics theory, which studies how individuals
amongst groups will exchange their goods to get each one better off [51]. The importance of
market equilibrium is evidenced by the 1972 Nobel Prize awarded to John R. Hicks and Kenneth
J. Arrow “for their pioneering contributions to general economic equilibrium theory and welfare
theory” [58]. Market equilibrium has wide application in fair allocation [32], as a few examples,
fairly assigning course seats to students [11] or dividing estates, rent, fares, and others [35]. Besides,
market equilibrium are also considered for ad auctions with budget constraints where money has real
value [15, 16].

Existing works often use traditional optimization method or online learning technique to solve market
equilibrium, which can tackle one market with around 400 buyers and goods in experiments [30, 52].
However, in realistic scenarios, there might be millions of buyers in one market (e.g. job market,
online shopping market). In these scenarios, the description complexity for the market is O(nm) and
it needs at least O(nm) cost to do one optimization step for the market, if there are n buyers and m
goods in the market, which is unacceptable when n is extremely large and potentially infinite. In this
case, and traditional optimization methods do not work anymore.

However, contextual models come to the rescue. The success of contextual auctions[21, 5] demon-
strate the power of contextual models, in which each bidder and item are represented as context and

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.

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

the value (or the distribution) of item to bidder is determined by the contexts. In this way, auctions
as well as other economic problems can be described in a more memory-efficient way, making it
possible to accelerate the computation on these problems. Inspired by the models of contextual
auctions, we propose the concept of contextual markets in a similar way. We verify that contextual
markets can be useful to model large-scale markets aforementioned, since the real market can be
assumed to be within some low dimension space, and the values of goods to buyers are often not
hard to speculate given the knowledge of goods and buyers [46, 45]. Besides, contextual models
never lose expressive power compared with raw models[7], giving contextual markets capabilities to
generalize over traditional markets.

This paper initiates the study of deep learning for contextual market equilibrium computation
with a large number of buyers. The description complexity of contextual markets is O(n + m),
if there are n buyers and m items in the market, making them memory-efficient and helpful for
follow-up equilibrium computation while holding the market structure. Following the framework of
differentiable economics [18, 26, 62], we propose a deep-learning based approach, MarketFCNet,
in which one optimization step costs only O(m) rather than O(nm) in traditional methods, greatly
accelerating the computation of market equilibrium. MarketFCNet takes the representations of one
buyer and one good as input, and outputs the allocation of the good to the buyer. The training on
MarketFCNet targets at an unbiased estimator of the objective function of EG-convex program, which
can be formed by independent samples of buyers. By this way, we optimize the allocation function
on “buyer space” implicitly, rather than optimizing the allocation to each buyer directly. Therefore,
MarketFCNet can reduce the algorithm complexity such that it becomes independent of n, i.e., the
number of buyers.

The effectiveness of MarketFCNet is demonstrated by our experimental results. As the market
scale expands, MarketFCNet delivers competitive performance and significantly lower running times
compared to existing methods in different experimental settings, demonstrating the potential of deep
learning-based methods to accelerate the approximation of large-scale contextual market equilibrium.

The contributions of this paper consist of three parts,

• We proposes a method, MarketFCNet, to approximate the contextual market equilibrium in

which the number of buyers is large.

• We proposes Nash Gap to quantify the deviation of the given allocation and price pair from

the market equilibrium.

• We conduct extensive experiments, demonstrating promising performance on the approxi-

mation measure and running time compared with existing methods.

2 Related Works

The history of market equilibrium arises from microeconomics theory, where the concept of com-
petitive equilibrium [51, §10] was proposed, and the existence of market equilibrium is guaranteed
in a general setting [3, 61]. Eisenberg and Gale [28] first considered the linear market case, and
proved that the solution of EG-convex program constitutes a market equilibrium, which lays the
polynomial-time algorithmic foundations for market equilibrium computation. Eisenberg [27] later
showed that EG program also works for a class of CCNH utility functions. Shmyrev program later is
also proposed to solve market equilibrium with linear utility with a perspective shift from allocation
to price [57], while Cole et al. [14] later found that Shmyrev program is the dual problem of EG
program with a change of variables. There are also a branch of literature that consider computational
perspective in more general settings such as indivisible goods [54, 19, 20] and piece-wise linear
utility [60, 33, 34].

There are abundant of works that present algorithms to solve the market equilibrium and shows
the convergence results theoretically [13]. Gao and Kroer [30] discusses the convergence rates of
first-order algorithms for EG convex program under linear, quasi-linear and Leontief utilities. Nan
et al. [52] later designs stochastic optimization algorithms for EG convex program and Shmyrev
program with convergence guarantee and show some economic insight. Jalota et al. [42] proposes an
ADMM algorithm for CCNH utilities and shows linear convergence results. Besides, researchers
are more engaged in designing dynamics that possess more economic insight. For example, PACE

2

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

dynamic [32, 48, 65] and proportional response dynamic [63, 66, 12], though the original idea of
PACE arise from auction design [16, 15].

With the fast growth of machine learning and neural network, many existing works aim at resolving
economic problem by deep learning approach, which falls into the differentiate economy framework
[26]. A mainstream is to approximate the optimal auction with differentiable models by neural
networks [25, 29, 36, 55]. The problem of Nash equilibrium computation in normal form games
[22, 50, 23] and optimal contract design [62] through deep learning also attracts researchers’ attentions.
Among these methodologies, transformer architecture [50, 21, 47] is widely used in solving economic
problems.

To the best of our knowledge, no existing works try to approximate market equilibrium through deep
learning. Besides, although some literature focuses on low-rank markets and representative markets
[46, 45], our works firstly propose the concept of contextual market. We believe that our approach
will pioneer a promising direction for large-scale contextual market equilibrium computation.

3 Contextual Market Modelling

In this section, we focus on the model of contextual market equilibrium in which goods are assumed to
be divisible. Let the market consist of n buyers, denoted as 1, ..., n, and m goods, denoted as 1, ..., m.
We denote [k] as the abbreviation of the set {1, 2, . . . , k}. Each buyer i ∈ [n] has a representation bi,
and each good j ∈ [m] has a representation gj. We assume that bi belongs to the buyer representation
space B, and gj belongs to the good representation space G. For a buyer with representation b ∈ B,
she has budget B(b) > 0. Denote Y (g) > 0 as the supply of good with representation g. Although
many existing works [30] assume that each good j has unit supply (i.e. Y (g) ≡ 1 for all g ∈ G)
without loss of generality, their results can be easily generalized to our settings.
An allocation is a matrix x = (xij)i∈[n],j∈[m] ∈ Rn×m
, where xij is the amount of good j allocated
to buyer i. We denote xi = (xi1, . . . , xim) as the vector of bundle of goods that is allocated to buyer
+ → R+, here u(bi; xi) denotes the utility of
i. The buyers’ utility function is denoted as u : B × Rm
buyer i with representation bi when she chooses to buy xi. We denote ui(xi) as an equivalent form
of u(bi; xi) and often refer them as the same thing. Similarly, B(bi), Y (gj) and Bi, Yj are often
referred to as the same thing, respectively.
Let p = (p1, . . . , pm) ∈ Rm
bi is defined as the set of utility-maximizing allocations within budget constraint.
+ , ⟨p, xi⟩ ≤ B(bi)(cid:9) .

+ be the prices of the goods, the demand set of buyer with representation

(cid:8)u(bi; xi) | xi ∈ Rm

D(bi; p) := arg max

(1)

+

xi

A contextual market is a 4-tuple: M = ⟨n, m, (bi)i∈[n], (gj)j∈[m]⟩, where buyer utility u(bi; xi) is
known given the information of the market. We also assume budget function B : B → R+ represents
the budget of buyers and capacity function Y : G → R+ represents the supply of goods. All of
u, B and Y are assumed to be public knowledge and excluded from a market representation. This
assumption mainly comes from two aspects: (1) these functions can be learned from historical data
and (2) budgets and supplies can be either encoded in b and g in some way.
The market equilibrium is represented as a pair (x, p), x ∈ Rn×m
following conditions.

+ , which satisfies the

, p ∈ Rm

+

• Buyer optimality: xi ∈ D(bi, p) for all i ∈ [n],
• Market clearance: (cid:80)n

i=1 xij ≤ Y (gj) for all j ∈ [m], and equality must hold if pj > 0.

We say that ui is homogeneous (with degree 1) if it satisfies ui(αxi) = αui(xi) for any xi ≥ 0 and
α > 0 [53, §6.2]. Following existing works, we assume that uis are CCNH utilities, where CCNH
represents for concave, continuous, non-negative, and homogeneous functions[30]. For CCNH
utilities, a market equilibrium can be computed using the following Eisenberg-Gale convex program
(EG):

n
(cid:88)

n
(cid:88)

max

Bi log ui(xi)

s.t.

xij ≤ Yj, x ≥ 0.

(EG)

134

Theorem 3.1 shows that the market equilibrium can be represented as the optimal solution of (EG).

i=1

i=1

3

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

162

163

164

Theorem 3.1 (Gao and Kroer [30]). Let ui be concave, continuous, non-negative and homogeneous
(CCNH). Assume ui(1) > 0 for all i. Then, (i) (EG) has an optimal solution and (ii) any optimal
solution x to (EG) together with its optimal Lagrangian multipliers p∗ ∈ Rm
+ constitute a market
equilibrium, up to arbitrary assignment of zero-price items. Furthermore, ⟨p∗, x∗
i ⟩ = Bi for all i.
Based on Theorem 3.1, it’s easy to find that we can always assume (cid:80)
the existence of market equilibrium, which states as follows.
Proposition 3.2. Following the assumptions in Theorem 3.1. For the following EG convex program
with equality constraints,

i∈[n] xij = Yj while preserving

max

n
(cid:88)

i=1

Bi log ui(xi)

s.t.

n
(cid:88)

i=1

xij = Yj, x ≥ 0.

(2)

Then, an optimal solution x∗ together with its Lagrangian multipliers p∗ ∈ Rm
+ constitute a market
equilibrium. Moreover, assume more that for each good j, there is some buyer i such that ∂ui
> 0
∂xij
always hold whenever ui(xi) > 0, then all prices are strictly positive in market equilibrium. As a
consequence, Equation (EG) and Equation (2) derive the same solution.

We leave all proofs to Appendix B. Since the additional assumption in Proposition 3.2 is fairly weak,
without further clarification, we always assume the conditions in Proposition 3.2 hold and the market
clearance condition becomes (cid:80)

i∈[n] xij = Y (gj), ∀j ∈ [m].

4 MarketFCNet

In this section, we introduce the MarketFCNet (denoted as Market Fully-Connected Network)
approach to solve the market equilibrium when the number of buyers is large and potentially infinite.
MarketFCNet is a sampling-based methodology, and the key point is to design an unbiased estimator
of an objective function whose solution coincides with the market equilibrium. The main advantage
is that it has the potential to fit the infinite-buyer case without scaling the computational complexity.
Therefore, MarketFCNet is scalable with the number of buyers varies.

4.1 Problem Reformulation

Following the idea of differentiable economics [26], we consider parameterized models to represent
the allocation of good j to buyer i, denoted as xθ(bi, gj), and call it allocation network, where θ is the
network parameter. Given buyer i and good j, the network can automatically compute the allocation
xij = xθ(bi, gj). The allocation to buyer i is represented as xi = xθ(bi, g) and the allocation
matrix is represented as x = xθ(b, g). Then the market clearance constraint can be reformulated as
(cid:80)
i∈[n] xθ(bi, gj) = Y (gj), ∀j ∈ [m] and the price constraint can be reformulated as xθ(b, g) ≥ 0.

Let b be uniformly distributed from B = {bi : i ∈ [n]}, then the EG program (EG) becomes,

OBJ(xθ) = Eb[B(b) log u(b; xθ(b, g))]

max
xθ
s.t. Eb[xθ(b, gj)] = Y (gj)/n, ∀j ∈ [m]

xθ(b, g) ≥ 0

(EG-FC)

165

For simplicity, we take Y (gj)/n ≡ 1 for all gj.

166

167

168

169

170

171

4.2 Optimization

The second constraint in (EG-FC) can be easily handled by the network architecture (for example,
network with a softplus layer σ(x) = log(1 + exp(x)). As for the first constraint, from Theorem 3.1,
we know the prices of goods are simply the Lagrangian multipliers for the first constraint in (EG-FC).
Therefore, we employ the Augmented Lagrange Multiplier Method (ALMM) to solve the problem
(EG-FC). We define Lρ(xθ, λ) as the Lagrangian, which has the form:
m
(cid:88)

m
(cid:88)

λj (Eb[xθ(b, gj)] − 1) +

(Eb[xθ(b, gj)] − 1)2

Lρ(xθ; λ) = − OBJ(xθ) +

(3)

ρ
2

j=1

j=1

4

Figure 1: Training process of MarketFCNet. On each iteration, the batch of M independent buyers
are drawn. each buyer and each good are represented as k-dimension context. The (i, j)’th element in
the allocation matrix represents the allocation computed from i’th buyer and j’th good. MarketFCNet
training process alternates between the training of allocation network and prices. The training of
allocation network need to achieve an unbiased estimator (cid:98)Lρ(xθ; λ) of the loss function Lρ(xθ; λ),
followed by gradient descent. The training of prices need to get an unbiased estimator (cid:98)∆λj of ∆λj,
followed by ALMM updating rule λj ← λj + βt (cid:98)∆λj.

Directly computing the objective function seems intractable due to the potentially infinite data size.
Therefore, we follow the framework in learning theory culture that we only guarantee to achieve an
unbiased gradient of the objective function [1, 8]. The training process of MarketFCNet is presented
in Figure 1.

To finish the ALMM algorithm, we need to obtain unbiased estimators of following two expressions.

• An unbiased estimator of Lρ(xθ; λ).
• An unbiased estimator of ∆λj, where ∆λj is given by ∆λj = ρ (Eb[xθ(b, gj)] − 1).

Unbiased estimator of ∆λj We aim to obtain an unbiased estimator of Eb[xθ(b, gj)]. By apply-
ing Monte Carlo method, we can choose batch size M and sample b1, b2, ..., bM ∼ U (B), then
1
M

i=1 xθ(bi, gj) forms an unbiased estimator.

(cid:80)M

Unbiased estimator of Lp(xθ; λ) For OBJ(xθ) and the second term, the technique to achieve an
unbiased estimator is similar. u(b; xθ(b, g)) in OBJ(xθ) can be calculated directly by summing over
all goods. For the last term, notice that

(Eb [xθ(b, gj)] − 1)2 = (Eb [xθ(b, gj)] − 1) · (Eb′ [xθ(b′, gj)] − 1)

1, ..., b′
Therefore, we can sample b1, ..., bM , b′
m
(cid:88)

M
(cid:88)

ρ
2

·

1
M

i=1

j=1

M ∼ U (B) and compute

(xθ(bi, gj) − 1) · (xθ(b′

i, gj) − 1)

(4)

(5)

which provides an unbiased estimator for the last term, capturing the squared deviation of output
allocations from the constraint.

5 Performance Measures of Market Equilibrium

In this section, we propose Nash Gap to measure the performance of an approximated market
equilibrium and show that Nash Gap preserves the economic interpretation for market equilibrium. To
introduce Nash Gap, we first introduce two types of welfare, Log Nash Welfare and Log Fixed-price
Welfare in Definition 5.1 and Definition 5.2, respectively.

5

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

Allocation networkBuyers𝑥!(𝑏",𝑔#)'ℒ$(𝑥!;λ)gradient descent𝑏%𝑏&𝑏'𝑔%𝑔(……𝑔&M * km * kM * mλ%λ&λ)……pricesUpdate λ……Goodsi.i.d.Update𝜃MarketFCNet193

Definition 5.1 (Log Nash Welfare). The Log Nash Welfare (abbreviated as LNW) is defined as

LNW(x) =

1
Btotal

(cid:88)

i∈[n]

Bi log ui(xi),

(6)

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

where Btotal = (cid:80)

i∈[n] Bi is the total budgets for buyers.

Notice that LNW(x) is identical to the objective function in Equation (EG), differing only in the
constant term coefficient.
Definition 5.2 (Fixed-price and Log Fixed-price Welfare). We define the fixed-price utility for buyer
i as,

˜u(bi; p) = max

{u(bi; xi) | xi ∈ Rm

+ , ⟨p, xi⟩ ≤ B(bi)}

xi

(7)

which represents the optimal utility that buyer i can obtain at the price level p, regardless of the
market clearance constraints. The Log Fixed-price Welfare (abbreviated as LFW) is defined as the
logarithm of Fixed-price Welfare,

LFW(p) =

1
Btotal

(cid:88)

i∈[n]

Bi log ˜ui(p)

(8)

Based on these definitions, we present the definition of Nash Gap.
Definition 5.3 (Nash Gap). We define Nash Gap (abbreviated as NG) as the difference of Log Nash
Welfare and Log Fixed-price Welfare, i.e.

NG(x, p) = LFW(p) − LNW(x)

(9)

5.1 Properties of Nash Gap

To show why NG is useful in the measure of market equilibrium, we first observe that,
Proposition 5.4 (Price constraints). If (x, p) constitute a market equilibrium, the following identity
always hold,

(cid:88)

j∈[m]

pjYj =

(cid:88)

Bi

i∈[n]

(10)

Below, we state the most important theorem in this paper.
Theorem 5.5. Let (x, p) be a pair of allocation and price. Assuming the allocation satisfies market
clearance and the price meets price constraint, then we have NG(x, p) ≥ 0.

Moreover, NG(x, p) = 0 if and only if (x, p) is a market equilibrium.

Theorem 5.5 show that Nash Gap is an ideal measure of the solution concept of market equilibrium,
since it holds following properties,

• NG(x, p) is continuous on the inputs (x, p).
• NG(x, p) ≥ 0 always hold. (under conditions in Theorem 5.5)
• NG(x, p) = 0 if and only if (x, p) meets the solution concept.
• The computation of NG does not require the knowledge of an equilibrium point (x∗, p∗)

Since some may argue that NG(x, p) is not intuitive to understand, we consider some more intuitive
measures, the Euclidean distance to the market equilibrium, i.e., ||x − x∗|| and ||p − p∗||, as
well as the difference on Weighted Social Welfare, |WSW(x) − WSW(x∗)|, where WSW(x) :=
(cid:80)

ui(xi), and show the connection between NG and these intuitive measures.

i∈[n]

Bi
Btotal

Proposition 5.6. Under some technical assumptions (which is presented in Appendix B.4), if
NG(x, p) = ε, we have:

• ||p − p∗|| = O(

√

ε).

6

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

256

257

258

259

260

261

262

263

264

• ||xi − x∗

i || = O(

√

ε) for all i.

• |WSW(x) − WSW(x∗)| = O(ε).

Finally, we give a saddle-point explaination for Nash Gap.
Corollary 5.7. Within market clearance and price constraint, we have

min
p

LFW(p) = max

x

LNW(x)

(11)

Corollary 5.7 provides an economic interpretation for GAP. Market equilibrium can be seen as the
saddle point over social welfare, and the social welfare for x can be actually implemented while
the social welfare for p is virtual and desired by buyers. Nash Gap measures the gap between the
“desired welfare” and the “implemented welfare” for buyers.

5.2 Measures in General Cases

Since NG only works for (x, p) that satisfies market clearance and price constraints, we generalize
the measure of NG to a more general case, which need to give a measure for all positive (x, p).

We first notice that any equilibrium must satisfy the conditions of market clearance and price
constraint, we first make a projection on arbitrary positive (x, p) to the space where these constraints
hold. Specifically, if we let

αj =

Vj
i xij

(cid:80)

,

˜xij = xij · αj

β =

(cid:80)
(cid:80)

i Bi
j Vjpj

,

˜pj = β · pj

(12)

then ( ˜x, ˜p) satisfies these constraints and we consider NG( ˜x, ˜p) as the equilibrium measure.

Besides, we also need to measure how far is the point (x, p) to the space within the conditions of
market clearance and price constraint. we propose following two measurement, called Violation of
Allocation (abbreviated as VoA) and Violation of Price (abbreviated as VoP), respectively.

VoA(x) :=

1
m

(cid:88)

j

| log αj|,

VoP(p) := | log β|

(13)

From the expressions of VoA and VoP, we know that these two constraints hold if and only if
VoA(x) = 0 and VoP(p) = 0.

We argue that this projection is of economic meaning. If (x, p) constitute a market equilibrium
and we scale budget with a factor of β, then (x, βp) constitute a market equilibrium in the new
market. Similarly, if we scale the value for each buyer with factor 1/α (here α can be a vector in
Rm
α p) constitute a market equilibrium in the new market.
These instances are evidence that market equilibrium holds a linear structure over market parameters.
Therefore, a linear projection can eliminate the effect from linear scaling, while preserving the effect
from orthogonal errors.

+ ) and capacity with factor α, then, (αx, 1

Notice that x = ˜x and p = ˜p if and only if VoA(x) = 0 and VoP(p) = 0, respectively. From
Theorem 5.5 We can easy derive following statements:
Proposition 5.8. For arbitrary x ∈ Rn×m
+ , we have VoA(x) ≥ 0, VoP(p) ≥
0, NG( ˜x, ˜p) ≥ 0 always hold. Moreover, (x, p) is a market equilibrium if and only if VoA(x) =
VoP(p) = NG( ˜x, ˜p) = 0.

, p ∈ Rm

+

Proposition 5.8 is a certificate that VoA(x), VoP(p), NG( ˜x, ˜p) together form a good measure for
market equilibrium. Therefore, in our experiments we compute these measures of solutions and
prefer a lower measure without further clarification.

6 Experiments

In this section, we present empirical experiments that evaluate the effectiveness of MarketFCNet.
Though briefly mentioned in this section, we leave the details of baselines, implementations, hyper-
parameters and experimental environments to Appendix C.

7

Table 1: Comparison of MarketFCNet with baselines: n = 1, 048, 576 buyers and m = 10 goods.
The GPU time for MarketFCNet represents the training time and testing time, respectively.

Methods

NG

Naïve

3.65e-1

VoA

0

VoP

GPU Time

0

3.57e-3

EG

2.17e-2

2.620e-1

7.031e-2

EG-m

2.49e-4

6.01e-2

9.77e-2

197

100

FC

1.63e-3

1.416e-2

6.750e-3

43.6; 9.63e-2

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

6.1 Experimental Settings

In our experiments, all utilities are chosen as CES utilities, which captures a wide utility class
including linear utilities, Cobb-Douglas utilities and Leontief utilities and has been widely studied in
literature [59, 4]. CES utilities have the form,





ui(xi) =



1/α

(cid:88)

ijxα
vα
ij



j∈[m]

with α ≤ 1. The fixed-price utilities for CES utility is derived in Appendix A.

In order to evaluate the performance of MarketFCNet, we compare them mainly with a baseline that
directly maximizes the objective in EG convex program with gradient ascent algorithm (abbreviated
as EG), which is widely used in the field of market equilibrium computation. Besides, we also
consider a momentum version of EG algorithm with momentum β = 0.9 (abbreviated as EG-m). We
move the details of all baselines, experimental environments and implementations of algorithms to
Appendix C.1 and Appendix C.2.

We also consider a naïve allocation and pricing rule (abbreviated as Naïve), which can be regarded as
the benchmark of the experiments:

xij = 1,

pj =

(cid:80)

i∈[n] Bi
mVj

,

for all i, j

(14)

278

279

In the following experiments, MarketFCNet is abbreviated as FC. Notice that Naïve always gives an
allocation that satisfies market clearance and price constraints, while EG, EG-m and FC do not.

280

6.2 Experiment Results

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

Comparing with Baselines We choose number of buyers n = 1, 048, 576 = 220, number of items
m = 10, CES utilities parameter α = 0.5 and representation with standard normal distribution as
the basic experimental environment of MarketFCNet; We consider NG( ˜x, ˜p), VoA(x), VoP(p) and
the running time of algorithms as the measures. Without special specification, these parameters are
default settings among other experiments. Results are presented in Table 1. From these results we
can see that the approximations of MarketFCNet are competitive with EG and EG-m and far better
than Naïve, which means that the solution of MarketFCNet are very close to market equilibrium.
MarketFCNet also achieve a much lower running time compared with EG and EG-m, which indicates
that these methods are more suitable to large-scale market equilibrium computation. In following
experiments, VoA and VoP measures are omitted and we only report NG and running time.

Experiments in different parameters settings
In this experiments, the market scale is chosen as
n = 4, 194, 304 and m = 10. We consider experiments on different distribution of representation,
including normal distribution, uniform distribution and exponential distribution. See (a) and (b)
in Figure 2 for results. We also consider different α in our experimental settings. Specifically,
our settings consist of: 1) α = 1, the utility functions are linear; 2) α = 0.5, where goods are
substitutes; 3) α = 0, where goods are neither substitutes or complements; 4) α = −1, where goods
are complements. More detailed results are shown in (c) and (d) Figure 2. The performance of
MarketFCNet is robust in both settings.

8

Figure 2: The Nash Gap and GPU running time for different algorithms: MarketFCNet, EG and
EG-m. Different colors represent for different algorithm. Market size is chosen as n = 4, 194, 304
buyers and m = 10 goods.

(a) Nash Gap on different
context distributions.

(b) GPU running time on
different context distribu-
tions.

(c) Nash Gap on different
CES utilities parameter α.

(d) GPU running time on
different CES utilities pa-
rameter α.

Figure 3: The Nash Gap and GPU running time for different algorithms: MarketFCNet, EG and
EG-m. Different colors represent for different algorithm. Market size is chosen as n = 218, 220, 222
buyers and m = 5, 10, 20 goods.

(a) Nash Gap on different market size,
n = 218, 220, 222 buyers and m =
5, 10, 20 goods.

(b) GPU running time on different mar-
ket size, n = 218, 220, 222 buyers and
m = 5, 10, 20 goods.

299

300

301

302

303

304

305

306

In this section we ask that how market size (here n
Different market scale for MarketFCNet
and m) will have impact on the efficiency of MarketFCNet. We set m = 5, 10, 20 and n = 218 =
262, 114, 220 = 1, 048, 576, 222 = 4, 194, 304 as the experimental settings. For each combination
of n and m, we train MarketFCNet and compared with EG and EG-m, see results in Figure 3. As
the market size varies, MarketFCNet has almost the same Nash Gap and running time, which shows
the robustness of MarketFCNet method over different market sizes. However, as the market size
increases, both EG and EG-m have larger Nash Gaps and longer running times, demonstrating that
MarketFCNet is more suitable to large-scale contextual market equilibrium computation.

307

7 Conclusions and Future Work

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

This paper initiates the problem of large-scale contextual market equilibrium computation from a deep
learning perspective. We believe that our approach will pioneer a promising direction for large-scale
contextual market equilibrium computation.

For future works, it would be promising to extend these methods to the case when only the number of
goods is large, or both the numbers of goods and buyers are large, which stays a blank throughout our
works. Since many existing works proposed dynamics for online market equilibrium computation,
it’s also promising to extend our approaches to tackle the online market setting with large buyers.
Besides, both existing works and ours consider sure budgets and values for buyers, and it would be
interesting to extend the fisher market and equilibrium concept when the budgets or values of buyers
are stochastic or uncertain.

9

NormalUniformExponentialDistribution1041031021011NGEGFCEG-mNormalUniformExponentialDistribution050100150200250300350400TimeEGFCEG-m10.50-11041031021011NGEGFCEG-m10.50-1050100150200250300350TimeEGFCEG-mn218220222m51020NG1051041031021011EGFCEG-mn218220222m51020time0200400600EGMarketFCNetEG-m318

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

359

360

361

362

363

364

References

[1] Shun-ichi Amari. Backpropagation and stochastic gradient descent method. Neurocomputing, 5

(4-5):185–196, 1993.

[2] Kenneth J Arrow. An extension of the basic theorems of classical welfare economics. In
Proceedings of the second Berkeley symposium on mathematical statistics and probability,
volume 2, pages 507–533. University of California Press, 1951.

[3] Kenneth J Arrow and Gerard Debreu. Existence of an equilibrium for a competitive economy.

Econometrica: Journal of the Econometric Society, pages 265–290, 1954.

[4] Kenneth J Arrow, Hollis B Chenery, Bagicha S Minhas, and Robert M Solow. Capital-labor
substitution and economic efficiency. The review of Economics and Statistics, pages 225–250,
1961.

[5] Santiago Balseiro, Christian Kroer, and Rachitesh Kumar. Contextual standard auctions with
budgets: Revenue equivalence and efficiency guarantees. Management Science, 69(11):6837–
6854, 2023.

[6] Siddhartha Banerjee, Vasilis Gkatzelis, Artur Gorokh, and Billy Jin. Online nash social welfare
maximization with predictions. In Proceedings of the 2022 Annual ACM-SIAM Symposium on
Discrete Algorithms (SODA), pages 1–19. SIAM, 2022.

[7] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning.
In Proceedings of the 26th annual international conference on machine learning, pages 41–48,
2009.

[8] Léon Bottou. Large-scale machine learning with stochastic gradient descent. In Proceedings
of COMPSTAT’2010: 19th International Conference on Computational StatisticsParis France,
August 22-27, 2010 Keynote, Invited and Contributed Papers, pages 177–186. Springer, 2010.

[9] Simina Brânzei, Yiling Chen, Xiaotie Deng, Aris Filos-Ratsikas, Søren Frederiksen, and Jie
In Proceedings of the AAAI

Zhang. The fisher market game: Equilibrium and welfare.
Conference on Artificial Intelligence, volume 28, 2014.

[10] Jonathan Brogaard, Terrence Hendershott, and Ryan Riordan. High-frequency trading and price

discovery. The Review of Financial Studies, 27(8):2267–2306, 2014.

[11] Eric Budish. The combinatorial assignment problem: Approximate competitive equilibrium

from equal incomes. Journal of Political Economy, 119(6):1061–1103, 2011.

[12] Yun Kuen Cheung, Richard Cole, and Yixin Tao. Dynamics of distributed updating in fisher
markets. In Proceedings of the 2018 ACM Conference on Economics and Computation, pages
351–368, 2018.

[13] Richard Cole and Lisa Fleischer. Fast-converging tatonnement algorithms for one-time and
ongoing market problems. In Proceedings of the Fortieth Annual ACM Symposium on Theory
of Computing, pages 315–324, 2008.

[14] Richard Cole, Nikhil Devanur, Vasilis Gkatzelis, Kamal Jain, Tung Mai, Vijay V Vazirani,
and Sadra Yazdanbod. Convex program duality, fisher markets, and nash social welfare. In
Proceedings of the 2017 ACM Conference on Economics and Computation, pages 459–460,
2017.

[15] Vincent Conitzer, Christian Kroer, Debmalya Panigrahi, Okke Schrijvers, Nicolas E Stier-Moses,
Eric Sodomka, and Christopher A Wilkens. Pacing equilibrium in first price auction markets.
Management Science, 68(12):8515–8535, 2022.

[16] Vincent Conitzer, Christian Kroer, Eric Sodomka, and Nicolas E Stier-Moses. Multiplicative

pacing equilibria in auction markets. Operations Research, 70(2):963–989, 2022.

[17] Michael Curry, Alexander R Trott, Soham Phade, Yu Bai, and Stephan Zheng. Finding general
equilibria in many-agent economic simulations using deep reinforcement learning. 2021.

10

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

405

406

407

408

409

410

411

[18] Michael Curry, Tuomas Sandholm, and John Dickerson. Differentiable economics for random-

ized affine maximizer auctions. arXiv preprint arXiv:2202.02872, 2022.

[19] Xiaotie Deng, Christos Papadimitriou, and Shmuel Safra. On the complexity of equilibria. In
Proceedings of the Thiry-fourth Annual ACM Symposium on Theory of Computing, pages 67–71,
2002.

[20] Xiaotie Deng, Christos Papadimitriou, and Shmuel Safra. On the complexity of price equilibria.

Journal of Computer and System Sciences, 67(2):311–324, 2003.

[21] Zhijian Duan, Jingwu Tang, Yutong Yin, Zhe Feng, Xiang Yan, Manzil Zaheer, and Xiaotie Deng.
A context-integrated transformer-based neural network for auction design. In International
Conference on Machine Learning, pages 5609–5626. PMLR, 2022.

[22] Zhijian Duan, Wenhan Huang, Dinghuai Zhang, Yali Du, Jun Wang, Yaodong Yang, and Xiaotie
Deng. Is nash equilibrium approximator learnable? In Proceedings of the 2023 International
Conference on Autonomous Agents and Multiagent Systems, pages 233–241, 2023.

[23] Zhijian Duan, Yunxuan Ma, and Xiaotie Deng. Are equivariant equilibrium approximators
beneficial? In Proceedings of the 40th International Conference on Machine Learning, ICML’23.
JMLR.org, 2023.

[24] Zhijian Duan, Haoran Sun, Yurong Chen, and Xiaotie Deng. A scalable neural network for
DSIC affine maximizer auction design. 2023. URL https://openreview.net/forum?id=
cNb5hkTfGC.

[25] Paul Dütting, Zhe Feng, Harikrishna Narasimhan, David Parkes, and Sai Srivatsa Ravindranath.
Optimal auctions through deep learning. In International Conference on Machine Learning,
pages 1706–1715. PMLR, 2019.

[26] Paul Dütting, Zhe Feng, Harikrishna Narasimhan, David C Parkes, and Sai Srivatsa Ravin-
dranath. Optimal auctions through deep learning: Advances in differentiable economics. Journal
of the ACM (JACM), 2023.

[27] Edmund Eisenberg. Aggregation of utility functions. Management Science, 7(4):337–350,

1961.

[28] Edmund Eisenberg and David Gale. Consensus of subjective probabilities: The pari-mutuel

method. The Annals of Mathematical Statistics, 30(1):165–168, 1959.

[29] Zhe Feng, Harikrishna Narasimhan, and David C Parkes. Deep learning for revenue-optimal
auctions with budgets. In Proceedings of the 17th International Conference on Autonomous
Agents and Multiagent Systems, pages 354–362, 2018.

[30] Yuan Gao and Christian Kroer. First-order methods for large-scale market equilibrium computa-

tion. Advances in Neural Information Processing Systems, 33:21738–21750, 2020.

[31] Yuan Gao and Christian Kroer. Infinite-dimensional fisher markets and tractable fair division.

Operations Research, 71(2):688–707, 2023.

[32] Yuan Gao, Alex Peysakhovich, and Christian Kroer. Online market equilibrium with application
to fair division. Advances in Neural Information Processing Systems, 34:27305–27318, 2021.

[33] Jugal Garg, Ruta Mehta, Vijay V Vazirani, and Sadra Yazdanbod. Settling the complexity of
leontief and plc exchange markets under exact and approximate equilibria. In Proceedings of
the 49th Annual ACM SIGACT Symposium on Theory of Computing, pages 890–901, 2017.

[34] Jugal Garg, Yixin Tao, and László A Végh. Approximating equilibrium under constrained
piecewise linear concave utilities with applications to matching markets. In Proceedings of
the 2022 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 2269–2284.
SIAM, 2022.

[35] Jonathan Goldman and Ariel D Procaccia. Spliddit: Unleashing fair division algorithms. ACM

SIGecom Exchanges, 13(2):41–46, 2015.

11

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

[36] Noah Golowich, Harikrishna Narasimhan, and David C Parkes. Deep learning for multi-facility
location mechanism design. In International Joint Conferences on Artificial Intelligence, pages
261–267, 2018.

[37] Xue-Zhong He and Shen Lin. Reinforcement learning equilibrium in limit order markets.

Journal of Economic Dynamics and Control, 144:104497, 2022.

[38] Howard Heaton, Daniel McKenzie, Qiuwei Li, Samy Wu Fung, Stanley Osher, and Wotao Yin.

Learn to predict equilibria via fixed point networks. arXiv preprint arXiv:2106.00906, 2021.

[39] Edward Hill, Marco Bardoscia, and Arthur Turrell. Solving heterogeneous general equilibrium
economic models with deep reinforcement learning. arXiv preprint arXiv:2103.16977, 2021.

[40] Zhiyi Huang, Minming Li, Xinkai Shu, and Tianze Wei. Online nash welfare maximization
In International Conference on Web and Internet Economics, pages

without predictions.
402–419. Springer, 2023.

[41] Devansh Jalota and Yinyu Ye. Stochastic online fisher markets: Static pricing limits and adaptive

enhancements. arXiv preprinted arXiv:2205.00825, 2023.

[42] Devansh Jalota, Marco Pavone, Qi Qi, and Yinyu Ye. Fisher markets with linear constraints:
Equilibrium properties and efficient distributed algorithms. Games and Economic Behavior,
141:223–260, 2023.

[43] Nils Kohring, Fabian Raoul Pieroth, and Martin Bichler. Enabling first-order gradient-based
learning for equilibrium computation in markets. In International Conference on Machine
Learning, pages 17327–17342. PMLR, 2023.

[44] Christian Kroer. Ai, games, and markets. 2023. https://www.columbia.edu/~ck2945/

files/main_ai_games_markets.pdf.

[45] Christian Kroer and Alexander Peysakhovich. Scalable fair division for’at most one’preferences.

arXiv preprint arXiv:1909.10925, 2019.

[46] Christian Kroer, Alexander Peysakhovich, Eric Sodomka, and Nicolas E Stier-Moses. Comput-
ing large market equilibria using abstractions. In Proceedings of the 2019 ACM Conference on
Economics and Computation, pages 745–746, 2019.

[47] Ningyuan Li, Yunxuan Ma, Yang Zhao, Zhijian Duan, Yurong Chen, Zhilin Zhang, Jian Xu,
Bo Zheng, and Xiaotie Deng. Learning-based ad auction design with externalities: The frame-
work and a matching-based approach. In Proceedings of the 29th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining, pages 1291–1302, 2023.

[48] Luofeng Liao, Yuan Gao, and Christian Kroer. Nonstationary dual averaging and online fair
allocation. Advances in Neural Information Processing Systems, 35:37159–37172, 2022.

[49] Yuxuan Lu, Qian Qi, and Xi Chen. A framework of transaction packaging in high-throughput

blockchains. arXiv preprint arXiv:2301.10944, 2023.

[50] Luke Marris, Ian Gemp, Thomas Anthony, Andrea Tacchetti, Siqi Liu, and Karl Tuyls. Tur-
bocharging solution concepts: Solving nes, ces and cces with neural equilibrium solvers.
Advances in Neural Information Processing Systems, 35:5586–5600, 2022.

[51] Andreu Mas-Colell, Michael Dennis Whinston, Jerry R Green, et al. Microeconomic theory,

volume 1. Oxford University Press New York, 1995.

[52] Tianlong Nan, Yuan Gao, and Christian Kroer. Fast and interpretable dynamics for fisher

markets via block-coordinate updates. arXiv preprint arXiv:2303.00506, 2023.

[53] Noam Nisan, Tim Roughgarden, Eva Tardos, and Vijay V Vazirani. Algorithmic game theory,

2007. Book available for free online, 2007.

[54] Christos Papadimitriou. Algorithms, games, and the internet. In Proceedings of the Thirty-third

Annual ACM Symposium on Theory of Computing, pages 749–753, 2001.

12

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

[55] Jad Rahme, Samy Jelassi, Joan Bruna, and S Matthew Weinberg. A permutation-equivariant
neural network architecture for auction design. In Proceedings of the AAAI Conference on
Artificial Intelligence, volume 35, pages 5664–5672, 2021.

[56] Weiran Shen, Sébastien Lahaie, and Renato Paes Leme. Learning to clear the market. In

International Conference on Machine Learning, pages 5710–5718. PMLR, 2019.

[57] Vadim I Shmyrev. An algorithm for finding equilibrium in the linear exchange model with fixed

budgets. Journal of Applied and Industrial Mathematics, 3:505–518, 2009.

[58] The Sveriges Riksbank Prize in Economic Sciences in Memory of Alfred Nobel 1972. No-
belprize.org. Nobel Prize Outreach AB 2024, Sun. 28 Jan 2024. https://www.nobelprize.
org/prizes/economic-sciences/1972/summary/.

[59] Hal R Varian and Hal R Varian. Microeconomic analysis, volume 3. Norton New York, 1992.

[60] Vijay V Vazirani and Mihalis Yannakakis. Market equilibrium under separable, piecewise-linear,

concave utilities. Journal of the ACM (JACM), 58(3):1–25, 2011.

[61] Leon Walras. Elements of pure economics. Routledge, 2013.

[62] Tonghan Wang, Paul Dütting, Dmitry Ivanov, Inbal Talgam-Cohen, and David C Parkes.
Deep contract design via discontinuous piecewise affine neural networks. arXiv preprint
arXiv:2307.02318, 2023.

[63] Fang Wu and Li Zhang. Proportional response dynamics leads to market equilibrium. In
Proceedings of the Thirty-ninth Annual ACM Symposium on Theory of Computing, pages
354–363, 2007.

[64] Ruitu Xu, Yifei Min, Tianhao Wang, Michael I Jordan, Zhaoran Wang, and Zhuoran Yang.
Finding regularized competitive equilibria of heterogeneous agent macroeconomic models via
reinforcement learning. In International Conference on Artificial Intelligence and Statistics,
pages 375–407. PMLR, 2023.

[65] Zongjun Yang, Luofeng Liao, and Christian Kroer. Greedy-based online fair allocation with
adversarial input: Enabling best-of-many-worlds guarantees. arXiv preprint arXiv:2308.09277,
2023.

[66] Li Zhang. Proportional response dynamics in the fisher market. Theoretical Computer Science,

412(24):2691–2698, 2011.

13

14

16

22

(15)

487

Appendix

488

A Derivation of Fixed-price Utility for CES Utility Functions

489

B Omitted Proofs

490

C Additional Experiments Details

491

A Derivation of Fixed-price Utility for CES Utility Functions

492

493

494

495

496

In this section we show the explicit expressions of Fixed-price Utility for CES utility functions.

We first consider the case α ̸= 0, 1, −∞. The optimization problem for consumer i is:

max
xij ,j∈[m]

ui(xi) =







1/α

(cid:88)

ijxα
vα
ij



j∈[m]

(cid:88)

s.t.

xijpj = Bi

j∈[m]
xij ≥ 0

(Budget Constraint)

(16)

Not hard to verify that in an optimal solution with Equation (Budget Constraint), Equation (16)
always holds, therefore we omit this constraint in our derivation.

We write the Lagrangian L(xi, λ)

L(xi, λ) = ui(xi) + λ(Bi −

(cid:88)

j∈[m]

xijpj)

497

By ∂L
∂xij

= 0, we have

498

We derive that

∂ui
∂x∗
ij

(xi) = λpj

∂ui
∂xij

(xi) =

1
α







1/α−1

(cid:88)

ijxα
vα
ij



· αvα

ijxα−1
ij

j∈[m]

ijxα−1
vα

ij =cpj

· · · let c = λ ·







1/α−1

(cid:88)

ijxα
vα
ij



j∈[m]

x∗
ij =

α
1−α
ij

v

1
1
1−α
1−α · p
j

c

499

Taking (21) into (Budget Constraint), we get

Bi =

α
1−α
ij
1
1−α

v

c

(cid:88)

j∈[m]

− α
· p
j

1−α

1

1−α =

c

(cid:19) α

1−α

1
Bi

(cid:88)

j∈[m]

(cid:18) vij
pj

14

(17)

(18)

(19)

(20)

(21)

(22)

(23)

500

Taking Equation (23) into Equation (21), we get

x∗
ij =

α
1−α
ij

v

1
1−α
j

p

·

Bi
c0

501

where c0 = (cid:80)

j∈[m]

(cid:17) α

1−α

(cid:16) vij
pj

502

Taking Equation (24) into Equation (15), we finally have

ui(x∗

i ) = (cid:2)vα


ijxα
ij

(cid:3) 1

α

(cid:88)

vα
ij

=




j∈[m]



cα
0




α2
1−α
ij

v

α
1−α
p
j

(cid:19) α

1−α

(cid:18) vij
pj



cα
0





=



(cid:88)

j∈[m]

=Bic

1−α
α

0

log ˜ui(p) = log ui(x∗

i ) = log Bi +

1 − α
α

log c0

(24)

(25)

503

504

For α = 1, by simple arguments we know that consumer will only buy the good that with largest
value-per-cost, i.e., vij/pj. Therefore, we have

log ˜ui(p) = log Bi + log max

j

vij
pj

505

506

For α = 0, we have log ui(xi) = 1
vt

Similarly, we have

(cid:80)

j∈[m] vij log xij where vt = (cid:80)

j∈[m] vij.

=

vij
xij

cpj =

x∗
ij =

∂ log ui
∂xij
vij
cpj

507

By solving budget constraints we have c = vt
Bi

, and therefore, x∗

ij = vij Bi
pj vt

and

log ui(x∗

i ) =

1
vt

(cid:88)

j∈[m]

(vij log

vijBi
pjvt

)

= log Bi +

(cid:88)

j∈[m]

vij
vt

log

vij
pjvt

(26)

(27)

(28)

(29)

(30)

508

For α = −∞, we can easily know that vijx∗

ij ≡ c for some c. By solving budget constraint we have

(cid:88)

j∈[m]

cpj
vij

= Bi

c = Bi





(cid:88)

j∈[m]



−1



pj
vij

log ˜ui(p) = log c = log Bi − log

(cid:88)

j∈[m]

pj
vij

15

(31)

(32)

(33)

509

Above all, the log Fixed-price Utility for CES functions is
log Bi + maxj log vij
pj
log Bi + (cid:80)
vij
vt
log Bi − log (cid:80)
log Bi + 1−α

log ˜ui(p) =





j∈[m]
α log c0

j∈[m]

log vij
pj vt
pj
vij

others

for α = 1

for α = 0

for α = −∞

510

B Omitted Proofs

511

512

B.1 Proof of Proposition 3.2

We consider Lagrangian multipliers p and use the KKT condition. The Lagrangian becomes

L(p, x) =

(cid:88)

i

Bi log ui(xi) −

(cid:88)

(cid:88)

pj(

j

i

xij − Yj)

513

and the partial derivative of xij is

514

By complementary slackness of xij ≥ 0, we have

∂L(p, xi)
∂xij

=

Bi
ui(xi)

∂ui
∂xij

− pj

Bi
ui(xi)

∂ui
∂xij

≤ pj for all i

(34)

(35)

(36)

(37)

515

516

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

By theorem 3.1, we know that if (x, p) is a market equilibrium, we must have ui(xi) > 0 for all i,
and by condition in Proposition 3.2, we can always select buyer i such that ∂ui
> 0. Therefore, we
∂xij
have pj > 0.
As a consequence, pj > 0 indicates that (cid:80)

j xij = Vj by market clearance condition.

B.2 Proof of Proposition 5.4

Consider the market equilibrium condition ⟨p∗, x∗
expression, we have (cid:80)
(cid:80)
i
(cid:80)n

i=1 xij = Yj in market equilibrium, so (cid:80)

i Bi. Then, (cid:80)
j pjYj = (cid:80)

j pjxij = (cid:80)

i ⟩ = Bi, we have (cid:80)
(cid:80)

j pjxij = Bi. sum over this
i Bi. Notice that we have

i xij = (cid:80)

j pj
i Bi, that completes the proof.

B.3 Proof of Theorem 5.5

Proof of Theorem 5.5. Denote (x, p) as the market equilibrium, p as the price for goods and x∗
as the optimal consumption set of buyer i when the price is p.

i (p)

We have following equation:

(cid:88)

j

xijpj =Bi

xi ∈x∗

i (p)

xij =Yj

(cid:88)

i∈[n]

ui(p) =ui(xi), ∀p ∈ Rm

+ , ∀xi ∈ x∗

i (p)

(38)

(39)

(40)

(41)

From Proposition 5.4 we know (cid:80)
Let p′ be some price for items such that (cid:80)
⟨p′, xi⟩. We know that

i∈[n] Bi = (cid:80)

j∈[m] Yjpj.

j∈[m] Yjp′

j = (cid:80)

i∈[n] Bi. Let x′

i ∈ x∗

i (p′) and B′

i =

(cid:88)

i∈[n]

B′

i = ⟨p′,

(cid:88)

i∈[n]

xi⟩ = ⟨p′, Y ⟩ =

(cid:88)

Bi

i∈[n]

(42)

16

530

531

532

533

For consumer i, xi costs B′
price p′, and x′ is the optimal consumption for buyer i. Then we have

i at price p′, thus Bi
B′
i

xi costs Bi at price p′. Besides, x′

i also costs Bi for

ui(p′) = ui(x′

i) ≥ ui(

Bi
B′
i

xi) =

Bi
B′
i

ui(xi)

(43)

where the last equation is from the homogeneity(with degree 1) of utility function.

Taking logarithm and weighted sum with Bi, we have

Bi log ui(p′) ≥

(cid:88)

i∈[n]

(cid:88)

i∈[n]

Bi log

Bi
B′
i

(cid:88)

+

i∈[n]

Bi log ui(xi)

(44)

534

Take Btotal = (cid:80)

i∈[n] Bi, the first term in RHS becomes

(cid:88)

i∈[n]

Bi log

=Btotal

(cid:88)

i∈[n]

Bi
B′
i
(cid:18) Bi
Btotal

(cid:19)

log

Bi/Btotal
B′
i/Btotal

535

Therefore,

=Btotal · KL(

B
Btotal

||

B′
Btotal

)

≥ 0

(cid:88)

i∈[n]

Bi log ui(p′) ≥

(cid:88)

i∈[n]

Bi log ui(xi)

536

For x′ that satisfies market clearance, by optimality of EG program(EG), we have

(cid:88)

i∈[n]

Bi log ui(xi) ≥

(cid:88)

i∈[n]

Bi log ui(x′
i)

(45)

(46)

(47)

(48)

(49)

(50)

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

Equation (49) and Equation (50) together complete the proof of the first part.

If (x, p) constitutes a market equilibrium, it’s obvious that LFW(p) and LNW(x) are identical,
therefore NG(x, p) = 0.

On the other hand, if (x, p) is not a market equilibrium, but NG(x, p) = 0, it means that the KL
convergence term must equal to 0, and Bi = B′
i for all i, which means that xi costs buyer i with
money Bi and xi are in the consumption set of buyer i. Since (x, p) is not a market equilibrium,
there is at least one buyer that can choose a better allocation x′
i to improve her utility, therefore
improve LFW(p), and it cannot be the case that LFW(p) = LNW(x), which makes a contradiction.

B.4 Proof of Proposition 5.6

We leave the formal presentation of Proposition 5.6 and proofs to three theorems below.
Lemma B.1. Assume that ui(xi) is twice differentiable and denote H(xi) as the Hessian matrix of
ui(xi). If following hold:

• H(x∗

i ) has rank m − 1

• ||xi − x∗

i || = ε for some i

• x∗

i > 0

then we have OPT − LNW(x) = Ω(ε2).

17

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

Lemma B.2. Denote ˜ui(p, Bi) and x∗
i (p, Bi) as the maximum utility buyer i can get and the
corresponding consumption for buyer i when her budget is Bi and prices are p. If following hold:

• ||p − p∗|| = ε

• x∗

i (p, Bi) is differentiable with p.

• HX := ((cid:80)

i∈[n]

∂x∗
ij
∂pk

(p∗, Bi))j,k∈[m] has full rank.

then we have LF W (p) − OP T = Ω(ε2).
Remark B.3. It’s worth notice that H(x∗
homogeneous and thus linear in the direction x. Therefore, we have H(xi)xi = 0 for all xi.
Let Ci = {xi ∈ Rm
with Ci, the condition that H(x∗
on the consumption set Ci.

+ : ⟨p, xi⟩ = Bi} be the consumption set of buyer i, since xi can not be parallel
i ) has rank m − 1 means that, H(xi) is strongly concave at point x∗
i

i ) can not has full rank m, since ui(x) is assumed to be

Besides, we emphasize that the conditions in Lemma B.1 and Lemma B.2 are satisfied for CES utility
with α < 1.

Corollary B.4. Under the assumptions in Lemma B.1 and Lemma B.2, if NG(x, p) = ε, we have:

• ||p − p∗|| = O(

√

ε)
√

• ||xi − x∗

i || = O(

ε) for all i

Proof of Corollary B.4. A direct inference from Lemma B.1 and Lemma B.2, notice that NG = ε
indicates that OPT − LNW(x) ≤ ε and LFW(p) − OPT ≤ ε.

Corollary B.4 states that, for a pair of (x, p) that satisfy market clearance and price constraints, a
small Nash Gap indicates that the point (x, p) is close to the equilibrium point (x∗, p∗), in the sense
of Euclidean distance.

Lemma B.5. Assume following hold:

• buyers have same utilities at x∗, i.e. ui(x∗

i ) = uj(x∗

j ) ≡ c for all i, j

• ||xi − x∗

i || ≤ ε for all i

then, we have |WSW(x) − WSW(x∗)| = O(ε2).

Remark B.6. These conditions can be held when buyers are homogeneous, i.e., Bi = Bj and
ui(x) = uj(x) for all i, j, x ∈ Rm
+ . Besides, consider buyers with same budgets, these conditions
can also be held if the market has some “equivariance property”, e.g., there is a n-cycle permutation of
buyers ρ and permutation of goods τ , such that ui(xi) = uρ(i)(τ (xρ(i))) for all i and τ (Y1, ..., Ym) =
(Y1, ..., Ym).
Corollary B.7. Under the assumptions in Lemma B.1 and Lemma B.5, if NG(x, p) = ε, we have

• |WSW(x) − WSW(x∗)| = O(ε).

586

Proof. A direct inference from Lemma B.1 and Lemma B.5.

587

588

B.4.1 Proof of Lemma B.1

Proof of Lemma B.1. We observe that

OPT − LNW(x) =

(cid:88)

i∈[n]

Bi [log ui(x∗

i ) − log ui(xi)]

18

589

Consider the Taylor expansion of log ui(xi) and ui(xi):

log ui(xi) = log ui(x∗

(ui(xi) − ui(x∗

i ))

i ) +

1
ui(x∗
i )
i )2 (ui(xi) − ui(x∗

i ))2

−

1
2ui(x∗

+O((ui(xi) − ui(x∗

i ))3)

ui(xi) =ui(x∗

i ) +

i )(xi − x∗
i )

(x∗

∂ui
∂xi
i )T H(x∗

+

1
2

(xi − x∗

i )(xi − x∗

i ) + O(||xi − x∗

i ||3)

590

Notice that ||xi − x∗

i || = ε, we have
log ui(xi) = log ui(x∗
i )
∂ui
∂xi

+

[

1
ui(x∗
i )
1
(xi − x∗
2

+

−

1
2ui(x∗
+O(ε3)

i )2

(x∗

i )(xi − x∗

i ) · · · ε term

i )T H(x∗
(cid:18) ∂ui
∂xi

i )(xi − x∗

i )] · · · ε2 term

(cid:19)2

(x∗

i )(xi − x∗
i )

· · · ε2 term

591

We next deal with Equation (51) to Equation (53) separately.

592

593

594

Derivation of Equation (51) Since x∗

i solves the buyer i’s problem, we must have

∂ui
∂xi

(x∗

i ) = λip∗

where λi is the Lagrangian Multipliers for buyer i.

We also know that ui(xi) is homogeneous with degree 1, by Euler formula, we derive

595

Combine Equation (54) and Equation (55) and take xi = x∗

i , we derive

⟨

∂ui
∂xi

(xi), xi⟩ = ui(xi)

λi⟨p∗, x∗

λi =

i ⟩ =ui(x∗
i )
ui(x∗
i )
Bi
ui(x∗
i )
Bi

i ) =

(x∗

p∗

596

Sum up over i for Equation (51), we have

∂ui
∂xi

(cid:88)

Bi

1
ui(x∗
i )

∂ui
∂xi

(x∗

i )(xi − x∗
i )

i∈[n]
(cid:88)

=p

(xi − x∗
i )

(51)

(52)

(53)

(54)

(55)

(56)

i∈[n]

=0 · · · by market clearance

597

598

Derivation of Equation (52) and Equation (53) Combining Equation (52) and Equation (53), we
have

(xi − x∗

i )T H(x∗

i )(xi − x∗

i ) −

1
2Bi

(xi − x∗

i )T (p∗p∗T )(xi − x∗
i )

Bi
2ui(x∗
i )
1
2Bi

=

(xi − x∗

i )T (

B2
i
ui(x∗
i )

H(x∗

i ) − p∗p∗T )(xi − x∗
i )

19

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

Denote H0(x∗

i ) = B2
i
ui(x∗

i ) H(x∗

i ) − p∗p∗T , next we assert that H0(x∗

i ) is negative definite.

i ) and −p∗p∗T are negative semi-definite, H0(x∗
i )) ≥ m − 1.

Since H(x∗
rank(H0(x∗
Let λ1 ≤ λ2 ≤ · · · ≤ λm−1 < λm = 0 be eigenvalues and v1, ..., vn = x∗
of H(x∗
i )) = m − 1, it means that x∗
eigenvalue 0. However, we have −p∗p∗T x∗
Therefore, we have rank(H0(x∗
n < 0 as the eigenvalues of H0(x∗
λi

i = −Bip∗ ̸= 0, which leads to a contradiction.
i ) is negative definite, we denote λi

i ), and k as the universal lower bound for |λi

i be eigenvectors
i has to be eigenvectors of −p∗p∗T with

i ) must be negative semi-definite with

1 ≤ ..., ≤
n|, then we have that,

i )) = m and H0(x∗

i ). If rank(H0(x∗

1
2

(xi − x∗

i )T H0(x∗

i )(xi − x∗

i ) ≤ −

k
2

ε2

By combining Equation (56) and Equation (57), we have

OPT − LNW(x) = −

(cid:88)

Bi

i∈[n]

(cid:20) 1
2Bi

(xi − x∗

i )T H0(x∗

i )(xi − x∗
i )

ε2 + O(ε3)

≥

k
2
=Ω(ε2)

(cid:21)

+ O(ε3)

(57)

(58)

B.4.2 Proof of Lemma B.2

Proof of Lemma B.2. The proof is similar with Appendix B.4.1 by using Taylor expansion technique.
Before that, we first derive some identities.

By Roy’s identity, we have

∂ ˜ui
∂pj

(p, Bi) = −x∗

ij(p, Bi)

∂ ˜ui
∂Bi

(p, Bi)

613

Since u(xi) is homogeneous with xi, it’s easy to derive that

614

Above all,

615

Besides,

∂ ˜ui
∂Bi

(p, Bi) =

˜ui(p, Bi)
Bi

∂ ˜ui
∂pj

(p, Bi) = −

1
Bi

x∗
ij(p, Bi)˜ui(p, Bi)

∂2 ˜ui
∂pj∂pk

(p, Bi) =

−

1
B2
i
1
Bi

ij(p, Bi)x∗
x∗

ik(p, Bi)˜ui(p, Bi)

x∗
ij(p, Bi)
∂pk

˜ui(p, Bi)

616

Next we consider the Taylor expansion,

log ˜ui(p) = log ˜ui(p∗)
∂ ˜ui
∂p

+

[

1
˜ui(p∗)
1
2

+

−

(p∗)(p − p∗) · · · ε term

(p − p∗)T Hp(p − p∗)] · · · ε2 term

1
2˜ui(p∗)2

(cid:20) ∂ ˜ui
∂p

(cid:21)2

(p∗)(p − p∗)

· · · ε2 term

(59)

(60)

(61)

617

where Hp is the Hessian matrix for ˜ui(p).

+O(ε3)

20

618

Derivation of Equation (59) We have

(p∗), (p − p∗)⟩

Bi

1
˜ui(p∗)

⟨

∂ ˜ui
∂p

⟨x∗

i , (p − p∗)⟩

(cid:88)

i∈[n]
(cid:88)

i∈[n]

=

=⟨1, (p − p∗)⟩ · · · by market clearance
=0 · · · by price constraints

619

Derivation of Equation (60) and Equation (61) These expressions become

˜ui(p∗)⟨x∗

i , p − p∗⟩2 −

1
Bi

˜ui(p∗)(p − p∗)T (

∂x∗
ij
∂pk

(p∗, Bi))j,k∈[m](p − p∗)]

1
B2
i
˜ui(p∗)2
B2
i

[

1
2˜ui(p∗)
1
2˜ui(p∗)2
1
2Bi

−

=

⟨x∗

i , p − p∗⟩2

(p − p∗)T (

∂x∗
ij
∂pk

(p∗, Bi))j,k∈[m](p − p∗)

620

Summing up over i, we derive that

LFW(p) − OPT =

(cid:88)

Bi

1
2Bi

(p − p∗)T (

∂x∗
ij
∂pk

(p∗, Bi))j,k∈[m](p − p∗) + O(ε3)

i∈[n]
1
2

=

(p − p∗)T HX (p − p∗) + O(ε3)

621

622

623

Since p∗ gets the minimum of LFW(p), we must have that HX is positive semi-definite. Together
with HX has full rank, we know that HX is positive definite. Denote λm as the minimum eigenvalues
of HX , we have

LFW(p) − OPT ≥

ε2λm
2
=Ω(ε2)

+ O(ε3)

624

625

626

B.4.3 Proof of Lemma B.5

Proof of Lemma B.5. Notice that

WSW(x) = WSW(x∗) +

(cid:88)

i∈[n]

⟨

∂WSW
∂xi

(x∗

i ), (xi − x∗

i )⟩ + O(ε2)

627

We have

628

Therefore,

(x∗
i )

∂WSW
∂xi
∂ui
∂xi
ui(x∗
i )
Bi

=Bi

=Bi

(x∗
i )

p∗

=cp∗

WSW(x) =WSW(x∗) +

(cid:88)

i∈[n]

c⟨p∗, xi − x∗

i ⟩ + O(ε2)

=WSW(x∗) + O(ε2) · · · market clearance

which completes the proof.

629

630

21

631

C Additional Experiments Details

632

633

634

635

636

637

638

639

640

641

642

643

644

645

646

647

648

649

650

651

652

653

654

655

656

657

658

659

660

661

662

663

664

665

666

C.1 More about baselines

EG program solver (abbreviated as EG) We propose the first baseline algorithm EG. Recall the
Eisenberg-Gale convex program(EG):

max

1
n

n
(cid:88)

i=1

Bi log ui(xi)

s.t.

1
n

n
(cid:88)

i=1

xij = 1, x ≥ 0.

(62)

We use the network module in pytorch to represent the parameters x ∈ Rn×m
, and softplus activation
function to satisfy x ≥ 0 automatedly. We use gradient ascent algorithm to optimize the parameters
x. For constraint 1
i∈[n] xij = 1, we introduce Lagrangian multipliers λj and minimize the
n
Lagrangian:

(cid:80)

+

Lρ(x; λ) = −

1
n

(cid:88)

i∈[n]


+

ρ
2

(cid:88)



j∈[m]

Bi log ui(xi) +



λj



(cid:88)

j∈[m]

1
n

(cid:88)

i∈[n]



xij − 1




2

xij − 1



1
n

(cid:88)

i∈[n]

(63)

(64)

(cid:17)
i∈[n] xij − 1
The updates of λ is λj ← λj + βtρ
, here βt is step size, which is identical with
that in MarketFCNet. The algorithm returns the final (x, λ) as the approximated market equilibrium.

(cid:16) 1
n

(cid:80)

EG program solver with momentum (abbreviated as EG-m) The program to solve is exactly
same with that in EG. The only difference is that we use gradient ascent with momentum to optimize
the parameters x.

C.2 More Experimental Details

Without special specification, we use the experiment settings as follows. All experiments are con-
ducted in one RTX 4090 graphics cards using 16 CPUs or 1 GPU. We set dimension of representations
of buyers and goods to be d = 5. Each elements in representation is i.i.d from N (0, 1) for normal
distribution (default) contexts, U [0, 1] for uniform distribution contexts and Exp(1) for exponential
distribution contexts. Budget is generated with B(b) = ||b||2, and valuation in utility function is
generated with v(b, g) = softplus(⟨b, g⟩), where softplus(x) = log(1 + exp(x)) is a smoothing
function that maps each real number to be positive. α in CES utility are chosen to be 0.5 by default.
MarketFCNet is designed as a fully connected network with depth 5 and width 256 per layer. ρ is
chosen to be 0.2 in Augmented Lagrange Multiplier Method and the step size βt is chosen to be 1√
.
t
We choose K = 100 as inner iteration for each epoch, and training for 30 epochs in MarketFCNet.
For EG and EG-m baselines, we choose the inner iteration K = 1000 when n > 1000 and K = 100
when n ≤ 1000 for each epoch. Baselines are enssembled with early stopping as long as NG is lower
than 10−3. Both baselines are optimized for 30 epochs in total.

We use Adam optimizer and learning rate 1e − 4 to optimize the allocation network in MarketFCNet.
When computing ∆λj in MarketFCNet, we directly compute ∆λj rather than generate an unbiased
estimator, since it does not cost too much to consider all buyers for one time. For those baselines,
we use gradient descent to optimize the parameters following existing works, and the step size is
fine-tuned to be 1e + 2 for α = 1, n > 1000; 1e + 3 for α < 1, n > 1000 and 1 for α < 1, n ≤ 1000
and 0.1 for α = 1, n ≤ 1000 for better performances of the baselines. Since that Lagrangian
multipliers λ ≤ 0 will indicate an illegal Nash Gap measure, therefore, we hard code EG algorithm
such that it will only return a result when it satisfies that the price λj > 0 for all good j. All baselines
are run in GPU when n > 1000 and CPU when n ≤ 1000.1

1We find in the experiments when market size is pretty large, baselines run slower on CPU than on GPU and
this phenomenon reverses when market size is small. Therefore, the hardware on which baselines run depend on
the market size and we always choose the faster one in experiments.

22

667

NeurIPS Paper Checklist

668

669

670

671

672

673

674

675

676

677

678

679

680

681

682

683

684

685

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

702

703

704

705

706

707

708

709

710

711

712

713

714

715

716

717

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?

Answer: [TODO][Yes]

Justification: [TODO]

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

Answer: [TODO][Yes]

Justification: [TODO]We discuss the limitations in Section 7.

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

Answer: [TODO][No]

23

718

719

720

721

722

723

724

725

726

727

728

729

730

731

732

733

734

735

736

737

738

739

740

741

742

743

744

745

746

747

748

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

Justification: [TODO]The answer is [Yes] except for Theorem 3.1. Theorem 3.1 is a
restated theorem of Gao and Kroer [30] and we do not cover that proof in this paper.
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
Answer: [TODO][Yes]
Justification: [TODO]We present the experimental details in Appendix C.
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

24

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

Answer: [TODO][No]
Justification: [TODO]The code need to be more finely organized before it goes public.
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
Answer: [TODO][Yes]
Justification: [TODO]These are presented in Appendix C
Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail

that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental

material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [TODO][No]
Justification: [TODO]Since the difference between baselines and our method is promi-
nent, we believe that one experiment on each setting is an enough certificate to show the
effectiveness of our method.
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

25

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

Answer: [TODO][Yes]

Justification: [TODO]See Appendix C.

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

Answer: [TODO][Yes]

Justification: [TODO]

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a

deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consid-

eration due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?

Answer: [TODO][Yes]

Justification: [TODO]The accelaration of market equilibrium computation is a positive
social impact.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal

impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

26

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

Answer: [TODO][NA]

Justification: [TODO]

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

Answer: [TODO][NA]

Justification: [TODO]

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

27

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

• If this information is not available online, the authors are encouraged to reach out to

the asset’s creators.

13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [TODO][NA]
Justification: [TODO]
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
Answer: [TODO][NA]
Justification: [TODO]
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
Answer: [TODO][NA]
Justification: [TODO]
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

28

