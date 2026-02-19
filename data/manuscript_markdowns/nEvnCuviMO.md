Metric Distortion Under Probabilistic Voting

Anonymous Author(s)

Abstract

Metric distortion in social choice provides a framework for assessing how well
voting rules minimize social cost in scenarios where voters and candidates exist
in a shared metric space, with voters submitting rankings and the rule outputting
a single winner. We expand this framework to include probabilistic voting. Our
extension encompasses a broad range of probability functions, including widely
studied models like Plackett-Luce (PL) and Bradley-Terry, and a novel "pairwise
quantal voting" model inspired by quantal response theory. We demonstrate that
distortion results under probabilistic voting better correspond with conventional
intuitions regarding popular voting rules such as Plurality, Copeland, and Random
Dictator (RD) than those under deterministic voting. For example, in the PL model
with candidate strength inversely proportional to the square of their metric distance,
we show that Copeland’s distortion is at most 2, whereas that of RD is Ω(
m) in
large elections, where m is the number of candidates. This contrasts sharply with
the classical model, where RD beats Copeland with a distortion of 3 versus 5 [1].

√

1

Introduction

Societies must make decisions collectively; different agents often have conflicting interests, and the
choice of the mechanism used for combining everyone’s opinions often makes a big difference to the
outcome. The machine learning community has applied social choice principles for AI alignment
[2, 3], algorithmic fairness [4, 5], and preference modelling [6, 7]. Over the last century, there has
been increasing interest in using computational tools to analyse and design voting rules [8–11]. One
prominent framework for evaluating voting rules is that of distortion [12], where the voting rule has
access to only the ordinal preferences of the voters. However, the figure of merit is the sum of all
voters’ cardinal utilities (or costs). The distortion of a voting rule is the worst-case ratio of the cost of
the alternative it selects and the cost of the truly optimal alternative.

An additional assumption is imposed in metric distortion [1] – that the voters and candidates all lie in
a shared (unknown) metric space, and costs are given by distances (thus satisfying non-negativity
and triangular inequality). This model is a generalization of a commonly studied spatial model of
voting in the Economics literature [13, 14], and has a natural interpretation of voters liking candidates
with a similar ideological position across many dimensions. While metric distortion is a powerful
framework and has led to the discovery and re-discovery of interesting voting rules (e.g. Plurality
Veto [15] and the study of Maximal Lotteries [16] for metric distortion by Charikar et al. [17]), its
outcomes sometimes do not correspond with traditional wisdom around popular voting rules. For
example, the overly simple Random Dictator (RD) rule (where the winner is the top choice of a
uniform randomly selected voter) beats the Copeland rule (which satisfies the Condorcet Criterion
[10] and other desirable properties) with a metric distortion of 3 versus 5 [1].

While not yet adopted in the metric distortion framework, there is a mature line of work on
Probabilistic voting (PV) [18–20]. Here, the focus is on the behavioural modelling of voters and
accounting for the randomness of their votes. Two sources of this randomness often cited in the
literature are the boundedness of the voters’ rationality and the noise in their estimates of candidates’
positions. A popular model for this behaviour is based on the Quantal Response Theory [20]. Another
closely related line of work is on Random Utility Models (RUMs) [21–23] in social choice where

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.

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

the hypothesis is that the candidates have ground-truth strengths. Voters make noisy observations of
these strengths and vote accordingly. We adopt these models of voting behaviour and study it within
the metric distortion framework. The questions we ask are:

Given a model of probabilistic voting, what is the metric distortion of popular voting rules?
How does this differ (qualitatively and quantitatively) from the deterministic model?

1.1 Preliminaries and Notation

Let N be a set of n voters and A be the set of m candidates. Let S be the set of total orders on A.
Each voter i ∈ N has a preference ranking σi ∈ S. A vote profile is a set of preference rankings
σN = (σ1, ..., σn) ∈ S n for all voters. The tuple (N , A, σN ) defines an instance of an election. Let
∆(A) denote the set of all probability distributions over the set of candidates.
Definition 1 (Voting Rule). A voting rule f : Sn → ∆(A) takes a vote profile σN and outputs a
probability distribution p over the alternatives.

For deterministic voting rules, we overload notation by saying that the rule’s output is a candidate
and not a distribution. We now define some voting rules [10]. Let I denote the indicator function.

Random Dictator Rule: Select a voter uniformly at random and output their top choice, i.e.,
RD(σN ) = p such that pj = 1
n

I(σi,1 = j).

(cid:80)

i∈N

Plurality Rule: Choose the candidate who is the top choice of the most voters, i.e., PLU(σN ) =
arg maxj∈A

I(σi,1 = j). Ties are broken arbitrarily.

(cid:80)

i∈N

i∈N

(cid:80)

I (cid:0)(cid:80)

j′∈A\{j}

(cid:1) . Ties are broken arbitrarily.

Copeland Rule: Choose the candidate who wins the most pairwise comparisons, i.e., COP(σN ) =
I(j ≻σi j′) > n
arg maxj∈A
2
Distance function d : (N ∪ A)2 → R≥0 satisfies triangular inequality (d(x, y) ≤ d(x, z) + d(z, y))
and symmetry (d(x, y) = d(y, x)). The distance between voter i ∈ N and candidate j ∈ A is also
referred to as the cost of j for i. We consider the most commonly studied social cost function, which
is the sum of the costs of all voters. SC(j, d) := (cid:80)
In deterministic voting, the preference ranking σi of voter i is consistent with the distances. That is,
d(i, j) > d(i, j′) =⇒ j′ ≻σi j for all voters i ∈ N and candidates j, j′ ∈ A. Let ρ(σN ) be the set
of distance functions d consistent with vote profile σN . The metric distortion of a voting rule is:

i∈N d(i, j).

Definition 2 (Metric Distortion). DIST(f ) = sup

N ,A,σN

sup
d∈ρ(σN )

E[SC(f (σN ),d)]

min
j∈A

SC(j,d)

.

1.2 Our Contributions

We extend the study of metric distortion to probabilistic voting (Definition 4). This extension is useful
since voters, in practice, have been shown to vote randomly [20]. We define axiomatic properties
of models of probabilistic voting which are suitable for studying metric distortion. These are scale-
freeness with distances (Axiom 1), pairwise order probabilities being independent of other candidates
(Axiom 2), and strict monotonicity of pairwise order probabilities in distances (Axiom 3).

All our results apply to a broad class of models of probabilistic models, as explained in § 2. We
provide distortion bounds for all n ≥ 3 and m ≥ 2, which are most salient in the limit n → ∞. For
large elections (m fixed, n → ∞), we provide matching upper and lower bounds on the distortion of
Plurality, an upper bound for Copeland, and a lower bound for RD. The distortion of plurality grows
linearly in m. The distortion upper bound of Copeland is constant. The distortion lower bound for
RD increases sublinearly in m where this rate depends on the probabilistic model. Crucially, our
results match those in deterministic voting in the limit where the randomness goes to zero.

The technique is as follows. For the problem of maximizing the distortion, we establish a critical
threshold of the expected fraction of votes on pairwise comparisons on all edges on a directed path
from a winner to the “true optimal" candidate for Copeland and Plurality. This path is one or two hops
for Copeland and one for Plurality. We then formulate a linear-fractional program which incorporates
this critical threshold. We linearize this program via the sub-level sets technique [24], and find a
feasible solution of the dual problem. Concentration inequalities on this solution provide an upper
bound on the distortion. We find a matching lower bound for Plurality by construction.

2

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

134

135

136

137

138

139

140

1.3 Related Work

Metric distortion Anshelevich et al. [1] initiated the study of metric distortion and showed that
any deterministic voting rule has a distortion of at least 3 and that Copeland has a distortion of 5.
The Plurality Veto Rule attains the optimal distortion of 3 [15]. Charikar and Ramakrishnan [25]
showed that any randomized voting rule has a distortion of at least 2.112. Charikar et al. [17] gave
a randomized voting rule with a distortion of at most 2.753. Anshelevich et al. [26] gave a useful
survey on distortion in social choice.

Distortion with Additional Information Abramowitz et al. [27] showed that deterministic voting
rules achieve a distortion of 2 when voters provide preference strengths as ratios of distances.
Amanatidis et al. [28] demonstrated that even a few queries from each voter can significantly improve
distortion in non-metric settings. Anshelevich et al. [29] examined threshold approval voting, where
voters approve candidates with utilities above a threshold. Our work relates to these studies since in
probabilistic voting, the likelihood of a voter switching the order of two candidates depends on the
relative strength of their preference, often resulting in lower distortion than deterministic methods.

Probabilisitc voting and random utility models (RUMs) Hinich [30] showed that the celebrated
Median Voter Theorem of [31] does not hold under probabilistic voting. Classical work has focused
on studying the equilibrium positions of voters and/or candidates in game-theoretic models of
probabilistic voting [20, 32–35]. McKelvey and Patty [20] adopt the quantal response model, a
popular way to model agents’ bounded rationality.

RUMs have mostly been studied in social choice [21, 23, 36] with the hypothesis that candidates have
universal ground-truth strengths, which voters make noisy observations of. Our model is the same as
RUM regarding the voters’ behaviour; however, voters have independent costs from candidates. The
Plackett-Luce (PL) model [37, 38] has been widely studied in social choice [39–41]. For probabilities
on pairwise orders, PL reduces to the Bradley-Terry (BT) model [42]. These probabilities are
proportional to candidates’ strengths (which we define as the inverse of powers of costs).

The widely studied Mallows model [43], based on Condorcet [44], flips the order of each candidate
pair (relative to a ground truth ranking) with a constant probability p ∈ (0, 1
2 ) [45, 46]. The process is
repeated if a linear order is not attained. In the context of metric distortion, a limitation of this model
is that it doesn’t account for the relative distance of candidates to the voter. For a comprehensive
review of RUM models, see Marden [47]. Critchlow et al. [48] does an axiomatic study of RUM
models; our axioms are grounded in metric distortion and are distinct from theirs.

Recently, there has been significant interest in smoothed analysis [49] of social choice. Here a small
amount of randomness is added to problem instances and its effect is studied on the satisfiability of
axioms [50–53] and the computational complexity of voting rules [54–56]. Baumeister et al. [50]
term this model as being ‘towards reality,’ highlighting the need to study the randomness in the
election instance generation processes. Unlike smoothed analysis where the voter and candidate
positions are randomized, we consider these positions fixed, but the submitted votes are random given
these positions. The technical difference appears in the benchmark (the “optimal" outcome in the
denominator of the distortion is unchanged in our framework and changes in smoothed analysis).

2 Axioms and Model

Under probabilistic voting, the submitted preferences may no longer be consistent with the underlying
distances. For a distribution P(d) over σN , let qP(d)(i, j, j′) denote the induced marginal probability
that voter i ranks candidate j higher than j′. We focus on these marginal probabilities on pairwise
orders and provide axioms for classifying which qP(d)(·) are suitable for studying distortion.
Axiom 1 (Scale-Freeness (SF)). The probability qP(d)(·) must be invariant to scaling of d. That is,
for any tuple (i, j, j′) and any constant κ > 0, we must have qP(d)(i, j, j′) = qP(κd)(i, j, j′).

Note that the metric distortion (Definition 2) for deterministic voting is scale-free. We want to retain
the same property in the probabilistic model as well. Conceptually, one may think of the voter’s
preferences as being a function of the relative (and not absolute) distances to the candidates.
Axiom 2 (Independence of Other Candidates (IOC)). The probability qP(d)(i, j, j′) must be
independent of the distance of voter i to all ‘other’ candidates, i.e., those in A \ {j, j′}.

3

Table 1: Axioms satisfied by commonly studied models of probabilistic voting

Axiom 1: SF Axiom 2: IOC Axiom 3: Strict Monotonocity

Mallows
PL/BT with exponential in d
PL/BT with powers of d
PQV

✓
×
✓
✓

×
✓
✓
✓

×
✓
✓
✓

This axiom extends Luce’s choice axioms [38], defined for selecting the top choice, to entire rankings.
IOC is reminiscent of the independence of irrelevant alternatives axiom for voting rules.
Axiom 3 (Strict Monotonicity (SM)). For every tuple (i, j, j′), for fixed distance d(i, j) > 0, the
probability qP(d)(i, j, j′) must be strictly increasing in d(i, j′) at all but at most finitely many points.

The monotonicity in d(i, j) follows since qP(d)(i, j′, j) = 1 − qP(d)(i, j, j′). This axiom is natural.

In the Mallows model [43], qP(d)(·) was derived by Busa-Fekete et al. [57] and is as follows:

qP(d)(i, j, j′) = h(rj′ − rj + 1, ϕ) − h(rj′ − rj, ϕ).
Mallows:
(1)
Here h(k, ϕ) = k
(1−ϕk) . Whereas rj and rj′ are the positions of j and j′ in the ground-truth (noiseless)
ranking, and the constant ϕ is a dispersion parameter. Observe that this model fails Axiom 2 since it
depends on the number of candidates between j and j′ in the noiseless ranking. It also fails Axiom 3
since it does not depend on the exact distances but only on the order of the distances.

Plackett-Luce Model: The PL model [37, 38] is ‘sequential’ in the following way. For each voter
i ∈ N , each candidate j ∈ A has a ‘strength’ si,j. In most of the literature on RUMs, a common
assumption is that si,j is the same for all voters i. However, we choose this more general model to
make it useful in the context of metric distortion. The voter chooses their top choice with probability
proportional to the strengths. Similarly, for every subsequent rank, they choose a candidate from
among the remaining ones with probabilities proportional to their strengths. In terms of the pairwise
order probabilities, the PL model reduces to the Bradley-Terry (BT) model [42], that is:

PL/BT:

qP(d)(i, j, j′) =

si,j
si,j + si,j′

(2)

Prima facie, in the metric distortion framework, any decreasing function of distance d(i, j) would
be a natural choice for si,j. However, not all such functions satisfy Axiom 1. The exponential
function is a popular choice in the literature employing BT or PL models. However, in general,
e−d(i,j)
e−d(i,j)+e−d(i,j′ ) ̸=
e−2d(i,j)+e−2d(i,j′) , thus failing the Scale-Freeness Axiom 1.
On the other hand, observe that all functions s = d−θ for any θ ∈ (0, ∞) satisfy our axioms. We use
the regime θ ∈ (1, ∞) for technical simplicity in this work.

e−2d(i,j)

We also define the following class of functions “PQV” for qP(d)(·) motivated by Quantal Response
Theory [58] and its use in probabilistic voting [20]. Observe that PQV satisfies all our axioms.
Definition 3 (Pairwise Quantal Voting (PQV)). Let the relative preference r(i, j, j′) be the ratio of
e−λ/r(i,j,j′ )
distances, d(i,j′)
e−λr(i,j,j′ )+e−λ/r(i,j,j′ ) .

d(i,j) . For constant λ > 0, PQV is as follows: qP(d)(i, j, j′) =

We now define a general class of functions for pairwise order probabilities in terms of the relative
preference (ratio of distances) r. Let G be a class of functions such that any G ∋ g : [0, ∞)∪{∞} →
[0, 1] has the following properties.

1. g is continuous and twice-differentiable.
2. g(0) = 0. Further, g′(r) > 0 ∀r ∈ (0, ∞) i.e. g(r) is strictly increasing in [0, ∞).
3. Define 1
4. There ∃c ∈ [0, ∞) s.t. g′′(r) > 0 ∀r ∈ (0, c) i.e. g is convex in the open interval (0, c).

r as +∞ when r = 0. Then we must have g(r) + g( 1

r ) = 1 ∀r ≥ 0.

Observe that PL (with g(r) = rθ
e−λr+e−λ/r , λ > 0) are in
G. Construction of distributions (if any exists) on rankings σN which generate pairwise order

1+rθ , θ > 1) and PQV (with g(r) =

e−λ/r

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

4

d(i,j) ). That is,
(cid:19)
(cid:18) d(i, B)
d(i, A)

Figure 1: A 1-d Euclidean example of voting probabilities. There are two candidates at 0 and 1. The
figure on the left shows the voter position between 0 and 1. In the right figure, the voter is in positions
to the left of 0. As the distance grows, both candidates look similar to the voter in the probabilistic
model but not in deterministic voting. The case of voter positions to the right of 1 is symmetric.

probabilities qP(d)(i, j, j′) = g( d(i,j′)
for our technical derivations. For PL, these distributions are known from prior work [40].

d(i,j) ) according to PQV is left for future work. We do not need it

We assume g ∈ G in the rest of the paper. Let M(N ∪ A) denote the set of valid distance functions
on (N , A). For any g and d ∈ M(N ∪ A) let ˆP (g)(d) denote the set of probability distributions on
σN for which the marginal pairwise order probabilities are g( d(i,j′)

∀P ∈ ˆP (g)(d), σN ∼ P =⇒ P[A ≻i B] = g

.

(3)

We assume that all voters vote independently of each other. We now define metric distortion under
probabilistic voting as a function of g for a given m and n.
Definition 4 (Metric Distortion under Probabilistic Voting).

DIST

(g)(f, n, m) := sup

N :|N |=n
A:|A|=m

sup
d∈M(N ∪A)

sup
P∈ ˆP (g)(d)

EσN ∼P [SC(f (σN ), d)]

min
A∈A

SC(A, d)

.

(4)

DIST(g)(f ) = supn,m DIST(g)(f, n, m) by supremizing over all possible n and m.
The expectation is both over the randomness in the votes and the voting rule f .

Observe that the distortion is a supremum over all distributions in ˆP (g)(d). Since we focus on large
elections (with large n and relatively small m), we define DIST

(g) as a function of m and n.

(cid:17)

(cid:16) x
1−x

As in Fig. 1, consider the 1-d Euclidean space with candidate X at the origin and Y at 1. Observe
denote the probability that a voter located at a distance x from X votes
that g
for Y when the voter is to the left and right of X respectively. Interestingly, this 1-d intuition extends
well for general metric spaces. Towards this, we define the following functions.

(cid:16) x
1+x

and g

(cid:17)

gMID(x) := g

(cid:18) x

(cid:19)

1 − x

∀x ∈ (0, 1) and gOUT(x) := g

(cid:18) x

(cid:19)

1 + x

∀x ∈ [0, ∞).

(5)

Lemma 1. gMID(x)

x

and gOUT(x)

x

have unique local maxima in (0, 1) and (0, ∞) respectively.

Denote the unique maximisers of gMID(x)
For simplifying notation, in the rest of the work, we use ˆgMID for gMID(x∗

MID and x∗

and gOUT(x)

by x∗

x

x

OUT respectively.

MID)

x∗

MID

and ˆgOUT for gOUT(x∗
x∗

OUT)

OUT

.

In the analysis in the rest of the paper, we will see ˆgMID and ˆgOUT appear many times, so we note these
2+1
2 ≈ 1.21 and
quantities for the PL and PQV models here. For the PL model with θ = 2, ˆgMID =
2−1
ˆgOUT =
2 ≈ 0.21. When θ = 4, ˆgMID ≈ 1.42 and ˆgOUT ≈ 0.06. When θ → ∞, ˆgMID → 2 and
ˆgOUT → 0. This limit is where PL resembles deterministic voting.

√

√

For PQV with λ = 1, ˆgMID ≈ 1.25 and ˆgOUT = 0.18. When λ → ∞, ˆgMID → 2 and ˆgOUT → 0.

5

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

0.00.10.20.30.40.50.60.70.80.91.0Position of voter0.00.10.20.30.40.50.60.70.80.91.0Probability of voting for Candidate at 1PQV: =1PQV: =2PQV: =4PL/BT: =2PL/BT: =4Deterministic0.10.3131030100Absolute value of position of voter0.00.10.20.30.40.50.60.70.80.91.0Probability of voting for Candidate at 1PQV: =1PQV: =2PQV: =4PL/BT: =2PL/BT: =4Deterministic200

201

202

203

204

3 Distortion of Plurality Rule Under Probabilistic Voting

In this section, we give upper and lower bounds on the distortion of the Plurality rule [59] (PLU).In
the limit the number of voters n → ∞ (“large election"), our upper and lower bounds match and are
linear in the number of candidates m. Let B represent the candidate that minimizes the social cost
(referred to as ‘best’), and let {Aj}j∈[m−1] denote the set of other candidates.

205

3.1 Upper bound on the distortion of Plurarity(PLU)

206

Theorem 1. For every ϵ > 0 and m ≥ 2 and n ≥ m2 we have

DIST(g)(PLU, n, m) ≤ m(m − 1) (ˆgMID + ˆgOUT) exp

(6)

(cid:18)

+ max

mˆgMID
(1 − n−( 1

2 −ϵ))

− 1,

(cid:16) −n( 1
(2n( 1
mˆgOUT
(1 − n−( 1

(cid:17)

2 +ϵ) + 2m
2 −ϵ) − 1)m
(cid:19)

+ 1

.

2 −ϵ))

207

Further,

lim
n→∞

DIST(g)(PLU, n, m) ≤ max (mˆgMID − 1, mˆgOUT + 1) .

208

209

210

211

To prove this theorem, we first give a lemma which upper bounds SC(W,d)
SC(B,d) under the constraint
that the expected number of voters that rank candidate W over B is given by α. This ratio will be
useful to bound the contribution of non-optimal candidate W to the distortion of PLU. We state an
optimization problem (7) below, which would be required to bound the ratio as a function of α.

Eα =

min
b,w∈Rn

≥0

s.t.






(cid:80)n
(cid:80)n

n
(cid:88)

i=1 bi
i=1 wi
(cid:18) bi
wi

g

i=1
max
i

|wi − bi| ≤ min

(wi + bi)

i

(cid:19)

≥ α

∀α ≥ 0

(7)

212

Lemma 2. For any two candidates W, B ∈ A which satisfy

n
(cid:80)
i=1

P[W ≻i B] = α, we have

SC(W, d)
SC(B, d)

≤

1
opt(Eα)

≤ max

(cid:16) n
α

ˆgMID − 1,

ˆgOUT + 1

(cid:17)

.

n
α

(8)

213

214

215

Our proof is via Lemmas 3 and 4. Lemma 3 shows that we can bound the ratio of social costs by the
inverse of the optimum value of Eα and Lemma 4 gives a lower bound on the optimum value of Eα.
Lemma 3. For any two candidates W, B ∈ A satisfying (cid:80)n

P[W ≻i B] = α, we have

i=1

SC(W, d)
SC(B, d)

≤

1
opt(Eα)

.

(9)

216

217

Proof. bi and wi in (7) represent the distances d(i, B) and d(i, W ). The last constraint is the triangle
inequality i.e. |d(i, B) − d(i, W )| ≤ d(B, W ) ≤ |d(i, B) + d(i, W )| for every voter i ∈ N .

218

Consider the following linearized version of (7).

Eµ,α =






(cid:33)

bi

− µ

(cid:32) n
(cid:88)

(cid:33)

wi

i=1

(cid:32) n
(cid:88)

i=1

n
(cid:88)

g

min
w,b∈Rn

≥0

s.t.

(cid:18) bi
wi

i=1
|bi − wi| ≤ 1 ∀i ∈ [n]
bi + wi ≥ 1 ∀i ∈ [n]

(cid:19)

≥ α

∀0 ≤ µ ≤ 1, α ≥ 0.

(10)

6

219

Lemma 4. opt(Eα) ≥ min

(cid:16)(cid:0) n

α ˆgMID − 1(cid:1)−1

, (cid:0) n

α ˆgOUT + 1(cid:1)−1(cid:17)

.

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

Our proof uses Lemma 5 and is by solving a linearized version of (7) in (10). This is done by
introducing an extra non-negative parameter µ ≤ 1. Note that it is sufficient to consider µ ≤ 1 since
opt(Eα) ≤ 1 because B minimises the social cost by definition. We find the smallest µ ∈ (0, 1) such
that its objective is non-negative.
Lemma 5. If opt(Eµ,α) ≥ 0, then opt(Eα) ≥ µ.

Further, opt(Eµ,α) ≥ 0 if µ = min

(cid:16)(cid:0) n

α ˆgMID − 1(cid:1)−1

, (cid:0) n

α ˆgOUT + 1(cid:1)−1(cid:17)

.

The first part follows since scaling each term by a constant r satisfies the constraints and also yields the
same objective. And thus we may replace the constraints by maxi |wi−bi| ≤ 1 and mini(wi+bi) ≥ 1
in equation (10). Further, the objective function is linearized as ((cid:80)n
The proof of the second part is technical and has been moved to Appendix B. It involves introducing
a Lagrangian multiplier λ and demonstrating that the objective function is non-negative for a suitably
chosen λ. To establish this, we show that minimising the Lagrangian over the boundaries of the
constraint set given by |bi − wi| = 1 and bi + wi = 1 is sufficient. This requires a careful analysis.

i=1 bi) − µ ((cid:80)n

i=1 wi).

The main technique used in proving Theorem 1 involves considering two cases for every non-optimal
candidate Aj: one where the expected number of voters ranking candidate Aj above B (call it αj)
exceeds a threshold of n
m and one where it does not. In the first case, the ratio of social costs
of Aj and B is bounded using Lemma 2 that naturally gives a bound on contribution of candidate Aj
to the distortion. In the later case, we use Chernoff bound to bound the probability of Aj being the
winner and multiply it with the ratio of social costs of Aj and B to bound the distortion. The proof of
Theorem 1 is in Appendix C.

m − nϵ+1/2

240

3.2 Lower bound on the distortion of Plurality

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

We now present a lower bound on the distortion of PLU for any m in the limit n tends to infinity. This
lower bound matches the upper bound of Theorem 1 in the limit. A full proof is in Appendix D. Note
that the proof has an adversarially chosen distribution over the rankings subject to the marginals on
pairwise relationships satisfying g (as in the definition of distortion under probabilistic voting 4).
This lower bound does not apply to the PL model, which has a specific distribution over rankings.

Theorem 2. For every m ≥ 2,

limn→∞ DIST(g)(PLU, n, m) ≥ max (mˆgMID − 1, mˆgOUT + 1) .

Proof Sketch. The proof is by an example in an Euclidean metric space in R3. One candidate “C" is
at (1, 0, 0). The other m − 1 candidates are “good" and are equidistantly placed on a circle of radius
ϵ on the y − z plane centred at (0, 0, 0). We call them G := {G1, G2, . . . , Gm−1}.

We present sketches of two constructions below for every ϵ, ζ > 0.
(cid:16)

√

(cid:17)

(cid:16)

Construction 1: Let qMID := g
. Each of the m −
1 candidates in G has ⌊aMIDn⌋ voters overlapping with it. The remaining voters (we call them
“ambivalent”) are placed at (x∗
MID, 0, 0). Clearly, each voter overlapping with a candidate votes for it
as the most preferred candidate with probability one. Each of the ambivalent voters votes as follows.

and aMID := 1

1 − 1+ζ
mqMID

m−1

MID

MID)2+ϵ2

(x∗
1−x∗

(cid:17)

– With probability qMID, vote for candidate C as the top choice and uniformly randomly permute the
other candidates in the rest of the vote.

– With probability 1 − qMID, vote for candidate C as the last choice and uniformly randomly permute
the other candidates in the rest of the vote.

We show that the probability that C wins tends to 1 as n → ∞ and the distortion is mˆgMID − 1.

Construction 2: We give a construction where the locations of the candidates are identical as in
Construction 1, and some voters are located with the “good" candidates. The ambivalent voters are at
OUT, 0, 0). We show that P[C wins] tends to 1 as n → ∞ and the distortion is mˆgOUT + 1.
(−x∗

This result establishes that the distortion of Plurality is bound to increase linearly with m even under
probabilistic voting, and is therefore not a good choice when m is even moderately large.

7

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

295

296

297

298

299

300

301

302

4 Distortion of Copeland Rule Under Probabilistic Voting

We now bound the distortion of the Copeland voting rule. We say that candidate W defeats candidate
Y if more than half of the voters rank W above Y .
Theorem 3. For every ϵ > 0, m ≥ 2 and n ≥ 4, we have

DIST(g)(COP, n, m) ≤ 4m(m − 1) exp

(cid:17)

(ˆgMID + ˆgOUT)2

(cid:16) −n( 1
2(2n( 1

2 +ϵ) + 8
2 −ϵ) − 1)
(cid:16)
(cid:17)2

− 1

,

+ max

(cid:16)(cid:16)

2ˆgMID
1 − n−( 1

2 −ϵ)

2ˆgOUT
1 − n−( 1

2 −ϵ)

(cid:17)2(cid:17)

.

+ 1

For every m ≥ 2, we have lim
n→∞

DIST(g)(COP, n, m) ≤ max (cid:0) (2ˆgMID − 1)2 , (2ˆgOUT + 1)2 (cid:1).

Proof Sketch. A Copeland winner belongs to the uncovered set in the tournament graph, as
demonstrated in [1, Theorem 15]. Recall that B denotes the candidate with the least social cost. For
a Copeland winner W , either W defeats B or it defeats a candidate Y who defeats B.

We now consider two exhaustive cases on candidate Aj and define event Ej for every j ∈ [m − 1]
by computing the expected fraction of votes on pairwise comparisons. The event Ej denotes the
existence of an at-most two hop directed path from a candidate Aj to candidate B for Copeland such
that the expected fraction of votes on all edges along that path exceed n

.

2 − n(1/2+ϵ)

2

If Ej holds true, we upper bound the ratio of social cost of candidate Aj and social cost of candidate
B using Lemma 2 which in-turn would give a bound on the distortion. Otherwise, we use union
bound and Chernoff’s bound to upper bound the probability of Aj being the winner. Multiplying the
probability bound with the ratio of social costs (one obtained from Lemma 2) leads to a bound on the
distortion. A detailed proof is in Appendix E.

5 Distortion of Random Dictator Rule Under Probabilistic Voting

We first give an upper bound on the distortion of RD; the proof is in Appendix F.
Theorem 4. DIST(g)(RD, m, n) ≤ (m − 1)ˆgMID + 1.

We now give a lower bound on the distortion of RD. We do this by constructing an example.
Theorem 5. For m ≥ 3 and n ≥ 2, DIST(g)(RD, m, n) ≥ 2 +

1

g−1(

m−1 ) − 2
n .

1

Proof. We have a 1-D Euclidean construction. Let B be at 0 and all other candidates A \ {B} be at
1. m − 1 voters are at 0 and one voter V is at ˜x = g−1( 1

m−1 )/(1 + g−1( 1

m−1 )).

The ranking for V is generated as follows: pick a candidate from A \ {B} as the top rank uniformly
at random. Keep B on the second rank. Permute the remaining candidates uniformly at random for
the remaining ranks. Observe that the marginal pairwise order probabilities are consistent with the
distance of V from B and each candidate in A \ {B}. In particular g( ˜x
m−1 . The distortion for
this instance is P[B wins]·1+P[B loses]· n−˜x
m−1 ) − 2
˜x = 1+ 1
n .

1−˜x ) = 1
˜x − 2
n = 2+

˜x = n−1

n + 1

g−1(

n−˜x

n

1

1

m−1 ) = (m−2)− 1

1+rθ , we have g−1(t) = ( t

θ . Then g−1( 1
θ − 2

For g(r) = rθ
1−t ) 1
bound is DIST(g)(RD, m, n) ≥ 2 + (m − 2) 1

θ , and the distortion lower
n , and limn→∞ DIST(g)(RD, m, n) ≥ 2 + (m − 2) 1
θ .
However, note that this result does not apply to the PL model! This is because the PL model has
a specific distribution on the rankings. In contrast, the above result is obtained by choosing an
adversarial distribution on rankings subject to the constraint that its marginals on pairwise relations
d(i,Aj )−θ
are given by g. In the PL model, P[Aj is top-ranked in σi] =
Ak ∈A d(i,Ak)−θ [45]. We have the
following result for the PL model. A proof via a similar construction as Theorem 5 is in Appendix G.
Theorem 6. Let DISTθ
P L(RD, m, n) denote the distortion when the voters’ rankings are generated
per the PL model with parameter θ. We have limn→∞ DISTθ

P L(RD, m, n) ≥ 1 + (m−1)1/θ

(cid:80)

.

2

8

303

6 Numerical Evaluations

Figure 2: Here, we illustrate how the distortion bounds on different voting rules vary with m and
with the randomness parameters of the two models, PL and PQV, in the limit n → ∞. Both the x and
y axes are on the log scale. We plot the upper bound for Copeland (Theorem 3), the lower bound for
RD (Theorem 5), and the matching bounds for Plurality (Theorem 1).

Recall that higher values of θ and λ correspond to lower randomness. From Figure 2, we observe that
under sufficient randomness, the more intricate voting rule Copeland outshines the simpler rule RD,
which only looks at a voter’s top choice. Moreover, its distortion is independent of m in the limit
n → ∞. This is in sharp contrast to RD, where the distortion is Ω(m1/θ) in the PL model, a sharp
rate of increase in m for low values of θ. The distortion of Plurality increases linearly in m.

An important observation is regarding the asymptotics when θ or λ increases. The distortion of RD
converges to its value under deterministic voting, i.e., 3. The distortion of Plurality also converges to
2m − 1, the same as in deterministic voting. Since our bound on Copeland is not tight, it converges
to 9 rather than 5. So far, in the study of metric distortion, the social choice community has looked
only at these asymptotic; here, we present insights available from looking at the ‘complete’ picture.
Interestingly, the distortion of RD increases with randomness, whereas that of Copeland decreases
up to a certain point and then increases again. The reason for the increases in the high randomness
regime is that the votes become too noisy to reveal the best candidate any more.

Since these plots have no abrupt transitions, this figure hints that smoothened analysis [52] (typically
done with small amounts of noise) is unlikely to give any new insights regarding metric distortion.

7 Discussion and Future Work

We extend the metric distortion framework in social choice in an important way – by capturing the
bounded rationality and randomness in voters’ behaviour. Consideration of this randomness shows
that, in general, the original metric distortion framework is too pessimistic on important voting rules,
most notably on Copeland. On the other hand, the simplistic voting rule Random Dictator, which
attains a distortion of 3 (at least as good as any deterministic rule [1]), is not so good when we look at
the full picture – its distortion increases with the number of candidates in our model. Our framework
opens up opportunities to revisit the metric distortion problem with a closer-to-reality view of voters.
It may hopefully lead to the development of new voting rules that consider the randomness of voters’
behaviour. For example, Liu and Moitra [46] take a learning theory approach to design voting rules
under the assumption of random voting per the Mallows model. However, technical analysis in our
framework may be challenging because of the interplay of the geometric structure of voters’ positions
and the probabilistic nature of their votes.

Future Work An interesting extension would be to other tournament graph-based voting rules
(weighted or unweighted). Our techniques are well-suited for this class of rules since it is based on
the expected weights of the edges of the tournament graph. Closing the gap for the distortion of
Copeland would be useful for getting deeper insights. Another open problem is the characterization
of the set of distributions on rankings that induce the pairwise probabilities per PQV.

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

9

1248163264 in PL model2351123DistortionPlurarity: m=3RD: m=3Plurarity: m=6RD: m=6Plurarity: m=12RD: m=12Copeland0.512481632 in PQV model2351123DistortionPlurarity: m=3RD: m=3Plurarity: m=6RD: m=6Plurarity: m=12RD: m=12Copeland337

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

References

[1] Elliot Anshelevich, Onkar Bhardwaj, and John Postl. Approximating optimal social choice
under metric preferences. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial
Intelligence, pages 777–783, 2015.

[2] Jessica Dai and Eve Fleisig. Mapping social choice theory to RLHF.

arXiv preprint

arXiv:2404.13038, 2024.

[3] Vincent Conitzer, Rachel Freedman, Jobst Heitzig, Wesley H Holliday, Bob M Jacobs, Nathan
Lambert, Milan Mossé, Eric Pacuit, Stuart Russell, Hailey Schoelkopf, et al. Social choice for
AI alignment: Dealing with diverse human feedback. arXiv preprint arXiv:2404.10271, 2024.

[4] Seth D Baum. Social choice ethics in artificial intelligence. AI & Society, 35(1):165–176, 2020.

[5] Jessie Finocchiaro, Roland Maio, Faidra Monachou, Gourab K Patro, Manish Raghavan, Ana-
Andreea Stoica, and Stratis Tsirtsis. Bridging machine learning and mechanism design towards
algorithmic fairness. In Proceedings of the 2021 ACM Conference on Fairness, Accountability,
and Transparency, pages 489–503, 2021.

[6] Francesca Rossi, Kristen Brent Venable, and Toby Walsh. A Short Introduction to Preferences:

Between AI and Social Choice. Morgan & Claypool Publishers, 2011.

[7] Meltem Öztürk, Alexis Tsoukiàs, and Philippe Vincke. Preference modelling. Multiple criteria

decision analysis: State of the art surveys, 78:27–59, 2005.

[8] Kenneth J Arrow. A difficulty in the concept of social welfare. Journal of political economy, 58

(4):328–346, 1950.

[9] Amartya Sen. Social choice theory. Handbook of mathematical economics, 3:1073–1181, 1986.

[10] Kenneth J Arrow, Amartya Sen, and Kotaro Suzumura. Handbook of social choice and welfare,

volume 2. Elsevier, 2010.

[11] Felix Brandt, Vincent Conitzer, Ulle Endriss, Jérôme Lang, and Ariel D Procaccia. Handbook

of computational social choice. Cambridge University Press, 2016.

[12] Ariel D Procaccia and Jeffrey S Rosenschein. The distortion of cardinal preferences in voting.
In International Workshop on Cooperative Information Agents, pages 317–331. Springer, 2006.

[13] James M Enelow and Melvin J Hinich. The spatial theory of voting: An introduction. CUP

Archive, 1984.

[14] Samuel Merrill and Bernard Grofman. A unified theory of voting: Directional and proximity

spatial models. Cambridge University Press, 1999.

[15] Fatih Erdem Kizilkaya and David Kempe. Plurality veto: A simple voting rule achieving
optimal metric distortion. Proceedings of the 31st International Joint Conference on Artificial
Intelligence (IJCAI), pages 349–355, 2022.

[16] Germain Kreweras. Aggregation of preference orderings. In Mathematics and Social Sciences I:
Proceedings of the seminars of Menthon-Saint-Bernard, France (1–27 July 1960) and of Gösing,
Austria (3–27 July 1962), pages 73–79, 1965.

[17] Moses Charikar, Prasanna Ramakrishnan, Kangning Wang, and Hongxun Wu. Breaking the
metric voting distortion barrier. In Proceedings of the 2024 Annual ACM-SIAM Symposium on
Discrete Algorithms (SODA), pages 1621–1640. SIAM, 2024.

[18] Peter J Coughlin. Probabilistic voting theory. Cambridge University Press, 1992.

[19] Kevin M Quinn, Andrew D Martin, and Andrew B Whitford. Voter choice in multi-party
democracies: a test of competing theories and models. American Journal of Political Science,
pages 1231–1247, 1999.

[20] Richard D McKelvey and John W Patty. A theory of voting in large elections. Games and

Economic Behavior, 57(1):155–180, 2006.

10

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

429

[21] Thomas Pfeiffer, Xi Gao, Yiling Chen, Andrew Mao, and David Rand. Adaptive polling for
information aggregation. In Proceedings of the AAAI conference on artificial intelligence,
volume 26, pages 122–128, 2012.

[22] David C Parkes, Houssein Azari Soufiani, and Lirong Xia. Random utility theory for social
choice. In Proceeedings of the 25th Annual Conference on Neural Information Processing
Systems. Curran Associates, Inc., 2012.

[23] Hossein Azari Soufiani, David C Parkes, and Lirong Xia. Preference elicitation for general
random utility models. In Proceedings of the Twenty-Ninth Conference on Uncertainty in
Artificial Intelligence, pages 596–605, 2013.

[24] Stephen P Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press,

2004.

[25] Moses Charikar and Prasanna Ramakrishnan. Metric distortion bounds for randomized social
choice. In Proceedings of the 2022 Annual ACM-SIAM Symposium on Discrete Algorithms
(SODA), pages 2986–3004. SIAM, 2022.

[26] Elliot Anshelevich, Aris Filos-Ratsikas, Nisarg Shah, and Alexandros A Voudouris. Distortion
in social choice problems: The first 15 years and beyond. In 30th International Joint Conference
on Artificial Intelligence, pages 4294–4301, 2021.

[27] Ben Abramowitz, Elliot Anshelevich, and Wennan Zhu. Awareness of voter passion greatly
improves the distortion of metric social choice. In International Conference on Web and Internet
Economics, pages 3–16. Springer, 2019.

[28] Georgios Amanatidis, Georgios Birmpas, Aris Filos-Ratsikas, and Alexandros A Voudouris.
Peeking behind the ordinal curtain: Improving distortion via cardinal queries. Artificial
Intelligence, 296:103488, 2021.

[29] Elliot Anshelevich, Aris Filos-Ratsikas, Christopher Jerrett, and Alexandros A Voudouris.
Improved metric distortion via threshold approvals. In Proceedings of the AAAI Conference on
Artificial Intelligence, volume 38, pages 9460–9468, 2024.

[30] Melvin J Hinich. Equilibrium in spatial voting: The median voter result is an artifact. Journal

of Economic Theory, 16(2):208–219, 1977.

[31] Duncan Black. On the rationale of group decision-making. Journal of political economy, 56(1):

23–34, 1948.

[32] Jeffrey S Banks and John Duggan. Probabilistic voting in the spatial model of elections: The
theory of office-motivated candidates. In Social Choice and Strategic Decisions: Essays in
Honor of Jeffrey S. Banks, pages 15–56. Springer, 2005.

[33] John Wiggs Patty. Local equilibrium equivalence in probabilistic voting models. Games and

Economic Behavior, 51(2):523–536, 2005.

[34] Peter Coughlin and Shmuel Nitzan. Electoral outcomes with probabilistic voting and nash

social welfare maxima. Journal of Public Economics, 15(1):113–121, 1981.

[35] Peter Coughlin and Shmuel Nitzan. Directional and local electoral equilibria with probabilistic

voting. Journal of Economic Theory, 24(2):226–239, 1981.

[36] Lirong Xia. Designing social choice mechanisms using machine learning. In Proceedings of the
international conference on Autonomous agents and multi-agent systems, pages 471–474, 2013.

[37] Robin L Plackett. The analysis of permutations. Journal of the Royal Statistical Society Series

C: Applied Statistics, 24(2):193–202, 1975.

[38] R Duncan Luce. Individual choice behavior: A theoretical analysis. Courier Corporation, 2005.

[39] Isobel Claire Gormley and Thomas Brendan Murphy. Analysis of Irish third-level college
applications data. Journal of the Royal Statistical Society Series A: Statistics in Society, 169(2):
361–379, 2006.

11

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

[40] Hossein Azari, David Parks, and Lirong Xia. Random utility theory for social choice. Advances

in Neural Information Processing Systems, 25, 2012.

[41] Isobel Claire Gormley and Thomas Brendan Murphy. A grade of membership model for rank

data. Bayesian Analysis, 1(1):1–32, 2004.

[42] Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. The

method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

[43] Colin L Mallows. Non-null ranking models. i. Biometrika, 44(1/2):114–130, 1957.

[44] Marquis de Condorcet. Essay on the application of analysis to the probability of majority

decisions. Paris: Imprimerie Royale, page 1785, 1785.

[45] Ioannis Caragiannis, Ariel D Procaccia, and Nisarg Shah. When do noisy votes reveal the truth?

ACM Transactions on Economics and Computation (TEAC), 4(3):1–30, 2016.

[46] Allen Liu and Ankur Moitra. Robust voting rules from algorithmic robust statistics.

In
Proceedings of the Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages
3471–3512. SIAM, 2023.

[47] John I Marden. Analyzing and modeling rank data. CRC Press, 1996.

[48] Douglas E Critchlow, Michael A Fligner, and Joseph S Verducci. Probability models on rankings.

Journal of mathematical psychology, 35(3):294–318, 1991.

[49] Daniel A Spielman and Shang-Hua Teng. Smoothed analysis of algorithms: Why the simplex
algorithm usually takes polynomial time. Journal of the ACM (JACM), 51(3):385–463, 2004.

[50] Dorothea Baumeister, Tobias Hogrebe, and Jörg Rothe. Towards reality: smoothed analysis
In Proceedings of the 19th International Conference on

in computational social choice.
Autonomous Agents and Multiagent Systems, pages 1691–1695, 2020.

[51] Bailey Flanigan, Daniel Halpern, and Alexandros Psomas. Smoothed analysis of social choice
revisited. In International Conference on Web and Internet Economics, pages 290–309. Springer,
2023.

[52] Lirong Xia. The smoothed possibility of social choice. Advances in Neural Information

Processing Systems, 33:11044–11055, 2020.

[53] Lirong Xia. Semi-random impossibilities of condorcet criterion. In Proceedings of the AAAI

Conference on Artificial Intelligence, volume 37, pages 5867–5875, 2023.

[54] Ao Liu and Lirong Xia. The semi-random likelihood of doctrinal paradoxes. In Proceedings of

the AAAI Conference on Artificial Intelligence, volume 36, pages 5124–5132, 2022.

[55] Lirong Xia and Weiqiang Zheng. The smoothed complexity of computing kemeny and slater
rankings. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages
5742–5750, 2021.

[56] Lirong Xia and Weiqiang Zheng. Beyond the worst case: Semi-random complexity analysis
of winner determination. In International Conference on Web and Internet Economics, pages
330–347. Springer, 2022.

[57] Róbert Busa-Fekete, Eyke Hüllermeier, and Balázs Szörényi. Preference-based rank elicitation
using statistical models: The case of mallows. In International conference on machine learning,
pages 1071–1079. PMLR, 2014.

[58] Richard D McKelvey and Thomas R Palfrey. Quantal response equilibria for normal form

games. Games and economic behavior, 10(1):6–38, 1995.

[59] Kenneth J. Arrow. Social Choice and Individual Values. Yale University Press, New Haven, 2

edition, 1963.

[60] John Canny. Chernoff bounds. URL https://people.eecs.berkeley.edu/~jfc/cs174/

lecs/lec10/lec10.pdf.

12

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

A Proof of Lemma 1

Lemma (Restatement of Lemma 1). gMID(x)

x

and gOUT(x)

x

have unique local maxima in (0, 1) and (0, ∞) respectively.

To prove Lemma 1, we first state and prove Lemma 6 which shows that gMID(x) and gOUT(x) change from convex to
concave in intervals (0, 1) and (0, ∞) respectively.
Lemma 6. • There ∃c1 ∈ [0, 1] s.t. gMID(x) is convex in [0, c1] and concave in [c1, 1].

• There ∃c2 ∈ [0, ∞) s.t. gOUT(x) is convex in [0, c2] and concave in [c2, ∞).

Proof. Observe that g′′(x) < 0 for x ≥ 1.

1

2

1

(cid:17)

(cid:17)

(cid:17)

1−x

1−x

thus, g′

MID(x) = g′ (cid:16) x
(1−x)4 . Observe that g′′

(cid:16) x
Recall that gMID(x) = g
1−x
(cid:17)
g′ (cid:16) x
(1−x)3 + g′′ (cid:16) x
must exist a c ∈ (0, 1) such that g′′
MID(c) = 0.
Now we show that there cannot exist two distinct c1, c2 ∈ (0, 1) such that g′′
this statement by contradiction assuming the contrary which implies that g′′
However, since g′ (cid:16) x
> 0 we must have g′′( x
1−x
for r ∈ (0, c) and g′′(r) < 0 for r ∈ (c, ∞).

(cid:17)

(1−x)2 and gMID(x) + gMID(1 − x) = 1 Thus, g′′

MID(x) =

1−x
MID(0) > 0 which implies limx→1 g′′

MID(x) < 0 and thus, there

MID(c1) = 0 and g′′
MID(c2) = 0. We prove
MID(x) must have changed its sign twice.
1−x ) changing its sign twice which is a contradiction since g′′(r) > 0

Now consider gOUT(x) = g
g′′ (cid:16) x

(cid:17)

1

1+x

(1+x)4 . Using a similar approach, we can also prove the second point in the Lemma.

(cid:17)

(cid:16) x
1+x

we have g′

OUT(x) = g′ (cid:16) x

1+x

(cid:17)

1

(1+x)2 . Thus, g′′

OUT(x) = −g′ (cid:16) x

1+x

(cid:17)

2
(1+x)3 +

Using Lemma 6, we now prove Lemma 1 showing the existence of unique maximas of gMID(x)

x

and gOUT(x)

x

.

x−0

x−0

, thus implying g′

MID(t) = gMID(x)−gMID(0)

MID(x) = g′(t) contradicting the fact that g′

Proof of Lemma 1. Recall from Lemma 6 that gMID(x) is convex in [0, c1] and concave in [c1, 1].
Since the first derivative equals zero at every local maxima, we must have xg′
MID(x) − g(x) = 0 for any local maxima
x. We now argue that such a maxima cannot exist in [0, c1]. Suppose such a maxima exists in that case, we must have
MID(x) = gMID(x)−gMID(0)
g′
for some x ∈ (0, c1). Applying LMVT in the interval [0, x]1, we must have some t ∈ (0, x)
s.t. g′
MID(r) is strictly increasing in
[0, c1].
MID(x) = gMID(c1)
Observe that gMID(t) − t gMID(c1)
for
c1
c1
gMID(t)
some x ∈ (0, c1). Since, g′
MID(x) is increasing in [0, c1], we must have g′
t = 1
and gMID(c1)
> 0 since c1g′
MID(c1) > gMID(c1) implying gMID(t)/t is increasing
at t = c1. Thus, gMID(t)/t must have at least one local maxima x∗ in the open interval (c1, ∞) and no local maxima
elsewhere.
We now argue that this local maxima x∗ is unique. Suppose we have two distinct local maximas at x1, x2 ∈ (c1, ∞)
and thus, we have x1g′
MID(x2) − gMID(x2) = 0. Rolle’s theorem would imply that there
exists t ∈ (x1, x2)2 s.t. tg′′

is zero at t = 0 and t = c1 and thus, by Rolle’s theorem, we have g′

MID(t) = 0 which is a contradiction since g′′

MID(x1) − gMID(x1) = 0 and x2g′

> 1. Also, we have d
dt

MID(x) < 0 in (c1, ∞).

MID(c1) > gMID(c1)

. Observe limt→1

(cid:17)(cid:12)
(cid:12)
(cid:12)t=c1

(cid:16) gMID(t)
t

c1

c1

507

Similarly, we can prove the result on the existence and uniqueness of maxima of the function

g( x
x+1 )
x

.

508

509

510

B Proof of Lemma 5

Lemma (Restatement of Lemma 5). If opt(Eµ,α) ≥ 0, then opt(Eα) ≥ µ.
α ˆgOUT + 1(cid:1)−1(cid:17)

Further, opt(Eµ,α) ≥ 0 if µ = min

α ˆgMID − 1(cid:1)−1

(cid:16)(cid:0) n

, (cid:0) n

.

1Observe that g(x)/x has a removable discontinuity at 0 since the limit is defined.
2W.L.O.G, we assume x1 < x2

13

g(ri) −

α
n
|1−ri| .

511

512

Proof. To lower bound the optimal value of Eµ,α, we first pre-multiply the first constraint by λ (and substitute
bi
wi

= ri ∀i ∈ [n]) and thus define,

F (r, b, λ) =

(cid:32) n
(cid:88)

i=1

(cid:33)

bi

− µ

(cid:32) n
(cid:88)

i=1

(cid:33)

bi
ri

− λ

(cid:32) n
(cid:88)

i=1

(cid:33)

g(ri) − α

.

513

Further, we define the set which satisfies the last two constraints in Eµ,α by C as

C := {(r, b) ∈ (Rn

≥0, Rn

≥0) : bi(1 + 1/ri) ≥ 1; |bi(1/ri − 1)| ≤ 1 ∀i ∈ [n]}.

514

From the theory of Lagrangian, we have the following

opt(Eµ,α) ≥ min
(r,b)∈C

max
λ≥0

F (r, b, λ) ≥ max
λ≥0

min
(r,b)∈C

F (r, b, λ).

(11)

(12)

(13)

515

516

Now for a fixed λ > 0, we minimise F (r, b, λ) over (r, b) ∈ C. Observe that for every i ∈ [n], it is sufficient to
minimise h(ri, bi) defined as follows.

h(ri, bi) := bi(1 − µ/ri) − λ

(cid:16)

(cid:17)

.

(14)

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

Observe that the constraints in C can be written as bi ≥ ri
1+ri

and bi ≤ ri

Observe that for a given ri, the function h(ri, bi) is monotonic in bi and thus the optimum point must lie on the boundary
and first optimize over bi(1 + 1/ri) = 1 (call it C MID

) and |bi(1 − 1/ri)| = 1 (call it C OUT

) respectively.

i

i

Recall from Lemma 6 that there exists c1, c2 s.t. gMID(x) is convex in (0, c1) and concave in (c1, 1) and gOUT(x) is
convex in (0, c2) and concave in (c2, ∞).

• Minimisation of h(ri, bi) over bi(1 + 1/ri) = 1.

We first substitute 1/ri = 1/bi − 1 in the function and thus, can write the function h(bi) = bi(µ + 1) − µ −
(cid:1).
λ

= bi(µ + 1) − µ − λ (cid:0)gMID(bi) − α

(cid:16)

(cid:17)

(cid:17)

g

(cid:16) bi
1−bi

− α
n

n

Observe that on optimizing over bi, we obtain two local minima, one at bi = 0 and the other at bi = ˜xMID(λ) ∈ (c1, ∞)
where ˜xMID(λ) satisfies the following equations if λ ≥ 1+µ

MID(c1) . Otherwise, we have a unique minima at bi = 0. 3

g′

g′
MID(˜xMID(λ)) = max

(cid:18) 1 + µ
λ

(cid:19)

, g′

MID(1−)

and g′′

MID(˜xMID(λ)) < 0.

(15)

Observe ˜xMID(λ) > c1 since gMID is concave only in [c1, 1]. Also observe that since g′
increasing, ˜xMID(λ) is monotonically increasing in λ.

MID(x) is monotonically

• Minimisation of h(ri, bi) over bi|(1 − 1/ri)| = 1.

On substituting, we write the function

h(bi) =




(1 − µ)bi − µ − λ



(1 − µ)bi + µ − λ

(cid:16)

(cid:16)

g

g

(cid:17)

(cid:17)

(cid:16) bi
1+bi
(cid:16) bi
bi−1

− α
n
− α
n

(cid:17)

= (1 − µ)bi − µ − λ (cid:0)gOUT(bi) − α
= (1 − µ)bi + µ − λ + λ (cid:0)gOUT(bi − 1) + α

(cid:1)

n

(cid:17) (a)

n

if ri ≥ 1

(cid:1)

otherwise

(16)

(a) follows from the fact that g(r) + g(1/r) = 1.

Since the second function has only a single minima at bi = 1, it is sufficient to consider only the first function in the
case ri ≥ 1.

Observe that on optimizing over bi, we obtain two local minima one at bi = 0 and one at bi = ˜xOUT(λ) ∈ (c2, ∞)
where ˜xOUT(λ) satisfies the equations if λ ≥ 1−µ

OUT(c2) . Otherwise, we have a unique minima at bi = 0. 4
g′

g′
OUT(˜xOUT(λ)) =

(cid:19)

(cid:18) 1 − µ
λ

and g′′

OUT(˜xOUT(λ)) < 0.

(17)

Thus, we have ˜xOUT(λ) > c2 since gOUT is concave only in [c2, ∞). Also observe that since g′
˜xOUT(λ) is monotonic in λ.

OUT(x) is monotonic,

3This follows from the fact that gMID(x) is monotonically decreasing in [c1, 1) and monotonically increasing in [0, c1).
4This follows from the fact that gOUT(x) is monotonically decreasing in [c2, ∞) and monotonically increasing in [0, c2). Since

g′
OUT(∞) = 0, the solution to (17) exists for every λ ∈

(cid:17)
(cid:16) 1−µ
OUT (c2) , ∞
g′

.

14

538

Since, this argument is true for every i ∈ [n], we obtain

F (r, b, λ) = n · min

−µ + λ

min
(r,b)

(cid:32)

,(µ + 1)˜xMID(λ) − µ − λ

(cid:16)

gMID(˜xMID(λ)) −

α
n

(1 − µ)˜xOUT(λ) − µ − λ

(cid:16)

gOUT(˜xOUT(λ)) −

(cid:17)

,

α
n

(cid:17)

(cid:33)
.

α
n

539

Since x∗

MID is the local maximiser of gMID(x)

, we have

x
gMID(x∗

MID) = x∗

MID ˆgMID and x∗

MID > c1.

540

Similarly,

gOUT(x∗

OUT) = x∗

OUT ˆgOUT and x∗

OUT > c2.

541

For the purpose of this analysis, we define two functions δMID(λ) and δOUT(λ) below.

542

543

We also define

δMID(λ) = (µ + 1)˜xMID(λ) − µ − λ

δOUT(λ) = (1 − µ)˜xOUT(λ) − µ − λ

(cid:16)

(cid:16)

gMID(˜xMID(λ)) −

gOUT(˜xOUT(λ)) −

(cid:17)

.

(cid:17)

.

α
n

α
n

µ∗ : = min

(cid:18)(cid:16) n
α

(cid:17)−1

ˆgMID − 1

,

(cid:16) n
α

ˆgOUT + 1

(cid:17)−1(cid:19)

,

and

λ∗ := µ∗ n
α

.

(18)

(19)

(20)

(21)

(22)

(23)

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

Recall that we aim to show opt(Eµ,α) ≥ 0 when µ = µ∗ and thus substitute µ = µ∗ in every subsequent
equation. Observe that it is sufficient to show δMID(λ∗) and δOUT(λ∗) are non-negative since this would imply that
maxλ≥0 min(r,b)∈C F (r, b, λ) is non-negative.
We now consider the following two exhaustive cases.

• Case 1: ˆgMID − α

n > ˆgOUT + α

n . Observe from Equation (15),

MID(˜xMID(λ∗)) = max
g′

µ∗ + 1
(d) follows from the fact that both ˜xMID(λ∗) and x∗

(cid:18) 1

(cid:18) n
α

(cid:19)

(cid:19)

, g′

MID(1−)

= g′

MID(x∗

MID)

(d)
=⇒ ˜xMID(λ∗) = x∗

MID.

(24)

MID exceed c1 and g′

MID(x) is monotonically decreasing for x ≥ c1.

δMID(λ∗) = (µ∗ + 1)˜xMID(λ∗) − µ∗ − λ∗ (cid:16)

gMID(˜xMID(λ∗)) −

(cid:17)

α
n

(b)
≥ (−µ∗ + λ∗α/n) + λ∗ (˜xMID(λ∗)g′

MID(˜xMID(λ∗)) − gMID(˜xMID(λ∗)))

(c)
≥ 0.

(25)

MID(˜xMID(λ∗)) = 1+µ∗

λ∗

(b) follows from g′
(c) follows from ˜xMID(λ∗) = x∗
µ∗ = λ∗ α

n . Now consider,

5 as stated in Equation (15).

MID (in Equation (24)) and gMID(x∗

MID) = x∗

MID ˆgMID (in Equation (19)) and the fact that

ˆgOUT

(d)
=

OUT)

gOUT(x∗
x∗

OUT

(e)
≤

MID)

gMID(x∗
x∗

MID

− 2α/n

(g)
=

1 − µ
λ∗

(h)
=⇒ g′

OUT(x∗

OUT) ≤ g′

OUT(˜xOUT(λ∗))

(i)
=⇒ x∗

OUT ≥ ˜xOUT(λ∗).

(d) follows from the fact that x∗
(e) follows from the fact that ˆgMID − α
n in Case 1.
(g) follows from the definition of λ∗ and that µ = λ∗ α
n .

n > ˆgOUT + α

OUT is the local maximiser of gOUT(x)/x,

(h) follows from the constraint in (17).

5This follows from the fact that 1+µ∗

λ∗ = ˆgMID = g′(x∗

MID) ≥ g′

MID(1−)

15

557

558

559

560

561

562

563

564

(i) follows from the fact that g′

OUT(x) is monotonically decreasing in x in [c2, ∞).
(cid:17)
α
n
OUT(˜xOUT(λ∗)) − gOUT(˜xOUT(λ∗)))

δOUT(λ∗) = (1 − µ)˜xOUT(λ∗) − µ − λ∗ (cid:16)
−µ + λ∗ α
n

gOUT(˜xOUT(λ∗)) −

+ λ∗(˜xOUT(λ∗)g′

(j)
=

(cid:17)

(cid:16)

(k)
≥ 0 + 0 ≥ 0.

(26)

(j) follows from g′

OUT(˜xOUT(λ)) = 1−µ
(k) follows from the following reasons:

λ as stated in Equation (17), and

– Observe that xg′

OUT(x) − gOUT(x) is monotonically decreasing in [c2, ∞) as gOUT is concave in this region. However,

since x∗

OUT ≥ ˜xOUT(λ∗) ≥ c2, we have

(˜xOUT(λ∗)g′

OUT(˜xOUT(λ∗)) − gOUT(˜xOUT(λ∗))) ≥ x∗

OUT ˆgOUT − gOUT(x∗

OUT) = 0

– λ∗ = µ∗ n

α follows from the definition of λ∗.

Thus, using (25) and (26) we show that for the chosen value of λ∗ = µ∗ n
min(r,w)∈C F (r, w, λ∗) ≥ 0 implying from (13) that opt(Eµ,α) ≥ 0.

α , we have

• Case 2: ˆgMID − α

n ≤ ˆgOUT + α

n

Choosing λ∗ = µ∗ n

α , we can prove opt(Eµ,α) ≥ 0 in a very similar manner whenever µ = µ∗.

565

C Proof of Theorem 1

566

Theorem (Restatement of Theorem 1). For every ϵ > 0 and m ≥ 2 and n ≥ m2 we have

DIST(g)(PLU, n, m) ≤ m(m − 1) (ˆgMID + ˆgOUT) exp

(27)

567

Further,

lim
n→∞

DIST(g)(PLU, n, m) ≤ max (mˆgMID − 1, mˆgOUT + 1) .

(cid:18)

+ max

mˆgMID
(1 − n−( 1

2 −ϵ))

− 1,

(cid:16) −n( 1
(2n( 1
mˆgOUT
(1 − n−( 1

(cid:17)

2 +ϵ) + 2m
2 −ϵ) − 1)m
(cid:19)

+ 1

.

2 −ϵ))

568

Proof. Recall that candidate B ∈ A minimises the social cost. The other candidates are denoted by {Aj}j∈[m−1].

DIST(g)(PLU, n, m) =

sup
d∈M(N ∪A)





m−1
(cid:88)

j=1

P[Aj wins]

SC(Aj, d)
SC(B, d)

+ P[B wins]





(28)

569

570

571

572

573

For every j ∈ [m − 1], we now bound the probability of Aj being the winner. This event implies that at least n
m voters
choose Aj as the top preference, implying that the same voters rank Aj over B. Further, we now define Bernoulli
random variables {Yi,j}n
i=1 each denoting the event that voter i ranks candidate Aj over B. Recall from Equation 3,
(cid:17)(cid:17)
(cid:16)
g
Yi,j ∼ Bern

. Therefore,

(cid:16) d(i,B)
d(i,Aj )

P[Aj wins] ≤ P

(cid:32) n
(cid:88)

i=1

Yi,j ≥

(cid:33)

.

n
m

(29)

Let αj be the expectation of the random variable (cid:80)n

i=1 Yi,j i.e. the expected number of voters ranking Aj over B.

αj :=

n
(cid:88)

i=1

E[Yi,j] =

n
(cid:88)

i=1

(cid:19)

g

(cid:18) d(i, B)
d(i, Aj)

16

for every j ∈ [m − 1].

(30)

574

575

576

577

578

579

580

581

582

Now we use Chernoff bounds on the sum of Bernoulli random variable for every j ∈ [m − 1] when αj ≤ n
to bound the probability of Aj being the winner.

m − n(1/2+ϵ)

m

If αj ≤

n
m

−

P[Aj wins] ≤ P

n(1/2+ϵ)
m
(cid:32) n
(cid:88)

Yi,j ≥

i=1

we have,

(cid:33)

n
m

= P

(cid:32) n
(cid:88)

i=1

(cid:18)

Yi,j ≥ αj

1 +

(cid:19)(cid:33)

− 1

n
mαj

−1)

(cid:33)αj

(cid:32)

(a)
≤

( n

mαj

e
( n
mαj

)n/mαj

(cid:17) n

m

n
m −αj

e

=

≤

(cid:16) mαj
n
mαj
n

(c)
≤

mαj
n

(cid:18) mαj
n


n
m

e



(cid:16)

(cid:18)

exp

−

αj
n/m − 1

(cid:19)(cid:19)( n

m −1)

n
m

e

1 − n−( 1

2 −ϵ)(cid:17)



exp

−

+ϵ)

n

m − n( 1
2
m
n/m − 1

=

(cid:16)

mαj
n

1 − n−( 1

2 −ϵ)(cid:17)(n/m−1)

exp

(cid:33)

(cid:32)

2 +ϵ)

n( 1
m

(cid:32)

(d)
≤

mαj
n

exp

(cid:32)

=

mαj
n

exp

−2n−( 1

2 − n−( 1

2 −ϵ)(n/m − 1)
2 −ϵ)
(cid:33)

−n( 1
(2n( 1

2 +ϵ) + 2m
2 −ϵ) − 1)m

.

(cid:33)

+

2 +ϵ)

n( 1
m





n
m −1





(31)

(32)

(33)

(34)

(35)

(36)

(37)

(38)

(a) follows from applying the Chernoff bound. We restate the bound from [60] below.
Suppose X1, X2, . . . , Xn be independent Bernoulli random variables with P(Xi) = µi for every i ∈ [n] and µ :=
(cid:80)n

i=1 µi, then we have

(cid:88)

P(

i

Xi ≥ (1 + δ)µ) ≤

(cid:18)

eδ
(1 + δ)1+δ

(cid:19)µ

(39)

(c) holds since xe−x is increasing in (0, 1) and because

α

n/m−1 ≤ 1 and α ≤ n

m − n( 1

2

+ϵ)

m , the maxima is attained at

α = n

m − n( 1

2

+ϵ)

m . (d) holds since log(1 + x) ≤ 2x

2+x for −1 < x ≤ 0.

Let S := {j ∈ [m − 1] : αj < n

m } i.e. S denotes the indices of candidates with αj less than n

m − n(1/2+ϵ)
m .

Now using Lemma 2 and αj ≥ n

for every j ∈ [m − 1] \ S, we have

m − n(1/2+ϵ)
m − n(1/2+ϵ)

m

SC(Aj, d)
SC(B, d)

(cid:18)

≤ max

mˆgMID
(1 − n−(1/2−ϵ))

− 1,

mˆgOUT
(1 − n−(1/2−ϵ))

(cid:19)

+ 1

(40)

583

We now have

DIST(g)(PLU, n, m)

=

sup
d∈M(N ∪A)

(cid:32)

(cid:88)

j∈[m−1]\S

(cid:18)

P[Aj wins]

(cid:19)

SC(Aj, d)
SC(B, d)

+ P[B wins] +

(cid:18)

(cid:88)

j∈S

P[Aj wins]

SC(Aj, d)
SC(B, d)

(cid:19)(cid:33)

(cid:18)

(a)
≤ max

max
j∈[m−1]\S

SC(Aj, d)
SC(B, d)

(cid:19)

, 1

+

(cid:32)

(cid:88)

j∈S

max

(b)
≤ m(m − 1) (ˆgMID + ˆgOUT) exp

(cid:32)

−n( 1
(2n( 1

2 +ϵ) + 2m
2 −ϵ) − 1)m

(cid:18) n
αj
(cid:33)

ˆgMID − 1,

n
αj

ˆgOUT + 1

(cid:19) mαj
n

(cid:32)

exp

−n( 1
(2n( 1

2 +ϵ) + 2m
2 −ϵ) − 1)m

(cid:33)(cid:33)

(cid:18)

mˆgMID
(1 − n−(1/2−ϵ))

− 1,

mˆgOUT
(1 − n−(1/2−ϵ))

(cid:19)

+ 1

.

+ max

17

584

(a) follows from the following observations.

585

• Apply Lemma 2 to bound SC(Aj ,d)

SC(B,d) . Since αj ≤ n

∀j ∈ S, apply Equation (31) to bound P[Aj wins].

586

• (cid:80)

j∈[m−1]\S

(cid:16)

P[Aj wins] SC(Aj ,d)
SC(B,d)

(cid:17)

+ P[B wins] ≤ max

max
j∈[m−1]\S

SC(Aj ,d)
SC(B,d) , 1

(cid:19)

.

m − n(1/2+ϵ)
(cid:18)

m

587

(b) follows from the fact that |S| ≤ m − 1 , max(a, b) ≤ a + b, and applying Equation (40).

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

D Proof of Theorem 2

Theorem (Restatement of Theorem 2). For every m ≥ 2,
max (mˆgMID − 1, mˆgOUT + 1) .

limn→∞ DIST(g)(PLU, n, m) ≥

Proof. The proof is by an example in an Euclidean metric space in R3. One candidate “C" is at (1, 0, 0). The other
m − 1 candidates are “good" and are equidistantly placed on a circle of radius ϵ on the y − z plane centred at (0, 0, 0).
We call them G := {G1, G2, . . . , Gm−1}.

We present two constructions below for every ϵ, ζ > 0.

Construction 1: Let qMID := g
. Each of the m − 1 candidates in G has
⌊aMIDn⌋ voters overlapping with it. The remaining voters (we call them “ambivalent”) are placed at (x∗
MID, 0, 0). Clearly,
each voter overlapping with a candidate votes for it as the most preferred candidate with probability one. Each of the
ambivalent voters votes as follows.

m−1

and aMID := 1

1 − 1+ζ
mqMID

MID

(cid:16)

(cid:17)

√

(cid:16)

(cid:17)

MID)2+ϵ2

(x∗
1−x∗

– With probability qMID, vote for candidate C as the top choice and uniformly randomly permute the other candidates in
the rest of the vote.

– With probability 1 − qMID, vote for candidate C as the last choice and uniformly randomly permute the other candidates
in the rest of the vote.

Observe that this satisfies the pairwise probability criterion in Equation 3. Since limn→∞⌊an⌋/n = a and that the
distance of a candidate in G from any non-ambivalent voter is at most 2ϵ, we have that for every j ∈ [m − 1],

lim
n→∞

SC(C, d)
SC(Gj, d)

≥

√

MID)(1 − (m − 1)aMID) + (m − 1)aMID

(1 − x∗
(1 − (m − 1)aMID)(cid:112)(x∗
√
(mqMID − (1 + ζ))
(1 + ζ)(cid:112)(x∗

1 + ϵ2
MID)2 + ϵ2 + 2(m − 2)aMIDϵ
1 + ϵ2 + (1 + ζ)(1 − x∗
MID)
MID)2 + ϵ2 + 2(m − 2)aMIDϵ

=

.

(41)

(42)

605

Clearly every candidate in G minimises the social cost and now we show that

lim
n→∞

P[C wins] = 1.

Let Bernoulli random variables {Yi}n
(cid:80)n

P[Yi = 1] = qMID(n − (m − 1)⌊aMIDn⌋) and thus

i=1

i=1 denote the events that voter i ∈ N ranks candidate C at the top. Here,

n
(cid:80)
i=1

P[Yi = 1]

=

1 + ζ
m

.

lim
n→∞
By the law of large numbers, we have that P[(cid:80)
to win, the event (cid:80)
i Yi ≥ n

n
i Yi ≥ n
m ] = 1 as n → ∞. Since every candidate in G is equally likely

m implies the event that C is the winner and thus, limn→∞ P[C wins] = 1. Thus,
√

lim
n→∞

DIST(g)(PLU, n, m) ≥

(mqMID − (1 + ζ))
(1 + ζ)(cid:112)(x∗

1 + ϵ2 + (1 + ζ)(1 − x∗
MID)2 + ϵ2 + 2(m − 2)aMIDϵ

MID)

.

(43)

Construction 2: Let qOUT := g

(cid:18) √

OUT)2+ϵ2

(x∗
1+x∗

OUT

(cid:19)

and aOUT := 1

m−1

(cid:16)

overlapping with it, and the remaining “ambivalent" voters are at (−x∗

(cid:17)

1 − 1+ζ
mqOUT
OUT, 0, 0).

. Each candidate in G has ⌊aOUTn⌋ voters

Clearly, each voter overlapping with a candidate votes for it as the most preferred candidate with probability one. Each
of the ambivalent voters votes as follows.

18

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

• With probability qOUT, vote for candidate C as the top choice and uniformly randomly permute the other candidates in

the rest of the vote.

• With probability 1−qOUT, vote for candidate C as the last choice and uniformly randomly permute the other candidates

in the rest of the vote.

616

This satisfies the pairwise probability criterion in Equation 3. For every j ∈ [m − 1],

lim
n→∞

SC(C, d)
SC(Gj, d)

≥

OUT)(1 − (m − 1)aOUT) + (m − 1)aOUT

(1 + x∗
(1 − (m − 1)aOUT)(cid:112)(x∗

1 + ϵ2
OUT)2 + ϵ2 + 2(m − 2)aOUTϵ

√

√

=

(1 + ζ)(1 + x∗

OUT) + (mqOUT − (1 + ζ))

1 + ϵ2

(1 + ζ)(cid:112)(x∗

OUT)2 + ϵ2 + 2(m − 2)aOUTϵ

.

(44)

(45)

617

618

619

620

621

Clearly, every candidate in G minimises the social cost. Now, we show that

lim
n→∞
i=1 denote the events that voter i ∈ N ranks candidate C at the top. We have
P[Yi=1]
= 1+ζ
m . Applying the law of large numbers,
n
m ] = 1 as n tends to ∞. However since every candidate in G is equally likely to win, the event

(cid:80)n

i=1

P[C wins] = 1.

P[Yi = 1] = qMID(n − (m − 1)⌊an⌋) and thus, limn→∞

i=1

LEt Bernoulli random variables {Yi}n
(cid:80)n
we get that P[(cid:80)
i
(cid:80)
i Yi ≥ n

Yi ≥ n

m corresponds to the event that C is the winner and thus, limn→∞ P[C wins] = 1. Therefore we have,

lim
n→∞

DIST(g)(PLU, n, m) ≥

√

(mqOUT − (1 + ζ))
(1 + ζ)(cid:112)(x∗

1 + ϵ2 + (1 + ζ)(1 + x∗
OUT)2 + ϵ2 + 2(m − 2)aOUTϵ

OUT)

.

(46)

622

623

On applying the limit ϵ, ζ → 0 and substituting for qMID and qOUT, we get the desired lower bound by combining the
results from the two constructions.

624

E Proof of Theorem 3

625

Theorem 7. Restatement of Theorem 3 For every ϵ > 0, m ≥ 2 and n ≥ 4, we have

DIST(g)(COP, n, m) ≤ 4m(m − 1) exp

(cid:16) −n( 1
2(2n( 1

2 +ϵ) + 8
2 −ϵ) − 1)
(cid:16)
(cid:17)2

− 1

,

(cid:17)

(ˆgMID + ˆgOUT)2

2ˆgOUT
1 − n−( 1

(cid:17)2(cid:17)

.

+ 1

+ max

(cid:16)(cid:16)

2ˆgMID
1 − n−( 1

2 −ϵ)
DIST(g)(COP, n, m) ≤ max (cid:0) (2ˆgMID − 1)2 , (2ˆgOUT + 1)2 (cid:1).

2 −ϵ)

626

For every m ≥ 2, we have lim
n→∞

627

Proof. Recall that B ∈ A minimises the social cost, and {Aj}j∈[m−1] denotes the set A \ B.

DIST(g)(COP, n, m) =

sup
d∈M(N ∪A)





m−1
(cid:88)

j=1

P[Aj wins]

SC(Aj, d)
SC(B, d)



+ P[B wins]



(47)

628

629

630

631

632

633

634

635

Consider a Copeland winner W . As noted by prior work [1], W must be in the uncovered set of the tournament graph,
and one of the following two cases must be true.

• W defeats B.

• There exists a candidate Y ∈ A s.t. W defeats Y and Y defeats B.

For every j ∈ [m − 1], we now bound the probability of Aj being the winner. For every j ∈ [m − 1], we define
Bernoulli random variables {Yi,j}n
i=1 denoting the event that voter i ranks candidate Aj over candidate B. From
. For every distinct j, k ∈ [m − 1], we define Bernoulli random
Equation 3, we have that Yi,j ∼ Bern

(cid:17)(cid:17)

(cid:16)

g

(cid:16) d(i,Aj )
d(i,B)

variables {Zi,j,k}n

i=1 denoting the event that voter i ranks candidate Aj over Ak. Zi,j,k ∼ Bern(g

19

(cid:16) d(i,Ak)
d(i,Aj )

(cid:17)

).

636

Observe that

P[Aj wins] ≤ P





n
(cid:88)

i=1

Yi,j ≥

n
2

(cid:91)

(cid:32) n
(cid:88)

Zi,j,k ≥

k∈[m−1]\{j}

i=1

n
2

∩

n
(cid:88)

i=1

Yi,k ≥

(cid:33)
 .

n
2

(48)

637

638

639

640

Let αj denote the expected value of the random variable (cid:80)n
candidate Aj over B.

n
(cid:88)

αj :=

E[Yi,j] =

n
(cid:88)

g

(cid:18) d(i, B)
d(i, Aj)

i=1

i=1

i=1 Yi,j, i.e., the expected number of voters who rank

(cid:19)

for every j ∈ [m − 1].

(49)

Let βj,k denote the expected value of the random variable (cid:80)n
candidate Aj over Ak.

i=1 Zi,j,k, i.e., the expected number of voters who rank

βj,k :=

n
(cid:88)

i=1

E[Zi,j,k] =

n
(cid:88)

i=1

(cid:19)

g

(cid:18) d(i, Ak)
d(i, Aj)

for every j ∈ [m − 1].

(50)

641

Similar to Equation (31), we have the following bound:

If αj ≤

n
2

−

, we have

n(1/2+ϵ)
2
(cid:32) n
(cid:88)

P

i=1

Yi,j ≥

(cid:33)

n
2

= P

(cid:32) n
(cid:88)

i=1

(cid:18)

Yi,j ≥ αj

1 +

(cid:19)(cid:33)

− 1

n
2αj

( n
2αj

−1)

(cid:33)αj

(a)
≤

≤

(cid:32)

e
( n
2αj
(cid:18) 2αj
n

(c)
≤

(cid:18) 2αj
n

)n/2αj
(cid:19)2 (cid:18) mαj
n


(cid:19)2

(cid:16)

n
2

e



(cid:18)

exp

−

αj
n/2 − 2

(cid:19)(cid:19)( n

2 −2)

n
m

e

1 − n−( 1

2 −ϵ)(cid:17)



exp

−

+ϵ)

n

2 − n( 1
2
2
n/2 − 2





n
2 −2





(cid:19)2 (cid:16)

=

(cid:18) 2αj
n

1 − n−( 1

2 −ϵ)(cid:17)(n/2−2)

exp

(cid:33)

(cid:32)

2 +ϵ)

n( 1
2

(cid:32)

(cid:19)2

=

(cid:18) 2αj
n

exp

642

From Equation (31) in the proof of Theorem 1, we have

(cid:19)2

(d)
≤

(cid:18) 2αj
n

exp

(cid:32)

(cid:33)

+

2 +ϵ)

n( 1
2

−2n−( 1

2 − n−( 1

2 −ϵ)(n/2 − 2)
2 −ϵ)
(cid:33)

−n( 1
(2n( 1

2 +ϵ) + 8
2 −ϵ) − 1)2

643

644

645

646

647

(cid:32) n
(cid:88)

P

i=1

Yi,j ≥

(cid:33)

n
2

≤

(cid:18) 2αj
n

(cid:19)2

(cid:32)

exp

−n( 1
2(2n( 1

2 +ϵ) + 8
2 −ϵ) − 1)

(cid:33)

if αj ≤

n
2

−

n(1/2+ϵ)
2

.

Similarly, P

(cid:32) n
(cid:88)

i=1

Zi,j,k ≥

(cid:33)

n
2

≤

(cid:18) 2βj,k
n

(cid:19)2

(cid:32)

exp

−n( 1
2(2n( 1

2 +ϵ) + 8
2 −ϵ) − 1)

(cid:33)

if βj,k ≤

n
2

−

n(1/2+ϵ)
2

.

Consider two exhaustive cases on candidate Aj and define an event Ej for every j ∈ [m − 1]. We compute the expected
fraction of votes on pairwise comparisons. The event Ej denotes the existence of an at-most two hop directed path
from a candidate Aj to candidate B for Copeland such that the expected fraction of votes on all edges along that path
exceed n

. Recall that we only considered one hop path for the case of PLU in the proof of Theorem 1.

2 − n(1/2+ϵ)

2

20

(51)

(52)

(53)

(54)

(55)

(56)

(57)

(58)

(59)

(cid:18)

Ej :=

αj ≥

n
2

−

n(1/2+ϵ)
2

(cid:19) (cid:91)

(cid:18)(cid:18)

k∈[m−1]\{j}

βj,k ≥

n
2

−

n(1/2+ϵ)
2

(cid:19) (cid:92) (cid:18)

αk ≥

(cid:19)(cid:19)

n
2

−

n(1/2+ϵ)
2

.

(60)

648

649

650

651

652

653

654

If Ej holds true, we can directly upper bound the ratio of the social cost of candidate Aj to the social cost of candidate
B using Lemma 2, which in turn provides a bound on the distortion. If Ej does not hold, we apply the union bound and
Chernoff’s bound to upper bound the probability of Aj being the winner. By multiplying this probability bound with
the ratio of social costs obtained from Lemma 2, we derive a bound on the distortion.

Define S := {j ∈ [m − 1] : Ej is not true}. Furthermore, we define K1(j) := {j ∈ [m − 1] : αk ≥ βj,k} and
K2(j) := {j ∈ [m − 1] : αk < βj,k} denotes complement of K1(j) for every j ∈ [m].

From Equations (58) and (59), both of the following conditions 1 and 2 are satisfied for every j ∈ S.

655

1. P (cid:0)(cid:80)n

i=1 Yi,j ≥ n
2

(cid:1) ≤

(cid:17)2

(cid:16) 2αj
n

exp

(cid:18)

656

2. For every k ∈ [m − 1] \ {j},

2

−n( 1
2(2n( 1

2

(cid:19)

+ϵ)+8
−ϵ)−1)

657

658

P (cid:0)(cid:80)n

i=1 Zi,j,k ≥ n
2

(cid:1) ≤

(cid:17)2

(cid:16) 2βj,k
n

exp

and, P (cid:0)(cid:80)n

i=1 Yi,k ≥ n
2

(cid:1) ≤ (cid:0) 2αk

n

(cid:1)2

exp

(cid:18)

2

−n( 1
2(2n( 1
(cid:18)

2

(cid:19)

+ϵ)+8
−ϵ)−1)

if k ∈ K1(j)

2

−n( 1
2(2n( 1

2

(cid:19)

+ϵ)+8
−ϵ)−1)

if k ∈ K2(j).

659

Furthermore, we define γj := max

max
k∈[m−1]\{j}

(min(αk, βj,k)) , αj

.

(cid:18)

(cid:19)

660

661

Since, for every Copeland winner W , it must either defeat B or there exists a Y ∈ A s.t. W defeats Y and Y defeats B.
Using union bound for every j ∈ S, we have

P[Aj wins] ≤ P

(cid:34) n
(cid:88)

i=1

Yi,j ≥

(cid:35)

n
2

+

(cid:88)

(cid:34)(cid:32) n
(cid:88)

P

Yi,k ≥

(cid:32) n
(cid:88)

∩

Zi,j,k ≥

(cid:33)(cid:35)

n
2

if j ∈ S

(cid:32)

(cid:19)2

(cid:18) 2αj
n

≤

exp

k∈[m−1]\{j}
(cid:33)

−n( 1
2(2n( 1

2 +ϵ) + 8
2 −ϵ) − 1)

i=1

(cid:88)

+

k∈K2(j)

(cid:18) 2αk
n

exp

i=1
(cid:32)

−n( 1
2(2n( 1
(cid:32)

(cid:33)

2 +ϵ) + 8
2 −ϵ) − 1)

(cid:88)

+

k∈K1(j)

(cid:19)2

(cid:18) 2βj,k
n

exp

−n( 1
2(2n( 1

2 +ϵ) + 8
2 −ϵ) − 1)

(cid:33)

n
2

(cid:19)2

≤m

(cid:19)2

(cid:18) 2γj
n

exp

(cid:32)

−n( 1
2(2n( 1

2 +ϵ) + 8
2 −ϵ) − 1)

(cid:33)

if j ∈ S.

662

663

The last inequality follows from the definition of γj.
Furthermore from Lemma 2 and the definition of γj, 6 we have

SC(Aj, d)
SC(B, d)

(cid:18)

≤

max

(cid:18) n
γj

ˆgMID − 1,

(cid:19)(cid:19)2

n
γj

ˆgOUT + 1

664

Using Equation (62) and (61) and applying max(a, b) ≤ a + b, we have

(cid:33)

if j ∈ S

(61)

(62)

P[Aj wins]

SC(Aj, d)
SC(B, d)

≤ 4m exp

(cid:32)

(cid:33)

−n( 1
2(2n( 1

2 +ϵ) + 8
2 −ϵ) − 1)

(ˆgMID + ˆgOUT)2 if j ∈ S.

(63)

6This follows on splitting SC(Aj ,d)

SC(B,d) = SC(Aj ,d)
SC(Ak,d) × SC(Ak,d)
(cid:17)
, 1
αj

1
βj,k

(cid:19)

)

,

max( 1
αk

1
γ = min

(cid:18)

min
k∈[m−1]\{j}

(cid:16)

SC(B,d) and applying the lemma separately. We further use the fact that

21

665

Recall that for every j ∈ [m − 1] \ S, Ej is satisfied. Let us further denote

ˆEj := αj ≥

n
2

−

n(1/2+ϵ)
2

and ˆDj,k :=

(cid:18)

βj,k ≥

n
2

−

n(1/2+ϵ)
2

(cid:19)

.

666

667

668

Observe that Ej being satisfied implies either a) ˆEj is satisfied or b) ∃k ∈ [m − 1] \ {j} s.t ˆEk and ˆDj,k are satisfied.
We consider both cases separately.
Suppose ˆEj is satisfied for some j ∈ [m − 1] \ S. Then we have from Lemma 2,

SC(Aj, d)
SC(B, d)

(cid:18)

≤ max

2ˆgMID
(1 − n−(1/2−ϵ))

− 1,

2ˆgOUT
(1 − n−(1/2−ϵ))

(cid:19)

+ 1

.

(64)

669

Now we consider case (b) where ˆEk and ˆDj,k are both satisfied for some k ∈ [m − 1] \ {j}. From Lemma 2 we have,

SC(Aj, d)
SC(B, d)

≤ max

(cid:32)(cid:18)

2ˆgMID
(1 − n−(1/2−ϵ))

(cid:19)2

(cid:18)

− 1

,

2ˆgOUT
(1 − n−(1/2−ϵ))

(cid:19)2(cid:33)

+ 1

.

(65)

670

Now combining Equations (63), (64), and (65), we have for any metric space d ∈ M(N ∪ A),

DIST(g)(COP, n, m) ≤

(cid:32)

(cid:88)

j∈S

(cid:18)

P[Aj wins]

(cid:19)

SC(Aj, d)
SC(B, d)

+ P[B wins] +

(cid:88)

(cid:18)

P[Aj wins]

(cid:19)(cid:33)

SC(Aj, d)
SC(B, d)

(cid:33)

(ˆgMID + ˆgOUT) + max

j∈[m−1]\S

(cid:18)

max
j∈[m−1]\S

SC(Aj, d)
SC(B, d)

(cid:19)

, 1

(a)
≤ 4(m − 1)m exp

(b)
≤ 4(m − 1)m exp

(cid:32)

−n( 1
2(2n( 1
(cid:16) −n( 1
2(2n( 1

2 +ϵ) + 8
2 −ϵ) − 1)

2 +ϵ) + 8
2 −ϵ) − 1)

(cid:17)

(ˆgMID + ˆgOUT) + max

(cid:16)(cid:16)

2ˆgMID
(1 − n−(1/2−ϵ))

(cid:17)2

(cid:16)

,

− 1

2ˆgOUT
(1 − n−(1/2−ϵ))

(cid:17)2(cid:17)

.

+ 1

671

(a) follows from Equation (61) and the fact that (cid:80)

(cid:16)

j∈S

P[Aj wins] SC(Aj ,d)
SC(B,d)

(cid:17)

+P[B wins] ≤ max

(cid:18)

max
j∈S

(cid:19)

SC(Aj ,d)
SC(B,d) , 1

.

672

(b) follows from combining Equations (63), (64), and (65).

673

F Proof of Theorem 4

674

Theorem (Restatement of Theorem 4). DIST(g)(RD, m, n) ≤ (m − 1)ˆgMID + 1.

Proof. The probability of voter i voting for candidate W as its top candidate is upper bounded by g
the probability that W is ranked over B. Therefore, under RD, the probability of W winning satisfies:

(cid:16) d(i,B)
d(i,W )

675

676

P[W wins] ≤

1
n

(cid:32) n
(cid:88)

i=1

(cid:18) d(i, B)
d(i, W )

g

(cid:19)(cid:33)

.

(cid:17)

which is

(66)

677

678

Recall that we define the set of candidates in A \ B as {A1, A2, . . . , Am−1}. In the rest of the analysis we denote
d(i, Aj) by yi,j (for all j ∈ [m − 1]) and d(i, B) by bi for every i ∈ [n]. We also denote d(B, Aj) by zj for every

22

(67)

(68)

(69)

(70)

(71)

(72)

(73)

(74)

679

j ∈ [m − 1]. Now for every metric d, we bound the distortion as follows.

DIST(g)(RD, m, n) ≤

m−1
(cid:88)

(cid:18)

P[Aj wins]

(cid:19)

(cid:80)n
(cid:80)n

i=1 yi,j
i=1 bi

+ (1 −

m−1
(cid:88)

j=1

P[Aj wins])

j=1

m−1
(cid:88)

j=1

=

m−1
(cid:88)

(a)
≤

j=1

m−1
(cid:88)

j=1

1
n

≤

m−1
(cid:88)

(d)
≤

j=1

P[Aj wins]

(cid:18) (cid:80)n
(cid:80)n

i=1 yi,j
i=1 bi

(cid:19)

− 1

+ 1

1
n

(cid:32) n
(cid:88)

i=1

g

(cid:18) bi
yi,j

(cid:19)(cid:33) (cid:80)n

i=1(yi,j − bi)
(cid:80)n
i=1 bi

+ 1

(cid:32) n
(cid:88)

i=1
(cid:16)(cid:80)n

g

(cid:18) bi/zj
yi,j/zj
(cid:16) bi/zj
yi,j /zj
i=1 bi/zj
(cid:17)

i=1 g
(cid:80)n

(cid:16) x∗
MID
1−x∗
x∗

MID

MID

(cid:19)(cid:33) (cid:80)n

i=1(yi,j/zj − bi/zj)
i=1 bi/zj

(cid:80)n

+ 1

(cid:17)(cid:17)

+ 1

+ 1 = (m − 1)ˆgMID + 1.

(e)
≤ (m − 1)

g

(a) follows from Equation (66).

680

681

682

683

When bi
zj

≤ 1 and thus, yi,j
zj

from triangle inequality. Similarly, we have yi,j
zj

− 1 when bi
zj

≥ 1. Thus,

(d) follows from the fact that yi,j − bi ≤ zj which follows from triangle inequality.
(e) follows from the following arguments by considering two cases namely bi
zj

≥ 1.

≤ 1 and bi
zj
≥ bi
zj

≥ 1 − bi
zj
(cid:17)

g

(cid:16) bi/zj
yi,j /zj
bi/zj

≤ max

(cid:17)

(cid:80)n

i=1 g
(cid:80)n

(cid:16) bi/zj
yi,j /zj
i=1 bi/zj

=⇒

(cid:33)

g( x
x−1 )
x

,

sup
x∈(1,∞)
(cid:17)


for every i ∈ [n]

(cid:32)

sup
x∈(0,1)


g( x
1−x )
x
(cid:16) x∗
MID
1−x∗
x∗

MID

g

≤ max



MID

, 1

 .

684

The last inequality follows from the fact that

g( x

x−1 )
x ≤ 1 when x ≥ 1. Further, we have ˆgMID ≥ 1 for all valid g.

685

686

687

688

689

690

691

692

693

G Proof of Theorem 6

Theorem (Restatement of Theorem 6). Let DISTθ
generated per the PL model with parameter θ. We have limn→∞ DISTθ

P L(RD, m, n) denote the distortion when the voters’ rankings are
P L(RD, m, n) ≥ 1 + (m−1)1/θ

.

2

Proof. We have a 1-D Euclidean construction. Let B be at 0 and all other candidates A \ {B} be at 1. m − 1 voters are
at 0, and one voter is at t. We will set t later by optimizing for the distortion.

(m−1)(1−t)−θ
t−θ+(m−1)(1−t)−θ

The distortion for this instance is P[B wins] · 1 + P[B loses] · n−t
t
1
. We drop the terms which are O(1/n) to obtain 1 +
n
1 + (m−1)tθ−1
bound of 1 + (m−1)1/θ

(1−t)θ+(m−1)tθ . This is lower bounded by 1 + (m−1)tθ−1

t−θ+(m−1)(1−t)−θ +
t(t−θ+(m−1)(1−t)−θ) . This simplifies to
1+(m−1)tθ . Setting t = (m − 1)−1/θ, we obtain a distortion lower

= n−1
n + 1
(m−1)(1−t)−θ

n−t
t

n

.

t−θ

2

23

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

NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions
and scope?
Answer: [Yes]
Justification: We took care to make sure the claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope.
Guidelines:

• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in
the paper and important assumptions and limitations. A No or NA answer to this question will not be
perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can

be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained

by the paper.

2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: We have added a future work section which lays the open questions and limitations.
Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that the paper has

limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these
assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic
approximations only holding locally). The authors should reflect on how these assumptions might be
violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few
datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which
should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a
facial recognition algorithm may perform poorly when image resolution is low or images are taken in
low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online
lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale

with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of

privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers
as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t
acknowledged in the paper. The authors should use their best judgment and recognize that individual
actions in favor of transparency play an important role in developing norms that preserve the integrity of
the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and
correct) proof?
Answer: [Yes]
Justification: All assumptions are mentioned clearly. All the proofs are provided, and we took care to make
them correct to the best of our understanding.
Guidelines:

24

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

• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the
supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs

provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results
of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether
the code and data are provided or not)?
Answer: [NA]
Justification: [NA]
Guidelines:

• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers:
Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their

results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the
contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution
is a specific model and empirical evaluation, it may be necessary to either make it possible for others
to replicate the model with the same dataset, or provide access to the model. In general. releasing code
and data is often one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language
model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide
some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For
example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that

algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture

clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way
to access this model for reproducing the results or a way to reproduce the model (e.g., with an
open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to
describe the particular way they provide for reproducibility. In the case of closed-source models, it
may be that access to the model is limited in some way (e.g., to registered users), but it should be
possible for other researchers to have some path to reproducing or verifying the results.

5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully
reproduce the main experimental results, as described in supplemental material?
Answer: [NA]
Justification: [NA]
Guidelines:

• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/guides/

CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so “No”
is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to
the contribution (e.g., for a new open-source benchmark).

25

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

• The instructions should contain the exact command and environment needed to run to reproduce the
results. See the NeurIPS code and data submission guidelines (https://nips.cc/public/guides/
CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw

data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method
and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted
from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is

recommended, but including URLs to data and code is permitted.

6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they
were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: [NA]

Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary

to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information
about the statistical significance of the experiments?

Answer: [NA]

Justification: [NA]

Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or

statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test
split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library

function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a
2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not
verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric

error bars that would yield results that are out of range (e.g. negative error rates).

• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated

and reference the corresponding figures or tables in the text.

8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type
of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: [NA]

Guidelines:

• The answer NA means that the paper does not include experiments.

26

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

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider,

including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as

well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments

reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of
Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]

Justification: [NA]

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the

Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws

or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the
work performed?

Answer: [Yes]

Justification: We have discussed the positive social impact of the design of voting rules and ways in which our
paper can be instrumental towards it.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the

paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation,
generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could
make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular
applications, let alone deployments. However, if there is a direct path to any negative applications,
the authors should point it out. For example, it is legitimate to point out that an improvement in the
quality of generative models could be used to generate deepfakes for disinformation. On the other hand,
it is not needed to point out that a generic algorithm for optimizing neural networks could enable people
to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended
and functioning correctly, harms that could arise when the technology is being used as intended but gives
incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g.,
gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse,
mechanisms to monitor how a system learns from feedback over time, improving the efficiency and
accessibility of ML).

11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data
or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped
datasets)?

Answer: [NA]

Justification: [NA]

Guidelines:

• The answer NA means that the paper poses no such risks.

27

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

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards
to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or
restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe

how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but

we encourage authors to take this into account and make a best faith effort.

12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly
credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [NA]
Justification: [NA]
Guidelines:

• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source

should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be
provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets.
Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if

it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided
alongside the assets?
Answer: [NA]
Justification: [NA]
Guidelines:

• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via

structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an

anonymized URL or include an anonymized zip file.

14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full
text of instructions given to participants and screenshots, if applicable, as well as details about compensation
(if any)?
Answer: [NA]
Justification: [NA]
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper

involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor

should be paid at least the minimum wage in the country of the data collector.

15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

28

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

Question: Does the paper describe potential risks incurred by study participants, whether such risks
were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent
approval/review based on the requirements of your country or institution) were obtained?
Answer:[NA]
Justification: [NA]
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required
for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and

we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such

as the institution conducting the review.

29

