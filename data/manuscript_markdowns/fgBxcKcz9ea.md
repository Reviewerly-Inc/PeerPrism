Differential Privacy of Dirichlet Posterior Sampling

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

We study the inherent privacy of releasing a single sample from a Dirichlet posterior
distribution. As a complement to the previous study that provides general theories
on the differential privacy of posterior sampling from exponential families, this
study focuses speciﬁcally on the Dirichlet posterior sampling and its privacy
guarantees. With the notion of truncated concentrated differential privacy (tCDP),
we are able to derive a simple privacy guarantee of the Dirichlet posterior sampling,
which effectively allows us to analyze its utility in various settings. Speciﬁcally,
we provide accuracy guarantees of the Dirichlet posterior sampling in Multinomial-
Dirichlet sampling and private normalized histogram publishing.

1

Introduction

The Bayesian framework provides a way to perform statistical analysis by combining prior beliefs
with real-life evidence. At a high level, the belief and the evidence are assumed to be described
by probabilistic models. As we receive new data, our belief is updated accordingly via the Bayes’
theorem, resulting in the so-called posterior belief. The posterior tells us how much we are uncertain
about the model’s parameters.

The Dirichlet distribution is usually chosen as the prior when performing Bayesian analysis on discrete
variables, as it is a conjugate prior to the categorical and multinomial distributions. Speciﬁcally,
Dirichlet distributions are often used in discrete mixture models, where a Dirichlet prior is put on
the mixture weights [LW92; MMR05]. Such models have applications in NLP [PB98], biophysical
systems [Hin15], accident analysis [de 06], and genetics [BHW00; PM01; CWS03]. In all of these
studies, samplings from Dirichlet posteriors arise when performing Markov chain Monte Carlo
methods for approximate Bayesian inference.

Dirichlet posterior sampling also appears in other learning tasks. For example, in Bayesian active
learning, it arises in Gibbs sampling, which is used to approximate the posterior of the classiﬁer over
the labeled sample [NLYCC13]. In Thompson sampling for multi-armed bandits, one repeatedly
draws a sample from the Dirichlet posterior of each arm, and picks the arm whose sample maximizes
the reward [ZHGSY20; AAFK20; NIK20]. And in Bayesian reinforcement learning, state-transition
probabilities are sampled from the Dirichlet posterior over past observed states [Str00; ORR13].

Dirichlet posterior sampling can also be used for data synthesis. Suppose that we have a histogram
(x1, . . . , xd) of actual data. An approximate discrete distribution of this histogram can be obtained by
drawing a sample Y from Dirichlet(x1 + α1, . . . , xd + αd), where α1, . . . , αd are prior parameters.
Then synthetic data is produced by repeatedly drawing from Multinomial(Y). There are many
studies on data synthesis that followed this approach [AV08; MKAGV08; RWZ14; PG14; SJGLY17].

Submitted to 35th Conference on Neural Information Processing Systems (NeurIPS 2021). Do not distribute.

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

In the above examples, the data that we integrate into these tasks might contain sensitive information.
Thus it is important to ask: how much of the information is protected from the Dirichlet samplings?
The goal of this study is to ﬁnd an answer to this question.

The mathematical framework of differential privacy (DP) [DMNS06] allows us to quantify how much
the privacy of the Dirichlet posterior sampling is affected by the prior parameters α1, . . . , αd. In the
deﬁnition of DP, the privacy of a randomized algorithm is measured by how much its distribution
changes upon perturbing a single data point of the input. Nonetheless, this notion might be too
strict for the Dirichlet distribution, as a small perturbation of a near-zero parameter can cause a large
distribution shift. Thus, it might be more appropriate to rely on one of several relaxed notions of
DP, such as approximate differential privacy, Rényi differential privacy, or concentrated differential
privacy. It is natural to wonder if the Dirichlet posterior sampling satisﬁes any of these deﬁnitions.

1.1 Overview of Our results

This study focuses on the privacy and utility of Dirichlet posterior sampling. In summary, we provide
a closed-form privacy guarantee of the Dirichlet posterior sampling, which in turn allows us to
effectively analyze its utility in various settings.

§3 Privacy. We study the role of the prior parameters in the privacy of the Dirichlet posterior
sampling. Theorem 1 is our main result, where we provide a guaranteed upper bound for truncated
concentrated differential privacy (tCDP) of the Dirichlet posterior sampling. In addition, we convert
the tCDP guarantee into an approximate differential privacy guarantee in Corollary 2.
§4 Utility. Using the tCDP guarantee, we investigate the utility of Dirichlet posterior sampling
applied in two speciﬁc applications:

• In Section 4.1, we consider one-time sampling from a Multinomial-Dirichlet distribution.
But instead of directly sampling from this distribution, we sample from another distribution
with larger prior parameters. The accuracy is then measured by the KL-divergence between
the original and the private distributions.

• In Section 4.2, we use the Dirichlet posterior sampling for a private release of a normalized
histogram. In this case, the accuracy is measured by the mean-squared error between the
sample and the original normalized histogram.

In both tasks, we compute the sample size that guarantees the desired level of accuracy. In the case
of private histogram publishing, we also compare the Dirichlet posterior sampling to the Gaussian
mechanism.

1.2 Related work

There are several studies on the differential privacy of posterior sampling. Wang, Fienberg, and
Smola [WFS15] showed that any posterior sampling with the log-likelihood bounded by B is 4B-
differentially private. However, the likelihoods that we study are not bounded away from zero; they
have the form (cid:81)
i which becomes small when one of the pi’s is close to zero. Dimitrakakis, Nelson,
Zhang, Mitrokotsa, and Rubinstein [DNZMR17] showed that if the condition on the log-likelihood is
relaxed to the Lipschitz continuity with high probability, then one can obtain the approximate DP.
Nonetheless, with the Dirichlet density, it is difﬁcult to compute the probability of events in which
the Lipschitz condition is satisﬁed.

i pxi

In the case that the sufﬁcient statistics x has ﬁnite (cid:96)1-sensitivity, Foulds, Geumlek, Welling and
Chaudhuri [FGWC16] suggested adding Laplace noises to x. Suppose that y is the output; they
showed that sampling from p(θ|y) is differentially private and as asymptotically efﬁcient as sampling
from p(θ|x). However, for a small sample size, the posterior over the noisy statistics might be too
far away from the actual posterior. Bernstein and Sheldon [BS18] thus proposed to approximate the
joint distribution p(θ, x, y) using Gibbs sampling, which is then integrated over x to obtain a more
accurate posterior over y.

2

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

Geumlek, Song, and Chaudhuri [GSC17] were the ﬁrst to study the posterior sampling with the
RDP. Even though they provided a general framework to ﬁnd (λ, (cid:15))-RDP guarantees for exponential
families, explicit forms of (cid:15) and the upper bound of λ were not given. In contrast, our tCDP guarantees
of the Dirichlet posterior sampling imply an explicit expression for (cid:15), and also an upper bound for λ.

The privacy of data synthesis via sampling from Multinomial(Y), where Y is a discrete distri-
bution drawn from the Dirichlet posterior, was ﬁrst studied by Machanavajjhala, Kifer, Abowd,
Gehrke, and Vilhuber [MKAGV08]. They showed that the data synthesis is (ε,δ)-probabilistic DP,
which implies (ε,δ)-approximate DP. However, as their privacy analysis includes the sampling from
Multinomial(Y), their privacy guarantee depends on the number of synthetic samples. In contrast,
we show that the one-time sampling from the Dirichlet posterior is approximate DP, which by the
post-processing property allows us to sample from Multinomial(Y) as many times as we want while
retaining the same privacy guarantee.

The Dirichlet mechanism was ﬁrst introduced by Gohari, Wu, Hawkins, Hale, and Topcu [GWHHT21].
Originally, the Dirichlet mechanism takes a discrete distribution p := (p1, . . . , pd) and draws one
sample Y ∼ Dirichlet(rp1, . . . , rpd). Note the absence of the prior parameters, which makes Y an
unbiased estimator of p. But this comes with a cost, as the worst case of privacy violation occurs
when almost all of the parameters are close to zero. The authors avoided this issue by restricting
the input space to a subset of the unit simplex, with some of the pi’s bounded below by a ﬁxed
positive constant. This results in complicated expressions for the privacy guarantees as they involve
a minimization problem over the restricted domain. In this study, we take a different approach by
adding prior parameters to the Dirichlet mechanism. As a result, we obtain a biased algorithm that
requires no assumption on the input space and has simpler forms of privacy guarantees.

103

1.3 Notations

104

105

106

107

108

109

110

111

112

≥0 be the set of d-tuples of non-negative real numbers and Rd

We let Rd
>0 be the set of d-tuples of
positive real numbers. We assume that all vectors are d-dimensional where d ≥ 2. The notations for
all vectors are always in bold. Speciﬁcally, x := (x1, . . . , xd) ∈ Rd
≥0 consists of sample statistics of
the data and α := (α1, . . . , αd) ∈ Rd
>0 consists of the prior parameters. The vector p := (p1, . . . , pd)
always satisﬁes (cid:80)
i xi
and α0 := (cid:80)
1, . . . , xd + x(cid:48)
i αi. For any vectors x, x(cid:48) and scalar r > 0, we write x + x(cid:48) := (x1 + x(cid:48)
d)
and rx := (rx1, . . . , rxd). For any positive reals x and x(cid:48), the notation x ∝ x(cid:48) means x = Cx(cid:48) for
some constant C > 0, x ≈ x(cid:48) means cx(cid:48) ≤ x ≤ Cx(cid:48) for some c, C > 0, and x (cid:46) x(cid:48) means x ≤ Cx(cid:48)
for some C > 0. Lastly, (cid:107)x(cid:107)∞ := maxi|xi| is the (cid:96)∞ norm of x.

i pi = 1. The number of observations is always N . We also denote x0 := (cid:80)

113

2 Background

114

2.1 Privacy models

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

Deﬁnition 2.1 (Pure and Approximate DP [DMNS06]). A randomized mechanism M : X n → Y
is (ε, δ)-differentially private ((ε, δ)-DP) if for any datasets x, x(cid:48) differing on a single entry, and all
events E ⊂ Y,

P[M (x) ∈ E] ≤ eεP[M (x(cid:48)) ∈ E] + δ.
If M is (ε, 0)-DP, then we say that it is ε-differential privacy (ε-DP).

The term pure differential privacy (pure DP) refers to (cid:15)-differential privacy, while approximate
differential privacy (approximate DP) refers to (ε, δ)-DP when δ > 0.

In contrast to pure and approximate DP, the next deﬁnitions of differential privacy are deﬁned in
terms of the Rényi divergence between M (x) and M (x(cid:48)):
Deﬁnition 2.2 (Rényi Divergence [Rén61]). Let P and Q be probability distributions. For λ ∈ (1, ∞)
the Rényi divergence of order λ between P and Q is deﬁned as
1
λ − 1

(cid:20) P (y)λ−1
Q(y)λ−1

P (y)λQ(y)1−λ dy =

Dλ(P (cid:107)Q) :=

1
λ − 1

(cid:19)
(cid:21)
.

E
y∼P

log

log

(cid:18)

(cid:90)

3

125

126

127

Deﬁnition 2.3 (tCDP and zCDP [BDRS18; BS16]). A randomized mechanism M : X n → Y is
ω-truncated ρ-concentrated differentially private ((ρ, ω)-tCDP) if for any datasets x, x(cid:48) differing on a
single entry and for all λ ∈ (1, ω),

Dλ(M (x)(cid:107)M (x(cid:48))) ≤ λρ.

128

If M is (ρ, ∞)-tCDP, then we say that it is ρ-zero-concentrated differential privacy (ρ-zCDP).

129

130

131

132

133

134

135

Note that both tCDP and zCDP have the composition and post-processing properties. Intuitively, ρ con-
trols the expectation and standard deviation of the privacy loss random variable: Z = log P [M (x)=Y ]
P [M (x(cid:48))=Y ] ,
where Y has density M (x), and ω controls the number of standard deviations for which Z concen-
trates like a Gaussian. A smaller ρ and larger ω correspond to a stronger privacy guarantee. It turns
out that tCDP implies approximate DP:
Lemma 1 (From tCDP to Approximate DP [BDRS18]). Let δ > 0. If M is a (ρ, ω)-tCDP mechanism,
then it also satisﬁes (ε, δ)-DP with

ε =

(cid:40)

ρ + 2(cid:112)ρ log(1/δ)
ρω + log(1/δ)

ω−1

if log(1/δ) ≤ (ω − 1)2ρ
if log(1/δ) > (ω − 1)2ρ

.

136

2.2 Dirichlet distribution

137

138

139

For α ∈ Rd
>0, the Dirichlet distribution Dirichlet(α) is a continuous distribution of d-dimensional
probability vectors i.e. vectors whose coordinate sum is equal to 1. The density function of Y ∼
Dirichlet(α) is given by:

p(y) =

1
B(α)

d
(cid:89)

i=1

yαi−1
i

,

140

where B(α) is the beta function, which can be written in terms of the gamma function:

B(α) =

(cid:81)
Γ((cid:80)

i Γ(αi)
i αi)

.

(1)

141

2.3 Dirichlet posterior sampling

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

We consider the prior Dirichlet(α) and the likelihood of the form p(x|y) ∝ (cid:81)d
x ∈ Rd
sampling:

i where
≥0 consists of sample statistics of the dataset. The Dirichlet posterior sampling is a one-time

i=1 yxi

Y ∼ Dirichlet(x + α).
There is a modiﬁcation of the sampling which introduces a concentration parameter r > 0, and
instead we sample from Dirichlet(rx + α) [GSC17; GWHHT21]. Smaller values of r make the
sampling more private, and larger values of r make Y a closer approximation of x. Even though the
case r = 1 is the main focus of this study, our main privacy results can be easily extended to other
values of r as we will see at the end of Section 3.1.

Consider a special case where x = p is an empirical distribution derived from the dataset, and we
want Y to be a private approximation of p; the sampling Y ∼ Dirichlet(rp + α) is called the
Dirichlet mechanism [GWHHT21]. It is interesting to note that the Dirichlet mechanism is a form of
the exponential mechanism [MT07]: let r > 0 be the privacy parameter, Dirichlet(α) be the prior,
and the negative KL-divergence be the score function of the exponential mechanism. Then the output
Y of this mechanism is distributed according to the following density function:




exp(−r DKL(p, y)) (cid:81)
(cid:82) exp(−r DKL(p, y)) (cid:81)

i

i yαi−1
i yαi−1

i

dy

∝ exp

r

(cid:88)

pi log(yi/pi)



(cid:89)

∝

i,pi(cid:54)=0
(cid:89)

yrpi
i

i,pi(cid:54)=0

i

yαi−1
i

=

(cid:89)

i

yrpi+αi−1
i

,

yαi−1
i

(cid:89)

i

156

which is exactly the density function of Dirichlet(rp + α).

4

157

2.4 Polygamma functions

158

159

160

161

In most of this study, we take advantage of several nice properties of the log-gamma function and its
derivatives. Speciﬁcally, ψ(x) := d
dx log Γ(x) is concave and increasing, while its derivative ψ(cid:48)(x) is
positive, convex, and decreasing. In addition, ψ(cid:48) can be approximated by the reciprocals:

which implies that ψ(cid:48)(x) ≈ 1

+

1
x

1
2x2 < ψ(cid:48)(x) <
x2 as x → 0 and ψ(cid:48)(x) ≈ 1

1
1
x2 ,
x
x as x → ∞.

+

162

3 Main privacy results

163

3.1 Truncated concentrated differential privacy

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

Theorem 1. Let α ∈ Rd
>0 and αm := mini αi. Let γ ∈ (0, αm). Let ∆2, ∆∞ > 0 be constants that
satisfy (cid:80)
i| ≤ ∆∞ whenever x, x(cid:48) ∈ R2
≥0 are sample statistics
of any two datasets differing on a single entry. The one-time sampling from Dirichlet(x + α) is
(ρ, ω)-tCDP, where ω = γ
∆∞

2 and maxi |xi − x(cid:48)

i(xi − x(cid:48)

i)2 ≤ ∆2

+ 1 and

ρ =

1
2

∆2

2ψ(cid:48)(αm − γ).

(2)

Note that (ρ, ∞)-tCDP is not obtainable, as the ratio between two Dirichlet densities blows up as
ω → ∞. We present here a short proof that skips some calculations (see Appendix 1 for a full proof).

proof. Consider any λ ∈
of Dirichlet(u) and P (cid:48)(y) be the density of Dirichlet(u(cid:48)). A quick calculation shows that:

. Let u := x + α and u(cid:48) := x(cid:48) + α(cid:48). Let P (y) be the density

+ 1

1, γ
∆∞

(cid:16)

(cid:17)

(cid:21)

(cid:20) P (y)λ−1
P (cid:48)(y)λ−1

Ey∼P (y)

B(u(cid:48))λ−1
B(u)λ−1 ·
We take the logarithm on both sides and apply the second-order Taylor expansion to the following
G(ui, u(cid:48)
i) terms that appear on the right-hand side. As a result, there exist ξ between
ui + (λ − 1)(ui − u(cid:48)
i such that

B(u + (λ − 1)(u − u(cid:48)))
B(u)

i) and ui, and ξ(cid:48) between ui and u(cid:48)

i) and H(ui, u(cid:48)

(3)

=

.

G(ui, u(cid:48)

i) := (λ − 1)(log Γ(u(cid:48)

i) − log Γ(ui))

H(ui, u(cid:48)

(λ − 1)(xi − x(cid:48)

= −(λ − 1)(xi − x(cid:48)

1
i)ψ(ui) +
2
i) := log Γ(ui + (λ − 1)(ui − u(cid:48)
i)) − log Γ(ui)
1
2

i)2ψ(cid:48)(ξ),
i, then ξ and ξ(cid:48) are bounded below by u(cid:48)
i, then ξ and ξ(cid:48) are bounded below by ui − (λ − 1)|ui − u(cid:48)

= (λ − 1)(xi − x(cid:48)

(λ − 1)2(xi − x(cid:48)

i)ψ(ui) +

i)2ψ(cid:48)(ξ(cid:48))

Note that ψ(cid:48) is increasing. If xi > x(cid:48)
other hand, if xi ≤ x(cid:48)
λ < γ
∆∞

+ 1 guarantees that ui − (λ − 1)|ui − u(cid:48)

i| > αm − γ. All cases considered, we have

i ≥ αm. On the
i|. The condition

(4)

(5)

G(ui, u(cid:48)

i) + H(ui, u(cid:48)

i) ≤

=

(cid:0)(λ − 1) + (λ − 1)2(cid:1)(xi − x(cid:48)

1
2
1
2
i, the same argument shows that G(u0, u(cid:48)

λ(λ − 1)(xi − x(cid:48)

i)2ψ(cid:48)(αm − γ).

i)2ψ(cid:48)(αm − γ)

0) + H(u0, u(cid:48)

0) > 0.

178

179

Denoting u0 := (cid:80)
Therefore,

i ui and u(cid:48)

0 := (cid:80)

i u(cid:48)

Dλ(P (y)(cid:107)P (cid:48)(y)) =

<

≤

1
λ − 1

1
λ − 1

(cid:32)

(cid:88)

(G(ui, u(cid:48)

i) + H(ui, u(cid:48)

i)) − G(u0, u(cid:48)

0) − H(u0, u(cid:48)
0)

(cid:33)

i
(cid:88)

(G(ui, u(cid:48)

i) + H(ui, u(cid:48)

i))

i

1
2

λ

(cid:88)

(xi − x(cid:48)

i)2ψ(cid:48)(αm − γ) ≤

i

5

1
2

λ∆2

2ψ(cid:48)(αm − γ).

Figure 1: Left: the actual values of ρ = 1
2 D2(P (cid:107)P (cid:48)) and the worst case (ρ, 2)-tCDP guarantees (2)
at ∆2
2 = ∆∞ = 1. Here, P and P (cid:48) are Dirichlet posterior densities over x = (11, 8, 65, 25, 38, 0),
x(cid:48) = (11, 8, 65, 25, 38, 1), and α = (α, . . . , α). Right: comparison between (ε, δ)-DP guarantees of
the Dirichlet posterior samplings (8) with different uniform priors: α = (α, . . . , α).

The guaranteed upper bound (2) is independent of the sample statistics. As a result, the bound applies
even in worst settings i.e., when xi = 0 and x(cid:48)
i = ∆∞, or vice versa, for some i. As we can see in
Figure 1, the upper bound is a close approximation to the actual value of ρ when x6 = 0 and x(cid:48)
6 = 1.
However, being a sample independent bound, the difference becomes substantial when all xi’s are
large. There is one way to get around this issue: if there is no privacy violation in assuming that
the sample statistics are always bounded below by some threshold τ , then we can incorporate the
threshold into the prior (thus ψ(cid:48)(αm − γ) in (2) is replaced by ψ(cid:48)(αm + τ − γ)).

The parameter γ allows us to adjust the moment bound ω as desired. Even though a higher ω usually
leads to a better privacy guarantee, there are two downsides to picking γ close to αm in this case.
First, note that ρ contains ψ(cid:48)(αm − γ); as γ → αm, the value of ρ diverges to ∞, leading to a weaker
privacy guarantee instead. Second, as the Taylor approximation (5) is accurate when ui is close to
ui + (λ − 1)(ui − u(cid:48)
i), having a large value of λ would push the guaranteed upper bound away from
the actual privacy loss. Thus it is recommended to pick γ so that γ/∆∞ ≥ 1 and αm − γ (cid:29) 0.
Alternatively, we can choose the value of γ that minimizes ε when converting from tCDP to (ε, δ)-DP
using Lemma 1—this method will be explored in the next subsection.

Theorem 1 can be easily applied to sampling from Dirichlet(rx + α). Replacing x with rx, we have
∆2 replaced by r∆2 and ∆∞ replaced by r∆∞. Consequently, the sampling is
-tCDP,
where ρ = 1
2ψ(cid:48)(αm − γ). In Appendix 4, we analyze the scaling of r in conjunction with αm
2 r2∆2
at a ﬁxed privacy budget ρ.

γ
r∆∞

+ 1

ρ,

(cid:16)

(cid:17)

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

3.2 Approximate differential privacy

200

201

We now convert the tCDP guarantee to an approximate DP guarantee. Let δ ∈ (0, 1). Using Lemma 1,
the Dirichlet posterior sampling with Dirichlet(α) as the prior is (ε, δ)-DP with

ε =

(cid:40)ρ(γ) + 2(cid:112)ρ(γ) log(1/δ)
+ 1

ρ(γ)

(cid:17)

(cid:16) γ
∆∞

+ log(1/δ)∆∞
γ

if log(1/δ) ≤ γ2ρ(γ)/∆2
∞
if log(1/δ) > γ2ρ(γ)/∆2
∞

,

(6)

202

203

204

205

206

207

where ρ(γ) = 1

2 ∆2

2ψ(cid:48)(αm − γ).

We try to minimize (cid:15) by adjusting the value of γ. First, we consider the case log(1/δ) ≤ γ2ρ(γ)/∆2
∞.
Since ρ(γ) is a strictly increasing function of γ, both ρ(γ) + 2(cid:112)ρ(γ) log(1/δ) and γ2ρ(γ)/∆2
∞
are both strictly increasing function of γ. Therefore, ε is minimized at the minimum possible
value of γ in this case, that is, at the unique γM that satisﬁes log(1/δ) = γ2
∞ =
1
2 γ2

2ψ(cid:48)(αm − γM )/∆2

M ρ(γM )/∆2

M ∆2

∞.

6

26101418101100True (,2)-tCDP upper bound0481216201010108106104102100=2=5=10208

209

210

211

212

213

214

215

216

Now we consider the second case, when γ < γM . As ρ(γ) is an increasing positive convex function
of γ, the function

f (γ) :=

1
2

∆2

2ψ(cid:48)(αm − γ)

(cid:18) γ
∆∞

(cid:19)

+ 1

+

log(1/δ)∆∞
γ

;

γ ∈ (0, γM ],

(7)

is also convex in γ, and thus has a unique minimizer γm ∈ (0, γM ]. Comparing to the ﬁrst case, we
have f (γm) ≤ f (γM ) = ρ(γM ) + 2(cid:112)ρ(γM ) log(1/δ). We then conclude that ε = f (γm).
Theorem 2. Let α ∈ R2
>0 and denote αm = mini αi. Let ∆2, ∆∞ > 0 be constants that satisfy
(cid:80)
≥0 are sample statistics of any
two datasets differing on a single entry. For any δ ∈ (0, 1), let γM be the solution to the equation
log(1/δ) = 1
∞. The one-time sampling from Dirichlet(x + α) is (ε, δ)-DP,
where

i| ≤ ∆∞ whenever x, x(cid:48) ∈ Rd

2 and maxi |xi − x(cid:48)

2ψ(cid:48)(αm − γ)/∆2

i(xi − x(cid:48)

i)2 ≤ ∆2

2 γ2∆2

217

Figure 1 shows how δ decays as a function of ε at three different values of αm.

ε = min

γ∈(0,γM ]

f (γ).

(8)

218

219

220

4 Utility

Using the results from the previous section, we analyze the Dirichlet posterior sampling’s utility in
two speciﬁc tasks.

221

4.1 Multinomial-Dirichlet sampling

Suppose that we are observing N trials, each of which has d possible outcomes. For each i ∈
{1, . . . , d}, let xi be the number of times the i-th outcome was observed. Then we have the
multinomial likelihood p(x|y) ∝ (cid:81)

. From this, we sample from the Dirichlet posterior:

i yxi

i

Y ∼ Dirichlet(x + α).

(9)

Suppose that we want to sample from a true distribution PX ∼ Dirichlet(x + α), but for privacy
reasons, we instead sample from Qx ∼ Dirichlet(x + α(cid:48)) where α(cid:48)
i > αi for all i. The utility of the
privacy scheme is then measured by the KL-divergence between Px and Qx. Assuming that x is an
observation of Multinomial(p), the following Theorem tells us that, on average, the KL-divergence
is small when the sample size is large, and the pi’s are evenly distributed.
Theorem 3. Let p := (p1, . . . , pd) where pi > 0 for all i and (cid:80)
i pi = 1. Deﬁne a random
variable X ∼ Multinomial(p). Let PX ∼ Dirichlet(X + α) and QX ∼ Dirichlet(X + α(cid:48)) where
α(cid:48)

i ≥ αi ≥ 1 for all i. The following estimate holds:
1
N + 1

EX[DKL(PX(cid:107)QX)] ≤

(cid:88)

(α(cid:48)

i − αi)2 ·

i

1
pi

.

(10)

233

234

The proof is given in Appendix 2. Let us consider a simple privacy scheme where we ﬁx s > 0 and
let α(cid:48)

i = αi + s for all i. Thus (10) becomes:

EX[DKL(PX(cid:107)QX)] ≤

G(p)s2
N + 1

,

(11)

where G(p) := (cid:80)
2ψ(cid:48)(α(cid:48)
and ρ(cid:48) = ∆2
the values of ψ(cid:48)(αm − γ) and ψ(cid:48)(α(cid:48)

i 1/pi. Now we take into account the privacy parameters. Let ρ = ∆2
m − γ), where αm = mini αi, α(cid:48)

2ψ(cid:48)(αm − γ)
i, and γ < αm. Here, we approximate

m = mini α(cid:48)
m − γ) under two regimes:

High-privacy regime: α(cid:48)
α(cid:48)
m − γ ≈ ∆2
for α − γ < 1. Thus we have the following bound for the right-hand side of (11):

2/ρ(cid:48). We also have αm − γ ≈ ∆2

m − γ > 1. We have ψ(cid:48)(α(cid:48)

m − γ) ≈ 1/(α(cid:48)

2/ρ for αm − γ ≥ 1 and αm − γ > (αm − γ)2 ≈ ∆2

m − γ), which implies
2/ρ

G(p)s2
N + 1

=

G(p)(α(cid:48)

m − αm)2

N + 1

(cid:19)2

1
ρ

<

∆4

2G(p)
ρ(cid:48)2(N + 1)

.

(12)

Consequently, we have DKL(P (cid:107)Q) < (cid:15) for N = Ω

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

235

236

237

238

239

240

241

(cid:46) ∆4

2G(p)
N + 1
(cid:16) ∆4

(cid:18) 1

ρ(cid:48) −
(cid:17)
.

2G(p)
ρ(cid:48)2(cid:15)

7

242

243

244

245

246

247

248

Low-privacy regime: 1 > α(cid:48)
m − γ ≈
∆2/ρ(cid:48)1/2 and αm − γ ≈ ∆2/ρ1/2. Similar computation as (12) shows that DKL(P (cid:107)Q) < (cid:15) when
N = Ω

m − γ > 0. This is similar as above, except we have α(cid:48)

(cid:16) ∆2

(cid:17)

.

2G(p)
ρ(cid:48)(cid:15)

We observe that, in both regimes, the sample size scales faster with respect to (cid:15) with a higher value of
G(p), which is associated with a higher number of outcomes d, and more concentrated multinomial
parameter p; this agrees with the result of our simulation in Appendix 3. Moreover, for small ρ(cid:48) the
sample size scales as 1/ρ(cid:48)2, while for large ρ(cid:48) the sample size scales as 1/ρ(cid:48).

249

4.2 Private normalized histograms

Let x = (x1, . . . , xd) be a histogram of N observations and p := x/N . We can privatize p by
sampling a probability vector: Y ∼ Dirichlet(x + α). Note that Y is a biased estimator of p.
Denoting α0 := (cid:80)

i αi, the bias of each component of Y is given by E[Y] − pi. Hence,

(cid:12)
(cid:12)
(cid:12)
(cid:12)
Since Yi ∼ Beta(xi + αi, N + α0 − xi − αi) is

xi + αi
N + α0

|Bias(Yi)| =

− pi

(cid:12)
(cid:12)
(cid:12)
(cid:12)

=

|xiα0 − N αi|
N (N + α0)

≤

N α0
N (N + α0)

=

α0
N + α0

.

1

4(N +α0+1) -sub-Gaussian [MA17], we have,

P[|Yi − pi| > t + |Bias(Yi)|] ≤ P[|Yi − E[Yi]| + |Bias(Yi)| > t + |Bias(Yi)|]

= P[|Yi − E[Yi]| > t]
≤ 2e−2t2(N +α0+1).
(cid:113) log(2d/β)

With the union bound, we plug in t =
accuracy guarantee of the private normalized histogram:
Theorem 4. Let Y ∼ Dirichlet(x + α), where x ∈ Rd
≥0 and α ∈ Rd
β ∈ (0, 1), with probability at least 1 − β, the following inequality holds:

2(N +α0+1) , for any β ∈ (0, 1), to obtain the following

>0, and p := x/N . For any

(cid:115)

(cid:107)Y − p(cid:107)∞ ≤

log(2d/β)
2(N + α0 + 1)

+

α0
N + α0

.

(13)

Given (cid:15) > 0, we use (13) to ﬁnd a lower bound for N that gives (cid:107)Y − p(cid:107)∞ < (cid:15) w.p. 1 − β when
Y is sampled with ρ-tCDP. For simplicity, we consider a uniform prior: αi = α > 0 for all i.
Thus, ρ = 1
2ψ(cid:48)(α − γ), where γ might be chosen according to Corollary 2. We consider the two
following regimes:

2 ∆2

High-privacy regime: α − γ > 1.
we have α ≈ ∆2

2/2ρ + γ. Replacing α0 by dα in (13) yields the sample size:

In this case, ψ(cid:48)(α − γ) ≈ 1/(α − γ). From ρ = 1

2 ∆2

2ψ(cid:48)(α − γ),

264

for the desired accuracy.

N = Ω

(cid:18) log(2d/β)
(cid:15)2

+

(cid:18) ∆2
2
2ρ

d
(cid:15)

(cid:19)(cid:19)

+ γ

,

(14)

Low-privacy regime: α − γ < 1. This is the same as above, except now we have ψ(cid:48)(α − γ) ≈
1/(α − γ)2, which implies α ≈ ∆2/(2ρ)1/2 + γ. The sample size that guarantees the desired
accuracy is:

N = Ω

(cid:18) log(2d/β)
(cid:15)2

+

d
(cid:15)

(cid:18) ∆2√
2ρ

(cid:19)(cid:19)

+ γ

.

(15)

Let us compare this result to the Gaussian mechanism, which adds a noise Z ∼ N (0, σ2Id) to the
normalized histogram p directly. Thus the (cid:96)2-sensitivity in this case is ∆2/N . We have that the
Gaussian mechanism is ρ-zCDP where ρ = ∆2
2N 2σ2 [BS16]. Using the same argument as above, with
probability at least 1 − β, the following inequality holds for all i:

(cid:107)Z(cid:107)∞ ≤

(cid:115)

log(2d/β)∆2
2
N 2ρ

.

8

(16)

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

265

266

267

268

269

270

271

Figure 2: The (cid:96)∞-accuracy, as a function of N , of Dirichlet posterior sampling (γ = 1) and Gaussian
mechanisms for private normalized histograms (∆2
2 = 2 and ∆∞ = 1). For each N, d and ρ, we
generated the inputs x1, . . . , x200, where xk ∼ Multinomial(qk) and qk ∼ Dirichlet(5, . . . , 5).

Hence, the sample size of N = Ω
ing this to (14), if we assume (cid:15) < 1, the AM-GM inequality tells us that

2/ρ(cid:15)2

(cid:16)(cid:112)log(2d/β)∆2

(cid:17)

guarantees the desired accuracy. Compar-

log(2d/β)
(cid:15)2

+

d∆2
2
ρ(cid:15)

>

log(2d/β)
(cid:15)2

+

∆2
2
ρ

≥ 2

(cid:115)

log(2d/β)∆2
2
ρ(cid:15)2

.

(17)

The inequality (17) implies that the Gaussian mechanism requires less sample than the Dirichlet
mechanism in order to guarantee the same level of accuracy. The Gaussian mechanism is also better
in the low-privacy regime as the ρ in (15) satisﬁes
2, leading to the same
inequality (17). Nonetheless, the decay in (16) is linear in d, while that in (13) has α0 = dα in
the denominators. This observation suggests that, when x is a sparse histogram i.e. when N ≤ d,
the (cid:96)∞-accuracy of the Dirichlet mechanism is smaller than that of the Gaussian mechanism. This
conclusion is supported by our simulation in Figure 2. We see that the (cid:96)∞-accuracy of the Dirichlet
mechanism is smaller than that of the Gaussian mechanism for small N when d = 1000. The code
for all experiments in this study can be found in the supplemental material.

ρ < ρ and ∆2 ≈ ∆2

√

Potential negative societal impacts

It is important to note that, when ρ becomes unacceptably large (e.g., ρ = 104), the sampling is far
away from being private. Thus any organization that deploys the posterior sampling on sensitive data
must not vacuously refer to this study and claim that its algorithm is private. It is the organization’s
responsibility to fully publish the prior parameters, and educate its users/customers on differential
privacy and how the privacy guarantees are calculated.

It is desirable that differentially private algorithms are accurate for the task at hand, especially when
the data is used for important decision-making. Thus, one needs to make sure that there is enough
sample to achieve the desired level of accuracy. For a large differentially private system, privacy
budgets need to be allocated to the parts that require accurate outputs.

Lastly, one must be careful with the choice of prior parameters; if a uniform prior is used, smaller
groups will suffer a relatively larger statistical bias. As a result, private statistics of small populations
(such as ethnic or racial minorities) will be relatively less accurate. One way to get around this issue
is to (privately) impose larger prior parameters on larger populations.

9

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

101103105Sample size (N)105103101101d=10,=0.01DirichletGaussian101103105Sample size (N)105103101101d=10,=0.1DirichletGaussian101103105Sample size (N)106104102100d=10,=1DirichletGaussian101103105Sample size (N)104102100d=1000,=0.01DirichletGaussian101103105Sample size (N)105103101101d=1000,=0.1DirichletGaussian101103105Sample size (N)105103101d=1000,=1DirichletGaussianReferences

[AAFK20]

[AV08]

[BDRS18]

[BHW00]

[BS16]

[BS18]

[CWS03]

[de 06]

[DMNS06]

[DNZMR17]

[FGWC16]

[GSC17]

[GWHHT21]

[Hin15]

[LW92]

[MA17]

[MKAGV08]

[MMR05]

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

347

348

349

350

351

352

353

354

355

I. Aykin, B. Akgun, M. Feng, and M. Krunz. “MAMBA: A Multi-armed Bandit Framework for
Beam Tracking in Millimeter-wave Systems”. In: 39th IEEE Conference on Computer Com-
munications, INFOCOM 2020, Toronto, ON, Canada, July 6-9, 2020. IEEE, 2020, pp. 1469–
1478.
J. M. Abowd and L. Vilhuber. “How Protective Are Synthetic Data?” In: Privacy in Statistical
Databases, UNESCO Chair in Data Privacy International Conference, PSD 2008, Istan-
bul, Turkey, September 24-26, 2008. Proceedings. Ed. by J. Domingo-Ferrer and Y. Saygin.
Vol. 5262. Lecture Notes in Computer Science. Springer, 2008, pp. 239–246.
M. Bun, C. Dwork, G. N. Rothblum, and T. Steinke. “Composable and versatile privacy via
truncated CDP”. In: Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of
Computing, STOC 2018, Los Angeles, CA, USA, June 25-29, 2018. Ed. by I. Diakonikolas,
D. Kempe, and M. Henzinger. ACM, 2018, pp. 74–86.
R. J. Boys, D. A. Henderson, and D. J. Wilkinson. “Detecting Homogeneous Segments in DNA
Sequences by Using Hidden Markov Models”. In: Journal of the Royal Statistical Society.
Series C (Applied Statistics) 49.2 (2000), pp. 269–285. ISSN: 00359254, 14679876.
M. Bun and T. Steinke. “Concentrated Differential Privacy: Simpliﬁcations, Extensions, and
Lower Bounds”. In: Theory of Cryptography. Ed. by M. Hirt and A. Smith. Berlin, Heidelberg:
Springer Berlin Heidelberg, 2016, pp. 635–658.
G. Bernstein and D. R. Sheldon. “Differentially Private Bayesian Inference for Exponential
Families”. In: Advances in Neural Information Processing Systems 31: Annual Conference on
Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal,
Canada. Ed. by S. Bengio, H. M. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and
R. Garnett. 2018, pp. 2924–2934.
J. Corander, P. Waldmann, and M. J. Sillanpää. “Bayesian Analysis of Genetic Differentiation
Between Populations”. In: Genetics 163.1 (Jan. 2003), pp. 367–374. ISSN: 1943-2631.
M. de Lapparent. “Empirical Bayesian analysis of accident severity for motorcyclists in large
French urban areas”. In: Accident Analysis & Prevention 38.2 (2006), pp. 260–268. ISSN:
0001-4575.
C. Dwork, F. Mcsherry, K. Nissim, and A. Smith. “Calibrating noise to sensitivity in private
data analysis”. In: TCC. 2006.
C. Dimitrakakis, B. Nelson, Z. Zhang, A. Mitrokotsa, and B. I. P. Rubinstein. “Differential
Privacy for Bayesian Inference through Posterior Sampling”. In: J. Mach. Learn. Res. 18
(2017), 11:1–11:39.
J. R. Foulds, J. Geumlek, M. Welling, and K. Chaudhuri. “On the Theory and Practice of
Privacy-Preserving Bayesian Data Analysis”. In: Proceedings of the Thirty-Second Conference
on Uncertainty in Artiﬁcial Intelligence, UAI 2016, June 25-29, 2016, New York City, NY, USA.
Ed. by A. T. Ihler and D. Janzing. AUAI Press, 2016.
J. Geumlek, S. Song, and K. Chaudhuri. “Renyi Differential Privacy Mechanisms for Posterior
Sampling”. In: Advances in Neural Information Processing Systems 30: Annual Conference
on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA.
Ed. by I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan,
and R. Garnett. 2017, pp. 5289–5298.
P. Gohari, B. Wu, C. Hawkins, M. T. Hale, and U. Topcu. “Differential Privacy on the Unit
Simplex via the Dirichlet Mechanism”. In: IEEE Trans. Inf. Forensics Secur. 16 (2021),
pp. 2326–2340.
K. Hines. “A Primer on Bayesian Inference for Biophysical Systems”. In: Biophysical Journal
108.9 (2015), pp. 2103–2113. ISSN: 0006-3495.
M. Lavine and M. West. “A Bayesian method for classiﬁcation and discrimination”. In:
Canadian Journal of Statistics 20.4 (1992), pp. 451–461.
O. Marchal and J. Arbel. “On the sub-Gaussianity of the Beta and Dirichlet distributions”. In:
Electronic Communications in Probability 22.none (2017), pp. 1 –14.
A. Machanavajjhala, D. Kifer, J. M. Abowd, J. Gehrke, and L. Vilhuber. “Privacy: Theory
meets Practice on the Map”. In: Proceedings of the 24th International Conference on Data
Engineering, ICDE 2008, April 7-12, 2008, Cancún, Mexico. Ed. by G. Alonso, J. A. Blakeley,
and A. L. P. Chen. IEEE Computer Society, 2008, pp. 277–286.
J.-M. Marin, K. Mengersen, and C. P. Robert. “Bayesian Modelling and Inference on Mixtures
of Distributions”. In: Bayesian Thinking. Ed. by D. Dey and C. Rao. Vol. 25. Handbook of
Statistics. Elsevier, 2005, pp. 459–507.

10

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

F. McSherry and K. Talwar. “Mechanism Design via Differential Privacy”. In: 48th Annual
IEEE Symposium on Foundations of Computer Science (FOCS 2007), October 20-23, 2007,
Providence, RI, USA, Proceedings. IEEE Computer Society, 2007, pp. 94–103.
I. Nasim, A. S. Ibrahim, and S. Kim. “Learning-Based Beamforming for Multi-User Vehicular
Communications: A Combinatorial Multi-Armed Bandit Approach”. In: IEEE Access 8 (2020),
pp. 219891–219902.
V. C. Nguyen, W. S. Lee, N. Ye, K. M. A. Chai, and H. L. Chieu. “Active Learning for
Probabilistic Hypotheses Using the Maximum Gibbs Error Criterion”. In: Advances in Neural
Information Processing Systems 26: 27th Annual Conference on Neural Information Processing
Systems 2013. Proceedings of a meeting held December 5-8, 2013, Lake Tahoe, Nevada,
United States. Ed. by C. J. C. Burges, L. Bottou, Z. Ghahramani, and K. Q. Weinberger. 2013,
pp. 1457–1465.
I. Osband, D. Russo, and B. V. Roy. “(More) Efﬁcient Reinforcement Learning via Poste-
rior Sampling”. In: Advances in Neural Information Processing Systems 26: 27th Annual
Conference on Neural Information Processing Systems 2013. Proceedings of a meeting held
December 5-8, 2013, Lake Tahoe, Nevada, United States. Ed. by C. J. C. Burges, L. Bottou,
Z. Ghahramani, and K. Q. Weinberger. 2013, pp. 3003–3011.
T. Pedersen and R. F. Bruce. “Knowledge Lean Word-Sense Disambiguation”. In: Proceedings
of the Fifteenth National Conference on Artiﬁcial Intelligence and Tenth Innovative Appli-
cations of Artiﬁcial Intelligence Conference, AAAI 98, IAAI 98, July 26-30, 1998, Madison,
Wisconsin, USA. Ed. by J. Mostow and C. Rich. AAAI Press / The MIT Press, 1998, pp. 800–
805.
Y. Park and J. Ghosh. “PeGS: Perturbed Gibbs Samplers that Generate Privacy-Compliant
Synthetic Data”. In: Trans. Data Priv. 7.3 (2014), pp. 253–282.
J. Pella and M. Masuda. “Bayesian methods for analysis of stock mixtures from genetic
characters”. English. In: Fishery Bulletin 99 (Jan. 2001). 1, p. 151. ISSN: 00900656.
A. Rényi. “On measures of entropy and information”. In: Proceedings of the Fourth Berkeley
Symposium on Mathematical Statistics and Probability, Volume 1: Contributions to the Theory
of Statistics. The Regents of the University of California. 1961.
J. P. Reiter, Q. Wang, and B. Zhang. “Bayesian Estimation of Disclosure Risks for Multiply
Imputed, Synthetic Data”. In: J. Priv. Conﬁdentiality 6.1 (2014).
M. J. Schneider, S. Jagpal, S. Gupta, S. Li, and Y. Yu. “Protecting customer privacy when
marketing with second-party data”. In: International Journal of Research in Marketing 34.3
(2017), pp. 593–603. ISSN: 0167-8116.
M. J. A. Strens. “A Bayesian Framework for Reinforcement Learning”. In: Proceedings of the
Seventeenth International Conference on Machine Learning (ICML 2000), Stanford University,
Stanford, CA, USA, June 29 - July 2, 2000. Ed. by P. Langley. Morgan Kaufmann, 2000,
pp. 943–950.
Y. Wang, S. E. Fienberg, and A. J. Smola. “Privacy for Free: Posterior Sampling and Stochastic
Gradient Monte Carlo”. In: Proceedings of the 32nd International Conference on Machine
Learning, ICML 2015, Lille, France, 6-11 July 2015. Ed. by F. R. Bach and D. M. Blei. Vol. 37.
JMLR Workshop and Conference Proceedings. JMLR.org, 2015, pp. 2493–2502.
J. Zhu, X. Huang, X. Gao, Z. Shao, and Y. Yang. “Multi-Interface Channel Allocation in Fog
Computing Systems using Thompson Sampling”. In: 2020 IEEE International Conference on
Communications, ICC 2020, Dublin, Ireland, June 7-11, 2020. IEEE, 2020, pp. 1–6.

[MT07]

[NIK20]

[NLYCC13]

[ORR13]

[PB98]

[PG14]

[PM01]

[Rén61]

[RWZ14]

[SJGLY17]

[Str00]

[WFS15]

[ZHGSY20]

Checklist

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reﬂect the paper’s
contributions and scope? [Yes] We gave a simple guaranteed upper bound of tCDP (2)
for the Dirichlet posterior sampling and illustrated how it can be used to derive accuracy
guarantees in Section 4.

(b) Did you describe the limitations of your work? [Yes] We discussed a limitation of
the guaranteed upper bound of tCDP in the paragraph following Theorem 1. We also
described a situation under which the Gaussian mechanism is preferable to the Dirichlet
posterior sampling at the end of Section 4.2.

(c) Did you discuss any potential negative societal impacts of your work? [Yes] See the

section on potential negative societal impacts at the end of the paper.

11

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

(d) Have you read the ethics review guidelines and ensured that your paper conforms to

them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [Yes]
(b) Did you include complete proofs of all theoretical results? [No] The proofs of all
theorems are given in the main paper, except that of Theorem 3 which is given in
Appendix 2.

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main experi-
mental results (either in the supplemental material or as a URL)? [Yes] The code and
the instructions for our simulations are included in the supplemental material.

(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they
were chosen)? [Yes] We speciﬁed the details of our simulations in the ﬁgures’ captions.
(c) Did you report error bars (e.g., with respect to the random seed after running experi-

ments multiple times)? [Yes] We reported the error bars in Figure 2

(d) Did you include the total amount of compute and the type of resources used (e.g.,
type of GPUs, internal cluster, or cloud provider)? [N/A] Our experiments are not
computationally intensive.

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [N/A]
(b) Did you mention the license of the assets? [N/A]
(c) Did you include any new assets either in the supplemental material or as a URL? [N/A]

(d) Did you discuss whether and how consent was obtained from people whose data you’re

using/curating? [N/A]

(e) Did you discuss whether the data you are using/curating contains personally identiﬁable

information or offensive content? [N/A]

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if

applicable? [N/A]

(b) Did you describe any potential participant risks, with links to Institutional Review

Board (IRB) approvals, if applicable? [N/A]

(c) Did you include the estimated hourly wage paid to participants and the total amount

spent on participant compensation? [N/A]

12

