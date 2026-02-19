Proximal and Federated Random Reshufﬂing

Anonymous Author(s)
Afﬁliation
Address
email

Abstract

Random Reshufﬂing (RR), also known as Stochastic Gradient Descent (SGD)
without replacement, is a popular and theoretically grounded method for ﬁnite-sum
minimization. We propose two new algorithms: Proximal and Federated Random
Reshufﬂing (ProxRR and FedRR). The ﬁrst algorithm, ProxRR, solves composite
ﬁnite-sum minimization problems in which the objective is the sum of a (potentially
non-smooth) convex regularizer and an average of n smooth objectives. ProxRR
evaluates the proximal operator once per epoch only. When the proximal operator
is expensive to compute, this small difference makes ProxRR up to n times faster
than algorithms that evaluate the proximal operator in every iteration, such as
proximal (stochastic) gradient descent. We give examples of practical optimization
tasks where the proximal operator is difﬁcult to compute and ProxRR has a clear
advantage. One such task is federated or distributed optimization, where the evalu-
ation of the proximal operator corresponds to communication across the network.
We obtain our second algorithm, FedRR, as a special case of ProxRR applied to
federated optimization, and prove it has a smaller communication footprint than
either distributed gradient descent or Local SGD. Our theory covers both constant
and decreasing stepsizes, and allows for importance resampling schemes that can
improve conditioning, which may be of independent interest. Our theory covers
both convex and nonconvex regimes. Finally, we corroborate our results with
experiments on real data sets.

1

Introduction

Modern theory and practice of training supervised machine learning models is based on the paradigm
of regularized empirical risk minimization (ERM) [Shalev-Shwartz and Ben-David, 2014]. While the
ultimate goal of supervised learning is to train models that generalize well to unseen data, in practice
only a ﬁnite data set is available during training. Settling for a model merely minimizing the average
loss on this training set—the empirical risk—is insufﬁcient, as this often leads to over-ﬁtting and poor
generalization performance in practice. Due to this reason, empirical risk is virtually always amended
with a suitably chosen regularizer whose role is to encode prior knowledge about the learning task at
hand, thus biasing the training algorithm towards better performing models.

The regularization framework is quite general and perhaps surprisingly it also allows us to consider
methods for federated learning (FL)—a paradigm in which we aim at training model for a number of
clients that do not want to reveal their data [Koneˇcný et al., 2016, McMahan et al., 2017, Kairouz
et al., 2019]. The training in FL usually happens on devices with only a small number of model
updates being shared with a global host. To this end, Federated Averaging algorithm has emerged
that performs Local SGD updates on the clients’ devices and periodically aggregates their average.
Its analysis usually requires special techniques and deliberately constructed sequences hindering the
research in this direction. We shall see, however, that the convergence of our FedRR follows from
merely applying our algorithm for regularized problems to a carefully chosen reformulation.

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

Formally, regularized ERM problems are optimization problems of the form

(cid:2)P (x) := 1

n

(cid:80)n

i=1 fi(x) + ψ(x)(cid:3),

min
x∈Rd

(1)

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

where fi : Rd → R is the loss of model parameterized by vector x ∈ Rd on the i-th training data
point, and ψ : Rd → R ∪ {+∞} is a regularizer. Let [n] := {1, 2, . . . , n}. We shall make the
following assumption throughout the paper without explicitly mentioning it:
Assumption 1. The functions fi are Li-smooth, and the regularizer ψ is proper, closed and convex.
Let Lmax := maxi∈[n] Li.

(cid:80)

In some results we will additionally assume that either the individual functions fi, or their average
f := 1
i fi, or the regularizer ψ are µ-strongly convex. Whenever we need such additional
n
assumptions, we will make this explicitly clear. While all these concepts are standard, we review
them brieﬂy in Section A.

Proximal SGD. When the number n of training data points is huge, as is increasingly common
in practice, the most efﬁcient algorithms for solving (1) are stochastic ﬁrst-order methods, such
as stochastic gradient descent (SGD) [Bordes et al., 2009], in one or another of its many variants
proposed in the last decade [Shang et al., 2018, Pham et al., 2020]. These method almost invariably
rely on alternating stochastic gradient steps with the evaluation of the proximal operator

proxγψ(x) := argminz∈Rd

(cid:8)γψ(z) + 1

2 (cid:107)z − x(cid:107)2(cid:9) .

54

The simplest of these has the form

k+1 = proxγkψ(xSGD
xSGD

k − γk∇fik (xSGD

k

)),

(2)

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

where ik is an index from {1, 2, . . . , n} chosen uniformly at random, and γk > 0 is a properly
chosen learning rate. Our understanding of (2) is quite mature; see [Gorbunov et al., 2020] for a
general treatment which considers methods of this form in conjunction with more advanced stochastic
gradient estimators in place of ∇fik .
Applications such as training sparse linear models [Tibshirani, 1996], nonnegative matrix factoriza-
tion [Lee and Seung, 1999], image deblurring [Rudin et al., 1992, Bredies et al., 2010], and training
with group selection [Yuan and Lin, 2006] all rely on the use of hand-crafted regularizes. For most of
them, the proximal operator can be evaluated efﬁciently, and SGD is near or at the top of the list of
efﬁcient training algorithms.

Random reshufﬂing. A particularly successful variant of SGD is based on the idea of random
shufﬂing (permutation) of the training data followed by n iterations of the form (2), with the index
ik following the pre-selected permutation [Bottou, 2012]. This process is repeated several times,
each time using a new freshly sampled random permutation of the data, and the resulting method is
known under the name Random Reshufﬂing (RR). When the same permutation is used throughout,
the technique is known under the name Shufﬂe-Once (SO).

One of the main advantages of this approach is rooted in its intrinsic ability to avoid cache misses when
reading the data from memory, which enables a signiﬁcantly faster implementation. Furthermore,
RR is often observed to converge in fewer iterations than SGD in practice. This can intuitively be
ascribed to the fact that while due to its sampling-with-replacement approach SGD can miss to learn
from some data points in any given epoch, RR will learn from each data point in each epoch.

Understanding the random reshufﬂing trick, and why it works, has been a non-trivial open problem
for a long time [Bottou, 2009, Recht and Ré, 2012, Gürbüzbalaban et al., 2019, Haochen and Sra,
2019]. Until recent development which lead to a signiﬁcant simpliﬁcation of the convergence
analysis technique and proofs [Mishchenko et al., 2020], prior state of the art relied on long and
elaborate proofs requiring sophisticated arguments and tools, such as analysis via the Wasserstein
distance [Nagaraj et al., 2019], and relied on a signiﬁcant number of strong assumptions about
the objective [Shamir, 2016, Haochen and Sra, 2019]. In alternative recent development, Ahn et al.
[2020] also develop new tools for analyzing the convergence of random reshufﬂing, in particular using
decreasing stepsizes and for objectives satisfying the Polyak-Łojasiewicz condition, a generalization
of strong convexity [Polyak, 1963, Lojasiewicz, 1963].

The difﬁculty of analyzing RR has been the main obstacle in the development of even some of the
most seemingly benign extensions of the method. Indeed, while all these are well understood in

2

Algorithm 1 Proximal Random Reshufﬂing (ProxRR) and Shufﬂe-Once (ProxSO)

Require: Stepsizes γt > 0, initial vector x0 ∈ Rd, number of epochs T
1: Sample a permutation π = (π0u, π1, . . . , πn−1) of [n] (Do step 1 only for ProxSO)
2: for epochs t = 0, 1, . . . , T − 1 do
3:

Sample a permutation π = (π0, π1, . . . , πn−1) of [n] (Do step 3 only for ProxRR)
x0
for i = 0, 1, . . . , n − 1 do

t = xt

4:
5:
6:
7:

xi+1
t − γt∇fπi (xi
t = xi
t)
xt+1 = proxγtnψ(xn
t )

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

combination with its much simpler-to-analyze cousin SGD, to the best of our knowledge, there exists
no theoretical analysis of proximal, parallel, and importance sampling variants of RR with both
constant and decreasing stepsizes, and in most cases it is not even clear how should such methods be
constructed. Empowered by and building on the recent advances of Mishchenko et al. [2020], in this
paper we address all these challenges.

2 Contributions

In this section we outline the key contributions of our work, and also offer a few intuitive explanations
motivating some of the development.

• New algorithm: ProxRR. Despite rich literature on Proximal SGD [Gorbunov et al., 2020], it is
not obvious how one should extend RR to solve problem (1) when a regularizer ψ is present. Indeed,
the standard practice for SGD is to apply the proximal operator after each stochastic step [Duchi and
Singer, 2009], i.e., in analogy with (2). On the other hand, RR is motivated by the fact that a data
pass better approximates the full gradient step. If we applied the proximal operator after each step of
RR, we would no longer approximate the full gradient after an epoch, as we illustrate next.
2 (cid:107)x(cid:107)2, f1(x) = (cid:104)c1, x(cid:105), f2(x) = (cid:104)c2, x(cid:105) with some c1, c2 ∈ Rd,
Example 1. Let n = 2, ψ(x) = 1
c1 (cid:54)= c2. Let x0 ∈ Rd, γ > 0 and deﬁne x1 = x0 − γ∇f1(x0), x2 = x1 − γ∇f2(x1). Then, we
have prox2γψ(x2) = prox2γψ(x0 − 2γ∇f (x0)). However, if ˜x1 = proxγψ(x0 − γ∇f1(x0)) and
˜x2 = proxγψ(x1 − γ∇f2(˜x1)), then ˜x2 (cid:54)= prox2γψ(x0 − 2γ∇f (x0)).

Motivated by this observation, we propose ProxRR (Algorithm 1), in which the proximal operator is
applied at the end of each epoch of RR, i.e., after each pass through all randomly reshufﬂed data. A
notable property of Algorithm 1 is that only a single proximal operator evaluation is needed during
each data pass. This is in sharp contrast with the way Proximal SGD works, and offers signiﬁcant
advantages in regimes where the evaluation of the proximal mapping is expensive (e.g., comparable
to the evaluation of n gradients ∇f1, . . . , ∇fn).

• Convergence of ProxRR (for strongly convex functions or regularizer). We establish several
convergence results for ProxRR, of which we highlight two here. Both offer a linear convergence rate
with a ﬁxed stepsize to a neighborhood of the solution. In both we reply on Assumption 1. Firstly, in
the case when in addition, each fi is µ-strongly convex, we prove the rate (see Theorem 2)

E

(cid:107)xT − x∗(cid:107)2(cid:105)
(cid:104)

≤ (1 − γµ)nT (cid:107)x0 − x∗(cid:107)2 + 2γ2σ2

rad

µ

,

where γt = γ ≤ 1/Lmax is the stepsize, and σ2
rad is a shufﬂing radius constant (for precise deﬁnition,
see (4)). In Theorem 1 we bound the shufﬂing radius in terms of (cid:107)∇f (x∗)(cid:107)2, n, Lmax and the more
i=1 (cid:107)∇fi(x∗) − ∇f (x∗)(cid:107)2. Secondly, if ψ is µ-strongly convex, and
common quantity σ2
we choose the stepsize γt = γ ≤ 1/Lmax, we prove the rate (see Theorem 3)

∗ := 1
n

(cid:80)n

E

(cid:104)

(cid:107)xT − x∗(cid:107)2(cid:105)

≤ (1 + 2γµn)−T (cid:107)x0 − x∗(cid:107)2 + γ2σ2

rad

µ

.

119

120

Both mentioned rates show exponential (linear in logarithmic scale) convergence to a neighborhood
whose size is proportional to γ2σ2
rad. Since we can choose γ to be arbitrarily small or periodically

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

decrease it, this implies that the iterates converge to x∗ in the limit. Moreover, we show in Section 4
that when γ = O( 1

T 2 ), which is superior to the O( 1

T ) the error is O( 1

T ) error of SGD.

• Results for SO. All of our results apply to the Shufﬂe-Once algorithm as well. For simplicity, we
center the discussion around RR, whose current theoretical guarantees in the nonconvex case are
better than that of SO. Nevertheless, the other results are the same for both methods, and ProxRR is
identical to ProxSO in terms of our theory too. A study of the empirical differences between RR and
SO can be found in [Mishchenko et al., 2020].

• Application to Federated Learning. In Section 6 we describe an application of our results to
federated learning [Koneˇcný et al., 2016, McMahan et al., 2017, Kairouz et al., 2019]. In this way we
obtain the FedRR method, which is similar to Local SGD, except the local solver is a single pass
of RR over the local data. Empirically, FedRR can be vastly superior to Local SGD (see Figure 2).
Remarkably, we also show that the rate of FedRR beats the best known lower bound for Local SGD
due to [Woodworth et al., 2020] (we needed to adapt it from the original online to the ﬁnite-sum
setting we consider in this paper) for large enough n. See Section F for more details.

• Nonconvex analysis. In the nonconvex regime, and under suitable assumptions, we establish (see
Theorems 5 and 8) an O( 1
γT ) rate up to a neighborhood of size O(γ2). For a certain stepsize it yields
an O( 1
ε3 ) convergence rate.

Besides the above results, we describe several extensions in the appendix, which we now outline.

• Extension 1: Decreasing stepsizes. The convergence of RR is not always exact and depends on
the parameters of the objective. Similarly, if the shufﬂing radius σ2
rad is positive, and we wish to ﬁnd
an ε-approximate solution, the optimal choice of a ﬁxed stepsize for ProxRR will depend on ε. This
deﬁciency can be ﬁxed by using decreasing stepsizes in both vanilla RR [Ahn et al., 2020] and in
SGD [Stich, 2019]. We adopt the same technique to our setting. However, we depart from [Ahn et al.,
2020] by only adjusting the stepsize once per epoch rather than at every iteration, similarly to the
concurrent work of Tran et al. [2020] on RR with momentum. For details, see Section I.

• Extension 2: Importance resampling for Proximal RR. While importance sampling is a well
established technique for speeding up the convergence of SGD [Zhao and Zhang, 2015, Khaled and
Richtárik, 2020], no importance sampling variant of RR has been proposed nor analyzed. This is not
surprising since the key property of importance sampling in SGD—unbiasedness—does not hold for
RR. Our approach to equip ProxRR with importance sampling is via a reformulation of problem (1)
into a similar problem with a larger number of summands. In particular, for each i ∈ [n] we include
fi, and then take average of all N = (cid:80)
ni copies of the function 1
i ni functions constructed this
ni
way. The value of ni depends on the “importance” of fi, described below. We then apply ProxRR
to this reformulation. If fi is Li-smooth for all i ∈ [n] and we let ¯L := 1
i Li, then we choose
n
ni = (cid:100)Li/¯L(cid:101). It is easy to show that N ≤ 2n, and hence our reformulation leads to at most a doubling
of the number of functions forming the ﬁnite sum. However, the overall complexity of ProxRR
applied to this reformulation will depend on ¯L instead of maxi Li (see Theorem 10), which can lead
to a signiﬁcant improvement. For details of the construction and our complexity results, see Section J.

(cid:80)

159

3 Preliminaries

160

161

162

163

164

165

166

167

In our analysis, we build upon the notions of limit points and shufﬂing variance introduced by
Mishchenko et al. [2020] for vanilla (i.e., non-proximal) RR. Given a stepsize γ > 0 (held constant
during each epoch) and a permutation π of {1, 2, . . . , n}, the inner loop iterates of RR/SO converge
∗, x2
to a neighborhood of intermediate limit points x1

∗ deﬁned by

∗, . . . , xn

∗ := x∗ − γ (cid:80)i−1
xi

j=0 ∇fπj (x∗),

i = 1, . . . , n.

(3)

The intuition behind this deﬁnition is fairly simple: if we performed i steps starting at x∗, we would
end up close to xi

∗. To quantify the closeness, we deﬁne the shufﬂing radius.

Deﬁnition 1 (Shufﬂing radius). Given a stepsize γ > 0 and a random permutation π of {1, 2, . . . , n}
used in Algorithm 1, deﬁne xi

∗(γ, π) as in (3). Then, the shufﬂing radius is deﬁned by

∗ = xi

σ2
rad(γ) := max

i=0,...,n−1

(cid:104) 1
γ2 Eπ

(cid:2)Dfπi

∗, x∗)(cid:3)(cid:105)

(xi

,

(4)

4

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

207

208

209

where the expectation is taken with respect to the randomness in the permutation π. If there are
multiple stepsizes γ1, γ2, . . . used in Algorithm 1, we take the maximum of all of them as the shufﬂing
radius, i.e., σ2

rad := maxt≥1 σ2

rad(γt).

The shufﬂing radius is related by a multiplicative factor in the stepsize to the shufﬂing variance
introduced by Mishchenko et al. [2020]. When the stepsize is held ﬁxed, the difference between the
two notions is minimal. When the stepsize is decreasing, however, the shufﬂing radius is easier to
work with, since it can be upper bounded by problem constants independent of the stepsizes.

Armed with a special lemma for sampling without replacement, we can upper bound the shufﬂing
radius using the smoothness constant Lmax, size of the vector ∇f (x∗), and the variance σ2
∗ of the
gradient vectors ∇f1(x∗), . . . , ∇fn(x∗).
Theorem 1 (Bounding the shufﬂing radius). For any stepsize γ > 0 and any random permutation π
(cid:1), where x∗ is a solution of Problem (1)
of {1, 2, . . . , n} we have σ2
and σ2

2 n(cid:0)n(cid:107)∇f (x∗)(cid:107)2 + 1

∗ is the population variance at the optimum

rad ≤ Lmax

2 σ2

∗

∗ := 1
σ2
n

(cid:80)n

i=1(cid:107)∇fi(x∗) − ∇f (x∗)(cid:107)2.

(5)

All proofs are relegated to the supplementary material. In order to better understand the bound
given by Theorem 1, note that if there is no proximal operator (i.e., ψ = 0) then ∇f (x∗) = 0 and
we get that σ2
. This recovers the existing upper bound on the shufﬂing variance of
Mishchenko et al. [2020] for vanilla RR. On the other hand, if ∇f (x∗) (cid:54)= 0 then we get an additive
term of size proportional to the squared norm of ∇f (x∗).

rad ≤ Lmaxnσ2

4

∗

4 Theory for strongly convex losses f1, . . . , fn

Our ﬁrst theorem establishes a convergence rate for Algorithm 1 applied with a constant stepsize to
Problem (1) when each objective fi is strongly convex. This assumption is commonly satisﬁed in
machine learning applications where each fi represents a regularized loss on some data points, as in
(cid:96)2 regularized linear regression and (cid:96)2 regularized logistic regression.
Theorem 2. Let Assumption 1 be satisﬁed. Further, assume that each fi is µ-strongly convex. If
Algorithm 1 is run with constant stepsize γt = γ ≤ 1/Lmax, then its iterates satisfy

E

(cid:104)

(cid:107)xT − x∗(cid:107)2(cid:105)

≤ (1 − γµ)nT (cid:107)x0 − x∗(cid:107)2 + 2γ2σ2

rad

µ

.

We can convert the guarantee of Theorem 2 to a convergence rate by properly tuning the stepsize
and using the upper bound of Theorem 1 on the shufﬂing radius. In particular, if we choose the
, and let κ := Lmax/µ and r0 := (cid:107)x0 − x∗(cid:107)2, then we obtain
stepsize as γ = min
(cid:107)xT − x∗(cid:107)2(cid:105)

= O (ε) provided that the total number of iterations KRR = nT is at least

εµ
2σrad

(cid:110) 1

Lmax

(cid:111)

E

(cid:104)

√

√

,

KRR ≥ [(κ +

√

√
κn√
εµ (

n (cid:107)∇f (x∗)(cid:107) + σ∗)] log (cid:0) 2r0

ε

(cid:1) .

(6)

Comparison with vanilla RR. If there is no proximal operator, then (cid:107)∇f (x∗)(cid:107) = 0 and we recover
the earlier result of Mishchenko et al. [2020] on the convergence of RR without proximal, which is
optimal in ε up to logarithmic factors. On the other hand, when the proximal operator is nonzero,
we get an extra term in the complexity proportional to (cid:107)∇f (x∗)(cid:107): thus, even when all the functions
are the same (i.e., σ∗ = 0), we do not recover the linear convergence of Proximal Gradient Descent
[Karimi et al., 2016, Beck, 2017]. This can be easily explained by the fact that Algorithm 1 performs
n gradient steps per one proximal step. Hence, even if f1 = · · · = fn, Algorithm 1 does not reduce
to Proximal Gradient Descent. We note that other algorithms for composite optimization which may
not take a proximal step at every iteration (for example, using stochastic projection steps) also suffer
from the same dependence [Patrascu and Irofti, 2021].

Comparison with proximal SGD. In order to compare (6) against the complexity of Proximal SGD
(Algorithm 2), we recall that Proximal SGD achieves E
= O (ε) if either f or ψ is
µ-strongly convex and

(cid:107)xK − x∗(cid:107)2(cid:105)
(cid:104)

KSGD ≥

log (cid:0) 2r0

ε

(cid:1) .

(7)

(cid:17)

(cid:16)

κ + σ2
∗
εµ2

5

Algorithm 2 Proximal SGD
Require: Stepsizes γk > 0, initial vector x0 ∈ Rd, number of steps K
1: for steps k = 0, 1, . . . , K − 1 do
2:
3:

Sample ik uniformly at random from [n]
xk+1 = proxγkψ(xk − γk∇fik (xk))

This result is standard [Needell et al., 2016, Gower et al., 2019], with the exception that we do not
know any proof in the literature for the case when ψ is strongly convex. For completeness, we prove
it in Appendix C, but since our proof is a minor modiﬁcation of that in [Gower et al., 2019], we do
not provide it here.

By comparing KSGD (given by (7)) and KRR (given by (6)), we see that ProxRR has milder
dependence on ε than Proximal SGD. In particular, ProxRR converges faster whenever the target
(cid:16)
. Furthermore, ProxRR is much
accuracy ε is small enough to satisfy ε ≤
better when we consider proximal iteration complexity (# of proximal operator access), in which case
the complexity of ProxRR (6) is reduced by a factor of n (because we take one proximal step every n
iterations), while the proximal iteration complexity of Proximal SGD remains the same as (7). In this
case, ProxRR is better whenever the accuracy ε satisﬁes

σ4
∗
n(cid:107)∇f (x∗)(cid:107)2+σ2
∗

1
Lmaxnµ

(cid:17)

(cid:104)

ε ≥ n

Lmaxµ

n(cid:107)∇f (x∗)(cid:107)2 + σ2
∗

(cid:105)

or

ε ≤ n

Lmaxµ

(cid:104)

σ4
∗
n(cid:107)∇f (x∗)(cid:107)2+σ2
∗

(cid:105)

.

We can see that if the target accuracy is large enough or small enough, and if the cost of proximal
operators dominates the computation, ProxRR is much quicker to converge than Proximal SGD.

5 Theory for strongly convex regularizer ψ

In Theorem 2, we assume that each fi is µ-strongly convex. This is motivated by the common practice
of using (cid:96)2 regularization in machine learning. However, applying (cid:96)2 regularization in every step
of Algorithm 1 can be expensive when the data are sparse and the iterates xi
t are dense, because it
requires accessing each coordinate of xi
t which can be much more expensive than computing sparse
gradients ∇fi(xi
t). Alternatively, we may instead choose to put the (cid:96)2 regularization inside ψ and
only ask that ψ be strongly convex—this way, we can save a lot of time as we need to access each
coordinate of the dense iterates xi
t only once per epoch rather than every iteration. Theorem 3 gives a
convergence guarantee in this setting.
Theorem 3. Let Assumption 1 hold and f1, . . . , fn be convex. Further, assume that ψ is µ-strongly
convex. If Algorithm 1 is run with constant stepsize γt = γ ≤ 1/Lmax, where Lmax = maxi Li, then
its iterates satisfy

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

232

233

234

E

(cid:104)

(cid:107)xT − x∗(cid:107)2(cid:105)

≤ (1 + 2γµn)−T (cid:107)x0 − x∗(cid:107)2 + γ2σ2

rad

µ

.

235

Using Theorem 3 and choosing the stepsize as

236

we get E

(cid:107)xT − x∗(cid:107)2(cid:105)
(cid:104)

= O (ε) provided that the total number of iterations satisﬁes

γ = min

(cid:110) 1

Lmax

√

εµ
σrad

,

(cid:111)

,

(8)

εµ + n
This can be converted to a bound similar to (6) by using Theorem 1, in which case the only difference
between the two cases is an extra n log (cid:0) 1
(cid:1) term when only the regularizer ψ is µ-strongly convex.
ε
Since for small enough accuracies the 1/√
ε term dominates, this difference is minimal.

K ≥

(9)

ε

(cid:16)

κ + σrad/µ
√

(cid:17)

log (cid:0) 2r0

(cid:1) .

237

238

239

240

6 FedRR: application of ProxRR to federated learning

241

242

Let us consider now the problem of minimizing the average of N = (cid:80)M
stored on M devices, which have N1, . . . , NM samples correspondingly,

m=1 Nm functions that are

min
x∈Rd

F (x) + R(x),

F (x) = 1
N

(cid:80)M

m=1Fm(x),

Fm(x) = (cid:80)Nm

j=1fmj(x).

(10)

6

0 ∈ Rd, number of epochs T

for m = 1, . . . , M locally in parallel do

Algorithm 3 Federated Random Reshufﬂing (FedRR)
Require: Stepsize γ > 0, initial vector x0 = x0
1: for epochs t = 0, 1, . . . , T − 1 do
2:
3:
4:
5:
6:
7:
8:

xi+1
t,m = xi
t,m = xNm
xn
t,m
(cid:80)M
xt+1 = 1
M

t,m − γ∇fπi,m (xi

m=1 xn

t,m = xt

t,m)

t,m

x0
Sample permutation π0,m, π1,m, . . . , πNm−1,m of {1, 2, . . . , Nm}
for i = 0, 1, . . . , Nm − 1 do

243

244

245

246

247

248

249

250

For example, fmj(x) can be the loss associated with a single sample (Xmj, ymj), where pairs
(Xmj, ymj) follow a distribution Dm that is speciﬁc to device m. An important instance of such for-
mulation is federated learning, where M devices train a shared model by communicating periodically
with a server. We normalize the objective in (10) by N as this is the total number of functions after
we expand each Fm into a sum. We denote the solution of (10) by x∗.

Extending the space. To rewrite the problem as an instance of (1), we are going to consider a bigger
product space, which is sometimes used in distributed optimization [Bianchi et al., 2015]. Let us
deﬁne n := max{N1, . . . , Nm} and introduce ψC, the consensus constraint, deﬁned via

ψC(x1, . . . , xM ) :=

(cid:26)0,

x1 = · · · = xM

+∞, otherwise

.

251

252

By introducing dummy variables x1, . . . , xM and adding the constraint x1 = · · · = xM , we arrive at
the intermediate problem

min
x1,...,xM ∈Rp

1
N

(cid:80)M

m=1 Fm(xm) + (R + ψC)(x1, . . . , xM ),

where R + ψC is deﬁned, with a slight abuse of notation, as (R + ψC)(x1, . . . , xM ) = R(x1) if
x1 = · · · = xM , and (R + ψC)(x1, . . . , xM ) = +∞ otherwise.

Since we have replaced R with a more complicated regularizer R + ψC, we need to understand how
to compute the proximal operator of the latter. We show (Lemma 7 in the supplementary) that the
proximal operator of (R + ψC) is merely the projection onto {(x1, . . . , xM ) | x1 = · · · = xM }
followed by the proximal operator of R with a smaller stepsize.

Reformulation. To have n functions in every Fm, we write Fm as a sum with extra n − Nm zero
functions, fmj(x) ≡ 0 for any j > Nm, so that Fm(xm) = (cid:80)n
j=1 fmj(xm) +
(cid:80)n
j=Nm+1 0. We can now stick the vectors together into x = (x1, . . . , xM ) ∈ RM ·d and multiply

j=1 fmj(xm) = (cid:80)Nm

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

the objective by N

n , which gives the following reformulation:
min
x∈RM ·d

i=1fi(x) + ψ(x),

(cid:80)n

1
n

(11)

263

where ψ(x) := N

n (R + ψC) and

fi(x) = fi(x1, . . . , xM ) :=

M
(cid:88)

fmi(xm).

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

In other words, function fi(x) includes i-th data sample from each device and contains at most
one loss from every device, while Fm(x) combines all data losses on device m. Note that the
solution of (11) is x∗ := (x(cid:62)
∗ )(cid:62) and the gradient of the extended function fi(x) is given
by ∇fi(x) = (∇f1i(x1)(cid:62), · · · , ∇fM i(xM )(cid:62))(cid:62). Therefore, a stochastic gradient step that uses
∇fi(x) corresponds to updating all local models with the gradient of i-th data sample, without any
communication.

∗ , . . . , x(cid:62)

m=1

Algorithm 1 for this speciﬁc problem can be written in terms of x1, . . . , xM , which results in
Algorithm 3. Note that since fmi(xi) depends only on xi, computing its gradient does not require
communication. Only once the local epochs are ﬁnished, the vectors are averaged as the result of
projecting onto the set {(x1, . . . , xM ) | x1 = · · · = xM }.

Reformulation properties. To analyze FedRR, the only thing that we need to do is understand the
properties of the reformulation (11) and then apply Theorem 2 or Theorem 3. The following lemma
gives us the smoothness and strong convexity properties of (11).

7

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

313

Lemma 1. Let function fmi be Li-smooth and µ-strongly convex for every m. Then, fi from
reformulation (11) is Li-smooth and µ-strongly convex.

The previous lemma shows that the conditioning of the reformulation is κ = Lmax
just as we
would expect. Moreover, it implies that the requirement on the stepsize remains exactly the same:
γ ≤ 1/Lmax. What remains unknown is the value of σ2
rad, which plays a key role in the convergence
bounds for ProxRR and ProxSO. To ﬁnd an upper bound on σ2
m,∗ := 1
σ2
Nm

rad, let us deﬁne
∇Fm(x∗)(cid:13)
2
(cid:13)

(cid:13)
(cid:13)∇fmj(x∗) − 1
Nm

(cid:80)n

j=1

µ

,

M
(cid:88)

which is the variance of local gradients on device m. This quantity characterizes the convergence rate
of local SGD [Yuan et al., 2020], so we should expect it to appear in our bounds too. The next lemma
explains how to use it to upper bound σ2
Lemma 2. The shufﬂing radius σ2

rad of the reformulation (11) is upper bounded by

rad.

(cid:16)

(cid:17)
.

σ2
rad ≤ Lmax ·

(cid:107)∇Fm(x∗)(cid:107)2 +

n
4
rad depends on the sum of local variances (cid:80)M

σ2
m,∗

m=1
The lemma shows that the upper bound on σ2
m,∗ as
well as on the local gradient norms (cid:80)M
m=1 (cid:107)∇Fm(x∗)(cid:107)2. Both of these sums appear in the existing
literature on convergence of Local GD/SGD [Khaled et al., 2019, Woodworth et al., 2020, Yuan et al.,
2020]. We are now ready to present formal convergence results. For simplicity, we will consider
heterogeneous and homogeneous cases separately and assume that N1 = · · · = NM = n. To further
illustrate generality of our results, we will present the heterogeneous assuming strong convexity R
and the homogeneous under strong convexity of functions fmi.

m=1 σ2

Heterogeneous data. In the case when the data are heterogeneous, we provide the ﬁrst local RR
method. We can apply either Theorem 2 or Theorem 3, but for brevity, we give only the corollary
obtained from Theorem 3.
Theorem 4. Assume that functions fmi are convex and Li-smooth for each m and i.
µ-strongly convex and γ ≤ 1/Lmax, then we have for the iterates produced by Algorithm 3

If R is

E

(cid:107)xT − x∗(cid:107)2(cid:105)
(cid:104)

≤ (1 + 2γµn)−T (cid:107)x0 − x∗(cid:107)2 + γ2Lmax

M µ

(cid:16)

(cid:80)M

m=1

(cid:107)∇Fm(x∗)(cid:107)2 + N

4M σ2

m,∗

(cid:17)

.

For nonconvex analysis, we consider R ≡ 0 and require the following standard assumption.
Assumption 2 (Bounded variance and dissimilarity). There exist constants σ, ζ > 0 such that for
any x ∈ Rd and
(cid:80)n

(cid:80)M

n ∇Fm(x) − ∇F (x)(cid:13)
(cid:13)
2
(cid:13) 1
(cid:13)

≤ ζ 2.

m=1

and

1
M

1
n

i=1

(cid:13)
(cid:13)∇fmi − 1

n ∇Fm(x)(cid:13)
2
(cid:13)
n ∇Fm(x) = 1
Nm
l=1 ∇Fl(x) is the full gradient on all data.

≤ σ2

(cid:80)M

Note that above 1
1
N
Theorem 5 (Nonconvex convergence). Let Assumptions 1 and 2 be satisﬁed, and R ≡ 0 (no prox).
Then, the communication complexity to achieve E

∇Fm(x) is the gradient of a local dataset and ∇F (x) =

≤ ε2 is

(cid:104)

(cid:16)(cid:16) 1

T = O

ε2 + σ√

nε3 + ζ

ε3

(cid:107)∇F (xT )(cid:107)2(cid:105)
(cid:17)

(F (x0) − F∗)

.

(cid:17)

Notice that by replicating the data locally on each device and thereby increasing the value of n
without changing the objective, we can improve the second term in the communication complexity.
In particular, if the data are not too dissimilar (σ (cid:29) ζ) and ε is small ( 1
ε2 ), the second term in
the complexity dominates, and it helps to have more local steps. However, if the data are less similar,
the nodes have to communicate more frequently to get more information about other objectives.

ε3 (cid:29) 1

Homogeneous data. For simplicity, in the homogeneous (i.e., i.i.d.) data case we provide guarantees
without the proximal operator. Since then we have F1(x) = · · · = FM (x), for any m it holds
∇Fm(x∗) = 0, and thus σ2

(cid:80)n

j=1 (cid:107)∇fmj(x∗)(cid:107)2. The full variance is then given by

m,∗ = 1
n
i=1 (cid:107)∇fmi(x∗)(cid:107)2 = N
m,∗ = 1
n
m=1 (cid:107)∇fmi(x∗)(cid:107)2 is the variance of the gradients over all data.

∗ = M σ2
∗,

n σ2

(cid:80)M

(cid:80)n

m=1

(cid:80)M

m=1 σ2
(cid:80)M

(cid:80)n

i=1

314

where σ2

∗ := 1
N

8

Figure 1: Experimental results for problem (12). The ﬁrst two plots show with average and conﬁdence intervals
estimated on 20 random seeds and clearly demonstrate that one can save a lot of proximal operator computations
with our method. The right plot shows the best/worst convergence of ProxSO over 20,000 sampled permutations.

Figure 2: FedRR vs Local-SGD and Scaffold: i.i.d. data (left) and heterogeneous data (middle and right). We
set λ1 = 0 and estimate the averages and standard deviations by running 10 random seeds for each method.

Theorem 6. Let R(x) ≡ 0 (no prox) and the data be i.i.d., that is ∇Fm(x∗) = 0 for any m, where
(cid:80)M
x∗ is the solution of (10). Let σ2
m=1 (cid:107)∇fmi(x∗)(cid:107)2. If each fmj is Lmax-smooth
and µ-strongly convex, then the iterates of Algorithm 3 satisfy

∗ := 1
N

(cid:80)n

i=1

E (cid:2)(cid:107)xT − x∗(cid:107)2(cid:3) ≤ (1 − γµ)nT (cid:107)x0 − x∗(cid:107)2 + γ2LmaxN σ2

∗

M µ

.

The most important part of this result is that the last term in Theorem 6 has a factor of M in the
denominator, meaning that the convergence bound improves with the number of devices involved.

7 Experiments1

ProxRR vs SGD. In Figure 1, we look at the logistic regression loss with the elastic net regularization,

(cid:80)N

1
N

2 (cid:107)x(cid:107)2,

i=1 fi(x) + λ1(cid:107)x(cid:107)1 + λ2

(12)
where each fi : Rd → R is deﬁned as fi(x) := −(cid:0)bi log (cid:0)h(a(cid:62)
i x)(cid:1)(cid:1),
and where (ai, bi) ∈ Rd × {0, 1}, i = 1, . . . , N are the data samples, h : t → 1/(1 + e−t) is the
sigmoid function, and λ1, λ2 ≥ 0 are parameters. We set minibatch sizes to 32 for all methods and
use theoretical stepsizes, without any tuning. We denote the heuristic version of RR that performs
proximal operator step after each iteration as ‘RR (iteration prox)’. From the experiments, we can see
that all methods behave more or less the same way. However, the algorithm that we propose needs
only a small fraction of proximal operator evaluations, which gives it a huge advantage whenever the
operator takes more time to compute than stochastic gradients.

i x)(cid:1) + (1 − bi) log (cid:0)1 − h(a(cid:62)

FedRR vs Local SGD and Scaffold. We also compare the performance of FedRR, Local SGD and
Scaffold Karimireddy et al. [2020] on homogeneous (i.e., i.i.d.) and heterogeneous data. Since Local
SGD and Scaffold require smaller stepsizes to converge, they are signiﬁcantly slower in the i.i.d.
regime, as can be seen in Figure 2. FedRR, however, does not need small initial stepsize and very
quickly converges to a noisy neighborhood of the solution. We obtain heterogeneous regime by
sorting data with respect to the labels and mixing the sorted dataset with the unsorted one. In this
scenario, we also use the same small stepsize for every method to address the data heterogeneity.
Clearly, Scaffold is the best in terms of functional values because it does variance reduction with
respect to the data. Extending FedRR in the same way might be useful too, but this goes beyond the
scope of our paper and we leave it for future work. We also note that in terms of distances from the
optimum, FedRR still performs much better than Local SGD and Scaffold.

1Our code is provided in the supplementary. More experimental details are in the appendix.

9

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

020040060080010001200Data passes10-510-410-310-210-1P(x)¡P¤SGDRR (iteration prox)RR (epoch prox)020000400006000080000Prox steps10-810-610-410-2100P(x)¡P¤SGDRR (iteration prox)RR (epoch prox)0.00.51.01.52.0Data passes0.00050.0010.002P(x)¡P¤AverageWorst shuffleBest shuffle02004006008001000Communication rounds10-610-510-410-310-210-1100f(x)¡f¤Local SGDScaffoldFedRR010000200003000040000Communication rounds10-310-210-1f(x)¡f¤Local SGDScaffoldFedRR010000200003000040000Communication rounds124102040kx¡x¤k2SGDScaffoldFedRR342

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

References

Kwangjun Ahn, Chulhee Yun, and Suvrit Sra. SGD with shufﬂing: optimal rates without component
convexity and large epoch requirements. arXiv preprint arXiv:2006.06946. Neural Information
Processing Systems (NeurIPS) 2020, 2020. (Cited on pages 2, 4, and 31)

Amir Beck. First-Order Methods in Optimization. Society for Industrial and Applied Mathematics,

Philadelphia, PA, 2017. doi: 10.1137/1.9781611974997. (Cited on page 5)

Pascal Bianchi, Walid Hachem, and Franck Iutzeler. A coordinate descent primal-dual algorithm and
application to distributed asynchronous optimization. IEEE Transactions on Automatic Control, 61
(10):2947–2957, 2015. (Cited on page 7)

Antoine Bordes, Léon Bottou, and Patrick Gallinari. SGD-QN: Careful quasi-Newton stochastic

gradient descent. 2009. (Cited on page 2)

Léon Bottou. Curiously fast convergence of some stochastic gradient descent algorithms. Unpublished
open problem offered to the attendance of the SLDS 2009 conference, 2009. URL http://leon.
bottou.org/papers/bottou-slds-open-problem-2009. (Cited on page 2)

Léon Bottou. Stochastic gradient descent tricks. In Neural Networks: Tricks of the Trade, pages

421–436. Springer, 2012. (Cited on page 2)

Kristian Bredies, Karl Kunisch, and Thomas Pock. Total generalized variation. SIAM Journal on

Imaging Sciences, 3(3):492–526, 2010. (Cited on page 2)

Gong Chen and Marc Teboulle. Convergence Analysis of a Proximal-Like Minimization Algorithm
Using Bregman Functions. SIAM Journal on Optimization, 3(3):538–543, 1993. doi: 10.1137/
0803026. (Cited on page 19)

John Duchi and Yoram Singer. Efﬁcient online and batch learning using forward backward splitting.

Journal of Machine Learning Research, 10(Dec):2899–2934, 2009. (Cited on page 3)

Eduard Gorbunov, Filip Hanzely, and Peter Richtárik. A Uniﬁed Theory of SGD: Variance Reduction,
Sampling, Quantization and Coordinate Descent. volume 108 of Proceedings of Machine Learning
Research, pages 680–690, Online, 26–28 Aug 2020. PMLR. (Cited on pages 2, 3, 18, and 34)

Robert M. Gower, Nicolas Loizou, Xun Qian, Alibek Sailanbayev, Egor Shulgin, and Peter Richtárik.
SGD: General Analysis and Improved Rates. In Kamalika Chaudhuri and Ruslan Salakhutdinov,
editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of
Proceedings of Machine Learning Research, pages 5200–5209, Long Beach, California, USA,
09–15 Jun 2019. PMLR. (Cited on page 6)

Robert M. Gower, Peter Richtárik, and Francis Bach. Stochastic quasi-gradient methods: variance
reduction via Jacobian sketching. Mathematical Programming, pages 1–58, 2020. ISSN 0025-5610.
doi: 10.1007/s10107-020-01506-0. (Cited on page 34)

Mert Gürbüzbalaban, Asuman Özda˘glar, and Pablo A. Parrilo. Why random reshufﬂing beats
ISSN 1436-4646. doi:

stochastic gradient descent. Mathematical Programming, Oct 2019.
10.1007/s10107-019-01440-w. (Cited on page 2)

Jeff Haochen and Suvrit Sra. Random Shufﬂing Beats SGD after Finite Epochs. In Kamalika
Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on
Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 2624–2633,
Long Beach, California, USA, 09–15 Jun 2019. PMLR. (Cited on page 2)

Peter Kairouz et al. Advances and open problems in federated learning.

arXiv preprint

arXiv:1912.04977, 2019. (Cited on pages 1 and 4)

Hamed Karimi, Julie Nutini, and Mark Schmidt. Linear Convergence of Gradient and Proximal-
Gradient Methods Under the Polyak-Łojasiewicz Condition. In European Conference on Machine
Learning and Knowledge Discovery in Databases - Volume 9851, ECML PKDD 2016, page
795–811, Berlin, Heidelberg, 2016. Springer-Verlag. (Cited on page 5)

10

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

430

431

432

433

434

435

436

Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian U. Stich, and
Ananda Theertha Suresh. SCAFFOLD: Stochastic controlled averaging for federated learning. In
International Conference on Machine Learning, pages 5132–5143. PMLR, 2020. (Cited on pages 9
and 30)

Ahmed Khaled and Peter Richtárik. Better theory for SGD in the nonconvex world. arXiv Preprint

arXiv:2002.03329, 2020. (Cited on pages 4 and 31)

Ahmed Khaled, Konstantin Mishchenko, and Peter Richtárik. First Analysis of Local GD on

Heterogeneous Data. arXiv preprint arXiv:1909.04715, 2019. (Cited on page 8)

Ahmed Khaled, Konstantin Mishchenko, and Peter Richtárik. Tighter theory for Local SGD on
In International Conference on Artiﬁcial Intelligence and

identical and heterogeneous data.
Statistics, pages 4519–4529. PMLR, 2020. (Cited on page 29)

Jakub Koneˇcný, H. Brendan McMahan, Felix Yu, Peter Richtárik, Ananda Theertha Suresh, and Dave
Bacon. Federated learning: strategies for improving communication efﬁciency. In NIPS Private
Multi-Party Machine Learning Workshop, 2016. (Cited on pages 1 and 4)

Daniel D. Lee and H. Sebastian Seung. Learning the parts of objects by non-negative matrix

factorization. Nature, 401(6755):788–791, 1999. (Cited on page 2)

Stanislaw Lojasiewicz. A topological property of real analytic subsets. Coll. du CNRS, Les équations

aux dérivées partielles, 117:87–89, 1963. (Cited on page 2)

H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas.
Communication-efﬁcient learning of deep networks from decentralized data. In Proceedings of the
20th International Conference on Artiﬁcial Intelligence and Statistics (AISTATS), 2017. (Cited on
pages 1 and 4)

Konstantin Mishchenko, Ahmed Khaled, and Peter Richtárik. Random Reshufﬂing: Simple Analysis
with Vast Improvements. arXiv preprint arXiv:2006.05988. Neural Information Processing Systems
(NeurIPS) 2020, 2020. (Cited on pages 2, 3, 4, 5, 16, 19, 20, 25, and 26)

Dheeraj Nagaraj, Prateek Jain, and Praneeth Netrapalli. SGD without Replacement: Sharper Rates
for General Smooth Convex Functions. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors,
Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings
of Machine Learning Research, pages 4703–4711, Long Beach, California, USA, 09–15 Jun 2019.
PMLR. (Cited on page 2)

Deanna Needell, Nathan Srebro, and Rachel Ward. Stochastic gradient descent, weighted sampling,
and the randomized Kaczmarz algorithm. Mathematical Programming, 155(1):549–573, Jan 2016.
ISSN 1436-4646. doi: 10.1007/s10107-015-0864-7. (Cited on pages 6 and 34)

Neal Parikh and Stephen Boyd. Proximal Algorithms. Foundations and Trends in Optimization, 1(3):
127–239, January 2014. ISSN 2167-3888. doi: 10.1561/2400000003. (Cited on pages 16 and 30)

Andrei Patrascu and Paul Irofti. Stochastic proximal splitting algorithm for composite minimization.

Optimization Letters, pages 1–19, 2021. (Cited on page 5)

Nhan H. Pham, Lam M. Nguyen, Dzung T. Phan, and Quoc Tran-Dinh. ProxSARAH: An efﬁcient
algorithmic framework for stochastic composite nonconvex optimization. Journal of Machine
Learning Research, 21(110):1–48, 2020. (Cited on page 2)

Boris T. Polyak. Gradient methods for minimizing functionals. Zhurnal Vychislitel’noi Matematiki i

Matematicheskoi Fiziki, 3(4):643–653, 1963. (Cited on page 2)

Benjamin Recht and Christopher Ré. Toward a noncommutative arithmetic-geometric mean in-
equality: Conjectures, case-studies, and consequences.
In S. Mannor, N. Srebro, and R. C.
Williamson, editors, Proceedings of the 25th Annual Conference on Learning Theory, volume 23,
page 11.1–11.24, 2012. Edinburgh, Scotland. (Cited on page 2)

Leonid I. Rudin, Stanley Osher, and Emad Fatemi. Nonlinear total variation based noise removal

algorithms. Physica D: nonlinear phenomena, 60(1-4):259–268, 1992. (Cited on page 2)

11

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

Shai Shalev-Shwartz and Shai Ben-David. Understanding machine learning: from theory to algo-

rithms. Cambridge University Press, 2014. (Cited on page 1)

Ohad Shamir. Without-replacement sampling for stochastic gradient methods. In Advances in neural

information processing systems, pages 46–54, 2016. (Cited on page 2)

Fanhua Shang, Licheng Jiao, Kaiwen Zhou, James Cheng, Yan Ren, and Yufei Jin. ASVRG:
Accelerated Proximal SVRG. In Jun Zhu and Ichiro Takeuchi, editors, Proceedings of Machine
Learning Research, volume 95, pages 815–830. PMLR, 14–16 Nov 2018. (Cited on page 2)

Sebastian U. Stich. Uniﬁed Optimal Analysis of the (Stochastic) Gradient Method. arXiv preprint

arXiv:1907.04232, 2019. (Cited on pages 4 and 31)

Ruo-Yu Sun. Optimization for Deep Learning: An Overview. Journal of the Operations Research
Society of China, 8(2):249–294, Jun 2020. ISSN 2194-6698. doi: 10.1007/s40305-020-00309-6.
(Cited on page 31)

Junqi Tang, Karen Egiazarian, Mohammad Golbabaee, and Mike Davies. The practicality of stochastic
optimization in imaging inverse problems. IEEE Transactions on Computational Imaging, 6:1471–
1485, 2020. (Cited on page 34)

Robert Tibshirani. Regression shrinkage and selection via the Lasso. Journal of the Royal Statistical

Society: Series B (Methodological), 58(1):267–288, 1996. (Cited on page 2)

Trang H. Tran, Lam M. Nguyen, and Quoc Tran-Dinh. Shufﬂing gradient-based methods with

momentum. arXiv preprint arXiv:2011.11884, 2020. (Cited on pages 4 and 31)

Blake Woodworth, Kumar Kshitij Patel, and Nathan Srebro. Minibatch vs Local SGD for Hetero-
geneous Distributed Learning. arXiv preprint arXiv:2006.04735. Neural Information Processing
Systems (NeurIPS) 2020, 2020. (Cited on pages 4, 8, and 24)

Honglin Yuan, Manzil Zaheer, and Sashank Reddi. Federated composite optimization. arXiv preprint

arXiv:2011.08474, 2020. (Cited on page 8)

Ming Yuan and Yi Lin. Model selection and estimation in regression with grouped variables. Journal
of the Royal Statistical Society: Series B (Statistical Methodology), 68(1):49–67, 2006. (Cited on
page 2)

Peilin Zhao and Tong Zhang. Stochastic optimization with importance sampling for regularized loss
minimization. In Proceedings of the 32nd International Conference on Machine Learning, PMLR,
volume 37, pages 1–9, 2015. (Cited on page 4)

12

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

Checklist

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reﬂect the paper’s

contributions and scope? [Yes]

(b) Did you describe the limitations of your work? [Yes]
(c) Did you discuss any potential negative societal impacts of your work? [N/A]
(d) Have you read the ethics review guidelines and ensured that your paper conforms to

them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [Yes]
(b) Did you include complete proofs of all theoretical results? [Yes]

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main experi-

mental results (either in the supplemental material or as a URL)? [Yes]

(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they

were chosen)? [Yes]

(c) Did you report error bars (e.g., with respect to the random seed after running experi-

ments multiple times)? [Yes]

(d) Did you include the total amount of compute and the type of resources used (e.g., type

of GPUs, internal cluster, or cloud provider)? [Yes]

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [Yes]
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

13

