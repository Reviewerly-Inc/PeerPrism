ISAAC Newton: Input-based Approximate
Curvature for Newton’s Method

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

We present ISAAC (Input-baSed ApproximAte Curvature), a novel method that
conditions the gradient using selected second-order information and has an asymp-
totically vanishing computational overhead, assuming a batch size smaller than
the number of neurons. We show that it is possible to compute a good conditioner
based on only the input to a respective layer without a substantial computational
overhead. The proposed method allows effective training even in small-batch
stochastic regimes, which makes it competitive to first-order as well as quasi-
Newton methods.

1

Introduction

While second-order optimization methods are traditionally much less explored than first-order
methods in large-scale machine learning (ML) applications due to their memory requirements and
prohibitive computational cost per iteration, they have recently become more popular in ML mainly
due to their fast convergence properties when compared to first-order methods [1]. The expensive
computation of an inverse Hessian (also known as pre-conditioning matrix) in the Newton step has
also been tackled via estimating the curvature from the change in gradients. Loosely speaking, these
algorithms are known as quasi-Newton methods and a comprehensive treatment can be found in
the textbook [2]. In addition, various new approximations to the pre-conditioning matrix have been
proposed in the recent literature [3]–[6]. From a theoretical perspective, second-order optimization
methods are not nearly as well understood as first-order methods. It is an active research direction to
fill this gap [7], [8].

Motivated by the task of training neural networks, and the observation that invoking local curvature
information associated with neural network objective functions can achieve much faster progress
per iteration than standard first-order methods [9]–[11], several methods have been proposed. One
of these methods, that received significant attention, is known as Kronecker-factored Approximate
Curvature (K-FAC) [12], whose main ingredient is a sophisticated approximation to the generalized
Gauss-Newton matrix and the Fisher information matrix quantifying the curvature of the underlying
neural network objective function, which then can be inverted efficiently.

Inspired by the K-FAC approximation and the Tikhonov regularization of the Newton method, we
introduce a novel two parameter regularized Kronecker-factorized Newton update step. The proposed
scheme disentangles the classical Tikhonov regularization and allows us to condition the gradient
using selected second-order information and has an asymptotically vanishing computational overhead.
While this property makes the presented method highly attractive from the computational complexity
perspective, we show that its achieved empirical performance on complicated high-dimensional
Machine Learning problems remains comparable to existing state-of-the-art methods.

The contributions of this paper can be summarized as follows: (i) we propose a novel two parameter
regularized K-FAC approximated Gauss-Newton update step; (ii) we show that asymptotically—as

Submitted to 36th Conference on Neural Information Processing Systems (NeurIPS 2022). Do not distribute.

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

both regularization parameters vanish—our method recovers the classical K-FAC scheme and in
the opposite setting—as both regularization parameters grow—our method asymptotically reduces
to classical gradient descent; (iii) we prove that for an arbitrary pair of regularization parameters,
the proposed update direction is always a direction of decreasing loss; (iv) in the limit, as one
regularization parameter grows, we obtain an efficient and effective conditioning of the gradient with
an asymptotically vanishing overhead; (v) we empirically analyze the presented method and find that
our efficient conditioning method maintains the performance of its more expensive counterpart; (vi)
we demonstrate the effectiveness of the presented method in the setting of small-batch stochastic
regimes and observe that it is competitive to first-order as well as quasi-Newton methods.

2 Preliminaries

In this section, we review aspects of second-order optimization, with a focus on generalized Gauss-
Newton methods. In combination with Kronecker factorization, this leads us to a new regularized
update scheme. We consider the training of an L-layer neural network f (x; θ) defined recursively as

zi ← ai−1W (i)

(pre-activations),

(1)
where a0 = x is the vector of inputs and aL = f (x; θ) is the vector of outputs. Unless noted otherwise,
we assume these vectors to be row vectors (i.e., in R1×n) as this allows for a direct extension to the
(batch) vectorized case (i.e., in Rb×n) introduced later. For any layer i, let W (i) ∈ Rdi−1×di be a
weight matrix and let ϕ be an element-wise nonlinear function. We consider a convex loss function
L(y, y′) that measures the discrepancy between y and y′. The training optimization problem is then
(2)

Ex,y [L(f (x; θ), y)] ,

(activations),

ai ← ϕ(zi)

arg min

θ

where θ = (cid:2)θ(1), . . . , θ(L)(cid:3) with θ(i) = vec(W (i)).
The classical Newton method for solving (2) is expressed as the update rule

θ′ = θ − η H−1

θ ∇θL(f (x; θ), y) ,

(3)
where η > 0 denotes the learning rate and Hθ is the Hessian corresponding to the objective function
in (2). The stability and efficiency of an estimation problem solved via the Newton method can be
improved by adding a Tikhonov regularization term [13] leading to a regularized Newton method
θ′ = θ − η (Hθ + λI)−1∇θL(f (x; θ), y) ,
(4)
where λ > 0 is the so-called Tikhonov regularization parameter. It is well-known [14], [15], that
under the assumption of approximating the model f with its first-order Taylor expansion, the Hessian
corresponds with the so-called generalized Gauss-Newton (GGN) matrix Gθ, and hence (4) can be
expressed as

θ′ = θ − η (Gθ + λI)−1∇θL(f (x; θ), y) .
(5)
A major practical limitation of (5) is the computation of the inverse term. A method that alleviates this
difficulty is known as Kronecker-Factored Approximate Curvature (K-FAC) [12] which approximates
the block-diagonal (i.e., layer-wise) empirical Hessian or GGN matrix. Inspired by K-FAC, there
have been other works discussing approximations of Gθ and its inverse [15]. In the following, we
discuss a popular approach that allows for (moderately) efficient computation.

The generalized Gauss-Newton matrix Gθ is defined as

(6)
where J and H denote the Jacobian and Hessian matrices, respectively. Correspondingly, the diagonal
block of Gθ corresponding to the weights of the ith layer W (i) is

f L(f (x; θ), y) Jθf (x; θ)(cid:3) ,

Gθ = E (cid:2)(Jθf (x; θ))⊤∇2

GW (i)=E(cid:2)(JW (i)f (x; θ))⊤∇2

f L(f (x; θ), y) JW (i) f (x; θ)(cid:3).

According to the backpropagation rule Jθ(i) f (x; θ) = Jzif (x; θ) ai−1, a⊤b = a ⊗ b, and the
mixed-product property, we can rewrite GW (i) as
GW (i)=E

f L(f (x; θ), y))1/2(cid:1)(cid:0)(∇2
= E(cid:2)(¯g⊤ai−1)⊤(¯g⊤ai−1)(cid:3) = E(cid:2)(¯g ⊗ ai−1)⊤(¯g ⊗ ai−1)(cid:3) = E(cid:2)(¯g⊤¯g) ⊗ (a⊤

(cid:104)(cid:0)(Jzi f (x; θ) ai−1)⊤(∇2

f L(f (x; θ), y))1/2 Jzif (x; θ) ai−1

(7)
i−1 ⊗ ai−1)(cid:3), (8)

(cid:1)(cid:105)

74

where

¯g = (Jzif (x; θ))⊤ (∇2

f L(f (x; θ), y))1/2 .

(9)

2

75

76

77

78

79

80

81

82

83

Remark 1 (Monte-Carlo Low-Rank Approximation for ¯g⊤¯g). As ¯g is a matrix of shape m × di
where m is the dimension of the output of f , ¯g is generally expensive to compute. Therefore, [12] use
a low-rank Monte-Carlo approximation to estimate Hf L(f (x; θ), y) and thereby ¯g⊤¯g. For this, we
need to use the distribution underlying the probabilistic model of our loss L (e.g., Gaussian for MSE
loss, or a categorical distribution for cross entropy). Specifically, by sampling from this distribution
pf (x) defined by the network output f (x; θ), we can get an estimator of Hf L(f (x; θ), y) via the
identity

Hf L(f (x; θ), y) = Eˆy∼pf (x)

(cid:2)∇f L(f (x; θ), ˆy)⊤∇f L(f (x; θ), ˆy)(cid:3) .

(10)

An extensive reference for this (as well as alternatives) can be found in Appendix A.2 of Dangel et
al. [15]. The respective rank-1 approximation (denoted by ≜) of Hf L(f (x; θ)) is

Hf L(f (x; θ), y) ≜ ∇f L(f (x; θ), ˆy)⊤∇f L(f (x; θ), ˆy) ,

84

where ˆy ∼ pf (x). Respectively, we can estimate ¯g⊤¯g using this rank-1 approximation with

¯g ≜ (Jzi f (x; θ))⊤ ∇f L(f (x; θ), ˆy) = ∇ziL(f (x; θ), ˆy) .

(11)

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

In analogy to ¯g, we introduce the gradient of training objective with respect to pre-activations zi as
gi = (Jzif (x; θ))⊤ ∇f L(f (x; θ), y) = ∇ziL(f (x; θ), y) .
In other words, for a given layer, let g ∈ R1×di denote the gradient of the loss between an output and
the ground truth and let ¯g ∈ Rm×di denote the derivative of the network f times the square root of
the Hessian of the loss function (which may be approximated according to Remark 1), each of them
with respect to the output zi of the given layer i. Note that ¯g is not equal to g and that they require one
backpropagation pass each (or potentially many for the case of ¯g). This makes computing ¯g costly.

(12)

Applying the K-FAC [12] approximation to (8) the expectation of Kronecker products can be
approximated as the Kronecker product of expectations as

G = E((¯g⊤¯g) ⊗ (a⊤a)) ≈ E(¯g⊤¯g) ⊗ E(a⊤a) ,

(13)

where, for clarity, we drop the index of ai−1 in (8) and denote it with a; similarly we denote GW (i)
as G. While the expectation of Kronecker products is generally not equal to the Kronecker product
of expectations, this K-FAC approximation (13) has been shown to be fairly accurate in practice
and to preserve the “coarse structure” of the GGN matrix [12]. The K-FAC decomposition in (13)
is convenient as the Kronecker product has the favorable property that for two matrices A, B the
identity (A ⊗ B)−1 = A−1 ⊗ B−1 which significantly simplifies the computation of an inverse.
In practice, E(¯g⊤¯g) and E(a⊤a) can be computed by averaging over a batch of size b as

E(¯g⊤¯g) ≃ ¯g¯g¯g⊤¯g¯g¯g/b,

E(a⊤a) ≃ a⊤a/b,

(14)

where we denote batches of g, ¯g and a, as g ∈ Rb×di, ¯g¯g¯g ∈ Rrb×di and a ∈ Rb×di−1 , where our layer
has di−1 inputs, di outputs, b is the batch size, and r is either the number of outputs m or the rank of
an approximation according to Remark 1. Correspondingly, the K-FAC approximation of the GGN
matrix and its inverse are concisely expressed as

G ≈ (¯g¯g¯g⊤¯g¯g¯g) ⊗ (a⊤a)/b2

G−1 ≈ (cid:0)¯g¯g¯g⊤¯g¯g¯g(cid:1)−1

⊗(cid:0)a⊤a(cid:1)−1

· b2 .

(15)

Equipped with the standard terminology and setting, we now introduce the novel, regularized update
step. First, inspired by the K-FAC approximation (13), the Tikhonov regularized Gauss-Newton
method (5) can be approximated by

θ(i)′ = θ(i) − η(¯g¯g¯g⊤¯g¯g¯g/b + λI)−1 ⊗ (a⊤a/b + λI)−1∇θ(i)L(f (x; θ)),
with regularization parameter λ > 0. A key observation, which is motivated by the structure of
the above update, is to disentangle the two occurrences of λ into two independent regularization
parameters λg, λa > 0. By defining the Kronecker-factorized Gauss-Newton update step as

(16)

ζζζ = λgλa(¯g¯g¯g⊤¯g¯g¯g/b + λgI)−1 ⊗ (a⊤a/b + λaI)−1∇θ(i)L(f (x; θ)),

110

we obtain the concise update equation

θ(i)′ = θ(i) − η∗ζζζ.

(17)

(18)

3

111

112

113

114

115

116

117

118

This update (18) is equivalent to update (16) when in the case of η∗ = η
and λ = λg = λa. This
equivalence does not restrict η∗, λg, λa in any way, and changing λg or λa does not mean that we
change our learning rate or step size η∗. Parameterizing ζζζ in (17) with the multiplicative terms λgλa
makes the formulation more convenient for analysis.

λgλa

In this paper, we investigate the theoretical and empirical properties of the iterative update rule (18)
and in particular show how the regularization parameters λg, λa affect the Kronecker-factorized
Gauss-Newton update step ζζζ. When analyzing the Kronecker-factorized Gauss-Newton update step
ζζζ, a particularly useful tool is the vector product identity,

(cid:16)(cid:0)¯g¯g¯g⊤¯g¯g¯g(cid:1)−1

⊗ (cid:0)a⊤a(cid:1)−1(cid:17)

vec(g⊤a) = vec

(cid:16)(cid:0)¯g¯g¯g⊤¯g¯g¯g(cid:1)−1

g⊤a (cid:0)a⊤a(cid:1)−1(cid:17)

,

(19)

119

where the gradient with respect to the weight matrix is g⊤a.

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

149

150

151

3 Theoretical Guarantees

In this section, we investigate the theoretical properties of the Kronecker-factorized Gauss-Newton
update direction ζζζ as defined in (17). We recall that ζζζ introduces a Tikonov regularization, as it is
commonly done in implementations of second order-based methods. Not surprisingly, we show that
by decreasing the regularization parameters λg, λa the update rule (18) collapses (in the limit) to the
classical Gauss-Newton method, and hence in the regime of small λg, λa the variable ζζζ describes the
Gauss-Newton direction. Moreover, by increasing the regularization strength, we converge (in the
limit) to the conventional gradient descent update step.
The key observation is that, as we disentangle the regularization of the two Kronecker factors ¯g¯g¯g⊤¯g¯g¯g
and a⊤a, and consider the setting where only one regularizer is large (λg → ∞ to be precise),
we obtain an update direction that can be computed highly efficiently. We show that this setting
describes an approximated Gauss-Newton update scheme, whose superior numerical performance is
then empirically demonstrated in Section 4.
Theorem 1 (Properties of ζζζ). The K-FAC based update step ζζζ as defined in (17) can be expressed as

(cid:32)

ζζζ =

Im −

(cid:18)

Ib +

1
bλg

¯g¯g¯g⊤

1
bλg

¯g¯g¯g¯g¯g¯g⊤

(cid:19)−1

(cid:33)

(cid:32)

¯g¯g¯g

· g⊤ ·

Ib −

(cid:18)

aa⊤

Ib +

1
bλa

1
bλa

aa⊤

(cid:19)−1 (cid:33)

· a .

(20)

Moreover, ζζζ admits the following asymptotic properties:

(i) In the limit of λg, λa → 0,

1
λgλa

ζζζ is the K-FAC approximation of the Gauss-Newton step, i.e.,

limλg,λa→0

1
λgλa

ζζζ ≈ G−1∇θ(i)L(f (x; θ)), where ≈ denotes the K-FAC approximation (15).

(ii) In the limit of λg, λa → ∞, ζζζ is the gradient, i.e., limλg,λa→∞ ζζζ = ∇θ(i)L(f (x; θ)).
The Proof is deferred to the Supplementary Material.

We want to show that ζζζ is well-defined and points in the correct direction, not only for λg and λa
numerically close to zero because we want to explore the full spectrum of settings for λg and λa.
Thus, we prove that ζζζ is a direction of increasing loss, independent of the choices of λg and λa.
Theorem 2 (Correctness of ζζζ is independent of λg and λa). ζζζ is a direction of increasing loss,
independent of the choices of λg and λa.

Proof. Recall that (λgIm +¯g¯g¯g⊤¯g¯g¯g/b) and (λaIn +a⊤a/b) are positive semi-definite (PSD) matrices by
definition. Their inverses (λgIm + ¯g¯g¯g⊤¯g¯g¯g/b)−1 and (λaIn + a⊤a/b)−1 are therefore also PSD. As the
Kronecker product of PSD matrices is PSD, the conditioning matrix ((λgIm + ¯g¯g¯g⊤¯g¯g¯g/b)−1 ⊗ (λaIn +
a⊤a/b)−1 ≈ G−1) is PSD, and therefore the direction of the update step remains correct.

From our formulation of ζζζ, we can find that, in the limit for λg → ∞, Equation (21) does not depend
on ¯g¯g¯g. This is computationally very beneficial as computing ¯g¯g¯g is costly as it requires one or even
many additional backpropagation passes. In addition, it allows conditioning the gradient update by
multiplying a b × b matrix between g⊤ and a, which can be done very fast.

4

152

153

Theorem 3 (Efficient Update Direction). In the limit of λg → ∞, the update step ζζζ converges to
limλg→∞ ζζζ = ζζζ ∗, where

(cid:32)

ζζζ ∗ = g⊤ ·

Ib −

(cid:18)

Ib +

aa⊤

1
bλa

1
bλa

aa⊤

(cid:19)−1 (cid:33)

· a .

(21)

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

(i) Here, the update direction ζζζ ∗ is based only on the inputs and does not require computing ¯g¯g¯g

(which would require a second backpropagation pass), making it efficient.

(ii) The computational cost of computing the update ζζζ ∗ lies in O(bn2 + b2n + b3), where n is the
number of neurons in each layer. This comprises the conventional cost of computing the gradient
∇ = g⊤x lying in O(bn2), and the overhead of computing ζζζ ∗ instead of ∇ lying in O(b2n + b3).
The overhead is vanishing, assuming n ≫ b. For b > n the complexity lies in O(bn2 + n3).

Proof. We first show the property (21). Note that according to (22), λg · (cid:0)λgIm + ¯g¯g¯g⊤¯g¯g¯g/b(cid:1)−1
verges in the limit of λg → ∞ to Im, and therefore (21) holds.
(i) The statement follows from the fact that the term ¯g¯g¯g does not appear in the equivalent characteriza-
tion (21) of ζζζ ∗.
(ii) We first note that the matrix aa⊤ is of dimension b × b, and can be computed in O(b2n) time.
Next, the matrix

con-

(cid:32)

Ib −

(cid:18)

Ib +

aa⊤

1
bλa

1
bλa

aa⊤

(cid:19)−1(cid:33)

166

is of shape b × b and can be multiplied with a in O(b2n) time.

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

Notably, (21) can be computed with a vanishing computational overhead and with only minor
modifications to the implementation. Specifically, only the g⊤a expression has to be replaced by (21)
in the backpropagation step. As this can be done independently for each layer, this lends itself also to
applying it only to individual layers.

As we see in the experimental section, in many cases in the mini-batch regime (i.e., b < n), the
optimal (or a good) choice for λg actually lies in the limit to ∞. This is a surprising result, leading to
the efficient and effective ζζζ ∗ = ζζζ λg→∞ optimizer.
Remark 2 (Relation between Update Direction ζζζ and ζζζ ∗). When comparing the update direction
ζζζ in (20) without regularization (i.e., λg → 0, λa → 0) with ζζζ ∗ (i.e., λg → ∞) as given in (21), it
can be directly seen that ζζζ ∗ corresponds to a particular pre-conditioning of ζζζ, since ζζζ ∗ = Mζζζ for
M = 1
bλg

¯g¯g¯g⊤¯g¯g¯g.

As the last theoretical property of our proposed update direction ζζζ ∗, we show that in specific networks
ζζζ ∗ coincides with the Gauss-Newton update direction.
Theorem 4 (ζζζ ∗ is Exact for the Last Layer). For the case of linear regression or, more generally, the
last layer of networks, with the mean squared error, ζζζ ∗ is the Gauss-Newton update direction.

Proof. The Hessian matrix of the mean squared error loss is the identity matrix. Correspondingly,
the expectation value of ¯g¯g¯g⊤¯g¯g¯g is I. Thus, ζζζ ∗ = ζζζ.

Remark 3. The direction ζζζ ∗ corresponds to the Gauss-Newton update direction with an approxima-
tion of G that can be expressed as G ≈ E (cid:2)I ⊗ (a⊤a)(cid:3) .
Remark 4 (Extension to the Natural Gradient). In some cases, it might be more desirable to use the
Fisher-based natural gradient instead of the Gauss-Newton method. The difference to this setting is
that in (5) the GGN matrix G is replaced by the empirical Fisher information matrix F.
We note that our theory also applies to F, and that ζζζ ∗ also efficiently approximates the natural
gradient update step F−1∇. The i-th diagonal block of F (Fθ(i) = E(cid:2)(g⊤
i−1 ⊗ ai−1)(cid:3)).
has the same form as a block of the GGN matrix G (Gθ(i) = E(cid:2)(¯g⊤
Thus, we can replace ¯g¯g¯g with g in our theoretical results to obtain their counterparts for F.

i−1 ⊗ ai−1)(cid:3)),

i gi) ⊗ (a⊤

i ¯gi) ⊗ (a⊤

5

Figure 1: Logarithmic training loss (top) and test accuracy (bottom) on the MNIST classification task. The
axes are the regularization parameters λg and λa in logarithmic scale with base 10. Training with a 5-layer
ReLU activated network with 100 (left, a, e), 400 (center, b, c, f, g), and 1 600 (right, d, h) neurons per layer.
The optimizer is SGD except for (c, g) where the optimizer is SGD with momentum. The top-left sector is
ζζζ, the top-right column is ζζζ ∗, and the bottom-right corner is ∇ (gradient descent). For each experiment and
each of the three sectors, we use one learning rate, i.e., ζζζ, ζζζ ∗, ∇ have their own learning rate to make a fair
comparison between the methods; within each sector the learning rate is constant. We can observe that in the
limit of λg → ∞ (i.e., in the limit to the right) the performance remains good, showing the utility of ζζζ ∗.

4 Experiments

In the previous section, we discussed the theoretical properties of the proposed update directions
ζζζ and ζζζ ∗ with the aspect that ζζζ ∗ would actually be “free” to compute in the mini-batch regime. In
this section, we provide empirical evidence that ζζζ ∗ is a good update direction, even in deep learning.
Specifically, we demonstrate that
(E1) ζζζ ∗ achieves similar performance to K-FAC, while being substantially cheaper to compute.
(E2) The performance of our proposed method can be empirically maintained in the mini-batch

regime (n ≫ b).

(E3) ζζζ ∗ may be used for individual layers, while for other layers only the gradient ∇ is used. This

still leads to improved performance.

(E4) ζζζ ∗ also improves the performance for training larger models such as BERT and ResNet.
(E5) The runtime and memory requirements of ζζζ ∗ are comparable to those of gradient descent.

E1: Impact of Regularization Parameters

For (E1), we study the dependence of the model’s performance on the regularization parameters λg
and λa. Here, we train a 5-layer deep neural network on the MNIST classification task [16] with a
batch size of 60 for a total of 40 epochs or 40 000 steps.

The plots in Figure 1 demonstrate that the advantage of training by conditioning with curvature
information can be achieved by considering both layer inputs a and gradients with respect to random
samples ¯g¯g¯g, but also using only layer inputs a. In the plot, we show the performance of ζζζ for different
choices of λg and λa, each in the range from 10−6 to 106. The right column shows ζζζ ∗, i.e., λg = ∞,
for different λa. The bottom-right corner is gradient descent, which corresponds to λg = ∞ and
λa = ∞.

Newton’s method or the general K-FAC approximation corresponds to the area with small λg and λa.
The interesting finding here is that the performance does not suffer by increasing λg toward ∞, i.e.,
from left to right in the plot.

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

210

211

212

213

214

215

216

217

6

-6-5-4-3-2-10+1+2+3+4+5+6log10g-6-5-4-3-2-10+1+2+3+4+5+6log10a(a)-6-5-4-3-2-10+1+2+3+4+5+6log10g(b)-6-5-4-3-2-10+1+2+3+4+5+6log10g(c)-6-5-4-3-2-10+1+2+3+4+5+6log10g(d)1412108642-6-5-4-3-2-10+1+2+3+4+5+6log10g-6-5-4-3-2-10+1+2+3+4+5+6log10a(e)-6-5-4-3-2-10+1+2+3+4+5+6log10g(f)-6-5-4-3-2-10+1+2+3+4+5+6log10g(g)-6-5-4-3-2-10+1+2+3+4+5+6log10g(h)0.9700.9720.9740.9760.9780.9800.9820.9840.986218

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

256

257

258

259

260

261

Figure 2: Training loss of the MNIST auto-encoder trained with gradient descent, K-FAC, ζζζ, and ζζζ ∗. Comparing
the performance per real-time (left) and per number of update steps (right). Runtimes are for a CPU core.

In addition, in Figure 3, we consider the case of regression with an auto-encoder trained with the
MSE loss on MNIST [16] and Fashion-MNIST [17]. Here, we follow the same principle as above
and also find that ζζζ ∗ performs well.

In Figure 7, we compare the loss for dif-
ferent methods. Here, we distinguish
between loss per time (left) and loss
per number of steps (right). We can ob-
serve that, for λ = 0.1, K-FAC, ζζζ, and
ζζζ ∗ are almost identical per update step
(right), while ζζζ ∗ is by a large margin
the fastest, followed by ζζζ, and the con-
ventional K-FAC implementation is the
slowest (left). On the other hand, for
λ = 0.01 we can achieve a faster con-
vergence than with λ = 0.1, but here
only the K-FAC and ζζζ methods are nu-
merically stable, while ζζζ ∗ is unstable in
this case. This means in the regime of
very small λ, ζζζ ∗ is not as robust as K-
FAC and ζζζ, however, it achieves good
performance with small but moderate
λ like λ = 0.1. For λ < 0.01, also
K-FAC and ζζζ become numerically un-
stable in this setting and, in general, we
observed that the smallest valid λ for
K-FAC is 0.01 or 0.001 depending on
model and task. Under consideration
of the runtime, ζζζ ∗ performs best as it is
almost as fast as gradient descent while
performing equivalent to K-FAC and ζζζ.
Specifically, a gradient descent step is
only about 10% faster than ζζζ ∗.

E2: Minibatch Regime

Figure 3: Training an auto-encoder on MNIST (left) and Fashion-
MNIST (right). The model is the same as used by Botev et al. [18],
i.e., it is a ReLU-activated 6-layer fully connected model with
dimensions 784-1000-500- 30-500-1000-784. Displayed is
the logarithmic training loss.

Figure 4: Training a 5-layer ReLU network with 400 neurons per
layer on the MNIST classification task (as in Figure 1) but with
the Adam optimizer [19].

For (E2), in Figure 1, we can see that training
performs well for n ∈ {100, 400, 1 600} neu-
rons per layer at a batch size of only 60. Also, in
all other experiments, we use small batch sizes
of between 8 and 100.

E3: ζζζ ∗ in Individual Layers

In Figure 5, we train the 5-layer fully connected
model with 400 neurons per layer. Here, we
consider the setting that we use ζζζ ∗ in some of
the layers while using the default gradient ∇
in other layers. Specifically, we consider the

Figure 5: Training on the MNIST classification task
using ζζζ ∗ only in selected layers. Runtimes are for CPU.

7

0200040006000800010000training time [s]0.000.010.020.030.040.050.060.07Training lossGradient descent*(a=0.1)K-FAC (a,g=0.1)K-FAC (a,g=0.01) (a,g=0.1) (a,g=0.01)05001000150020002500300035004000Steps0.000.010.020.030.040.050.060.07Training lossGradient descent*(a=0.1)K-FAC (a,g=0.1)K-FAC (a,g=0.01) (a,g=0.1) (a,g=0.01)-6-5-4-3-2-10+1+2+3+4+5+6log10g-6-5-4-3-2-10+1+2+3+4+5+6log10a(a)-6-5-4-3-2-10+1+2+3+4+5+6log10g(b)5.55.04.54.03.53.02.5-6-5-4-3-2-10+1+2+3+4+5+6log10g-6-5-4-3-2-10+1+2+3+4+5+6log10a(a)-6-5-4-3-2-10+1+2+3+4+5+6log10g(b)025050075010001250150017502000training time [s]181614121086420log. training errorGradient descent* for layers 1, 2, 3, 4, 5* for layers 1* for layers 5* for layers 1, 2, 3* for layers 3, 4, 5* for layers 1, 3, 5* for layers 2, 4Table 1: BERT results for fine-tuning pre-trained BERT-Base (B-B) and BERT-Mini (B-M) models on the
COLA, MRPC, and STSB text classification tasks. Larger values are better for all metrics. MCC is the Matthews
correlation. Results averaged over 10 runs.

Method / Setting

CoLA (B-B)

CoLA (B-M)

MRPC (B-B)

STS-B (B-M)

Metric

MCC

MCC

Acc.

F1

Pearson

Spearman

Gradient baseline 54.20 ± 7.56 21.08 ± 2.88 82.52 ± 1.22 87.88 ± 0.74 76.98 ± 1.10 76.88 ± 0.79
ζζζ∗
57.62 ± 1.59 24.67 ± 2.62 83.28 ± 0.89 88.28 ± 0.70 81.09 ± 1.58 80.82 ± 1.57

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

settings, where all, the first, the final, the first three, the final three, the odd numbered, and the
even numbered layers are updated by ζζζ ∗. We observe that all settings with ζζζ ∗ perform better than
plain gradient descent, except for “ζζζ ∗ for layers 3,4,5” which performs approximately equivalent to
gradient descent.

E4: Large-scale Models
BERT To demonstrate the utility of ζζζ ∗ also in large-scale models, we evaluate it for fine-tuning
BERT [20] on three natural language tasks. In Table 1, we summarize the results for the BERT
fine-tuning task. For the “Corpus of Linguistic Acceptability” (CoLA) [21] data set, we fine-tune
both the BERT-Base and the BERT-Mini models and find that we outperform the gradient descent
baseline in both cases. For the “Microsoft Research Paraphrase Corpus” (MRPC) [22] data set, we
fine-tune the BERT-Base model and find that we outperform the baseline both in terms of accuracy
and F1-score. Finally, on the “Semantic Textual Similarity Benchmark” (STS-B) [23] data set, we
fine-tune the BERT-Mini model and achieve higher Pearson and Spearman correlations than the
baseline. While for training with CoLA and MRPC, we were able to use the Adam optimizer [19]
(which is recommended for this task and model) in conjunction with ζζζ ∗ in place of the gradient,
for STS-B Adam did not work well. Therefore, for STS-B, we evaluated it using the SGD with
momentum optimizer. For each method, we performed a grid search over the hyperparameters. We
note that we use a batch size of 8 in all BERT experiments.

ResNet
In addition, we conduct an experiment
where we train the last layer of a ResNet with
ζζζ ∗, while the remainder of the model is up-
dated using the gradient ∇. Here, we train a
ResNet-18 [24] on CIFAR-10 [25] using SGD
with a batch size of 100 in a vanilla setting, i.e.,
without additional tricks employed in by He et
al. [24] and others. Specifically, we use (i) a
constant learning rate for each training (optimal
from (1, 0.3, 0.1, 0.03, 0.01)) and (ii) vanilla
SGD and not momentum-based SGD. The rea-
son behind this is that we want a vanilla experi-
ment and with aspects such as extensively tuning
multiple parameters of learning rate scheduler
would make the evaluation less transparent; how-
ever, therefore, all accuracies are naturally lower than SOTA. In Figure 6, we plot the test accuracy
against time. The results show that the proposed method outperforms vanilla SGD when applied
to the last layer of a ResNet-18. To validate that the learning rate is not the cause for the better
performance, we also plot the neighboring learning rates and find that even with a too small or too
large learning rate ζζζ ∗ outperforms gradient descent with the optimal learning rate.

Figure 6: ResNet-18 trained on CIFAR-10. Runtimes
are for a GPU. Results are averaged over 5 runs.

E5: Runtime and Memory

Finally, we also evaluate the runtime and memory requirements of each method. The runtime
evaluation is displayed in Table 2. We report both CPU and GPU runtime using PyTorch [26] and
(for K-FAC) the backpack library [15]. Note that the CPU runtime is more representative of the
pure computational cost, as for the first rows of the GPU runtime the overhead of calling the GPU
is dominant. When comparing runtimes between the gradient and ζζζ ∗ on the GPU, we can observe
that we have an overhead of around 2.5 s independent of the model size. The overhead for CPU time
is also very small at less than 1% for the largest model, and only 1.3 s for the smallest model. In

8

025050075010001250150017502000training time [s]0.730.740.750.760.770.78Test Acc.Gradient descent, lr=0.1Gradient descent, lr=0.3Gradient descent, lr=1.0* for last layer, lr=0.1* for last layer, lr=0.3* for last layer, lr=1.0308

309

310

311

312

313

314

contrast, the runtime of ζζζ ∗ is around 4 times the runtime of the gradient, and K-FAC has an even
substantially larger runtime. Regarding memory, ζζζ ∗ (contrasting the other approaches) also requires
only a small additional footprint.
Remark 5 (Implementation). The implementation of ζζζ ∗ can be done by replacing the backpropagation
step of a respective layer by (21). As all “ingredients” are already available in popular deep learning
frameworks, it requires only little modification (contrasting K-FAC and ζζζ, which require at least one
additional backpropagation.)

Table 2: Runtimes and memory requirements for different models. Runtime is the training time per epoch on
MNIST at a batch size of 60, i.e., for 1 000 training steps. The K-FAC implementation is from the backpack
library [15]. The GPU is an Nvidia A6000.

Model

CPU time GPU time Memory CPU time

GPU t.

Memory CPU time GPU t.

Memory

CPU t. GPU t. Memory

Gradient

K-FAC

ζζζ

ζζζ∗

2.05 s
5 layers w/ 100 n.
23.74 s
5 layers w/ 400 n.
187.87 s
5 layers w/ 1 600 n.
5 layers w/ 6 400 n. 3439.59 s

1.79 s
1.84 s
1.93 s
8.22 s 691.0 MB

1.0 MB
62.78 s
4.8 MB 218.48 s
51.0 MB 6985.48 s

17.63 s
32.00 s
156.48 s

1.0 MB
4.9 MB
51.4 MB
— 1320.81 s 3155.3 MB 9673 s 31.87 s 1197.8 MB 3451.61 s 10.24 s 692.5 MB

11.5 MB
8.65 s 11.76 s
22.4 MB 38.67 s 12.62 s
212.2 MB 665.80 s 12.53 s

3.34 s
13.62 s
85.8 MB 291.01 s

4.07 s
4.19 s
4.49 s

1.6 MB
7.7 MB

Auto-Encoder

78.61 s

2.20 s

16.2 MB 1207.58 s

74.09 s

70.7 MB 193.25 s 14.19 s

33.8 MB

87.39 s

4.93 s

16.5 MB

We will publish the source code of our implementation. In the appendix, we give a PyTorch [26]
implementation of the proposed method (ζζζ ∗).

5 Related Work

Our methods are related to K-FAC by Martens and Grosse [12]. K-FAC uses the approximation
(13) to approximate the blocks of the Hessian of the empirical risk of neural networks. In most
implementations of K-FAC, the off-diagonal blocks of the Hessian are also set to zero. One of the
main claimed benefits of K-FAC is its speed (compared to stochastic gradient descent) for large-batch
size training. That said, recent empirical work has shown that this advantage of K-FAC disappears
once the additional computational costs of hyperparameter tuning for large batch training is accounted
for. There is a line of work that extends the basic idea of K-FAC to convolutional layers [27]. Botev et
al. [18] further extend these ideas to present KFLR, a Kronecker factored low-rank approximation,
and KFRA, a Kronecker factored recursive approximation of the Gauss-Newton step. Singh and
Alistarh [28] propose WoodFisher, a Woodbury matrix inverse-based estimate of the inverse Hessian,
and apply it to neural network compression. Yao et al. [29] propose AdaHessian, a second-order
optimizer that incorporates the curvature of the loss function via an adaptive estimation of the Hessian.
Frantar et al. [6] propose M-FAC, a matrix-free approximation of the natural gradient through a queue
of the (e.g., 1 000) recent gradients. These works fundamentally differ from our approach in that their
objective is to approximate the Fisher or Gauss-Newton matrix inverse vector products. In contrast,
this work proposes to approximate the Gauss-Newton matrix by only one of its Kronecker factors,
which we find to achieve good performance at a substantial computational speedup and reduction of
memory footprint. For an overview of this area, we refer to Kunstner et al. [30] and Martens [31].
For an overview of the technical aspects of backpropagation of second-order quantities, we refer to
Dangel et al. [15], [32]

Taking a step back, K-FAC is one of many Newton-type methods for training neural networks.
Other prominent examples of such methods include subsampled Newton methods [33], [34] (which
approximate the Hessian by subsampling the terms in the empirical risk function and evaluating the
Hessian of the subsampled terms) and sketched Newton methods [3]–[5] (which approximate the
Hessian by sketching, e.g., by projecting the Hessian to a lower-dimensional space by multiplying it
with a random matrix). The main features that distinguish K-FAC from this group of methods are
K-FAC’s superior empirical performance and K-FAC’s lack of theoretical justification.

6 Conclusion

In this work, we presented ISAAC Newton, a novel approximate curvature method based on layer-
inputs. We demonstrated it to be a special case of the regularization-generalized Gauss-Newton
method and empirically demonstrate its utility. Specifically, our method features an asymptotically
vanishing computational overhead in the mini-batch regime, while achieving competitive empirical
performance on various benchmark problems.

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

9

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

403

404

405

References

[1] N. Agarwal, B. Bullins, and E. Hazan, “Second-order stochastic optimization for machine
learning in linear time,” Journal on Machine Learning Research, vol. 18, no. 1, pp. 4148–4187,
2017.
J. Nocedal and S. J. Wright, Numerical Optimization, 2e. New York, NY, USA: Springer, 2006.
[2]
[3] A. Gonen and S. Shalev-Shwartz, “Faster SGD using sketched conditioning,” arXiv preprint,

arXiv:1506.02649, 2015.

[4] M. Pilanci and M. J. Wainwright, “Newton sketch: A near linear-time optimization algorithm

with linear-quadratic convergence,” SIAM Journal on Optimization, vol. 27, 2017.

[5] M. A. Erdogdu and A. Montanari, “Convergence rates of sub-sampled Newton methods,” in

Proc. Neural Information Processing Systems (NeurIPS), 2015.

[6] E. Frantar, E. Kurtic, and D. Alistarh, “M-FAC: Efficient matrix-free approximations of
second-order information,” in Proc. Neural Information Processing Systems (NeurIPS), 2021.
[7] N. Doikov and Y. Nesterov, “Convex Optimization based on Global Lower Second-order
Models,” in Proc. Neural Information Processing Systems (NeurIPS), Curran Associates, Inc.,
2020.

[8] Y. Nesterov and B. T. Polyak, “Cubic regularization of Newton method and its global perfor-

mance,” Mathematical Programming, vol. 108, 2006.

[9] S. Becker and Y. Lecun, “Improving the convergence of back-propagation learning with

second-order methods,” 1989.

[10] T. Schaul, S. Zhang, and Y. LeCun, “No more pesky learning rates,” in International Conference

on Machine Learning (ICML), 2013.

[11] Y. Ollivier, “Riemannian metrics for neural networks i: Feedforward networks,” Information

[12]

and Inference, vol. 4, pp. 108–153, Jun. 2015.
J. Martens and R. Grosse, “Optimizing neural networks with Kronecker-factored approximate
curvature,” in International Conference on Machine Learning (ICML), 2015.

[13] A. N. Tikhonov and V. Y. Arsenin, Solutions of Ill-posed problems. W.H. Winston, 1977.
[14] P. Chen, “Hessian matrix vs. Gauss—Newton Hessian matrix,” SIAM Journal on Numerical

Analysis, 2011.

[15] F. Dangel, F. Kunstner, and P. Hennig, “Backpack: Packing more into backprop,” in Interna-

tional Conference on Learning Representations, 2020.

[16] Y. LeCun, C. Cortes, and C. Burges, “MNIST Handwritten Digit Database,” ATT Labs, 2010.
[17] H. Xiao, K. Rasul, and R. Vollgraf, “Fashion-MNIST: A novel image dataset for benchmarking

machine learning algorithms,” arXiv, 2017.

[18] A. Botev, H. Ritter, and D. Barber, “Practical Gauss-Newton optimisation for deep learning,”

in International Conference on Machine Learning (ICML), 2017.

[19] D. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in International Confer-

[20]

ence on Learning Representations (ICLR), 2015.
J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional
transformers for language understanding,” in North American Chapter of the Association for
Computational Linguistics: Human Language Technologies (NAACL-HLT), 2018.

[21] A. Warstadt, A. Singh, and S. R. Bowman, “Neural network acceptability judgments,” Trans-

actions of the Association for Computational Linguistics, vol. 7, 2019.

[22] W. B. Dolan and C. Brockett, “Automatically constructing a corpus of sentential paraphrases,”
in Proceedings of the Third International Workshop on Paraphrasing (IWP2005), 2005.
[23] D. Cer, M. Diab, E. Agirre, I. Lopez-Gazpio, and L. Specia, “SemEval-2017 task 1: Semantic
textual similarity multilingual and crosslingual focused evaluation,” in Proceedings of the
11th International Workshop on Semantic Evaluation (SemEval-2017), Vancouver, Canada:
Association for Computational Linguistics, 2017.

[24] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in
Proc. International Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
[25] A. Krizhevsky, V. Nair, and G. Hinton, “Cifar-10 (Canadian Institute for Advanced Research),”

2009.

[26] A. Paszke, S. Gross, F. Massa, et al., “Pytorch: An imperative style, high-performance deep
learning library,” in Proc. Neural Information Processing Systems (NeurIPS), 2019.

10

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

[27] R. Grosse and J. Martens, “A Kronecker-factored approximate Fisher matrix for convolution

layers,” in International Conference on Machine Learning (ICML), 2016.

[28] S. P. Singh and D. Alistarh, “Woodfisher: Efficient second-order approximation for neural

network compression,” in Proc. Neural Information Processing Systems (NeurIPS), 2020.

[29] Z. Yao, A. Gholami, S. Shen, M. Mustafa, K. Keutzer, and M. W. Mahoney, “Adahessian:
An adaptive second order optimizer for machine learning,” in AAAI Conference on Artificial
Intelligence, 2021.

[30] F. Kunstner, L. Balles, and P. Hennig, “Limitations of the empirical Fisher approximation for
natural gradient descent,” in Proc. Neural Information Processing Systems (NeurIPS), 2019.
J. Martens, “New insights and perspectives on the natural gradient method,” Journal of Machine
Learning Research, 2020.

[31]

[32] F. Dangel, S. Harmeling, and P. Hennig, “Modular block-diagonal curvature approximations
for feedforward architectures,” in International Conference on Artificial Intelligence and
Statistics (AISTATS), 2020.

[33] F. Roosta-Khorasani and M. W. Mahoney, “Sub-Sampled Newton Methods I: Globally Con-

vergent Algorithms,” arXiv: 1601.04737, 2016.

[34] P. Xu, J. Yang, F. Roosta, C. R´e, and M. W. Mahoney, “Sub-sampled Newton Methods with
Non-uniform Sampling,” in Proc. Neural Information Processing Systems (NeurIPS), 2016.

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

Checklist

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reflect the paper’s

contributions and scope? [Yes]

(b) Did you describe the limitations of your work? [Yes]
(c) Did you discuss any potential negative societal impacts of your work? [N/A]
(d) Have you read the ethics review guidelines and ensured that your paper conforms to them?

[Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [Yes]
(b) Did you include complete proofs of all theoretical results? [Yes]

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main experimental
results (either in the supplemental material or as a URL)? [Yes] / [No] We include a
Python / PyTorch implementation of the method in the supplementary material. We will
publicly release full source code for the experiments.

(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were

chosen)? [Yes]

(c) Did you report error bars (e.g., with respect to the random seed after running experiments

multiple times)? [Yes]

(d) Did you include the total amount of compute and the type of resources used (e.g., type of

GPUs, internal cluster, or cloud provider)? [Yes]

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [Yes]
(b) Did you mention the license of the assets? [N/A]
(c) Did you include any new assets either in the supplemental material or as a URL? [N/A]
(d) Did you discuss whether and how consent was obtained from people whose data you’re

using/curating? [N/A]

(e) Did you discuss whether the data you are using/curating contains personally identifiable

information or offensive content? [N/A]

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if

applicable? [N/A]

(b) Did you describe any potential participant risks, with links to Institutional Review Board

(IRB) approvals, if applicable? [N/A]

(c) Did you include the estimated hourly wage paid to participants and the total amount spent

on participant compensation? [N/A]

12

461

A PyTorch Implementation

462

463

We display a PyTorch [26] implementation of ISAAC for a fully-connected layer below. Here, we
mark the important part (i.e., the part beyond the boilerplate) with a red rectangle.

import torch

class ISAACLinearFunction(torch.autograd.Function):

@staticmethod
def forward(ctx, input, weight, bias, la, inv_type):

ctx.save_for_backward(input, weight, bias)
ctx.la = la
if inv_type == 'cholesky_inverse':

ctx.inverse = torch.cholesky_inverse

elif inv_type == 'inverse':

ctx.inverse = torch.inverse

else:

raise NotImplementedError(inv_type)

return input @ weight.T + (bias if bias is not None else 0)

@staticmethod
def backward(ctx, grad_output):

input, weight, bias = ctx.saved_tensors
if ctx.needs_input_grad[0]:

grad_0 = grad_output @ weight

else:

grad_0 = None

if ctx.needs_input_grad[1]:

aaT = input @ input.T / grad_output.shape[0]
I_b = torch.eye(aaT.shape[0], device=aaT.device, dtype=aaT.dtype)
aaT_IaaT_inv = aaT @ ctx.inverse(aaT / ctx.la + I_b)
grad_1 = grad_output.T @ (

I_b - 1. / ctx.la * aaT_IaaT_inv

) @ input

else:

grad_1 = None

return (

grad_0,
grad_1,
grad_output.mean(0, keepdim=True) if bias is not None else None,
None, None, None,

)

class ISAACLinear(torch.nn.Linear):

def __init__(self, in_features, out_features,

la, inv_type='inverse', **kwargs):

super(ISAACLinear, self).__init__(

in_features=in_features, out_features=out_features, **kwargs

)
self.la = la
self.inv_type = inv_type

def forward(self, input: torch.Tensor) -> torch.Tensor:

return ISAACLinearFunction.apply(

input, self.weight,

13

self.bias.unsqueeze(0) if self.bias is not None else None,
self.la,
self.inv_type

)

B Implementation Details

Unless noted differently,
for all experiments, we tune the learning rate on a grid of
(1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001). We verified this range to cover the full reasonable range of
learning rates. Specifically, for every single experiment, we made sure that there is no learning rate
outside this range which performs better.

For all language model experiments, we used the respective Huggingface PyTorch implementation.

All other hyperparameter details are given in the main paper.

The code will be made publicly available.

C Additional Proofs

Proof of Theorem 1. We first show, that ζζζ as defined in (17) can be expressed as in (20). Indeed by
using (19), the Woodbury matrix identity and by regularizing the inverses, we can see that

ζζζ = λgλa(¯g¯g¯g⊤¯g¯g¯g/b + λgI)−1 ⊗ (a⊤a/b + λaI)−1g⊤a

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

= λgλa · (cid:0)λgIm + ¯g¯g¯g⊤¯g¯g¯g/b(cid:1)−1
1
λg

= λgλa ·

1
bλg

2 ¯g¯g¯g⊤

Im −

(cid:32)

g⊤a (cid:0)λaIn + a⊤a/b(cid:1)−1
(cid:33)
(cid:18)

(cid:19)−1

Ib +

¯g¯g¯g¯g¯g¯g⊤

¯g¯g¯g

1
bλg

(cid:32)

1
λa

In −

(cid:18)

Ib +

2 a⊤

1
bλa

aa⊤

(cid:19)−1

(cid:33)

a

g⊤a

(cid:32)

1
bλa
(cid:18)

=

Im −

¯g¯g¯g⊤

Ib +

1
bλg

(cid:19)−1

(cid:33)

¯g¯g¯g

· g⊤

1
bλg

¯g¯g¯g¯g¯g¯g⊤

(cid:32)

· a ·

In −

(cid:18)

Ib +

1
bλa

a⊤

1
bλa

aa⊤

(cid:19)−1

(cid:33)

a

(cid:32)

=

Im −

(cid:18)

Ib +

1
bλg

¯g¯g¯g⊤

1
bλg

¯g¯g¯g¯g¯g¯g⊤

(cid:19)−1

(cid:33)

¯g¯g¯g

· g⊤

(cid:32)

·

a −

(cid:18)

aa⊤

Ib +

1
bλa

1
bλa

aa⊤

(cid:19)−1

(cid:33)

(cid:32)

=

Im −

(cid:32)

·

Ib −

1
bλg

1
bλa

(cid:18)

¯g¯g¯g⊤

Ib +

(cid:19)−1

1
bλg

¯g¯g¯g¯g¯g¯g⊤

(cid:18)

aa⊤

Ib +

1
bλa

aa⊤

(cid:19)−1(cid:33)

· a

a

(cid:33)

¯g¯g¯g

· g⊤

475

To show Assertion (i), we note that according to (17)

lim
λg,λa→0

1
λgλa

ζζζ

= lim

λg,λa→0

(¯g¯g¯g⊤¯g¯g¯g/b + λgI)−1 ⊗ (a⊤a/b + λaI)−1g⊤a

= (¯g¯g¯g⊤¯g¯g¯g)−1 ⊗ (a⊤a)−1g⊤a
≈ G−1g⊤a,

14

476

477

478

where the first equality uses the definition of ζζζ in (17). The second equality is due to the continuity of
the matrix inversion and the last approximate equality follows from the K-FAC approximation (15).

To show Assertion (ii), we consider limλg→∞ and limλa→∞ independently, that is

λg · (cid:0)λgIm + ¯g¯g¯g⊤¯g¯g¯g/b(cid:1)−1

lim
λg→∞

(cid:18)

= lim

λg→∞

Im +

(cid:19)−1

¯g¯g¯g⊤¯g¯g¯g

1
bλg

= Im,

λa · (cid:0)λaIn + a⊤a/b(cid:1)−1

lim
λa→∞

(cid:18)

= lim

λa→∞

In +

(cid:19)−1

a⊤a

= In.

1
bλa

lim
λg,λa→∞

λg

(cid:0)λgIm + ¯g¯g¯g⊤¯g¯g¯g/b(cid:1)−1

· g⊤

· a · λa

(cid:0)λaIn + a⊤a/b(cid:1)−1

= Im · g⊤a · In = g⊤a,

479

and

480

This then implies

481

which concludes the proof.

(22)

(23)

(24)

15

482

D Additional Experiments

Figure 7: Training loss of the MNIST auto-encoder trained with gradient descent, K-FAC, ζζζ, ζζζ ∗, as well as SGD
w/ momentum, SGD with a 10× larger batch size (600), K-FAC with a 10× larger batch size (600), and Adam.
Comparing the performance per real-time (left) and per number of epochs (right). We display both the training
loss (top) as well as the test loss (bottom) Runtimes are for a CPU core.

Figure 8: ResNet-18 trained on CIFAR-10 with image augmentation and a cosine learning rate schedule. The
first line (blue) uses the hyperparameters of a public implementation. To ablate the optimizer, two additional
settings are added, specifically, without weight decay and without momentum. Results are averaged over 5 runs
and the standard deviation is indicated with the colored areas.

16

0200040006000800010000training time [s]0.000.010.020.030.040.050.060.07Train lossGradient descent*(a=0.1)K-FAC (a,g=0.1)K-FAC (a,g=0.01) (a,g=0.1) (a,g=0.01)SGD w/ MomentumSGD (bs=600)K-FAC (a,g=0.01) (bs=600)Adam0510152025303540Epochs0.000.010.020.030.040.050.060.07Train lossGradient descent*(a=0.1)K-FAC (a,g=0.1)K-FAC (a,g=0.01) (a,g=0.1) (a,g=0.01)SGD w/ MomentumSGD (bs=600)K-FAC (a,g=0.01) (bs=600)Adam0200040006000800010000training time [s]0.000.010.020.030.040.050.060.07Test lossGradient descent*(a=0.1)K-FAC (a,g=0.1)K-FAC (a,g=0.01) (a,g=0.1) (a,g=0.01)SGD w/ MomentumSGD (bs=600)K-FAC (a,g=0.01) (bs=600)Adam0510152025303540Epochs0.000.010.020.030.040.050.060.07Test lossGradient descent*(a=0.1)K-FAC (a,g=0.1)K-FAC (a,g=0.01) (a,g=0.1) (a,g=0.01)SGD w/ MomentumSGD (bs=600)K-FAC (a,g=0.01) (bs=600)Adam0255075100125150175200Epochs0.800.820.840.860.880.900.92Test Acc.Gradient descent, SGD w/ momentum + weight decay* for last layer, SGD w/ momentum + weight decayGradient descent, SGD w/ momentum* for last layer, SGD w/ momentumGradient descent, SGD* for last layer, SGDFigure 9: Test accuracy for training on the MNIST classification task using ζζζ ∗ only in selected layers. Runtimes
are for CPU.

17

025050075010001250150017502000training time [s]0.9600.9650.9700.9750.9800.9850.990Test accuracyGradient descent* for layers 1, 2, 3, 4, 5* for layers 1* for layers 5* for layers 1, 2, 3* for layers 3, 4, 5* for layers 1, 3, 5* for layers 2, 4