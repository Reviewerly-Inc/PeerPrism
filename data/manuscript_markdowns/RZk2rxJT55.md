Hamiltonian Mechanics of Feature Learning:
Bottleneck Structure in Leaky ResNets

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

We study Leaky ResNets, which interpolate between ResNets ( ˜L = 0) and Fully-
Connected nets ( ˜L → ∞) depending on an ’effective depth’ hyper-parameter ˜L.
In the infinite depth limit, we study ’representation geodesics’ Ap: continuous
paths in representation space (similar to NeuralODEs) from input p = 0 to output
p = 1 that minimize the parameter norm of the network. We give a Lagrangian
and Hamiltonian reformulation, which highlight the importance of two terms: a
kinetic energy which favors small layer derivatives ∂pAp and a potential energy
that favors low-dimensional representations, as measured by the ’Cost of Identity’.
The balance between these two forces offers an intuitive understanding of feature
learning in ResNets. We leverage this intuition to explain the emergence of a
bottleneck structure, as observed in previous work: for large ˜L the potential energy
dominates and leads to a separation of timescales, where the representation jumps
rapidly from the high dimensional inputs to a low-dimensional representation,
move slowly inside the space of low-dimensional representations, before jumping
back to the potentially high-dimensional outputs. Inspired by this phenomenon, we
train with an adaptive layer step-size to adapt to the separation of timescales.

1

Introduction

Feature learning is generally considered to be at the center of the recent successes of deep neural
networks (DNNs), but it also remains one of the least understood aspects of DNN training.

There is a rich history of empirical analysis of the features learned by DNNs, for example the
appearance of local edge detections in CNNs with a striking similarity to the biological visual cortex
[19], feature arithmetic properties of word embeddings [22], similarities between representations
at different layers [18, 20], or properties such as Neural Collapse [24] to name a few. While some
of these phenomenon have been studied theoretically [3, 8, 27], a more general theory of feature
learning in DNNs is still lacking.

For shallow networks, there is now strong evidence that the first weight matrix is able to recognize a
low-dimensional projection of the inputs that determines the output (assuming this structure is present)
[4, 2, 1]. A similar phenomenon appears in linear networks, where the network is biased towards
learning low-rank functions and low-dimensional representations in its hidden layers [13, 21, 29].
But in both cases the learned features are restricted to depend linearly on the inputs, and the feature
learning happens in the very first weight matrix, whereas it has been observed that features increase
in complexity throughout the layers [31].

The linear feature learning ability of shallow networks has inspired a line of work that postulates that
the weight matrices learn to align themselves with the backward gradients and that by optimizing for
this alignment directly, one can achieve similar feature learning abilities even in deep nets [5, 25].

Submitted to 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Do not distribute.

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

For deep nonlinear networks, a theory that has garnered a lot of interest is the Information Bottleneck
[28], which observed amongst other things that the inner representations appear to maximize their
mutual information with the outputs, while minimizing the mutual information with the inputs. A
limitation of this theory is its reliance on the notion of mutual information which has no obvious
definition for empirical distributions, which lead to some criticism [26].

theory that

A recent
is similar to the Information Bottleneck but with a focus on the
dimensionality/rank of the representations and weight matrices rather than the mutual information is
the Bottleneck rank/Bottleneck structure [16, 15, 30]: which describes how, for large depths, most of
the representations will have approximately the same low dimension, which equals the Bottleneck
rank of the task (the minimal dimension that the inputs can be projected to while still allowing
for fitting the outputs). The intuitive explanation for this bias is that a smaller parameter norm is
required to (approximately) represent the identity on low-dimensional representations rather than
high dimensional ones. Some other types of low-rank bias have been observed in recent work [9, 14].

In this paper we will focus on describing the Bottleneck structure in ResNets, and formalize the
notion of ‘cost of identity’ as a driving force for the bias towards low dimensional representation.
The ResNet setup allows us to consider the continuous paths in representation space from input to
output, similar to the NeuralODE [6], and by adding weight decay, we can analyze representation
geodesics, which are paths that minimize parameter norm, as already studied in [23].

1.1 Leaky ResNets

Our goal is to study a variant of the NeuralODE [6, 23] approximation of ResNet with leaky skip
connections and with L2-regularization. The classical NeuralODE describes the continuous evolution
of the activations αp(x) ∈ Rw starting from α0(x) = x at the input layer p = 0 and then follows
∂pαp(x) = Wpσ(αp(x))
for the w × (w + 1) matrices Wp and the nonlinearity σ : Rw → Rw+1 which maps a vector z to
σ(z) = ( [z1]+ . . .
[zw]+ 1 ) , applying the ReLU nonlinearity entrywise and appending a
new entry with value 1. Thanks to the appended 1 we do not need any explicit bias, since the last
column Wp,·w+1 of the weights replaces the bias.

This can be thought of as a continuous version of the traditional ResNet with activations αℓ(x) for
ℓ = 1, . . . , L: αℓ+1(x) = αℓ(x) + Wℓσ(αℓ(x)).

We will focus on Leaky ResNets, a variant of ResNets that interpolate between ResNets and FCNNs,
by tuning the strength of the skip connections leading to the following ODE with parameter ˜L:

∂pαp(x) = − ˜Lαp(x) + Wpσ(αp(x)).
This can be thought of as the continuous version of αℓ+1(x) = (1 − ˜L)αℓ(x) + Wℓσ(αℓ(x)). As we
will see, the parameter ˜L plays a similar role as the depth in a FCNN.
Finally we will be interested describing the paths that minimize a cost with L2-regularization

min
Wp

1
N

N
(cid:88)

i=1

∥f ∗(xi) − α1(xi)∥2 +

λ
2 ˜L

(cid:90) 1

0

∥Wp∥2

F dp.

The scaling of λ
˜L

for the regularization term will be motivated in Section 1.2.

This type of optimization has been studied in [23] without leaky connections, but we will describe in
this paper large ˜L behavior which leads to a so-called Bottleneck structure [16, 15] as a result of a
separation of time scales in p.

1.2 A Few Symmetries

Changing the leakage parameter ˜L is equivalent (up to constants) to changing the integration range
[0, 1] or to scaling the outputs.
Integration range: Consider the weights Wp on the range [0, 1] and leakage parameter ˜L, leading
to activations αp. Then stretching the weights to a new range [0, c], by defining W ′
c Wq/c for

q = 1

2

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

102

103

104

105

106

107

108

q ∈ [0, c], and dividing the leakage parameter by c, stretches the activations α′

q = αp/c:

∂qα′

q(x) = −

˜L
c

α′

q(x) +

1
c

Wq/cσ(α′

q(x)) =

∂pαq/2(x),

1
c
(cid:82) 1
0 ∥Wp∥2 dp.

(cid:13)
2
(cid:13)

(cid:13)
(cid:13)W ′
q

dq = 1
c

and the parameter norm is simply divided by c:(cid:82) c
0
This implies that a path on the range [0, c] with leakage parameter ˜L = 1 is equivalent to a path on
the range [0, 1] with leakage parameter ˜L = c up to a factor of c in front of the parameter weights.
For this reason, instead of modeling different depths as changing the integration range, we will keep
the integration range to [0, 1] for convenience but change the leakage parameter ˜L instead. To get rid
of the factor in front of the integral, we choose a regularization term of the form λ
. From now on, we
˜L
call ˜L the (effective) depth of the network.
Note that this also suggests that in the absence of leakage ( ˜L = 0), changing the range of integration
has no effect on the effective depth, since 2 ˜L = 0 too. Instead, in the absence of leakage, the effective
depth can be increased by scaling the outputs as we now show.

Output scaling: Given a path Wp on the [0, 1] (for simplicity, we assume that there are no bias, i.e.
Wp,·w+1 = 0), then increasing the leakage by a constant ˜L → ˜L + c leads to a scaled down path
p = e−cpαp. Indeed we have α′
α′
p(x) = −( ˜L + c)α′

p(x)) = e−cp (∂pαp(x) − cαp(x)) = ∂p(e−cpαp(x)).

0(x) = α0(x) and

p(x) + Wpσ(α′

∂pα′

Thus a nonleaky ResNet ˜L = 0 with very large outputs α1(x) is equivalent to a leaky ResNet ˜L > 0
with scaled down outputs e− ˜Lα1(x). Such large outputs are common when training on cross-entropy
loss, and other similar losses that are only minimized at infinitely large outputs. When trained on
such losses, it has been shown that the outputs of neural nets will keep on growing during training
[12, 7], suggesting that when training ResNets on such a loss, the effective depth increases during
training (though quite slowly).

1.3 Lagrangian Reformulation

The optimization of Leaky ResNets can be reformulated, leading to a Lagrangian form.

100

101

First observe that the weights Wp at any minimizer can be expressed in terms of the matrix of
activations Ap = αp(X) ∈ Rw×N over the whole training set X ∈ Rw×N (similar to [17]):

Wp = ( ˜LAp + ∂pAp)σ(Ap)+

where (·)+ is the pseudo-inverse.

We therefore consider the equivalent optimization over the activations Ap:

min
Ap:A0=X

1
N

(cid:90) 1

∥f ∗(X) − A1∥2 +

(cid:13)
˜LAp + ∂pAp
(cid:13)
(cid:13)

λ
2 ˜L
Kp
= ∥M σ(Ap)+∥F corresponding to the scalar
p B(cid:3) for Kp = σ(Ap)T σ(Ap) that will play a central role in our
= ∞ if M does not lie in the image of Kp,

(cid:13)
2
(cid:13)
(cid:13)

dp.

0

This is our first encounter with the norm ∥M ∥Kp
= Tr (cid:2)AK +
product ⟨A, B⟩Kp
upcoming analysis. By convention, we say that ∥M ∥Kp
i.e. ImM T ⊈ ImKp.
It can be helpful to decompose this loss along the different neurons

min
Ap:A0=X

w
(cid:88)

i=1

1
N

∥f ∗

i (X) − A1,i∥2 +

λ
2 ˜L

(cid:90) 1

0

(cid:13)
˜LAp,i· + ∂pAp,i·
(cid:13)
(cid:13)

(cid:13)
2
(cid:13)
(cid:13)

Kp

dp,

109

110

111

Leading to a particle flow behavior, where the neurons Ap,i· ∈ RN are the particles. At first glance, it
appears that there is no interaction between the particles, but remember that the norm ∥·∥Kp
depends
on the covariance Kp = (cid:80)w
i=1 σ(Ai·)σ(Ai·)T , leading to a global interaction between the neurons.

3

112

If we assume that ImAT

p ⊂ Imσ(Ap)T , we can decompose the inside of the integral as three terms:

1
2 ˜L

(cid:13)
˜LAp + ∂pAp
(cid:13)
(cid:13)

(cid:13)
2
(cid:13)
(cid:13)

K+
p

=

˜L
2

∥Ap∥2
Kp

+ ˜L ⟨∂pAp, Ap⟩K+

p

+

1
2 ˜L

∥∂pAp∥2
Kp

.

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

p

/ potential energy − ˜L

plays a relatively minor role in our analysis1, so we focus more on

The middle term ⟨∂pAp, Ap⟩K+
the two other terms:
Cost of identity ∥Ap∥2
2 ∥Ap∥2
: This term can be interpreted as a form of
Kp
potential energy, since it only depends on the representation Ap and not its derivative ∂pAp. We call
it the cost of identity (COI), since it is the Frobenius norm of the smallest weight matrix Wp such that
Wpσ(Ap) = Ap. The COI can be interpreted as measuring the dimensionality of the representation,
inspired by the fact if the representations Ap is non-negative (and there is no bias β = 0), then
Ap = σ(Ap) and the COI simply equals the rank ∥Ap∥2
= RankAp (this interpretation is further
Kp
justified in Section 1.4). We follow the convention of defining the potential energy as the negative of
the term that appears in the Lagrangian, so that the Hamiltonian equals the sum of these two energies.

Kp

Kinetic energy 1
: This term measures the size of the representation derivative ∂pAp
2 ˜L
w.r.t. the Kp norm. It favors paths p (cid:55)→ Ap that do not move too fast, especially along directions
where σ(Ap) is small.

∥∂pAp∥2
Kp

This suggests that the local optimal paths must balance two objectives that are sometimes opposed:
the kinetic energy favors going from input representation to output representation in a ‘straight line’
that minimizes the path length, the COI on the other hand favors paths that spends most of the path in
low-dimensional representations that have a low COI. The balance between these two goals shifts
as the depth ˜L grows, and for large depths it becomes optimal for the network to rapidly move to a
representation of smallest possible dimension (not too small that it becomes impossible to map back
to the outputs), remain for most of the layers inside the space of low-dimensional representations,
and finally move rapidly to the output representation; even if this means doing a large ‘detour’ and
having a large kinetic energy. The main goal of this paper is to describe this general behavior.
Note that one could imagine that as ˜L → ∞ it would always be optimal to first go to the minimal
COI representation which is the zero representation Ap = 0, but once the network reaches a zero
representation, it can only learn constant representations afterwards (the matrix Kp = 11T is then
rank 1 and its image is the space of constant vectors). So the network must find a representation that
minimizes the COI under the condition that there is a path from this representation to the outputs.

Remark. While this interpretation and decomposition is a pleasant and helpful intuition, it is rather
difficult to leverage for theoretical proofs directly. The problem is that we will focus on regimes
where the representations Ap and σ(Ap) are approximately low-dimensional (since those are the
representations that locally minimize the COI), leading to an unbounded pseudo-inverse σ(Ap)+.
This is balanced by the fact that ( ˜LAp + ∂pAp) is small along the directions where σ(Ap)+ explodes,
. But the suppression of ( ˜LAp + ∂pAp)
ensuring a finite weight matrix norm
along these bad directions usually comes from cancellations, i.e. ∂pAp ≈ − ˜LAp. In such cases, the
decomposition in three terms of the Lagrangian is ill adapted since all three terms are infinite and

(cid:13)
˜LAp + ∂pAp
(cid:13)
(cid:13)

(cid:13)
2
(cid:13)
(cid:13)

K+
p

cancel each other to yield a finite sum

. One of our goal is to save this intuition

and prove a similar decomposition with stable equivalent to the cost of identity and kinetic energy
where K +

p is replaced by the bounded (Kp + γI)+ for the right choice of γ.

(cid:13)
˜LAp + ∂pAp
(cid:13)
(cid:13)

(cid:13)
2
(cid:13)
(cid:13)

Kp

id it

1In linear networks σ
=
integrable
0 Tr (cid:2)∂pApσ(Ap)+σ(Ap)+T AT
(cid:3) dp = log |A1|+ − log |A0|+, where |·|+ is pseudo-determinant,
(cid:82) 1
the product of the non-zero singular values. Since its integral only depends on the endpoints, it has no impact on
the representation path in between, which is the focus of this paper. In nonlinear networks, we are not able to
discard in such a manner, but we will see that in the rest of analysis the two other terms play a central role, while
the second term plays less role.

can actually be discarded,

since

is

it

p

4

151

1.4 Cost of Identity as a Measure of Dimensionality

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

The cost of identity can be thought of as a measure of dimensionality of the representation. It is
obvious for non-negative representations because ∥Ap∥2
F = RankAp, but in general,
K+
p
it can be shown to upper bound a notion of ‘stable rank’:
Proposition 1. ∥Aσ(A)+∥2

for the nuclear norm ∥A∥∗ = (cid:80)RankA

F ≥ ∥A∥2

= ∥ApAp

si(A).

+∥2

i=1

∗
∥A∥2
F

Proof. We know that ∥σ(A)∥F ≤ ∥A∥F , therefore ∥Aσ(A)+∥2
√
is minimized when B = ∥A∥F√
∥A∥∗

A, yielding the result.

F ≥ min∥B∥F ≤∥A∥F

∥AB+∥2

F which

is upper bounded by RankA, with equality if all non-zero singular values

The stable rank ∥A∥2
∗
∥A∥2
F
of A are equal, and it is lower bound by the more common notion of stable rank ∥A∥2
F
∥A∥2
op
(cid:80) si max si ≥ (cid:80) s2
Note that in contrast to the COI which is a very unstable quantity because of the pseudo-inverse, the
ratio ∥A∥2
is continuous except at A = 0. This also makes it much easier to compute empirically
∗
∥A∥2
F
than the COI itself.

i for the singular values si.

, because

We know that the COI matches the dimension or rank for positive representations, but it turns out that
the local minima of the COI that are stable under the addition of a new neuron are all positive:
Proposition 2. A local minimum of A (cid:55)→ ∥Aσ(A)+∥2
(cid:18) A
0

F is said to be stable if it remains a local
∈ R(w+1)×N . All stable minima are

minimum after concatenating a zero vector A′ =

(cid:19)

non-negative, and satisfy ∥Aσ(A)+∥2

F = RankA.

Proof. The COI of the nearby point

(cid:19)

(cid:18) A
ϵz
(AT A + ϵ2zzT ) (cid:0)(σ(A)T σ(A) + ϵ2σ(z)σ(z)T (cid:1)+(cid:105)
(cid:13)zT σ(A)+(cid:13)
(cid:13)Aσ(A)+(cid:13)
2
2
(cid:13)
(cid:13)

+ ϵ2 (cid:13)

− ϵ2 (cid:13)

(cid:104)

for z ∈ Imσ(A)T equals

Tr
= (cid:13)

(cid:13)σ(z)T σ(A)+σ(A)+T AT (cid:13)
2
(cid:13)

+ O(ϵ4).

170

171

Assume by contradiction that there is a i = 1, . . . , N such that σ(A·i) ̸= A·i, then choosing
z = σ(A)T σ(A·i) we have σ(z) = z and the two ϵ2 terms are negative:

172

which implies that A′ it is not a local minimum.

ϵ2 ∥σ(Ai)∥2 − ϵ2 ∥Ai∥2 < 0,

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

These stable minima will play a significant role in the rest of our analysis, as we will see that for large
˜L the representations Ap of most layers will be close to one such local minimum. Now we are not
able to rule out the existence of non-stable local minima (nor guarantee that they are avoided with
high probability), but one can show that all strict local minima of wide enough networks are stable.
Actually we can show something stronger, starting from any non-stable local minimum there is a
constant loss path that connects it to a saddle:
Proposition 3. If w > N (N + 1) then if ˆA ∈ Rw×N is local minimum of A (cid:55)→ ∥Aσ(A)+∥2
F that is
not non-negative, then there is a continuous path At of constant COI such that A0 = ˆA and A1 is a
saddle.

This could explain why a noisy GD would avoid such negative/non-stable minima, since there is
no ‘barrier’ between the minima and a lower one, one could diffuse along the path described in
Proposition 3 until reaching a saddle and going towards a lower COI minima. But there seems to be
something else that pushes away from such non-negative minima, as in our experiments with full
population GD we have only observed stable/non-negative local minimas.

5

(a) Hamiltonian measures across ˜L

(b) Bottleneck structure

(c) Hamiltonian dynamics

Figure 1: Leaky ResNet structures: We train equidistant networks with a fixed L = 20 over a
range of effective depths ˜L. The true function f ∗ : R30 → R30 is the composition of two random
FCNNs g1, g2 mapping from dim. 30 to 3 to 30. (a) Estimates of the Hamiltonian constants for
networks trained with different ˜L. The Hamiltonian refers to − 2
H which estimates the true rank
˜L
k∗. The COI refers to minp ||Ap||. The trend line follows the median estimate for − 2
H across each
˜L
network’s layers, whereas the error bars signify its minimum and maximum over p ∈ [0, 1]. The
"stable" Hamiltonians utilize the relaxation from Theorem 4. (b) Spectra of the representations Ap
and weights Wp respectively for ˜L = 7. (c) Hamiltonian dynamics of the network in (b).

187

1.5 Hamiltonian Reformulation

188

189

190

191

192

193

194

We can further reformulate the evolution of the optimal representations Ap in terms of a Hamiltonian,
similar to Pontryagin’s maximum principle.
Let us define the backward pass variables Bp = − 1
2 ∥f ∗(X)−A∥2
F ,
which play the role of the ‘momenta’ of Ap in this Hamiltonian interpretation, which follows the
backward differential equation

λ ∂Ap C(A1) for the cost C(A) = 1

B1 = −

1
λ
−∂pBp = ˙σ(Ap) ⊙ (cid:2)W T

∂A1 C(A1) =

p Bp

(f ∗(X) − A1)

2
λN
(cid:3) − ˜LBp.

Now at any critical point, we have that ∂Wp C(A1) + λ
˜L
− ˜L

λ ∂Ap C(A1)σ(Ap)T = ˜LBpσ(Ap)T , leading to joint dynamics for Ap and Bp:

Wp = 0 and thus Wp =

∂pAp = ˜L(Bpσ(Ap)T σ(Ap) − Ap)
p Bp

−∂pBp = ˜L (cid:0) ˙σ(Ap) ⊙ (cid:2)σ(Ap)BT

(cid:3) − Bp

(cid:1) .

195

These are Hamiltonian dynamics ∂pAp = ∂Bp H and −∂pBp = ∂Ap H w.r.t. the Hamiltonian

H(Ap, Bp) =

˜L
2

(cid:13)
(cid:13)Bpσ(Ap)T (cid:13)
2
(cid:13)

− ˜LTr (cid:2)BpAT

p

(cid:3) .

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

The Hamiltonian is a conserved quantity, i.e. it is constant in p. It will play a significant role in
describing a separation of timescales that appears for large depths ˜L. Another significant advantage
of the Hamiltonian reformulation over the Lagrangian approach is the absence of the unstable
pseudo-inverses σ(Ap)+.
Remark. Note that the Lagrangian and Hamiltonian reformulations have already appeared in previous
work [23] for non-leaky ResNets. Our main contributions are the description in the next section of the
Hamiltonian as the network becomes leakier ˜L → ∞, the connection to the cost of identity, and the
appearance of a separation of timescales. These structures are harder to observe in non-leaky ResNets
(though they could in theory still appear since increasing the scale of the outputs is equivalent to
increasing the effective depth ˜L as shown in Section 1.2).

The Lagrangian and Hamiltonian are also very similar to the ones in [10, 11], and the separation of
timescales and rapid jumps that we will describe also bear a strong similarity. Though a difference
with our work is that the norm ∥·∥Kp

depends on Ap and can be degenerate.

6

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

2 Bottleneck Structure in Representation Geodesics

A recent line of work [16, 15] studies the appearance of a so-called Bottleneck structure in large
depth fully-connected networks, where the weight matrices and representations of ‘almost all’ layers
of the layers are approximately low-rank/low-dimensional as the depth grows. This dimension k is
consistent across layers, and can be interpreted as being equal to the so-called Bottleneck rank of the
learned function. This structure has been shown to extend to CNNs in [30], and we will observe a
similar structure in our leaky ResNets, further showcasing its generality.

More generally, our goal is to describe the ‘representation geodesics’ of DNNs:
the paths in
representation space from input to output representation. The advantage of ResNets (leaky or
not) over FCNNs is that these geodesics can be approximated by continuous paths and are described
by differential equations (as described by the Hamiltonian reformulation).

This section provides an approximation of the Hamiltonian that illustrates the separation of timescales
that appears for large depths, with slow layers with low COI/dimension, and fast layers with high
COI/dimension.

223

2.1 Separation of Timescales

224

If ImAT

p ⊂ Imσ(Ap)T , then the Hamiltonian equals the sum of the kinetic and potential energies:

H =

1
2 ˜L

∥∂pAp∥2
Kp

−

˜L
2

∥Ap∥2
Kp

.

225

226

227

228

229

230

231

232

233

(cid:113)

(cid:113)

= ˜L

+ 2
˜L

∥Ap∥2
Kp

∥Ap∥2
Kp

H. On the other hand, ∂pAp will

H which implies that for large ˜L, the derivative
is close to − 2
˜L

This implies that ∥∂pAp∥Kp
∂pAp is only finite at ps where the COI ∥Ap∥2
Kp
+ 2
H > 0 between the COI and the Hamiltonian. This
blow up for all p with a finite gap
˜L
suggests a separation of timescales as ˜L → ∞, with slow dynamics in layers whose COI/dimension
is close to − 2
˜L
p ⊂ Imσ(Ap)T seems to rarely be true in practice, and both kinetic and
But the assumption ImAT
COI appear to be often infinite in practice. But up to a few approximations, the same argument can
be made for stable versions of the kinetic energy/COI:
(cid:13)
(cid:13)
2
(cid:13)B ˜L
(cid:13)
(cid:13)
(cid:13)

H and fast dynamics in the high COI/dimension layers.

Theorem 4. For sequence A ˜L

≤ c < ∞, and any γ > 0, we have

p of geodesics with

p

−

(cid:18) 1
˜L

√

ℓγ, ˜L +

(cid:19)2

γc

≤ −

H − min

p

(cid:13)
(cid:13)A ˜L
(cid:13)

p

2
˜L

(cid:13)
2
(cid:13)
(cid:13)

(Kp+γI)

≤ γc,

234

for the path length ℓγ, ˜L = (cid:82) 1

0

(cid:13)
(cid:13)∂pA ˜L
(cid:13)

p

(cid:13)
(cid:13)
(cid:13)(Kp+γI)

dp. Finally

√

− ˜L

γc ≤ ∥∂pAp∥(Kp+γi) − ˜L

(cid:114)

∥Ap∥2

(Kp+γI) +

H ≤ 2 ˜L

√

γc.

2
˜L

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

p )∥2

op = γ0∥Kp∥2

Note that the size of ∥B ˜L
p ∥2 can vary a lot throughout the layers, we therefore suggest choosing
a p-dependent γ: γp = γ0∥σ(A ˜L
op. There are two motivations for this: first it is
natural to have γ scale with Kp, ; and second, since Wp = ˜LBpσ(Ap)T is of approximately constant
size (thanks to balancedness, see Appendix A.3), we typically have that the size of Bp is inversely
proportional to that of σ(Ap), so that γp∥Bp∥2 should keep roughly the same size for all p.
Theorem 4 shows that for large ˜L (and choosing e.g. γ = ˜L−1), the Hamiltonian is close to the
minimal COI along the path. Second, the norm of the derivative ∥∂pAp∥(Kp+γi) is close to ˜L times
∥Ap∥2
the ‘extra-COI’
(Kq+γI), which describes
the separation of timescales, with slow (∼ 1) dynamics at layers p where the COI is almost optimal
and fast (∼ ˜L) dynamics everywhere the COI is far from optimal.

(Kp+γI) − minq ∥Aq∥2

(Kp+γI) + 2
˜L

∥Ap∥2

H ≈

(cid:113)

(cid:113)

7

(a) Test performance versus depth

(b) Bottleneck structure and adaptivity.

(c) Paths

Figure 2: Discretization: We train networks with a fixed ˜L = 3 over a range of depths L and
definitions of ρℓs. The true function f ∗ : R30 → R30 is the composition of three random ResNets
g1, g2, g3 mapping from dim. 30 to 6 to 3 to 30. (a) Test error as a function of L for different
discretization schemes. (b) Weight spectra across layers for adaptive ρℓ (L = 18), grey vertical lines
represents the steps pℓ (c) 2D projection of the representation paths Ap for L = 18. Observe how
adaptive ρℓs appears to better spread out the steps.

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

271

Assuming a finite length ℓγ, ˜L < ∞, the norm of the derivative must be finite at almost all layers,
meaning that the COI/dimensionality is optimal in almost all layers, with only a countable number
of short high COI/dimension jumps. These jumps typically appear at the beginning and end of the
network, because the input and output dimensionality and COI are (mostly) fixed, so it will typically
be non-optimal, and so there will often be fast regions close to the beginning and end of the network.
We have actually never observed any jump in the middle of the network, though we are not able to
rule them out theoretically.
If we assume that the paths A ˜L
that the representations in the slow layers (‘inside the Bottleneck’) will be non-negative:
Proposition 5. Let A ˜L
p be a uniformly bounded sequence of local minima for increasing ˜L, at
any p0 ∈ (0, 1) such that ∥∂pAp∥ is uniformly bounded in a neighborhood of p0 for all ˜L, then
A∞
p0

p are stable under adding a neuron, then we can additionally guarantee

= lim ˜L A ˜L

is non-negative.

p0

2 k∗.

We therefore know that the optimal COI minq ∥Aq∥2
(Kq+γI) is close to the dimension of the limiting
p0 , i.e. it must be an integer k∗ which we call the Bottleneck rank of the sequence
representations A∞
of minima since it is closely related to the Bottleneck rank introduced in [16]. The Hamiltonian H is
then close to − ˜L
the Hamiltonian (and the stable Hamiltonians Hγ =
Figure 1 illustrates these phenomena:
1
(Kp+γI)) approach the rank k∗ = 3 from below, while the minimal
2 ˜L
COI approaches it from above; The kinetic energy is proportional to the extra COI, and they are both
large towards the beginning and end of the network where the weights Wp are higher dimensional.
We see in Figure 1c that the (stable) Hamiltonian are not exactly constant, but it still varies much less
than its components, the kinetic and potential energies.

(Kp+γI) − ˜L

2 ∥Ap∥2

∥∂pAp∥2

Because of the non-convexity of the loss we are considering, one can imagine that there could exist
distinct sequences of local minima as ˜L → ∞, which could have different rank, depending on what
low-dimension they reach inside their bottleneck. Indeed in our experiments we have seen that the
number of dimensions that are kept inside the bottleneck can vary by 1 or 2, and in FCNN distinct
sequences of depth increasing minima with different ranks have been observed in [15].

272

3 Discretization Scheme

273

274

To use such Leaky ResNets in practice, we need to discretize over the range [0, 1]. For this we
choose a set of layer-steps ρ1, . . . , ρL with (cid:80) ρℓ = 1, and define the activations at the locations

8

275

pℓ = ρ1 + · · · + ρℓ ∈ [0, 1] recursively as

αp0(x) = x
αpℓ(x) = (1 − ρℓ ˜L)αpℓ−1 (x) + ρℓWpℓσ (cid:0)αpℓ−1 (x)(cid:1)

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

313

314

315

316

317

318

319

320

ℓ=1 ρℓ ∥Wpℓ∥2, for the parameters θ =
and the regularized cost L(θ) = C(α1(X)) + λ
2 ˜L
(Wp1, . . . , WpL ). Note that it is best to ensure that ρℓ ˜L remains smaller than 1 so that the prefactor
(1 − ρℓ ˜L) does not become negative, though we will also discuss certain setups where it might be
okay to take larger layer-steps.

(cid:80)L

Now comes the question of how to choose the ρℓs. We consider three options:
Equidistant: The simplest choice is to choose equidistant points ρℓ = 1
L . Note that the condition
ρℓL < 1 then becomes L > ˜L. But this choice might be ill adapted in the presence of a Bottleneck
structure due to the separation of timescales.

Irregular: Since we typically observe that the fast layers appear close to the inputs and outputs with
a slow bottleneck in the middle, one could simply choose the ρℓ to be go from small to large and back
to small as ℓ ranges from 1 to L. This way there are many discretized layers in the fast regions close
to the input and output and not too many layers inside the Bottleneck where the representations are
(cid:12)
changing less. More concretely one can choose ρℓ = 1
(cid:12)) for a ∈ [0, 1), the choice
a = 0 leads to an equidistant mesh, but increasing a will lead to more points close to the inputs and
outputs. To guarantee ρℓ ˜L < 1, we need L > (1 + a 1
Adaptive: But this can be further improved by choosing the ρℓ to guarantee that the distances
∥Aℓ − Aℓ−1∥ /∥Ap∥ are approximately the same for all ℓ (we divide by the size of Ap since
it can vary a lot throughout the layers). Since the rate of change of Ap is proportional to ρℓ
(∥Aℓ − Aℓ−1∥ /∥Ap∥ = ρℓcℓ), it is optimal to choose ρℓ = c−1
for cℓ = ∥Aℓ−Aℓ−1∥/ρℓ∥Ap∥. The
ℓ
(cid:80) c−1
ℓ
update ρℓ ← c−1
i
(cid:80) c−1
i

can be done at every training step or every few training steps.

4 − (cid:12)
L − 1
(cid:12) ℓ

L + a

4 ) ˜L.

L ( 1

2

Note that the condition ρℓ ˜L < 1 might not be necessary inside the bottleneck since we have the
approximation Wpσ(Apℓ−1) ≈ ˜LApℓ−1, canceling out the negative direction. In particular with the
adaptive layer-steps that we propose, a large ρℓ is only possible for layers where cℓ is small, which is
only possible when Wpσ(Apℓ−1) ≈ ˜LApℓ−1 .
Figure 2 illustrates the effect of the choice of ρℓ for different depths L, we see a small but consistent
advantage in the test error when using adaptive or irregular ρℓs. Looking at the resulting Bottleneck
structure, we see that the adaptive ρℓs result in more steps especially in the beginning of the network,
but also at the end. This because the ‘true function’ f ∗ : R30 → R30 we are fitting in these
experiments is of the form f ∗ = g3 ◦ g2 ◦ g1 where the first inner dimension is 6 and the second is 3,
thus resulting in a rank of k∗ = 3. But before reaching this minimal dimension, the network needs to
represent g2 ◦ g1, which requires more layers, and one can almost see that the weight matrices are
roughly 6-dimensional around p = 0.3. The adaptivity to this structure could explain the advantage
in the test error.

4 Conclusion

We have given a description of the representation geodesics Ap of Leaky ResNets. We have identified
an invariant, the Hamiltonian, which is the sum of the kinetic and potential energy, where the kinetic
energy measures the size of the derivative ∂pAp, while the potential energy is inversely proportional
to the cost of identity, which is a measure of dimensionality of the representations. As the effective
depth of the network grows, the potential energy dominates and we observe a separation of timescales.
At layers with minimal dimensionality over the path, the kinetic energy (and thus the derivative ∂pAp)
is finite. Conversely, at layers where the representation is higher-dimensional, the kinetic energy must
scale with ˜L. This leads to a Bottleneck structure, with a short, high-dimensional jump from the input
representation to a low dimensional representation, followed by slow dynamics inside the space of
low-dimensional representations followed by a final high-dimensional jump to the high dimensional
outputs.

9

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

365

366

367

368

References

[1] Emmanuel Abbe, Enric Boix Adsera, and Theodor Misiakiewicz. The merged-staircase property:
a necessary and nearly sufficient condition for sgd learning of sparse functions on two-layer
neural networks. In Conference on Learning Theory, pages 4782–4887. PMLR, 2022.

[2] Emmanuel Abbe, Enric Boix-Adserà, Matthew Stewart Brennan, Guy Bresler, and
Dheeraj Mysore Nagaraj. The staircase property: How hierarchical structure can guide deep
learning. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances
in Neural Information Processing Systems, 2021.

[3] Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, and Andrej Risteski. A latent
variable model approach to pmi-based word embeddings. Transactions of the Association
for Computational Linguistics, 4:385–399, 2016.

[4] Francis Bach. Breaking the curse of dimensionality with convex neural networks. The Journal

of Machine Learning Research, 18(1):629–681, 2017.

[5] Daniel Beaglehole, Adityanarayanan Radhakrishnan, Parthe Pandit, and Mikhail Belkin.
arXiv preprint
feature learning in convolutional neural networks.

Mechanism of
arXiv:2309.00570, 2023.

[6] Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary

differential equations. Advances in neural information processing systems, 31, 2018.

[7] Lénaïc Chizat and Francis Bach. Implicit bias of gradient descent for wide two-layer neural
networks trained with the logistic loss. In Jacob Abernethy and Shivani Agarwal, editors,
Proceedings of Thirty Third Conference on Learning Theory, volume 125 of Proceedings of
Machine Learning Research, pages 1305–1338. PMLR, 09–12 Jul 2020.

[8] Kawin Ethayarajh, David Duvenaud, and Graeme Hirst. Towards understanding linear word

analogies. arXiv preprint arXiv:1810.04882, 2018.

[9] Tomer Galanti, Zachary S Siegel, Aparna Gupte, and Tomaso Poggio. Sgd and weight decay
provably induce a low-rank bias in neural networks. arXiv preprint arXiv:2206.05794, 2022.

[10] Tobias Grafke, Rainer Grauer, T Schäfer, and Eric Vanden-Eijnden. Arclength parametrized
hamilton’s equations for the calculation of instantons. Multiscale Modeling & Simulation,
12(2):566–580, 2014.

[11] Tobias Grafke and Eric Vanden-Eijnden. Numerical computation of rare events via large
deviation theory. Chaos: An Interdisciplinary Journal of Nonlinear Science, 29(6):063118, 06
2019.

[12] Suriya Gunasekar, Jason Lee, Daniel Soudry, and Nathan Srebro. Characterizing implicit bias
in terms of optimization geometry. In Jennifer Dy and Andreas Krause, editors, Proceedings of
the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine
Learning Research, pages 1832–1841. PMLR, 10–15 Jul 2018.

[13] Suriya Gunasekar, Jason D Lee, Daniel Soudry, and Nati Srebro. Implicit bias of gradient
descent on linear convolutional networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman,
N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems,
volume 31. Curran Associates, Inc., 2018.

[14] Florentin Guth, Brice Ménard, Gaspar Rochette, and Stéphane Mallat. A rainbow in deep

network black boxes. arXiv preprint arXiv:2305.18512, 2023.

[15] Arthur Jacot. Bottleneck structure in learned features: Low-dimension vs regularity tradeoff. In
A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in
Neural Information Processing Systems, volume 36, pages 23607–23629. Curran Associates,
Inc., 2023.

[16] Arthur Jacot. Implicit bias of large depth networks: a notion of rank for nonlinear functions. In

The Eleventh International Conference on Learning Representations, 2023.

10

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

[17] Arthur Jacot, Eugene Golikov, Clément Hongler, and Franck Gabriel. Feature learning in
l2-regularized dnns: Attraction/repulsion and sparsity. In Advances in Neural Information
Processing Systems, volume 36, 2022.

[18] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural
network representations revisited. In International Conference on Machine Learning, pages
3519–3529. PMLR, 2019.

[19] A. Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep

convolutional neural networks. Communications of the ACM, 60:84 – 90, 2012.

[20] Jianing Li and Vardan Papyan. Residual alignment: Uncovering the mechanisms of residual

networks. Advances in Neural Information Processing Systems, 36, 2024.

[21] Zhiyuan Li, Yuping Luo, and Kaifeng Lyu. Towards resolving the implicit bias of gradient
descent for matrix factorization: Greedy low-rank learning. In International Conference on
Learning Representations, 2020.

[22] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word

representations in vector space. arXiv preprint arXiv:1301.3781, 2013.

[23] Houman Owhadi. Do ideas have shape? plato’s theory of forms as the continuous limit of

artificial neural networks. arXiv preprint arXiv:2008.03920, 2020.

[24] Vardan Papyan, XY Han, and David L Donoho. Prevalence of neural collapse during the
terminal phase of deep learning training. Proceedings of the National Academy of Sciences,
117(40):24652–24663, 2020.

[25] Adityanarayanan Radhakrishnan, Daniel Beaglehole, Parthe Pandit, and Mikhail Belkin.
Mechanism for feature learning in neural networks and backpropagation-free machine learning
models. Science, 383(6690):1461–1467, 2024.

[26] Andrew Michael Saxe, Yamini Bansal, Joel Dapello, Madhu Advani, Artemy Kolchinsky,
Brendan Daniel Tracey, and David Daniel Cox. On the information bottleneck theory of deep
learning. In International Conference on Learning Representations, 2018.

[27] Peter Súkeník, Marco Mondelli, and Christoph H Lampert. Deep neural collapse is provably
optimal for the deep unconstrained features model. Advances in Neural Information Processing
Systems, 36, 2024.

[28] Naftali Tishby and Noga Zaslavsky. Deep learning and the information bottleneck principle. In

2015 ieee information theory workshop (itw), pages 1–5. IEEE, 2015.

[29] Zihan Wang and Arthur Jacot. Implicit bias of SGD in l2-regularized linear DNNs: One-
In The Twelfth International Conference on Learning

way jumps from high to low rank.
Representations, 2024.

[30] Yuxiao Wen and Arthur Jacot. Which frequencies do cnns need? emergent bottleneck structure

in feature learning. to appear at ICML, 2024.

[31] Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In
Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September
6-12, 2014, Proceedings, Part I 13, pages 818–833. Springer, 2014.

11

408

A Proofs

409

A.1 Cost of Identity

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

Proposition 6 (Proposition 3 in the main.). If w > N (N + 1) then if ˆA ∈ Rw×N is local minimum
of A (cid:55)→ ∥Aσ(A)+∥2
F that is not non-negative, then there is a continuous path At of constant COI
such that A0 = ˆA and A1 is a saddle.

i=1 ai( ˆAi· ˆAT

( ˆK, ˆK σ) belongs

ˆK σ = σ( ˆA)T σ( ˆA).
i· , σ( ˆAi·)σ( ˆAi·)T ) : i = 1, . . . , w

Proof. The local minimum ˆA leads to a pair of N × N covariance matrices ˆK =
ˆAT ˆA and
The pair
to the conical hull
(cid:110)
(cid:111)
( ˆAi· ˆAT
Cone
. Since this cones lies in a N (N + 1)-dimensional
space (the space of pairs of symmetric N × N matrices), we know by Caratheodory’s
there is a conical combination ( ˆK, ˆK σ − β21N ×N ) =
theorem (for convex cones) that
(cid:80)w
i· , σ( ˆAi·)σ( ˆAi·)T ) such that no more than N (N + 1) of the coefficients are non-
zero. We now define At to have lines At,i· = (cid:112)(1 − t) + tai ˆAi·, so that At=0 = ˆA and at t = 1 at
least one line of At=1 is zero (since at least one of the ais is zero). First note that the covariance pairs
i· = (1 − t) ˆK + t ˆK = ˆK
remain constant over the path: Kt = AT
(cid:3) is constant
and similarly K σ
too. Second, since a representation A is non-negative iff the covariances satisfy K = K σ, the
representation path At cannot be non-negative either since it has the same kernel pairs ( ˆK, ˆK σ) with
ˆK ̸= ˆK σ.
Now (the converse of) Proposition 2 tells us that if At=1 is not non-negative and has a zero line, then
it is not a local minimum, which implies that it is a saddle.

t At = (cid:80)w
t = ˆK σ, which implies that the cost ∥Atσ(At)+∥2

i=1((1 − t) + tai) ˆAi· ˆAT

F = Tr (cid:2)KtK σ+

t

428

A.2 Bottleneck

429

430

Theorem 7. For any uniformly bounded sequence A ˜L
and any γ > 0, we have

p of geodesics, i.e.

(cid:13)
(cid:13)A ˜L
(cid:13)

p

(cid:13)
2
(cid:13)
(cid:13)

(cid:13)
(cid:13)B ˜L
(cid:13)

p

(cid:13)
2
(cid:13)
(cid:13)

,

≤ c < ∞,

−

(cid:18) 1
˜L

√

ℓγ, ˜L +

(cid:19)2

γc

≤ −

H − min

p

(cid:13)
(cid:13)A ˜L
(cid:13)

p

2
˜L

(cid:13)
2
(cid:13)
(cid:13)

(Kp+γI)

≤ γc,

431

for the path length ℓγ, ˜L = (cid:82) 1

0

(cid:13)
(cid:13)∂pA ˜L
(cid:13)

p

(cid:13)
(cid:13)
(cid:13)(Kp+γI)

dp. Finally

√

− ˜L

γc ≤ ∥∂pAp∥(Kp+γi) − ˜L

(cid:114)

∥Ap∥2

(Kp+γI) +

H ≤ 2 ˜L

√

γc.

2
˜L

432

Proof. First observe that

(cid:13)
(cid:13)
(cid:13)
(cid:13)

1
˜L

∂pAp + γBp

(cid:13)
2
(cid:13)
(cid:13)
(cid:13)

(Kp+γI)

= ∥Bp(Kp + γ) − Ap∥2

(Kp+γI)

+ γ ∥Bp∥2 − 2Tr (cid:2)BpAT

p

(cid:3) + ∥Ap∥2

(Kp+γI)

= (cid:13)
(cid:13)Bpσ(Ap)T (cid:13)
2
(cid:13)
2
˜L

=

H + γ ∥Bp∥2 + ∥Ap∥2

(Kp+γI)

433

and thus we have

−

2
˜L

H = ∥Ap∥2

(Kp+γI) −

(cid:13)
(cid:13)
(cid:13)
(cid:13)

1
˜L

∂pAp + γBp

(cid:13)
2
(cid:13)
(cid:13)
(cid:13)

(Kp+γI)

+ γ ∥Bp∥2 .

12

(1) The upper bound − 2
˜L

H − minp

(cid:13)
(cid:13)A ˜L
(cid:13)

p

For the lower bound, first observe that

(cid:13)
2
(cid:13)
(cid:13)

(Kp+γI)

434

435

≤ γc then follows from the fact that ∥Bp∥2 ≤ c.

1
˜L

∥∂pAp∥(Kp+γI) ≥

(cid:13)
1
(cid:13)
(cid:13)
˜L
(cid:13)
(cid:114)

∂pAp + γBp

− ∥γBp∥(Kp+γI)

∥Ap∥2

(Kp+γI) +

H + γ ∥Bp∥2 −

√

γc

(cid:13)
(cid:13)
(cid:13)
(cid:13)(Kp+γI)
2
˜L
2
˜L

H −

∥Ap∥2

(Kp+γI) +

√

γc,

(1)

≥

≥

(cid:114)

436

and therefore

(cid:90) 1

0
(cid:114)

1
˜L
(cid:90) 1

0
(cid:114)

1
˜L

ℓγ, ˜L =

≥

≥

∥∂pAp∥(Kp+γI) dp

∥Ap∥2

(Kp+γI) +

H −

√

γcdp

min
p

∥Ap∥2

(Kp+γI) +

H −

√

γc

2
˜L
2
˜L

which implies the lower bound.

(2) The lower bound follows from equation 1. The upper bound follows from

1
˜L

∥∂pAp∥(Kp+γI) ≤

(cid:13)
1
(cid:13)
(cid:13)
˜L
(cid:13)
(cid:114)

≤

≤

≤

(cid:114)

(cid:114)

∂pAp + γBp

+ ∥γBp∥(Kp+γI)

∥Ap∥2

(Kp+γI) +

H + γ ∥Bp∥2 +

√

γc

∥Ap∥2

(Kp+γI) +

√

γ ∥Bp∥ +

√

γc

∥Ap∥2

(Kp+γI) +

H + 2

√

γc.

(cid:13)
(cid:13)
(cid:13)
(cid:13)(Kp+γI)
2
˜L
2
˜L
2
˜L

H +

Proposition 8 (Proposition 5 in the main.). Let A ˜L
p be a uniformly bounded sequence of local minima
for increasing ˜L, at any p0 ∈ (0, 1) such that ∥∂pAp∥ is uniformly bounded in a neighborhood of p0
for all ˜L, then A∞
p0

= lim ˜L A ˜L

is non-negative.

p0

Proof. Given a path Ap with corresponding weight matrices Wp corresponding to a width w, then
(cid:19)
(cid:18) A
0

. Our goal is to show that for sufficiently large

is a path with weight matrix

(cid:18) Wp
0

0
0

(cid:19)

depths, one can under certain assumptions slightly change the weights to obtain a new path with the
same endpoints but a slightly lower loss, thus ensuring that if certain assumptions are not satisfied
then the path cannot be locally optimal.

Let us assume that ∥∂pAp∥ ≤ c1 in a neighborhood of a p0 ∈ (0, 1), and assume by contradiction
that there is an input index i = 1, . . . , N such that Ap0,·i has at least one negative entry, and therefore
∥Ap0,·i∥2 − ∥σ(Ap0,·i)∥2 = c0 > 0 for all ˜L.
We now consider the new weights

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

(cid:18) Wp − ˜Lϵ2t(p)Ap,·iσ(Ap,·i)T
ϵ ˜Lt(p)σ(Ap,·i)

(cid:19)

ϵ ˜Lt(p)Ap,·i
0

452

for t(p) = max{0, 1 − |p−p0|

r

} a triangular function centered in p0 and for an ϵ > 0.

13

453

For ϵ and rsmall enough, the parameter norm will decrease:

(cid:90) 1

0

=

(cid:13)
(cid:13)
(cid:13)
(cid:13)
(cid:90) 1

0

Wp − ˜Lϵ2t(p)Ap,·iσ(Ap,·i)T
ϵ ˜Lt(p)σ(Ap,·i)
(cid:18)

∥Wp∥2 + ˜L2ϵ2t(p)2

−

2
˜L

ϵ ˜Lt(p)Ap,·i
0

(cid:13)
2
(cid:13)
(cid:13)
(cid:13)

dp

AT

p,·iWpσ(Ap,·i) + ∥Ap,·i∥2 + ∥σ(Ap,·i)∥2

(cid:19)

dp.

454

Now since Wpσ(Ap,·i) = ∂pAp,·i + ˜LAp,·i, this simplifies to

∥Wp∥2 + ˜L2ϵ2t(p)2

(cid:18)

(cid:90) 1

0

− ∥Ap,·i∥2 + ∥σ(Ap,·i)∥2 −

(cid:19)

AT

p,·i∂pAp,·i

1
˜L

dp + O(ϵ4).

455

456

457

458

459

460

461

462

By taking r small enough, we can guarantee that − ∥Ap,·i∥2 + ∥σ(Ap,·i)∥2 < − c0
t(p) > 0, and for ˜L large enough we can guarantee that
we can guarantee that the parameter norm will be strictly smaller for ϵ small enough.
(cid:18) Ap
ϵap

We will now show that with these new weights the path becomes approximately

p,·i∂pAp,·i

AT

1
˜L

(cid:12)
(cid:12)
(cid:12)

(cid:12)
(cid:12) is smaller then c0
(cid:12)

2 for all p such that
4 , so that

(cid:19)

where

ap = ˜L

(cid:90) p

0

t(q)Kp,i·e ˜L(q−p)dq.

Note that ap is positive for all p since Kp has only positive entries. Also note that as ˜L → ∞,
ap → t(p)Kp,i· and so that a0 → 0 and a1 → 1.

On one hand, we have the time derivative
(cid:19)

(cid:18) Ap
ϵap

∂p

=

(cid:18) Wpσ(Ap) − ˜LAp
ϵ ˜L (t(p)Kp,i· − ap)

(cid:19)

.

On the other hand the actual derivative as determined by the new weights:
(cid:19)

(cid:18) Wp − ˜Lϵ2t(p)Ap,·iσ(Ap,·i)T
ϵ ˜Lt(p)σ(Ap,·i)

ϵ ˜Lt(p)Ap,·i
0

(cid:19) (cid:18) σ(Ap)
ϵσ(ap)

− ˜L

(cid:19)

(cid:18) Ap
ϵap

=

(cid:18) Wpσ(Ap) − ˜LAp − ˜Lϵ2t(p)2Ap,·iKp,i· + ˜Lϵ2t(p)Ap,·iap
ϵ ˜Lt(p)Kp,i· − ϵ ˜La(p)

(cid:19)

.

463

The only difference is the two terms

− ˜Lϵ2t(p)2Ap,·iKi· + ˜Lϵ2t(p)Ap,·iap = ˜Lϵ2t(p)Ap,·i (t(p)Ki· − ap) .
One can guarantee with a Grönwall type of argument that the representation path resulting from the

464

465

new weights must be very close to the path

(cid:18) Ap
ϵap

(cid:19)

.

466

A.3 Balancedness

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

This paper will heavily focus on the Hamiltonian Hp that is constant throughout the layers p ∈ [0, 1],
and how it can be interpreted. Note that the Hamiltonian we introduce is distinct from an already
known invariant, which arises as the result of so-called balancedness, which we introduce now.

Though this balancedness also appears in ResNets, it is easiest to understand in fullyconnected
networks. First observe that for any neuron i ∈ 1, . . . , w at a layer ℓ one can multiply the incoming
weights (Wℓ,i·, bℓ,i) by a scalar α and divide the outcoming weights Wℓ+1,·i by the same scalar
α without changing the subsequent layers. One can easily see that the scaling that minimize the
contribution to the parameter norm is such that the norm of incoming weights equals the norm
of the outcoming weights ∥Wℓ,i·∥2 + ∥bℓ,i∥2 = ∥Wℓ+1,·i∥2. Summing over the is we obtain
∥Wℓ∥2
F , which means that the
norm of the weights is increasing throughout the layers, and in the absence of bias, it is even constant.

F + ∥bℓ∥2 = ∥Wℓ+1∥2

F and thus ∥Wℓ∥2

F = ∥W1∥2

F + (cid:80)ℓ−1

k=1 ∥bk∥2

Leaky ResNet exhibit the same symmetry:

14

479

480

481

482

483

484

Proposition 9. At any critical Wp, we have ∥Wp∥2 = ∥W0∥2 + ˜L (cid:82) p

0 ∥Wp,·w+1∥2 dq.

Proof. This proofs handles the bias Wp,·(w+1) differently to the rest of the weights Wp,·(1:w), to
simplify notations, we write Vp = Wp,·(1:w) and bp = Wp,·(w+1) for the bias.
First let us show that choosing the weight matrices ˜Vq = r′(q)Vr(q) and bias ˜bq = r′(q)e ˜L(r(q)−q)br(q)
leads to the path ˜Aq = e ˜L(r(q)−q)Ar(q). Indeed the path ˜Aq = e ˜L(r(q)−q)Ar(q) has the right value
when p = 0 and it then satisfies the right differential equation:

∂q ˜Aq = ˜L(r′(q) − 1) ˜Aq + e ˜L(r(q)−q)r′(q)∂pAr(q)
= ˜L(r′(q) − 1) ˜Aq + e ˜L(r(q)−q)r′(q)

(cid:16)

− ˜LAr(q) + Vr(q)σ(Ar(q)) + br(q)

(cid:17)

= − ˜L ˜Zq + r′(q)Ar(q)σ
(cid:17)

= ˜Vqσ

(cid:16) ˜Aq

+ ˜bq − ˜L ˜Aq

(cid:17)

(cid:16) ˜Zq

+ e ˜L(r(q)−q)r′(q)br(q)

485

The optimal reparametrization r(q) is therefore the one that minimizes

(cid:90) 1

0

(cid:13)
˜Wq
(cid:13)
(cid:13)

(cid:13)
2
(cid:13)
(cid:13)

+

(cid:13)
˜bq
(cid:13)
(cid:13)

(cid:13)
2
(cid:13)
(cid:13)

dq =

(cid:90) 1

0

r′(q)2 (cid:16)(cid:13)

(cid:13)Wr(q)

(cid:13)
2
(cid:13)

+ e2 ˜L(r(q)−q) (cid:13)

(cid:13)br(q)

2(cid:17)

(cid:13)
(cid:13)

dq

486

For the identity reparametrization r(q) = q to be optimal, we need

(cid:90) 1

0

2dr′(p)

(cid:16)

∥Wp∥2 + ∥bp∥2(cid:17)

+ 2 ˜Ldr(p) ∥bp∥2 dp = 0

487

for all dr(q) with dr(0) = dr(1) = 0. Since

(cid:90) 1

0

dr′(p)

∥Wp∥2 + ∥bp∥2(cid:17)
(cid:16)

dp = −

(cid:90) 1

0

dr(p)∂p

(cid:16)

∥Wp∥2 + ∥bp∥2(cid:17)

dq,

488

we need

489

and thus for all p

(cid:90) 1

0

dr(p)

(cid:104)

−∂p

(cid:16)

∥Wp∥2 + ∥bp∥2(cid:17)

+ ˜L ∥bp∥2(cid:105)

dp = 0

(cid:16)

∥Wp∥2 + ∥bp∥2(cid:17)

∂p

= ˜L ∥bp∥2 .

490

Integrating, we obtain as needed

∥Wp∥2 + ∥bp∥2 = ∥W0∥2 + ∥b0∥2 + ˜L

(cid:90) p

0

∥bq∥2 dq.

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

B Experimental Setup

Our experiments make use of synthetic data to train leaky ResNets so that the Bottleneck rank k∗ is
known for our experiments. The synthetic data is generated by teacher networks for a given true rank
k∗. To construct a bottleneck, the teacher network is a composition of networks for which the the
inner-dimension is k∗. Our experiments used an input and output dimension of 30, and a bottleneck
of k∗ = 3. For data, we sampled a thousand data points for training, and another thousand for testing
which are collectively augmented by demeaning and normalization.

To train the leaky ResNets, it is important for them to be wide, usually wider than the input or output
dimension, we opted for a width of 100. However, the width of the representation must be constant
to implement leaky residual connections, so we introduce a single linear mapping at the start, and
another at the end, of the forward pass to project the representations into a higher dimension for the
paths. These linear mappings can be either learned or fixed.

15

Figure 3: Various properties of the Hamiltonian dynamics of Leaky ResNets which remain bounded

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

523

To achieve a tight convergence in training, we train primarily using Adam using Mean Squared Error
as a loss function, and our custom weight decay function. After training on Adam (we found 5000
epochs to work well), we then train briefly (usually 1000 epochs) using SGD with a smaller learning
rate to tighten the convergence.

The bottleneck structure of a trained network, as seen in Figure 1b and 2b, can be observed in the
spectra of both the representations Ap and the weight matrices Wp at each layer. As long as the
training is not over-regularized (λ too large) then the spectra reveals a clear separation between k∗
number of large values as the rest decay. In our experiments, λ = 0.001
to get good results. To
facilitate the formation of the bottleneck structure, L should be large, for our experiments we usually
use L = 20. Figure 2a shows how larger L, which have better separation between large and small
singular values, lead to improved test performance.

˜L

As first noted in section 1.3, solving for the Cost Of Identity, the kinetic energy, and the Hamiltonian
H is difficult due to the instability of the pseudo-inverse. Although the relaxation (Kp + γI) improves
the stability, we also utilize the solve function to avoid computing a pseudo-inverse altogether. The
stability of these computations rely on the boundedness of some additional properties: the path length
(cid:82) ||∂pAp|| dp, as well as the magnitudes of Bp, and Bpσ(Ap)T from the Hamiltonian reformulation.
Figure 3 shows how their respective magnitudes remains relatively constant as the effective depth ˜L
grows.

For compute resources, these small networks are not particularly resource intensive. Even on a CPU,
it only takes a couple minutes to fully train a leaky ResNet.

16

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

569

570

571

572

573

574

575

NeurIPS Paper Checklist

1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The contribution section accurately describes our contributions, and all
theorems/propositions are proven in the main or the appendix.
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
Justification: We discuss limitations of our results and approach after we state them.
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

• While the authors might fear that complete honesty about limitations might be used
by reviewers as grounds for rejection, a worse outcome might be that reviewers
discover limitations that aren’t acknowledged in the paper. The authors should use
their best judgment and recognize that individual actions in favor of transparency play
an important role in developing norms that preserve the integrity of the community.
Reviewers will be specifically instructed to not penalize honesty concerning limitations.

3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [Yes]

17

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

616

617

618

619

620

621

622

623

624

625

626

627

628

629

Justification: All assumptions are stated in the Theorem statements.
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

Question: Does the paper fully disclose all the information needed to reproduce the
main experimental results of the paper to the extent that it affects the main claims and/or
conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: The experimental setup is described in the Appendix.
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

• While NeurIPS does not require releasing code, the conference does require all
submissions to provide some reasonable avenue for reproducibility, which may depend
on the nature of the contribution. For example
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

Question: Does the paper provide open access to the data and code, with sufficient
instructions to faithfully reproduce the main experimental results, as described in
supplemental material?

18

630

631

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

667

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

Answer: [No]
Justification: We use synthetic data, with a description of how to build this synthetic data.
The code is not the main contribution of the paper, so there is little reason to publish it.
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

Question: Does the paper specify all the training and test details (e.g., data splits,
hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand
the results?
Answer: [Yes]
Justification: Most details are given in the experimental setup section in the Appendix.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail

that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental

material.

7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [No]
Justification: The numerical experiments are mostly there as a visualization of the theoretical
results, our main goal is therefore clarity, which would be hurt by putting error bars
everywhere.
Guidelines:

• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars,
confidence intervals, or statistical significance tests, at least for the experiments that
support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula,

call to a library function, bootstrap, etc.)

19

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

Question: For each experiment, does the paper provide sufficient information on the
computer resources (type of compute workers, memory, time of execution) needed to
reproduce the experiments?
Answer: [Yes]
Justification: In the experimental setup section of the Appendix.
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
Justification: We have read the Code of Ethics and see no issue.
Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a

deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special

consideration due to laws or regulations in their jurisdiction).

10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [NA]
Justification: The paper is theoretical in nature, so it has no direct societal impact that can
be meaningfully discussed.
Guidelines:

• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal

impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.

20

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

Justification: Not relevant to our paper.

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

Justification: We only use our own synthetic data.

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

21

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

• If this information is not available online, the authors are encouraged to reach out to

the asset’s creators.

13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [NA]
Justification: We do not release any new assets.
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
Justification: Not relevant to this paper.
Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with

human subjects.

• Including this information in the supplemental material is fine, but if the main
contribution of the paper involves human subjects, then as much detail as possible
should be included in the main paper.

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
Justification: Not relevant to this paper.
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

22

