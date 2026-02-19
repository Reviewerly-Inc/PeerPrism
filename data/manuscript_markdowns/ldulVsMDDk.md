Towards a Better Theoretical Understanding of
Independent Subnetwork Training

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

Modern advancements in large-scale machine learning would be impossible without
the paradigm of data-parallel distributed computing. Since distributed computing
with large-scale models imparts excessive pressure on communication channels, a
lot of recent research was directed towards co-designing communication compres-
sion strategies and training algorithms with the goal of reducing communication
costs. While pure data parallelism allows better data scaling, it suffers from poor
model scaling properties. Indeed, compute nodes are severely limited by memory
constraints, preventing further increases in model size. For this reason, the latest
achievements in training giant neural network models rely on some form of model
parallelism as well. In this work, we take a closer theoretical look at Independent
Subnetwork Training (IST), which is a recently proposed and highly effective
technique for solving the aforementioned problems. We identify fundamental
differences between IST and alternative approaches, such as distributed methods
with compressed communication, and provide a precise analysis of its optimization
performance on a quadratic model.

1

Introduction

A huge part of today’s machine learning success drives from the possibility to build more and more
complex models and train them on increasingly larger datasets. This fast progress has become
feasible due to advancements in distributed optimization, which is necessary for proper scaling
when the training data sizes grow [50]. In a typical scenario data parallelism is used for efficiency
which consists of sharding the dataset across computing devices. This allowed very efficient scaling
and accelerating of training moderately sized models by using additional hardware [19]. Though,
such data parallel approach can suffer from communication bottleneck, which sparked a lot of
research on distributed optimization with compressed communication of the parameters between
nodes [3, 27, 38].

1.1 The need for model parallel

Despite the efficiency gains of data parallelism, it has some fundamental limitations when it comes to
scaling up the model size. As the model dimension grows, the amount of memory required to store
and update the parameters also increases, which becomes problematic due to resource constraints
on individual devices. This has led to the development of model parallelism [11, 37], which splits
a large model across multiple nodes, with each node responsible for computations of model parts
[15, 47]. However, naive model parallelism also poses challenges because each node can only update
its portion of the model based on the data it has access to. This creates a need for a very careful
management of communication between devices. Thus, a combination of both data and model
parallelism is often necessary to achieve efficient and scalable training of huge models.

Submitted to 37th Conference on Neural Information Processing Systems (NeurIPS 2023). Do not distribute.

i = Ck

Select submodels wk
for i = 1, . . . , n in parallel do

Algorithm 1 Distributed Submodel (Stochastic) Gradient Descent
1: Parameters: learning rate γ > 0; sketches C1, . . . , Cn; initial model x0 ∈ Rd
2: for k = 0, 1, 2 . . . do
3:
4:
5:
6:
7:
8:
9:
10: end for

Compute local (stochastic) gradient w.r.t. submodel: Ck
Take (maybe multiple) gradient descent step z+
Send z+
i

end for
Aggregate/merge received submodels: xk+1 = 1
n

i ∇fi(wk
i )
i ∇fi(wk
i )

i xk for i ∈ [n] and broadcast to all computing nodes

to the server

i − γCk

i = wk

i=1 z+

(cid:80)n

i

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

IST.
Independent Subnetwork Training (IST) is a technique which suggests dividing the neural
network into smaller independent sub-parts, training them in a distributed parallel fashion and then
aggregating the results to update the weights of the whole model. According to IST, every subnetwork
is operational on its own, has fewer parameters than the full model, and this not only reduces the load
on computing nodes but also results in faster synchronization. A generalized analog of the described
method is formalized as an iterative procedure in Algorithm 1. This paradigm was pioneered by
[45] for networks with fully-connected layers and was later extended to ResNets [14] and Graph
architectures [43]. Previous experimental studies have shown that IST is a very promising approach
for various applications as it allows to effectively combine data with model parallelism and train
larger models with limited compute. In addition, [28] performed theoretical analysis of IST for
overparameterized single hidden layer neural networks with ReLU activations. The idea of IST was
also recently extended to the federated setting via an asynchronous distributed dropout [13] technique.

Federated Learning. Another important setting when the data is distributed (due to privacy reasons)
is Federated Learning [22, 27, 31]. In this scenario computing devices are often heterogeneous and
more resource-constrained [5] (e.g. mobile phones) in comparison to data-center setting. Such
challenges prompted extensive research efforts into selecting smaller and more efficient submodels
for local on-device training [2, 6, 8, 12, 20, 21, 29, 35, 42, 44]. Many of these works propose
approaches to adapt submodels, often tailored to specific neural network architectures, based on
the capabilities of individual clients for various machine learning tasks. However, there is a lack of
comprehension regarding the theoretical properties of these methods.

1.2 Summary of contributions

When reviewing the literature, we have found that a rigorous understanding of IST convergence
virtually does not exist, which motivates our work. The main contributions of this paper include

• A novel approach to analyzing distributed methods that combine data and model parallelism

by operating with sparse submodels for a quadratic model.

• The first analysis of independent subnetwork training in homogeneous and heterogeneous

scenarios without restrictive assumptions on gradient estimators.

• Identification of settings when IST can optimize very efficiently or converge not to the
optimal solution but only to an irreducible neighborhood which is also tightly characterized.
• Experimental validation of the proposed theory through carefully designed illustrative
experiments. Due to space limitations, the results (and proofs) are provided in the Appendix.

2 Formalism and Setup

We consider the standard optimization formulation of distributed/federated learning problem [41],

(cid:34)

min
x∈Rd

f (x) :=

(cid:35)

fi(x)

,

1
n

n
(cid:88)

i=1

(1)

69

70

where n is the number of clients/workers, each fi : Rd → Rd represents the loss of the model
parameterized by vector x ∈ Rd on the data of client i.

2

71

A typical Stochastic Gradient Descent (SGD) type method for solving this problem has the form

xk+1 = xk − γgk,

n
(cid:80)
i=1
i is a suitably constructed estimator of ∇fi(xk). In the distributed
where γ > 0 is a stepsize and gk
setting, computation of gradient estimators gk
i is typically performed by clients, sent to the server,
which subsequently performs aggregation via averaging gk = 1
i . The result is then used to
n
update the model xk+1 via a gradient-type method (2), and at the next iteration the model is broadcast
back to the clients. The process is repeated iteratively until a model of suitable qualities is found.

gk = 1
n

i=1 gk

gk
i ,

(cid:80)n

(2)

One of the main techniques used to accelerate distributed training is lossy communication compres-
sion [3, 27, 38]. It suggests applying a (possibly randomized) lossy compression mapping C to a
vector/matrix/tensor x before it is transmitted. This saves bits sent per every communication round
at the cost of transmitting a less accurate estimate C(x) of x. The error caused by this routine also
causes convergence issues, and to the best of our knowledge, convergence of IST-based techniques is
for this reason not yet understood.
Definition 1 (Unbiased compressor). A randomized mapping C : Rd → Rd is an unbiased compres-
sion operator (C ∈ U(ω) for brevity) if for some ω ≥ 0 and ∀x ∈ Rd

E [C(x)] = x,

E (cid:2)∥C(x) − x∥2(cid:3) ≤ ω∥x∥2.

(3)

A notable example of a mapping from this class is the random sparsification (Rand-q for q ∈
{1, . . . , d}) operator defined by

CRand-q(x) := Cqx = d
q

(cid:80)
i∈S

eie⊤

i x,

(4)

where e1, . . . , ed ∈ Rd are standard unit basis vectors in Rd, and S is a random subset of [d] :=
{1, . . . , d} sampled from the uniform distribution on the all subsets of [d] with cardinality q. Rand-q
belongs to U (d/q − 1), which means that the more elements are “dropped” (lower q), the higher is
the variance ω of the compressor.

In this work, we are mainly interested in a somewhat more general class of operators than mere
sparsifiers. In particular, we are interested in compressing via the application of random matrices, i.e.,
i ∈ Rd×d can be used to represent submodel computations in the following
via sketching. A sketch Ck
way:

where we require Ck
i
corresponds to computing the local gradient with respect to a sparse submodel model Ck
additionally sketching the resulting gradient with the same matrix Ck
update lies in the lower-dimensional subspace.

(5)
to be a symmetric positive semidefinite matrix. Such gradient estimate
i xk, and
i to guarantee that the resulting

gk
i := Ck

i ∇fi(Ck

i xk),

Using this notion, Algorithm 1 (with one local gradient step) can be represented in the following form

xk+1 = 1
n

n
(cid:80)
i=1

(cid:2)Ck

i xk − γCk

i ∇fi(Ck

i xk)(cid:3) ,

(6)

which is equivalent to the SGD-type update (2) when perfect reconstruction property holds

Ck

Ck := 1
n

n
(cid:80)
i=1
where I is the identity matrix (with probability one). This property holds for a specific class of
compressors that are particularly useful for capturing the concept of an independent subnetwork
partition.
Definition 2 (Permutation sketch). Assume that model size is greater than number of clients d ≥ n
and d = qn, where q ≥ 1 is an integer1. Let π = (π1, . . . , πd) be a random permutation of [d]. Then
for all x ∈ Rd and each i ∈ [n] we define Perm-q operator

i = I,

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

j=q(i−1)+1
1While this condition may look restrictive it naturally holds for distributed learning in a data-center setting.

Ci := n ·

qi
(cid:80)

eπj e⊤
πj

.

(7)

For other scenarios [40] generalized it for n ≥ d and block permutation case.

3

106

107

108

109

110

111

Perm-q is unbiased and can be conveniently used for representing (non-overlapping) structured
decomposition of the model such that every client i is responsible for computations over a submodel
Cixk.
Our convergence analysis relies on assumption previously used for coordinate descent type methods.
Assumption 1 (Matrix smoothness). A differentiable function f : Rd → R is L-smooth, if there
exists a positive semi-definite matrix L ∈ Rd×d such that

f (x + h) ≤ f (x) + ⟨∇f (x), h⟩ +

1
2

⟨Lh, h⟩ ,

∀x, h ∈ Rd.

(8)

112

Standard L-smoothness condition is obtained as a special case of (8) for L = L · I.

113

2.1

Issues with existing approaches

114

Consider the simplest gradient type method with compressed model in the single node setting

xk+1 = xk − γ∇f (C(xk)).

(9)

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

Algorithms belonging to this family require a different analysis in comparison to SGD [16, 18],
Distributed Compressed Gradient Descent [3, 26] and Randomized Coordinate Descent [34, 36] type
methods because the gradient estimator is no longer unbiased

E [∇f (C(x))] ̸= ∇f (x) = E [C(∇f (x))] .

(10)

That is why such kind of algorithms are harder to analyze. So, prior results for unbiased SGD [25]
can not be directly reused. Furthermore, the nature of the bias in this type of gradient estimator does
not exhibit additive (zero-mean) noise, thereby preventing the application of previous analyses for
biased SGD [1].

An assumption like bounded stochastic gradient norm extensively used in previous works [30, 48]
hinders an accurate understanding of such methods. This assumption hides the fundamental difficulty
of analyzing biased gradient estimator:

(cid:104)

∥∇f (C(x))∥2(cid:105)

E

≤ G

(11)

and may not hold even for quadratic functions f (x) = x⊤Ax.
In addition, in the distributed
setting such condition can result in vacuous bounds [23] as it does not allow to accurately capture
heterogeneity.

3 Results in the Interpolation Case

To conduct a thorough theoretical analysis of methods that combine data with model parallelism,
we simplify the algorithm and problem setting to isolate the unique effects of this approach. The
following considerations are made:

(1) We assume that every node i computes the true gradient at the submodel Ci∇fi(Cixk).
(2) A notable difference from the original IST algorithm 1 is that workers perform single

gradient descent step (or just gradient computation).

(3) Finally, we consider a special case of quadratic model (12) as a loss function (1).

Condition (1) is mainly for the sake of simplicity and clarity of exposition and can be potentially
generalized to stochastic gradient computations. (2) is imposed because local steps did not bring
any theoretical efficiency improvements for heterogeneous settings until very recently [32]. And
even then, only with the introduction of additional control variables, which goes against resource-
constrained device setting. The reason behind (3) is that despite the seeming simplicity quadratic
problem has been used extensively to study properties of neural networks [46, 49]. Moreover, it is a
non-trivial model which allows to understand complex optimization algorithms [4, 10, 17]. It serves
as a suitable problem for observing complex phenomena and providing theoretical insights, which
can also be observed in practical scenarios.

4

148

149

150

151

152

153

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

145

Having said that we consider a special case of problem (1)

f (x) = 1
n

n
(cid:80)
i=1

fi(x),

fi(x) ≡ 1

2 x⊤Lix − b⊤

i x.

(12)

146

In this case, f (x) is L-smooth, and ∇f (x) = L x − b, where L = 1
n

(cid:80)n

i=1 Li and b := 1
n

(cid:80)n

i=1 bi.

147

3.1 No linear term: problems and solutions

First, let us examine the case of bi ≡ 0, which we call interpolation for quadratics, and perform the
analysis for general sketches Ck
n
(cid:80)
i=1

i . In this case the gradient estimator (2) takes the form

k
i xk = B

i xk) = 1
n

i ∇fi(Ck

gk = 1
n

i LiCk

n
(cid:80)
i=1

Ck

Ck

(13)

xk

k
where B

(cid:80)n

:= 1
n

i=1 Ck

i LiCk

i . We prove the following result for a method with such an estimator.
Theorem 1. Consider the method (2) with estimator (13) for a quadratic problem (12) with L ≻ 0
and bi ≡ 0. Then if W := 1
2

⪰ 0 and there exists constant θ > 0:

k
+ B

k
L B

(cid:105)
L

E

(cid:104)

(cid:104)
k
B

E

k(cid:105)

L B

⪯ θ W,

and the step size is chosen as 0 < γ ≤ 1

θ , the iterates satisfy

1
K

K−1
(cid:80)
k=0

E

(cid:104)(cid:13)
(cid:13)∇f (xk)(cid:13)
2
(cid:13)
L−1 W L−1

(cid:105)

≤

2(f (x0)−E[f (xK )])
γK

,

154

and

(cid:104)

E

∥xk − x⋆∥2
L

(cid:16)

(cid:105)

≤

1 − γλmin

(cid:16)

− 1

L

2 W L

− 1

2 (cid:17)(cid:17)k

∥x0 − x⋆∥2
L

.

(14)

(15)

(16)

This theorem establishes an O(1/K) convergence rate with constant step size up to a stationary point
and linear convergence for the expected distance to the optimum. Note that we employ weighted
norms in our analysis, as the considered class of loss functions satisfies the matrix L-smoothness
Assumption 1. The use of standard Euclidean distance may result in loose bounds that do not recover
correct rates for special cases like Gradient Descent.

It is important to highlight that inequality (14) may not hold (for any θ > 0) in the general case
as the matrix W is not guaranteed to be positive (semi-)definite in the case of general sampling.
i can result in gradient estimator gk, which is
The intuition behind it is that arbitrary sketches Ck
misaligned with the true gradient ∇f (xk). Specifically, the inner product (cid:10)∇f (xk), gk(cid:11) can be
negative, and there is no expected descent after one step.

Next, we give examples of samplings for which the inequality (14) can be satisfied.

3
k
1. Identity. Consider Ci ≡ I. Then B
= L
satisfied for θ = λmax(L). So, (15) says that if we choose γ = 1

k
= L, B

k
L B

2
, W = L
θ , then

≻ 0 and hence (14) is

1
K

K−1
(cid:80)
k=0

(cid:13)
(cid:13)∇f (xk)(cid:13)
2
I ≤
(cid:13)

2λmax(L)(f (x0)−f (xK ))
K

,

which exactly matches the rate of Gradient Descent in the non-convex setting. As for iterates
convergence, the rate in (16) is λmax(L)/λmin(L) corresponding to precise Gradient Descent result for
strongly convex functions.
2. Permutation. Assume n = d2 and the use of Perm-1 (special case of Definition 2) sketch
Ck

n) is a random permutation of [n]. Then

, where πk = (πk

1 , . . . , πk

i = neπk

i

e⊤
πk
i
(cid:104)
B

E

k(cid:105)

= 1
n

n
(cid:80)
i=1

n2E (cid:2)Ck

i LiCk
i

(cid:3) = 1

n

n
(cid:80)
i=1

nDiag(Li) = (cid:80)n

i=1 Di = n D,

2This is done mainly for simplifying the presentation. Results can be generalized to the case of n ̸= d in the

similar way as done in [40] which can be found in the Appendix.

5

171

where D := 1
n

(cid:80)n

i=1 Di, Di := Diag(Li). Then inequality (14) leads to

n D L D ⪯ θ
2

(cid:0)L D + D L(cid:1) ,

(17)

172

173

174

175

176

177

which may not always hold as L D + D L is not guaranteed to be positive definite even in case of
L ≻ 0. However, such kind of condition can be enforced via a slight modification of permutation
sketches { ˜Ci}n
i=1, which is done in Section 3.1.2. The limitation of such an approach is that
compressors ˜Ci become no longer unbiased.
Remark 1. Matrix W in case of permutation sketches may not be positive-definite. Consider the
following homogeneous (Li ≡ L) two-dimensional problem example

L =

(cid:20) a c
b
c

(cid:21)

.

178

Then

W = 1
2

(cid:2)L D + D L(cid:3) =

(cid:20)

a2
c(a + b)/2

c(a + b)/2
b2

(cid:21)

,

(18)

(19)

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

which for c > 2ab

a+b has det(W) < 0, and thus W ⊁ 0 according to Sylvester’s criterion.

Next, we focus on the particular case of Permutation sketches, which are the most suitable for
model partitioning according to Independent Subnetwork Training (IST). At the rest of the section,
we discuss how the condition (14) can be enforced via a specially designed preconditioning of the
problem (12) or modification of sketch mechanism (7).

3.1.1 Homogeneous problem preconditioning

To start consider a homogeneous setting fi(x) = 1
diagonal matrix with elements equal to diagonal of L. Then problem can be converted to

2 x⊤Lx, so Li ≡ L. Now define D = Diag(L) –

fi(D− 1

2 x) = 1
2

(cid:16)

D− 1

2 x

(cid:17)⊤

(cid:16)

L

D− 1

2 x

(cid:17)

= 1

2 x⊤ (cid:16)
(cid:124)

D− 1

2

2 LD− 1
(cid:123)(cid:122)
˜L

x,

(cid:17)

(cid:125)

(20)

which is equivalent to the original problem after a change of variables ˜x := D− 1
2 x. Note that
D = Diag(L) is positive definite as L ≻ 0, and therefore ˜L ≻ 0. Moreover, the preconditioned
matrix ˜L has all ones on the diagonal: Diag(˜L) = I. If we now combine it with Perm-1 sketches

(cid:104)

E

k(cid:105)

B

= E

(cid:104) 1
n

(cid:80)n

i=1 Ci ˜L Ci

(cid:105)

= nDiag(˜L) = nI.

Therefore, inequality (14) takes the form ˜W = n ˜L ⪰ 1
side of (15) can be transformed the following way

θ n2 ˜L, which holds for θ ≥ n, and left hand

(cid:13)∇f (xk)(cid:13)
(cid:13)
2
˜L−1 ˜W ˜L−1 ≥ nλmin
(cid:13)

−1(cid:17) (cid:13)

(cid:16)˜L

I = nλmax(˜L) (cid:13)
(cid:13)∇f (xk)(cid:13)
2
(cid:13)

(cid:13)∇f (xk)(cid:13)
2
(cid:13)
I

192

for an accurate comparison to standard methods. The resulting convergence guarantee

1
K

K−1
(cid:80)
k=0

E

(cid:104)(cid:13)
(cid:13)∇f (xk)(cid:13)
2
(cid:13)
I

(cid:105)

≤

2λmax( ˜L)(f (x0)−E[f (xK )])
K

,

193

which matches classical Gradient Descent.

(21)

(22)

194

195

196

197

198

3.1.2 Heterogeneous sketch preconditioning

In contrast to homogeneous case the heterogeneous problem fi(x) = 1
2 x⊤Lix can not be so easily
preconditioned by a simple change of variables ˜x := D− 1
2 x, as every client i has its own matrix
Li. However, this problem can be fixed via the following modification of Perm-1, which scales the
output according to the diagonal elements of local smoothness matrix Li:

˜Ci :=

√

n

(cid:104)
L− 1

2

i

(cid:105)

πi,πi

eπie⊤
πi

.

(23)

6

In this case E

(cid:104) ˜CiLi ˜Ci

(cid:105)

= I, E

(cid:104)

k(cid:105)

B

= I, and W = L. Then inequality (14) is satisfied for θ ≥ 1.

If one plugs these results into (15), such convergence guarantee can be obtained
(cid:104)(cid:13)
(cid:13)∇f (xk)(cid:13)
2
(cid:13)
I

2λmax(L)(f (x0)−E[f (xK )])
K

1
K

≤

E

(cid:105)

,

K−1
(cid:80)
k=0

(24)

which matches the Gradient Descent result as well. Thus we can conclude that heterogeneity does not
bring such a fundamental challenge in this scenario. In addition, a method with Perm-1 is significantly
better in terms of computational and communication complexity as it requires calculating the local
gradients with respect to much smaller submodels and transmits only sparse updates.

This construction also shows that for γ = 1/θ = 1

(cid:16)

γλmin

L

− 1

2 W L

− 1

2 (cid:17)

(cid:16)

= λmin

L

− 1

2 L L

− 1

2 (cid:17)

= 1,

(25)

which after plugging into the bound for the iterates (16) shows that the method basically converges in
1 iteration. This observation that sketch preconditioning can be extremely efficient, although it uses
only the diagonal elements of matrices Li.
Now when we understand that the method can perform very well in the special case of ˜bi ≡ 0 we can
move on to a more complicated situation.

4

Irreducible Bias in the General Case

Now we look at the most general heterogeneous case with different matrices and linear terms
fi(x) ≡ 1

2 x⊤Lix − x⊤ bi . In this instance gradient estimator (2) takes the form

gk = 1
n

Ck

i ∇fi(Ck

i xk) = 1
n

n
(cid:80)
i=1

Ck
i

(cid:0)LiCk

i xk − bi

(cid:1) = B

k

xk − Cb,

(26)

where Cb = 1
n
(23) like in Section 3.1.2 Then E
B
Furthermore expected gradient estimator (26) results in E (cid:2)gk(cid:3) = xk − 1√
the following way

i bi. Herewith let us use a heterogeneous permutation sketch preconditioner
bi.

n
n (cid:103)D b and can be transformed

n (cid:103)D b, where (cid:103)D b := 1

= I and E (cid:2)Cb(cid:3) = 1√

i=1 D− 1

(cid:80)n

k(cid:105)

(cid:104)

i

2

n
(cid:80)
i=1
i=1 Ck

(cid:80)n

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

E (cid:2)gk(cid:3) = L

−1

L xk ± L

−1

b − 1√

−1

n (cid:103)D b = L

∇f (xk) + L

−1

b −

1
√
n

(cid:103)D b
,

(cid:124)

(cid:123)(cid:122)
h

(cid:125)

(27)

218

219

which reflects the decomposition of the estimator into optimally preconditioned true gradient and a
bias, depending on the linear terms bi.

220

4.1 Bias of the method

221

Estimator (27) can be directly plugged (with proper conditioning) into general SGD update (2)

E (cid:2)xk+1(cid:3) = xk − γE (cid:2)gk(cid:3) = (1 − γ)xk + γ√

n (cid:103)D b = (1 − γ)k+1 x0 + γ√

n (cid:103)D b

k
(cid:80)
j=0

(1 − γ)j. (28)

222

223

The resulting recursion (28) is exact, and its asymptotic limit can be analyzed. Thus for constant
γ < 1 by using the formula for the sum of the first k terms of a geometric series, one gets

E (cid:2)xk(cid:3) = (1 − γ)k x0 + 1−(1−γ)k

√

n (cid:103)D b −→
k→∞

1√
n (cid:103)D b,

which shows that in the limit, the first initialization term (with x0) vanishes while the second converges
to 1√

n (cid:103)D b. This reasoning shows that the method does not converge to the exact solution

xk → x∞ ̸= x⋆ ∈ arg min

x∈Rd

2 x⊤ L x − x⊤ b(cid:9) ,
(cid:8) 1

224

225

which for the positive-definite L can be defined as x⋆ = L
bi. So,
in general, there is an unavoidable bias. However, in the limit case: n = d → ∞, the bias diminishes.

b, while x∞ = 1
√
n

n

(cid:80)n

i=1 D− 1

i

2

−1

7

226

4.2 Generic convergence analysis

227

228

229

230

231

232

233

While the analysis in Section 4.1 is precise, it does not allow us to compare the convergence of IST
to standard optimization methods. Due to this, we also analyze the non-asymptotic behavior of the
method to understand the convergence speed. Our result is formalized in the following theorem.

Theorem 2. Consider the method (2) with estimator (26) for a quadratic problem (12) with the
positive definite matrix L ≻ 0. Assume that for every Di := Diag(Li) matrices D− 1
exist, scaled
permutation sketches (23) are used and heterogeneity is bounded as E
≤ σ2.
Then for step size is chosen as

(cid:104)(cid:13)
(cid:13)gk − E (cid:2)gk(cid:3)(cid:13)
2
(cid:13)
L

(cid:105)

i

2

0 < γ ≤ γc,β := 1/2−β
β+1/2 ,

(29)

234

where γc,β ∈ (0, 1] for β ∈ (0, 1/2), the iterates satisfy

1
K

K−1
(cid:80)
k=0

E

(cid:104)(cid:13)
(cid:13)∇f (xk)(cid:13)
2
(cid:13)
L−1

(cid:105)

≤

2(f (x0)−E[f (xK )])
γK

+ (cid:0)2β−1 (1 − γ) + γ(cid:1) ∥h∥2

L

+ γσ2,

(30)

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

where L = 1
n

(cid:80)n

i=1 Li, h = L

−1

b − 1√
n

1
n

(cid:80)n

i=1 D− 1

i

2

bi and b = 1
n

(cid:80)n

i=1 bi.

Note that the derived convergence upper bound has a neighborhood proportional to the bias of
the gradient estimator h and level of heterogeneity σ2. Some of these terms with factor γ can be
eliminated via decreasing learning rate schedule (e.g., ∼ 1/
k). However, such a strategy does not
diminish the term with a multiplier 2β−1 (1 − γ), making the neighborhood irreducible. Moreover,
this term can be eliminated for γ = 1, which also minimizes the first term that decreases as 1/K.
Though, such step size choice maximizes the terms with factor γ. Furthermore, there exists an
inherent trade-off between convergence speed and the size of the neighborhood.

√

In addition, convergence to the stationary point is measured in the weighted by L
squared norm of
the gradient. At the same time, the neighborhood term depends on the weighted by L norm of h. This
fine-grained decoupling is achieved by carefully applying Fenchel-Young inequality and provides a
tighter characterization of the convergence compared to using standard Euclidean distances.

−1

In this scenario, every worker has access to the all data fi(x) ≡ 1

2 x⊤Lx−x⊤ b.
Homogeneous case.
Then diagonal preconditioning of the problem can be used as in the previous Section 3.1.1. This
results in a gradient ∇f (x) = ˜L x− ˜b for ˜L = D− 1
2 LD− 1
2 b. If it is further combined
√
neπie⊤
with a scaled by 1/

2 and ˜b = D− 1
πi , the resulting gradient estimator is

n Permutation sketch Ci :=

√

gk = xk − 1√
n

−1

˜b = ˜L

∇f (xk) + ˜h,

(31)

˜b.
(cid:105)

In this case heterogeneity term σ2 from upper bound (30) disappears

−1 ˜b − 1√
for ˜h = ˜L
n
(cid:104)(cid:13)
(cid:13)gk − E (cid:2)gk(cid:3)(cid:13)
2
as E
= 0, thus the neighborhood size can significantly decrease. However,
(cid:13)
L
the bias term depending on ˜h still remains as the method does not converge to the exact solution
√
xk → x∞ ̸= x⋆ = ˜L
n
˜b, which means that ˜b is the right eigenvector of
and solution x⋆ can coincide when ˜L
matrix ˜L

−1 ˜b for positive-definite ˜L. Nevertheless the method’s fixed point x∞ = ˜b /

with eigenvalue 1√

−1 ˜b = 1√

−1

n

n .

Let us contrast obtained result (30) with non-convex rate of SGD [25] with constant step size γ for
L-smooth and lower-bounded f

min
k∈{0,...,K−1}

(cid:13)
(cid:13)∇f (xk)(cid:13)
2
(cid:13)

≤

6(f (x0)−inf f)
γK

+ γLC,

(32)

where constant C depends, for example, on the variance of stochastic gradient estimates. Observe
that the first term in the compared upper bounds (32) and (30) is almost identical and decreases with
speed 1/K. But unlike (30) the neighborhood for SGD can be completely eliminated by reducing the
step size γ. This highlights a fundamental difference of our results to unbiased methods.

8

263

The intuition behind this issue is that for SGD-type methods like Compressed Gradient Descent

264

the gradient estimate is unbiased and enjoys the property that variance

E (cid:2)∥C(∇f (xk)) − ∇f (xk)∥2(cid:3) ≤ ω∥∇f (xk)∥2

xk+1 = xk − C(∇f (xk))

(33)

(34)

265

266

goes down to zero as the method progresses because ∇f (xk) → ∇f (x⋆) = 0 in the unconstrained
case. In addition, any stationary point x⋆ ceases to be a fixed point of the iterative procedure as

x⋆ ̸= x⋆ − ∇f (C(x⋆)),

(35)

267

268

269

in the general case, unlike for Compressed Gradient Descent with both biased and unbiased compres-
sors C. So, even if the method (computing gradient at sparse model) is initialized from the solution
after one gradient step, it may get away from there.

270

4.3 Comparison to previous works

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

Independent Subnetwork Training [45]. There are several improvements over the previous works
that tried to theoretically analyze the convergence of Distributed IST.

The first difference is that our results allow for an almost arbitrary level of model sparsification,
i.e., work for any ω ≥ 0 as permutation sketches can be viewed as a special case of compression
operators (1). This improves significantly over the work of [45], which demands3 ω ≲ µ2/L2. Such a
requirement is very restrictive as the condition number L/µ of the loss function f is typically very
large for any non-trivial optimization problem. Thus, the sparsifier’s (4) variance ω = d/q − 1 has to
be very close to 0 and q ≈ d. So, the previous theory allows almost no compression (sparsification)
because it is based on the analysis of Gradient Descent with Compressed Iterates [24].

The second distinction is that the original IST work [45] considered a single node setting and thus
their convergence bounds did not capture the effect of heterogeneity, which we believe is of crucial
importance for distributed setting [9, 39]. Besides, they consider Lipschitz continuity of the loss
function f , which is not satisfied for a simple quadratic model. A more detailed comparison including
additional assumptions on the gradient estimator made in [45] is presented in the Appendix.

FL with Model Pruning.
In a recent work [48] made an attempt to analyze a variant of the FedAvg
algorithm with sparse local initialization and compressed gradient training (pruned local models).
They considered a case of L-smooth loss and sparsification operator satisfying a similar condition to
(1). However, they also assumed that the squared norm of stochastic gradient is uniformly bounded
(11), which is “pathological” [23] especially in the case of local methods as it does not allow to
capture the very important effect of heterogeneity and can result in vacuous bounds.

In the Appendix we show some limitations of other relevant previous approaches to training with
compressed models: too restrictive assumptions on the algorithm [33] or not applicability in our
problem setting [7].

5 Conclusions and Future Work

In this study, we introduced a novel approach to understanding training with combined model and
data parallelism for a quadratic model. This framework allowed to shed light on distributed submodel
optimization which revealed the advantages and limitations Independent Subnetwork Training (IST).
Moreover, we accurately characterized the behavior of the considered method in both homogeneous
and heterogeneous scenarios without imposing restrictive assumptions on gradient estimators.

In future research, it would be valuable to explore extensions of our findings to settings that are closer
to practical scenarios, such as cross-device federated learning. This could involve investigating partial
participation support, leveraging local training benefits, and ensuring robustness against stragglers.
Additionally, it would be interesting to generalize our results to non-quadratic scenarios without
relying on pathological assumptions.

3µ refers to constant from Polyak-Łojasiewicz (or strong convexity) condition. In case of a quadratic problem

with positive-definite matrix A: µ = λmin(A)

9

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

References

[1] Ahmad Ajalloeian and Sebastian U Stich. On the convergence of SGD with biased gradients.

arXiv preprint arXiv:2008.00051, 2020.

[2] Samiul Alam, Luyang Liu, Ming Yan, and Mi Zhang. FedRolex: Model-heterogeneous federated
learning with rolling sub-model extraction. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave,
and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022.

[3] Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. QSGD:
Communication-efficient SGD via gradient quantization and encoding. In Advances in Neural
Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

[4] Yossi Arjevani, Ohad Shamir, and Nathan Srebro. A tight convergence analysis for stochastic
gradient descent with delayed updates. In Algorithmic Learning Theory, pages 111–132. PMLR,
2020.

[5] Sebastian Caldas, Jakub Koneˇcny, H Brendan McMahan, and Ameet Talwalkar. Expanding
the reach of federated learning by reducing client resource requirements. arXiv preprint
arXiv:1812.07210, 2018.

[6] Zachary Charles, Kallista Bonawitz, Stanislav Chiknavaryan, Brendan McMahan, et al. Fed-
erated select: A primitive for communication-and memory-efficient federated learning. arXiv
preprint arXiv:2208.09432, 2022.

[7] El Mahdi Chayti and Sai Praneeth Karimireddy. Optimization with access to auxiliary informa-

tion. arXiv preprint arXiv:2206.00395, 2022.

[8] Yuanyuan Chen, Zichen Chen, Pengcheng Wu, and Han Yu. Fedobd: Opportunistic block
dropout for efficiently training large-scale neural networks through federated learning. arXiv
preprint arXiv:2208.05174, 2022.

[9] Sélim Chraibi, Ahmed Khaled, Dmitry Kovalev, Peter Richtárik, Adil Salim, and Martin Takáˇc.
Distributed fixed point methods with compressed iterates. arXiv preprint arXiv:2102.07245,
2019.

[10] Leonardo Cunha, Gauthier Gidel, Fabian Pedregosa, Damien Scieur, and Courtney Paque-
tte. Only tails matter: Average-case universality and robustness in the convex regime. In
International Conference on Machine Learning, pages 4474–4491. PMLR, 2022.

[11] Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao, Marc’aurelio
Ranzato, Andrew Senior, Paul Tucker, Ke Yang, et al. Large scale distributed deep networks.
Advances in neural information processing systems, 25, 2012.

[12] Enmao Diao, Jie Ding, and Vahid Tarokh. Heterofl: Computation and communication efficient

federated learning for heterogeneous clients. arXiv preprint arXiv:2010.01264, 2020.

[13] Chen Dun, Mirian Hipolito, Chris Jermaine, Dimitrios Dimitriadis, and Anastasios Kyrillidis.
Efficient and light-weight federated learning via asynchronous distributed dropout. arXiv
preprint arXiv:2210.16105, 2022.

[14] Chen Dun, Cameron R Wolfe, Christopher M Jermaine, and Anastasios Kyrillidis. ResIST:
In Uncertainty in Artificial

Layer-wise decomposition of resnets for distributed training.
Intelligence, pages 610–620. PMLR, 2022.

[15] Philipp Farber and Krste Asanovic. Parallel neural network training on multi-spert. In Proceed-
ings of 3rd International Conference on Algorithms and Architectures for Parallel Processing,
pages 659–666. IEEE, 1997.

[16] Eduard Gorbunov, Filip Hanzely, and Peter Richtárik. A unified theory of SGD: Variance
reduction, sampling, quantization and coordinate descent. In International Conference on
Artificial Intelligence and Statistics, pages 680–690. PMLR, 2020.

10

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

[17] Baptiste Goujaud, Damien Scieur, Aymeric Dieuleveut, Adrien B Taylor, and Fabian Pedregosa.
Super-acceleration with cyclical step-sizes. In International Conference on Artificial Intelligence
and Statistics, pages 3028–3065. PMLR, 2022.

[18] Robert Mansel Gower, Nicolas Loizou, Xun Qian, Alibek Sailanbayev, Egor Shulgin, and Peter
Richtárik. SGD: General analysis and improved rates. Proceedings of the 36th International
Conference on Machine Learning, Long Beach, California, 2019.

[19] Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola,
Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch SGD: Training
imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2018.

[20] Samuel Horvath, Stefanos Laskaridis, Mario Almeida, Ilias Leontiadis, Stylianos Venieris, and
Nicholas Lane. FjORD: Fair and accurate federated learning under heterogeneous targets with
ordered dropout. Advances in Neural Information Processing Systems, 34:12876–12889, 2021.

[21] Yuang Jiang, Shiqiang Wang, Victor Valls, Bong Jun Ko, Wei-Han Lee, Kin K Leung, and
Leandros Tassiulas. Model pruning enables efficient federated learning on edge devices. IEEE
Transactions on Neural Networks and Learning Systems, 2022.

[22] Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Ar-
jun Nitin Bhagoji, Kallista A. Bonawitz, Zachary Charles, Graham Cormode, Rachel Cum-
mings, Rafael G. L. D’Oliveira, Hubert Eichner, Salim El Rouayheb, David Evans, Josh Gardner,
Zachary Garrett, Adrià Gascón, Badih Ghazi, Phillip B. Gibbons, Marco Gruteser, Zaïd Har-
chaoui, Chaoyang He, Lie He, Zhouyuan Huo, Ben Hutchinson, Justin Hsu, Martin Jaggi, Tara
Javidi, Gauri Joshi, Mikhail Khodak, Jakub Koneˇcný, Aleksandra Korolova, Farinaz Koushanfar,
Sanmi Koyejo, Tancrède Lepoint, Yang Liu, Prateek Mittal, Mehryar Mohri, Richard Nock,
Ayfer Özgür, Rasmus Pagh, Hang Qi, Daniel Ramage, Ramesh Raskar, Mariana Raykova, Dawn
Song, Weikang Song, Sebastian U. Stich, Ziteng Sun, Ananda Theertha Suresh, Florian Tramèr,
Praneeth Vepakomma, Jianyu Wang, Li Xiong, Zheng Xu, Qiang Yang, Felix X. Yu, Han Yu,
and Sen Zhao. Advances and open problems in federated learning. Found. Trends Mach. Learn.,
14(1-2):1–210, 2021.

[23] Ahmed Khaled, Konstantin Mishchenko, and Peter Richtárik. Tighter theory for local SGD on
identical and heterogeneous data. In International Conference on Artificial Intelligence and
Statistics, pages 4519–4529. PMLR, 2020.

[24] Ahmed Khaled and Peter Richtárik. Gradient descent with compressed iterates. arXiv preprint

arXiv:1909.04716, 2019.

[25] Ahmed Khaled and Peter Richtárik. Better theory for SGD in the nonconvex world. Transactions

on Machine Learning Research, 2023. Survey Certification.

[26] Sarit Khirirat, Hamid Reza Feyzmahdavian, and Mikael Johansson. Distributed learning with

compressed gradients. arXiv preprint arXiv:1806.06573, 2018.

[27] Jakub Koneˇcný, H. Brendan McMahan, Felix X. Yu, Peter Richtárik, Ananda Theertha Suresh,
and Dave Bacon. Federated learning: Strategies for improving communication efficiency. NIPS
Private Multi-Party Machine Learning Workshop, 2016.

[28] Fangshuo Liao and Anastasios Kyrillidis. On the convergence of shallow neural network training

with randomly masked neurons. Transactions on Machine Learning Research, 2022.

[29] Rongmei Lin, Yonghui Xiao, Tien-Ju Yang, Ding Zhao, Li Xiong, Giovanni Motta, and
Françoise Beaufays. Federated pruning: Improving neural network efficiency with federated
learning. arXiv preprint arXiv:2209.06359, 2022.

[30] Tao Lin, Sebastian U Stich, Luis Barba, Daniil Dmitriev, and Martin Jaggi. Dynamic model
pruning with feedback. In International Conference on Learning Representations, 2019.

[31] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas.
Communication-Efficient Learning of Deep Networks from Decentralized Data. In Proceedings
of the 20th International Conference on Artificial Intelligence and Statistics, volume 54 of
Proceedings of Machine Learning Research, pages 1273–1282, 20–22 Apr 2017.

11

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

441

442

443

444

445

446

447

448

[32] Konstantin Mishchenko, Grigory Malinovsky, Sebastian Stich, and Peter Richtárik. ProxSkip:
Yes! local gradient steps provably lead to communication acceleration! finally! In International
Conference on Machine Learning, pages 15750–15769. PMLR, 2022.

[33] Amirkeivan Mohtashami, Martin Jaggi, and Sebastian Stich. Masked training of neural networks
with partial gradients. In International Conference on Artificial Intelligence and Statistics,
pages 5876–5890. PMLR, 2022.

[34] Yu Nesterov. Efficiency of coordinate descent methods on huge-scale optimization problems.

SIAM Journal on Optimization, 22(2):341–362, 2012.

[35] Xinchi Qiu, Javier Fernandez-Marques, Pedro PB Gusmao, Yan Gao, Titouan Parcollet, and
Nicholas Donald Lane. ZeroFL: Efficient on-device training for federated learning with local
sparsity. In International Conference on Learning Representations, 2022.

[36] Peter Richtárik and Martin Takáˇc. Iteration complexity of randomized block-coordinate descent
methods for minimizing a composite function. Mathematical Programming, 144(1-2):1–38,
2014.

[37] Peter Richtárik and Martin Takáˇc. Distributed coordinate descent method for learning with big

data. Journal of Machine Learning Research, 17(75):1–25, 2016.

[38] Frank Seide, Hao Fu, Jasha Droppo, Gang Li, and Dong Yu. 1-bit stochastic gradient descent
and its application to data-parallel distributed training of speech dnns. In Fifteenth Annual
Conference of the International Speech Communication Association, 2014.

[39] Egor Shulgin and Peter Richtárik. Shifted compression framework: Generalizations and
improvements. In The 38th Conference on Uncertainty in Artificial Intelligence, 2022.

[40] Rafał Szlendak, Alexander Tyurin, and Peter Richtárik. Permutation compressors for prov-
In International Conference on Learning

ably faster distributed nonconvex optimization.
Representations, 2022.

[41] Jianyu Wang, Zachary Charles, Zheng Xu, Gauri Joshi, H Brendan McMahan, Maruan Al-
Shedivat, Galen Andrew, Salman Avestimehr, Katharine Daly, Deepesh Data, et al. A field
guide to federated optimization. arXiv preprint arXiv:2107.06917, 2021.

[42] Dingzhu Wen, Ki-Jun Jeon, and Kaibin Huang. Federated dropout—a simple approach for
enabling federated learning on resource constrained devices. IEEE Wireless Communications
Letters, 11(5):923–927, 2022.

[43] Cameron R Wolfe, Jingkang Yang, Arindam Chowdhury, Chen Dun, Artun Bayer, Santiago
Segarra, and Anastasios Kyrillidis. Gist: Distributed training for large-scale graph convolutional
networks. arXiv preprint arXiv:2102.10424, 2021.

[44] Tien-Ju Yang, Dhruv Guliani, Françoise Beaufays, and Giovanni Motta. Partial variable training
for efficient on-device federated learning. In ICASSP 2022-2022 IEEE International Conference
on Acoustics, Speech and Signal Processing (ICASSP), pages 4348–4352. IEEE, 2022.

[45] Binhang Yuan, Cameron R Wolfe, Chen Dun, Yuxin Tang, Anastasios Kyrillidis, and Chris
Jermaine. Distributed learning of fully connected neural networks using independent subnet
training. Proceedings of the VLDB Endowment, 15(8):1581–1590, 2022.

[46] Guodong Zhang, Lala Li, Zachary Nado, James Martens, Sushant Sachdeva, George Dahl, Chris
Shallue, and Roger B Grosse. Which algorithmic choices matter at which batch sizes? insights
from a noisy quadratic model. Advances in neural information processing systems, 32, 2019.

[47] Xiru Zhang, Michael Mckenna, Jill Mesirov, and David Waltz. An efficient implementation
of the back-propagation algorithm on the connection machine cm-2. Advances in neural
information processing systems, 2, 1989.

[48] Hanhan Zhou, Tian Lan, Guru Venkataramani, and Wenbo Ding. On the convergence of
heterogeneous federated learning with arbitrary adaptive online model pruning. arXiv preprint
arXiv:2201.11803, 2022.

12

449

450

451

452

[49] Libin Zhu, Chaoyue Liu, Adityanarayanan Radhakrishnan, and Mikhail Belkin. Quadratic
models for understanding neural network dynamics. arXiv preprint arXiv:2205.11787, 2022.

[50] Martin Zinkevich, Markus Weimer, Lihong Li, and Alex Smola. Parallelized stochastic gradient

descent. Advances in neural information processing systems, 23, 2010.

13

