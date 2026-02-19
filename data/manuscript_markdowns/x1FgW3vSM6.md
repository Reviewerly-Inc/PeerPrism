Flag Aggregator: Distributed Training under Failures
and Augmented Losses using Convex Optimization

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

Modern ML applications increasingly rely on complex deep learning models and
large datasets. There has been an exponential growth in the amount of computa-
tion needed to train the largest models. Therefore, to scale computation and data,
these models are inevitably trained in a distributed manner in clusters of nodes,
and their updates are aggregated before being applied to the model. However, a
distributed setup is prone to Byzantine failures of individual nodes, components,
and software. With data augmentation added to these settings, there is a critical
need for robust and efficient aggregation systems. We define the quality of workers
as reconstruction ratios ∈ (0, 1], and formulate aggregation as a Maximum Like-
lihood Estimation procedure using Beta densities. We show that the Regularized
form of log-likelihood wrt subspace can be approximately solved using iterative
least squares solver, and provide convergence guarantees using recent Convex
Optimization landscape results. Our empirical findings demonstrate that our ap-
proach significantly enhances the robustness of state-of-the-art Byzantine resilient
aggregators. We evaluate our method in a distributed setup with a parameter server,
and show simultaneous improvements in communication efficiency and accuracy
across various tasks.

1

Introduction

How to Design Aggregators? We consider the problem of designing aggregation functions that can
be written as optimization problems of the form,

A(g1, . . . , gp) ∈ arg min
Y ∈C

Ag1,...,gp (Y ),

(1)

where {gi}p
i=1 ⊆ Rn are given estimates of an unknown summary statistic used to compute the
Aggregator Y ∗. If we choose A to be a quadratic function that decomposes over gi’s, and C = Rn,
then we can see A is simply the standard mean operator. There is a mature literature of studying such
functions for various scientific computing applications [1]. More recently, from the machine learning
standpoint there has been a plethora of work [2, 3, 4, 5] on designing provably robust aggregators A
for mean estimation tasks under various technical assumptions on the distribution or moments of gi.

Distributed ML Use Cases. Consider training a model with a large dataset such as ImageNet-1K
[6] or its augmented version which would require data to be distributed over p workers and uses
back propagation. Indeed, in this case, gi’s are typically the gradients computed by individual
workers at each iteration. In settings where the training objective is convex, the convergence and
generalization properties of distributed optimization can be achieved by defining A as a weighted
combination of gradients facilitated by a simple consensus matrix, even if some gi’s are noisy [7, 8].
In a distributed setup, as long as the model is convex we can simultaneously minimize the total
iteration or communication complexity to a significant extent i.e., it is possible to achieve convergence

Submitted to 37th Conference on Neural Information Processing Systems (NeurIPS 2023). Do not distribute.

Figure 1: Robust gradient aggregation in our distributed training framework. In our applications, each of
the p workers provides gradients computed using a random sample obtained from given training data, derived
synthetic data from off-the-shelf Diffusion models, and random noise in each iteration. Our Flag Aggregator
(FA) removes high frequency noise components by using few rounds of Singular Value Decomposition of the
concatenated Gradient Matrix G, and provides new update Y ∗.

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

and robustness under technical assumptions on the moments of (unknown) distribution from which
gi’s are drawn. However, it is still an open problem to determine the optimality of these procedures
in terms of either convergence or robustness [9, 10].

Potential Causes of Noise. When data is distributed among workers, hardware and software failures
in workers [11, 12, 13] can cause them to send incorrect gradients, which can significantly mislead
the model [14]. To see this, let’s consider a simple experiment with 15 workers, that f of them
produce uniformly random gradients. Figure 2 shows that the model accuracy is heavily impacted
when f > 0 when mean is used to aggregate the gradients.

The failures can occur due to component or software failures and
their probability increases with the scale of the system [15, 16, 17].
Reliability theory is used to analyze such failures, see Chapter 9
in [18], but for large-scale training, the distribution of total system
failures is not independent over workers, making the total noise in
gradients dependent and a key challenge for large-scale training.
Moreover, even if there are no issues with the infrastructure, our
work is motivated by the prevalence of data augmentation, including
hand-chosen augmentations. Since number of parameters n is often
greater than number of samples, data augmentation improves the
generalization capabilities of large-scale models under technical con-
ditions [19, 20, 21]. In particular, Adversarial training is a common
technique that finds samples that are close to training samples but
classified as a different class at the current set of parameters, and
then use such samples for parameter update purposes [22]. Unfortunately, computing adversarial
samples is often difficult [23], done using randomized algorithms [24] and so may introduce depen-
dent (across samples) noise themselves. In other words, using adversarial training paradigm, or the
so-called inner optimization can lead to noise in gradients, which can cause or simulate dependent
“Byzantine” failures in the distributed context.

Figure 2: Tolerance to f
Byzantine workers for a non-
robust aggregator (mean).

Available Computational Solutions. Most existing open source implementations of A rely just
on (functions of) pairwise distances to filter gradients from workers using suitable neighborhood
based thresholding schemes, based on moment conditions [25, 26, 27]. While these may be a good
strategy when the noise in samples/gradients is somewhat independent, these methods are suboptimal
when the noise is dependent or nonlinear, especially when n is large. Moreover, choosing discrete

2

…𝐺!:𝑤!×…𝑈∈𝑅"×"Σ∈𝑅"×$𝑉%∈𝑅$×$𝑌=𝑈[:,1:𝑚]𝐺=𝐺!|𝐺&|⋯|𝐺"××𝑌1𝑝𝑌𝑌%𝐺1𝑑𝐺'(,1≤𝑖≤𝑝,1≤𝑗≤𝑛SVD∗5𝑔!!𝑔!"𝑔!#𝑔"!𝑔""𝑔"#𝑔$!𝑔$"𝑔$#………𝑔!(𝑡)𝑔"(𝑡)𝑔#(𝑡)……𝑔!(𝑡+1)𝑔"(𝑡+1)𝑔#(𝑡+1)……𝑑)*!𝑑)𝑔'1≤𝑖≤𝑝Left	singular	vectorsiteration𝑡Augmented	DataStable	Diffusion𝑔!+𝑛,𝑛~𝑁(0,𝜎"𝐼)𝑔!+𝑛,𝑛~𝑁(0,𝑊)Augmented	DataStable	Diffusion𝑔!+𝑛,𝑛~𝑁(0,𝜎"𝐼)𝑔!+𝑛,𝑛~𝑁(0,𝑊)iteration𝑡+1Right	singular	vectorsSingular	valuesConcatenated	gradient	matrixGradients	from	workersWeights	for	workers’	gradient	subspaces𝐺&:𝑤&×𝐺":𝑤"×…Flag	AggregatorEstimate	Subspace	for	Aggregation01020304050Epoch102030405060Top-1 Accuracy (%)Mean, f=0Mean, f=1Mean, f=2Mean, f=367

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

hyperparameters such as number of neighbors is impractical in our use cases since they hamper
convergence of the overall training procedure. To mitigate the suboptimality of existing aggregation
schemes, we explicitly estimate a subspace Y spanned by “most” of the gradient workers, and then
use this subspace to estimate that a sparse linear combination of gi gradients, acheiving robustness.

We present a new optimization based formulation for generalized gradient aggregation purposes in
the context of distributed training of deep learning architectures, as shown in Figure 1.

Summary of our Contributions. From the theoretical perspective, we present a simple Maximum
Likelihood Based estimation procedure for aggregation purposes, with novel regularization functions.
Algorithmically, we argue that any procedure used to solve Flag Optimization can be directly used to
obtain the optimal summary statistic Y ∗ for our aggregation purposes. Experimentally, our results
show resilience against Byzantine attacks, encompassing physical failures, while effectively managing
the stochasticity arising from data augmentation schemes. In practice, we achieve a significantly
(≈ 20%) better accuracy on standard datasets. Our implementation offers substantial advantages in
reducing communication complexity across diverse noise settings through the utilization of our novel
aggregation function, making it applicable in numerous scenarios.

2 Robust Aggregators as Orthogonality Constrained Optimization

In this section, we first provide the basic intuition of our proposed approach to using subspaces for
aggregation purposes using linear algebra, along with connections of our approach standard eigende-
composition based denoising approaches. We then present our overall optimization formulation in
two steps, and argue that it can be optimized using existing methods.

2.1 Optimal Subspace Hypothesis for Distributed Descent

We will use lowercase letters y, g to denote vectors, and uppercase letters Y, G to denote ma-
trices. We will use boldfont 1 to denote the vector of all ones in appropriate dimensions.
Let gi ∈ Rn is the gradient vector from worker i, and Y ∈ Rn×m
is an orthogonal matrix representation of a subspace that gradients
could live in such that m ≤ p. Now, we may interpret each column
of Y as a basis function that act on gi ∈ Rn, i.e., j−th coordinate of
(Y T g)j for 1 ≤ j ≤ m is the application of j−th basis or column
of Y on g. Recall that by definition of dot product, we have that
if Y:,j ⊥ x, then (Y T g)j will be close to zero. Equivalently, if
g ∈ span(Y ), then (Y T g)T Y T g will be bounded away from zero,
see Chapter 2 in [28]. Assuming that G ∈ Rn×p is the gradient
matrix of p workers, Y Y T G ∈ Rn×p is the reconstruction of G
using Y as basis. That is, ith column of Y T G specifies the amount
of gradient from worker i as a function of Y , and high l2 norm of
Y T gi implies that there is a basis in Y such that Y ̸⊥ gi. So it is
easy to see that the average over columns of Y Y T G would give the final gradient for update.
Explained Variance of worker i. If we denote zi = Y T gi ∈ Rm representing the transformation
i Y Y T gi is a scalar,
2 = zT
of gradient gi to zi using Y , then, 0 ≤ ∥zi∥2
(cid:1). Moreover, when Y is orthogonal, we have 0 ≤ ∥zi∥2 =
and so is equal to its trace tr (cid:0)gT
∥Y T gi∥2 ≤ ∥Y ∥2∥gi∥2 ≤ ∥gi∥2 since the operator norm (or largest singular value) ∥Y ∥2 of Y is at
most 1. Our main idea is to use ∥zi∥2
2 to define the quality of the subspace Y for aggregation,
as is done in some previous works for Robust Principal Component Estimation [29] – the quantity
2/∥gi∥2
∥zi∥2
2 is called as Explained/Expressed variance of subspace Y wrt i−th worker [30, 31] – we
2/∥gi∥2
refer to ∥zi∥2
2 as the “value” of i−th worker. In Figure 3, we can see from the spike near 1.0
that if we choose the subspace carefully (blue) as opposed to merely choosing the mean gradient
(with unit norm) of all workers, then we can increase the value of workers.

Figure 3: Distributions of Ex-
plained Variances on Minibatches

i zi = (Y T g)T Y T g = gT

i Y Y T gi

2, ∥gi∥2

Advantages of Subspace based Aggregation. We can see that using subspace Y , we can easily: 1.
handle different number of gradients from each worker, 2. compute gradient reconstruction Y Y T G
efficiently whenever Y is constrained to be orthogonal Y = (cid:80)
i where yi is the i−th column
of Y , otherwise have to use eigendecomposition of Y to measure explained variance which can
be time consuming. In (practical) distributed settings, the quality (or noise level) of gradients in

i yiyT

3

0.00.20.40.60.81.0Value of Workers (v)025050075010001250150017502000FrequencyOptimalSubspaceSuboptimalSubspace119

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

i zi) instead.

each worker may be different, and/or each worker may use a different batch size. In such cases,
handcrafted aggregation schemes may be difficult to maintain, and fine-tune. For these purposes with
an Orthogonal Subspace Y , we can simply reweigh gradients of worker i according to its noise level,
and/or use gi ∈ Rn×bi where bi is the batch size of i−th worker with tr(zT
Why is optimizing over subspaces called “Flag” Optimization? Recent optimization results
suggest that we can exploit the finer structure available in Flag Manifold to specify Y more precisely
[32]. For example, Y ∈ Rm×n can be parametrized directly as a subspace of dimension m or
as a nested sequence of Yk ∈ Rmk×n, k = 1, ..., K where mk < mk+1 ≤ p ≤ n such that
span(Yk) ⊆ span(Yk+1) with YK ∈ Rm×n. When mk+1 = mk = 1, we have the usual (real)
Grassmanian Manifold (quotient of orthogonal group) whose coordinates can be used for optimization,
please see Section 5 in [33] for details. In fact, [34] used this idea to extend median in one-dimensional
vector spaces to different finite dimensional subspaces using the so-called chordal distance between
them. In our distributed training context, we use the explained variance of each worker instead. Here,
workers may specify dimensions along which gradient information is relevant for faster convergence
– an advantage currently not available in existing aggregation implementations – which may be used
for smart initialization also. We use “Flag” to emphasize this additional nested structure available in
our formulation for distributed training purposes.

136

2.2 Approximate Maximum Likelihood Estimation of Optimal Subspace

137

138

139

140

141

142

143

Now that we can evaluate a subspace Y on individual gradients gi, we now show that finding subspace
Y can be formulated using standard maximum likelihood estimation principles [35]. Our formulation
reveals that regularization is critical for aggregation especially in distributed training. In order to
write down the objective function for finding optimal Y , we proceed in the following two steps:

Step 1. Assume that each worker provides a single gradient for simplicity. Now, denoting the value of
information v of worker i by vi = zT
i zi
, we have vi ∈ [0, 1]. Now by assuming that vi’s are observed
gT
i gi
2 (for simplicity), we can see that the likelihood P(vi) is,
from Beta distribution with α = 1 and β = 1

P(vi) :=

(cid:16)

2

(1 − vi)− 1
B(1, 1
2 )

=

(cid:17)− 1

2

1 − zT
i zi
gT
i gi
B(1, 1
2 )

,

(2)

144

145

where B(a, b) is the normalization constant. Then, the total log-likelihood of observing gradients gi
as a function of Y (or vi’s) is given by taking the log of product of P(vi)’s as (ignoring constants),

log

(cid:32) p
(cid:89)

i=1

(cid:33)

P(vi)

=

p
(cid:88)

i=1

log (P(vi)) = −

1
2

p
(cid:88)

i=1

log(1 − vi).

(3)

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

Step 2. Now we use Taylor’s series with constant a > 0 to approximate individual worker log-
likelihoods log(1 − vi) ≈ a(1 − vi) 1
i . On
the other hand, using Taylor expansion of exp about the origin (so large a > 1 is better), we have that
exp
i which immediately implies

. Whence, we have that 1 + log(vi)

a − a as follows: first, we know that exp

≈ 1 + log(vi)

(cid:16) log(vi)
a

(cid:16) log(vi)
a

a ≈ v

= v

(cid:17)

(cid:17)

1
a

1
a

a

that log(vi) ≈ av
i − a. So, by substituting the Taylor series approximation of log in Equation 3, we
obtain the negative log-likelihood approximation to be minimized for robust aggregation purposes as,

1
a

− log

(cid:33)

P(vi)

≈

(cid:32) p
(cid:89)

i=1

1
2

p
(cid:88)

(cid:16)

i=1

a (1 − vi)

(cid:17)

1
a − a

,

(4)

where a > 1 is a sufficiently large constant. In the above mentioned steps, the first step is standard.
Our key insight is using Taylor expansion in (4) with a sufficiently large a to eliminate log optimization
which are known to be computationally expensive to solve, and instead solve smooth ℓa, a > 1 norm
based optimization problems which can be done efficiently by modifying existing procedures [36].
Extension to general beta distributions, and gradients α > 0, β > 0, gi ∈ Rn×k. Note that our
derivation in the above two steps can be extended to any beta shape parameters α > 0, β > 0 – there
will be two terms in the final negative log-likelihood expression in our formulation (4), one for each
α, β. Similarly, by simply using vi = tr (cid:0)gT
(cid:1) to define value of worker i in equation (2), and
then in our estimator in (4), we can easily handle multiple k gradients from a single worker i for Y .

i Y Y T gi

4

Algorithm 1 Distributed SGD with proposed Flag Aggregator (FA) at the Parameter Server
Input: Number of workers p, loss functions l1, l2, ..., lp, per-worker minibatch size B, learning rate

schedule αt, initial parameters w0, number of iterations T

Output: Updated parameters wT from any worker

1 for t = 1 to T do
2

for p = 1 to p in parallel on machine p do

3

4

5

6

7

8

Select a minibatch: ip,1,t, ip,2,t,. . . ,ip,B,t gp,t ← 1
B

(cid:80)B

b=1 ∇lip,b,t (wt−1)

Gt ← {g1,t, · · · , gp,t} // Parameter Server receives gradients from p workers
ˆYt ← IRLS( ˆGt) with ˆGt = Gt + λ∇R(Y )1T // Do IRLS at the Parameter Server for ˆY
Obtain gradient direction dt: dt = 1
p
for p = 1 to p in parallel on machine p do
update model: wt ← wt−1 − αt · dt

t Gt1 // Compute, Send dt to all p machines

ˆYt ˆY T

9 Return wT

161

2.3 Flag Aggregator for Distributed Optimization

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

i=1

(cid:112)1 − gT

It is now easy to see that by choosing a = 2, in equation (4), we obtain the negative loglikelihood
(ignoring constants) as ((cid:80)p
i Y Y T gi) showing that Flag Median can indeed be seen as
an Maximum Likelihood Estimator (MLE). In particular, Flag Median can be seen as an MLE of
Beta Distribution with parameters α = 1 and β = 1
2 . Recent results suggest that in many cases, MLE
is ill-posed, and regularization is necessary, even when the likelihood distribution is Gaussian [37].
So, based on the Flag Median estimator for subspaces, we propose an optimization based subspace
estimator Y ∗ for aggregation purposes. We formulate our Flag Aggregator (FA) objective function
with respect to Y as a regularized sum of likelihood based (or data) terms in (4) using trace operators
tr(·) as the solution to the following constrained optimization problem:

min
Y :Y T Y =I

A(Y ) :=

p
(cid:88)

i=1

(cid:32)

(cid:118)
(cid:117)
(cid:117)
(cid:116)

1 −

(cid:33)

i Y (cid:1)

tr (cid:0)Y T gigT
∥gi∥2
2

+ λR(Y )

(5)

where λ > 0 is a regularization hyperparameter. In our analysis, and implementation, we provide
support for two possible choices for R(Y ):

(1) Mathematical norms: R(Y ) can be a form of norm-based regularization other than ∥Y ∥2

Fro since
it is constant over the feasible set in (5). For example, it could be convex norm with efficient
subgradient oracle such as, i.e. element-wise: (cid:80)n

j=1 ∥Yij∥1 or (cid:80)m

i=1 ∥Yi,i∥1,

(cid:80)m

i=1

(2) Data-dependent norms: Following our subspace construction in Section 2.1, we may choose

(cid:114)(cid:16)

(cid:80)p

i,j=1,i̸=j

1 − tr(Y T (gi−gj )(gi−gj )T Y )

R(Y ) = 1
p−1

2 denotes the
distance between gradient vectors gi, gj from workers i, j. Intuitively, the pairwise terms in our
loss function (5) favors subspace Y that also reconstructs the pairwise vectors gi − gj that are close
to each other. So, by setting λ = Θ(p), that is, the pairwise terms dominate the objective function
in (5). Hence, λ regularizes optimal solutions Y ∗ of (5) to contain gi’s with low pairwise distance
in its span – similar in spirit to AggregaThor in [38].

ij = ∥gi − gj∥2

where D2

D2
ij

(cid:17)

Convergence of Flag Aggregator (FA) Algorithm 1. With these, we can state our main algorithmic
result showing that our FA (5) can be solved efficiently using standard convex optimization proof
techniques. In particular, in supplement, we present a smooth Semi-Definite Programming (SDP)
relaxation of FA in equation (5) using the Flag structure. This allows us to view the IRLS procedure
in 1 as solving the low rank parametrization of the smooth SDP relaxation, thus guaranteeing fast
convergence to second order optimal (local) solutions. Importantly, our SDP based proof works for
any degree of approximation of the constant a in equation (4) and only relies on smoothness of the
loss function wrt Y , although speed of convergence is reduced for higher values of a ̸= 2, see [39].
We leave determining the exact dependence of a on rate of convergence for future work.

How is FA aggregator different from (Bulyan and Multi-Krum)? Bulyan is a strong Byzantine
resilient gradient aggregation rule for p ≥ 4f + 3 where p is the total number of workers and f is

5

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

the number of Byzantine workers. Bulyan is a two-stage algorithm. In the first stage, a gradient
aggregation rule R like coordinate-wise median [40] or Krum [9] is recursively used to select
θ = p − 2f gradients. The process uses R to select gradient vector gi which is closest to R’s output
(e.g. for Krum, this would be the gradient with the top score, and hence the exact output of R). The
chosen gradient is removed from the received set and added to the selection set S repeatedly until
|S| = θ. The second stage produces the resulting gradient. If β = θ − 2f , each coordinate would
be the average of β-nearest to the median coordinate of the θ gradients in S. In matrix terms, if we
consider S ∈ Rp×m as a matrix with each column having one non-zero entry summing to 1, Bulyan
m ReLU(GS)1m, where 1m ∈ Rm is the vector of all ones, while FA would return
would return 1
1
p Y Y T G1p. Importantly, the gradient matrix is being right-multiplied in Bulyan, but left-multiplied
in FA, before getting averaged. While this may seem like a discrepancy, in supplement we show that
by observing the optimality conditions of (5) wrt Y , we show that 1
m Y Y T G can be seen as a right
multiplication by a matrix parametrized by lagrangian multipliers associated with the orthogonality
constraints in (5). This means it should be possible to combine both approaches for faster aggregation.

208

3 Experiments

209

210

211

212

213

214

215

216

217

In this section, we conduct experiments to test our proposed FA in the context of distributed training
in two testbeds. First, to test the performance of our FA scheme solved using IRLS (Flag Mean) on
standard Byzantine benchmarks. Then, to evaluate the ability of existing state-of-the-art gradient
aggregators we augment data via two techniques that can be implemented with Sci-kit package.

Implementation Details. We implement FA in Pytorch [41], which is popular but does not support
Byzantine resilience natively. We adopt the parameter server architecture and employ Pytorch’s
distributed RPC framework with TensorPipe backend for machine-to-machine communication. We
extend Garfield’s Pytorch library [42] with FA and limit our IRLS convergence criteria to a small
error, 10−10, or 5 iterations of flag mean for SVD calculation. We set m = ⌈ p+1

2 ⌉.

218

3.1 Setup

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

Baselines: We compare FA to several existing aggregation rules: (1) coordinate-wise Trimmed
Mean [40] (2) coordinate-wise Median [40] (3) mean-around-median (MeaMed) [43] (4) Phocas
[44] (5) Multi-Krum [9] (6) Bulyan [45].

Accuracy: The fraction of correct predictions among all predictions, using the test dataset (top-1
cross-accuracy).

Testbed: We used 4 servers as our experimental platform. Each server has 2 Intel(R) Xeon(R) Gold
6240 18-core CPU @ 2.60GHz with Hyper-Threading and 384GB of RAM. Servers have a Tesla
V100 PCIe 32GB GPU and employ a Mellanox ConnectX-5 100Gbps NIC to connect to a switch.
We use one of the servers as the parameter server and instantiate 15 workers on other servers, each
hosting 5 worker nodes, unless specified differently in specific experiments. For the experiments
designed to show scalability, we instantiate 60 workers.

Dataset and model: We focus on the image classification task since it is a widely used task for
benchmarking in distributed training [46]. We train ResNet-18 [47] on CIFAR-10 [48] which has
60,000 32 × 32 color images in 10 classes. For the scalability experiment, we train a CNN with two
convolutional layers followed by two fully connected layers on MNIST [49] which has 70,000 28 ×
28 grayscale images in 10 classes. We also run another set of experiments on Tiny ImageNet [50] in
the supplement. We use SGD as the optimizer, and cross-entropy to measure loss. The batch size
for each worker is 128 unless otherwise stated. Also, we use a learning decay strategy where we
decrease the learning rate by a factor of 0.2 every 10 epochs.

Threat models: We evaluate FA under two classes of Byzantine workers. They can send uniformly
random gradients that are representative of errors in the physical setting, or use non-linear augmented
data described as below.

Evaluating resilience against nonlinear data augmentation: In order to induce Byzantine behavior
in our workers we utilize ODE solvers to approximately solve 2 non-linear processes, Lotka Volterra

6

(a) f = 1

(b) f = 2

(c) f = 3

Figure 4: Tolerance to the number of Byzantine workers for robust aggregators for batch size 128.

[51] and Arnold’s Cat Map [52], as augmentation methods. Since the augmented samples are
deterministic, albeit nonlinear functions of training samples, the “noise” is dependent across samples.

In Lotka Volterra, we use the following linear gradient transformation of 2D pixels:

(x, y) → (αx − βxy, δxy − γy),

where α, β, γ and δ are hyperparameters. We choose them to be 2

3 , 4

3 , −1 and −1 respectively.

Second, we use a nonsmooth transformation called Arnold’s Cat Map as a data augmentation scheme.
Once again, the map can be specified using a two-dimensional matrix as,

(x, y) →

(cid:18) 2x + y
N

,

x + y
N

(cid:19)

mod 1,

243

244

245

246

247

248

249

250

251

where mod represents the modulus operation, x and y are the coordinates or pixels of images and N
is the height/width of images (assumed to be square). We also used a smooth approximation of the
Cat Map obtained by approximating the mod function as,

(x, y) →

1
n
n , α2 = x+y

where α1 = 2x+y
our data augmentation experiments.

(cid:18)

2x + y
(1 + exp(−m log(α1)

,

x + y
(1 + exp(−m log(α2)

(cid:19)

,

n , and m is the degree of approximation, which we choose to be 0.95 in

How to perform nonlinear data augmentation? In all three cases, we used SciPy’s [53] solve_ivp
method to solve the differential equations, by using the LSODA solver. In addition to the setup
described above, we also added a varying level of Gaussian noise to each of the training images. All
the images in the training set are randomly chosen to be augmented with varying noise levels of the
above mentioned augmentation schemes. We have provided the code that implements all our data
augmentation schemes in the supplement zipped folder.

252

253

254

255

256

257

258

259

260

3.2 Results

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

Tolerance to the number of Byzantine workers: In this experiment, we show the effect of Byzantine
behavior on the convergence of different gradient aggregation rules in comparison to FA. Byzantine
workers send random gradients and we vary the number of them from 1 to 3. Figure 4 shows that for
some rules, i.e. Trimmed Mean, the presence of even a single Byzantine worker has a catastrophic
impact. For other rules, as the number of Byzantine workers increases, filtering out the outliers
becomes more challenging because the amount of noise increases. Regardless, FA remains more
robust compared to other approaches.

Marginal utility of larger batch sizes under a fixed noise level:
We empirically verified the batch size required to identify our optimal Y ∗ - the FA matrix at each
iteration. In particular, we fixed the noise level to f = 3 Byzantine workers and varied batch sizes.
We show the results in Figure 5. Our results indicate that, in cases where a larger batch size is
a training requirement, FA achieves a significantly better accuracy compared to the existing
state of the art aggregators. This may be useful in some large scale vision applications, see [54, 55]
for more details. Empirically, we can already see that our spectral relaxation to identify gradient
subspace is effective in practice in all our experiments.

7

Flag AggregatorBulyanMulti-KrumMeaMedMedianTrimmed MeanPhocas010203040Epoch10203040506070Top-1 Accuracy (%)010203040Epoch102030405060Top-1 Accuracy (%)010203040Epoch1020304050Top-1 Accuracy (%)(a) bs = 64

(b) bs = 128

(c) bs = 192

Figure 5: Marginal utility of larger batch sizes under a fixed noise level f = 3.

(a) p = 15, f = 3

(b) p = 11, f = 2

(c) p = 13, f = 2

(d) p = 15, f = 2

Figure 6: We present results under two different gradient attacks. The attack in (a) corresponds to
simply dropping 10% of gradients from f workers. The attacks in (b)-(d) correspond to generic f
workers sending random gradient vectors, i.e. we simply fix noise level while adding more workers.

276

277

278

279

280

281

Tolerance to communication loss: To analyze the effect of unreliable communication channels
between the workers and the parameter server on convergence, we design an experiment where the
physical link between some of the workers and the parameter server randomly drops a percentage of
packets. Here, we set the loss rate of three links to 10% i.e., there are 3 Byzantine workers in our
setting. The loss is introduced using the netem queuing discipline in Linux designed to emulate the
properties of wide area networks [56]. The two main takeaways in Figure 6a are:

1. FA converges to a significantly higher accuracy than other aggregators, and thus is more
robust to unreliable underlying network transports.
2. Considering time-to-accuracy for comparison, FA reaches a similar accuracy in less total
number of training iterations, and thus is more robust to slow underlying network transports.

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

Analyzing the marginal utility of additional workers. To see the effect of adding more workers
to a fixed number of Byzantine workers, we ran experiments where we fixed f , and increased p.
Our experimental results shown in Figures 6b-6d indicate that our FA algorithm possesses strong
resilience property for reasonable choices of p.

The effect of having augmented data during training in Byzantine workers: Figure 7 shows FA
can handle nonlinear data augmentation in a much more stable fashion. Please see supplement for
details on the level of noise, and exact solver settings that were used to obtain augmented images.

The effect of the regularization parameter in FA: The data-dependent regularization parameter λ
in FA provides flexibility in the loss function to cover aggregators that benefit from pairwise distances
such as Bulyan and Multi-Krum. To verify whether varying λ can interpolate Bulyan and Multi-Krum,
we change λ in Figure 8. We can see when FA improves or performs similarly for a range of λ. Here,
we set p and f to satisfy the strong Byzantine resilience condition of Bulyan, i.e, p ≥ 4f + 3.

Scaling out to real-world situations with more workers: In distributed ML, p and f are usually
large. To test high-dimensional settings commonly dealt in Semantic Vision with our FA, we used
ResNet-18. Now, to specifically test the scalability of FA, we fully utilized our available GPU servers
and set up to p = 60 workers (up to f = 14 Byzantine) with the MNIST dataset and a simple CNN
with two convolutional layers followed by two fully connected layers (useful for simple detection).
Figure 9 shows evidence that FA is feasible for larger setups.

8

Flag AggregatorBulyanMulti-KrumMeaMedMedianTrimmed MeanPhocas0102030Epoch102030405060Top-1 Accuracy (%)0102030Epoch1020304050Top-1 Accuracy (%)0102030Epoch1020304050Top-1 Accuracy (%)Flag AggregatorBulyanMulti-KrumMeaMedMedianTrimmed MeanPhocas01020Epoch102030405060Top-1 Accuracy (%)0102030Epoch102030405060Top-1 Accuracy (%)0102030Epoch102030405060Top-1 Accuracy (%)0102030Epoch102030405060Top-1 Accuracy (%)Figure 7: Accuracy of us-
ing augmented data in f =
3 workers

Figure 8: CIFAR10 with
ResNet-18, p = 7, and
f = 1

Figure 9: Scaling FA to
larger setups

(a) Cropped Time-to-accuracy

(b) Iteration time

(c) Total Time-to-accuracy

Figure 10: Wall clock time comparison

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

4 Discussion and Limitation

Is it possible to fully “offload” FA computation to switches? Recent work propose that aggregation
be performed entirely on network infrastructure to alleviate any communication bottleneck that may
arise [57, 58]. However, to the best of our knowledge, switches that are in use today only allow
limited computation to be performed on gradient gi as packets whenever they are transmitted [59, 60].
That is, programmability is restrictive at the moment— switches used in practice have no floating
point, or loop support, and are severely memory/state constrained. Fortunately, solutions seem near.
For instance, [61] have already introduced support for floating point arithmetic in programmable
switches. We may use quantization approaches for SVD calculation with some accuracy loss [62] to
approximate floating point arithmetic. Offloading FA to switches has great potential in improving
its computational complexity because the switch would perform as a high-throughput streaming
parameter server to synchronize gradients over the network. Considering that FA’s accuracy currently
outperforms its competition in several experiments, an offloaded FA can reach their accuracy even
faster or it could reach a higher accuracy in the same amount of time.

Potential Limitation. Because in every iteration of FA, we perform SVD, the complexity of the
algorithm would be O(nNδ((cid:80)p
i=1 ki)2) with Nδ being the number of iterations for the algorithm.
Figure 10 show the wall clock time it takes for FA to reach a certain accuracy (10a) or epoch(10b)
compared to other methods under a fixed amount of random noise f = 3 with p = 15 workers.
Although the iteration complexity of FA is higher, here each iteration has a higher utility as reflected in
the time-to-accuracy measures. This makes FA comparable to others in a shorter time span, however,
if there is more wall clock time to spare, FA converges to a better state as shown in Figure 10c where
we let the same number of total iterations finish for all methods.

5 Conclusion

In this paper we proposed Flag Aggregator (FA) that can be used for robust aggregation of gradients
in distributed training. FA is an optimization-based subspace estimator that formulates aggregation as
a Maximum Likelihood Estimation procedure using Beta densities. We perform extensive evaluations
of FA and show it can be effectively used in providing Byzantine resilience for gradient aggregation.
Using techniques from convex optimization, we theoretically analyze FA and with tractable relaxations
show its amenability to be solved by off-the-shelf solvers or first-order reweighing methods.

9

Flag AggregatorBulyanMulti-KrumMeaMedMedianTrimmed MeanPhocas0102030Epoch102030405060Top-1 Accuracy (%)05101520Epoch10203040506070Top-1 Accuracy (%)FA, =0FA, =4FA, =8FA, =16BulyanMulti-Krum05101520Epoch20406080100Top-1 Accuracy (%)FA, p=15,f=3FA, p=30,f=6FA, p=60,f=122004006008001000Time (s)1020304050Top-1 Accuracy (%)Flag AggregatorBulyanMulti-KrumMeaMedMedianTrimmed MeanPhocas010203040Epoch050010001500200025003000Time (s)Flag AggregatorBulyanMulti-KrumMeaMedMedianTrimmed MeanPhocas50010001500200025003000Time (s)1020304050Top-1 Accuracy (%)Flag AggregatorBulyanMulti-KrumMeaMedMedianTrimmed MeanPhocas329

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

References

[1] Michel Grabisch, Jean-Luc Marichal, Radko Mesiar, and Endre Pap. Aggregation functions,

volume 127. Cambridge University Press, 2009.

[2] Sivaraman Balakrishnan, Simon S. Du, Jerry Li, and Aarti Singh. Computationally effi-
cient robust sparse estimation in high dimensions.
In Satyen Kale and Ohad Shamir, edi-
tors, Proceedings of the 2017 Conference on Learning Theory, volume 65 of Proceedings
of Machine Learning Research, pages 169–212. PMLR, 07–10 Jul 2017. URL https:
//proceedings.mlr.press/v65/balakrishnan17a.html.

[3] Ilias Diakonikolas, Daniel Kane, Sushrut Karmalkar, Eric Price, and Alistair Stewart. Outlier-
robust high-dimensional sparse estimation via iterative filtering. Advances in Neural Information
Processing Systems, 32, 2019.

[4] Yu Cheng, Ilias Diakonikolas, Rong Ge, Shivam Gupta, Daniel M. Kane, and Mahdi
Soltanolkotabi. Outlier-robust sparse estimation via non-convex optimization. Advances
in Neural Information Processing Systems, 2022.

[5] Ilias Diakonikolas, Daniel M. Kane, Sushrut Karmalkar, Ankit Pensia, and Thanasis Pittas.
Robust sparse mean estimation via sum of squares. In Po-Ling Loh and Maxim Raginsky,
editors, Proceedings of Thirty Fifth Conference on Learning Theory, volume 178 of Proceedings
of Machine Learning Research, pages 4703–4763. PMLR, 02–05 Jul 2022. URL https:
//proceedings.mlr.press/v178/diakonikolas22e.html.

[6] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng
Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei.
ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision
(IJCV), 115(3):211–252, 2015. doi: 10.1007/s11263-015-0816-y.

[7] Konstantinos I Tsianos and Michael G Rabbat. Distributed strongly convex optimization. In
2012 50th Annual Allerton Conference on Communication, Control, and Computing (Allerton),
pages 593–600. IEEE, 2012.

[8] Tao Yang, Xinlei Yi, Junfeng Wu, Ye Yuan, Di Wu, Ziyang Meng, Yiguang Hong, Hong Wang,
Zongli Lin, and Karl H Johansson. A survey of distributed optimization. Annual Reviews in
Control, 47:278–305, 2019.

[9] Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer. Machine learning
with adversaries: Byzantine tolerant gradient descent. In Proceedings of the 31st Interna-
tional Conference on Neural Information Processing Systems, NIPS’17, page 118–128. Curran
Associates Inc., 2017. ISBN 9781510860964.

[10] Sadegh Farhadkhani, Rachid Guerraoui, Nirupam Gupta, Rafael Pinot, and John Stephan.
Byzantine machine learning made easy by resilient averaging of momentums. In Kamalika
Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, edi-
tors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of
Proceedings of Machine Learning Research, pages 6246–6283. PMLR, 17–23 Jul 2022. URL
https://proceedings.mlr.press/v162/farhadkhani22a.html.

[11] Leonardo Bautista-Gomez, Ferad Zyulkyarov, Osman Unsal, and Simon McIntosh-Smith.
Unprotected computing: A large-scale study of dram raw error rate on a supercomputer. In SC
’16: Proceedings of the International Conference for High Performance Computing, Networking,
Storage and Analysis, pages 645–655, 2016. doi: 10.1109/SC.2016.54.

[12] Bianca Schroeder and Garth A. Gibson. Disk failures in the real world: What does an mttf
of 1,000,000 hours mean to you? In Proceedings of the 5th USENIX Conference on File and
Storage Technologies, FAST ’07, page 1–es, USA, 2007. USENIX Association.

[13] Phillipa Gill, Navendu Jain, and Nachiappan Nagappan. Understanding network failures
in data centers: Measurement, analysis, and implications. SIGCOMM Comput. Commun.
Rev., 41(4):350–361, aug 2011. ISSN 0146-4833. doi: 10.1145/2043164.2018477. URL
https://doi.org/10.1145/2043164.2018477.

10

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

[14] Gilad Baruch, Moran Baruch, and Yoav Goldberg. A little is enough: Circumventing defenses
for distributed learning. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox,
and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32.
Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper/2019/
file/ec1c59141046cd1866bbbcdfb6ae31d4-Paper.pdf.

[15] Guosai Wang, Lifei Zhang, and Wei Xu. What can we learn from four years of data center
hardware failures? In 2017 47th Annual IEEE/IFIP International Conference on Dependable
Systems and Networks (DSN), pages 25–36, 2017. doi: 10.1109/DSN.2017.26.

[16] Devesh Tiwari, Saurabh Gupta, James Rogers, Don Maxwell, Paolo Rech, Sudharshan Vazhku-
dai, Daniel Oliveira, Dave Londo, Nathan DeBardeleben, Philippe Navaux, Luigi Carro, and
Arthur Bland. Understanding gpu errors on large-scale hpc systems and the implications for
system design and operation. In 2015 IEEE 21st International Symposium on High Performance
Computer Architecture (HPCA), pages 331–342, 2015. doi: 10.1109/HPCA.2015.7056044.

[17] Bin Nie, Devesh Tiwari, Saurabh Gupta, Evgenia Smirni, and James H. Rogers. A large-scale
study of soft-errors on gpus in the field. In 2016 IEEE International Symposium on High
Performance Computer Architecture (HPCA), pages 519–530, 2016. doi: 10.1109/HPCA.2016.
7446091.

[18] Sheldon M Ross. Introduction to probability models. Academic press, 2014.

[19] Fanny Yang, Zuowen Wang, and Christina Heinze-Deml.

Invariance-inducing regulariza-
tion using worst-case transformations suffices to boost accuracy and spatial robustness.
In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Gar-
nett, editors, Advances in Neural Information Processing Systems, volume 32. Curran
Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper/2019/file/
1d01bd2e16f57892f0954902899f0692-Paper.pdf.

[20] Christina Heinze-Deml and Nicolai Meinshausen. Conditional variance penalties and domain

shift robustness, 2017. URL https://arxiv.org/abs/1710.11469.

[21] Saeid Motiian, Marco Piccirilli, Donald A. Adjeroh, and Gianfranco Doretto. Unified deep su-
pervised domain adaptation and generalization. In IEEE International Conference on Computer
Vision (ICCV), 2017.

[22] Sravanti Addepalli, Samyak Jain, et al. Efficient and effective augmentation strategy for
adversarial training. Advances in Neural Information Processing Systems, 35:1488–1501, 2022.

[23] Eric Wong, Leslie Rice, and J Zico Kolter. Fast is better than free: Revisiting adversarial

training. arXiv preprint arXiv:2001.03994, 2020.

[24] Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized
smoothing. In international conference on machine learning, pages 1310–1320. PMLR, 2019.

[25] Youssef Allouah, Rachid Guerraoui, Nirupam Gupta, Rafael Pinot, and John Stephan. Dis-
tributed learning with curious and adversarial machines. arXiv preprint arXiv:2302.04787,
2023.

[26] Youssef Allouah, Sadegh Farhadkhani, Rachid Guerraoui, Nirupam Gupta, Rafaël Pinot, and
John Stephan. Fixing by mixing: A recipe for optimal byzantine ml under heterogeneity. In
International Conference on Artificial Intelligence and Statistics, pages 1232–1300. PMLR,
2023.

[27] Sadegh Farhadkhani, Rachid Guerraoui, Nirupam Gupta, Rafael Pinot, and John Stephan.
Byzantine machine learning made easy by resilient averaging of momentums. In International
Conference on Machine Learning, pages 6246–6283. PMLR, 2022.

[28] P-A Absil. Optimization algorithms on matrix manifolds. Princeton University Press, 2008.

[29] John Wright, Arvind Ganesh, Shankar Rao, Yigang Peng, and Yi Ma. Robust principal
component analysis: Exact recovery of corrupted low-rank matrices via convex optimization.
Advances in neural information processing systems, 22, 2009.

11

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

[30] Matthias Hein and Thomas Bühler. An inverse power method for nonlinear eigenproblems with
applications in 1-spectral clustering and sparse pca. Advances in neural information processing
systems, 23, 2010.

[31] Rudrasis Chakraborty, Soren Hauberg, and Baba C Vemuri. Intrinsic grassmann averages for
online linear and robust subspace learning. In Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition, pages 6196–6204, 2017.

[32] D. Monk. The geometry of flag manifolds. Proceedings of the London Mathematical So-
ciety, s3-9(2):253–286, 1959. doi: https://doi.org/10.1112/plms/s3-9.2.253. URL https:
//londmathsoc.onlinelibrary.wiley.com/doi/abs/10.1112/plms/s3-9.2.253.

[33] Ke Ye, Ken Sze-Wai Wong, and Lek-Heng Lim. Optimization on flag manifolds. Mathematical

Programming, 194(1-2):621–660, 2022.

[34] Nathan Mankovich, Emily J King, Chris Peterson, and Michael Kirby. The flag median
and flagirls. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 10339–10347, 2022.

442

[35] Kevin P Murphy. Probabilistic machine learning: an introduction. MIT press, 2022.

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

[36] Massimo Fornasier, Holger Rauhut, and Rachel Ward. Low-rank matrix recovery via iteratively
reweighted least squares minimization. SIAM Journal on Optimization, 21(4):1614–1640, 2011.

[37] Toni Karvonen and Chris J Oates. Maximum likelihood estimation in gaussian process regression

is ill-posed. Journal of Machine Learning Research, 24(120):1–47, 2023.

[38] Georgios Damaskinos, El-Mahdi El-Mhamdi, Rachid Guerraoui, Arsany Guirguis, and Sébastien
Rouault. Aggregathor: Byzantine machine learning via robust gradient aggregation. Proceedings
of Machine Learning and Systems, 1:81–106, 2019.

[39] Yuxin Chen, Yuejie Chi, Jianqing Fan, Cong Ma, and Yuling Yan. Noisy matrix completion:
Understanding statistical guarantees for convex relaxation via nonconvex optimization. SIAM
journal on optimization, 30(4):3098–3121, 2020.

[40] Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. Byzantine-robust distributed
learning: Towards optimal statistical rates. In Proceedings of the 35th International Conference
on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 5650–
5659. PMLR, 10–15 Jul 2018.

[41] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan,
Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas
Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy,
Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style,
high-performance deep learning library. In Advances in Neural Information Processing Systems,
volume 32, 2019.

[42] Rachid Guerraoui, Arsany Guirguis, Jérémy Plassmann, Anton Ragot, and Sébastien Rouault.
Garfield: System support for byzantine machine learning (regular paper). In 2021 51st Annual
IEEE/IFIP International Conference on Dependable Systems and Networks (DSN), pages 39–51,
2021. doi: 10.1109/DSN48987.2021.00021.

467

[43] Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. Generalized byzantine-tolerant sgd, 2018.

468

469

470

471

472

473

[44] Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. Phocas: dimensional byzantine-resilient

stochastic gradient descent, 2018.

[45] El Mahdi El Mhamdi, Rachid Guerraoui, and Sébastien Rouault. The hidden vulnerability of
distributed learning in Byzantium. In Proceedings of the 35th International Conference on
Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 3521–3530.
PMLR, 10–15 Jul 2018.

12

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

[46] Trishul Chilimbi, Yutaka Suzue, Johnson Apacible, and Karthik Kalyanaraman. Project
adam: Building an efficient and scalable deep learning training system. In Proceedings of the
11th USENIX Conference on Operating Systems Design and Implementation, OSDI’14, page
571–582, USA, 2014.

[47] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
pages 770–778, 2016. doi: 10.1109/CVPR.2016.90.

[48] Alex Krizhevsky. Learning multiple layers of features from tiny images, 2009. URL https:

//www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf.

[49] Yann LeCun and Corinna Cortes. MNIST handwritten digit database, 2010. URL http:

//yann.lecun.com/exdb/mnist/.

485

[50] Ya Le and Xuan S. Yang. Tiny imagenet visual recognition challenge, 2015.

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

517

518

519

520

[51] David Kelly. Rough path recursions and diffusion approximations. The Annals of Applied

Probability, 26(1):425–461, 2016.

[52] Jianghong Bao and Qigui Yang. Period of the discrete arnold cat map and general cat map.

Nonlinear Dynamics, 70(2):1365–1375, 2012.

[53] Fundamental algorithms for scientific computing in python. https://scipy.org/, 2023.

[54] Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping
Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima.
In International Conference on Learning Representations, 2017. URL https://openreview.
net/forum?id=H1oyRlYgg.

[55] Yang You, Jonathan Hseu, Chris Ying, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh.
Large-batch training for lstm and beyond. In Proceedings of the International Conference for
High Performance Computing, Networking, Storage and Analysis, SC ’19, 2019.

[56] Kevin Hsieh, Aaron Harlap, Nandita Vijaykumar, Dimitris Konomis, Gregory R. Ganger,
Phillip B. Gibbons, and Onur Mutlu. Gaia: Geo-distributed machine learning approaching lan
speeds. In Proceedings of the 14th USENIX Conference on Networked Systems Design and
Implementation, page 629–647, 2017.

[57] Amedeo Sapio, Marco Canini, Chen-Yu Ho, Jacob Nelson, Panos Kalnis, Changhoon Kim,
Arvind Krishnamurthy, Masoud Moshref, Dan Ports, and Peter Richtárik. Scaling distributed
machine learning with {In-Network} aggregation. In 18th USENIX Symposium on Networked
Systems Design and Implementation (NSDI 21), pages 785–808, 2021.

[58] ChonLam Lao, Yanfang Le, Kshiteej Mahajan, Yixi Chen, Wenfei Wu, Aditya Akella, and
Michael Swift. {ATP}: In-network aggregation for multi-tenant learning. In 18th USENIX
Symposium on Networked Systems Design and Implementation (NSDI 21), pages 741–761,
2021.

[59] Pat Bosshart, Glen Gibb, Hun-Seok Kim, George Varghese, Nick McKeown, Martin Izzard,
Fernando Mujica, and Mark Horowitz. Forwarding metamorphosis: Fast programmable match-
action processing in hardware for sdn. In Proceedings of the ACM SIGCOMM 2013 Conference
on SIGCOMM, SIGCOMM ’13, page 99–110, New York, NY, USA, 2013. Association for
Computing Machinery. ISBN 9781450320566. doi: 10.1145/2486001.2486011. URL https:
//doi.org/10.1145/2486001.2486011.

[60] N McKeown. Pisa: Protocol independent switch architecture. In P4 Workshop, 2015.

[61] Yifan Yuan, Omar Alama, Jiawei Fei, Jacob Nelson, Dan RK Ports, Amedeo Sapio, Marco
Canini, and Nam Sung Kim. Unlocking the power of inline {Floating-Point} operations on
In 19th USENIX Symposium on Networked Systems Design and
programmable switches.
Implementation (NSDI 22), pages 683–700, 2022.

13

521

522

523

524

525

526

[62] Zhourui Song, Zhenyu Liu, and Dongsheng Wang. Computation error analysis of block floating
point arithmetic oriented convolution neural network accelerator design. In Proceedings of the
Thirty-Second AAAI Conference on Artificial Intelligence and Thirtieth Innovative Applications
of Artificial Intelligence Conference and Eighth AAAI Symposium on Educational Advances in
Artificial Intelligence, AAAI’18/IAAI’18/EAAI’18. AAAI Press, 2018. ISBN 978-1-57735-
800-8.

14

