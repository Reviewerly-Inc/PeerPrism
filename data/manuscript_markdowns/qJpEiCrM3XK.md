A Curriculum Perspective of Robust Loss Functions

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

37

Learning with noisy labels is a fundamental problem in machine learning. A large
body of work aims to design loss functions robust against label noise. However,
it remain open questions why robust loss functions can underfit and why loss
functions deviating from theoretical robustness conditions can appear robust. To
tackle these questions, we show that a broad array of loss functions differs only in
the implicit sample-weighting curriculums they induce. We then adopt the resulting
curriculum perspective to analyze how robust losses interact with various training
dynamics, which helps elucidate the above questions. Based on our findings, we
propose simple fixes to make robust losses that severely underfit competitive to
state-of-the-art losses. Notably, our novel curriculum perspective complements the
common theoretical approaches focusing on bounding the risk minimizers.1

1

Introduction

Labeling errors are non-negligible [1] in datasets from expert annotation [2, 3], crowd-sourcing
[4] and automatic annotation [5, 6]. The resulting noisy labels can hamper generalization, as over-
parameterized neural networks can memorize all training samples [7]. To combat the impact of
noisy labels, a large body of research aims to design loss functions robust against label noise [8–13].
Theoretical results show that loss functions satisfying certain robustness conditions [9, 11] will lead
to the same optimum with clean or noisy labels.

Existing approaches focus on bounding the risk minimizer of a loss function [9–11, 14, 15] with the
presence of label noise, which are agnostic to the training dynamics. Though theoretically appealing,
they may fail to fully characterize the performance of robust losses with noisy labels. In particular,
it has been shown that (1) robust losses can underfit difficult tasks [1, 10, 12, 13], while (2) losses
failing to satisfy theoretical robustness conditions [12, 13, 16] can exhibit robustness. The reasons
behind these observations remain open questions. For (1), existing explanations [10, 17] can be
limited as discussed in §2.3. For (2), to our knowledge, there is no work directly addressing it.

To tackle the above questions, we consider training dynamics in our analysis, which complements
existing theoretical approaches [9–11]. By rewriting loss functions into a standard form, we show
that many loss function differs in the implicitly sample-weighting curriculums they induce (§3),
which connects robust losses to the seemingly distinct [1] curriculum learning approaches [18–22]
for noise-robust training. The original definition [23] of curriculum learning aims to present training
samples with gradually increasing difficulty and diversity to ease learning. We adopt a generalized
definition of curriculum [24], i.e., a curriculum specifies a sequence of re-weighting of training sample
distributions, which can manifest as sample weighting [18–20] or sample selection [21, 22, 25].

The curriculum perspective helps elucidate underfitting and noise robustness from the interaction
between the sample-weighting curriculums and various training dynamics. We first attribute un-
derfitting to the marginal average sample weights with the implicit curriculums (§4.1). We then
show that an increased number of classes can lead to marginal initial sample weights with some loss

1Our code will be available at github.

Submitted to 36th Conference on Neural Information Processing Systems (NeurIPS 2022). Do not distribute.

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

functions (§4.2). By adapting their curriculums accordingly, we make robust losses that severely
underfit perform competitively to state-of-the-art loss functions (§4.2). Finally, we attribute the noise
robustness of loss functions to higher average sample weights for clean samples compared to noisy
ones (§4.3). We hypothesize that clean samples can receive higher weights with sample-weighting
curriculums that magnify the learning speed differences and neglect unlearnt samples, which explains
our empirical observations (§4.3). Inspired by this hypothesis, we find two unexpected results when
viewed from existing theoretical robustness guarantees: by simply changing the learning rate schedule,
robust losses can be vulnerable to label noise and cross entropy can appear robust (§4.3).

2 Background

After formulating classification with label noise, we briefly review typical sufficient conditions and
loss functions for noise robustness to set the context for our novel curriculum perspective. We then
summarizing open questions to be addressed in this work.

2.1 Classification with Label Noise and Noise Robustness

The k-ary classification problem with input x ∈ Rd can be solved with classifier arg maxi si,
where si is the score of the i-th class from the class scoring function sθ : Rd → Rk. The class
scores sθ(x) can be turned into class probabilities with the softmax function pi = esi/((cid:80)k
j=1 esj ),
where pi is the probability for class i. Given a loss function L(sθ(x), y) and data (x, y) with
ground truth label y ∈ {1, . . . , k}, the parameter θ of sθ can be estimated with risk minimization
Ex,yL(sθ(x), y), whose solution are called risk minimizers. For notation simplicity, we
arg minθ
omit the dependence on θ and x if possible.
The annotation process may introduce errors, resulting in a potentially corrupted label ˜y following

˜y =

(cid:26) y,

with probability P (˜y = y|x, y)
i, i ̸= y with probability P (˜y = i|x, y)

Label noise is symmetric (or uniform) if P (˜y = i|x, y) = η/(k − 1), ∀i ̸= y, with η = P (˜y ̸= y) the
noise rate constant. Label noise is asymmetric (or class-conditional) if P (˜y = i|x, y) = P (˜y = i|y).
Given data (x, ˜y) with noisy label ˜y, a loss function L is robust against label noise if

arg min
θ

Ex,˜yL(sθ(x), ˜y) = arg min

θ

Ex,yL(sθ(x), y)

(1)

Most existing work [9–11, 14, 15] aim to derive bounds for the difference between risk minimizers
obtained using noisy and clean data, i.e., ensuring Eq. (1) holds with some conditions. As typical
examples, loss functions satisfying the symmetric [9] or asymmetric [11] conditions are theoretically
guaranteed to be robust. A loss function L is called symmetric if

(cid:88)

i

L(sθ(x), i) = C, ∀x, sθ

(2)

where C is a constant. When noise rate η < (k − 1)/k, a symmetric loss is robust against symmetric
label noise [9]. Such stringent condition is later relaxed by Zhou et al. [11]. Suppose a loss function
can be written as a function of softmax probability pi, i.e., L(sθ(x), i) = l(pi). As an equivalent
rephrase of the sufficient condition, L is called asymmetric if

max
i̸=y

P (˜y = i|x, y)
P (˜y = y|x, y)

= ˜r ≤ r =

inf
0≤pi,∆p≤1
pi+∆p≤1

l(pi) − l(pi + ∆p)
l(0) − l(∆p)

(3)

where ∆p is a valid increment of pi. When clean labels dominate the data, i.e., ˜r < 1, an asymmetric
loss is robust against generic label noise. Notably, both symmetric and asymmetric conditions for
noise robustness are agnostic to training dynamics to reach the risk minimizers.

2.2 Review of Selected Loss Functions

In addition to cross entropy (CE) that is vulnerable to label noise [9], we review typical loss functions
for later analysis. We ignore differences in constant scaling factors and constant additive bias in the
loss functions. They are either equivalent to learning rate scaling in SGD or irrelevant in the gradient
computation. See Table 1 for the exact formulas and Appendix A for an extended review.

2

Type

Name

CE

MAE/RCE

Sym.

NCE

AUL

AGCE

GCE

SCE

Asym.

Comb.

Function

− log py

1 − py

(cid:80)k

− log py
i=1 − log pi

(a−py )q −(a−1)q
q
(a+1)−(a+py )q
q

1−pq
y
q

Sample Weight w

Constraints

1 − py

py(1 − py)
(cid:0)wCE + k−1

k ϵNCE

(cid:1)

γNCE

py(1 − py)(a − py)q−1

a > 1, q > 0

py(a + py)q−1(1 − py)

a > 0, q > 0

pq
y(1 − py)

0 < q ≤ 1

(1 − q) · LCE + q · LMAE

(1 − q + q · py)(1 − py)

0 < q < 1

NCE+MAE

(1 − q) · LNCE + q · LMAE

(1 − q) · wNCE + q · wMAE

0 < q < 1

Table 1: Expressions, constraints of hyperparameters and sample weights of the implicit curriculums
(§3.1) for loss functions reviewed in §2.2. Note that wNCE is an approximation as discussed in §3.2.

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

Symmetric Loss The mean absolute error (MAE) [9] and the subsequent reverse cross entropy
(RCE) [13] are essentially equivalent, both satisfying Eq. (2). Ma et al. [10] normalize generic
loss functions satisfying L(s, i) > 0, ∀i ∈ {1, . . . , K} into symmetric losses with LN(s, y) =
L(s, y)/((cid:80)k
i=1 L(s, i)). We include normalized cross entropy (NCE) as an example.
Asymmetric Loss We include two asymmetric losses [11] for our analysis: asymmetric generalized
cross entropy (AGCE) and asymmetric unhinged loss (AUL). Notably, AGCE with q ≥ 1 and AUL
with q ≤ 1 are both completely asymmetric [11], i.e., Eq. (3) always holds when ˜r < 1.
Combined Loss Loss functions can be combined for both robust and sufficient learning. For
example, generalized cross entropy (GCE) [12] can be viewed as a smooth interpolation between
CE and MAE. Alternatively, symmetric cross entropy (SCE) [13] uses a weighted average of CE
and RCE (MAE). Finally, Ma et al. [10] argue that robust and sufficient training requires a balanced
combination of active and passive losses. Suppose loss function L can be rewritten into

L(s, y) =

k
(cid:88)

i=1

l(s, i)

(4)

where l is a function of scores s and any possible label i. An active loss requires ∀i ̸= y, l(s, i) = 0,
which focuses on learning the target label. In contrast, a passive one satisfies ∃j ̸= y, l(s, i) ̸= 0,
which can improve by unlearning non-target labels. Accordingly, CE and NCE are active while MAE
(RCE) is passive. We use NCE+MAE as an example.

2.3 Open Questions

Why do robust losses underfit? Ma et al. [10] attribute underfitting to failure in balancing active-
passive components. However, different specifications of Eq. (4) can lead to ambiguities in the
active-passive dichotomy. For example, with LMAE(s, y) ∝ (cid:80)k
i=1 |I(i = y) − py| where I(·) is the
indicator function, MAE is passive; yet the equivalent LMAE(s, y) ∝ (cid:80)k
I(i = y)(1 − py) makes
MAE an active loss. Wang et al. [17] instead view ∥∇sL(s, y)∥1 as weights for sample gradients
and attribute underfitting to their low variance, making clean and noisy samples less distinguishable.
However, as we show in §4.1, MAE also underfits on clean datasets. Why robust losses underfit thus
remains an open question.
What affects the robustness of a loss function? Although combined losses such as GCE and SCE
fail to satisfy existing robustness conditions (Eq. (2) and (3)), it is unclear why they exhibit robustness
against label noise [12, 13]. Furthermore, it is unclear how training dynamics, which are irrelevant in
many theoretical robustness guarantees [9–11, 14, 15], affect the noise robustness of a loss function.

i=1

3

107

108

109

110

111

112

3

Implicit Curriculums of Robust Loss Functions

We derive the standard form of reviewed loss functions and show that each implicitly induces a
sample-weighting curriculum, which helps examine how they interact with various training dynamics.

3.1 The Implicit Sample-Weighting Curriculums

Loss functions in Table 1 are generally functions of the target softmax probability py, i.e., L(s, y) =
l(py). Note that py can be rewritten as

113

where

py =

esy
i=1 esi

(cid:80)k

=

1

elog (cid:80)

i̸=y esi −sy + 1

=

1
e−∆y + 1

∆y = sy − log

(cid:88)

i̸=y

esi ≤ sy − max
i̸=y

si = ∆∗
y

(5)

(6)

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

is the soft margin between sy and the maximum of other scores, a smooth approximation of the hard
margin ∆∗
y. ∆y indicates how well a sample is learnt given classifier arg maxi si, as ∆y ≥ 0 leads to
∆∗
y ≥ 0, ensuring successful classification with scores s. Since ∇sl(py) = l′(py) · p′
y(∆y) · ∇s∆y,
these loss functions can be rewritten into a standard form with equivalent gradients:

L(s, y) = − stop_grad[w(∆y)] · ∆y

(7)
where stop_grad(·) avoids backpropagating through w(∆y) = l′(py) · p′
y(∆y). The equivalence
is valid only with first-order derivatives. Each loss function in the form of Eq. (7) thus implicitly
induces a sample-weighting curriculum, where w(∆y) is the sample weight and ∆y the implicit loss.
By examining how w(∆y) interacts with different training dynamics, we can elucidate the reasons
behind underfitting and noise robustness. Table 1 summarizes w(∆y) for the reviewed loss functions.
Wang et al. [16, 17] treat ∥∇sL(s, y)∥1 as weights for sample gradients, which share similar formulas
as w(∆y) in Table 1. Instead of directly weighting sample gradients, our derivation identifies the
implicit loss ∆y, making our sample-weighting scheme compatible with the definition of curriculum
learning [24]. In addition, the extracted ∆y and ∆∗
y can serve as direct metrics for sample performance
in curriculums, compared to loss [26, 27] and gradient magnitude [28] that are affected by preference
from w(∆y) of a loss function. Finally, the ∆y distribution is essential in analyzing the interaction
between loss functions and training dynamics in §4.

3.2 The Additional Entropy-Reducing Curriculum of NCE

Due to normalization, LNCE(s, y) in Table 1 additionally depends on ∆i where i ̸= y, which cannot
be be trivially rewritten into Eq. (7). A derivation of the gradient gives

∇sLNCE(s, y) =

(cid:40)

1
i=1 LCE(s, i)

(cid:80)k

∇sLCE(s, y) +

kLCE(s, y)
i=1 LCE(s, i)

(cid:80)k

(cid:34)

· ∇s

−

1
k

k
(cid:88)

i=1

(cid:35)(cid:41)

LCE(s, i)

133

Thus NCE can be rewritten as

= γNCE · [∇sLCE(s, y) + ϵNCE · ∇sRNCE(s)]

134

135

136

137

138

139

140

141

LNCE(s, y) = γNCE · LCE(s, y) + γNCE · ϵNCE · RNCE(s)
(8)
In this equivalent form, there is no backpropagation through the computation of γNCE and ϵNCE.
The first term results in a similar sample-weighting curriculum as CE, with an additional factor
γNCE = 1/((cid:80)k

i=1 − log pi) ≤ 1/(k log k). The second term is a regularizer

RNCE(s) = −

1
k

k
(cid:88)

i=1

LCE(s, i)

(9)

which reduces the entropy of the softmax output. The regularizer has per-sample weights ϵNCE =
k(− log py)/((cid:80)k
i=1 − log pi). It can thus be interpreted as a regularization curriculum. Notably, the
two curriculums work synergically in reducing the entropy of the softmax output.

The extra regularizer makes NCE incompatible to Eq. (7). However, as shown in Appendix C, since
∆y induces gradients with constant L1 norm, we can approximate the upperbound of wNCE with

wNCE =

∥∇sLNCE(s, y)∥1
∥∇s∆y∥1

(cid:18)

≤ γNCE

wCE +

(cid:19)

k − 1
k

ϵNCE

(10)

142

See Appendix C for derivations. Note that directions of ∇sLNCE(s, y) and ∇s∆y may be different.

4

Underfitting

Loss

CIFAR100

Acc.

No

Moderate

Severe

CE
GCE
SCE
NCE+MAE

NCE
AUL
AGCE

MAE
AUL†
AGCE†

71.33 ± 0.23
69.95 ± 0.40
71.36 ± 0.39
68.89 ± 0.23

43.18 ± 1.55
58.75 ± 1.07
49.27 ± 1.03

3.69 ± 0.59
3.13 ± 0.43
1.62 ± 0.69

¯α∗
t

8.183
8.861
9.541
2.971

1.769
5.278
4.537

0.035
0.033
0.009

CIFAR10

Acc.

92.76 ± 0.30
92.96 ± 0.13
93.17 ± 0.06
92.37 ± 0.33

91.28 ± 0.22
92.43 ± 0.19
92.61 ± 0.18

91.56 ± 0.11
91.13 ± 0.06
87.14 ± 4.96

¯α∗
t

5.541
6.151
7.018
2.414

1.072
5.171
5.225

2.492
2.308
1.701

Table 2: With clean labels, robust losses can underfit CIFAR100 but CIFAR10. Hyperparameters of
loss functions are tuned on CIFAR100 and listed in Table 7. We report test accuracy and average
effective learning rate ¯α∗
t (scaled by 103) at the final training step with 3 different runs, using learning
rate α = 0.1. AUL† and AGCE† with inferior hyperparamters are included as reference. See
Appendix D for results with α = 0.01.

(a) Severe underfitting

(b) Moderate underfitting

(c) No underfitting

Figure 1: Sample-weighting functions w(∆y) for selected loss functions and hyperparameters in
Table 2. We include the initial distributions of ∆y on CIFAR10 and CIFAR100 for reference.

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

4 Understanding Robust Losses with Their Implicit Curriculums

We empirically investigate the interaction between sample-weighting curriculums and various training
dynamics for questions in §2.3. Experiments are conducted on MNIST [29] and CIFAR10/100 [30]
with synthetic symmetric and asymmetric label noises following standard settings [10, 11]. We also
include real human noisy labels provided by Wei et al. [31] on CIFAR10/100. We use a 4-layer CNN
for MNIST, an 8-layer CNN for CIFAR10 and a ResNet-34 [32] for CIFAR100. By default, models
are trained with a fixed number of epochs using SGD with momentum, weight decay and cosine
learning rate annealing. See Appendix B for more experimental details. Different from standard
settings, we rescale w(∆y) to have unit maximum to avoid complications, since hyperparameters of
loss functions can change the maximum of w(∆y), essentially adjusting the learning rate of SGD.

153

4.1 Underfitting of Robust Losses from a Sample-Weighting Curriculum Perspective

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

Robust losses can underfit. We confirm that on difficult tasks like CIFAR100 [10, 12, 13], underfiting
results from robust losses themselves rather than inferior hyperparameters. We tune hyperparameters
of loss functions on CIFAR100 and report results on CIFAR100 and CIFAR10 without label noise. As
shown in Table 2, the performance of NCE, AGCE and AUL lag behind CE by a nontrivial margin on
CIFAR100. Notably, MAE performs much worse compared to CE, similar to AGCE† and AUL† with
inferior hyperparameters. In contrast, all loss functions fit CIFAR10 well. See Table 8 in Appendix D
for similar results with a smaller learning rate.
Marginal effective learning rate explains underfitting. We attribute underfitting to the diminishing
effective learning rate α∗
t = αt · ¯wt, where ¯wt is the average sample weight of the batch and αt the
learning rate at step t. We use the average effective learning rate up to step t, ¯α∗
i /t, to
characterize the overall α∗
t . In Table 2, for loss functions that heavily underfit on CIFAR100, their ¯α∗
t
at the final step is marginal compare to CE.

t = (cid:80)t

i=1 α∗

5

(a) AUL with inferior/superior hyperparameters.

(b) NCE with estimated weight upperbound.

Figure 2: Different causes of underfitting: (a) marginal initial sample weights; (b) fast diminishing
sample weights. We plot the average effective learning rate ¯α∗
t at different training steps t with
selected loss functions on CIFAR100.

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

Marginal effective learning rate due to marginal initial sample weights. In Fig. 1 we compare
sample-weighting functions w(∆y) of robust losses to the ∆y distribution of CIFAR10 and CIFAR100
at initialization. For robust losses that severely underfit (Fig. 1a), the ∆y distribution of CIFAR100
concentrates at regions with marginal sample weights, resulting in small effective learning rate α∗
t . It
can be hard for these samples to escape the region with marginal weights before the learning rate
attenuates. In contrast, loss functions with non-trivial initial sample weights (Fig. 1b and 1c) result in
moderate or no underfitting in Table 2. As a corroboration, we plot the average effective learning
rate ¯α∗
t of AUL with different hyperparameters in Fig. 2a. With superior hyperparameters (AUL
in Table 2), ¯α∗
t quickly increase to a non-negligible value before annealing. In contrast, ¯α∗
t stays
marginal with inferior hyperparameters (AUL† in Table 2).
Marginal effective learning rate due to fast diminishing sample weights. In Fig. 2b, different
from other robust losses but similar to CE, the effective learning rate of NCE peaks at initialization.
However, it decreases much faster compared to CE, which can be attributed to the synergy between
the two implicit curriculums of NCE in reducing wNCE. As ∆y improves, γNCE, ϵNCE and wCE all
decreases. In addition, the regularizer RNCE(s) further decreases the entropy of softmax output and
thus γNCE. Thus wNCE decreases much faster compared to wCE, leading to faster attenuating α∗
t .
Loss combination mitigates marginal initial sample weights. As wCE and wNCE peak at ini-
tialization, they compensate the marginal initial sample weights when combined with other robust
losses, helping initial learning and thus avoiding underfitting. In Table 2, the effective learning rate
on CIFAR100 is substantially increased when combining MAE with CE and NCE. Interestingly, CE
and NCE are both “active” as their sample weights peak at initialization, while other robust losses are
“passive” due to their marginal initial sample weights. Such dichotomy based on sample-weighting
curriculums complements the active-passive dichotomy [10] from a distinct perspective.

4.2 Addressing Underfitting by Adapting the Sample-Weighting Curriculums

As shown in Table 2, robust losses can underfit on CIFAR100 but CIFAR10. Such difference has
been vaguely attributed to the increased task difficulty [1, 12]. We further show that with static
sample-weighting curriculums, loss functions suffer from marginal initial sample weights due to the
increased number of classes k. By adapting the curriculums accordingly, robust losses that severely
underfit can become competitive with the state-of-the-art. We leave the fix for NCE to future work,
and use MAE as a typical example for illustration.

Intuitively, the larger number of classes, the more subtile differences to be distinguished, thus the
harder the task is. In addition, the number of classes k determines the ∆y distribution at initialization.
Assuming that class scores si at initialization are i.i.d. variables following the normal distribution, i.e.,
si ∼ N (µ, σ). In particular, µ = 0 and σ = 1 for most neural networks with standard initializations
[33] and normalization layers [34, 35]. See Appendix E for comparisons between simulations and
real settings. The expected ∆y can be approximated with

E[∆y] ≈ − log(k − 1) − σ2/2 +

eσ2
− 1
2(k − 1)

(11)

6

(a) Simulated initial distributions

(b) Add E[∆y] to ∆y’s

(c) Shifted/rescaled wMAE(∆y)

Figure 3: (a). Simulated initial ∆y distributions with different k assuming si ∼ N (µ, σ). We include
the plot of wMAE(∆y) for reference. (b). Adding E[∆y] to ∆y’s centers simulated distributions in
(a) to the origin. (c). The shifted and rescaled wMAE(∆y) with a = 2.6 and k = 100. We include
the initial ∆y distribution of CIFAR100 for reference.

Loss

CE [11]
GCE [11]
NCE [11]
NCE+AUL [11]

AGCE
AGCE shift
AGCE rescale

MAE
MAE shift
MAE rescale

Clean

η = 0

71.33 ± 0.43
63.09 ± 1.39
29.96 ± 0.73
68.96 ± 0.16

49.27 ± 1.03
67.50 ± 1.48
67.20 ± 0.79

3.69 ± 0.59
69.02 ± 0.78
69.95 ± 1.21

Symmetric

Asymmetric

η = 0.4

η = 0.8

η = 0.4

Human

η = 0.4

39.92 ± 0.10
56.11 ± 1.35
19.54 ± 0.52
59.25 ± 0.23

47.76 ± 1.75
53.33 ± 1.08
56.32 ± 0.59

1.29 ± 0.50
44.60 ± 0.24
60.70 ± 0.30

7.59 ± 0.20
17.42 ± 0.06
8.55 ± 0.37
23.03 ± 0.64

16.03 ± 0.59
10.47 ± 0.57
12.75 ± 1.10

1.00 ± 0.00
8.08 ± 0.26
10.79 ± 0.97

40.17 ± 1.31
40.91 ± 0.57
20.64 ± 0.40
38.59 ± 0.48

33.40 ± 1.57
38.37 ± 1.55
40.00 ± 0.27

2.53 ± 1.34
40.57 ± 0.47
39.22 ± 1.54

30.45 ± 1.50
44.44 ± 1.39
49.08 ± 0.74

2.09 ± 0.55
48.31 ± 0.31
54.65 ± 0.73

Table 3: Shifting or rescaling ∆y mitigates underfitting on CIFAR100 with different noise types and
noise rate η. Human noisy labels are from CIFAR100-N [31]. Test accuracies are reported with 3
different runs. We use a = 4.5 for AGCE and a = 2.6 for MAE. Results from [11] are included as
context. See Appendix E for results on WebVision and CIFAR100 with additional noise rates.

202

203

204

205

206

207

208

We leave detailed derivations to Appendix E. With more output classes, the ∆y distribution will
have smaller expectation, corresponding to diminishing initial sample weights with the fixed MAE
curriculum, as shown in Fig. 3a. In Fig. 3b, subtracting E(∆y) from ∆y centers distributions to 0.
Shifting or rescaling w(∆y) mitigates underfitting from increased number of classes. To assign
nontrivial sample weights at initialization, the sample-weighting curriculum of robust losses should
be adapted according to the number of classes k. A simple strategy is to make the expected initial
sample weights agnostic to k. Given a sample-weighting function w(∆y), we can either shift

wshift(∆y) = w(∆y + E[∆y] − a)

(12)

209

or rescale

(13)

wrescale(∆y) = w(∆y/E[∆y] · a)
it, where a > 0 is a hyperparameter. The shifted and scaled wMAE(∆y) are shown in Fig. 3c as an
illustration. Intuitively, shifting or scaling with E[∆y] can cancel the effect of increased k on the
expected initial sample weights. With smaller a, samples will get higher weights at initialization.
In Table 3, we test our fixes with different noise types and noise rates on CIFAR100. See Appendix E
for more results on the large scale noisy dataset WebVision [36] and CIFAR100 with different
synthetic noise rates. Rescaling and shifting alleviate the underfitting issues, making MAE and AGCE
perform comparable to the previous best (NCE+AUL) [11]. Notably, the performance of MAE is
substantially improved. Interestingly, despite being effective fixes for underfitting, simply scaling or
shifting w(∆y)’s can risk assigning large weights for noisy samples, which have lower ∆y in general
as discuss in §4.3, thus hampering the noise robustness of loss functions. Under symmetric label
noise with η = 0.8, the performance of AGCE decreases after applying the fixes.

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

7

Clean

Symmetric

Loss

CE

SCE
GCE

MAE

AUL
AGCE

η = 0.2

η = 0.4

η = 0.6

η = 0.8

Acc

∆acc

90.49

-15.85

91.06
90.85

90.56

90.79
90.56

-8.10
-2.02

-1.96

-1.90
-4.28

snr

0.39

0.76
3.25

3.46

3.51
3.11

∆acc

-32.34

-21.55
-5.59

-8.25

-5.06
-4.47

snr

0.58

1.03
3.16

3.15

3.40
3.29

∆acc

-51.57

-43.86
-14.16

-12.31

-13.43
-17.76

snr

0.77

1.29
2.95

2.88

3.01
2.69

∆acc

-71.14

-71.10
-50.10

-38.11

-50.99
-44.87

snr

0.95

1.32
2.29

2.53

1.79
2.04

Human

η = 0.4

∆acc

-28.18

-22.96
-12.52

-22.49

-22.36
-21.62

snr

0.53

0.74
1.14

1.00

1.02
1.02

Table 4: Robust losses assign larger weights to clean samples. We report snr and drop in test accuracy
with symmetric and human label noise on CIFAR10 at the final step with 3 different runs. We use the
“worst” version of CIFAR10-N [31] as human label noise. Standard deviation are omitted due to space
limitation. Hyperparameters of loss functions are tuned with noise rate η = 0.6. See Appendix B for
detailed hyperparameters.

(a) CE: 58.05

(b) SCE: 69.62

(c) MAE: 85.86

Figure 4: How ∆y distribution of noisy (green, left) and clean (orange, right) samples evolve during
training on CIFAR10 with 40% symmetric label noise. We include w(∆y) curves for reference, and
omit vertical axes denoting probability density for brevity. Vertical axes are scaled to the peak of
histograms for better readability, with epoch number (axis scaling factor) denoted on the right of
each subplot. We also include the final accuracy of the corresponding run for each loss function as
reference. See Appendix F for results of more loss functions with human label noise.

221

4.3 Noise Robustness from a Sample-Weighting Curriculum Perspective

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

i,t

I(˜yi,t = yi,t)αtwi,t/((cid:80)

Intuitively, loss functions exhibiting noise robustness should weight clean samples more than noisy
ones. We provide an explanation based on how w(∆y) interacts with two training dynamics.
Robust losses assign larger weights to clean samples. The average weight assigned to noisy samples
during training, adjusted by learning rate αt, is ¯wnoise = (cid:80)
I(˜yi,t ̸=
yi,t)αt), where wi,t denotes the weight of i-th sample of the batch at step t. ¯wclean for clean samples
can be defined similarly. The ratio snr = ¯wclean/ ¯wnoise characterizes their relative contribution
during training. We report snr and the drop in test accuracy under different label noise on CIFAR10
in Table 4. Loss functions with less performance drop have higher snr in general.
To explain what leads to a high snr, we first examine how ∆y distributions of noisy and clean samples
evolve during training on CIFAR10 with symmetric label noise in Fig. 4. See Appendix F for results
of more loss functions with human label noise. When trained using loss functions with increased
robustness (Fig. 4b and 4c), the noisy and clean distributions of ∆y gets better separated and more
spread. In addition, ∆y’s of some noisy samples gets decreased, suggesting that noisy samples can
be unlearnt. In contrast, with CE (Fig. 4a), the noisy and clean distributions of ∆y are less separated
and more compact.

i,t

8

(a) α = 0.01 with different (η, loss).

(b) CE with different (η, α).

Figure 5: Learning curves with fixed learning rate and extended training epochs on MNIST, where α
is the learning rate and η the symmetric label noise rate. Vertical axes are scaled for readability.

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

We now give a possible explanation for Fig. 4 with the following two training dynamics: (D1) clean
samples are learnt faster than noisy samples; (D2) noisy samples can be unlearnt when trained
on clean samples. D1 is identified in [7, 37], which later manifests itself in curriculum-based robust
training [38, 39]. It can result from the dominance of clean samples (˜r < 1) in the expected gradient.
In addition, gradients of clean samples are more correlated than those of noisy samples [40]. Thus
performance on clean samples can be improved when training on one another, leading to D1. D2 only
become apparent when examining Fig. 4b and 4c, which can result from generalization with clean
samples. Suppose in MNIST, a sample of 0 is erroneously labeled as 9. Then a model well-trained
with clean samples of class 9 and 0 can result in a low ∆y for this noisy sample. D1 and D2 can act
in synergy to separate the clean and noisy distributions of ∆y, as shown in Fig. 4.
We hypothesis that robust losses enhance the synergy of D1 and D2. In Table 1, w(∆y) of loss
functions can be decomposed into f (∆y)·g(∆y), where f (∆y) is a monotonically increasing function
and g(∆y) a decreasing one. For example, fCE(∆y) degenerates to constant 1 and gCE(∆y) = 1−py,
while fMAE(∆y) = py and gMAE(∆y) = 1 − py. Notably, g(∆y) shared by all loss functions
converges to 0 as ∆y increases, preventing ∆y from growing infinitely large. In addition, a non-
degenerated f (∆y) can enhance the synergy between D1 and D2. Since the initial ∆y distribution
generally lies on the monotonically increasing part of w(∆y) determined by f (∆y), faster learning
of samples results in their larger weights during training. Thus robust losses magnify the difference
in learning speed between clean and noisy samples, which can also account for the substantially
spread ∆y distributions in Fig. 4b and 4c. As w(∆y) can assign negligible sample weights with
low ∆y due to the monotonically increasing f (∆y), unlearnt noisy samples are neglected with
diminishing weights, which can account for the decrease of ∆y’s for noisy samples in Fig. 4b and 4c.
In contrast, as wCE(∆y) assign high sample weights for small ∆y’s, it compensates the synergy of
D1 and D2, thus results in compact ∆y distribution, larger ∆y’s for noisy samples, and less separated
∆y distributions in Fig. 4a.
With sufficient training, clean samples will eventually have high ∆y’s with diminishing sample
weights thanks to g(∆y). Noisy samples will then dominate the expected gradient and can lead to
overfitting, leading to two unexpected results when viewed from robustness conditions [9, 11]:
Robust losses are vulnerable to label noise with extended training. In Fig. 5a we show the learning
curve of CE and MAE using constant learning rate under different symmetric noises on MNIST.
Although enjoying theoretically guaranteed noise robustness [9, 11], similar to CE, MAE eventually
overfits to noisy samples, becoming vulnerable to label noise as weights of clean samples diminish.
Loss functions can become robust by adjusting the learning rate schedule. Interestingly, in Fig. 4a,
despite the compensation of wCE(∆y), the synergy between D1 and D2 still results in partially-
separated ∆y distributions of noisy and clean samples. We can thus improve the noise robustness of
CE by preventing the weights of clean samples from diminishing due to g(∆y), which can be achieve
by slowing down the convergence or early stopping [41]. In Fig. 5b we show the learning curve of
CE using fixed learning rates under symmetric noise on MNIST. By simply increasing or decreasing
the learning rate, which strengthens the implicit regularization of SGD [42] or directly slows down
the convergence, the noise robustness of CE can be substantially improved.

9

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

321

322

323

324

325

326

327

5 Related Work

Our work is closely related to robust loss functions [8–13] for robust training with noisy labels [1].
Theoretical results [9, 11] derive sufficient conditions for robustness against label noise without con-
sidering the training dynamics. We complement these results by considering the interaction between
robust losses and various training dynamics. The underfitting of robust losses has been heuristically
mitigated with loss combination [10, 12, 13]. We further elucidate the cause of underfitting from a
curriculum perspective, based on which we provide an effective solution.

Curriculum-based approaches combat label noise with either sample selection [21, 22] or sample-
weighting [18–20]. In particular, sample weights are designed [16–18] or predicted by a model
trained on a separated dataset [19, 20]. In contrast, the sample-weighting curriculums considered
in this work are implicitly induced by robust loss functions. Most related to our work, Wang et al.
[16] identifies gradient norms as weights for sample gradients of each robust loss. In contrast, as
discussed in §3.1, we explicitly extract the implicit loss, which helps draw the connection to standard
curriculum learning [24] and facilitates analysis of training dynamics.

Our work is also related to the ongoing debate [24, 43] on strategies for selecting or weighting
samples in curriculum learning: whether easier first [23, 26] or harder first [27, 44] is better. The
implicit curriculums of robust losses in this work differ in two important ways. First, the implicit
loss identified in §3.1 more directly measures sample difficulty than loss value [26, 27] and gradient
magnitude [28]. Second, the implicit sample-weighting curriculums can be viewed as a combination
of both weighting strategies by emphasizing moderately difficult samples, as discussed in §4.3.

6 Conclusion

We identify the implicit sample-weighting curriculums of selected loss functions. By decoupling
the implicit loss as a direct sample performance metric and sample weights specifying the implicit
sample preference, we can analyze how robust loss functions and curriculums interact with different
training dynamics. Such a perspective complements existing research on theoretical bounds for
the risk minimizer, and connects robust loss functions to the seemingly distinct approaches based
on curriculum learning. Following the curriculum perspective, we elucidate the reasons behind
underfitting and robustness against label noise for existing robust loss functions, and design a simple
approach to address the underfitting issue.

References

[1] Hwanjun Song, Minseok Kim, Dongmin Park, Yooju Shin, and Jae-Gil Lee. Learning from
noisy labels with deep neural networks: A survey. ArXiv preprint, abs/2007.08199, 2020. URL
https://arxiv.org/abs/2007.08199.

[2] Pete Bridge, Andrew Fielding, Pamela Rowntree, and Andrew Pullar. Intraobserver variability:

should we worry? Journal of medical imaging and radiation sciences, 47(3):217–220, 2016.

[3] Yoshihide Kato and Shigeki Matsubara. Correcting errors in a treebank based on synchronous
tree substitution grammar. In Proceedings of the ACL 2010 Conference Short Papers, pages
74–79, Uppsala, Sweden, 2010. Association for Computational Linguistics. URL https:
//aclanthology.org/P10-2014.

[4] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng
Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-
Fei. Imagenet large scale visual recognition challenge. arXiv:1409.0575 [cs], 2015. URL
http://arxiv.org/abs/1409.0575. arXiv: 1409.0575.

[5] Kun Liu, Yao Fu, Chuanqi Tan, Mosha Chen, Ningyu Zhang, Songfang Huang, and Sheng Gao.
Noisy-labeled NER with confidence estimation. In Proceedings of the 2021 Conference of the
North American Chapter of the Association for Computational Linguistics: Human Language
Technologies, pages 3437–3445, Online, 2021. Association for Computational Linguistics. doi:
10.18653/v1/2021.naacl-main.269. URL https://aclanthology.org/2021.naacl-main.
269.

[6] Huda Khayrallah and Philipp Koehn. On the impact of various types of noise on neural machine
translation. In Proceedings of the 2nd Workshop on Neural Machine Translation and Generation,

10

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

369

370

371

372

373

374

375

376

377

pages 74–83, Melbourne, Australia, 2018. Association for Computational Linguistics. doi:
10.18653/v1/W18-2709. URL https://aclanthology.org/W18-2709.

[7] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding
deep learning requires rethinking generalization. In 5th International Conference on Learning
Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings.
OpenReview.net, 2017. URL https://openreview.net/forum?id=Sy8gdB9xx.

[8] N. Manwani and P. S. Sastry. Noise tolerance under risk minimization. IEEE Transactions
on Cybernetics, 43(3):1146–1151, 2013. doi: 10.1109/tsmcb.2012.2223460. URL https:
//doi.org/10.1109%2Ftsmcb.2012.2223460.

[9] Aritra Ghosh, Himanshu Kumar, and P. S. Sastry. Robust loss functions under label noise for
deep neural networks. In Satinder P. Singh and Shaul Markovitch, editors, Proceedings of the
Thirty-First AAAI Conference on Artificial Intelligence, February 4-9, 2017, San Francisco,
California, USA, pages 1919–1925. AAAI Press, 2017. URL http://aaai.org/ocs/index.
php/AAAI/AAAI17/paper/view/14759.

[10] Xingjun Ma, Hanxun Huang, Yisen Wang, Simone Romano, Sarah M. Erfani, and James Bailey.
Normalized loss functions for deep learning with noisy labels. In Proceedings of the 37th
International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event,
volume 119 of Proceedings of Machine Learning Research, pages 6543–6553. PMLR, 2020.
URL http://proceedings.mlr.press/v119/ma20c.html.

[11] Xiong Zhou, Xianming Liu, Junjun Jiang, Xin Gao, and Xiangyang Ji. Asymmetric loss
functions for learning with noisy labels. In Marina Meila and Tong Zhang, editors, Proceedings
of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021,
Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 12846–12856.
PMLR, 2021. URL http://proceedings.mlr.press/v139/zhou21f.html.

[12] Zhilu Zhang and Mert R. Sabuncu. Generalized cross entropy loss for training deep
neural networks with noisy labels.
In Samy Bengio, Hanna M. Wallach, Hugo
Larochelle, Kristen Grauman, Nicolò Cesa-Bianchi, and Roman Garnett, editors, Advances
in Neural Information Processing Systems 31: Annual Conference on Neural Informa-
tion Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada,
pages 8792–8802, 2018. URL https://proceedings.neurips.cc/paper/2018/hash/
f2925f97bc13ad2852a7a551802feea0-Abstract.html.

[13] Yisen Wang, Xingjun Ma, Zaiyi Chen, Yuan Luo, Jinfeng Yi, and James Bailey. Symmetric
cross entropy for robust learning with noisy labels. In 2019 IEEE/CVF International Conference
on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 - November 2, 2019, pages
322–330. IEEE, 2019. doi: 10.1109/ICCV.2019.00041. URL https://doi.org/10.1109/
ICCV.2019.00041.

[14] Yang Liu and Hongyi Guo. Peer loss functions: Learning from noisy labels without knowing
noise rates. (arXiv:1910.03231), Aug 2020. doi: 10.48550/arXiv.1910.03231. URL http:
//arxiv.org/abs/1910.03231. arXiv:1910.03231 [cs, stat].

[15] Lei Feng, Senlin Shu, Zhuoyi Lin, Fengmao Lv, Li Li, and Bo An. Can cross entropy loss
be robust to label noise? In Proceedings of the Twenty-Ninth International Joint Conference
on Artificial Intelligence, page 2206–2212, Yokohama, Japan, Jul 2020. International Joint
Conferences on Artificial Intelligence Organization. ISBN 978-0-9992411-6-5. doi: 10.24963/
ijcai.2020/305. URL https://www.ijcai.org/proceedings/2020/305.

[16] Xinshao Wang, Elyor Kodirov, Yang Hua, and Neil M. Robertson. Derivative manipulation
for general example weighting. ArXiv preprint, abs/1905.11233, 2019. URL https://arxiv.
org/abs/1905.11233.

[17] Xinshao Wang, Yang Hua, Elyor Kodirov, and Neil M. Robertson. Imae for noise-robust learning:
Mean absolute error does not treat examples equally and gradient magnitude’s variance matters.
ArXiv preprint, abs/1903.12141, 2019. URL https://arxiv.org/abs/1903.12141.

11

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

429

430

[18] Haw-Shiuan Chang, Erik G. Learned-Miller, and Andrew McCallum. Active bias: Training
more accurate neural networks by emphasizing high variance samples. In Isabelle Guyon, Ulrike
von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman
Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference
on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA,
pages 1002–1012, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/
2f37d10131f2a483a8dd005b3d14b0d9-Abstract.html.

[19] Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, and Li Fei-Fei. Mentornet: Learning
data-driven curriculum for very deep neural networks on corrupted labels. In Jennifer G. Dy
and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine
Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80
of Proceedings of Machine Learning Research, pages 2309–2318. PMLR, 2018. URL http:
//proceedings.mlr.press/v80/jiang18c.html.

[20] Mengye Ren, Wenyuan Zeng, Bin Yang, and Raquel Urtasun. Learning to reweight examples
for robust deep learning. In Jennifer G. Dy and Andreas Krause, editors, Proceedings of the 35th
International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm,
Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pages
4331–4340. PMLR, 2018. URL http://proceedings.mlr.press/v80/ren18a.html.

[21] Tianyi Zhou, Shengjie Wang, and Jeff A. Bilmes. Robust curriculum learning: from clean
label detection to noisy label self-correction. In 9th International Conference on Learning
Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.
URL https://openreview.net/forum?id=lmTWnm3coJJ.

[22] Pengfei Chen, Benben Liao, Guangyong Chen, and Shengyu Zhang. Understanding and
utilizing deep neural networks trained with noisy labels. In Kamalika Chaudhuri and Ruslan
Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning,
ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of
Machine Learning Research, pages 1062–1070. PMLR, 2019. URL http://proceedings.
mlr.press/v97/chen19g.html.

[23] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning.
In Andrea Pohoreckyj Danyluk, Léon Bottou, and Michael L. Littman, editors, Proceedings
of the 26th Annual International Conference on Machine Learning, ICML 2009, Montreal,
Quebec, Canada, June 14-18, 2009, volume 382 of ACM International Conference Proceeding
Series, pages 41–48. ACM, 2009. doi: 10.1145/1553374.1553380. URL https://doi.org/
10.1145/1553374.1553380.

[24] Xin Wang, Yudong Chen, and Wenwu Zhu. A survey on curriculum learning. ArXiv preprint,

abs/2010.13166, 2020. URL https://arxiv.org/abs/2010.13166.

[25] Jinchi Huang, Lie Qu, Rongfei Jia, and Binqiang Zhao. O2u-net: A simple noisy label detection
approach for deep neural networks. In 2019 IEEE/CVF International Conference on Computer
Vision (ICCV), pages 3325–3333, 2019. doi: 10.1109/ICCV.2019.00342.

[26] M. Pawan Kumar, Benjamin Packer, and Daphne Koller. Self-paced learning for latent variable
models.
In John D. Lafferty, Christopher K. I. Williams, John Shawe-Taylor, Richard S.
Zemel, and Aron Culotta, editors, Advances in Neural Information Processing Systems 23:
24th Annual Conference on Neural Information Processing Systems 2010. Proceedings of a
meeting held 6-9 December 2010, Vancouver, British Columbia, Canada, pages 1189–1197.
Curran Associates, Inc., 2010. URL https://proceedings.neurips.cc/paper/2010/
hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html.

[27] Ilya Loshchilov and Frank Hutter. Online batch selection for faster training of neural networks.
ArXiv preprint, abs/1511.06343, 2015. URL https://arxiv.org/abs/1511.06343.

[28] Siddharth Gopal. Adaptive sampling for SGD by exploiting side information. In Maria-Florina
Balcan and Kilian Q. Weinberger, editors, Proceedings of the 33nd International Conference
on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016, volume 48
of JMLR Workshop and Conference Proceedings, pages 364–372. JMLR.org, 2016. URL
http://proceedings.mlr.press/v48/gopal16.html.

12

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

480

481

[29] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document
recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998. doi: 10.1109/5.726791.

[30] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images.

2009.

[31] Jiaheng Wei, Zhaowei Zhu, Hao Cheng, Tongliang Liu, Gang Niu, and Yang Liu. Learning
with noisy labels revisited: A study using real-world human annotations. Mar 2022. URL
https://openreview.net/forum?id=TBWA6PLJZQm.

[32] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR
2016, Las Vegas, NV, USA, June 27-30, 2016, pages 770–778. IEEE Computer Society, 2016.
doi: 10.1109/CVPR.2016.90. URL https://doi.org/10.1109/CVPR.2016.90.

[33] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward
In Proceedings of the Thirteenth International Conference on Artificial
neural networks.
Intelligence and Statistics, page 249–256. JMLR Workshop and Conference Proceedings, 2010.
URL https://proceedings.mlr.press/v9/glorot10a.html.

[34] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training
by reducing internal covariate shift. In Francis R. Bach and David M. Blei, editors, Proceedings
of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July
2015, volume 37 of JMLR Workshop and Conference Proceedings, pages 448–456. JMLR.org,
2015. URL http://proceedings.mlr.press/v37/ioffe15.html.

[35] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization, 2016. URL

https://arxiv.org/abs/1607.06450.

[36] Wen Li, Limin Wang, Wei Li, Eirikur Agustsson, and Luc Van Gool. Webvision database:
Visual learning and understanding from web data, 2017. URL https://arxiv.org/abs/
1708.02862.

[37] Devansh Arpit, Stanislaw Jastrzebski, Nicolas Ballas, David Krueger, Emmanuel Bengio,
Maxinder S. Kanwal, Tegan Maharaj, Asja Fischer, Aaron C. Courville, Yoshua Bengio, and
Simon Lacoste-Julien. A closer look at memorization in deep networks. In Doina Precup and
Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning,
ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine
Learning Research, pages 233–242. PMLR, 2017. URL http://proceedings.mlr.press/
v70/arpit17a.html.

[38] Quanming Yao, Hansi Yang, Bo Han, Gang Niu, and James Kwok. Searching to exploit
memorization effect in learning from corrupted labels, 2019. URL https://arxiv.org/abs/
1911.02377.

[39] Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor W. Tsang,
and Masashi Sugiyama. Co-teaching: Robust training of deep neural networks with
extremely noisy labels.
In Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kris-
ten Grauman, Nicolò Cesa-Bianchi, and Roman Garnett, editors, Advances in Neu-
ral Information Processing Systems 31: Annual Conference on Neural Information
Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada,
pages 8536–8546, 2018. URL https://proceedings.neurips.cc/paper/2018/hash/
a19744e268754fb0148b017647355b7b-Abstract.html.

[40] Satrajit Chatterjee and Piotr Zielinski. On the generalization mystery in deep learning. ArXiv

preprint, abs/2203.10036, 2022. URL https://arxiv.org/abs/2203.10036.

[41] Hwanjun Song, Minseok Kim, Dongmin Park, and Jae-Gil Lee. How does early stopping help
generalization against label noise?, 2019. URL https://arxiv.org/abs/1911.08059.

[42] Samuel L. Smith, Benoit Dherin, David G. T. Barrett, and Soham De. On the origin of implicit
regularization in stochastic gradient descent. In 9th International Conference on Learning
Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.
URL https://openreview.net/forum?id=rq_Qr0c1Hyo.

13

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

524

525

526

527

528

529

530

[43] Guy Hacohen and Daphna Weinshall. On the power of curriculum learning in training deep
networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th
International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach,
California, USA, volume 97 of Proceedings of Machine Learning Research, pages 2535–2544.
PMLR, 2019. URL http://proceedings.mlr.press/v97/hacohen19a.html.

[44] Xuan Zhang, Gaurav Kumar, Huda Khayrallah, Kenton Murray, Jeremy Gwinnup, Marianna J.
Martindale, Paul McNamee, Kevin Duh, and Marine Carpuat. An empirical exploration of
curriculum learning for neural machine translation. ArXiv preprint, abs/1811.00739, 2018. URL
https://arxiv.org/abs/1811.00739.

[45] Tsung-Yi Lin, Priya Goyal, Ross B. Girshick, Kaiming He, and Piotr Dollár. Focal loss for
dense object detection. In IEEE International Conference on Computer Vision, ICCV 2017,
Venice, Italy, October 22-29, 2017, pages 2999–3007. IEEE Computer Society, 2017. doi:
10.1109/ICCV.2017.324. URL https://doi.org/10.1109/ICCV.2017.324.

[46] Hao Cheng, Zhaowei Zhu, Xingyu Li, Yifei Gong, Xing Sun, and Yang Liu. Learning with
instance-dependent label noise: A sample sieve approach, 2020. URL https://arxiv.org/
abs/2010.02347.

[47] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna.
Rethinking the inception architecture for computer vision, 2015. URL https://arxiv.org/
abs/1512.00567.

[48] Michal Lukasik, Srinadh Bhojanapalli, Aditya Menon, and Sanjiv Kumar. Does label smooth-
ing mitigate label noise? In Proceedings of the 37th International Conference on Machine
Learning, page 6448–6458. PMLR, Nov 2020. URL https://proceedings.mlr.press/
v119/lukasik20a.html.

[49] Jiaheng Wei, Hangyu Liu, Tongliang Liu, Gang Niu, Masashi Sugiyama, and Yang Liu. To
smooth or not? when label smoothing meets noisy labels. (arXiv:2106.04149), Jun 2022. doi:
10.48550/arXiv.2106.04149. URL http://arxiv.org/abs/2106.04149. arXiv:2106.04149
[cs].

[50] Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, and Lizhen Qu.
Making deep neural networks robust to label noise: A loss correction approach. In 2017 IEEE
Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July
21-26, 2017, pages 2233–2241. IEEE Computer Society, 2017. doi: 10.1109/CVPR.2017.240.
URL https://doi.org/10.1109/CVPR.2017.240.

[51] Yee Whye Teh, David Newman, and Max Welling. A collapsed variational bayesian
inference algorithm for latent dirichlet allocation.
In Bernhard Schölkopf, John C.
Platt, and Thomas Hofmann, editors, Advances in Neural Information Processing Sys-
tems 19, Proceedings of the Twentieth Annual Conference on Neural Information Process-
ing Systems, Vancouver, British Columbia, Canada, December 4-7, 2006, pages 1353–
1360. MIT Press, 2006. URL https://proceedings.neurips.cc/paper/2006/hash/
532b7cbe070a3579f424988a040752f2-Abstract.html.

[52] Barry Cobb, Rafael Rumí, and Antonio Salmerón. Approximating the distribution of a sum of

log-normal random variables. 2012.

Checklist

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reflect the paper’s

contributions and scope? [Yes]

(b) Did you describe the limitations of your work? [Yes] In §3.1 we state that the curriculum
view is valid when considering the first order derivatives. We also analyze the exception
with NCE in §3.2.

(c) Did you discuss any potential negative societal impacts of your work? [N/A]

14

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

576

577

578

(d) Have you read the ethics review guidelines and ensured that your paper conforms to

them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [Yes] In §4.2 we

explicitly state the assumed distributions of si when deriving E[∆y].

(b) Did you include complete proofs of all theoretical results? [Yes] In Appendix C we
include the detailed derivations of ∥∇s∆y∥1 and wNCE; in Appendix E we include the
detailed derivation of E[∆y].

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main experi-

mental results (either in the supplemental material or as a URL)? [Yes]

(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they
were chosen)? [Yes] All default settings are in Appendix B, specific hyperparameters
deviation from default settings are stated near each result.

(c) Did you report error bars (e.g., with respect to the random seed after running experi-
ments multiple times)? [Yes] Tables 2 to 4, 8 and 10. We omit error bars for figures to
improve readability.

(d) Did you include the total amount of compute and the type of resources used (e.g., type

of GPUs, internal cluster, or cloud provider)? [Yes] Appendix B

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...
(a) If your work uses existing assets, did you cite the creators? [Yes] At the beginning of

§4, we cite MNIST, CIFAR10/100.

(b) Did you mention the license of the assets? [N/A] MNIST and CIFAR10/100 are classic

benchmarks

(c) Did you include any new assets either in the supplemental material or as a URL? [N/A]

(d) Did you discuss whether and how consent was obtained from people whose data you’re

using/curating? [N/A]

(e) Did you discuss whether the data you are using/curating contains personally identifiable

information or offensive content? [N/A]

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if

applicable? [N/A]

(b) Did you describe any potential participant risks, with links to Institutional Review

Board (IRB) approvals, if applicable? [N/A]

(c) Did you include the estimated hourly wage paid to participants and the total amount

spent on participant compensation? [N/A]

A Extended Review of Loss Functions

As a general reference, we provide an extended review of loss functions for classification that is
relevant to the standard form Eq. (7), complementing review in §2.2. Loss functions and their
sample-weighting functions are summarized in Table 5. We plot how hyperparameters affect their
sample-weighting functions in Fig. 6.

A.1 Loss Functions without Robustness Guarantees

Cross Entropy (CE)

LCE(s, y) = − log py

is the standard loss function for classification.
Focal Loss (FL) [45]

LFL(s, y) = −(1 − py)q log py
aims to address the label imbalance in object detection. Note that both CE and FL are neither
symmetric [10] nor asymmetric [11].

15

579

580

A.2 Symmetric Losses

Mean Absolute Error (MAE) [9]

LMAE(s, y) =

1
k

k
(cid:88)

i=1

|I(i = y) − pi| = 2 − 2py ∝ 1 − py

581

582

is a classic symmetric loss, where I(i = y) is the indicator function.
Reverse Cross Entropy (RCE) [13]

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

LRCE(s, y) =

k
(cid:88)

i=1

pi log 1(i = y) =

(cid:88)

i̸=y

piA = (1 − py)A ∝ 1 − py = LMAE(s, y)

is equivalent to MAE in implementation, where log 0 is truncated to a negative constant A to avoid
numerical overflow.
Ma et al. [10] argued that any generic loss functions with L(s, i) > 0, ∀i ∈ {1, . . . , k} can become
symmetric by simply normalizing them. As an example,
Normalized Cross Entropy (NCE)

LNCE(s, y) =

LCE(s, y)
i=1 LCE(s, i)
is a symmetric loss [10]. However, NCE does not follow the standard form of Eq. (7). It involves an
additional regularizer as discussed in §3.2 and Appendix C, thus being more relevant to discussions
in Appendix A.4.

− log py
i=1 − log pi

(cid:80)k

(cid:80)k

=

A.3 Asymmetric Losses

Zhou et al. [11] derived the asymmetric condition for noise robustness, and propose an array of
asymmetric losses:
Asymmetric Generalized Cross Entropy (AGCE)

LAGCE(s, y) =

(a + 1) − (a + py)q
q

where a > 0 and q > 0. It is asymmetric when I(q ≤ 1)( a+1
Asymmetric Unhinged Loss (AUL)

a )1−q + I(q > 1) ≤ 1/˜r.

where a > 1 and q > 0. It is asymmetric when I(q ≤ 1)( a

LAUL(s, y) =

(a − py)q − (a − 1)q
q
a−1 )q−1 + I(q ≤ 1) ≤ 1/˜r.

Asymmetric Exponential Loss (AEL)

LAEL(s, y) = e−py/q

where q > 0. It is assymetric when e1/q ≤ 1/˜r.

A.3.1 Combined Losses

Loss functions can be combined to enjoy better learning.
Generalized Cross Entropy (GCE) [12]

LGCE(s, y) =

1 − pq
y
q

can be viewed as a smooth interpolation between CE and MAE, where 0 < q ≤ 1. CE or MAE can
be recovered by setting q → 0 or q = 1.
Symmetric Cross Entropy (SCE) [13]

LSCE(s, y) = a · LCE(s, y) + b · LRCE(s, y) ∝ (1 − q) · (− log pi) + q · (1 − pi)

16

Name

CE

FL

Function

− log py

Sample Weight w

Constraints

1 − py

−(1 − py)q log py

(1 − py)q(1 − py − qpy log py)

q > 0

MAE/RCE

1 − py

py(1 − py)

AUL

AGCE

AEL

GCE

SCE

TCE

(a+1)−(a+py )q
q
(a−py )q −(a−1)q
q
e−py /q

(1 − pq

y)/q

py(1 − py)(a − py)q−1

a > 1, q > 0

py(a + py)q−1(1 − py)

a > 0, q > 0

1

q py(1 − py)e−py /q

pq
y(1 − py)

q > 0

0 < q ≤ 1

−(1 − q) log py + q(1 − py)

(1 − q + q · py)(1 − py)

0 < q < 1

(cid:80)q

i=1(1 − py)i/i

py

(cid:80)q

i=1(1 − py)i

q ≥ 1

Table 5: Expressions, constraints of hyperparameters and sample-weighting functions of loss functions
in Appendix A that follows the standard form Eq. (7).

606

607

is a weighted average of CE and RCE (MAE), where a > 0, b > 0, and 0 < q < 1.
Taylor Cross Entropy (TCE) [15]

LTCE(s, y) =

q
(cid:88)

i=1

(1 − py)i
i

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

is originally derived from Taylor series of the log function. TCE reduces to MAE when q = 1.
Interestingly, the summand of TCE (1 − py)i/i with i > 2 is proportional to AUL with a = 1 and
q = i. Thus TCE can be viewed as a combination of symmetric and asymmetric losses.
Ma et al. [10] propose to additively combine active and passive loss functions. We review NCE+MAE
as an example:

LNCE+MAE(s, y) = a · LNCE(s, y) + b · LMAE(s, y) ∝ (1 − q) ·

− log py
i=1 − log pi

(cid:80)k

+ q · (1 − py)

where a > 0, b > 0, and 0 < q < 1.

A.4 Loss Functions with Additional Regularizers

We additionally review loss functions that implicitly involve a regularizer and a primary loss function
that fits the standard form Eq. (7). See Table 6 for a summary. We leave investigation on how these
additional regularizers affect noise robustness for future work.
Mean Square Error (MSE) [9]

LMSE(s, y) =

k
(cid:88)

(I(i = y) − pi)2 = 1 − 2py +

k
(cid:88)

i=1

p2
i

i=1

∝ 1 − py +

1
2

·

k
(cid:88)

i=1

p2
i = LMAE(s, y) + α · RMSE(s)

is argued [9] to be more robust than CE, where α = 1

2 and the regularizer
k
(cid:88)

RMSE(s) =

p2
i

(14)

620

621

reduces the entropy of the softmax output. We can generalize α to a hyperparamter, making MSE a
combination of MAE and an entropy regularizer RMSE.

i=1

17

(a) FL

(b) AUL, a = 2.0

(c) AUL, q = 0.1

(d) AGCE, a = 1.0

(e) AGCE, q = 0.5

(f) AEL

(g) GCE

(h) SCE

(i) TCE

Figure 6: How hyperparameters affect the sample-weighting functions of loss functions in Table 5.
The initial ∆y distribution of CIFAR100 are included as reference.

622

Given a generic loss function L(s, y), Peer Loss (PL) [14]

623

624

625

626

627

LPL(s, y) = L(s, y) − L(sn1, yn2 )
can make it robust against label noise, where sn1 and yn2 denote scores (of input xn1) and labels
randomly sampled from the noisy data. PL is inspired by the peer prediction mechanism to truthfully
elicit information when there is no ground truth verification. Its noise robustness is theoretically
established for binary classification and extended to multi-class setting [14]. Cheng et al. [46] later
show that PL in its expectation is equivalent to the original loss plus a Confidence Regularizer (CR):
RCR(s) = −E˜y[L(s, ˜y)]

628

Substituting L with the standard LCE, RCR(s) becomes

RCR(s) = −E˜y[− log p˜y] =

k
(cid:88)

i=1

P (˜y = i) log pi

(15)

629

630

631

632

Minimizing RCR(s) thus makes the softmax output distribution pi’s deviate from the prior label
distribution of the noisy dataset P (˜y = i)’s, reducing the entropy of the softmax output.
Label smoothing [47] has been shown to mitigate overfitting with label noise [48]. With the standard
cross entropy, Generalized Label Smoothing (GLS) [49]

LGLS+CE(s, y) =

k
(cid:88)

i=1

−[I(i = y)(1 − α) +

α
k

] log pi

= −(1 − α) log py − α ·

1
k

k
(cid:88)

i=1

log pi

∝ − log py −

α
1 − α

·

1
k

k
(cid:88)

i=1

log pi = LCE(s, y) + α′ · RGLS(s)

18

Name

MSE

Original

Primary Loss

Regularizer

1 − 2py + (cid:80)k

i=1 p2

i

PL
GLS − (cid:80)k

NCE

− log py + log pyn2 |xn1
i=1[I(i = y)(1 − α) + α
− log py
i=1 − log pi

(cid:80)k

k ] log pi

1 − py

− log py

− log py
(cid:16)

(cid:80)k

1
i=1 log pi

(cid:80)k

i=1 p2

i

(cid:80)k

i=1 P (˜y = i) log pi
± (cid:80)k
1
k log pi
(cid:80)k

i=1

1
k log pi

i=1

(cid:17)

log pi

stop_grad

Table 6: Original expressions, primary losses following the standard form Eq. (7) and regularizers
for loss functions reviewed in Appendix A.4. We view PL in its expectation to derive its regularizer.
pyn2 |xn1
is the softmax output with a random input xn1 and a random label yn2 sampled from the
noisy data.

Loss

CIFAR10

CIFAR100

SCE
GCE
NCE+MAE
AUL
AGCE

q = 0.7
q = 0.3
q = 0.3
a = 1.1, q = 5
a = 0.1, q = 0.1

q = 0.95
q = 0.9
q = 0.9
a = 7.0, q = 0.5
a = 3.0, q = 1.2

AUL†
AGCE†

FL
AEL
TCE

a = 3.0, q = 0.7
a = 1.6, q = 2.0

/
/

/
/
/

q = 2
q = 1.5
q = 6

Table 7: Hyperparameters of each loss function on different datasets. AUL† and AGCE† are with
inferior hyperparameters.

633

where α′ = α/(1 − α), has regularizer RGLS

RGLS(s) = −

k
(cid:88)

i=1

1
k

log pi

(16)

634

635

636

637

With α′ > 0, RGLS corresponds to the original label smoothing [47], which increases the entropy of
softmax outputs. In contrast, α′ < 0 corresponding to negative label smoothing [49], which decreases
the output entropy similar to RCR.
Finally, with equivalent derivatives, NCE discussed in §3.2 and Appendix C can be decomposed into

LNCE(s, y) =

1
i=1 − log pi

(cid:80)k

(cid:40)

− log py +

k log py
i=1 log pi

(cid:80)k

·

(cid:34)

1
k

k
(cid:88)

i=1

(cid:35)(cid:41)

log pi

= stop_grad(γNCE) · [LCE(s, y) + stop_grad(ϵNCE) · RNCE(s)]

638

where

RNCE(s) =

k
(cid:88)

i=1

1
k

log pi

(17)

639

is the same regularizer as RGLS with a negative weight −ϵNCE.

640

641

642

643

644

645

B Detailed Experimental Settings

Our settings follow [10, 11], with differences explicitly stated in the main text. All models on
CIFAR10/100 and MNIST are trained on NVIDIA 2080ti gpus with FP32. For models on the large
scale dataset WebVision [36], we use FP16 to accelerate training.
Synthetic noise generation The noisy labels are generated following [10, 11, 50]. For symmetric
label noise, the training labels are randomly flipped to a different class with with probabilities

19

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

η ∈ {0.2, 0.4, 0.6, 0.8}. Asymmetric label noise are generated by a class-dependent flipping pattern.
On CIFAR-100, the 100 classes are grouped into 20 super-classes each having 5 sub-classes. Each
class are flipped within the same super-class into the next in a circular fashion. The flip probabilities
are η ∈ {0.1, 0.2, 0.3, 0.4}.
Models and Training We use a 4-layer CNN for MNIST, an 8-layer CNN for CIFAR10, a
ResNet-34 [32] for CIFAR100, and a ResNet-50 [32] for WebVision, all with batch normalization
[34]. Data augmentation including random width/height shift and horizontal flip are applied to
CIFAR10/100. On WebVision, we additionally include random cropping and color jittering. Without
further specifications, all models are trained using SGD with momentum 0.9 and batch size 128
for 50, 120, 200 and 250 epochs on MNIST, CIFAR10, CIFAR100 and WebVision, respectively.
Learning rates with cosine annealing are 0.01 on MNIST and CIFAR10, 0.1 on CIFAR100, and 0.2
on WebVision. Weight decays are 10−3 on MNIST, 10−4 on CIFAR10, 10−5 on CIFAR100 and
3 × 10−5 on WebVision. Notably, all loss functions are normalized to have unit maximum in sample
weights, which is different from [10]. See Table 7 for hyperparameters of loss functions on different
datasets.

C Deriving the Upperbound of Sample Weights of NCE

We provide detailed derivations for results in §3.2.
Constant Norm of ∥∇s∆y∥1: Since

∂∆y
∂si

=

(cid:40) 1,

− esi
(cid:80)

k̸=y esk = pi

1−py

i = y
i ̸= y

,

664

then

∥∇s∆y∥1 =

(cid:88)

i

|

∂∆y
∂si

| = 1 +

(cid:88)

i̸=y

esi
k̸=y esk

(cid:80)

= 1 + 1 = 2

665

Approximating upperbound of wNCE in Eq. (10):

wNCE =

∥∇sLNCE(s, y)∥1
∥∇s∆y∥1

=

1
2

∥∇sLNCE(s, y)∥1

≤

≤

1
2

1
2

γNCE · (∥∇sLCE(s, y)∥1 + ϵNCE · ∥∇sRCE(s)∥1)



γNCE ·

∥∇sLCE(s, y)∥1 + ϵNCE ·

∥∇sLCE(s, j)∥1





1
k

(cid:88)

j

(cid:18)

= γNCE

wCE +

(cid:19)

k − 1
k

ϵNCE

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

The derivation is based on the inequality |x ± y| ≤ |x| + |y|, following the intuition [16, 17] that
∥∇sLNCE(s, y)∥1 can be regarded as sample weights.

D Underfitting of Robust Losses: Additional Results

In Table 8 we report similar results as Table 2 in §4.1 with smaller learning rates. Although settings
that severe underfits slightly improve, the performance gap compared to CE is still substantial. Such
results further confirms that underfitting results from robust losses themselves.

E Fixing Underfitting: Derivations and Additional Results

We include detailed derivations and additional results for §4.2.
Simulated ∆y’s well approximate real settings. We compare the simulated ∆y distributions to that
of real datasets at initialization in Fig. 7. Although less accurate with the variance, the simulated
expectations mostly follow real settings, which supports the analysis in §4.2.

20

Underfitting

No

No

Moderate

Severe

Loss

CE

GCE
SCE
NCE+MAE

NCE
AUL
AGCE

MAE
AUL†
AGCE†

CIFAR100

Acc.

¯α∗
t

CIFAR10

Acc.

¯α∗
t

68.76 ± 0.21

0.962

90.24 ± 0.14

0.624

69.00 ± 0.24
68.89 ± 0.05
68.21 ± 0.51

57.95 ± 0.26
47.98 ± 3.48
43.51 ± 2.58

9.11 ± 0.83
10.04 ± 2.33
5.34 ± 0.67

0.956
1.165
0.520

0.330
0.485
0.406

0.025
0.023
0.008

90.83 ± 0.20
91.07 ± 0.09
90.14 ± 0.09

85.96 ± 0.21
88.94 ± 0.29
90.71 ± 0.19

90.65 ± 0.10
90.77 ± 0.04
81.59 ± 8.55

0.644
0.726
0.344

0.206
0.604
0.549

0.355
0.337
0.243

Table 8: Similar results as Table 2 except with learning rate α = 0.01. See Table 7 for detailed
hyperparameters. AUL† and AGCE† with inferior hyperparamters are included as reference. Robust
losses can underfit regardless of hyperparameters of training.

(a) CIFAR10

(b) CIFAR100

(c) WebVision10

(d) WebVision50

(e) WebVision200

(f) WebVision400

Figure 7: Comparing simulated and real ∆y distributions at initialization. We simulate with class
scores following standard normal distribution, i.e., si ∼ N (0, 1). Histograms are real distributions
while the curves are from simulations, with the vertical axis denoting probability density f (∆y).

677

Deriving E(∆y) in Eq. (11) :

E(∆y) = E[sy − log

(cid:88)

j̸=y

esj ] = µ − E[log

(cid:88)

esj ]

≈1 µ − log E[

(cid:88)

esj ] +

j̸=y

V[(cid:80)
2E[(cid:80)

j̸=y
j̸=y esj ]
j̸=y esj ]2

=2 µ − log{(k − 1)E[esy ]} +

=3 µ − log[(k − 1)eµ+σ2/2] +

(k − 1)V[esy ]
2{(k − 1)E[esy ]}2
(k − 1)(eσ2

− 1)e2µ+σ2

2[(k − 1)eµ+σ2/2]2

= − log(k − 1) − σ2/2 +

eσ2
− 1
2(k − 1)

678

679

V[X]
where ≈1 follows E[log X] ≈ log E[X] −
2E[X]2 [51], =2 utilize properties of sum of log-normal
variables [52], and =3 substitutes the expression of E[esy ] and V[esy ] for log-normal distributions.

21

Loss

CE [11]
GCE [11]
NCE [11]
NCE+AUL [11]

AGCE
AGCE shift
AGCE rescale

MAE
MAE shift
MAE rescale

Clean
η = 0

71.33 ± 0.43
63.09 ± 1.39
29.96 ± 0.73
68.96 ± 0.16

49.27 ± 1.03
67.50 ± 1.48
67.20 ± 0.79

3.69 ± 0.59
69.02 ± 0.78
69.95 ± 1.21

Symmetric Noise (Noise Rate η)

η = 0.2

η = 0.4

η = 0.6

η = 0.8

56.51 ± 0.39
61.57 ± 1.06
25.27 ± 0.32
65.36 ± 0.20

49.17 ± 2.15
61.95 ± 1.48
64.28 ± 1.27

2.92 ± 0.46
59.75 ± 0.84
66.42 ± 0.71

39.92 ± 0.10
56.11 ± 1.35
19.54 ± 0.52
59.25 ± 0.23

47.76 ± 1.75
53.33 ± 1.08
56.32 ± 0.59

1.29 ± 0.50
44.60 ± 0.24
60.70 ± 0.30

21.39 ± 1.17
45.28 ± 0.61
13.51 ± 0.65
46.34 ± 0.21

38.17 ± 1.43
33.26 ± 0.37
38.52 ± 1.67

2.27 ± 1.24
24.27 ± 0.26
45.17 ± 2.37

7.59 ± 0.20
17.42 ± 0.06
8.55 ± 0.37
23.03 ± 0.64

16.03 ± 0.59
10.47 ± 0.57
12.75 ± 1.10

1.00 ± 0.00
8.08 ± 0.26
10.79 ± 0.97

Table 9: Shifting or rescaling ∆y mitigates underfitting on CIFAR100 with symmetric label noise.
We use a = 2.6 for MAE and AGCE and a = 4.5 for AGCE. Test accuracies are reported with 3
different runs. We also include results from [11] as context.

Loss

CE [11]
GCE [11]
NCE [11]
NCE+AUL [11]

AGCE
AGCE-shift
AGCE-rescale

MAE
MAE-shift
MAE-rescale

Clean
η = 0

71.33 ± 0.43
63.09 ± 1.39
29.96 ± 0.73
68.96 ± 0.16

49.27 ± 1.03
67.50 ± 1.48
67.20 ± 0.79

3.69 ± 0.59
69.02 ± 0.78
69.95 ± 1.21

Asymmetric Noise (Noise Rate η)

η = 0.1

η = 0.2

η = 0.3

η = 0.4

64.85 ± 0.37
63.01 ± 1.01
27.59 ± 0.54
66.62 ± 0.09

47.53 ± 0.73
64.07 ± 0.90
65.69 ± 0.24

3.59 ± 0.56
63.82 ± 0.84
68.01 ± 1.08

58.11 ± 0.32
59.35 ± 1.10
25.75 ± 0.50
63.86 ± 0.18

46.77 ± 2.37
56.16 ± 1.44
60.80 ± 0.77

3.19 ± 0.98
56.38 ± 0.45
65.71 ± 0.47

50.68 ± 0.55
53.83 ± 0.64
24.28 ± 0.80
50.38 ± 0.32

39.82 ± 2.70
46.73 ± 1.39
48.72 ± 1.39

2.11 ± 1.93
48.93 ± 0.53
57.40 ± 0.35

40.17 ± 1.31
40.91 ± 0.57
20.64 ± 0.40
38.59 ± 0.48

33.40 ± 1.57
38.37 ± 1.55
40.00 ± 0.27

2.53 ± 1.34
40.57 ± 0.47
39.22 ± 1.54

Table 10: Shifting or rescaling ∆y mitigates underfitting on CIFAR100 with asymmetric label noise.
We use a = 2.6 for MAE and AGCE and a = 4.5 for AGCE. Test accuracies are reported with 3
different runs. We also include results from [11] as context.

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

Additional results of shifted and rescaled fix to robust losses. We report results with symmetric
(Table 9) and asymmetric (Table 10) label noise with diverse noise rates η. For real world noisy
datasets, we subsample WebVision following standard settings [10, 11] with different number of
classes, and report results with MAE and ResNet50 in Table 11. See Appendix B for detailed
experimental settings. Notably, WebVision50 corresponds to the mini setting adopted in previous
work [10, 11]. Shift and rescale ∆y mitigate underfitting of MAE and AGCE in general, resulting in
performance similar to the state-of-the-arts.

F Understanding Robustness: Additional Results

As a more extended exploration to Fig. 4 in §4.3, in Fig. 8 we plot how distribution of ∆y evolve with
more loss functions and more number of epochs on human label noise of CIFAR10-N [31]. They all
follow similar trends as in Fig. 4.

22

k = 10
a = 2.2

k = 50
a = 2.0

k = 200
a = 1.8

k = 400
a = 1.6

CE
MAE
MAE-shift
MAE-rescale

62.40
10.0
58.40
48.40

66.40
3.68
60.76
66.72

70.26
0.50
59.31
71.92

/
/
/
/

Table 11: Shifting or rescaling ∆y mitigates underfitting on real noisy dataset WebVision [36] with
different number of classes. Due to the scale of the dataset, we only report test accuracy with a single
run.

(a) CE: 61.96

(b) FL: 61.02

(c) MAE: 68.34

(d) AUL: 68.51

(e) SCE: 68.61

(f) TCE: 72.42

(g) AGCE: 73.19

(h) AEL: 78.13

(i) GCE: 78.78

Figure 8: Additional results as Fig. 4 for more loss functions in Table 5 on CIFAR10-N [31] with
“worst” noisy labels (η = 0.4). Note that CE and FL do not enjoy robustness guarantees.

23

