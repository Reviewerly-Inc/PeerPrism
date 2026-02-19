Distributed Learning with Strategic Users:
A Repeated Game Approach

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

34

We consider a distributed learning setting where strategic users are incentivized, by
a cost-sensitive fusion center, to train a learning model based on local data. The
users are not obliged to provide their true gradient updates and the fusion center
is not capable of validating the authenticity of reported updates. Thus motivated,
we formulate the interactions between the fusion center and the users as repeated
games, manifesting an under-explored interplay between machine learning and
game theory. We then develop an incentive mechanism for the fusion center based
on a joint gradient estimation and user action classiﬁcation scheme, and study its
impact on the convergence performance of distributed learning. Further, we devise
an adaptive zero-determinant (ZD) strategy, thereby generalizing the celebrated ZD
strategy to the repeated games with time-varying stochastic errors. Theoretical and
empirical analysis show that the fusion center can incentivize the strategic users to
cooperate and report informative gradient updates, thus ensuring the convergence.

1

Introduction

Distributed machine learning is becoming increasingly important in large-scale problems with data-
intensive applications [18, 21, 25, 37]. Notably, federated learning has emerged as an attractive
distributed computing paradigm that aims to learn an accurate model without collecting data from the
owners and storing it in the cloud: The training data is kept locally on the computing devices which
participate in the model training and report gradient updates (or its variants) based on local data [19].

In this work, we study a distributed learning scheme in which privacy-aware users train a global model
with a fusion center. We consider the users to be rational, self-interested and risk-neutral. The users
are not compelled to contribute their resources unconditionally, unless they are sufﬁciently rewarded,
and the system may reach a noncooperative Nash equilibrium where the users do not participate in
training. This departs from conventional distributed learning schemes where the agents directly follow
the lead of the fusion center (FC)1 and send their gradients. Since the users are strategic, a paramount
objective for the FC is to design an effective reward mechanism to incentivize self-interested users to
provide informative gradient updates. The repeated game enriches the distributed learning framework
with the idea of many agents interacting within a common uncertain environment, and this framework
provides a new perspective to specify how agents can strategically choose the learning updates how
the resulting changes impact the performance of the learning efforts.

Challenges and Contributions. There are a number of challenges in distributed learning with
strategic users. First, the users are not obliged to entirely dedicate their resources and they may not
fulﬁll their roles in the training of the algorithm if it were not for their own interest. Secondly, the
FC cannot directly validate data driven gradient updates due to their stochastic nature. The quality

1We refer to the fusion center as “she" and a user as “he".

Submitted to 34th Conference on Neural Information Processing Systems (NeurIPS 2020). Do not distribute.

Figure 1: The fusion center (FC) trains the learning model with strategic users who are not obliged
to report their gradients. (a) The objective of the FC is to incentivize users to cooperate by giving
rewards so as to learn the model. (b) If the user is cooperative, he reports a privacy-preserved version
of his gradient signal. Otherwise, the user is defective and sends an arbitrary uninformative signal.
(c) The FC and the user each choose to cooperate or defect with respective payoffs as shown.

of the updates can vary over time and across the users since each user can control his own dataset.
The interactions among users and the FC are repeated, and each user is capable of devising intricate
strategies based on the past interactions. From a game-theoretic perspective, the fusion center’s ability
to reciprocate against non-cooperative user actions is signiﬁcantly restricted since she cannot directly
observe the user actions. Finally, the FC is not allowed to impose penalties on the users and positive
rewards are the only options at her disposal to incentivize user participation. The work proposed here
is, to the best of our knowledge, the ﬁrst distributed learning framework to consider these challenges.

In this study, we model the interactions (in terms of gradient reporting and reward) between the
FC and the users as repeated games, which intertwine with the updates in distributed learning. We
propose a reward mechanism for the fusion center, based on an adaptive zero-determinant strategy,
thereby generalizing the celebrated ZD strategy to the repeated games with time-varying stochastic
errors. To tackle the challenge that the FC cannot directly verify the received reported gradients,
we devise a gradient estimation and user action classiﬁcation. Our ﬁndings demonstrate that, by
employing adaptive ZD strategies, the FC can incentivize the strategic users to cooperate and report
informative gradient updates, thus ensuing the convergence of distributed learning.

Detailed discussion on related work is relegated to Appendix A, due to space limitation.

2 Distributed Learning with Strategic Users as Repeated Games

We consider a distributed learning setting with K strategic users K = {1, . . . , K} and a fusion center
(FC), and the optimization problem is given as follows:

min
θ∈Rn

F (θ) :=

1
K

K
(cid:88)

k=1

EZk∼D

(cid:2)L(θ; Zk)(cid:3),

(1)

(cid:80)s

(cid:1), where zi

i=1 ∇θL(cid:0)θt; zi

where L(·) is the loss function.
In each iteration, each user gets a mini-batch of s i.i.d. sam-
ples from an unknown distribution D, and computes the stochastic gradient signal as Xk,t :=
1
s
Stage Game Formulation: Actions and Payoffs. The action and the reported signal of user k in
iteration t are denoted with Bk,t ∈ {c, d} and Yk,t, respectively. As depicted in Fig. 1, a user is
cooperative (Bk,t = c) if he is sending the privacy-preserved version of his gradient Xk,t. Otherwise,
the user is defective and sends a noise signal Υk,t ∼ N (0, Ξt) independent of Xk,t:

k,t is the ith sampled data of user k at time t.

k,t

Yk,t =

(cid:26)Xk,t + Nk,t,

Υk,t,

if Bk,t = c (cooperative);
if Bk,t = d (defective).

(2)

Remark 1. Note that Nk,t is independent of Xk,t and Nk,t ∼ N ((cid:126)0, ν2
t I). If (cid:107)∇θL(θ; z)(cid:107)2 ≤ (cid:96) for all
θ and z, then this privacy-preservation mechanism enjoys (cid:15)t-differential privacy, with (cid:15)t = (cid:96)2(cid:14)s2ν2
t
for mini-batch size s. The details are provided in Appendix.

2

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

Fusion CenterFusion CenterCurrent ModelCurrent ModelStochasticGradientStochasticGradientkkReportedGradientReportedGradientPrivacyPreservedPrivacyPreservedArbitraryArbitraryUserUserFusion CenterFusion CenterCCDDccddFusion CenterFusion Center11......UsersUsersKK(a)(a)(b)(b)(c)(c)Random SignalRandom Signal64

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

The payoff structure of a single interplay between the fusion center and a user is depicted in Fig 1b.
In iteration t, when a user cooperates, he provides an information gain R to the FC at his privacy
cost VUR with 0 < VU ≤ 1. When a user defects, he does not provide any information gain and does
not incur any privacy cost. The FC may distribute rewards at the end of each iteration to incentivize
the users. We denote the action of the FC toward user k as Ak,t ∈ {C, D}. The FC is cooperative
(Ak,t= C) if she makes a payment r to the user at her cost rVFC with 0 < VFC ≤ 1. The FC is defective
(Ak,t= D), if she does not make any payment to the user. The factor VFC captures the difference in the
valuation of the reward between the FC and the user; for instance, the reward can be a coupon which
may be redeemed in the future. Denote the FC’s payoff vector by SFC = [R−rVFC, −rVFC, R, 0]
and that of the users by SU = [r−VUR, r, −VUR, 0]. In this paper, we only analyze the case where
R > rVFC and r > VUR. Otherwise, the FC or users do not have any incentive to cooperate.

The FC cannot observe the actions of the users and her realized payoffs. We assume that users do
not communicate or collude with each other. They cannot observe the actions of other users and the
actions of the FC toward other users. Next, we will discuss how to devise effective strategies for the
FC to incentivize cooperative user action for the repeated game in a cost-effective manner.

Repeated Games between Users and Fusion Center. A salient feature of 2 × 2 repeated games
is that players with longer memories of the history of the game have no advantage over those with
shorter ones when each stage game is identically repeated inﬁnite times [31]. Thus, without loss of
generality, we assume the user strategies only depend on the outcomes of the last round. Let q1, q2, q3
and q4 denote the probabilities of cooperation for the user conditioned on the joint action pair of
the previous iteration, that is (Ak,t−1, Bk,t−1), in the order of (C, c), (C, d), (D, c) and (D, d). The
user’s strategy vector is deﬁned as q = [q1, q2, q3, q4].

Analogous to the user strategies, let p1, p2, p3 and p4 denote the probabilities of cooperation for
the FC conditioned on (Ak,t−1, Bk,t), in the order of (C, c), (C, d), (D, c) and (D, d). The fusion
center’s strategy vector is deﬁned as p = [p1, p2, p3, p4]. The joint action pair of the user and the
FC is considered as the state of the game in iteration t: (Ak,t, Bk,t). The strategy vectors p and q
imply a Markov state transition matrix as follows:




Ω =




q1p1
q2p1
q3p3
q4p3

(1 − q1)p2
(1 − q2)p2
(1 − q3)p4
(1 − q4)p4

q1(1 − p1)
q2(1 − p1)
q3(1 − p3)
q4(1 − p3)

(1 − q1)(1 − p2)
(1 − q2)(1 − p2)
(1 − q3)(1 − p4)
(1 − q4)(1 − p4)


 .

(3)

Let Λ∗ be the stationary vector of the transition matrix Ω, i.e., Λ∗ = Λ∗Ω. We can ﬁnd the expected
payoffs of the FC and the user in the stationary state as s∗
U. The FC sets
her strategy p satisfying, for some real values ϕ0, ϕ1 and ϕ2, the equation

FC = Λ∗S(cid:62)

U = Λ∗S(cid:62)

FC and s∗

[p1 − 1, p2 − 1, p3, p4] = ϕ0SFC + ϕ1SU + ϕ21.
This class of strategies are called zero-determinant (ZD) strategies, which enforce a linear relation
between the expected payoffs, given by ϕ0s∗
FC +ϕ1s∗
U +ϕ2 = 0, regardless of the user strategy [31].
Remark 2. The ZD strategy is a powerful tool to incentivize the users cooperation for the FC
because she can unilaterally set s∗
FC.
Against such an FC strategy, the user’s best response which maximizes his payoff is full cooperation,
q∗ = [1 1 1 1]. The details are provided in Appendix C.

U or establish an extortionate linear relation between s∗

U and s∗

(4)

Against the FC who is equipped with the ZD strategy, the user can increase his expected payoff only
by cooperating more often, and consequently his best response is full cooperation. Assuming that
there are sufﬁciently many participating users, the FC has the absolute leverage against any single
user who tries to negotiate with her. Nevertheless, the FC cannot directly employ the ZD strategy
since she cannot observe the true actions of the users. In the next section, we will study the use of ZD
strategy can be extended in the scope of distributed learning.

3 Distributed Stochastic Gradient Descent with Strategic Users

For the ease of exposition, in this paper we focus on an interesting variant of the classical stochastic
gradient descent algorithm using the gradient signals reported by strategic users (SGD-SU). In each
iteration, the FC collects the reported gradients of the users and update the model as follows:

θt = θt−1 − ηt · (cid:98)mt(Yt),

(5)

3

Algorithm 1: Stochastic Gradient Descent with Strategic Users (SGD-SU)

1 for t = 1, 2, . . . , T − 1 do
2

Fusion Center: broadcast the current iterate θt−1 to all the users
forall k ∈ {1, 2, . . . , K} do

User k: compute the gradient Xk,t and Yk,t ←

(cid:26)Xk,t + Nk,t

Υk,t

1
K(Λ1Ωt−1)q(cid:62)

cooperative action,
defective action,
k=1 Yk,t

(cid:80)K

Fusion Center: form the gradient estimate (cid:98)mt(Yt) ←
update model parameter θt ← θt−1 − ηt (cid:98)mt(Yt)

(cid:40)

classify the users (cid:98)Bk,t ( (cid:98)mt, Yk,t) ←

(cooperative)

ˆc
ˆd (defective)

k,t (cid:98)mt > (cid:107) 1

2 (cid:98)mt(cid:107)2

2

if Y (cid:62)
else

(7)

compute the detection and false alarm probabilities using (8) and (11)
compute the adaptive strategies (9) and distribute the rewards accordingly

3

4

5

6

7

8

9

110

111

112

113

114

115

116

117

118

where Yt = [Y1,t . . . YK,t], ηt is the step size and (cid:98)mt is the gradient estimator. The FC cannot directly
observe user actions and verify the reported gradients. This gives rise to two coupled challenges:

• The gradient estimator (cid:98)mt should be resilient against the uninformative reports of defective users.
• Although the ZD strategies are powerful tools to incentivize user cooperation, the FC cannot

directly employ a ZD strategy because she cannot observe the users’ actions.

To tackle these difﬁculties, we will ﬁrst introduce a gradient estimation and user classiﬁcation scheme
and discuss the impact of user action classiﬁcation errors on the dynamics of repeated games. As
outlined in Algorithm 1. we will develop adaptive FC strategies which generalize the classical ZD
strategies to the repeated games with time-varying stochastic errors.

119

3.1

Joint Gradient Estimation and User Action Classiﬁcation

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

The stochastic gradients can be decomposed as Xk,t = mt + Wk,t where mt := ∇θF (θt) is the
population gradient and Wk,t is the zero-mean noise term [30]. The unknown parameter mt is the
mean of the reported gradient Yk,t when the user is cooperative (Bk,t = c). The defective users
send zero-mean random noise as their reported gradients. The FC needs to classify the reported
gradients and obtain an estimate of mt for the SGD-SU update in (5). These two problems are
coupled with each other, and the joint scheme is, therefore, comprised of a gradient estimator (cid:98)mt,
and a classiﬁcation rule (cid:98)Bk,t. To tackle this difﬁcult problem, we ﬁrst investigate gradient estimation.
Let Λ1 be the initial state distribution of the games between the users and the FC. A modiﬁed
empirical mean based gradient estimator can be employed as follows:

(cid:98)mt(Yt) :=

1
K(Λ1Ωt−1)q(cid:62)
It is easy to verify that (cid:98)mt(·) is an unbiased estimator if the FC is able to employ her strategies p
without any errors and the state distribution of the repeated games are governed by the state transition
matrix Ω as in (3) without any perturbations.
Using the gradient estimator (cid:98)mt(·), the FC can form the user action classiﬁcation rule as

Yk,t.

(6)

k=1

(cid:88)K

(cid:98)Bk,t ( (cid:98)mt(Yt), Yk,t) =

if Y (cid:62)

k,t (cid:98)mt >

1
2

(cid:40)
(cid:98)c
(cid:98)d else;

(cid:107) (cid:98)mt(cid:107)2,

(7)

where ˆd (or ˆc) is the defective (or cooperative) label. The noise in the stochastic gradients, Wk,t,
can be approximated as a zero mean Gaussian r.v. [17, 22, 26, 36]. Recall from (2) that cooperative
users send the privacy-preserved versions of their gradient. This implies Yk,t ∼ N (mt, Σt), given
Bk,t = c, where Σt := cov[Wk,t] + ν2
t I. Thus, the detection and false alarm probabilities of the
classiﬁer, denoted by Φt and Ψt respectively, can be found as
(cid:32)

(cid:33)

(cid:33)

Φt = 1 − Q

m(cid:62)

2 (cid:107) (cid:98)mt(cid:107)2

t (cid:98)mt − 1
(cid:113)
(cid:98)m(cid:62)

t Σt (cid:98)mt

(cid:32) 1
(cid:113)

2 (cid:107) (cid:98)mt(cid:107)2
(cid:98)m(cid:62)

t Ξt (cid:98)mt

.

(8)

and Ψt = Q

4

138

139

140

141

142

Remark 3. The linear classiﬁer (7) is an effective tool under the homoscedasticity assumption. If
that is violated, the FC can employ different classiﬁers. The details are provided in Appendix for the
Classiﬁer Design.

In the next subsection, we discuss how the FC can devise her strategies building on the joint gradient
estimation and user action classiﬁcation scheme.

143

3.2 Adaptive Strategies for Fusion Center

144

145

146

147

148

149

150

Although the ZD strategies, p, provide the FC an efﬁcient and powerful mechanism to encourage
the user’s cooperation; the FC cannot directly use p since they are conditioned on the user’s ac-
tion, Bk,t, which is not observable to her. Alternatively, the FC can use the classiﬁcation results
after carefully adapting her strategies to mitigate the adverse effects of inevitable classiﬁcation
errors. Let πt,1, πt,2, πt,3 and πt,4 denote the probabilities of cooperation for the FC conditioned
on (Ak,t−1, (cid:98)Bk,t), in the order of (C, ˆc), (C, ˆd), (D, ˆc) and (D, ˆd). These are referred to as adaptive
strategies and the FC sets these probabilities satisfying the following system of equations:

p1 = πt,1Φt + πt,2(1 − Φt),
p3 = πt,3Φt + πt,4(1 − Φt),

p2 = πt,1Ψt + πt,2(1 − Ψt),
p4 = πt,3Ψt + πt,4(1 − Ψt).

151

Suppose

Φt
Ψt

≥

p1
p2

and

Φt
Ψt

≥

p3
p4

. Then the unique solution to the system above is given by

πt,1 =

πt,3 =

p1(1 − Ψt) − p2(1 − Φt)
Φt − Ψt
p3(1 − Ψt) − p4(1 − Φt)
Φt − Ψt

,

,

πt,2 =

πt,4 =

p2Φt − p1Ψt
Φt − Ψt
p4Φt − p3Ψt
Φt − Ψt

,

.

(9a)

(9b)

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

Remark 4. If the FC directly employed the ZD strategies without any adaptation, i.e., she cooperates
with probability pi conditioned on classiﬁcation output; the repeated games may not converge to
the stationary state Λ∗ and a linear relation between the expected payoffs (4) may not be enforced
because the classiﬁcation errors yield an additive disturbance on the state transition matrix as follows

Ω − (p1 − p2) (cid:8)q(cid:62)[1−Φt 0 1−Φt 0] + (1 − q)(cid:62)[0 Ψt 0 Ψt](cid:9) .
(10)
Adaptive strategies (9) cancel out this adverse disturbance on the dynamics of the repeated games.

In the absence of classiﬁcation errors (Φt = 1 and Ψt = 0), the adaptive strategies reduce to the ZD
strategies, i.e., πt = p. Classiﬁcation errors force the FC to be more retaliatory than dictated by the
ZD strategy p, i.e., πt,1 > p1, πt,3 > p3, πt,2 < p2 and πt,4 < p4. In general, detection and false alarm
probabilities, Φt and Ψt, are time-varying; thus the adaptive strategies also change over time.

162

3.3 The Impact of Estimation Errors on Repeated Game Dynamics

163

164

165

The proposed adaptive strategies (9) requires the knowledge of detection probability, Φt. However,
the FC cannot exactly compute Φt using (8) since she does not have the knowledge of mt. Instead,
she can form her estimate (cid:98)Φt using (cid:98)mt:

(cid:98)Φt = 1 − Q

(cid:33)

(cid:32) 1
(cid:113)

2 (cid:107) (cid:98)mt(cid:107)2
(cid:98)m(cid:62)

t Σt (cid:98)mt

(11)

166

167

168

Due to the inevitable gradient estimation errors, in general, we have (cid:98)Φt (cid:54)= Φt. As a result, the FC
cannot exactly employ the adaptive FC strategies dictated by Eq. 9. With several steps of variable
substitutions, this yields an additive perturbation on the state transition matrix as follows:

(cid:101)Ωt = Ω + VtΩ⊥ with Vt := (cid:98)Φt − Φt
(cid:98)Φt − Ψt

and Ω⊥ := (p1 − p2)q(cid:62)[−1 0 1 0].

(12)

169

170

Let ˜Λt be the probability distribution over the state space of the games {Cc, Cd, Dc, Dd} at the start
of iteration t. According to (12), the state distributions follow the transition rule such that

(cid:101)Λt+1 = (cid:101)Λt (cid:101)Ωt = (cid:101)Λt

(cid:0)Ω + VtΩ⊥(cid:1) .

5

171

172

173

174

Note that Λt can be considered as the state distribution of the repeated games in the absence of
perturbations on the state transition matrix. For the FC, Λt is the designed state distribution in which
the ZD strategy dominates against any user strategy.
Next, we study the time-varying perturbation terms. Using (8) and (11), Vt can be found as2:

Vt =

(cid:98)Φt −Φt
(cid:98)Φt −Ψt

=

(cid:32)

(cid:98)m(cid:62)
t
(cid:113)

Q

(cid:16)
mt − 1

(cid:33)
(cid:17)
2 (cid:98)mt

(cid:32) 1
(cid:113)

−Q

(cid:98)m(cid:62)
(cid:32) 1
(cid:113)

t Σt (cid:98)mt
2 (cid:107) (cid:98)mt(cid:107)2
(cid:98)m(cid:62)

t Σt (cid:98)mt

1−Q

(cid:33)

−Q

(cid:32) 1
(cid:113)

2 (cid:107) (cid:98)mt(cid:107)2
(cid:98)m(cid:62)
t Σt (cid:98)mt
2 (cid:107) (cid:98)mt(cid:107)2
(cid:98)m(cid:62)
t Ξt (cid:98)mt

(cid:33)

(cid:33) =

Q

+ 1

(cid:32) (cid:99)mt(mt−(cid:99)mt)
(cid:107)mt(cid:107)
(cid:112)Ray(Σt, (cid:98)mt)
(cid:18)

2 (cid:107) (cid:98)mt(cid:107)

(cid:33)

(cid:18)

− Q

(cid:19)

1
2 (cid:107) (cid:98)mt(cid:107)
(cid:112)Ray(Σt, (cid:98)mt)

(cid:19) .

1−Q

1
2 (cid:107) (cid:98)mt(cid:107)
(cid:112)Ray(Σt, (cid:98)mt)

(cid:18)

(cid:19)

−Q

(cid:107)mt(cid:107)
(cid:112)Ray(Ξt, (cid:98)mt)

175

176

177

178

179

180

181

182

183

In the presence of these perturbations, to establish stability guarantees on the dynamics of the repeated
games, we impose the following assumption on the norm of the gradient estimator:
Assumption 1. We assume that (cid:107) (cid:98)mt(cid:107) ≥ max (cid:8)2(cid:112)Ray( (cid:98)mt, Σt), 2(cid:112)Ray( (cid:98)mt, Ξt), (cid:112)
t (mt − (cid:98)mt)(cid:9).
(cid:98)m(cid:62)
Note that these conditions are primarily associated to the accuracy of the linear classiﬁer (7) which
operates effectively when the mean vectors of the classes are sufﬁciently separated. The following
result indicates that, due to the perturbations on the state transition matrix, the real state distribution
˜Λt is a noisy version of Λt.
Lemma 1. Let Λ1 denote the initial state distributions of the games between the FC and the users.
Under Assumption 1, we have that

˜Λt = Λt + Λ1

(cid:88)t−1
i=1

ViΩi−1Ω⊥Ωt−1−i.

(13)

184

185

186

This noise on the state distributions will manifest as a novel bias term in the gradient estimation. In
the next subsection, we will provide the convergence analysis of SGD-SU which will mainly focus
on the characterization of this bias term.

187

3.4 Convergence Results

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

In this section, we provide the convergence guarantee for SGD-SU (5). Let Ft denote the σ-algebra,
generated by {θ1, Yi, i < t}. In particular, Ft should be interpreted as the history of SGD-SU up to
iteration t, just before Yt is generated. Thus, conditioning on Ft can be thought of as conditioning
on {θ1, (cid:101)Λ1, Y1, . . . , θt−1, (cid:101)Λt−1, Yt−1, θt, (cid:101)Λt}. For convenience, denote Et[·] := Et[·|Ft]. Observe
that, we can decompose the gradient estimator (cid:98)mt as follows:
(cid:98)mt(·) = mt(1 + ζt) + Et,
where ζt is the estimation bias term due to the perturbations on the state transition matrix, given by

(14)

ζt =

1
mt

(Et[ (cid:98)mt] − mt) =

(cid:80)K

k=1 P(Bk,t = c|Ft)
K(Λtq(cid:62))

− 1

and Et is the estimation noise term, given by Et = (cid:98)mt −Et[ (cid:98)mt]. Conditioned on Ft, the probability of
a user taking the cooperative action, in iteration t, is given by P(Bk,t = c|Ft) = (cid:101)Λtq(cid:62). The bias term,
ζt, can be found as follows:

ζt = (cid:101)Λtq(cid:62)

Λtq(cid:62) − 1.

(15)

From Lemma 1 and (15), it is clear that the perturbations on the state transition matrix (12), directly
translates into a bias in the gradient estimation rule.

To establish convergence guarantees for the SGD-SU in (5), Λtq(cid:62) and (cid:101)Λtq(cid:62) must meet the following
criteria during the course of the algorithm:
Assumption 2. We assume that Λtq(cid:62) > 1

2 and (cid:101)Λtq(cid:62) > 0, for all t ∈ {1, 2, . . . , T }.

2The Rayleigh’s quotient for a symmetric matrix M and nonzero vector x is deﬁned as Ray(M, x) =

x(cid:62)M x
x(cid:62)x

6

210

By (16) and (17), we have that

202

203

204

205

206

207

208

209

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

The ﬁrst condition Λtq(cid:62) ≥ 0.5 is very mild in the sense that it merely requires that the probability
of user cooperation dictated by the memory-1 strategies p and q (1 × 4 vectors), in the absence
of perturbations, is larger than 0.5. The second condition (cid:101)Λtq(cid:62) > 0 states that, in the presence of
perturbations, the probability of user cooperation is always positive3.

By Assumption 2, there exists a positive constant HT such that

0 < |ζt| < HT < 1, ∀t ∈ {1, . . . , T }.

(16)

Further, we have the following lemma characterizing the properties of estimation noise.
Lemma 2. Conditioned on Ft, the estimation noise in iteration t, denoted Et, is a zero-mean random
vector with the mean square error given by
(cid:16)

Et[(cid:107)Et(cid:107)2] =

1
K (cid:0)Λtq(cid:62)(cid:1)

(ζt + 1)tr (Σt − Ξt) +

1

(cid:17)
Λtq(cid:62) tr (Ξt)

.

(17)

Et

(cid:2)(cid:107)Et(cid:107)2(cid:3) ≤

ET
K

with ET :=

1
Λtq(cid:62)

(cid:20)
(cid:0)HT + 1(cid:1)tr(Σt − Ξt) +

1

Λtq(cid:62) tr(Ξt)

(cid:21)
.

(18)

We impose the following assumption on the objective function, which is standard for performance
analysis of stochastic gradient-based methods [3, 28].
Assumption 3. The objective function F and the SGD-SU satisfy the following:

(i) F is L−smooth, that is, F is differentiable and its gradient is L−Lipschitz:

(cid:107)∇F (θ) − ∇F (θ(cid:48))(cid:107) ≤ L(cid:107)θ − θ(cid:107), ∀θ, θ(cid:48) ∈ Rn.

(ii) The sequence of iterates {θt} is contained in an open set over which F is bounded below by

a scalar Finf .

Our next result describes the behavior of the sequence of gradients of F when ﬁxed step sizes are
employed.
Theorem 1. Under Assumptions 2 and 3, suppose that the SGD-SU (5) is run for T iterations with a
ﬁxed stepsize ¯β satisfying

1
L(1 + HT )
Then, the SGD algorithm with strategic users satisﬁes that

0 < ¯β ≤

.

(19)

E

(cid:20) 1
T

(cid:88)T

t=1

(cid:21)

(cid:107)∇F (θt)(cid:107)2

≤

LET
K(1 − HT )

+

2(F (θ1) − Finf )
¯βT (1 − HT )

.

Theorem 1 illustrates the impact of the perturbations on the state transition matrix (12) on the
convergence rate of SGD-SU. When HT is close to 0, SGD-SU performs similar to the basic
minibatch SGD. On the other hand, if HT is close to 1, the optimality gap may be large. Our next
result will characterize the gradient estimation bias term ζt. First, we have the following assumption
on the state transition matrix Ω.
Assumption 4. The state transition matrix Ω can be diagonalized as Ω = ΓUΓ−1 with U has the
eigenvalues of Ω in descending order of magnitude: 1 ≥ |u2| ≥ |u3| ≥ |u4| ≥ 0.

Denote the element of Γ−1 in the ith row and jth column as Γ−1
(cid:126)γ1, . . . , (cid:126)γ4. Next, we deﬁne δ as

ij . Denote the four rows of Γ−1 by

(cid:18)

δ :=

max
j∈{2,3,4}

(cid:19)(cid:18)

(cid:12)
(cid:12)Γ3j − Γ1j

(cid:12)
(cid:12)

max
j∈{2,3,4}

2(cid:19)
(cid:12)
(cid:12)(cid:126)γjq(cid:62)(cid:12)
.
(cid:12)

231

Further, the ﬁrst order Taylor approximation of the scalar variable Vt can be found as follows:

Vt =

m(cid:62)

t ( (cid:98)mt −mt)
(cid:107)mt(cid:107)2

ht(mt) with ht(mt) :=

(cid:107)mt(cid:107)
(cid:112)2πRay(Σt, mt)
(cid:32)

(cid:107)mt(cid:107)
2(cid:112)Ray(Σt, mt)

(cid:32)

exp

−

(cid:33)

− Q

(cid:33)

1

(cid:107)mt(cid:107)2
8
Ray(Σt, mt)
(cid:32)

(cid:107)mt(cid:107)
2(cid:112)Ray(Ξt, mt)

1 − Q

(cid:33) . (20)

3A sufﬁcient condition for this requirement is that user strategies are forgiving in nature, i.e., q1, q2, q3, q4 > 0.

7

232

233

234

t

:= maxi∈{1,...,t} hi(mi). Our next result indicates that, the estimation bias term ζt can

Deﬁne hmax
be found in terms of the past gradient estimation errors.
Theorem 2. Under Assumptions 1, 2 and 4, the gradient estimation bias term ζt, can be found as

235

with

ζt = (p1 − p2)

(cid:88)t−1
i=1

Λiq(cid:62)
Λtq(cid:62)

i Ei

m(cid:62)
(cid:107)mi(cid:107)2 hi(mi)∆i,t

|∆i,t| ≤ δ|u2|t−1−i + δ2hmax

t−1 |u2|t−2−i(t − i − 1).

236

Further, for some 0 < η < 1 we have

P (|ζt| < η|α1, . . . , αt−1) > 1 −

(cid:80)t−1

i=1 α2
i
Kη2

237

with

2

α2

i =

(cid:12)
(cid:12)
(ν2
(cid:12)
(cid:12)

i − ξ2

m(cid:62)

(cid:12)
(cid:12)
i ) +
(cid:12)
(cid:12)
(cid:107)mi(cid:107)2 (cid:0)Λiq(cid:62)(cid:1)

i Σimi
(cid:107)mi(cid:107)2

+

ξ2
i
Λiq(cid:62)

(cid:21)2

(cid:20) Λiq(cid:62)
Λtq(cid:62)

i ∆2
h2

i,t.

(21a)

(21b)

(22a)

(22b)

238

239

Note that Eq. (21) indicates that, the estimation bias term ζt can be expanded in terms of past gradient
estimation errors. We prove that the absolute values of the coefﬁcients, |∆i,t|’s, are bounded as

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

|∆i,t| ≤ δ|u2|t−1−i + δ2hmax

t−1 |u2|t−2−i(t − i − 1),

where u2 is the eigenvalue of Ω with the second highest absolute value. Since Ω is a row stochastic
matrix, |u2| ≤ 1. When |u2| is strictly less than 1, ∆i,t’s decay fast as t − i grows. This can also be
interpreted as the impact of past gradient estimation errors fade away quickly. Using this result, in
Eq.(22), we derive a high probability upper bound on the estimation bias term ζt.

4 Experiments

In this section, we evaluate the performance of SGD-SU (5) using real-life datasets. All the results in
the preceding section assert convergence for the SG method (5) under the assumption that the FC can
access Σt and Ξt. In a real-life machine learning setting with strategic users, this information may
not be available to the FC. For convenience, deﬁne (cid:98)Kc
t as the sets of users who are classiﬁed
as cooperative (ˆc) and defective ( ˆd) at iteration t. Based on the user action classiﬁcation, the FC can
form her estimates for the covariance matrices under the cooperative and defective actions as follows:

t and (cid:98)Kd

(cid:98)Σt =

1
| (cid:98)Kc
t |

k∈ (cid:98)Kc
t
(cid:80)

(cid:88)

(cid:0)Yk,t − ¯Y c

t

(cid:1) (cid:0)Yk,t − ¯Y c

t

(cid:1)(cid:62) and (cid:98)Ξt =

1
| (cid:98)Kd
t |

(cid:88)

k∈ (cid:98)Kd
t

(cid:0)Yk,t − ¯Y d

t

(cid:1)(cid:0)Yk,t − ¯Y d

t

(cid:1)(cid:62),

(23)

where ¯Y c

t = 1
| ˆKc
t |

k∈ ˆKc
t

Yk,t and ¯Y d

t = 1
| ˆKd
t |

(cid:80)

k∈ ˆKc
t

Yk,t.

In our ﬁrst set of experiments, we consider a binary logistic classiﬁcation problem and use the KDD-
Cup 04 dataset [6]. The goal of binary logistic classiﬁcation experiments is to learn a classiﬁcation
rule that differentiates between two types of particles generated in high energy collider experiments
based on 78 attributes [6]. In our second set of experiments, we consider a neural network trained on
the MNIST dataset. The number of users is chosen as K = 50 and mini-batch size is s = 10. In the
experiments, we have tested the performance of two different ZD strategies, namely equalizer and
extortion[31].

For the logistic classiﬁcation problem, Fig. 4a and 4b, depict the optimality gap under four different
user strategies: q = [0.9 0.15 0.9 0.15] (stubborn), q = [0.9 0.9 0.15 0.15] (tit-for-tat, ), q =
[0.9 0.15 0.15 0.9] (win-stay-lose-switch) and q = [0.9 0.9 0.9 0.9] (full cooperation). For the full
cooperation, coin toss, tit-for-tat and stubborn user strategies, SGSU converges quickly. For Pavlov
user strategies, SGSU can eventually approach, albeit more slowly than other cases. Fig 4c and 4d
illustrate the probability of user cooperation, (cid:101)Λtq(cid:62), across different user strategies. The experimental
results validate Lemma 1 and the empirical user cooperation probabilities match the theoretical except
when the users are Pavlov. Unsurprisingly, when the users follow full cooperation (or coin toss)
strategy, they cooperate with probability 0.9 (or 0.5) regardless of the actual states of the repeated

8

(a) Equalizer

(b) Extortion

(c) Equalizer

(d) Extortion

(e) Equalizer

(f) Extortion

(g) Equalizer

(h) Extortion

Figure 2: Stochastic Descent Algorithm with Strategic Users

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

games. For the cases with stubborn and tit-for-tat users, the games quickly converge to the steady
state distribution. Interestingly, for the cases with Pavlov users, the probability of user cooperation
decreases over time. This is associated to the performance of the linear classiﬁer. For the image
classiﬁcation problem, Fig 4e-h depict the training loss and testing accuracy across iterations for
different FC and user strategies. In all experiments, SGSU converges in the presence of strategic
users. Further details regarding the Experimental results are relegated to Appendix.

5 Future Directions

In this work, we study a distributed learning framework where strategic users train a learning model
with a fusion center. The main objective of the FC is to encourage users to be cooperative by
distributing rewards. Based on this, we devise a reward mechanism for the FC based on the ZD-
strategies. Further, we examine the performance of SGD algorithm in the presence of strategic users.
Our ﬁndings reveal that the algorithm has provable convergence and our empirical results verify our
theoretical analysis.

We are also working on the development of robust estimation tools in distributed learning with
strategic users. The geometric median is a reliable estimation technique when the collected data
contain outliers of large magnitude [10, 14, 24, 27]:

Med(Yt) := arg min
y∈Rn

(cid:88)K

k=1

(cid:107)y − Yk,t(cid:107)2.

(24)

The FC can use Med as a robust gradient estimator, especially when the variance of the uninformative
signals, ξ2
t , reported by the defective users, is very high. The geometric median (24) can be computed
by the Weiszfeld’s algorithm [34, 35], which is a special case of iteratively reweighted least squares.
In contrast, with the knowledge of q, the modiﬁed sample mean estimator (6) allows the FC to trade
robustness for overall tractability of the algorithm with reduced computational complexity.

The linear classiﬁer is vulnerable to vanishing gradients as the stochastic gradient descent algorithm
with strategic users (SGD-SU) converges to θ∗. This can be addressed by modifying the classiﬁer
to incorporate the information contained in the norm of the reported gradients. Furthermore, we
discuss how to extend the convergence guarantee for SGSU to allow heterogeneous user strategies.
The details are presented in Appendix.

References

[1] ALISTARH, D., ALLEN-ZHU, Z., AND LI, J. Byzantine stochastic gradient descent.

In

Advances in Neural Inform. Proc. Systems 31 (2018), NIPS’18, pp. 4613–4623.

9

0500100015002000-0.8-0.6-0.4-0.200.20.40.60.8StubbornTit-For-TatPavlovCoin TossFull Coop0500100015002000-0.8-0.6-0.4-0.200.20.40.60.8StubbornTit-For-TatPavlovCoin TossFull Coop05001000150020000.40.50.60.70.80.91StubbornTit-For-TatPavlovCoin TossFull Coop05001000150020000.30.40.50.60.70.80.91StubbornTit-For-TatPavlovCoin TossFull Coop050100150200Iterations0.20.40.60.81.0Testing AccuracyFull CoopStubbornTit-for-TatPavlov050100150200Iterations0.51.01.52.0Training LossFull CoopStubbornTit-for-TatPavlov050100150200Iterations0.51.01.52.0Training LossFull CoopStubbornTit-for-TatPavlov050100150200Iterations0.20.40.60.81.0Testing AccuracyFull CoopStubbornTit-for-TatPavlov297

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

[2] BLANCHARD, P., EL MHAMDI, E. M., GUERRAOUI, R., AND STAINER, J. Machine learning
with adversaries: Byzantine tolerant gradient descent. In Advances in Neural Inform. Proc.
Systems 30 (2017), NIPS’17, pp. 119–129.

[3] BOTTOU, L., CURTIS, F. E., AND NOCEDAL, J. Optimization methods for large-scale machine

learning. SIAM Review 60, 2 (2018), 223–311.

[4] CAI, Y., DASKALAKIS, C., AND PAPADIMITRIOU, C. Optimum statistical estimation with

strategic data sources. Journal of Machine Learning Research 40, 2015 (2015), 1–17.

[5] CARAGIANNIS, I., PROCACCIA, A. D., AND SHAH, N. Truthful univariate estimators. 33rd

International Conference on Machine Learning, ICML 2016 1 (2016), 200–210.

[6] CARUANA, R., JOACHIMS, T., AND BACKSTROM, L. Kdd-cup 2004: Results and analysis.

SIGKDD Explor. Newsl. 6, 2 (Dec. 2004), 95–108.

[7] CHEN, Y., IMMORLICA, N., LUCIER, B., SYRGKANIS, V., AND ZIANI, J. Optimal data
acquisition for statistical estimation. In Proceedings of the 2018 ACM Conference on Economics
and Computation (New York, NY, USA, 2018), EC ’18, Association for Computing Machinery,
p. 27–44.

[8] CHEN, Y., PODIMATA, C., PROCACCIA, A. D., AND SHAH, N. Strategyproof Linear
Regression in High Dimensions. In Proceedings of the 2018 ACM Conference on Economics
and Computation (New York, NY, USA, jun 2018), vol. 76, ACM, pp. 9–26.

[9] CHEN, Y., SU, L., AND XU, J. Distributed statistical machine learning in adversarial settings:

Byzantine gradient descent. Proc. ACM Meas. Anal. Comput. Syst. 1, 2 (Dec. 2017).

[10] COHEN, M. B., LEE, Y. T., MILLER, G., PACHOCKI, J., AND SIDFORD, A. Geometric
median in nearly linear time. In Proc. ACM Symp. on Theory of Comp. (New York, NY, USA,
2016), STOC ’16, ACM, p. 9–21.

[11] CUMMINGS, R., IOANNIDIS, S., AND LIGETT, K. Truthful linear regression. Journal of

Machine Learning Research 40, 2015 (2015), 1–36.

[12] DEKEL, O., FISCHER, F., AND PROCACCIA, A. D. Incentive compatible regression learning.

Journal of Computer and System Sciences 76, 8 (2010), 759–777.

[13] DWORK, C. Differential privacy. In Proc. Int. Conf. Automata, Languages and Programming -

Volume Part II (Berlin, Heidelberg, 2006), ICALP’06, Springer-Verlag, pp. 1–12.

[14] FLETCHER, P. T., VENKATASUBRAMANIAN, S., AND JOSHI, S. Robust statistics on rieman-
nian manifolds via the geometric median. In 2008 IEEE Conference on Computer Vision and
Pattern Recognition (2008), pp. 1–8.

[15] HAO, D., RONG, Z., AND ZHOU, T. Extortion under uncertainty: Zero-determinant strategies

in noisy games. Phys. Rev. E 91 (May 2015), 052803.

[16] HORN, R., HORN, R., AND JOHNSON, C. Matrix Analysis. Cambridge University Press, 1990.

[17] JASTRZKEBSKI, S., KENTON, Z., ARPIT, D., BALLAS, N., FISCHER, A., BENGIO, Y., AND
STORKEY, A. J. Three factors inﬂuencing minima in SGD. arXiv:1711.04623v3 [cs.LG] (Nov.
2017).

[18] JORDAN, M. I., LEE, J. D., AND YANG, Y. Communication-efﬁcient distributed statistical

inference. Journal of the American Statistical Association 114, 526 (2019), 668–681.

[19] KONE ˇCNÝ, J., MCMAHAN, H. B., RAMAGE, D., AND RICHTARIK, P. Federated optimization:
Distributed machine learning for on-device intelligence. arXiv:1610.02527 [cs.LG] (Oct. 2016).

[20] KONG, Y., SCHOENEBECK, G., TAO, B., AND YU, F.-Y. Information elicitation mechanisms
for statistical estimation. Proceedings of the AAAI Conference on Artiﬁcial Intelligence 34, 02
(Apr. 2020), 2095–2102.

[21] LI, M., ANDERSEN, D. G., PARK, J. W., SMOLA, A. J., AHMED, A., JOSIFOVSKI, V., LONG,
J., SHEKITA, E. J., AND SU, B.-Y. Scaling distributed machine learning with the parameter
server. In Proc. of the 11th USENIX Conf. on Operating Systems Design and Implementation
(USA, 2014), OSDI’14, USENIX Association, p. 583–598.

[22] LIN, T., STICH, S. U., PATEL, K. K., AND JAGGI, M. Don’t use large mini-batches, use local

sgd. In Int. Conf. Learning Representations (2020), ICLR’20.

10

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

[23] LIU, Y., AND WEI, J. Incentives for Federated Learning: A Hypothesis Elicitation Approach.

arXiv:2007.10596v1 [cs.LG] (July 2020).

[24] LOPUHAA, H. P., AND ROUSSEEUW, P. J. Breakdown points of afﬁne equivariant estimators
of multivariate location and covariance matrices. Ann. Statist. 19, 1 (03 1991), 229–248.
[25] LOW, Y., BICKSON, D., GONZALEZ, J., GUESTRIN, C., KYROLA, A., AND HELLERSTEIN,
J. M. Distributed graphlab: A framework for machine learning and data mining in the cloud.
Proc. VLDB Endow. 5, 8 (Apr. 2012), 716–727.

[26] MANDT, S., HOFFMAN, M. D., AND BLEI, D. M. A variational analysis of stochastic gradient
algorithms. In Proc. Int. Conf. Machine Learning (2016), vol. 48 of ICML’16, JMLR.org,
p. 354–363.

[27] MINSKER, S. Geometric median and robust estimation in banach spaces. Bernoulli 21, 4 (11

2015), 2308–2335.

[28] NEMIROVSKI, A., JUDITSKY, A., LAN, G., AND SHAPIRO, A. Robust stochastic approxima-
tion approach to stochastic programming. SIAM J. on Optimization 19, 4 (2009), 1574–1609.
[29] NG, K. L., CHEN, Z., LIU, Z., YU, H., LIU, Y., AND YANG, Q. A multi-player game for
studying federated learning incentive schemes. In Proceedings of the Twenty-Ninth International
Joint Conference on Artiﬁcial Intelligence, IJCAI-20 (7 2020), C. Bessiere, Ed., International
Joint Conferences on Artiﬁcial Intelligence Organization, pp. 5279–5281.

[30] POLYAK, B. T., AND JUDITSKY, A. B. Acceleration of stochastic approximation by averaging.

SIAM Journal on Control and Optimization 30, 4 (1992), 838–855.

[31] PRESS, W. H., AND DYSON, F. J. Iterated prisoner’s dilemma contains strategies that dominate

any evolutionary opponent. Proc. Natl. Acad. Sci 109, 26 (2012), 10409–10413.

[32] RICHARDSON, A., FILOS-RATSIKAS, A., AND FALTINGS, B. Budget-Bounded Incentives for

Federated Learning. Springer International Publishing, Cham, 2020, pp. 176–188.

[33] SU, L., AND XU, J. Securing distributed gradient descent in high dimensional statistical

learning. SIGMETRICS Perform. Eval. Rev. 47, 1 (Dec. 2019), 83–84.

[34] VARDI, Y., AND ZHANG, C.-H. The multivariate L1-median and associated data depth. Proc.

Natl. Acad. Sci. 97, 4 (2000), 1423–1426.

[35] WEISZFELD, E. Sur un probléme de minimum dans l’espace. Tohoku Math. J. 42 (1936),

274–280.

[36] XING, C., ARPIT, D., TSIRIGOTIS, C., AND BENGIO, Y. A Walk with SGD. arXiv:1802.08770

[stat.ML] (Feb. 2018).

[37] XING, E. P., HO, Q., XIE, P., AND WEI, D. Strategies and principles of distributed machine

learning on big data. Engineering 2, 2 (2016), 179 – 195.

11

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

Checklist

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reﬂect the paper’s

contributions and scope?

(b) Did you describe the limitations of your work?
(c) Did you discuss any potential negative societal impacts of your work?
(d) Have you read the ethics review guidelines and ensured that your paper conforms to

them?

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results?
(b) Did you include complete proofs of all theoretical results? Proofs are included in the

Appendix.
3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main experi-
mental results (either in the supplemental material or as a URL)? The code is included
in supplementary material complying to NeurIPS instructions with details on how to
run the code.

(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they

were chosen)?

(c) Did you report error bars (e.g., with respect to the random seed after running experi-

ments multiple times)?

(d) Did you include the total amount of compute and the type of resources used (e.g., type
of GPUs, internal cluster, or cloud provider)? The hardware used is described in the
Appendix

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators?
(b) Did you mention the license of the assets?
(c) Did you include any new assets either in the supplemental material or as a URL? Our

code is included in the supplementary material.

(d) Did you discuss whether and how consent was obtained from people whose data you’re

using/curating?

(e) Did you discuss whether the data you are using/curating contains personally identiﬁable

information or offensive content?

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if

applicable?

(b) Did you describe any potential participant risks, with links to Institutional Review

Board (IRB) approvals, if applicable?

(c) Did you include the estimated hourly wage paid to participants and the total amount

spent on participant compensation?

12

