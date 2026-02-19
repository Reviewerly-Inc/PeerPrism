Federated Hypergradient Descent

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

35

In this work, we explore combining automatic hyperparameter tuning and opti-
mization for federated learning (FL) in an online, one-shot procedure. We apply
a principled approach on a method for adaptive client learning rate, number of
local steps, and batch size. In our federated learning applications, our primary
motivations are minimizing communication budget as well as local computational
resources in the training pipeline. Conventionally, hyperparameter tuning meth-
ods involve at least some degree of trial-and-error, which is known to be sample
inefﬁcient. In order to address our motivations, we propose FATHOM (Federated
AuTomatic Hyperparameter OptiMization) as a one-shot online procedure. We
investigate the challenges and solutions of deriving analytical gradients with respect
to the hyperparameters of interest. Our approach is inspired by the fact that we
have full knowledge of all components involved in our training process, and this
fact can be exploited in our algorithm impactfully. We show that FATHOM is
more communication efﬁcient than Federated Averaging (FedAvg) with optimized,
static valued hyperparameters, and is also more computationally efﬁcient overall.
As a communication efﬁcient, one-shot online procedure, FATHOM solves the
bottleneck of costly communication and limited local computation, by eliminat-
ing a potentially wasteful tuning process, and by optimizing the hyperparamters
adaptively throughout the training procedure without trial-and-error. We show
our numerical results through extensive empirical experiments with the Federated
EMNIST-62 (FEMNIST) and Federated Stack Overﬂow (FSO) datasets, using
FedJAX as our baseline framework.

1

Introduction

Federated learning (FL) for on-device applications has its obvious social implications, due to its
inherent privacy-protection feature. It opens up a broad range of opportunities to allow a massive
number of devices to collaborate in developing a shared model by retaining private data on the
devices. The ubiquity of machine learning (ML) on consumer data, coupled with the growth of privacy
concerns, has pushed researchers and developers to look for new ways to protect and beneﬁt end-users.
In order for FL to deliver its promise in deployed applications, there are still many open challenges
remained to be solved. We are especially interested in the overall communication efﬁciency of the FL
pipeline for it to be realistically deployed in a unique communication environment over expensive
links. To begin, consider a typical step in a machine learning (ML) pipeline: hyperparameter tuning.
Whether it is in a centralized, distributed or federated setting, it is an essential step to achieve an
optimal operation for the training process. At the heart of an ML training process is the optimization
algorithm. In particular, we are interested in using Federated Averaging (FedAvg) as our baseline

Submitted to 36th Conference on Neural Information Processing Systems (NeurIPS 2022). Do not distribute.

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

federated optimization algorithm for our work. This is because, despite all the recent innovations in
FL since its introduction in 2016 by McMahan et al. [2016], FedAvg remains the de facto standard in
federated optimization for both research and practice, due to its simplicity and empirical effectiveness.
In order for FedAvg to operate effectively, it requires properly tuned hyperparameter values.

Our work focuses speciﬁcally on hyperparameter optimization (HPO) of: 1) client learning rate,
2) number of local steps, as well as 3) batch size, for FedAvg. We propose FATHOM (Federated
AuTomatic Hyperparameter OptiMization), which is an online algorithm that operates as a one-shot
procedure. In the rest of this paper, we will go through a few notable recent state-of-the-art works on
this topic, and make justiﬁcations for our new approach. Then we will derive a few key steps for our
algorithm, followed by a theoretical convergence bound for adaptive learning rate and number of local
steps in the non-convex regime. Lastly, we present numerical results on our empirical experiments
with neural networks on the FEMNIST and FSO datasets.

Our contributions are as follows:

• We derive gradients with respect to client learning rate and number of local steps for FedAvg,
for an online optimization procedure. We propose FATHOM, a practical one-shot procedure
for joint-optimization of hyperparameters and model parameters, for FedAvg.

• We derive a new convergence upper-bound with a relaxed condition (see Section 4 and
remark 2), to highlight the beneﬁts from the extra degree-of-freedom that FATHOM delivers
for performance gains.

• We present empirical results that show state-of-the-art performance. To our knowledge,
we are the ﬁrst to show gain from an online HPO procedure over a well-tuned equivalent
procedure with ﬁxed hyperparameter values.

2 Related Work and Justiﬁcations for FATHOM

We explore the question whether the FATHOM approach is justiﬁed over the more recent, state-of-
the-art methods that are designed for the same goal: a single-shot online hyperparamter optimization
procedure for FL. Zhou et al. [2022] proposed Federated Loss SuRface Aggregation (FLoRA), a
general single-shot HPO for FL, which works by treating HPO as a black-box problem and by
performing loss surface aggregation for training the global model. Khodak et al. [2021] draws
inspiration from weight-sharing in Neural Architectural Search (Pham et al. [2018], Cai et al. [2019]),
and proposed FedEx, which is an online hyperparameter tuning algorithm that uses exponentiated
gradients to update hyperparameters. On the other hand, Mostafa [2019]’s RMAH and Guo et al.
[2022]’s Auto-FedRL both use REINFORCE (Williams [1992]) in their agents to update hyperparam-
eters in an online manner, by using relative loss as their trial rewards. One basic assumption among
these methods, is that at least some of the gradients with respect to the hyperparameters are unavail-
able directly. Generalized techniques are used to update these quantities, involving Monte-Carlo
sampling and evaluation with held-out data. One key beneﬁt with techniques such as these is their
generalizability for a wide range of different hyperparameters. On the other hand, we identify a few
areas with these methods that we would like to improve on. One, information about the internals of
the procedure can and should be exploited. Two, communication overhead becomes a concern, since
sufﬁcient Monte-Carlo sampling is required for some of these techniques to converge, an example
being the re-parametrization trick (Kingma and Welling [2013]) which is used for FedEx, RMAH and
Auto-FedRL. From initial observations of their empirical results, while these methods are successful
in hyperparameter tuning and reaching target model accuracy as shown in these works, these goals are
achieved in unspeciﬁed numbers of total communication rounds from works based on RL approaches
such as Mostafa [2019] and Guo et al. [2022].

The above observations justify exploring our problem differently from previous approaches. Our
method exploits full knowledge of the training process, and it does not require sufﬁcient trials
at potential expense of communication budget. Inspired by the hypergradient descent techniques
developed by Baydin et al. [2017] and Amid et al. [2022] for centralized optimization learning rate,
we develop FATHOM by directing deriving analytical gradients with respect to the hyperparameters
of interest. The result is a sample efﬁcient method which offers both improvements in communication
efﬁciency and reduced local computation in a single-shot online optimization procedure. Meanwhile,
FATHOM is not as ﬂexibly applicable in optimizing a wide range of hyperparameters, since each

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

gradient needs to be derived separately to take advantage of our full knowledge of the training process.
We believe this approach is a performance advantage, at the expense of its ﬂexibility.

There are other notable relevant works. Charles and Koneˇcný [2020] and Li et al. [2019] proved that
reducing the client learning rate during training is necessary to reach the true objective. Yet, a line of
interesting works, such as Dai et al. [2020] and Holly et al. [2021]) applies Bayesian Optimization
(BO) on federated hyperparamter tuning, by treating it as a closed-box optimization problem. Dai
et al. [2021] further updates their use of BO in FL by incorporating differential privacy. However,
these BO-based works do not consider adaptive hyperparameters. Yet, another work (Wang and Joshi
[2018]) shares similarity to our approach of optimally adapting the number of local steps, with their
adaptive communication strategy, AdaComm, in the distributed setting. However, their main interest
is reducing wall-clock time. Lastly, around the same time of this writing, Wang et al. [2022] publishes
their benchmark suite for FL HPO, called FedHPO-B, which would be valuable to our future work.

3 Methodology

In this section we formalize the problem of hyperparameter optimization (HPO) for FL. We ﬁrst
review FedAvg, a de facto standard of federated optimization methods for research baseline and
practice. Then, we present our method for online-tuning of its hyperparameters, speciﬁcally client
learning rate, number of local steps, and batch size. We call our method FATHOM (Federated
AuTomatic Hyperparameter OptiMization).

107

3.1 Problem Deﬁnition

108

109

In this paper, we consider the empirical risk minimization (ERM) across all the client data, as an
unconstrained optimization problem:

f ⇤ := min

f (x) :=

x

2Rd "

m

fi(x)

1
m

#

(1)

where fi : Rd
dimension of the parameters x, m is number of clients, and f ⇤ = f (x
solution to the ERM problem in eq(1).

i=1
X
R is the loss function for data stored in local client index i with d being the
is a stationary

) where x

!

⇤

⇤

To facilitate some of the discussions that follow, it helps to deﬁne assumptions here as we do
throughout the rest of this paper:
Assumption 1. (Unbiased Local Gradient Estimator) Let gi(x) be the unbiased, local gradient
estimator of

fi(x), i.e., E[gi(x)] =

x, and i

fi(x),

[m].

r

r

8

2

117

3.2 Federated Optimization and Tuning of Hyperparameters

b

c

E⌫i/B

Federated Averaging (FedAvg) We describe the operations of FedAvg from McMahan et al.
[2016], as follows. At any round t, each of the m clients takes a total of Ki local SGD steps, where
Ki =
, and where ⌫i is the number of data samples from client index i, B is batch size,
with epoch number E = 1 being a common baseline. In this version of FedAvg, heterogeneous data
size is accommodated across clients, and the number of local steps can be manipulated via E and
B as hyperparameters. Each local SGD step updates the local model parameters of each client i as
follows: xi
[K] is the local step
t,k), where ⌘L is the local learning rate and k
index. To conclude each round, these clients return the local parameters xi
t,Ki to the server where
it updates its global model, with xt+1 =
i ⌫i. To facilitate some of the
discussions that follow, we deﬁne the following quantities:

t,K/⌫ where ⌫ =

t,k+1 = xi

⌘Lgi(xi

i ⌫ixi

t,k  

2

 t , xt+1  

xt =

m

i=1
X

 i

t where  i

t ,  

⌘L,tgi(xi,k
t )

(2)

P
1
Ki 

Xk=0

Ofﬂine Hyperparameter Tuning Ofﬂine tuning is best to be summarized as follows. We ﬁrst
deﬁne U =
V . We also deﬁne
C. Ofﬂine tuning would have the following objective:
C = U

with ⌘L 2
 
V , and c = (⌘L, K), where c

U , and V =

with K

2I |

u
{

0
}

1
}

v
{

 

R

2

2

u

v

|

⇥

P
⌫i
⌫

2

3

110

111

112

113

114

115

116

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

2

C fvalid(x, c) s.t. x = argminz

2Rd ftrain(z, c) . With abuse of notation, we use fvalid for the
minc
objective function calculated from a validation dataset which is usually held-out before the procedure,
and ftrain for the objective from training data which usually is just local client data. A few notable
ofﬂine tuning methods are as follows. Global grid-search from Holly et al. [2021] is an example
of ofﬂine tuning that iterates over the entire search grid deﬁned as C, completing an optimization
process for each grid point and evaluating the result with a held-out validation set. Global Bayesian
Optimization from Holly et al. [2021] is another similar example of ofﬂine tuning that follows the
same template and objective. Instead of brute-force grid-search, c is sampled from a distribution
DC
over C, i.e. c

⇠D C, that updates after every iteration.

Online Hyperparameter Optimization We are interested in an online procedure that combines
hyperparameter optimization and model parameter optimization, with the following objective:

ftrain(x, c)

(3)

This formulation is the objective of our method, FATHOM, which we will discuss shortly in detail. It
has the advantage of joint optimization in a one-shot procedure. Furthermore, it does not assume the
availability of a validation dataset.

min
d
x
2R
C
c
2

145

3.3 Our Method: FATHOM

In this section we will introduce our method, FATHOM (Federated AuTomatic Hyperparameter
OptiMization). Recall from our joint objective, eq(3), that both the model parameters, x, and
hyperparameters of the optimization algorithm, c, are optimized jointly to minimize our objective
function. An alternative view is to treat c as part of the parameters being optimized in a classic
formulation, i.e. minyf (y) with y = (x, c). As previously mentioned, our method is inspired by
hypergradient descent from Baydin et al. [2017] and by exponentiated gradient from Amid et al.
[2022], both proposed for centralized learning rate optimization. We will present how FATHOM
exploits our knowledge of analytical gradients to update client learning rate, number of local steps, as
well as batch size, for an online, one-shot optimization procedure.
Assumption 2. (Convexity w.r.t. ⌘L and K) We assume Et(f (xt)) is convex w.r.t. ⌘L and K, even
though we assume non-convexity w.r.t. xt). Speciﬁcally, convexity w.r.t. K follows the deﬁnition in
Murota [1998], to accommodate the integer space where K is deﬁned.
Remark 1. Assumption 2 is necessary to guarantee the existence of subgradients derived in Theorems
1 and 2, and it will be assumed for this work. In problems dealing with deep neural networks, it is
reasonable to not assume convexity w.r.t. hyperparameters. However, from our empirical results, we
claim that the proposed algorithm is still able to operate as desired under this condition.

3.3.1 Hypergradient for Client Learning Rate

In this section, we derive the hypergradient for client learning rate in a similar fashion as Baydin et al.
[2017], with the difference being that they are mainly concerned with the centralized optimization
problem, and that we are concerned with the distributed setting where clients take local steps. We
derive the following hypergradient of the objective function as deﬁned in eq(1), taken with respect to
the learning rate ⌘L,t

 

1 such that it can be updated to obtain ⌘L,t:
@f (xt)
@⌘L,t

@f (xt)
@xt

 
@⌘L,t

1 +  t

@(xt

1)

=

=

 

1

1

Ht =

f (xt)

1

@ t
 
@⌘L,t

·
where  t is the update step for the global parameters xt as deﬁned in eq(2), leading to @ t
@⌘L,t
⌘L,tr
t ). We also make the approximation xt+1  

xt =  t ⇡  
 
can then write the normalized update, H t, similar to Amid et al. [2022], as follows:

k=0 gi(xi,k

m
i=1

⌫i
⌫

r

K

 

 

 

 

1

1

·

(4)

=

=  t
⌘L,t
f (xt). We

H t = r
kr

f (xt)
f (xt)

·

k

1

@ t
 
@⌘L,t

1

1

@ t
 
@⌘L,t

1

⇡  

 t
 tk
k

·

 t
 t
k

1k

⇣

. 
The resulting hypergradient is a scalar, as expected, and can be used efﬁciently as part of the update
 
 
rule for ⌘L, which we will see in Section 3.3.4. The implementation is communication efﬁcient, since
in each round, each client needs one extra scalar to send back to the server, and likewise the server
needs to broadcast one extra scalar back to the clients. It is also computationally efﬁcient since it
avoids calculating the full local gradient

 
 
 

f (xt).

⌘

 

 

 

1

 

(5)

P

P

r

4

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

3.3.2 Hypergradient for Number of Local Steps

Since the number of local steps is an integer, i.e. K =
, this means f (xt) does not
exist for non-integer values of K. We formulate a subgradient as a surrogate of the hypergradient
@f (xt)/@K, as follows. We will call this a hyper-subgradient.
Theorem 1. When a piecewise function Lt is deﬁned for every value of K0 2
0.0

Kt = K0:

[K] on l, such that
l < 1.0, we claim, under Assumption 2, that the following is a subgradient of f (xt) at

k
{

1
}

 

2

k

I

|

@Lt
@l

f (xt)

=

r

·

 

⌘L,t

gi(xi,Kt 
t
 

1

1

)

⌫i
⌫

(6)

where l represents the marginal fraction of local steps beyond K0. We leave the proof (with an
illustration in Figure 2) in the Appendix section beginning in eq(20).

 

 

m

i=1
X

The result from Theorem 1 is not sufﬁciently communication-efﬁcient for implementing an update
rule for K. This is because it would require the quantity gi(xi,Kt 
) to be communicated from
each client i to the server. To save communication, let us reuse what the server has in memory:
 t =

. If we let:

Kt 
k=0 gi(xi,k
t )

m
i=1

⌘L

 

1

1

1

t

⌫i
⌫

 

 

P

P
St =

f (xt)

r

Nt =

@St
@l

f (xt)

=

r

·

·

N t = r
kr

f (xt)
f (xt)

·

k

 

 

 

 

 

m

i=1
X
m

⌘L,t

⌘L,t

i=1
X
1

 

1k

 

 t
 t
k

Kt 

1

Xk=0
1
Kt 

⌫i
⌫

⌫i
⌫

gi(xi,k
t
 

1)

l

 

gi(xi,k
t
 

1)

Xk=0
 t
 tk

k

 
1

 

1k

 

 t
 t
k

·

⇡  

f (xt)

=

r

 t

1

 

·

(7)

(8)

(9)

where eq(9) is the normalized update as in Amid et al. [2022]. We claim that eq(8) is a positively-
biased version of eq(6), which has its practical importance due to the fact that the last term in eq(6)
from Theorem 1 results in zero-mean, noisy gradients, when the local functions are nearing their local
solutions, when in fact, this is the area where more local work is not needed. Thus, a positive bias is
desirable to drive the number of local steps down. This result is also useful from a communication
efﬁciency perspective in its implementation, because the server has all the components to calculate
this quantity, and would not require additional communication.

3.3.3 Regularization for Number of Local Steps

One of the goals for FATHOM is savings in local computation. To avoid excessive number of local
steps, we further develop a regularization term for local computation against excessive K, which is a
proxy for the hypergradient of the local client functions at the end of each round : @fi(xi,K
Theorem 2. When a piecewise function Jt is deﬁned for every value of K0 2
0.0
at Kt = K0:

l < 1.0, we claim, under Assumption 2, that the following is a subgradient of

t
[K] on l, such that
i=1 fi(xi,Kt
)

)/@K.



m

t

@Jt
@l

=

⌘L,t

 

⌫i
⌫

gi(xi,K0 
t

1

)

E

gi(xi,Kt
t

)

·

⌘L,t

⇡  

⌫i
⌫

m

i=1
X

m

Kt 

1

i=1
X

Xk=0

gi(xi,k
t )

·

P
gi(xi,Kt
t

)

(10)

where l represents the marginal fraction of local steps beyond K0. We leave the proof in the Appendix
section beginning in eq(24).

⇥

⇤

In our algorithm, we use the normalized update based on the following biased proxy, since eq(10)
tends to be noisy from gi(xi,Kt

).

t

Gt =

⌘L,t

 

Gt =

⌘L,t

 

⌫i
⌫

⌫i
⌫

m

i=1
X
m

i=1
X

min
Kt
K


K

1

 

⇣

Xk=0

min
K


Kt   P
 
  P
5

gi(xi,k
t )

·

gi(xi,K
t

)

1

K

 

k=0 gi(xi,k
t )
k=0 gi(xi,k
t )

K

 

1

⌘
gi(xi,K
t
gi(xi,K
t
k

)

)

·

!

k

 
 

(11)

(12)

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

where Gt is the normalized update. The proxy yields a bias towards smaller number of local steps,
which is desirable for reducing local computation. We use this biased proxy against using a more
typical regularization such as L2 for the number of local steps, based on initial empirical results for
better performance..

3.3.4 Normalized Exponentiated Gradient Updates

For the update rules of the hyperparameters ⌘L (client learning rate) and K (number of client local
steps), we use the normalized exponentiated gradient descent method (EGN) with no momentum,
rather than a conventional linear update method such as the additive update of hypergradient descent
proposed in Baydin et al. [2017]. It is reasonable to use exponentiated gradient (EG) methods for
updates of hyperparameters that are strictly positive in value. EG methods also enjoy signiﬁcantly
faster convergence properties when only a small subset of the dimensions are relevant, according to
Amid et al. [2022].

EG methods have been proposed in previous works for a variety of applications (Khodak et al. [2021],
Amid et al. [2022], Li et al. [2020]), and analyzed in depth (Ghai et al. [2019]), where its convergence
has been studied and validated (Li and Cevher [2018]). Recently, Amid et al. [2022] showed that EGN
is the same as the multiplicative update for hypergradient descent proposed in Baydin et al. [2017],
when the approximation exp(
is made. From our observations, we believe that momentum
is not needed for the effectiveness of EGN in our application, as validated in our numerical results.
We also opted-out of adding further complexity such as extra weights and activation functions to
model the relationships between ⌘L,t and Kt, because it would require more samples to optimize and
because FATHOM is a one-shot procedure. Furthermore, due to the non-stationary nature of these
values, we opt for a simpler scheme for faster performance.

1 +

⇡

)

·

·

Hence, for the update rule of client learning rate, ⌘L, we have:

⌘L,t+1 = ⌘L,t exp

 ⌘H t

 

(13)

where H t is as deﬁned in eq(5). For number of local steps, we observe that it is related to batch
size in round t, Bt, as follows. To accommodate heterogeneity of local dataset sizes among clients,
we have number of local data samples from client i to be ⌫i. The number of local steps for client i
is Ki =
, where Et is number of epochs, with Et = 1 meaning the entire local dataset
for each client to be processed once per round. We derive update rules for Et and Bt globally to
optimize the number of local steps, without having to make any changes to our theoretical analysis to
accommodate the heterogeneity of local dataset sizes:

⌫iEt/Btc

b

 

 

Et+1 = Et exp

 E

N t + Gt

 

(14)

237

and

 
Bt+1 = Bt exp

 
 B

  

(15)
where Nt and Gt are deﬁned in eq(9) and eq(12), respectively. These update rules accomplish the goal
 
 
of updating the number of local steps via Et/Bt with Et+1
.
Bt+1
 E)Gt becomes a tunable regularization term as discussed at the
 E, ( B  
Typically, with  B  
 
end of Section 3.3.3.

 EN t  

  
= Et
Bt

 E  

exp

 B

Gt

 

 

 

 

 

 

Gt

3.3.5 Client Sampling

We present our method, FATHOM, as shown in Algorithm 1. One practical factor we have not
considered in our discussions is partial client sampling. For our implementation to handle the
1 for calculating H t in eq(5) and N t in eq(9) is
stochastic nature of client sampling, the metric  t
↵) t, which is a
1,sm + (1
modiﬁed by a smoothing function for noise ﬁltering, i.e.  t,sm = ↵ t
single-pole inﬁnite impulse response ﬁlter (Oppenheim and Schafer [2009]Oppenheimer et al. [2009])
with no bias compensation. We use the notation "sm" for smoothed, and after many experiments, we
decide to use ↵ = 0.5 for all of our numerical results.

 

 

 

6

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

Algorithm 1: FATHOM : gi(x) is deﬁned in Assumptions 1, and m is the number of clients.
Input: Server initializes global model xt=1, T as the end communication round, and:

 t=0,sm = 0 ; ↵ = 0.5 ;  ⌘ = 0.01 ;  E = 0.01 ;  B = 0.1

Output: xT , as well as ⌘L,t, Et and Bt for all t
for t = 1, . . . , T do

[T ]

2

Sample client set St out of m clients.
For each client i
Set  i = 0, and  i = +
for k = 0, . . . , Kt,i  

St, initialize: xi,k=0

.
1
1 do

2

t

= xt and Kt,i =

⌫iEt/Bt

.

c

b

For each client i, compute in parallel an unbiased stochastic gradient gi(xi,k
t ).
For each client i, calculate  i = min( i, gi(xi,k
xt
t )
t  
⌘L,tgi(xi,k
For each client i, update in parallel its local solution: xi,k+1
t )

 i) where  i = xi,k

= xi,k

·

t

t  

⌫i, where ⌫i is the size of client i dataset.

end
Server calcualtes ⌫ =
Server calculates  t =
Server updates global model xt+1 = xt  
Server calculates H t = N t =

P

P

2
i

St

St

2

i

 i(⌫i/⌫); see eq(2)

 t

 t
 tk ·

 t
 t

 

1,sm
1,smk

 
Server calculates Gt; see eq(12
Server updates client learning rate ⌘L,t+1, epochs, Et+1, and batch size Bt+1 for the next
round; see eq(13), eq(14), and eq(15).
Server updates  t,sm = (1

1,sm for the next round

↵) t + ↵ t

k

k

 

, modiﬁed from eq(5) and eq(9)

 

 

end

4 Theoretical Convergence

A standard approach to theoretical analysis of an online optimization method such as ours, is through
analyzing the regret bound (Zinkevich [2003], Khodak et al. [2019], Kingma and Ba [2014], and
Mokhtari et al. [2016]). Nonetheless, this approach does not tell us the impact on communication
efﬁciency by the online updates introduced from FATHOM. Therefore, we take an alternative
approach by extending the guarantees of FedAvg performance (Wang et al. [2021], Reddi et al.
[2020], Gorbunov et al. [2020], Yang et al. [2021], Li et al. [2019], etc) to include both adaptive
learning rate and adaptive number of local steps. We assume the special case in our analysis to have
full client participation. We prove that adaptive learning rate and adaptive number of local steps does
not impact asymptotic convergence, despite the given relaxed conditions.

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

4.1 Assumptions

262

263

264

265

266

267

268

L

kr

  r

fi(y)

Assumption 3. (L-Lipschitz Continuous Gradient for Parameters xt) There exists a constant L > 0,
[m], where x and y are the
x, y
fi(x)
such that
parameters in eq(1.
Assumption 4. (Bounded Local Variance) There exist a constant  L > 0, such that the variance of
[m].
each local gradient estimator is bounded by E
Assumption 5. (Bounded Second Moment) There exists a constant G > 0, such that Etkr
G, i

Rd, and i

x, and i

fi(xt)

fi(x)

gi(x)

 2
L,

x
k

k 

[m],

2
k

k 

kr

 



 

2

2

2

8

8

k

y

,

xt.

2

8

269

4.2 Convergence Results

270

271

272

273

Theorem 3. Under Assumptions 1-5 and with full client participation, when FATHOM as shown
in Algorithm 1 is used to ﬁnd a solution x
to the unconstrained problem deﬁned in eq(1), the
xt}
sequence of outputs
satisﬁes the following upper-bound, where, with slight abuse of notation,
{
2
f (xt)
2:
[T ] Etkr
k

= mint

E

2

⇤

Ef athom =

O

✓s

 2
L + G2
mKT

+ 3
s

 2
L
KT 2

G2
T 2

+ 3
r

◆

(16)

7

274

275

with the following conditions: ⌘L = min

and ⌘L,t 

1/L for all t, where

2 0mD

 1KLT ( 2

L+G2)

, 3

q

q

 0D
2

2.5 2K

L2 2

LT

, 3

q

 0D
3

2.5 3K

L2G2T !

⌘L ,

1
T

T

t=1
X

⌘L,t

and

K ,

1
T

T

t=1
X

Kt

276

and where

 0 =

T [ 1
T

t ⌘L,tKt
t ⌘L,t][ 1
P
T
2
1
t ⌘L,t
P
T
L,tK 2
t ⌘3
t
⇤
P

⇥

P

P

t Kt]
t Kt

⇤

,  1 =

,  3 =

P

P

t ⌘L,t

1
t ⌘L,tKt
T
t ⌘2
L,tKt
⇥
P
1
t ⌘L,tKt
P
T

⇤

2

t ⌘L,t
L,tK 3
t ⌘3
t
⇤
P

⇥

 2 =

t ⌘L,tKt

1
T

P

⇥

(17)

(18)

(19)

1
T

2

t Kt

⇥

P

⇤

277

We leave the proof in the Appendix beginning in eq(29).

P

P

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

The values of  0,  1,  2,  3, and  4 are dependent on the relative changes over the adaptive process
of these components, according to Chebyshev’s Sum Inequalities (Hardy et al. [1988]). A special
case is when these quantities equal to 1 when both ⌘L,t and Kt are constant, which recovers the
standard upperbound for FedAvg from eq(16).
Remark 2. The deﬁnitions in eq(17) combined with the conditions for ⌘L above is called the relaxed
conditions in this paper for the hyperparameters ⌘L,t and Kt. The values of ⌘L,t and Kt are adaptive
during the optimization process between rounds t = 1 and t = T , as long as the above conditions are
satisﬁed for the guarantee in eq(31) to hold. This relaxation presents opportunities for a scheme such
as FATHOM to exploit for performance gain. For example, suppose T approaches
for a prolonged
training session. Then ⌘L would necessarily be sufﬁciently small for
Ef athom to be bounded by
T ⌘L can be reasonably large and still
eq(16). However, for early rounds i.e. small t values, ⌘L,t 
can satisfy eq(17), for the beneﬁt of accelerated learning and convergence progress early on. Similar
strategy can be used for number of local steps to minimize local computations towards later rounds.
In any case, these strategies are mere guidelines meant to remain within the worst case guarantee.
However, Theorem 3 offers the ﬂexibility otherwise not available. We will now show the empirical
performance gained by taking advantage of this ﬂexibility.

1

Figure 1: Test Accuracy Performance with various values of initial client learning rate (LR_0), initial
batch size (BatchSize_0), and number of clients per round (NumClients). Top row: FSO sims. Bottom
row: FEMNIST sims. Baseline values for FEMNIST: LR_0=0.1, BatchSize_0=20, NumClients=10.
Baseline values for FSO: LR_0=0.32, BatchSize_0=16, NumClients=50.

8

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

5 Empirical Evaluation and Numerical Results

We present an empirical evaluation of FATHOM proposed in Section 3 and outlined in Algorithm
1. We conduct extensive simulations of federated learning in character recognition on the federated
EMNIST-62 dataset (FEMNIST) (Cohen et al. [2017]) with a CNN, and in natural language next-word
prediction on the federated Stack Overﬂow dataset (FSO) (TensorFlow-Federated-Authors [2019])
with a RNN. We defer most of the details of the experiment setup in Appendix Section C.1. Our
choice of datasets, tasks and models, are exactly the same as the "EMNIST CR" task and the "SO
NWP" task from Reddi et al. [2020]. See Figure 1 and Table 1 and their captions for details of the
experiment results. Our evaluation lacks comparison with a few one-shot FL HPO methods discussed
earlier in the paper because of a lack of standardized benchmark (until FedHPO- B Wang et al. [2022]
was published concurrently as this work) to be fair and comprehensive.

The underlying principle behind these experiments is evaluating the robustness of FATHOM versus
FedAvg under various initial settings, to mirror realistic usage scenarios where the optimal hyperpa-
rameter values are unknown. For FATHOM, we start with the same initial hyperparameter values
as FedAvg. The test accuracy progress with respect to communication rounds is shown in Figure 1
from these experiments. We also pick test accuracy targets for the two tasks. For FEMNIST CR we
use 86% and for FSO NWP we use 23%. Table 1 shows a table of resource utilization metrics with
respect to reaching these targets in our experiments, highlighting the communication efﬁciency as
well as reduction in local computation from FATHOM in comparison to FedAvg. To our knowledge,
we are the ﬁrst to show gain from an online HPO procedure over a well-tuned equivalent procedure
with ﬁxed hyperparameter values.

The federated learning simulation framework on which we build our algorithms for our experiments
is FedJAX (Ro et al. [2021]) which is under the Apache License. The server that runs the experiments
is equipped with Nvidia Tesla V100 SXM2 GPUs.

Table 1: Resource utilization in communication and local computation to reach speciﬁed test
accuracy target for each task. All evalutions are run for ten trials. Bold numbers highlight better
performance. NA means target was not reached within 1500 rounds for FSO NWP and 2000 rounds
for FEMNIST CR, in any of our trials. LR_0 is initial client learning rate, BS_0 is initial batch size,
and NCPR is number of clients per round. All experiments use baseline initial values except where
indicated. For clariﬁcation, M is used in place for "million", and K for "thousand".
Baseline_fso : (LR_0 = 0.32, BS_0 = 16, NCPR = 50)
Baseline_femnist : (LR_0 = 0.10, BS_0 = 20, NCPR = 10)

Tasks

Experiments

FSO NWP
Target@23%

FEMNIST CR
Target@86%

Baseline_fso
LR_0 = 0.05
BS_0 = 4
BS_0 = 256
NCPR = 25
NCPR = 200
Baseline_femnist
LR_0 = 0.05
BS_0 = 4
BS_0 = 256
NCPR = 100
NCPR = 200

18

580

±
NA

±
NA

FedAvg
11
971

Number of Rounds To
Reach Target Test Accuracy
FATHOM
12
562
7
871
43
758
28
801
49
970
17
396
24
739
21
905
17
708
20
736
16
777
16
790

1283
684
1098
1574
885

±
±
±
±
±
NA

33
26
15
19
41

±
±
±
±
±
±
±
±
±
±
±
±

1436
1481

18
33

±
±

Local Gradients Calculated To
Reach Target Test Accuracy
FedAvg
FATHOM

85M
138M
93M
174M
63M
280M
1.5M
1.7M
1.2M
2.0M
22M
57M

1.2M
±
3.2M
±
2.8M
±
18M
±
2.7M
±
45M
±
36K
±
28K
±
28K
±
44K
±
0.27M
1.0M

±
±

124M

±
NA

1.3M

74M

2.5M

±
NA

82M
350M
2.2M
3.1M
1.7M

±
±
±
±
±
NA

3.8M
13M
64K
28K
88K

28M
59M

±
±

0.39K
1.3M

6 Conclusion and Future Work

In this work, we propose FATHOM for adaptive hyperparameters in federated optimization, speciﬁ-
cally for FedAvg. We analyze theoretically and evaluate empirically its potential beneﬁts in conver-
gence behavior as measured in test accuracy, and in reduction of local computations, by automatically
adapting the three main hyperparameters of FedAvg: client learning rate, and number of local steps
via epochs and batch size. An example of future efforts to extend this work is using a standardized
benchmark such as Wang et al. [2022] for performance comparison against other FL HPO methods.

9

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

369

370

References

E. Amid, R. Anil, C. Fifty, and M. K. Warmuth. Step-size adaptation using exponentiated gradient updates,

2022.

A. G. Baydin, R. Cornish, D. M. Rubio, M. Schmidt, and F. Wood. Online learning rate adaptation with

hypergradient descent, 2017.

H. Cai, L. Zhu, and S. Han. ProxylessNAS: Direct neural architecture search on target task and hardware.
In International Conference on Learning Representations, 2019. URL https://arxiv.org/pdf/1812.
00332.pdf.

Z. Charles and J. Koneˇcný. On the outsized importance of learning rates in local update methods, 2020.

G. Cohen, S. Afshar, J. Tapson, and A. Van Schaik. Emnist: Extending mnist to handwritten letters. In 2017

international joint conference on neural networks (IJCNN), pages 2921–2926. IEEE, 2017.

Z. Dai, K. H. Low, and P. Jaillet. Federated bayesian optimization via thompson sampling, 2020.

Z. Dai, B. K. H. Low, and P. Jaillet. Differentially private federated bayesian optimization with distributed

exploration, 2021.

U. Ghai, E. Hazan, and Y. Singer. Exponentiated gradient meets gradient descent, 2019.

E. Gorbunov, F. Hanzely, and P. Richtárik. Local sgd: Uniﬁed theory and new efﬁcient methods, 2020.

P. Guo, D. Yang, A. Hatamizadeh, A. Xu, Z. Xu, W. Li, C. Zhao, D. Xu, S. Harmon, E. Turkbey, et al. Auto-fedrl:
Federated hyperparameter optimization for multi-institutional medical image segmentation. arXiv preprint
arXiv:2203.06338, 2022.

G. Hardy, J. Littlewood, and G. Pólya. Inequalities. Cambridge Mathematical Library. Cambridge University
Press, 1988. ISBN 9781107647398. URL https://books.google.com/books?id=EfvZAQAAQBAJ.

S. Holly, T. Hiessl, S. R. Lakani, D. Schall, C. Heitzinger, and J. Kemnitz. Evaluation of hyperparameter-

optimization approaches in an industrial federated learning system, 2021.

M. Khodak, M.-F. Balcan, and A. Talwalkar. Adaptive gradient-based meta-learning methods, 2019.

M. Khodak, R. Tu, T. Li, L. Li, N. Balcan, V. Smith, and A. Talwalkar. Federated hyperparameter tuning:
Challenges, baselines, and connections to weight-sharing. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. W.
Vaughan, editors, Advances in Neural Information Processing Systems, 2021. URL https://openreview.
net/forum?id=p99rWde9fVJ.

D. P. Kingma and J. Ba. Adam: A method for stochastic optimization, 2014.

D. P. Kingma and M. Welling. Auto-encoding variational bayes, 2013.

L. Li, M. Khodak, M.-F. Balcan, and A. Talwalkar. Geometry-aware gradient algorithms for neural architecture

search, 2020.

X. Li, K. Huang, W. Yang, S. Wang, and Z. Zhang. On the convergence of fedavg on non-iid data, 2019.

Y.-H. Li and V. Cevher. Convergence of the exponentiated gradient method with armijo line search. Journal
of Optimization Theory and Applications, 181(2):588–607, Dec 2018. ISSN 1573-2878. doi: 10.1007/
s10957-018-1428-9. URL http://dx.doi.org/10.1007/s10957-018-1428-9.

H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas. Communication-efﬁcient learning of

deep networks from decentralized data, 2016.

A. Mokhtari, S. Shahrampour, A. Jadbabaie, and A. Ribeiro. Online optimization in dynamic environments:
Improved regret rates for strongly convex problems. 2016 IEEE 55th Conference on Decision and Control
(CDC), Dec 2016. doi: 10.1109/cdc.2016.7799379. URL http://dx.doi.org/10.1109/cdc.2016.
7799379.

H. Mostafa. Robust federated learning through representation matching and adaptive hyper-parameters, 2019.

K. Murota. Discrete convex analysis. Mathematical Programming, 83:313–371, 1998.

A. V. Oppenheim and R. W. Schafer. Discrete-Time Signal Processing. Prentice Hall Press, USA, 3rd edition,

2009. ISBN 0131988425.

10

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

413

414

415

416

417

418

H. Pham, M. Guan, B. Zoph, Q. Le, and J. Dean. Efﬁcient neural architecture search via parameters sharing.
In J. Dy and A. Krause, editors, Proceedings of the 35th International Conference on Machine Learning,
volume 80 of Proceedings of Machine Learning Research, pages 4095–4104. PMLR, 10–15 Jul 2018. URL
https://proceedings.mlr.press/v80/pham18a.html.

S. Reddi, Z. Charles, M. Zaheer, Z. Garrett, K. Rush, J. Koneˇcný, S. Kumar, and H. B. McMahan. Adaptive

federated optimization, 2020.

J. H. Ro, A. T. Suresh, and K. Wu. Fedjax: Federated learning simulation with jax. arXiv preprint

arXiv:2108.02117, 2021.

TensorFlow-Federated-Authors. Tensorﬂow federated stack overﬂow dataset, 2019. URL https://www.

tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow.

J. Wang and G. Joshi. Adaptive communication strategies to achieve the best error-runtime trade-off in local-

update sgd, 2018.

J. Wang, Z. Charles, Z. Xu, G. Joshi, H. B. McMahan, B. A. y Arcas, M. Al-Shedivat, G. Andrew, S. Avestimehr,
K. Daly, D. Data, S. Diggavi, H. Eichner, A. Gadhikar, Z. Garrett, A. M. Girgis, F. Hanzely, A. Hard, C. He,
S. Horvath, Z. Huo, A. Ingerman, M. Jaggi, T. Javidi, P. Kairouz, S. Kale, S. P. Karimireddy, J. Konecny,
S. Koyejo, T. Li, L. Liu, M. Mohri, H. Qi, S. J. Reddi, P. Richtarik, K. Singhal, V. Smith, M. Soltanolkotabi,
W. Song, A. T. Suresh, S. U. Stich, A. Talwalkar, H. Wang, B. Woodworth, S. Wu, F. X. Yu, H. Yuan,
M. Zaheer, M. Zhang, T. Zhang, C. Zheng, C. Zhu, and W. Zhu. A ﬁeld guide to federated optimization, 2021.

Z. Wang, W. Kuang, C. Zhang, B. Ding, and Y. Li. Fedhpo-b: A benchmark suite for federated hyperparameter

optimization, 2022.

R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning.
Mach. Learn., 8(3–4):229–256, may 1992. ISSN 0885-6125. doi: 10.1007/BF00992696. URL https:
//doi.org/10.1007/BF00992696.

H. Yang, M. Fang, and J. Liu. Achieving linear speedup with partial worker participation in non-iid federated

learning, 2021.

Y. Zhou, P. Ram, T. Salonidis, N. Baracaldo, H. Samulowitz, and H. Ludwig. Single-shot hyper-parameter

optimization for federated learning: A general algorithm and analysis, 2022.

M. Zinkevich. Online convex programming and generalized inﬁnitesimal gradient ascent. In Proceedings of
the Twentieth International Conference on International Conference on Machine Learning, ICML’03, page
928–935. AAAI Press, 2003. ISBN 1577351894.

Checklist

The checklist follows the references. Please read the checklist guidelines carefully for information on
how to answer these questions. For each question, change the default [TODO] to [Yes] , [No] , or
[N/A] . You are strongly encouraged to include a justiﬁcation to your answer, either by referencing
the appropriate section of your paper or providing a brief inline description. For example:

• Did you include the license to the code and datasets? [Yes] The code is MIT licensed.

Please do not modify the questions and only use the provided macros for your answers. Note that the
Checklist section does not count towards the page limit. In your paper, please delete this instructions
block and only keep the Checklist section heading above along with the questions/answers below.

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reﬂect the paper’s

contributions and scope? [Yes]

(b) Did you describe the limitations of your work? [Yes] Please refer to Sections 2 and 6
(c) Did you discuss any potential negative societal impacts of your work? [No] Not
speciﬁcally, but it is alluded to how FL applications have social implications in the
introductory section.

(d) Have you read the ethics review guidelines and ensured that your paper conforms to

them? [Yes]

11

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

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [Yes] Please refer to

Assumptions 1, 2, 3, 4, and 5

(b) Did you include complete proofs of all theoretical results? [Yes] Yes, in the supple-

mental material.

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main exper-
imental results (either in the supplemental material or as a URL)? [Yes] Yes, in the
supplemental material.

(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they

were chosen)? [Yes] Yes, in the supplemental material.

(c) Did you report error bars (e.g., with respect to the random seed after running experi-

ments multiple times)? [Yes] Yes, see Table 1

(d) Did you include the total amount of compute and the type of resources used (e.g., type
of GPUs, internal cluster, or cloud provider)? [Yes] Yes, it is mentioned in Section 5

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...
(a) If your work uses existing assets, did you cite the creators? [Yes] Yes, it is mentioned

in Section 5

(b) Did you mention the license of the assets? [Yes] Yes, it is mentioned in Section 5
(c) Did you include any new assets either in the supplemental material or as a URL? [Yes]

Yes, in the supplemental material.

(d) Did you discuss whether and how consent was obtained from people whose data you’re

using/curating? [N/A]

(e) Did you discuss whether the data you are using/curating contains personally identiﬁable

information or offensive content? [No]

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if

applicable? [N/A]

(b) Did you describe any potential participant risks, with links to Institutional Review

Board (IRB) approvals, if applicable? [N/A]

(c) Did you include the estimated hourly wage paid to participants and the total amount

spent on participant compensation? [N/A]

12

