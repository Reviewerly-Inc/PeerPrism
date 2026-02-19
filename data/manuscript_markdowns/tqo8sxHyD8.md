Finite-Time Error Bounds for Distributed Linear
Stochastic Approximation

Abstract

This paper considers a novel multi-agent linear stochastic approximation algorithm
driven by Markovian noise and general consensus-type interaction, in which each
agent evolves according to its local stochastic approximation process which depends
on the information from its neighbors. The interconnection structure among the
agents is described by a time-varying directed graph. While the convergence of
consensus-based stochastic approximation algorithms when the interconnection
among the agents is described by doubly stochastic matrices (at least in expectation)
has been studied, less is known about the case when the interconnection matrix is
simply stochastic. For any uniformly strongly connected graph sequences whose
associated interaction matrices are stochastic, the paper derives ﬁnite-time bounds
on the mean-square error, deﬁned as the deviation of the output of the algorithm
from the unique equilibrium point of the associated ordinary differential equation.
For the case of interconnection matrices being stochastic, the equilibrium point
can be any unspeciﬁed convex combination of the local equilibria of all the agents
in the absence of communication. Both the cases with constant and time-varying
step-sizes are considered. In the case when the convex combination is required
to be a straight average and interaction between any pair of neighboring agents
may be uni-directional, so that doubly stochastic matrices cannot be implemented
in a distributed manner, the paper proposes a push-type distributed stochastic
approximation algorithm and provides its ﬁnite-time bounds for the performance by
leveraging the analysis for the consensus-type algorithm with stochastic matrices.

1

Introduction

The use of reinforcement learning (RL) to obtain policies that describe solutions to a Markov decision
process (MDP) in which an autonomous agent interacting with an unknown environment aims to
optimize its long term reward is now standard [1]. Multi-agent or distributed reinforcement learning
is useful when a team of agents interacts with an unknown environment or system and aims to
collaboratively accomplish tasks involving distributed decision-making. Distributed here implies that
agents exchange information only with their neighbors according to a certain communication graph.
Recently, many distributed algorithms for multi-agent RL have been proposed and analyzed [2].
The basic result in such works is of the type that if the graph describing the communication among
the agents is bi-directional (and hence can be represented by a doubly stochastic matrix), then an
algorithm that builds on traditional consensus algorithms converges to a solution in terms of policies
to be followed by the agents that optimize the sum of the utility functions of all the agents; further,
both ﬁnite and inﬁnite time performance of such algorithms can be characterized [3, 4].

This paper aims to relax the assumption of requiring bi-directional communication among agents
in a distributed RL algorithm. This assumption is arguably restrictive and will be violated due to
reasons such as packet drops or delays, differing privacy constraints among the agents, heterogeneous
capabilities among the agents in which some agents may be able to communicate more often or with
more power than others, adversarial attacks, or even sophisticated resilient consensus algorithms
being used to construct the distributed RL algorithm. A uni-directional communication graph can
be represented through a (possibly time-varying) stochastic – which may not be doubly stochastic –
matrix being used in the algorithm. As we discuss in more detail below, relaxing the assumption of a

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

doubly stochastic matrix to simply a stochastic matrix in the multi-agent and distributed RL algorithms
that have been proposed in the literature, however, complicates the proofs of their convergence and
ﬁnite time performance characterizations. The main result in this paper is to provide a ﬁnite time
bound on the mean square error for a multi-agent linear stochastic approximation algorithm in which
the agents interact over a time-varying directed graph characterized by a stochastic matrix. This paper,
thus, extends the applicability of distributed and multi-agent RL algorithms presented in the literature
to situations such as those mentioned above where bidirectional communication at every time step
cannot be guaranteed. As we shall see, this extension is technically challenging and requires new
proof techniques that may be of independent interest.

Related Work A key tool used for designing and analyzing RL algorithms is stochastic approxima-
tion [5], e.g., for policy evaluation, including temporal difference (TD) learning as a special case [6].
Convergence study of stochastic approximation based on ordinary differential equation (ODE) meth-
ods has a long history [7]. Notable examples are [8, 9] which prove asymptotic convergence of TD(λ).
Recently, ﬁnite-time performance of single-agent stochastic approximation and TD algorithms has
been studied in [10–18]; many other works have now appeared that perform ﬁnite-time analysis for
other RL algorithms, see, e.g., [19–28], just to name a few.

Many distributed and multi-agent reinforcement learning algorithms have now been proposed in
the literature. In this setting, each agent can receive information only from its neighbors, and no
single agent can solve the problem alone or by ‘taking the lead’. A backbone of almost all distributed
RL algorithms proposed in the literature is the consensus-type interaction among the agents, dating
back at least to [29]. Many works have analyzed asymptotic convergence of such RL algorithms
using ODE methods [4, 30–32]. This can be viewed as an application of ideas from distributed
stochastic approximation [33–38]. Finite-time performance guarantees for distributed RL have also
(cid:3)
been provided in works, most notably in [3, 39–43].

The assumption that is the central concern of this paper and is made in all the existing ﬁnite-time
analyses for distributed RL algorithms is that the consensus interaction is characterized by doubly
stochastic matrices [3, 39–43] at every time step, or at least in expectation, i.e., W 1 = 1 and
1(cid:62)E(W ) = 1(cid:62) [37]. Intuitively, doubly stochastic matrices imply symmetry in the communication
graph, which almost always requires bidirectional communication graphs. More formally, the
assumption of doubly stochastic matrices is restrictive since distributed construction of a doubly
stochastic matrix needs to either invoke algorithms such as the Metropolis algorithm [44] which
requires bi-directional communication of each agent’s degree information; or to utilize an additional
distributed algorithm [45] which signiﬁcantly increases the complexity of the whole algorithm
design. Doubly stochastic matrices in expectation can be guaranteed via so-called broadcast gossip
algorithms which still requires bi-directional communication for convergence [37]. In a realistic
network, especially with mobile agents such as autonomous vehicles, drones, or robots, uni-directional
communication is inevitable due to various reasons such as asymmetric communication and privacy
constraints, non-zero communication failure probability between any two agents at any given time,
and application of resilient consensus in the presence of adversary attacks [46, 47], all leading to
an interaction among the agents characterized by a stochastic matrix, which may further be time-
varying. The problem of design of distributed RL algorithms with time-varying stochastic matrices
and characterizing either their asymptotic convergence or ﬁnite time analysis remains open.

As a step towards solving this problem, we propose a novel distributed stochastic approximation
algorithm and provide its convergence analyses when a time-dependent stochastic matrix is being
used due to uni-directional communication in a dynamic network. One of the ﬁrst guarantees to be
lost as the assumption of doubly stochastic matrices is removed is that the algorithm converges to a
“policy” that maximizes the sum of reward functions of all the agents. Instead, the convergence is to a
set of policies that optimize a convex combination of the network-wise accumulative reward, with
the exact combination depending on the limit product of the inﬁnite sequence of stochastic matrices.
Nonetheless, by deﬁning the error as the deviation of the output of the algorithm from the eventual
equilibrium point, we derive ﬁnite-time bounds on the mean squared error. We consider both the
cases with constant and time-varying step sizes. In the important special case where the goal is to
optimize the average of the individual accumulative rewards of all the agents, we provide a distributed
stochastic approximation algorithm, which builds on the push-sum idea [48] that has been used to
solve distributed averaging problem over strongly connected graphs, and characterize its ﬁnite-time
performance. Thus, this paper provides the ﬁrst distributed algorithm that can be applied (e.g., in
TD learning) to converge to the policy maximizing the team objective of the sum of the individual

2

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

utility functions over time-varying, uni-directional, communication graphs, and characterizes the
ﬁnite-time bounds on the mean squared error of the algorithm output from the equilibrium point
under appropriate assumptions.

Technical Innovation and Contributions There are two main technical challenges in removing
the assumption of doubly stochastic matrices being used in the analysis of distributed stochastic
approximation algorithms. The ﬁrst is in the direction of ﬁnite-time analysis. For distributed RL
algorithms, ﬁnite-time performance analysis essentially boils down to two parts, namely bounding
the consensus error and bounding the “single-agent” mean-square error. For the case when consensus
interaction matrices are all doubly stochastic, the consensus error bound can be derived by analyzing
the square of the 2-norm of the deviation of the current state of each agent from the average of the
states of the agents. With consensus in the presence of doubly stochastic matrices, the average of the
states of the agents remains invariant. Thus, it is possible to treat the average value as the state of a
ﬁctitious agent to derive the mean-square consensus error bound with respect to the limiting point.
More formally, this process relies on two properties of a double stochastic matrix W , namely that
(1) 1(cid:62)W = 1(cid:62), and (2) if xt+1 = W xt, then (cid:107)xt+1 − (1(cid:62)xt+1)1(cid:107)2 ≤ σ2(W )(cid:107)xt − (1(cid:62)xt)1(cid:107)2
where σ2(W ) denotes the second largest singular value of W (which is strictly less than one if W is
irreducible). Even if the doubly stochastic matrix is time-varying (denoted by Wt), property (1) still
holds and property (2) can be generalized as in [49]. Thus, the square of the 2-norm (cid:107)xt − (1(cid:62)xt)1(cid:107)2
2
is a quadratic Lyapunov function for the average consensus processes. Doubly stochastic matrices in
expectation can be treated in the same way by looking at the expectation. This is the core on which
all the existing ﬁnite-time analyses of distributed RL algorithms are based.

However, if each consensus interaction matrix is stochastic, and not necessarily doubly stochastic, the
above two properties may not hold. In fact, it is well known that quadratic Lyapunov functions for
general consensus processes xt+1 = Stxt, with St being stochastic, do not exist [50]. This breaks
down all the existing analyses and provides the ﬁrst technical challenge that we tackle in this paper.
Speciﬁcally, we appeal to the idea of quadratic comparison functions for general consensus processes.
This was ﬁrst proposed in [51] and makes use of the concept of “absolute probability sequences”. We
provide a general analysis methodology and results that subsume the existing ﬁnite-time analyses for
single-timescale distributed linear stochastic approximation and TD learning as special cases.

The second technical challenge arises from the fact that with stochastic matrices, the distributed RL
algorithms may not converge to the policies that maximize the average of the utility functions of the
agents. To regain this property, we propose a new algorithm that utilizes a push-sum protocol for
consensus. However, ﬁnite-time analysis for such a push-based distributed algorithm is challenging.
Almost all, if not all, the existing push-based distributed optimization works build on the analysis
in [52]; however, that analysis assumes that a convex combination of the entire history of the states
of each agent (and not merely the current state of the agent) is being calculated. This assumption
no longer holds in our case. To obtain a direct ﬁnite-time error bound without this assumption, we
propose a new approach to analyze our push-based distributed algorithm by leveraging our consensus-
based analyses to establish direct ﬁnite-time error bounds for stochastic approximation. Speciﬁcally,
we tailor an “absolute probability sequence” for the push-based stochastic approximation algorithm
and exploit its properties. Such properties have never been found in the existing literature and may be
of independent interest for analyzing any push-sum based distributed algorithm.

We now list the main contributions of our work. We propose a novel consensus-based distributed
linear stochastic approximation algorithm driven by Markovian noise in which each agent evolves
according to its local stochastic approximation process and the information from its neighbors. We
assume only a (possibly time-varying) stochastic matrix being used during the consensus phase,
which is a more practical assumption when only unidirectional communication is possible among
agents. We establish both convergence guarantees and ﬁnite-time bounds on the mean-square error,
deﬁned as the deviation of the output of the algorithm from the unique equilibrium point of the
associated ordinary differential equation. The equilibrium point can be an “uncontrollable” convex
combination of the local equilibria of all the agents in the absence of communication. We consider
both the cases of constant and time-varying step-sizes. Our results subsume the existing results on
convergence and ﬁnite-time analysis of distributed RL algorithms that assume doubly stochastic
matrices and bi-directional communication as special cases. In the case when the convex combination
is required to be a straight average and interaction between any pair of neighboring agents may be
uni-directional, we propose a push-type distributed stochastic approximation algorithm and establish
its ﬁnite-time performance bound. It is worth emphasizing that it is straightforward to extend our

3

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

algorithm from the straight average point to any pre-speciﬁed convex combination. Since it is well
known that TD algorithms can be viewed as a special case of linear stochastic approximation [8], our
distributed linear stochastic approximation algorithms and their ﬁnite-time bounds can be applied to
TD algorithms in a straight-forward manner.

Notation We use Xt to represent that a variable X is time-dependent and t ∈ {0, 1, 2, . . .} is
the discrete time index. The ith entry of a vector x will be denoted by xi and, also, by (x)i when
convenient. The ijth entry of a matrix A will be denoted by aij and, also, by (A)ij when convenient.
We use 1n to denote the vectors in IRn whose entries all equal to 1’s, and I to denote the identity
matrix, whose dimension is to be understood from the context. Given a set S with ﬁnitely many
elements, we use |S| to denote the cardinality of S. We use (cid:100)·(cid:101) to denote the ceiling function.

A vector is called a stochastic vector if its entries are nonnegative and sum to one. A square
nonnegative matrix is called a row stochastic matrix, or simply stochastic matrix, if its row sums all
equal one. Similarly, a square nonnegative matrix is called a column stochastic matrix if its column
sums all equal one. A square nonnegative matrix is called a doubly stochastic matrix if its row sums
and column sums all equal one. The graph of an n × n matrix is a direct graph with n vertices and a
directed edge from vertex i to vertex j whenever the ji-th entry of the matrix is nonzero. A directed
graph is strongly connected if it has a directed path from any vertex to any other vertex. For a strongly
connected graph G, the distance from vertex i to another vertex j is the length of the shortest directed
path from i to j; the longest distance among all ordered pairs of distinct vertices i and j in G is
called the diameter of G. The union of two directed graphs, Gp and Gq, with the same vertex set,
written Gp ∪ Gq, is meant the directed graph with the same vertex set and edge set being the union of
the edge set of Gp and Gq. Since this union is a commutative and associative binary operation, the
deﬁnition extends unambiguously to any ﬁnite sequence of directed graphs with the same vertex set.

2 Distributed Linear Stochastic Approximation

Consider a network consisting of N agents. For the purpose of presentation, we label the agents
from 1 through N . The agents are not aware of such a global labeling, but can differentiate between
their neighbors. The neighbor relations among the N agents are characterized by a time-dependent
directed graph Gt = (V, Et) whose vertices correspond to agents and whose directed edges (or arcs)
depict neighbor relations, where V = {1, . . . , N } is the vertex set and Et = V × V is the edge set
at time t. Speciﬁcally, agent j is an in-neighbor of agent i at time t if (j, i) ∈ Et, and similarly,
agent k is an out-neighbor of agent i at time t if (i, k) ∈ Et. Each agent can send information to its
out-neighbors and receive information from its in-neighbors. Thus, the directions of edges represent
the directions of information ﬂow. For convenience, we assume that each agent is always an in- and
out-neighbor of itself, which implies that Gt has self-arcs at all vertices for all time t. We use N i
t and
N i−
t

to denote the in- and out-neighbor set of agent i at time t, respectively, i.e.,

t = {j ∈ V : (j, i) ∈ Et}, N i−

t = {k ∈ V : (i, k) ∈ Et}.

It is clear that N i

are nonempty as they both contain index i.

N i
t and N i−

t

We propose the following distributed linear stochastic approximation over a time-varying neighbor
graph sequence {Gt}. Each agent i has control over a random vector θi

t which is updated by

θi
t+1 =

(cid:88)

j∈N i
t

wij

t θj

t + αt

(cid:18)

A(Xt)

(cid:88)

j∈N i
t

wij

t θj

t + bi(Xt)

(cid:19)

,

i ∈ V,

t ∈ {0, 1, 2, . . .},

(1)

where wij
t are consensus weights, αt is the step-size at time t, A(Xt) is a random matrix and bi(Xt)
is a random vector, both generated based on the Markov chain {Xt} with state spaces X . It is worth
noting that the update of each agent only uses its in-neighbors’ information and thus is distributed.

Remark 1 The work of [33] considers a different consensus-based networked linear stochastic
approximation as follows:

θi
t+1 =

(cid:88)

j∈N i
t

wij

t θj

t + αt

(cid:0)A(Xt)θi

t + bi(Xt)(cid:1) ,

i ∈ V,

t ∈ {0, 1, 2, . . .},

(2)

200

201

whose state form is Θt+1 = WtΘt +αtΘtA(Xt)(cid:62) +αtB(Xt), and mainly focuses on asymptotically
weakly convergence for the ﬁxed step-size case (i.e., αt = α for all t). Under the similar set of

4

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

conditions, with its condition (C3.4’) being a stochastic analogy for Assumption 6, Theorem 3.1
in [33] shows that (2) has a limit which can be veriﬁed to be the same as θ∗, the limit of (1). How to
apply the ﬁnite-time analysis tools in this paper to (2) has so far eluded us. The two updates (1) and
(2) are analogous to the “combine-then-adapt” and “adapt-then-combine” diffusion strategies in
(cid:3)
distributed optimization [53].

We impose the following assumption on the weights wij
literature [54–56].

t which has been widely adopted in consensus

Assumption 1 There exists a constant β > 0 such that for all i, j ∈ V and t, wij
j ∈ N i
t = 1.

t . For all i ∈ V and t, (cid:80)

wij

j∈N i
t

t ≥ β whenever

Let Wt be the N × N matrix whose ijth entry equals wij
t and zero otherwise. From
t
Assumption 1, each Wt is a stochastic matrix that is compliant with the neighbor graph Gt. Since
each agent i is always assumed to be an in-neighbor of itself, all diagonal entries of Wt are positive.
Thus, if Gt is strongly connected, Wt is irreducible and aperiodic. To proceed, deﬁne


if j ∈ N i







Θt =

t )(cid:62)
(θ1
...
t )(cid:62)
(θN
Then, the N linear stochastic recursions in (1) can be combined and written as
Θt+1 = WtΘt + αtWtΘtA(Xt)(cid:62) + αtB(Xt),

(b1(Xt))(cid:62)
...
(bN (Xt))(cid:62)


 , B(Xt) =


 .







t ∈ {0, 1, 2, . . .}.

(3)

The goal of this section is to characterize the ﬁnite-time performance of (1), or equivalently (3), with
the following standard assumptions, which were adopted e.g. in [3, 13].

Assumption 2 There exists a matrix A and vectors bi, i ∈ V, such that

lim
t→∞

lim
t→∞
Deﬁne bmax = maxi∈V supx∈X (cid:107)bi(x)(cid:107)2 < ∞ and Amax = supx∈X (cid:107)A(x)(cid:107)2 < ∞. Then, (cid:107)A(cid:107)2 ≤
Amax and (cid:107)bi(cid:107)2 ≤ bmax, i ∈ V.

E[A(Xt)] = A,

i ∈ V.

E[bi(Xt)] = bi,

Assumption 3 Given a positive constant α, we use τ (α) to denote the mixing time of the Markov
chain {Xt} for which






(cid:107)E[A(Xt) − A|X0 = X](cid:107)2 ≤ α, ∀X, ∀t ≥ τ (α),

(cid:107)E[bi(Xt) − bi|X0 = X](cid:107)2 ≤ α, ∀X, ∀t ≥ τ (α), ∀i ∈ V.

The Markov chain {Xt} mixes at a geometric rate, i.e., there exists a constant C such that τ (α) ≤
−C log α.

Assumption 4 All eigenvalues of A have strictly negative real parts, i.e., A is a Hurwitz matrix.
Then, there exists a symmetric positive deﬁnite matrix P , such that A(cid:62)P + P A = −I. Let γmax and
γmin be the maximum and minimum eigenvalues of P , respectively.
Assumption 5 The step-size sequence {αt} is positive, non-increasing, and satisﬁes (cid:80)∞
and (cid:80)∞

t=0 αt = ∞

t=0 α2

t < ∞.

230

To state our ﬁrst main result, we need the following concepts.

231

232

233

234

235

Deﬁnition 1 A graph sequence {Gt} is uniformly strongly connected if there exists a positive integer
Gk is strongly connected. If such an integer exists,
L such that for any t ≥ 0, the union graph ∪t+L−1
k=t
we sometimes say that {Gt} is uniformly strongly connected by sub-sequences of length L.

Remark 2 Two popular joint connectivity deﬁnitions in consensus literature are “B-connected” [57]
and “repeatedly jointly strongly connected” [58]. A graph sequence {Gt} is B-connected if there

5

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

t=kB

exists a positive integer B such that the union graph ∪(k+1)B−1
Gt is strongly connected for each
integer k ≥ 0. Although the uniformly strongly connectedness looks more restrictive compared
with B-connectedness at ﬁrst glance, they are in fact equivalent. To see this, ﬁrst it is easy to see
that if {Gt} is uniformly strongly connected, {Gt} must be B-connected; now supposing {Gt} is
Gk must be strongly connected, and thus {Gt} is
B-connected, for any ﬁx t, the union graph ∪t+2B−1
uniformly strongly connected by sub-sequences of length 2B. Thus, the two deﬁnitions are equivalent.
It is also not hard to show that the uniformly strongly connectedness is equivalent to “repeatedly
jointly strongly connectedness” provided the graphs under consideration all have self-arcs at all
vertices, as “repeatedly jointly strongly connectedness” is deﬁned upon “graph composition”. (cid:3)

k=t

Deﬁnition 2 Let {Wt} be a sequence of stochastic matrices. A sequence of stochastic vectors {πt}
is an absolute probability sequence for {Wt} if π(cid:62)

t+1Wt for all t ≥ 0.

t = π(cid:62)

This deﬁnition was ﬁrst introduced by Kolmogorov [59]. It was shown by Blackwell [60] that every
sequence of stochastic matrices has an absolute probability sequence. In general, a sequence of
stochastic matrices may have more than one absolute probability sequence; when the sequence of
stochastic matrices is “ergodic”, it has a unique absolute probability sequence [56]. It is easy to see
that when Wt is a ﬁxed irreducible stochastic matrix W , πt is simply the normalized left eigenvector
of W for eigenvalue one. More can be said.

Lemma 1 Suppose that Assumption 1 holds. If {Gt} is uniformly strongly connected, then there
exists a unique absolute probability sequence {πt} for the matrix sequence {Wt} and a constant
πmin ∈ (0, 1) such that πi

t ≥ πmin for all i and t.

Let (cid:104)θ(cid:105)t = (cid:80)N
that (cid:104)θ(cid:105)t = (π(cid:62)

tθi
i=1 πi
t Θt)(cid:62) = Θ(cid:62)

t, which is a column vector and convex combination of all θi

t. It is easy to see

t πt. From Deﬁnition 2 and (3), we have

t+1Θt+1 = π(cid:62)
π(cid:62)
= π(cid:62)

t+1WtΘt + αtπ(cid:62)
t Θt + αtπ(cid:62)

t+1WtΘtA(Xt)(cid:62) + αtπ(cid:62)
t+1B(Xt),

t ΘtA(Xt)(cid:62) + αtπ(cid:62)

t+1B(Xt)

258

which implies that

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

(cid:104)θ(cid:105)t+1 = (cid:104)θ(cid:105)t + αtA(Xt)(cid:104)θ(cid:105)t + αtB(Xt)(cid:62)πt+1.

(4)

Asymptotic performance of (1) with any uniformly strongly connected neighbor graph sequence is
characterized by the following two theorems.

Theorem 1 Suppose that Assumptions 1, 2 and 5 hold. Let {θi
is uniformly strongly connected, then limt→∞ (cid:107)θi

t − (cid:104)θ(cid:105)t(cid:107)2 = 0 for all i ∈ V.

t}, i ∈ V, be generated by (1). If {Gt}

Theorem 1 only shows that all the sequences {θi
t}, i ∈ V, generated by (1) will ﬁnally reach a
consensus, but not necessarily convergent or bounded. To guarantee the convergence of the sequences,
we further need the following assumption, whose validity is discussed in Remark 3.

Assumption 6 The absolute probability sequence {πt} for the stochastic matrix sequence {Wt} has
a limit, i.e., there exists a stochastic vector π∞ such that limt→∞ πt = π∞.

Theorem 2 Suppose that Assumptions 1–6 hold. Let {θi
unique equilibrium point of the ODE

t}, i ∈ V, be generated by (1) and θ∗ be the

˙θ = Aθ + b,

b =

N
(cid:88)

i=1

∞bi,
πi

(5)

where A and bi are deﬁned in Assumption 2 and π∞ is deﬁned in Assumption 6. If {Gt} is uniformly
strongly connected, then all θi

t will converge to θ∗ both with probability 1 and in mean square.

Remark 3 Though Assumption 6 may look restrictive at ﬁrst glance, simple simulations show that the
sequences {θi
t}, i ∈ V, do not converge if the assumption does not hold. It is worth emphasizing that
the existence of π∞ does not imply the existence of limt→∞ Wt, though the converse is true. Indeed,
the assumption subsumes various cases including (a) all Wt are doubly stochastic matrices, and (b)

6

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

all Wt share the same left eigenvector for eigenvalue 1, which may arise from the scenario when
the number of neighbors of each agent does not change over time [61]. An important implication
of Assumption 6 is when the consensus interaction among the agents, characterized by {Wt}, is
replaced by resilient consensus algorithms such as [46,47] in order to attenuate the effect of unknown
malicious agents, the resulting dynamics of non-malicious agents, in general, will not converge,
because the resulting interaction stochastic matrices among the non-malicious agents depend on
the state values transmitted by the malicious agents, which can be arbitrary, and thus the resulting
stochastic matrix sequence, in general, does not have a convergent absolute probability sequence; of
course, in this case, the trajectories of all the non-malicious agents will still reach a consensus as
(cid:3)
long as the step-size is diminishing, as implied by Theorem 1.

We now study the ﬁnite-time performance of the proposed distributed linear stochastic approximation
(1) for both ﬁxed and time-varying step-size cases. Its ﬁnite-time performance is characterized by the
following theorem.

Let ηt = (cid:107)πt − π∞(cid:107)2 for all t ≥ 0. From Assumption 6, ηt converges to zero as t → ∞.

Theorem 3 Let the sequences {θi
t}, i ∈ V, be generated by (1). Suppose that Assumptions 1–4, 6
hold and {Gt} is uniformly strongly connected by sub-sequences of length L. Let qt and mt be the
unique integer quotient and remainder of t divided by L, respectively. Let δt be the diameter of
∪t+L−1
k=t

Gk, δmax = maxt≥0 δt, and

(cid:18)

(cid:15) =

1 +

2bmax
Amax

−

πminβ2L
2δmax

(cid:19)

(1 + αAmax)2L −

2bmax
Amax

(1 + αAmax)L,

(6)

where 0 < α < min{K1,

log 2
Amaxτ (α) ,

0.1
K2γmax

}.

1) Fixed step-size: Let αt = α for all t ≥ 0. For all t ≥ T1,

N
(cid:88)

i=1

πi
tE

(cid:104)(cid:13)
(cid:13)θi

t − θ∗(cid:13)
2
(cid:13)
2

(cid:105)

≤ 2(cid:15)qt

N
(cid:88)

i=1

πi
mt

E

(cid:104)(cid:13)
(cid:13)θi

mt

− (cid:104)θ(cid:105)mt

(cid:105)

(cid:13)
2
(cid:13)
2

(cid:18)

+ C1

1 −

0.9α
γmax

(cid:19)t−T1

+ C2.

(7)

2) Time-varying step-size: Let αt = α0

t+1 with α0 ≥ γmax

0.9 . For all t ≥ LT2,

N
(cid:88)

i=1

πi
tE

(cid:104)(cid:13)
(cid:13)θi

t − θ∗(cid:13)
2
(cid:13)
2

(cid:105)

≤ 2(cid:15)qt−T2

N
(cid:88)

i=1

πi
LT2+mt

E

(cid:104)(cid:13)
(cid:13)θi

LT2+mt

− (cid:104)θ(cid:105)LT2+mt

(cid:105)

(cid:13)
2
(cid:13)
2

+ C3

(cid:16)

α0(cid:15)

qt−1

2 + α(cid:100) qt−1

2 (cid:101)L

(cid:17)

+

(cid:18)

1
t

C4 log2 (cid:16) t
α0

(cid:17)

+ C5

t
(cid:88)

k=LT2

(cid:19)

ηk + C6

.

(8)

Here T1, T2, K1, K2, C1 − C6 are ﬁnite constants whose deﬁnitions are given in Appendix A.1.

Since πi
bound holds for each individual E[(cid:107)θi
following remark.

t is uniformly bounded below by πmin ∈ (0, 1) from Lemma 1, it is easy to see that the above
2]. To better understand the theorem, we provide the

t − θ∗(cid:107)2

Remark 4 In Appendix B.2.1, we show that both (cid:15) and (1 − 0.9α
) lie in the interval (0, 1). It is easy
γmax
to show that (cid:15) is monotonically increasing for δmax and L, monotonically decreasing for β and πmin.
Therefore, the summands in the ﬁnite-time bound (7) for the ﬁxed step-size case are exponentially
decaying expect for the constant C2, which implies that lim supt→∞
2] ≤ C2,
providing a constant limiting bound. From Appendix A, C2 depends on L, γmin, γmax, Amax, bmax.

t − θ∗(cid:107)2

tE[(cid:107)θi

i=1 πi

(cid:80)N

In Appendix B.2.2, we show that limt→∞
(8) for the time-varying step-size case converges to zero as t → ∞.

k=1 ηk = 0, which implies that the ﬁnite-time bound

(cid:80)t

1
t

We next comment on 0.1 in the inequality deﬁning α. Actually, we can replace 0.1 with any constant
c ∈ (0, 1), which will affect the value of (cid:15) and the feasible set of α, with the latter becoming
}. Thus, the smaller the value of c is, the smaller is the
0 < α < min{K1,
feasible set of α, though the feasible set is always nonempty. For convenience, we simply pick c = 0.1
in this paper; that is why we also have 0.9 in (7).

log 2
Amaxτ (α) ,

c
K2γmax

7

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

Lastly, we comment on α0 in the time-varying step-size case. We set α0 ≥ γmax
getting a cleaner expression of the ﬁnite-time bound. For α0 < γmax
works, but will yield a more complicated expression. The same is true for Theorem 5.

0.9 for the purpose of
0.9 , our analysis approach still
(cid:3)

Technical Challenge and Proof Sketch As described in the introduction, the key challenge of ana-
lyzing the ﬁnite-time performance of the distributed stochastic approximation (1) lies in the condition
that consensus interaction matrix is time-varying and stochastic (not necessarily doubly stochastic).
To tackle this, we appeal to the absolute probability sequence πt of the time-varying interaction matrix
sequence and introduce the quadratic Lyapunov comparison function (cid:80)N
2]. Then,
using the inequality (cid:80)N
i=1 πi
t −θ∗(cid:107)2
2], the next
step is to ﬁnd the ﬁnite-time bounds of (cid:80)N
tE[(cid:107)θi
2], respectively.
The latter term is essentially the “single-agent” mean-square error. Our main analysis contribution
here is to bound the former term for both ﬁxed and time-varying step-size cases.

tE[(cid:107)θi
2]+2E[(cid:107)(cid:104)θ(cid:105)t −θ∗(cid:107)2

2] ≤ 2 (cid:80)N
i=1 πi

2] and E[(cid:107)(cid:104)θ(cid:105)t − θ∗(cid:107)2

tE[(cid:107)θi
t − (cid:104)θ(cid:105)t(cid:107)2

t −(cid:104)θ(cid:105)t(cid:107)2

t − θ∗(cid:107)2

tE[(cid:107)θi

i=1 πi

i=1 πi

3 Push-SA

The preceding section shows that the limiting state of consensus-based distributed stochastic approxi-
mation depends on π∞, which leads to a convex combination of the local equilibria of all the agents in
the absence of communication, but the convex combination is in general “uncontrollable”. Note that
this convex combination will correspond to a convex combination of the network-wise accumulative
rewards in applications such as distributed TD learning. In an important case when the convex
combination is desired to be the straight average, the existing literature e.g. [3, 39] relies on doubly
stochastic matrices whose corresponding π∞ = (1/N )1N . As mentioned in the introduction, doubly
stochastic matrices implicitly require bi-directional communication between any pair of neighboring
agents; see e.g. gossiping [62] and the Metropolis algorithm [44]. A popular method to achieve the
straight average target while allowing uni-directional communication between neighboring agents
is to appeal to the idea so-called “push-sum” [48], which was tailored for solving the distributed
averaging problem over directed graphs and has been applied to distributed optimization [52]. In this
section, we will propose a push-based distributed stochastic approximation algorithm tailored for
uni-directional communication and establish its ﬁnite-time error bound.
Each agent i has control over three variables, namely yi
with initial value 1, ˜θi
t can be arbitrarily initialized, and θi
sends its weighted current values ˆwji
out-neighbors j ∈ N i−

t is scalar-valued
0. At each time t ≥ 0, each agent i
t + αtA(Xt)θt + αtbi(Xt)) to each of its current

t, ˜θi
0 = ˜θi

t, in which yi

t and θi

t and ˆwji
, and updates its variables as follows:


t (˜θi

t yi

t

yi
t+1 =

(cid:88)

ˆwij

t yj
t ,

yi
0 = 1,




j∈N i
t
(cid:88)

˜θi
t+1 =

ˆwij
t

(cid:104)˜θj

t + αt

(cid:16)

j∈N i
t
˜θi
t+1
yi
t+1

,

θi
t+1 =

0 = ˆθi
θi
0,

A(Xt)θj

t + bj(Xt)

(cid:17)(cid:105)

,

(9)

where ˆwij
be aware of the number of its out-neighbors.

t = 1/|N j−

t

|. It is worth noting that the algorithm is distributed yet requires that each agent

Asymptotic performance of (9) with any uniformly strongly connected neighbor graph sequence is
characterized by the following theorem.

Theorem 4 Suppose that Assumptions 2–5 hold. Let {θi
unique equilibrium point of the ODE

t}, i ∈ V, be generated by (9) and θ∗ be the

˙θ = Aθ +

1
N

N
(cid:88)

i=1

bi,

(10)

350

351

where A and bi are deﬁned in Assumption 2. If {Gt} is uniformly strongly connected, then θi
converge to θ∗ in mean square for all i ∈ V.

t will

8

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

(cid:80)N

In this section, we deﬁne (cid:104)˜θ(cid:105)t = 1
i=1 θi
t. To help understand these
N
deﬁnitions, let ˆWt be the N × N matrix whose ij-th entry equals ˆwij
if j ∈ N i
t , otherwise equals
t
zero. It is easy to see that each ˆWt is a column stochastic matrix whose diagonal entries are all
N 1N for all t ≥ 0 can be regarded as an absolute probability sequence of { ˆWt}.
positive. Then, πt = 1
Thus, the above two deﬁnitions are intuitively consistent with (cid:104)θ(cid:105)t in the previous section.

˜θi
t and (cid:104)θ(cid:105)t = 1
N

i=1

(cid:80)N

Finite-time performance of (9) with any uniformly strongly connected neighbor graph sequence is
characterized by the following theorem.
Let µt = (cid:107)A(Xt)((cid:104)θ(cid:105)t − (cid:104)˜θ(cid:105)t)(cid:107)2. In Appendix B.3, we show that (cid:107)(cid:104)θ(cid:105)t − (cid:104)˜θ(cid:105)t(cid:107)2 converges to zero
as t → ∞, so does µt.

Theorem 5 Suppose that Assumptions 2–4 hold and {Gt} is uniformly strongly connected by sub-
sequences of length L. Let {θi
0.9 . Then,
there exists a nonnegative ¯(cid:15) ≤ (1 − 1

t}, i ∈ V, be generated by (9) with αt = α0

L such that for all t ≥ ¯T ,

t+1 and α0 ≥ γmax

N N L ) 1

N
(cid:88)

i=1

(cid:104)(cid:13)
(cid:13)θi

t+1 − θ∗(cid:13)
2
(cid:13)
2

(cid:105)

E

≤ C7¯(cid:15)t + C8

(cid:16)

α0¯(cid:15)

t

2 + α(cid:100) t
2 (cid:101)

(cid:17)

+ C9αt

+

(cid:18)

1
t

C10 log2 (cid:16) t
α0

(cid:17)

+ C11

t
(cid:88)

k= ¯T

(cid:19)

µk + C12

,

(11)

where ¯T and C7 − C12 are ﬁnite constants whose deﬁnitions are given in Appendix A.2.

(cid:80)t

1
t

i=1 E[(cid:107)θi

t+1 − θ∗(cid:107)2

i=1 E[(cid:107)θi
i=1 E[(cid:107)θi

2] and E[(cid:107)(cid:104)˜θ(cid:105)t − θ∗(cid:107)2

t+1 − (cid:104)˜θ(cid:105)t(cid:107)2
t+1 − (cid:104)˜θ(cid:105)t(cid:107)2

In Appendix B.3, we show that limt→∞
k=1 µk = 0, which implies that the ﬁnite-time bound
(11) converges to zero as t → ∞. It is worth mentioning that the theorem does not consider the ﬁxed
step-size case, as our current analysis approach cannot be directly apply for this case.
Proof Sketch and Technical Challenge Using the inequality (cid:80)N
2] ≤
2 (cid:80)N
2] + 2N E[(cid:107)(cid:104)˜θ(cid:105)t − θ∗(cid:107)2
2], our goal is to derive the ﬁnite-time bounds of
(cid:80)N
2], respectively. Although this looks similar to the proof
of Theorem 3, the derivation is quite different. First, the iteration of (cid:104)˜θ(cid:105)t is a single-agent SA plus a
disturbance term (cid:104)θ(cid:105)t − (cid:104)˜θ(cid:105)t, so we cannot directly apply the existing single-agent SA ﬁnite-time
analyses to bound E[(cid:107)(cid:104)˜θ(cid:105)t − θ∗(cid:107)2
2]; instead, we have to show that (cid:104)θ(cid:105)t − (cid:104)˜θ(cid:105)t will diminish and
quantify the diminishing “speed”. Second, both the proof of showing diminishing (cid:104)θ(cid:105)t − (cid:104)˜θ(cid:105)t and
derivation of bounding (cid:80)N
2] involve a key challenge: to prove the sequence
{θi
t} generated from the Push-SA (9) is bounded almost surely. To tackle this, we introduce a
novel way to constructing an absolute probability sequence for the Push-SA as follows. From (9),
t+1 = (cid:80)N
j=1 ˜wij
t = ( ˆwij
θi
t ). We show
that each matrix ˜Wt = [ ˜wij
t ] is stochastic, and there exists a unique absolute probability sequence
{˜πt} for the matrix sequence { ˜Wt} such that ˜πi
t ≥ ˜πmin for all i ∈ V and t ≥ 0, with the con-
stant ˜πmin ∈ (0, 1). Most importantly, we show two critical properties of { ˜Wt} and {˜πt}, namely
limt→∞(Πt
N for all i, j ∈ V and t ≥ 0, which have never been
reported in the existing literature though push-sum based algorithms have been extensively studied.

t + αtA(Xt) θj
t
yj
t

t+1 − (cid:104)˜θ(cid:105)t(cid:107)2

], where ˜wij

N and ˜πi

˜Ws) = 1

t )/((cid:80)N

i=1 E[(cid:107)θi

N 1N 1(cid:62)

bj (Xt)
yj
t

k=1 ˆwik

t [θj

t yk

t yj

= 1

+ αt

t
yi
t

s=0

4 Concluding Remarks

In this paper, we have established both asymptotic and non-asymptotic analyses for a consensus-based
distributed linear stochastic approximation algorithm over uniformly strongly connected graphs, and
proposed a push-based variant for coping with uni-directional communication. Both algorithms and
their analyses can be directly applied to TD learning. One limitation of our ﬁnite-time bounds is that
they involve quite a few constants which are well deﬁned and characterized but whose values are not
easy to compute. Future directions include leveraging the analyses for resilience in the presence of
malicious agents and extending the tools to more complicated RL.

9

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

437

438

439

440

441

442

References

[1] R.S. Sutton and A.G. Barto. Reinforcement Learning: An Introduction. Cambridge: MIT press,

1998.

[2] K. Zhang, Z. Yang, and T. Ba¸sar. Multi-agent reinforcement learning: A selective overview of

theories and algorithms. arXiv preprint arXiv:1911.10635, 2019.

[3] T.T. Doan, S.T. Maguluri, and J. Romberg. Finite-time analysis of distributed TD(0) with
linear function approximation on multi-agent reinforcement learning. In 36th International
Conference on Machine Learning, pages 1626–1635, 2019.

[4] K. Zhang, Z. Yang, H. Liu, T. Zhang, and T. Ba¸sar. Fully decentralized multi-agent reinforcement
learning with networked agents. In 35th International Conference on Machine Learning, pages
5872–5881, 2018.

[5] H. Robbins and S. Monro. A stochastic approximation method. The annals of mathematical

statistics, pages 400–407, 1951.

[6] R.S. Sutton and A.G. Barto. Reinforcement Learning: An Introduction. MIT Press, 2018.

[7] V. S. Borkar and S. P. Meyn. The ODE method for convergence of stochastic approximation
and reinforcement learning. SIAM Journal on Control and Optimization, 38(2):447–469, 2000.

[8] J.N. Tsitsiklis and B. Van Roy. An analysis of temporal-difference learning with function

approximation. IEEE Transactions on Automatic Control, 42(5):674–690, 1997.

[9] P. Dayan. The convergence of TD (λ) for general λ. Machine Learning, 8(3-4):341–362, 1992.

[10] G. Dalal, B. Szörényi, G. Thoppe, and S. Mannor. Finite sample analyses for TD(0) with
function approximation. In 32nd AAAI Conference on Artiﬁcial Intelligence, pages 6144–6160,
2018.

[11] C. Lakshminarayanan and C. Szepesvari. Linear stochastic approximation: How far does con-
stant step-size and iterate averaging go? In International Conference on Artiﬁcial Intelligence
and Statistics, pages 1347–1355, 2018.

[12] J. Bhandari, D. Russo, and R. Singal. A ﬁnite time analysis of temporal difference learning
with linear function approximation. In 31st Conference on Learning Theory, pages 1691–1692,
2018.

[13] R. Srikant and L. Ying. Finite-time error bounds for linear stochastic approximation and TD
learning. In 32nd Conference on Learning Theory, volume 99, pages 2803–2830. Proceedings
of Machine Learning Research, 25–28 Jun 2019.

[14] H. Gupta, R. Srikant, and L. Ying. Finite-time performance bounds and adaptive learning rate
selection for two time-scale reinforcement learning. In 33rd Conference on Neural Information
Processing System, pages 4706–4715, 2019.

[15] Y. Wang, W. Chen, Y. Liu, Z. Ma, and T. Liu. Finite sample analysis of the GTD policy
evaluation algorithms in markov setting. In 31st Conference on Neural Information Processing
Systems, pages 5504–5513, 2017.

[16] S. Ma, Y. Zhou, and S. Zou. Variance-reduced off-policy TDC learning: Non-asymptotic

convergence analysis. In 34th Conference on Neural Information Processing Systems, 2020.

[17] T. Xu, S. Zou, and Y. Liang. Two time-scale off-policy TD learning: Non-asymptotic analysis
over markovian samples. In 33rd Conference on Neural Information Processing Systems, pages
10634–10644, 2019.

[18] Z. Chen, S. T. Maguluri, S. Shakkottai, and K. Shanmugam. Finite-sample analysis of contractive
In 34th Conference on Neural

stochastic approximation using smooth convex envelopes.
Information Processing Systems, 2020.

[19] S. Zou, T. Xu, and Y. Liang. Finite-sample analysis for SARSA with linear function approx-
imation. In 33rd Conference on Neural Information Processing Systems, pages 8668–8678,
2019.

[20] G. Qu and A. Wierman. Finite-time analysis of asynchronous stochastic approximation and Q-
learning. In 33rd Conference on Learning Theory, volume 125, pages 3185–3205. Proceedings
of Machine Learning Research, 09–12 Jul 2020.

10

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

[21] Y. Wu, W. Zhang, P. Xu, and Q. Gu. A ﬁnite time analysis of two time-scale actor critic methods.

In 34th Conference on Neural Information Processing Systems, 2020.

[22] P. Xu and Q. Gu. A ﬁnite-time analysis of Q-learning with neural network function approxima-

tion. In 37th International Conference on Machine Learning, 2020.

[23] W. Weng, H. Gupta, N. He, L. Ying, and R. Srikant. The mean-squared error of double

Q-learning. In 34th Conference on Neural Information Processing Systems, 2020.

[24] Y. Wang and S. Zou. Finite-sample analysis of Greedy-GQ with linear function approximation
under markovian noise. In Jonas Peters and David Sontag, editors, Proceedings of the 36th
Conference on Uncertainty in Artiﬁcial Intelligence (UAI), volume 124 of Proceedings of
Machine Learning Research, pages 11–20. PMLR, 03–06 Aug 2020.

[25] S. Chen, A. M. Devraj, A. Buši´c, and S. Meyn. Explicit mean-square error bounds for monte-
carlo and linear stochastic approximation. In Proceedings of the Twenty Third International
Conference on Artiﬁcial Intelligence and Statistics, volume 108 of Proceedings of Machine
Learning Research, pages 4173–4183. PMLR, 26–28 Aug 2020.

[26] G. Wang, B. Li, and G. B. Giannakis. A multistep lyapunov approach for ﬁnite-time analysis of

biased stochastic approximation. arXiv preprint arXiv:1909.04299, 2019.

[27] G. Dalal, G. Thoppe, B. Szörényi, and S. Mannor. Finite sample analysis of two-timescale
In Conference On

stochastic approximation with applications to reinforcement learning.
Learning Theory, pages 1199–1233. PMLR, 2018.

[28] V. S. Borkar and S. Pattathil. Concentration bounds for two time scale stochastic approximation.
In 56th Annual Allerton Conference on Communication, Control, and Computing, pages 504–
511, 2018.

[29] J. N. Tsitsiklis. Problems in Decentralized Decision Making and Computation. PhD thesis,

Department of Electrical Engineering and Computer Science, MIT, 1984.

[30] Y. Zhang and M.M. Zavlanos. Distributed off-policy actor-critic reinforcement learning with
policy consensus. In 58th IEEE Conference on Decision and Control, pages 4674–4679, 2019.

[31] W. Suttle, Z. Yang, K. Zhang, Z. Wang, T. Ba¸sar, and J. Liu. A multi-agent off-policy actor-critic

algorithm for distributed reinforcement learning. In 21st IFAC World Congress, 2020.

[32] K. Zhang, Z. Yang, and T. Ba¸sar. Networked multi-agent reinforcement learning in continuous

spaces. In 57th IEEE Conference on Decision and Control, pages 2771–2776, 2018.

[33] H.J. Kushner and G. Yin. Asymptotic properties of distributed and communicating stochastic
approximation algorithms. SIAM Journal on Control and Optimization, 25(5):1266–1290, 1987.

[34] S. S. Stankovi´c, M. S. Stankovi´c, and D. Stipanovi´c. Decentralized parameter estimation by
consensus based stochastic approximation. IEEE Transactions on Automatic Control, 56(3):531–
543, 2010.

[35] M. Huang. Stochastic approximation for consensus: a new approach via ergodic backward

products. IEEE Transactions on Automatic Control, 57(12):2994–3008, 2012.

[36] M.S. Stankovi´c and S.S. Stankovi´c. Multi-agent temporal-difference learning with linear func-
tion approximation: Weak convergence under time-varying network topologies. In American
Control Conference, pages 167–172, 2016.

[37] P. Bianchi, G. Fort, and W. Hachem. Performance of a distributed stochastic approximation

algorithm. IEEE Transactions on Information Theory, 59(11):7405–7418, 2013.

[38] M. S. Stankovi´c, N. Ili´c, and S. S. Stankovi´c. Distributed stochastic approximation: weak
convergence and network design. IEEE Transactions on Automatic Control, 61(12):4069–4074,
2016.

[39] T. T. Doan, S. T. Maguluri, and J. Romberg. Finite-time performance of distributed temporal-
difference learning with linear function approximation. SIAM Journal on Mathematics of Data
Science, 3(1):298–320, 2021.

[40] G. Wang, S. Lu, G. Giannakis, G. Tesauro, and J. Sun. Decentralized TD tracking with linear
function approximation and its ﬁnite-time analysis. In 34th Conference on Neural Information
Processing Systems, 2020.

11

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

[41] K. Zhang, Z. Yang, H. Liu, T. Zhang, and T. Ba¸sar. Finite-sample analysis for decentralized batch
multi-agent reinforcement learning with networked agents. IEEE Transactions on Automatic
Control, 2021.

[42] J. Sun, G. Wang, G. B. Giannakis, Q. Yang, and Z. Yang. Finite-time analysis of decentralized
temporal-difference learning with linear function approximation. In International Conference
on Artiﬁcial Intelligence and Statistics, pages 4485–4495. PMLR, 2020.

[43] S. Zeng, T. T. Doan, and J. Romberg. Finite-time analysis of decentralized stochastic approxima-
tion with applications in multi-agent and multi-task learning. arXiv preprint arXiv:2010.15088,
2020.

[44] L. Xiao, S. Boyd, and S. Lall. A scheme for robust distributed sensor fusion based on average
consensus. In Proceedings of the 4th International Conference on Information Processing in
Sensor Networks, pages 63–70, 2005.

[45] B. Gharesifard and J. Cortés. Distributed strategies for generating weight-balanced and doubly

stochastic digraphs. European Journal of Control, 18(6):539–557, 2012.

[46] N. H. Vaidya, L. Tseng, and G. Liang. Iterative approximate byzantine consensus in arbitrary
directed graphs. In Proceedings of the 2012 ACM symposium on Principles of distributed
computing, pages 365–374, 2012.

[47] H. J. LeBlanc, H. Zhang, X. Koutsoukos, and S. Sundaram. Resilient asymptotic consensus in
robust networks. IEEE Journal on Selected Areas in Communications, 31(4):766–781, 2013.

[48] D. Kempe, A. Dobra, and J. Gehrke. Gossip-based computation of aggregate information. In

44th IEEE Symposium on Foundations of Computer Science, pages 482–491, 2003.

[49] A. Nedi´c, A. Olshevsky, and M. G. Rabbat. Network topology and communication-computation
tradeoffs in decentralized optimization. Proceedings of the IEEE, 106(5):953–976, 2018.

[50] A. Olshevsky and J. N. Tsitsiklis. On the nonexistence of quadratic lyapunov functions for
consensus algorithms. IEEE Transactions on Automatic Control, 53(11):2642–2645, 2008.
[51] B. Touri. Product of Random Stochastic Matrices and Distributed Averaging. Springer Science

& Business Media, 2012.

[52] A. Nedi´c and A. Olshevsky. Distributed optimization over time-varying directed graphs. IEEE

Transactions on Automatic Control, 60(3):601–615, 2014.

[53] J. Chen and A. H. Sayed. Diffusion adaptation strategies for distributed optimization and
learning over networks. IEEE Transactions on Signal Processing, 60(8):4289–4305, 2012.

[54] A. Jadbabaie, J. Lin, and A. S. Morse. Coordination of groups of mobile autonomous agents
using nearest neighbor rules. IEEE Transactions on Automatic Control, 48(6):988–1001, 2003.

[55] R. Olfati-Saber, J. A. Fax, and R. M. Murray. Consensus and cooperation in networked

multi-agent systems. Proc. IEEE, 95(1):215–233, 2007.

[56] A. Nedi´c and J. Liu. On convergence rate of weighted-averaging dynamics for consensus

problems. IEEE Transactions on Automatic Control, 62(2):766–781, 2017.

[57] A. Nedi´c, A. Olshevsky, A. Ozdaglar, and J. N. Tsitsiklis. On distributed averaging algorithms

and quantization effects. IEEE Transactions on automatic control, 54(11):2506–2517, 2009.

[58] M. Cao, A. S. Morse, and B. D. O. Anderson. Reaching a consensus in a dynamically changing
environment: a graphical approach. SIAM Journal on Control and Optimization, 47(2):575–600,
2008.

[59] A. Kolmogoroff. Zur theorie der markoffschen ketten. Mathematische Annalen, 112(1):155–160,

1936.

[60] D. Blackwell. Finite non-homogeneous chains. Annals of Mathematics, 46(4):594–599, 1945.

[61] A. Olshevsky and J. N. Tsitsiklis. Degree ﬂuctuations and the convergence time of consensus

algorithms. IEEE Transactions on Automatic Control, 58(10):2626–2631, 2013.

[62] S. Boyd, A. Ghosh, B. Prabhakar, and D. Shah. Randomized gossip algorithms.

IEEE

Transactions on Information Theory, 52(6):2508–2530, 2006.

[63] Adwaitvedant S Mathkar and Vivek S Borkar. Nonlinear gossip. SIAM Journal on Control and

Optimization, 54(3):1535–1557, 2016.

12

545

546

547

548

[64] J. Hajnal and M. S. Bartlett. Weak ergodicity in non-homogeneous Markov chains. Mathematical

Proceedings of the Cambridge Philosophical Society, 54:233–246, 1958.

[65] A. Nedi´c and A. Olshevsky. Distributed optimization over time-varying directed graphs. IEEE

Transactions on Automatic Control, 60(3):601–615, 2015.

13

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

Checklist

For each question, change the default [TODO] to [Yes] , [No] , or [N/A] . You are strongly encouraged
to include a justiﬁcation to your answer, either by referencing the appropriate section of your paper
or providing a brief inline description. For example:

• Did you include the license to the code and datasets? [Yes] See Section ??.
• Did you include the license to the code and datasets? [No] The code and the data are

proprietary.

• Did you include the license to the code and datasets? [N/A]

Please do not modify the questions and only use the provided macros for your answers. Note that the
Checklist section does not count towards the page limit. In your paper, please delete this instructions
block and only keep the Checklist section heading above along with the questions/answers below.

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reﬂect the paper’s

contributions and scope? [Yes] See Abstract and Section 1.

(b) Did you describe the limitations of your work? [Yes] See Section 4 and the paragraph

after Theorem 5.

(c) Did you discuss any potential negative societal impacts of your work? [N/A]
(d) Have you read the ethics review guidelines and ensured that your paper conforms to

them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [Yes] See Section 2.
(b) Did you include complete proofs of all theoretical results? [Yes] See Appendix B.

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main experi-

mental results (either in the supplemental material or as a URL)? [N/A]

(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they

were chosen)? [N/A]

(c) Did you report error bars (e.g., with respect to the random seed after running experi-

ments multiple times)? [N/A]

(d) Did you include the total amount of compute and the type of resources used (e.g., type

of GPUs, internal cluster, or cloud provider)? [N/A]

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

14

